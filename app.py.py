import io
import os
import pickle
import re
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="교통사고 예상 사망자 수 예측 및 스쿨존 안전 신호등",
    page_icon="🚦",
    layout="wide",
)

MODEL_PATH = "traffic_casualty_model.pkcls"


# ---------------------------------------------------------
# 2. 외부 ML 패키지(Orange3, sklearn) 없이 .pkcls 직접 로드
# ---------------------------------------------------------
class _DummyObj:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _PurePythonUnpickler(pickle.Unpickler):
    """Orange3 및 scikit-learn 설치 여부와 무관하게 .pkcls 내부 가중치만 안전하게 추출"""

    def find_class(self, module, name):
        if (
            module.startswith("Orange")
            or module.startswith("sklearn")
            or module.startswith("numpy.random")
        ):
            return type(name, (_DummyObj,), {"__module__": module})
        return super().find_class(module, name)


@st.cache_resource
def load_model_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    with open(model_path, "rb") as f:
        raw_bytes = f.read()

    orange_wrapper = _PurePythonUnpickler(io.BytesIO(raw_bytes)).load()
    skl_model = orange_wrapper.skl_model

    # 신경망 가중치(coefs_)와 절편(intercepts_) 추출
    coefs = [np.asarray(w, dtype=np.float64) for w in skl_model.coefs_]
    intercepts = [np.asarray(b, dtype=np.float64) for b in skl_model.intercepts_]

    # 시도 카테고리 목록 추출
    sido_values = ("경기", "부산", "서울")
    for attr in orange_wrapper.original_domain.attributes:
        if getattr(attr, "_name", "") == "시도" and len(attr._args) >= 4:
            sido_values = tuple(attr._args[3])

    # 전처리 표준화(offset, factor) 계수 추출
    norm_params = {}
    for attr in orange_wrapper.domain.attributes:
        name = getattr(attr, "_name", "")
        cv = getattr(attr, "_compute_value", None)
        if cv is not None and hasattr(cv, "offset") and hasattr(cv, "factor"):
            norm_params[name] = (float(cv.offset), float(cv.factor))

    return {
        "coefs": coefs,
        "intercepts": intercepts,
        "sido_values": list(sido_values),
        "norm_params": norm_params,
    }


def predict_casualties(
    bundle: dict,
    sido: str,
    accidents: float,
    serious_injuries: float,
    minor_injuries: float,
    reported_injuries: float,
) -> float:
    """Orange3 전처리(원-핫 인코딩 + 정규화) 및 MLP(ReLU -> Identity) 순전파 실행"""
    norm = bundle["norm_params"]

    def scale(key: str, raw_val: float) -> float:
        offset, factor = norm[key]
        return (raw_val - offset) * factor

    # 입력 피처 벡터 생성
    a = np.array(
        [[
            scale("시도=경기", 1.0 if sido == "경기" else 0.0),
            scale("시도=부산", 1.0 if sido == "부산" else 0.0),
            scale("시도=서울", 1.0 if sido == "서울" else 0.0),
            scale("사고건수", float(accidents)),
            scale("중상자수", float(serious_injuries)),
            scale("경상자수", float(minor_injuries)),
            scale("부상신고자수", float(reported_injuries)),
        ]],
        dtype=np.float64,
    )

    # 신경망 은닉층(ReLU) -> 출력층(Identity) 순전파
    coefs = bundle["coefs"]
    intercepts = bundle["intercepts"]
    for i in range(len(coefs) - 1):
        a = np.maximum(0.0, a @ coefs[i] + intercepts[i])
    out = a @ coefs[-1] + intercepts[-1]

    return max(0.0, float(out[0, 0]))


# ---------------------------------------------------------
# 3. 위험도 등급 및 안전 수칙 데이터 정의
# ---------------------------------------------------------
def get_safety_status(predicted_deaths: float):
    if predicted_deaths < 7.0:
        return {
            "level": "SAFE",
