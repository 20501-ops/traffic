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
# 2. Orange3 설치 없이 .pkcls 내부 MLP 모델 및 전처리 추출
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


class _OrangeFreeUnpickler(pickle.Unpickler):
    """Orange3 패키지가 없는 Linux 클라우드 환경에서도 .pkcls를 열기 위한 언피클러"""

    def find_class(self, module, name):
        if module.startswith("Orange") or module.startswith("numpy.random"):
            return type(name, (_DummyObj,), {"__module__": module})
        return super().find_class(module, name)


@st.cache_resource
def load_model_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    with open(model_path, "rb") as f:
        raw_bytes = f.read()

    orange_wrapper = _OrangeFreeUnpickler(io.BytesIO(raw_bytes)).load()
    skl_model = orange_wrapper.skl_model

    # 도메인에서 시도 카테고리 목록과 각 변수의 정규화(offset, factor) 계수 추출
    sido_values = ("경기", "부산", "서울")
    for attr in orange_wrapper.original_domain.attributes:
        if getattr(attr, "_name", "") == "시도" and len(attr._args) >= 4:
            sido_values = tuple(attr._args[3])

    norm_params = {}
    for attr in orange_wrapper.domain.attributes:
        name = getattr(attr, "_name", "")
        cv = getattr(attr, "_compute_value", None)
        if cv is not None and hasattr(cv, "offset") and hasattr(cv, "factor"):
            norm_params[name] = (float(cv.offset), float(cv.factor))

    return {
        "skl_model": skl_model,
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
    """Orange3 전처리 파이프라인(원-핫 인코딩 + 표준화)을 동일하게 적용해 예측합니다."""
    norm = bundle["norm_params"]

    def scale(key: str, raw_val: float) -> float:
        offset, factor = norm[key]
        return (raw_val - offset) * factor

    # Orange3 Neural Network 전처리 피처 순서:
    # [시도=경기, 시도=부산, 시도=서울, 사고건수, 중상자수, 경상자수, 부상신고자수]
    x_vec = np.array(
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

    pred = float(bundle["skl_model"].predict(x_vec)[0])
    return max(0.0, pred)


# ---------------------------------------------------------
# 3. 위험도 등급 및 안전 수칙 데이터 정의
# ---------------------------------------------------------
def get_safety_status(predicted_deaths: float):
    if predicted_deaths < 7.0:
        return {
            "level": "SAFE",
            "title": "🟢 안전 (SAFE)",
            "subtitle": "예상 사고 치명도가 상대적으로 낮은 수준입니다.",
            "accent_color": "#16a34a",
            "bg_color": "#f0fdf4",
            "border_color": "#86efac",
            "rules": [
                "어린이 보호구역(스쿨존) 진입 시 규정 속도 **시속 30km 이하**를 준수하세요.",
                "횡단보도 앞에서는 보행자가 없더라도 **일시정지 후 좌우를 확인**하세요.",
                "등·하교 시간대 학교 정문 주변 주·정차를 자제해 어린이 시야를 확보해 주세요.",
                "어린이는 차도로 갑자기 뛰어들 수 있으므로 충분한 안전거리를 유지하세요.",
            ],
        }
    elif predicted_deaths < 12.0:
        return {
            "level": "CAUTION",
            "title": "🟡 주의 (CAUTION)",
            "subtitle": "사고 발생 시 중대 인명 피해로 이어질 가능성이 있습니다.",
            "accent_color": "#d97706",
            "bg_color": "#fffbeb",
            "border_color": "#fcd34d",
            "rules": [
                "스쿨존 내 **시속 20~30km 이하 서행** 및 전방 주시를 대폭 강화하세요.",
                "신호등 없는 횡단보도 앞에서는 **무조건 일시정지**하여 보행자 횡단 여부를 확인하세요.",
                "우회전 시 사각지대에 키가 작은 어린이가 있는지 반사경과 육안으로 재확인하세요.",
                "불법 주·정차 차량 사이에서 어린이가 튀어나올 수 있으니 브레이크에 발을 올리고 주행하세요.",
            ],
        }
    else:
        return {
            "level": "DANGER",
            "title": "🔴 위험 (DANGER)",
            "subtitle": "고위험 구간입니다. 스쿨존 및 교차로 특별 감속 운행이 필요합니다.",
            "accent_color": "#dc2626",
            "bg_color": "#fef2f2",
            "border_color": "#fca5a5",
            "rules": [
                "**즉시 감속 운행(시속 20km 이하)**을 실시하고, 가급적 어린이 보호구역 우회 경로를 이용하세요.",
                "스쿨존 내 주·정차는 전면 금지되어 있으며, 위반 시 일반 도로 대비 최대 3배의 과태료가 부과됩니다.",
                "교차로 우회전 시 **보행자 신호와 관계없이 일시정지** 후 서행 통과하세요.",
                "비·눈·야간 등 시야 확보가 어려운 상황에서는 전조등을 점등하고 비상 방어운전을 실천하세요.",
            ],
        }


def render_traffic_light(active_level: str) -> str:
    red_style = (
        "background:#ef4444; box-shadow:0 0 35px #ef4444, 0 0 70px rgba(239,68,68,0.6);"
        if active_level == "DANGER"
        else "background:#3f1d1d; opacity:0.35;"
    )
    yellow_style = (
        "background:#f59e0b; box-shadow:0 0 35px #f59e0b, 0 0 70px rgba(245,158,11,0.6);"
        if active_level == "CAUTION"
        else "background:#3f3214; opacity:0.35;"
    )
    green_style = (
        "background:#22c55e; box-shadow:0 0 35px #22c55e, 0 0 70px rgba(34,197,94,0.6);"
        if active_level == "SAFE"
        else "background:#143822; opacity:0.35;"
    )

    return f"""
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        background:#1e293b;
        border:4px solid #334155;
        border-radius:36px;
        padding:24px 20px;
        width:170px;
        margin:0 auto;
        box-shadow:0 12px 28px rgba(0,0,0,0.25);
    ">
        <div style="width:96px; height:96px; border-radius:50%; margin:10px 0; transition:all 0.25s ease; {red_style}"></div>
        <div style="width:96px; height:96px; border-radius:50%; margin:10px 0; transition:all 0.25s ease; {yellow_style}"></div>
        <div style="width:96px; height:96px; border-radius:50%; margin:10px 0; transition:all 0.25s ease; {green_style}"></div>
    </div>
    """


# ---------------------------------------------------------
# 4. 메인 UI 구성
# ---------------------------------------------------------
st.title("🚦 교통사고 예상 사망자 수 예측 및 스쿨존 안전 대시보드")
st.caption("Orange3 신경망(MLP) 회귀 모델(`traffic_casualty_model.pkcls`) 기반 실시간 시뮬레이터")

try:
    bundle = load_model_bundle(MODEL_PATH)
except Exception as e:
    st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.info("GitHub 저장소 루트 경로에 `traffic_casualty_model.pkcls` 파일이 함께 업로드되어 있는지 확인해 주세요.")
    st.stop()

st.divider()

left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    st.subheader("1️⃣ 지역 및 사고 조건 설정")

    sido_options = bundle["sido_values"]
    default_sido_idx = sido_options.index("서울") if "서울" in sido_options else 0

    selected_sido = st.selectbox(
        "시도 선택",
        options=sido_options,
        index=default_sido_idx,
        help="예측을 수행할 광역시·도를 선택하세요.",
    )

    accidents = st.slider(
        "사고건수 (건)",
        min_value=200,
        max_value=4600,
        value=1080,
        step=10,
        help="지역 내 발생한 교통사고 건수를 조절하세요.",
    )

    minor_injuries = st.slider(
        "경상자수 (명)",
        min_value=200,
        max_value=4800,
        value=1050,
        step=10,
        help="교통사고로 인한 경상자 수를 조절하세요.",
    )

    with st.expander("추가 모델 입력 변수 조절 (중상자수 · 부상신고자수)"):
        st.caption("모델 학습 시 함께 사용된 보조 변수입니다. 기본값은 학습 데이터의 중앙값입니다.")
        serious_injuries = st.slider(
            "중상자수 (명)",
            min_value=50,
            max_value=1000,
            value=261,
            step=5,
        )
        reported_injuries = st.slider(
            "부상신고자수 (명)",
            min_value=0,
            max_value=700,
            value=90,
            step=5,
        )

# 실시간 모델 예측 실행
predicted_deaths = predict_casualties(
    bundle=bundle,
    sido=selected_sido,
    accidents=accidents,
    serious_injuries=serious_injuries,
    minor_injuries=minor_injuries,
    reported_injuries=reported_injuries,
)
status = get_safety_status(predicted_deaths)

with right_col:
    st.subheader("2️⃣ 실시간 예측 결과 및 위험도 신호등")

    metric_col, light_col = st.columns([1.2, 1.0], vertical_alignment="center")

    with metric_col:
        st.metric(
            label=f"[{selected_sido}] 예상 사망자 수",
            value=f"{predicted_deaths:.2f} 명",
            delta=status["title"],
            delta_color="off",
        )
        st.markdown(
            f"""
            <div style="
                background:{status['bg_color']};
                border-left:6px solid {status['accent_color']};
                padding:14px 16px;
                border-radius:10px;
                margin-top:10px;
            ">
                <div style="font-size:1.25rem; font-weight:800; color:{status['accent_color']};">
                    {status['title']}
                </div>
                <div style="font-size:0.95rem; color:#334155; margin-top:4px;">
                    {status['subtitle']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("※ 신호등 기준: 🟢 안전(7명 미만) · 🟡 주의(7~12명 미만) · 🔴 위험(12명 이상)")

    with light_col:
        st.markdown(render_traffic_light(status["level"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    rules_html = "".join(
        [f"<li style='margin-bottom:8px; line-height:1.55;'>{rule}</li>" for rule in status["rules"]]
    )
    rules_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", rules_html)

    st.markdown(
        f"""
        <div style="
            background:{status['bg_color']};
            border:2px solid {status['border_color']};
            border-radius:16px;
            padding:22px 24px;
            box-shadow:0 4px 14px rgba(0,0,0,0.05);
        ">
            <div style="
                display:flex;
                align-items:center;
                justify-content:space-between;
                margin-bottom:12px;
                border-bottom:1px solid {status['border_color']};
                padding-bottom:10px;
            ">
                <span style="font-size:1.2rem; font-weight:800; color:#0f172a;">
                    🚸 어린이 보호구역(스쿨존) 핵심 안전 수칙
                </span>
                <span style="
                    background:{status['accent_color']};
                    color:white;
                    font-weight:700;
                    font-size:0.85rem;
                    padding:4px 12px;
                    border-radius:999px;
                ">
                    {status['title']}
                </span>
            </div>
            <ul style="margin:0; padding-left:20px; color:#1e293b; font-size:0.98rem;">
                {rules_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
