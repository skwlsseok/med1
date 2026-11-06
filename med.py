import streamlit as st
import requests
import urllib.parse
from PIL import Image
import numpy as np
import os
import sys
import cv2
from datetime import datetime, time, date
from paddleocr import PaddleOCR

# ------------------------------------------
# 경량 LLM 코어(core.py) 불러오기: 같은 폴더, ./medguard_server, ../medguard_server 순서로 찾음
# ------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (_here, os.path.join(_here, "medguard_server"), os.path.join(os.path.dirname(_here), "medguard_server")):
    if os.path.exists(os.path.join(_cand, "core.py")):
        sys.path.insert(0, _cand)
        break
import core

# 페이지 설정
st.set_page_config(
    page_title="테스트용 의약품 정보 시스템",
    page_icon="💊",
    layout="centered"
)

# ==========================================
# 0. 사이드바 테마 설정 (다크 모드 / 라이트 모드)
# ==========================================
st.sidebar.header("🎨 화면 테마 설정")
theme_mode = st.sidebar.radio("모드 선택", ["라이트 모드 (Light)", "다크 모드 (Dark)"], index=0)

# 테마별 색상 변수 정의 (Selectbox 반전 대비 포함)
if theme_mode == "다크 모드 (Dark)":
    app_bg = "#0e1117"
    container_bg = "#161b22"
    text_color = "#ffffff"
    border_color = "#30363d"
    sub_text = "#8b949e"
    # [다크 모드 Selectbox] 바탕(검정) 대비 -> 하얀색 박스 + 검정색 글씨
    sb_bg, sb_text, sb_border = "#ffffff", "#000000", "#ffffff"
    dropdown_hover = "#e9ecef"
else:
    app_bg = "#f0f2f5"
    container_bg = "#ffffff"
    text_color = "#212529"
    border_color = "#e1e4e8"
    sub_text = "#6c757d"
    # [라이트 모드 Selectbox] 바탕(흰색) 대비 -> 진한 흑회색 박스 + 하얀색 글씨
    sb_bg, sb_text, sb_border = "#343a40", "#ffffff", "#212529"
    dropdown_hover = "#495057"

# ==========================================
# 0-1. 사이드바 AI 설정 (경량화 LLM)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 설정")
_profile_keys = list(core.MODEL_PROFILES.keys())
selected_profile = st.sidebar.selectbox(
    "AI 모델 크기 (기기 사양에 맞게)",
    _profile_keys,
    index=1,
    key="llm_profile"
)
API_URL = os.environ.get("MEDGUARD_API_URL", "").rstrip("/")
API_KEY = os.environ.get("MEDGUARD_APP_KEY", "")
DUR_KEY = os.environ.get("DUR_API_KEY", "")

if API_URL:
    st.sidebar.caption(f"서버 연동 모드: {API_URL}")
else:
    st.sidebar.caption("로컬 모델 모드: 이 PC에서 직접 LLM을 실행합니다.")
st.sidebar.caption(
    "DUR 조회: 연결됨" if DUR_KEY else "DUR 조회: DUR_API_KEY 미설정 (내장 성분 규칙만 사용)"
)
st.sidebar.warning("⚠️ 참고용 정보입니다. 복용 전 의사·약사 등 전문가 상담이 필요합니다.")

# 다크/라이트 모드에 따른 동적 CSS 주입 (Selectbox 반전 디자인 포함)
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {app_bg};
    }}
    .mobile-container {{
        max-width: 480px;
        margin: auto;
        background-color: {container_bg};
        padding: 20px;
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}
    .top-banner {{
        background: linear-gradient(135deg, #0066ff 0%, #004080 100%);
        color: white;
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 20px;
    }}
    /* 글자 가독성 보장 */
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: {text_color} !important;
    }}
    .top-banner h4, .top-banner p {{
        color: white !important;
    }}
    
    /* ⭐ Selectbox (선택 상자) 본체 강력한 반전 디자인 ⭐ */
    div[data-baseweb="select"] > div {{
        background-color: {sb_bg} !important;
        border: 2px solid {sb_border} !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        padding: 4px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    /* Selectbox 내부 선택된 텍스트 */
    div[data-baseweb="select"] span {{
        color: {sb_text} !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }}
    /* Selectbox 우측 화살표(아이콘) */
    div[data-baseweb="select"] svg {{
        fill: {sb_text} !important;
    }}
    
    /* ⭐ 드롭다운 리스트(메뉴) 테마 동기화 ⭐ */
    ul[data-baseweb="menu"] {{
        background-color: {sb_bg} !important;
        border-radius: 12px !important;
        padding: 5px 0 !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
    }}
    ul[data-baseweb="menu"] li {{
        color: {sb_text} !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        background-color: transparent !important;
    }}
    ul[data-baseweb="menu"] li:hover {{
        background-color: {dropdown_hover} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 (메뉴 이동용)
if "menu" not in st.session_state:
    st.session_state.menu = "home"
if "scan_ctx" not in st.session_state:
    st.session_state.scan_ctx = None
if "llm_text" not in st.session_state:
    st.session_state.llm_text = ""
if "llm_timing" not in st.session_state:
    st.session_state.llm_timing = {}

# 의료 면책 문구
MEDICAL_DISCLAIMER = (
    "⚠️ **의료 면책 안내**: 본 결과는 공공데이터 및 AI 기반 참고용 정보이며, "
    "의학적 진단이나 처방을 대체하지 않습니다. 복용 전 반드시 의사·약사 등 전문가에게 진단·상담을 받으세요. "
    "이상 증상(발진, 호흡곤란, 출혈, 심한 복통 등)이 있으면 즉시 진료를 받거나 119에 연락하세요."
)

@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    # torch와 CUDA/cuDNN DLL이 충돌하지 않도록 OCR은 CPU로 실행 (WinError 127 방지)
    options = dict(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    try:
        return PaddleOCR(device="cpu", **options)
    except TypeError:        # device 옵션을 지원하지 않는 구버전
        return PaddleOCR(**options)

DRUG_INFO = {
    "타이레놀 500mg": {"ingredients": ["아세트아미노펜"], "description": "해열 및 진통 목적 의약품", "warning": "매일 술을 마시는 경우 간 손상 유발 가능성"},
    "테라플루 나이트열정": {"ingredients": ["아세트아미노펜"], "description": "감기 제증상 완화", "warning": "졸음을 유발할 수 있으므로 운전 주의"},
    "아스피린 프로텍트정": {"ingredients": ["아세틸살리실산"], "description": "혈소판 응집 억제 및 진통", "warning": "위장관 출혈 위험 주의"},
    "이지엔6 이부프로펜": {"ingredients": ["이부프로펜"], "description": "소염진통제", "warning": "위장장애 주의"}
}
DRUG_LIST = list(DRUG_INFO.keys())

def extract_text_from_image(uploaded_file):
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        ocr_engine = get_ocr_engine()
        results = ocr_engine.predict(image_cv)
        texts = []
        for result in results:
            res_data = getattr(result, "json", lambda: result)()
            if isinstance(res_data, dict):
                rec_texts = res_data.get("res", res_data).get("rec_texts", [])
                texts.extend(rec_texts)
        return "\n".join(texts) if texts else "글자를 인식하지 못했습니다."
    except Exception as e:
        return f"OCR 오류: {e}"

def get_drug_ingredients(drug_name):
    for key in DRUG_INFO:
        if key in drug_name:
            return DRUG_INFO[key]["ingredients"]
    return []

def short_name(drug_name):
    """DUR 조회용 제품 기본명 (예: '타이레놀 500mg' -> '타이레놀')"""
    return drug_name.split()[0] if drug_name else drug_name

def build_dur_msg_for_llm(ctx):
    """LLM에 전달할 근거 문구 (DUR 조회 성공/불가/미조회를 구분)"""
    dur = ctx["dur"]
    dup = ctx["duplicate"]
    builtin = (
        f"내장 성분 정보 기준 동일 성분 중복: {', '.join(dup)}" if dup
        else "내장 성분 정보 기준 동일 성분 중복 없음"
    )
    if dur is None:
        return builtin + " (식약처 DUR 병용금기는 조회하지 않음)"
    if dur.get("status") == "ok":
        return dur["message"] + " / " + builtin
    return (
        "식약처 DUR 조회 불가: " + dur["message"] + " / " + builtin
        + " (병용금기 여부는 '확인 불가'로 안내할 것)"
    )

def run_llm_via_api(ctx):
    """서버(FastAPI) 연동 모드: /api/v1/analyze 호출 (스트리밍 없음)"""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    payload = {
        "base_drug": short_name(ctx["base"]),
        "second_drug": short_name(ctx["second"]) if ctx["second"] else "",
        "ocr_text": ctx["ocr_text"],
        "base_ingredients": ctx["base_ings"],
        "second_ingredients": ctx["second_ings"],
    }
    resp = requests.post(f"{API_URL}/api/v1/analyze", json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()["report"]


# 모바일 UI 프레임 시작
st.markdown('<div class="mobile-container">', unsafe_allow_html=True)

# 상단 앱바
col_top1, col_top2 = st.columns([4, 1])
with col_top1:
    st.markdown("### 🧪 **약품 정보 시스템** <span style='font-size:11px; background-color:#e6f0ff; color:#0066ff; padding:2px 6px; border-radius:8px;'>테스트 버전</span>", unsafe_allow_html=True)
with col_top2:
    if st.button("🏠 홈", key="home_btn"):
        st.session_state.menu = "home"
        st.rerun()

st.markdown("---")

# ------------------------------------------
# [화면 1] 홈 화면
# ------------------------------------------
if st.session_state.menu == "home":
    st.markdown("""
        <div class="top-banner">
            <h4 style="margin:0; color:white;">의약품 성분 및 복용 관리 (테스트)</h4>
            <p style="margin:5px 0 0 0; font-size:12px; opacity:0.9;">처방전 스캔과 성분 충돌, 복용 알림을 검토하세요!</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### **테스트 주요 기능**")

    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        if st.button("📄 **처방전 스캔**\n\nOCR 및 성분 분석", width="stretch"):
            st.session_state.menu = "scan"
            st.rerun()
    with r1_c2:
        if st.button("📅 **복용 일정 및 알림**\n\n달력 & 시간 설정", width="stretch"):
            st.session_state.menu = "alarm"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(MEDICAL_DISCLAIMER)

# ------------------------------------------
# [화면 2] 처방전 스캔 및 성분 충돌 분석 화면
# ------------------------------------------
elif st.session_state.menu == "scan":
    st.markdown("#### 📄 **처방전 / 약품 라벨 스캔**")

    uploaded_file = st.file_uploader("약봉지 또는 약품 사진 업로드", type=["jpg", "jpeg", "png"])
    selected_drug = st.selectbox("기준 약품 선택", DRUG_LIST)
    secondary_drug = st.selectbox("함께 복용 중인 약물", ["없음"] + DRUG_LIST)

    ocr_text = ""
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", width="stretch")
        with st.spinner("📖 OCR로 글자 읽는 중..."):
            ocr_text = extract_text_from_image(uploaded_file)
        st.text_area("OCR 인식 결과", ocr_text, height=80)

    if st.button("🚀 종합 성분 충돌 검토 실행", type="primary", width="stretch"):
        base_ings = get_drug_ingredients(selected_drug)
        has_second = secondary_drug != "없음"
        second_ings = get_drug_ingredients(secondary_drug) if has_second else []
        duplicate = sorted(set(base_ings) & set(second_ings)) if has_second else []

        dur_result = None
        if has_second and DUR_KEY:
            dur_result = core.check_dur_contraindication(
                short_name(selected_drug), short_name(secondary_drug), DUR_KEY
            )

        st.session_state.scan_ctx = {
            "base": selected_drug,
            "second": secondary_drug if has_second else "",
            "base_ings": base_ings,
            "second_ings": second_ings,
            "duplicate": duplicate,
            "dur": dur_result,
            "ocr_text": ocr_text if not ocr_text.startswith(("OCR 오류", "글자를 인식하지")) else "",
        }
        st.session_state.llm_text = ""      # 새 검토를 하면 이전 AI 설명은 지움
        st.session_state.llm_timing = {}

    ctx = st.session_state.scan_ctx
    if ctx:
        st.markdown("---")
        st.markdown("##### 🔬 **분석 결과 리포트**")
        st.success(f"**기준 약품**: {ctx['base']} (주성분: {', '.join(ctx['base_ings']) or '등록 정보 없음'})")

        if ctx["second"]:
            # 1) 내장 성분 규칙 (식약처 DUR이 아닙니다)
            if ctx["duplicate"]:
                st.error(
                    f"🚨 **동일 성분 중복**: {', '.join(ctx['duplicate'])} — "
                    "함께 복용하면 1일 최대 허용량을 넘길 수 있어 주의가 필요합니다. (내장 성분 정보 기준)"
                )
            else:
                st.success("✅ 내장 성분 정보 기준으로 직접적인 성분 중복은 확인되지 않았습니다.")

            # 2) 약리 계열 확인 (사전 기준, 코드가 판정)
            class_text = core.build_class_facts(ctx["base_ings"], ctx["second"], ctx["second_ings"])
            with st.expander("🧬 약리 계열 확인 (사전 기준)"):
                st.markdown(class_text)

            # 3) 식약처 DUR 조회
            dur = ctx["dur"]
            if dur is None:
                st.info("ℹ️ DUR_API_KEY가 설정되지 않아 식약처 DUR 병용금기는 조회하지 않았습니다. (위 결과는 내장 성분 규칙 기준)")
            elif dur.get("status") != "ok":
                st.warning(f"⚠️ **{dur['risk_level']}**\n\n{dur['message']}")
            elif dur["has_conflict"]:
                st.error(f"🚨 **{dur['risk_level']}**\n\n{dur['message']}")
            else:
                st.info(f"ℹ️ **{dur['risk_level']}**\n\n{dur['message']}")
        else:
            st.info("함께 복용하는 약물이 선택되지 않았습니다.")

        # ------------------------------------------
        # 경량 LLM: 근거(사전·DUR)를 쉬운 말로 설명 (RAG)
        # ------------------------------------------
        st.markdown("##### 🧠 **AI 복약 설명**")
        st.caption(
            f"{core.MODEL_PROFILES[selected_profile]['short']} · "
            "사전·식약처 DUR 근거를 먼저 넣고 그 근거로만 쉬운 말로 풀이합니다."
        )

        if st.button("🧠 AI 복약 설명 생성", width="stretch", key="llm_btn"):
            dur_msg = build_dur_msg_for_llm(ctx)
            try:
                if API_URL:
                    with st.spinner("서버에서 AI 설명을 생성 중입니다..."):
                        st.session_state.llm_text = run_llm_via_api(ctx)
                    st.session_state.llm_timing = {}
                    st.rerun()
                else:
                    timing = {}
                    status_box = st.empty()
                    status_box.info("⏳ AI 분석 중... (최초 1회는 모델 로딩 시간이 걸립니다)")

                    def _stream():
                        first = True
                        for piece in core.analyze_with_local_llm_stream(
                            ocr_text=ctx["ocr_text"],
                            base_drug=short_name(ctx["base"]),
                            second_drug=short_name(ctx["second"]) if ctx["second"] else "",
                            base_ingredients=ctx["base_ings"],
                            dur_msg=dur_msg,
                            profile_key=selected_profile,
                            timing=timing,
                            second_ingredients=ctx["second_ings"],
                        ):
                            if first:
                                status_box.empty()
                                first = False
                            yield piece

                    st.session_state.llm_text = st.write_stream(_stream())
                    status_box.empty()
                    st.session_state.llm_timing = dict(timing)
            except Exception as e:
                st.error(f"❌ AI 설명 생성 중 오류가 발생했습니다: {e}")
                st.session_state.llm_text = ""
        elif st.session_state.llm_text:
            st.markdown(st.session_state.llm_text)

        tm = st.session_state.llm_timing
        if tm:
            st.caption(
                f"⏱ 모델 로딩 {tm.get('load', 0):.1f}초 · "
                f"첫 글자까지 {tm.get('first_token', 0):.1f}초 · "
                f"전체 생성 {tm.get('total', 0):.1f}초"
            )

        st.markdown(f"<br>{MEDICAL_DISCLAIMER}", unsafe_allow_html=True)

    if st.button("⬅️ 메인으로 돌아가기"):
        st.session_state.menu = "home"
        st.rerun()

# ------------------------------------------
# [화면 3] 달력 및 정밀 시간 예약 기능 화면
# ------------------------------------------
elif st.session_state.menu == "alarm":
    st.markdown("#### 📅 **복용 일정 및 알림 시간 설정**")
    st.caption("복용을 시작할 날짜와 세부 알림 시간을 선택하세요.")

    target_medicine = st.selectbox("알림 설정할 약품 선택", DRUG_LIST)
    selected_date = st.date_input("복용 시작일 선택", value=date.today())

    st.markdown("##### ⏰ **복용 시간 상세 설정**")
    t_col1, t_col2, t_col3 = st.columns(3)

    with t_col1:
        ampm = st.selectbox("오전/오후", ["오전", "오후"])
    with t_col2:
        hour = st.selectbox("시간", [str(i) for i in range(1, 13)], index=8)
    with t_col3:
        minute = st.selectbox("분", ["00", "10", "20", "30", "40", "50"])

    st.markdown(f"""
        <div style="background-color:rgba(0,102,255,0.1); border-radius:10px; padding:10px; margin-top:10px; font-size:13px;">
            📌 <b>설정 요약</b><br>
            - 약품명: <b>{target_medicine}</b><br>
            - 시작일자: <b>{selected_date.strftime('%Y년 %m월 %d일')}</b><br>
            - 알림시간: <b>{ampm} {hour}시 {minute}분</b>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔔 복용 알림 및 일정 등록 완료", type="primary", width="stretch"):
        st.success(f"[{target_medicine}] 복용 알림이 성공적으로 등록되었습니다!")
        st.balloons()

    if st.button("⬅️ 메인으로 돌아가기"):
        st.session_state.menu = "home"
        st.rerun()

# 하단 내비게이션 바
st.markdown("---")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("🏠 홈", width="stretch"):
        st.session_state.menu = "home"
        st.rerun()
with nav2:
    if st.button("📄 스캔", width="stretch"):
        st.session_state.menu = "scan"
        st.rerun()
with nav3:
    if st.button("📅 일정 설정", width="stretch"):
        st.session_state.menu = "alarm"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
