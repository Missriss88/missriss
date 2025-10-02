# app1.py — Streamlit Cloud 종합본
# - 모드: 웹캠 / 이미지 업로드 / 동영상 업로드
# - 리포트: 규칙 기반(기본) + GOOGLE_API_KEY 있으면 Gemini로 요약
# - webrtc는 video_frame_callback 방식(안정), 버전 호환 셔임 포함

import os, math
from typing import Tuple, List
import google.generativeai as genai
import cv2
import av
import numpy as np
import streamlit as st

# ===== Streamlit<->webrtc 구버전 호환 셔임 (experimental_rerun 호출 대비) =====
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    st.experimental_rerun = st.rerun  # webrtc 내부가 experimental_rerun을 부르면 rerun을 대리 실행

from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ===== 프로젝트 탐지 모듈 =====
import fire_detector       # load_fire_model(path), detect_fire(frame, model)
import person_detector     # load_person_model(path), detect_person(frame, model)

st.set_page_config(page_title="실시간 화재/인명 감지", page_icon="🔥", layout="wide")

# -------------------------------------------------------
# (필요 시 수정) 모델 경로 / 기본 파라미터
# -------------------------------------------------------
FIRE_PT   = os.path.abspath("fire2.pt")      # 예: ./fire2.pt
PERSON_PT = os.path.abspath("yolov8n.pt")    # 예: ./yolov8n.pt (YOLOv5면 yolov5s.pt)
PROXIMITY_PX_DEFAULT = 150                   # 화재-사람 중심거리 경고 임계(px)

# -------------------------------------------------------
# 유틸
# -------------------------------------------------------
def center_distance(a, b) -> float:
    ax = (a[0]+a[2])/2.0; ay = (a[1]+a[3])/2.0
    bx = (b[0]+b[2])/2.0; by = (b[1]+b[3])/2.0
    return math.hypot(ax-bx, ay-by)

def draw_boxes(frame_bgr, fire_boxes, person_boxes, warning: bool):
    out = frame_bgr.copy()
    # Fire
    for (x1,y1,x2,y2) in fire_boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(out,"FIRE",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    # Person
    for (x1,y1,x2,y2) in person_boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,200,0),2)
        cv2.putText(out,"PERSON",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,0),2)
    # Warning banner
    if warning:
        h,w = out.shape[:2]
        cv2.rectangle(out,(0,0),(w,36),(0,0,255),-1)
        cv2.putText(out,"WARNING: PERSON NEAR FIRE",(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    return out

# -------------------------------------------------------
# 리포트(규칙 기반 + LLM 폴백)
# -------------------------------------------------------
def rule_based_report(fire_count:int, person_count:int, warning:bool)->str:
    if fire_count==0 and person_count==0:
        risk="안전"; summary="화재와 사람 모두 탐지되지 않았습니다."; action="별도 조치 필요 없습니다."
    elif fire_count>0 and person_count==0:
        risk="주의"; summary="작은 불이 탐지되었지만 주변에 사람은 없습니다."; action="상황을 관찰하세요. 확산/연기 증가 시 진화 조치를 고려하세요."
    elif fire_count>0 and person_count>0 and not warning:
        risk="주의"; summary="불이 있으나 사람은 안전거리에서 관찰 중입니다."; action="안전거리를 유지하고 소화기 등 대비 상태를 확인하세요."
    else:
        risk="경고"; summary="불 근처에서 사람이 감지되었습니다."; action="즉시 안전거리를 확보하고 필요 시 소화/대피를 안내하세요."
    return f"""[상황 요약]
- 화재: {fire_count}건 / 인원: {person_count}명 / 근접 경고: {'발생' if warning else '없음'}

[위험 평가] {risk}
{summary}

[권장 조치]
- {action}
"""

def generate_report_llm(fire_count:int, person_count:int, warning:bool, base_text:str)->str:
    """
    GOOGLE_API_KEY가 있으면 Gemini로 상황-맞춤 간결 요약 생성.
    - base_text(규칙 기반)를 컨텍스트로 제공해 과장 방지
    - 실패/키없음 => base_text 반환
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return base_text
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
        client = genai.Client(api_key=api_key)

        prompt = (
            "당신은 과장을 피하고 현장을 차분히 전달하는 안전 도우미입니다.\n"
            "다음 규칙 기반 요약을 존중하되 표현만 자연스럽고 간결하게 정리하세요.\n"
            "소규모 화염(예: 드럼통에 나무 태움)은 '주의' 수준으로 표현합니다.\n"
            "불필요한 공포 조성 금지, 한글 5~7문장 내.\n\n"
            f"규칙 기반 요약:\n{base_text}\n\n"
            f"탐지치: 화재={fire_count}, 인원={person_count}, 근접경고={'Y' if warning else 'N'}\n"
            "출력: 상황 요약 / 위험 평가(안전·주의·경고·심각 중) / 권장 조치(최소한)"
        )

        resp = client.models.generate_content(
            model="gemini-2.0-flash",  # 1.5에서 404 났던 이력 → 2.0 권장
            contents=prompt,
            config=GenerateContentConfig(max_output_tokens=400),
        )
        text = (resp.text or "").strip()
        return text or base_text
    except Exception:
        return base_text  # 폴백

# -------------------------------------------------------
# 모델 로딩(캐시)
# -------------------------------------------------------
@st.cache_resource
def load_models(fire_pt_path: str, person_pt_path: str):
    fire_model = fire_detector.load_fire_model(fire_pt_path)
    person_model = person_detector.load_person_model(person_pt_path)
    return fire_model, person_model

def detect_once(frame_bgr: np.ndarray, fire_model, person_model, proximity_px:int):
    fire_boxes   = fire_detector.detect_fire(frame_bgr, fire_model) or []
    person_boxes = person_detector.detect_person(frame_bgr, person_model) or []
    warning = False
    for fb in fire_boxes:
        for pb in person_boxes:
            if center_distance(fb, pb) <= proximity_px:
                warning = True
                break
        if warning: break
    annotated = draw_boxes(frame_bgr, fire_boxes, person_boxes, warning)
    return annotated, len(fire_boxes), len(person_boxes), warning

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🔥 실시간 화재/인명 감지")

# (디버그) 버전 출력 — 실제 배포 시 제거 가능
try:
    import streamlit_webrtc, streamlit
    st.caption(f"ST {streamlit.__version__} | WRTC {streamlit_webrtc.__version__}")
except Exception:
    pass

# 0) 모델 로딩
try:
    with st.spinner("AI 모델 로딩 중..."):
        fire_model, person_model = load_models(FIRE_PT, PERSON_PT)
    st.success("모델 로딩 완료 ✅")
except Exception as e:
    st.error("AI 모델 로딩 실패 ❌ .pt 파일과 경로를 확인하세요.")
    st.exception(e)
    st.stop()

mode = st.radio("모드 선택", ["웹캠", "이미지 업로드", "동영상 업로드"], horizontal=True)

# ===== 상태 보드(웹캠) =====
status_fire = st.empty(); status_person = st.empty(); status_warn = st.empty()

# =========================
# 1) 웹캠
# =========================
if mode == "웹캠":
    st.subheader("웹캠 실시간 감지")
    proximity_px = st.slider("위험 근접 거리(px)", 50, 500, PROXIMITY_PX_DEFAULT)

    # 리포트 상태
    if "report_text" not in st.session_state: st.session_state.report_text = ""
    if "report_done" not in st.session_state: st.session_state.report_done = False

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        annotated, f_cnt, p_cnt, warn = detect_once(frm, fire_model, person_model, proximity_px)

        # 상태 보드
        status_fire.metric("탐지된 화재", f_cnt)
        status_person.metric("탐지된 인원", p_cnt)
        status_warn.metric("근접 경고", "발생" if warn else "없음")

        # 경고 최초 발생 시 1회 LLM 리포트(폴백 포함)
        if warn and not st.session_state.report_done:
            base = rule_based_report(f_cnt, p_cnt, warn)
            st.session_state.report_text = generate_report_llm(f_cnt, p_cnt, warn, base)
            st.session_state.report_done = True

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    if not ctx or not ctx.state.playing:
        st.info("좌하단 **START** 버튼 클릭 후, 브라우저의 **카메라 권한 허용**을 눌러주세요.")

    if st.session_state.report_done and st.session_state.report_text:
        st.warning("🚨 위험 상황 감지됨. 아래 리포트를 확인하세요.")
        st.text_area("AI 리포트(과장 금지, 상황 맞춤)", st.session_state.report_text, height=260)
        c1, c2 = st.columns(2)
        if c1.button("리포트 초기화"):
            st.session_state.report_text = ""; st.session_state.report_done = False; st.rerun()
        if c2.button("근접 임계값 +20px"):
            proximity_px = min(500, proximity_px + 20)
            st.toast(f"임계값을 {proximity_px-20} → {proximity_px}px로 변경했습니다.")

# =========================
# 2) 이미지 업로드
# =========================
elif mode == "이미지 업로드":
    st.subheader("이미지 업로드 분석")
    file = st.file_uploader("이미지 선택(jpg/png)", type=["jpg","jpeg","png"])
    proximity_px = st.slider("위험 근접 거리(px)", 50, 500, PROXIMITY_PX_DEFAULT, key="imgprox")
    if file:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, f_cnt, p_cnt, warn = detect_once(img, fire_model, person_model, proximity_px)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="분석 결과", use_container_width=True)
        base = rule_based_report(f_cnt, p_cnt, warn)
        st.markdown(generate_report_llm(f_cnt, p_cnt, warn, base))

# =========================
# 3) 동영상 업로드
# =========================
else:
    st.subheader("동영상 업로드 분석")
    file = st.file_uploader("동영상 선택(mp4/avi/mov)", type=["mp4","avi","mov"])
    proximity_px = st.slider("위험 근접 거리(px)", 50, 500, PROXIMITY_PX_DEFAULT, key="vidprox")
    if file:
        # 임시 저장 → OpenCV 처리
        tmp_path = os.path.abspath("uploaded_video.tmp")
        with open(tmp_path, "wb") as f: f.write(file.read())
        cap = cv2.VideoCapture(tmp_path)

        viewer = st.empty()
        info = st.empty()
        frame_count = 0
        fire_total = 0
        person_total = 0
        warn_any = False

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame_count += 1
            annotated, f_cnt, p_cnt, warn = detect_once(frame, fire_model, person_model, proximity_px)
            fire_total += f_cnt; person_total += p_cnt; warn_any = warn_any or warn

            viewer.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            info.text(f"프레임: {frame_count} | 화재:{f_cnt} 인원:{p_cnt} 경고:{'Y' if warn else 'N'}")
            cv2.waitKey(1)

        cap.release()
        os.remove(tmp_path)
        st.success("분석 완료")
        base = rule_based_report(fire_total, person_total, warn_any)
        st.markdown(generate_report_llm(fire_total, person_total, warn_any, base))
