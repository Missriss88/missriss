# app1.py
import os
import math
from typing import Tuple, List

import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

import fire_detector
import person_detector

st.set_page_config(page_title="실시간 화재/인명 감지", page_icon="🔥", layout="wide")

# -----------------------------
# 유틸
# -----------------------------
def abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def iou_or_center_dist(a, b) -> float:
    # 중심거리(px) 계산
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)

# -----------------------------
# 규칙 기반 리포트 (과장 금지)
# -----------------------------
def rule_based_report(fire_count: int, person_count: int, warning: bool) -> str:
    if fire_count == 0 and person_count == 0:
        risk = "안전"
        summary = "화재와 사람 모두 탐지되지 않았습니다."
        action = "별도 조치 필요 없습니다."
    elif fire_count > 0 and person_count == 0:
        risk = "주의"
        summary = "작은 불이 탐지되었지만 주변에 사람은 없습니다."
        action = "상황을 관찰하세요. 불이 확산되거나 연기가 많아지면 진화 조치를 고려하세요."
    elif fire_count > 0 and person_count > 0 and not warning:
        risk = "주의"
        summary = "불이 있으나 사람은 안전거리에서 관찰 중입니다."
        action = "안전거리를 유지하고 소화기 등 대비 상태를 확인하세요."
    else:
        # fire>0, person>0, warning True
        risk = "경고"
        summary = "불 근처에서 사람이 감지되었습니다."
        action = "즉시 안전거리를 확보하고 필요 시 소화/대피를 안내하세요."
    return f"""[상황 요약]
- 화재: {fire_count}건 / 인원: {person_count}명 / 근접 경고: {'발생' if warning else '없음'}

[위험 평가] {risk}
{summary}

[권장 조치]
- {action}
"""

# -----------------------------
# 모델 로드 (캐시)
# -----------------------------
@st.cache_resource
def load_models(fire_pt_path: str, person_pt_path: str):
    fire_model = fire_detector.load_fire_model(fire_pt_path)
    person_model = person_detector.load_person_model(person_pt_path)
    return fire_model, person_model

# -----------------------------
# 한 프레임 분석 및 그리기
# -----------------------------
def analyze_and_draw(
    frame_bgr: np.ndarray,
    fire_model,
    person_model,
    proximity_px: int = 150
) -> Tuple[np.ndarray, int, int, bool]:
    """프레임을 받아 탐지/표시/경고여부 반환"""
    # 탐지
    fire_boxes: List[List[int]] = fire_detector.detect_fire(frame_bgr, fire_model) or []
    person_boxes: List[List[int]] = person_detector.detect_person(frame_bgr, person_model) or []

    # 경고 판정(최소 중심거리)
    warning = False
    for fb in fire_boxes:
        for pb in person_boxes:
            if iou_or_center_dist(fb, pb) <= proximity_px:
                warning = True
                break
        if warning:
            break

    # 그리기
    annotated = frame_bgr.copy()
    for (x1, y1, x2, y2) in fire_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, "FIRE", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, "PERSON", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # 경고 표시
    if warning:
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 36), (0, 0, 255), -1)
        cv2.putText(annotated, "WARNING: PERSON NEAR FIRE",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return annotated, len(fire_boxes), len(person_boxes), warning

# -----------------------------
# UI
# -----------------------------
st.title("🔥 실시간 화재/인명 감지")

# 사이드바: 모델 경로/옵션
with st.sidebar:
    st.header("모델 설정")
    fire_pt_rel = st.text_input("화재 모델(.pt) 경로", value="fire2.pt")
    person_pt_rel = st.text_input("사람 모델(.pt) 경로", value="yolov8n.pt")  # YOLOv8 예시
    proximity_px = st.slider("위험 근접 거리(px)", 50, 500, 150)
    st.caption("화재/사람 중심거리(px)가 이 값보다 가까우면 '경고'로 표시합니다.")

# 절대경로 변환
fire_pt = abs_path(fire_pt_rel)
person_pt = abs_path(person_pt_rel)

# 모델 로드
try:
    with st.spinner("AI 모델 로딩 중..."):
        fire_model, person_model = load_models(fire_pt, person_pt)
    st.success("모델 로딩 완료 ✅")
except Exception as e:
    st.error("AI 모델 로딩 실패 ❌ .pt 경로나 환경을 확인하세요.")
    st.exception(e)
    st.stop()

st.divider()
st.subheader("웹캠")

# 리포트 상태
if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "report_done" not in st.session_state:
    st.session_state.report_done = False

# 최신 탐지 요약(사이드)
status_fire = st.empty()
status_person = st.empty()
status_warn = st.empty()

# ---- WebRTC 콜백 ----
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frm = frame.to_ndarray(format="bgr24")
    frm = cv2.flip(frm, 1)

    annotated, f_cnt, p_cnt, warn = analyze_and_draw(
        frm, fire_model, person_model, proximity_px
    )

    # 사이드 상태 갱신
    status_fire.metric("탐지된 화재", f_cnt)
    status_person.metric("탐지된 인원", p_cnt)
    status_warn.metric("근접 경고", "발생" if warn else "없음")

    # 경고 처음 발생 시 리포트 1회 생성(규칙 기반, 과장 금지)
    if warn and not st.session_state.report_done:
        st.session_state.report_text = rule_based_report(f_cnt, p_cnt, warn)
        st.session_state.report_done = True

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---- WebRTC 실행 ----
webrtc_streamer(
    key="webcam",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

# 리포트 표시
if st.session_state.report_done and st.session_state.report_text:
    st.warning("🚨 위험 상황이 감지되었습니다. 아래 리포트를 확인하세요.")
    st.text_area("현장 리포트(간결, 과장 금지)", st.session_state.report_text, height=250)
    cols = st.columns(2)
    if cols[0].button("리포트 초기화"):
        st.session_state.report_text = ""
        st.session_state.report_done = False
        st.rerun()
    if cols[1].button("근접 임계값 +20px"):
        # 상황에 따라 민감도 조절
        proximity_px_new = min(500, proximity_px + 20)
        st.toast(f"근접 임계값을 {proximity_px} → {proximity_px_new}px 로 높였습니다.")
