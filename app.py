import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import google.generativeai as genai
import av
from streamlit_webrtc import webrtc_streamer
import tempfile
import os
import requests
import yt_dlp
from collections import Counter
from datetime import datetime

# --- 설정 ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("오류: .streamlit/secrets.toml 파일에 'GOOGLE_API_KEY'가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {path} 파일을 찾을 수 없거나 손상되었습니다.")
        return None

model = load_yolo_model('fire2.pt')
if model is None:
    st.stop()

# --- Gemini LLM 연동 함수 ---
def get_llm_response(detection_summary):
    genai.configure(api_key=GOOGLE_API_KEY)
    if not detection_summary: return "탐지된 객체가 없습니다."
    prompt = (
        f"화재 감지 시스템에서 다음과 같은 객체들이 탐지되었습니다: '{detection_summary}'.\n"
        "이 정보를 바탕으로 다음 항목들을 분석해주세요:\n"
        "1. 즉시 필요한 조치 (예: '소화기 확인', '즉시 대피', '119 신고')\n"
        "2. 사진속의 상황을 정확하게 판단하여 답을 줄것(이미 진화되었는지, 아직 화재중인지)\n"
        "답변은 한국어로, 각 항목을 간단하게 요약해서 설명해주세요.  "
    )
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 호출 중 오류가 발생했습니다: {e}"

# --- 커스텀 드로잉 함수 ---
CLASS_COLORS = {"fire": (0, 0, 255), "smoke": (128, 128, 128)}
def draw_detections(image, results):
    img_copy = image.copy()
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_names = results[0].names
        for box, cls_id, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[cls_id]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_copy, (x1, y1 - h - 10), (x1 + w, y1 - 5), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img_copy

# --- 스트림릿 UI 및 세션 관리 ---
st.title("🔥 실시간 화재 탐지 시스템 (with Gemini AI)")
st.write("---")
st.sidebar.header("⚙️ 입력 소스 선택")
source_option = st.sidebar.selectbox("다음 중에서 입력 소스를 선택하세요.", ("이미지", "동영상", "실시간 웹캠"))

if 'detections' not in st.session_state: st.session_state.detections = []
if 'video_processed' not in st.session_state: st.session_state.video_processed = False
if 'source_option' not in st.session_state: st.session_state.source_option = source_option
if st.session_state.source_option != source_option:
    st.session_state.video_processed = False
    st.session_state.detections = []
    st.session_state.source_option = source_option

# --- 동영상/URL 공통 처리 함수 ---
def process_video_stream(cap):
    st.session_state.detections = []
    stframe = st.empty()
    with st.spinner("영상을 처리하며 실시간 탐지 중입니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = model(frame)
            annotated_frame = draw_detections(frame, results)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if results[0].boxes is not None:
                st.session_state.detections.extend([model.names[int(cls)] for cls in results[0].boxes.cls])
    cap.release()
    st.success("영상 분석이 완료되었습니다. 아래 버튼을 눌러 리포트를 생성하세요.")
    st.session_state.video_processed = True

# --- 이미지 처리 기능 ---
if source_option == "이미지":
    uploaded_file = st.sidebar.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        results = model(image)
        annotated_image = draw_detections(image, results)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="탐지 결과 이미지")
        detections = [model.names[int(cls)] for cls in results[0].boxes.cls] if results[0].boxes is not None else []
        if detections:
            detection_counts = Counter(detections)
            detection_summary = ", ".join([f"{obj} {count}개" for obj, count in detection_counts.items()])
            st.info(f"탐지 요약: **{detection_summary}**")
            with st.spinner("Gemini AI가 상황을 상세 분석 중입니다..."):
                description = get_llm_response(detection_summary)
                st.subheader("🤖 AI 상세 분석")
                st.markdown(description)

# --- 동영상 처리 기능 ---
elif source_option == "동영상":
    uploaded_video = st.sidebar.file_uploader("동영상 파일을 업로드하세요.", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("업로드된 동영상 분석 시작"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                video_path = tfile.name
            if video_path:
                process_video_stream(cv2.VideoCapture(video_path))
                os.remove(video_path)
    
    # ⭐️ 리포트 생성 버튼을 '동영상' 블록 안으로 이동
    if st.session_state.video_processed:
        if st.button("최종 상황 정리 리포트 생성"):
            if st.session_state.detections:
                detection_counts = Counter(st.session_state.detections)
                detection_summary = ", ".join([f"{obj} {count}개" for obj, count in detection_counts.items()])
                st.info(f"최종 탐지 요약: **{detection_summary}**")
                with st.spinner("Gemini AI가 종합 리포트를 생성 중입니다..."):
                    description = get_llm_response(detection_summary)
                    st.subheader("🤖 AI 종합 분석 리포트")
                    st.markdown(description)
            else:
                st.info("영상 전체에서 객체가 탐지되지 않았습니다.")

# --- URL 처리 기능 ---
elif source_option == "URL":
    url = st.sidebar.text_input("분석할 동영상 주소(URL)를 입력하세요.")
    if url and st.button("URL 동영상 분석 시작"):
        try:
            ydl_opts = {'format': 'best[ext=mp4]/best', 'noplaylist': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_url = info['url']
            st.video(video_url)
            process_video_stream(cv2.VideoCapture(video_url))
        except Exception as e:
            st.error(f"URL 처리 중 오류: {e}")

    # ⭐️ 리포트 생성 버튼을 'URL' 블록 안으로 이동
    if st.session_state.video_processed:
        if st.button("최종 상황 정리 리포트 생성"):
            if st.session_state.detections:
                detection_counts = Counter(st.session_state.detections)
                detection_summary = ", ".join([f"{obj} {count}개" for obj, count in detection_counts.items()])
                st.info(f"최종 탐지 요약: **{detection_summary}**")
                with st.spinner("Gemini AI가 종합 리포트를 생성 중입니다..."):
                    description = get_llm_response(detection_summary)
                    st.subheader("🤖 AI 종합 분석 리포트")
                    st.markdown(description)
            else:
                st.info("영상 전체에서 객체가 탐지되지 않았습니다.")

# --- 실시간 웹캠 처리 기능 ---
elif source_option == "실시간 웹캠":
    st.subheader("📹 실시간 웹캠 탐지")
    st.write("아래 'START' 버튼을 누르면 웹캠이 활성화됩니다.")
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_frame = draw_detections(img, results)
        detections = [model.names[int(cls)] for cls in results[0].boxes.cls] if results[0].boxes is not None else []
        detection_counts = Counter(detections)
        fire_count = detection_counts.get("fire", 0)
        smoke_count = detection_counts.get("smoke", 0)
        status_text_fire = f"Fire: {fire_count}"
        status_text_smoke = f"Smoke: {smoke_count}"
        cv2.rectangle(annotated_frame, (5, 5), (150, 65), (0, 0, 0), -1)
        cv2.putText(annotated_frame, status_text_fire, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, status_text_smoke, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        if fire_count > 0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert_text_time = f"Time: {now}"
            alert_text_situation = f"Situation: Fire({fire_count}), Smoke({smoke_count})"
            cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 380, 5), (annotated_frame.shape[1] - 5, 65), (0, 0, 255), -1)
            cv2.putText(annotated_frame, alert_text_time, (annotated_frame.shape[1] - 375, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, alert_text_situation, (annotated_frame.shape[1] - 375, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="webcam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    st.info("웹캠이 실행 중입니다. 화재 감지 시 우측 상단에 경고가 표시됩니다.")