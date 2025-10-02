# app.py
import streamlit as st
import cv2
import numpy as np
import toml
import google.generativeai as genai
from PIL import Image
import tempfile
import time
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 제공된 탐지 모듈 임포트
from fire_detector import load_fire_model, detect_fire
from person_detector import load_person_model, detect_person

# --- 초기 설정 및 모델 로딩 ---

st.set_page_config(page_title="🔥 AI 화재 및 인명 안전 관제 시스템", layout="wide")
st.title("🔥 AI 화재 및 인명 안전 관제 시스템")
st.write("이미지, 동영상 또는 웹캠을 통해 화재와 사람을 감지하고 위험 상황 시 AI 분석 리포트를 생성합니다.")

# @st.cache_resource: Streamlit 앱의 성능 최적화를 위해 모델을 캐시에 저장
@st.cache_resource
def load_models():
    """YOLO 모델들을 로드하고 캐시에 저장합니다."""
    try:
        fire_model = load_fire_model('fire2.pt')
        person_model = load_person_model('yolov8n.pt')
        return fire_model, person_model
    except FileNotFoundError as e:
        st.error(f"모델 파일 로딩 오류: {e}. 'fire2.pt'와 'yolov8n.pt' 파일이 프로젝트 폴더에 있는지 확인하세요.")
        return None, None

fire_model, person_model = load_models()

# Google API 키 로드 및 Gemini 설정
try:
    secrets = toml.load("secret.toml")
    GOOGLE_API_KEY = secrets.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("secret.toml 파일에서 Google API 키를 찾을 수 없습니다.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
except FileNotFoundError:
    st.error("'secret.toml' 파일을 찾을 수 없습니다. API 키를 설정해주세요.")
    st.stop()

# --- 핵심 기능 함수 ---

def is_near(person_box, fire_box, threshold):
    """사람과 불의 바운딩 박스 중심점 사이의 거리를 계산하여 근접 여부를 판단합니다."""
    px, py = (person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2
    fx, fy = (fire_box[0] + fire_box[2]) / 2, (fire_box[1] + fire_box[3]) / 2
    distance = math.sqrt((px - fx)**2 + (py - fy)**2)
    return distance < threshold

def process_frame(frame, proximity_threshold):
    """단일 프레임에 대해 화재 및 사람 감지, 위험 분석을 수행합니다."""
    fire_boxes = detect_fire(frame, fire_model)
    person_boxes = detect_person(frame, person_model)
    
    dangerous_persons = set()

    # 위험 상황 분석
    for i, p_box in enumerate(person_boxes):
        for f_box in fire_boxes:
            if is_near(p_box, f_box, proximity_threshold):
                dangerous_persons.add(i)
                break

    # 시각화: 바운딩 박스 및 경고 그리기
    for i, p_box in enumerate(person_boxes):
        x1, y1, x2, y2 = p_box
        if i in dangerous_persons:
            # 위험에 처한 사람: 빨간색
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "DANGER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # 안전한 사람: 녹색
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for f_box in fire_boxes:
        x1, y1, x2, y2 = f_box
        # 화재: 주황색
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # 화면 상단에 경고 메시지 표시
    if dangerous_persons:
        warning_text = f"WARNING: {len(dangerous_persons)} person(s) near fire!"
        cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)
        
    return frame, len(fire_boxes), len(person_boxes), len(dangerous_persons)


def generate_ai_report(image, fire_count, person_count, danger_count):
    """Gemini API를 사용하여 AI 분석 리포트를 생성합니다."""
    st.info("AI 리포트를 생성 중입니다. 잠시만 기다려주세요...")
    
    model = genai.GenerativeModel('gemini-pro-vision')
    
    prompt = f"""
    당신은 최첨단 재난 분석 시스템입니다. 아래 이미지는 실제 재난 상황을 시뮬레이션한 것입니다.
    
    분석 데이터:
    - 탐지된 화재 수: {fire_count}
    - 탐지된 사람 수: {person_count}
    - 화재 근처 위험 인원 수: {danger_count}
    
    위 데이터와 이미지를 바탕으로 다음 항목에 대해 상세하고 전문적인 보고서를 작성해주세요:
    1.  **상황 개요**: 현재 이미지에 나타난 상황을 객관적으로 요약합니다.
    2.  **위험 평가**: 화재의 규모, 위험에 처한 사람의 수 등을 기반으로 현재 상황의 심각성을 평가합니다.
    3.  **권장 조치**: 이 상황에서 즉시 취해야 할 행동(예: 대피 경로 안내, 소방서 신고, 특정 인물 우선 구조 등)을 구체적으로 제안합니다.
    """
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    try:
        response = model.generate_content([prompt, pil_image])
        st.success("AI 리포트 생성 완료!")
        return response.text
    except Exception as e:
        st.error(f"AI 리포트 생성 중 오류가 발생했습니다: {e}")
        return None

# --- Streamlit UI 구성 ---

# 사이드바 설정
st.sidebar.title("⚙️ 설정")
proximity_threshold = st.sidebar.slider("위험 감지 임계값 (거리)", 50, 500, 150, 10,
                                        help="화재와 사람 사이의 거리가 이 값보다 가까우면 '위험'으로 판단합니다 (픽셀 단위).")

input_method = st.sidebar.radio("입력 방식 선택", ('이미지 업로드', '동영상 업로드', '실시간 웹캠'))

# AI 리포트 저장을 위한 세션 상태 초기화
if 'ai_report' not in st.session_state:
    st.session_state.ai_report = ""

if fire_model is None or person_model is None:
    st.warning("모델이 로드되지 않아 앱을 실행할 수 없습니다.")
else:
    if input_method == '이미지 업로드':
        uploaded_file = st.file_uploader("이미지 파일을 선택하세요...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            st.image(frame, channels="BGR", caption="원본 이미지")
            
            with st.spinner('이미지 분석 중...'):
                processed_frame, fire_count, person_count, danger_count = process_frame(frame.copy(), proximity_threshold)
            
            st.image(processed_frame, channels="BGR", caption="분석 결과")
            
            if st.button("AI 분석 리포트 생성"):
                report = generate_ai_report(processed_frame, fire_count, person_count, danger_count)
                st.session_state.ai_report = report


    elif input_method == '동영상 업로드':
        uploaded_file = st.file_uploader("동영상 파일을 선택하세요...", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            # AI 리포트 생성을 위한 프레임 캡쳐
            report_capture_frame = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, fire_count, person_count, danger_count = process_frame(frame.copy(), proximity_threshold)
                st_frame.image(processed_frame, channels="BGR")
                
                # 위험 상황 발생 시 첫 프레임을 리포트용으로 캡쳐
                if danger_count > 0 and report_capture_frame is None:
                    report_capture_frame = processed_frame.copy()

            cap.release()
            
            if report_capture_frame is not None:
                st.warning("동영상에서 위험 상황이 감지되었습니다.")
                if st.button("캡쳐된 위험 상황으로 AI 리포트 생성"):
                     report = generate_ai_report(report_capture_frame, fire_count, person_count, danger_count)
                     st.session_state.ai_report = report
            else:
                st.info("동영상 분석 완료. 감지된 위험 상황이 없습니다.")

    elif input_method == '실시간 웹캠':
        st.info("웹캠을 시작합니다. 브라우저에서 카메라 접근 권한을 허용해주세요.")

        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.proximity_threshold = 150 # 초기값

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # 사이드바 값 실시간 반영 (이 부분이 의도대로 동작하지 않음)
                self.proximity_threshold = proximity_threshold 
                
                processed_img, _, _, danger_count = process_frame(img, self.proximity_threshold)
                
                # 실시간 위험 경고
                if danger_count > 0:
                    cv2.putText(processed_img, "!! REAL-TIME DANGER !!", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)
                
                return processed_img
        
        webrtc_streamer(key="webcam", video_processor_factory=VideoTransformer)

    # AI 리포트 출력 영역
    if st.session_state.ai_report:
        st.markdown("---")
        st.subheader("🤖 AI 분석 리포트")
        st.markdown(st.session_state.ai_report)
        if st.button("리포트 초기화"):
            st.session_state.ai_report = ""
            st.rerun()
