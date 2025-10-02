import streamlit as st
import cv2
import fire_detector
import person_detector
import google.generativeai as genai
import numpy as np
import tempfile
from PIL import Image
import queue

# streamlit-webrtc 라이브러리를 임포트합니다.
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 화재 및 인명 안전 시스템",
    page_icon="🚨",
    layout="wide",
)

# --- 앱 제목 ---
st.title("🚨 AI 화재 및 인명 감지 시스템 (CCTV용 모델)")
st.markdown("---")

# --- AI 리포트 결과 저장을 위한 세션 상태 초기화 ---
if "webrtc_report" not in st.session_state:
    st.session_state.webrtc_report = None

# --- 사이드바 설정 ---
st.sidebar.title("⚙️ 제어판")

# Gemini API 키 입력
st.sidebar.header("API 키 설정")
try:
    # Streamlit Cloud의 Secrets에서 키를 먼저 시도
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    # 없으면 사용자에게 직접 입력받음
    GOOGLE_API_KEY = st.sidebar.text_input(
        "Google Gemini API 키를 입력하세요.", type="password", help="API 키가 없으면 리포트 생성이 불가합니다."
    )

# 작업 모드 선택
st.sidebar.header("탐지 모드")
app_mode = st.sidebar.selectbox(
    "작업 모드를 선택하세요",
    ["실시간 웹캠 감지", "파일 업로드 및 분석"]
)
st.sidebar.markdown("---")

# --- 모델 로딩 ---
@st.cache_resource
def load_models():
    """AI 모델을 로드하는 함수 (캐싱하여 성능 최적화)"""
    fire_model = fire_detector.load_fire_model('fire2.pt')
    person_model = person_detector.load_person_model('yolov5s.pt')
    return fire_model, person_model

with st.spinner('AI 모델을 로딩 중입니다...'):
    fire_model, person_model = load_models()

if fire_model is None or person_model is None:
    st.error("AI 모델 로딩에 실패했습니다. .pt 파일들을 확인해주세요.")
    st.stop()

# --- 공통 함수: 프레임 분석 및 시각화 ---
def analyze_and_draw_on_frame(frame, proximity_threshold):
    """프레임 내에서 객체를 탐지하고 위험상황을 분석하여 시각화합니다."""
    fire_boxes = fire_detector.detect_fire(frame, fire_model)
    person_boxes = person_detector.detect_person(frame, person_model)
    is_warning = False

    if fire_boxes:
        for f_box in fire_boxes:
            cv2.rectangle(frame, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (0, 0, 255), 3)
            cv2.putText(frame, 'Fire', (f_box[0], f_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            for p_box in person_boxes:
                fire_center_x = f_box[0] + (f_box[2] - f_box[0]) / 2
                person_center_x = p_box[0] + (p_box[2] - p_box[0]) / 2
                if abs(fire_center_x - person_center_x) < proximity_threshold:
                    is_warning = True
                    cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 165, 255), 4)
                else:
                    cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)
                cv2.putText(frame, 'Person', (p_box[0], p_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        for p_box in person_boxes:
            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)
            cv2.putText(frame, 'Person', (p_box[0], p_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if is_warning:
        cv2.putText(frame, "WARNING: Person Near Fire!", (50, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 3)
        
    return frame, len(fire_boxes), len(person_boxes), is_warning

# --- 공통 함수: AI 리포트 생성 ---
def generate_report(fire_count, person_count, is_warning, image_frame):
    """탐지 결과를 바탕으로 Gemini AI 리포트를 생성합니다."""
    if not GOOGLE_API_KEY:
        st.error("Gemini API 키가 설정되지 않았습니다. 리포트를 생성할 수 없습니다.")
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
        
        prompt_parts = [
            pil_image,
            "당신은 CCTV 이미지를 분석하는 AI 안전 전문가입니다.",
            "\n## 지시사항",
            "첨부된 이미지와 아래 요약 데이터를 바탕으로 상황을 분석하고 안전 리포트를 작성하세요.",
            "\n## 분석 데이터",
            f"- 화재 객체 수: {fire_count}",
            f"- 사람 객체 수: {person_count}",
            f"- 위험 경고 (사람-화재 근접): {'발생' if is_warning else '없음'}",
            "\n## 분석 가이드라인",
            "1.  **[상황 맥락 파악]** 이미지 속 불이 통제된 상황(예: 드럼통 안의 모닥불, 캠프파이어)인지, 통제되지 않은 위험한 화재(예: 건물 화재, 산불)인지 먼저 판단하세요. 주변 환경(실내/실외)과 사람들의 행동(불을 쬐는 중/대피 중)을 근거로 제시하세요.",
            "2.  **[위험도 평가]** 위 맥락에 따라 위험도를 '안전', '주의', '경고', '심각' 4단계로 평가하고, 그 이유를 구체적으로 설명하세요. (예: '드럼통 내부의 통제된 불이며 주변에 인화물질이 없어 안전 단계임')",
            "3.  **[권장 조치]** 평가된 위험도에 맞는 현실적인 조치를 1~2가지 제안하세요. 심각한 상황이 아니라면, '안전거리 유지', '소화기 위치 확인' 등 예방적 조치를 권고하세요. 과장된 경고는 피하세요.",
            "\n위 가이드라인에 따라 리포트를 생성해주세요."
        ]
        
        # 안정적인 최신 모델 사용
        model = genai.GenerativeModel('gemini-1.5-flash') 
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"AI 리포트 생성 중 오류가 발생했습니다: {e}")
        return None

# --- 모드 1: 실시간 웹캠 감지 (안정적인 방식으로 수정) ---
if app_mode == "실시간 웹캠 감지":
    st.header("실시간 웹캠 감지")
    st.info("웹캠 실행 버튼을 누르면 감지가 시작됩니다. 브라우저에서 카메라 권한을 허용해주세요.")

    proximity_threshold = st.slider(
        "위험 근접 거리 설정 (px)", 50, 500, 150, 
        help="화재와 사람 사이의 거리가 이 값보다 가까우면 위험으로 판단합니다."
    )

    result_queue = queue.Queue()

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.report_triggered = False

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated_frame, f_count, p_count, is_warning = analyze_and_draw_on_frame(img, proximity_threshold)
            if is_warning and not self.report_triggered:
                self.report_triggered = True
                result_queue.put((f_count, p_count, is_warning, annotated_frame))
            return annotated_frame

    ctx = webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if not result_queue.empty() and st.session_state.webrtc_report is None:
        with st.spinner("⚠️ 위험 상황 감지! AI 리포트를 생성합니다..."):
            f_count, p_count, is_warning, frame_for_report = result_queue.get()
            report = generate_report(f_count, p_count, is_warning, frame_for_report)
            if report:
                st.session_state.webrtc_report = report
    
    if st.session_state.webrtc_report:
        st.text_area("AI 생성 리포트", st.session_state.webrtc_report, height=400)
        if st.button("리포트 초기화"):
            st.session_state.webrtc_report = None
            st.rerun()

# --- 모드 2: 파일 업로드 및 분석 ---
elif app_mode == "파일 업로드 및 분석":
    st.header("파일 업로드 및 분석")
    uploaded_file = st.file_uploader(
        "이미지 또는 동영상 파일을 업로드하세요.",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        # --- 이미지 파일 처리 (안전 장치 추가) ---
        if file_type == "image":
            image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, 1)

            if frame is None:
                st.error("업로드한 파일을 이미지로 변환할 수 없습니다. 파일이 손상되었거나 유효하지 않은 형식입니다.")
            else:
                annotated_frame, f_count, p_count, warn = analyze_and_draw_on_frame(frame, 150)
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="분석된 이미지", use_container_width=True)
                
                st.markdown("---")
                st.subheader("📋 분석 결과")
                st.write(f"🔥 탐지된 화재: **{f_count}** 건")
                st.write(f"👨‍👩‍👧‍👦 탐지된 인원: **{p_count}** 명")
                st.warning("🚨 위험 상황(화재 근접 인원) 발생!" if warn else "✅ 위험 상황 없음.")
                
                if st.button("AI 안전 리포트 생성"):
                    with st.spinner("AI가 리포트를 작성 중입니다..."):
                        report = generate_report(f_count, p_count, warn, annotated_frame)
                        if report:
                            st.text_area("AI 생성 리포트", report, height=300)

        # --- 동영상 파일 처리 (안전 장치 추가) ---
        elif file_type == "video":
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            video_capture = cv2.VideoCapture(tfile.name)
            frame_placeholder = st.empty()
            
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            
            max_fire, max_person, is_any_warning, last_frame_for_report = 0, 0, False, None

            for i in range(total_frames):
                success, frame = video_capture.read()
                if not success:
                    break
                
                if frame is None: # 안전 장치
                    continue

                annotated_frame, f_count, p_count, warn = analyze_and_draw_on_frame(frame, 150)
                last_frame_for_report = annotated_frame
                
                max_fire = max(max_fire, f_count)
                max_person = max(max_person, p_count)
                if warn: is_any_warning = True

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, caption=f"동영상 분석 중... ({i+1}/{total_frames})", use_container_width=True)
                progress_bar.progress((i + 1) / total_frames)

            video_capture.release()
            
            st.success("동영상 분석이 완료되었습니다.")
            st.markdown("---")
            st.subheader("📋 전체 동영상 분석 요약")
            st.write(f"🔥 탐지된 최대 화재 수: **{max_fire}** 건")
            st.write(f"👨‍👩‍👧‍👦 탐지된 최대 인원 수: **{max_person}** 명")
            st.warning("🚨 위험 상황(화재 근접 인원)이 한 번 이상 발생했습니다!" if is_any_warning else "✅ 전체 영상에서 위험 상황은 감지되지 않았습니다.")

            if last_frame_for_report is not None:
                if st.button("AI 안전 리포트 생성"):
                    with st.spinner("AI가 리포트를 작성 중입니다..."):
                        report = generate_report(max_fire, max_person, is_any_warning, last_frame_for_report)
                        if report:
                            st.text_area("AI 생성 리포트", report, height=300)
