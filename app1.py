import gradio as gr
import cv2
import fire_detector
import person_detector
import google.generativeai as genai
import numpy as np
import tempfile
from PIL import Image
import os

# 모델 로딩 (기존과 동일, 캐싱 대신 앱 시작 시 로드)
def load_models():
    """AI 모델을 로드하는 함수"""
    try:
        fire_model = fire_detector.load_fire_model('fire2.pt')
        person_model = person_detector.load_person_model('yolov8n.pt')
        return fire_model, person_model
    except Exception as e:
        return None, None, f"모델 로딩 중 오류 발생: {e}"

fire_model, person_model, load_error = load_models()
if fire_model is None or person_model is None:
    raise ValueError(load_error)

# 공통 함수: 프레임 분석 및 시각화 (기존과 동일)
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

# 공통 함수: AI 리포트 생성 (기존과 동일)
def generate_report(fire_count, person_count, is_warning, image_frame, api_key):
    """탐지 결과를 바탕으로 Gemini AI 리포트를 생성합니다."""
    if not api_key:
        return "Gemini API 키가 설정되지 않았습니다. 리포트를 생성할 수 없습니다."
    try:
        genai.configure(api_key=api_key)
        pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))

        # 사용자님이 선호하셨던 상세 프롬프트
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
            "1. **[상황 맥락 파악]** 이미지 속 불이 통제된 상황(예: 드럼통 안의 모닥불, 캠프파이어)인지, 통제되지 않은 위험한 화재(예: 건물 화재, 산불)인지 먼저 판단하세요. 주변 환경과 사람들의 행동을 근거로 제시하세요.",
            "2. **[위험도 평가]** 위 맥락에 따라 위험도를 '안전', '주의', '경고', '심각' 4단계로 평가하고, 그 이유를 구체적으로 설명하세요.",
            "3. **[권장 조치]** 평가된 위험도에 맞는 현실적인 조치를 1~2가지 제안하세요. '경고' 또는 '심각' 단계일 경우, 구체적인 대피 요령을 반드시 포함시키세요.",
            "\n위 가이드라인에 따라 리포트를 생성해주세요."
        ]
        
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"AI 리포트 생성 중 오류가 발생했습니다: {e}"

# 웹캠 처리 함수
def webcam_analysis(img, proximity_threshold, api_key, generate_report_flag):
    if img is None:
        return None, "웹캠 이미지를 캡처해주세요.", ""
    
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_frame, f_count, p_count, is_warning = analyze_and_draw_on_frame(frame, proximity_threshold)
    result_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    summary = f"🔥 탐지된 화재: {f_count} 건\n👨‍👩‍👧‍👦 탐지된 인원: {p_count} 명\n{'🚨 위험 상황 발생!' if is_warning else '✅ 위험 상황 없음.'}"
    
    report = ""
    if generate_report_flag and is_warning:
        report = generate_report(f_count, p_count, is_warning, annotated_frame, api_key)
    
    return result_img, summary, report

# 파일 업로드 처리 함수 (이미지)
def image_upload_analysis(uploaded_file, api_key, generate_report_flag):
    if uploaded_file is None:
        return None, "이미지 파일을 업로드해주세요.", ""
    
    image = Image.open(uploaded_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_frame, f_count, p_count, is_warning = analyze_and_draw_on_frame(frame, 150)
    result_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    summary = f"🔥 탐지된 화재: {f_count} 건\n👨‍👩‍👧‍👦 탐지된 인원: {p_count} 명\n{'🚨 위험 상황 발생!' if is_warning else '✅ 위험 상황 없음.'}"
    
    report = ""
    if generate_report_flag:
        report = generate_report(f_count, p_count, is_warning, annotated_frame, api_key)
    
    return result_img, summary, report

# 파일 업로드 처리 함수 (비디오)
def video_upload_analysis(uploaded_file, api_key, generate_report_flag):
    if uploaded_file is None:
        return None, "비디오 파일을 업로드해주세요.", ""
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_capture = cv2.VideoCapture(tfile.name)
    
    max_fire, max_person, is_any_warning = 0, 0, False
    last_warn_frame = None
    last_frame = None
    
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        
        annotated_frame, f_count, p_count, warn = analyze_and_draw_on_frame(frame, 150)
        last_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        max_fire = max(max_fire, f_count)
        max_person = max(max_person, p_count)
        if warn:
            is_any_warning = True
            last_warn_frame = annotated_frame.copy()
    
    video_capture.release()
    os.unlink(tfile.name)
    
    summary = f"🔥 최대 화재 수: {max_fire} 건\n👨‍👩‍👧‍👦 최대 인원 수: {max_person} 명\n{'🚨 위험 상황 발생!' if is_any_warning else '✅ 위험 상황 없음.'}"
    
    report = ""
    if generate_report_flag:
        report_frame = last_warn_frame if last_warn_frame is not None else last_frame
        report = generate_report(max_fire, max_person, is_any_warning, report_frame, api_key)
    
    return last_frame, summary, report  # 마지막 프레임 반환 (전체 비디오 출력은 별도 고려 필요)

# Gradio 인터페이스
with gr.Blocks(title="AI 화재 및 인명 안전 시스템") as demo:
    gr.Markdown("# 🚨 AI 화재 및 인명 감지 시스템")
    gr.Markdown("이미지/동영상 파일을 업로드하거나 실시간 웹캠을 통해 화재 및 인명 위험을 감지하고 AI 리포트를 생성합니다.")
    
    api_key = gr.Textbox(label="Google Gemini API 키 입력", type="password", placeholder="API 키를 입력하세요.")
    
    with gr.Tabs():
        with gr.Tab("실시간 웹캠 감지"):
            webcam_input = gr.Image(source="webcam", label="웹캠 입력 (실시간 캡처)")
            proximity_threshold = gr.Slider(minimum=50, maximum=500, value=150, label="위험 근접 거리 설정 (px)")
            generate_report_checkbox = gr.Checkbox(label="위험 시 AI 리포트 생성")
            output_image = gr.Image(label="분석된 이미지")
            summary_text = gr.Textbox(label="분석 결과")
            report_text = gr.Textbox(label="AI 리포트")
            
            webcam_input.change(
                fn=webcam_analysis,
                inputs=[webcam_input, proximity_threshold, api_key, generate_report_checkbox],
                outputs=[output_image, summary_text, report_text]
            )
        
        with gr.Tab("파일 업로드 및 분석"):
            file_upload = gr.File(label="이미지 또는 동영상 업로드 (jpg, png, mp4 등)")
            generate_report_checkbox_upload = gr.Checkbox(label="AI 리포트 생성")
            output_image_upload = gr.Image(label="분석된 이미지/마지막 프레임")
            summary_text_upload = gr.Textbox(label="분석 결과")
            report_text_upload = gr.Textbox(label="AI 리포트")
            
            def process_upload(file, api_key, generate_flag):
                if file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return image_upload_analysis(file, api_key, generate_flag)
                elif file.name.lower().endswith(('.mp4', '.mov', '.avi')):
                    return video_upload_analysis(file, api_key, generate_flag)
                else:
                    return None, "지원되지 않는 파일 형식입니다.", ""
            
            file_upload.change(
                fn=process_upload,
                inputs=[file_upload, api_key, generate_report_checkbox_upload],
                outputs=[output_image_upload, summary_text_upload, report_text_upload]
            )

demo.launch()
