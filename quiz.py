import streamlit as st

# 퀴즈 데이터에서 이미지 URL 부분을 제거했습니다.
quiz_data = [
    {
        "hint": "적벽대전에서 동남풍을 불게 하여 조조의 대군을 물리치는 데 결정적인 역할을 한 책사입니다.",
        "options": ["사마의", "제갈량", "주유", "방통"],
        "answer": "제갈량",
        "explanation": "제갈량은 삼국시대 촉나라의 재상으로, 뛰어난 지략과 통찰력으로 유비를 도와 촉한을 세우는 데 큰 공을 세웠습니다. '삼고초려', '칠종칠금' 등의 고사로도 유명합니다."
    },
    {
        "hint": "'무신(武神)'이라 불리며, 청룡언월도를 들고 적토마를 탄 모습으로 유명한 蜀나라의 장수입니다.",
        "options": ["장비", "조운", "마초", "관우"],
        "answer": "관우",
        "explanation": "관우는 유비, 장비와 도원결의를 맺은 형제로, 의리와 용맹의 상징입니다. 그의 붉은 얼굴과 긴 수염은 그의 트레이드마크이기도 합니다."
    },
    {
        "hint": "난세의 간웅이라 불리며, 위나라의 기틀을 닦은 강력한 군주입니다. '내가 천하를 버릴지언정, 천하가 나를 버리게 하지는 않겠다'는 명언을 남겼습니다.",
        "options": ["조조", "유비", "손권", "원소"],
        "answer": "조조",
        "explanation": "조조는 후한 말의 정치가, 군사가이자 시인으로, 북중국을 통일하고 위나라의 기초를 세웠습니다. 능력 위주의 인재 등용으로도 유명합니다."
    },
    {
        "hint": "방천화극을 휘두르며 당대 최고의 무력을 자랑했던 장수입니다. 하지만 여러 주군을 배신하여 '삼성을 섬긴 종'이라는 비판을 받기도 했습니다.",
        "options": ["손책", "여포", "전위", "감녕"],
        "answer": "여포",
        "explanation": "여포는 '인중여포, 마중적토(사람 중에는 여포, 말 중에는 적토마)'라는 말이 있을 정도로 개인의 무력이 출중했던 인물입니다. 그의 무력은 타의 추종을 불허했습니다."
    },
    {
        "hint": "강동의 호랑이라 불린 손견의 아들이자, 오나라를 건국한 손권의 형입니다. 젊은 나이에 강동을 평정하여 '소패왕'이라 불렸습니다.",
        "options": ["손책", "주유", "손견", "육손"],
        "answer": "손책",
        "explanation": "손책은 아버지 손견의 유지를 이어받아 단기간에 강동 지역을 제패하며 오나라의 기반을 마련한 뛰어난 군주입니다. 주유와는 의형제를 맺은 막역한 사이였습니다."
    },
    {
        "hint": "장판파에서 유비의 아들 유선을 품에 안고 조조의 대군을 뚫어낸, 용맹하고 충성스러운 장수입니다.",
        "options": ["황충", "위연", "조운", "강유"],
        "answer": "조운",
        "explanation": "조운(자: 자룡)은 촉의 오호대장군 중 한 명으로, 실수는 없으면서도 큰 공을 여러 번 세워 유비의 깊은 신임을 받았습니다. 백마를 탄 모습으로 자주 묘사됩니다."
    },
    {
        "hint": "한나라의 황손을 자처하며 '인(仁)'을 기치로 내걸고 촉한을 세운 초대 황제입니다.",
        "options": ["유선", "유표", "유비", "유장"],
        "answer": "유비",
        "explanation": "유비(자: 현덕)는 덕을 중시하는 군주로, 수많은 인재들이 그의 인품에 이끌려 따랐습니다. 관우, 장비와 맺은 '도원결의'는 천하에 유명합니다."
    },
    {
        "hint": "장판교 위에서 단신으로 조조의 대군을 막아선, 불같은 성격과 엄청난 용맹을 자랑한 장수입니다.",
        "options": ["장비", "허저", "하후돈", "장료"],
        "answer": "장비",
        "explanation": "장비(자: 익덕)는 유비, 관우의 의형제로, 그의 용맹은 만 명의 군사와 맞먹는다고 평가받았습니다. 장팔사모라는 긴 창을 주 무기로 사용했습니다."
    },
    {
        "hint": "형 손책의 뒤를 이어 강동을 안정적으로 다스렸으며, 붉은 수염과 푸른 눈을 가졌다고 묘사되는 오나라의 초대 황제입니다.",
        "options": ["손견", "손권", "손등", "손호"],
        "answer": "손권",
        "explanation": "손권(자: 중모)은 젊은 나이에 강동의 주인이 되었으나, 뛰어난 인재 관리와 정치력으로 오나라를 반석 위에 올려놓았습니다. 적벽대전에서 유비와 연합하여 조조를 격파했습니다."
    },
    {
        "hint": "제갈량의 숙명의 라이벌이자, 위나라의 권력을 장악하여 훗날 손자가 세울 진나라의 기틀을 마련한 인물입니다.",
        "options": ["가후", "곽가", "순욱", "사마의"],
        "answer": "사마의",
        "explanation": "사마의(자: 중달)는 뛰어난 지략과 인내심으로 제갈량의 북벌을 여러 차례 막아냈습니다. '고평릉 사변'을 통해 위나라의 실권을 장악했습니다."
    }
]

# 세션 상태 초기화
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answered = False
    st.session_state.user_answer = None

# --- 앱 UI 구성 ---

st.title("⚔️ 삼국지 인물 퀴즈 (텍스트 버전) ⚔️")
st.write("힌트를 보고 삼국지의 영웅들을 맞춰보세요!")
st.write("---")

# 퀴즈 종료 화면
if st.session_state.current_question >= len(quiz_data):
    st.header(f"🎉 퀴즈 종료! 당신의 최종 점수는: {st.session_state.score} / {len(quiz_data)}점 🎉")
    
    score_percentage = st.session_state.score / len(quiz_data)
    if score_percentage == 1.0:
        st.success("대단합니다! 당신은 진정한 삼국지 전문가입니다!")
        st.balloons()
    elif score_percentage >= 0.7:
        st.info("훌륭해요! 삼국지에 대해 많이 알고 계시는군요.")
    else:
        st.warning("조금 아쉽네요. 다시 도전해서 삼국지 영웅들과 더 친해져 보세요!")

    if st.button("다시 풀기"):
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.answered = False
        st.session_state.user_answer = None
        st.rerun()
else:
    # 퀴즈 진행 화면
    q_idx = st.session_state.current_question
    question = quiz_data[q_idx]
    
    st.subheader(f"문제 {q_idx + 1} / {len(quiz_data)}")
    # st.image(...) 라인을 삭제했습니다.
    st.info(f"**힌트:** {question['hint']}")

    if not st.session_state.answered:
        with st.form(key=f"quiz_form_{q_idx}"):
            user_answer = st.radio(
                "이 인물은 누구일까요?",
                options=question["options"],
                index=None
            )
            submit_button = st.form_submit_button("정답 확인")

        if submit_button and user_answer:
            st.session_state.answered = True
            st.session_state.user_answer = user_answer
            st.rerun()

    if st.session_state.answered:
        correct_answer = question["answer"]
        
        if st.session_state.user_answer == correct_answer:
            if 'score_counted' not in st.session_state or not st.session_state.score_counted:
                 st.session_state.score += 1
                 st.session_state.score_counted = True
            st.success("정답입니다! 🎉")
        else:
            st.error(f"오답입니다. 정답은 '{correct_answer}'입니다. 😥")
            st.session_state.score_counted = True
        
        st.markdown(f"**해설:** {question['explanation']}")

        if st.button("다음 문제로"):
            st.session_state.current_question += 1
            st.session_state.answered = False
            st.session_state.user_answer = None
            st.session_state.score_counted = False
            st.rerun()