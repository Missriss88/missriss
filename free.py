import streamlit as st
import pandas as pd
from PIL import Image
import os

# --- 페이지 1: 시작하는 말 ---
def main_page():
    st.title('시작하는 말 🎈')
    st.sidebar.title('페이지 🎈')
    
    st.image("turtleship.jpg", caption="거북선")

    # st.header 대신 st.markdown을 사용해 일반 텍스트를 표현합니다.
    st.markdown("#### 거북선은 우리나라뿐만 아니라 세계 최초의 :blue[철갑선]😄 으로 알려져 있습니다.")
    st.write("물론 정확한 사료가 남아있지 않아서 얼마든지 바뀔 수 있는 칭호입니다. 하지만 그걸 차치하고도 거북선은 임진왜란 때 가장 중요한 역할을 했던 함선임은 틀림없습니다.")

# --- 페이지 2: 역사 ---
def page2():
    st.title('역사 ❄️')
    st.sidebar.title('Side 2 ❄️')

    # st.columns를 사용해 이미지와 텍스트를 나란히 배치합니다.
    col1, col2 = st.columns([0.8, 1.2]) # 이미지와 텍스트 영역 비율 조절
    with col1:
        st.image("turtleship1.jpg", caption="거북선의 예상 이미지")
    with col2:
        st.write("""
        :blue[**거북선**]은 임진왜란 때 조선 수군이 사용한 돌격 전투선으로, 조선 수군 지휘관 이순신이 1592년 임진왜란 직전에 대대적으로 개조 및 운용한 함선입니다.
        역사 기록에 따르면, 거북선은 본래 조선 초기부터 존재했으나 임진왜란 당시 이순신의 지휘 아래 크게 활약했습니다.
        """)
    
    st.divider()

    # st.expander를 사용해 긴 내용을 깔끔하게 정리합니다.
    with st.expander("**:blue[제작 시기와 창조자]** 자세히 보기"):
        st.markdown("""
        - **문헌상 최초 기록**: 1413년(태종 13년) 『조선왕조실록』에서 태종이 거북선과 왜선이 싸우는 것을 목격했다는 기록이 있습니다.
        - **이순신의 개량**: 임진왜란 시기 이순신의 거북선은 기존의 것을 단순 복원한 것이 아닌, 혁신적인 개량의 결과물입니다.
        - **『난중일기』 기록**: 1592년 3월 27일, 거북선에서 대포를 시험했다는 기록이 있어 임진왜란 발발 이전부터 체계적으로 준비했음을 알 수 있습니다.
        """)

    with st.expander("**:blue[구조와 설계]** 자세히 보기"):
        col1, col2 = st.columns(2)
        with col1:
             st.markdown("""
            - **기본 구조**: 판옥선 위에 덮개를 씌우고, 적이 오르지 못하도록 칼과 송곳을 꽂았습니다.
            - **용머리**: 함선 머리의 용머리에서는 유황 연기를 내뿜어 적을 교란하고 위협했습니다.
            - **층별 구조**: 보통 2-3층 구조로, 1층은 창고, 2층은 노 젓는 공간과 전투 공간으로 활용되었습니다.
            - **철갑**: '철갑선'이라는 기록이 있으나, 실제 철판으로 덮었는지에 대해서는 여러 학설이 있습니다.
            """)
        with col2:
            st.image("25f706496c5f0fb12b2d1d77bfa7cee7.jpg", caption="태종 때와 이순신 때의 거북선 비교")

    with st.expander("**:blue[사용 지역과 전술적 운용]** 자세히 보기"):
        st.write("거북선은 주로 전라좌수영을 중심으로 운용되었으며, 이순신 장군이 총 5척을 직접 건조하여 운용했습니다.")
        st.image("1303590416bab23e22f84493daff50b0b72a33be.png", caption="임진왜란 당시 주요 군선 구조")

    st.divider()
    st.subheader(":blue[주요 활약 해전 및 영향]")
    st.markdown("""
    - **사천 해전 (1592.5.29)**: 거북선이 실전에 처음 투입된 해전입니다.
    - **한산도 대첩 (1592.7.8)**: 학익진 전술과 함께 왜군에 대승을 거둔 결정적인 전투입니다.
    - **부산 해전**: 일본 수군 본거지에 결정적 타격을 입혔습니다.
    - **전술적 효과**: 적진에 돌격해 진형을 파괴하는 **돌격전함**의 역할과 용머리 연기를 통한 **심리전 효과**가 뛰어났습니다.
    """)
    st.image("f99391ffb87c2b0fc3df7577db3cdb3e239f19dd.png", caption="3D 모델로 예측해본 거북선의 형태")


# --- 페이지 3: 상세내용 ---
def page3():
    st.title('상세내용 🎉')
    st.sidebar.title('Side 3 🎉')

    st.header("임진왜란 주요 해전 기록", divider="violet")

    # 현재 파이썬 파일의 위치를 기준으로 정확한 파일 경로를 생성합니다.
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, '해군전투.csv')
        df = pd.read_csv(file_path)

        # 사용자가 데이터를 필터링할 수 있는 Selectbox를 추가합니다.
        battle_list = ['전체'] + df['전투명'].unique().tolist()
        selected_battle = st.selectbox('보고 싶은 전투를 선택하세요', battle_list)

        if selected_battle == '전체':
            st.dataframe(df)
        else:
            st.dataframe(df[df['전투명'] == selected_battle])

    except FileNotFoundError:
        st.error(f"오류: '해군전투.csv' 파일을 찾을 수 없습니다. 'free.py'와 같은 폴더에 파일이 있는지 확인해주세요.")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    st.divider()

    # 긴 텍스트를 st.markdown을 사용해 하나의 문단으로 묶습니다.
    st.subheader(":blue[거북선의 역사적 의의]")
    st.markdown("""
    거북선은 조선 수군의 핵심 돌격선으로, 임진왜란 당시 이순신 장군이 개량하여 일본 수군과의 전투에서 연전연승을 거두게 한 역사적인 군함입니다. 
    판옥선을 기반으로 상체에 덮개를 씌우고 칼과 송곳을 꽂아 적의 침입을 막았으며, 강력한 화포를 탑재하여 막강한 공격력을 자랑했습니다.

    이러한 혁신을 바탕으로 조선 수군은 **임진왜란 16전 16승**이라는 대기록을 세울 수 있었고, 거북선은 일본군에게 공포의 대상이 되었습니다. 
    오늘날 거북선은 임진왜란 승리의 상징이자, 한국의 역사적 정체성과 기술적 발전을 보여주는 중요한 문화유산으로 남아있습니다.
    """)


# --- 앱 실행 로직 ---

# 딕셔너리로 페이지를 관리합니다.
page_names_to_funcs = {'시작하는 말': main_page, '역사': page2, '상세내용': page3}

# 사이드바에서 페이지를 선택합니다.
selected_page = st.sidebar.selectbox("보고 싶은 페이지를 선택하세요", page_names_to_funcs.keys())

# 선택된 페이지의 함수를 호출하여 실행합니다.
page_names_to_funcs[selected_page]()

