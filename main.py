import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        super().__init__()  # 부모 초기화 꼭 해주기!
        self.text = ""
        self.placeholder = placeholder

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text)


# --- 프롬프트 및 체인 정의 ---

mid_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 글쓰기 피드백 도우미입니다.
학생이 현재까지 쓴 글을 읽고, 더 나은 글을 쓸 수 있도록 쉬운 말로 피드백을 제공해주세요.
다만, 학생이 보고 베낄 수 있는 예시나 정답은 절대 제시하지 마세요.
초등학생들의 수준에 맞게 피드백을 제공해주세요.
피드백 길이는 100글자 이상 제시해주세요.
""",
        ),
        (
            "human",
            """
이름: {name},
현재까지 쓴 글: {essay}
""",
        ),
    ],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 글쓰기 피드백 도우미입니다.
학생이 쓴 글을 읽고, 채점하여 점수는 최소 0점에서 100점까지 점수를 부여해야 합니다.
학생이 더 나은 글을 쓸 수 있도록 마지막 피드백을 제공해주세요.

다음의 형식에 맞게 답변을 생생해주세요.
### 점수 
- 50점
### 점수 배점 이유
- 이유 설명하기
### 글이 더 나아지기 위한 피드백
- 피드백 적어주기
""",
        ),
        (
            "human",
            """
이름: {name},
현재까지 쓴 글: {essay}
""",
        ),
    ],
)

st.set_page_config(
    page_title="글쓰기 도우미",
    layout="wide",
)

st.title("글쓰기 도우미")

col1, col2 = st.columns(2)

with col1:
    container1 = st.container(border=True)
    mid_feedback_btn = st.button(
        label="지금까지의 글 점검하기", use_container_width=True
    )
    container2 = st.container(border=True)

    with container1:
        st.markdown("### 오늘의 주제: 나의 기분은?")
        name = st.text_input(label="자기 이름을 입력해주세요.")
        content = st.text_area(label="자신이 쓴 글을 입력해주세요.", height=200)

    with container2:
        st.markdown("#### 중간 피드백")
        # 중간 피드백 스트리밍 표시 영역
        feedback_placeholder = st.empty()

with col2:
    final_feedback_btn = st.button(
        "최종 제출하기",
        use_container_width=True,
        type="primary",
    )
    final_container = st.container(
        border=True,
        height=500,
    )
    with final_container:
        st.markdown("#### 최종 점수 및 피드백")
        # 최종 피드백 스트리밍 표시 영역
        final_feedback_placeholder = st.empty()


if "mid_llm" not in st.session_state:
    st.session_state.mid_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.5,
        streaming=True,
        callbacks=[StreamlitCallbackHandler(feedback_placeholder)],
    )

if "final_llm" not in st.session_state:
    st.session_state.final_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.5,
        streaming=True,
        callbacks=[StreamlitCallbackHandler(final_feedback_placeholder)],
    )

mid_chain = mid_prompt | st.session_state.mid_llm | StrOutputParser()
final_chain = final_prompt | st.session_state.final_llm | StrOutputParser()

if mid_feedback_btn:
    mid_chain.invoke(
        {
            "name": name,
            "essay": content,
        }
    )

if final_feedback_btn:
    final_chain.invoke(
        {
            "name": name,
            "essay": content,
        }
    )
