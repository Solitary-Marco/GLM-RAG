import streamlit as st
from modules import *
from langchain_community.embeddings import ZhipuAIEmbeddings 
import os
from langchain_community.document_loaders import PyPDFLoader

your_api_key = ""
if not os.getenv("ZHIPUAI_API_KEY"):
    os.environ["ZHIPUAI_API_KEY"] = your_api_key
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
)

# 加载文档
loader = PyPDFLoader("./数据挖掘与人工智能导论.pdf")
documents = loader.load()

# 创建/载入向量数据
# vector = Chroma(collection_name="Database", embedding_function=embeddings, persist_directory="knowledge_db/")


def main():
    st.set_page_config(page_title="Your Learning Assistant", page_icon="👨‍🏫", layout="wide")
    st.title("Your Learning Assistant 👨‍🏫")
  
    # 顶部并列按钮，使用四列并添加空白列来实现居中效果
    col0, col1, col2, col3, col4 = st.columns([3, 1, 1, 1, 3])
    with col1:
        if st.button("知识问答"):
            st.session_state["nav"] = "知识问答"
    with col2:
        if st.button("练习题生成"):
            st.session_state["nav"] = "练习题生成"
    with col3:
        if st.button("作业批改"):
            st.session_state["nav"] = "作业批改"

    # 确保 nav 键在 session_state 中有一个默认值
    if "nav" not in st.session_state:
        st.session_state["nav"] = "知识问答"

    nav_selection = st.session_state["nav"]

    # 页面布局
    col1, col2 = st.columns([1, 5])

    with col1:
        st.subheader("历史记录")
        # 示例历史记录，可以根据需要替换为实际内容
        st.write("这里是历史记录内容。")
    with col2:
        if nav_selection == "知识问答":
            show_qa_module(documents)
        elif nav_selection == "练习题生成":
            show_practice_module(documents)
        elif nav_selection == "作业批改":
            show_assignment_module(documents)  
    

if __name__ == "__main__":
    main()

