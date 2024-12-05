import PyPDF2
import docx2txt
import streamlit as st
from Question_Answer import generate_response
from Homework_Correction import generate_result
from Exercise_Generation import generate_exercise


def show_qa_module(documents):
    st.header("知识问答(Q&A)")
    message = st.text_area("请输入您要咨询的问题：", height=68)

    if st.button("发送 🡆"):
        if message:
            st.write("答案生成中...")
            result = generate_response(documents, message)
            st.info(result)
        else:
            st.warning("请先输入您的问题！")
    

def show_practice_module(documents):
    st.header("练习题生成(Exercise generation)")
    long_text = st.text_area("请输入您的具体需求：", height=68)
    uploaded_file = st.file_uploader("上传相关文件", type=["pdf", "docx", "txt"])

    uploaded_file_text = []
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            uploaded_file_text = uploaded_file.read().decode("utf-8")
            st.text(uploaded_file_text)
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                uploaded_file_text.append(page.extract_text())
            st.write("文件上传成功！")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            uploaded_file_text = docx2txt.process(uploaded_file)
            st.write("文件上传成功！")

    if st.button("发送 🡆"):
        if long_text:
            st.write("练习题生成中...")
            result = generate_exercise(documents, long_text, uploaded_file_text)
            st.info(result)
        else:
            st.warning("请先输入您的需求！")

def show_assignment_module(vector):
    st.header("作业批改(Homework correction)")
    
    uploaded_file = st.file_uploader("请上传作业：", type=["pdf", "docx", "txt"])
    exercise = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            exercise = uploaded_file.read().decode("utf-8")
            st.text(exercise)
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                exercise += page.extract_text()
            st.write("作业上传成功！")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            exercise = docx2txt.process(uploaded_file)
            st.write("作业上传成功！")

    uploaded_file2 = st.file_uploader("请上传参考资料/答案：", type=["pdf", "docx", "txt"])
    reference_answer = ""
    if uploaded_file2 is not None:
        if uploaded_file2.type == "text/plain":
            reference_answer = uploaded_file2.read().decode("utf-8")
            st.text(reference_answer)
        elif uploaded_file2.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file2)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                reference_answer += page.extract_text()
            st.write("参考资料/答案上传成功！")
        elif uploaded_file2.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            reference_answer = docx2txt.process(uploaded_file2)
            st.write("参考资料/答案上传成功！")
    
    if st.button("发送 🡆"):
        if uploaded_file:
            st.write("作业正在批改中...")
            result = generate_result(vector, exercise, reference_answer)
            st.info(result)
        else:
            st.warning("请先上传相关文件！")
