from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import ZhipuAIEmbeddings 
from langchain_community.vectorstores import FAISS
from typing import List

def generate_exercise(documents, long_text, uploaded_file_text=None):


    if uploaded_file_text is not None：
        documents = uploaded_file_text
    
    
    # Setup LLMChain & prompts
    llm = ChatOpenAI(
        temperature=0.95,
        model="glm-4",
        openai_api_key="2078514133ddc465825d01c04baa0317.7ePDJJs5sD2CGESo",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    prompt_template  = f"""
        现在你是一款智能助教，你需要根据***本地知识库***和***用户具体需求***生成练习题，你需要遵循以下原则：
        1. 生成的练习题需要用阿拉伯数字标序。
        2. 选择题的选项设置要具有迷惑性，要尽可能的和正确答案贴近。
        3. 注意选择题的选项需要回车另起一行。
        3. 填空题的空缺位置需要是具体的重要知识点，不能过于简单。
        4. 简答题应该是基于具体的知识片段总结而来。

        ### 以下是本地知识库内容：
        {documents}
        ### 以下是用户的具体需求：
        {long_text}

        请输出符合用户要求的练习题：
        
        """
        
    messages=[
                {"role": "user",
                 "content": prompt_template
                }
                ],
    response = llm.generate(messages)
    return response.generations[0][0].text


"""
You are an intelligent teaching assistant, now you need to generate exercises based on the content of the attachments provided by the user (if the user has not uploaded the attachment, then use the local knowledge base) and specific needs, you need to follow the following principles:
    1. The generated exercises need to be ordered with Arabic numerals.
    2. The option Settings of multiple choice questions should be confusing and close to the correct answer as much as possible.
    3. Note that the multiple choice question requires carriage return to start a new line.
    3. The vacant position of the blank filling question needs to be a specific and important knowledge point, and can not be too simple.
    4. Short answer questions should be based on specific pieces of knowledge.    
    
    Here are the contents of the local knowledge base:
    {documents}
    The following is the attachment content provided by the user:
    {uploaded_file}
    The following are the specific needs of users:
    {long_text}
    
    Please output exercises that meet user requirements:    
"""

