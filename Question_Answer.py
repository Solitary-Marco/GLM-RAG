# from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import ZhipuAIEmbeddings 
from langchain_community.vectorstores import FAISS


def generate_response(documents, message):
     
    embeddings = ZhipuAIEmbeddings(
        model="embedding-3",
    )
    # db = Chroma(collection_name="GLM_RAG", embedding_function=embeddings, persist_directory="knowledge_db/")
    db = FAISS.from_documents(documents, embeddings) 
    # Function for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    

    # Setup LLMChain & prompts
    llm = ChatOpenAI(
        temperature=0.95,
        model="glm-4",
        openai_api_key="59b91f4e9535c6bda16ec6bbbb01e751.uQchf7BLyxHkgslV",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    prompt = ChatPromptTemplate.from_template(
        """
        现在,你是一个专业课程助理,我将给你一个关于课程的问题,你要根据课程材料和你自己的理解给我最好的答案，然后我会把它反馈给学生。
        你必须遵守以下所有规则：
        1. 答案应该与所提供的课程材料相应的知识点在长度、语调、逻辑论证等细节方面非常相似，甚至完全相同；
        2. 你可以用自己的知识和理解来补充从课程材料中得到的答案；
        3. 如果现有知识库中没有与该问题相关的答案，则需要进行回复“这个问题我答不上来，请向老师求助吧！”
        
        以下是我从学生那里收到的问题：
        {question}
        以下是您需要熟悉的课程材料：
        {context}
        
        请你给出最佳答复：
    
        """
        ) 
    
    retriever_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = retriever_chain.invoke(message)
    return response
