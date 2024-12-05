from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings 
import os
from langchain_chroma import Chroma

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
# from googletrans import Translator, LANGUAGES
# from langchain_openai import OpenAIEmbeddings
# from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
# from langdetect import detect


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


your_api_key = "2078514133ddc465825d01c04baa0317.7ePDJJs5sD2CGESo"
if not os.getenv("ZHIPUAI_API_KEY"):
    os.environ["ZHIPUAI_API_KEY"] = your_api_key

# 加载文档
loader = PyPDFLoader("D:/Desktop/1. 中共二十大报告.pdf",)
documents = loader.load()


# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(documents)


# 向量化存储
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
)

# 创建/载入collection名为“Database”的向量数据库
vector = Chroma(collection_name="Database", embedding_function=embeddings, persist_directory="knowledge_db/")

retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="2078514133ddc465825d01c04baa0317.7ePDJJs5sD2CGESo",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """
            现在,你是一个专业课程助理,我将给你一个关于课程的问题,你要根据课程材料和你自己的理解给我最好的答案，然后我会把它反馈给学生。
            你必须遵守以下所有规则：
            
            1. 答案应该与所提供的课程材料相应的知识点在长度、语调、逻辑论证等细节方面非常相似，甚至完全相同；
            
            2. 你可以用自己的知识和理解来补充从课程材料中得到的答案；
            
            3. 如果现有知识库中没有与该问题相关的答案，则需要进行回复“这个问题我答不上来，请向老师求助吧！”
            
            以下是我从学生那里收到的问题：
            {{message}}
            
            以下是您需要熟悉的课程材料：
            {{retriever}}
            
            请写出我应该发给学生的最佳答复：
            
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
response = conversation.invoke({"question": "中共二十大报告主题是什么？"})
# message = "中共二十大报告主题是什么？"
# response = retriever_chain.invoke(message)
# best_practice = retrieve_info(message)
# response = chain.run(message=message, best_practice=best_practice)
print(response['text'])





