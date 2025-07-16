from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import ZhipuAIEmbeddings 
from langchain_community.vectorstores import FAISS


def generate_result(documents, exercise, reference_answer):
    
    context = reference_answer if reference_answer is not None else documents
    
    
    # Setup LLMChain & prompts
    llm = ChatOpenAI(
        temperature=0.95,
        model="glm-4",
        openai_api_key="",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    prompt_template  = f"""
        现在你是一位专业的授课教师，我将向你提供一份学生的作业题，作业题可能会包含选择题、填空题和简答题，每种题型都会给出分值，
        你需要结合本地知识对该学生的作业进行批阅，然后计算输出学生的总得分，对于学生回答错误的题目，需要给出解析并标注原题号。

        以下是学生的作业：
        {exercise}
        以下是本地知识:
        {context}

        请完成学生作业的批阅并反馈结果:
        
        """
        
    messages=[
                {"role": "user",
                 "content": prompt_template
                }
                ],
    response = llm.generate(messages)
    return response.generations[0][0].text
 
    
    
    
    
    
    

"""
Now you are a professional teacher, I will provide you with a student's homework, the homework may include multiple choice questions, fill in the blanks and short answer questions, each type of question will give a score, you need to combine local knowledge or refer to the attached content to review the student's homework, and then calculate the total score of the student, for the student's wrong answer questions, It is necessary to analyze and mark the original question number.
    
    Here are the students' assignments:
    {exercise}
    The following is the attachment:
    {reference_answer}
    Here's the local knowledge:
    {documents}
    
    Please complete student assignments and feedback results:    
"""
