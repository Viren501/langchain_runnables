from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", # google/gemma-4-26B-A4B-it  zai-org/GLM-5.2  
    task="text-generation"
) 

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)
# LCEL Pipe operator
# report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)), #(condition, runnable),
    # (lambda x: len(x.split())>300, prompt2 | model | parser), #(condition, runnable),
    RunnablePassthrough()#default
)
# Change the value to 500 or something to address difference

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))
