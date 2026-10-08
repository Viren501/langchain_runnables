from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.runnables.base import RunnableLambda

load_dotenv()

# def word_counter(text):
#     return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke('Hi there how are you'))

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", # google/gemma-4-26B-A4B-it  zai-org/GLM-5.2  
    task="text-generation"
) 

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

# OR
# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(lambda x: len(x.split()))
# })

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# print(final_chain.invoke({'topic': 'AI'}))

result = final_chain.invoke({'topic': 'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
