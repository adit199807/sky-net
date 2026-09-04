from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.environ.get('MINI_MODEL', '')

reflectionPrompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""
        You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet
        Always provide detailed recommendations, including requests for length, virality, style, etc
        """),
        MessagesPlaceholder(variable_name = 'messageHistory')   
    ]
)


generationPrompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""
        You are a twitter techie influencer assistant tasked with writing excellent twitter posts
        Generate the best twitter post possible for the user's request
        If the user provides critique, respond with a revised version of your previous attempts
        """),
        MessagesPlaceholder(variable_name = 'messageHistory')   
    ]
)

llm = ChatOpenAI(model=MODEL)
generationChain = generationPrompt | llm
reflectionChain = reflectionPrompt | llm


if __name__ == '__main__':
    print('....')