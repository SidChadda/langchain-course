from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

from typing import List

from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent   # <-- correct import
from tavily import TavilyClient
from langchain_tavily import TavilySearch 

tavily=TavilyClient()


# @tool
# def get_weather(query: str) -> str:
#     """Return the results for the user query"""
#     print("running Tavily ")
#     return tavily.search(query)


class Source(BaseModel):
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The answer to the user query")
    sources: List[Source] = Field(default_factory=list, description="The sources used to generate the answer")



# llm = ChatOpenAI(model="gpt-5")  # <-- correct model name
llm=ChatOllama(model="qwen3.5:4b", temperature=0) 

tools = [TavilySearch(tavily=tavily)]  # <-- correct tool initialization
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)   # <-- correct call


def main():
    print("in main")
    result = agent.invoke({"messages": [HumanMessage(content="Find 5 Data Analyst jobs in Calgary?")]})
    print(result)

if __name__ == "__main__":
    main()
