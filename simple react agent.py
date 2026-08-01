from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent   # <-- correct import


@tool
def get_weather(city: str) -> str:
    """Return the weather for a given city."""
    return f"The weather in {city} is sunny with a temperature of 25°C."


#llm = ChatOpenAI() 
llm=ChatOllama(model="qwen3.5:4b", temperature=0) 
tools = [get_weather]
agent = create_agent(model=llm, tools=tools)   # <-- correct call


def main():
    print("in main")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in New York?")]})
    print(result)

if __name__ == "__main__":
    main()
