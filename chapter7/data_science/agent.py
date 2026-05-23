"""Agent functionality."""
import pandas as pd
from config import set_environment
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from data_science.prompts import PROMPT

set_environment()

def create_agent(csv_file: str) -> AgentExecutor:
    """
    Create data agent.

    Args:
        csv_file: The path to the CSV file.

    Returns:
        An agent executor.
    """
    llm = ChatOpenAI()
    df = pd.read_csv(csv_file)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent

def query_agent(agent: AgentExecutor, query: str) -> str:
    """Query an agent and return the response."""
    prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
    formatted_prompt = prompt.format(query=query)
    return agent.run(formatted_prompt)
