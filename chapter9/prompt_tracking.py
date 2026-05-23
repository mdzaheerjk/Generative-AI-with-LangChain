"""Prompt tracking with PromptWatch.io."""
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from promptwatch import PromptWatch

from config import set_environment

set_environment()

# Setup the chain using the modern Runnables API
prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
model = ChatOpenAI()
chain = prompt_template | model | StrOutputParser()

# Track prompt execution with PromptWatch
with PromptWatch() as pw:
    result = chain.invoke({"input": "The quick brown fox jumped over"})
    print(f"Result: {result}")

print("Check PromptWatch.io dashboard for detailed trace information")

