from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

# Load the environment variables from .env file
load_dotenv()

# Parse the command line arguments
p = argparse.ArgumentParser(description="Code generation using OpenAI's GPT-3")
p.add_argument("--task", default="return the sum of two numbers", type=str)
p.add_argument("--language", default="golang", type=str)
args = p.parse_args()

llm = OpenAI()

# Define the first prompt templates
firstCodePrompt = PromptTemplate(
    template = "Write a {language} function that will {task}",
    input_variables=["language", "task"],
)

# Define the second prompt template
createUnitTestsPrompt = PromptTemplate(
    template = "Write unit tests for the following {language} function: {code}",
    input_variables=["language", "code"],
)

# Create a chain for the first prompt template
code_chain = LLMChain(
    llm=llm,
    prompt=firstCodePrompt,
    output_key="code",
)

# Create a chain for the second prompt template
createUnitTestsPromptChain = LLMChain(
    llm=llm,
    prompt=createUnitTestsPrompt,
    output_key="unit_tests",
)

# Create a sequential chain of the firstCodePrompt and createUnitTestsPrompt
chain = SequentialChain(
    chains=[code_chain, createUnitTestsPromptChain],
    input_variables=["language", "task"],
    output_variables=["code", "unit_tests"]
)

# Run the code generation chain
result = chain({
    "language": args.language,
    "task": args.task,
})  

# Print Dictionary
print(result)

# Print Code
print(result["code"])

# Print Unit Tests
print(result["unit_tests"])
