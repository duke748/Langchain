from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

# Load the environment variables from .env file
load_dotenv()

# Parse the command line arguments
p = argparse.ArgumentParser(description="Code generation using OpenAI's GPT-3")
p.add_argument("--task", default="return the sum of two numbers", type=str)
p.add_argument("--language", default="python", type=str)
args = p.parse_args()

llm = OpenAI()

firstCodePrompt = PromptTemplate(
    template = "Write a {language} function that will {task}",
    input_variables=["language", "task"],
)

createUnitTestsPrompt = PromptTemplate(
    template = "Write unit tests for the following {language} function: {code}",
    input_variables=["language", "code"],
)


code_chain = LLMChain(
    llm=llm,
    prompt=firstCodePrompt,
    output_key="code",
)


result = code_chain({
    "language": args.language,
    "task": args.task,
})  
# Print Dictionary
print(result)
# Print Code
print(result["code"])