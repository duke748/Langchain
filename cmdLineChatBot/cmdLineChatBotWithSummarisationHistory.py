from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from dotenv import load_dotenv


# Load the environment variables from .env file
load_dotenv()

# Create a chatbot using OpenAI's GPT-4 model
chat = ChatOpenAI(
    # model_name="gpt-4",
)

# Create ConversationSummaryBufferMemory to store the messages. Return the messages as well.
# Use FileChatMessageHistory to store the messages in a file.
# mem = ConversationSummaryBufferMemory(
#     memory_key="messages",
#     return_messages=True,
#     # chat_memory=FileChatMessageHistory("ConversationSummaryBufferMemoryMessageHistory.json"),
#     llm=chat
# )

mem = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
)

# Define the prompt template
prompt = ChatPromptTemplate(
    input_variables=["userInput", "messages"],
    messages=[
        MessagesPlaceholder("messages"),
        HumanMessagePromptTemplate.from_template("{userInput}"), # Take the content for the human message
    ]
)

# Create a chain for the prompt template
# The chain will take in the user input and the messages from the memory and return the response from the chatbot.
#   The response will be stored in the memory.
chain = LLMChain(
    llm=chat, # Use the chatbot model
    prompt=prompt, # Use the prompt template above
    memory=mem, # Use the ConversationBufferMemory to store the messages
    verbose=True
)

# Set up a simple shell chatbot that can take in user input and respond back to the user.
while True:
    cmdLineInput = input(">> ")

    result = chain({
        "userInput": cmdLineInput
    })

    try:
        print(mem.load_memory_variables({}))
    except:
        pass
        print("No messages in memory")

    try:
        messages = mem.chat_memory.messages
        previous_summary = ""
        print("Prediction: " + mem.predict_new_summary(messages, previous_summary))
    except:
        pass
        print("Prediction failed: No messages in memory")
   
    print("BOT: " + result["text"])

    