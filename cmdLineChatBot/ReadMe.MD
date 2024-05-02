## Simple command line based conversational GenAI bot in Python with history

The code use conversationmBufferMemory to keep a store of the conversation between the user and the bot.

It uses the chatPromptTemplate which takes a SystemPromptTemplate and HumanMessagePromptTemplate to pass the information to the LLM

![chatPromptTemplate](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/duke748/Langchain/main/cmdLineChatBot/plantUML/multiChainExample.puml)

The general sequence of operation is as follows

![Sequence Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/duke748/Langchain/main/cmdLineChatBot/plantUML/sequenceDiagram.puml)

Below shows how the chatPromptTemplate links in the entire scope.

![Sequence Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/duke748/Langchain/main/cmdLineChatBot/plantUML/ChatPromptTemplate.puml)

