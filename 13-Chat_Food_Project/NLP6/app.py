import os
from langchain_openai import ChatOpenAI
from parser import Llama_document_parser
from db_creator import df_to_dbtbl
import lancedb
from langchain_google_genai import ChatGoogleGenerativeAI
from combind_graph import combined_graph_builder
from pydantic import BaseModel, Field
import chainlit as cl
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_together import ChatTogether
from IPython.display import Image, display


memory = MemorySaver()

doc_dfs = Llama_document_parser()

db = lancedb.connect(".lancedb")
tbl = df_to_dbtbl(db, doc_dfs)
import logging

with open(r'..\Codes\log.txt', 'w') as file:
    # Opening in write mode automatically clears the file
    file.write("")

# Set up logging configuration
logging.basicConfig(filename=r'..\Codes\log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create a file handler and set its level and format
file_handler = logging.FileHandler(r'..\Codes\log.txt')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Add the file handler to the logger
logger.addHandler(file_handler)

logger.info("---CHAT FOOD---")
# logger.handlers[0].flush()  # Forces flushing logs to disk
file_handler.flush()

os.environ['TOGETHER_API_KEY'] = "API KEY"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
gemini_chat = ChatOpenAI(
  api_key=os.environ["TOGETHER_API_KEY"],
  base_url="https://api.together.xyz/v1",
  model=MODEL_NAME
)

graph = combined_graph_builder(gemini_chat, tbl, logger)

@cl.on_chat_start
async def on_chat_start():
    model = graph
    cl.user_session.set("runnable", model)
    cl.Message(content="Hello! I am a chatbot. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()

    # graph.get_state(config)
    final_answer = cl.Message(content="")
    this_time_messages = []
    for msg, metadata in graph.stream({"messages": HumanMessage(content=message.content)}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        this_time_messages.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        await final_answer.stream_token(msg.content)
    cl.user_session.set("messages", this_time_messages)
    await final_answer.send()
