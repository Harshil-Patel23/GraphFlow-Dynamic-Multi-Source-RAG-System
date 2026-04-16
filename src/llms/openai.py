# """
# OpenAI LLM initialization and configuration.
# """

# import os

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# llm = ChatOpenAI(model="gpt-4o")

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

llm = ChatGroq(model="llama-3.3-70b-versatile")
