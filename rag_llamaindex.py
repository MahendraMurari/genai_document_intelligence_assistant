import sys 
import torch 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import Settings 
import json
from RAGMathFunctions import *
from typing import Sequence, List 
from llama_index.core.llms import ChatMessage 
from llama_index.core.tools import BaseTool, FunctionTool 
from llama_index.core.agent import ReActAgent
import nest_asyncio 
from llama_index.core import ( 
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage, 
) 
from llama_index.core.tools import QueryEngineTool, ToolMetadata 
import LLMTasks 
from LLMInit import initialize_LLM 
import math

# Define math functions
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b

def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def compute_circle_area(radius: float) -> float:
    """Compute the area of a circle given its radius"""
    return math.pi * radius * radius

#Ref: https://docs.llamaindex.ai 
def main(): 
    try:
        llm = initialize_LLM() # initialize Llama-3 
        print('----------------\n') 
        statement = "The fundamental idea in low rank adaptation is: " 
        response = LLMTasks.LLM_generation_complete(llm,statement) 
        print(response) # the response is reasonable 

        #-----------------Simple Text based RAG with Llama-3----------------------
        input_files=["C:/Users/mahi/Downloads/data_for_RAG_Assignment/data_for_RAG_Assignment/MSAI.txt"] # document contains info about MS in AI at UB
        topk = 2 # experiment by changing this number
        prompt = "What are the specializations offered in the MS in AI at University of Bridgeport?"
        #prompt = "How many credits are required for MS in AI at UB?"
        response = LLMTasks.LLM_RAG_simple(llm, prompt, input_files, topk)
        print('\n-----RAG Responsee based on MSAI.txt document--------\n',response)

        # Adding the new code you provided:
        prompt = "What are the specializations offered in the MS in AI at University of Bridgeport?"
        response = LLMTasks.LLM_generation_complete(llm, prompt)
        print(response) # the response is not proper as Llama does not have domain knowledge for UB
        # This can be improved by the use of RAG as you will discover later

        #---------------Having LLMs do Math based on custom RAG Math functions----
        nest_asyncio.apply()
        multiply_tool = FunctionTool.from_defaults(fn=multiply)
        add_tool = FunctionTool.from_defaults(fn=add)
        subtract_tool = FunctionTool.from_defaults(fn=subtract)
        divide_tool = FunctionTool.from_defaults(fn=divide)
        circle_area_tool = FunctionTool.from_defaults(fn=compute_circle_area)
        
        # ReAct agent with tools - doing math
        agent = ReActAgent.from_tools(
            [multiply_tool, add_tool, subtract_tool, divide_tool, circle_area_tool],
            llm=llm,
            verbose=True
        )
        response = agent.chat("What is 100*6-5+10 ?") # result should be 605
        print(str(response))
        response = agent.chat("What is the area of a circle with radius 10") # 314.15
        print(str(response))

        #------------- ReAct Agent With RAG QueryEngine Tools------------------ 
        #----using pdf documents as domain knowledge for RAG------------------- 
        print("\nLoading PDF documents...")
        lyft_docs = SimpleDirectoryReader(
            input_files=["C:/Users/mahi/Downloads/data_for_RAG_Assignment/data_for_RAG_Assignment/lyft_2021.pdf"]
        ).load_data()
        uber_docs = SimpleDirectoryReader(
            input_files=["C:/Users/mahi/Downloads/data_for_RAG_Assignment/data_for_RAG_Assignmentuber_2021.pdf"]
        ).load_data()
        ub_docs = SimpleDirectoryReader(
            input_files=["C:/Users/mahi/Downloads/data_for_RAG_Assignment/data_for_RAG_Assignment/catalog-2022-2023.pdf"]
        ).load_data()

        # create indices 
        print("Creating indices...")
        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        uber_index = VectorStoreIndex.from_documents(uber_docs)
        ub_index = VectorStoreIndex.from_documents(ub_docs)

        # create query engines 
        lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
        uber_engine = uber_index.as_query_engine(similarity_top_k=3)
        ub_engine = ub_index.as_query_engine(similarity_top_k=3)

        # create query engine tools 
        query_engine_tools = [
            QueryEngineTool(
                query_engine=lyft_engine,
                metadata=ToolMetadata(
                    name="lyft_10k",
                    description=(
                        "Provides information about Lyft financials for year 2021. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=uber_engine,
                metadata=ToolMetadata(
                    name="uber_10k",
                    description=(
                        "Provides information about Uber financials for year 2021. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=ub_engine,
                metadata=ToolMetadata(
                    name="ub_catalog",
                    description=(
                        "University of Bridgeport catalog 2022-2023. "
                        "Use a plain text question as input to the tool."
                    ),
                ),
            ),
        ]

        # Create ReAct Agent using RAG QueryEngine Tools 
        print("\nCreating and querying ReAct Agent...")
        agent = ReActAgent.from_tools(
            query_engine_tools,
            llm=llm,
            verbose=True,
        )
        
        print("\nQuerying Lyft revenue...")
        response = agent.chat("What was Lyft's revenue in 2021?")
        print(str(response))
        
        print("\nQuerying Uber revenue...")
        response = agent.chat("What was Uber's revenue in 2021?")
        print(str(response))
        
        print("\nQuerying UB Computer Vision course...")
        response = agent.chat("Can you give me brief description about Computer Vision course at University of Bridgeport?")
        print(str(response))

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print("\nPress Enter to exit...")
        input()  # This will keep the window open
        return 0

if __name__ == "__main__": 
    sys.exit(int(main() or 0))
