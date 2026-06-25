from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def LLM_generation_complete(llm, statement):
    """Generate a completion using the LLM"""
    try:
        response = llm.complete(statement)
        return response
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return None

def LLM_generation_chat(llm, messages):
    """Handle chat-based interactions with the LLM"""
    try:
        response = llm.chat(messages)
        return response
    except Exception as e:
        print(f"Error in chat generation: {e}")
        return None

def LLM_RAG_simple(llm, prompt, rag_input_files, topk):
    """
    Perform RAG using provided documents and HuggingFace model
    """
    try:
        # Load documents
        documents = SimpleDirectoryReader(
            input_files=rag_input_files
        ).load_data()
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Create service context with our specific LLM
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model
        )
        
        # Create index with our service context
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        
        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=topk)
        
        # Execute query
        response = query_engine.query(prompt)
        return response
        
    except Exception as e:
        print("Error in RAG processing:")
        print("******")
        print(str(e))
        print("******")
        return None