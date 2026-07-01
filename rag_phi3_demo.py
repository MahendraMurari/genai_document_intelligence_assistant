import sys
from anyio import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from pathlib import Path
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.chains.question_answering import load_qa_chain
#from langchain.prompts import PromptTemplate

def ask(question, retriever, text_generator):
    context_docs = retriever.invoke(question)

    print("\nRETRIEVED SOURCE CONTEXT:")
    for idx, doc in enumerate(context_docs[:2], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        print(f"Source {idx}: {source} | Page {page}")

    context = "\n\n".join(doc.page_content for doc in context_docs[:3])

    prompt = f"""
You are a document intelligence assistant. Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = text_generator(
        prompt,
        max_new_tokens=250,
        do_sample=False,
        return_full_text=False
    )

    return response[0]["generated_text"]

def main():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=-1,
        max_new_tokens=300
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    # Load the PDF document
    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR.parent / "RAGLama3_1204215" / "data" / "uber_2021.pdf"

    loader = PyPDFLoader(str(pdf_path), extract_images=False)
    pages = loader.load_and_split()
    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  
        chunk_overlap=20,  
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    # Store data into the database
    db = Chroma.from_documents(chunks, embeddings, persist_directory="new_db_index")
    vectordb = Chroma(persist_directory="new_db_index", embedding_function=embeddings)
    # Load the retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
   
    while True:
        user_question = input("\nUser Question: ")
        if user_question.lower() == "end":
            break
        answer = ask(user_question, retriever, pipe)
        print("Answer:---------\n", answer)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
