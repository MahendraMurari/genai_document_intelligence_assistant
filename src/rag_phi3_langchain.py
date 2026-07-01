import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def ask(question, retriever, chain):
    context = retriever.invoke(question)
    print(context)
    answer = chain.invoke({"input_documents": context, "question": question}, return_only_outputs=True)['output_text']
    return answer

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
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=0,
        max_new_tokens=300
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    # Load the PDF document
    link_to_pdf = "https://www.bridgeport.edu/files/docs/academics/catalogs/catalog-2022-2023.pdf"
    loader = PyPDFLoader(link_to_pdf, extract_images=False)
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
    # Define the prompt template
    qna_prompt_template = """<|system|>
You have been provided with the context and a question, try to find out the answer to
the question only using the context information. If the answer to the question is not
found within the context, return "I dont know" as the response.<|end|>
<|user|>
Context:
{context}
Question: {question}<|end|>
<|assistant|>"""
    PROMPT = PromptTemplate(
        template=qna_prompt_template, input_variables=["context", "question"]
    )
    # Define the QNA chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    while True:
        user_question = input("\nUser Question: ")
        if user_question.lower() == "end":
            break
        answer = ask(user_question, retriever, chain)
        print("Answer:---------\n", answer)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
