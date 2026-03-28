import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# Loading environment variables 
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to your .env file.")
    st.stop()

# Reads all uploaded PDF files and extract plain text

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            # page.extract_text() can sometimes return None
            page_text = page.extract_text() or ""
            text += page_text
    return text

# Split large text into smaller chunks:  This helps embeddings work better
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Converts text chunks into embeddings and save FAISS
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Building Gemini answer chain, prompt | model is the modern LangChain style
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You answer questions using only the provided context. "
            "If the answer is not in the context, say: "
            "\"answer is not available in the context\"."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ])
    # This creates a runnable chain
    chain = prompt | model
    return chain

# Asking a question using the saved FAISS index
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    context = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke({
        "context": context,
        "question": user_question
    })

    st.write("Reply:", response.content)

# Streamlit ui

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with Different PDFs")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click Submit & Process",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")


if __name__ == "__main__":
    main()