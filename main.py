import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GoogleGenerativeAI

# ---------------------- SETTINGS ----------------------
load_dotenv()

st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# Style
st.markdown("""
    <style>
    body {background-color: #f0f2f6;}
    .stTextInput>div>div>input {
        color: #333;
        background: #e0f7fa;
    }
    .stButton>button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
    }
    .stTextArea textarea {
        background-color: #fffde7;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- USER CREDENTIALS ----------------------

USER_CREDENTIALS = {
    "admin": "admin123",
    "test": "test123"
}

# ---------------------- LOGIN ----------------------

def show_login():
    st.title("🔐 Secure Login")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password")

# ---------------------- MAIN CHATBOT ----------------------

def main_chatbot():
    st.title("💬 Gemini RAG Chatbot")
    st.write("Ask questions based on uploaded PDF content.")

    # API KEY
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        st.error("❗ Please set GENAI_API_KEY in your Streamlit secrets or .env")
        st.stop()

    # Upload PDF
    pdf_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

    if pdf_file:
        with st.spinner("🔍 Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())

            loader = PyMuPDFLoader("temp.pdf")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            db = FAISS.from_documents(docs, embeddings)

            st.success("✅ PDF processed! Ask your question below:")

            query = st.text_input("❓ Ask a question")
            if query:
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                matching_docs = retriever.get_relevant_documents(query)

                llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=matching_docs, question=query)

                st.markdown(f"### 🤖 Answer:\n{response}")

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------------- ROUTER ----------------------

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_chatbot()
else:
    show_login()
