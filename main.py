import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# -------------- ENV & PAGE CONFIG --------------
load_dotenv()
st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# -------------- MODERN STYLING --------------
st.markdown("""
    <style>
    html, body {
        background-color: #eef2f7;
    }
    .stApp {
        background: linear-gradient(to bottom right, #e3f2fd, #fce4ec);
        border-radius: 12px;
        padding: 20px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input, .stTextArea textarea {
        background-color: #ffffffdd;
        border-radius: 8px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 16px;
    }
    .stTitle {
        color: #1a237e;
    }
    </style>
""", unsafe_allow_html=True)

# -------------- LOGIN SYSTEM --------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "test": "test123"
}

def show_login():
    st.title("🔐 Login to Gemini Chatbot")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password")

# -------------- MAIN CHATBOT APP --------------
def main_chatbot():
    st.title("💬 Gemini RAG Chatbot")
    st.write("You can upload a PDF and ask questions about it **OR** just chat generally with the AI.")

    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        st.error("🚫 API key not found. Set GENAI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()

    pdf_file = st.file_uploader("📄 Upload a PDF file (optional)", type=["pdf"])

    vector_db = None

    if pdf_file:
        with st.spinner("🔍 Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())

            loader = PyMuPDFLoader("temp.pdf")
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)

            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                vector_db = FAISS.from_documents(docs, embeddings)
                st.success("✅ PDF processed successfully! Ask your question below.")
            except ImportError:
                st.warning("⚠️ FAISS library not found. PDF-based QA is disabled.")

    query = st.text_input("❓ Ask a question (from PDF or general)")

    if query:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

        if pdf_file and vector_db:
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            matching_docs = retriever.get_relevant_documents(query)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=matching_docs, question=query)
        else:
            response = llm.invoke(query)

        st.markdown("### 🤖 Answer:")
        st.success(response)

    # Logout Button
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# -------------- APP ROUTER --------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_chatbot()
else:
    show_login()
