import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings

# Load API key
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")

st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# Simple login credentials
USER_CREDENTIALS = {"admin": "admin123", "test": "test123"}

def show_login():
    st.title("🔐 Login to Gemini Chatbot")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USER_CREDENTIALS and USER_CREDENTIALS[user] == pwd:
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid credentials")

def show_chat_bubble(user_msg, bot_msg):
    st.markdown(f"<div style='text-align:right; background:#c5e1a5; border-radius:8px; margin:8px 0; padding:10px;'>{user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:left; background:#b3e5fc; border-radius:8px; margin:8px 0; padding:10px;'>🤖 {bot_msg}</div>", unsafe_allow_html=True)

def pdf_chatbot():
    st.header("📄 PDF Q&A Mode")
    pdf = st.file_uploader("Upload a PDF...", type=["pdf"])
    if pdf and api_key:
        with open("temp.pdf", "wb") as f:
            f.write(pdf.read())
        docs = PyMuPDFLoader("temp.pdf").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        q = st.text_input("Ask about the PDF:")
        if q:
            retr = db.as_retriever(search_kwargs={"k": 4})
            relevant = retr.get_relevant_documents(q)
            llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", google_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            ans = chain.run(input_documents=relevant, question=q)
            show_chat_bubble(q, ans)

def general_chatbot():
    st.header("💬 General Chat Mode")
    if "history" not in st.session_state:
        st.session_state.history = []
    uq = st.text_input("Ask anything:")
    if uq:
        st.session_state.history.append(("You", uq))
        llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", google_api_key=api_key)
        resp = llm.invoke(uq)
        bot_response = resp if isinstance(resp, str) else resp.content
        st.session_state.history.append(("Bot", bot_response))
    for sender, msg in st.session_state.history:
        bgcolor = "#c5e1a5" if sender == "You" else "#b3e5fc"
        align = "right" if sender == "You" else "left"
        st.markdown(f"<div style='text-align:{align}; background:{bgcolor}; border-radius:8px; margin:8px 0; padding:10px;'>{msg}</div>", unsafe_allow_html=True)

def main():
    if not api_key:
        st.error("Set your GENAI_API_KEY in .env or Streamlit secrets")
        return
    st.sidebar.title("Select Mode")
    mode = st.sidebar.radio("", ["📄 PDF Q&A", "💬 General Chat"])
    if st.sidebar.button("Logout"):
        st.session_state.clear()
    if mode == "📄 PDF Q&A":
        pdf_chatbot()
    else:
        general_chatbot()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main()
else:
    show_login()
