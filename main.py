import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# ---------------------- Load API Key ----------------------
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# ---------------------- User Credentials ----------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "test": "test123"
}

# ---------------------- Fancy UI Styling ----------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0f7fa, #f8bbd0);
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .message-bubble {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 10px 0;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    .user-bubble {
        background-color: #c5e1a5;
        text-align: right;
    }
    .bot-bubble {
        background-color: #b3e5fc;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Login Page ----------------------
def show_login():
    st.title("🔐 Login to Gemini Chatbot")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    login = st.button("Login")

    if login:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
        else:
            st.error("❌ Invalid username or password")

# ---------------------- PDF-based RAG Chatbot ----------------------
def pdf_chatbot():
    st.header("📄 PDF-Based Gemini RAG Chatbot")

    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if not api_key:
        st.error("❗ Please set GENAI_API_KEY in .env")
        return

    if pdf_file:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        loader = PyMuPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_db = FAISS.from_documents(chunks, embeddings)

        query = st.text_input("❓ Ask a question about the PDF:")
        if query:
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            related_docs = retriever.get_relevant_documents(query)

            llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", google_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=related_docs, question=query)

            st.markdown(f"<div class='message-bubble bot-bubble'>🤖 {answer}</div>", unsafe_allow_html=True)

# ---------------------- General Gemini Chatbot ----------------------
def general_chatbot():
    st.header("💬 General Gemini Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("💬 Say something:")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", google_api_key=api_key)
        response = llm.invoke(user_input)
        st.session_state.chat_history.append(("bot", response))

    for sender, msg in st.session_state.chat_history:
        bubble_class = "user-bubble" if sender == "user" else "bot-bubble"
        st.markdown(f"<div class='message-bubble {bubble_class}'>{msg}</div>", unsafe_allow_html=True)

# ---------------------- Main App ----------------------
def main():
    st.sidebar.title("🔧 Options")
    selected_mode = st.sidebar.radio("Choose Mode", ["📄 PDF Chat", "💬 General Chat"])
    logout = st.sidebar.button("🚪 Logout")

    if logout:
        st.session_state.clear()

    if selected_mode == "📄 PDF Chat":
        pdf_chatbot()
    else:
        general_chatbot()

# ---------------------- Routing ----------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main()
else:
    show_login()
