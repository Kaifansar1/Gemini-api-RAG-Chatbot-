import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# ---------------------- SETTINGS ----------------------
load_dotenv()
st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# ---------------------- USER CREDENTIALS ----------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "test": "test123"
}

# ---------------------- LOGIN SCREEN ----------------------
def show_login():
    st.markdown("""
        <style>
            body {
                background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            }
            .login-box {
                background: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                max-width: 400px;
                margin: auto;
                text-align: center;
            }
            .login-box h1 {
                font-size: 2rem;
                margin-bottom: 1rem;
                color: #333;
            }
            .stTextInput>div>div>input {
                border-radius: 10px;
                border: 1px solid #ccc;
                padding: 10px;
                font-size: 16px;
            }
            .stButton>button {
                background: linear-gradient(to right, #ff758c, #ff7eb3);
                color: white;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<h1>🔒 Login to Gemini Chatbot</h1>', unsafe_allow_html=True)
        
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state["authenticated"] = True
                st.success("✅ Login successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- CHAT BUBBLE UI ----------------------
def show_chat_bubble(user_message, bot_response):
    st.markdown("""
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-top: 20px;
            }
            .bubble {
                padding: 12px 16px;
                border-radius: 20px;
                max-width: 80%;
                font-size: 15px;
                line-height: 1.5;
                word-wrap: break-word;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .user-bubble {
                background: linear-gradient(to right, #43cea2, #185a9d);
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 0;
            }
            .bot-bubble {
                background: linear-gradient(to right, #f7971e, #ffd200);
                color: black;
                align-self: flex-start;
                border-bottom-left-radius: 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="bubble user-bubble">{user_message}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bubble bot-bubble">{bot_response}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- MAIN CHATBOT APP ----------------------
def main_chatbot():
    st.title("💬 Gemini RAG Chatbot")
    st.write("Ask questions based on a PDF or general chat.")

    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        st.error("❗ Please set your GENAI_API_KEY in .env or Streamlit secrets")
        st.stop()

    # PDF Chat Section
    st.markdown("## 📄 PDF Question Answering")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file:
        with st.spinner("Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())

            loader = PyMuPDFLoader("temp.pdf")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            db = FAISS.from_documents(docs, embeddings)

            st.success("✅ PDF processed! Ask questions below:")

            pdf_query = st.text_input("Ask a question about the PDF")
            if pdf_query:
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                matching_docs = retriever.get_relevant_documents(pdf_query)

                llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=matching_docs, question=pdf_query)
                show_chat_bubble(pdf_query, response)

    # General Chatbot Section
    st.markdown("## 💬 General Chatbot (No PDF required)")
    general_query = st.text_input("Ask anything...")

    if general_query:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        response = llm.invoke(general_query)
        show_chat_bubble(general_query, response.content)

    # Logout Button
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------------- APP ROUTING ----------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_chatbot()
else:
    show_login()
