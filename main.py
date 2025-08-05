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
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                text-align: center;
            }
            input {
                background-color: #fffde7 !important;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<h1>🔒 Login to Gemini Chatbot</h1>', unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    login_btn = st.button("Login")

    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
        else:
            st.error("❌ Invalid username or password")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- MAIN CHATBOT ----------------------
def main_chatbot():
    st.markdown("""
        <style>
            .chat-bubble {
                padding: 10px;
                border-radius: 10px;
                margin: 10px 0;
                max-width: 80%;
            }
            .user-bubble {
                background-color: #e0f7fa;
                align-self: flex-end;
                margin-left: auto;
            }
            .bot-bubble {
                background-color: #ffe0b2;
                align-self: flex-start;
                margin-right: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("💬 Gemini RAG Chatbot")
    st.write("Ask questions from PDF or general queries.")

    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        st.error("❗ Please set your GENAI_API_KEY in Streamlit secrets or .env file")
        st.stop()

    mode = st.radio("Select Mode:", ["📄 PDF Q&A", "🌐 General Chat"])

    if mode == "📄 PDF Q&A":
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

                st.success("✅ PDF processed. Ask your questions now.")

                query = st.text_input("❓ Your question about PDF:")
                if query:
                    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                    matching_docs = retriever.get_relevant_documents(query)

                    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
                    chain = load_qa_chain(llm, chain_type="stuff")

                    response = chain.run(input_documents=matching_docs, question=query)

                    st.markdown(f'<div class="chat-bubble user-bubble">{query}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-bubble bot-bubble">🤖 {response}</div>', unsafe_allow_html=True)

    elif mode == "🌐 General Chat":
        query = st.text_input("❓ Ask me anything:")
        if query:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
            response = llm.invoke(query)

            st.markdown(f'<div class="chat-bubble user-bubble">{query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble bot-bubble">🤖 {response}</div>', unsafe_allow_html=True)

    # Logout Button
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------------- ROUTING ----------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_chatbot()
else:
    show_login()
