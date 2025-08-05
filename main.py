import os
import streamlit as st
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# ------------------ Load API Key ------------------ #
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------ User Authentication ------------------ #
USERS = {
    "admin": "admin123",
    "guest": "guest123"
}

def authenticate(username, password):
    return USERS.get(username) == password

# ------------------ Login UI ------------------ #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Gemini Chatbot Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.stop()

# ------------------ Custom CSS ------------------ #
st.markdown("""
<style>
body, html {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: linear-gradient(to right, #eef2ff, #fef9f9);
    padding: 10px;
}
h1, h2 {
    text-align: center;
    color: #1e3a8a;
    font-weight: bold;
}
.chat-container {
    max-height: 75vh;
    overflow-y: auto;
    padding: 15px;
    border-radius: 12px;
    background-color: #ffffffdd;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.chat-bubble {
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px;
    display: flex;
    align-items: flex-end;
    max-width: 85%;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.user-bubble {
    background-color: #dcfce7;
    color: #065f46;
    margin-left: auto;
    justify-content: flex-end;
    border-top-right-radius: 0;
}
.bot-bubble {
    background-color: #e0e7ff;
    color: #1e3a8a;
    margin-right: auto;
    justify-content: flex-start;
    border-top-left-radius: 0;
}
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    margin: 0 10px;
}
.user-avatar {
    background: url('https://i.imgur.com/QnQ3fYQ.png') no-repeat center center / cover;
}
.bot-avatar {
    background: url('https://i.imgur.com/N7yXh8C.png') no-repeat center center / cover;
}
@media screen and (max-width: 768px) {
    .chat-bubble {
        font-size: 15px;
        max-width: 95%;
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------ Initialize Chat History ------------------ #
if "chat" not in st.session_state:
    model = genai.GenerativeModel("gemini-pro")
    st.session_state.chat = model.start_chat(history=[])
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------ App UI ------------------ #
st.title("🧠 Gemini Chatbot with RAG + Modern UI")

# Document uploader
with st.expander("📄 Upload .txt or .pdf Files for RAG"):
    uploaded_files = st.file_uploader("Upload text or PDF files", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        documents = []
        os.makedirs("temp_docs", exist_ok=True)
        for file in uploaded_files:
            path = os.path.join("temp_docs", file.name)
            with open(path, "wb") as f:
                f.write(file.read())

            if file.name.endswith(".txt"):
                loader = TextLoader(path)
            elif file.name.endswith(".pdf"):
                loader = PyMuPDFLoader(path)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            documents.extend(loader.load())

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        st.success("✅ Files embedded successfully!")

# ------------------ Display Chat ------------------ #
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.chat.history:
    if msg.role == "user":
        st.markdown(f"""
            <div class="chat-bubble user-bubble">
                <div class="avatar user-avatar"></div>
                <div>{msg.parts[0].text}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-bubble bot-bubble">
                <div class="avatar bot-avatar"></div>
                <div>{msg.parts[0].text}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ User Input ------------------ #
query = st.chat_input("Ask your question here...")

if query:
    st.session_state.chat.send_message(query)
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(query, k=3)
        context = "\\n".join([doc.page_content for doc in docs])
        prompt = f"Answer using context:\n{context}\n\nQuestion: {query}"
        response = st.session_state.chat.send_message(prompt)
    else:
        response = st.session_state.chat.send_message(query)

    st.experimental_rerun()
