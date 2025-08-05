import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader

# Load API key securely
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please add your Google API Key in Streamlit Secrets!")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="💬 Gemini Chatbot", layout="wide")

# Sidebar: Mode selection
st.sidebar.title("🔧 Choose Mode")
mode = st.sidebar.radio("", ["💬 General Chat", "📄 PDF Chat"])

# ────────────────────────────────
# 💬 General Chat Mode
# ────────────────────────────────
if mode == "💬 General Chat":
    st.title("💬 Gemini Chatbot")
    st.markdown("Ask anything:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("You:", key="general_input")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))

        try:
            response = model.generate_content(user_query)
            st.session_state.chat_history.append(("bot", response.text))
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Gemini:** {msg}")

# ────────────────────────────────
# 📄 PDF Q&A Mode
# ────────────────────────────────
elif mode == "📄 PDF Chat":
    st.title("📄 PDF Q&A with Gemini")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        st.success("✅ PDF uploaded and text extracted!")

        user_q = st.text_input("Ask a question about the PDF:")

        if user_q:
            prompt = f"Answer the question based on the PDF content:\n\n{text}\n\nQuestion: {user_q}"
            try:
                response = model.generate_content(prompt)
                st.markdown(f"**🤖 Answer:** {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
