import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import os

# ------------------ SETUP ------------------
st.set_page_config(page_title="Gemini Chatbot", layout="centered")

api_key = st.secrets["GENAI_API_KEY"] if "GENAI_API_KEY" in st.secrets else st.text_input("🔑 Enter your Gemini API Key", type="password")
if not api_key:
    st.stop()

genai.configure(api_key=api_key)

# ------------------ PDF Text Extraction ------------------
def extract_pdf_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ------------------ Chat UI Helper ------------------
def chat_bubble(sender, msg):
    bg = "#DCF8C6" if sender == "You" else "#F1F0F0"
    align = "flex-end" if sender == "You" else "flex-start"
    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align};">
            <div style="background:{bg}; padding:10px 15px; border-radius:12px; margin:5px; max-width:80%;">
                <strong>{sender}:</strong><br>{msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------ General Chat Mode ------------------
def general_chat():
    st.subheader("💬 General Chat Mode")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask anything:")
    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(user_input)
        st.session_state.chat_history.append(("Gemini", response.text))

    for sender, msg in st.session_state.chat_history:
        chat_bubble(sender, msg)

# ------------------ PDF Chat Mode ------------------
def pdf_chat():
    st.subheader("📄 PDF Q&A with Gemini")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if pdf_file:
        text = extract_pdf_text(pdf_file)
        st.success("PDF uploaded and text extracted!")

        if "pdf_history" not in st.session_state:
            st.session_state.pdf_history = []

        query = st.text_input("Ask a question about the PDF:")

        if query:
            st.session_state.pdf_history.append(("You", query))
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"""You are a helpful assistant. The following is content from a PDF:\n\n{text}\n\nQuestion: {query}"""
            response = model.generate_content(prompt)
            st.session_state.pdf_history.append(("Gemini", response.text))

        for sender, msg in st.session_state.pdf_history:
            chat_bubble(sender, msg)

# ------------------ Main UI ------------------
st.title("💬 Gemini Chatbot")

mode = st.radio("🔧 Choose Mode", ["💬 General Chat", "📄 PDF Chat"])

if mode == "💬 General Chat":
    general_chat()
else:
    pdf_chat()
