import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# 🔑 Load your Gemini API key
genai.configure(api_key="YOUR_API_KEY_HERE")

# 📘 Model for chat
model = genai.GenerativeModel("gemini-pro")


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def general_chat():
    st.subheader("💬 General Chat Mode")
    user_input = st.text_input("Ask anything:")
    if user_input:
        response = model.generate_content(user_input)
        st.markdown(f"**Gemini:** {response.text}")


def pdf_chat():
    st.subheader("📄 PDF Q&A with Gemini")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        st.success("PDF uploaded and text extracted!")
        text = extract_pdf_text(uploaded_file)

        question = st.text_input("Ask a question about the PDF:")
        if question:
            prompt = f"Answer the question based on this text:\n\n{text}\n\nQuestion: {question}"
            response = model.generate_content(prompt)
            st.markdown(f"**Gemini:** {response.text}")


# 🔀 UI Logic
st.title("💬 Gemini Chatbot")
mode = st.radio("🔧 Choose Mode", ["💬 General Chat", "📄 PDF Chat"])

if mode == "💬 General Chat":
    general_chat()
else:
    pdf_chat()
