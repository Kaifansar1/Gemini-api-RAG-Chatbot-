import streamlit as st
import google.generativeai as genai
import PyPDF2

# Set up the Gemini API key securely from secrets.toml
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Set up Gemini model (make sure you are using correct version/model)
model = genai.GenerativeModel(model_name="gemini-pro")

# Page setup
st.set_page_config(page_title="💬 Gemini Chatbot", layout="centered")
st.title("💬 Gemini Chatbot")

# Sidebar for choosing chat mode
mode = st.sidebar.radio("🔧 Choose Mode", ["💬 General Chat", "📄 PDF Chat"])

# General Chat mode
if mode == "💬 General Chat":
    st.subheader("💬 General Chat Mode")
    user_prompt = st.text_input("Ask anything:")
    if user_prompt:
        try:
            response = model.generate_content(user_prompt)
            st.success("Gemini says:")
            st.write(response.text)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# PDF Chat mode
if mode == "📄 PDF Chat":
    st.subheader("📄 PDF Q&A with Gemini")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()
            st.success("✅ PDF uploaded and text extracted.")
        except Exception as e:
            st.error(f"❌ PDF reading error: {str(e)}")
            pdf_text = ""

        question = st.text_input("Ask a question based on the PDF:")
        if question and pdf_text:
            try:
                prompt = f"Based on this PDF content:\n\n{pdf_text}\n\nAnswer this question:\n{question}"
                response = model.generate_content(prompt)
                st.success("Gemini says:")
                st.write(response.text)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
