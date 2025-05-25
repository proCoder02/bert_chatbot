import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="QA Chatbot", layout="centered")

# Load model pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="./qa_model", tokenizer="./qa_model")

qa_pipeline = load_qa_pipeline()

st.title("📚 Ask Questions from Your Text")

context = st.text_area("📄 Enter Context (Knowledge Base Text):", height=300)

question = st.text_input("❓ Your Question")

if st.button("Get Answer") and context and question:
    with st.spinner("Searching..."):
        result = qa_pipeline(question=question, context=context)
        st.success(f"💡 Answer: {result['answer']}")
