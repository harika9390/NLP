import streamlit as st
from ext import extractive_summarize
from abs import summarize_text
from PyPDF2 import PdfReader
import io

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Text Summarization App")
    st.sidebar.title("Options")

    input_option = st.sidebar.radio("Choose input type:", ("Enter text", "Upload file"))
    summarization_type = st.sidebar.radio("Choose summarization type:", ("Extractive", "Abstractive"))
    num_sentences = st.sidebar.number_input("Number of sentences for summarization:", min_value=1, value=4)

    text = ""
    if input_option == "Enter text":
        text = st.text_area("Enter text:")
    elif input_option == "Upload file":
        uploaded_file = st.file_uploader("Upload a text file or PDF", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                text = read_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")

    if st.button("Summarize"):
        if text:
            try:
                if summarization_type == "Extractive":
                    summary = extractive_summarize(text, num_sentences=num_sentences)
                elif summarization_type == "Abstractive":
                    summary = summarize_text(text, num_sentences=num_sentences)

               
                st.write("**Original Text:**")
                st.write(text)

               
                st.write("**Summary:**")
                st.write(summary)

            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.write("Please provide text or upload a file.")

if __name__ == "__main__":
    main()
