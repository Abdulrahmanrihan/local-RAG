import streamlit as st
from rag_app import pdf_loader_streamlit, build_qa_chain, ask_question

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

st.title("Books Information Retrieval")
st.write("""This is a simple RAG application used to retrieve relevant information to your questions
         from your pdf books, In other words: we are trying to have you chat with your books.""")

st.write("Steps to use")

st.markdown("""
1. Enter a valid OpenAI API key in the sidebar's text input field
2. Upload your pdf files and click the "process" button
3. Type your prompt and click "generate".
Watch the magic happen!
""")

with st.form('upload_form'):
    # Uploading the pdfs part
    loaders = []
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            st.write(f"Uploaded file: {uploaded_file.name}")
            st.pdf(uploaded_file)
            
    file_upload_button = st.form_submit_button("process")
    if file_upload_button:
        if uploaded_files is not None:
            loaders = pdf_loader_streamlit(uploaded_files)
        else:
            st.warning('You have to upload pdf files first')
            
with st.form('prompt_form'):
    # Taking the prompt as a string and feeding it to the retrieval model
    text = st.text_area('Enter your prompt here')
    submitted = st.form_submit_button('Submit')
    
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-') and loader is not None:
        qa_chain = build_qa_chain(openai_api_key, loaders)
        result = (qa_chain, text)
        st.write(result['result'])
        st.write("Here is the resources I used")
        st.write(result["source_documents"])
