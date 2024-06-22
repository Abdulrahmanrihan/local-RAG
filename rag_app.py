import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

DATA_FOLDER = './data'

# This function is for parsing pdf data using PyPDF
def pdf_loader(data_folder=DATA_FOLDER):
    print([fn for fn in os.listdir(data_folder) if fn.endswith('.pdf')])
    loaders = [PyPDFLoader(os.path.join(data_folder, fn))
               for fn in os.listdir(data_folder) if fn.endswith('.pdf')]
    print(f'{len(loaders)} file loaded')
    return loaders

# This function does the same as the above one but this is to be used on the streamlit app
def pdf_loader_streamlit(uploaded_files):
    loaders = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(uploaded_file)
            loaders.append(loader)
    print(f'{len(loaders)} file(s) loaded')
    return loaders


# We're just building the two qa models and passing the chunk size and chunk overlap parameters to them
# Notice that we're using their default embedding models, which is generally a good practice since each model 
# has a context length limit

def build_qa_chain(openai_key, loaders, chunk_size: int = 1000, chunk_overlap: int = 50) -> RetrievalQA:
    embedding = OpenAIEmbeddings()
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    llm = OpenAI(model_name="text-davinci-003",
                temperature=0.9,
                openai_api_key=openai_key,
                max_tokens=256)
        
    # Building the vectorstore where we will store our vector embedding of our data
    # It is implemented using chromadb by default, which only supports similarity search only and not exact.
    index = VectorstoreIndexCreator(embedding=embedding, text_splitter=splitter).from_loaders(loaders)

    # The agent takes in the user’s query, embeds the query into a vector, retrieves relevant document chunks from  
    # the vector store, sends the relevant document chunks to the LLM and eventually passes the LLM completion to the user.
    return RetrievalQA.from_chain_type(llm=llm, 
                                   chain_type="stuff", 
                                   retriever=index.vectorstore.as_retriever(search_type="similarity",
                                   search_kwargs={"k": 4}),
                                   return_source_documents=True,
                                   input_key="question")

def ask_question(qa_chain, prompt):
    result = qa_chain({'question': prompt, 'include_run_info': True})
    return(result)
