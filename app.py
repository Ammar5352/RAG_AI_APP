import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

st.title('RAG Document Query AI App')
with st.sidebar:
    st.header("Note")
    st.markdown("""
    This is a document query AI app. Please upload a PDF file to start.
    Once uploaded, you can enter a query, and the app will provide answers based on the document content.
    Use the **"Reset Files"** button to clear the uploaded files and start fresh.
    You can select multiples files to get context data from your files😉!
    """)
    st.header("Note")
    st.markdown("""
         Please Manually **Cancel** Uploaded File or Click on **Browse Files** to Upload New Files!       
                
                """)

uploaded_file = st.file_uploader("Please Upload a PDF File", type="pdf",accept_multiple_files=True)
dir_name = 'trial'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

llm = ChatGroq(model='llama3-8b-8192')
prompt = ChatPromptTemplate.from_template(
    """
    Answer all the following questions accurately.
    Read context properly and try to answer it"
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

import time
def delete_files_in_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            

def vector_store_query():
    with st.spinner("Preprocessing File!"):
        loader = PyPDFDirectoryLoader(dir_name)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
        text_splitter = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings()
        vectors = Chroma.from_documents(text_splitter, embeddings)
        return vectors  

if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'count' not in st.session_state:
    st.session_state.count = 0

if uploaded_file:
    st.write("Uploaded!")
    for file in uploaded_file:
        with open(os.path.join(dir_name, file.name), 'wb') as f:
            f.write(file.getbuffer())
        
    if st.session_state.count == 0 or st.session_state.vectors is None:
        st.session_state.vectors = vector_store_query()
        st.session_state.count += 1
        st.write("Vectors Embedded Successfully")

    user_prompt = st.text_input("Enter your amazing query. I will try to solve it 😉")
    if st.button("Generate Response"):
        start = time.process_time()
        if user_prompt:
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                rag_chain = create_retrieval_chain(retriever, document_chain)
                response = rag_chain.invoke({'input': user_prompt})
                st.write(response['answer'])
                st.write("Response time:", time.process_time() - start)

                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response['context']):
                        st.write(doc.page_content)
                        st.write("---------------------------------")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.button("Reset Embeddings"):
        st.session_state.vectors = None
        st.session_state.count = 0
        delete_files_in_directory(dir_name)
        st.write("Successfully deleted vectors embeddings. Create another by uploading another files!")
