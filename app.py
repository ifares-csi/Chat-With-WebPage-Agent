import streamlit as st
import validators
import requests
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

def initialize_env():
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

initialize_env()

def check_url_accessibility(url):
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text_from_url(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs if docs else "Error: No content extracted."
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_webpage(url, user_prompt):
    '''
     Steps 
        Step 1: load the page >> Docs
        Step 2: Split data>> Chunks documents
        Step 3:  text
        Step 4: Vectores>> vectore embedding >> Store in DB: FAISS
        Step 5: Query >> Vector >> Similarity Search >> FAISS
        Step 6: Create document stuff
        Step 7: Create Retrieval by Victor Store
        Step 7: Prompt >> LLM >> invoke
        Step 8: Return the repsonse
         ### Load Data--> Docs-->Divide our Docuemnts into chunks dcouments-->text-->vectors-->
                           Vector Embeddings--->Vector Store DB
    '''
    try:
        docs = extract_text_from_url(url)
        if isinstance(docs, str) and docs.startswith("Error"):
            return docs

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)

        #  Embedding & Storing >> Vector Embeddings
        embeddings = OpenAIEmbeddings()
        vectorstoredb = FAISS.from_documents(documents, embeddings)
        
        llm = ChatOpenAI(model="gpt-4o")
        
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Input--->Retriever--->vectorstoredb
        retriever = vectorstoredb.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        
        ## Get the response form the LLM
        response = retrieval_chain.invoke({"input": user_prompt})
        return response if response else "No relevant summary found."
    except Exception as e:
        return f"Error: {str(e)}"

st.set_page_config(page_title="Chat With WebPage", layout="wide")

# Center the title using HTML
st.markdown(
    """
    <h1 style='text-align: center;'>Chat With WebPage</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h5 style='text-align: center;'>Enter a webpage URL and your question to get relevant answers!</h5>
    """,
    unsafe_allow_html=True
)

# st.markdown("### Enter a webpage URL and your question to get relevant answers!")

url = st.text_input("Enter webpage URL")
prompt = st.text_area("Enter your question", "Example: Summarize this webpage in a concise manner.")


if url and not validators.url(url):
    st.error("Invalid URL. Please enter a valid webpage URL.")

if st.button("Submit"):
    if url and validators.url(url):
        if not check_url_accessibility(url):
            st.error("Error: Unable to access the webpage.")
        else:
            with st.spinner("Processing..."):
                summary = summarize_webpage(url, prompt)
            st.markdown("---")
            st.subheader("Response:")
            st.write(summary['answer'])
            st.markdown("---")
            if "hidden_object" not in st.session_state:
                st.session_state.hidden_object = {"context": summary}  # Replace with your object
    else:
        st.error("Please enter a valid URL.")