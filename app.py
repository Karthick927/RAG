import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate API key exists
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Please set it in your environment variables or .env file")
    st.stop()

# Set Groq API key
os.environ["GROQ_API_KEY"] = groq_api_key

st.set_page_config(page_title="RAG Q&A System", layout="wide")
st.title("📚 RAG Q&A System")
st.write("Upload a document and ask questions about its content!")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None

def create_rag_pipeline(file_path: str):
    """Create a RAG pipeline with LangChain and Groq AI"""
    
    # 1. Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create embeddings using HuggingFace (free, open-source)
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. Create the Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
    )
    
    # 5. Define custom prompt
    system_prompt = """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 6. Create RAG chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            "context": vector_store.as_retriever(search_kwargs={"k": 3}) | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, vector_store

# Sidebar for file upload
st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a text file", type=["txt"])

if uploaded_file is not None:
    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue().decode("utf-8"))
        tmp_file_path = tmp_file.name
    
    # Create RAG pipeline
    with st.spinner("Processing document..."):
        try:
            qa_chain, vector_store = create_rag_pipeline(tmp_file_path)
            st.session_state.qa_chain = qa_chain
            st.session_state.vector_store = vector_store
            st.session_state.document_name = uploaded_file.name
            st.sidebar.success(f"✅ '{uploaded_file.name}' processed successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing document: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

# Main content area
if st.session_state.qa_chain is not None:
    st.success(f"✅ Document loaded: '{st.session_state.document_name}' | You can now ask questions.")
    
    # Question input
    st.header("❓ Ask a Question")
    question = st.text_area("Enter your question:", height=100, placeholder="What would you like to know about the document?")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        submit_button = st.button("🔍 Get Answer", use_container_width=True)
    
    if submit_button and question.strip():
        with st.spinner("Finding answer..."):
            try:
                # Get answer from chain
                answer = st.session_state.qa_chain.invoke(question)
                
                # Get source documents
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                
                # Display answer
                st.header("📝 Answer")
                st.write(answer)
                
                # Display source documents
                st.header("📚 Source Documents")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Source {i}"):
                        st.write(doc.page_content)
                        
            except Exception as e:
                st.error(f"❌ Error getting answer: {str(e)}")
    elif submit_button:
        st.warning("Please enter a question.")
else:
    st.info("👈 Please upload a document first to get started!")
