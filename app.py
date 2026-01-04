import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle

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

st.set_page_config(page_title="RAG Q&A System", layout="wide", initial_sidebar_state="expanded")
st.title("📚 RAG Q&A System with Groq")
st.write("Upload a document and ask questions about its content!")

# Initialize session state
if "document_content" not in st.session_state:
    st.session_state.document_content = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create the Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
)

# Define custom prompt
system_prompt = """You are a helpful assistant. Answer questions based on the provided document context.
If the answer is not in the document, say you don't know.

Document Content:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(system_prompt)

# Create QA chain
qa_chain = prompt | llm | StrOutputParser()

# Sidebar for file upload
st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a text file", type=["txt"])

if uploaded_file is not None:
    # Read file content
    file_content = uploaded_file.read().decode("utf-8")
    st.session_state.document_content = file_content
    st.session_state.document_name = uploaded_file.name
    st.session_state.chat_history = []  # Reset chat history
    st.sidebar.success(f"✅ '{uploaded_file.name}' loaded successfully!")

# Main content area
if st.session_state.document_content is not None:
    st.success(f"✅ Document loaded: **{st.session_state.document_name}**")
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.container():
                st.write(f"**Q{i+1}:** {q}")
                st.write(f"**A{i+1}:** {a}")
                st.divider()
    
    # Question input
    st.header("❓ Ask a Question")
    question = st.text_area("Enter your question:", height=100, placeholder="What would you like to know about the document?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        submit_button = st.button("🔍 Get Answer", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and question.strip():
        with st.spinner("🤔 Finding answer..."):
            try:
                # Prepare context (use first 4000 chars to avoid token limit)
                context = st.session_state.document_content[:4000]
                
                # Get answer from chain
                answer = qa_chain.invoke({
                    "context": context,
                    "question": question
                })
                
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                
                # Display answer
                st.header("📝 Answer")
                st.write(answer)
                
                # Display relevant excerpt
                st.header("📚 Document Excerpt")
                # Find relevant lines from document
                doc_lines = st.session_state.document_content.split('\n')
                relevant_lines = [line for line in doc_lines if any(word in line.lower() for word in question.lower().split())]
                
                if relevant_lines:
                    for line in relevant_lines[:5]:
                        st.write(f"• {line}")
                else:
                    st.write(st.session_state.document_content[:500] + "...")
                        
            except Exception as e:
                st.error(f"❌ Error getting answer: {str(e)}")
    elif submit_button:
        st.warning("⚠️ Please enter a question.")
else:
    st.info("👈 Please upload a document from the sidebar to get started!")
    
    # Display some instructions
    with st.expander("📖 How to use?"):
        st.write("""
        1. **Upload a text file** (.txt) from the sidebar
        2. **Ask questions** about the document content
        3. Get **instant answers** powered by Groq AI
        4. View **relevant excerpts** from your document
        """)
