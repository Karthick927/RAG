import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from io import BytesIO
from PIL import Image
import asyncio

# For vector database
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For Text-to-Speech
import edge_tts

# Load environment variables
load_dotenv()

# Get API keys
groq_api_key = os.getenv("GROQ_API_KEY")
ocr_api_key = os.getenv("OCR_SPACE_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found!")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

st.set_page_config(page_title="RAG Q&A System", layout="wide", initial_sidebar_state="expanded")
st.title("📚 Advanced RAG Q&A System with Groq")
st.write("Upload documents (text/images/audio) and ask questions!")

# Available voices
VOICES = {
    "1": ("en-US-JennyNeural", "🇺🇸 Jenny - US Female"),
    "2": ("en-US-GuyNeural", "🇺🇸 Guy - US Male"),
    "3": ("en-GB-SoniaNeural", "🇬🇧 Sonia - British Female"),
    "4": ("en-IE-EmilyNeural", "🇮🇪 Emily - Irish Female"),
    "5": ("en-AU-NatashaNeural", "🇦🇺 Natasha - Australian Female"),
    "6": ("en-IN-NeerjaNeural", "🇮🇳 Neerja - Indian Female"),
}

# Initialize session state
if "document_content" not in st.session_state:
    st.session_state.document_content = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "4"  # Default to Emily

# Create Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=1024,
)

# Text-to-Speech function
async def text_to_speech_async(text, voice_id):
    """Convert text to speech using edge-tts"""
    try:
        voice_code, _ = VOICES[voice_id]
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        # Generate speech
        communicate = edge_tts.Communicate(text, voice_code)
        await communicate.save(temp_path)
        
        return temp_path
    
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def text_to_speech(text, voice_id):
    """Wrapper for async TTS function"""
    return asyncio.run(text_to_speech_async(text, voice_id))

# OCR.space API function
def extract_text_from_image(image_file):
    """Extract text from image using OCR.space API"""
    try:
        url = "https://api.ocr.space/parse/image"
        
        if ocr_api_key:
            payload = {'apikey': ocr_api_key, 'language': 'eng'}
        else:
            payload = {'language': 'eng'}
        
        files = {'file': image_file}
        
        response = requests.post(url, files=files, data=payload)
        result = response.json()
        
        if result.get('IsErroredOnProcessing'):
            st.error(f"OCR Error: {result.get('ErrorMessage', 'Unknown error')}")
            return None
        
        text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
        return text
    
    except Exception as e:
        st.error(f"Error during OCR: {str(e)}")
        return None

# Audio transcription using Groq Whisper
def transcribe_audio(audio_file):
    """Transcribe audio using Groq Whisper API"""
    try:
        from groq import Groq
        
        client = Groq(api_key=groq_api_key)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        with open(tmp_path, 'rb') as audio:
            transcription = client.audio.transcriptions.create(
                file=(tmp_path, audio.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        
        os.unlink(tmp_path)
        
        return transcription
    
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

# Create vector store
def create_vectorstore(text):
    """Create vector database from text"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings
        )
        
        return vectorstore
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Define QA prompt
system_prompt = """You are a helpful assistant. Answer questions based on the provided context.

The context below may be from:
- A text document
- Transcribed audio (speech-to-text)
- Extracted text from an image (OCR)

When someone asks "what's in the audio/image/document", provide the content from the context.
If a specific question cannot be answered from the context, say you don't know.

Context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(system_prompt)
qa_chain = prompt | llm | StrOutputParser()

# Sidebar
st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["txt", "jpg", "jpeg", "png", "mp3", "wav", "m4a"]
)

# Voice selection
st.sidebar.header("🔊 Voice Settings")
voice_options = {k: v[1] for k, v in VOICES.items()}
selected_voice = st.sidebar.selectbox(
    "Choose voice:",
    options=list(voice_options.keys()),
    format_func=lambda x: voice_options[x],
    index=3  # Default to Emily (index 3 = key "4")
)
st.session_state.selected_voice = selected_voice

enable_voice = st.sidebar.checkbox("🎤 Enable voice output", value=True)

if uploaded_file is not None:
    file_type = uploaded_file.type
    st.sidebar.info(f"Processing: {uploaded_file.name}")
    
    with st.spinner("🔄 Processing file..."):
        if file_type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
        
        elif file_type.startswith("image"):
            st.sidebar.info("🖼️ Extracting text from image...")
            file_content = extract_text_from_image(uploaded_file)
            if not file_content:
                st.sidebar.error("Failed to extract text from image")
                st.stop()
        
        elif file_type.startswith("audio"):
            st.sidebar.info("🎵 Transcribing audio...")
            file_content = transcribe_audio(uploaded_file)
            if not file_content:
                st.sidebar.error("Failed to transcribe audio")
                st.stop()
        
        else:
            st.sidebar.error("Unsupported file type")
            st.stop()
        
        st.session_state.document_content = file_content
        st.session_state.document_name = uploaded_file.name
        st.session_state.chat_history = []
        
        st.sidebar.info("🧠 Creating vector database...")
        st.session_state.vectorstore = create_vectorstore(file_content)
        
        st.sidebar.success(f"✅ '{uploaded_file.name}' processed successfully!")

# Main content
if st.session_state.document_content is not None:
    st.success(f"✅ Document loaded: **{st.session_state.document_name}**")
    
    with st.expander("📄 Document Preview"):
        st.text(st.session_state.document_content[:1000] + "...")
    
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.container():
                st.write(f"**Q{i+1}:** {q}")
                st.write(f"**A{i+1}:** {a}")
                st.divider()
    
    st.header("❓ Ask a Question")
    question = st.text_area("Enter your question:", height=100)
    
    col1, col2 = st.columns([1, 1])
    
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
                if st.session_state.vectorstore:
                    docs = st.session_state.vectorstore.similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                else:
                    context = st.session_state.document_content[:4000]
                
                answer = qa_chain.invoke({
                    "context": context,
                    "question": question
                })
                
                st.session_state.chat_history.append((question, answer))
                
                st.header("📝 Answer")
                st.write(answer)
                
                # Generate and play voice
                if enable_voice:
                    with st.spinner("🎤 Generating voice..."):
                        audio_path = text_to_speech(answer, st.session_state.selected_voice)
                        
                        if audio_path and os.path.exists(audio_path):
                            st.audio(audio_path, format='audio/mp3')
                            # Clean up
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
                
                st.header("📚 Relevant Context")
                st.write(context[:500] + "...")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif submit_button:
        st.warning("⚠️ Please enter a question.")
else:
    st.info("👈 Upload a document from the sidebar to get started!")
    
    with st.expander("📖 How to use?"):
        st.write("""
        1. **Upload a file**:
           - 📄 Text files (.txt)
           - 🖼️ Images (.jpg, .png) - Text will be extracted
           - 🎵 Audio files (.mp3, .wav) - Will be transcribed
        
        2. **Choose a voice** from the sidebar (Emily - Irish is default!)
        3. **Ask questions** about the content
        4. Get **AI-powered answers** with voice output
        5. Uses **vector database** for better context retrieval
        
        **Setup:**
        - Add `GROQ_API_KEY` to your .env file
        - (Optional) Add `OCR_SPACE_API_KEY` for better OCR
        """)
