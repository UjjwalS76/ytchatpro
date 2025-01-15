import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# Configure Streamlit
st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("🎥 Chat with a YouTube Video's Transcript")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Input for YouTube URL
youtube_url = st.sidebar.text_input(
    "Enter a YouTube video URL (e.g. https://www.youtube.com/watch?v=VIDEO_ID)",
    value=""
)

def get_google_api_key():
    """
    Retrieve the Google API key from Streamlit's secrets.
    If not found, stop the app with an error.
    """
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("❌ Google API key not found in Streamlit secrets.")
        st.stop()
    return st.secrets["GOOGLE_API_KEY"]

def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from a given URL.
    Supports typical YouTube link formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r"youtu\.be/([^#&\?]+)",
        r"youtube\.com.*v=([^#&\?]+)",
        r"youtube\.com/embed/([^#&\?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""

def fetch_transcript_text(video_id: str) -> str:
    """
    Attempt to fetch the transcript from YouTube for the given video_id.
    We try multiple language lists to include auto-generated transcripts.
    If no transcript is found or disabled, an exception is raised.
    """
    # Try these languages first (manual captions in English or variants)
    lang_priority = ["en", "en-US", "en-GB"]
    
    # If manual transcripts fail, we'll attempt an auto-generated fallback
    # 'a' is a hack that triggers auto-generated transcripts in youtube_transcript_api.
    lang_autogen = ["a"]

    transcript_text = None
    
    # First, try priority languages:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_priority)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        pass
    except Exception as e:
        # If some other error occurs, re-raise it
        raise e

    # If that fails, attempt auto-generated transcripts
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_autogen)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text
    except Exception as e:
        # If it fails again, we pass back the original error
        raise e

def split_text_into_docs(text: str):
    """
    Split a string of text into documents using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return docs

def get_embeddings(api_key: str):
    """
    Initialize GoogleGenerativeAIEmbeddings with the given API key.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=api_key
    )

def get_llm(api_key: str):
    """
    Initialize the ChatGoogleGenerativeAI LLM (Gemini model).
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",
        api_key=api_key
    )

def create_faiss_vectorstore(docs, embedding_fn):
    """
    Create a FAISS vector store from documents (docs).
    """
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return FAISS.from_texts(texts, embedding_fn, metadatas=metadatas)

# Main logic
if youtube_url:
    # Extract the video_id
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("❌ Unable to extract a valid video ID. Please check the URL.")
        st.stop()

    # Fetch transcript
    try:
        with st.spinner("🎬 Fetching transcript..."):
            transcript_text = fetch_transcript_text(video_id)
    except TranscriptsDisabled:
        st.error("❌ Subtitles are disabled for this video. No transcript available.")
        st.stop()
    except NoTranscriptFound:
        st.error("❌ No transcript found (manual or auto-generated). This video may not support transcripts.")
        st.stop()
    except VideoUnavailable:
        st.error("❌ This video is unavailable or private. Cannot retrieve a transcript.")
        st.stop()
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        st.stop()

    # If transcript_text is still None or empty, stop
    if not transcript_text:
        st.error("❌ No transcript retrieved; please try another video.")
        st.stop()

    # Split into documents
    with st.spinner("🛠️ Splitting transcript into chunks..."):
        docs = split_text_into_docs(transcript_text)

    # Retrieve Google API key from Streamlit secrets
    google_api_key = get_google_api_key()

    # Get embeddings and vector store
    embeddings = get_embeddings(google_api_key)
    with st.spinner("📦 Creating FAISS vector store..."):
        vectorstore = create_faiss_vectorstore(docs, embeddings)

    # Initialize LLM
    llm = get_llm(google_api_key)

    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # Chat Interface
    st.header("💬 Ask questions about the video:")
    user_input = st.text_input("You:", key="input")

    if st.button("Send") and user_input.strip():
        with st.spinner("🧠 Generating response..."):
            try:
                response = qa_chain({"question": user_input})
                answer = response['answer']
                st.session_state.setdefault('questions', []).append(user_input)
                st.session_state.setdefault('responses', []).append(answer)
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Display chat history
    if 'responses' in st.session_state and st.session_state.responses:
        st.markdown("### 🗨️ Chat History")
        for i in range(len(st.session_state.responses)):
            st.markdown(f"**You:** {st.session_state.questions[i]}")
            st.markdown(f"**Bot:** {st.session_state.responses[i]}")
else:
    st.info("🔄 Please enter a valid YouTube URL to get started.")
