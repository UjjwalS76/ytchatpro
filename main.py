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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Input for YouTube URL
youtube_url = st.sidebar.text_input(
    "Enter a YouTube video URL",
    placeholder="https://www.youtube.com/watch?v=VIDEO_ID"
)

def get_google_api_key():
    """Retrieve the Google API key from Streamlit's secrets."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google API key not found in Streamlit secrets.")
        st.stop()
    return api_key

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from various URL formats."""
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

def get_transcript(video_id: str) -> str:
    """Fetch and combine transcript segments."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to get manual transcripts first, fall back to auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript()
        except:
            transcript = transcript_list.find_generated_transcript()
            
        segments = transcript.fetch()
        return " ".join(segment["text"] for segment in segments)
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {str(e)}")
        return ""

def create_conversational_chain(transcript_text: str, api_key: str):
    """Create the conversational chain with the transcript."""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.create_documents([transcript_text])

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Create vector store
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-002",
            google_api_key=api_key,
            temperature=0.7
        )

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        
        return chain
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {str(e)}")
        return None

def main():
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check the format.")
            return

        # Get transcript
        transcript_text = get_transcript(video_id)
        if not transcript_text:
            return

        # Get API key and create chain
        api_key = get_google_api_key()
        qa_chain = create_conversational_chain(transcript_text, api_key)
        if not qa_chain:
            return

        # Chat interface
        st.header("💬 Ask questions about the video:")
        user_input = st.text_input("Your question:", key="user_input")

        if st.button("Send", key="send_button") and user_input:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = qa_chain({"question": user_input})
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "answer": response["answer"]
                    })
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 📝 Chat History")
            for entry in st.session_state.chat_history:
                with st.container():
                    st.markdown(f"**You:** {entry['question']}")
                    st.markdown(f"**Bot:** {entry['answer']}")
                    st.markdown("---")
    else:
        st.info("👋 Enter a YouTube URL to start chatting about the video!")

if __name__ == "__main__":
    main()
