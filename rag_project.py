import os
import streamlit as st
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# load api key
load_dotenv()
api_key = os.getenv("RAG_API_KEY")

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# extracting video id
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    return None

# Streamlit interface
st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")

st.title("🎥 YouTube Video Summarizer (RAG)")
st.write("Paste a YouTube link and get summary using RAG + Gemini")

video_url = st.text_input("Enter YouTube URL")
out_language = st.selectbox("Select Desirable Output Language", ["English", "Hindi", "Bhojpuri"])

if st.button("Summarize"):

    if not video_url:
        st.warning("Please enter a valid URL")
        st.stop()

    video_id = get_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    # STEP 1A: INDEXING (Document Injection)
    try:
        api = YouTubeTranscriptApi()
        # Fetch any available transcript (prioritizing English, then Hindi)
        transcript_data = api.fetch(video_id, languages=["en", "en-US", "en-IN", "en-GB", "hi", "hi-IN"])

        # Raw document (transcript)
        transcript = " ".join(chunk.text for chunk in transcript_data)

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        st.stop()

    # STEP 1B: INDEXING (Text Splitting)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Convert document → chunks
    chunks = splitter.create_documents([transcript])

    # STEP 2: embedding generating
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # STEP 3: VECTOR STORE (Storage)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # STEP 4: Retriever (Similarity Search)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}   
    )

    # STEP 5: AUGMENTATION (Context + Prompt)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        Your final answer MUST be in {language}.

        {context}

        Question: {question}
        """,
        input_variables=["context", "question", "language"]
    )

    # Format retrieved docs into context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Combine retrieval + user query
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "language": lambda x: out_language
    })

    parser = StrOutputParser()

    # STEP 6: GENERATION (Final Answer)
    main_chain = parallel_chain | prompt | llm | parser

    # Generate summary
    with st.spinner("Processing..."):
        result = main_chain.invoke("Summarize the video")

    # OUTPUT
    st.subheader("📄 Summary")
    st.write(result)