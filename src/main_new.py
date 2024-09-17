import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS # we can use chromadb as well
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load the environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = UnstructuredPDFLoader(file_path)
    #elif file_type == "docx":
   #     loader = UnstructuredDocxLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain

st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with Doc - LLAMA 3.1")

# Initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Allow file upload for PDFs, DOCX, or TXT files
uploaded_file = st.file_uploader(label="Upload your file", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    file_type = uploaded_file.name.split(".")[-1]

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Set up vectorstore and conversation chain if not already initialized
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path, file_type))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Ask Llama...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
