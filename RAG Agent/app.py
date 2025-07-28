import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.llms.base import LLM
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, List, Mapping, Any

# -------------------------------
# 1. Load API Key
# -------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------------
# 2. Gemini LLM Wrapper (LangChain Compatible)
# -------------------------------
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"  # ✅ Updated to Gemini 1.5 flash

    @property
    def _llm_type(self):
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return genai.GenerativeModel(self.model).generate_content(prompt).text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

# -------------------------------
# 3. Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖")
st.title("🤖 Agentic RAG Chatbot (Gemini + FAISS)")
st.write("Upload a PDF and ask questions. The agent decides whether to search the document or answer directly.")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF temporarily
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # -------------------------------
    # 4. Load and Process the PDF
    # -------------------------------
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # -------------------------------
    # 5. Create a Search Tool for Agent
    # -------------------------------
    def search_docs(query: str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    tools = [
        Tool(
            name="DocumentSearch",
            func=search_docs,
            description="Use this to search and retrieve information from the uploaded PDF."
        )
    ]

    # -------------------------------
    # 6. Initialize the Agent (with conversational memory)
    # -------------------------------
    agent = initialize_agent(
        tools,
        GeminiLLM(),
        agent="chat-conversational-react-description",
        verbose=True
    )

    st.success("✅ PDF processed successfully! You can now ask questions.")

    # -------------------------------
    # 7. Chat Interface with Proper Memory
    # -------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Take user input
    if prompt := st.chat_input("Ask your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })["output"]
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # ✅ Store as proper LangChain message types
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))

else:
    st.warning("Please upload a PDF to start chatting.")
