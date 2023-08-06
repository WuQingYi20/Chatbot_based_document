import os
import openai
import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain

# Load OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_db(file, chain_type, k):
    """Load database from a PDF file."""
    loader = PyPDFLoader(file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

class ChatBot:
    """Chatbot class."""
    def __init__(self):
        self.chat_history = []
        self.qa = None

    def load_file(self, file):
        self.qa = load_db(file, "stuff", 4)

    def convchain(self, query):
        """Ask the chatbot a question."""
        if not query or not self.qa:
            return '', ''
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        answer = result['answer']
        return query, answer

    def clr_history(self):
        """Clear the chat history."""
        self.chat_history = []

cb = ChatBot()

st.title('ChatWithYourData_Bot')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    cb.load_file("temp.pdf")

query = st.text_input("Enter your question:")

if st.button('Ask'):
    user_query, bot_reply = cb.convchain(query)
    st.write(f'You: {user_query}')
    st.write(f'Bot: {bot_reply}')

if st.button('Clear History'):
    cb.clr_history()