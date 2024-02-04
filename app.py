import re
import time
from io import BytesIO
from typing import Any, Dict, List
import os
from dotenv import load_dotenv

import openai
import streamlit as st
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
import pinecone

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from pypdf import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough



load_dotenv('.env')

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  



# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output



# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks



# Define a function for the embeddings
@st.cache_resource
def test_embed(_txt_docs):
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Indexing
    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  
        environment=os.getenv("PINECONE_ENV"),  
    )

    index_name = "test-index"
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        # index = FAISS.from_documents(_txt_docs, embeddings)
        # First, check if our index already exists. If it doesn't, we create it
        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(name=index_name, metric="dotproduct", dimension=384)
        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
        index = Pinecone.from_documents(_txt_docs, embeddings, index_name=index_name)
    st.success("Embeddings done.", icon="✅")
    return index


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the Streamlit app

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    2. Add your openAI API key
    2. Perform Q&A

    """
)



# Allow the user to input the OpenAI API key
user_input_api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"], accept_multiple_files = True)

if uploaded_file and user_input_api_key:
    
    openai.api_key = user_input_api_key

    text_docs = []
    for file in uploaded_file:
        name_of_file = file.name
        doc = parse_pdf(file) # returns list of texts 
        pages = text_to_docs(doc)
        text_docs.extend(pages)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Test the embeddings and save the index in a vector database
    index = test_embed(text_docs)
    
    
    # Set up the question-answering system
    
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": index.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history"
        )

    # Allow the user to enter a query and generate a response
    query = st.text_input(
        "**What's on your mind?**",
        placeholder="Ask me anything from the files",
    )

    if query:
        with st.spinner(
            "Generating Answer to your Query : `{}` ".format(query)
        ):
            # res = agent_chain.run(query)
            res = rag_chain.invoke(query)
            st.info(res, icon="🤖")

    # Allow the user to view the conversation history and other information stored in the agent's memory
    with st.expander("History/Memory"):
        st.session_state.memory
