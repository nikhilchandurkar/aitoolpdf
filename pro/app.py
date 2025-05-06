


import os
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Create embeddings and vector store"""
    # Use the correct model format for embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def get_gemini_response(vector_store, query, api_key):
    """Get response from Gemini AI"""
    
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-8b",  
        google_api_key=api_key,
        temperature=0.9
    )
    
    # Search for relevant chunks
    docs = vector_store.similarity_search(query)
    
    # Load QA chain
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Get response
    response = chain.run(input_documents=docs, question=query)
    return response

def main():
    # Load environment variables
    load_dotenv()
    
    # Get the API key
    api_key = os.getenv("GOOGLE_API_KEY")  # Fixed: Standard environment variable name
    
    # Check for Google API key
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in a .env file.")
        st.stop()
    
    # App title and sidebar
    st.set_page_config(page_title="PDF Q&A with Gemini AI", layout="wide")
    
    # Sidebar contents
    with st.sidebar:
        st.title('💬 PDF Summarizer and Q&A App')
        st.markdown("Upload your PDF and ask questions about it!")
        add_vertical_space(2)
        st.write('Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading.')
        add_vertical_space(2)
        st.write('Made by ***Nikkii***')
    
    # Main content
    st.header("Ask About Your PDF 🤷‍♀️💬")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # PDF upload
    pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
    
    # Process PDFs button
    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store with API key
                    st.session_state.vector_store = get_vector_store(text_chunks, api_key)
                    
                    st.success("PDF processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
        else:
            st.warning("Please upload PDF files first.")
    
    # Reset chat button
    if st.button("Reset Chat"):
        # Clear chat history
        st.session_state.chat_history = []
    
    # Create a container for chat history display
    chat_container = st.container()
    
    # Display chat history in the container
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # User query with chat_input for a more conversational feel
    user_question = st.chat_input("Ask a question about your PDF...")
    
    # Process query
    if user_question and st.session_state.vector_store:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner("Thinking..."):
            # Get Gemini response with API key
            try:
                response = get_gemini_response(st.session_state.vector_store, user_question, api_key)
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Error getting response: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Update the display by showing the latest messages
        with chat_container:
            with st.chat_message("user"):
                st.write(user_question)
            with st.chat_message("assistant"):
                st.write(st.session_state.chat_history[-1]["content"])
                
    elif user_question and not st.session_state.vector_store:
        st.warning("Please upload and process a PDF first.")
        
    # Add an initial greeting if chat history is empty
    if not st.session_state.chat_history and st.session_state.vector_store:
        st.info("Your PDF is processed! Ask me anything about it.")

if __name__ == "__main__":
    main()