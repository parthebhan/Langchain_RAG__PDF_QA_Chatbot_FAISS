import logging
import os
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load API Key from Environment
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI app
app = FastAPI()

def get_pdf_text(pdf_docs: List[UploadFile]) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks created. Check the input text.")
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise

class QueryRequest(BaseModel):
    question: str

@app.post("/process_pdfs/")
async def process_pdfs(files: List[UploadFile]):
    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return {"message": "PDFs processed and vector store created successfully"}
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing PDFs")

@app.post("/query/")
async def query(request: QueryRequest):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(request.question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": request.question}, return_only_outputs=True)
        return {"answer": response["output_text"]}
    except Exception as e:
        logger.error(f"Error querying PDFs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing your query")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
