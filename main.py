from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
import PyPDF2
import docx
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Document Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Set up API keys and configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = "e16b0490-ed22-48e1-89a7-a634df9e6979"
INDEX_NAME = "image-qa-index"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embeddings dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

# Get the index
index = pc.Index(INDEX_NAME)

# Set up components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Text splitting configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def process_pdf(file_path: str) -> List[str]:
    """Extract text from PDF and split into chunks"""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text_splitter.split_text(text)

def process_docx(file_path: str) -> List[str]:
    """Extract text from DOCX and split into chunks"""
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text_splitter.split_text(text)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file temporarily
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file based on extension
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension == 'pdf':
            chunks = process_pdf(file_path)
        elif file_extension == 'docx':
            chunks = process_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload PDF or DOCX files only.")
        
        # Add chunks to Pinecone
        vectorstore.add_texts(chunks)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return {"message": "Document processed successfully!"}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        # Create the conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            verbose=True
        )
        
        # Get the answer to the question
        result = qa_chain({"question": question})
        
        return {"answer": result["answer"]}
    
    except Exception as e:
        return {"error": str(e)}

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
