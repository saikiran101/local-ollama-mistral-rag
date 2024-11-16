import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_path
import io
from PIL import Image
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.evaluation import QAEvalChain

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow Vite React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your PDF document
pdf_path = r"C:\Users\sai51\Documents\Python Programming\SaikiranYadavNamsani.pdf"

# Load and process the PDF
print(f"Loading content from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# Use HuggingFaceEmbeddings with GPU support
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    collection_name="pdf-content-rag",
    persist_directory="ollama_pdf_rag/store/vector_database"
)
vector_store.persist()

# Use GPU-accelerated model if available
llm = ChatOllama(model="mistral:latest", device=device)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

template = """You are an AI assistant answering questions based on the given context. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide your reasoning and cite the relevant parts of the context.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QuestionInput(BaseModel):
    text: str

class AnswerOutput(BaseModel):
    answer: str
    confidence: float
    page_numbers: List[int]

class EvaluationExample(BaseModel):
    question: str
    answer: str

class EvaluationResult(BaseModel):
    accuracy: float
    results: List[Dict[str, str]]

@app.post("/ask", response_model=AnswerOutput)
async def ask_question(question: QuestionInput):
    try:
        answer = chain.invoke(question.text)
        
        # Simple heuristic for confidence
        confidence = min(1.0, max(0.0, len(answer) / 500))
        
        # Extract page numbers from the context
        page_numbers = list(set([doc.metadata['page'] for doc in compression_retriever.get_relevant_documents(question.text)]))
        
        return AnswerOutput(answer=answer, confidence=confidence, page_numbers=page_numbers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_page/{page_number}")
async def get_page(page_number: int):
    try:
        if page_number < 1:
            raise ValueError("Page number must be 1 or greater.")
        
        # Convert the specific page to an image
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        
        if not images:
            raise ValueError(f"Page {page_number} not found in the PDF.")
        
        img = images[0]
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return FileResponse(img_byte_arr, media_type="image/png", filename=f"page_{page_number}.png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_model(examples: List[EvaluationExample]):
    try:
        eval_chain = QAEvalChain.from_llm(llm=llm)
        
        questions = [ex.question for ex in examples]
        ground_truths = [ex.answer for ex in examples]
        
        # Generate predictions
        predictions = [chain.invoke(q) for q in questions]
        
        # Evaluate
        graded_outputs = eval_chain.evaluate(
            questions,
            predictions,
            ground_truths
        )
        
        # Calculate accuracy
        correct = sum(1 for output in graded_outputs if output['results'] == 'CORRECT')
        accuracy = correct / len(graded_outputs)
        
        return EvaluationResult(
            accuracy=accuracy,
            results=[
                {
                    "question": q,
                    "prediction": p,
                    "ground_truth": gt,
                    "evaluation": go['results']
                }
                for q, p, gt, go in zip(questions, predictions, ground_truths, graded_outputs)
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
