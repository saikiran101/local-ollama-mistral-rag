# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings 
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma

# # Path to your local PDF file
# local_path = r"C:\Users\sai51\Documents\Python Programming\SaikiranYadavNamsani.pdf"
# print(local_path)

# # Load the local PDF file
# if local_path:
#     loader = UnstructuredPDFLoader(file_path=local_path)
#     data = loader.load()
#     print(data)  # Show the loaded data for debugging
# else:
#     print("Upload a PDF file")
    
# print(data[0].page_content)

# # Split and chunk the data
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
# chunks = text_splitter.split_documents(data)

# # Add documents to vector database
# vector_db = Chroma.from_documents(
#     documents=chunks, 
#     embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
#     collection_name="local-rag",
#     persist_directory="ollama_pdf_rag/store/vector_database" 
# )
# # Call this when you're done adding to the vector database to ensure it's saved
# vector_db.persist()

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# URL of the website you want to load
url = "https://www.freecodecamp.org/news/the-docker-handbook/"  # Replace with your desired URL
print(f"Loading content from: {url}")

# Load the web content
loader = WebBaseLoader(url)
data = loader.load()
print(f"Loaded {len(data)} document(s)")

# Print the first few characters of the content for debugging
print(data[0].page_content[:500] + "...")

# Split and chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
print(f"Split into {len(chunks)} chunks")

# Add documents to vector database
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="web-content-rag",
    persist_directory="ollama_web_rag/store/vector_database" 
)

# Persist the vector database
vector_db.persist()
print("Vector database persisted successfully")
print(f"Loading content from: {url}")

# Load the web content
loader = WebBaseLoader(url)
data = loader.load()
print(f"Loaded {len(data)} document(s)")

# Print the first few characters of the content for debugging
print(data[0].page_content[:500] + "...")

# Split and chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
print(f"Split into {len(chunks)} chunks")

# Add documents to vector database
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="web-content-rag",
    persist_directory="ollama_web_rag/store/vector_database" 
)

# Persist the vector database
vector_db.persist()
print("Vector database persisted successfully")