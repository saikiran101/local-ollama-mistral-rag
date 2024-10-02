# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.embeddings import OllamaEmbeddings 

# # Reload the vector database from the persistent directory and pass in the embedding model
# vector_db = Chroma(
#     persist_directory="ollama_pdf_rag/store/vector_database",
#     embedding_function=OllamaEmbeddings(model="nomic-embed-text", show_progress=True)  # Ensure to pass the same embedding model
# )


# # Load local model for LLM
# local_model = "mistral"
# llm = ChatOllama(model=local_model)
# print(llm)

# # Define query prompt template
# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# # Create retriever using MultiQueryRetriever
# retriever = MultiQueryRetriever.from_llm(
#     vector_db.as_retriever(), 
#     llm,
#     prompt=QUERY_PROMPT
# )

# # RAG (Retrieve and Generate) prompt template
# template = """Answer the question based ONLY on the following context:
# {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Define the chain for query execution
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Use a while loop to keep prompting for input until user exits
# while True:
#     question = input("Please enter your question (or type 'exit' to quit): ")
#     if question.lower() == 'exit':
#         break

#     # Invoke the chain with the user's input
#     answer = chain.invoke({"question": question})
#     print(f"Answer: {answer}")
    
# import subprocess
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.embeddings import OllamaEmbeddings

# # Check GPU availability for Ollama
# def check_ollama_gpu():
#     try:
#         result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
#         return "GPU" in result.stdout
#     except FileNotFoundError:
#         print("Ollama is not installed or not in PATH")
#         return False

# # Initialize embedding model
# embedding_model = OllamaEmbeddings(
#     model="nomic-embed-text",
#     show_progress=True
# )

# # Reload the vector database
# vector_db = Chroma(
#     persist_directory="ollama_pdf_rag/store/vector_database",
#     embedding_function=embedding_model
# )

# # Check GPU availability and select appropriate model
# if check_ollama_gpu():
#     print("GPU is available for Ollama")
#     local_model = "mistral:latest"  # or another GPU-optimized model
# else:
#     print("GPU is not available for Ollama, using CPU")
#     local_model = "mistral:latest"  # You can choose a different model for CPU if needed

# # Initialize the language model
# llm = ChatOllama(model=local_model)
# print(f"Using model: {local_model}")

# # Define query prompt template
# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# # Create retriever using MultiQueryRetriever
# retriever = MultiQueryRetriever.from_llm(
#     vector_db.as_retriever(),
#     llm,
#     prompt=QUERY_PROMPT
# )

# # RAG prompt template
# template = """Answer the question based ONLY on the following context: {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Define the chain for query execution
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Use a while loop to keep prompting for input until user exits
# while True:
#     question = input("Please enter your question (or type 'exit' to quit): ")
#     if question.lower() == 'exit':
#         break
    
#     # Invoke the chain with the user's input
#     answer = chain.invoke({"question": question})
#     print(f"Answer: {answer}")
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# URL of the website you want to load
url = "https://www.freecodecamp.org/news/the-docker-handbook/"  # Replace with your desired URL
print(f"Loading content from: {url}")

# Load the web content
loader = WebBaseLoader(url)
data = loader.load()
print(f"Loaded {len(data)} document(s)")

# Split and chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)
print(f"Split into {len(chunks)} chunks")

# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Create or load the vector store
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    collection_name="web-content-rag",
    persist_directory="ollama_web_rag/store/vector_database"
)
vector_store.persist()
print("Vector store created and persisted")

# Initialize the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize the language model
llm = ChatOllama(model="mistral:latest")
print(f"Using model: {llm.model}")

# Define an improved prompt template
template = """You are an AI assistant answering questions based on the given context. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Define the chain for query execution
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use a while loop to keep prompting for input until user exits
while True:
    question = input("Please enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    
    # Invoke the chain with the user's input
    answer = chain.invoke(question)
    print(f"Answer: {answer}")