
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# Function: Load and Split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    return docs

# Function: Create Vector Store
def create_vector_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Function: Initialize LLM
def initialize_llm():
    return ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)

# Tool Function for RAG — expects 'context' in both prompt and function
def create_rag_tool(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for school safeguarding based on local policy."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{context}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    @tool
    def safeguarding_rag_tool(context: str) -> str:
        """Search safeguarding policy and give appropriate guidance"""
        response = retrieval_chain.invoke({"context": context, "chat_history": []})
        return response["answer"]

    return safeguarding_rag_tool

# Function: Create Agent Executor — expects 'input'
def create_agent_executor(llm, vectorstore):
    rag_tool = create_rag_tool(llm, vectorstore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for school safeguarding based on local policy."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, [rag_tool], prompt)
    return AgentExecutor(agent=agent, tools=[rag_tool], verbose=True)
