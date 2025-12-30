from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import pinecone

#Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Document chunking strategy 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Vector store setup
pc = pinecone.Pinecone(api_key="your_key")
index = pc.Index("your-index")

vectorstore = Pinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Hybrid search with reranking
from langchain.retrievers import ContextualCompressionRetriever 
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank(model="rerank-english-v2.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k":20})
)

# Query rewriting for better retrieval
def rewrite_query(original_query: str, llm) -> str:
    prompt = f"""Rewrite this query to be more specific and include relevant keywords:
    Original: {original_query}
    Rewritten:"""
    return llm.predict(prompt)

# RAG chain with optimized prompts
llm = ChatOpenAI(model="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    chain_type_kwargs={
        "prompt":"""Use the following context to answer the question.
        If you dont know, say so. Dont hallucinate.
        
        Context: {context}
        Question: {question}
        Answer:"""
    }
)