# core.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n"]
    )
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_documents([doc]))
    return chunks

def load_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_db(chunks, emb, persist_directory, collection_name):
    return Chroma.from_documents(
        chunks, emb,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

def connect_llm(model_name):
    return OllamaLLM(model=model_name)

def search_chunks(vectordb, frage, k):
    return vectordb.similarity_search(frage, k=k)

def generate_answer(llm, context, frage):
    prompt = f"""Nutze ausschließlich den nachfolgenden Kontext, um die Frage zu beantworten. 
    Gib eine *mittellange, prägnante Antwort* (idealerweise 5–7 Sätze).
    Füge die zugehörigen Dokumentnamen als Quellen in eckigen Klammern an.
    {context}

    Frage: {frage}
    Antwort:"""
    return llm.invoke(prompt)
