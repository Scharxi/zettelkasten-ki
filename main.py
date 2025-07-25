# async_app.py

import asyncio
import aioconsole
import aiofiles
from pathlib import Path
from core import (chunk_documents, load_embedding_model, create_vector_db,
                  connect_llm, search_chunks, generate_answer)
from langchain.schema import Document

zettelkasten_path = "/Users/lucas/Documents/Zettelkasten"
embedding_model = "multi-qa-MiniLM-L6-cos-v1"
persist_directory = "/Users/lucas/Development/projects/python/zettelkasten-ki/db"
collection_name = "zettelkasten"
llm_modelname = "llama3.1"

async def load_markdown_files(md_files):
    documents = []
    for i, md_file in enumerate(md_files, 1):
        async with aiofiles.open(md_file, "r", encoding="utf-8") as f:
            content = await f.read()
            documents.append(Document(page_content=content, metadata={"source": str(md_file)}))
    return documents

async def main():
    md_files = list(Path(zettelkasten_path).rglob("*.md"))
    documents = await load_markdown_files(md_files)
    chunked_docs = await asyncio.to_thread(chunk_documents, documents)
    emb = await asyncio.to_thread(load_embedding_model, embedding_model)
    vectordb = await asyncio.to_thread(create_vector_db, chunked_docs, emb, persist_directory, collection_name)
    llm = await asyncio.to_thread(connect_llm, llm_modelname)

    async def frage_zettelkasten(frage, k=4):
        relevante_chunks = await asyncio.to_thread(search_chunks, vectordb, frage, k)
        context = "\n---\n".join([chunk.page_content for chunk in relevante_chunks])
        return await asyncio.to_thread(generate_answer, llm, context, frage)

    while True:
        user_input = await aioconsole.ainput("Frage: ")
        if not user_input.strip():
            break
        antwort = await frage_zettelkasten(user_input)
        print(f"Antwort: {antwort}")

if __name__ == "__main__":
    asyncio.run(main())
