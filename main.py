import asyncio
import aioconsole
import aiofiles
import json
import hashlib
from pathlib import Path

import langchain_chroma

from core import chunk_documents, load_embedding_model, create_vector_db, connect_llm, search_chunks, generate_answer
from langchain.schema import Document


zettelkasten_path = "/Users/lucas/Documents/Zettelkasten"
embedding_model = "multi-qa-MiniLM-L6-cos-v1"
persist_directory = "/Users/lucas/Development/projects/python/zettelkasten-ki/db"
collection_name = "zettelkasten"
llm_modelname = "llama3.1"
hash_json_path = "zettel_hashes.json"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def calc_all_hashes(md_files):
    return {str(p): file_hash(p) for p in md_files}


def load_prev_hashes(json_path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_hashes(hash_dict, json_path):
    with open(json_path, "w") as f:
        json.dump(hash_dict, f)


async def load_markdown_files(md_files):
    documents = []
    for i, md_file in enumerate(md_files, 1):
        async with aiofiles.open(md_file, "r", encoding="utf-8") as f:
            content = await f.read()
            documents.append(Document(page_content=content, metadata={"source": str(md_file)}))
    return documents


async def main():
    md_files = list(Path(zettelkasten_path).rglob("*.md"))
    db_dir = Path(persist_directory)
    db_exists = db_dir.exists() and any(db_dir.iterdir())

    current_hashes = calc_all_hashes(md_files)
    prev_hashes = load_prev_hashes(hash_json_path)

    # Geänderte oder neue Dateien ermitteln
    new_or_changed = [p for p in md_files
                          if (str(p) not in prev_hashes) or
                          (current_hashes[str(p)] != prev_hashes.get(str(p)))]

    # Hauptlogik: Nur Chunks neu berechnen, wenn sich Notizen geändert haben oder kein DB da ist
    if not db_exists or new_or_changed:
        print("Neue oder geänderte Dateien entdeckt – Embedding & Datenbankaktualisierung...")
        if new_or_changed:
            documents = await load_markdown_files(new_or_changed)
        else:
            documents = await load_markdown_files(md_files)
        chunked_docs = await asyncio.to_thread(chunk_documents, documents)
        emb = await asyncio.to_thread(load_embedding_model, embedding_model)
        vectordb = await asyncio.to_thread(
            create_vector_db, chunked_docs, emb, persist_directory, collection_name
        )
        save_hashes(current_hashes, hash_json_path)
        print("Datenbank, Embeddings und Hashes aktualisiert.")
    else:
        print("Keine Änderungen an den Notizen – lade bestehende Datenbank...")
        emb = await asyncio.to_thread(load_embedding_model, embedding_model)
        # Laden ohne neue Chunks (Chroma lädt die alten Einträge)
        from langchain_chroma import Chroma
        vectordb = langchain_chroma.Chroma(
            persist_directory=persist_directory,
            embedding_function=emb,
            collection_name=collection_name
        )

        print("Vektordatenbank geladen.")

    llm = await asyncio.to_thread(connect_llm, llm_modelname)

    async def frage_zettelkasten(frage, k=4):
        relevante_chunks = await asyncio.to_thread(search_chunks, vectordb, frage, k)
        context = "\n---\n".join([chunk.page_content for chunk in relevante_chunks])
        return await asyncio.to_thread(generate_answer, llm, context, frage)

    while True:
        user_input = await aioconsole.ainput("Frage: ")
        if not user_input.strip():
            print("Auf Wiedersehen!")
            break
        antwort = await frage_zettelkasten(user_input)
        print(f"Antwort: {antwort}")


if __name__ == "__main__":
    asyncio.run(main())
