from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document

# --- Einstellungen ---
zettelkasten_path = "/Users/lucas/Documents/Zettelkasten"  # Passe dies auf deinen Ordner an
embedding_model = "multi-qa-MiniLM-L6-cos-v1"  # Kleines, schnelles Modell

print("Zettelkasten KI wird initialisiert...")

# --- Schritt 1: Markdown-Dateien einlesen ---
print("Lade Markdown-Dateien...")
md_files = list(Path(zettelkasten_path).rglob("*.md"))
print(f"   ✓ {len(md_files)} Markdown-Dateien gefunden")

documents = []
for i, md_file in enumerate(md_files, 1):
    print(f"   Verarbeite Datei {i}/{len(md_files)}: {md_file.name}")
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": str(md_file)}
        ))

# --- Schritt 2: Chunking der Texte ---
print("Teile Texte in Chunks auf...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n## ","\n### ", "\n\n", "\n"]
)
chunked_docs = []
for doc in documents:
    chunks = text_splitter.split_documents([doc])
    chunked_docs.extend(chunks)

print(f"   ✓ {len(chunked_docs)} Text-Chunks erstellt")

# --- Schritt 3: Embedding-Modell initialisieren ---
print("Lade Embedding-Modell...")
emb = HuggingFaceEmbeddings(model_name=embedding_model)
print("   ✓ Embedding-Modell geladen")

# --- Schritt 4: Chroma-Vektor-DB aufbauen ---
print("Erstelle Vektor-Datenbank...")
vectordb = Chroma.from_documents(chunked_docs, emb, collection_name="zettelkasten")
print("   ✓ Vektor-Datenbank erstellt")

# --- Schritt 5: Ollama (Llama 3) initialisieren ---
print("Verbinde mit Ollama...")
llm = OllamaLLM(model="llama3.1")
print("   ✓ Ollama-Verbindung hergestellt")

# --- Schritt 6: Abfragefunktion ---
def frage_zettelkasten(frage: str, k=4):
    print("Suche relevante Informationen...")
    relevante_chunks = vectordb.similarity_search(frage, k=k)
    context = "\n---\n".join([chunk.page_content for chunk in relevante_chunks])
    prompt = f"""Beantworte die folgende Frage mithilfe dieses Kontexts aus meinem Zettelkasten und gebe die Namen der Dokumente als Quelle an in denen du die Antwort findest:
{context}

Frage: {frage}
Antwort:"""
    print("Generiere Antwort...")
    antwort = llm.invoke(prompt)
    return antwort

# --- Schritt 7: CLI-Interface ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Zettelkasten KI ist bereit!")
    print("Stelle jetzt deine Fragen (leere Eingabe zum Beenden)")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Frage: ")
        if not user_input.strip():
            print("Auf Wiedersehen!")
            break
        print()
        antwort = frage_zettelkasten(user_input)
        print(f"Antwort: {antwort}")
        print("-" * 50)
