import os
import faiss
import gradio as gr
import numpy as np
import requests

from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found. Please set it in Hugging Face Space settings.")

GOOGLE_DRIVE_PDF_LINKS = [
    "https://drive.google.com/uc?id=1VcnmcluZ58HOVX9cVA4KCmpwbb9eMm5M"
]

# =========================
# INIT MODELS
# =========================
client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = None
doc_chunks = []

# =========================
# DATA PIPELINE
# =========================
def download_pdfs():
    paths = []
    for i, link in enumerate(GOOGLE_DRIVE_PDF_LINKS):
        path = f"kb_{i}.pdf"
        if not os.path.exists(path):
            response = requests.get(link)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
        paths.append(path)
    return paths


def load_text_from_pdfs(paths):
    text = ""
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def build_vector_store(chunks):
    global faiss_index, doc_chunks

    doc_chunks = chunks
    embeddings = embedder.encode(chunks)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))


# =========================
# BUILD KNOWLEDGE BASE (ON STARTUP)
# =========================
pdf_paths = download_pdfs()
kb_text = load_text_from_pdfs(pdf_paths)

if not kb_text.strip():
    raise ValueError("Knowledge base PDFs contain no extractable text.")

chunks = chunk_text(kb_text)
build_vector_store(chunks)

# =========================
# RAG CORE
# =========================
def retrieve_context(query, k=4):
    query_embedding = embedder.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding), k)

    return "\n\n".join([doc_chunks[i] for i in indices[0]])


def ask_llm(question):
    context = retrieve_context(question)

    prompt = f"""
You are a factual assistant.
Rules:
- Use ONLY the provided context.
- If the answer is not clearly present, say exactly:
  "I do not know based on the provided documents."
- Do NOT add external information.
- Be concise and structured.
Context:
{context}
Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = completion.choices[0].message.content.strip()

    return f"""
## ✅ Answer
{answer}
"""

# =========================
# GRADIO UI (HCI-ALIGNED)
# =========================
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="pink"
    ),
    css="""
    .container { max-width: 900px; margin: auto; }
    .answer-box {
        border-radius: 12px;
        padding: 20px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        line-height: 1.6;
    }
    """
) as demo:

    gr.Markdown("""
    # 📚 Knowledge Base Assistant  
    Ask questions **only** from the internal documents.
    """)

    question = gr.Textbox(
        label="Your Question",
        placeholder="e.g. What is Retrieval Augmented Generation?",
        lines=2
    )

    ask_button = gr.Button("Ask")

    answer = gr.Markdown(
        value="### Awaiting your question…",
        elem_classes="answer-box"
    )

    ask_button.click(ask_llm, question, answer)

demo.launch()
