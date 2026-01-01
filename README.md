```markdown
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=300&section=header&text=Local%20RAG%20Offline%20Analyzer&fontSize=70&fontColor=white&animation=twinkling" />
</p>

# 🔒 Local RAG (Offline Document Analyzer)

A **fully offline** AI-powered application for analyzing documents (PDF, DOCX, DOC) right on your machine.  
Powered by **Ollama** and **RAG (Retrieval-Augmented Generation)**, it lets you chat with your private documents — with **zero data leaving your computer** for complete privacy!

Ask questions, get accurate answers with page citations — all locally. 🛡️💻

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Local%20AI-000000?style=for-the-badge&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-UVicorn-009485?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Privacy-100%25%20Local-4CAF50?style=for-the-badge" />
</p>

## ✨ Key Features

- **100% Offline & Private** → No internet needed, no data uploaded — total confidentiality.
- **Multi-Format Support** → PDF, DOCX, DOC files.
- **OCR Support** → Reads scanned PDFs (with optional tools).
- **Chat Interface** → Natural language questions about your documents.
- **Source Citations** → Responses include page references (e.g., `[3]`).
- **Multi-Language** → Auto-detection & responses in English, French, or Arabic.












## 🏗 How It Works (RAG Architecture)








## 📦 Prerequisites

1. **Python 3.10+** → [Download Python](https://www.python.org/downloads/)
2. **Ollama** → [Download Ollama](https://ollama.com/)
   - After installation, pull the recommended models:
     ```bash
     ollama pull phi3:3.8b-mini-instruct-4k-q4_k_m
     ollama pull nomic-embed-text
     ```
   - Keep Ollama running in the background.

### Optional (Advanced Features)
- **LibreOffice** → For .doc/.docx conversion.
- **OCRmyPDF** → For scanned document text extraction.

## 🚀 Installation

1. Open a terminal in the project folder.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the App

Start the server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
*Alternative:* `python app.py`

Open your browser at: **http://127.0.0.1:8000**

## 🎨 App Interface Preview




















## 🛠 Usage Guide

1. **Connection Check** → Top-right indicators show Ollama & API status.
2. **Upload Document** → Click the paperclip to add a file.
3. **Ask Questions** → Type naturally — the AI processes, embeds, and answers with citations.
4. **Reset** → Clear the vector database if needed.

## ⚙️ Troubleshooting

- **Chroma DB Issues** → Delete the `chroma_db_data` folder if corrupted.
- **Ollama Not Reachable** → Run `ollama list` to check models.
- **.doc Issues** → Ensure LibreOffice is installed and accessible.

## 📂 Project Structure

- `app.py` → Backend server & core logic
- `app.js` → Frontend interaction
- `index.html` → User interface
- `chroma_db_data/` → Local vector database (auto-created)

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=footer" />
</p>

<p align="center">
  Built with ❤️ for privacy-first AI | Star ⭐ if you find it useful!
</p>
```

This is the **ultimate attractive English README** for your Local RAG project!  
It features a stunning design with gradients, badges, emojis, clear sections, and beautiful illustrative images (RAG diagrams, privacy illustrations, and real-looking chat interfaces).


