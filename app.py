# app.py — Local PDF/Word Analyst (Ollama-only, offline-capable)
# Cleaned v18.5 — Fixed streaming issue where words were removed by aggressive cleanup logic.

import os
import re
import shutil
import tempfile
import textwrap
import hashlib
import time
import unicodedata
import urllib.request
import subprocess
import json
import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import deque
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings

# ---------------- Config ----------------
APP_DB = "chroma_db_data"  # Changed from "db" to avoid conflicting locks/corruption
os.makedirs(APP_DB, exist_ok=True)    
INDEX_PATH = os.path.join(APP_DB, "index.json")

# --- Default Environment Variables (Cleaned) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "phi3:3.8b-mini-instruct-4k-q4_k_m")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Speed/profile 
CTX = int(os.getenv("CTX", "2048"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
AUTOCONTINUE = os.getenv("AUTOCONTINUE", "0") == "1"
MAX_Q_LEN = int(os.getenv("MAX_Q_LEN", "4000")) 

# RAG knobs (tuned)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVER = int(os.getenv("CHUNK_OVERLAP", "80"))
INGEST_PAGES = int(os.getenv("INGEST_MAX_PAGES", "5"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

# Strict mode defaults
CHATGPT_MODE = os.getenv("CHATGPT_MODE", "0") == "1"
STRICT_LANG = os.getenv("STRICT_LANG", "1") == "1"
REQUIRE_CITATIONS = os.getenv("REQUIRE_CITATIONS", "1") == "1"
SHOW_TAGS = os.getenv("SHOW_TAGS", "0") == "1"
FORCE_LANG = os.getenv("FORCE_LANG", "").strip()

CHATGPT_FALLBACK = os.getenv("CHATGPT_FALLBACK", "0") == "1"

# OCR: skip | force | redo
OCRMYPDF_MODE = os.getenv("OCRMYPDF_MODE", "skip").strip().lower()

# Per-document language memory (process-lifetime)
DOC_LANG: Dict[str, str] = {}

# ---------- Optional DOCX fallback ----------
try:
    import docx2txt
    HAVE_DOCX2TXT = True
except Exception:
    HAVE_DOCX2TXT = False

# --------------- Language tools ---------------
_AR = re.compile(r"[\u0600-\u06FF]")

def detect_language(text: str) -> str:
    t = text or ""
    if _AR.search(t): return "Arabic"
    tl = t.lower()
    if any(ch in tl for ch in "éèàùâêîôûçëïüœ") or re.search(
        r"\b(le|la|les|des|un|une|et|pour|avec|comment|quoi|bonjour|salut)\b", tl):
        return "French"
    return "English"

def pick_lang(question: str, doc_id: Optional[str]) -> str:
    if FORCE_LANG in {"Arabic","French","English"}:
        return FORCE_LANG
    if doc_id and doc_id in DOC_LANG:
        return DOC_LANG[doc_id]
    return detect_language(question)

def lang_pack(lang: str) -> Dict[str, str]:
    if lang == "Arabic":
        return {"lang":"Arabic","nf":"غير متوفر في الوثيقة.","h1":"تحليل مفصل:","h2":"توصيات عملية:"}
    if lang == "French":
        return {"lang":"French","nf":"Non trouvé dans le PDF.","h1":"Analyse détaillée :","h2":"Recommandations actionnables :"}
    return {"lang":"English","nf":"Not found in the PDF.","h1":"Detailed analysis:","h2":"Actionable recommendations:"}

_LATIN_KEEP = r"\s0-9A-Za-zÀ-ÖØ-öø-ÿĀ-ſƀ-ɏ\u0300-\u036F\u1E00-\u1EFF"
_ARABIC_KEEP = r"\s0-9\u0660-\u0669\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
_PUNCT_KEEP = r"\.,;:!?\-%\(\)\[\]\{\}«»\"'\/\\\+\*=…–—،؛؟"

def _filter_to_lang(text: str, lang: str) -> str:
    if not STRICT_LANG: return text
    allowed = (_ARABIC_KEEP if lang == "Arabic" else _LATIN_KEEP) + _PUNCT_KEEP
    return re.sub(rf"[^{allowed}]", "", text)

def _dedupe_lines(text: str) -> str:
    seen = set(); out = []
    for raw in text.splitlines():
        s = unicodedata.normalize("NFKC", raw).strip()
        key = re.sub(r"\s+", " ", s).casefold()
        if key and key not in seen:
            seen.add(key); out.append(raw)
    return "\n".join(out)

# --- tag helpers (citations) ---
_TAG_RE = re.compile(r"(?:\[\s*\d+\s*\]|\(p?\s*\d+\)|\[ص\s*\d+\])", re.I)
_ONLY_TAG_LINE = re.compile(r"^\s*(?:\[\s*\d+\s*\]|\(p?\s*\d+\)|\[ص\s*\d+\])\s*$", re.I)

# --- simple list markers to ignore mid-stream (Western + Arabic-Indic) ---
ONLY_NUMBER_LINE = re.compile(r"^\s*[\d\u0660-\u0669]+\s*[.\u066B]\s*$", re.M)
NUM_TOKEN = re.compile(r"^\s*[\d\u0660-\u0669]+\s*[.\u066B]\s*$")
NUM_TOKEN_RUN_TAIL = re.compile(r"(?:\s|^)(?:[\d\u0660-\u0669]+\s*[.\u066B\u066C،؛:]\s*)+$")
ONLY_NUMBER_RUN_FULL = re.compile(r"^\s*(?:[\d\u0660-\u0669]+\s*[.\u066B\u066C،؛:]\s*)+$")

# --- Orphan numeric runs (Western + Arabic-Indic) ----------------------------
_ORPH_DIGIT = r"[0-9\u0660-\u0669]"
_ORPH_SEP = r"[.\u066B\u066C،؛:]"

# FIXED REGEXES:
ORPHAN_RUN_LINE = re.compile(rf"(?m)^\s*(?:{_ORPH_DIGIT}+\s*{_ORPH_SEP}\s*){{2,}}{_ORPH_DIGIT}\.?\s$")
ORPHAN_RUN_ANY = re.compile(rf"(?<!\S)(?:{_ORPH_DIGIT}+\s*{_ORPH_SEP}\s*){{2,}}{_ORPH_DIGIT}*\.?(?!\S)")
ORPHAN_CHAIN_ANY = re.compile(r'(?<!\S)(?:[\d\u0660-\u0669]+\s*[.\u066B\u066C،؛:]\s*){2,}[\d\u0660-\u0669]*\.?(?!\S)')

def _is_orphan_run(s: str) -> bool:
    st = (s or "").strip()
    if not st:
        return False
    if ORPHAN_RUN_LINE.fullmatch(st) or ORPHAN_RUN_ANY.fullmatch(st):
        return True
    noise = re.sub(rf"[{_ORPH_DIGIT}{_ORPH_SEP}\s]", "", st)
    return len(noise) <= max(1, int(0.2 * len(st)))

def _kill_orphan_runs(t: str) -> str:
    if not t:
        return t
    t = ORPHAN_RUN_LINE.sub("", t)
    t = ORPHAN_CHAIN_ANY.sub("", t)
    t = ORPHAN_RUN_ANY.sub("", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()

def _strip_naked_tags(text: str) -> str:
    lines = (text or "").splitlines()
    return "\n".join(ln for ln in lines if not _ONLY_TAG_LINE.match(ln))

def _has_any_tag(text: str) -> bool:
    return bool(_TAG_RE.search(text or ""))

def _keep_paras_if_tagged(text: str, meta: Dict[str,str]) -> str:
    if not text.strip(): return text
    h1 = meta["h1"].split(":")[0].strip()
    h2 = meta["h2"].split(":")[0].strip()
    paras = re.split(r'\n{2,}', text.strip())
    kept = []
    for p in paras:
        sp = p.strip()
        if not sp: continue
        if sp.startswith(h1) or sp.startswith(h2) or _TAG_RE.search(sp):
            kept.append(p)
    return "\n\n".join(kept)

def _hide_tags(text: str) -> str:
    t = re.sub(r'\s*\[\d+\]', '', text)
    t = re.sub(r'\s*\(p?\s*\d+\)', '', t, flags=re.I)
    t = re.sub(r'\s*\[ص\s*\d+\]', '', t)
    return re.sub(r'[ \t]{2,}', ' ', t).strip()

# --------- Sentence-level de-dupers ---------
_SENT_SPLIT = re.compile(r'(?<=[\.!?…]|[؟]|[؛])\s+')

def _norm_sentence(s: str) -> str:
    s = _hide_tags(s)
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def _dedupe_sentences(text: str) -> str:
    parts = _SENT_SPLIT.split(text)
    seen = set(); out = []
    for sent in parts:
        ns = _norm_sentence(sent)
        if ns and ns not in seen:
            seen.add(ns); out.append(sent)
    return " ".join(out).strip()

def _ngram_block(text: str, n: int = 7) -> str:
    toks = re.split(r'\s+', (text or "").strip())
    seen = set(); keep = []
    for i in range(len(toks)):
        ng = tuple(toks[i:i+n])
        if len(ng) < n:
            keep.append(toks[i]); continue
        if ng in seen:
            continue
        seen.add(ng); keep.append(toks[i])
    return " ".join(keep)

# --------- Renumber recommendations (under H2) ---------

def _renumber_h2(text: str, meta: Dict[str, str]) -> str:
    h2_head = meta["h2"].split(":")[0].strip()
    lines = text.splitlines()
    before, h2_lines, after = [], [], []
    state = "before"
    for ln in lines:
        if state == "before":
            before.append(ln)
            if ln.strip().startswith(h2_head):
                state = "h2"
        elif state == "h2":
            if ln.strip().startswith(meta["h1"].split(":")[0].strip()):
                after.append(ln); state = "after"
            else:
                h2_lines.append(ln)
        else:
            after.append(ln)

    if not h2_lines:
        return text

    h2_blob = "\n".join(h2_lines)
    h2_blob = re.sub(r'(?m)^\s*[\d\u0660-\u0669]+\s*[.\u066B]?\s*$', '', h2_blob)
    h2_blob = re.sub(r'(?m)^\s*[\d\u0660-\u0669]+\s*[.\u066B]?\s*', '', h2_blob)
    h2_blob = re.sub(r'\s+[\d\u0660-\u0669]+\s*[.\u066B]\s+', ' ', h2_blob)

    raw_items = [s.strip() for s in _SENT_SPLIT.split(h2_blob) if s.strip()]
    items = [re.sub(r'\s+', ' ', s).strip() for s in raw_items if len(s.strip()) > 2]
    numbered = [f"{i}. {s}" for i, s in enumerate(items, 1)]

    out = []
    out.extend(before); out.extend(numbered); out.extend(after)
    return "\n".join(out)

# --------- Cleaners (strict vs chatty) ---------

def _clean_output_strict(text: str, meta: Dict[str, str], enforce_citations: Optional[bool]=None) -> str:
    t = (text or "").strip()
    t = re.sub(r'^\s*(1\)\s*)?brief\s+analysis.?:\s*', meta["h1"]+" ", t, flags=re.I|re.M)
    t = re.sub(r'^\s*(2\)\s*)?actionable\s+recommendations.?:\s*', meta["h2"]+" ", t, flags=re.I|re.M)
    t = _filter_to_lang(t, meta["lang"])
    t = _strip_naked_tags(t)
    t = re.sub(r'(?:\s*\[\d+\]){2,}', lambda m: re.findall(r'\[\d+\]', m.group(0))[-1], t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = _dedupe_lines(t).strip()
    t = re.sub(r'(?m)^\s*(?:[\d\u0660-\u0669]+\s*[.\u066B]\s*){1,}\s*$', '', t)
    t = _keep_paras_if_tagged(t, meta)
    t = re.sub(r'(?m)^\s*[\d\u0660-\u0669]+\s*[.\u066B]\s*$', '', t)
    enforce = REQUIRE_CITATIONS if enforce_citations is None else enforce_citations
    if enforce and (not t or not _has_any_tag(t)): return meta["nf"]
    if not SHOW_TAGS: t = _hide_tags(t)
    t = _kill_orphan_runs(t)
    t = _dedupe_sentences(t)
    t = _ngram_block(t, n=7)
    t = _renumber_h2(t, meta)
    return t


def _clean_output_chatty(text: str, meta: Dict[str, str]) -> str:
    t = (text or "").strip()
    t = re.sub(r'^\s*(1\)\s*)?brief\s+analysis.?:\s*', meta["h1"]+" ", t, flags=re.I|re.M)
    t = re.sub(r'^\s*(2\)\s*)?actionable\s+recommendations.?:\s*', meta["h2"]+" ", t, flags=re.I|re.M)
    t = _filter_to_lang(t, meta["lang"])
    t = _strip_naked_tags(t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = _dedupe_lines(t).strip()
    t = re.sub(r'(?m)^\s*(?:[\d\u0660-\u0669]+\s*[.\u066B]\s*){1,}\s*$', '', t)
    t = re.sub(r'(?m)^\s*[\d\u0660-\u0669]+\s*[.\u066B]\s*$', '', t)
    if not SHOW_TAGS: t = _hide_tags(t)
    t = _kill_orphan_runs(t)
    t = _dedupe_sentences(t)
    t = _ngram_block(t, n=7)
    t = _renumber_h2(t, meta)
    return t

# --------- Prompts - OPTIMIZED FOR STRICTNESS AND SPEED ---------

def build_prompt_strict() -> ChatPromptTemplate:
    system = (
        "You are a careful PDF analyst. Your only goal is to answer the question using ONLY the provided PDF context below. "
        "Do not invent content or use external knowledge. Avoid all redundancy. "
        "Every sentence containing a non-trivial claim MUST include a numeric page tag like [3] at the END of that sentence. "
        "NEVER output a line that is only a tag. You must use the {language} ONLY. "
        "If a needed fact is missing or cannot be fully grounded with a page tag, reply exactly: {nf}\n"
        "Output MUST have this two-section structure. Use Markdown newlines to separate lists.\n"
        "Headings:\n"
        "## 1) {h1}\n"
        "## 2) {h2}\n" 
        "Under {h1}: 3–6 concise paragraphs, each with tags. Separate paragraphs with a blank line.\n"
        "Under {h2}: 6–10 numbered, actionable recommendations (1-2 sentences per item), each with tags. format as a list like '1. content'.\n"
        "Never print a bare list marker like '6.' without the sentence on the same line."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question:\n{question}\n\nContext (with page tags):\n{context}")
    ])


def build_prompt_chatty() -> ChatPromptTemplate:
    system = (
        "You are a precise document analyst. Use ONLY the provided context. "
        "Answer in {language}. Do NOT invent facts, and avoid repeating ideas or phrases. "
        "Write two sections with the EXACT headings:\n"
        "## 1) {h1}\n"
        "## 2) {h2}\n"
        "Rules:\n"
        "- Under {h1}: 4–6 short, distinct paragraphs. Each paragraph covers a different point. Separate paragraphs with a blank line.\n"
        "- Under {h2}: 6–10 numbered recommendations. Exactly one sentence per item. No duplicates. Format as a markdown list '1. '.\n"
        "- Never print a bare list marker like '6.' without the sentence on the same line.\n"
        "- Avoid generic filler. Prefer concrete, document-grounded points. Omit uncertain points."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])

# Sentence end detector + "should we continue?" helper
_SENT_END = re.compile(r'[\.!?]\s*$|[؟]\s*$|[؛]\s*$|…\s*$')

def _needs_continue(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return False
    last = t.split()[-1]
    return not _SENT_END.search(t) and len(last) > 1

# --------------- Index helpers ---------------

def _index_load() -> Dict[str, Any]:
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"docs": []}


def _index_save(data: Dict[str, Any]) -> None:
    os.makedirs(APP_DB, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------------- Vector DB - REFACTORED FOR STABILITY ---------------

@lru_cache(maxsize=1)
def get_vectordb() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    settings = ChromaSettings(anonymized_telemetry=False)

    try:
        # 1. Attempt to connect to existing DB
        client = PersistentClient(path=APP_DB, settings=settings)
        return Chroma(
            client=client,
            collection_name="pdf",
            embedding_function=embeddings,
        )
    except Exception as e:
        # 2. If connection fails, attempt full repair/reset
        print(f"[ERROR] Chroma DB failed to load. Attempting reset. Error: {e}")
        try:
            shutil.rmtree(APP_DB, ignore_errors=True)
            if os.path.exists(APP_DB):
               # Windows fallback: if deletion failed, try renaming (if not locked by us) 
               # or just warn user.
               print(f"[WARN] Could not delete {APP_DB}. Proceeding anyway, but errors may persist if file is locked.")
            
            os.makedirs(APP_DB, exist_ok=True)
            # Re-create client and collection
            client = PersistentClient(path=APP_DB, settings=settings)
            return Chroma(
                client=client,
                collection_name="pdf",
                embedding_function=embeddings,
            )
        except Exception as e2:
             print("[FATAL] Could not allow ChromaDB to start. If you see this, STOP the app and DELETE the folder.")
             raise RuntimeError(f"Chroma DB repair failed: {e2}") from e2


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    threads = max(1, (os.cpu_count() or 2) - 1)
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=CHAT_MODEL,
        temperature=0.1,
        top_p=0.85,
        repeat_penalty=1.18,
        num_ctx=CTX,
        num_predict=MAX_TOKENS,
        num_thread=threads,
        keep_alive="10m",
    )


def _fmt_doc_preview(d: Document) -> str:
    page = d.metadata.get("page")
    if isinstance(page, int): page = page + 1
    head = textwrap.shorten((d.page_content or "").replace("\n", " "), width=220)
    src = d.metadata.get("source", "")
    tag = f" [{page}]" if page is not None else ""
    return f"{head}{tag}\n-- {src}"


def _sha256(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

# --------------- OCR helper ---------------

def _ocr_pdf_if_needed(pdf_path: str, docs: List[Document]) -> List[Document]:
    total_chars = sum(len((d.page_content or "").strip()) for d in docs)
    if total_chars > 100: return docs
    exe = shutil.which("ocrmypdf") or shutil.which("ocrmypdf.exe")
    if not exe: return docs
    tmp_out = pdf_path + ".ocr.pdf"
    mode = (OCRMYPDF_MODE or "skip").lower()
    if mode == "force": mode_flag = ["--force-ocr"]
    elif mode == "redo": mode_flag = ["--redo-ocr"]
    else: mode_flag = ["--skip-text"]
    try:
        cmd = [exe, "--quiet", *mode_flag, "--language", os.getenv("OCR_LANGS", "eng+fra+ara"), pdf_path, tmp_out]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=300)
        if os.path.exists(tmp_out):
            return PyPDFLoader(tmp_out).load()
    except Exception:
        pass
    return docs

# --------------- Word/PDF loaders ---------------

SAFE_EXTS = {".pdf", ".docx", ".doc"}

def _safe_filename(name: str) -> str:
    base = Path(name).name
    if any(ch in base for ch in ['..', '/', '\\', '%', ':', '\x00']):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return base


def _to_pdf_with_soffice(in_path: str, out_dir: str) -> Optional[str]:
    exe = shutil.which("soffice") or shutil.which("soffice.exe") or r"C:\\Program Files\\LibreOffice\\program\\soffice.exe"
    if not exe or not os.path.exists(exe): return None
    cmd = [exe, "--headless", "--convert-to", "pdf", "--outdir", out_dir, in_path]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90)
        base = os.path.splitext(os.path.basename(in_path))[0] + ".pdf"
        out_pdf = os.path.join(out_dir, base)
        return out_pdf if os.path.exists(out_pdf) else None
    except Exception:
        return None


def _load_docs_any(fpath: str, filename: str) -> List[Document]:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        docs = PyPDFLoader(fpath).load()
        docs = _ocr_pdf_if_needed(fpath, docs)
        return docs[:max(1, INGEST_PAGES)]
    if ext in ("docx", "doc"):
        out_pdf = _to_pdf_with_soffice(fpath, os.path.dirname(fpath))
        if out_pdf:
            docs = PyPDFLoader(out_pdf).load()
            docs = _ocr_pdf_if_needed(out_pdf, docs)
            return docs[:max(1, INGEST_PAGES)]
        if ext == "docx" and HAVE_DOCX2TXT:
            raw = docx2txt.process(fpath) or ""
            text = re.sub(r"\r\n?", "\n", raw).strip()
            paras = [p.strip() for p in text.split("\n") if p.strip()]
            pages, buf, limit = [], "", 1400
            for p in paras:
                if buf and (len(buf) + 2 + len(p) > limit):
                    pages.append(buf.strip()); buf = p
                else:
                    buf = (buf + "\n\n" + p) if buf else p
            if buf: pages.append(buf.strip())
            if not pages: pages = [text] if text else ["(empty)"]
            return [Document(page_content=pg, metadata={"source": filename, "page": i+1})
                        for i, pg in enumerate(pages[:max(1, INGEST_PAGES)])]
        raise HTTPException(
            status_code=400,
            detail=("To ingest .doc/.docx, install LibreOffice (soffice). For .docx fallback: pip install docx2txt")
        )
    raise HTTPException(status_code=400, detail="Unsupported file. Upload .pdf, .docx or .doc.")

# --------------- ingest ---------------

def _ingest_bytes(contents: bytes, filename: str) -> Dict[str, Any]:
    if len(contents) / (1024*1024) > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB).")
    if not filename.lower().endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(status_code=400, detail="Please upload a .pdf, .docx, or .doc file.")

    doc_id = _sha256(contents)
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, filename)
        with open(fpath, "wb") as f: f.write(contents)
        try:
            docs = _load_docs_any(fpath, filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read the document. Error: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER,
            separators=["\n\n","\n",". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        ids = []
        for i, c in enumerate(chunks):
            cid = f"{doc_id}:{i}"
            c.metadata.update({"doc_id": doc_id, "source": filename, "chunk_id": cid})
            ids.append(cid)

        vectordb = get_vectordb()
        try: vectordb.delete(where={"doc_id": doc_id})
        except Exception: pass
        if chunks: vectordb.add_documents(chunks, ids=ids)

        # remember language for this doc
        sample = " ".join((d.page_content or "") for d in docs[:2])
        lang = detect_language(sample)
        DOC_LANG[doc_id] = lang

        # index.json bookkeeping
        idx = _index_load()
        idx["docs"] = [d for d in idx.get("docs", []) if d.get("doc_id") != doc_id]
        idx["docs"].append({
            "doc_id": doc_id,
            "file": filename,
            "lang": lang,
            "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
            "pages_indexed": len(docs),
        })
        _index_save(idx)

        return {"status":"ok","file":filename,"doc_id":doc_id,"lang":lang}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --------------- search (MMR) ---------------

def _search(question: str, k: int, doc_id: Optional[str]) -> List[Document]:
    vectordb = get_vectordb()
    filt = {"doc_id": doc_id} if doc_id else None
    try:
        return vectordb.max_marginal_relevance_search(
            question, k=max(2, k), fetch_k=max(32, k * 12), filter=filt
        )
    except Exception:
        return vectordb.similarity_search(question, k=max(2, k), filter=filt)

# --------------- helpers ---------------

def _format_context(docs: List[Document]) -> str:
    context = "\n\n".join(_fmt_doc_preview(d) for d in docs)
    return (context[:3000] + "…") if len(context) > 3000 else context


def _snippets_from_docs(docs: List[Document], limit: int = 8) -> List[Dict[str, Any]]:
    snips = []
    for d in docs[:limit]:
        page = d.metadata.get("page")
        if isinstance(page, int): page = page + 1
        snips.append({
            "text": textwrap.shorten((d.page_content or "").replace("\n"," "), width=320),
            "page": page,
            "source": d.metadata.get("source", "")
        })
    return snips


def _answer_strict_once(question: str, docs: List[Document], meta: Dict[str,str]) -> str:
    llm = get_llm(); llm.num_predict = MAX_TOKENS
    messages = build_prompt_strict().format_messages(
        question=question, context=_format_context(docs), language=meta["lang"],
        nf=meta["nf"], h1=meta["h1"], h2=meta["h2"]
    )
    resp = llm.invoke(messages)
    return (resp.content or "").strip()


def _answer_chatty(question: str, docs: List[Document], meta: Dict[str,str]) -> str:
    llm = get_llm(); llm.num_predict = MAX_TOKENS
    prompt = build_prompt_chatty()
    resp = llm.invoke(prompt.format_messages(
        question=question, context=_format_context(docs),
        language=meta["lang"], h1=meta["h1"], h2=meta["h2"]
    ))
    return (resp.content or "").strip()

# --------------- answer ---------------

def _answer(question: str, k: int = 6, doc_id: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    question = (question or "")[:MAX_Q_LEN]
    docs = _search(question, k=k, doc_id=doc_id)
    if not docs:
        raise HTTPException(status_code=400, detail="No indexed content found. Upload a document first.")
    t1 = time.time()

    lang = pick_lang(question, doc_id)
    meta = lang_pack(lang)

    try:
        if CHATGPT_MODE:
            text = _answer_chatty(question, docs, meta)
            clean = _clean_output_chatty(text, meta)
        else:
            text = _answer_strict_once(question, docs, meta)
            clean = _clean_output_strict(text, meta)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    if (not CHATGPT_MODE) and clean == meta["nf"] and REQUIRE_CITATIONS and CHATGPT_FALLBACK:
        try:
            text2 = _answer_chatty(question, docs, meta)
            clean = _clean_output_chatty(text2, meta)
        except Exception:
            pass

    if AUTOCONTINUE and _needs_continue(clean) and clean != meta["nf"]:
        cont_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Continue in {meta['lang']} only. Complete the last sentence and keep the same two-section structure."),
            ("human", clean + "\n\n[Continue]")
        ])
        try:
            llm = get_llm(); llm.num_predict = 160
            cont = (llm.invoke(cont_prompt.format_messages()).content or "").strip()
            tail = _clean_output_chatty(cont, meta) if CHATGPT_MODE else _clean_output_strict(
                cont, meta, enforce_citations=REQUIRE_CITATIONS and not CHATGPT_FALLBACK
            )
            clean = (clean + " " + tail).strip()
        except Exception:
            pass

    t2 = time.time()
    print(f"[TIMING] search={t1-t0:.2f}s llm={t2-t1:.2f}s total={time.time()-t0:.2f}s")
    return {"answer": clean, "markdown": clean, "text": clean, "snippets": _snippets_from_docs(docs)}

# ---------------- FastAPI ----------------

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks  # Fix for PydanticUndefinedAnnotation: Callbacks

# ... imports ...

# Fix pydantic error with langchain
ChatOllama.model_rebuild()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmer code...
    try:
        _ = get_vectordb()
        # ... existing warmup ...
    except Exception:
        pass
    yield

app = FastAPI(
    title="Local PDF/Word Analyst (Ollama)",
    version="18.5.2-fixed",
    lifespan=lifespan,
)

# --- CORS (local dev friendly) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Serve static files (css, js, logo)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.options("/{rest_of_path:path}")
def any_options(rest_of_path: str):
    return Response(status_code=200)

# --- helper routes ---
@app.get("/")
def root():
    # Redirect root to index.html or serve it directly
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"ok": True, "message": "API is running. Open index.html manually if not found."}

@app.get("/index.html")
def get_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return Response(status_code=404)

@app.get("/app.js")
def get_js():
    return FileResponse("app.js")

@app.get("/styles.css")
def get_css():
    return FileResponse("styles.css")

@app.get("/logo.jpg")
def get_logo():
    if os.path.exists("logo.jpg"):
        return FileResponse("logo.jpg")
    return Response(status_code=404)

@app.get("/favicon.ico")
def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/health/alive")
def health_alive():
    return {"ok": True, "service": "api"}

@app.get("/health/ollama_only")
def health_ollama_only():
    try:
        url = OLLAMA_URL.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=8) as r:
            return {"ok": r.status == 200, "llm_reply": "ollama up" if r.status == 200 else f"status {r.status}"}
    except Exception as e:
        return {"ok": False, "error": f"ollama unreachable: {e}"}

@app.get("/models")
def list_models():
    try:
        url = OLLAMA_URL.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=8) as r:
            if r.status == 200:
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
                return {"ok": True, "models": [m.get("name") for m in data.get("models", [])]}
            return {"ok": False, "status": r.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- Ingest ----------
@app.post("/ingest")
async def ingest(pdf: UploadFile = File(...)):
    if not pdf.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    ext = Path(pdf.filename).suffix.lower()
    if ext not in SAFE_EXTS:
        raise HTTPException(status_code=400, detail="Please upload .pdf, .docx, or .doc file.")
    safe_name = _safe_filename(pdf.filename)
    contents = await pdf.read()
    return _ingest_bytes(contents, safe_name)

# ---------- Ask (non-streaming) ----------
@app.post("/ask")
async def ask(
    question: str = Form(...),
    k: int = Form(6),
    doc_id: Optional[str] = Form(default=None)
):
    question = (question or "")[:MAX_Q_LEN]
    return _answer(question, k, doc_id)



# ---------- Ask (streaming) ----------
@app.post("/ask_stream")
async def ask_stream(
    question: str = Form(...),
    k: int = Form(6),
    doc_id: Optional[str] = Form(default=None)
):
    question = (question or "")[:MAX_Q_LEN]
    docs = _search(question, k=k, doc_id=doc_id)
    if not docs:
        raise HTTPException(status_code=400, detail="No indexed content found. Upload a document first.")
    lang = pick_lang(question, doc_id)
    meta = lang_pack(lang)

    llm = get_llm(); llm.num_predict = MAX_TOKENS
    if CHATGPT_MODE:
        messages = build_prompt_chatty().format_messages(
            question=question, context=_format_context(docs),
            language=meta["lang"], h1=meta["h1"], h2=meta["h2"]
        )
    else:
        messages = build_prompt_strict().format_messages(
            question=question, context=_format_context(docs), language=meta["lang"],
            nf=meta["nf"], h1=meta["h1"], h2=meta["h2"]
        )
    stream = llm.stream(messages)

    def gen():
        buf = ""
        last_flush = time.time()
        SENT_END = re.compile(r'([\.\!?…]|[؟]|[؛])\s+')
        recent = deque(maxlen=32)
        in_h1 = False
        in_h2 = False

        def should_emit(sentence: str) -> bool:
            ns = _norm_sentence(sentence)
            if not ns: return False
            if ns in recent: return False
            recent.append(ns)
            if CHATGPT_MODE: return True
            return (_TAG_RE.search(sentence)
                    or sentence.strip().startswith(meta["h1"].split(":")[0])
                    or sentence.strip().startswith(meta["h2"].split(":")[0]))

        for chunk in stream:
            raw_piece = (chunk.content or "")
            piece = _filter_to_lang(raw_piece, meta["lang"]) if STRICT_LANG else raw_piece
            if not piece: continue

            buf += piece
            # FIX 1: Removed aggressive stream cleanup here. We only keep the minimal cleanup needed.
            # buf = _kill_orphan_runs(buf) # <-- REMOVED from the main loop
            buf = NUM_TOKEN_RUN_TAIL.sub(" ", buf) 

            parts = SENT_END.split(buf)
            sentences = []
            i = 0
            while i + 1 < len(parts):
                seg, sep = parts[i], parts[i+1]
                sentences.append((seg + sep).strip())
                i += 2
            remainder = parts[-1] if (len(parts) % 2 == 1) else ""

            emits = []
            for s in sentences:
                st = s.strip()
                if not st: continue
                if NUM_TOKEN.match(st):
                    continue

                if st.startswith(meta["h1"].split(":")[0]): in_h1, in_h2 = True, False
                elif st.startswith(meta["h2"].split(":")[0]): in_h1, in_h2 = False, True

                display = _hide_tags(st) if not SHOW_TAGS else st
                display = _kill_orphan_runs(display)
                if _is_orphan_run(display):
                    continue
                if not (in_h1 or in_h2):
                    continue
                if should_emit(display):
                    emits.append(display)

            if emits:
                yield "\n".join(emits) + "\n"
                buf = remainder
                last_flush = time.time()
            else:
                if len(buf) > 250 or (time.time() - last_flush) > 0.9:
                    tail = _hide_tags(buf) if not SHOW_TAGS else buf
                    # FIX 2: Removed aggressive cleanup logic for flushing buffer
                    # tail = _kill_orphan_runs(tail) # <-- REMOVED
                    # if _is_orphan_run(tail): # <-- REMOVED
                    #     tail = "" # <-- REMOVED
                    
                    if (in_h1 or in_h2) and tail.strip():
                        if not NUM_TOKEN.match(tail.strip()) and not ONLY_NUMBER_RUN_FULL.match(tail.strip()):
                            yield tail
                    buf = ""
                    last_flush = time.time()

        # FIX 3: Removed aggressive cleanup from the final buffer
        if buf.strip() and not _ONLY_TAG_LINE.match(buf) and not NUM_TOKEN.match(buf.strip()):
            out = _hide_tags(buf) if not SHOW_TAGS else buf
            # out = _kill_orphan_runs(out) # <-- REMOVED
            if not _is_orphan_run(out):
                out = _dedupe_sentences(out)
                if not ONLY_NUMBER_RUN_FULL.match(out.strip()):
                    yield out

    return StreamingResponse(gen(), media_type="text/plain")

# ---------- Ask + Upload ----------
@app.post("/ask_upload")
async def ask_upload(
    question: str = Form(""),
    pdf: UploadFile = File(...),
    k: int = Form(6),
):
    t0 = time.time()
    if not pdf.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    ext = Path(pdf.filename).suffix.lower()
    if ext not in SAFE_EXTS:
        raise HTTPException(status_code=400, detail="Please upload .pdf, .docx, or .doc file.")
    contents = await pdf.read()
    info = _ingest_bytes(contents, _safe_filename(pdf.filename))
    t1 = time.time()

    auto_q = {
        "Arabic": "قدّم تحليلاً مفصلًا للوثيقة مع توصيات عملية واضحة.",
        "French": "Rédige une analyse détaillée du document et propose des recommandations actionnables.",
        "English": "Write a detailed analysis of the document and provide actionable recommendations."
    }[info["lang"]]
    q = (question or "").strip() or auto_q

    result = _answer(question=q[:MAX_Q_LEN], k=k, doc_id=info["doc_id"])
    t2 = time.time()
    print(f"[TIMING] ingest={t1-t0:.2f}s qa={t2-t1:.2f}s total={t2-t0:.2f}s")
    return {"doc_id": info["doc_id"], "file": info["file"], **result}

# ---------- Docs list ----------
@app.get("/docs")
def list_docs():
    return _index_load()

# ---------- Export (download .md) ----------
@app.post("/export")
async def export_report(
    question: str = Form(""),
    doc_id: Optional[str] = Form(default=None),
    k: int = Form(6),
):
    if not question.strip():
        question = "Write a detailed analysis of the document and provide actionable recommendations."
    res = _answer(question[:MAX_Q_LEN], k=k, doc_id=doc_id)
    content = res.get("markdown") or res.get("text") or ""
    fname = f"report_{(doc_id or 'latest')[:8]}_{int(time.time())}.md"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return PlainTextResponse(content=content, headers=headers)

# ---------- Delete single doc (optional utility) ----------
@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: str):
    try:
        vectordb = get_vectordb()
        vectordb.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    idx = _index_load()
    idx["docs"] = [d for d in idx.get("docs", []) if d.get("doc_id") != doc_id]
    _index_save(idx)
    try:
        if doc_id in DOC_LANG: del DOC_LANG[doc_id]
    except Exception:
        pass
    return {"status": "ok", "deleted": doc_id}

# ---------- Reset ----------
@app.post("/reset")
async def reset():
    try:
        shutil.rmtree(APP_DB, ignore_errors=True)
    finally:
        os.makedirs(APP_DB, exist_ok=True)
        get_vectordb.cache_clear(); get_llm.cache_clear()
        DOC_LANG.clear()
        try: os.remove(INDEX_PATH)
        except Exception: pass
    return {"status": "ok", "message": "Vector DB reset."}


if __name__ == "__main__":
    import uvicorn
    print("Starting Local RAG Server...")
    print("Open http://127.0.0.1:8000/index.html in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
