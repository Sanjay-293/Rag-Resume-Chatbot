import os
import re
import io
import tempfile
from typing import List, Dict, Tuple, Optional

import streamlit as st

# LangChain / Vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLMs
try:
    from langchain_openai import ChatOpenAI
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except Exception:
        ChatOpenAI = None  # type: ignore

try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# ----------------------------- Utilities ----------------------------- #

def clean_filename_name(stem: str) -> str:
    stem = re.sub(r"[_\-]+", " ", os.path.splitext(stem)[0])
    stem = re.sub(r"\s+cv|\s+resume|\s+profile|\s+updated", "", stem, flags=re.I)
    tokens = [t for t in stem.split() if t]
    if not tokens:
        return ""
    cand = " ".join([t if t.isupper() else t.capitalize() for t in tokens])
    cand = re.sub(r"\b(202\d|20\d\d|v\d+)\b", "", cand)
    cand = re.sub(r"\s+", " ", cand).strip()
    return cand


def guess_name_from_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()][:12]
    for l in lines:
        if any(ch.isdigit() for ch in l):
            continue
        words = l.split()
        if 2 <= len(words) <= 4 and all(w[0:1].isalpha() for w in words):
            caps_ratio = sum(1 for w in words if (w.istitle() or w.isupper())) / len(words)
            if caps_ratio >= 0.75:
                return " ".join(w.capitalize() if not w.isupper() else w for w in words)
    return ""


def extract_candidate_name(first_page_text: str, filename: str) -> str:
    name_from_text = guess_name_from_text(first_page_text)
    name_from_file = clean_filename_name(os.path.basename(filename))
    if name_from_text and len(name_from_text.split()) >= 2:
        return name_from_text
    if name_from_file and len(name_from_file.split()) >= 2:
        return name_from_file
    return name_from_file or "Unknown"


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_pdf_to_docs(file_path: str, source_label: str) -> Tuple[List[Document], str]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    first_page_text = pages[0].page_content if pages else ""
    candidate_name = extract_candidate_name(first_page_text, source_label)
    for i, d in enumerate(pages, start=1):
        d.metadata = d.metadata or {}
        d.metadata.update({
            "source": source_label,
            "page": i,
            "candidate_name": candidate_name,
        })
    return pages, candidate_name


def chunk_documents(pages: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(pages)


def build_index_from_uploads(files: List[io.BytesIO]) -> Tuple[Optional[FAISS], Dict[str, str]]:
    if not files:
        return None, {}

    tmpdir = tempfile.mkdtemp(prefix="resumes_")
    all_chunks: List[Document] = []
    name_map: Dict[str, str] = {}

    for f in files:
        fname = f.name
        save_path = os.path.join(tmpdir, fname)
        with open(save_path, "wb") as out:
            out.write(f.read())
        pages, candidate_name = load_pdf_to_docs(save_path, fname)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)
        name_map[fname] = candidate_name

    if not all_chunks:
        return None, {}

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(all_chunks, embeddings)

    return vectordb, name_map


def get_llm(llm_choice: str):
    if llm_choice == "OpenAI" and ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    if llm_choice == "Ollama" and ChatOllama and os.getenv("RUN_LOCAL_OLLAMA") == "true":
        return ChatOllama(model="llama3", temperature=0)
    return None


def llm_answer(question: str, docs: List[Document], llm_choice: str) -> str:
    llm = get_llm(llm_choice)
    if not llm:
        joined = "\n\n".join(
            f"[Source: {d.metadata.get('candidate_name','?')} — {d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )
        return (
            "No LLM configured. Showing top relevant snippets:\n\n" + joined
        )

    prompt_tmpl = PromptTemplate(
        template=(
            "You are a recruiter assistant. Given the user's question and resume excerpts for a specific candidate, "
            "answer concisely in bullet points. Cite sources as (filename p.x). If a fact isn't in context, say you "
            "don't have enough info.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
        input_variables=["question", "context"],
    )

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

    prompt = prompt_tmpl.format(question=question, context=context)
    resp = llm.invoke(prompt)
    try:
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        return str(resp)


def retrieve(vectordb: FAISS, query: str, k: int, candidate: Optional[str]) -> List[Tuple[Document, float]]:
    docs_and_scores = vectordb.similarity_search_with_score(query, k=k*2)
    if candidate:
        docs_and_scores = [(d, s) for d, s in docs_and_scores if d.metadata.get("candidate_name") == candidate]
    return docs_and_scores[:k]


def get_available_llms():
    available_llms = []
    if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        available_llms.append("OpenAI")
    if ChatOllama and os.getenv("RUN_LOCAL_OLLAMA") == "true":
        available_llms.append("Ollama")
    if not available_llms:
        available_llms = ["None"]
    return available_llms

# ----------------------------- Streamlit UI ----------------------------- #
st.set_page_config(page_title="RAG Resume Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 RAG Resume Chatbot — LangChain + Streamlit")
st.caption("Upload multiple resume PDFs, then ask questions. Optionally pick a candidate to focus retrieval.")

with st.sidebar:
    st.header("1) Upload Resumes (PDF)")
    uploads = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True)
    build = st.button("Rebuild Index", type="primary")
    st.markdown("---")
    st.header("Settings")
    top_k = st.slider("Top-K chunks", 2, 12, 6)

    llm_choice = st.selectbox("Choose LLM backend", options=get_available_llms())

if build:
    st.session_state.pop("vectordb", None)
    st.session_state.pop("name_map", None)

if ("vectordb" not in st.session_state) and uploads:
    with st.spinner("Indexing resumes…"):
        vectordb, name_map = build_index_from_uploads(uploads)
        st.session_state["vectordb"] = vectordb
        st.session_state["name_map"] = name_map
        if vectordb:
            st.success(f"Indexed {len(name_map)} resumes with {len(vectordb.index_to_docstore_id)} chunks.")

vectordb = st.session_state.get("vectordb")
name_map = st.session_state.get("name_map", {})

st.markdown("---")
col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Candidate Filter")
    candidate = st.selectbox(
        "Pick a candidate (optional)",
        options=[""] + sorted(set(name_map.values())),
        index=0,
        format_func=lambda x: "(All candidates)" if x == "" else x,
    )
    if candidate:
        st.info(f"Filtering results to: **{candidate}**")

with col1:
    st.subheader("2) Ask a question")
    question = st.text_input("e.g., Skills, experience with Python, latest company, contact email, etc.")
    ask = st.button("Ask 👇", type="primary", disabled=not (vectordb and question))

if ask and vectordb:
    with st.spinner("Retrieving…"):
        hits = retrieve(vectordb, question, k=top_k, candidate=candidate or None)
        docs = [d for d, _ in hits]
        answer = llm_answer(question, docs, llm_choice)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Show sources"):
        for i, (doc, score) in enumerate(hits, start=1):
            meta = doc.metadata or {}
            st.markdown(
                f"**{i}. {meta.get('candidate_name','?')}** — `{meta.get('source','?')}` p.{meta.get('page','?')}  ")
            st.write(doc.page_content)
            st.caption(f"Similarity score: {score:.4f}")

st.markdown("---")
with st.expander("ℹ️ How candidate names are detected"):
    st.write(
        """
        We use simple heuristics:
        - First page lines: look for a 2–4 word capitalized line without digits (commonly the name header).
        - Filename cleanup: use the PDF filename (minus words like 'resume', 'cv', versions, years).
        If both fail, we mark as 'Unknown'. You can still filter by any detected name.
        """
    )

st.caption("Built with LangChain, FAISS, and sentence-transformers/all-MiniLM-L6-v2.")
