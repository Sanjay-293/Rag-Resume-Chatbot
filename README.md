# 🧠 RAG Resume Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built with
**LangChain** and **Streamlit**, designed to search and analyze **bulk
resumes in PDF format**.\
Upload multiple resumes, then ask natural language questions about
candidates (e.g., skills, work experience, contact info).\
The chatbot retrieves relevant resume chunks and answers using either
**OpenAI GPT** or a **free local Ollama LLM**.

------------------------------------------------------------------------

## ✨ Features

-   📂 **Bulk PDF Upload** -- Upload multiple resume files at once.\
-   🔍 **Semantic Search** -- Uses Chroma vector database + MiniLM
    embeddings.\
-   🏷 **Auto Candidate Detection** -- Extracts candidate names from the
    first page and filename heuristics.\
-   🎯 **Candidate Filter** -- Ask questions about all candidates or
    just one.\
-   🤖 **Flexible LLM** -- Choose between:
    -   **OpenAI GPT (gpt-4o-mini)** (requires API key)
    -   **Ollama local LLM (llama3, mistral, etc.)** (free & private)\
-   📑 **Cited Sources** -- Each answer includes resume file + page
    references.\
-   🖥 **Streamlit UI** -- Clean, interactive interface for recruiters.

------------------------------------------------------------------------

## 🚀 Setup

### 1. Clone the repo

``` bash
git clone https://github.com/yourusername/rag-resume-chatbot.git
cd rag-resume-chatbot
```

### 2. Install dependencies

``` bash
pip install -U streamlit langchain langchain-community langchain-openai chromadb sentence-transformers pypdf
```

### 3. Configure an LLM

#### Option A: OpenAI (cloud, requires API key)

-   Get an API key from [OpenAI](https://platform.openai.com).

-   Set it in your terminal:

    ``` bash
    export OPENAI_API_KEY=sk-xxxx   # Mac/Linux
    setx OPENAI_API_KEY "sk-xxxx"  # Windows PowerShell
    ```

#### Option B: Ollama (free local LLM)

-   Install [Ollama](https://ollama.com/download).

-   Pull a model:

    ``` bash
    ollama pull llama3
    ```

    (Other models: `ollama pull mistral`, `ollama pull phi3`, etc.)

-   Verify:

    ``` bash
    ollama list
    ```

### 4. Run the chatbot

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🖼 Demo

1.  Upload resume PDFs in the sidebar.\
2.  Choose a candidate (optional).\
3.  Ask a question like:
    -   *"What are John Doe's Python skills?"*\
    -   *"Which candidate has experience in data science?"*\
4.  The chatbot retrieves relevant chunks and answers with sources.

------------------------------------------------------------------------

## 🛠 Tech Stack

-   [LangChain](https://www.langchain.com) -- RAG pipeline\
-   [Chroma](https://www.trychroma.com) -- Vector store\
-   [Sentence Transformers](https://www.sbert.net) -- Embeddings
    (`all-MiniLM-L6-v2`)\
-   [Streamlit](https://streamlit.io) -- Web UI\
-   [Ollama](https://ollama.com) / [OpenAI
    GPT](https://platform.openai.com) -- LLMs

------------------------------------------------------------------------

## 📌 Notes

-   Index is **ephemeral** (in-memory). Rebuild index after uploading
    new files.\
-   If **no LLM** is configured, the app will return top relevant text
    snippets only.\
-   Works fully offline with Ollama.

------------------------------------------------------------------------

## 📜 License

MIT License -- feel free to use and adapt.

