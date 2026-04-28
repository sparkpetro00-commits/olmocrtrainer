# SCAL LangExtract + Offline RAG Setup

## 1) Create environment

```bash
conda create -n scal-rag python=3.11 -y
conda activate scal-rag
```

## 2) Install dependencies

```bash
pip install pandas openpyxl langextract
```

Optional for local Ollama model serving:

- Install Ollama from https://ollama.com/
- Pull a local model:

```bash
ollama pull gemma2:2b
ollama serve
```

## 3) Run the GUI

```bash
python scal_langextract_rag_gui.py
```

## 4) Workflow

1. Open extracted `.txt` or `.json` report.
2. (Optional) Click **Parse HTML Tables** to build a unified table dataset from `<table>` blocks.
3. Click **Run LangExtract** (default uses local Ollama endpoint).
4. Click **Build RAG Index**.
5. Use **Offline RAG Chat** tab to ask questions about SCAL report values.
6. Export to Excel/CSV.

## Notes

- If `langextract` is not installed, the app still supports table parsing + offline retrieval chat.
- Offline RAG in this app is retrieval-first (no cloud dependency).
