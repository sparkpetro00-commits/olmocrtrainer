"""
SCAL LangExtract + Offline RAG GUI

What this app does:
1) Load unstructured extraction files (.txt/.json/.md)
2) Run schema-style extraction using Google LangExtract
3) Parse HTML tables from extracted OCR text into a unified dataset
4) Build an offline retriever (RAG-style) and chat over report content
5) Export structured outputs to Excel / CSV / JSON

Run:
    python scal_langextract_rag_gui.py

Recommended local model for fully-offline LangExtract:
    Ollama + gemma2:2b (or another local model)
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import pandas as pd

try:
    import langextract as lx

    LANGEXTRACT_AVAILABLE = True
except Exception:
    LANGEXTRACT_AVAILABLE = False


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page: int | None
    text: str
    bow: Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\.%-]+", text.lower())


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        if k in b:
            dot += v * b[k]
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def load_text_or_json(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Attempt JSON flattening if file is JSON-like
    try:
        data = json.loads(content)
    except Exception:
        return content

    def flatten(v: Any) -> list[str]:
        out: list[str] = []
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (int, float, bool)):
            out.append(str(v))
        elif isinstance(v, list):
            for item in v:
                out.extend(flatten(item))
        elif isinstance(v, dict):
            for k in (
                "raw_response",
                "text",
                "content",
                "extracted_text",
                "results",
                "pages",
                "raw_extraction",
                "raw_extractions",
            ):
                if k in v:
                    out.extend(flatten(v[k]))
        return out

    parts = flatten(data)
    if parts:
        return "\n\n".join(parts)
    return json.dumps(data, indent=2)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[tuple[list[str], list[list[str]]]] = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_headers: list[str] = []
        self.current_rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.header_done = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.current_headers = []
            self.current_rows = []
            self.header_done = False
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ("th", "td") and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.in_cell:
            text = "".join(self.current_cell).strip()
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                if not self.header_done:
                    self.current_headers = self.current_row[:]
                    self.header_done = True
                else:
                    self.current_rows.append(self.current_row[:])
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.tables.append((self.current_headers[:], self.current_rows[:]))
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def parse_html_tables(text: str) -> list[pd.DataFrame]:
    parser = _TableParser()
    parser.feed(text)
    frames: list[pd.DataFrame] = []
    for headers, rows in parser.tables:
        if not rows:
            continue
        width = max(len(headers), *(len(r) for r in rows))
        normalized_headers = headers + [f"col_{i+1}" for i in range(len(headers), width)]
        norm_rows = [r + [""] * (width - len(r)) for r in rows]
        frames.append(pd.DataFrame(norm_rows, columns=normalized_headers[:width]))
    return frames


def split_into_page_chunks(text: str) -> list[tuple[int | None, str]]:
    # Supports files saved by the existing GUI ("PAGE X | ...")
    pattern = re.compile(r"=+\s*\nPAGE\s+(\d+)\s*\|.*?\n=+\s*\n", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        # fallback paragraph chunking
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[tuple[int | None, str]] = []
        cur = []
        cur_len = 0
        for p in paras:
            if cur_len + len(p) > 1400 and cur:
                chunks.append((None, "\n\n".join(cur)))
                cur = []
                cur_len = 0
            cur.append(p)
            cur_len += len(p)
        if cur:
            chunks.append((None, "\n\n".join(cur)))
        return chunks

    chunks = []
    for idx, m in enumerate(matches):
        page = int(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append((page, body))
    return chunks


class OfflineRAG:
    def __init__(self):
        self.chunks: list[Chunk] = []

    def build(self, raw_text: str, structured_records: list[dict] | None = None):
        self.chunks = []
        cid = 1
        for page, body in split_into_page_chunks(raw_text):
            self.chunks.append(
                Chunk(cid, "raw_report", page, body, Counter(tokenize(body)))
            )
            cid += 1

        if structured_records:
            for rec in structured_records:
                t = json.dumps(rec, ensure_ascii=False)
                self.chunks.append(
                    Chunk(cid, "structured_record", None, t, Counter(tokenize(t)))
                )
                cid += 1

    def query(self, question: str, top_k: int = 5) -> list[tuple[float, Chunk]]:
        q = Counter(tokenize(question))
        scored = [(cosine_counter(q, ch.bow), ch) for ch in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x for x in scored[:top_k] if x[0] > 0]


class SCALApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SCAL Extractor + Offline RAG")
        self.root.geometry("1300x850")

        self.loaded_file: str | None = None
        self.raw_text = ""
        self.langextract_records: list[dict] = []
        self.unified_table_df: pd.DataFrame | None = None
        self.rag = OfflineRAG()

        self._build_ui()
        self.log("Ready")

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Open TXT/JSON", command=self.open_file).pack(side=tk.LEFT)
        ttk.Button(top, text="Build RAG Index", command=self.build_rag).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Parse HTML Tables", command=self.parse_tables).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=(6, 0))

        self.file_label = ttk.Label(top, text="No file loaded", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=(12, 0))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        tab_input = ttk.Frame(notebook)
        tab_extract = ttk.Frame(notebook)
        tab_rag = ttk.Frame(notebook)
        notebook.add(tab_input, text="Input")
        notebook.add(tab_extract, text="LangExtract")
        notebook.add(tab_rag, text="Offline RAG Chat")

        # Input tab
        self.input_text = scrolledtext.ScrolledText(tab_input, font=("Consolas", 9))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # LangExtract tab
        cfg = ttk.LabelFrame(tab_extract, text="Extraction Config", padding=6)
        cfg.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(cfg, text="Model ID:").grid(row=0, column=0, sticky="w")
        self.model_id_var = tk.StringVar(value="gemma2:2b")
        ttk.Entry(cfg, textvariable=self.model_id_var, width=30).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(cfg, text="Model URL (Ollama):").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.model_url_var = tk.StringVar(value="http://localhost:11434")
        ttk.Entry(cfg, textvariable=self.model_url_var, width=30).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(cfg, text="Extraction Classes:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.classes_var = tk.StringVar(value="sample,depth,permeability,porosity,resistivity,cec,qv,saturation")
        ttk.Entry(cfg, textvariable=self.classes_var, width=70).grid(row=1, column=1, columnspan=3, sticky="we", padx=4, pady=(6, 0))

        ttk.Label(cfg, text="Prompt Description:").grid(row=2, column=0, sticky="nw", pady=(6, 0))
        self.prompt_text = scrolledtext.ScrolledText(cfg, height=6, font=("Consolas", 9))
        self.prompt_text.grid(row=2, column=1, columnspan=3, sticky="we", padx=4, pady=(6, 0))
        self.prompt_text.insert(
            "1.0",
            (
                "Extract SCAL/core-analysis facts and table values in order of appearance. "
                "Use exact spans from text where possible. Include sample id, depth, measurement name, "
                "value, unit, and test condition when present."
            ),
        )

        btns = ttk.Frame(tab_extract)
        btns.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btns, text="Run LangExtract", command=self.run_langextract).pack(side=tk.LEFT)

        self.extract_out = scrolledtext.ScrolledText(tab_extract, font=("Consolas", 9))
        self.extract_out.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # RAG tab
        rag_top = ttk.Frame(tab_rag, padding=4)
        rag_top.pack(fill=tk.X)
        ttk.Label(rag_top, text="Ask a question about SCAL report data:").pack(side=tk.LEFT)
        self.chat_entry = ttk.Entry(rag_top)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.chat_entry.bind("<Return>", lambda e: self.ask())
        ttk.Button(rag_top, text="Ask", command=self.ask).pack(side=tk.LEFT)

        self.chat_out = scrolledtext.ScrolledText(tab_rag, font=("Consolas", 9))
        self.chat_out.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.status = ttk.Label(self.root, text="", anchor="w", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, padx=6, pady=(0, 6))

    def log(self, msg: str):
        self.status.config(text=msg)
        self.root.update_idletasks()

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open extraction file",
            filetypes=[("Text/JSON", "*.txt;*.json;*.md"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            text = load_text_or_json(path)
            self.loaded_file = path
            self.raw_text = text
            self.file_label.config(text=Path(path).name, foreground="black")
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert("1.0", text)
            self.log(f"Loaded {Path(path).name} ({len(text):,} chars)")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def load_text(self, text: str, source_name: str = "in_memory_extraction"):
        """Load text content directly (for integration from other GUIs)."""
        self.raw_text = text or ""
        self.loaded_file = source_name
        self.file_label.config(text=source_name, foreground="black")
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", self.raw_text)
        self.log(f"Loaded {source_name} ({len(self.raw_text):,} chars)")

    def _default_examples(self):
        # High-signal examples for SCAL style docs
        if not LANGEXTRACT_AVAILABLE:
            return []
        return [
            lx.data.ExampleData(
                text="Sample ID 2 Depth 3184.0 meter Permeability to air 0.134 md Porosity 0.116 fraction",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="sample",
                        extraction_text="Sample ID 2",
                        attributes={"sample_id": "2"},
                    ),
                    lx.data.Extraction(
                        extraction_class="depth",
                        extraction_text="3184.0 meter",
                        attributes={"depth_m": "3184.0"},
                    ),
                    lx.data.Extraction(
                        extraction_class="permeability",
                        extraction_text="0.134 md",
                        attributes={"value": "0.134", "unit": "md"},
                    ),
                    lx.data.Extraction(
                        extraction_class="porosity",
                        extraction_text="0.116 fraction",
                        attributes={"value": "0.116", "unit": "fraction"},
                    ),
                ],
            )
        ]

    def _normalize_langextract(self, result: Any) -> list[dict]:
        # Works with object/dataclass or dict-like payloads
        if result is None:
            return []

        exts = []
        if hasattr(result, "extractions"):
            exts = list(getattr(result, "extractions"))
        elif isinstance(result, dict) and "extractions" in result:
            exts = list(result.get("extractions") or [])

        rows: list[dict] = []
        for e in exts:
            if isinstance(e, dict):
                cls = e.get("extraction_class")
                txt = e.get("extraction_text")
                attrs = e.get("attributes") or {}
            else:
                cls = getattr(e, "extraction_class", None)
                txt = getattr(e, "extraction_text", None)
                attrs = getattr(e, "attributes", {}) or {}

            row = {"extraction_class": cls, "extraction_text": txt}
            for k, v in attrs.items():
                row[f"attr_{k}"] = v
            rows.append(row)

        return rows

    def run_langextract(self):
        if not self.raw_text.strip():
            messagebox.showwarning("No input", "Load a TXT/JSON extraction first")
            return
        if not LANGEXTRACT_AVAILABLE:
            messagebox.showerror(
                "Missing dependency",
                "langextract is not installed.\nInstall with: pip install langextract",
            )
            return

        self.log("Running LangExtract...")

        def work():
            try:
                model_id = self.model_id_var.get().strip() or "gemma2:2b"
                model_url = self.model_url_var.get().strip() or "http://localhost:11434"
                prompt = self.prompt_text.get("1.0", tk.END).strip()

                result = lx.extract(
                    text_or_documents=self.raw_text,
                    prompt_description=prompt,
                    examples=self._default_examples(),
                    model_id=model_id,
                    model_url=model_url,
                    fence_output=False,
                    use_schema_constraints=False,
                )
                rows = self._normalize_langextract(result)
                self.root.after(0, lambda: self._on_langextract_done(rows))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("LangExtract Error", str(e)))
                self.root.after(0, lambda: self.log("LangExtract failed"))

        threading.Thread(target=work, daemon=True).start()

    def _on_langextract_done(self, rows: list[dict]):
        self.langextract_records = rows
        self.extract_out.delete("1.0", tk.END)
        self.extract_out.insert("1.0", json.dumps(rows, indent=2, ensure_ascii=False))
        self.log(f"LangExtract complete: {len(rows)} records")

    def parse_tables(self):
        if not self.raw_text.strip():
            messagebox.showwarning("No input", "Load a TXT/JSON extraction first")
            return
        frames = parse_html_tables(self.raw_text)
        if not frames:
            messagebox.showinfo("No tables", "No HTML <table> blocks were found")
            return

        unified_rows = []
        for i, df in enumerate(frames, start=1):
            d = df.copy()
            d.insert(0, "table_id", i)
            unified_rows.append(d)
        self.unified_table_df = pd.concat(unified_rows, ignore_index=True)
        self.log(f"Parsed {len(frames)} HTML tables, unified rows: {len(self.unified_table_df)}")

        preview = self.unified_table_df.head(80).to_string(index=False)
        self.extract_out.delete("1.0", tk.END)
        self.extract_out.insert("1.0", preview)

    def build_rag(self):
        if not self.raw_text.strip() and not self.langextract_records:
            messagebox.showwarning("No data", "Load file or run extraction first")
            return
        self.rag.build(self.raw_text, self.langextract_records)
        self.log(f"RAG index built: {len(self.rag.chunks)} chunks")

    def ask(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        if not self.rag.chunks:
            messagebox.showwarning("No index", "Click 'Build RAG Index' first")
            return

        self.chat_entry.delete(0, tk.END)
        self.chat_out.insert(tk.END, f"You: {q}\n")

        hits = self.rag.query(q, top_k=5)
        if not hits:
            self.chat_out.insert(tk.END, "Assistant: I could not find relevant evidence in the indexed report.\n\n")
            self.chat_out.see(tk.END)
            return

        lines = ["Assistant: Based on retrieved report evidence:"]
        for score, ch in hits:
            loc = f"page {ch.page}" if ch.page is not None else ch.source
            snippet = ch.text.replace("\n", " ").strip()
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            lines.append(f"- ({score:.3f}, {loc}) {snippet}")

        lines.append("\nCitations:")
        for score, ch in hits:
            loc = f"page {ch.page}" if ch.page is not None else ch.source
            lines.append(f"- chunk {ch.chunk_id}: {loc}")

        self.chat_out.insert(tk.END, "\n".join(lines) + "\n\n")
        self.chat_out.see(tk.END)

    def _build_export_frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        if self.langextract_records:
            frames["langextract_records"] = pd.DataFrame(self.langextract_records)
        if self.unified_table_df is not None and not self.unified_table_df.empty:
            frames["unified_tables"] = self.unified_table_df
        if self.raw_text.strip():
            frames["raw_text"] = pd.DataFrame([{"text": self.raw_text}])
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": self.loaded_file or "",
            "langextract_records": len(self.langextract_records),
            "has_unified_table": self.unified_table_df is not None,
            "rag_chunks": len(self.rag.chunks),
        }
        frames["_metadata"] = pd.DataFrame([meta])
        return frames

    def export_excel(self):
        frames = self._build_export_frames()
        if len(frames) <= 1:
            messagebox.showwarning("No data", "Nothing to export yet")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            title="Export structured dataset to Excel",
        )
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for name, df in frames.items():
                    safe_name = name[:31]
                    df.to_excel(writer, sheet_name=safe_name, index=False)
            self.log(f"Exported Excel: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_csv(self):
        if self.unified_table_df is None and not self.langextract_records:
            messagebox.showwarning("No data", "Nothing to export yet")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Export main dataset to CSV",
        )
        if not path:
            return
        try:
            if self.unified_table_df is not None and not self.unified_table_df.empty:
                self.unified_table_df.to_csv(path, index=False)
            else:
                pd.DataFrame(self.langextract_records).to_csv(path, index=False)
            self.log(f"Exported CSV: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


def main():
    root = tk.Tk()
    app = SCALApp(root)
    if not LANGEXTRACT_AVAILABLE:
        messagebox.showwarning(
            "LangExtract not installed",
            "langextract package is missing.\nInstall with: pip install langextract\n\n"
            "You can still use table parsing and offline retrieval chat.",
        )
    root.mainloop()


if __name__ == "__main__":
    main()
