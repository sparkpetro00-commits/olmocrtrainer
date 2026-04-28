"""
olmOCR Agentic Document Extraction GUI - Integrated Version
=========================================================
Uses olmOCR package for PDF rendering and optimized prompts

Run from: C:/Users/Mining/Downloads/olmocr-main
Usage: python olmocr_agentic_gui.py
"""

import os
import sys
import importlib.util
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import json
import threading
from datetime import datetime
from PIL import Image, ImageTk
import torch
import gc
import time
import tempfile
import base64
import io
import pandas as pd

# Add current + parent directories to path
# - current dir: local companion modules (e.g., scal_langextract_rag_gui.py)
# - parent dir: installed/local olmocr package resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    OLMCOCR_AVAILABLE = True
except ImportError as e:
    OLMCOCR_AVAILABLE = False
    print(f"olmOCR package not found: {e}")

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try importing openai for Groq
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional integration: SCAL LangExtract + Offline RAG window
try:
    from scal_langextract_rag_gui import SCALApp
    SCAL_RAG_AVAILABLE = True
    SCAL_RAG_IMPORT_ERROR = ""
except Exception:
    SCAL_RAG_AVAILABLE = False
    SCAL_RAG_IMPORT_ERROR = str(sys.exc_info()[1])

MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_OLMOCR_PROMPT = build_no_anchoring_v4_yaml_prompt() if OLMCOCR_AVAILABLE else """Attached is one page of a document. Return markdown with front matter: primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram."""

TABLE_PROMPT = """Extract table data from the image as JSON array. If no table: {"no_table": true}. Include all rows and columns."""

# Structure prompt for LLM to convert extracted text to structured format
STRUCTURE_PROMPT = """You are a data extraction expert. Convert the extracted document text into a well-structured JSON format.

Extract:
- Tables as JSON arrays
- Key-value pairs as objects
- Lists as arrays
- Headers and sections

Return ONLY valid JSON, no explanation."""

# ===== PROMPT OPTIMIZER SYSTEM PROMPT =====
OPTIMIZER_SYSTEM = """You are an expert at writing extraction prompts for olmOCR, \
a vision-language model fine-tuned for document OCR.

The model's capabilities (stay strictly within these):
- Extracts natural text, preserving reading order
- Converts equations to LaTeX (inline: \\( \\), block: \\[ \\])
- Converts tables to HTML (<table>, <th>, <tr>, <td>)
- Labels figures/charts with markdown: ![description](page_x_y_w_h.png)
- Returns YAML front matter: primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram
- Can describe visible chart content and list visible axis labels, tick values, legend text, and series names

The model CANNOT:
- Digitize graph data points with pixel-level precision
- Perform computations, unit conversions, or transformations
- Output formats other than markdown with YAML front matter
- Guarantee extraction of values not visually present on the page

IMMUTABLE BASE TEMPLATE (must appear word-for-word in the output):
---
Attached is one page of a document that you must process.
Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LaTeX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.
---

TASK: Given the user's document type and extraction goal, produce a single optimized prompt that:
1. Contains the FULL base template above, word-for-word, as the opening
2. Appends the user's specific extraction instruction AFTER the last line of the template
3. Frames the instruction in terms of what the model can actually do
4. If the goal involves chart/graph data: instructs the model to describe all visible axis labels, tick values, legend entries, and data series in an HTML table
5. Does NOT promise outputs beyond model capability — only report what is visually present
6. Returns ONLY the final prompt text, nothing else"""


# ===== POST-PROCESS PROMPT BUILDER =====
def build_postprocess_prompt(raw_text: str, columns: list = None) -> str:
    """Build the post-process prompt.

    Args:
        raw_text: Full compiled OCR text from all pages.
        columns: List of column names from the user template.
                 If None or empty, the LLM infers its own schema.
    """
    if columns:
        cols_str = ", ".join(columns)
        schema_instruction = (
            f"Return a JSON array of objects using EXACTLY these column names: [{cols_str}]\n"
            f"- Do not add extra columns beyond those listed\n"
            f"- Use null for any column where the value is not found in the text"
        )
    else:
        schema_instruction = (
            "Infer an appropriate schema from the content.\n"
            "- Use descriptive snake_case column names\n"
            "- Group related values into the same record where they clearly belong together\n"
            "- Return a JSON array of objects — each object is one record/row"
        )

    return f"""You are a data extraction specialist working on multi-page OCR output.

The text below is compiled from all pages of a scanned document.
Your task: extract ALL structured records (rows of data) from this text.

{schema_instruction}

General rules:
- Only extract values that explicitly appear in the text — do NOT infer or hallucinate
- Numbers must be numeric type, not strings
- If multiple records/rows exist, include ALL of them as separate objects in the array
- Ignore page headers, footers, and VLM metadata lines (primary_language, is_rotation_valid, etc.)
- Return ONLY a valid JSON array — no explanation, no markdown code fences, no preamble

COMPILED DOCUMENT TEXT:
{raw_text}"""


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def resize_image(img, size=1288):
    """Resize image to target longest dimension (as per olmOCR spec)"""
    w, h = img.size
    if max(w, h) == size:
        return img
    if w > h:
        return img.resize((size, int(h * size / w)), Image.LANCZOS)
    return img.resize((int(w * size / h), size), Image.LANCZOS)

class VLMExtractor:
    def __init__(self, model_name=MODEL_ID):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.loaded = False

    def load_model(self):
        if self.loaded:
            return "already loaded"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Loading {self.model_name} on {gpu_mem:.0f}GB GPU...")

        # Processor from base Qwen model — FP8 variant doesn't include its own
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        # Load FP8 model — compressed-tensors handles quantization automatically
        # torch_dtype="auto" reads dtype from model config (FP8), no upcasting
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",   # respects FP8 weights from compressed-tensors
            device_map="auto",    # spreads across available VRAM
        ).eval()

        torch.cuda.empty_cache()
        gc.collect()

        self.loaded = True
        return f"loaded {self.model_name} ({gpu_mem:.0f}GB)"

    def extract(self, image_base64=None, image_path=None, prompt=None):
        if not self.loaded:
            self.load_model()

        # Load image — render_pdf_to_base64png already outputs at 1288px, no resize needed
        if image_base64:
            img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        elif image_path:
            img = Image.open(image_path).convert("RGB")
            img = resize_image(img, 1288)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            raise ValueError("Must provide image_base64 or image_path")

        if prompt is None:
            prompt = DEFAULT_OLMOCR_PROMPT

        # Official message format from HuggingFace model card
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ]}]

        txt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[txt],
            images=[img],
            padding=True,
            return_tensors="pt",
        )

        # Move to model device — no dtype override, let FP8 stay as-is
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                temperature=0.0,    # greedy — faster and matches official pipeline
                max_new_tokens=8000,
                do_sample=False,    # greedy decoding, no sampling overhead
            )

        output_tokens = out.shape[1] - input_ids.shape[1]
        decoded = self.processor.tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        del inputs, out, msgs
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "raw_response": decoded,
            "prompt_used": prompt,
            "input_tokens": input_ids.shape[1],
            "output_tokens": output_tokens,
            "total_tokens": input_ids.shape[1] + output_tokens,
        }

class IntelligentAssistant:
    def __init__(self, model_name=LLM_MODEL_ID):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.history = []

    def load_model(self):
        if self.loaded:
            return "already loaded"
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).to("cuda").eval()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self.loaded = True
        return f"loaded {self.model_name} ({gpu_mem:.0f}GB)"

    def chat(self, message, system_context=None):
        if not self.loaded:
            self.load_model()
        
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        
        for msg in self.history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to("cuda", dtype=torch.long)  # Fix dtype
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=256, temperature=0.7, do_sample=True, top_p=0.9)
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        del inputs, outputs
        torch.cuda.empty_cache()
        
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        
        return response

    def optimize_prompt(self, doc_type, goal, example_fmt=None):
        if not self.loaded:
            self.load_model()
        
        system_msg = "You are an expert at writing prompts for document extraction with VLM."
        
        user_msg = f"""Create an optimized extraction prompt for:
- Document Type: {doc_type}
- Extraction Goal: {goal}
"""
        if example_fmt:
            user_msg += f"- Expected Format: {example_fmt}"
        
        user_msg += "\n\nRespond ONLY with the prompt text."
        
        return self.chat(user_msg, system_context=system_msg)

    def clear_history(self):
        self.history = []


class APILLM:
    """API-based LLM (Groq/OpenAI) for structuring output"""
    
    def __init__(self, provider="groq", api_key=None, model="llama-3.3-70b-versatile"):
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.loaded = False
        self.history = []
    
    def load_model(self):
        if not self.api_key:
            raise RuntimeError(f"No API key set for {self.provider}")
        
        if self.provider == "groq":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
            self.model = "llama-3.3-70b-versatile"
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-4o-mini"
        
        self.loaded = True
        return f"loaded {self.provider} ({self.model})"
    
    def chat(self, message, system_context=None):
        if not self.loaded:
            self.load_model()
        
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        
        for msg in self.history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content
        
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": result})
        
        return result
    
    def structure_data(self, extracted_text, format_type="json"):
        """Convert extracted text to structured format"""
        if format_type == "json":
            prompt = f"""Convert this extracted document text to clean JSON. Extract tables as arrays, key-value pairs as objects.

Text:
{extracted_text}

Return ONLY valid JSON, no explanation."""
        elif format_type == "excel":
            prompt = f"""Convert this extracted document text to a table format (CSV-like with columns).

Text:
{extracted_text}

Return as CSV format with headers."""
        else:
            prompt = extracted_text
        
        return self.chat(prompt, system_context=STRUCTURE_PROMPT)
    
    def clear_history(self):
        self.history = []


class PostProcessAgent:
    """Runs a single LLM call to map raw extracted text to a user-defined column schema.
    Works with both IntelligentAssistant (local) and APILLM (API) backends.
    Anti-hallucination: only extracts values explicitly present in the text."""

    def __init__(self, llm):
        self.llm = llm  # IntelligentAssistant or APILLM instance

    def run(self, raw_text: str, columns: list = None) -> list:
        """Returns list[dict] with one dict per extracted record.

        Args:
            raw_text: Full compiled OCR text.
            columns: Optional column schema. If None, LLM infers schema.
        """
        prompt = build_postprocess_prompt(raw_text, columns)

        # Use chat() for both backends — both expose it
        response = self.llm.chat(prompt, system_context=(
            "You are a precise data extraction assistant. "
            "Return only what is explicitly present in the text as a JSON array. "
            "Never fabricate or infer values."
        ))

        return self._parse(response, columns, raw_text)

    def _parse(self, response: str, columns: list, raw_text: str) -> list:
        """Parse JSON from response. One retry on failure."""
        cleaned = self._strip_fences(response)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                data = [data]
            return data
        except json.JSONDecodeError:
            pass

        # Retry: ask LLM to fix the JSON
        try:
            fix_prompt = (
                f"The following is not valid JSON. Fix it so it is a valid JSON array "
                f"using columns [{', '.join(columns)}]. Return ONLY the fixed JSON array.\n\n"
                f"{response}"
            )
            fixed = self.llm.chat(fix_prompt)
            data = json.loads(self._strip_fences(fixed))
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception:
            return [{"_error": "Could not parse LLM response as JSON", "_raw": response[:500]}]

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences like ```json ... ```"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first and last fence lines
            inner = lines[1:] if lines[0].startswith("```") else lines
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            text = "\n".join(inner).strip()
        return text


class OlmoCRAgenticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("olmOCR Agentic Document Extraction")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        self.root.resizable(True, True)

        self.mode = tk.StringVar(value="single")
        self.selected_files = []
        self.pdf_pages = []
        self.selected_pages = []
        self.extracted_data = []
        self.output_dir = None
        self.current_page_idx = 0
        
        self.vlm = None
        self.llm = None
        self.stop_flag = False
        self.extracting = False
        self.current_prompt = DEFAULT_OLMOCR_PROMPT
        
        # LLM settings
        self.llm_provider = tk.StringVar(value="local")
        self.api_key = ""
        self.api_llm = None
        self.structured_data = None

        # Custom page range for extraction (From-To, 1-based index as user input)
        self.custom_range_from = tk.StringVar(value="")
        self.custom_range_to = tk.StringVar(value="")
        self.custom_range_applied = False

        # Post-process state
        self.template_columns = []
        self.pp_records = []
        self.pp_model_used = ""
        
        self.setup_ui()
        
        self.log("🤖 olmOCR Agentic Extraction Ready!")
        self.log(f"olmOCR package: {'Loaded' if OLMCOCR_AVAILABLE else 'Not found'}")
        self.log("Workflow: Load VLM → Extract → Process Output → Export")

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top bar - just title and status
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="📄 olmOCR Agentic Document Extraction", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        self.model_status = ttk.Label(top_frame, text="Models: Not loaded", 
                                      foreground="gray", font=('Arial', 9))
        self.model_status.pack(side=tk.RIGHT, padx=10)
        
        # Control bar - compact
        control_frame = ttk.Frame(self.root, padding=(5, 2, 5, 2))
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="Load VLM", command=self.cmd_load_vlm, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Load LLM", command=self.cmd_load_llm, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Label(control_frame, text="Mode:").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(control_frame, text="Single", variable=self.mode, value="single").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Batch", variable=self.mode, value="batch").pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Select Files", command=self.select_files, width=12).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(control_frame, text="No files", font=('Arial', 8))
        self.file_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="Select All", command=self.select_all_pages, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Deselect", command=self.deselect_all_pages, width=8).pack(side=tk.LEFT, padx=2)
        self.page_label = ttk.Label(control_frame, text="0 pages", font=('Arial', 8))
        self.page_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="Output", command=self.select_output, width=8).pack(side=tk.LEFT, padx=5)
        self.output_label = ttk.Label(control_frame, text="Not set", font=('Arial', 8))
        self.output_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Label(control_frame, text="LLM:").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(control_frame, text="Local", variable=self.llm_provider, value="local").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Groq", variable=self.llm_provider, value="groq").pack(side=tk.LEFT)
        
        self.api_key_var = tk.StringVar(value=os.getenv("GROQ_API_KEY", ""))
        ttk.Entry(control_frame, textvariable=self.api_key_var, width=10, show="*").pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Set", command=self.set_api_key, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="SCAL RAG", command=self.cmd_open_scal_rag, width=10).pack(side=tk.LEFT, padx=4)
        
        # Main content - Scrollable Canvas
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.main_canvas = tk.Canvas(self.canvas_frame, bg='#f0f0f0')
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scroll_content = ttk.Frame(self.main_canvas)
        self.canvas_window = self.main_canvas.create_window(0, 0, window=self.scroll_content, anchor="nw")
        
        self.scroll_content.bind("<Configure>", self._on_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # PanedWindow inside scrollable content
        main_paned = ttk.PanedWindow(self.scroll_content, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        self._create_left_panel(left_frame)
        
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        self._create_right_panel(right_frame)
        
        # Bottom
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=(0, 2))
        
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padding=2)
        self.status_label.pack(fill=tk.X, padx=5, pady=(0, 5))
    
    def _on_frame_configure(self, event=None):
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        self.main_canvas.itemconfig(self.canvas_window, width=event.width)

    def _create_left_panel(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="📄 Preview", padding=3)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(nav_frame, text="◀", command=self.prev_page, width=5).pack(side=tk.LEFT, padx=2)
        self.page_nav_label = ttk.Label(nav_frame, text="Page: 0 / 0", width=12)
        self.page_nav_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="▶", command=self.next_page, width=5).pack(side=tk.LEFT, padx=2)
        
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_canvas = tk.Canvas(canvas_frame, bg='#2d2d2d')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill=tk.Y)
        
        thumb_frame = ttk.LabelFrame(parent, text="Page Thumbnails", padding=5)
        thumb_frame.pack(fill=tk.X, pady=5)
        
        self.thumb_canvas = tk.Canvas(thumb_frame, bg='#1e1e1e', height=60)
        thumb_scroll = ttk.Scrollbar(thumb_frame, orient="horizontal", command=self.thumb_canvas.xview)
        self.thumb_canvas.configure(xscrollcommand=thumb_scroll.set)
        
        self.thumb_canvas.pack(side="left", fill=tk.X, expand=True)
        thumb_scroll.pack(fill=tk.X)
        
        self.thumb_frame_inner = ttk.Frame(self.thumb_canvas)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_frame_inner, anchor="nw")

    def _create_right_panel(self, parent):
        # Use notebook for right panel sections
        right_notebook = ttk.Notebook(parent)
        right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Extraction (Prompt + Results)
        extract_tab = ttk.Frame(right_notebook)
        right_notebook.add(extract_tab, text="🔍 Extraction")
        self._create_extraction_tab(extract_tab)
        
        # Tab 2: Chat
        chat_tab = ttk.Frame(right_notebook)
        right_notebook.add(chat_tab, text="💬 Chat")
        self._create_chat_tab(chat_tab)
        
        # Tab 3: Error Log
        error_tab = ttk.Frame(right_notebook)
        right_notebook.add(error_tab, text="❌ Error Log")
        self._create_error_tab(error_tab)
        
        # Tab 4: Status Log
        status_tab = ttk.Frame(right_notebook)
        right_notebook.add(status_tab, text="📋 Status Log")
        self._create_status_tab(status_tab)
    
    def _create_extraction_tab(self, parent):
        # Prompt section
        prompt_frame = ttk.LabelFrame(parent, text="📝 Extraction Prompt", padding="5")
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: preset selector + save
        prompt_top = ttk.Frame(prompt_frame)
        prompt_top.pack(fill=tk.X)

        ttk.Label(prompt_top, text="Preset:").pack(side=tk.LEFT, padx=(0, 4))
        self.prompt_choice = ttk.Combobox(prompt_top, width=18, state="readonly")
        self.prompt_choice['values'] = ("Default (olmOCR v4)", "Table Extraction", "Custom")
        self.prompt_choice.current(0)
        self.prompt_choice.pack(side=tk.LEFT, padx=(0, 8))
        self.prompt_choice.bind('<<ComboboxSelected>>', lambda e: self.on_prompt_change())
        ttk.Button(prompt_top, text="💾 Save", command=self.save_prompt, width=8).pack(side=tk.LEFT)
        ttk.Button(prompt_top, text="↩ Reset", command=self.reset_prompt, width=8).pack(side=tk.LEFT, padx=4)

        # Custom Page Range (From-To) for extraction
        range_frame = ttk.LabelFrame(prompt_frame, text="Custom Page Range (From-To, 1-based)", padding=4)
        range_frame.pack(fill=tk.X, padx=5, pady=(6, 2))
        range_row = ttk.Frame(range_frame)
        range_row.pack(fill=tk.X)
        ttk.Label(range_row, text="From:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(range_row, width=6, textvariable=self.custom_range_from).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(range_row, text="To:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(range_row, width=6, textvariable=self.custom_range_to).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(range_row, text="Apply", command=self._apply_custom_range, width=8).pack(side=tk.LEFT, padx=(6, 0))

        # Prompt text box
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=4, font=('Consolas', 9))
        self.prompt_text.pack(fill=tk.X, pady=(6, 4))
        self.prompt_text.insert("1.0", DEFAULT_OLMOCR_PROMPT)

        # Row 2: Optimizer inputs
        opt_frame = ttk.LabelFrame(prompt_frame, text="✨ Prompt Optimizer", padding="4")
        opt_frame.pack(fill=tk.X, pady=(0, 4))

        opt_row1 = ttk.Frame(opt_frame)
        opt_row1.pack(fill=tk.X)
        ttk.Label(opt_row1, text="Doc type:").pack(side=tk.LEFT, padx=(0, 4))
        self.opt_doctype = ttk.Entry(opt_row1, width=18)
        self.opt_doctype.pack(side=tk.LEFT, padx=(0, 8))
        self.opt_doctype.insert(0, "e.g. Lab Report")
        ttk.Label(opt_row1, text="Goal:").pack(side=tk.LEFT, padx=(0, 4))
        self.opt_goal = ttk.Entry(opt_row1)
        self.opt_goal.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.opt_goal.insert(0, "e.g. extract all depth and porosity values")

        opt_row2 = ttk.Frame(opt_frame)
        opt_row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(opt_row2, text="LLM:").pack(side=tk.LEFT, padx=(0, 4))
        self.opt_provider = ttk.Combobox(opt_row2, width=10, state="readonly",
                                          values=("Local", "Groq", "OpenAI"))
        self.opt_provider.current(0)
        self.opt_provider.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(opt_row2, text="API Key:").pack(side=tk.LEFT, padx=(0, 4))
        self.opt_api_key = ttk.Entry(opt_row2, width=28, show="*")
        self.opt_api_key.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(opt_row2, text="✨ Optimize Prompt",
                   command=self.cmd_optimize_prompt).pack(side=tk.LEFT)
        self.opt_status = ttk.Label(opt_row2, text="", foreground="gray")
        self.opt_status.pack(side=tk.LEFT, padx=8)

        # Row 3: Extraction controls
        btn_frame = ttk.Frame(prompt_frame)
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        self.extract_btn = ttk.Button(btn_frame, text="▶ START EXTRACTION", command=self.start_extraction)
        self.extract_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self.stop_extraction, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Output
        output_frame = ttk.LabelFrame(parent, text="📊 Extraction Results", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.response_notebook = ttk.Notebook(output_frame)
        self.response_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Raw Response
        raw_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(raw_frame, text="Raw Response")

        # Save-all toolbar
        raw_toolbar = ttk.Frame(raw_frame)
        raw_toolbar.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(raw_toolbar, text="💾 Save All Pages (TXT)",
                   command=self.cmd_save_all_pages_txt).pack(side=tk.LEFT, padx=4)
        ttk.Button(raw_toolbar, text="💾 Save All Pages (JSON)",
                   command=self.cmd_save_all_pages_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(raw_toolbar, text="🔍 Render Tables",
                   command=lambda: self._refresh_table_render(switch_tab=True)).pack(side=tk.LEFT, padx=4)

        self.raw_text = scrolledtext.ScrolledText(raw_frame, font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        self.raw_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Prompt Used
        prompt_tab_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(prompt_tab_frame, text="Prompt Used")

        self.used_prompt_text = scrolledtext.ScrolledText(prompt_tab_frame, font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        self.used_prompt_text.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Page Timings
        timing_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(timing_frame, text="Page Timings")

        self.timing_text = scrolledtext.ScrolledText(timing_frame, font=('Consolas', 9), bg='#1e1e1e', fg='#cdd6f4')
        self.timing_text.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Post-Process
        self._create_postprocess_tab()

        # Tab 5: Rendered Tables
        self._create_table_render_tab()

        # Token / timing summary bar
        self.page_token_var = tk.StringVar(value="Tokens: -  |  Duration: -")
        ttk.Label(output_frame, textvariable=self.page_token_var, font=('Consolas', 9)).pack(anchor=tk.W, pady=(4, 0))
    
    def _create_chat_tab(self, parent):
        chat_input_frame = ttk.Frame(parent)
        chat_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(chat_input_frame, text="You:").pack(side=tk.LEFT, padx=5)
        self.chat_input = ttk.Entry(chat_input_frame)
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.chat_input.bind('<Return>', lambda e: self.send_chat())
        ttk.Button(chat_input_frame, text="Send", command=self.send_chat).pack(side=tk.LEFT, padx=5)
        
        self.chat_text = scrolledtext.ScrolledText(parent, font=('Consolas', 9), bg='#1e1e2e', fg='#cdd6f4')
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_text.tag_config("user", foreground="#89b4fa")
        self.chat_text.tag_config("assistant", foreground="#a6e3a1")
        self.chat_text.tag_config("system", foreground="#9399b2")
    
    def _create_error_tab(self, parent):
        self.error_text = scrolledtext.ScrolledText(parent, font=('Consolas', 9), bg='#2d1f1f', fg='#f0a0a0')
        self.error_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.error_text.tag_config("error", foreground="#ff6666")
    
    def _create_status_tab(self, parent):
        self.log_text = scrolledtext.ScrolledText(parent, font=('Consolas', 9), bg='#1e2d1e', fg='#a6e3a1')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.tag_config("info", foreground="#a6e3a1")

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=msg, foreground="black")
        entry = f"[{timestamp}] {msg}\n"
        self.log_text.insert(tk.END, entry, "info")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def log_error(self, msg, exc_info=None):
        import traceback
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_msg = f"[{timestamp}] {msg}\n"
        if exc_info:
            error_msg += f"{traceback.format_exc()}\n"
        self.error_text.insert(tk.END, error_msg, "error")
        self.error_text.see(tk.END)
        # Also echo to status log in red
        self.log_text.insert(tk.END, f"[{timestamp}] ERROR: {msg}\n", "status")
        self.log_text.see(tk.END)
        self.status_label.config(text=f"Error: {msg[:80]}", foreground="red")

    def log_chat(self, msg, tag="system"):
        self.chat_text.insert(tk.END, msg + "\n", tag)
        self.chat_text.see(tk.END)

    # ===== FILE OPERATIONS =====
    
    def select_files(self):
        if self.mode.get() == "single":
            filename = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf"), ("Images", "*.png *.jpg *.jpeg")])
            if filename:
                self.selected_files = [filename]
                self.file_label.config(text=Path(filename).name)
                self.load_pdf_preview(filename)
        else:
            folder = filedialog.askdirectory()
            if folder:
                pdf_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.lower().endswith('.pdf')]
                if pdf_files:
                    self.selected_files = pdf_files
                    self.file_label.config(text=f"{len(pdf_files)} PDFs")
                    self.log(f"Found {len(pdf_files)} PDFs")
                    self.clear_preview()

    def load_pdf_preview(self, path):
        self.log(f"Loading {Path(path).name}...")

        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            num_pages = len(reader.pages)
        except Exception as e:
            self.log_error(f"Could not read PDF page count: {e}")
            return

        self.pdf_pages = []
        try:
            for page_num in range(1, num_pages + 1):
                image_base64 = render_pdf_to_base64png(path, page_num, target_longest_image_dim=1288)
                img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
                self.pdf_pages.append(img)
            self.display_preview()
            self.display_thumbnails()
            self.log(f"✓ Loaded {len(self.pdf_pages)} pages")
        except Exception as e:
            self.log_error(f"PDF rendering failed: {e}")

    def display_preview(self):
        self.preview_canvas.delete("all")
        
        if not self.pdf_pages or self.current_page_idx >= len(self.pdf_pages):
            return
        
        canvas_w = self.preview_canvas.winfo_width() or 480
        canvas_h = self.preview_canvas.winfo_height() or 600
        
        img = self.pdf_pages[self.current_page_idx]
        img_w, img_h = img.size
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.9
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        img_display = img.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_display)
        
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        self.page_nav_label.config(text=f"Page: {self.current_page_idx + 1} / {len(self.pdf_pages)}")
        
        if self.current_page_idx < len(self.extracted_data):
            self.display_result(self.extracted_data[self.current_page_idx])

    def display_result(self, result):
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert("1.0", result.get("raw_response", ""))
        
        self.used_prompt_text.delete("1.0", tk.END)
        self.used_prompt_text.insert("1.0", result.get("prompt_used", ""))
        
        tokens = result.get("total_tokens", 0)
        in_tok = result.get("input_tokens", 0)
        out_tok = result.get("output_tokens", 0)
        self.page_token_var.set(f"Tokens: In={in_tok} | Out={out_tok} | Total={tokens}")

        # Auto-refresh table renderer when result is displayed
        # Use after() so the raw_text widget is fully updated before parsing
        self.root.after(50, self._refresh_table_render)

    def display_thumbnails(self):
        for w in self.thumb_frame_inner.winfo_children():
            w.destroy()
        
        self.thumb_vars = []
        
        for i, page_img in enumerate(self.pdf_pages):
            thumb = page_img.copy()
            thumb.thumbnail((80, 100), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            
            var = tk.BooleanVar(value=False)
            frame = ttk.Frame(self.thumb_frame_inner)
            
            cb = ttk.Checkbutton(frame, text=f"P{i+1}", variable=var, command=self.update_page_selection)
            cb.pack()
            
            lbl = ttk.Label(frame, image=photo)
            lbl.image = photo
            lbl.pack()
            
            def make_view(idx):
                return lambda e: self.view_page(idx)
            lbl.bind("<Button-1>", make_view(i))
            
            frame.pack(side=tk.LEFT, padx=2, pady=2)
            self.thumb_vars.append((i, var, photo))
        
        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))

    def view_page(self, idx):
        self.current_page_idx = idx
        self.display_preview()

    def prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.display_preview()

    def next_page(self):
        if self.current_page_idx < len(self.pdf_pages) - 1:
            self.current_page_idx += 1
            self.display_preview()

    def update_page_selection(self):
        self.selected_pages = [i for i, var, _ in self.thumb_vars if var.get()]
        self.page_label.config(text=f"{len(self.selected_pages)} pages")

    def select_all_pages(self):
        if hasattr(self, 'thumb_vars'):
            for _, var, _ in self.thumb_vars:
                var.set(True)
            self.update_page_selection()

    def deselect_all_pages(self):
        if hasattr(self, 'thumb_vars'):
            for _, var, _ in self.thumb_vars:
                var.set(False)
            self.update_page_selection()

    def clear_preview(self):
        self.preview_canvas.delete("all")
        self.pdf_pages = []
        self.selected_pages = []
        self.current_page_idx = 0
        self.page_nav_label.config(text="Page: 0 / 0")

    def select_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = folder
            self.output_label.config(text=Path(folder).name)
    
    # ===== API & OUTPUT PROCESSING =====
    
    def set_api_key(self):
        key = self.api_key_var.get().strip()
        if key:
            self.api_key = key
            self.log(f"API key set for {self.llm_provider.get()}")
        else:
            self.log("Please enter an API key")
    
    def process_output(self):
        """Process extracted data using LLM to structure it"""
        if not self.extracted_data:
            self.log("No extracted data to process. Run extraction first.")
            return
        
        provider = self.llm_provider.get()
        
        if provider == "local":
            # Use local LLM
            if not self.llm or not self.llm.loaded:
                self.log("Loading local LLM...")
                self.cmd_load_llm()
                time.sleep(2)
            
            if not self.llm or not self.llm.loaded:
                self.log("Local LLM not available")
                return
            llm = self.llm
        else:
            # Use API LLM
            if not self.api_key:
                self.log("Please set API key first")
                return
            
            try:
                if not hasattr(self, 'api_llm') or not self.api_llm:
                    self.api_llm = APILLM(provider=provider, api_key=self.api_key)
                    self.api_llm.load_model()
                llm = self.api_llm
            except Exception as e:
                self.log_error(f"API LLM error: {e}")
                return
        
        self.log("Processing output to structured format...")
        
        # Combine all extracted text
        all_text = ""
        for result in self.extracted_data:
            all_text += result.get("raw_response", "") + "\n\n"
        
        def do_structure():
            try:
                # Both IntelligentAssistant and APILLM expose .chat()
                # APILLM also has .structure_data() but we use .chat() for portability
                struct_prompt = (
                    f"Convert this extracted document text to clean JSON. "
                    f"Extract tables as arrays, key-value pairs as objects.\n\n"
                    f"Text:\n{all_text}\n\nReturn ONLY valid JSON, no explanation."
                )
                structured = llm.chat(struct_prompt, system_context=STRUCTURE_PROMPT)
                self.structured_data = structured
                self.root.after(0, lambda: self.log("✓ Output structured! Click 'Export' to save."))
                self.root.after(0, lambda: self.display_structured_output(structured))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_error(f"Structure error: {err}"))
        
        threading.Thread(target=do_structure, daemon=True).start()
    
    def display_structured_output(self, data):
        """Show structured output in the post-process preview tab."""
        self.pp_output_text.delete("1.0", tk.END)
        self.pp_output_text.insert("1.0", data)
        # Switch to Post-Process tab so the user sees the result
        self.response_notebook.select(3)
    
    def export_to_excel(self):
        """Export post-processed records to Excel with Data + _metadata sheets."""
        records = getattr(self, 'pp_records', [])
        if not records:
            # Fall back to raw structured_data if post-process hasn't run
            if not self.structured_data:
                self.log("Run Post-Process first, or use Export JSON for raw results")
                return
            try:
                records = json.loads(self.structured_data)
                if not isinstance(records, list):
                    records = [records]
            except Exception:
                self.log_error("Could not parse structured data as JSON for Excel export")
                return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not filename:
            return

        try:
            # Build data sheet — use template column order if available
            df = pd.DataFrame(records)
            if self.template_columns:
                # Reorder to match template, keep extra cols at end
                ordered = [c for c in self.template_columns if c in df.columns]
                extras = [c for c in df.columns if c not in self.template_columns]
                df = df[ordered + extras]

            # Auto-infer numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')

            # Build metadata sheet
            source = self.selected_files[0] if self.selected_files else "unknown"
            total_tokens = sum(r.get("total_tokens", 0) for r in self.extracted_data)
            meta_df = pd.DataFrame([{
                "source_file": source,
                "model": MODEL_ID,
                "post_process_model": getattr(self, 'pp_model_used', ''),
                "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pages_extracted": len(self.extracted_data),
                "total_tokens": total_tokens,
                "template_columns": ", ".join(self.template_columns),
                "records_extracted": len(records),
            }])

            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Data", index=False)
                meta_df.to_excel(writer, sheet_name="_metadata", index=False)

            self.log(f"✓ Exported {len(records)} record(s) to {Path(filename).name}")
        except Exception as e:
            self.log_error(f"Excel export failed: {e}")

    def export_to_csv(self):
        """Export post-processed records to CSV."""
        records = getattr(self, 'pp_records', [])
        if not records:
            self.log("Run Post-Process first, then export CSV")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not filename:
            return

        try:
            df = pd.DataFrame(records)
            if self.template_columns:
                ordered = [c for c in self.template_columns if c in df.columns]
                extras = [c for c in df.columns if c not in self.template_columns]
                df = df[ordered + extras]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            df.to_csv(filename, index=False)
            self.log(f"✓ Exported {len(records)} record(s) to {Path(filename).name}")
        except Exception as e:
            self.log_error(f"CSV export failed: {e}")

    def export_to_json(self):
        """Export full extraction + post-process results to JSON."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not filename:
            return

        try:
            source = self.selected_files[0] if self.selected_files else "unknown"
            total_tokens = sum(r.get("total_tokens", 0) for r in self.extracted_data)
            output = {
                "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": source,
                "model": MODEL_ID,
                "pages_extracted": len(self.extracted_data),
                "total_tokens": total_tokens,
                "post_process_model": getattr(self, 'pp_model_used', ''),
                "template_columns": self.template_columns,
                "post_processed_records": getattr(self, 'pp_records', []),
                "raw_extraction": self.extracted_data,
            }
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            self.log(f"✓ Exported to {Path(filename).name}")
        except Exception as e:
            self.log_error(f"JSON export failed: {e}")

    # ===== PROMPT =====
    
    def on_prompt_change(self):
        idx = self.prompt_choice.current()
        if idx == 0:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", DEFAULT_OLMOCR_PROMPT)
        elif idx == 1:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", TABLE_PROMPT)

    def save_prompt(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            with open(filename, 'w') as f:
                f.write(self.prompt_text.get("1.0", tk.END))
            self.log("Prompt saved")

    def reset_prompt(self):
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", DEFAULT_OLMOCR_PROMPT)
        self.prompt_choice.current(0)

    def cmd_optimize_prompt(self):
        """Optimize the prompt using selected LLM backend, inline (no dialog)."""
        doc_type = self.opt_doctype.get().strip()
        goal = self.opt_goal.get().strip()

        if not goal:
            self.opt_status.config(text="Enter a goal first", foreground="red")
            return

        provider = self.opt_provider.get().lower()
        api_key = self.opt_api_key.get().strip()

        self.opt_status.config(text="Optimizing...", foreground="gray")
        self.root.update_idletasks()

        def _run():
            try:
                if provider == "local":
                    if not self.llm or not self.llm.loaded:
                        self.root.after(0, lambda: self.opt_status.config(
                            text="Loading local LLM...", foreground="gray"))
                        self.cmd_load_llm()
                        time.sleep(2)
                    llm = self.llm
                else:
                    llm = APILLM(provider=provider, api_key=api_key or self.api_key)
                    llm.load_model()

                user_msg = (
                    f"Document type: {doc_type or 'unknown'}\n"
                    f"Extraction goal: {goal}\n\n"
                    f"Write the optimized prompt."
                )
                optimized = llm.chat(user_msg, system_context=OPTIMIZER_SYSTEM)
                optimized = optimized.strip()

                def _apply():
                    self.prompt_text.delete("1.0", tk.END)
                    self.prompt_text.insert("1.0", optimized)
                    self.prompt_choice.current(2)
                    self.opt_status.config(text="✓ Done", foreground="green")
                    self.log("✓ Prompt optimized")

                self.root.after(0, _apply)

            except Exception as e:
                self.root.after(0, lambda err=str(e): (
                    self.opt_status.config(text=f"Error: {err[:50]}", foreground="red"),
                    self.log_error(f"Prompt optimizer failed: {err}")
                ))

        threading.Thread(target=_run, daemon=True).start()

    # ===== POST-PROCESS TAB =====

    def _create_postprocess_tab(self):
        """Build the Post-Process tab inside response_notebook.

        This tab operates on the FULL compiled extraction (all pages of the document),
        not on individual pages. It uses an LLM to clean, arrange, and structure
        the OCR output into a tabular dataset ready for export.
        """
        pp_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(pp_frame, text="Post-Process")

        # --- Info banner ---
        info = ttk.Label(
            pp_frame,
            text="Post-processing compiles ALL extracted pages into a structured dataset. "
                 "Run extraction first, then optionally upload a column template.",
            foreground="#888888", font=('Arial', 8, 'italic'), wraplength=800, justify="left",
        )
        info.pack(fill=tk.X, padx=8, pady=(4, 2))

        # --- Document stats row (populated after extraction) ---
        self.pp_doc_stats = ttk.Label(pp_frame, text="No extraction data loaded.",
                                       foreground="gray", font=('Consolas', 8))
        self.pp_doc_stats.pack(fill=tk.X, padx=8, pady=(0, 4))

        # --- Template section ---
        tpl_frame = ttk.LabelFrame(pp_frame, text="Column Schema  (optional — leave blank to auto-infer)", padding=4)
        tpl_frame.pack(fill=tk.X, padx=6, pady=(0, 4))

        tpl_row = ttk.Frame(tpl_frame)
        tpl_row.pack(fill=tk.X)

        ttk.Label(tpl_row, text="Template:").pack(side=tk.LEFT, padx=(0, 4))
        self.pp_template_label = ttk.Label(tpl_row, text="None (auto-infer schema)",
                                           foreground="gray", width=34, anchor="w")
        self.pp_template_label.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tpl_row, text="📂 Upload .xlsx/.csv",
                   command=self.cmd_upload_template).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tpl_row, text="✕ Clear",
                   command=self.cmd_clear_template).pack(side=tk.LEFT)

        self.pp_cols_label = ttk.Label(tpl_frame, text="Columns: auto-infer",
                                        foreground="gray", font=('Consolas', 8))
        self.pp_cols_label.pack(fill=tk.X, pady=(2, 0))

        # --- LLM selector + run ---
        run_frame = ttk.LabelFrame(pp_frame, text="Run", padding=4)
        run_frame.pack(fill=tk.X, padx=6, pady=(0, 4))

        run_row = ttk.Frame(run_frame)
        run_row.pack(fill=tk.X)

        ttk.Label(run_row, text="LLM:").pack(side=tk.LEFT, padx=(0, 4))
        self.pp_provider = ttk.Combobox(run_row, width=10, state="readonly",
                                         values=("Local", "Groq", "OpenAI"))
        self.pp_provider.current(0)
        self.pp_provider.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(run_row, text="API Key:").pack(side=tk.LEFT, padx=(0, 4))
        self.pp_api_key = ttk.Entry(run_row, width=28, show="*")
        self.pp_api_key.pack(side=tk.LEFT, padx=(0, 8))

        self.pp_run_btn = ttk.Button(run_frame, text="▶  Run Post-Process on Full Document",
                                      command=self.cmd_run_postprocess)
        self.pp_run_btn.pack(fill=tk.X, pady=(4, 0))

        self.pp_status = ttk.Label(run_frame, text="", foreground="gray")
        self.pp_status.pack(anchor=tk.W)

        # --- Output preview ---
        out_frame = ttk.LabelFrame(pp_frame, text="Structured Output Preview (JSON)", padding=4)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))

        self.pp_output_text = scrolledtext.ScrolledText(
            out_frame, font=('Consolas', 8), bg='#1e1e2e', fg='#cdd6f4')
        self.pp_output_text.pack(fill=tk.BOTH, expand=True)

        # --- Export buttons ---
        exp_row = ttk.Frame(pp_frame)
        exp_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(exp_row, text="Export:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(exp_row, text="📊 Excel (.xlsx)",
                   command=self.export_to_excel).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(exp_row, text="📄 CSV (.csv)",
                   command=self.export_to_csv).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(exp_row, text="📋 JSON (.json)",
                   command=self.export_to_json).pack(side=tk.LEFT)

    # ===== SAVE ALL PAGES =====

    def cmd_save_all_pages_txt(self):
        """Concatenate raw responses from all pages into one .txt file."""
        if not self.extracted_data:
            self.log("No extracted data yet. Run extraction first.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All", "*.*")],
            title="Save All Pages as Text",
        )
        if not filename:
            return
        try:
            lines = []
            for r in self.extracted_data:
                pg = r.get("page_number", "?")
                lines.append(f"{'='*60}")
                lines.append(f"PAGE {pg}  |  {r.get('duration_s', 0):.1f}s  |  {r.get('total_tokens', 0)} tokens")
                lines.append(f"{'='*60}")
                lines.append(r.get("raw_response", ""))
                lines.append("")
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.log(f"✓ Saved {len(self.extracted_data)} pages to {Path(filename).name}")
        except Exception as e:
            self.log_error(f"Save TXT failed: {e}")

    def cmd_save_all_pages_json(self):
        """Save all page results (raw extraction) to a single JSON file."""
        if not self.extracted_data:
            self.log("No extracted data yet. Run extraction first.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            title="Save All Pages as JSON",
        )
        if not filename:
            return
        try:
            source = self.selected_files[0] if self.selected_files else "unknown"
            total_tokens = sum(r.get("total_tokens", 0) for r in self.extracted_data)
            output = {
                "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": source,
                "model": MODEL_ID,
                "pages_extracted": len(self.extracted_data),
                "total_tokens": total_tokens,
                "pages": self.extracted_data,
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            self.log(f"✓ Saved {len(self.extracted_data)} pages to {Path(filename).name}")
        except Exception as e:
            self.log_error(f"Save JSON failed: {e}")

    # ===== TABLE RENDERER =====

    def _create_table_render_tab(self):
        """Build the Rendered Tables tab — parses <table> HTML from raw responses."""
        import html.parser

        tab_frame = ttk.Frame(self.response_notebook)
        self.response_notebook.add(tab_frame, text="Rendered Tables")

        ctrl = ttk.Frame(tab_frame)
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(ctrl, text="🔄 Refresh from Current Page",
                   command=lambda: self._refresh_table_render(switch_tab=True)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="🔄 Refresh from All Pages",
                   command=lambda: self._refresh_table_render(all_pages=True, switch_tab=True)).pack(side=tk.LEFT, padx=(0, 6))
        self.table_render_status = ttk.Label(ctrl, text="No tables found", foreground="gray")
        self.table_render_status.pack(side=tk.LEFT, padx=8)

        # Scrollable container for multiple treeviews
        canvas = tk.Canvas(tab_frame, bg='#1e1e1e')
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.table_render_inner = ttk.Frame(canvas)
        self.table_render_window = canvas.create_window((0, 0), window=self.table_render_inner, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.table_render_inner.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(self.table_render_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        self._table_render_canvas = canvas

    def _parse_html_tables(self, html_text: str) -> list:
        """Parse all <table> blocks from html_text. Returns list of (headers, rows) tuples."""
        import html.parser
        import html as html_module

        tables = []

        class TableParser(html.parser.HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_table = False
                self.in_row = False
                self.in_cell = False
                self.cell_is_header = False
                self.current_row = []
                self.current_headers = []
                self.current_rows = []
                self.current_cell_text = []
                self.headers_done = False

            def handle_starttag(self, tag, attrs):
                if tag == "table":
                    self.in_table = True
                    self.current_headers = []
                    self.current_rows = []
                    self.headers_done = False
                elif tag == "tr" and self.in_table:
                    self.in_row = True
                    self.current_row = []
                elif tag in ("th", "td") and self.in_row:
                    self.in_cell = True
                    self.cell_is_header = (tag == "th")
                    self.current_cell_text = []

            def handle_endtag(self, tag):
                if tag in ("th", "td") and self.in_cell:
                    text = html_module.unescape("".join(self.current_cell_text).strip())
                    self.current_row.append(text)
                    self.in_cell = False
                elif tag == "tr" and self.in_row:
                    # Decide: header row or data row
                    if not self.headers_done and self.current_row:
                        # Use first row as headers (mix of th/td is fine)
                        self.current_headers = self.current_row[:]
                        self.headers_done = True
                    elif self.current_row:
                        self.current_rows.append(self.current_row[:])
                    self.in_row = False
                elif tag == "table" and self.in_table:
                    tables.append((self.current_headers[:], self.current_rows[:]))
                    self.in_table = False

            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell_text.append(data)

        parser = TableParser()
        parser.feed(html_text)
        return tables

    def _refresh_table_render(self, all_pages=False, switch_tab=False):
        """Parse HTML tables from raw response(s) and display as Treeview grids.

        Args:
            all_pages: If True, parse all extracted pages instead of the current one.
            switch_tab: If True, switch to the Rendered Tables tab after rendering.
        """
        # Clear previous widgets
        for w in self.table_render_inner.winfo_children():
            w.destroy()

        # Gather text to parse
        if all_pages and self.extracted_data:
            combined = "\n".join(r.get("raw_response", "") for r in self.extracted_data)
            source_label = f"All {len(self.extracted_data)} pages"
        else:
            # Use currently displayed raw text (includes whatever page is shown)
            combined = self.raw_text.get("1.0", tk.END)
            source_label = f"Page {self.current_page_idx + 1}"

        tables = self._parse_html_tables(combined)

        if not tables:
            self.table_render_status.config(
                text=f"No <table> elements found in {source_label}", foreground="orange")
            ttk.Label(self.table_render_inner, text="No tables found in the current text.",
                      foreground="gray").pack(padx=10, pady=10)
            if switch_tab:
                self.response_notebook.select(4)
            return

        self.table_render_status.config(
            text=f"Found {len(tables)} table(s) in {source_label}", foreground="green")

        for tbl_idx, (headers, rows) in enumerate(tables, start=1):
            # Title
            ttk.Label(self.table_render_inner,
                      text=f"Table {tbl_idx} — {len(rows)} row(s), {len(headers)} column(s)",
                      font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=8, pady=(8, 2))

            if not headers:
                headers = [f"Col {i+1}" for i in range(max((len(r) for r in rows), default=1))]

            # Pad rows to header length
            ncols = len(headers)
            padded_rows = [r + [""] * (ncols - len(r)) for r in rows]

            tv_frame = ttk.Frame(self.table_render_inner)
            tv_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

            tv = ttk.Treeview(tv_frame, columns=headers, show="headings",
                               height=min(len(padded_rows) + 1, 20))
            for col in headers:
                tv.heading(col, text=col)
                # Auto-width: max of header len and cell values
                max_len = max(
                    len(str(col)),
                    max((len(str(r[headers.index(col)])) for r in padded_rows), default=0)
                )
                tv.column(col, width=min(max(max_len * 8, 60), 300), anchor="w")

            for row in padded_rows:
                tv.insert("", tk.END, values=row)

            tv_scroll_x = ttk.Scrollbar(tv_frame, orient="horizontal", command=tv.xview)
            tv.configure(xscrollcommand=tv_scroll_x.set)
            tv.pack(fill=tk.X)
            tv_scroll_x.pack(fill=tk.X)

        if switch_tab:
            self.response_notebook.select(4)

    def cmd_upload_template(self):
        """Upload .xlsx or .csv, read column headers from row 1."""
        filetypes = [("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, nrows=0)
            else:
                df = pd.read_excel(path, nrows=0)
            self.template_columns = list(df.columns)
            self.pp_template_label.config(
                text=Path(path).name, foreground="black")
            cols_preview = ", ".join(self.template_columns[:8])
            if len(self.template_columns) > 8:
                cols_preview += f" ... (+{len(self.template_columns)-8} more)"
            self.pp_cols_label.config(
                text=f"Columns ({len(self.template_columns)}): {cols_preview}",
                foreground="black")
            self.log(f"✓ Template loaded: {len(self.template_columns)} columns from {Path(path).name}")
        except Exception as e:
            self.log_error(f"Template load failed: {e}")

    def cmd_clear_template(self):
        self.template_columns = []
        self.pp_template_label.config(text="None (auto-infer schema)", foreground="gray")
        self.pp_cols_label.config(text="Columns: auto-infer", foreground="gray")

    def _update_pp_doc_stats(self):
        """Refresh the doc-stats label in the Post-Process tab."""
        if not self.extracted_data:
            self.pp_doc_stats.config(text="No extraction data loaded.", foreground="gray")
            return
        pages = len(self.extracted_data)
        chars = sum(len(r.get("raw_response", "")) for r in self.extracted_data)
        tokens = sum(r.get("total_tokens", 0) for r in self.extracted_data)
        source = Path(self.selected_files[0]).name if self.selected_files else "unknown"
        self.pp_doc_stats.config(
            text=f"Ready: {pages} page(s)  |  ~{chars:,} chars  |  {tokens:,} tokens  |  {source}",
            foreground="black",
        )

    def cmd_run_postprocess(self):
        """Run the post-process agent on the FULL compiled extraction (all pages)."""
        if not self.extracted_data:
            self.pp_status.config(text="No extracted data yet — run extraction first", foreground="red")
            return

        # Update stats display
        self._update_pp_doc_stats()

        provider = self.pp_provider.get().lower()
        api_key = self.pp_api_key.get().strip() or self.api_key

        # Concatenate all pages
        raw_text = "\n\n--- PAGE BREAK ---\n\n".join(
            r.get("raw_response", "") for r in self.extracted_data
        )

        # Warn if very large (likely to hit LLM context limits)
        char_count = len(raw_text)
        WARN_CHARS = 60_000
        if char_count > WARN_CHARS:
            proceed = messagebox.askyesno(
                "Large Document Warning",
                f"The compiled text is {char_count:,} characters (~{char_count//4:,} tokens).\n\n"
                f"This may exceed the context window of the selected LLM and produce incomplete results.\n\n"
                f"Consider using only a page subset (deselect pages before extraction) "
                f"or use a model with a larger context window (e.g. Groq llama-3.3-70b).\n\n"
                f"Continue anyway?",
            )
            if not proceed:
                return

        self.pp_status.config(text="Running...", foreground="gray")
        self.pp_run_btn.config(state=tk.DISABLED)
        self.pp_output_text.delete("1.0", tk.END)

        # Use template columns if loaded, else None (auto-infer)
        columns = self.template_columns if self.template_columns else None
        schema_note = f"{len(columns)} columns from template" if columns else "auto-infer schema"
        self.log(f"Post-processing {len(self.extracted_data)} page(s) | {char_count:,} chars | {schema_note}")

        def _run():
            try:
                # Build LLM backend
                if provider == "local":
                    if not self.llm or not self.llm.loaded:
                        self.root.after(0, lambda: self.pp_status.config(
                            text="Loading local LLM...", foreground="gray"))
                        self.cmd_load_llm()
                        time.sleep(2)
                    llm = self.llm
                else:
                    llm = APILLM(provider=provider, api_key=api_key)
                    llm.load_model()

                agent = PostProcessAgent(llm)
                records = agent.run(raw_text, columns)

                self.pp_records = records
                self.structured_data = json.dumps(records, indent=2)
                self.pp_model_used = f"{provider}:{getattr(llm, 'model', 'local')}"

                preview = json.dumps(records, indent=2)

                def _update():
                    self.pp_output_text.delete("1.0", tk.END)
                    self.pp_output_text.insert(tk.END, preview)
                    self.pp_status.config(
                        text=f"✓ {len(records)} record(s) extracted | ready to export",
                        foreground="green")
                    self.pp_run_btn.config(state=tk.NORMAL)
                    self.log(f"✓ Post-process done: {len(records)} record(s)")

                self.root.after(0, _update)

            except Exception as e:
                def _err(err=str(e)):
                    self.pp_status.config(text=f"Error: {err[:60]}", foreground="red")
                    self.pp_run_btn.config(state=tk.NORMAL)
                    self.log_error(f"Post-process error: {err}")
                self.root.after(0, _err)

        threading.Thread(target=_run, daemon=True).start()

    # ===== EXTRACTION =====
    
    def start_extraction(self):
        if not self.vlm or not self.vlm.loaded:
            self.log("Loading VLM...")
            self.cmd_load_vlm()
            time.sleep(2)
        
        self.stop_flag = False
        self.extracting = True
        self.extract_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._run_extraction)
        thread.daemon = True
        thread.start()

    def stop_extraction(self):
        self.stop_flag = True
        self.log("Stopping...")

    def _run_extraction(self):
        try:
            if not self.output_dir:
                if self.selected_files:
                    self.output_dir = str(Path(self.selected_files[0]).parent / "extraction_output")
                    Path(self.output_dir).mkdir(exist_ok=True)
                    self.root.after(0, lambda: self.output_label.config(text=Path(self.output_dir).name))
            
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            self.extracted_data = []
            total_start = time.time()
            page_timings = []  # list of {page, duration_s, tokens}

            # Clear timing tab
            self.root.after(0, lambda: (
                self.timing_text.delete("1.0", tk.END),
                self.timing_text.insert(tk.END, f"{'Page':<8}{'Duration':>12}{'Tokens':>10}\n"),
                self.timing_text.insert(tk.END, "-" * 32 + "\n"),
            ))

            self.root.after(0, lambda: self.log("Extracting..."))

            if self.mode.get() == "single":
                self.current_page_idx = 0
                # Custom range: if applied, override to specified subset
                if self.custom_range_applied:
                    try:
                        start = int(self.custom_range_from.get())
                        end = int(self.custom_range_to.get())
                    except Exception:
                        start = end = None
                    if start and end:
                        start = max(1, start)
                        end = min(len(self.pdf_pages), end)
                        if start <= end:
                            pages_to_process = list(range(start - 1, end))
                            self.log(f"Using custom page range: {start}-{end}")
                        else:
                            pages_to_process = []
                            self.log("Invalid custom range: start > end; no pages will be processed")
                    else:
                        pages_to_process = self.selected_pages if self.selected_pages else list(range(len(self.pdf_pages)))
                else:
                    pages_to_process = self.selected_pages if self.selected_pages else list(range(len(self.pdf_pages)))

                for i, page_idx in enumerate(pages_to_process):
                    if self.stop_flag:
                        break

                    self.root.after(0, lambda p=i+1, t=len(pages_to_process): self.log(f"Processing page {p}/{t}..."))

                    page_start = time.time()

                    if OLMCOCR_AVAILABLE and self.selected_files:
                        try:
                            image_base64 = render_pdf_to_base64png(self.selected_files[0], page_idx + 1, target_longest_image_dim=1288)
                            result = self.vlm.extract(image_base64=image_base64, prompt=prompt)
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            self.root.after(0, lambda err=str(e), t=tb: self.log_error(f"Render error: {err}\nTraceback: {t[-500:]}"))
                            try:
                                temp_path = Path(self.output_dir or ".") / f"temp_p{page_idx}.png"
                                self.pdf_pages[page_idx].save(temp_path)
                                result = self.vlm.extract(image_path=str(temp_path), prompt=prompt)
                                temp_path.unlink()
                            except Exception as e2:
                                self.root.after(0, lambda err=str(e2): self.log_error(f"Fallback also failed: {err}"))
                                continue
                    else:
                        temp_path = Path(self.output_dir or ".") / f"temp_p{page_idx}.png"
                        self.pdf_pages[page_idx].save(temp_path)
                        result = self.vlm.extract(image_path=str(temp_path), prompt=prompt)
                        temp_path.unlink()

                    if result is None:
                        self.root.after(0, lambda: self.log_error(f"VLM extraction returned None for page {page_idx + 1}"))
                        continue

                    page_dur = time.time() - page_start
                    page_tokens = result.get("total_tokens", 0)
                    result["page_number"] = page_idx + 1
                    result["duration_s"] = round(page_dur, 2)
                    page_timings.append({
                        "page": page_idx + 1,
                        "duration_s": round(page_dur, 2),
                        "tokens": page_tokens,
                    })

                    # Update timing tab and status log
                    timing_line = f"Page {page_idx+1:<4}  {page_dur:>8.1f}s  {page_tokens:>8} tok\n"
                    self.root.after(0, lambda ln=timing_line: (
                        self.timing_text.insert(tk.END, ln),
                        self.timing_text.see(tk.END),
                    ))
                    self.root.after(0, lambda p=page_idx+1, d=page_dur, tk_=page_tokens:
                        self.log(f"Page {p} done — {d:.1f}s, {tk_} tokens"))

                    self.extracted_data.append(result)
                    self.root.after(0, lambda idx=page_idx: self.view_page(idx))

            total_dur = time.time() - total_start
            total_tokens = sum(r.get("total_tokens", 0) for r in self.extracted_data)

            # Final row in timing tab
            summary_line = (
                f"{'-'*32}\n"
                f"{'TOTAL':<8}  {total_dur:>8.1f}s  {total_tokens:>8} tok\n"
            )
            self.root.after(0, lambda ln=summary_line: (
                self.timing_text.insert(tk.END, ln),
                self.timing_text.see(tk.END),
            ))

            # Update summary bar
            self.root.after(0, lambda: self.page_token_var.set(
                f"Total tokens: {total_tokens}  |  Total duration: {total_dur:.1f}s  |  Pages: {len(self.extracted_data)}"
            ))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(self.output_dir or ".") / f"extraction_{timestamp}.json"

            output_data = {
                "extraction_timestamp": timestamp,
                "prompt": prompt,
                "pages_processed": len(self.extracted_data),
                "total_tokens": total_tokens,
                "total_duration_s": round(total_dur, 2),
                "avg_duration_per_page_s": round(total_dur / max(len(self.extracted_data), 1), 2),
                "page_timings": page_timings,
                "results": [{k: v for k, v in r.items()} for r in self.extracted_data],
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            self.root.after(0, lambda: self.log(
                f"✓ Done! {len(self.extracted_data)} pages | {total_tokens} tokens | {total_dur:.1f}s total → {output_file.name}"))
        
        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log_error(f"Extraction error: {err}"))
        finally:
            self.root.after(0, self._extraction_done)

    def _extraction_done(self):
        self.extracting = False
        self.progress_bar.stop()
        self.extract_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        # Refresh post-process stats now that extraction is complete
        self._update_pp_doc_stats()

    def cmd_open_scal_rag(self):
        """Open integrated SCAL LangExtract + Offline RAG window.

        If extraction results already exist, pre-load them into the SCAL window.
        """
        try:
            # Lazy-load module on click to avoid startup hard-failure
            module_path = os.path.join(_CURRENT_DIR, "scal_langextract_rag_gui.py")
            if not os.path.exists(module_path):
                self.log_error(f"SCAL RAG module not found at: {module_path}")
                return

            spec = importlib.util.spec_from_file_location("scal_langextract_rag_gui", module_path)
            if spec is None or spec.loader is None:
                self.log_error("Unable to create import spec for scal_langextract_rag_gui.py")
                return
            scal_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scal_module)
            SCALAppLocal = getattr(scal_module, "SCALApp", None)
            if SCALAppLocal is None:
                self.log_error("SCALApp class not found in scal_langextract_rag_gui.py")
                return

            win = tk.Toplevel(self.root)
            win.title("SCAL LangExtract + Offline RAG")
            app = SCALAppLocal(win)

            # Pre-load compiled extraction text from current workflow if available
            if self.extracted_data:
                compiled = "\n\n".join(r.get("raw_response", "") for r in self.extracted_data)
                source_name = Path(self.selected_files[0]).name if self.selected_files else "olmocr_extracted_pages"
                app.load_text(compiled, source_name=source_name)
                app.build_rag()
                self.log("Opened SCAL RAG window with current extraction pre-loaded")
            else:
                self.log("Opened SCAL RAG window")
        except Exception as e:
            self.log_error(f"Failed to open SCAL RAG window: {e}\n{traceback.format_exc()}")

    def _apply_custom_range(self):
        try:
            s = self.custom_range_from.get().strip()
            t = self.custom_range_to.get().strip()
            if not s or not t:
                self.log("Custom range cleared: using full document")
                self.custom_range_applied = False
                return
            start = int(s)
            end = int(t)
            if start < 1 or end < 1:
                self.log("Custom range must be 1-based and >= 1")
                self.custom_range_applied = False
                return
            if len(self.pdf_pages) > 0:
                start = max(1, start)
                end = min(len(self.pdf_pages), end)
                if start > end:
                    self.log("Invalid range: start > end")
                    self.custom_range_applied = False
                    return
                self.custom_range_applied = True
                self.log(f"Custom page range applied: {start}-{end}")
            else:
                self.log("No PDF pages loaded yet; range will apply after loading.")
                self.custom_range_applied = False
        except Exception as e:
            self.log_error(f"Invalid range: {e}")
            self.custom_range_applied = False

    # ===== CHAT =====
    
    def send_chat(self):
        msg = self.chat_input.get().strip()
        if not msg:
            return
        
        self.chat_input.delete(0, tk.END)
        self.log_chat(f"You: {msg}", "user")
        
        thread = threading.Thread(target=self._process_chat, args=(msg,))
        thread.daemon = True
        thread.start()

    def _process_chat(self, msg):
        msg_lower = msg.lower()
        
        if "load vlm" in msg_lower:
            self.root.after(0, self.cmd_load_vlm)
            return
        if "load llm" in msg_lower:
            self.root.after(0, self.cmd_load_llm)
            return
        if msg_lower in ["extract", "start", "run"]:
            self.root.after(0, self.start_extraction)
            return
        if "stop" in msg_lower:
            self.root.after(0, self.stop_extraction)
            return
        if "help" in msg_lower:
            self.log_chat("Commands: 'load vlm', 'load llm', 'extract', 'stop', 'optimize'", "assistant")
            return
        
        if not self.llm or not self.llm.loaded:
            self.log_chat("Load LLM first: 'load llm'", "assistant")
            return
        
        try:
            response = self.llm.chat(msg)
            self.root.after(0, lambda r=response: self.log_chat(f"Assistant: {r}", "assistant"))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log_error(f"Chat error: {err}"))

    # ===== MODEL LOADING =====
    
    def cmd_load_vlm(self):
        if self.vlm and self.vlm.loaded:
            self.log("VLM already loaded")
            return
        
        self.log("Loading VLM (olmOCR-2-7B-1025-FP8)...")
        
        def load():
            try:
                self.vlm = VLMExtractor()
                result = self.vlm.load_model()
                self.root.after(0, lambda r=result: self._on_vlm_loaded(r))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_error(f"VLM Error: {err}"))
        
        threading.Thread(target=load, daemon=True).start()

    def _on_vlm_loaded(self, result):
        self.log(f"✓ {result}")
        self.update_model_status()

    def cmd_load_llm(self):
        if self.llm and self.llm.loaded:
            self.log("LLM already loaded")
            return
        
        self.log("Loading LLM (Qwen2.5-3B)...")
        
        def load():
            try:
                self.llm = IntelligentAssistant()
                result = self.llm.load_model()
                self.root.after(0, lambda r=result: self._on_llm_loaded(r))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_error(f"LLM Error: {err}"))
        
        threading.Thread(target=load, daemon=True).start()

    def _on_llm_loaded(self, result):
        self.log(f"✓ {result}")
        self.log_chat("Hi! I can help optimize prompts. Type 'help' for commands.", "assistant")
        self.update_model_status()

    def update_model_status(self):
        status = []
        if self.vlm and self.vlm.loaded:
            status.append("VLM")
        if self.llm and self.llm.loaded:
            status.append("LLM")
        
        if status:
            self.model_status.config(text=f"Models: {', '.join(status)}", foreground="green")
        else:
            self.model_status.config(text="Models: Not loaded", foreground="gray")

def _check_optional_deps():
    """Warn about optional but important packages at startup."""
    missing = []
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        missing.append("openpyxl  (Excel export)  →  pip install openpyxl")
    try:
        import pypdf  # noqa: F401
    except ImportError:
        missing.append("pypdf  (PDF page count)  →  pip install pypdf")
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "Optional Dependencies Missing",
            "Some features may not work:\n\n" + "\n".join(missing),
        )
        root.destroy()


def main():
    if not TRANSFORMERS_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing Dependencies", 
            "pip install torch transformers pillow pandas pypdf")
        return
    
    if not torch.cuda.is_available():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("CUDA Required", "olmOCR needs CUDA GPU")
        return

    _check_optional_deps()
    
    root = tk.Tk()
    app = OlmoCRAgenticGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
