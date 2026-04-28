olmOCR Trainer + Agentic GUI Bundle

This folder is a self-contained working bundle for:
- `olmocr_trainer.py`
- `olmocr_agentic_gui.py`

Included dependency files used by these scripts:
- `olmocr/prompts/__init__.py`
- `olmocr/prompts/prompts.py`
- `olmocr/prompts/anchor.py`
- `olmocr/data/renderpdf.py`
- `scal_langextract_rag_gui.py` (optional companion UI import for agentic GUI)

Environment launchers:
- `run_olmocr_trainer_conda.bat`
- `run_olmocr_agentic_gui_conda.bat`

Requirements file:
- `requirements_olmocr_agentic_gui.txt`

Notes:
- Install Poppler tools (`pdfinfo`, `pdftoppm`) for robust PDF page rendering.
- Keep trainer/agentic environments separate from inference/webapp envs to avoid torch conflicts.
