# AutoGrade AI: Intelligent Assessment Pipeline

An end-to-end automated grading pipeline for handwritten mathematical answers, leveraging the power of **Retrieval-Augmented Generation (RAG)**, **Vision LLMs**, and **Vector Databases**.

## Features

- **Intelligent OCR Engine:** Uses NVIDIA's Llama 3.2 90B Vision model via the NIM API to extract messy handwritten mathematical steps into structured JSON and LaTeX.
- **Syllabus RAG System:** Parses NCERT textbook PDFs and indexes semantic concepts into a local **Qdrant** vector database using NVIDIA Nemotron.
- **Automated Board Examiner:** Audits student answers against retrieved official syllabus methods, identifies errors, and provides the canonical step-by-step solution with step marks.

## Tech Stack

- **Languages:** Python
- **AI & Machine Learning:** Retrieval-Augmented Generation (RAG), Prompt Engineering, Large Language Models (LLMs), NVIDIA NIM APIs (Llama 3.2 Vision, Nemotron-3 30B)
- **Databases & Tools:** Qdrant (Vector DB), PyPDF, OpenAI Python SDK

## Setup & Local Execution

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt` (or manually install `openai`, `qdrant-client`, `python-dotenv`, `pypdf`).
4. Create a `.env` file in the `engines/` directory and add your NVIDIA API Key:
   ```
   NVIDIA_API_KEY=your_api_key_here
   ```
5. Run the pipeline: `python engines/run_pipeline.py`
