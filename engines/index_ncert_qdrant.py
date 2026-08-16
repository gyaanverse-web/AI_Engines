import os
import json
import uuid
from typing import List, Dict, Any
import pypdf
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# We import the embedding function and qdrant_client from the testing engine
from testing_engine import get_embedding, QDRANT_COLLECTION_NAME, qdrant_client

EMBEDDING_VECTOR_SIZE = int(os.getenv("EMBEDDING_VECTOR_SIZE", "4096"))

# Use a specific test collection for NCERT data if not provided
COLLECTION_NAME = QDRANT_COLLECTION_NAME or "ncert_local_test"

def init_qdrant() -> QdrantClient:
    client = qdrant_client
    
    # Check if collection exists, if not create it
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating Qdrant collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_VECTOR_SIZE, 
                distance=Distance.COSINE
            )
        )
    return client

def extract_ncert_concepts(text_chunk: str) -> List[Dict[str, Any]]:
    """
    Uses NVIDIA Nemotron to extract semantic meaning, formulas, and concepts from a text chunk.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = """
    You are an expert curriculum developer and CBSE Board Examiner. I will provide a raw chunk of text from an NCERT textbook.
    Your task is to analyze the text and extract its core "Semantic Meaning", mathematical formulas, and concepts.
    
    If the text contains multiple distinct topics or questions, you can return multiple objects. If it's all one topic, return a single object.
    
    Output a JSON array matching this exact schema:
    [
      {
        "Semantic_Meaning": "A clear, high-level summary of the concepts, theories, and exact meaning of the text. Include the core question or problem if one exists.",
        "Concepts": ["Concept 1", "Concept 2"],
        "Formulas": ["Formula 1 (in LaTeX)", "Formula 2"],
        "Topic": "The chapter or topic name if discernible"
      }
    ]
    
    Output strictly valid JSON. Do not output markdown codeblocks. Just the raw JSON array.
    """

    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the text chunk:\n\n{text_chunk}"}
        ],
        temperature=0.3,
        max_tokens=4000
    )
    
    response_content = completion.choices[0].message.content
    if not response_content:
        return []
    
    response_text = response_content.strip()
    
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
        
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from LLM: {e}")
        return []

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extracts text from a PDF file page by page and returns it as a list of lines."""
    lines = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lines.extend(text.split('\n'))
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return lines

def ingest_ncert_file(file_path: str, max_chunks: int = None):
    """
    Reads the file, chunks it, structures via LLM, and embeds into Qdrant.
    """
    qdrant = init_qdrant()
    
    if file_path.lower().endswith('.pdf'):
        print(f"Extracting text from PDF: {os.path.basename(file_path)}")
        lines = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Simple chunking for demonstration (reading first N lines)
            # In production, we'd use a proper text splitter
            lines = f.readlines()
        
    print(f"Loaded {len(lines)} lines from {os.path.basename(file_path)}.")
    
    # We will process chunks of ~200 lines with an overlap of 50 lines 
    # to ensure concepts don't get cut in half at chunk boundaries.
    chunk_size = 200
    overlap = 50
    chunks_processed = 0
    
    step = chunk_size - overlap
    for i in range(0, len(lines), step):
        if max_chunks is not None and chunks_processed >= max_chunks:
            print(f"Reached max_chunks ({max_chunks}). Stopping early for testing.")
            break
            
        chunk_text = "".join(lines[i:i+chunk_size])
        print(f"\n--- Processing Chunk {chunks_processed + 1} ---")
        
        concepts_list = extract_ncert_concepts(chunk_text)
        print(f"Extracted {len(concepts_list)} semantic concepts from this chunk.")
        
        for concept_obj in concepts_list:
            semantic_meaning = concept_obj.get('Semantic_Meaning', '')
            if not semantic_meaning:
                continue
                
            print(f"Embedding Meaning: {semantic_meaning[:50]}...")
            
            # Embed the semantic meaning extracted by the LLM
            embed_text = semantic_meaning
            
            try:
                vector = get_embedding(embed_text)
                
                # Add the raw text to the payload so LLM has the exact NCERT source at runtime
                payload = concept_obj
                payload["Raw_Text"] = chunk_text
                
                # Insert into Qdrant
                point_id = str(uuid.uuid4())
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload
                        )
                    ]
                )
                print(f"Successfully upserted into Qdrant! ID: {point_id}")
            except Exception as e:
                print(f"Failed to embed/upsert: {e}")
                
        chunks_processed += 1

if __name__ == "__main__":
    # Explicitly target the Data/NCERT folder
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'Data', 'NCERT')
    
    if not os.path.exists(data_folder):
        print(f"Data folder not found at {data_folder}")
    else:
        files_to_process = []
        for root, dirs, files in os.walk(data_folder):
            for filename in files:
                if filename.lower().endswith('.txt') or filename.lower().endswith('.pdf'):
                    files_to_process.append(os.path.join(root, filename))
                
        if not files_to_process:
            print(f"No .txt or .pdf files found in {data_folder}")
        else:
            print(f"Found {len(files_to_process)} files to ingest.")
            for data_file in files_to_process:
                print(f"\n========== Starting Ingestion for {os.path.basename(data_file)} ==========")
                ingest_ncert_file(data_file)
                print(f"========== Completed Ingestion for {os.path.basename(data_file)} ==========\n")
