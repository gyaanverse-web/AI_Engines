import os
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any, List
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from testing_engine import get_embedding

# Explicitly use local Qdrant database to prevent .env remote crashes and preserve our NCERT data
qdrant_client = QdrantClient(path="qdrant_local_db")
COLLECTION_NAME = "ncert_local_test"

def search_ncert_solution(question_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Takes a question string, embeds it via NVIDIA, and retrieves
    the matching Semantic Concepts from Qdrant.
    """
    print(f"Generating NVIDIA embedding for search query...")
    
    # Step 1: Embed the search query
    try:
        query_vector = get_embedding(question_text)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return []
    
    # Step 2: Search Qdrant Collection
    print(f"Searching Qdrant database...")
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=0.60 # relaxed to 60% since we are semantic searching
    ).points
    
    # Step 3: Extract the Payload
    results = []
    for hit in search_results:
        results.append({
            "score": hit.score,
            "payload": hit.payload
        })
        
    return results

def extract_topic_from_question(question_text: str) -> str:
    """
    Uses NVIDIA LLM to extract the core NCERT Topic/Chapter from the question.
    This helps us search Qdrant for the syllabus specifically, rather than messy steps.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    prompt = f"""
    You are an AI syllabus mapper. Read the following question and identify the core NCERT Chapter or Topic it belongs to (e.g., "Set Theory", "Uniformly Accelerated Motion", "Quadratic Equations").
    Return ONLY a short string containing the Topic name. No extra text, no markdown.
    
    Question: {question_text}
    """
    
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=50
    )
    
    content = completion.choices[0].message.content
    topic = content.strip() if content else ""
    return topic if topic else question_text

def generate_board_solution(student_question: str, student_steps: List[Dict[str, str]] = None, retrieved_context: List[Dict[str, Any]] = None) -> str:
    """
    Feeds the student question, their handwritten steps, and retrieved NCERT concepts to NVIDIA LLM.
    Audits the student's solution against the Board constraints.
    """
    if retrieved_context is None:
        retrieved_context = []
        
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Build context string
    context_str = ""
    if retrieved_context:
        for i, res in enumerate(retrieved_context):
            payload = res['payload']
            context_str += f"\n--- NCERT Context {i+1} ---\n"
            context_str += f"Semantic Meaning: {payload.get('Semantic_Meaning')}\n"
            context_str += f"Formulas: {payload.get('Formulas')}\n"
            context_str += f"Raw Book Text: {payload.get('Raw_Text')}\n"
        
    # Calculate Confidence Factor
    confidence_percentage = 0.0
    if retrieved_context:
        avg_score = sum(res['score'] for res in retrieved_context) / len(retrieved_context)
        confidence_percentage = round(avg_score * 100, 2)
        
        confidence_rule = ""
        if confidence_percentage >= 80:
            confidence_rule = f"\n        [RAG CONFIDENCE: HIGH ({confidence_percentage}%)]. The provided NCERT Context is highly relevant. STRICTLY enforce these exact methods and do not deviate."
        else:
            confidence_rule = f"\n        [RAG CONFIDENCE: MEDIUM ({confidence_percentage}%)]. The provided NCERT Context is somewhat relevant. Use it as a strong guideline, but rely on your internal Board knowledge if it seems incomplete."

        system_prompt = rf"""You are a strict CBSE Board Examiner auditing a student's answer sheet.
        You will be provided with a student's question, the student's step-by-step mathematical solution, and relevant context extracted directly from the NCERT textbook.
        
        CRITICAL RULES:
        1. Verify if the mathematical METHOD used by the student exists in the provided NCERT context. Evaluate the student's steps strictly based on that syllabus.
        2. Identify exact errors in the student's solution. If an error is found, point it out and provide the correct step that should have followed.
        3. If the student uses JEE shortcuts, out-of-syllabus methods (methods not found in the NCERT context), or skips crucial steps, explicitly state: "This method/shortcut is not allowed in Boards. Marks deducted."
        4. Whether the student's solution is correct or incorrect, ALWAYS provide the complete canonical correct step-by-step solution. For each step, explicitly assign 'step marks' (e.g., 1/2 mark, 1 mark).
        5. Provide alternate solutions or concepts IF they are valid under the NCERT syllabus.
        6. ANTI-CIRCULAR LOGIC: Ensure that any Alternate Solutions suggested are logically sound and do NOT use circular reasoning (e.g., do not use a variable to solve step 1 that is only calculated in a later step).
        7. FORMATTING: Output strictly in plain text. Do NOT use LaTeX (no \( \) or $$). Use standard text for math (e.g., x^2, sqrt(2)). Do NOT use Markdown tables. Write clearly in plain text paragraphs.
        {confidence_rule}
        """
        
        steps_str = "\n".join([f"Step {s.get('stepId', '')}: {s.get('text', '')}" for s in student_steps]) if student_steps else "No steps provided."
        user_prompt = f"Student Question: {student_question}\n\nStudent Steps:\n{steps_str}\n\nNCERT Context:\n{context_str}"
        print(f"\nAsking NVIDIA LLM to audit based on {len(retrieved_context)} concepts (Confidence: {confidence_percentage}%)...")
    else:
        system_prompt = r"""You are a strict CBSE Board Examiner auditing a student's answer sheet.
        You have NOT been provided with specific NCERT text, so you must rely on your internal knowledge of the standard CBSE/State Board syllabus.
        
        CRITICAL RULES:
        1. Verify if the mathematical METHOD used by the student is a standard Board method. Evaluate the student's steps strictly based on standard syllabus.
        2. Identify exact errors in the student's solution. If an error is found, point it out and provide the correct step that should have followed.
        3. If the student uses JEE shortcuts, out-of-syllabus methods, or skips crucial steps, explicitly state: "This method/shortcut is not allowed in Boards. Marks deducted."
        4. Whether the student's solution is correct or incorrect, ALWAYS provide the complete canonical correct step-by-step solution. For each step, explicitly assign 'step marks' (e.g., 1/2 mark, 1 mark).
        5. Provide alternate solutions or concepts IF they are valid under the NCERT syllabus.
        6. ANTI-CIRCULAR LOGIC: Ensure that any Alternate Solutions suggested are logically sound and do NOT use circular reasoning (e.g., do not use a variable to solve step 1 that is only calculated in a later step).
        7. FORMATTING: Output strictly in plain text. Do NOT use LaTeX (no \( \) or $$). Use standard text for math (e.g., x^2, sqrt(2)). Do NOT use Markdown tables. Write clearly in plain text paragraphs.
        """
        steps_str = "\n".join([f"Step {s.get('stepId', '')}: {s.get('text', '')}" for s in student_steps]) if student_steps else "No steps provided."
        user_prompt = f"Student Question: {student_question}\n\nStudent Steps:\n{steps_str}"
        print("\nNo NCERT context found. Asking NVIDIA LLM to audit using strict CBSE Board internal knowledge...")
    
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=4000
    )
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    # This simulates a student uploading a picture of a question
    test_question = """Given the sets A = {1, 3, 5}, B = {2, 4, 6} and C = {0, 2, 4, 6, 8}, which of the
following may be considered as universal set (s) for all the three sets A, B and C
(i) {0, 1, 2, 3, 4, 5, 6}
(ii)
(iii)
{0,1,2,3,4,5,6,7,8,9,10}
(iv) {1,2,3,4,5,6,7,8}"""
    
    # Test data representing OCR output
    test_student_steps = [
        {"stepId": "1", "text": "Distance = sqrt((-3-2)^2 + (7-3)^2 + (2-5)^2)"},
        {"stepId": "2", "text": "Distance = sqrt(25 + 16 + 9)"},
        {"stepId": "3", "text": "Distance = sqrt(50)"}
    ]
    
    print(f"\nSTUDENT QUESTION: {test_question}")
    print("STUDENT STEPS:")
    for s in test_student_steps:
        print(f"Step {s['stepId']}: {s['text']}")
    print("")
    
    # 1. Retrieve Semantic Context
    results = search_ncert_solution(test_question, top_k=2)
    
    if not results:
        print("⚠️ No matching NCERT concepts found in the local database. Relying entirely on LLM's internal Board knowledge.")
    else:
        print(f"✅ Found {len(results)} relevant NCERT concepts (Max Score: {results[0]['score']:.4f})")
        
    # 2. Generate Board Solution
    final_answer = generate_board_solution(test_question, test_student_steps, results)
    
    print("\n" + "="*80)
    print("🧑‍🏫 BOARD EXAMINER EVALUATION (NVIDIA LLM)")
    print("="*80)
    print(final_answer)
    print("="*80)
