import os
import sys
import json
from dotenv import load_dotenv

from nvidia_ocr_engine import get_nvidia_json_ocr
from board_engine import search_ncert_solution, generate_board_solution, extract_topic_from_question

def evaluate_student_answer(image_path: str = None, pre_extracted_data: dict = None):
    print("=" * 80)
    print("🚀 STARTING END-TO-END BOARD EVALUATION PIPELINE")
    print("=" * 80)
    
    if pre_extracted_data:
        print("\n[1/4] Using Pre-Extracted Data (Bypassing OCR)...")
        ocr_result = pre_extracted_data
    else:
        if not image_path or not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return
            
        print(f"\n[1/4] Running NVIDIA OCR on {os.path.basename(image_path)}...")
        try:
            ocr_result = get_nvidia_json_ocr(image_path)
        except Exception as e:
            print(f"❌ OCR Engine Failed: {e}")
            return
            
    question = ocr_result.get("question", "")
    steps = ocr_result.get("ocr_data", [])
    ocr_confidence = ocr_result.get("ocr_confidence", "N/A")
    ocr_reason = ocr_result.get("ocr_confidence_reason", "No reason provided")
    
    if not question:
        print("❌ OCR could not detect a valid question in the image.")
        return
        
    print("\n📝 EXTRACTED STUDENT DATA:")
    print(f"OCR Confidence: {ocr_confidence}%")
    print(f"OCR Note: {ocr_reason}")
    
    # OCR Confidence Guard
    try:
        if ocr_confidence != "N/A" and float(ocr_confidence) < 50.0:
            print("\n❌ System Error: Handwriting is too messy to accurately grade.")
            print("Action: Pipeline halted. Please upload a clearer image.")
            return
    except ValueError:
        pass
    print(f"Question: {question}")
    print(f"Steps Found: {len(steps)}")
    for s in steps:
        print(f"  Step {s.get('stepId', '?')}: {s.get('text', '')}")
        
    print("\n[2/4] Extracting Core Topic from Question...")
    topic = extract_topic_from_question(question)
    print(f"Detected Topic: '{topic}'")
        
    print("\n[3/4] Searching Qdrant Database for official NCERT Syllabus info...")
    context_results = search_ncert_solution(topic, top_k=3)
    
    if not context_results:
        print("⚠️ No matching NCERT concepts found in the local database. Relying entirely on LLM's internal Board knowledge.")
    else:
        print(f"✅ Found {len(context_results)} relevant NCERT concepts (Max Score: {context_results[0]['score']:.4f})")
        
    print("\n[4/4] Generating Final Board Evaluation Report...")
    final_evaluation = generate_board_solution(question, steps, context_results)
    
    print("\n" + "="*80)
    print("🧑‍🏫 FINAL BOARD EXAMINER REPORT")
    print("="*80)
    print(final_evaluation)
    print("="*80)

if __name__ == "__main__":
    load_dotenv()
    
    # Check if a specific image was passed as a command line argument
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default test image based on the nvidia_ocr_engine.py test code
        img_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Test_Image9.jpeg')
        
    evaluate_student_answer(img_path)
