import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from run_pipeline import evaluate_student_answer

def test_shortcut_logic():
    load_dotenv()
    
    # Simulating a student using L'Hopital's Rule (a common JEE shortcut) for a basic CBSE Limit question
    fake_ocr_data = {
        "question": "Evaluate the limit as x approaches 2 for (x^2 - 4)/(x - 2)",
        "ocr_data": [
            {"stepId": "1", "text": "Using L'Hopital's rule, differentiate numerator and denominator"},
            {"stepId": "2", "text": "\\lim_{x \\to 2} \\frac{2x}{1}"},
            {"stepId": "3", "text": "= 4"}
        ]
    }
    
    print("\n--- RUNNING SHORTCUT SIMULATION ---")
    evaluate_student_answer(pre_extracted_data=fake_ocr_data)

if __name__ == "__main__":
    test_shortcut_logic()
