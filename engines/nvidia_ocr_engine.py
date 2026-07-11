import os
import json
import base64
import re
from typing import Dict, Any
from openai import OpenAI

def get_nvidia_json_ocr(image_path: str) -> Dict[str, Any]:
    """
    Takes an image of a student's handwritten math answer.
    Extracts the full question and the step-by-step mathematical calculations.
    Returns a strict JSON dictionary.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    if not client.api_key:
        raise ValueError("NVIDIA_API_KEY is missing from environment variables.")

    # Encode the image
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"

    prompt_text = """
You are an expert Math OCR Engine. 
Analyze the provided image containing handwritten or printed mathematical steps.

Extract the content and output a strictly valid JSON object matching this schema exactly. Do not output markdown code blocks (e.g., ```json), just output the raw JSON object.

{
  "question": "Extract the full main question text here as a string",
  "ocr_confidence": 95,
  "ocr_confidence_reason": "State why you gave this confidence score. Is the handwriting very messy, are some fractions unreadable, or is it clear?",
  "ocr_data": [
    {
      "stepId": "1",
      "text": "Extract the mathematical step in LaTeX format"
    },
    {
      "stepId": "2",
      "text": "Extract the next mathematical step in LaTeX format"
    }
  ]
}

CRITICAL RULES FOR MESSY HANDWRITING:
1. Include EVERY step written by the student, exactly as it appears. Do not skip or summarize lines.
2. Format all mathematical equations and formulas in clean LaTeX.
3. Pay extreme attention to poor handwriting. If a number looks like a 0 or an 8, use context. If a word is scribbled out, try your absolute best to transcribe what the student intended to write.
4. Do NOT miss subscripts, exponents (e.g., 10^-5, r^2), or unit conversions (e.g., cm to m).
5. If the student writes implicit conversions or jumps steps (e.g., d/2 = 1.00 cm / 2 = 0.005m), transcribe the whole line exactly as written.
6. Provide an accurate `ocr_confidence` score from 0 to 100 based on the legibility of the image.
7. Output MUST be valid JSON only. Do not add any conversational text.
8. CRITICAL: You MUST double-escape all LaTeX backslashes in the JSON output so it parses correctly (e.g. write \\\\pi instead of \\pi, and \\\\times instead of \\times).
"""

    completion = client.chat.completions.create(
        model="meta/llama-3.2-90b-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_string}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=4000,
        stream=False
    )
    
    # Optional: print reasoning for debugging
    reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
    if reasoning:
        print("--- Reasoning Trace ---")
        print(reasoning)
        print("-----------------------\n")
        
    raw_content = completion.choices[0].message.content.strip()
    
    # Robust JSON extraction: Look for a JSON block that actually contains the expected keys
    json_match = re.search(r"(\{\s*\"question\"[\s\S]*\})", raw_content)
    
    if json_match:
        response_text = json_match.group(1)
    else:
        # Fallback to last { to } if standard search fails
        start_idx = raw_content.find('{')
        end_idx = raw_content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response_text = raw_content[start_idx:end_idx+1]
        else:
            response_text = raw_content
        
    # Attempt to fix single unescaped backslashes which break Python's json.loads
    # e.g. \pi -> \\pi, but avoid changing already escaped \\pi
    response_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', response_text)
        
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        print("Raw response:")
        print(response_text)
        return {"question": "", "ocr_data": []}

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    test_image = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Test_Image9.jpeg')
    
    if os.path.exists(test_image):
        print(f"Running NVIDIA OCR on {os.path.basename(test_image)}...\n")
        try:
            result = get_nvidia_json_ocr(test_image)
            print("--- Final Extracted JSON ---")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test image not found at {test_image}")
