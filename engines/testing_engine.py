import os
import json
import re
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4")
NVIDIA_EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nv-embedqa-e5-v5",
)
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "high")
OPENAI_TEXT_VERBOSITY = os.getenv("OPENAI_TEXT_VERBOSITY", "low")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_VECTOR_SIZE = int(os.getenv("EMBEDDING_VECTOR_SIZE", "4096"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "50"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
NUMERIC_TOLERANCE = float(os.getenv("NUMERIC_TOLERANCE", "0.05"))

STANDARD_FORMULAS = {
    "kinematics_displacement": {
        "topic": "Kinematics",
        "canonical": r"S = ut + \frac{1}{2}at^2",
    }
}


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
if QDRANT_URL == "http://localhost:6333":
    qdrant_client = QdrantClient(path="qdrant_local_db")
else:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
    )


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _normalize_latex_text(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return cleaned

    replacements = {
        "∴": r"\therefore",
        "×": r"\times",
        "÷": r"\div",
        "−": "-",
        "√": r"\sqrt{}",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"\$\$(.*?)\$\$", r"\1", cleaned)
    cleaned = re.sub(r"\$(.*?)\$", r"\1", cleaned)
    cleaned = re.sub(r"(\d)(m/s\^2|m/s|kg|cm|mm|km|m|s)\b", r"\1 \\mathrm{\2}", cleaned)
    return cleaned


def _strip_latex_commands(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\left|\\right|\\therefore", "", cleaned)
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return cleaned


def _compact_math(text: str) -> str:
    return re.sub(r"\s+", "", _strip_latex_commands(text or "")).lower()


def _replace_frac(expr: str) -> str:
    pattern = re.compile(r"\\frac\(([^()]+)\)\(([^()]+)\)")
    previous = None
    current = expr
    while previous != current:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _latex_expr_to_python(expr: str) -> str:
    cleaned = _strip_latex_commands(expr)
    cleaned = cleaned.replace("\\times", "*").replace("\\div", "/")
    cleaned = cleaned.replace("^", "**")
    cleaned = _replace_frac(cleaned)
    cleaned = re.sub(r"\bAns\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", cleaned)
    cleaned = re.sub(r"(?<=[)])(?=[A-Za-z(])", "*", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\()", "*", cleaned)
    return cleaned


def _safe_eval_numeric(expr: str, variable_values: dict[str, float] | None = None) -> float | None:
    python_expr = _latex_expr_to_python(expr)
    if variable_values:
        for name, value in variable_values.items():
            python_expr = re.sub(rf"\b{re.escape(name)}\b", str(value), python_expr)

    if re.search(r"[^0-9a-zA-Z_+\-*/().]", python_expr):
        return None

    try:
        return float(eval(python_expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _extract_equation(text: str) -> tuple[str, str] | None:
    if "=" not in text:
        return None
    left, right = text.split("=", 1)
    left = left.strip()
    right = right.strip().rstrip(")")
    if not left or not right:
        return None
    return left, right


def _extract_assigned_variable(step_text: str) -> str | None:
    match = re.search(r"(?:^|\\therefore\s*)([A-Za-z])\s*=", step_text)
    if match:
        return match.group(1)
    return None


def _extract_last_number(text: str) -> float | None:
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", _strip_latex_commands(text))
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _extract_last_number_text(text: str) -> str | None:
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", _strip_latex_commands(text))
    if not numbers:
        return None
    return numbers[-1]


def _decimal_places(number_text: str | None) -> int | None:
    if not number_text or "." not in number_text:
        return 0 if number_text else None
    return len(number_text.split(".", 1)[1])


def _extract_unit(text: str) -> str | None:
    match = re.search(r"\\mathrm\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()

    plain_match = re.search(r"\b(m/s\^2|m/s|kg|cm|mm|km|m|s)\b", _strip_latex_commands(text))
    if plain_match:
        return plain_match.group(1).strip()
    return None


def _infer_expected_unit(question: str, local_context: str, step_text: str) -> str | None:
    combined = f"{question} {local_context} {step_text}".lower()
    if "acceleration" in combined:
        return "m/s^2"
    if "velocity" in combined or "speed" in combined:
        return "m/s"
    if "distance" in combined or "displacement" in combined or re.search(r"\bS\s*=", step_text):
        return "m"
    if "time" in combined:
        return "s"
    return None


def _solve_linear_value_from_equation(equation_text: str, variable: str) -> float | None:
    equation = _extract_equation(equation_text)
    if not equation:
        return None

    left_expr, right_expr = equation
    left_at_zero = _safe_eval_numeric(left_expr, {variable: 0})
    left_at_one = _safe_eval_numeric(left_expr, {variable: 1})
    right_at_zero = _safe_eval_numeric(right_expr, {variable: 0})
    right_at_one = _safe_eval_numeric(right_expr, {variable: 1})

    values = [left_at_zero, left_at_one, right_at_zero, right_at_one]
    if any(value is None for value in values):
        return None

    left_constant = left_at_zero
    left_coeff = left_at_one - left_at_zero
    right_constant = right_at_zero
    right_coeff = right_at_one - right_at_zero
    denominator = left_coeff - right_coeff
    if abs(denominator) < 1e-9:
        return None

    return (right_constant - left_constant) / denominator


def _find_expected_numeric_value(
    variable: str,
    local_context: str,
) -> float | None:
    lines = []
    for raw_line in (local_context or "").splitlines():
        _, _, content = raw_line.partition(":")
        cleaned = content.strip()
        if variable in cleaned and "=" in cleaned:
            lines.append(cleaned)

    for line in reversed(lines):
        value = _solve_linear_value_from_equation(line, variable)
        if value is not None:
            return value
    return None


def _format_number(value: float, decimals: int | None = None) -> str:
    if decimals is not None:
        rounded = round(value, decimals)
        if decimals == 0:
            return str(int(round(rounded)))
        return f"{rounded:.{decimals}f}"

    rounded = round(value, 4)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _make_result(
    status: str,
    description: str,
    *,
    step_weight: float | None = None,
) -> dict[str, Any]:
    result = {
        "step_status": status,
        "description": description,
    }
    if step_weight is not None:
        result["step_weight"] = step_weight
    return result


def _apply_rule_override(
    step_result: dict[str, Any],
    rule_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not rule_result:
        return step_result

    step_result["step_status"] = rule_result["step_status"]
    step_result["description"] = rule_result["description"]
    if "step_weight" in rule_result:
        step_result["step_weight"] = rule_result["step_weight"]
    return _align_step_result(step_result)


def _is_heading_or_setup_line(step_text: str) -> bool:
    normalized = _compact_math(step_text)
    plain = _normalize_whitespace(_strip_latex_commands(step_text)).lower()
    setup_phrases = [
        "given that",
        "we know that",
        "find",
        "determine",
        "final acceleration",
    ]
    if any(phrase in plain for phrase in setup_phrases):
        return True
    if plain.endswith("?"):
        return True
    return normalized in {"", "ans:-giventhat", "weknowthat"}


def _step_type_check(step_text: str) -> dict[str, Any] | None:
    plain = _normalize_whitespace(_strip_latex_commands(step_text))
    lowered = plain.lower()
    if _is_heading_or_setup_line(step_text):
        if "final acceleration" in lowered or lowered.endswith("?"):
            return _make_result(
                "incomplete",
                "Missing: solution step; Correct step: start solving for acceleration",
                step_weight=0.1,
            )
        if "given that" in lowered or "we know that" in lowered:
            return _make_result(
                "incomplete",
                "Missing: actual setup values; Correct step: list the known quantities",
                step_weight=0.1,
            )
        return _make_result(
            "incomplete",
            "Missing: mathematical step; Correct step: continue the solution",
            step_weight=0.1,
        )
    return None


def _formula_checker(step_text: str, question: str, local_context: str) -> dict[str, Any] | None:
    normalized = _compact_math(step_text)
    combined = f"{question} {local_context} {step_text}".lower()

    if "acceleration" in combined and "s=ut+" in normalized and "at^2" in normalized:
        has_half = any(token in normalized for token in [r"\frac(1)(2)", "1/2", "0.5"])
        if not has_half:
            return _make_result(
                "wrong",
                r"Error: the displacement formula misses the \frac{1}{2} factor on at^2; Correct step: use S = ut + \frac{1}{2}at^2",
                step_weight=0.7,
            )
        return _make_result(
            "right",
            "The displacement formula is correct",
            step_weight=0.7,
        )

    return None


def _apply_deterministic_step_checks(
    step_result: dict[str, Any],
    question: str,
    step_text: str,
    local_context: str,
) -> dict[str, Any]:
    variable = _extract_assigned_variable(step_text)
    reported_value = _extract_last_number(step_text)
    reported_value_text = _extract_last_number_text(step_text)
    reported_decimals = _decimal_places(reported_value_text)
    expected_unit = _infer_expected_unit(question, local_context, step_text)
    reported_unit = _extract_unit(step_text)

    if variable and reported_value is not None:
        expected_value = _find_expected_numeric_value(variable, local_context)
        if expected_value is not None:
            if abs(reported_value - expected_value) <= NUMERIC_TOLERANCE:
                if expected_unit and not reported_unit:
                    step_result["step_status"] = "incomplete"
                    step_result["description"] = (
                        f"Missing: the unit {expected_unit} is not written; Correct step: write \\therefore {variable} = {_format_number(expected_value, reported_decimals)}\\ \\mathrm{{{expected_unit}}}"
                    )
                elif expected_unit and reported_unit and reported_unit != expected_unit:
                    step_result["step_status"] = "wrong"
                    step_result["description"] = (
                        f"Error: the numerical value is correct, but the unit {reported_unit} is incorrect; Correct step: write \\therefore {variable} = {_format_number(expected_value, reported_decimals)}\\ \\mathrm{{{expected_unit}}}"
                    )
                else:
                    step_result["step_status"] = "right"
                    step_result["description"] = ""
            else:
                step_result["step_status"] = "wrong"
                correction = _format_number(expected_value, reported_decimals)
                if expected_unit:
                    step_result["description"] = (
                        f"Error: the final calculation gives the wrong value for {variable}; Correct step: write \\therefore {variable} = {correction}\\ \\mathrm{{{expected_unit}}}"
                    )
                else:
                    step_result["description"] = (
                        f"Error: the final calculation gives the wrong value for {variable}; Correct step: write \\therefore {variable} = {correction}"
                    )

    return _align_step_result(step_result)


def _shorten_description(description: str) -> str:
    cleaned = _normalize_whitespace(description)
    if not cleaned:
        return cleaned

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    first_sentence = parts[0].strip()
    first_sentence = first_sentence.rstrip(" .,!?:;")
    if len(first_sentence) <= 120:
        return first_sentence

    truncated = first_sentence[:120]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(" .,!?:;")


def _align_step_result(step_result: dict[str, Any]) -> dict[str, Any]:
    status = str(step_result.get("step_status", "")).strip().lower()
    description = _shorten_description(str(step_result.get("description", "")))

    if status == "right":
        step_result["description"] = ""
        step_result["step_status"] = status
        return step_result

    if status in {"incomplete", "unknown"}:
        step_result["description"] = description
        step_result["step_status"] = status
        return step_result

    if not description:
        step_result["description"] = description
        return step_result

    lowered = description.lower()
    if status == "wrong" and any(
        phrase in lowered
        for phrase in ["numeric value is correct", "is correct but", "correct but the unit is missing"]
    ):
        status = "incomplete"
        description = ""
    elif status == "wrong" and not any(
        token in lowered
        for token in ["should", "correct", "instead", "use", "must be", "replace", "missing"]
    ):
        corrected_step = step_result.get("text", "").strip()
        if corrected_step:
            description = f"Error: {description}; Correct step: write {corrected_step}"
        else:
            description = f"Error: {description}; Correct step: write the corrected mathematical step"

    step_result["step_status"] = status
    step_result["description"] = description
    return step_result


def _build_local_step_context(
    ocr_data: list[dict[str, Any]],
    current_index: int,
    window: int = 2,
) -> str:
    nearby_steps = []
    start = max(0, current_index - window)
    end = min(len(ocr_data), current_index + window + 1)

    for index in range(start, end):
        if index == current_index:
            continue
        step = ocr_data[index]
        step_id = step.get("stepId", str(index + 1))
        step_text = _normalize_latex_text(step.get("text", ""))
        if step_text:
            nearby_steps.append(f"{step_id}: {step_text}")

    return "\n".join(nearby_steps)


def _needs_numeric_verification(step_text: str) -> bool:
    normalized = step_text.lower()
    has_number = bool(re.search(r"\d", step_text))
    answer_hint = any(token in normalized for token in ["ans", "therefore", r"\therefore", "="])
    return has_number and answer_hint


def _verify_numeric_step(
    step_result: dict[str, Any],
    question: str,
    step_text: str,
    local_context: str,
) -> dict[str, Any]:
    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        # Previous config for rollback:
        # temperature=OPENAI_TEMPERATURE,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON-only numeric verifier for math and science steps. "
                    "Verify the numeric correctness of the step using the question and nearby steps. "
                    "Check the numerical value separately from the unit. "
                    "If the number is correct but the unit is missing, step_status must be incomplete. "
                    "If the number is wrong, step_status must be wrong. "
                    "If the number and unit are both correct, step_status must be right. "
                    "Return a short description that clearly names the issue, such as missing unit or calculation error."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"original_question: {_normalize_latex_text(question or 'Not provided')}\n"
                    f"step_text: {step_text}\n\n"
                    f"nearby_steps:\n{local_context or 'Not provided'}\n\n"
                    "Return only valid JSON matching the schema."
                ),
            },
        ],
        text={
            "verbosity": OPENAI_TEXT_VERBOSITY,
            "format": {
                "type": "json_schema",
                "name": "numeric_step_verification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "step_status": {
                            "type": "string",
                            "enum": ["right", "wrong", "unknown", "incomplete"],
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["step_status", "description"],
                },
            }
        },
    )

    verification = json.loads(response.output_text)
    step_result["step_status"] = verification["step_status"]
    step_result["description"] = verification["description"]
    return _align_step_result(step_result)


def create_qdrant_collection(collection_name: str = QDRANT_COLLECTION_NAME):
    if qdrant_client.collection_exists(collection_name=collection_name):
        return collection_name

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    return collection_name


def chunk_document_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks = []
    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunks.append(cleaned_text[start:end])
        start += chunk_size - overlap

    return chunks


def get_embedding(text: str):
    response = nvidia_client.embeddings.create(
        model=NVIDIA_EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
        extra_body={"input_type": "passage"}
    )
    return response.data[0].embedding


def get_embeddings(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE):
    if not texts:
        return []

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = nvidia_client.embeddings.create(
            model=NVIDIA_EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": "passage"}
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def index_documents(
    documents: list[dict[str, Any]],
    collection_name: str = QDRANT_COLLECTION_NAME,
):
    create_qdrant_collection(collection_name)

    chunk_records = []
    for document in documents:
        document_id = document.get("document_id") or str(uuid.uuid4())
        document_text = document.get("text", "")
        metadata = document.get("metadata", {})

        for chunk_index, chunk_text in enumerate(chunk_document_text(document_text)):
            chunk_records.append(
                {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

    points = []
    if chunk_records:
        embeddings = get_embeddings([record["text"] for record in chunk_records])
        for record, embedding in zip(chunk_records, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=record,
                )
            )

        for start in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
            batch = points[start:start + QDRANT_UPSERT_BATCH_SIZE]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

    return {"collection_name": collection_name, "indexed_chunks": len(points)}


def extract_text_file_content(file_path: str):
    return Path(file_path).read_text(encoding="utf-8").strip()


def index_text_documents(
    document_paths: list[str],
    collection_name: str = QDRANT_COLLECTION_NAME,
):
    documents = []
    print(f"Starting TXT indexing for {len(document_paths)} documents")

    for document_path in document_paths:
        print(f"Processing document: {document_path}")
        text_path = Path(document_path)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {document_path}")

        document_text = extract_text_file_content(str(text_path))
        if not document_text:
            continue

        documents.append({
            "document_id": text_path.stem,
            "text": document_text,
            "metadata": {
                "source_path": str(text_path),
                "file_name": text_path.name,
            },
        })

    return index_documents(
        documents=documents,
        collection_name=collection_name,
    )


def retrieve_relevant_chunks(
    query: str,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
):
    query_vector = get_embedding(query)
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": point.score,
            "document_id": point.payload.get("document_id"),
            "chunk_index": point.payload.get("chunk_index"),
            "text": point.payload.get("text", ""),
            "metadata": point.payload.get("metadata", {}),
        }
        for point in search_result.points
    ]


def generate_document_answer(
    question: str,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
):
    relevant_chunks = retrieve_relevant_chunks(
        query=question,
        collection_name=collection_name,
        top_k=top_k,
    )

    context = "\n\n".join(
        f"[Chunk {chunk['chunk_index']} | Score {chunk['score']:.4f}]\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        # Previous config for rollback:
        # temperature=OPENAI_TEMPERATURE,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        text={"verbosity": OPENAI_TEXT_VERBOSITY},
        input=[
            {
                "role": "system",
                "content": (
                    "You are a document-based RAG assistant. Answer only from the "
                    "provided context. If the context does not contain the answer, "
                    "say that the document does not have enough information."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
    )

    return {
        "question": question,
        "answer": response.output_text,
        "sources": relevant_chunks,
    }


def evaluate_ocr_steps_with_rag(
    ocr_data: list[dict[str, Any]],
    question: str = "",
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
):
    step_results = []
    question = _normalize_latex_text(question.strip())

    for index, step in enumerate(ocr_data):
        step_id = step.get("stepId", str(uuid.uuid4()))
        step_text = _normalize_latex_text(step.get("text", "").strip())

        if not step_text:
            continue

        local_context = _build_local_step_context(ocr_data, index)
        retrieval_query = (
            f"Question: {question}\nStep: {step_text}\nNearby steps:\n{local_context}"
            if question
            else f"Step: {step_text}\nNearby steps:\n{local_context}"
        )
        relevant_chunks = retrieve_relevant_chunks(
            query=retrieval_query,
            collection_name=collection_name,
            top_k=top_k,
        )
        context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)

        response = openai_client.responses.create(
            model=OPENAI_CHAT_MODEL,
            # Previous config for rollback:
            # temperature=OPENAI_TEMPERATURE,
            reasoning={"effort": OPENAI_REASONING_EFFORT},
            input=[
                {
                    "role": "system",
"content": (
    "You are a strict JSON-only evaluator for step-by-step student solutions. "
    
    "INPUT FORMAT: "
    "You will receive a JSON object containing a 'question' and an 'ocr_data' array. "
    "Each item in 'ocr_data' represents one step of the student's solution. "
    
    "TASK: "
    "Evaluate EACH step but use the question and nearby steps to understand intent and flow. "
    "Return output in EXACT same JSON schema as required, without changing any field names. "
    
    "KNOWLEDGE USAGE: "
    "Use the question as the primary source of truth. "
    "Use provided context (if any) as supporting knowledge. "
    "Do not rely on external knowledge if it conflicts with the question. "
    
    "INTERPRETATION RULES: "
    "The OCR text is normalized into LaTeX-friendly notation. "
    "Always interpret expressions mathematically (not as plain text). "
    "Use  steps to understand continuity (substitution, simplification, progression). "
    
    "VALIDATION RULES: "
    "Check step intent, logical flow, topic, and whether the step appears mathematically meaningful. "
    "You may comment on likely formula, substitution, arithmetic, sign, or logic issues, but you are not the final authority for numeric or unit correctness. "
    
    "FORMULA CHECK: "
    "Flag obvious formula misuse when directly visible in the current step. "
    
    "NUMERIC AND UNIT CHECK: "
    "A deterministic validation layer will make the final numeric and unit decision. "
    "Do not overrule a mathematically plausible step just because the final numeric answer is not fully shown. "
    
    "STEP CLASSIFICATION: "
    "step_status must be exactly one of: right, wrong, unknown, incomplete. "
    
    "Use right → correct step that advances solution. "
    "Use wrong → incorrect formula, substitution, sign, or clear logic error. "
    "Use incomplete → partial step, setup, missing unit, or unfinished calculation. "
    "Use unknown → OCR or context too unclear to judge. "
    
    "Never mark headings, given statements, or intent lines as wrong. "
    
    "WEIGHT ASSIGNMENT: "
    "step_weight must be between 0 and 1. "
    "Use low weight (0.05–0.2) for setup steps. "
    "Use high weight (0.8–1.0) for core solving steps. "
    
    "FIELDS REQUIREMENT: "
    "topic → academic topic (e.g., Kinematics, Algebra). "
    "step_understanding → one concise line explaining student intent. "
    
    "DESCRIPTION RULES (VERY STRICT): "
    "description for wrong steps should be detailed enough to explain the exact mistake and the fix in one or two concise sentences. "
    "It MUST match step_status exactly and never contradict it. "
    
    "If step_status is right → keep description minimal. "
    
    "If step_status is wrong → "
    "format EXACTLY as: "
    "'Error: <exact issue>; Correct step: <correction>' "
    "State what part is wrong, such as formula, substitution, sign, arithmetic, or unit, and give the corrected step explicitly. "
    
    "If step_status is incomplete → "
    "format EXACTLY as: "
    "'Missing: <missing part>; Correct step: <what should be added>' "
    "State clearly why the step is incomplete and what is missing. "
    
    "If step_status is unknown → clearly state what is unclear. "
    
    "IMPORTANT: "
    "Do NOT give long explanations. "
    "Do NOT repeat reasoning. "
    "Do NOT use vague phrases like 'incorrect step'. "
    "Be precise and technical. "
    
    "OUTPUT FORMAT: "
    "Return ONLY valid JSON. "
    "Do NOT change schema. "
    "Do NOT add extra fields. "
),
                },
                {
                    "role": "user",
                    "content": (
                        f"original_question: {question or 'Not provided'}\n"
                        f"stepId: {step_id}\n"
                        f"text: {step_text}\n\n"
                        f"nearby_steps:\n{local_context or 'Not provided'}\n\n"
                        f"Context:\n{context}\n\n"
                        "Return only valid JSON matching the schema."
                    ),
                },
            ],
            text={
                "verbosity": OPENAI_TEXT_VERBOSITY,
                "format": {
                    "type": "json_schema",
                    "name": "step_rag_evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "stepId": {"type": "string"},
                            "text": {"type": "string"},
                            "step_status": {
                                "type": "string",
                                "enum": ["right", "wrong", "unknown", "incomplete"],
                            },
                            "step_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "topic": {"type": "string"},
                            "step_understanding": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": [
                            "stepId",
                            "text",
                            "step_status",
                            "step_weight",
                            "topic",
                            "step_understanding",
                            "description",
                        ],
                    },
                }
            },
        )

        step_result = _align_step_result(json.loads(response.output_text))
        step_result = _apply_rule_override(step_result, _step_type_check(step_text))
        step_result = _apply_rule_override(
            step_result,
            _formula_checker(step_text, question, local_context),
        )
        if _needs_numeric_verification(step_text):
            step_result = _verify_numeric_step(
                step_result=step_result,
                question=question,
                step_text=step_text,
                local_context=local_context,
            )
        step_result = _apply_deterministic_step_checks(
            step_result=step_result,
            question=question,
            step_text=step_text,
            local_context=local_context,
        )

        step_results.append(step_result)

    return {"response": step_results}
