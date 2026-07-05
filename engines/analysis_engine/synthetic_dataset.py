import json
from itertools import product
from pathlib import Path

from .categories import CATEGORY_KEYS
from .config import DEFAULT_DATASET_PATH
from .question_skill_weightage import analyze_question_skill_weightage


QUESTION_TEMPLATES = [
    {
        "chapter": "Simple Interest",
        "questions": [
            "Calculate the simple interest on ₹{principal} at {rate}% per annum for {time} years.",
            "Find the amount when the principal is ₹{principal}, rate is {rate}% and time is {time} years.",
            "Determine the simple interest for ₹{principal} invested for {time} years at {rate}%.",
        ],
        "values": {
            "principal": [1200, 5000, 9200],
            "rate": [4, 5, 8],
            "time": [2, 3, 5],
        },
    },
    {
        "chapter": "Compound Interest",
        "questions": [
            "Calculate the compound interest on ₹{principal} at {rate}% for {time} years.",
            "Find the amount compounded annually for ₹{principal} at {rate}% over {time} years.",
            "Compute the compound interest when principal is ₹{principal}, rate is {rate}% and time is {time} years.",
        ],
        "values": {
            "principal": [2500, 7000, 12500],
            "rate": [5, 6, 10],
            "time": [2, 3, 4],
        },
    },
    {
        "chapter": "Mensuration",
        "questions": [
            "Find the area of a circle with radius {radius} cm.",
            "Calculate the volume of a cuboid of length {length} cm, breadth {breadth} cm and height {height} cm.",
            "Determine the perimeter of a rectangle of sides {length} cm and {breadth} cm.",
        ],
        "values": {
            "radius": [7, 10, 14],
            "length": [8, 12, 15],
            "breadth": [4, 6, 9],
            "height": [3, 5, 8],
        },
    },
    {
        "chapter": "Photosynthesis",
        "questions": [
            "Explain the process of photosynthesis in plants.",
            "Describe why sunlight is necessary for photosynthesis.",
            "Write a note on the effect of chlorophyll in photosynthesis.",
        ],
        "values": {},
    },
    {
        "chapter": "Grammar",
        "questions": [
            "Define noun and describe its kinds with examples.",
            "Explain the difference between simple present tense and present continuous tense.",
            "Write a paragraph using adjectives correctly.",
        ],
        "values": {},
    },
    {
        "chapter": "Proof",
        "questions": [
            "Prove that the opposite angles of a cyclic quadrilateral are supplementary.",
            "Show that the sum of the interior angles of a triangle is 180 degrees.",
            "Derive the formula for the area of a triangle.",
        ],
        "values": {},
    },
    {
        "chapter": "Data Interpretation",
        "questions": [
            "Study the following table and answer which year had the highest sales.",
            "Observe the graph given below and analyze the data trend.",
            "Study the chart and compare the values shown in the data.",
        ],
        "values": {},
    },
    {
        "chapter": "Physics",
        "questions": [
            "Draw and label the ray diagram for a concave mirror.",
            "Draw the circuit diagram and label its components.",
            "Explain with a figure how a periscope works.",
        ],
        "values": {},
    },
    {
        "chapter": "Application of Percentage",
        "questions": [
            "A shopkeeper gives {discount}% discount on a bag priced at ₹{price}. Find the sale price in a real life situation.",
            "In daily life, how do you calculate profit percentage when the cost price is ₹{cost} and selling price is ₹{sell}?",
            "A practical case study: find the loss percentage when an item bought for ₹{cost} is sold for ₹{sell}.",
        ],
        "values": {
            "discount": [10, 15, 20],
            "price": [500, 1200, 2400],
            "cost": [800, 1000, 2500],
            "sell": [720, 1100, 2300],
        },
    },
    {
        "chapter": "Reasoning",
        "questions": [
            "Analyze the situation and justify which is correct if both statements are given.",
            "Compare the two cases and infer the best explanation.",
            "Assertion reason: choose the correct conclusion because both statements may be true.",
        ],
        "values": {},
    },
]


def _build_questions(template_group: dict[str, object]) -> list[str]:
    templates = template_group["questions"]
    values = template_group["values"]
    if not values:
        return list(templates)

    value_keys = list(values.keys())
    combinations = product(*(values[key] for key in value_keys))
    built_questions = []

    for template in templates:
        for combination in combinations:
            payload = dict(zip(value_keys, combination))
            built_questions.append(template.format(**payload))

    return built_questions


def build_synthetic_dataset_records() -> list[dict[str, object]]:
    records = []

    for template_group in QUESTION_TEMPLATES:
        chapter = template_group["chapter"]
        for question in _build_questions(template_group):
            labels = analyze_question_skill_weightage(
                {
                    "question": question,
                    "chapter": chapter,
                },
                use_ml=False,
            )
            records.append(
                {
                    "question": question,
                    "chapter": chapter,
                    "labels": {category: int(labels[category]) for category in CATEGORY_KEYS},
                }
            )

    return records


def write_synthetic_dataset(path: Path | None = None) -> Path:
    output_path = path or DEFAULT_DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_synthetic_dataset_records()
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path
