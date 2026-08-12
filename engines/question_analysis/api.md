# Question Analysis Engine API

The repository application mounts this engine at:

```text
/question_analysis
```

## Health check

```http
GET /question_analysis/
```

## Analyze question skill weightage

```http
POST /question_analysis/analyze
Content-Type: application/json
```

Request:

```json
{
  "question": "Calculate the simple interest on ₹5000 at 5% per annum for 2 years.",
  "chapter": "Simple Interest",
  "use_ml": true
}
```

`question` and `chapter` are required. `use_ml` is optional:

- `true` enables rule-plus-ML blending for the request.
- `false` forces rule-only scoring.
- omitted uses the server configuration.

Response:

```json
{
  "concept_based": 10,
  "formula_based": 35,
  "calculation_based": 50,
  "reasoning_based": 5,
  "memory_based": 0,
  "diagram_based": 0,
  "data_interpretation_based": 0,
  "proof_or_derivation_based": 0,
  "application_based": 0,
  "language_or_explanation_based": 0
}
```

The response always contains exactly ten category keys whose integer values total `100`. The `X-Question-Analysis-Mode` response header identifies whether rule-only or rule-plus-ML scoring was used.

Validation failures return HTTP `400` with an `error` field.
