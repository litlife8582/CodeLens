import os
import json
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
from langchain_openai import ChatOpenAI

load_dotenv()


# ============================================================
# Evaluator LLM
# ============================================================

evaluator_llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0,
)


# ============================================================
# Data Schemas
# ============================================================

class ClaimVerification(BaseModel):
    statement: str = Field(
        description="The atomic claim extracted from the answer"
    )
    is_supported: bool = Field(
        description="True if the retrieved context directly supports the claim"
    )
    reasoning: str = Field(
        description="Brief explanation of why the claim is or is not supported"
    )


class RagasExtraction(BaseModel):
    claims: List[ClaimVerification] = Field(
        description="All atomic claims extracted from the answer"
    )


class FaithJudgeClassification(BaseModel):
    unsupported_claim: str
    severity: Literal["Unwanted", "Benign", "Questionable"]
    diagnosis: str = Field(
        description="Explanation of the impact on codebase logic"
    )


class FaithJudgeBatch(BaseModel):
    evaluations: List[FaithJudgeClassification]


# ============================================================
# JSON Utilities
# ============================================================

def extract_json_object(raw_text: str) -> dict:
    """
    Extract the first valid JSON object from an LLM response.

    Handles:
    - plain JSON
    - ```json ... ```
    - accidental text before/after JSON
    """

    if not raw_text:
        raise ValueError("Evaluator returned an empty response.")

    raw_text = raw_text.strip()

    # Remove Markdown code fences
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    raw_text = raw_text.strip()

    # First attempt: entire response is JSON
    try:
        parsed = json.loads(raw_text)

        if not isinstance(parsed, dict):
            raise ValueError("Evaluator JSON is not an object.")

        return parsed

    except json.JSONDecodeError:
        pass

    # Second attempt: find JSON object embedded in other text
    start = raw_text.find("{")

    if start == -1:
        raise ValueError(
            f"No JSON object found in evaluator response: {raw_text[:300]}"
        )

    # Try progressively larger closing positions.
    for end in range(len(raw_text), start, -1):
        candidate = raw_text[start:end].strip()

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Could not parse valid JSON from evaluator response: "
        f"{raw_text[:500]}"
    )


def invoke_json_evaluator(prompt: str, label: str, retries: int = 2) -> dict:
    """
    Invoke the evaluator without LangChain structured-output mode.

    This is intentionally used because openrouter/free can return
    non-JSON text even when structured output is requested.
    """

    last_error = None

    for attempt in range(1, retries + 1):

        try:
            response = evaluator_llm.invoke(prompt)

            raw = response.content

            # Some providers may return a list of content blocks.
            if isinstance(raw, list):
                parts = []

                for item in raw:
                    if isinstance(item, dict):
                        if "text" in item:
                            parts.append(str(item["text"]))
                    else:
                        parts.append(str(item))

                raw = "".join(parts)

            raw = str(raw).strip()

            print(
                f"\n{label} evaluator attempt "
                f"{attempt}/{retries}"
            )

            print("Evaluator raw response:")
            print(raw[:1000])

            return extract_json_object(raw)

        except Exception as e:
            last_error = e

            print(
                f"\n[WARNING] {label} evaluator attempt "
                f"{attempt} failed: {e}"
            )

            if attempt < retries:
                print("Retrying evaluator...")

    raise RuntimeError(
        f"{label} evaluator failed after {retries} attempts. "
        f"Last error: {last_error}"
    )


# ============================================================
# Tier 1: RAGAS Atomic Faithfulness
# ============================================================

def compute_ragas_faithfulness(
    query: str,
    context: str,
    answer: str
) -> dict:

    ragas_prompt = f"""
You are an objective evaluator measuring factual faithfulness
of an AI answer about a software codebase.

Your task is ONLY to evaluate the Generated Answer against the
Retrieved Context.

IMPORTANT RULES:

1. Break the Generated Answer into atomic claims.
2. A claim is supported ONLY when the Retrieved Context directly
   provides enough evidence for that claim.
3. Do NOT use outside knowledge.
4. Do NOT assume facts that are not present in the Retrieved Context.
5. If a claim cannot be verified from the Retrieved Context,
   mark is_supported as false.
6. Do not reward plausible guesses.
7. Do not punish the answer for omitting information.
8. Extract every factual claim that appears in the answer.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return ONLY valid JSON.

Required format:

{{
  "claims": [
    {{
      "statement": "atomic claim",
      "is_supported": true,
      "reasoning": "brief evidence-based explanation"
    }}
  ]
}}

Do not use Markdown.
Do not use code fences.
Do not output commentary.
Do not output safety labels.
Do not output anything before or after the JSON.
"""

    data = invoke_json_evaluator(
        ragas_prompt,
        "RAGAS",
        retries=2
    )

    try:
        result = RagasExtraction.model_validate(data)

    except ValidationError as e:
        print("\n❌ RAGAS JSON failed schema validation.")
        print(data)
        raise RuntimeError(
            f"RAGAS evaluator returned invalid schema: {e}"
        )

    total_claims = len(result.claims)

    verified_claims = [
        c for c in result.claims
        if c.is_supported
    ]

    unverified_claims = [
        c for c in result.claims
        if not c.is_supported
    ]

    score = (
        len(verified_claims) / total_claims
        if total_claims > 0
        else 1.0
    )

    return {
        "faithfulness_score": round(score, 3),
        "total_claims": total_claims,
        "verified_count": len(verified_claims),
        "unverified_claims": unverified_claims
    }


# ============================================================
# Tier 2: FaithJudge Diagnosis
# ============================================================

def classify_hallucinations_faithjudge(
    context: str,
    unverified_claims: list
) -> List[FaithJudgeClassification]:

    if not unverified_claims:
        return []

    claims_text = "\n".join(
        [
            (
                f'- Unsupported claim: "{c.statement}"\n'
                f'  Evidence gap: {c.reasoning}'
            )
            for c in unverified_claims
        ]
    )

    faithjudge_prompt = f"""
You are an expert software QA evaluator implementing the
FaithJudge hallucination taxonomy.

You will classify EVERY unsupported claim.

The classification MUST be exactly one of:

"Unwanted"
"Benign"
"Questionable"

DEFINITIONS:

Unwanted:
A hallucination that contradicts or invents concrete codebase
facts and could affect execution logic.

Examples:
- fabricated imports
- non-existent dependencies
- non-existent components
- non-existent props
- incorrect function calls
- fabricated files
- fabricated APIs

Benign:
A harmless statement that does not materially affect execution
logic.

Examples:
- generic stylistic descriptions
- harmless commentary
- general best-practice observations

Questionable:
A claim whose truth cannot be determined from the retrieved
context alone, without assuming additional codebase information.

IMPORTANT:

- Classify EVERY unsupported claim.
- Do NOT invent additional claims.
- Do NOT use outside knowledge.
- Judge only using the Retrieved Context.
- Do not use Markdown.
- Do not output commentary.
- Do not output safety labels.

RETRIEVED CONTEXT:
{context}

UNSUPPORTED CLAIMS:
{claims_text}

Return ONLY valid JSON in exactly this structure:

{{
  "evaluations": [
    {{
      "unsupported_claim": "exact unsupported claim",
      "severity": "Questionable",
      "diagnosis": "brief explanation"
    }}
  ]
}}

The severity MUST be exactly one of:

"Unwanted"
"Benign"
"Questionable"
"""

    data = invoke_json_evaluator(
        faithjudge_prompt,
        "FaithJudge",
        retries=2
    )

    try:
        result = FaithJudgeBatch.model_validate(data)

    except ValidationError as e:
        print("\n❌ FaithJudge JSON failed schema validation.")
        print(data)
        raise RuntimeError(
            f"FaithJudge evaluator returned invalid schema: {e}"
        )

    return result.evaluations


# ============================================================
# Unified Evaluation Pipeline
# ============================================================

def evaluate_rag_output(
    system_name: str,
    query: str,
    context: str,
    answer: str
):

    print(f"\n{'=' * 60}")
    print(f"Evaluating {system_name}")
    print(f"{'=' * 60}")

    print("\nAnswer being evaluated:")
    print(answer)

    # --------------------------------------------------------
    # Tier 1
    # --------------------------------------------------------

    ragas_result = compute_ragas_faithfulness(
        query,
        context,
        answer
    )

    print("\n--- RAGAS Faithfulness ---")
    print(
        f"Score: {ragas_result['faithfulness_score']}"
    )
    print(
        f"Claims: {ragas_result['total_claims']}"
    )
    print(
        f"Verified: {ragas_result['verified_count']}"
    )

    # --------------------------------------------------------
    # Tier 2
    # --------------------------------------------------------

    diagnoses = classify_hallucinations_faithjudge(
        context,
        ragas_result["unverified_claims"]
    )

    print("\n--- FaithJudge Diagnosis ---")

    if diagnoses:
        for diagnosis in diagnoses:
            print(
                f"[{diagnosis.severity}] "
                f"{diagnosis.unsupported_claim}"
            )
            print(
                f"  {diagnosis.diagnosis}"
            )
    else:
        print("No unsupported claims.")

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    return {
        "system": system_name,
        "score": ragas_result["faithfulness_score"],
        "total_claims": ragas_result["total_claims"],
        "verified_claims": ragas_result["verified_count"],
        "unverified_claims": [
            c.model_dump()
            for c in ragas_result["unverified_claims"]
        ],
        "diagnoses": [
            d.model_dump()
            for d in diagnoses
        ]
    }