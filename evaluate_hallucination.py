import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()

# Evaluator LLM
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)

# ==========================================
# Data Schemas
# ==========================================
class ClaimVerification(BaseModel):
    statement: str = Field(description="The atomic claim extracted from the answer")
    is_supported: bool = Field(description="True if context directly supports the claim, else False")
    reasoning: str = Field(description="Brief explanation of why it is or isn't supported")

class RagasExtraction(BaseModel):
    claims: List[ClaimVerification] = Field(description="List of all atomic claims extracted and verified")

class FaithJudgeClassification(BaseModel):
    unsupported_claim: str
    severity: Literal["Unwanted", "Benign", "Questionable"]
    diagnosis: str = Field(description="Explanation of the impact on codebase logic")

# ==========================================
# Tier 1: RAGAS Atomic Faithfulness
# ==========================================
def compute_ragas_faithfulness(query: str, context: str, answer: str) -> dict:
    ragas_prompt = f"""
You are an objective judge evaluating factual faithfulness in software codebases.
Break down the Generated Answer into atomic, self-contained statements.
For each statement, determine whether it can be completely verified solely using the Retrieved Context.

Retrieved Context:
{context}

Generated Answer:
{answer}
"""
    structured_llm = evaluator_llm.with_structured_output(RagasExtraction)
    result = structured_llm.invoke(ragas_prompt)
    
    total_claims = len(result.claims)
    verified_claims = [c for c in result.claims if c.is_supported]
    unverified_claims = [c for c in result.claims if not c.is_supported]
    
    score = (len(verified_claims) / total_claims) if total_claims > 0 else 1.0
    
    return {
        "faithfulness_score": round(score, 3),
        "total_claims": total_claims,
        "verified_count": len(verified_claims),
        "unverified_claims": unverified_claims
    }

# ==========================================
# Tier 2: FaithJudge In-Context Diagnosis
# ==========================================
def classify_hallucinations_faithjudge(context: str, unverified_claims: list) -> List[FaithJudgeClassification]:
    if not unverified_claims:
        return []
        
    claims_text = "\n".join([f"- {c.statement} (Context gap: {c.reasoning})" for c in unverified_claims])
    
    faithjudge_prompt = f"""
You are an expert software QA evaluator implementing the FaithJudge taxonomy for hallucination analysis.
Classify each unsupported claim into one of three categories based on these peer-annotated code guidelines:

1. 'Unwanted': Hallucinations that break execution logic (e.g., fabricating non-existent imports, missing dependencies, or non-existent props).
2. 'Benign': Hallucinations that do not break execution (e.g., generic comments, correct syntax best-practices not explicitly mentioned in the context).
3. 'Questionable': Ambiguous statements where factual truth cannot be determined without wider codebase context.

Retrieved Context:
{context}

Unsupported Claims to Evaluate:
{claims_text}
"""
    class FaithJudgeBatch(BaseModel):
        evaluations: List[FaithJudgeClassification]

    structured_llm = evaluator_llm.with_structured_output(FaithJudgeBatch)
    diagnosis = structured_llm.invoke(faithjudge_prompt)
    return diagnosis.evaluations

# ==========================================
# Unified Evaluation Pipeline
# ==========================================
def evaluate_rag_output(system_name: str, query: str, context: str, answer: str):
    print(f"\nEvaluating {system_name}...")
    
    # 1. RAGAS calculation
    ragas_result = compute_ragas_faithfulness(query, context, answer)
    
    # 2. FaithJudge classification on failure spans
    diagnoses = classify_hallucinations_faithjudge(context, ragas_result["unverified_claims"])
    
    return {
        "system": system_name,
        "score": ragas_result["faithfulness_score"],
        "total_claims": ragas_result["total_claims"],
        "verified_claims": ragas_result["verified_count"],
        "diagnoses": [d.dict() for d in diagnoses]
    }