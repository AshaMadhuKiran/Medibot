"""SQL RAG: answer analytical questions over the relational database.

``sql_rag_chain(question)`` is a plain Python function implementing the three
explicit steps required by the assignment:

  1. Translate the natural-language question into SQL using the LLM.
  2. Clean the raw LLM output to extract *only* the SQL statement.
  3. Execute the (read-only) SQL against ``mediassist.db`` and pass the result
     back to the LLM to produce a natural-language answer.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import List

import pandas as pd

from . import llm
from .config import settings

SCHEMA = """
Table: claims
- claim_id (TEXT)            e.g. 'CLM-2024-1000'
- patient_id (TEXT)
- patient_name (TEXT)
- department (TEXT)         one of: nephrology, cardiology, neurology, gynaecology,
                            orthopaedics, general_medicine, emergency
- claim_type (TEXT)         e.g. reimbursement, cashless
- diagnosis_code (TEXT)     ICD-10 code, e.g. 'N17.9'
- insurer (TEXT)
- claimed_amount (REAL)
- approved_amount (REAL)    NULL until resolved
- status (TEXT)             one of: pending, submitted, approved, rejected, escalated
- submitted_date (TEXT)     ISO date 'YYYY-MM-DD'
- resolved_date (TEXT)      ISO date or NULL

Table: maintenance_tickets
- ticket_id (TEXT)          e.g. 'TKT-2024-2000'
- equipment_name (TEXT)
- equipment_id (TEXT)
- category (TEXT)           one of: sterilisation, infusion, radiology, monitoring,
                            surgical, laboratory
- campus (TEXT)
- issue_type (TEXT)         e.g. preventive_maintenance, breakdown
- fault_code (TEXT)
- raised_by (TEXT)
- raised_date (TEXT)        ISO date 'YYYY-MM-DD'
- resolved_date (TEXT)      ISO date or NULL
- status (TEXT)             one of: open, in_progress, resolved, escalated
- resolution_note (TEXT)
"""

_FORBIDDEN = ("drop", "delete", "truncate", "alter", "update", "insert", "replace")


class SqlRagError(Exception):
    pass


@dataclass
class SqlRagResult:
    answer: str
    sql: str
    row_count: int
    retrieval_type: str = "sql_rag"


# --------------------------------------------------------------------------- #
# Step 1 — natural language -> SQL
# --------------------------------------------------------------------------- #
def generate_sql(question: str) -> str:
    prompt = f"""You are an expert SQLite analyst for a hospital network.

Database schema:
{SCHEMA}

Rules:
1. Return ONLY a single SQLite SELECT statement.
2. Do not explain. Do not use markdown fences.
3. Only query the tables and columns listed above.
4. Use case-insensitive matching for text filters where helpful (LOWER(col) = 'value').

Question: {question}
"""
    return llm.complete(prompt, temperature=0.0)


# --------------------------------------------------------------------------- #
# Step 2 — clean the raw LLM output to a bare SQL statement
# --------------------------------------------------------------------------- #
def clean_sql(raw: str) -> str:
    text = raw.strip()

    # Strip ```sql ... ``` or ``` ... ``` fences if the model added them.
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Keep from the first SELECT/WITH onward (drop any leading prose).
    match = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :]

    # Take only the first statement.
    text = text.split(";")[0].strip()

    if not text:
        raise SqlRagError("Could not extract a SQL statement from the model output.")

    lowered = text.lower()
    for word in _FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise SqlRagError(f"Refusing to run unsafe SQL containing '{word}'.")
    return text


# --------------------------------------------------------------------------- #
# Step 3 — execute and explain
# --------------------------------------------------------------------------- #
def run_sql(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(settings.db_path))
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def explain_result(question: str, df: pd.DataFrame) -> str:
    table_md = df.head(50).to_markdown(index=False) if not df.empty else "(no rows)"
    prompt = f"""Question: {question}

SQL result:
{table_md}

Write a concise, direct natural-language answer to the question based only on
this result. Include the relevant numbers. Do not mention SQL.
"""
    return llm.complete(prompt, temperature=0.0)


def sql_rag_chain(question: str) -> SqlRagResult:
    """End-to-end analytical answer: generate -> clean -> execute -> explain."""
    raw_sql = generate_sql(question)
    sql = clean_sql(raw_sql)

    try:
        df = run_sql(sql)
    except Exception as exc:  # noqa: BLE001
        raise SqlRagError(f"SQL execution failed: {exc}") from exc

    answer = explain_result(question, df)
    return SqlRagResult(answer=answer, sql=sql, row_count=len(df))


# Convenience list used by the README / tests to demonstrate the chain.
DEMO_QUESTIONS: List[str] = [
    "How many billing claims were escalated?",
    "What is the total claimed amount for cardiology?",
    "Which equipment category has the most open maintenance tickets?",
    "How many claims are still pending and what is their average claimed amount?",
]
