import dspy
from typing import Literal

# 1. Search Query Generator
class GenerateSearchQuery(dspy.Signature):
    """
    Take a complex user question and generate a concise, keyword-rich search query
    to be used for document retrieval. Focus on nouns, entities, and key terms.
    """
    question: str = dspy.InputField(desc="The user's original question.")
    search_query: str = dspy.OutputField(desc="A short, keyword-focused search query.")

# 2. Router Signature
class ClassifyQuestion(dspy.Signature):
    """
    Classify the question into exactly one category. Output ONLY the label, nothing else.
    
    - rag: Policy questions, return windows, KPI definitions (no database needed)
    - sql: Pure data questions with standard columns (revenue, top customers, counts)
    - hybrid: Questions referencing named campaigns or doc-defined formulas that need lookup first
    """
    
    question: str = dspy.InputField(desc="The user's natural language query.")
    label: str = dspy.OutputField(desc="Output exactly one word: rag, sql, or hybrid")

# 3. Planner Signature (For Hybrid/RAG extraction)
class ExtractSearchTerms(dspy.Signature):
    """
    You are a planner. Your goal is to extract structured constraints from retrieved documents to help a SQL writer.
    Read the provided context and the user's question. 
    Output specific values like Date Ranges (YYYY-MM-DD), Formula adjustments, or Category names.
    If the context contains a definition (like 'Gross Margin'), summarize the formula.
    """
    
    context: str = dspy.InputField(desc="Retrieved document chunks.")
    question: str = dspy.InputField(desc="The original question.")
    
    search_terms: str = dspy.OutputField(
        desc="A concise summary of constraints (e.g., 'Date Range: 1997-06-01 to 1997-06-30', 'Formula: Price * 0.7')."
    )

# 4. NL-to-SQL Signature
class GenerateSQL(dspy.Signature):
    """
    Generate SQLite query. Output ONLY the SQL, nothing else.
    
    Table name: "Order Details" (with quotes and space)
    Revenue: SUM(UnitPrice * Quantity * (1 - Discount))
    """
    
    question: str = dspy.InputField()
    schema: str = dspy.InputField()
    constraints: str = dspy.InputField()
    error_feedback: str = dspy.InputField(default="")
    
    sql: str = dspy.OutputField(desc="SQL query only")

# 5. Synthesizer Signature
class GenerateAnswer(dspy.Signature):
    """
    Answer question using data. Match format_hint exactly.
    Use sql_result if not empty, else use context.
    """
    
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    sql_query: str = dspy.InputField()
    sql_result: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    
    answer: str = dspy.OutputField()
    why: str = dspy.OutputField(desc="1-2 sentences")