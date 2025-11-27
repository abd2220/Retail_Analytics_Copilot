import dspy
from typing import Literal

# --- NEW SIGNATURE ---
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
    You are a routing agent for a retail analytics system.
    Your job is to decide the best strategy to answer the user's question.
    
    Strategies:
    - 'rag': Use this for questions about static policies, definitions, definitions of KPIs, or marketing calendars (e.g., "return policy", "meaning of AOV", "dates for summer campaign"). No database access needed.
    - 'sql': Use this for pure data questions where all math and entities are obvious standard database columns (e.g., "total revenue", "top 5 customers", "inventory counts").
    - 'hybrid': Use this when the question refers to a specific named campaign, date range alias, or custom KPI defined in documents that requires looking up a definition FIRST before writing SQL (e.g., "sales during 'Summer Beverages'", "top margin products using the KPI doc definition").
    """
    
    question: str = dspy.InputField(desc="The user's natural language query.")
    label: Literal['rag', 'sql', 'hybrid'] = dspy.OutputField(desc="The chosen strategy label.")

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
    You are a SQLite expert for the Northwind database.
    Generate a valid SQLite query to answer the question.
    
    Rules:
    1. Use ONLY the provided schema.
    2. 'Order Details' table must be quoted as "Order Details".
    3. Use the 'constraints' field for specific date ranges or formula logic.
    4. Return ONLY the raw SQL query. No markdown formatting or explanations.
    """
    
    question: str = dspy.InputField(desc="The user's data question.")
    schema: str = dspy.InputField(desc="Schema of available tables and columns.")
    constraints: str = dspy.InputField(desc="Specific constraints (dates, formulas) from documentation.")
    error_feedback: str = dspy.InputField(desc="Previous error message (if any) to fix.", default="")
    
    sql_query: str = dspy.OutputField(desc="The executable SQLite query string.")

# 5. Synthesizer Signature
class GenerateAnswer(dspy.Signature):
    """
    You are an information extraction machine. 
    You MUST answer the question using ONLY the provided text from 'context' or 'sql_result'.
    DO NOT use any outside knowledge or make assumptions.

    Your process:
    1.  Carefully read the user's question to understand the exact entities involved (e.g., "unopened Beverages", "return window").
    2.  Scan the 'context' for the single sentence that contains ALL of these entities.
    3.  Extract the specific value (e.g., a number, a name) from that exact sentence.
    4.  If no sentence contains the required information, state that the information is not available.

    Example:
    Question: "return window for unopened Beverages"
    Context: "...- Perishables: 3-7 days. ...- Beverages unopened: 14 days. ...- Non-perishables: 30 days."
    Your thought process: The sentence "- Beverages unopened: 14 days." matches all the keywords. The number is 14.

    The 'answer' field must strictly match the `format_hint`.
    The 'explanation' should be 1-2 sentences explaining exactly which document and sentence you used.
    """
    
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved text context (if any).")
    sql_query: str = dspy.InputField(desc="The query that was run.")
    sql_result: str = dspy.InputField(desc="The rows returned by the database.")
    format_hint: str = dspy.InputField(desc="The required format (int, float, list, etc).")
    
    answer: str = dspy.OutputField(desc="The precise value matching the format hint.")
    explanation: str = dspy.OutputField(desc="A brief justification (max 2 sentences).")