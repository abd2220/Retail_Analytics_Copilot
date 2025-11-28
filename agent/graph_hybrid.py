import os
import json
import re
import dspy
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Internal imports
from .tools.sqlite_tool import SQLiteTool
from .rag.retrieval import LocalRetriever
from .dspy_signatures import GenerateSearchQuery, ClassifyQuestion, ExtractSearchTerms, GenerateSQL, GenerateAnswer

# --- 0. LM CONFIGURATION ---
lm = dspy.LM(model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434")
dspy.settings.configure(lm=lm)

# --- 1. SETUP & STATE DEFINITION ---

class AgentState(TypedDict):
    id: str
    question: str
    format_hint: str
    route: str
    search_query: str
    search_terms: str
    retrieved_docs: List[Dict]
    sql_query: str
    sql_results: Dict
    retries: int
    error_feedback: str
    final_output: Dict

# Initialize Tools
db_tool = SQLiteTool()
retriever = LocalRetriever()

# --- 2. DSPy MODULE LOADING ---

query_gen_module = dspy.Predict(GenerateSearchQuery)
planner_module = dspy.Predict(ExtractSearchTerms)
synthesizer_module = dspy.Predict(GenerateAnswer)

# Load Optimized Router Module
# NOTE: Disabled loading optimized module to improve stability with Phi-3.5
print("--- Using Baseline (Unoptimized) Router for Stability ---")
router_module = dspy.Predict(ClassifyQuestion)

# sql_module_path = "agent/dspy_modules/optimized_sql.json"
# ... (removed old SQL loading logic) ...

# SQL Generator Module (DSPy Predict - required by assignment)
sql_gen_module = dspy.Predict(GenerateSQL)

# --- 3. NODE IMPLEMENTATIONS ---

def router_node(state: AgentState):
    print(f"\n--- [Router] Processing: {state['question'][:50]}... ---")
    try:
        pred = router_module(question=state["question"])
        raw_label = pred.label.lower().strip()
        
        if "hybrid" in raw_label:
            route = "hybrid"
        elif "sql" in raw_label:
            route = "sql"
        elif "rag" in raw_label:
            route = "rag"
        else:
            print(f"   (Router output unclear: '{raw_label}', defaulting to hybrid)")
            route = "hybrid"
            
    except Exception as e:
        print(f"   (Router crashed: {e}, defaulting to hybrid)")
        route = "hybrid"
    
    print(f"--- [Router] Classified as: {route} ---")
    return {"route": route}

def search_query_generation_node(state: AgentState):
    """Generates a keyword-rich search query."""
    print("--- [Query Generator] Creating keyword search query... ---")
    pred = query_gen_module(question=state["question"])
    return {"search_query": pred.search_query}

def retriever_node(state: AgentState):
    print(f"--- [Retriever] Searching docs with query: '{state['search_query']}' ---")
    results = retriever.search(state["search_query"], top_k=3)
    return {"retrieved_docs": results}

def planner_node(state: AgentState):
    """Extracts constraints (Requirement #3)."""
    print("--- [Planner] Extracting constraints... ---")
    context_str = "\n\n".join([d['text'] for d in state.get("retrieved_docs", [])])
    
    # Even if context is empty, we pass through planner to be safe
    if not context_str:
        return {"search_terms": ""}
        
    pred = planner_module(context=context_str, question=state["question"])
    return {"search_terms": pred.search_terms}

def fix_order_details_table(sql: str) -> str:
    """
    Post-process SQL to fix common LLM mistakes with 'Order Details' table name.
    The table MUST be quoted as "Order Details" (with space).
    """
    import re
    
    # Common wrong patterns the LLM generates
    wrong_patterns = [
        r'\bOrderDetails\b',
        r'\bOrder_Details\b', 
        r'\borderdetails\b',
        r'\border_details\b',
        r'`Order Details`',  # MySQL style
        r'\[Order Details\]',  # SQL Server style
    ]
    
    for pattern in wrong_patterns:
        sql = re.sub(pattern, '"Order Details"', sql, flags=re.IGNORECASE)
    
    return sql

def sql_generator_node(state: AgentState):
    """
    Generate SQL using DSPy Predict (required by assignment).
    Uses simplified prompting to work with small Phi-3.5 model.
    """
    attempt = state.get("retries", 0) + 1
    print(f"--- [SQL Generator] Attempt {attempt} ---")
    
    schema = db_tool.get_schema()
    constraints = state.get("search_terms", "")
    error_msg = state.get("error_feedback", "") if state.get("retries", 0) > 0 else ""
    
    # Add explicit hint about Order Details if we got that error
    if "OrderDetails" in error_msg or "no such table" in error_msg.lower():
        error_msg += ' IMPORTANT: Use "Order Details" (with space and double quotes) not OrderDetails.'
    
    try:
        pred = sql_gen_module(
            question=state["question"],
            schema=schema,
            constraints=constraints,
            error_feedback=error_msg
        )
        
        # Extract SQL (field name is 'sql' in signature)
        raw_sql = pred.sql if hasattr(pred, 'sql') else pred.sql_query if hasattr(pred, 'sql_query') else ""
        
        # Clean markdown and fix table names
        clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
        
        # Remove explanatory text if model added it
        # SQL should not contain phrases like "Here is" or "This query"
        lines = clean_sql.split('\n')
        sql_lines = [l for l in lines if l.strip() and not any(phrase in l.lower() for phrase in ['here is', 'this query', 'explanation', 'note:', 'the query'])]
        clean_sql = '\n'.join(sql_lines) if sql_lines else clean_sql
        
        clean_sql = fix_order_details_table(clean_sql)
        
        print(f"   Generated SQL: {clean_sql[:100]}...")
        
    except Exception as e:
        print(f"   SQL generation error: {e}")
        # Fallback - try to generate something simple
        clean_sql = "SELECT 1"  # Will trigger repair loop
    
    return {"sql_query": clean_sql}

def executor_node(state: AgentState):
    print("--- [Executor] Running SQL... ---")
    query = state["sql_query"]
    results = db_tool.execute_sql(query)
    
    current_retries = state.get("retries", 0)
    if results.get("error"):
        print(f"   (Error: {results['error']})")
        return {"sql_results": results, "retries": current_retries + 1, "error_feedback": str(results['error'])}
    
    return {"sql_results": results}

def synthesizer_node(state: AgentState):
    print("--- [Synthesizer] Finalizing answer... ---")
    sql_res = state.get("sql_results", {})
    
    # Pass actual document text for RAG questions, keep it short
    docs = state.get("retrieved_docs", [])
    if docs:
        # Take top doc text, truncate if needed
        context_str = docs[0].get('text', '')[:500]
    else:
        context_str = ""
    
    # Format SQL results cleanly
    sql_rows = sql_res.get("rows", [])
    sql_result_str = str(sql_rows) if sql_rows else "No results"
    if len(sql_result_str) > 300:
        sql_result_str = sql_result_str[:300] + "..."
    
    pred = synthesizer_module(
        question=state["question"],
        context=context_str,
        sql_query=state.get("sql_query", ""),
        sql_result=sql_result_str,
        format_hint=state["format_hint"]
    )
    
    raw_answer = pred.answer
    hint = state["format_hint"]
    final_val = raw_answer
    
    try:
        if hint == "int":
            nums = re.findall(r'\d+', raw_answer)
            final_val = int(nums[0]) if nums else 0
        elif hint == "float":
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_answer)
            final_val = round(float(nums[0]), 2) if nums else 0.0
        elif "list" in hint or "{" in hint:
             if "{" in raw_answer or "[" in raw_answer:
                try:
                    final_val = json.loads(raw_answer)
                except:
                    pass
    except:
        pass
    
    citations = [doc["id"] for doc in state.get("retrieved_docs", [])]
    if sql_query := state.get("sql_query"):
        for tbl in ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]:
            if tbl in sql_query or f'"{tbl}"' in sql_query: citations.append(tbl)
            
    confidence = 1.0 - (0.2 * state.get("retries", 0)) if not sql_res.get("error") else 0.0
    
    # Extract explanation (field name is 'why' in signature)
    explanation_text = ""
    if hasattr(pred, 'why'):
        explanation_text = pred.why[:200] if pred.why else ""
    elif hasattr(pred, 'reason'):
        explanation_text = pred.reason[:200] if pred.reason else ""
    elif hasattr(pred, 'explanation'):
        explanation_text = pred.explanation[:200] if pred.explanation else ""
    
    output_obj = {
        "id": state["id"],
        "final_answer": final_val,
        "sql": state.get("sql_query", ""),
        "confidence": round(max(0.0, confidence), 2),
        "explanation": explanation_text,
        "citations": list(set(citations))
    }
    
    return {"final_output": output_obj}

# --- 4. GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("search_query_generator", search_query_generation_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("planner", planner_node) # Planner is back!
workflow.add_node("sql_generator", sql_generator_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("router")

# Router -> (Search Query Generator OR SQL Generator)
def route_after_router(state):
    # RAG and Hybrid need docs, so they go to Search Query -> Retriever
    if state["route"] in ["rag", "hybrid"]:
        return "search_query_generator"
    # Pure SQL skips retrieval AND planner
    return "sql_generator"

workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "search_query_generator": "search_query_generator",
        "sql_generator": "sql_generator"
    }
)

workflow.add_edge("search_query_generator", "retriever")
workflow.add_edge("retriever", "planner")

# Planner -> (Synthesizer OR SQL Generator)
def route_after_planner(state):
    # RAG skips SQL
    if state["route"] == "rag":
        print("--- [Graph] Route is RAG, skipping SQL... ---")
        return "synthesizer"
    return "sql_generator"

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "synthesizer": "synthesizer",
        "sql_generator": "sql_generator"
    }
)

workflow.add_edge("sql_generator", "executor")

# Executor -> (Retry OR Synthesizer)
def route_after_executor(state):
    res = state.get("sql_results", {})
    if res.get("error"):
        if state.get("retries", 0) <= 2:
            print(f"!!! Triggering Repair Loop ({state.get('retries', 0)}/2) !!!")
            return "retry"
        else:
            print("!!! Max retries reached. Moving to synthesis.")
    return "finalize"

workflow.add_conditional_edges(
    "executor",
    route_after_executor,
    {
        "retry": "sql_generator",
        "finalize": "synthesizer"
    }
)

workflow.add_edge("synthesizer", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)