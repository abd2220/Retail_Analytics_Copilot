# Retail Analytics Copilot

A local, private AI agent that answers retail analytics questions by combining **RAG** (Retrieval-Augmented Generation) over local policy documents and **SQL** over a Northwind SQLite database. Built with **LangGraph** and **DSPy**, running locally on **Phi-3.5-mini**.

## 🧠 Graph Design
The agent uses a stateful **LangGraph** workflow with a self-correcting repair loop:

*   **Hybrid Routing:** A DSPy Router classifies questions as `rag` (policy), `sql` (pure data), or `hybrid` (requiring doc lookups to filter DB queries).
*   **Planner Node:** For hybrid queries, the Planner first extracts specific constraints (e.g., "Summer Beverages 1997" -> `1997-06-01` to `1997-06-30`) from documents before SQL generation.
*   **Resilience (Repair Loop):** If SQL execution fails (syntax error) or returns no data, the graph loops back to the Generator with the specific error message for up to **2 retries**.
*   **Strict Synthesis:** The final Synthesizer node enforces strict type casting (int, float, list) to match the Output Contract and validates citations against the actual tables/docs used.

## 🚀 DSPy Optimization
I optimized the **NL-to-SQL (GenerateSQL)** module using the `BootstrapFewShot` optimizer. This was critical for the small Phi-3.5 model to learn schema nuances (like quoting `"Order Details"`).

*   **Module:** `GenerateSQL`
*   **Training Set:** 20 handcrafted examples covering joins, aggregations, and schema ambiguities.
*   **Metric:** Exact SQL Execution Match (verifying data rows match gold standard).

| Metric | Baseline (Zero-Shot) | Optimized (Few-Shot) | Delta |
| :--- | :--- | :--- | :--- |
| **Valid SQL Rate** | **40.0%** | **55.0%** | **+15.0%** |

*(Note: The optimizer learned to correctly handle table quoting and join conditions which were the primary failure points in the baseline.)*

## ⚖️ Trade-offs & Assumptions
*   **Gross Margin Approximation:** Per assignment hints, since the Northwind database lacks cost data, I assume **`CostOfGoods ≈ 0.7 * UnitPrice`**. This rule is documented in `docs/kpi_definitions.md` and dynamically applied by the Planner/SQL Generator.
*   **Context Limits:** To keep prompt tokens low for the local model, SQL results passed to the Synthesizer are truncated if they exceed 500 characters.
*   **Memory:** The agent uses a `MemorySaver` checkpointer to maintain state across the repair loop, ensuring the "Retry" attempts are aware of previous errors.

## 🛠️ Setup & Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ensure Ollama is running:**
    ```bash
    ollama run phi3.5:3.8b-mini-instruct-q4_K_M
    ```
3.  **Run the Agent:**
    ```bash
    python run_agent_hybrid.py \
      --batch sample_questions_hybrid_eval.jsonl \
      --out outputs_hybrid.jsonl
    ```