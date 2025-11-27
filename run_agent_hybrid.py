import click
import json
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph_hybrid import app

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch, out):
    """
    Main entry point for the Retail Analytics Copilot.
    Runs the agent over a batch of questions and outputs results.
    """
    print(f"--- Starting Batch Run ---")
    print(f"Input: {batch}")
    print(f"Output: {out}")

    # Clear output file if exists to start fresh
    if os.path.exists(out):
        os.remove(out)

    with open(batch, 'r', encoding='utf-8') as f_in, open(out, 'a', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        total = len(lines)
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            q_data = json.loads(line)
            q_id = q_data.get("id")
            question_text = q_data.get("question")
            format_hint = q_data.get("format_hint", "str")
            
            print(f"\n[{i+1}/{total}] Processing ID: {q_id}")
            
            # 1. Initialize State
            initial_state = {
                "id": q_id,
                "question": question_text,
                "format_hint": format_hint,
                "retries": 0,
                "sql_results": {},
                "retrieved_docs": []
            }
            
            # 2. Run Agent
            # We use the Question ID as the thread_id to keep states separate
            config = {"configurable": {"thread_id": str(q_id)}}
            
            try:
                # Invoke the graph
                # recursion_limit=20 prevents infinite loops if something goes wrong
                final_state = app.invoke(initial_state, config=config, recursion_limit=20)
                
                # 3. Extract Output
                output_payload = final_state.get("final_output")
                
                # Fallback if something crashed internally and didn't produce output
                if not output_payload:
                    output_payload = {
                        "id": q_id,
                        "final_answer": None,
                        "sql": "",
                        "confidence": 0.0,
                        "explanation": "Agent failed to produce an output payload.",
                        "citations": []
                    }
                
            except Exception as e:
                print(f"Error processing {q_id}: {e}")
                output_payload = {
                    "id": q_id,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"System Error: {str(e)}",
                    "citations": []
                }

            # 4. Write to file immediately (Line-by-Line)
            f_out.write(json.dumps(output_payload) + "\n")
            f_out.flush() # Ensure it's saved even if script crashes later

    print(f"\n--- Batch Run Complete. Results saved to {out} ---")

if __name__ == '__main__':
    main()