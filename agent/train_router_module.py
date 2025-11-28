"""
DSPy Router Optimization Script

Uses BootstrapFewShot optimizer with conservative settings for Phi-3.5 model.
"""

import dspy
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dspy.teleprompt import BootstrapFewShot
from dspy_signatures import ClassifyQuestion

# --- 1. TRAINING DATASET (20 examples as required) ---
train_examples = [
    # RAG examples (7)
    dspy.Example(
        question="What is the return policy for unopened beverages?",
        label="rag"
    ).with_inputs("question"),
    dspy.Example(
        question="According to the product policy, how many days can I return perishables?",
        label="rag"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the definition of Average Order Value?",
        label="rag"
    ).with_inputs("question"),
    dspy.Example(
        question="How is Gross Margin calculated according to the KPI docs?",
        label="rag"
    ).with_inputs("question"),
    dspy.Example(
        question="What are the dates for the Winter Classics campaign?",
        label="rag"
    ).with_inputs("question"),
    dspy.Example(
        question="What categories are mentioned in the catalog?",
        label="rag"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the return window for non-perishable items?",
        label="rag"
    ).with_inputs("question"),
    
    # SQL examples (7)
    dspy.Example(
        question="How many customers are there in total?",
        label="sql"
    ).with_inputs("question"),
    dspy.Example(
        question="What are the top 5 products by unit price?",
        label="sql"
    ).with_inputs("question"),
    dspy.Example(
        question="List all orders shipped to Germany.",
        label="sql"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the total revenue from all orders?",
        label="sql"
    ).with_inputs("question"),
    dspy.Example(
        question="Top 3 products by total revenue all-time.",
        label="sql"
    ).with_inputs("question"),
    dspy.Example(
        question="How many orders were placed in 1997?",
        label="sql"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the average freight cost per order?",
        label="sql"
    ).with_inputs("question"),
    
    # HYBRID examples (6)
    dspy.Example(
        question="What was the total revenue during Summer Beverages 1997?",
        label="hybrid"
    ).with_inputs("question"),
    dspy.Example(
        question="Using the AOV definition, what was the Average Order Value in Winter Classics 1997?",
        label="hybrid"
    ).with_inputs("question"),
    dspy.Example(
        question="Which category had the highest sales during Summer Beverages 1997?",
        label="hybrid"
    ).with_inputs("question"),
    dspy.Example(
        question="Who was the top customer by gross margin in 1997 using the KPI definition?",
        label="hybrid"
    ).with_inputs("question"),
    dspy.Example(
        question="Total revenue from Beverages during the Summer Beverages campaign.",
        label="hybrid"
    ).with_inputs("question"),
    dspy.Example(
        question="Calculate AOV for Winter Classics 1997 as defined in the docs.",
        label="hybrid"
    ).with_inputs("question"),
]

# --- 2. METRIC ---
def router_metric(example, pred, trace=None):
    """
    Simple metric that's tolerant of model instability.
    Returns True if predicted label matches gold label.
    """
    try:
        gold = example.label.lower().strip()
        predicted = pred.label.lower().strip()
        
        # Extract label from potentially verbose output
        if "hybrid" in predicted:
            predicted = "hybrid"
        elif "sql" in predicted:
            predicted = "sql"
        elif "rag" in predicted:
            predicted = "rag"
        
        return gold == predicted
    except:
        return False

# --- 3. MANUAL EVALUATION ---
def evaluate_router(router, examples, name="Router"):
    """Synchronous evaluation with error handling"""
    correct = 0
    errors = 0
    total = len(examples)
    
    print(f"\nEvaluating {name} on {total} examples...")
    for i, ex in enumerate(examples):
        try:
            time.sleep(0.3)  # Small delay between calls
            pred = router(question=ex.question)
            
            # Extract label
            raw_label = pred.label.lower().strip()
            if "hybrid" in raw_label:
                predicted = "hybrid"
            elif "sql" in raw_label:
                predicted = "sql"
            elif "rag" in raw_label:
                predicted = "rag"
            else:
                predicted = "unknown"
            
            is_correct = (predicted == ex.label)
            if is_correct:
                correct += 1
            
            symbol = "✓" if is_correct else "✗"
            print(f"  [{i+1}/{total}] {ex.question[:30]:30s}... | Exp: {ex.label:6s} | Got: {predicted:6s} {symbol}")
            
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{total}] ERROR: {str(e)[:50]}...")
    
    score = (correct / total) * 100 if total > 0 else 0
    print(f"\nResult: {correct}/{total} correct = {score:.1f}% ({errors} errors)")
    return score

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("=" * 70)
    print("DSPy Router Optimization - BootstrapFewShot")
    print("=" * 70)
    
    # Setup LLM
    lm = dspy.LM(model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434")
    dspy.settings.configure(lm=lm)
    
    # Step 1: Baseline
    print("\n--- STEP 1: Baseline Router (Zero-Shot) ---")
    baseline_router = dspy.Predict(ClassifyQuestion)
    
    eval_set = train_examples[:20]
    score_before = evaluate_router(baseline_router, eval_set, "Baseline")
    
    # Step 2: Optimize with BootstrapFewShot
    print("\n--- STEP 2: DSPy BootstrapFewShot Optimization ---")
    print("Settings: max_bootstrapped_demos=1, max_labeled_demos=1, max_rounds=1")
    
    optimizer = BootstrapFewShot(
        metric=router_metric,
        max_bootstrapped_demos=1,  # Ultra-conservative
        max_labeled_demos=1,
        max_rounds=1
    )
    
    try:
        print("Training on first 8 examples...")
        optimized_router = optimizer.compile(baseline_router, trainset=train_examples[:8])
        print("Optimization completed!")
        
        # Step 3: Evaluate optimized
        print("\n--- STEP 3: Optimized Router Evaluation ---")
        score_after = evaluate_router(optimized_router, eval_set, "Optimized")
        
        # Results
        print("\n" + "=" * 70)
        print("*** DSPy Optimization Results ***")
        print(f"Baseline Score:  {score_before:.1f}%")
        print(f"Optimized Score: {score_after:.1f}%")
        print(f"Change:          {score_after - score_before:+.1f}%")
        print("=" * 70)
        
        # Save the better one
        save_path = "agent/dspy_modules/optimized_router.json"
        if not os.path.exists("agent/dspy_modules"):
            os.makedirs("agent/dspy_modules")
        
        if score_after >= score_before:
            print(f"\nSaving optimized router (improved performance)")
            optimized_router.save(save_path)
        else:
            print(f"\nSaving baseline router (optimization didn't improve)")
            baseline_router.save(save_path)
        
        print(f"Saved to: {save_path}")
        
    except Exception as e:
        print(f"\n!!! Optimization failed: {e}")
        print("Saving baseline router...")
        
        score_after = score_before
        
        print("\n" + "=" * 70)
        print("*** DSPy Optimization Results ***")
        print(f"Baseline Score:  {score_before:.1f}%")
        print(f"Optimized Score: N/A (failed)")
        print(f"Change:          0.0% (baseline deployed)")
        print("=" * 70)
        
        save_path = "agent/dspy_modules/optimized_router.json"
        if not os.path.exists("agent/dspy_modules"):
            os.makedirs("agent/dspy_modules")
        baseline_router.save(save_path)
        print(f"Saved baseline to: {save_path}")
