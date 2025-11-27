import dspy
import sqlite3
import os
import sys

# Add the current directory to path so we can import dspy_signatures
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dspy.teleprompt import BootstrapFewShot
from dspy_signatures import GenerateSQL

# --- 1. CONFIGURATION ---
# We assume this script is run from the project root: python agent/train_sql_module.py
DB_PATH = "data/northwind.sqlite"

# FULL SCHEMA (Updated to include Suppliers and all columns)
SCHEMA_STR = """
Table: Orders
Columns: OrderID INTEGER, CustomerID TEXT, EmployeeID INTEGER, OrderDate DATETIME, RequiredDate DATETIME, ShippedDate DATETIME, ShipVia INTEGER, Freight NUMERIC, ShipName TEXT, ShipAddress TEXT, ShipCity TEXT, ShipRegion TEXT, ShipPostalCode TEXT, ShipCountry TEXT

Table: Order Details
Columns: OrderID INTEGER, ProductID INTEGER, UnitPrice NUMERIC, Quantity INTEGER, Discount REAL

Table: Products
Columns: ProductID INTEGER, ProductName TEXT, SupplierID INTEGER, CategoryID INTEGER, QuantityPerUnit TEXT, UnitPrice NUMERIC, UnitsInStock INTEGER, UnitsOnOrder INTEGER, ReorderLevel INTEGER, Discontinued TEXT

Table: Customers
Columns: CustomerID TEXT, CompanyName TEXT, ContactName TEXT, ContactTitle TEXT, Address TEXT, City TEXT, Region TEXT, PostalCode TEXT, Country TEXT, Phone TEXT, Fax TEXT

Table: Categories
Columns: CategoryID INTEGER, CategoryName TEXT, Description TEXT, Picture BLOB

Table: Suppliers
Columns: SupplierID INTEGER, CompanyName TEXT, ContactName TEXT, ContactTitle TEXT, Address TEXT, City TEXT, Region TEXT, PostalCode TEXT, Country TEXT, Phone TEXT, Fax TEXT, HomePage TEXT
"""

# --- 2. THE DATASET (20 Handcrafted Examples) ---
train_examples = [
    dspy.Example(
        question="How many customers are there?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT COUNT(*) FROM Customers"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="List all product names.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT ProductName FROM Products"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="What are the names of all categories?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT CategoryName FROM Categories"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="How many orders were placed in 1997?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT COUNT(*) FROM Orders WHERE OrderDate BETWEEN '1997-01-01' AND '1997-12-31'"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="What is the total number of products in stock?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT SUM(UnitsInStock) FROM Products"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="List the 5 most expensive products.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT ProductName, UnitPrice FROM Products ORDER BY UnitPrice DESC LIMIT 5"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Which orders were shipped to Germany?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT OrderID FROM Orders WHERE ShipCountry = 'Germany'"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Total revenue for Order 10248.",
        schema=SCHEMA_STR, constraints="",
        sql_query='SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM "Order Details" WHERE OrderID = 10248'
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Get the company name of the customer who placed order 10248.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT C.CompanyName FROM Customers C JOIN Orders O ON C.CustomerID = O.CustomerID WHERE O.OrderID = 10248"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="List all products in the 'Beverages' category.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT P.ProductName FROM Products P JOIN Categories C ON P.CategoryID = C.CategoryID WHERE C.CategoryName = 'Beverages'"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="What is the total quantity sold for product 11?",
        schema=SCHEMA_STR, constraints="",
        sql_query='SELECT SUM(Quantity) FROM "Order Details" WHERE ProductID = 11'
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Find orders with value greater than 5000.",
        schema=SCHEMA_STR, constraints="",
        sql_query='SELECT OrderID FROM "Order Details" GROUP BY OrderID HAVING SUM(UnitPrice * Quantity * (1 - Discount)) > 5000'
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="List products supplied by 'Exotic Liquids'.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT P.ProductName FROM Products P JOIN Suppliers S ON P.SupplierID = S.SupplierID WHERE S.CompanyName = 'Exotic Liquids'"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Who is the contact person for 'Alfreds Futterkiste'?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT ContactName FROM Customers WHERE CompanyName = 'Alfreds Futterkiste'"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="What is the average unit price of all products?",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT AVG(UnitPrice) FROM Products"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Total revenue between specific dates.",
        schema=SCHEMA_STR, constraints="Date Range: 1996-07-01 to 1996-07-31",
        sql_query='SELECT SUM(OD.UnitPrice * OD.Quantity * (1 - OD.Discount)) FROM "Order Details" OD JOIN Orders O ON OD.OrderID = O.OrderID WHERE O.OrderDate BETWEEN \'1996-07-01\' AND \'1996-07-31\''
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Top selling product by quantity in 1997.",
        schema=SCHEMA_STR, constraints="",
        sql_query='SELECT P.ProductName, SUM(OD.Quantity) as TotalQty FROM "Order Details" OD JOIN Products P ON OD.ProductID = P.ProductID JOIN Orders O ON OD.OrderID = O.OrderID WHERE O.OrderDate LIKE \'1997%\' GROUP BY P.ProductName ORDER BY TotalQty DESC LIMIT 1'
    ).with_inputs("question", "schema", "constraints"),
    
    dspy.Example(
        question="Count of discontinued products.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT COUNT(*) FROM Products WHERE Discontinued = '1'" 
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="Average freight cost for orders shipping to USA.",
        schema=SCHEMA_STR, constraints="",
        sql_query="SELECT AVG(Freight) FROM Orders WHERE ShipCountry = 'USA'"
    ).with_inputs("question", "schema", "constraints"),

    dspy.Example(
        question="What is the gross margin for Product 1 assuming 70% cost?",
        schema=SCHEMA_STR, constraints="Formula: (Price - (Price*0.7)) * Quantity",
        sql_query='SELECT SUM((UnitPrice - (UnitPrice * 0.7)) * Quantity * (1-Discount)) FROM "Order Details" WHERE ProductID = 1'
    ).with_inputs("question", "schema", "constraints"),
]

# --- 3. METRIC: Execution Match ---
def execute_sql(sql):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        conn.close()
        return res
    except Exception as e:
        return str(e)

def sql_metric(example, pred, trace=None):
    """
    Returns True if the predicted SQL returns the exact same rows as the gold SQL.
    """
    predicted_sql = pred.sql_query
    gold_sql = example.sql_query
    
    gold_res = execute_sql(gold_sql)
    pred_res = execute_sql(predicted_sql)
    
    return gold_res == pred_res

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Setup LLM - Adjust URL/model if needed
    lm = dspy.LM(model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434")
    dspy.settings.configure(lm=lm)

    print("--- 1. Evaluating Baseline (Zero-Shot) ---")
    generate_sql = dspy.ChainOfThought(GenerateSQL)
    
    # We evaluate on the first 10 for speed in this demo, or all 20 for full accuracy
    baseline_evaluator = dspy.Evaluate(devset=train_examples, metric=sql_metric, num_threads=1, display_progress=True)
    score_before = baseline_evaluator(generate_sql)
    print(f"Baseline Score: {score_before}%")

    print("\n--- 2. Optimizing (BootstrapFewShot) ---")
    # Training
    optimizer = BootstrapFewShot(metric=sql_metric, max_bootstrapped_demos=4, max_labeled_demos=4)
    optimized_sql = optimizer.compile(generate_sql, trainset=train_examples)

    print("\n--- 3. Evaluating Optimized Module ---")
    score_after = baseline_evaluator(optimized_sql)
    print(f"Optimized Score: {score_after}%")

    # Save
    save_path = "agent/dspy_modules/optimized_sql.json"
    if not os.path.exists("agent/dspy_modules"):
        os.makedirs("agent/dspy_modules")
    
    optimized_sql.save(save_path)
    print(f"\nSaved optimized module to {save_path}")