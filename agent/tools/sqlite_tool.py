import sqlite3
from typing import Dict, List, Tuple, Optional, Any

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path

    def get_schema(self, table_names: Optional[List[str]] = None) -> str:
        """
        Returns a markdown-formatted string of the database schema.
        This is crucial for the LLM to understand table structures.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # If no specific tables requested, get the main ones relevant to the assignment
        if not table_names:
            table_names = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]

        schema_str = []
        
        for table in table_names:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not cursor.fetchone():
                continue

            # Get column info using PRAGMA
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            
            # Format: TableName(ColumnName Type, ColumnName Type, ...)
            col_strs = [f"{col[1]} {col[2]}" for col in columns]
            schema_str.append(f"Table: {table}\nColumns: {', '.join(col_strs)}")

        conn.close()
        return "\n\n".join(schema_str)

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        Executes a SQL query and returns the results.
        Returns a dictionary containing:
        - columns: List[str]
        - rows: List[Tuple]
        - error: str (None if successful)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # Retrieve results
            rows = cursor.fetchall()
            
            # Retrieve column names from description
            if cursor.description:
                columns = [description[0] for description in cursor.description]
            else:
                columns = []
                
            conn.close()
            
            return {
                "columns": columns,
                "rows": rows,
                "error": None
            }
            
        except sqlite3.Error as e:
            conn.close()
            return {
                "columns": [],
                "rows": [],
                "error": str(e) # This is vital for the Repair Loop
            }

# Quick test block to verify it works when you run this file directly
if __name__ == "__main__":
    tool = SQLiteTool()
    print("--- SCHEMA ---")
    print(tool.get_schema())
    
    print("\n--- TEST QUERY ---")
    result = tool.execute_sql("SELECT * FROM Products LIMIT 2;")
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"Columns: {result['columns']}")
        print(f"Rows: {result['rows']}")