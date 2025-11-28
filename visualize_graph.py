import sys
import os

# Add current directory to path to import agent modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph_hybrid import app

def generate_graph_image():
    print("Generating graph visualization...")
    try:
        # Get the graph object
        graph = app.get_graph()
        
        # Generate PNG binary data
        png_data = graph.draw_mermaid_png()
        
        # Save to file
        output_file = "agent_graph.png"
        with open(output_file, "wb") as f:
            f.write(png_data)
            
        print(f"Success! Graph saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        print("\nTip: Ensure you have the correct dependencies installed.")
        print("Try running: pip install grandalf")

if __name__ == "__main__":
    generate_graph_image()
