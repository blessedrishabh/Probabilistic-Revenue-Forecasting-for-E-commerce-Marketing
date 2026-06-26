import json
import os
import sys

# Make sure we can import llm_Integration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_Integration.graph import build_graph

def main():
    graph = build_graph()
    
    # Initial state (inputs are largely handled by load_context from disk)
    initial_state = {}
    
    print("Running LangGraph workflow...")
    result = graph.invoke(initial_state)
    
    final_output = result.get("final_output", {})
    
    out_path = os.path.join(os.path.dirname(__file__), 'causal_output.json')
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Workflow complete. Output saved to {out_path}")
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()
