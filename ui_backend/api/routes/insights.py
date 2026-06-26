import os
import json
import sys
import traceback
from fastapi import APIRouter, HTTPException
from ui_backend.api.schemas import InsightsResponse

# Ensure llm_Integration is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

router = APIRouter()
OUTPUT_FILE = os.path.join(project_root, 'llm_Integration', 'causal_output.json')

@router.post("", response_model=InsightsResponse)
def generate_insights():
    try:
        from llm_Integration.graph import build_graph
        graph = build_graph()
        result = graph.invoke({})
        final_output = result.get("final_output", {})
        
        # Save it for reference
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final_output, f, indent=2)
            
        return final_output
    except Exception as e:
        error_detail = f"LangGraph execution failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[Insights API] {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=InsightsResponse)
def get_insights():
    if not os.path.exists(OUTPUT_FILE):
        raise HTTPException(status_code=404, detail="No insights found. Run POST /api/insights first.")
    with open(OUTPUT_FILE, 'r') as f:
        return json.load(f)
