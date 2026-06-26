import subprocess
import sys
import os

def main():
    print("Starting AIgnition Web Application (FastAPI serving SPA)...")
    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Run ONLY uvicorn. UI is served at /
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "ui_backend.api.app:app", "--port", "8000"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        env=backend_env
    )

if __name__ == "__main__":
    main()
