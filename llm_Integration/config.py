import os
from dotenv import load_dotenv

# Load variables from .env file (looks in parent directories)
load_dotenv()

# Model configuration
MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "qwen-2.5-32b-it")

# Groq API Keys for Round-Robin
GROQ_API_KEYS = [
    os.environ.get("GROQ_API_KEY_1"),
    os.environ.get("GROQ_API_KEY_2")
]

# Ensure keys are actually found
if not all(GROQ_API_KEYS):
    raise ValueError("Missing GROQ_API_KEY_1 or GROQ_API_KEY_2 in .env file")

# Thresholds
CONFIDENCE_INTERVAL_RATIO_THRESHOLD = 0.40  # P90-P10 / P50
