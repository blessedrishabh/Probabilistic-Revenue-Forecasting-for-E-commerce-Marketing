import itertools
from typing import List

class APIRotator:
    """Round-robin rotator for distributing API calls across multiple keys."""
    
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one API key must be provided.")
        self.api_keys = api_keys
        self._iterator = itertools.cycle(self.api_keys)
        self._call_count = 0

    def get_next_key(self) -> str:
        """Returns the next API key in the round-robin sequence."""
        self._call_count += 1
        key = next(self._iterator)
        print(f"  [APIRotator] LLM call #{self._call_count} -> Key ...{key[-6:]}")
        return key

def _create_rotator() -> APIRotator:
    """Lazy factory — only imports config when actually called."""
    from llm_Integration import config
    return APIRotator(config.GROQ_API_KEYS)

# Module-level singleton, lazily initialized
_rotator_instance: APIRotator = None

def get_rotator() -> APIRotator:
    """Returns the singleton APIRotator, creating it on first call."""
    global _rotator_instance
    if _rotator_instance is None:
        _rotator_instance = _create_rotator()
    return _rotator_instance
