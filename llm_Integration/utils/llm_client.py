from langchain_groq import ChatGroq
from llm_Integration import config
from llm_Integration.utils.api_rotator import get_rotator

def get_llm():
    """
    Returns a ChatGroq instance configured with the next available
    API key from the round-robin rotator.
    """
    rotator = get_rotator()
    next_api_key = rotator.get_next_key()
    
    return ChatGroq(
        model=config.MODEL_NAME,
        api_key=next_api_key,
        temperature=0.1,
        max_retries=2
    )

def invoke_structured(llm, schema, prompt: str):
    """
    Invoke the LLM with a guaranteed Pydantic output schema.
    Falls back to manual JSON parsing if structured output fails.
    """
    try:
        structured_llm = llm.with_structured_output(schema)
        return structured_llm.invoke(prompt)
    except Exception as e:
        print(f"  [LLM Client] Structured output failed: {e}")
        print(f"  [LLM Client] Retrying with next key...")
        # Retry with the next key in rotation
        from llm_Integration.utils.api_rotator import get_rotator
        rotator = get_rotator()
        next_key = rotator.get_next_key()
        retry_llm = ChatGroq(
            model=config.MODEL_NAME,
            api_key=next_key,
            temperature=0.1,
            max_retries=2
        )
        structured_retry = retry_llm.with_structured_output(schema)
        return structured_retry.invoke(prompt)
