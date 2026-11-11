import functools
import os
import time
import logging

logger = logging.getLogger("PDFBrainbox.Tracing")

# Check if LangSmith tracing is requested and available
_LANGSMITH_ACTIVE = False
_traceable = None

if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" or os.getenv("LANGSMITH_API_KEY"):
    try:
        from langsmith import traceable as ls_traceable
        _traceable = ls_traceable
        _LANGSMITH_ACTIVE = True
        logger.info("LangSmith tracing active and initialized.")
    except ImportError:
        logger.info("LangSmith requested but 'langsmith' package not installed. Falling back to local tracing.")

def trace_step(name: str, run_type: str = "chain"):
    """
    Universal tracing decorator.
    If LangSmith is active, registers the span with LangSmith.
    Otherwise, tracks execution latency and structured metadata locally.
    """
    if _LANGSMITH_ACTIVE and _traceable is not None:
        return _traceable(name=name, run_type=run_type)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            try:
                result = fn(*args, **kwargs)
                latency = round((time.time() - t_start) * 1000, 2)
                return result
            except Exception as e:
                latency = round((time.time() - t_start) * 1000, 2)
                logger.error(f"Step '{name}' failed after {latency}ms: {e}")
                raise
        return wrapper
    return decorator

