import numpy as np
import pandas as pd
from tools.guardrails import validate_code

SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ["len", "range", "sum", "min", "max", "abs", "round", "sorted", "list",
                           "dict", "set", "tuple", "str", "int", "float", "bool", "enumerate",
                           "zip", "any", "all", "print", "isinstance"]}


def run_pandas(df: pd.DataFrame, code: str) -> dict:
    """Runs validated code on a COPY of df. Code must assign its answer to `result`."""
    ok, reason = validate_code(code)
    if not ok:
        return {"error": f"Blocked by guardrails: {reason}"}
    env = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, env)
    except Exception as e:  # report to the LLM so it can fix its own code
        return {"error": f"{type(e).__name__}: {e}"}
    if "result" not in env:
        return {"error": "Code must assign the answer to a variable named `result`."}
    r = env["result"]
    if isinstance(r, (pd.DataFrame, pd.Series)):
        return {"table": r.head(50).reset_index() if isinstance(r, pd.Series) else r.head(50)}
    return {"value": r}
