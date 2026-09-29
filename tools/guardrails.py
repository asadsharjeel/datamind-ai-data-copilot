import ast
import re

INJECTION_PATTERNS = [
    r"ignore (all|any|your|previous|the above).{0,30}(instruction|rule|prompt)",
    r"reveal (your|the) (system )?prompt",
    r"you are now",
    r"disregard .{0,30}(instruction|rule)",
    r"api[_ ]?key",
]
DESTRUCTIVE = [r"\bdelete (the )?(file|data|folder)", r"\bformat (the )?(disk|drive)", r"\brm -rf\b"]


def check_user_input(text: str):
    """Returns (ok, message). Blocks obvious prompt-injection and destructive requests."""
    low = text.lower()
    for p in INJECTION_PATTERNS + DESTRUCTIVE:
        if re.search(p, low):
            return False, "I can't do that. I only analyse the uploaded data, read-only, and I can't change my rules."
    if len(text) > 2000:
        return False, "Question is too long. Please shorten it."
    return True, ""


BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals",
                "locals", "getattr", "setattr", "delattr", "vars", "breakpoint", "exit", "quit"}


def validate_code(code: str):
    """Static safety check for LLM-written pandas code. Returns (ok, reason)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Imports are not allowed."
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False, f"'{node.id}' is not allowed."
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, "Private/dunder attributes are not allowed."
        if isinstance(node, ast.Attribute) and node.attr in {
            "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "read_csv", "read_pickle", "eval", "query_file"
        }:
            return False, f"'.{node.attr}' is not allowed (read-only analysis)."
    return True, ""
