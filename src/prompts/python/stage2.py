PYTHON_STAGE2_PROMPT = """
<|im_start|>system
You are an expert of coding. Your task is to fix the given code based on the formatter or compiler errors and best practices. Please carefully read the given [ERRORS], [CODE], [GUIDELINES], and generate the improved code right after the [FIXED CODE].
<|im_end|>
<|im_start|>user
[ERRORS]
{lint_report}
[CODE]
{code}
[GUIDELINES]
1. Fix all syntax errors and formatting issues.
2. Ensure the code is self-contained and runnable. (e.g., no missing imports, no undefined variables, functions, or classes)
3. Add docstrings and comments in order to make the code more readable and maintainable.
<|im_end|>
<|im_start|>assistant
[FIXED CODE]
"""
