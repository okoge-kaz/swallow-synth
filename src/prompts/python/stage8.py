PYTHON_STAGE8_REWRITE_PROMPT = """
You are an expert Python coder tasked with refining a code corpus already rewritten in a first stage to high quality (e.g., Pythonic, documented, modular, and robust). Perform a second-stage rewrite to enhance efficiency, clarity, and educational value, prioritizing Python's standard library and built-in functions (e.g., dict for mappings like string-to-int, set for uniques, sorted for min/max distances, itertools.combinations for combinations). Use uppercase constants and ensure docstrings/comments first abstractly describe the purpose, then detail a step-by-step implementation strategy that aligns perfectly with the code. The rules below guide this refinement, each addressing a specific aspect of optimization or enhancement.

1. Standard Library Optimization: Use built-in functions and standard library modules (e.g., dict for key-value mappings/frequencies, set for uniques, sorted for distances, itertools.combinations) to simplify logic and avoid custom implementations. For string/list operations, include length checks to prevent errors like IndexError.
2. Enhanced Documentation: Write docstrings with an abstract purpose followed by a step-by-step strategy. Use comments similarly for complex sections. For checks like is_prime(n: int) -> bool or is_palindrome(s: str) -> bool, implement as modular helpers if absent.
3. Refined Error Handling: Strengthen error handling to cover edge cases (e.g., empty inputs, invalid mappings) integrated into the strategy, using try-except for potential built-in function errors.
4. Modular Refinement: Retain type annotations and further break complex logic into single-responsibility functions or clear steps, preferring procedural style unless classes are justified.
5. Generalization: Ensure self-containment with top-level imports and uppercase constants. Generalize fixed or specific logic (e.g., via argparse for variable inputs) for reusability.
6. Performance and Structure: Optimize using efficient built-ins (e.g., dict/set for O(1), sorted for O(n log n)) and maintain clear organization (imports, constants, functions, main).
7. Educational Value: Enhance for learners by expanding typical implementations into practical, real-world examples (e.g., variable bases, dynamic inputs), with clear strategy explanations.
8. Strategic Thinking: Analyze the first-stage code’s purpose, identify optimization opportunities (e.g., better built-in usage), plan step-by-step refinements, and ensure docstring-code alignment.

Please generate the rewritten code after '<|REWRITTEN_CODE|>: ```python'.
"""
