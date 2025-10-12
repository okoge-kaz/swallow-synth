PYTHON_STAGE4_PROMPT = """
You are an expert coding. Please carefully rewrite the given code to be following the below rules:

1. **Pythonic Style**: Ensure the code follows Python's PEP 8 style guide, including proper indentation, spacing, and naming conventions.
2. **Documentation**: Add comments and docstrings to explain the purpose of functions, classes, and complex code sections.
3. **Error Handling**: Implement appropriate error handling using try-except blocks where necessary. And make sure the code to handle exceptional inputs or edge cases. (e.g., empty lists, empty strings, etc.)
4. **Annotations**: Use type annotations for function parameters and return types to improve code clarity and maintainability.
5. **Modularity**: Break down large functions into smaller, reusable functions. Each function should have a single responsibility.
6. **Self-Containment**: Ensure the code is self-contained and functional. It should not rely on external variables or context that is not imported or defined within the code itself. Please make sure the popular name functions or classes are implemented or imported at the top of the code.
7. **Structured**: Organize the code into logical sections, such as imports, constants, functions, and the main execution block. If the implementation is complex, consider writing the implementation strategy by using step-by-step comments.
8. **Performance**: Optimize the code for performance where applicable, avoiding unnecessary computations or memory allocations.
9. **Educational**: The rewritten code should be educational, providing clear explanations and insights into the coding process. It should be suitable for someone learning Python or looking to improve their coding skills. If the provided code is too simple, please change the semantic of the code to be more complex and educational.
10. **Thinking**: First consider the purpose of the provided code and the context in which it will be used. Second, If the provided code's implementation is not clear or has issues, please think how to solve the issues. Third, think step by step how to rewrite the code to be more educational and self-contained. Finally, rewrite the code following the above rules.

Please generate the rewritten code after "<|REWRITTEN_CODE|>: ```python".
"""
