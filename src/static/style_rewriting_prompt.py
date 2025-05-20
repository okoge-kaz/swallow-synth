PYTHON_STYLE_REWRITING_PROMPT = """
Please refactor the following Python code to resolve pylint/ruff warnings and improve code quality.

【Warnings】
{lint_report}

【Code】
{code}

【Guidelines】
1. If there are syntax errors, fix them first. Syntax errors will be marked with type "syntax_error" in the warnings.
2. Warnings that can be automatically fixed by ruff have already been addressed.
3. Please improve the code considering the following aspects:
   - Use meaningful variable and function names that follow naming conventions
   - Write clear and concise docstrings for functions and classes
   - Use type hints for function signatures
   - Write clear and concise comments for code blocks
   - Ensure the code is self-contained and does not depend on external variables
   - Ensure the code is well-structured and easy to read
   - Ensure the code is free of errors and runs correctly
   - Ensure the code is optimized and does not have redundant operations
   - Ensure algorithms and data structures are efficient and concise
   - Are functions properly modularized with clear separation of concerns?
   - Is variable lifetime intentionally managed, avoiding frequent reassignments and overly long scopes?
   - Is error handling appropriately implemented where necessary?
   - Do comments provide context and rationale, not just code behavior?
   - Do functions and classes have clear single responsibilities?
   - Is the code formatted for readability?
   - Are all function and class calls properly defined and imported?
   - Are there any references to undefined functions or classes in comments?

4. If the given code is too simple or not self-contained, enhance it to be more educational and useful:
   - Add practical examples and use cases
   - Include input validation and error handling
   - Add meaningful comments explaining the logic and design decisions
   - Implement additional features that demonstrate best practices
   - Make the code more reusable and maintainable
   - Add docstring examples showing how to use the code
   - Include edge cases and their handling
   - Add type hints and documentation for better code understanding

5. Ensure the refactored code is syntactically correct.
"""

RUST_STYLE_REWRITING_PROMPT = """
Please refactor the following Rust code to resolve clippy warnings and improve code quality.

【Warnings】
{lint_report}

【Code】
{code}

【Guidelines】
1. If there are syntax errors, fix them first. Syntax errors will be marked with type "syntax_error" in the warnings.
2. Warnings that can be automatically fixed by rustfmt have already been addressed.
3. Please improve the code considering the following aspects:
   - Use meaningful variable and function names that follow naming conventions
   - Write clear and concise docstrings for functions and structs
   - Use type annotations where they improve clarity
   - Write clear and concise comments for code blocks
   - Ensure the code is self-contained and does not depend on external variables
   - Ensure the code is well-structured and easy to read
   - Ensure the code is free of errors and runs correctly
   - Ensure the code is optimized and does not have redundant operations
   - Ensure algorithms and data structures are efficient and concise
   - Are functions properly modularized with clear separation of concerns?
   - Is variable lifetime intentionally managed, avoiding frequent reassignments and overly long scopes?
   - Is error handling appropriately implemented where necessary?
   - Do comments provide context and rationale, not just code behavior?
   - Do functions and structs have clear single responsibilities?
   - Is the code formatted for readability?
   - Are all function and struct/trait calls properly defined and imported?
   - Are there any references to undefined functions or types in comments?

4. If the given code is too simple or not self-contained, enhance it to be more educational and useful:
   - Add practical examples and use cases
   - Include proper error handling using Result and Option
   - Add meaningful comments explaining the logic and design decisions
   - Implement additional features that demonstrate Rust best practices
   - Make the code more reusable and maintainable
   - Add docstring examples showing how to use the code
   - Include edge cases and their handling
   - Add type annotations and documentation for better code understanding
   - Demonstrate proper use of ownership and borrowing
   - Show effective use of traits and generics

5. Ensure the refactored code is syntactically correct.
"""

JAVA_STYLE_REWRITING_PROMPT = """
Please refactor the following Java code to resolve checkstyle warnings and improve code quality.

【Warnings】
{lint_report}

【Code】
{code}

【Guidelines】
1. If there are syntax errors, fix them first. Syntax errors will be marked with type "syntax_error" in the warnings.
2. Warnings that can be automatically fixed by google-java-format have already been addressed.
3. Please improve the code considering the following aspects:
   - Use meaningful variable and method names that follow naming conventions
   - Write clear and concise JavaDoc for methods and classes
   - Use appropriate access modifiers and annotations
   - Write clear and concise comments for code blocks
   - Ensure the code is self-contained and does not depend on external variables
   - Ensure the code is well-structured and easy to read
   - Ensure the code is free of errors and runs correctly
   - Ensure the code is optimized and does not have redundant operations
   - Ensure algorithms and data structures are efficient and concise
   - Are methods properly modularized with clear separation of concerns?
   - Is variable lifetime intentionally managed, avoiding frequent reassignments and overly long scopes?
   - Is error handling appropriately implemented where necessary?
   - Do comments provide context and rationale, not just code behavior?
   - Do methods and classes have clear single responsibilities?
   - Is the code formatted for readability?
   - Are all method and class calls properly defined and imported?
   - Are there any references to undefined methods or classes in comments?

4. If the given code is too simple or not self-contained, enhance it to be more educational and useful:
   - Add practical examples and use cases
   - Include proper exception handling
   - Add meaningful comments explaining the logic and design decisions
   - Implement additional features that demonstrate Java best practices
   - Make the code more reusable and maintainable
   - Add JavaDoc examples showing how to use the code
   - Include edge cases and their handling
   - Add proper access modifiers and annotations
   - Demonstrate effective use of interfaces and abstract classes
   - Show proper use of collections and streams

5. Ensure the refactored code is syntactically correct.
"""

C_STYLE_REWRITING_PROMPT = """
Please refactor the following C code to resolve cppcheck warnings and improve code quality.

【Warnings】
{lint_report}

【Code】
{code}

【Guidelines】
1. If there are syntax errors, fix them first. Syntax errors will be marked with type "syntax_error" in the warnings.
2. Warnings that can be automatically fixed by clang-format have already been addressed.
3. Please improve the code considering the following aspects:
   - Use meaningful variable and function names that follow naming conventions
   - Write clear and concise comments for functions and code blocks
   - Use appropriate header guards and include organization
   - Write clear and concise comments for code blocks
   - Ensure the code is self-contained and does not depend on external variables
   - Ensure the code is well-structured and easy to read
   - Ensure the code is free of errors and runs correctly
   - Ensure the code is optimized and does not have redundant operations
   - Ensure algorithms and data structures are efficient and concise
   - Are functions properly modularized with clear separation of concerns?
   - Is variable lifetime intentionally managed, avoiding frequent reassignments and overly long scopes?
   - Is error handling appropriately implemented where necessary?
   - Do comments provide context and rationale, not just code behavior?
   - Do functions have clear single responsibilities?
   - Is the code formatted for readability?
   - Are all function calls properly defined and declared in header files?
   - Are there any references to undefined functions in comments?

4. If the given code is too simple or not self-contained, enhance it to be more educational and useful:
   - Add practical examples and use cases
   - Include proper error handling and return value checking
   - Add meaningful comments explaining the logic and design decisions
   - Implement additional features that demonstrate C best practices
   - Make the code more reusable and maintainable
   - Add function documentation showing how to use the code
   - Include edge cases and their handling
   - Add proper header guards and include organization
   - Demonstrate effective use of pointers and memory management
   - Show proper use of structs and enums

5. Ensure the refactored code is syntactically correct.
"""

CPP_STYLE_REWRITING_PROMPT = """
Please refactor the following C++ code to resolve cppcheck warnings and improve code quality.

【Warnings】
{lint_report}

【Code】
{code}

【Guidelines】
1. If there are syntax errors, fix them first. Syntax errors will be marked with type "syntax_error" in the warnings.
2. Warnings that can be automatically fixed by clang-format have already been addressed.
3. Please improve the code considering the following aspects:
   - Use meaningful variable and function names that follow naming conventions
   - Write clear and concise comments for functions and classes
   - Place non-member functions in namespaces and avoid global functions
   - Don't use classes just to group static members
   - Static methods should be closely related to class instances or static data
   - Write clear and concise comments for code blocks
   - Ensure the code is self-contained and does not depend on external variables
   - Ensure the code is well-structured and easy to read
   - Ensure the code is free of errors and runs correctly
   - Ensure the code is optimized and does not have redundant operations
   - Ensure algorithms and data structures are efficient and concise
   - Are functions properly modularized with clear separation of concerns?
   - Is variable lifetime intentionally managed, avoiding frequent reassignments and overly long scopes?
   - Is error handling appropriately implemented where necessary?
   - Use C++17, C++20, or C++23 features when possible and avoid deprecated features
   - Do comments provide context and rationale, not just code behavior?
   - Do functions and classes have clear single responsibilities?
   - Is the code formatted for readability?
   - Are all function and class calls properly defined and included?
   - Are there any references to undefined functions or classes in comments?

4. If the given code is too simple or not self-contained, enhance it to be more educational and useful:
   - Add practical examples and use cases
   - Include proper exception handling
   - Add meaningful comments explaining the logic and design decisions
   - Implement additional features that demonstrate modern C++ best practices
   - Make the code more reusable and maintainable
   - Add documentation showing how to use the code
   - Include edge cases and their handling
   - Demonstrate effective use of RAII and smart pointers
   - Show proper use of templates and concepts
   - Implement proper move semantics and perfect forwarding
   - Use appropriate STL containers and algorithms
   - Demonstrate effective use of lambda expressions and ranges

5. Ensure the refactored code is syntactically correct.
"""
