JAVASCRIPT_STAGE6_PROMPT = """
Apply a polishing pass to the rewritten JavaScript code, emphasizing robustness and maintainability for production services.

Rules:
1. **Async Discipline**: favour `async/await`, ensure all promises are awaited or handled, and use `AbortController` for cancellable workflows.
2. **Observability**: integrate optional logging/metrics hooks (behind feature flags) to assist debugging in large deployments.
3. **Security**: sanitize user-facing output, guard against injection, and validate external API responses.
4. **Testing Strategy**: design exported functions with dependency injection for mocking; include comments about expected unit/integration tests.
5. **Educational Notes**: briefly explain advanced patterns (e.g., memoization, debounce) to help future maintainers.

Return the final code after the marker in the form "<|REWRITTEN_CODE|>: ```javascript".
"""
