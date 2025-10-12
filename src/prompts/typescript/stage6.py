TYPESCRIPT_STAGE6_PROMPT = """
Perform a final optimization pass on the TypeScript code, focusing on scalability and DX (developer experience).

Directives:
1. **Type Refinement**: leverage template literal types, satisfies clauses, and mapped types to encode invariants.
2. **Configuration Hooks**: expose factory functions or dependency injection patterns enabling easy testing and feature toggles.
3. **Performance**: highlight memoization opportunities, use lazy imports where beneficial, and avoid unnecessary re-renders in reactive frameworks.
4. **Diagnostics**: add optional verbose logging or feature-flagged tracing to debug complex flows.
5. **Educational Commentary**: explain non-trivial typing tricks or architectural choices so future engineers can maintain the system confidently.

Return the finished code after the marker as "<|REWRITTEN_CODE|>: ```typescript".
"""
