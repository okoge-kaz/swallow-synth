GO_STAGE6_PROMPT = """
Apply a second refinement pass to the previously rewritten Go code, focusing on performance insights and production resilience.

Guidelines:
1. **Performance Awareness**: note allocation hotspots, consider slice capacity preallocation, and prefer `sync.Pool` or batching when useful.
2. **Context Propagation**: ensure every public function that performs I/O or blocking work accepts a `context.Context`.
3. **Observability**: add optional structured logging hooks and metrics (guarded behind interfaces) for tracing in large systems.
4. **Failure Modes**: classify errors with sentinel values or wrapping, and document retry semantics where appropriate.
5. **Educational Notes**: briefly explain advanced idioms (e.g., worker pools, generics) so future maintainers understand the approach.

Emit the refined code after the marker as "<|REWRITTEN_CODE|>: ```go".
"""
