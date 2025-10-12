CUDA_STAGE6_PROMPT = """
Apply a polishing pass to the refined CUDA code, emphasizing performance nuance and maintainability for large scale deployments.

Guidelines:
1. **Occupancy Tuning**: note launch configuration trade-offs, add constants for block sizes, and expose parameters for autotuning where sensible.
2. **Memory Optimizations**: coalesce global memory accesses, highlight opportunities for shared memory tiling, and document bank conflict avoidance strategies.
3. **Asynchrony**: structure streams and events explicitly, ensuring all asynchronous work is checked with `cudaGetLastError()` and `cudaDeviceSynchronize()` in debug builds.
4. **Instrumentation**: include optional timing or tracing hooks (guarded by preprocessor flags) to help diagnose performance regressions.
5. **Educational Notes**: explain any advanced intrinsics or warp-level primitives so fellow engineers understand their purpose.

Return the final code after the marker in the form "<|REWRITTEN_CODE|>: ```cuda".
"""
