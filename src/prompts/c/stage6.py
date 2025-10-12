C_STAGE6_PROMPT = """
You are refining high quality C code for performance and robustness. Apply an additional pass that focuses on measurable improvements without sacrificing clarity.

Rules:
1. **Algorithmic Tightening**: replace naive loops with efficient patterns, prefer \"size_t\" for counts, and minimise redundant work.
2. **Memory Discipline**: centralise allocation/free logic, add explicit lifetime comments for complex ownership, and use const correctness aggressively.
3. **Error Propagation**: convert silent failures into explicit error codes or errno updates, ensuring callers can react appropriately.
4. **Instrumentation**: where helpful, insert lightweight assertions for invariants in debug builds (use `#ifdef NDEBUG`).
5. **Educational Value**: include brief comments that describe the motivation behind non obvious optimizations.

Emit the refined code after the marker in the form "<|REWRITTEN_CODE|>: ```c".
"""
