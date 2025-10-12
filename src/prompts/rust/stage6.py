RUST_STAGE6_PROMPT = """
Perform a final optimization pass on the refined Rust code, emphasizing zero-cost abstractions and clarity.

Guidelines:
1. **Zero-Cost Abstractions**: replace dynamic dispatch with generics where it improves performance; highlight trade-offs in comments.
2. **Diagnostics**: add `debug_assert!` or `trace!` hooks (guarded by feature flags) for difficult-to-reproduce issues.
3. **Unsafe Justification**: if unsafe blocks are necessary, encapsulate them in safe APIs with explicit safety documentation.
4. **Crate Structure**: ensure modules expose minimal public API; use `pub(crate)`/`pub(super)` as needed.
5. **Educational Remarks**: explain advanced borrow checker workarounds or lifetime annotations so maintainers can reason about them.

Return the final code after the marker as "<|REWRITTEN_CODE|>: ```rust".
"""
