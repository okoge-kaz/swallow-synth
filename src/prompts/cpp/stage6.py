CPP_STAGE6_PROMPT = """
Perform a second pass on the previously refined C++ code, concentrating on advanced expressiveness and diagnostics.

Rules:
1. **Template Expressiveness**: lift duplicate logic into templated helpers or concepts (C++20 optional) while keeping the interface intuitive.
2. **Static Guarantees**: introduce `static_assert` checks and strong type aliases to encode invariants at compile time.
3. **Error Surfaces**: integrate `std::expected` or outcome-style result types to unify error handling, documenting failure modes in comments.
4. **Benchmark Mindset**: highlight hotspots and add TODO notes with profiling guidance where further tuning might be warranted.
5. **Educational Commentary**: annotate advanced idioms (e.g., perfect forwarding, CRTP) so future readers understand the rationale.

Return the enhanced code after the marker using "<|REWRITTEN_CODE|>: ```cpp".
"""
