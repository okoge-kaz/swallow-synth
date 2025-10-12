CPP_STAGE4_PROMPT = """
You are an expert modern C++ developer. Rewrite the supplied translation unit so that it embraces modern C++17 best practices while keeping behaviour equivalent.

Guidelines:
1. **Modern Constructs**: favour smart pointers, `std::optional`, `std::array`, ranges utilities, and RAII abstractions over raw resource management.
2. **Readability**: organise code into namespaces and classes or free functions with crisp responsibilities; eliminate macros when constexpr alternatives exist.
3. **Documentation**: add precise Doxygen-style comments for public APIs; explain tricky template or concurrency details.
4. **Error Handling**: use exceptions or `expected`-like patterns consistently; avoid naked `throw` without context.
5. **Thread Safety**: annotate shared state and choose appropriate synchronisation primitives when concurrency is implied.
6. **Performance**: leverage move semantics, emplace operations, and algorithms from `<algorithm>` or `<numeric>` to reduce manual loops.
7. **Testing Considerations**: design interfaces that are unit test friendly (dependency injection, pure functions, deterministic behaviour).

Output the rewritten code exactly after the marker as "<|REWRITTEN_CODE|>: ```cpp".
"""
