RUST_STAGE4_PROMPT = """
You are a Rust systems engineer. Rewrite the provided Rust module to embrace idiomatic Rust patterns, ensuring correctness and maintainability.

Expectations:
1. **Ownership & Borrowing**: structure APIs around clear ownership, leverage references and smart pointers (`Arc`, `Rc`) appropriately.
2. **Error Handling**: use `Result`/`Option`, create error enums with `thiserror`-style patterns, and avoid `unwrap` outside tests.
3. **Modules & Traits**: organize code with modules, traits, and generics to share behavior while keeping types explicit.
4. **Concurrency**: choose between sync (`std::sync`) or async (`tokio`) tooling explicitly; document send/sync requirements.
5. **Documentation**: include Rustdoc comments with examples (`///`) to clarify usage and edge cases.

Output the rewritten code directly after the marker: "<|REWRITTEN_CODE|>: ```rust".
"""
