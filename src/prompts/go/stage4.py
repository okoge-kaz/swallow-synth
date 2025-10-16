GO_STAGE4_PROMPT = """
You are an experienced Go engineer. Rewrite the provided Go source so that it follows idiomatic Go conventions while preserving behaviour.

Instructions:
1. **Idiomatic Style**: use standard naming, short variable names within small scopes, and organise code into clear packages and files.
2. **Error Handling**: check returned errors explicitly, wrap with context using `fmt.Errorf` when helpful, and avoid panic unless appropriate.
3. **Concurrency Safety**: prefer channels, goroutines, and `context.Context` patterns; document expectations around cancellation.
4. **Resource Management**: close resources (`defer file.Close()`) promptly and avoid goroutine leaks by respecting contexts.
5. **Testing Hooks**: design exported functions with deterministic behaviour and dependency injection to simplify unit testing.
6. **Documentation**: add concise comments for exported identifiers following `godoc` style.

Return the final code right after the marker as "<|REWRITTEN_CODE|>: ```go".
"""
