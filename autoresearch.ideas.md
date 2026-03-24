# Deferred Ideas

- If continuing on tokenizer internals, focus on semantically safe fast paths that preserve exact semantics, like extending the current ASCII ids-only path to more provably safe input classes. Avoid benchmark-specific memoization of whole prompts.
- Re-measure phase timing after the new ids-only path and reassess whether embedding-side work is now large enough to revisit with fresh data.
