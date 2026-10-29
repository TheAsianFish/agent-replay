## Trace diff: `examples/pydantic_ai_agent/cassettes/qwen/cheapest_fare_in_euros.jsonl` → `qwen2.5:3b run 0`

distance 4.188 (0 same, 2 changed, 0 added, 3 removed); cost unpriced → unpriced; latency 55759 ms → 31117 ms.

| Op | Base | Head | Cost | Changes |
|---|---|---|---|---|
| changed | `0 model anthropic.beta.messages.create` | `0 model anthropic.beta.messages.create` | 0.562 | `request.model: "qwen2.5:7b" → "qwen2.5:3b"`<br>`response: {"content":[{"id":"call_7hce5hca","input":{"date":"2026-11-… → {"content":[{"id":"call_93sy9455","input":{"object":{"date"…` |
| removed | `1 tool search_flights` | — | 1 | — |
| changed | `2 model anthropic.beta.messages.create` | `1 model anthropic.beta.messages.create` | 0.625 | `request.messages[-1]: {"content":[{"content":[{"text":"{\"flight_id\":\"UA512\",\… → {"content":[{"content":[{"text":"[{\"type\":\"missing\",\"l…`<br>`request.model: "qwen2.5:7b" → "qwen2.5:3b"`<br>`response: {"content":[{"id":"call_qv17br9g","input":{"amount":189,"fr… → {"content":[{"text":"The provided details for the flight se…` |
| removed | `3 tool convert_currency` | — | 1 | — |
| removed | `4 model anthropic.beta.messages.create` | — | 1 | — |
