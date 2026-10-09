# Results

Every number in README.md comes from this file. Each one lists its dataset,
method, the command that produced it and the raw output. Sections are added as
the evaluations land. PLAN.md, "Evaluation", lists the full plan.

## Live recording smoke test (M1)

Checks that the recorder works against the real Anthropic API and that a
real recording replays offline exactly. This is a single run, not a
benchmark: the latencies are one sample each.

- **Dataset:** one task for the toy agent, "Order A1001 arrived damaged. Can I
  get a refund?"
- **Model:** `claude-opus-5-5`, `output_config={"effort": "low"}`,
  `max_tokens=4096`. Cost uses the prices in `agent_replay/pricing.py`
  (captured 2026-10-08: $4 / $20 per MTok input / output).
- **Date:** 2026-10-09.
- **Command:** `uv run replay record examples.toy_agent.agent:run --task "Order A1001 arrived damaged. Can I get a refund?" -o examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl`

Raw output:

```
#  kind   name                       status  request
0  model  anthropic.messages.create  live    "1 message(s)"
1  tool   lookup_order               live    {"order_id":"A1001"}
2  tool   get_refund_policy          live    {}
3  model  anthropic.messages.create  live    "3 message(s)"
4  tool   issue_refund               live    {"amount":42.5,"order_id":"A1001"}
5  model  anthropic.messages.create  live    "5 message(s)"
Recorded 6 envelope(s) to
examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl (2687 tokens,
$0.0140)
Output: I'm sorry your order arrived damaged — I've refunded the full $42.50 for
order A1001 (refund ID RF-A1001).
```

Per-call usage, read from the committed cassette:

| Call | Content returned | Input tokens | Output tokens | Cost (USD) | Latency (ms) |
|---|---|---|---|---|---|
| 1 | 2 × `tool_use` (parallel) | 655 | 80 | 0.004220 | 11478.5 |
| 2 | 1 × `tool_use` | 856 | 77 | 0.004964 | 11233.9 |
| 3 | `text` | 972 | 47 | 0.004828 | 5156.2 |

Offline replay of that cassette (`uv run replay run examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl`):
all 6 steps `exact`, `PASS tool_sequence`, exit 0. The same test runs as
`make test-live` (`tests/test_live_anthropic.py`). It records a fresh live run,
then replays it offline. It passed twice in this session, before and after
the review fixes (last run: `1 passed, 132 deselected in 5.71s`). Its cost was
not captured separately. The mutant
(`--agent examples.toy_agent.mutants:first_tool_only`) also fails on the live
cassette (exit 1).

## To measure

Nothing yet. Key-requiring evaluations (judge kappa, LLMBar, flake suite) are
listed here, with exact commands, when their harnesses exist (M3 to M6).
