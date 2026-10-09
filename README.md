# Replay — record-and-replay regression testing for LLM agents

**Status: in progress (planning complete, building starts at M1). Nothing below
is installable yet.**

Replay records a real run of your tool-using agent (every model call, tool
call, clock read and random draw) into a cassette, then replays it on every PR
with the cut points you choose: everything mocked for a zero-network test of
your orchestration code, or the model live against a frozen world of recorded
tool results to catch prompt and model regressions. It asserts on the tool-call
sequence, structured-output schema, rule checks, cost and latency budgets, and a
rubric-driven LLM judge. Because agents are not deterministic, a scenario is
not pass/fail from one run: Replay re-runs it, estimates the pass probability
with a confidence interval, stops early with a sequential test, and tells flaky
from broken. Because judges are not oracles, judge checks are advisory until the
judge has been calibrated against human labels (Cohen's kappa with a bootstrap
CI and bias probes). Ships as `pip install agent-replay`, a GitHub Action that
fails PRs on regressions, and a self-hosted React dashboard with aligned trace
diffs.

## Planned quickstart

```bash
pip install agent-replay              # not published yet
replay record --agent myagent:run --task "cancel order #123" -o cassettes/cancel.jsonl
replay run cassettes/cancel.jsonl --mode full          # zero network
replay ci --mode model-live --spec replay.yaml          # statistical gate
replay calibrate --golden eval/golden --rubric rubrics/retail.yaml
```

## Documents

- [PLAN.md](PLAN.md): goal, architecture, evaluation plan, milestones.
- [CONTRIBUTING.md](CONTRIBUTING.md): commands, conventions and quality gates for contributors.
- RESULTS.md (from M4): every measured number with its command and raw output.

## License

MIT.
