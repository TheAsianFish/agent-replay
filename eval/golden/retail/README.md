# Retail golden set

The golden set that the retail judge (`examples/retail_agent/rubric.yaml`) is
calibrated against. Rebuild it with `make golden`; the build is offline, never
runs the judge, and re-running it leaves both files byte-identical.

## What is in it

`make golden` replays the six live `claude-opus-5-5` retail recordings
(`examples/retail_agent/replay-live.toml`, τ-bench retail tasks) in `full` mode
with the real agent and with the three seeded mutants that `eval/judge_smoke.py`
also uses (`wrong_write_argument`, `skips_order_lookup`, `first_tool_only`):
24 runs. Runs that the judge would see identically become one item, so the set
holds 21 items.

- `items.jsonl`: one item per line, the judge's view of a run (task, tool calls
  with inputs and results, final output) and the runs it came from (`sources`:
  suite, scenario, agent variant, cassette, the model that drove the agent).
- `labels.jsonl`: append-only labels; for each item, criterion, source
  (`objective` or `human`) and labeller the last line counts.

## Labels

**Objective** labels (`source: objective`, `labeller: checks`, criterion
`_overall`) come from the suite's deterministic checks, which include τ-bench's
ground truth (`matches_ground_truth`, `db_matches_ground_truth`): an item passes
when the agent finished and every check of the run's behaviour passed. `budget`
checks (cost, tokens, latency and model-call counts against a baseline) do not
count: a transcript shows none of that. Their `note` lists the checks that
failed, and `labelled_at` is the recording time of the cassette.

**Human** labels answer one rubric criterion each. Label the set with

```
uv run replay label eval/golden/retail --rubric examples/retail_agent/rubric.yaml --labeller <name>
```

(or set `REPLAY_LABELLER`). Labelling is blind: the session shows the transcript
and the criterion, never the judge's verdict, the objective label or which
variant produced the run. Answers are saved as they are given; run the command
again to resume (skipped questions are asked again first). Editing a criterion's text makes its earlier labels stale
(`uv run replay golden show eval/golden/retail --rubric examples/retail_agent/rubric.yaml`
counts them), and they need relabelling.
