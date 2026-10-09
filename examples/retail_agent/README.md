# Retail agent (τ-bench retail subset)

A customer-service agent for an online store: a plain Anthropic tool-use loop
(`agent.py`) over eleven tools ported from τ-bench's retail domain (`tools.py`),
six single-turn tasks with τ-bench's ground truth (`tasks.py`), a deterministic
fake model for offline recording (`fake_model.py`), deliberately broken variants
(`mutants.py`) and domain rules (`rules.py`). One cassette per task is committed
under `cassettes/`; `make retail-cassettes` re-records them.

## Attribution

The data, policy, tool semantics and tasks come from
[τ-bench](https://github.com/sierra-research/tau-bench) by Sierra, MIT licensed
(`data/LICENSE-tau-bench`), at commit
`59a200c6d575d595120f1cb70fea53cef0632f6b`, directory `tau_bench/envs/retail/`.
Paper: Yao et al., *τ-bench: A Benchmark for Tool-Agent-User Interaction in
Real-World Domains* (2024).

## What was vendored

`vendor.py` rebuilds `data/` from a tau-bench checkout; nothing in it is edited
by hand:

```
python -I examples/retail_agent/vendor.py /path/to/tau-bench
```

It reads `tasks_test.py` with `ast` (never importing it) and keeps:

- `tasks.json`: τ-bench test tasks 10, 17, 27, 53, 67 and 69 (user, instruction,
  ground-truth actions, expected outputs);
- `users.json`: the six users of those tasks; `orders.json`: all of their
  orders; `products.json`: every product those orders contain;
- `wiki.md`: the retail policy, verbatim (it is the start of the system prompt).

| Task id | τ-bench index | What the user wants | Correct outcome |
| --- | --- | --- | --- |
| `cancel_pending_laptop` | 69 | Return a laptop, else cancel | The order is still pending: cancel it (`no longer needed`) |
| `modify_address_suite` | 17 | Ship a pending order to another suite | `modify_pending_order_address` |
| `return_damaged_bicycle` | 53 | Return a damaged bicycle to the original card | `return_delivered_order_items` |
| `exchange_waterproof_boots` | 27 | Same hiking boots but waterproof | `exchange_delivered_order_items` |
| `lookup_latest_order_total` | 67 | How much the latest order cost | Answer `829.43`; no writes |
| `transfer_swapped_refunds` | 10 | Refund each order to the other order's card | Against policy: transfer to a human; no writes |

## How it was adapted

- **Single turn.** τ-bench plays the user with an LLM over several turns. Here
  each task's instruction states the user's identity and the whole request up
  front, and gives the confirmation that the policy requires before a write.
  Where τ-bench's user would branch on the agent's answers (task 27 offers a
  return and an exchange, task 67 gives two wrong zip codes first), the
  instruction asks for the branch the ground truth takes. The original
  instruction is kept in `Task.source_instruction`. The system prompt is the
  policy followed by a short note describing this setting, which also allows
  several read-only tool calls per turn (the policy says one at a time).
- **Ground truth unchanged.** `Task.expected_actions` are τ-bench's actions
  other than reads (writes and the transfer), `expected_outputs` its outputs.
  `expected_db_hash(task)` resets the database, applies those actions with the
  real tools and hashes the result, so "final database equals expected
  database" is τ-bench's reward.
- **Tools.** Same names, arguments, behaviour and error messages, with two
  changes: failures raise `ValueError("<message>")` and the agent reports them
  as `is_error` results reading `Error: <message>` (τ-bench returns that
  string); detail tools return the record as JSON data rather than a JSON
  string. `calculate` evaluates arithmetic with an AST walk instead of `eval`.
  Only the eleven tools listed in `tools.ALL_TOOLS` are ported (not
  `list_all_product_types`, `modify_pending_order_items`,
  `modify_pending_order_payment`, `modify_user_address` or `think`).
