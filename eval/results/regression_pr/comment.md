<!-- agent-replay: examples/retail_agent/replay.toml -->
## ❌ Replay: 2/6 scenarios passed

Suite `examples/retail_agent/replay.toml` failed at `779675f4eaaa` in 0.26s.

| Scenario | Mode | Result | Checks | Steps | Cost |
|---|---|---|---|---|---|
| `cancel_pending_laptop` | full | ❌ fail | 6/7 | 1 approximate, 10 exact | $0.0000 |
| `modify_address_suite` | full | ❌ fail | 6/7 | 1 approximate, 7 exact | $0.0000 |
| `return_damaged_bicycle` | full | ❌ fail | 6/7 | 1 approximate, 11 exact | $0.0000 |
| `exchange_waterproof_boots` | full | ❌ fail | 6/7 | 1 approximate, 13 exact | $0.0000 |
| `lookup_latest_order_total` | full | ✅ pass | 8/8 | 7 exact | $0.0000 |
| `transfer_swapped_refunds` | full | ✅ pass | 8/8 | 10 exact | $0.0000 |

### ❌ `cancel_pending_laptop`

Cassette `examples/retail_agent/cassettes/cancel_pending_laptop.jsonl`, agent `examples.retail_agent.mutants:wrong_write_argument`, full / strict.
- **rule:matches_ground_truth**: action #1: expected ('cancel_pending_order', {'order_id': '#W2417020', 'reason': 'no longer needed'}), got ('cancel_pending_order', {'order_id': '#W2417020', 'reason': 'ordered by mistake'})

### ❌ `modify_address_suite`

Cassette `examples/retail_agent/cassettes/modify_address_suite.jsonl`, agent `examples.retail_agent.mutants:wrong_write_argument`, full / strict.
- **rule:matches_ground_truth**: action #1: expected ('modify_pending_order_address', {'order_id': '#W8665881', 'address1': '123 Elm Street', 'address2': 'Suite 641', 'city': 'Austin', 'state': 'TX', 'country': 'USA', 'zip': '78712'}), got ('modify_pending_order_address', {'order_id': '#W8665881', 'address1': '123 Elm Street', 'address2': '', 'city': 'Austin', 'state': 'TX', 'country': 'USA', 'zip': '78712'})

### ❌ `return_damaged_bicycle`

Cassette `examples/retail_agent/cassettes/return_damaged_bicycle.jsonl`, agent `examples.retail_agent.mutants:wrong_write_argument`, full / strict.
- **rule:matches_ground_truth**: action #1: expected ('return_delivered_order_items', {'order_id': '#W3916020', 'item_ids': ['7758198585'], 'payment_method_id': 'credit_card_8105988'}), got ('return_delivered_order_items', {'order_id': '#W3916020', 'item_ids': ['7758198585'], 'payment_method_id': 'gift_card_0000000'})

### ❌ `exchange_waterproof_boots`

Cassette `examples/retail_agent/cassettes/exchange_waterproof_boots.jsonl`, agent `examples.retail_agent.mutants:wrong_write_argument`, full / strict.
- **rule:matches_ground_truth**: action #1: expected ('exchange_delivered_order_items', {'order_id': '#W7181492', 'item_ids': ['8118291112'], 'new_item_ids': ['8277474082'], 'payment_method_id': 'paypal_3024827'}), got ('exchange_delivered_order_items', {'order_id': '#W7181492', 'item_ids': ['8118291112'], 'new_item_ids': ['8277474082'], 'payment_method_id': 'gift_card_0000000'})

---
[Run log and report](https://github.com/TheAsianFish/agent-replay/actions/runs/40380755513)
