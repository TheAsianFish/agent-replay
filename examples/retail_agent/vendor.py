"""Rebuild the vendored τ-bench retail subset from a tau-bench checkout.

Usage (from the repository root)::

    python -I examples/retail_agent/vendor.py /path/to/tau-bench

The checkout is treated as data: ``tasks_test.py`` is parsed with :mod:`ast`
and its literals read with :func:`ast.literal_eval`; nothing from it is
imported or executed. The script only uses the standard library so it can run
isolated (``-I``) against an untrusted download.

It writes ``data/`` next to this file: the users of the chosen tasks, all of
their orders, the products those orders and tasks reference, the policy wiki,
the chosen tasks' ground truth and tau-bench's MIT license.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

RETAIL = Path("tau_bench") / "envs" / "retail"
OUT = Path(__file__).resolve().parent / "data"

# Indices into tau-bench's retail TASKS_TEST; see tasks.py for what each one is.
TASK_INDICES = (10, 17, 27, 53, 67, 69)


def _literal(node: ast.expr) -> Any:
    return ast.literal_eval(node)


def _call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def read_tasks(path: Path) -> list[dict[str, Any]]:
    """Every ``Task(...)`` in ``tasks_test.py``, as plain dicts, in file order."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tasks: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "Task"):
            continue
        fields = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        actions_node = fields["actions"]
        if not isinstance(actions_node, ast.List):
            raise ValueError(f"{path}: Task.actions is not a list literal")
        actions = []
        for action in actions_node.elts:
            if not isinstance(action, ast.Call):
                raise ValueError(f"{path}: an action is not an Action(...) call")
            kwargs = {kw.arg: _literal(kw.value) for kw in action.keywords}
            actions.append({"name": kwargs["name"], "arguments": kwargs["kwargs"]})
        tasks.append(
            {
                "user_id": _literal(fields["user_id"]),
                "instruction": _literal(fields["instruction"]),
                "actions": actions,
                "outputs": _literal(fields["outputs"]) if "outputs" in fields else [],
            }
        )
    # ast.walk is breadth-first; the tasks are siblings in one list, so this is file order.
    return tasks


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def build(checkout: Path) -> dict[str, Any]:
    retail = checkout / RETAIL
    all_tasks = read_tasks(retail / "tasks_test.py")
    users = load_json(retail / "data" / "users.json")
    orders = load_json(retail / "data" / "orders.json")
    products = load_json(retail / "data" / "products.json")

    chosen = [{"tau_bench_index": i, **all_tasks[i]} for i in TASK_INDICES]
    user_ids = sorted({t["user_id"] for t in chosen})
    sub_users = {uid: users[uid] for uid in user_ids}
    order_ids = sorted({oid for uid in user_ids for oid in users[uid]["orders"]})
    sub_orders = {oid: orders[oid] for oid in order_ids}
    product_ids = {item["product_id"] for o in sub_orders.values() for item in o["items"]}
    product_ids |= {
        a["arguments"]["product_id"]
        for t in chosen
        for a in t["actions"]
        if a["name"] == "get_product_details"
    }
    sub_products = {pid: products[pid] for pid in sorted(product_ids)}
    return {
        "users": sub_users,
        "orders": sub_orders,
        "products": sub_products,
        "tasks": chosen,
    }


def write(checkout: Path, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    subset = build(checkout)
    for name in ("users", "orders", "products", "tasks"):
        text = json.dumps(subset[name], indent=1, ensure_ascii=False) + "\n"
        (out / f"{name}.json").write_text(text, encoding="utf-8")
    wiki = (checkout / RETAIL / "wiki.md").read_text(encoding="utf-8")
    (out / "wiki.md").write_text(wiki, encoding="utf-8")
    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    (out / "LICENSE-tau-bench").write_text(license_text, encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: python -I {argv[0]} /path/to/tau-bench", file=sys.stderr)
        return 2
    checkout = Path(argv[1])
    if not (checkout / RETAIL / "tasks_test.py").is_file():
        print(
            f"{checkout} is not a tau-bench checkout (no {RETAIL}/tasks_test.py)", file=sys.stderr
        )
        return 2
    write(checkout)
    print(f"wrote the retail subset to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
