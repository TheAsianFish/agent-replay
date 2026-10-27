# Travel desk: a pydantic-ai agent under Replay

A tool-using [pydantic-ai](https://github.com/pydantic/pydantic-ai) (MIT) agent
that searches flights, converts currencies, books and cancels. It is here to
show Replay on a framework it did not write: nothing in `agent.py` knows about
recording except the `@tool` decorator on the tools in `tools.py`.

pydantic-ai takes an Anthropic client through its provider and calls
`client.beta.messages.create` on it, which Replay records:

```python
from anthropic import AsyncAnthropic
from pydantic_ai.providers.anthropic import AnthropicProvider
import agent_replay as replay

client = replay.wrap(AsyncAnthropic(base_url="http://localhost:11434", api_key="ollama"))
provider = AnthropicProvider(anthropic_client=client)  # see build_agent() in agent.py
```

The `base_url` is a local [Ollama](https://ollama.com) server, which speaks the
Anthropic Messages API, so the live model is `qwen2.5:7b` on your CPU and costs
nothing. Point it at the Anthropic API instead and nothing else changes.

## Try it

```bash
uv run replay ci examples/pydantic_ai_agent/replay.toml        # fake-model recordings, offline
uv run replay ci examples/pydantic_ai_agent/replay-qwen.toml   # qwen2.5:7b recordings, offline

# With Ollama: `ollama serve` and `ollama pull qwen2.5:7b`
make eval-pydantic-ai     # replay ci in full, then replay flake in model-live (10 runs each)
make record-pydantic-ai   # re-record the qwen2.5:7b cassettes
```

| File | What it is |
|---|---|
| `agent.py` | The agent (`run`, and `run_streaming` through `agent.run_stream`), and `make_client` for Ollama (`REPLAY_OLLAMA_URL`, default `http://localhost:11434`) |
| `tools.py` | Five tools over a fixed world of six flights and two bookings |
| `tasks.py` | Three tasks and their right answers |
| `rules.py` | `rule` checks on outcomes: the writes, the figure in the answer, lookup before write |
| `replay.toml`, `replay-qwen.toml` | The suites: `in_order` sequences and the rules, over the fake and the qwen2.5:7b recordings |
| `fake_model.py` | A deterministic model, so the fake cassettes record with no server |
| `record_all.py` | Re-records either set; `--live` keeps the first attempt per task that passes the checks and logs every attempt |
| `environment.py` | The CPU, Ollama version and model digest a live run is reported with |

The checks judge outcomes, not the recorded path: a sequence check in
`in_order` mode lets the model look things up more than once, and the rules
check the bookings and the figure in the answer. An `exact` sequence check
would measure agreement with one recording, which a different model fails for
the wrong reason.

The measured results (what broke on first contact, and the `model-live` flake
on qwen2.5:7b) are in [RESULTS.md](../../RESULTS.md).
