"""A retail customer-service agent on a vendored τ-bench retail subset (MIT, Sierra).

``agent.run`` is the agent, ``tools`` the ported τ-bench tools over a resettable
in-memory database, ``tasks`` six single-turn tasks with tau-bench's ground
truth, ``fake_model.make_client`` a deterministic model for offline recording,
``mutants`` deliberately broken variants and ``rules`` domain checks. See
README.md for attribution and how the data was adapted.
"""
