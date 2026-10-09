"""A small customer-support agent: look up an order, check policy, maybe refund.

``agent.run`` is the agent, ``fake_model.make_client`` a deterministic model for
offline recording, and ``mutants`` holds deliberately broken variants that the
replay checks must catch.
"""
