"""Alembic environment for the run store.

Migrations only run through :func:`agent_replay.store.migrate`, which passes an
open connection already inside the transaction (and lock) it wants them in.
There is no ``alembic.ini``: run ``replay db upgrade`` to migrate a store by hand.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from agent_replay.store import Base

connection = context.config.attributes.get("connection")
if not isinstance(connection, Connection):
    raise RuntimeError(
        "the run store's migrations run through agent_replay.store.migrate(); "
        "use `replay db upgrade` instead of the alembic command line"
    )
context.configure(
    connection=connection,
    target_metadata=Base.metadata,
    transactional_ddl=True,
    render_as_batch=True,
)
with context.begin_transaction():
    context.run_migrations()
