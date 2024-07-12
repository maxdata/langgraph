#!/usr/bin/env python
# coding: utf-8

# # How to create a custom checkpointer using Postgres
# 
# When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.
# 
# This example shows how to use `Postgres` as the backend for persisting checkpoint state.
# 
# NOTE: this is just an example implementation. You can implement your own checkpointer using a different database or modify this one as long as it conforms to the `BaseCheckpointSaver` interface.

# ## Checkpointer implementation



get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install -U psycopg psycopg-pool langgraph\n')




"""Implementation of a langgraph checkpoint saver using Postgres."""
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Optional,
    Union,
    Tuple,
    List,
)

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from psycopg_pool import AsyncConnectionPool, ConnectionPool


class JsonAndBinarySerializer(JsonPlusSerializer):
    def _default(self, obj):
        if isinstance(obj, (bytes, bytearray)):
            return self._encode_constructor_args(
                obj.__class__, method="fromhex", args=[obj.hex()]
            )
        return super()._default(obj)

    def dumps(self, obj: Any) -> tuple[str, bytes]:
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj

        return "json", super().dumps(obj)

    def loads(self, s: tuple[str, bytes]) -> Any:
        if s[0] == "bytes":
            return s[1]
        elif s[0] == "bytearray":
            return bytearray(s[1])
        elif s[0] == "json":
            return super().loads(s[1])
        else:
            raise NotImplementedError(f"Unknown serialization type: {s[0]}")


@contextmanager
def _get_sync_connection(
    connection: Union[psycopg.Connection, ConnectionPool, None],
) -> Generator[psycopg.Connection, None, None]:
    """Get the connection to the Postgres database."""
    if isinstance(connection, psycopg.Connection):
        yield connection
    elif isinstance(connection, ConnectionPool):
        with connection.connection() as conn:
            yield conn
    else:
        raise ValueError(
            "Invalid sync connection object. Please initialize the check pointer "
            f"with an appropriate sync connection object. "
            f"Got {type(connection)}."
        )


@asynccontextmanager
async def _get_async_connection(
    connection: Union[psycopg.AsyncConnection, AsyncConnectionPool, None],
) -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Get the connection to the Postgres database."""
    if isinstance(connection, psycopg.AsyncConnection):
        yield connection
    elif isinstance(connection, AsyncConnectionPool):
        async with connection.connection() as conn:
            yield conn
    else:
        raise ValueError(
            "Invalid async connection object. Please initialize the check pointer "
            f"with an appropriate async connection object. "
            f"Got {type(connection)}."
        )


class PostgresSaver(BaseCheckpointSaver):
    sync_connection: Optional[Union[psycopg.Connection, ConnectionPool]] = None
    """The synchronous connection or pool to the Postgres database.
    
    If providing a connection object, please ensure that the connection is open
    and remember to close the connection when done.
    """
    async_connection: Optional[
        Union[psycopg.AsyncConnection, AsyncConnectionPool]
    ] = None
    """The asynchronous connection or pool to the Postgres database.
    
    If providing a connection object, please ensure that the connection is open
    and remember to close the connection when done.
    """

    def __init__(
        self,
        sync_connection: Optional[Union[psycopg.Connection, ConnectionPool]] = None,
        async_connection: Optional[
            Union[psycopg.AsyncConnection, AsyncConnectionPool]
        ] = None,
    ):
        super().__init__(serde=JsonPlusSerializer())
        self.sync_connection = sync_connection
        self.async_connection = async_connection

    @contextmanager
    def _get_sync_connection(self) -> Generator[psycopg.Connection, None, None]:
        """Get the connection to the Postgres database."""
        with _get_sync_connection(self.sync_connection) as connection:
            yield connection

    @asynccontextmanager
    async def _get_async_connection(
        self,
    ) -> AsyncGenerator[psycopg.AsyncConnection, None]:
        """Get the connection to the Postgres database."""
        async with _get_async_connection(self.async_connection) as connection:
            yield connection

    CREATE_TABLES_QUERY = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        thread_ts TEXT NOT NULL,
        parent_ts TEXT,
        checkpoint BYTEA NOT NULL,
        metadata BYTEA NOT NULL,
        PRIMARY KEY (thread_id, thread_ts)
    );
    """

    @staticmethod
    def create_tables(connection: Union[psycopg.Connection, ConnectionPool], /) -> None:
        """Create the schema for the checkpoint saver."""
        with _get_sync_connection(connection) as conn:
            with conn.cursor() as cur:
                cur.execute(PostgresSaver.CREATE_TABLES_QUERY)

    @staticmethod
    async def acreate_tables(
        connection: Union[psycopg.AsyncConnection, AsyncConnectionPool], /
    ) -> None:
        """Create the schema for the checkpoint saver."""
        async with _get_async_connection(connection) as conn:
            async with conn.cursor() as cur:
                await cur.execute(PostgresSaver.CREATE_TABLES_QUERY)

    @staticmethod
    def drop_tables(connection: psycopg.Connection, /) -> None:
        """Drop the table for the checkpoint saver."""
        with connection.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS checkpoints;")

    @staticmethod
    async def adrop_tables(connection: psycopg.AsyncConnection, /) -> None:
        """Drop the table for the checkpoint saver."""
        async with connection.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS checkpoints;")

    UPSERT_CHECKPOINT_QUERY = """
    INSERT INTO checkpoints 
        (thread_id, thread_ts, parent_ts, checkpoint, metadata)
    VALUES 
        (%s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, thread_ts)
    DO UPDATE SET checkpoint = EXCLUDED.checkpoint,
                  metadata = EXCLUDED.metadata;
    """

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.
        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self.UPSERT_CHECKPOINT_QUERY,
                    (
                        thread_id,
                        checkpoint["ts"],
                        parent_ts if parent_ts else None,
                        self.serde.dumps(checkpoint),
                        self.serde.dumps(metadata),
                    ),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.
        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    self.UPSERT_CHECKPOINT_QUERY,
                    (
                        thread_id,
                        checkpoint["ts"],
                        parent_ts if parent_ts else None,
                        self.serde.dumps(checkpoint),
                        self.serde.dumps(metadata),
                    ),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

    LIST_CHECKPOINTS_QUERY_STR = """
    SELECT checkpoint, metadata, thread_ts, parent_ts
    FROM checkpoints
    {where}
    ORDER BY thread_ts DESC
    """

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Generator[CheckpointTuple, None, None]:
        """Get all the checkpoints for the given configuration."""
        where, args = self._search_where(config, filter, before)
        query = self.LIST_CHECKPOINTS_QUERY_STR.format(where=where)
        if limit:
            query += f" LIMIT {limit}"
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                cur.execute(query, tuple(args))
                for value in cur:
                    checkpoint, metadata, thread_ts, parent_ts = value
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        },
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }
                        if parent_ts
                        else None,
                    )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Get all the checkpoints for the given configuration."""
        where, args = self._search_where(config, filter, before)
        query = self.LIST_CHECKPOINTS_QUERY_STR.format(where=where)
        if limit:
            query += f" LIMIT {limit}"
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                await cur.execute(query, tuple(args))
                async for value in cur:
                    checkpoint, metadata, thread_ts, parent_ts = value
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        },
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }
                        if parent_ts
                        else None,
                    )

    GET_CHECKPOINT_BY_TS_QUERY = """
    SELECT checkpoint, metadata, thread_ts, parent_ts
    FROM checkpoints
    WHERE thread_id = %(thread_id)s AND thread_ts = %(thread_ts)s
    """

    GET_CHECKPOINT_QUERY = """
    SELECT checkpoint, metadata, thread_ts, parent_ts
    FROM checkpoints
    WHERE thread_id = %(thread_id)s
    ORDER BY thread_ts DESC LIMIT 1
    """

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.
            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                if thread_ts:
                    cur.execute(
                        self.GET_CHECKPOINT_BY_TS_QUERY,
                        {
                            "thread_id": thread_id,
                            "thread_ts": thread_ts,
                        },
                    )
                    value = cur.fetchone()
                    if value:
                        checkpoint, metadata, thread_ts, parent_ts = value
                    return CheckpointTuple(
                        config=config,
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }
                        if thread_ts
                        else None,
                    )
                else:
                    cur.execute(
                        self.GET_CHECKPOINT_QUERY,
                        {
                            "thread_id": thread_id,
                        },
                    )
                    value = cur.fetchone()
                    if value:
                        checkpoint, metadata, thread_ts, parent_ts = value
                        return CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": thread_ts,
                                }
                            },
                            checkpoint=self.serde.loads(checkpoint),
                            metadata=self.serde.loads(metadata),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": parent_ts,
                                }
                            }
                            if parent_ts
                            else None,
                        )
        return None

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.
            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                if thread_ts:
                    await cur.execute(
                        self.GET_CHECKPOINT_BY_TS_QUERY,
                        {
                            "thread_id": thread_id,
                            "thread_ts": thread_ts,
                        },
                    )
                    value = await cur.fetchone()
                    if value:
                        checkpoint, metadata, thread_ts, parent_ts = value
                    return CheckpointTuple(
                        config=config,
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }
                        if thread_ts
                        else None,
                    )
                else:
                    await cur.execute(
                        self.GET_CHECKPOINT_QUERY,
                        {
                            "thread_id": thread_id,
                        },
                    )
                    value = await cur.fetchone()
                    if value:
                        checkpoint, metadata, thread_ts, parent_ts = value
                        return CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": thread_ts,
                                }
                            },
                            checkpoint=self.serde.loads(checkpoint),
                            metadata=self.serde.loads(metadata),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": parent_ts,
                                }
                            }
                            if parent_ts
                            else None,
                        )

        return None

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for given config, filter, and before parameters.
        Args:
            config (Optional[RunnableConfig]): The config to use for filtering.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria.
            before (Optional[RunnableConfig]): A config to limit results before a certain timestamp.
        Returns:
            Tuple[str, Sequence[Any]]: A tuple containing the WHERE clause and parameter values.
        """
        wheres = []
        param_values = []

        # Add predicate for config
        if config is not None:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])

        if filter:
            raise NotImplementedError()

        # Add predicate for limiting results before a certain timestamp
        if before is not None:
            wheres.append("thread_ts < %s")
            param_values.append(before["configurable"]["thread_ts"])

        where_clause = "WHERE " + " AND ".join(wheres) if wheres else ""
        return where_clause, param_values


# ## Setup environment



import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")


# ## Setup model and tools for the graph



from typing import Literal
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o", temperature=0)


# ## Use sync connection



DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"


# ### With a connection pool



from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
)

checkpointer = PostgresSaver(sync_connection=pool)
checkpointer.create_tables(pool)




graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}
res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)




res




checkpointer.get(config)


# ### With a connection



from psycopg import Connection

with Connection.connect(DB_URI) as conn:
    checkpointer = PostgresSaver(sync_connection=conn)

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuple = checkpointer.get_tuple(config)




checkpoint_tuple


# ## Use async connection

# ### With a connection pool



from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
)

checkpointer = PostgresSaver(async_connection=pool)
await checkpointer.acreate_tables(pool)




graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "3"}}
res = await graph.ainvoke(
    {"messages": [("human", "what's the weather in nyc")]}, config
)




checkpoint_tuple = await checkpointer.aget_tuple(config)




checkpoint_tuple


# ### Use connection



from psycopg import AsyncConnection

async with await AsyncConnection.connect(DB_URI) as conn:
    checkpointer = PostgresSaver(async_connection=conn)
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "4"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]




checkpoint_tuples






