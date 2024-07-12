#!/usr/bin/env python
# coding: utf-8

# # Get/Update State
# 
# When running LangGraph agents, you can easily save good threads and use them in the future.
# 
# **Note:** this requires passing in a checkpointer.
# 

# ## Setup
# 
# First we need to install the packages required
# 



get_ipython().system('%pip install --quiet -U langgraph langchain langchain_openai tavily-pythonvily-python')


# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# 



import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")


# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.
# 



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")


# ## Set up the tools
# 
# We will first define the tools we want to use.
# For this simple example, we will use a built-in search tool via Tavily.
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/v0.2/docs/how_to/custom_tools) on how to do that.
# 



from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]


# We can now wrap these tools in a simple ToolNode.
# This is a prebuilt node that extracts tool calls from the most recent AIMessage, executes them, and returns a ToolMessage with the results.
# 



from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)


# ## Set up the model
# 
# Now we need to load the chat model we want to use.
# Importantly, this should satisfy two criteria:
# 
# 1. It should work with messages. We will represent all agent state in the form of messages, so it needs to be able to work well with them.
# 2. It should work with OpenAI function calling. This means it should either be an OpenAI model or a model that exposes a similar interface.
# 
# Note: these model requirements are not requirements for using LangGraph - they are just requirements for this one example.
# 



from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this using the `.bind_tools()` method, common to many of LangChain's chat models.
# 



model = model.bind_tools(tools)


# ## Define the nodes
# 
# We now need to define a few different nodes in our graph.
# In `langgraph`, a node can be either a function or a [runnable](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel).
# There are two main nodes we need for this:
# 
# 1. The agent: responsible for deciding what (if any) actions to take.
# 2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action.
# 
# We will also need to define some edges.
# Some of these edges may be conditional.
# The reason they are conditional is that based on the output of a node, one of several paths may be taken.
# The path that is taken is not known until that node is run (the LLM decides).
# 
# 1. Conditional Edge: after the agent is called, we should either:
#    a. If the agent said to take an action, then the function to invoke tools should be called
#    b. If the agent said that it was finished, then it should finish
# 2. Normal Edge: after the tools are invoked, it should always go back to the agent to decide what to do next
# 
# Let's define the nodes, as well as a function to decide how what conditional edge to take.
# 



# Define the function that determines whether to continue or not
def should_continue(state):
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# ## Define the graph
# 
# We can now put it all together and define the graph!
# 



from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.managed.few_shot import FewShotExamples


class BaseState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    examples: Annotated[list, FewShotExamples]


def _render_message(m):
    if isinstance(m, HumanMessage):
        return "Human: " + m.content
    elif isinstance(m, AIMessage):
        _m = "AI: " + m.content
        if len(m.tool_calls) > 0:
            _m += f" Tools: {m.tool_calls}"
        return _m
    elif isinstance(m, ToolMessage):
        return "Tool Result: ..."
    else:
        raise ValueError


def _render_messages(ms):
    m_string = [_render_message(m) for m in ms]
    return "\n".join(m_string)


# Define a new graph
workflow = StateGraph(BaseState)


def _agent(state: BaseState):
    if len(state["examples"]) > 0:
        _examples = "\n\n".join(
            [
                f"Example {i}: " + _render_messages(e["messages"])
                for i, e in enumerate(state["examples"])
            ]
        )
        system_message = """You are a helpful assistant. Below are some examples of interactions you had with users. \
These were good interactions where the final result they got was the desired one. As much as possible, you should learn from these interactions and mimic them in the future. \
Pay particularly close attention to when tools are called, and what the inputs are.!

{examples}

Assist the user as they require!""".format(
            examples=_examples
        )

    else:
        system_message = """You are a helpful assistant"""
    output = model.invoke([SystemMessage(content=system_message)] + state["messages"])
    return {"messages": [output]}


# Define the two nodes we will cycle between
workflow.add_node("agent", _agent)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")


# **Persistence**
# 
# To add in persistence, we pass in a checkpoint when compiling the graph
# 



from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")




# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])


# ## Preview the graph
# 



from IPython.display import Image

Image(app.get_graph().draw_png())


# ## Interacting with the Agent
# 
# We can now interact with the agent. Between interactions you can get and update state.
# 



thread = {"configurable": {"thread_id": "1"}}
for event in app.stream(
    {"messages": [HumanMessage(content="what's the weather in sf?")]}, thread
):
    for v in event.values():
        print(v)




current_values = app.get_state(thread)
current_values.values




current_values.values["messages"][-1].tool_calls[0]["args"][
    "query"
] = "weather in San Francisco, Accuweather"




app.update_state(thread, current_values.values)




app.get_state(thread)




for event in app.stream(None, thread):
    for v in event.values():
        print(v)




chkpnt_tuple = memory.get_tuple({"configurable": {"thread_id": "1"}})
config = chkpnt_tuple.config
checkpoint = chkpnt_tuple.checkpoint
metadata = chkpnt_tuple.metadata

# mark as "good"
metadata["score"] = 1
memory.put(config, checkpoint, metadata)




examples = list(memory.search({"score": 1}))




examples




thread = {"configurable": {"thread_id": "7"}}
for event in app.stream(
    {"messages": [HumanMessage(content="what's the weather in la?")]}, thread
):
    for v in event.values():
        print(v)




for event in app.stream(None, thread):
    for v in event.values():
        print(v)






