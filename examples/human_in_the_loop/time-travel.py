#!/usr/bin/env python
# coding: utf-8

# # How to view and update past graph state
# 
# Once you start [checkpointing](../persistence.ipynb) your graphs, you can easily **get** or **update** the state of the agent at any point in time. This permits a few things:
# 
# 1. You can surface a state during an interrupt to a user to let them accept an action.
# 2. You can **rewind** the graph to reproduce or avoid issues.
# 3. You can **modify** the state to embed your agent into a larger system, or to let the user better control its actions.
# 
# The key methods used for this functionality are:
# 
# - [get_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.get_state): fetch the values from the target config
# - [update_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.update_state): apply the given values to the target state
# 
# **Note:** this requires passing in a checkpointer.
# 
# Below is a quick example.

# ## Setup
# 
# First we need to install the packages required



get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install --quiet -U langgraph langchain_anthropic\n')


# Next, we need to set API keys for Anthropic (the LLM we will use)



import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")


# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.



os.environ["LANGCHAIN_TRACING_V2"] = "true"
_set_env("LANGCHAIN_API_KEY")


# ## Build the agent
# 
# We can now build the agent. We will build a relatively simple ReAct-style agent that does tool calling. We will use Anthropic's models and a fake tool (just for demo purposes).



# Set up the tool
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]


tools = [search]
tool_node = ToolNode(tools)

# Set up the model

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
model = model.bind_tools(tools)


# Define nodes and conditional edges


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
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

# Set up memory
memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable

# We add in `interrupt_before=["action"]`
# This will add a breakpoint before the `action` node is called
app = workflow.compile(checkpointer=memory)


# ## Interacting with the Agent
# 
# We can now interact with the agent. Let's ask it for the weather in SF.
# 



from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}
input_message = HumanMessage(content="Use the search tool to look up the weather in SF")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


# ## Checking history
# 
# Let's browse the history of this thread, from start to finish.



all_states = []
for state in app.get_state_history(config):
    print(state)
    all_states.append(state)
    print("--")


# ## Replay a state
# 
# We can go back to any of these states and restart the agent from there! Let's go back to right before the tool call gets executed.



to_replay = all_states[2]




to_replay.values




to_replay.next


# To replay from this place we just need to pass its config back to the agent. Notice that it just resumes from right where it left all - making a tool call.



for event in app.stream(None, to_replay.config):
    for v in event.values():
        print(v)


# ## Branch off a past state
# 
# Using LangGraph's checkpointing, you can do more than just replay past states. You can branch off previous locations to let the agent explore alternate trajectories or to let a user "version control" changes in a workflow.
# 
# Let's show how to do this to edit the state at a particular point in time. Let's update the state to change the input to the tool



# Let's now get the last message in the state
# This is the one with the tool calls that we want to update
last_message = to_replay.values["messages"][-1]

# Let's now update the args for that tool call
last_message.tool_calls[0]["args"] = {"query": "current weather in SF"}

branch_config = app.update_state(
    to_replay.config,
    {"messages": [last_message]},
)


# We can then invoke with this new `branch_config` to resume running from here with changed state. We can see from the log that the tool was called with different input.



for event in app.stream(None, branch_config):
    for v in event.values():
        print(v)


# Alternatively, we could update the state to not even call a tool!



from langchain_core.messages import AIMessage

# Let's now get the last message in the state
# This is the one with the tool calls that we want to update
last_message = to_replay.values["messages"][-1]

# Let's now get the ID for the last message, and create a new message with that ID.
new_message = AIMessage(content="its warm!", id=last_message.id)

branch_config = app.update_state(
    to_replay.config,
    {"messages": [new_message]},
)




branch_state = app.get_state(branch_config)




branch_state.values




branch_state.next


# You can see the snapshot was updated and now correctly reflects that there is no next step.





