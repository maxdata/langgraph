#!/usr/bin/env python
# coding: utf-8

# # How to add summary of the conversation history
# 
# One of the most common use cases for persistence is to use it to keep track of conversation history. This is great - it makes it easy to continue conversations. As conversations get longer and longer, however, this conversation history can build up and take up more and more of the context window. This can often be undesirable as it leads to more expensive and longer calls to the LLM, and potentially ones that error. One way to work around that is to create a summary of the conversation to date, and use that with the past N messages. This guide will go through an example of how to do that.
# 
# This will involve a few steps:
# - Check if the conversation is too long (can be done by checking number of messages or length of messages)
# - If yes, the create summary (will need a prompt for this)
# - Then remove all except the last N messages
# 
# A big part of this is deleting old messages. For an in depth guide on how to do that, see [this guide](./delete-messages.ipynb)

# ## Setup
# 
# First, let's set up the packages we're going to want to use



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


# ## Build the chatbot
# 
# Let's now build the chatbot.



from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END

memory = SqliteSaver.from_conn_string(":memory:")

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str

# We will use this model for both the conversation and the summarization
model = ChatAnthropic(model_name="claude-3-haiku-20240307")

# Define the logic to call the model
def call_model(state: State):
    # If a summary exists, we add this in as a system message
    summary = state.get('summary', '')
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state['messages']
    else:
        messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END


def summarize_conversation(state: State):
    # First, we summarize the conversation
    summary = state.get('summary', '')
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
        
    messages = state['messages'] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {
        "summary": response.content,
        "messages": delete_messages
    }
    


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", END)

# Finally, we compile it!
app = workflow.compile(checkpointer=memory)


# ## Using the graph



def print_update(update):
    for k, v in update.items():
        for m in v['messages']:
            m.pretty_print()
        if 'summary' in v:
            print(v['summary'])




from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="hi! I'm bob")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the celtics!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)


# We can see that so far no summarization has happened - this is because there are only six messages in the list.



values = app.get_state(config).values
values


# Now let's send another message in



input_message = HumanMessage(content="i like how much they win")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)


# If we check the state now, we can see that we have a summary of the conversation, as well as the last two messages



values = app.get_state(config).values
values


# We can now resume having a conversation! Note that even though we only have the last two messages, we can still ask it questions about things mentioned earlier in the conversation (because we summarized those)



input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)




input_message = HumanMessage(content="what NFL team do you think I like?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)




input_message = HumanMessage(content="i like the patriots!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)






