#!/usr/bin/env python
# coding: utf-8

# # How to stream state updates of your graph

# LangGraph supports multiple streaming modes. The main ones are:
# 
# - `values`: This streaming mode streams back values of the graph. This is the **full state of the graph** after each node is called.
# - `updates`: This streaming mode streams back updates to the graph. This is the **update to the state of the graph** after each node is called.
# 
# This guide covers `stream_mode="updates"`.

# ## Setup

# We'll be using a simple ReAct agent for this guide.



get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install -U langgraph langchain-openai\n')




import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")




from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
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
graph = create_react_agent(model, tools)


# ## Stream updates



inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        print(f"Receiving update from node: '{node}'")
        print(values)
        print("\n\n")






