#!/usr/bin/env python
# coding: utf-8

# # How to visualize your graph
# 
# This notebook walks through how to visualize the graphs you create. This works with ANY [Graph](https://langchain-ai.github.io/langgraph/reference/graphs/).



get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install -U langgraph\n')


# ## Set up Graph
# 
# You can visualize any arbitrary Graph, including StateGraph's and MessageGraph's. Let's have some fun by drawing fractals :).



import random
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


class MyNode:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state: State):
        return {"messages": [("assistant", f"Called node {self.name}")]}


def route(state) -> Literal["entry_node", "__end__"]:
    if len(state["messages"]) > 10:
        return "__end__"
    return "entry_node"


def add_fractal_nodes(builder, current_node, level, max_level):
    if level > max_level:
        return

    # Number of nodes to create at this level
    num_nodes = random.randint(1, 3)  # Adjust randomness as needed
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]
        node_name = f"node_{current_node}_{nm}"
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)

        # Recursively add more nodes
        r = random.random()
        if r > 0.2 and level + 1 < max_level:
            add_fractal_nodes(builder, node_name, level + 1, max_level)
        elif r > 0.05:
            builder.add_conditional_edges(node_name, route, node_name)
        else:
            # End
            builder.add_edge(node_name, "__end__")


def build_fractal_graph(max_level: int):
    builder = StateGraph(State)
    entry_point = "entry_node"
    builder.add_node(entry_point, MyNode(entry_point))
    builder.add_edge(START, entry_point)

    add_fractal_nodes(builder, entry_point, 1, max_level)

    # Optional: set a finish point if required
    builder.add_edge(entry_point, END)  # or any specific node

    return builder.compile()


app = build_fractal_graph(3)


# ## Ascii
# 
# We can easily visualize this graph in ascii



app.get_graph().print_ascii()


# ## Mermaid
# 
# We can also convert a graph class into Mermaid syntax.



print(app.get_graph().draw_mermaid())


# ## PNG
# 
# If preferred, we could render the Graph into a  `.png`. Here we could use three options:
# 
# - Using Mermaid.ink API (does not require additional packages)
# - Using Mermaid + Pyppeteer (requires `pip install pyppeteer`)
# - Using graphviz (which requires `pip install graphviz`)
# 
# 
# ### Using Mermaid.Ink
# 
# By default, `draw_mermaid_png()` uses Mermaid.Ink's API to generate the diagram.



from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeColors

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


# ### Using Mermaid + Pyppeteer



get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install --quiet pyppeteer\n%pip install --quiet nest_asyncio\n')




import nest_asyncio

nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

display(
    Image(
        app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeColors(start="#ffdfba", end="#baffc9", other="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
    )
)


# ### Using Graphviz



get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install pygraphviz\n')




display(Image(app.get_graph().draw_png()))

