from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from orders_graph import orders_graph_builder
from suggestor_graph import suggestor_graph_builder
from search_food_graph import search_food_graph_builder
from food_information_graph import food_information_graph_builder
from typing import TypedDict, Optional, Literal


messages = []
node_states = []

def combined_graph_builder(gemini_chat, tbl, logger):
    messages = []
    node_states = []

    class CombinedMyState(TypedDict):
        messages: str

    def node_initial(state: CombinedMyState) -> CombinedMyState:
        """
        A placeholder function that returns the current state as it is.
        """
        logger.info("Node Initial function called.")
        return state

    def node_other(state: CombinedMyState) -> CombinedMyState:
        """
        If the clinet wanted something else.
        """
        logger.info("---Node Other Requests---")
        global node_states, messages
        node_states.append("node_other")
        if len(node_states)>1 and node_states[-1] != node_states[-2]:
            logger.info("New Node")
            messages = [messages[-1]]
            node_states = [node_states[-1]]
        logger.info("Please keep your request in this 3 field:\nCancel The Order\nComment Registeration\nTrack Order Status")

        return state


    class CombinedIsRelated(BaseModel):
        is_related_flag: str = Field(
            description="The decision that the query is related to which one of these nodes:\n\n"
                        "1-'advanced_order_graph': requests about an order and manipulation on it.\n"
                        "2-'advanced_search_graph': requests about food's details, like its price, the restaurant and similar details.\n"
                        "3-'react_graph': requests which have a suggestion about food inside them.\n"
                        "4-'food_information_graph': If the client wanted some general or professional details about food, like asking recipie.\n"
                        "5-'node_other': If the client wanted something else.\n"
                        "Do not put ' or \" in the response."
        )


    def combined_is_related(state: CombinedMyState) -> Literal["advanced_order_graph", "advanced_search_graph", "react_graph", "food_information_graph", "node_other"]:
        query = state["messages"]
        print(query)
        logger.info("Determining the target relation for the query.")
        logger.debug(f"User query: {query}")

        messages = [
            (
                "system",
                "The decision that the query is related to which one of these nodes:\n\n"
                        "1-'advanced_order_graph': requests about an order and manipulation on it.\n"
                        "2-'advanced_search_graph': requests about food's details, like its price, the restaurant and similar details.\n"
                        "3-'react_graph': requests which have a suggestion about food inside them.\n"
                        "4-'food_information_graph': If the client wanted some general or professional details about food, like asking recipie.\n"
                        "5-'node_other': If the client wanted something else.\n"
                        "Do not put ' or \" in the response, the only output should be among this, do not return long response.",
            ),
            ("human", query.content),
        ]

        print(messages)

        try:
            # Invoke the LLM with structured output
            # llm_with_structured_output = gemini_chat.with_structured_output(CombinedIsRelated)
            # print(f"\n\n{query.content}\n\n")
            response = gemini_chat.invoke(messages)
            logger.info("Received response from LLM.")
            logger.debug(f"LLM response: {response}")

            # Determine the target node based on the LLM response
            if response.content in ["1", "advanced_order_graph", "1-advanced_order_graph", "1_advanced_order_graph", "1-'advanced_order_graph'"]:
                logger.info("Target node determined: advanced_order_graph")
                print(1)            
                return "advanced_order_graph"

            elif response.content in ["2", "advanced_search_graph", "2-advanced_search_graph", "2_advanced_search_graph", "2-'advanced_search_graph'"]:
                logger.info("Target node determined: advanced_search_graph")
                print(2)            
                return "advanced_search_graph"

            elif response.content in ["3", "react_graph", "3-react_graph", "3_react_graph", "3-'react_graph'"]:
                logger.info("Target node determined: react_graph")
                print(3)            
                return "react_graph"

            elif response.content in ["4", "food_information_graph", "4-food_information_graph", "4_food_information_graph", "4-'food_information_graph'"]:
                logger.info("Target node determined: food_information_graph")
                print(4)
                return "food_information_graph"

            elif response.content in ["5", "node_other", "5-node_other", "5_node_other"]:
                logger.info("Target node determined: node_other")
                print(5)            
                return "node_other"

            else:
                logger.warning(f"Unexpected response flag: {response.content}. Defaulting to 'node_other'.")
                print(6)
                return "node_other"

        except Exception as e:
            logger.error(f"Error while determining the target relation: {e}")
            return "node_other"


    memory = MemorySaver()

    # Build the graph
    builder = StateGraph(CombinedMyState)
    advanced_order_graph = orders_graph_builder(gemini_chat, memory, logger)
    advanced_search_graph = search_food_graph_builder(gemini_chat, memory, logger)
    react_graph = suggestor_graph_builder(gemini_chat, memory, tbl, logger)
    food_information_graph = food_information_graph_builder(gemini_chat, memory, tbl, logger)

    # Add nodes
    builder.add_node("node_initial", node_initial)
    builder.add_node("advanced_order_graph", advanced_order_graph)
    builder.add_node("advanced_search_graph", advanced_search_graph)
    builder.add_node("react_graph", react_graph)
    builder.add_node("food_information_graph", food_information_graph)
    builder.add_node("node_other", node_other)

    # Add edges (linear flow)
    builder.add_edge(START, "node_initial")

    builder.add_conditional_edges("node_initial", combined_is_related)
    builder.add_edge("advanced_order_graph", END)
    builder.add_edge("advanced_search_graph", END)
    builder.add_edge("react_graph", END)
    builder.add_edge("food_information_graph", END)
    builder.add_edge("node_other", END)

    # Compile the graph
    combined_graph = builder.compile(checkpointer=memory)

    return combined_graph

