from langgraph.graph import StateGraph, START, END, MessagesState
from typing import TypedDict, Optional

from db_manager import food_search


def search_food_graph_builder(gemini_chat, memory, logger):
    class MyState_Food_Search(TypedDict):
        messages: str
        food_id: Optional[str]
        food_name: Optional[str]
        food_category: Optional[str]
        restaurant_name: Optional[str]
        price: Optional[str]
        edit_distance: Optional[str]

    def node_search_food(state: MyState_Food_Search) -> MyState_Food_Search:
        """
        Gives the details of a food by its name or restaurant name.
        """
        logger.info("---Search Food Node---")

        user_query = state["messages"]
        logger.debug(f"User query: {user_query}")

        # Step 1: Ask LLM to extract food_name and restaurant_name
        details = f"Please extract the food_name and the restaurant_name from this query in this format: food_name,restaurant_name. Please return None for each of them if it was not found.:\n\n{user_query}"
        logger.debug(f"Sending request to LLM to extract details: {details}")

        invoke = gemini_chat.invoke(details).content.strip()
        logger.info(f"LLM response for details extraction: {invoke}")

        try:
            food_name, restaurant_name = invoke.split(",")
            logger.debug(f"Extracted food_name: {food_name}, restaurant_name: {restaurant_name}")
        except ValueError as e:
            logger.error(f"Error splitting LLM response: {invoke}. Exception: {e}")
            food_name, restaurant_name = "None", "None"

        # Step 2: Perform search using extracted food_name and restaurant_name
        matches = food_search(food_name, restaurant_name)
        logger.info(f"Search results for food_name '{food_name}' and restaurant_name '{restaurant_name}': {matches}")

        # Step 3: Generate response using search matches
        generated_text = (f"Please consider this query: {user_query}\n\n"
                          f"Generate a text considering the query and following matches:\n{matches}\n\n"
                          "Generate sentences and consider multiple details including the price if there were multiple matches.")
        logger.debug(f"Sending request to LLM to generate text: {generated_text}")

        response = gemini_chat.invoke(generated_text).content
        logger.info(f"Generated response from LLM: {response}")

        # Store or process response (currently just returning state as per the original code)
        return state

    # Build the graph
    builder = StateGraph(MyState_Food_Search)

    # Add nodes
    builder.add_node("node_search_food", node_search_food)

    # Add edges (linear flow)
    builder.add_edge(START, "node_search_food")
    builder.add_edge("node_search_food", END)

    # Compile the graph
    advanced_search_graph = builder.compile(checkpointer=memory)

    return advanced_search_graph
