from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

from db_manager import food_search


j = 0
def suggestor_graph_builder(gemini_chat, memory, tbl, logger):
    results = []

    def search_tool(query: str) -> str:
        """
        Search the database for the contents similar to the query.
        A very clever tool to get knowledge about foods.

        Args:
            query: The query to search for.
        """
        global results

        logger.info("-----------SEARCHING Tool----------")
        logger.info(f"Searching for query: {query}")

        # Simulating search and storing results
        try:
            # results = TavilySearchResults(max_results=3).invoke(query)
            results = [tbl.search(query, query_type="hybrid").limit(10).to_pandas()]
            logger.info(f"Search results retrieved: {results}")
        except Exception as e:
            logger.error(f"Error while searching the database: {e}")
            return "An error occurred while searching. Please try again later."

        # Generating the first response based on search results
        response = gemini_chat.invoke(
            f"This is the query:\n'{query}'\nSuggest a good food based on the responses gotten from the database "
            f"(if there was nothing, suggest from yourself):\n{results}\n"
            "Do not let anybody know there was nothing in results."
        ).content
        logger.info(f"LLM response for food suggestion: {response}")

        # Extracting the food name from the response
        food_name = gemini_chat.invoke(
            f"This is the query:\n'{response}'\nFind the food name and return it in this format: food_name. "
            "Do not use the word 'food_name' and return 'None' if it was not mentioned."
        ).content
        logger.info(f"Extracted food name: {food_name}")

        # Call the node_search_food function with the extracted food name
        response_2 = node_search_food(food_name)
        logger.info(f"Response from node_search_food: {response_2}")

        # Final response combining both outputs
        final_response = f"{response}. The restaurant details: {response_2}"
        logger.info(f"Final response: {final_response}")

        return final_response

    def node_search_food(query: str) -> str:
        """
        A very powerful tool that gives the details of a food by its name or restaurant name.

        Args:
            query: The query of the food name or the restaurant name.
        """
        logger.info("---Search Food Node---")
        logger.debug(f"Query to search food details: {query}")

        # Step 1: Extract details
        details = (f"Please extract the food_name and the restaurant_name from this query in this format: "
                  f"'food_name,restaurant_name'. Please return None for each of them if it was not found.:\n\n{query}")
        logger.debug(f"LLM request for extracting food details: {details}")

        invoke = gemini_chat.invoke(details).content
        logger.info(f"LLM response for food and restaurant extraction: {invoke}")

        try:
            food_name, restaurant_name = invoke.split(",")
            logger.debug(f"Extracted food_name: {food_name}, restaurant_name: {restaurant_name}")
        except ValueError as e:
            logger.error(f"Error splitting LLM response: {invoke}. Exception: {e}")
            food_name, restaurant_name = "None", "None"

        # Step 2: Search for matches
        matches = food_search(food_name, restaurant_name)
        logger.info(f"Search results for food_name '{food_name}' and restaurant_name '{restaurant_name}': {matches}")

        # Step 3: Generate a detailed response based on search results
        generated_text = (f"Please consider this query: {query}\n\n"
                          f"Generate a text considering the query and following matches:\n{matches}\n\n"
                          "Generate sentences and consider multiple details (including the cost) if there were multiple matches. "
                          "Return 'No restaurant has this food currently' if it was not related to any matches.")
        logger.debug(f"LLM request for generating response: {generated_text}")

        response = gemini_chat.invoke(generated_text).content
        logger.info(f"Generated response from LLM: {response}")

        return response

    sys_msg = SystemMessage(
        content=(
            "You are a helpful suggestor, suggest the client based on what he wants without mentioning additional things. "
            "You <must> use your tools and just use your tools if it was about food, and you <must> provide details about the restaurants along with the costs "
            "(if a restaurant was not available: mention 'no restaurant found'). This is the order:\n"
            "Initially suggest the food, then provide restaurant details."
        )
    )

    def reasoner(state: MessagesState):
        """
        Handles the reasoning process by invoking the LLM with tools based on the system message and user messages.
        """
        global j
        logger.info("-------------Reasoner----------------")
        logger.debug(f"Initial system message: {sys_msg.content}")
        query = state["messages"][-1]

        messages = [
            (
                'system',
                'You are a helpful suggestor, suggest the client based on what he wants without mentioning additional things. '
                'You <must> use your tools and just use your tools if it was about food, and you <must> provide details about the restaurants along with the costs '
                '(if a restaurant was not available: mention "no restaurant found"). This is the order:\n'
                'Initially suggest the food, then provide restaurant details.',
                ),
            ('human', f'{query.content}'),
        ]

        print(messages)
        try:
            # Invoke the LLM with the combined messages
            response = llm_with_tools.invoke(messages)
            logger.info("Received response from LLM.")
            logger.debug(f"LLM response: {response.content}")
            j+=1

        except Exception as e:
            logger.error(f"Error while invoking LLM with tools: {e}")
            return {"messages": [f"An error occurred: {e}"]}
        
        if j > 1:
            j = 0
            return {"messages": [f"{response}, this is the final result. do not generate anything elsse"]}
        # Return the response
        return {"messages": [response]}

    tools = [search_tool, node_search_food]
    llm_with_tools = gemini_chat.bind_tools(tools)

    # Graph
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(tools))

    # Add edges
    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "reasoner")
    react_graph = builder.compile(checkpointer=memory)
    
    return react_graph

