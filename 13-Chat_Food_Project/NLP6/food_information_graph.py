from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

i = 0
def food_information_graph_builder(gemini_chat, memory, tbl, logger):
    results = []

    def db_search_tool(query: str) -> str:
        """A clever tool to search in the DataBase to find professional details about a food.
        Args:
            query: The query to search for.
        """
        global results
        print("-----------Data Base SEARCHING Tool For Information----------")
        print("searching for", query)

        context_list = tbl.search(query, query_type="hybrid").limit(5).to_list()
        context = ''.join([f"{c['text']}\n\n" for c in context_list])

        # print("results", results)
        response = gemini_chat.invoke(f"This is the query:\n'{query}'\nAnswer base on the following context:\n{context}.")

        print("response", response)

        return response

    def web_search_tool(query: str) -> str:
        """A very clever tool to search in the web to find professional details about a food.
        Args:
            query: The query to search for.
        """
        global results
        global client
        print("-----------Web SEARCHING Tool For Information----------")
        print("searching for", query)
        # results = TavilySearchResults(max_results=3).invoke(query)
        results = client.search(
        query=query
    )

        # print("results", results)
        response = gemini_chat.invoke(f"This is the query:\n'{query}'\nAnswer base on the following results:\n{results}.")

        print("response", response)

        return response

    def food_info_reasoner(state: MessagesState):
        """
        Handles the reasoning process by invoking the LLM with tools based on the system message and user messages.
        """
        global i
        query = state["messages"][-1]
        
        logger.info("-------------Food Information Reasoner----------------")
        logger.debug(f"Initial system message: {sys_msg.content}")

        # Combine system message with user messages
        # messages = [sys_msg] + state["messages"]

        messages = [
            (
                "system",
                "You are a wise assistant to answer the general and special informations about foods, using your tools "
                "Initially you try to find the answers base on the database using db_search_tool.\n"
                "If the content was about the foods information and you did not find the results, you <must> use your web_search_tool to search the web without asking anything from the client.",
                ),
            ("human", query.content),
        ]


        # logger.debug(f"Combined messages for LLM invocation: {[msg.content for msg in messages]}")

        # try:
        # Invoke the LLM with the combined messages
        response = food_info_llm_with_tools.invoke(messages)
        i+=1

        logger.info("Received response from LLM.")
        logger.debug(f"LLM response: {response.content}")

        # except Exception as e:
        #     logger.error(f"Error while invoking LLM with tools: {e}")
        #     return {"messages": [f"An error occurred: {e}"]}

        if i>1:
            i = 0
            return {"messages": [f"{response}, this is the final result. do not generate anything elsse"]}
        # Return the response
        return {"messages": [response]}

    sys_msg = SystemMessage(
        content=(
            "You are a wise assistant to answer the general and special informations about foods. "
            "Initially you try to find the answers base on the database using db_search_tool.\n"
            "If the content was about the foods information and you did not find the results, you <must> use your web_search_tool to search the web without asking anything from the client."
        )
    )

    tools = [web_search_tool, db_search_tool, web_search_tool]
    food_info_llm_with_tools = gemini_chat.bind_tools(tools)

    food_info_memory = MemorySaver()
    # Graph
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("food_info_reasoner", food_info_reasoner)
    builder.add_node("tools", ToolNode(tools))

    # Add edges
    builder.add_edge(START, "food_info_reasoner")
    builder.add_conditional_edges(
        "food_info_reasoner",
        # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "food_info_reasoner")
    food_information_graph = builder.compile(checkpointer=food_info_memory)

    return food_information_graph

    