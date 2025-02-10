from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState

from db_manager import cancel_order, comment_order, check_order_status


messages = []
node_states = []

def orders_graph_builder(gemini_chat, memory, logger):
    class MyState(TypedDict):
        messages: str
        costumer_order_id: Optional[str]
        phone_number: Optional[str]
        status: Optional[str]
        person_name: Optional[str]
        comment: Optional[str]
        related_node: Optional[str]


    class IsRelated(BaseModel):
        is_related_flag: str = Field(
            description="The decision that the query is related to which one of these nodes:\n\n"
                        "1-'node_cancel_order': Cancels an order if the client requests.\n"
                        "2-'node_comment_registeration': Registers the client's idea about an order.\n"
                        "3-'node_order_status': If the client wanted to be aware of the order status.\n"
                        "4-'node_other': If the client wanted something else.\n"
                        "Do not put ' or \"."
        )


    def is_related(state: MyState) -> Literal["node_cancel_order", "node_comment_registeration", "node_order_status", "node_other"]:
        query = state["messages"]
        print(query)

        logger.info("Received query for determining node relation.")
        logger.debug(f"User query: {query}")

        # # Initialize the LLM with structured output
        # llm_with_structured_output = gemini_chat.with_structured_output(IsRelated)

        messages = [
            (
                "system",
                "The decision that the query is related to which one of these nodes:\n\n"
                        "1-'node_cancel_order': Cancels an order if the client requests.\n"
                        "2-'node_comment_registeration': Registers the client's idea about an order.\n"
                        "3-'node_order_status': If the client wanted to be aware of the order status.\n"
                        "4-'node_other': If the client wanted something else.\n"
                        "Do not put ' or \".",
            ),
            ("human", query.content),
        ]

        # Invoke the LLM to get a response
        response = gemini_chat.invoke(messages)
        logger.info("Received response from LLM.")
        logger.debug(f"LLM response: {response}")

        # Log the target node relation decision
        logger.info("-----Target Relation-----")
        logger.info(f"Response flag: {response.content}")

        # Determine and return the appropriate node
        if response.content in ["1", "node_cancel_order", "1-node_cancel_order"]:
            logger.info("Selected node: node_cancel_order")
            return "node_cancel_order"

        elif response.content in ["2", "node_comment_registeration", "2-node_comment_registeration"]:
            logger.info("Selected node: node_comment_registeration")
            return "node_comment_registeration"

        elif response.content in ["3", "node_order_status", "3-node_order_status"]:
            logger.info("Selected node: node_order_status")
            return "node_order_status"

        elif response.content in ["4", "node_other", "4-node_other"]:
            logger.info("Selected node: node_other")
            return "node_other"

        # Log if an unexpected flag is received
        logger.warning(f"Unexpected response flag: {response.content}. Returning 'node_other' by default.")
        return "node_other"


    def node_initial(state: MyState) -> MyState:
        """
        Gathers the Order ID for the next nodes.
        """
        logger.info("---Initial Node---")
        user_query = state["messages"]

        logger.info(f"User query: {user_query}")

        details = f"Please extract the details from this text in this format: 'order_id,phone_number,person_name'. Please return None for each of them if it was not filled. Your output should be just the values, not order_id, phone_number and person_name words:\n\n{user_query}"

        invoke = gemini_chat.invoke(details).content
        logger.info(f"Invoke response: {invoke}")
        order_id, phone_number, person_name = invoke.split(",")

        state["costumer_order_id"] = order_id
        state["phone_number"] = phone_number
        state["person_name"] = person_name

        return state


    def node_cancel_order(state: MyState) -> MyState:
        """
        Cancels an order if the client requests.
        """
        logger.info("---Node Cancel Order---")
        global node_states, messages
        node_states.append("node_cancel_order")

        if len(node_states)>1 and len(messages)>1 and node_states[-1] != node_states[-2]:
            logger.info("New Node")
            messages = [messages[-1]]
            node_states = [node_states[-1]]

        if state["costumer_order_id"] == 'None' and state["phone_number"] == 'None':
            logger.warning("Order canceled failed; Please provide your order ID and Phone Number.")
            state["status"] = "not exist"

        elif state["costumer_order_id"] == 'None':
            logger.warning("Order canceled failed; Please provide your order ID too.")
            state["status"] = "not exist"

        elif state["phone_number"] == 'None':
            logger.warning("Order canceled failed; Please provide Phone Number too.")
            state["status"] = "not exist"

        elif "delivered" in cancel_order(state["costumer_order_id"], state["phone_number"]):
            logger.warning("Order canceled failed; Order is already delivered")
            state["status"] = "delivered"

        elif "not exist." in cancel_order(state["costumer_order_id"], state["phone_number"]):
            logger.warning("Order canceled failed; Order not exists.")
            state["status"] = "not exist"

        elif "canceled" in cancel_order(state["costumer_order_id"], state["phone_number"]):
            logger.warning("Order is already canceled.")
            state["status"] = "canceled"

        else:
            logger.info("Order canceled successfully.")
            state["status"] = "canceled"

        return state

    def node_comment_registeration(state: MyState) -> MyState:
        """
        Registers the client's idea about an order.
        """
        logger.info("---Node Comment Registeration---")
        global node_states, messages
        user_query = state["messages"]
        node_states.append("node_comment_registeration")

        if len(node_states)>1 and len(messages)>1 and node_states[-1] != node_states[-2]:
            logger.info("New Node")
            messages = [messages[-1]]
            node_states = [node_states[-1]]

        comment = f"Please extract the comment from this text in this format: 'comment'. Please return None if it was not filled:\n\n{user_query}"

        invoke = gemini_chat.invoke(comment).content
        if invoke != "None" and state["costumer_order_id"] != "None":
            comment = invoke
            logger.info(f"Comment: {comment}")
            logger.info(comment_order(state["costumer_order_id"], state["person_name"], comment))

        elif state["person_name"] == "None":
            logger.warning("Comment Registration failed; Please provide your name.")
            comment = "None"

        elif state["costumer_order_id"] == "None":
            logger.warning("Comment Registration failed; Please provide your order ID.")
            comment = "None"

        else:
            logger.info(comment_order(state["costumer_order_id"], state["person_name"], comment))
            state["comment"] = comment

        return state

    def node_order_status(state: MyState) -> MyState:
        """
        If the client wanted to be aware of the order status.
        """
        logger.info("---Node Track Order Status---")
        global node_states, messages

        node_states.append("node_order_status")
        if len(node_states)>1 and len(messages)>1 and node_states[-1] != node_states[-2]:
            logger.info("New Node")
            messages = [messages[-1]]
            node_states = [node_states[-1]]

        if state["costumer_order_id"] != "None":
            logger.info(f"Order Status: {check_order_status(state['costumer_order_id'])}")

        else:
            logger.warning("Order status failed; Please provide your order ID.")

        return state

    def node_other(state: MyState) -> MyState:
        """
        If the client wanted something else.
        """
        logger.info("---Node Other Requests---")
        global node_states, messages
        node_states.append("node_other")
        if len(node_states)>1 and len(messages)>1 and node_states[-1] != node_states[-2]:
            logger.info("New Node")
            messages = [messages[-1]]
            node_states = [node_states[-1]]
        logger.info("Please keep your request in this 3 field:\nCancel The Order\nComment Registeration\nTrack Order Status")

        return state


    # Build the graph
    builder = StateGraph(MyState)

    # Add nodes
    builder.add_node("node_initial", node_initial)
    builder.add_node("node_cancel_order", node_cancel_order)
    builder.add_node("node_comment_registeration", node_comment_registeration)
    builder.add_node("node_order_status", node_order_status)
    builder.add_node("node_other", node_other)

    # Add edges (linear flow)
    builder.add_edge(START, "node_initial")

    builder.add_conditional_edges("node_initial", is_related)
    builder.add_edge("node_cancel_order", END)
    builder.add_edge("node_comment_registeration", END)
    builder.add_edge("node_order_status", END)
    builder.add_edge("node_other", END)

    # Compile the graph
    advanced_order_graph = builder.compile()

    return advanced_order_graph


