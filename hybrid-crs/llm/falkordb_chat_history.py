import json
from falkordb import FalkorDB

# Connect to local FalkorDB
db = FalkorDB(host='localhost', port=6379)
graph = db.select_graph('chat_history')

def store_conversation(conv_id: str, messages: list[dict]):
    """
    Creates a new conversation node with the full message history (JSON-encoded).
    """
    content_json = json.dumps(messages)
    query = f"""CREATE (c:Conversation {{id: '{conv_id}', content: '{content_json}'}})"""
    print(query)
    return graph.query(query)

def get_conversation(conv_id: str):
    """
    Retrieves the full message history for a conversation ID.
    """
    query = f"""
    MATCH (c:Conversation {{id: '{conv_id}'}})
    RETURN c.content
    """
    result = graph.query(query)

    if result.result_set:
        raw_json = result.result_set[0][0]
        return json.loads(raw_json)
    return None

def append_message(conv_id: str, new_message: dict):
    """
    Appends a new message to an existing conversation.
    """
    query = f"""
    MATCH (c:Conversation {{id: '{conv_id}'}})
    RETURN c
    """
    result = graph.query(query)

    if not result.result_set:
        print(f"Conversation '{conv_id}' not found.")
        return

    conv_node = result.result_set[0][0]
    current_content = json.loads(conv_node.properties['content'])

    current_content.append(new_message)
    updated_json = json.dumps(current_content)

    update_query = f"""
    MATCH (c:Conversation {{id: '{conv_id}'}})
    SET c.content = '{updated_json.replace("'", "\\'")}'
    """
    graph.query(update_query)

def list_conversations():
    """
    Returns a list of all conversation IDs and their content.
    """
    query = "MATCH (c:Conversation) RETURN c.id, c.content"
    result = graph.query(query)

    conversations = []
    for row in result.result_set:
        conv_id = row[0]
        content = json.loads(row[1])
        conversations.append({'id': conv_id, 'messages': content})

    return conversations

def delete_conversation(conv_id: str):
    """
    Deletes a conversation node.
    """
    query = f"""
    MATCH (c:Conversation {{id: '{conv_id}'}})
    DETACH DELETE c
    """
    graph.query(query)


if __name__ == "__main__":
    # Initial chat
    chat = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help?"}
    ]

    conversation_id = "conv-001"

    # Store
    res = store_conversation(conversation_id, chat)
    print(res)

    # Append message
    append_message(conversation_id, {"role": "user", "content": "What's the time?"})

    # Get conversation
    full_chat = get_conversation(conversation_id)
    print(f"Conversation '{conversation_id}':\n", full_chat)

    # List all conversations
    all_convs = list_conversations()
    print(f"\nAll conversations:")
    for conv in all_convs:
        print(f"- ID: {conv['id']}, Messages: {len(conv['messages'])}")

    # Delete
    delete_conversation(conversation_id)
