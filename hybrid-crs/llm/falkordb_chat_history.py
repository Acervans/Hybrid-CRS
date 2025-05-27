""" Chat history storage using FalkorDB as graph database 
"""

import json
from falkordb import FalkorDB


TIMEOUT = 5 * 1000 * 60  # 5 minutes


class FalkorDBChatHistory:

    def __init__(self, graph_name: str = "chat_history", **falkordb_kwargs):
        """
        Initializes the chat history storage.
        Creates a graph if it doesn't exist.

        Args:
            graph_name (str): Name of the graph to store chat history.
            **falkordb_kwargs: Additional keyword arguments for FalkorDB connection.
        """
        self.db = FalkorDB(host="localhost", port=6379, **falkordb_kwargs)
        self.g = self.db.select_graph(graph_name)

        try:
            self.g.query("CREATE INDEX ON :Chat(id)", timeout=TIMEOUT)
        except Exception:
            pass

    def store_chat(self, chat_id: str, messages: list[dict]) -> None:
        """
        Creates a new chat node with the full message history (JSON-encoded).

        Args:
            chat_id (str): Unique identifier for the chat.
            messages (list[dict]): List of messages in the chat.
        """
        content_json = json.dumps(messages)
        query = "CREATE (c:Chat {id: $chat_id, content: $content})"
        self.g.query(
            query, params={"chat_id": chat_id, "content": content_json}, timeout=TIMEOUT
        )

    def get_chat(self, chat_id: str) -> list[dict] | None:
        """
        Retrieves the full message history for a chat ID.

        Args:
            chat_id (str): Unique identifier for the chat.
        Returns:
            list[dict]: List of messages in the chat, or None if not found.
        """
        query = """
        MATCH (c:Chat {id: $chat_id})
        RETURN c.content
        """
        result = self.g.query(query, params={"chat_id": chat_id}, timeout=TIMEOUT)

        if result.result_set:
            raw_json = result.result_set[0][0]
            return json.loads(raw_json)
        return None

    def append_message(self, chat_id: str, new_message: dict) -> None:
        """
        Appends a new message to an existing chat.

        Args:
            chat_id (str): Unique identifier for the chat.
            new_message (dict): The new message to append.
        """
        query = """
        MATCH (c:Chat {id: $chat_id})
        RETURN c
        """
        result = self.g.query(query, params={"chat_id": chat_id}, timeout=TIMEOUT)

        if not result.result_set:
            print(f"Chat '{chat_id}' not found.")
            return

        conv_node = result.result_set[0][0]
        current_content = json.loads(conv_node.properties["content"])

        current_content.append(new_message)
        updated_json = json.dumps(current_content)

        update_query = """
        MATCH (c:Chat {id: $chat_id})
        SET c.content = $content
        """
        self.g.query(
            update_query,
            params={"chat_id": chat_id, "content": updated_json},
            timeout=TIMEOUT,
        )

    def list_chats(self) -> list[dict]:
        """
        Returns a list of all chat IDs and their content.

        Returns:
            list[dict]: List of dictionaries with chat IDs and their messages.
        """
        query = "MATCH (c:Chat) RETURN c.id, c.content"
        result = self.g.query(query, timeout=TIMEOUT)

        chats = []
        for row in result.result_set:
            chat_id = row[0]
            content = json.loads(row[1])
            chats.append({"id": chat_id, "messages": content})

        return chats

    def delete_chat(self, chat_id: str) -> None:
        """
        Deletes a chat node.

        Args:
            chat_id (str): Unique identifier for the chat to delete.
        """
        query = """
        MATCH (c:Chat {id: $chat_id})
        DETACH DELETE c
        """
        self.g.query(query, params={"chat_id": chat_id}, timeout=TIMEOUT)


if __name__ == "__main__":

    def test():
        # Initial chat
        chat = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]

        chat_id = "chat-001"

        ch = FalkorDBChatHistory()
        # Store
        res = ch.store_chat(chat_id, chat)
        print(res)

        # Append message
        ch.append_message(chat_id, {"role": "user", "content": "What's the time?"})

        # Get chat
        full_chat = ch.get_chat(chat_id)
        print(f"Chat '{chat_id}':\n", full_chat)

        # List all chats
        all_convs = ch.list_chats()
        print(f"\nAll chats:")
        for conv in all_convs:
            print(f"- ID: {conv['id']}, Messages: {len(conv['messages'])}")

        # Delete
        ch.delete_chat(chat_id)

    test()
