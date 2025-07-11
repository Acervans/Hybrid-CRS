"""Chat history storage using FalkorDB as graph database"""

import json
from falkordb import FalkorDB


TIMEOUT = 5 * 1000 * 60  # 5 minutes


class FalkorDBChatHistory:

    def __init__(
        self,
        graph_name: str = "chat-history",
        db: FalkorDB | None = None,
        **falkordb_kwargs,
    ):
        """
        Chat history storage using FalkorDB as graph database.
        Creates a graph if it doesn't exist.

        Args:
            graph_name (str): Name of the graph to store chat history
            db (FalkorDB | None): Optional existing FalkorDB connection
            **falkordb_kwargs: Additional keyword arguments for FalkorDB connection
        """
        self.db = db or FalkorDB(host="localhost", port=6379, **falkordb_kwargs)
        self.g = self.db.select_graph(graph_name)

        try:
            self.g.query("CREATE INDEX ON :Chat(id)", timeout=TIMEOUT)
        except Exception:
            pass

    def store_chat(self, chat_id: int, messages: list[dict]) -> None:
        """
        Creates a new chat node with the full message history (JSON-encoded).

        Args:
            chat_id (int): Unique identifier for the chat
            messages (list[dict]): List of messages in the chat
        """
        content_json = json.dumps(messages, indent=None, separators=(",", ":"))
        query = "CREATE (c:Chat {id: $chat_id, content: $content})"
        self.g.query(
            query, params={"chat_id": chat_id, "content": content_json}, timeout=TIMEOUT
        )

    def get_chat(self, chat_id: int) -> list[dict] | None:
        """
        Retrieves the full message history for a chat ID.

        Args:
            chat_id (int): Unique identifier for the chat
        Returns:
            list[dict]: List of messages in the chat, or None if not found
        """
        query = """
        MATCH (c:Chat {id: $chat_id})
        RETURN c.content
        """
        result = self.g.ro_query(query, params={"chat_id": chat_id}, timeout=TIMEOUT)

        if result.result_set:
            raw_json = result.result_set[0][0]
            return json.loads(raw_json)
        return None

    def append_message(self, chat_id: int, new_message: dict) -> None:
        """
        Appends a new message to an existing chat.

        Args:
            chat_id (int): Unique identifier for the chat
            new_message (dict): The new message to append
        """
        query = """
        MATCH (c:Chat {id: $chat_id})
        RETURN c
        """
        result = self.g.query(query, params={"chat_id": chat_id}, timeout=TIMEOUT)

        if not result.result_set:
            print(f"Chat '{chat_id}' not found.")
            return

        current_content = result.result_set[0][0].properties.get("content", "[]")

        # Remove "]", append, restore "]"
        comma = "," if len(current_content) > 2 else ""
        updated_content = (
            current_content[:-1]
            + comma
            + json.dumps(new_message, indent=None, separators=(",", ":"))
            + "]"
        )

        update_query = """
        MATCH (c:Chat {id: $chat_id})
        SET c.content = $content
        """
        self.g.query(
            update_query,
            params={"chat_id": chat_id, "content": updated_content},
            timeout=TIMEOUT,
        )

    def list_chats(self) -> list[dict]:
        """
        Returns a list of all chat IDs and their content.

        Returns:
            list[dict]: List of dictionaries with chat IDs and their messages
        """
        query = "MATCH (c:Chat) RETURN c.id, c.content"
        result = self.g.ro_query(query, timeout=TIMEOUT)

        chats = []
        for row in result.result_set:
            chat_id = row[0]
            content = json.loads(row[1])
            chats.append({"id": chat_id, "messages": content})

        return chats

    def delete_chat(self, chat_id: int) -> None:
        """
        Deletes a chat node.

        Args:
            chat_id (int): Unique identifier for the chat to delete
        """
        query = """
        MATCH (c:Chat {id: $chat_id})
        DETACH DELETE c
        """
        self.g.query(query, params={"chat_id": chat_id}, timeout=TIMEOUT)
