import pytest
from .falkordb_chat_history import FalkorDBChatHistory


@pytest.fixture(scope="module")
def chat_history():
    ch = FalkorDBChatHistory(graph_name="test-chat-history")

    yield ch

    ch.g.delete()


def test_store_chat(chat_history):
    chat_id = 1
    chat = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    chat_history.store_chat(chat_id, chat)
    stored = chat_history.get_chat(chat_id)
    assert stored == chat


def test_append_message(chat_history):
    chat_id = 2
    initial = [{"role": "user", "content": "Start"}]
    new_message = {"role": "assistant", "content": "Go on"}

    chat_history.store_chat(chat_id, initial)
    chat_history.append_message(chat_id, new_message)

    updated = chat_history.get_chat(chat_id)
    assert updated is not None
    assert len(updated) == 2
    assert updated[-1] == new_message


def test_get_chat_not_found(chat_history):
    assert chat_history.get_chat(9999) is None


def test_list_chats(chat_history):
    chat_id = 3
    chat = [{"role": "user", "content": "Ping"}]
    chat_history.store_chat(chat_id, chat)

    all_chats = chat_history.list_chats()
    assert any(c["id"] == chat_id for c in all_chats)
    assert all(isinstance(c["messages"], list) for c in all_chats)


def test_delete_chat(chat_history):
    chat_id = 4
    chat = [{"role": "user", "content": "Bye"}]
    chat_history.store_chat(chat_id, chat)
    chat_history.delete_chat(chat_id)
    assert chat_history.get_chat(chat_id) is None
