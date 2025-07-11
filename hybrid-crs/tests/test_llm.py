import pytest

from llm.falkordb_chat_history import FalkorDBChatHistory
from llm.user_profile import UserProfile, ContextPreference, ContextType

############################
# falkordb_chat_history.py #
############################


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


###################
# user_profile.py #
###################


@pytest.fixture
def sample_user_profile():
    return UserProfile(
        user_id=1,
        context_prefs={
            "genre": ContextPreference(
                type=ContextType.DICT, data={"action": True, "horror": False}
            ),
            "min_rating": ContextPreference(type=ContextType.NUM, data=7.0),
        },
        item_prefs={101: 5.0, 102: 4.5},
    )


def test_add_context_def(sample_user_profile):
    new_context = ContextPreference(type=ContextType.BOOL, data=True)
    sample_user_profile.add_context_def("new_context", new_context)
    assert "new_context" in sample_user_profile.context_prefs

    with pytest.raises(ValueError):
        sample_user_profile.add_context_def("genre", new_context)


def test_remove_context_def(sample_user_profile):
    sample_user_profile.remove_context_def("genre")
    assert "genre" not in sample_user_profile.context_prefs

    with pytest.raises(ValueError):
        sample_user_profile.remove_context_def("nonexistent")


def test_update_context_preference_dict(sample_user_profile):
    sample_user_profile.update_context_preference("genre", {"comedy": True})
    updated = sample_user_profile.context_prefs["genre"].data
    assert updated == {"action": True, "horror": False, "comedy": True}


def test_update_context_preference_non_dict(sample_user_profile):
    sample_user_profile.update_context_preference("min_rating", 9.0)
    assert sample_user_profile.context_prefs["min_rating"].data == 9.0

    with pytest.raises(ValueError):
        sample_user_profile.update_context_preference("unknown", 1)


def test_remove_context_preference_keys(sample_user_profile):
    sample_user_profile.remove_context_preference("genre", ["action"])
    assert "action" not in sample_user_profile.context_prefs["genre"].data


def test_remove_context_preference_all(sample_user_profile):
    sample_user_profile.remove_context_preference("genre", None)
    assert sample_user_profile.context_prefs["genre"].data == {}

    sample_user_profile.remove_context_preference("min_rating", None)
    assert sample_user_profile.context_prefs["min_rating"].data is None


def test_add_item_preferences(sample_user_profile):
    sample_user_profile.add_item_preferences([103, 104], [3.0, 2.5])
    assert sample_user_profile.item_prefs[103] == 3.0
    assert sample_user_profile.item_prefs[104] == 2.5


def test_add_item_preferences_mismatched_lengths(sample_user_profile):
    sample_user_profile.add_item_preferences([105], [1.0])  # OK
    assert sample_user_profile.item_prefs[105] == 1.0


def test_remove_item_preferences(sample_user_profile, capsys):
    sample_user_profile.remove_item_preferences([101])
    assert 101 not in sample_user_profile.item_prefs

    sample_user_profile.remove_item_preferences([999])  # Not in prefs
    captured = capsys.readouterr()
    assert "Item 999 not found" in captured.out
