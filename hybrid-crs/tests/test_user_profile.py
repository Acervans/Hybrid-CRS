import pytest

from llm.user_profile import UserProfile, ContextPreference, ContextType

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
