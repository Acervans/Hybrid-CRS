from typing import Any, Optional, List
from pydantic import BaseModel
from enum import Enum


class ContextType(Enum):
    """Type of the contextual preferences."""

    DICT = 0
    BOOL = 1
    NUM = 2
    STR = 3


class ContextPreference(BaseModel):
    """Class that defines a contextual preference in the user profile.

    Attributes:
        type (ContextType): The type of the contextual preference (e.g., dict, bool, num, str).
        data (Any): The data for the preference represented in `type`.
    """

    type: ContextType
    data: Any


class UserProfile(BaseModel):
    """Class that stores a user's preferences during a conversational recommendation session

    Attributes:
        user_id (int | str): The ID of the user.
        context_prefs (dict[str, ContextPreference]): The contextual preferences of the user.
        item_prefs (dict[int, float]): The item preferences of the user.
    """

    user_id: int | str
    context_prefs: dict[str, ContextPreference]
    item_prefs: dict[int, float]

    def __init__(
        self,
        user_id: int | str,
        context_prefs: dict[str, ContextPreference] = {},
        item_prefs: dict[int, float] = {},
    ):
        super().__init__(
            user_id=user_id, context_prefs=context_prefs, item_prefs=item_prefs
        )

    def add_context_def(self, context: str, definition: ContextPreference):
        """Add a context definition to the user profile.

        Args:
            context (str): The context name.
            definition (dict): The context definition, including type and initial data.

        Raises:
            ValueError: If the context already exists in the preferences.
        """
        if context in self.context_prefs:
            raise ValueError(f"Context {context} already in preferences")
        self.context_prefs[context] = definition

    def remove_context_def(self, context: str):
        """Remove a context definition from the user profile.

        Args:
            context (str): The context name.

        Raises:
            ValueError: If the context does not exist in the preferences.
        """
        if context not in self.context_prefs:
            raise ValueError(f"Context {context} not found in preferences")
        del self.context_prefs[context]

    def update_context_preference(
        self, context: str, value: Any
    ):  # value can be a dict to be merged into context
        """Update a contextual preference in the user profile.

        Args:
            context (str): The context name.
            value (Any): The new value for the context preference.

        Raises:
            ValueError: If the context does not exist in the preferences.
        """
        if context not in self.context_prefs:
            raise ValueError(f"Context {context} not found in preferences")

        context_dict = self.context_prefs.get(context)
        if context_dict.type == ContextType.DICT:
            context_dict.data.update(value)
        else:
            context_dict.data = value

    def remove_context_preference(self, context: str, keys: Optional[list[Any]]):
        """Remove a contextual preference from the user profile.

        Args:
            context (str): The context name.
            keys (Optional[list[Any]]): The keys to remove from the context preference.

        Raises:
            ValueError: If the context does not exist in the preferences.
        """
        if context not in self.context_prefs:
            raise ValueError(f"Context {context} not found in preferences")

        context_dict = self.context_prefs.get(context)
        if context_dict.type == ContextType.DICT:
            if keys:
                for key in keys:
                    del context_dict.data[key]
            else:
                context_dict.data = {}
        else:
            context_dict.data = None

    def add_item_preferences(self, item_ids: List[int], ratings: List[float]):
        """Add item preferences to the user profile.

        Args:
            item_ids (List[int]): The list of item IDs.
            ratings (List[float]): The list of ratings corresponding to the item IDs.

        Raises:
            ValueError: If the lengths of item_ids and ratings do not match.
        """
        for i in range(len(item_ids)):
            self.item_prefs[item_ids[i]] = ratings[i]

    def remove_item_preferences(self, item_ids: List[int]):
        """Remove item preferences from the user profile.

        Logs a message if an item ID does not exist in the preferences.

        Args:
            item_ids (List[int]): The list of item IDs to remove.
        """
        for item_id in item_ids:
            if item_id in self.item_prefs:
                del self.item_prefs[item_id]
            else:
                print(f"Item {item_id} not found in preferences")
