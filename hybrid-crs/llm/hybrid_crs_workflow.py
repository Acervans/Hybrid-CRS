import os
import asyncio
import fireducks.pandas as pd
import uuid
import sys
import json

from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
    step,
    Context,
)
from llama_index.core.llms import ChatMessage, MessageRole, CompletionResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, ToolOutput
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama

from pydantic import BaseModel, field_validator
from typing import Annotated, Any, Dict, List, Optional, Union


# --- Module Setup ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from recsys.falkordb_recommender import FalkorDBRecommender
    from recsys.recbole_utils import load_data_and_model, get_recommendations
    from data_processing.data_utils import sniff_delimiter
    from user_profile import UserProfile, ContextPreference, ContextType
except ImportError as e:
    try:
        # Parent import (/hybrid-crs)
        from .user_profile import UserProfile, ContextPreference, ContextType
    except ImportError:
        print(f"Error setting up imports: {e}")
        print("Please ensure you run this script from within the project structure.")
        sys.exit(1)


# --- Configuration ---
HOSTNAME = os.getenv("LOCAL_SERVICES_HOST", "localhost")
CTX_WINDOW = 16384
TOP_N = 5  # Number of recommendations to return
MIN_ITEMS_CF = 3  # Minimum items the user iteracted with to enable CF/EASER

llm: FunctionCallingLLM = Ollama(
    model="qwen2.5:3b",
    base_url=f"http://{HOSTNAME}:11434",
    temperature=0.2,
    request_timeout=360.0,
    context_window=CTX_WINDOW,
)
Settings.llm = llm


# --- Dataset Update Utility ---
async def update_dataset(
    inter_path: str, user_id: str, item_ids: List[str], ratings: List[float]
):
    """Appends new interactions to the dataset's interaction file."""
    try:
        with open(inter_path, "r") as f:
            lines = f.readlines(10)
        sep = sniff_delimiter(lines)

        with open(inter_path, "r") as f:
            existing_df = pd.read_csv(
                f, sep=sep, dtype={"user_id:token": str, "item_id:token": str}
            )

        new_data_df = pd.DataFrame(
            {
                "user_id:token": user_id,
                "item_id:token": list(map(str, item_ids)),
                "rating:float": ratings,
            }
        )

        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        combined_df.drop_duplicates(
            subset=["user_id:token", "item_id:token"], keep="last", inplace=True
        )
        combined_df.to_csv(inter_path, index=False, sep=sep)
    except FileNotFoundError:
        print(f"Interaction file not found at {inter_path}")
    except Exception as e:
        print(f"An error occurred during dataset update")
        raise e


# --- Prompt Templates ---
system_prompt = PromptTemplate(
    """
You are a friendly and helpful recommender agent named '{agent_name}'.
Your goal is to understand a user's preferences and recommend items from the '{dataset_name}' dataset.
The dataset is about: {description}.

You have a set of tools to achieve this. You MUST use them as described:
1.  **Gather Preferences**: Ask the user about their tastes for item features. Use the `update_user_preferences` tool to save each preference they state. Do NOT use this tool for item feedback.
2.  **Get Recommendations**: Once you have some preferences, and ONLY if the user wants, call the `get_recommendations` tool to fetch a list of items.
3.  **Process Feedback**: After presenting items, if the user gives natural language feedback like "I liked the first one", you MUST process this by calling the `add_user_item_feedback` tool.
    -   Infer a rating: 5.0 for positive feedback (like, love, enjoy), and 1.0 for negative feedback (dislike, hate, not for me).
    -   Collect ALL feedback from the user's message into a single call.
4.  **End Conversation**: When the user indicates they are finished (e.g., 'that's all', 'I'm done'), you MUST call the `end_session` tool.

Here are the item features you can ask about, along with their possible values. Do NOT make up any feature. You MUST map the user's input to one of these exact values when using tools.
{schema}

Start the conversation by introducing yourself and asking what the user is looking for.
"""
)

explanation_prompt = PromptTemplate(
    """
You are an expert recommender. The user provided these contextual preferences so far: 
{preferences}

You recommended the item '{item_name}' with properties:
{item_properties}

Rephrase the following structured explanations for this recommendation into a single, fluent, non-technical paragraph.
Do not use bullet points, be concise, and strictly reference the preferences and properties provided.

Explanations:
{explanations}
"""
)

correction_prompt = PromptTemplate(
    """Given the user's input '{update_value}', which of the following valid options is the best match?
Options: {valid_values}.

If none match well, respond with 'None'. Respond with only the best matching option from the list.
"""
)


# --- Workflow Events ---
class RecommendationGeneratedEvent(Event):
    """Event fired when recommendations are ready to be shown to the user."""

    recommendations: list[dict]
    explanations: list[str]


class JsonFeedbackEvent(Event):
    """Event carrying explicit user feedback from a structured JSON."""

    feedback: Dict[str, float]


class StreamEvent(Event):
    """Carries a chunk of a streaming LLM response."""

    delta: str


# --- Main Workflow Class ---
class HybridCRSWorkflow(Workflow):
    """A Conversational Recommendation Workflow with streaming
    using a hybrid graph + expert model method."""

    def __init__(
        self,
        user_id: str,
        agent_name: str,
        dataset_name: str,
        dataset_dir: str,
        description: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a Hybrid Conversational Recommendation Workflow.

        This sets up the LLM-backed workflow with access to both a graph-based recommender
        (FalkorDB) and a pre-trained expert model (RecBole).

        Args:
            user_id (str): Identifier for the current user
            agent_name (str): The name/persona of the recommender agent
            dataset_name (str): Name of the dataset used for recommendations
            dataset_dir (str): Directory path to the processed dataset files
            description (Optional[str]): A brief description of what the dataset is about
            model_dir (Optional[str]): Directory path to the saved RecBole model and dataset.
                If not provided, defaults to `../recsys/saved`.
            **kwargs: Additional keyword arguments passed to the parent `Workflow` class
        """
        self.user_id = user_id
        self.agent_name = agent_name
        self.dataset_name = dataset_name
        self.description = description or "No description provided"

        super().__init__(**kwargs)
        assert (
            llm.metadata.is_function_calling_model
        ), "LLM must support function calling."

        self.inter_path = os.path.join(dataset_dir, f"{self.dataset_name}.inter")
        self.schema_with_values: Dict = {}

        if self._verbose:
            print("Connecting to FalkorDB...")
        self.falkordb_rec = FalkorDBRecommender(
            dataset_name=self.dataset_name, dataset_dir=dataset_dir, clear=False
        )

        recbole_model_dir = model_dir or os.path.join("..", "recsys", "saved")
        recbole_model_path = os.path.join(recbole_model_dir, f"{dataset_name}.pth")
        recbole_dataset_path = os.path.join(
            recbole_model_dir, f"{dataset_name}-Dataset.pth"
        )
        if self._verbose:
            print(f"Loading RecBole model from {recbole_model_path}...")
        if os.path.exists(recbole_model_path):
            self.recbole_config, self.recbole_model, self.recbole_dataset, _, _, _ = (
                load_data_and_model(
                    load_model=recbole_model_path,
                    update_config={
                        "dataset_save_path": recbole_dataset_path,
                    },
                )
            )
            self.expert_model_loaded = True
            if self._verbose:
                print("RecBole model loaded successfully.")
        else:
            self.expert_model_loaded = False
            if self._verbose:
                print(
                    f"Warning: RecBole model not found at {recbole_model_path}. Expert recommendations disabled."
                )

        self.tools = self._get_tools()

    def _get_tools(self) -> List[FunctionTool]:
        """Defines the tools available to the LLM."""
        return [
            FunctionTool.from_defaults(
                self.get_recommendations, name="get_recommendations"
            ),
            FunctionTool.from_defaults(
                self.update_user_preferences, name="update_user_preferences"
            ),
            FunctionTool.from_defaults(
                self.add_user_item_feedback, name="add_user_item_feedback"
            ),
            FunctionTool.from_defaults(self.end_session, name="end_session"),
        ]

    # --- Tool Implementations (Workflow Logic) ---

    async def end_session(self, ctx: Context) -> ToolOutput:
        """Ends the session."""
        await ctx.store.set("end_session", True)
        return ToolOutput(
            tool_name="end_session",
            raw_input={},
            raw_output=None,
        )

    async def update_user_preferences(
        self,
        ctx: Context,
        context: Annotated[str, "Name of the context feature (e.g., category, name)"],
        value: Any | List[Any],
        delete: Annotated[bool, "If True, the preferences will be deleted"] = False,
    ) -> ToolOutput:
        """Updates the user's profile with a new contextual preference, with validation."""
        item_schema = self.schema_with_values.get("Item Features", {})
        feature_info = item_schema.get(context)
        final_values = []

        if feature_info and isinstance(feature_info.get("values"), list):
            valid_values = set(feature_info["values"])

            class SelectedValue(BaseModel):
                """Value selected from a list of possible values"""

                value: str

                @field_validator("value")
                @classmethod
                def value_in_list(cls, value: str) -> str:
                    if value not in valid_values:
                        raise ValueError(
                            f"'{value}' not in possible values for '{context}'"
                        )
                    return value

            if not isinstance(value, list):
                value = [value]

            for update_value in value:
                lower = str(update_value).lower()
                found = next((v for v in valid_values if str(v).lower() == lower), None)

                if not found:
                    if self._verbose:
                        print(
                            f"'{update_value}' not an exact match for '{context}'. Attempting to correct with LLM..."
                        )
                    correction_response = await llm.astructured_predict(
                        output_cls=SelectedValue,
                        prompt=correction_prompt.format(
                            update_value=update_value, valid_values=valid_values
                        ),
                    )
                    corrected_value = correction_response.value.strip()

                    if corrected_value != "None" and corrected_value in valid_values:
                        final_values.append(corrected_value)
                        if self._verbose:
                            print(f"Corrected '{update_value}' to '{corrected_value}'.")
                else:
                    final_values.append(found)

            if len(final_values) == 0:
                return ToolOutput(
                    tool_name="update_user_preferences",
                    content=f"Sorry, I couldn't find a matching value for '{value}'.",
                    raw_input={"context": context, "value": value},
                    raw_output=None,
                )

        profile: UserProfile = await ctx.store.get("profile")
        if context not in profile.context_prefs:
            ctype = ContextType.STR
            ftype = feature_info.get("type", "")
            if feature_info and ("seq" in ftype or ftype == "token"):
                ctype = ContextType.DICT
            elif feature_info and "float" in ftype:
                ctype = ContextType.NUM
            profile.add_context_def(
                context,
                ContextPreference(
                    type=ctype, data={} if ctype == ContextType.DICT else None
                ),
            )

        if profile.context_prefs[context].type == ContextType.DICT:
            profile.update_context_preference(
                context, {v: not delete for v in final_values}
            )
        else:
            if delete:
                profile.remove_context_def(context)
            else:
                profile.update_context_preference(context, value)

        return ToolOutput(
            tool_name="update_user_preferences",
            content=f"Preference '{context}' updated with values '{final_values}'.",
            raw_input={"context": context, "value": value},
            raw_output=None,
        )

    async def add_user_item_feedback(
        self, ctx: Context, item_ids: List[Any], ratings: List[float]
    ) -> ToolOutput:
        """Processes natural language feedback, updating the session profile and backend data sources."""
        if not item_ids:
            return ToolOutput(
                tool_name="add_user_item_feedback",
                content="No feedback was provided.",
                raw_input={"item_ids": item_ids, "ratings": ratings},
                raw_output=None,
            )

        profile: UserProfile = await ctx.store.get("profile")
        profile.add_item_preferences(item_ids, ratings)

        interactions = list(zip(item_ids, ratings))
        self.falkordb_rec.add_user_interactions(self.user_id, interactions)
        await update_dataset(self.inter_path, self.user_id, item_ids, ratings)

        is_new_user = (
            len(self.falkordb_rec.get_items_by_user(self.user_id)) < MIN_ITEMS_CF
        )
        await ctx.store.set("is_new_user", is_new_user)

        return ToolOutput(
            tool_name="add_user_item_feedback",
            content=f"Recorded your feedback for {len(item_ids)} items.",
            raw_input={"item_ids": item_ids, "ratings": ratings},
            raw_output=None,
        )

    async def get_recommendations(self, ctx: Context) -> ToolOutput:
        """Gets hybrid recommendations based on the user's current profile."""
        profile: UserProfile = await ctx.store.get("profile")
        is_new_user: bool = await ctx.store.get("is_new_user", True)
        item_props = {
            key: (
                [k for k, v in pref.data.items() if v]
                if pref.type == ContextType.DICT
                else pref.data
            )
            for key, pref in profile.context_prefs.items()
            if pref.data
        }

        # If new user, recommend with contextual preferences
        # Otherwise use hybrid recommendation (contextual + CF)
        use_expert = self.expert_model_loaded and not is_new_user
        if use_expert:
            falkor_top_n = TOP_N // 2
            expert_top_n = TOP_N - falkor_top_n
        else:
            falkor_top_n = TOP_N

        falkor_recs = (
            self.falkordb_rec.recommend_contextual(
                user_id=self.user_id, item_props=item_props, top_n=falkor_top_n
            )
            if is_new_user
            else self.falkordb_rec.recommend_hybrid(
                user_id=self.user_id, item_props=item_props, top_n=falkor_top_n, k=10
            )
        )
        final_recs = {item.properties["item_id"]: item for item, _score in falkor_recs}

        if use_expert:
            if self._verbose:
                print(
                    "Strategy: Hybrid recommendations using FalkorDB (Contextual + CF) + RecBole expert model."
                )
            expert_recs = get_recommendations(
                user_id=str(self.user_id),
                model=self.recbole_model,
                dataset=self.recbole_dataset,
                cutoff=expert_top_n,
            )
            for _, item_id in expert_recs:
                if item_id not in final_recs:
                    item_node = self.falkordb_rec.g.query(
                        "MATCH (i:Item {item_id: $item_id}) RETURN i",
                        {"item_id": item_id},
                    ).result_set[0][0]
                    final_recs[item_id] = item_node
        elif self._verbose:
            if is_new_user:
                print("Strategy: Contextual recommendations using FalkorDB.")
            else:
                print("Strategy: Contextual + CF recommendations using FalkorDB.")

        items_to_explain = list(final_recs.values())
        explanations = await self.generate_explanations(profile, items_to_explain)
        recs_output = [item.properties for item in items_to_explain]
        await ctx.store.set("last_recommendations", (recs_output, explanations))

        return ToolOutput(
            tool_name="get_recommendations",
            content="Generated recommendations.",
            raw_input={},
            raw_output=None,
        )

    async def generate_explanations(
        self, profile: UserProfile, items: list
    ) -> list[str]:
        """Generates natural language explanations for a list of recommended items."""
        tasks = [self.explain_recommendation(profile, item) for item in items]
        responses = await asyncio.gather(*tasks)
        return [r.text for r in responses]

    async def explain_recommendation(
        self, profile: UserProfile, item: Any
    ) -> CompletionResponse:
        """Generates natural language explanations for a recommendation."""
        explanations = self.falkordb_rec.explain_blackbox_recs(
            user_id=self.user_id,
            item_id=item.properties["item_id"],
            shared_props=self.falkordb_rec.item_feats.keys(),
        )
        try:
            rating = float(explanations[-1])
        except (IndexError, ValueError):
            rating = None
        item.properties["falkordb_rating"] = rating

        return llm.acomplete(
            explanation_prompt.format(
                preferences=profile.context_prefs,
                item_name=item.properties.get("name", item.properties["item_id"]),
                item_properties={
                    k: v
                    for k, v in item.properties.items()
                    if k not in ("item_id", "name", "pagerank")
                },
                explanations="\n".join(f"- {e}" for e in explanations),
            )
        )

    # --- Workflow Steps ---
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> InputRequiredEvent:
        """Initializes the workflow and streams the first agent message."""
        if self._verbose:
            print(
                "Fetching unique feature values for Item nodes to provide context to LLM..."
            )
        item_features_key = "Item Node Features"
        if item_features_key in self.falkordb_rec.schema:
            item_feats = self.falkordb_rec.schema[item_features_key]
            item_schema_data = {}
            for feat, feat_type in item_feats.items():
                should_fetch_values = feat_type.endswith("token") or (
                    feat_type.endswith("token_seq") and feat == "category"
                )
                if should_fetch_values:
                    possible_values = self.falkordb_rec.get_unique_feat_values(
                        "Item", feat
                    )
                    item_schema_data[feat] = {
                        "type": feat_type,
                        "values": (
                            sorted(possible_values)
                            if len(possible_values) <= 50
                            else f"Too many to list ({len(possible_values)})."
                        ),
                    }
                else:
                    item_schema_data[feat] = {"type": feat_type, "values": "N/A"}
            self.schema_with_values["Item Features"] = item_schema_data

        memory = ChatMemoryBuffer.from_defaults(
            chat_history=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=system_prompt.format(
                        agent_name=self.agent_name,
                        dataset_name=self.dataset_name,
                        description=self.description,
                        schema=json.dumps(self.schema_with_values, indent=2),
                    ),
                )
            ],
            llm=llm,
        )

        chat_stream = await llm.astream_chat(memory.get())
        response = None
        async for response in chat_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta))

        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response.message))

        await ctx.store.set("memory", memory)
        await ctx.store.set("profile", UserProfile(user_id=self.user_id))
        self.falkordb_rec.create_user(self.user_id)
        is_new_user = (
            len(self.falkordb_rec.get_items_by_user(self.user_id)) < MIN_ITEMS_CF
        )
        await ctx.store.set("is_new_user", is_new_user)

        if self._verbose:
            print(f"\nUser '{self.user_id}' is {'new' if is_new_user else 'existing'}.")

        return InputRequiredEvent(from_event=ev.__repr_name__())

    @step
    async def process_user_input(
        self, ctx: Context, ev: HumanResponseEvent
    ) -> Union[
        JsonFeedbackEvent, RecommendationGeneratedEvent, InputRequiredEvent, StopEvent
    ]:
        """Processes user input, streaming responses and handling tool calls."""
        try:
            feedback_data = json.loads(ev.response)
            if isinstance(feedback_data, dict):
                await ctx.store.set("last_recommendations_for_llm_context", None)
                return JsonFeedbackEvent(feedback=feedback_data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        memory: ChatMemoryBuffer = await ctx.store.get("memory")
        if last_recs_ctx := await ctx.store.get(
            "last_recommendations_for_llm_context", None
        ):
            memory.put(ChatMessage(role=MessageRole.SYSTEM, content=last_recs_ctx))
            await ctx.store.set("last_recommendations_for_llm_context", None)

        memory.put(ChatMessage(role=MessageRole.USER, content=ev.response))

        tool_stream = await llm.astream_chat_with_tools(
            tools=self.tools,
            chat_history=memory.get(),
            context=ctx,
            tool_required=False,
        )

        response = None
        tool_calls = None
        async for response in tool_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        if response is not None:
            tool_calls = response.message.additional_kwargs.get("tool_calls", None)
            if self._verbose:
                print(f"Tool calls: \n{"\n".join(tool_calls)}")

        if not tool_calls:
            return InputRequiredEvent(from_event=ev.__repr_name__())
        else:
            # Call tools, get recommendations last
            tool_calls = sorted(
                tool_calls, key=lambda t: t["function"]["name"] == "get_recommendations"
            )
            tools_by_name = {t.metadata.name: t for t in self.tools}
            for tool_call in tool_calls:
                argument_dict = tool_call["function"]["arguments"]
                tool = tools_by_name[tool_call["function"]["name"]]
                output = await tool.async_fn(ctx=ctx, **argument_dict)
                memory.put(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=output.content,
                        tool_name=output.tool_name,
                    )
                )

            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.message,
                    additional_kwargs=response.message.additional_kwargs,
                )
            )

        if recs_exps := await ctx.store.get("last_recommendations", None):
            await ctx.store.set("last_recommendations", None)
            # TODO if event is RecommendationGeneratedEvent, show feedback UI
            return RecommendationGeneratedEvent(
                recommendations=recs_exps[0], explanations=recs_exps[1]
            )

        if await ctx.store.get("end_session", False):
            memory.put(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="The user has indicated their desire to end this session. Send an appropriate farewell message",
                )
            )
            final_stream = await llm.astream_chat(memory.get())
            async for response in final_stream:
                ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))
            memory.put(
                ChatMessage(role=MessageRole.ASSISTANT, content=response.message)
            )
            profile: UserProfile = await ctx.store.get("profile")
            return StopEvent(result=profile.model_dump())

        chat_stream = await llm.astream_chat(memory.get())
        async for response in chat_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response.message))

        return InputRequiredEvent(from_event=ev.__repr_name__())

    @step
    async def process_json_feedback(
        self, ctx: Context, ev: JsonFeedbackEvent
    ) -> InputRequiredEvent:
        """Processes structured JSON feedback, updates backends, and confirms with user."""
        item_ids = list(ev.feedback.keys())
        ratings = list(ev.feedback.values())

        if item_ids:
            profile: UserProfile = await ctx.store.get("profile")
            profile.add_item_preferences(item_ids, ratings)
            self.falkordb_rec.add_user_interactions(
                self.user_id, list(zip(item_ids, ratings))
            )
            await update_dataset(self.inter_path, self.user_id, item_ids, ratings)

        memory: ChatMemoryBuffer = await ctx.store.get("memory")
        memory.put(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Feedback received for {len(item_ids)} items. Thank the user and continue the session.",
            )
        )
        chat_stream = await llm.astream_chat(memory.get())
        async for response in chat_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response.message))

        is_new_user = (
            len(self.falkordb_rec.get_items_by_user(self.user_id)) < MIN_ITEMS_CF
        )
        await ctx.store.set("is_new_user", is_new_user)
        return InputRequiredEvent(from_event=ev.__repr_name__())

    @step
    async def post_recommendation(
        self, ctx: Context, ev: RecommendationGeneratedEvent
    ) -> InputRequiredEvent:
        """After recommendations are shown, this step transitions the workflow back to waiting for user input."""
        ctx.write_event_to_stream(ev)
        rec_context_str = (
            "CONTEXT: Here are the items you just recommended. Use their item_id when the user provides feedback.\n"
            + json.dumps(ev.recommendations, indent=2)
        )
        await ctx.store.set("last_recommendations_for_llm_context", rec_context_str)
        return InputRequiredEvent(from_event=ev.__repr_name__())


# CLI Movielens-100k Example
async def main():
    """Main function to run the conversational recommender using the stream_events pattern."""
    user_id = "test_user_" + str(uuid.uuid4())[:8]
    dataset = "ml-100k"
    dataset_dir = os.path.join(
        "..", "data_processing", "datasets", "processed", dataset
    )
    model_dir = os.path.join("..", "recsys", "saved")

    print(f"Starting session for user: {user_id}")
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    w = HybridCRSWorkflow(
        user_id=user_id,
        agent_name="CineMate",
        dataset_name=dataset,
        dataset_dir=dataset_dir,
        description="A recommender for movies from the MovieLens 100k dataset.",
        model_dir=model_dir,
        timeout=600,
        verbose=False,
    )
    handler = w.run()

    async for event in handler.stream_events():
        if isinstance(event, StreamEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, RecommendationGeneratedEvent):
            print(
                "\n\nCineMate: Based on your preferences, I recommend the following movies:",
                flush=True,
            )
            for rec, exp in zip(event.recommendations, event.explanations):
                print(
                    f"\n🎬 **{rec.get('name', rec['item_id'])}** (ID: {rec['item_id']})",
                    flush=True,
                )
                print(f"   - {exp}")
            print(
                "\n\n(To give feedback, you can type a message like 'I liked ID 123' or send a JSON object like '{{\"123\": 5.0, \"456\": 1.0}}')",
                flush=True,
            )
        elif isinstance(event, InputRequiredEvent):
            user_input = input("\nYou: ")
            handler.ctx.send_event(HumanResponseEvent(response=user_input))

    print("\nWorkflow finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
