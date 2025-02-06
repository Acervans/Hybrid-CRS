import uuid
import asyncio
from typing import List, Optional, Dict, Any

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
    Workflow,
    step,
)
from llama_index.retrievers import ChromaDBRetriever


# Constants
CTX_WINDOW = 16384

# ollama
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=360.0, context_window=CTX_WINDOW)

QUESTION_TEMPLATE = """
Based on the dataset with the following metadata: {metadata}, what are you interested in?
"""

K = 5  # Number of top items to recommend

class Metadata(BaseModel):
    name: str
    num_items: int
    description: str

class ChatEvent(Event):
    response: str

class RecommendEvent(Event):
    recommendations: List[str]

class Intent(BaseModel):
    intent: str

    @validator('intent')
    def validate_intent(cls, v):
        allowed_intents = ["recommendation", "chat"]
        if v.lower() not in allowed_intents:
            raise ValueError(f"Invalid intent: {v}. Allowed intents are: {allowed_intents}")
        return v

class HybridCRSWorkflow(Workflow):
    def __init__(self, llm, retriever, max_steps: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps

    @step
    async def check_intent(self, ctx: Context, ev: StartEvent | HumanResponseEvent) -> Event:
        intent_str = await self.llm.predict("What is the user's intent?", ctx.chat_history)
        try:
            intent = Intent(intent=intent_str)
        except ValueError as e:
            return ChatEvent(response=str(e))

        if intent.intent == "chat":
            return ChatEvent(response="How can I assist you today?")
        return await self.start_recommendation(ctx, ev)

    @step
    async def start_recommendation(self, ctx: Context, ev: StartEvent | HumanResponseEvent) -> InputRequiredEvent:
        if not ctx.chat_history:
            metadata = Metadata(name="Sample Dataset", num_items=100, description="A dataset of sample items.")
            question = QUESTION_TEMPLATE.format(metadata=metadata)
            return InputRequiredEvent(prefix=question)
        return await self.elicit_preferences(ctx, ev)

    @step
    async def elicit_preferences(self, ctx: Context, ev: HumanResponseEvent) -> InputRequiredEvent | RecommendEvent:
        preferences = await self.llm.predict("What are the user's preferences?", ctx.chat_history)
        if self.has_sufficient_preferences(preferences):
            return await self.recommend_items(ctx, preferences)
        return InputRequiredEvent(prefix="Tell me more about your preferences.")

    def has_sufficient_preferences(self, preferences: str) -> bool:
        # Implement logic to check if preferences are sufficient
        return len(preferences.split()) > 5

    @step
    async def recommend_items(self, ctx: Context, preferences: str) -> RecommendEvent:
        items = self.retriever.retrieve(preferences)
        # Implement specific recommender system logic here
        top_items = self.llm.rerank(items, preferences, top_k=K)
        return RecommendEvent(recommendations=top_items)

if __name__ == "__main__":
    async def main():
        from llama_index.core.settings import Settings
        from llama_index.llms.ollama import Ollama

        llm = Settings.llm = Ollama(
            model="qwen2.5:3b", request_timeout=30.0, temperature=0.75
        )
        retriever = ChromaDBRetriever(index_path="/path/to/chromadb")

        w = HybridCRSWorkflow(llm, retriever, max_steps=3, timeout=None)
        handler = w.run()

        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                print(event.prefix)
                choice = input("\nYour response: ")
                handler.ctx.send_event(HumanResponseEvent(response=choice))
            elif isinstance(event, ChatEvent):
                print(event.response)
            elif isinstance(event, RecommendEvent):
                print("Top recommendations:")
                for item in event.recommendations:
                    print(item)

        await handler

    asyncio.run(main())
