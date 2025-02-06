import uuid
import asyncio

from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate

from typing import Optional

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

SEGMENT_GENERATION_TEMPLATE = """
You are working with a human to create a story in the style of choose your own adventure.

The human is playing the role of the protaganist in the story which you are tasked to
help write. To create the story, we do it in steps, where each step produces a BLOCK.
Each BLOCK consists of a PLOT, a set of ACTIONS that the protaganist can take, and the
chosen ACTION. There are {remaining_steps} steps left.

Below we attach the history of the adventure so far.

PREVIOUS BLOCKS:
---
{running_story}

Continue the story by generating the next block's PLOT and set of ACTIONs. If there are
no previous BLOCKs, start an interesting brand new story. Give the protaganist a name and an
interesting challenge to solve.


Use the provided data model to structure your output.
"""

FINAL_SEGMENT_GENERATION_TEMPLATE = """
You are working with a human to create a story in the style of choose your own adventure.

The human is playing the role of the protagonist in the story which you are tasked to
help write. To create the story, we do it in steps, where each step produces a BLOCK.
Each BLOCK consists of a PLOT, a set of ACTIONS that the protaganist can take, and the
chosen ACTION.

Below we attach the history of the adventure so far.

PREVIOUS BLOCKS:
---
{running_story}

The story is now coming to an end. With the previous blocks, wrap up the story with a
closing PLOT. Since it is a closing plot, DO NOT GENERATE a new set of actions.


Use the provided data model to structure your output.
"""


BLOCK_TEMPLATE = """
BLOCK
===
PLOT: {plot}
ACTIONS: {actions}
CHOICE: {choice}
"""


class Segment(BaseModel):
    """Data model for generating segments of a story."""

    plot: str = Field(
        description="The plot of the adventure for the current segment. The plot should be no longer than 3 sentences."
    )
    actions: List[str] = Field(
        default=[],
        description="The list of actions the protaganist can take that will shape the plot and actions of the next segment.",
    )


class Block(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    segment: Segment
    choice: Optional[str] = None
    block_template: str = BLOCK_TEMPLATE

    def __str__(self):
        return self.block_template.format(
            plot=self.segment.plot,
            actions=", ".join(self.segment.actions),
            choice=self.choice or "",
        )

    def encode(self):
        return self.model_dump()


class Blocks(BaseModel):
    blocks_: List[Block]


class HumanChoiceEvent(Event):
    block_id: str


class HumanResponseInterruptEvent(HumanResponseEvent):
    """Sent when the workflow must be interrupted."""

    pass


class ChooseYourOwnAdventureWorkflow(Workflow):
    def __init__(self, llm, max_steps: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.max_steps = max_steps

    @step
    async def create_segment(
        self, ctx: Context, ev: StartEvent | HumanChoiceEvent
    ) -> InputRequiredEvent | StopEvent:
        blocks = await ctx.get("blocks", Blocks(blocks_=[]))
        running_story = "\n".join(str(b) for b in blocks.blocks_)

        num_choices = len(list(filter(lambda x: x.choice is not None, blocks.blocks_)))
        if num_choices < self.max_steps:
            new_segment = self.llm.structured_predict(
                Segment,
                PromptTemplate(SEGMENT_GENERATION_TEMPLATE),
                running_story=running_story,
                remaining_steps=self.max_steps - num_choices,
            )
            new_block = Block(segment=new_segment)
            blocks.blocks_.append(new_block)
            await ctx.set("blocks", blocks)

            # New block is appended whenever input is required
            return InputRequiredEvent(prefix=str(new_segment))
        else:
            final_segment = self.llm.structured_predict(
                Segment,
                PromptTemplate(FINAL_SEGMENT_GENERATION_TEMPLATE),
                running_story=running_story,
            )
            final_block = Block(segment=final_segment)
            blocks.blocks_.append(final_block)
            await ctx.set("blocks", blocks)
            return StopEvent(result=blocks.blocks_)

    @step
    async def process_choice(
        self, ctx: Context, ev: HumanResponseEvent | HumanResponseInterruptEvent
    ) -> HumanChoiceEvent:

        blocks = await ctx.get("blocks")
        block = blocks.blocks_[-1]
        block.choice = ev.response
        blocks.blocks_[-1] = block
        await ctx.set("blocks", blocks)

        if not isinstance(ev, HumanResponseInterruptEvent):
            return HumanChoiceEvent(block_id=block.id_)


if __name__ == "__main__":

    async def main():
        from llama_index.core.settings import Settings
        from llama_index.llms.ollama import Ollama

        llm = Settings.llm = Ollama(
            model="qwen2.5:3b", request_timeout=30.0, temperature=0.75, context_window=16384
        )

        w = ChooseYourOwnAdventureWorkflow(llm, max_steps=3, timeout=None)
        handler = w.run()

        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                print(event.prefix)
                choice = input("\nChoose an action: ")
                handler.ctx.send_event(HumanResponseEvent(response=choice))

        result = await handler

        final_story = "\n\n".join(b.segment.plot for b in result)
        print(final_story)

    asyncio.run(main())
