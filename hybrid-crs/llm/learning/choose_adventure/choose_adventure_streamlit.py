import streamlit as st
import asyncio
import time
import uuid

from choose_adventure_workflow import (
    ChooseYourOwnAdventureWorkflow,
    HumanResponseInterruptEvent,
    InputRequiredEvent,
    StopEvent,
    Blocks,
    Block,
    Segment,
)

from llama_index.core.workflow import Context


async def run_workflow():
    """Iterate the workflow and handle events."""

    handler = st.session_state.handler
    try:
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                blocks = (
                    await st.session_state.handler.ctx.store.get(
                        "blocks", Blocks(blocks_=[])
                    )
                ).blocks_
                if len(blocks) > 0:
                    st.session_state.current_block = blocks[-1]
                    st.session_state.choices = blocks[-1].segment.actions
                st.rerun()
            elif isinstance(event, StopEvent):
                st.session_state.choices = None
                st.session_state.story_finished = True

                final_block = (
                    await st.session_state.handler.ctx.store.get(
                        "blocks", Blocks(blocks_=[])
                    )
                ).blocks_[-1]
                final_block.segment.actions = []
                st.session_state.current_block = final_block
                st.rerun()

        await handler

    except StopAsyncIteration:
        st.session_state.story_finished = True
        st.rerun()


def handle_start():
    st.session_state.started = True


def handle_action(action):
    # Record the choice
    st.session_state.current_block.choice = action
    st.session_state.blocks.append(st.session_state.current_block)
    st.session_state.handler.ctx.send_event(
        HumanResponseInterruptEvent(response=action)
    )
    st.session_state.choices = None


def stream_text(text: str, sleep_secs: float = 0.003):
    for tok in text:
        yield tok
        time.sleep(sleep_secs)


async def main(llm):

    st.set_page_config(page_title="Choose Your Adventure", page_icon=":sparkles:")
    st.title("Choose Your Own Adventure")
    st.caption("🚀 Create and explore your own adventure story interactively!")

    if "handler" not in st.session_state:
        # Initialize session state variables
        st.session_state.started = False
        st.session_state.workflow = None
        st.session_state.handler = None
        st.session_state.wf_task = None

        st.session_state.blocks = []
        st.session_state.current_block = None
        st.session_state.choices = None
        st.session_state.story_finished = False
        st.session_state.initial_premise = None
        st.session_state.max_steps = 3

    # Show optional inputs for initial premise and max steps only if no blocks are set
    st.session_state.initial_premise = st.text_input(
        "Initial premise for your adventure:",
        placeholder="Leave blank to start a random adventure.",
        disabled=st.session_state.started,
    )
    st.session_state.max_steps = st.slider(
        "Maximum number of steps for your adventure:",
        min_value=1,
        max_value=20,
        value=3,  # Default value
        step=1,
        disabled=st.session_state.started,
    )
    if st.button(
        "Start Adventure", on_click=handle_start, disabled=st.session_state.started
    ):
        # Initialize the workflow with max_steps
        workflow = ChooseYourOwnAdventureWorkflow(
            llm=llm,
            max_steps=st.session_state.max_steps,
            timeout=None,
        )
        st.session_state.workflow = workflow
        st.session_state.handler = workflow.run()
        st.session_state.context = st.session_state.handler.ctx

        # If an initial premise is provided, use it to initialize the story
        if st.session_state.initial_premise:
            initial_block = Block(
                segment=Segment(plot=st.session_state.initial_premise, actions=[])
            )
            st.session_state.blocks.append(initial_block)
            st.session_state.current_block = initial_block
            await st.session_state.handler.ctx.store.set(
                "blocks", Blocks(blocks_=[initial_block])
            )

        st.session_state.wf_task = asyncio.create_task(run_workflow())
        await st.session_state.wf_task

    # Display previous story blocks as chat
    for block in st.session_state.blocks:
        st.chat_message("assistant").write(block.segment.plot)

        for act in block.segment.actions:
            st.button(
                act,
                key=str(uuid.uuid4()),
                disabled=True,
                icon="✅" if act == block.choice else None,
            )
        if block.choice:
            st.chat_message("user").write(block.choice)

    if st.session_state.current_block and (
        st.session_state.choices or st.session_state.story_finished
    ):
        # Show the current plot
        st.chat_message("assistant").write_stream(
            stream_text(st.session_state.current_block.segment.plot)
        )

        # Display action buttons for choices
        for action in st.session_state.choices or []:
            st.button(action, on_click=handle_action, args=(action,))

    if st.session_state.story_finished:
        st.write("The story has concluded!")
        st.button(
            "Start New Adventure",
            on_click=lambda: st.session_state.clear() and st.rerun(),
        )

    elif (
        st.session_state.workflow
        and not st.session_state.choices
        and st.session_state.current_block
        and st.session_state.current_block.choice
    ):
        # Resume after choices
        context = Context.from_dict(
            st.session_state.workflow,
            data=st.session_state.context.to_dict(),
        )

        st.session_state.handler = st.session_state.workflow.run(ctx=context)
        st.session_state.context = context

        st.session_state.wf_task = asyncio.create_task(run_workflow())
        await st.session_state.wf_task


if __name__ == "__main__":
    from llama_index.core.settings import Settings
    from llama_index.llms.ollama import Ollama

    llm = Settings.llm = Ollama(
        model="qwen2.5:3b", request_timeout=30.0, temperature=0.75, context_window=16384
    )

    asyncio.run(main(llm))
