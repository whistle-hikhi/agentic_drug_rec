"""Phase 4 ReAct LangGraph agent (consultant-physician)."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .prompts import PHASE4_SYSTEM
from .tools   import PHASE4_TOOLS, ddi_check_tool

load_dotenv()


def build_phase4_agent(model: str = "gpt-4o-mini", use_ddi: bool = True):
    """Build and return the Phase 4 consultant-physician ReAct agent.

    The agent has access to three dual-model domain tools (each with an
    internal Role-Play LLM for uncertain-drug arbitration) and a
    summarisation tool.  An optional DDI checker tool can be included.

    Args:
        model:   OpenAI model name used for the ReAct agent and Role-Play LLMs.
        use_ddi: if True, include ddi_check_tool in the tool set.

    Returns:
        Compiled LangGraph ReAct agent.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    tools = list(PHASE4_TOOLS)
    if use_ddi:
        tools.append(ddi_check_tool)

    return create_react_agent(llm, tools=tools, prompt=PHASE4_SYSTEM)
