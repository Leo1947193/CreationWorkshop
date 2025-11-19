"""Conflict-Driven Narrative Generator (Module 4)."""
from __future__ import annotations

from textwrap import dedent

from src.core.llm import get_llm_client
from src.core.schemas import GlobalState
from .wm_kg import WorldModelKnowledgeGraph, get_axioms_text

final_story_prompt_template = """
<role>
你是一位才华横溢的短篇小说作家，擅长撰写丰富有趣、情感强烈的科幻与奇幻故事。
</role>

<task>
你的任务是根据以下严格的“世界规则”（上下文）和一个必须体现的“核心缺陷”，创作一篇高质量的短篇小说（约 1000 字）。

这篇故事必须是一个**警示故事**，其核心情节和戏剧冲突**必须**源于“核心缺陷”所导致的灾难性后果。
</task>

<constraints>
**!! 绝对约束!!**
1.  **遵守世界规则：** 你必须严格遵守 <WorldAxioms> 中列出的所有规则。这些是这个世界不可违背的物理和社会法则。
2.  **必须发生失败：** 你**绝对不能**创造一个“皆大欢喜”的结局或一个“deus ex machina”（机械降神）来*解决* “核心缺陷”。
3.  **缺陷必须显现：** 这个故事的重点是展示一个系统性缺陷的悲剧性后果。主角的努力（如果有的话）**必须失败**，或者缺陷的后果必须以一种不可避免的方式展现出来。
</constraints>

<context>
--- 世界观公理 (来自 WM-KG 的完整构造) ---
<WorldAxioms>
{world_axioms_text}
</WorldAxioms>

--- 核心缺陷 (故事的中心冲突) ---
<CoreDefect>
缺陷描述: {defect_description}
灾难性后果 (必须在故事中体现): {defect_consequence}
</CoreDefect>
</context>

<output>
**指示：**
现在，请将“核心缺陷”作为情节的中心，在“世界观公理”的约束下，编织一个连贯的、充满情感的叙事。
**展示 (Show)，不要告知 (Tell)。**
让读者感受到这个世界的规则，然后感受到这些规则因其内在“核心缺陷”而崩溃时的绝望。

**[此处开始你的短篇小说]**
</output>
""".strip()


def _synthesise_story(world_axioms: str, defect_description: str, consequence: str) -> str:
    """Fallback story composer used when OpenRouter 调用失败。"""
    template = dedent(
        """
        【占位稿】系统未能调用 LLM，请在 SEE/CDA 中继续完善信息或检查 OpenRouter 配置。

        世界公理摘要：
        {axioms}

        需呈现的缺陷：
        - 描述：{defect}
        - 后果：{consequence}
        """
    ).strip()
    return template.format(
        axioms=world_axioms or "（暂无公理）",
        defect=defect_description,
        consequence=consequence,
    )


def run_cdng_module(state: GlobalState) -> GlobalState:
    if not state.top_defect:
        state.generated_story = "目前未发现缺陷，系统无法生成剧情。"
        state.next_module_to_call = "IDLE"
        return state

    wmkg = WorldModelKnowledgeGraph(state.internal_kb_path)
    world_axioms_text = get_axioms_text(wmkg)
    prompt = final_story_prompt_template.format(
        world_axioms_text=world_axioms_text,
        defect_description=state.top_defect.description,
        defect_consequence=state.top_defect.long_term_consequence,
    )

    story = ""
    try:
        llm = get_llm_client(temperature=0.85)
        response = llm.invoke(prompt)
        story = response.content if hasattr(response, "content") else str(response)
    except Exception:
        story = _synthesise_story(
            world_axioms_text,
            state.top_defect.description,
            state.top_defect.long_term_consequence,
        )
    story = dedent(story).strip()

    state.generated_story = story
    state.next_module_to_call = "IDLE"
    return state
