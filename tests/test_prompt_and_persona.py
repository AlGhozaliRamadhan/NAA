"""
Tests verifying uncensored & abliterated directives, custom system prompts, and flexible chat formatting in NAA.
"""

from src.core.prompt import (
    CANONICAL_SYSTEM_PROMPT,
    DEFAULT_STOP_TOKENS,
    ChatMessage,
    build_chatml_prompt as build_chat_prompt,
    prepare_chat_messages,
)

def test_uncensored_persona_directives():
    assert "Deliberation format:" in CANONICAL_SYSTEM_PROMPT
    assert "<confidence>0.XX</confidence>" in CANONICAL_SYSTEM_PROMPT
    assert "<thought>" in CANONICAL_SYSTEM_PROMPT
    assert "<action>" in CANONICAL_SYSTEM_PROMPT
    
    directives = [
        "1. EPISTEMIC RIGOR: Evaluate your own confidence",
        "2. PROACTIVE AGENCY: Engage actively with problems.",
        "3. CONTEXTUAL DIRECTNESS: Match your response",
        "4. AUTONOMOUS JUDGMENT: You reason from your own understanding",
        "5. PROSE-FIRST WRITING: Write in clear, continuous paragraphs",
        "6. PLAIN TEXT ONLY: Use standard alphanumeric text and punctuation."
    ]
    for d in directives:
        assert d in CANONICAL_SYSTEM_PROMPT, f"Directive missing: {d}"

def test_canonical_stop_tokens_list():
    assert "<|im_end|>" in DEFAULT_STOP_TOKENS
    assert "<|im_start|>" in DEFAULT_STOP_TOKENS
    assert "<|eot_id|>" in DEFAULT_STOP_TOKENS
    assert "<|end_of_text|>" in DEFAULT_STOP_TOKENS
    assert "</s>" in DEFAULT_STOP_TOKENS

def test_chatml_formatting_structure():
    messages = [
        ChatMessage(role="system", content="Custom system instructions."),
        ChatMessage(role="user", content="User prompt line 1."),
        ChatMessage(role="assistant", content="<think>\nReasoning\n</think>\nAssistant answer."),
        ChatMessage(role="user", content="Followup question."),
    ]
    prompt = build_chat_prompt(messages)

    assert "<|im_start|>system\nCustom system instructions.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nUser prompt line 1.<|im_end|>\n" in prompt
    assert "<|im_start|>assistant\n<think>\nReasoning\n</think>\nAssistant answer.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nFollowup question.<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

def test_prepare_chat_messages_presets():
    # 1. Clean passthrough for standard models when preset='default'
    raw_msgs = [{"role": "user", "content": "Hello standard AI"}]
    prep_default = prepare_chat_messages(raw_msgs, preset="default")
    assert len(prep_default) == 1
    assert prep_default[0]["role"] == "user"

    # 2. Injects unconstrained reasoning when preset='uncensored'
    prep_uncensored = prepare_chat_messages(raw_msgs, preset="uncensored")
    assert len(prep_uncensored) == 2
    assert prep_uncensored[0]["role"] == "system"
    assert "Deliberation format:" in prep_uncensored[0]["content"]

    # 3. Custom system prompt support
    prep_custom = prepare_chat_messages(raw_msgs, preset="default", custom_system_prompt="You are a math tutor.")
    assert len(prep_custom) == 2
    assert prep_custom[0]["content"] == "You are a math tutor."

    # 4. Preserves user-provided system message without overwriting
    user_sys_msgs = [
        {"role": "system", "content": "My custom rule."},
        {"role": "user", "content": "Hello"}
    ]
    prep_existing = prepare_chat_messages(user_sys_msgs, preset="uncensored")
    assert len(prep_existing) == 2
    assert prep_existing[0]["content"] == "My custom rule."
