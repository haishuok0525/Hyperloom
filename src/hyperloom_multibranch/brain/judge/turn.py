# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One judge round of Brain v0: the observation rendered as one user prompt, one model call, its reply read.

The user prompt is the session state as indented JSON, then the digest of similar past sessions, the open areas and
the playbook where the session has them, then :data:`~hyperloom_multibranch.brain.judge.prompt.REPLY_FORMAT`. A failed
call and an unreadable reply give no answer, and the exchange says why.
"""

from __future__ import annotations

import json

from hyperloom_multibranch.brain.judge.prompt import REPLY_FORMAT, JudgePrompts
from hyperloom_multibranch.brain.judge.reply import parse_answer
from hyperloom_multibranch.contracts.brain import Exchange, JudgeObservation, JudgeRound
from hyperloom_multibranch.contracts.llm import Completion, CompletionFailed

#: Why an exchange whose reply holds no answer gave none.
NO_ANSWER = "the reply holds no well-formed decision object"


def user_prompt(observation: JudgeObservation) -> str:
    """The user prompt for ``observation``."""

    state = json.dumps(dict(observation.state), indent=2, sort_keys=True)
    history = f"How similar past sessions went:\n{observation.history}\n\n" if observation.history else ""
    sections = "".join(f"{text}\n\n" for text in (observation.open_areas, observation.playbook) if text)
    return f"Session state:\n{state}\n\n{history}{sections}{REPLY_FORMAT}"


async def judge_round(observation: JudgeObservation, llm: Completion, prompts: JudgePrompts) -> JudgeRound:
    """Ask ``llm`` once with the system prompt of ``prompts`` the observation's session plays under."""

    prompt = user_prompt(observation)
    try:
        reply = (await llm(prompts.system(plays_playbook=observation.plays_playbook), prompt)).text
    except CompletionFailed as exc:
        return JudgeRound(None, (Exchange(prompt, "", str(exc)),))
    answer = parse_answer(reply)
    return JudgeRound(answer, (Exchange(prompt, reply, "" if answer is not None else NO_ANSWER),))


__all__ = ["NO_ANSWER", "judge_round", "user_prompt"]
