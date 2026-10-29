# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain contract: the tools, events and snapshot the body and a brain exchange. Owned by the body.

It imports only the standard library, and every entity it defines survives a JSON round trip, so a brain can run in
another process and see nothing else of the body.
"""

from __future__ import annotations

from hyperloom.brain_api.brain import (
    CALL_SITE_METHODS,
    CONTRACT_VERSION,
    Brain,
    BrainContractError,
    LlmCallError,
    LlmHandle,
    LlmReply,
    Port,
    TurnFailed,
    validate_brain,
)
from hyperloom.brain_api.events import (
    TOPICS,
    VERDICTS,
    BrainTurn,
    CallSite,
    Event,
    Result,
    StateSnapshot,
    TickBatch,
    Verdict,
)
from hyperloom.brain_api.tools import (
    INTENT_TOOLS,
    QUERY_NAMES,
    TOOLS,
    Caller,
    ToolCall,
    ToolKind,
    ToolSpec,
    tool_for_action,
)
from hyperloom.brain_api.wire import WireError

__all__ = [
    "CALL_SITE_METHODS",
    "CONTRACT_VERSION",
    "INTENT_TOOLS",
    "QUERY_NAMES",
    "TOOLS",
    "TOPICS",
    "VERDICTS",
    "Brain",
    "BrainContractError",
    "BrainTurn",
    "CallSite",
    "Caller",
    "Event",
    "LlmCallError",
    "LlmHandle",
    "LlmReply",
    "Port",
    "Result",
    "StateSnapshot",
    "TickBatch",
    "ToolCall",
    "ToolKind",
    "ToolSpec",
    "TurnFailed",
    "Verdict",
    "WireError",
    "tool_for_action",
    "validate_brain",
]
