# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist domains a proposal is asked in, and the domain a gap points at.

The cards are the body's catalogue entries for the domains that author throughput candidates; the research,
enablement and operator-defined domains answer with findings or PR leads rather than candidates, so a proposal
cannot be asked in them.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from hyperloom.brain_api import Gap

from .inputs import DomainCard

_CARDS = (
    DomainCard(
        key="serving_specialist",
        layer="sglang / vllm scheduler / cuda_graph / kv_cache",
        kb_anchor="framework",
        description=(
            "Reads sglang/vllm source, focuses on scheduler, cuda graph, "
            "kv cache, batching, chunked prefill, max-num-seqs."
        ),
    ),
    DomainCard(
        key="kernel_switch_specialist",
        layer="aiter / sglang kernels / triton",
        kb_anchor="kernel_agent",
        description=(
            "Reads aiter / sglang kernels / triton source; focuses on attention, MoE, GEMM, fused attention paths."
        ),
    ),
    DomainCard(
        key="comm_specialist",
        layer="RCCL / NCCL / QuickReduce / AllReduce",
        kb_anchor="communication",
        description="Focuses on collective communication, allreduce algorithms, QuickReduce, topology.",
    ),
    DomainCard(
        key="compiler_specialist",
        layer="torch.compile / inductor / triton",
        kb_anchor="compiler",
        description="Focuses on torch.compile, inductor, triton codegen, AMDGCN, register pressure.",
    ),
    DomainCard(
        key="system_specialist",
        layer="KFD / driver / memory / dispatch overhead",
        kb_anchor="systems",
        description=(
            "Fixes launch latency, dispatch overhead, device "
            "synchronization and host-blocking calls; tunes KFD/driver "
            "env vars, numactl, HSA_ENABLE_SDMA, memory fragmentation."
        ),
    ),
    DomainCard(
        key="speculative_decoding_specialist",
        layer="speculative decoding: MTP / EAGLE / block drafters / draft model / n-gram",
        kb_anchor="speculative_decoding",
        description=(
            "Decides whether speculative decoding (MTP, EAGLE/EAGLE-3, block "
            "drafters, draft model, n-gram/suffix) should be on and how, for "
            "memory-bound decode: small batch or long context, judged from "
            "profiling and roofline evidence."
        ),
    ),
)

#: Domain key -> the card a dispatch in it is briefed under.
DOMAINS: Mapping[str, DomainCard] = MappingProxyType({card.key: card for card in _CARDS})

#: The card of a ``scope='freeform'`` dispatch: the ask's task is the whole mandate.
FREEFORM = DomainCard(
    key="freeform_specialist",
    layer="(free-form — not bound to the domain catalogue)",
    kb_anchor="framework",
    description=(
        "Free-form specialist: not bound to the domain catalogue. The Orchestration task_description is the whole mandate."
    ),
)

_BY_ANCHOR: Mapping[str, str] = MappingProxyType({card.kb_anchor: card.key for card in _CARDS})

# A gap's layer (the body's gap ledger and roofline bottleneck directions) -> the domain that works on it.
_BY_LAYER: Mapping[str, str] = MappingProxyType(
    {
        "framework": "serving_specialist",
        "scheduler": "serving_specialist",
        "memory": "serving_specialist",
        "kernel": "kernel_switch_specialist",
        "compute": "kernel_switch_specialist",
        "comm": "comm_specialist",
        "communication": "comm_specialist",
        "compiler": "compiler_specialist",
        "system": "system_specialist",
        "systems": "system_specialist",
        "host": "system_specialist",
        "idle": "system_specialist",
    }
)


def gap_domain(gap: Gap | None) -> str:
    """The domain ``gap`` points at: its own domain when that names one (by key, then by KB anchor), else its
    layer's, else ``""``."""
    if gap is None:
        return ""
    named = gap.domain.strip()
    if named in DOMAINS:
        return named
    return _BY_ANCHOR.get(named) or _BY_LAYER.get(gap.layer.strip().lower(), "")
