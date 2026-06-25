"""
Backward-compatible re-export shim — new code should import from deft.dci.

This file exists only so that external code or cached bytecode still using
``from deft.conditioning import ...`` continues to work during the migration
window.  It will be removed in a future cleanup.
"""

from .dci import (  # noqa: F401
    FiLMLayer,
    FiLMWrapper,
    LoRALayer,
    LoRAWrapper,
    PromptFiLMLayer,
    PromptFiLMWrapper,
)
