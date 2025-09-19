"""
Mochi-Moo Package Initialization
Author: Cazandra Aporbo MS
The pastel singularity begins here
"""

from mochi_moo.core import (
    MochiCore,
    PastelPalette,
    CognitiveMode,
    EmotionalContext,
    KnowledgeSynthesizer,
    ForesightEngine,
    MochiTrace,
    create_mochi,
    process_request,
)

__version__ = "1.0.0"
__author__ = "Cazandra Aporbo MS"
__email__ = "becaziam@gmail.com"
__description__ = (
    "A superintelligent assistant who dreams in matte rainbow and thinks in "
    "ten-dimensional pastel origami"
)

__all__ = [
    "MochiCore",
    "PastelPalette",
    "CognitiveMode",
    "EmotionalContext",
    "KnowledgeSynthesizer",
    "ForesightEngine",
    "MochiTrace",
    "create_mochi",
    "process_request",
]

# Initialize default configuration
DEFAULT_CONFIG = {
    "palette": "pastel_ombre",
    "foresight_depth": 10,
    "coherence_threshold": 0.75,
    "emotional_tracking": True,
    "privacy_mode": "strict",
    "trace_persistence": True,
    "whisper_threshold": 0.7,
    "session_timeout_minutes": 30,
}


def initialize() -> MochiCore:
    """
    Initialize Mochi-Moo with default settings.

    Returns:
        MochiCore: a configured instance ready for interaction.
    """
    mochi = create_mochi()

    # Warm up cognitive systems
    import asyncio

    async def warmup():
        await mochi.process(
            "initialization sequence",
            emotional_context=False,
        )

    asyncio.run(warmup())
    return mochi
