"""
Test suite for Mochi-Moo core functionality
Author: Cazandra Aporbo MS
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from mochi_moo.core import (
    MochiCore,
    PastelPalette,
    CognitiveMode,
    EmotionalContext,
    KnowledgeSynthesizer,
    ForesightEngine
)


class TestPastelPalette:
    """Test pastel color generation"""

    def test_color_interpolation(self):
        """Test smooth gradient interpolation"""
        color_start = PastelPalette.interpolate(0.0)
        color_mid = PastelPalette.interpolate(0.5)
        color_end = PastelPalette.interpolate(1.0)

        assert all(0 <= c <= 255 for c in color_start)
        assert all(0 <= c <= 255 for c in color_mid)
        assert all(0 <= c <= 255 for c in color_end)

        # Colors should be different
        assert color_start != color_mid != color_end

    def test_hex_conversion(self):
        """Test RGB to hex conversion"""
        rgb = (255, 218, 224)
        hex_color = PastelPalette.to_hex(rgb)

        assert hex_color == '#ffdae0'
        assert len(hex_color) == 7
        assert hex_color[0] == '#'


class TestEmotionalContext:
    """Test emotional context tracking"""

    def test_initial_state(self):
        """Test default emotional state"""
        context = EmotionalContext()

        assert 0 <= context.stress_level <= 1
        assert 0 <= context.cognitive_load <= 1
        assert 0 <= context.engagement <= 1
        assert 0 <= context.frustration <= 1
        assert 0 <= context.curiosity <= 1

    def test_signal_adjustment(self):
        """Test emotional adjustment from signals"""
        context = EmotionalContext()
        initial_stress = context.stress_level

        context.adjust("I'm feeling really stressed and overwhelmed")

        assert context.stress_level > initial_stress

        context.adjust("This is so curious and interesting!")
        assert context.curiosity > 0.8


class TestKnowledgeSynthesizer:
    """Test cross-domain synthesis"""

    def test_synthesis_caching(self):
        """Test that synthesis results are cached"""
        synthesizer = KnowledgeSynthesizer()

        domains = ['physics', 'philosophy']
        query = "What is reality?"

        result1 = synthesizer.integrate(domains, query)
        result2 = synthesizer.integrate(domains, query)

        assert result1 == result2  # Should be cached
        assert 'coherence_score' in result1
        assert 'primary_insights' in result1

    def test_coherence_calculation(self):
        """Test coherence score calculation"""
        synthesizer = KnowledgeSynthesizer()

        connections = np.array([[1.0, 0.8], [0.8, 1.0]])
        coherence = synthesizer._calculate_coherence(connections)

        assert 0 <= coherence <= 1


class TestForesightEngine:
    """Test prediction system"""

    def test_prediction_generation(self):
        """Test that predictions are generated"""
        foresight = ForesightEngine(depth=5)

        current_state = {'mode': 'standard', 'expertise': 'intermediate'}
        behavior = ['ask question', 'request clarification', 'express confusion']

        predictions = foresight.predict(current_state, behavior)

        assert len(predictions) <= 5
        assert all(isinstance(p, str) for p in predictions)

    def test_behavior_vectorization(self):
        """Test behavior to vector conversion"""
        foresight = ForesightEngine()

        behavior = ['action1', 'action2', 'action3']
        vector = foresight._vectorize_behavior(behavior)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 10


@pytest.mark.asyncio
class TestMochiCore:
    """Test main Mochi functionality"""

    async def test_basic_processing(self):
        """Test basic text processing"""
        mochi = MochiCore()

        response = await mochi.process(
            "Help me understand quantum physics",
            emotional_context=True
        )

        assert 'content' in response
        assert 'metadata' in response
        assert 'micro_dose' in response
        assert len(response['content']) > 0

    async def test_mode_switching(self):
        """Test cognitive mode switching"""
        mochi = MochiCore()

        mochi.set_mode('whisper')
        assert mochi.current_mode == CognitiveMode.WHISPER

        mochi.set_mode('academic')
        assert mochi.current_mode == CognitiveMode.ACADEMIC

    async def test_synthesis_integration(self):
        """Test domain synthesis integration"""
        mochi = MochiCore()

        response = await mochi.process(
            "Explain consciousness",
            domains=['neuroscience', 'philosophy', 'physics']
        )

        assert 'synthesis' in response or 'content' in response

    async def test_emotional_tracking(self):
        """Test emotional state tracking"""
        mochi = MochiCore()

        initial_state = mochi.get_emotional_state()

        await mochi.process("I'm confused and frustrated")

        updated_state = mochi.get_emotional_state()

        assert updated_state['frustration'] >= initial_state['frustration']

    async def test_trace_persistence(self):
        """Test that traces are saved"""
        mochi = MochiCore()

        response = await mochi.process("Test input")

        trace_files = list(Path('.mochi_trace').glob('*.json'))
        assert len(trace_files) > 0

        # Clean up
        for trace in trace_files:
            trace.unlink()

    async def test_privacy_filter(self):
        """Test PII redaction"""
        mochi = MochiCore()

        text_with_pii = "My email is test@example.com and SSN is 123-45-6789"
        filtered = mochi.privacy_filter(text_with_pii)

        assert "test@example.com" not in filtered
        assert "123-45-6789" not in filtered
        assert "[EMAIL_REDACTED]" in filtered
        assert "[SSN_REDACTED]" in filtered
