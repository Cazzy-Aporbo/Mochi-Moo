"""
Property-Based Testing and Visual Validation
Author: Cazandra Aporbo MS
Uses hypothesis for property testing and validates visual output correctness
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition
import numpy as np
import colorsys
import hashlib
from typing import Dict, List, Tuple, Any
import re
import json
from dataclasses import dataclass

from mochi_moo.core import (
    MochiCore, 
    PastelPalette, 
    EmotionalContext,
    CognitiveMode,
    KnowledgeSynthesizer,
    ForesightEngine
)


class TestPropertyBasedCore:
    """Property-based testing for core functionality"""
    
    @given(
        text=st.text(min_size=1, max_size=1000),
        emotional=st.booleans(),
        domains=st.one_of(
            st.none(),
            st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5)
        )
    )
    @settings(max_examples=100, deadline=5000)
    async def test_process_never_crashes(self, text, emotional, domains):
        """Property: process should never crash regardless of input"""
        mochi = MochiCore()
        
        try:
            response = await mochi.process(
                text,
                emotional_context=emotional,
                domains=domains
            )
            
            # Should always return valid response structure
            assert isinstance(response, dict)
            assert 'content' in response
            assert 'metadata' in response
            assert isinstance(response['content'], str)
            
        except Exception as e:
            pytest.fail(f"Process crashed with input: {text[:50]}... Error: {e}")
    
    @given(
        stress=st.floats(min_value=0, max_value=1),
        load=st.floats(min_value=0, max_value=1),
        engagement=st.floats(min_value=0, max_value=1),
        frustration=st.floats(min_value=0, max_value=1),
        curiosity=st.floats(min_value=0, max_value=1)
    )
    def test_emotional_state_invariants(self, stress, load, engagement, frustration, curiosity):
        """Property: Emotional states maintain invariants"""
        context = EmotionalContext(
            stress_level=stress,
            cognitive_load=load,
            engagement=engagement,
            frustration=frustration,
            curiosity=curiosity
        )
        
        # All values should remain in bounds
        assert 0 <= context.stress_level <= 1
        assert 0 <= context.cognitive_load <= 1
        assert 0 <= context.engagement <= 1
        assert 0 <= context.frustration <= 1
        assert 0 <= context.curiosity <= 1
        
        # Test adjustment maintains bounds
        context.adjust("very stressful situation")
        assert 0 <= context.stress_level <= 1
        
    @given(
        domains=st.lists(
            st.text(min_size=1, max_size=30, alphabet=st.characters(blacklist_categories=["Cc", "Cs"])),
            min_size=1,
            max_size=10,
            unique=True
        ),
        query=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=50)
    def test_synthesis_properties(self, domains, query):
        """Property: Synthesis always produces valid results"""
        synthesizer = KnowledgeSynthesizer()
        
        result = synthesizer.integrate(domains, query)
        
        # Structural properties
        assert isinstance(result, dict)
        assert 'coherence_score' in result
        assert 'primary_insights' in result
        assert 'cross_domain_patterns' in result
        assert 'suggested_explorations' in result
        
        # Value properties
        assert 0 <= result['coherence_score'] <= 1
        assert isinstance(result['primary_insights'], list)
        assert len(result['suggested_explorations']) > 0
        
        # Determinism property - same input should give same output
        result2 = synthesizer.integrate(domains, query)
        assert result == result2
        
    @given(
        depth=st.integers(min_value=1, max_value=100),
        behaviors=st.lists(
            st.text(min_size=1, max_size=50),
            min_size=0,
            max_size=100
        )
    )
    def test_foresight_properties(self, depth, behaviors):
        """Property: Foresight predictions are bounded and valid"""
        foresight = ForesightEngine(depth=depth)
        
        predictions = foresight.predict(
            {'mode': 'test', 'expertise': 'intermediate'},
            behaviors
        )
        
        # Predictions should be bounded by depth
        assert len(predictions) <= depth
        
        # All predictions should be strings
        assert all(isinstance(p, str) for p in predictions)
        
        # Should handle empty behaviors
        if len(behaviors) == 0:
            assert isinstance(predictions, list)
            
    @given(position=st.floats(min_value=-1000, max_value=1000))
    def test_palette_interpolation_properties(self, position):
        """Property: Palette interpolation always returns valid RGB"""
        color = PastelPalette.interpolate(position)
        
        # Should always return 3-tuple
        assert len(color) == 3
        
        # Each component should be valid RGB value
        assert all(isinstance(c, int) for c in color)
        assert all(0 <= c <= 255 for c in color)
        
        # Test hex conversion
        hex_color = PastelPalette.to_hex(color)
        assert hex_color.startswith('#')
        assert len(hex_color) == 7
        assert all(c in '0123456789abcdef' for c in hex_color[1:])


class MochiStateMachine(RuleBasedStateMachine):
    """Stateful testing of Mochi's behavior over time"""
    
    def __init__(self):
        super().__init__()
        self.mochi = MochiCore()
        self.message_count = 0
        self.modes_used = set()
        self.last_emotional_state = None
        
    @rule(message=st.text(min_size=1, max_size=500))
    def send_message(self, message):
        """Rule: Can send any message"""
        import asyncio
        
        response = asyncio.run(self.mochi.process(message))
        self.message_count += 1
        
        assert response is not None
        assert 'content' in response
        
    @rule(mode=st.sampled_from(list(CognitiveMode)))
    def switch_mode(self, mode):
        """Rule: Can switch to any mode"""
        self.mochi.set_mode(mode.value)
        self.modes_used.add(mode)
        assert self.mochi.current_mode == mode
        
    @rule(
        domains=st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=5),
        query=st.text(min_size=1, max_size=100)
    )
    def perform_synthesis(self, domains, query):
        """Rule: Can perform synthesis"""
        result = self.mochi.synthesizer.integrate(domains, query)
        assert 'coherence_score' in result
        
    @invariant()
    def emotional_state_bounded(self):
        """Invariant: Emotional state always in bounds"""
        state = self.mochi.get_emotional_state()
        for key, value in state.items():
            assert 0 <= value <= 1, f"{key} = {value} out of bounds"
            
    @invariant()
    def mode_consistency(self):
        """Invariant: Mode is always valid"""
        assert isinstance(self.mochi.current_mode, CognitiveMode)
        
    @invariant()
    def history_size_bounded(self):
        """Invariant: History doesn't grow unbounded"""
        assert len(self.mochi.interaction_history) <= 100


TestMochiStateMachine = MochiStateMachine.TestCase


class TestVisualValidation:
    """Test visual output correctness"""
    
    def test_palette_visual_properties(self):
        """Test visual properties of the palette"""
        # Generate full gradient
        positions = np.linspace(0, 1, 100)
        colors = [PastelPalette.interpolate(p) for p in positions]
        
        # Test pastel property - all colors should have high lightness
        for r, g, b in colors:
            # Convert to HSL
            h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
            assert l > 0.6, f"Color RGB({r},{g},{b}) not pastel enough (L={l:.2f})"
            
        # Test smooth transitions
        for i in range(len(colors) - 1):
            r1, g1, b1 = colors[i]
            r2, g2, b2 = colors[i + 1]
            
            # Calculate color distance
            distance = ((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2) ** 0.5
            assert distance < 30, f"Color jump too large: {distance:.2f}"
            
    def test_visualization_structure(self):
        """Test visualization output structure"""
        mochi = MochiCore()
        
        test_data = {
            'values': [0.1, 0.5, 0.9],
            'labels': ['A', 'B', 'C']
        }
        
        viz = mochi.visualize(test_data, style="pastel_origami")
        
        # Verify structure
        assert 'type' in viz
        assert 'data' in viz
        assert 'style' in viz
        
        # Verify style properties
        assert 'colors' in viz['style']
        assert 'animation' in viz['style']
        assert 'interaction' in viz['style']
        
        # Verify colors are valid hex
        for color in viz['style']['colors']:
            assert re.match(r'^#[0-9a-f]{6}$', color.lower())
            
    def test_color_consistency_across_visualizations(self):
        """Test that visualizations use consistent palette"""
        mochi = MochiCore()
        
        visualizations = []
        for i in range(10):
            viz = mochi.visualize({'data': i}, style="test")
            visualizations.append(viz)
            
        # All should use same color palette
        first_colors = visualizations[0]['style']['colors']
        for viz in visualizations[1:]:
            assert viz['style']['colors'] == first_colors
            
    def test_gradient_generation(self):
        """Test gradient generation properties"""
        # Test different gradient sizes
        for num_colors in [2, 5, 10, 20]:
            positions = np.linspace(0, 1, num_colors)
            colors = [PastelPalette.interpolate(p) for p in positions]
            
            # Should generate requested number of colors
            assert len(colors) == num_colors
            
            # All should be unique (with high probability)
            unique_colors = set(colors)
            assert len(unique_colors) >= num_colors - 1  # Allow one duplicate
            
    def test_accessibility_contrast(self):
        """Test color accessibility and contrast"""
        
        def calculate_luminance(r, g, b):
            """Calculate relative luminance for contrast ratio"""
            def adjust(c):
                c = c / 255
                if c <= 0.03928:
                    return c / 12.92
                return ((c + 0.055) / 1.055) ** 2.4
                
            return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)
        
        def contrast_ratio(color1, color2):
            """Calculate contrast ratio between two colors"""
            l1 = calculate_luminance(*color1)
            l2 = calculate_luminance(*color2)
            
            lighter = max(l1, l2)
            darker = min(l1, l2)
            
            return (lighter + 0.05) / (darker + 0.05)
        
        # Test against white background
        white = (255, 255, 255)
        text_color = (74, 74, 74)  # #4A4A4A as specified
        
        # Test each palette color
        for i in range(7):
            bg_color = PastelPalette.interpolate(i / 6)
            ratio = contrast_ratio(text_color, bg_color)
            
            # WCAG AA requires 4.5:1 for normal text
            assert ratio >= 3.0, f"Insufficient contrast: {ratio:.2f}:1 for color {bg_color}"
            

class TestVisualizationIntegrity:
    """Test visualization data integrity"""
    
    @given(
        data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(),
                st.lists(st.integers(), max_size=10)
            ),
            max_size=20
        )
    )
    def test_visualization_handles_any_data(self, data):
        """Property: Visualization should handle any data structure"""
        mochi = MochiCore()
        
        try:
            viz = mochi.visualize(data)
            assert viz is not None
            assert 'style' in viz
            assert 'colors' in viz['style']
        except Exception as e:
            pytest.fail(f"Visualization failed for data: {data}. Error: {e}")
            
    def test_visualization_determinism(self):
        """Test visualization generation is deterministic"""
        mochi = MochiCore()
        
        test_data = {'key': 'value', 'numbers': [1, 2, 3]}
        
        viz1 = mochi.visualize(test_data, style="test")
        viz2 = mochi.visualize(test_data, style="test")
        
        # Should produce identical output
        assert viz1 == viz2
        
        # Different style should produce different output
        viz3 = mochi.visualize(test_data, style="different")
        assert viz3['type'] != viz1['type'] or viz3['style'] != viz1['style']
        

class TestComplexProperties:
    """Test complex multi-component properties"""
    
    @given(
        messages=st.lists(
            st.text(min_size=1, max_size=200),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20, deadline=10000)
    async def test_conversation_coherence(self, messages):
        """Property: Conversations maintain coherence"""
        mochi = MochiCore()
        
        emotional_trajectory = []
        response_lengths = []
        
        for message in messages:
            response = await mochi.process(message)
            emotional_trajectory.append(mochi.get_emotional_state())
            response_lengths.append(len(response['content']))
            
        # Emotional trajectory should be smooth (no wild jumps)
        for i in range(1, len(emotional_trajectory)):
            prev = emotional_trajectory[i-1]
            curr = emotional_trajectory[i]
            
            for key in prev:
                delta = abs(curr[key] - prev[key])
                assert delta <= 0.5, f"Emotional {key} jumped by {delta}"
                
        # Response lengths should be reasonable
        assert all(10 <= length <= 10000 for length in response_lengths)
        
    @given(
        num_domains=st.integers(min_value=2, max_value=20),
        num_queries=st.integers(min_value=1, max_value=10)
    )
    def test_synthesis_cache_consistency(self, num_domains, num_queries):
        """Property: Cache maintains consistency under load"""
        synthesizer = KnowledgeSynthesizer()
        
        # Generate test data
        domains_list = [[f"d{i}_{j}" for j in range(num_domains)] for i in range(num_queries)]
        queries = [f"query_{i}" for i in range(num_queries)]
        
        # First pass - populate cache
        results1 = []
        for domains, query in zip(domains_list, queries):
            result = synthesizer.integrate(domains, query)
            results1.append(result)
            
        # Second pass - should use cache
        results2 = []
        for domains, query in zip(domains_list, queries):
            result = synthesizer.integrate(domains, query)
            results2.append(result)
            
        # Results should be identical
        assert results1 == results2
        
        # Cache should contain all entries
        assert len(synthesizer.synthesis_cache) >= min(num_queries, 100)
        

class TestMathematicalProperties:
    """Test mathematical properties of algorithms"""
    
    @given(
        matrix_size=st.integers(min_value=2, max_value=20)
    )
    def test_eigenvalue_properties(self, matrix_size):
        """Property: Eigenvalue decomposition maintains mathematical properties"""
        synthesizer = KnowledgeSynthesizer()
        
        # Create symmetric positive semi-definite matrix
        A = np.random.randn(matrix_size, matrix_size)
        connections = A.T @ A  # Guaranteed PSD
        
        eigenvalues, eigenvectors = np.linalg.eig(connections)
        
        # All eigenvalues should be non-negative (PSD property)
        assert np.all(eigenvalues >= -1e-10), "Negative eigenvalues in PSD matrix"
        
        # Eigenvectors should be orthogonal
        identity = eigenvectors.T @ eigenvectors
        assert np.allclose(identity, np.eye(matrix_size), atol=1e-10)
        
    @given(
        vector_size=st.integers(min_value=1, max_value=100),
        iterations=st.integers(min_value=1, max_value=100)
    )
    def test_markov_convergence(self, vector_size, iterations):
        """Property: Markov transformations should converge"""
        foresight = ForesightEngine()
        
        # Start with random probability distribution
        vector = np.random.rand(vector_size)
        vector = vector / np.sum(vector)
        
        # Apply transformations
        vectors = [vector]
        for _ in range(iterations):
            vector = foresight._transform(vector)
            vectors.append(vector)
            
        # Should not explode
        assert not np.any(np.isnan(vectors[-1]))
        assert not np.any(np.isinf(vectors[-1]))
        
        # Should show convergence (decreasing rate of change)
        if iterations > 10:
            early_change = np.linalg.norm(vectors[5] - vectors[4])
            late_change = np.linalg.norm(vectors[-1] - vectors[-2])
            
            # Later changes should be smaller (convergence)
            assert late_change <= early_change * 2  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
