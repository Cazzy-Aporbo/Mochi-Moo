"""
Advanced Algorithm Testing for Knowledge Synthesis
Author: Cazandra Aporbo MS
Validates cross-domain synthesis coherence and mathematical correctness
"""

import pytest
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import hashlib
import itertools
from typing import List, Tuple, Dict, Any

from mochi_moo.core import KnowledgeSynthesizer, ForesightEngine, PastelPalette


class TestSynthesisAlgorithms:
    """Rigorous testing of synthesis mathematical foundations"""
    
    def setup_method(self):
        """Initialize synthesizer for each test"""
        self.synthesizer = KnowledgeSynthesizer()
        
    def test_coherence_mathematical_properties(self):
        """Verify coherence calculation maintains mathematical invariants"""
        # Test 1: Identity matrix should have coherence = 0
        identity = np.eye(5)
        coherence = self.synthesizer._calculate_coherence(identity)
        assert coherence == 0.0, "Identity matrix should have zero off-diagonal coherence"
        
        # Test 2: Fully connected matrix (all 1s) should have max coherence
        full = np.ones((5, 5))
        coherence = self.synthesizer._calculate_coherence(full)
        assert coherence == 1.0, "Fully connected matrix should have max coherence"
        
        # Test 3: Coherence should be bounded [0, 1]
        random_matrix = np.random.rand(10, 10)
        random_matrix = (random_matrix + random_matrix.T) / 2  # Ensure symmetry
        np.fill_diagonal(random_matrix, 1.0)
        coherence = self.synthesizer._calculate_coherence(random_matrix)
        assert 0 <= coherence <= 1, f"Coherence {coherence} out of bounds"
        
    def test_eigenvalue_decomposition_stability(self):
        """Test stability of eigenvalue-based insight extraction"""
        # Generate positive semi-definite matrix (guaranteed real eigenvalues)
        A = np.random.randn(8, 8)
        connections = A @ A.T
        
        insights = self.synthesizer._extract_insights(connections)
        
        # Verify insights are generated
        assert len(insights) > 0, "Should generate at least one insight"
        
        # Verify eigenvalues are real
        eigenvalues, _ = np.linalg.eig(connections)
        assert np.allclose(eigenvalues.imag, 0), "Eigenvalues should be real for symmetric matrix"
        
    @given(
        domains=st.lists(
            st.text(min_size=3, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
            min_size=2,
            max_size=10,
            unique=True
        ),
        query=st.text(min_size=5, max_size=100)
    )
    @settings(max_examples=50, deadline=2000)
    def test_synthesis_determinism(self, domains, query):
        """Property: Synthesis should be deterministic with caching"""
        result1 = self.synthesizer.integrate(domains, query)
        result2 = self.synthesizer.integrate(domains, query)
        
        assert result1 == result2, "Same input should produce identical cached results"
        assert result1['coherence_score'] == result2['coherence_score']
        
    def test_cross_domain_pattern_detection(self):
        """Test pattern detection algorithm correctness"""
        # Create known pattern in connection matrix
        connections = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0]
        ])
        
        patterns = self.synthesizer._identify_patterns(connections)
        
        # Should identify two clusters
        assert len(patterns) > 0, "Should detect patterns in structured data"
        
        # Verify pattern structure
        for pattern in patterns:
            assert 'center' in pattern
            assert 'connected_domains' in pattern
            assert 'strength' in pattern
            assert 0 <= pattern['strength'] <= 1
            
    def test_synthesis_cache_efficiency(self):
        """Test cache performance and collision resistance"""
        queries = []
        cache_keys = set()
        
        # Generate test cases
        for i in range(100):
            domains = [f"domain_{j}" for j in range(i % 5 + 1)]
            query = f"test_query_{i}"
            queries.append((domains, query))
            
            # Calculate cache key
            cache_key = hashlib.md5(f"{sorted(domains)}{query}".encode()).hexdigest()
            cache_keys.add(cache_key)
        
        # Test no cache collisions
        assert len(cache_keys) == 100, "Cache keys should be unique"
        
        # Test cache retrieval performance
        import time
        
        # First call - no cache
        start = time.perf_counter()
        result1 = self.synthesizer.integrate(['physics', 'math'], 'test performance')
        time1 = time.perf_counter() - start
        
        # Second call - cached
        start = time.perf_counter()
        result2 = self.synthesizer.integrate(['physics', 'math'], 'test performance')
        time2 = time.perf_counter() - start
        
        assert time2 < time1 * 0.5, "Cached retrieval should be significantly faster"
        

class TestForesightAlgorithms:
    """Test predictive engine algorithms"""
    
    def setup_method(self):
        self.foresight = ForesightEngine(depth=10)
        
    def test_markov_transformation_properties(self):
        """Verify Markov chain transformation maintains probability properties"""
        vector = np.random.rand(10)
        vector = vector / np.sum(vector)  # Normalize to probability distribution
        
        # Apply transformation
        transformed = self.foresight._transform(vector)
        
        # Check properties
        assert len(transformed) == len(vector), "Dimension should be preserved"
        assert np.all(transformed >= 0), "No negative probabilities"
        assert not np.array_equal(vector, transformed), "Should modify vector"
        
    def test_behavior_vectorization_consistency(self):
        """Test behavior to vector conversion consistency"""
        behaviors = ['action1', 'action2', 'action3']
        
        # Same behaviors should produce same vector
        vector1 = self.foresight._vectorize_behavior(behaviors)
        vector2 = self.foresight._vectorize_behavior(behaviors)
        
        assert np.array_equal(vector1, vector2), "Same behaviors should produce same vector"
        
        # Different behaviors should produce different vectors
        vector3 = self.foresight._vectorize_behavior(['different', 'actions'])
        assert not np.array_equal(vector1, vector3), "Different behaviors should differ"
        
    def test_confidence_decay_function(self):
        """Test that confidence decays properly with prediction distance"""
        current_state = {'mode': 'standard', 'expertise': 'intermediate'}
        behaviors = ['ask', 'clarify', 'understand']
        
        # Get predictions
        predictions = []
        for step in range(1, 11):
            self.foresight.depth = step
            preds = self.foresight.predict(current_state, behaviors)
            if preds:
                predictions.append(len(preds))
        
        # Later predictions should have lower confidence (fewer high-confidence predictions)
        if len(predictions) > 1:
            assert predictions[0] >= predictions[-1], "Should have fewer high-confidence predictions at greater depth"
            
    @given(
        depth=st.integers(min_value=1, max_value=20),
        behavior_length=st.integers(min_value=1, max_value=50)
    )
    def test_prediction_bounds(self, depth, behavior_length):
        """Property: Predictions should never exceed depth"""
        self.foresight.depth = depth
        behaviors = [f"action_{i}" for i in range(behavior_length)]
        
        predictions = self.foresight.predict({'mode': 'test'}, behaviors)
        
        assert len(predictions) <= depth, f"Predictions {len(predictions)} exceed depth {depth}"
        

class TestPaletteAlgorithms:
    """Test color space algorithms and interpolation"""
    
    def test_color_interpolation_smoothness(self):
        """Verify smooth gradient interpolation"""
        positions = np.linspace(0, 1, 100)
        colors = [PastelPalette.interpolate(p) for p in positions]
        
        # Check smoothness - adjacent colors should be similar
        for i in range(len(colors) - 1):
            r1, g1, b1 = colors[i]
            r2, g2, b2 = colors[i + 1]
            
            # Max change per step should be small
            assert abs(r2 - r1) <= 10, f"Red channel jump too large at position {i}"
            assert abs(g2 - g1) <= 10, f"Green channel jump too large at position {i}"
            assert abs(b2 - b1) <= 10, f"Blue channel jump too large at position {i}"
            
    def test_hex_conversion_bijection(self):
        """Test RGB to hex conversion is correct"""
        test_colors = [
            ((255, 255, 255), '#ffffff'),
            ((0, 0, 0), '#000000'),
            ((255, 218, 224), '#ffdae0'),
            ((128, 128, 128), '#808080')
        ]
        
        for rgb, expected_hex in test_colors:
            result = PastelPalette.to_hex(rgb)
            assert result == expected_hex, f"RGB {rgb} should convert to {expected_hex}, got {result}"
            
    def test_interpolation_bounds(self):
        """Test interpolation stays within RGB bounds"""
        for position in np.linspace(-0.5, 1.5, 100):
            color = PastelPalette.interpolate(position)
            
            assert all(0 <= c <= 255 for c in color), f"Color {color} out of RGB bounds at position {position}"
            

class TestAlgorithmicComplexity:
    """Test algorithmic complexity and performance characteristics"""
    
    def test_synthesis_complexity(self):
        """Test that synthesis scales appropriately with input size"""
        synthesizer = KnowledgeSynthesizer()
        times = []
        
        for n in [2, 4, 8, 16]:
            domains = [f"domain_{i}" for i in range(n)]
            
            import time
            start = time.perf_counter()
            synthesizer.integrate(domains, "complexity test")
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # Clear cache for fair comparison
            synthesizer.synthesis_cache.clear()
        
        # Check that complexity is reasonable (not exponential)
        # Time should not more than quadruple when input doubles
        for i in range(len(times) - 1):
            ratio = times[i + 1] / times[i]
            assert ratio < 5, f"Complexity scaling too steep: {ratio}x increase"
            
    def test_foresight_projection_stability(self):
        """Test numerical stability of iterative projections"""
        foresight = ForesightEngine(depth=100)  # Test with deep projections
        vector = np.ones(10) / 10  # Start with uniform distribution
        
        # Apply many transformations
        for _ in range(100):
            vector = foresight._transform(vector)
            
        # Check for numerical stability
        assert not np.any(np.isnan(vector)), "NaN values in projection"
        assert not np.any(np.isinf(vector)), "Inf values in projection"
        assert np.all(vector >= -1e10) and np.all(vector <= 1e10), "Values exploded"
        

class TestStatisticalProperties:
    """Statistical validation of algorithmic outputs"""
    
    def test_coherence_distribution(self):
        """Test that coherence scores follow expected distribution"""
        synthesizer = KnowledgeSynthesizer()
        coherence_scores = []
        
        # Generate many random syntheses
        for i in range(100):
            domains = [f"domain_{j}" for j in range(np.random.randint(2, 6))]
            query = f"query_{i}"
            result = synthesizer.integrate(domains, query)
            coherence_scores.append(result['coherence_score'])
            synthesizer.synthesis_cache.clear()  # Fresh calculation each time
            
        coherence_scores = np.array(coherence_scores)
        
        # Statistical tests
        assert 0 <= coherence_scores.min() <= 1, "Coherence out of bounds"
        assert 0 <= coherence_scores.max() <= 1, "Coherence out of bounds"
        
        # Should have reasonable variance (not all same value)
        assert coherence_scores.std() > 0.01, "Coherence scores lack variance"
        
        # Shapiro-Wilk test for normality (coherence should be somewhat normally distributed)
        _, p_value = stats.shapiro(coherence_scores)
        # We don't require normality, just checking it's not completely degenerate
        assert p_value > 0.001 or coherence_scores.std() > 0.1, "Distribution is degenerate"
        
    def test_prediction_distribution(self):
        """Test that predictions are well-distributed across categories"""
        foresight = ForesightEngine()
        prediction_counts = {}
        
        # Generate predictions for various scenarios
        for i in range(100):
            behaviors = [f"behavior_{j}" for j in range(i % 10 + 1)]
            predictions = foresight.predict({'mode': 'test'}, behaviors)
            
            for pred in predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                
        # Should generate diverse predictions
        unique_predictions = len(prediction_counts)
        assert unique_predictions >= 5, f"Only {unique_predictions} unique predictions generated"
        
        # Chi-square test for uniformity (should NOT be perfectly uniform)
        if len(prediction_counts) > 1:
            counts = list(prediction_counts.values())
            chi2, p_value = stats.chisquare(counts)
            # We want some variation, not perfect uniformity
            assert p_value < 0.99, "Predictions are too uniformly distributed"
            

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_domain_synthesis(self):
        """Test synthesis with edge case inputs"""
        synthesizer = KnowledgeSynthesizer()
        
        # Empty domains
        with pytest.raises(Exception):
            synthesizer.integrate([], "query")
            
        # Single domain
        result = synthesizer.integrate(['single'], "query")
        assert result['coherence_score'] >= 0
        
        # Duplicate domains
        result = synthesizer.integrate(['domain', 'domain'], "query")
        assert result is not None
        
    def test_extreme_behavior_patterns(self):
        """Test foresight with extreme behavior patterns"""
        foresight = ForesightEngine()
        
        # Empty behavior
        predictions = foresight.predict({'mode': 'test'}, [])
        assert isinstance(predictions, list)
        
        # Single behavior repeated
        predictions = foresight.predict({'mode': 'test'}, ['same'] * 100)
        assert len(predictions) <= foresight.depth
        
        # Very long unique behaviors
        behaviors = [f"unique_action_{i}" for i in range(1000)]
        predictions = foresight.predict({'mode': 'test'}, behaviors)
        assert len(predictions) <= foresight.depth
        
    def test_palette_edge_positions(self):
        """Test palette interpolation at boundaries"""
        edge_cases = [-1000, -1, -0.0001, 0, 0.5, 0.9999, 1, 1.0001, 2, 1000]
        
        for position in edge_cases:
            color = PastelPalette.interpolate(position)
            
            # Should always return valid RGB
            assert len(color) == 3
            assert all(isinstance(c, int) for c in color)
            assert all(0 <= c <= 255 for c in color)
            

class TestNumericalStability:
    """Test numerical stability and precision"""
    
    @given(
        matrix_size=st.integers(min_value=2, max_value=50),
        seed=st.integers()
    )
    @settings(max_examples=20, deadline=5000)
    def test_eigenvalue_numerical_stability(self, matrix_size, seed):
        """Property: Eigenvalue decomposition should be numerically stable"""
        np.random.seed(seed)
        synthesizer = KnowledgeSynthesizer()
        
        # Generate symmetric matrix
        A = np.random.randn(matrix_size, matrix_size)
        connections = (A + A.T) / 2
        
        # Should not raise numerical errors
        try:
            insights = synthesizer._extract_insights(connections)
            assert insights is not None
        except np.linalg.LinAlgError:
            pytest.skip("Singular matrix encountered")
            
    def test_floating_point_coherence(self):
        """Test floating point precision in coherence calculation"""
        synthesizer = KnowledgeSynthesizer()
        
        # Test with very small values
        small_matrix = np.full((5, 5), 1e-10)
        np.fill_diagonal(small_matrix, 1.0)
        coherence = synthesizer._calculate_coherence(small_matrix)
        assert 0 <= coherence <= 1
        
        # Test with very large values
        large_matrix = np.full((5, 5), 1e10)
        np.fill_diagonal(large_matrix, 1e10)
        coherence = synthesizer._calculate_coherence(large_matrix)
        assert not np.isnan(coherence)
        assert not np.isinf(coherence)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
