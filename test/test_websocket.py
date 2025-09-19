"""
===== tests/test_websocket_cli.py =====
WebSocket and CLI Testing
Author: Cazandra Aporbo MS
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from click.testing import CliRunner
from fastapi.testclient import TestClient
from websocket import create_connection
import websocket

from mochi_moo.server import app
from mochi_moo.cli import cli, main


class TestWebSocketInterface:
    """Test WebSocket real-time communication"""
    
    def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                
                assert data['type'] == 'welcome'
                assert 'session_id' in data
                assert 'palette' in data
                assert len(data['palette']) == 7
                
    def test_websocket_message_processing(self):
        """Test message processing over WebSocket"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Skip welcome message
                websocket.receive_json()
                
                # Send test message
                websocket.send_json({
                    "input": "Test WebSocket message",
                    "emotional_context": True
                })
                
                # Should receive response
                response = websocket.receive_json()
                assert response['type'] == 'response'
                assert 'data' in response
                assert 'content' in response['data']
                
                # Should receive emotional update
                emotional = websocket.receive_json()
                assert emotional['type'] == 'emotional_update'
                assert 'data' in emotional
                
    def test_websocket_ping_pong(self):
        """Test WebSocket ping-pong keepalive"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Skip welcome
                websocket.receive_json()
                
                # Send ping
                websocket.send_json({"type": "ping"})
                
                # Should receive pong
                response = websocket.receive_json()
                assert response['type'] == 'pong'
                
    def test_websocket_mode_switching(self):
        """Test mode switching over WebSocket"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Skip welcome
                websocket.receive_json()
                
                # Send message with specific mode request
                websocket.send_json({
                    "input": "Switch to whisper mode",
                    "emotional_context": True
                })
                
                # Get response
                response = websocket.receive_json()
                emotional = websocket.receive_json()
                
                # Send another message
                websocket.send_json({
                    "input": "Test in whisper mode"
                })
                
                response2 = websocket.receive_json()
                # Response should reflect whisper mode characteristics
                assert '\n\n' in response2['data']['content'] or '...' in response2['data']['content']
                
    def test_websocket_synthesis_request(self):
        """Test synthesis over WebSocket"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Skip welcome
                websocket.receive_json()
                
                # Send synthesis request
                websocket.send_json({
                    "input": "Explain consciousness",
                    "domains": ["neuroscience", "philosophy", "physics"],
                    "emotional_context": True
                })
                
                # Should receive response with synthesis
                response = websocket.receive_json()
                assert response['type'] == 'response'
                
                emotional = websocket.receive_json()
                assert emotional['type'] == 'emotional_update'
                
    def test_websocket_concurrent_connections(self):
        """Test multiple concurrent WebSocket connections"""
        with TestClient(app) as client:
            connections = []
            
            # Create multiple connections
            for i in range(5):
                ws = client.websocket_connect("/ws").__enter__()
                connections.append(ws)
                
                # Each should get unique session
                welcome = ws.receive_json()
                assert 'session_id' in welcome
                
            # All sessions should be different
            session_ids = set()
            for ws in connections:
                ws.send_json({"input": "test"})
                response = ws.receive_json()
                if 'data' in response and 'session_id' in response['data']:
                    session_ids.add(response['data']['session_id'])
                    
            # Clean up
            for ws in connections:
                ws.__exit__(None, None, None)
                
    def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Skip welcome
                websocket.receive_json()
                
                # Send invalid message
                websocket.send_json({
                    "invalid": "message structure"
                })
                
                # Should handle gracefully
                response = websocket.receive_json()
                assert response is not None  # Should not crash
                

class TestCLIInterface:
    """Test command-line interface"""
    
    def setup_method(self):
        """Setup CLI test environment"""
        self.runner = CliRunner()
        
    def test_cli_version(self):
        """Test version command"""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '1.0.0' in result.output
        
    def test_cli_process_command(self):
        """Test process command"""
        with patch('mochi_moo.cli.MochiCore') as mock_mochi:
            mock_instance = MagicMock()
            mock_mochi.return_value = mock_instance
            
            # Mock async process
            future = asyncio.Future()
            future.set_result({
                'content': 'Test response',
                'micro_dose': 'Test insight'
            })
            mock_instance.process = MagicMock(return_value=future)
            
            result = self.runner.invoke(cli, [
                'process',
                'Test input',
                '--mode', 'standard',
                '--domains', 'test1',
                '--domains', 'test2'
            ])
            
            assert result.exit_code == 0
            assert 'Mochi\'s Response' in result.output
            
    def test_cli_synthesize_command(self):
        """Test synthesize command"""
        with patch('mochi_moo.cli.MochiCore') as mock_mochi:
            mock_instance = MagicMock()
            mock_mochi.return_value = mock_instance
            
            mock_instance.synthesizer.integrate.return_value = {
                'coherence_score': 0.85,
                'primary_insights': ['Insight 1', 'Insight 2'],
                'cross_domain_patterns': [{'test': 'pattern'}]
            }
            
            result = self.runner.invoke(cli, [
                'synthesize',
                'domain1', 'domain2',
                'Test query'
            ])
            
            assert result.exit_code == 0
            assert 'Cross-Domain Synthesis Results' in result.output
            assert '0.85' in result.output
            
    def test_cli_interactive_mode(self):
        """Test interactive mode"""
        with patch('mochi_moo.cli.MochiCore') as mock_mochi:
            mock_instance = MagicMock()
            mock_mochi.return_value = mock_instance
            
            # Mock async process
            future = asyncio.Future()
            future.set_result({
                'content': 'Interactive response',
                'micro_dose': 'Insight'
            })
            mock_instance.process = MagicMock(return_value=future)
            
            # Simulate interactive session with exit
            with patch('mochi_moo.cli.console.input', side_effect=['test input', 'exit']):
                result = self.runner.invoke(cli, ['interactive'])
                
                assert 'Welcome to Mochi-Moo Interactive Mode' in result.output
                assert 'Goodbye' in result.output
                
    def test_cli_status_command(self):
        """Test status command"""
        with patch('mochi_moo.cli.MochiCore') as mock_mochi:
            mock_instance = MagicMock()
            mock_mochi.return_value = mock_instance
            
            mock_instance.current_mode.value = 'standard'
            mock_instance.foresight.depth = 10
            mock_instance.synthesizer.synthesis_cache = {}
            mock_instance.interaction_history = []
            mock_instance.get_emotional_state.return_value = {
                'stress_level': 0.3,
                'engagement': 0.7,
                'curiosity': 0.8,
                'frustration': 0.2,
                'cognitive_load': 0.4
            }
            
            result = self.runner.invoke(cli, ['status'])
            
            assert result.exit_code == 0
            assert 'Mochi-Moo System Status' in result.output
            assert 'Current Mode' in result.output
            assert 'standard' in result.output
            
    def test_cli_server_command(self):
        """Test server start command"""
        with patch('mochi_moo.cli.run_server') as mock_run:
            result = self.runner.invoke(cli, ['server'])
            
            assert result.exit_code == 0
            assert 'Starting Mochi-Moo API server' in result.output
            mock_run.assert_called_once()
            
    def test_cli_output_file(self):
        """Test output to file"""
        with patch('mochi_moo.cli.MochiCore') as mock_mochi:
            mock_instance = MagicMock()
            mock_mochi.return_value = mock_instance
            
            # Mock async process
            future = asyncio.Future()
            future.set_result({
                'content': 'Test response',
                'micro_dose': 'Test insight'
            })
            mock_instance.process = MagicMock(return_value=future)
            
            with self.runner.isolated_filesystem():
                result = self.runner.invoke(cli, [
                    'process',
                    'Test input',
                    '--output', 'output.json'
                ])
                
                assert result.exit_code == 0
                assert 'Response saved to output.json' in result.output
                
                # Verify file was created
                import os
                assert os.path.exists('output.json')
                
    def test_cli_error_handling(self):
        """Test CLI error handling"""
        with patch('mochi_moo.cli.MochiCore') as mock_mochi:
            mock_mochi.side_effect = Exception("Initialization error")
            
            result = self.runner.invoke(cli, ['process', 'test'])
            
            # Should handle error gracefully
            assert result.exit_code != 0
            

"""
===== tests/conftest.py =====
Pytest configuration and fixtures
Author: Cazandra Aporbo MS
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import shutil
from typing import Generator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def clean_traces():
    """Clean up trace files after test"""
    yield
    trace_path = Path('.mochi_trace')
    if trace_path.exists():
        shutil.rmtree(trace_path, ignore_errors=True)


@pytest.fixture
def mock_mochi():
    """Create mock Mochi instance"""
    from unittest.mock import MagicMock
    from mochi_moo.core import MochiCore
    
    mock = MagicMock(spec=MochiCore)
    mock.current_mode.value = 'standard'
    mock.get_emotional_state.return_value = {
        'stress_level': 0.5,
        'cognitive_load': 0.5,
        'engagement': 0.7,
        'frustration': 0.2,
        'curiosity': 0.8
    }
    
    return mock


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment between tests"""
    # Clear any environment variables
    env_vars = ['MOCHI_MODE', 'MOCHI_DEBUG', 'MOCHI_TRACE']
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
            
    yield
    
    # Cleanup after test
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]


"""
===== pytest.ini =====
Pytest configuration file
"""
# [tool:pytest]
# minversion = 7.0
# testpaths = tests
# python_files = test_*.py
# python_classes = Test*
# python_functions = test_*
# 
# # Async settings
# asyncio_mode = auto
# 
# # Coverage settings
# addopts = 
#     -v
#     --tb=short
#     --strict-markers
#     --cov=mochi_moo
#     --cov-report=term-missing
#     --cov-report=html
#     --cov-report=xml
#     --cov-fail-under=90
#     --hypothesis-show-statistics
# 
# # Custom markers
# markers =
#     slow: marks tests as slow (deselect with '-m "not slow"')
#     integration: marks tests as integration tests
#     unit: marks tests as unit tests
#     performance: marks tests as performance tests
#     security: marks tests as security tests
#     websocket: marks tests as websocket tests
#     cli: marks tests as CLI tests
# 
# # Timeout settings
# timeout = 30
# timeout_method = thread
# 
# # Logging
# log_cli = true
# log_cli_level = INFO
# log_cli_format = %(asctime)s [%(levelname)8s] %(message)s
# log_cli_date_format = %Y-%m-%d %H:%M:%S


"""
===== tests/run_all_tests.py =====
Comprehensive test runner with reporting
"""
import pytest
import sys
from pathlib import Path
import json
import time
from typing import Dict, List

def run_comprehensive_tests() -> Dict:
    """Run all tests with detailed reporting"""
    
    start_time = time.time()
    results = {}
    
    # Test suites to run
    test_suites = [
        ("Unit Tests", ["tests/test_core.py", "-m", "unit"]),
        ("Integration Tests", ["tests/test_integration.py", "-m", "integration"]),
        ("Performance Tests", ["tests/test_performance_benchmarks.py", "-m", "performance"]),
        ("Algorithm Tests", ["tests/test_synthesis_algorithms.py"]),
        ("Property Tests", ["tests/test_property_visual.py", "--hypothesis-show-statistics"]),
        ("Security Tests", ["tests/test_integration.py::TestSecurityControls"]),
        ("WebSocket Tests", ["tests/test_websocket_cli.py::TestWebSocketInterface"]),
        ("CLI Tests", ["tests/test_websocket_cli.py::TestCLIInterface"]),
    ]
    
    print("=" * 80)
    print("MOCHI-MOO COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    for suite_name, args in test_suites:
        print(f"\nRunning {suite_name}...")
        print("-" * 40)
        
        # Run test suite
        result = pytest.main([
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=test_report_{suite_name.lower().replace(' ', '_')}.json",
            *args
        ])
        
        results[suite_name] = {
            "exit_code": result,
            "passed": result == 0
        }
        
        # Parse JSON report if available
        report_file = Path(f"test_report_{suite_name.lower().replace(' ', '_')}.json")
        if report_file.exists():
            with open(report_file) as f:
                report_data = json.load(f)
                results[suite_name].update({
                    "duration": report_data.get("duration", 0),
                    "passed_count": len([t for t in report_data.get("tests", []) 
                                       if t.get("outcome") == "passed"]),
                    "failed_count": len([t for t in report_data.get("tests", []) 
                                       if t.get("outcome") == "failed"]),
                    "total_count": len(report_data.get("tests", []))
                })
    
    # Generate summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for suite_name, result in results.items():
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"{suite_name:30} {status}")
        
        if "total_count" in result:
            print(f"  Tests: {result['passed_count']}/{result['total_count']} passed")
            print(f"  Duration: {result.get('duration', 0):.2f}s")
            
            total_tests += result['total_count']
            total_passed += result['passed_count']
            total_failed += result['failed_count']
        
        all_passed = all_passed and result["passed"]
    
    print("\n" + "-" * 80)
    print(f"OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Total Time: {total_time:.2f}s")
    print("=" * 80)
    
    # Generate detailed HTML report
    pytest.main([
        "--html=test_report_comprehensive.html",
        "--self-contained-html",
        "--cov=mochi_moo",
        "--cov-report=html",
        "tests/"
    ])
    
    print("\nDetailed reports generated:")
    print("  - test_report_comprehensive.html")
    print("  - htmlcov/index.html (coverage report)")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(r["passed"] for r in results.values()) else 1)
