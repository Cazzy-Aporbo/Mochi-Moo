"""
Performance Benchmarking and Stress Testing
Author: Cazandra Aporbo MS
Validates system performance under various load conditions
"""

import pytest
import asyncio
import time
import memory_profiler
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
import psutil
import gc
import statistics

from mochi_moo.core import MochiCore, KnowledgeSynthesizer, ForesightEngine


class TestPerformanceBenchmarks:
    """Comprehensive performance testing suite"""
    
    @classmethod
    def setup_class(cls):
        """Setup performance monitoring"""
        cls.process = psutil.Process()
        cls.baseline_memory = cls.process.memory_info().rss / 1024 / 1024  # MB
        
    def teardown_method(self):
        """Force garbage collection between tests"""
        gc.collect()
        
    @pytest.mark.asyncio
    async def test_response_latency_distribution(self):
        """Test response time distribution meets SLA requirements"""
        mochi = MochiCore()
        latencies = []
        
        test_inputs = [
            "Simple query",
            "Complex philosophical question about the nature of consciousness and reality",
            "Technical question requiring synthesis",
            "Emotional support request",
            "Creative writing prompt"
        ] * 20  # 100 total requests
        
        for input_text in test_inputs:
            start = time.perf_counter()
            await mochi.process(input_text, emotional_context=True)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        # Performance requirements
        assert p50 < 100, f"P50 latency {p50:.2f}ms exceeds 100ms target"
        assert p95 < 200, f"P95 latency {p95:.2f}ms exceeds 200ms target"
        assert p99 < 500, f"P99 latency {p99:.2f}ms exceeds 500ms target"
        
        # Log performance metrics
        print(f"\nLatency Percentiles: P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")
        print(f"Mean={statistics.mean(latencies):.2f}ms, StdDev={statistics.stdev(latencies):.2f}ms")
        
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test system under concurrent load"""
        mochi = MochiCore()
        concurrent_requests = 50
        
        async def make_request(idx: int) -> float:
            start = time.perf_counter()
            await mochi.process(f"Concurrent request {idx}")
            return time.perf_counter() - start
            
        # Launch concurrent requests
        tasks = [make_request(i) for i in range(concurrent_requests)]
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Performance assertions
        avg_time = statistics.mean(results)
        max_time = max(results)
        
        assert avg_time < 0.5, f"Average request time {avg_time:.2f}s too high under load"
        assert max_time < 2.0, f"Max request time {max_time:.2f}s indicates thread starvation"
        assert total_time < concurrent_requests * 0.1, f"Total time {total_time:.2f}s indicates poor concurrency"
        
        throughput = concurrent_requests / total_time
        print(f"\nConcurrency Test: {throughput:.2f} requests/second")
        
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""
        mochi = MochiCore()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Perform many operations
        for i in range(1000):
            mochi.synthesizer.integrate(['domain1', 'domain2'], f"query_{i}")
            if i % 100 == 0:
                gc.collect()  # Force garbage collection periodically
                
        # Clear caches
        mochi.synthesizer.synthesis_cache.clear()
        gc.collect()
        
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory: Initial={initial_memory:.2f}MB, Final={final_memory:.2f}MB, Growth={memory_growth:.2f}MB")
        
        # Allow for some growth but not unbounded
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB, indicating potential leak"
        
    def test_cache_performance_impact(self):
        """Measure cache effectiveness and performance improvement"""
        synthesizer = KnowledgeSynthesizer()
        
        # Test data
        test_cases = [(f"domain_{i%5}", f"domain_{(i+1)%5}") for i in range(100)]
        
        # First pass - cold cache
        cold_times = []
        for domains in test_cases:
            start = time.perf_counter()
            synthesizer.integrate(list(domains), "performance test")
            cold_times.append(time.perf_counter() - start)
            
        # Second pass - warm cache
        warm_times = []
        for domains in test_cases:
            start = time.perf_counter()
            synthesizer.integrate(list(domains), "performance test")
            warm_times.append(time.perf_counter() - start)
            
        # Calculate speedup
        avg_cold = statistics.mean(cold_times)
        avg_warm = statistics.mean(warm_times)
        speedup = avg_cold / avg_warm
        
        print(f"\nCache Performance: Cold={avg_cold*1000:.2f}ms, Warm={avg_warm*1000:.2f}ms, Speedup={speedup:.2f}x")
        
        assert speedup > 10, f"Cache speedup {speedup:.2f}x is less than expected 10x"
        assert avg_warm < avg_cold * 0.1, "Cache not providing sufficient performance benefit"
        
    def test_foresight_scaling(self):
        """Test foresight engine scaling with depth"""
        execution_times = {}
        
        for depth in [1, 5, 10, 20, 50]:
            foresight = ForesightEngine(depth=depth)
            behaviors = ['action1', 'action2', 'action3'] * 10
            
            start = time.perf_counter()
            for _ in range(100):
                foresight.predict({'mode': 'test'}, behaviors)
            elapsed = time.perf_counter() - start
            
            execution_times[depth] = elapsed
            
        # Check scaling is reasonable (not exponential)
        for i, depth in enumerate([1, 5, 10, 20]):
            if depth * 2 in execution_times:
                time_ratio = execution_times[depth * 2] / execution_times[depth]
                assert time_ratio < 3, f"Scaling from depth {depth} to {depth*2} is {time_ratio:.2f}x (too steep)"
                
        print(f"\nForesight Scaling: {execution_times}")
        
    @pytest.mark.asyncio
    async def test_emotional_tracking_overhead(self):
        """Measure overhead of emotional context tracking"""
        mochi = MochiCore()
        test_input = "Test query for performance measurement" * 10
        
        # With emotional tracking
        with_emotion_times = []
        for _ in range(100):
            start = time.perf_counter()
            await mochi.process(test_input, emotional_context=True)
            with_emotion_times.append(time.perf_counter() - start)
            
        # Without emotional tracking
        without_emotion_times = []
        for _ in range(100):
            start = time.perf_counter()
            await mochi.process(test_input, emotional_context=False)
            without_emotion_times.append(time.perf_counter() - start)
            
        avg_with = statistics.mean(with_emotion_times)
        avg_without = statistics.mean(without_emotion_times)
        overhead = ((avg_with - avg_without) / avg_without) * 100
        
        print(f"\nEmotional Tracking Overhead: {overhead:.2f}%")
        
        assert overhead < 20, f"Emotional tracking overhead {overhead:.2f}% exceeds 20% threshold"
        
    def test_synthesis_complexity_scaling(self):
        """Test synthesis performance with increasing domain count"""
        synthesizer = KnowledgeSynthesizer()
        
        results = []
        for num_domains in [2, 4, 8, 16, 32]:
            domains = [f"domain_{i}" for i in range(num_domains)]
            
            start = time.perf_counter()
            for _ in range(10):
                synthesizer.integrate(domains, "scaling test")
                synthesizer.synthesis_cache.clear()  # Force recalculation
            elapsed = time.perf_counter() - start
            
            avg_time = elapsed / 10
            results.append((num_domains, avg_time))
            
        # Check complexity class (should be polynomial, not exponential)
        for i in range(len(results) - 1):
            n1, t1 = results[i]
            n2, t2 = results[i + 1]
            
            # Time complexity should be at most O(n^3)
            expected_max = t1 * ((n2/n1) ** 3)
            assert t2 <= expected_max * 1.5, f"Complexity worse than O(n^3) from {n1} to {n2} domains"
            
        print(f"\nSynthesis Scaling: {results}")
        

class TestStressConditions:
    """Test system under stress conditions"""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_sustained_load(self):
        """Test system under sustained load for extended period"""
        mochi = MochiCore()
        errors = []
        latencies = []
        
        start_time = time.time()
        request_count = 0
        
        # Run for 10 seconds
        while time.time() - start_time < 10:
            try:
                req_start = time.perf_counter()
                await mochi.process(f"Sustained load test {request_count}")
                latencies.append(time.perf_counter() - req_start)
                request_count += 1
            except Exception as e:
                errors.append(str(e))
                
        # Calculate metrics
        error_rate = len(errors) / request_count * 100
        avg_latency = statistics.mean(latencies) * 1000  # ms
        throughput = request_count / 10  # requests per second
        
        print(f"\nSustained Load Test:")
        print(f"  Requests: {request_count}")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Avg Latency: {avg_latency:.2f} ms")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        assert error_rate < 1, f"Error rate {error_rate:.2f}% exceeds 1% threshold"
        assert throughput > 10, f"Throughput {throughput:.2f} req/s below minimum"
        
    @pytest.mark.asyncio
    async def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        mochi = MochiCore()
        
        # Create memory pressure by filling caches
        for i in range(10000):
            domains = [f"domain_{j}_{i}" for j in range(5)]
            mochi.synthesizer.integrate(domains, f"query_{i}")
            
        # Measure memory
        memory_used = self.process.memory_info().rss / 1024 / 1024
        
        # System should still be responsive
        start = time.perf_counter()
        await mochi.process("Memory pressure test")
        response_time = time.perf_counter() - start
        
        print(f"\nMemory Pressure: {memory_used:.2f}MB used, response time {response_time*1000:.2f}ms")
        
        assert response_time < 1.0, f"Response time {response_time:.2f}s too high under memory pressure"
        
    def test_cpu_intensive_operations(self):
        """Test CPU-intensive mathematical operations"""
        synthesizer = KnowledgeSynthesizer()
        
        # Generate large matrices
        sizes = [10, 50, 100, 200]
        times = []
        
        for size in sizes:
            # Create large symmetric matrix
            A = np.random.randn(size, size)
            connections = A @ A.T
            
            start = time.perf_counter()
            synthesizer._extract_insights(connections)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
        # Check CPU efficiency
        for i, (size, elapsed) in enumerate(zip(sizes, times)):
            print(f"Matrix size {size}: {elapsed*1000:.2f}ms")
            if i > 0:
                # Should scale roughly as O(n^3) for eigenvalue decomposition
                expected = times[0] * (size / sizes[0]) ** 3
                assert elapsed < expected * 2, f"CPU scaling worse than expected for size {size}"
                
    @pytest.mark.asyncio
    async def test_burst_traffic(self):
        """Test handling of traffic bursts"""
        mochi = MochiCore()
        
        # Simulate burst
        burst_size = 100
        tasks = []
        
        start = time.perf_counter()
        for i in range(burst_size):
            tasks.append(mochi.process(f"Burst request {i}"))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        burst_time = time.perf_counter() - start
        
        # Count errors
        errors = sum(1 for r in results if isinstance(r, Exception))
        error_rate = errors / burst_size * 100
        
        print(f"\nBurst Test: {burst_size} requests in {burst_time:.2f}s")
        print(f"  Throughput: {burst_size/burst_time:.2f} req/s")
        print(f"  Error rate: {error_rate:.2f}%")
        
        assert error_rate < 5, f"Error rate {error_rate:.2f}% during burst exceeds threshold"
        assert burst_time < burst_size * 0.1, f"Burst handling too slow: {burst_time:.2f}s"
        

class TestResourceUtilization:
    """Test resource utilization and efficiency"""
    
    def test_thread_pool_efficiency(self):
        """Test thread pool utilization"""
        mochi = MochiCore()
        
        def process_sync(idx):
            import asyncio
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(mochi.process(f"Thread test {idx}"))
            loop.close()
            return result
            
        with ThreadPoolExecutor(max_workers=10) as executor:
            start = time.perf_counter()
            futures = [executor.submit(process_sync, i) for i in range(50)]
            results = [f.result() for f in futures]
            elapsed = time.perf_counter() - start
            
        throughput = 50 / elapsed
        print(f"\nThread Pool Efficiency: {throughput:.2f} req/s with 10 workers")
        
        assert throughput > 20, f"Thread pool throughput {throughput:.2f} req/s too low"
        assert all(r is not None for r in results), "Some requests failed"
        
    def test_cpu_affinity(self):
        """Test CPU utilization across cores"""
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        mochi = MochiCore()
        
        # Monitor CPU usage
        initial_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Generate load
        for _ in range(100):
            mochi.synthesizer.integrate(['d1', 'd2', 'd3'], 'cpu test')
            
        final_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Calculate utilization
        cpu_usage = [f - i for i, f in zip(initial_cpu, final_cpu)]
        avg_usage = statistics.mean(cpu_usage)
        
        print(f"\nCPU Utilization across {cpu_count} cores:")
        print(f"  Average: {avg_usage:.2f}%")
        print(f"  Per-core: {cpu_usage}")
        
        # Should distribute load somewhat evenly
        if cpu_count > 1:
            std_dev = statistics.stdev(cpu_usage)
            assert std_dev < avg_usage, "CPU load not well distributed across cores"
            
    def test_io_efficiency(self):
        """Test I/O operation efficiency"""
        mochi = MochiCore()
        
        # Measure I/O operations
        io_counters_start = self.process.io_counters()
        
        # Perform operations that involve I/O
        for i in range(100):
            response = asyncio.run(mochi.process(f"IO test {i}"))
            # Traces are saved to disk
            
        io_counters_end = self.process.io_counters()
        
        read_bytes = io_counters_end.read_bytes - io_counters_start.read_bytes
        write_bytes = io_counters_end.write_bytes - io_counters_start.write_bytes
        
        print(f"\nI/O Efficiency:")
        print(f"  Read: {read_bytes/1024:.2f} KB")
        print(f"  Write: {write_bytes/1024:.2f} KB")
        
        # Should not generate excessive I/O
        assert write_bytes < 10 * 1024 * 1024, f"Excessive write I/O: {write_bytes/1024/1024:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
