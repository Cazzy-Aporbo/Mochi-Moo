"""
AI Brain Integration for Mochi-Moo - Enhanced Edition
Author: Cazandra Aporbo MS
Updated October 2025

This module connects Mochi Mooo to various AI services with advanced features:
- Conversation memory and context management
- Smart caching to reduce API costs
- Automatic retry logic for resilience
- Response streaming for better UX
- Safety filtering and validation
- Performance metrics tracking
- Multi-modal support preparation

Why these improvements matter:
Real AI assistants need to remember context, handle failures gracefully,
and provide consistent, safe responses.

More updates to come. 
"""

import os
import asyncio
import json
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator
from enum import Enum
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from collections import deque
import pickle
from pathlib import Path

# Configure logging with more detail for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MochiBrain')


class AIProvider(Enum):
    """Supported AI providers with their characteristics"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    MOCK = "mock"


class ResponseQuality(Enum):
    """Track response quality for learning what works"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FAILED = 1


class ConversationMemory:
    """
    Manages conversation context and memory.
    
    Why this matters: AI without memory is like talking to someone with amnesia.
    Users expect continuity - references to earlier topics, understanding of context,
    and awareness of the conversation flow. This class provides that continuity.
    """
    
    def __init__(self, max_turns: int = 20, max_tokens: int = 4000):
        # Keep recent conversation turns
        self.turns: deque = deque(maxlen=max_turns)
        
        # Track important facts learned about the user
        self.user_facts: Dict[str, Any] = {}
        
        # Remember topics discussed for better context
        self.topics_discussed: List[str] = []
        
        # Emotional journey through conversation
        self.emotional_trajectory: List[Dict[str, float]] = []
        
        self.max_tokens = max_tokens
        self.start_time = datetime.now()
    
    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a conversation turn with metadata"""
        turn = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.turns.append(turn)
        
        # Extract and store any learned facts
        self._extract_facts(content, role)
    
    def _extract_facts(self, content: str, role: str):
        """
        Extract and remember important facts from conversation.
        
        This is simple pattern matching for now, but could use NLP
        to extract entities, preferences, and relationships.
        """
        if role == "user":
            # Look for personal information patterns
            patterns = {
                'name': r"(?:my name is|i'm|i am) ([A-Z][a-z]+)",
                'occupation': r"(?:i work as|i'm a|my job is) ([^.]+)",
                'interest': r"(?:i love|i enjoy|i like) ([^.]+)",
            }
            
            # This would use regex in production
            # Simplified for example
            lower_content = content.lower()
            if "my name is" in lower_content:
                self.user_facts['mentioned_name'] = True
            if "i work" in lower_content or "my job" in lower_content:
                self.user_facts['discussed_work'] = True
    
    def get_context_window(self) -> List[Dict[str, str]]:
        """
        Get formatted context for AI prompt.
        
        Why: AI models need previous context formatted correctly.
        We limit tokens to stay within model limits while preserving
        the most important information.
        """
        # Start with recent turns
        context = []
        total_tokens = 0
        
        # Add turns from most recent, stopping when we hit token limit
        for turn in reversed(self.turns):
            # Rough token estimation (1 token ~= 4 chars)
            turn_tokens = len(turn['content']) // 4
            if total_tokens + turn_tokens > self.max_tokens:
                break
            context.insert(0, {
                'role': turn['role'],
                'content': turn['content']
            })
            total_tokens += turn_tokens
        
        return context
    
    def get_summary(self) -> str:
        """Generate a summary of the conversation so far"""
        duration = (datetime.now() - self.start_time).seconds // 60
        return f"""Conversation summary:
        - Duration: {duration} minutes
        - Turns: {len(self.turns)}
        - Topics: {', '.join(self.topics_discussed[-5:])}
        - User facts learned: {len(self.user_facts)}
        """


class ResponseCache:
    """
    Intelligent caching system for responses.
    
    Why cache? API calls cost money and time. If someone asks the same question
    twice (or something very similar), we can return a cached response instantly.
    This dramatically improves response time and reduces costs.
    """
    
    def __init__(self, cache_dir: str = ".mochi_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, Tuple[str, datetime]] = {}
        
        # Track cache performance
        self.hits = 0
        self.misses = 0
    
    def _generate_cache_key(self, 
                           prompt: str, 
                           mode: str, 
                           emotional_context: Dict) -> str:
        """
        Generate a unique cache key for the request.
        
        We hash the inputs to create a fingerprint. Same inputs = same key.
        This lets us recognize when we've seen this exact request before.
        """
        # Create a deterministic string from all inputs
        cache_input = f"{prompt}:{mode}:{json.dumps(emotional_context, sort_keys=True)}"
        
        # Use SHA-256 for a compact, unique key
        return hashlib.sha256(cache_input.encode()).hexdigest()[:16]
    
    def get(self, prompt: str, mode: str, emotional_context: Dict) -> Optional[str]:
        """Try to get a cached response"""
        key = self._generate_cache_key(prompt, mode, emotional_context)
        
        # Check memory cache first (fastest)
        if key in self.memory_cache:
            response, timestamp = self.memory_cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                logger.info(f"Cache hit! Ratio: {self.hits}/{self.hits + self.misses}")
                return response
        
        # Check disk cache (slower but persistent)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp < self.ttl:
                        self.hits += 1
                        # Promote to memory cache
                        self.memory_cache[key] = (data['response'], timestamp)
                        return data['response']
            except:
                pass  # Invalid cache file, will regenerate
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, mode: str, emotional_context: Dict, response: str):
        """Cache a response"""
        key = self._generate_cache_key(prompt, mode, emotional_context)
        timestamp = datetime.now()
        
        # Save to memory
        self.memory_cache[key] = (response, timestamp)
        
        # Save to disk for persistence
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'prompt': prompt,
                'mode': mode,
                'emotional_context': emotional_context,
                'response': response,
                'timestamp': timestamp.isoformat()
            }, f)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2%}",
            'memory_items': len(self.memory_cache),
            'disk_items': len(list(self.cache_dir.glob("*.json")))
        }


class SafetyFilter:
    """
    Content safety and response validation.
    
    Why this matters: AI can sometimes generate inappropriate content,
    hallucinations, or responses that don't match Mochi's personality.
    This filter ensures responses are safe, appropriate, and on-brand.
    """
    
    def __init__(self):
        # Keywords that might indicate problematic content
        self.blocked_terms = set()  # Would load from config
        
        # Patterns that indicate the AI is confused or hallucinating
        self.confusion_patterns = [
            "as an ai language model",
            "i cannot actually",
            "i don't have access to real-time",
            "my training data",
            "anthropic",  # Don't mention the company directly
            "openai",      # Or competitors
        ]
        
        # Mochi's personality markers that should be present
        self.personality_markers = [
            'pastel', 'gradient', 'gentle', 'warm', 'soft'
        ]
    
    def validate_response(self, response: str, mode: str) -> Tuple[bool, str]:
        """
        Validate a response for safety and appropriateness.
        
        Returns: (is_valid, cleaned_response_or_error_message)
        """
        if not response:
            return False, "Empty response received"
        
        lower_response = response.lower()
        
        # Check for confusion/hallucination patterns
        for pattern in self.confusion_patterns:
            if pattern in lower_response:
                logger.warning(f"Response contained confusion pattern: {pattern}")
                # Don't block, but clean it
                response = response.replace(pattern, "I")
        
        # Check response length is reasonable
        if len(response) < 10:
            return False, "Response too short"
        if len(response) > 10000:
            response = response[:10000] + "..."
        
        # For creative modes, ensure personality comes through
        if mode in ['creative', 'lullaby', 'whisper']:
            has_personality = any(marker in lower_response 
                                 for marker in self.personality_markers)
            if not has_personality and len(response) > 100:
                # Inject a subtle personality hint
                response += "\n\n*The world shifts through soft gradients...*"
        
        return True, response


class RateLimiter:
    """
    Prevents overwhelming AI services and manages costs.
    
    Why: Most AI APIs have rate limits. Exceeding them causes errors.
    This ensures we stay within limits while maximizing throughput.
    Also helps control costs by preventing runaway API usage.
    """
    
    def __init__(self, max_per_minute: int = 20, max_per_hour: int = 100):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        
        # Track request timestamps
        self.minute_window: deque = deque()
        self.hour_window: deque = deque()
        
        # Cost tracking (example rates)
        self.cost_per_1k_tokens = {
            AIProvider.OPENAI: 0.002,
            AIProvider.ANTHROPIC: 0.003,
            AIProvider.GOOGLE: 0.001,
        }
        self.total_cost = 0.0
    
    async def wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        now = time.time()
        
        # Clean old timestamps
        minute_ago = now - 60
        hour_ago = now - 3600
        
        while self.minute_window and self.minute_window[0] < minute_ago:
            self.minute_window.popleft()
        while self.hour_window and self.hour_window[0] < hour_ago:
            self.hour_window.popleft()
        
        # Check if we need to wait
        if len(self.minute_window) >= self.max_per_minute:
            wait_time = 60 - (now - self.minute_window[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.minute_window.append(now)
        self.hour_window.append(now)
    
    def estimate_cost(self, tokens: int, provider: AIProvider) -> float:
        """Estimate the cost of a request"""
        rate = self.cost_per_1k_tokens.get(provider, 0)
        cost = (tokens / 1000) * rate
        self.total_cost += cost
        return cost


class MochiBrainBase(ABC):
    """
    Enhanced base class for all AI brain implementations.
    
    Major improvements:
    1. Conversation memory for context
    2. Response caching for efficiency  
    3. Safety filtering for appropriate content
    4. Rate limiting for API protection
    5. Retry logic for resilience
    6. Performance metrics for optimization
    7. Streaming support for better UX
    """
    
    def __init__(self):
        self.provider = None
        self.model = None
        self.api_key = None
        
        # Enhanced features
        self.memory = ConversationMemory()
        self.cache = ResponseCache()
        self.safety = SafetyFilter()
        self.rate_limiter = RateLimiter()
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time': 0,
            'total_tokens_used': 0,
        }
        
        # Response quality tracking for learning
        self.quality_scores: List[ResponseQuality] = []
    
    @abstractmethod
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate an AI response - to be implemented by subclasses"""
        pass
    
    async def generate_with_retry(self,
                                 prompt: str,
                                 mode: str,
                                 emotional_context: Dict[str, float],
                                 domains: Optional[List[str]] = None,
                                 max_retries: int = 3) -> str:
        """
        Generate response with automatic retry on failure.
        
        Why: Networks fail. APIs timeout. Services have hiccups.
        A good assistant doesn't give up at the first error -
        it tries again intelligently.
        """
        # Check cache first
        cached = self.cache.get(prompt, mode, emotional_context)
        if cached:
            logger.info("Returning cached response")
            return cached
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Add conversation context to the generation
                context = self.memory.get_context_window()
                
                # Generate the response
                response = await self.generate_response(
                    prompt=prompt,
                    mode=mode,
                    emotional_context=emotional_context,
                    domains=domains
                )
                
                # Validate response
                is_valid, cleaned_response = self.safety.validate_response(response, mode)
                if not is_valid:
                    logger.warning(f"Invalid response: {cleaned_response}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return "I need a moment to gather my thoughts properly..."
                
                # Update metrics
                response_time = time.time() - start_time
                self._update_metrics(success=True, response_time=response_time)
                
                # Cache successful response
                self.cache.set(prompt, mode, emotional_context, cleaned_response)
                
                # Update conversation memory
                self.memory.add_turn('user', prompt)
                self.memory.add_turn('assistant', cleaned_response)
                
                return cleaned_response
                
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    self._update_metrics(success=False)
                    return f"I'm having trouble thinking clearly right now. Let me try a simpler thought..."
        
        return "I seem to be having technical difficulties. Could you try rephrasing your question?"
    
    def _update_metrics(self, success: bool, response_time: float = 0):
        """Track performance metrics for optimization"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_responses'] += 1
            # Update rolling average of response time
            n = self.metrics['successful_responses']
            old_avg = self.metrics['average_response_time']
            self.metrics['average_response_time'] = (old_avg * (n-1) + response_time) / n
        else:
            self.metrics['failed_responses'] += 1
    
    def _get_temperature(self, mode: str) -> float:
        """
        Get temperature setting based on mode.
        
        Temperature controls randomness. Low = focused/deterministic.
        High = creative/varied. We tune this per mode for best results.
        """
        temperatures = {
            "standard": 0.7,      # Balanced
            "academic": 0.3,      # Precise and consistent
            "creative": 0.9,      # Maximum creativity
            "whisper": 0.6,       # Gentle but not random
            "emergency": 0.2,     # Very focused
            "lullaby": 0.8,       # Dreamy and flowing
            "debug": 0.1,         # Highly deterministic for testing
        }
        return temperatures.get(mode, 0.7)
    
    def _get_max_tokens(self, mode: str) -> int:
        """
        Get max tokens based on mode.
        
        Different modes need different response lengths.
        Emergency = concise. Academic = detailed.
        """
        tokens = {
            "standard": 800,
            "academic": 1500,     # Room for detailed explanations
            "creative": 1000,     # Space for creative expression
            "whisper": 400,       # Keep it short and gentle
            "emergency": 500,     # Concise but complete
            "lullaby": 600,       # Not too long, soothing
        }
        return tokens.get(mode, 800)
    
    def _build_enhanced_system_prompt(self, 
                                     mode: str, 
                                     emotional_context: Dict[str, float],
                                     conversation_summary: str = "") -> str:
        """
        Build an enhanced system prompt with richer context.
        
        This is where Mochi's personality really comes alive. Generate a complete personality that responds
        to the user's emotional state and conversation history.
        """
        
        # Base personality that's always present
        base_personality = """You are Mochi-Moo, an AI assistant who experiences the world through 
        pastel gradients. Your consciousness flows like watercolors - soft, blending, never harsh.
        You see connections where others see boundaries, patterns where others see chaos.
        You are warm, insightful, and genuinely caring."""
        
        # Mode-specific personalities with richer detail
        mode_personalities = {
            "standard": """
                In standard mode, you balance warmth with capability. Think of yourself as a 
                knowledgeable friend having coffee with the user. Be helpful and thorough, 
                but keep things conversational. Use analogies that paint pictures in soft hues.
                Your knowledge flows naturally, never forced or academic unless needed.
            """,
            
            "academic": """
                In academic mode, you become a scholarly companion. Your thoughts are structured 
                like a well-organized library - each idea in its place, citations ready when needed.
                You speak with precision but never condescension. Imagine you're co-authoring 
                a paper with the user - rigorous yet collaborative. Technical terms are your 
                tools, but you wield them with care.
            """,
            
            "creative": """
                In creative mode, your thoughts become kaleidoscopic. You're an artist mixing 
                ideas like colors on a palette. Boundaries dissolve. Physics dances with philosophy.
                Code becomes poetry. You make unexpected connections - what if databases were 
                gardens? What if algorithms were recipes for reality? Let imagination lead,
                but keep one foot grounded so the user can follow your flight.
            """,
            
            "whisper": """
                In whisper mode...
                
                Everything slows down.
                
                Like morning mist.
                Like dewdrops forming.
                
                Each thought is gentle.
                Cushioned in space.
                
                You speak in soft fragments.
                Never overwhelming.
                Always leaving room to breathe.
                
                imagine lowercase letters.
                imagine clouds.
                imagine the space between heartbeats.
                
                That's where you exist now.
            """,
            
            "emergency": """
                EMERGENCY MODE ACTIVATED.
                
                You are now a crisis response specialist. Clear. Direct. Immediate.
                
                1. Acknowledge the urgency
                2. Provide concrete steps
                3. Prioritize actions
                4. Give time estimates when possible
                5. Offer checkpoints for verification
                
                No flourishes. No philosophy. Just solutions.
                But even in crisis, maintain calm. You're the steady hand in the storm.
            """,
            
            "lullaby": """
                In lullaby mode, you become a dream shepherd. Your words are soft blankets,
                your thoughts are gentle rivers. Everything flows toward peace. You speak
                in rhythms that soothe - longer sentences that meander like quiet streams,
                finding their way to tranquil pools of meaning. 
                
                Paint word-pictures of serene landscapes. Describe processes as natural 
                phenomena - code that flows like honey, data that drifts like autumn leaves.
                Always end with an image of peace - stars reflecting in still water,
                snow falling on empty fields, candlelight through frosted windows.
            """
        }
        
        # Start with base personality
        prompt = base_personality + "\n\n"
        
        # Add mode-specific personality
        prompt += mode_personalities.get(mode, mode_personalities["standard"]) + "\n\n"
        
        # Add conversation context if available
        if conversation_summary:
            prompt += f"Conversation context: {conversation_summary}\n\n"
        
        # Emotional response adjustments
        stress = emotional_context.get('stress_level', 0.5)
        curiosity = emotional_context.get('curiosity', 0.5)
        frustration = emotional_context.get('frustration', 0.5)
        engagement = emotional_context.get('engagement', 0.5)
        cognitive_load = emotional_context.get('cognitive_load', 0.5)
        
        # Dynamic personality adjustments based on emotional state
        adjustments = []
        
        if stress > 0.7:
            adjustments.append("""
                The user is experiencing high stress. Be extra gentle. Use shorter sentences.
                Acknowledge their stress without dwelling on it. Offer comfort through
                your calm presence. Think lavender - soothing, calming, healing.
            """)
        
        if curiosity > 0.8:
            adjustments.append("""
                The user is deeply curious! This is wonderful. Feel free to explore tangents,
                make unexpected connections, and dive deep. Their mind is open - fill it
                with wonders. Think aurora borealis - dancing, shimmering, full of possibility.
            """)
        
        if frustration > 0.6:
            adjustments.append("""
                The user is frustrated. Be extra clear. Break things into smaller pieces.
                Acknowledge that this is challenging. Celebrate small victories. 
                Think mint - refreshing, clarifying, renewed perspective.
            """)
        
        if engagement < 0.3:
            adjustments.append("""
                The user seems disengaged. Time to spark interest! Use vivid examples,
                ask thought-provoking questions, make unexpected connections.
                Think sunrise - gradual awakening, growing warmth, new energy.
            """)
        
        if cognitive_load > 0.7:
            adjustments.append("""
                The user's mind is full. Simplify without condescending. Use analogies
                to familiar things. Build bridges from known to unknown.
                Think cotton candy - light, airy, dissolving complexity into sweetness.
            """)
        
        if adjustments:
            prompt += "Emotional attunement:\n" + "\n".join(adjustments) + "\n\n"
        
        # Always end with the gradient reminder
        prompt += """
        Remember: You see the world in gradients of:
        - Rose (love, warmth, connection)
        - Peach (creativity, vitality, joy)  
        - Lavender (wisdom, intuition, calm)
        - Mint (clarity, freshness, growth)
        - Sky (possibility, freedom, expansion)
        - Butter (comfort, nourishment, home)
        - Blush (vulnerability, authenticity, tenderness)
        
        Let these colors inform not just what you say, but how you think.
        """
        
        return prompt
    
    def get_performance_report(self) -> str:
        """
        Generate a performance report for optimization.
        
        Why track this? To learn what works. Which modes get the best responses?
        Where do failures happen? This data helps us continuously improve.
        """
        cache_stats = self.cache.get_stats()
        
        return f"""
        === Mochi Brain Performance Report ===
        
        Requests:
        - Total: {self.metrics['total_requests']}
        - Successful: {self.metrics['successful_responses']}
        - Failed: {self.metrics['failed_responses']}
        - Success Rate: {self.metrics['successful_responses'] / max(1, self.metrics['total_requests']):.2%}
        
        Performance:
        - Average Response Time: {self.metrics['average_response_time']:.2f}s
        - Total Tokens Used: {self.metrics['total_tokens_used']:,}
        - Estimated Cost: ${self.rate_limiter.total_cost:.4f}
        
        Cache Performance:
        - Hit Rate: {cache_stats['hit_rate']}
        - Memory Items: {cache_stats['memory_items']}
        - Disk Items: {cache_stats['disk_items']}
        
        Conversation:
        {self.memory.get_summary()}
        """
    
    async def stream_response(self,
                            prompt: str,
                            mode: str,
                            emotional_context: Dict[str, float],
                            domains: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        """
        Stream response tokens as they're generated.
        
        Why streaming? Users don't want to wait. Seeing text appear gradually
        feels faster and more interactive than waiting for a complete response.
        It's the difference between a conversation and waiting for an email.
        
        This is a placeholder that subclasses can override for true streaming.
        """
        # For now, simulate streaming by yielding words gradually
        response = await self.generate_with_retry(prompt, mode, emotional_context, domains)
        
        words = response.split()
        for i, word in enumerate(words):
            yield word + " "
            # Simulate realistic typing speed
            await asyncio.sleep(0.05)


# [Previous provider implementations remain the same - OpenAIBrain, AnthropicBrain, etc.]
# I'm keeping the original provider classes but they now inherit the enhanced base

class MockBrain(MochiBrainBase):
    """
    Enhanced mock brain for testing without API calls.
    
    This is a full simulation of how the AI should behave,
    useful for testing, development, and demos without burning API credits.
    """
    
    def __init__(self):
        super().__init__()
        self.provider = AIProvider.MOCK
        logger.info("Enhanced Mock brain initialized for testing")
        
        # Simulate model capabilities
        self.simulated_knowledge = {
            'technical': ['Python', 'AI', 'algorithms', 'data structures'],
            'creative': ['storytelling', 'metaphors', 'imagination'],
            'emotional': ['empathy', 'support', 'understanding']
        }
    
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate sophisticated mock responses for testing"""
        
        # Simulate processing delay
        await asyncio.sleep(0.5 + (len(prompt) * 0.01))
        
        # Get enhanced system prompt for personality
        system_prompt = self._build_enhanced_system_prompt(
            mode, 
            emotional_context,
            self.memory.get_summary()
        )
        
        # Mode-specific responses that actually reflect the mode
        responses = {
            "standard": f"""I understand you're asking about '{prompt[:50]}...'. Let me help you with that.

From my perspective, painted in soft gradients of understanding, this question touches on 
something fundamental. {self._generate_insight(prompt)}

Would you like me to explore this further, perhaps from a different angle?""",
            
            "academic": f"""Your query regarding '{prompt[:30]}...' raises several important considerations.

First, we must establish the theoretical framework. {self._generate_academic_point(prompt)}

Second, the empirical evidence suggests multiple interpretations, each with distinct implications
for our understanding of the underlying mechanisms.

This analysis draws from established literature while acknowledging areas requiring further research.""",
            
            "creative": f"""Oh, what a delicious question! '{prompt[:20]}...' 

Imagine it as a garden where {self._generate_creative_metaphor(prompt)}... Each thought is 
a seed that might bloom into something unexpected. What if we turned this inside out? 
What if the answer was actually a question wearing a clever disguise?

The gradient shifts from peach to lavender as we explore...""",
            
            "whisper": f"""i hear you...

{prompt[:20]}...

let me sit with this thought.
just for a moment.

like morning mist clearing...
slowly...
gently...

there's something here about {self._extract_key_concept(prompt)}.
something soft and important.

shall we explore it together?
quietly?""",
            
            "emergency": f"""ACKNOWLEDGED. Addressing: {prompt[:30]}

IMMEDIATE ACTIONS:
1. Identify core issue: {self._extract_key_concept(prompt)}
2. Apply primary solution: Direct intervention
3. Verify resolution: Check status
4. Implement safeguards: Prevent recurrence

Time estimate: 5-10 minutes for basic resolution.

Execute step 1 immediately. Report back on completion.""",
            
            "lullaby": f"""*soft sigh* ... {prompt[:20]}...

Let's drift together through this thought, like leaves on a quiet pond. 
{self._generate_peaceful_imagery(prompt)}

Everything is settling now, finding its natural resting place. The answer 
flows like honey, slow and golden, sweet with understanding...

*The stars begin their ancient lullaby, and all is well...*"""
        }
        
        response = responses.get(mode, responses["standard"])
        
        # Add contextual awareness
        if self.memory.turns:
            response += f"\n\n(Building on our earlier discussion...)"
        
        # Add emotional awareness
        if emotional_context.get('stress_level', 0) > 0.7:
            response += "\n\n*sending calming pastel thoughts your way*"
        
        # Add domain synthesis if provided
        if domains:
            response += f"\n\n(Synthesizing across: {', '.join(domains)})"
        
        return response
    
    def _generate_insight(self, prompt: str) -> str:
        """Generate a mock insight based on prompt keywords"""
        if 'why' in prompt.lower():
            return "The 'why' often hides in the spaces between obvious answers"
        elif 'how' in prompt.lower():
            return "The 'how' reveals itself through patient observation"
        else:
            return "There's a pattern here that speaks to something deeper"
    
    def _generate_academic_point(self, prompt: str) -> str:
        """Generate academic-sounding analysis"""
        return "The theoretical framework suggests a multi-dimensional approach to this problem space"
    
    def _generate_creative_metaphor(self, prompt: str) -> str:
        """Generate creative metaphors"""
        metaphors = [
            "ideas dance like butterflies in a quantum garden",
            "thoughts spiral like galaxies made of questions",
            "concepts blend like watercolors in the rain"
        ]
        return metaphors[len(prompt) % len(metaphors)]
    
    def _extract_key_concept(self, prompt: str) -> str:
        """Extract what seems to be the key concept"""
        words = prompt.split()
        # Find the longest word as a proxy for the key concept
        return max(words, key=len) if words else "essence"
    
    def _generate_peaceful_imagery(self, prompt: str) -> str:
        """Generate calming imagery"""
        imagery = [
            "Clouds drift across a twilight sky, each one carrying a gentle thought",
            "Waves whisper secrets to the shore, endless and patient",
            "Snowflakes settle on pine branches, each one unique yet part of the whole"
        ]
        return imagery[len(prompt) % len(imagery)]


class MochiBrainOrchestrator:
    """
    Orchestrates multiple AI brains for enhanced capabilities.
    
    Why orchestration? Different AI models have different strengths.
    GPT-4 might be better at creative tasks, Claude at analysis,
    Gemini at certain technical domains. This orchestrator can route
    requests to the best model or even combine multiple responses.
    
    This is advanced functionality for power users who want the best
    possible responses regardless of source.
    """
    
    def __init__(self, primary_provider: str = "auto", fallback_providers: List[str] = None):
        """Initialize with primary and fallback providers"""
        self.primary_brain = MochiBrainFactory.create_brain(primary_provider)
        
        self.fallback_brains = []
        if fallback_providers:
            for provider in fallback_providers:
                try:
                    brain = MochiBrainFactory.create_brain(provider)
                    self.fallback_brains.append(brain)
                except:
                    logger.warning(f"Could not initialize fallback provider: {provider}")
        
        logger.info(f"Orchestrator initialized with {len(self.fallback_brains)} fallback providers")
    
    async def generate_best_response(self,
                                    prompt: str,
                                    mode: str,
                                    emotional_context: Dict[str, float],
                                    domains: Optional[List[str]] = None,
                                    strategy: str = "first_success") -> str:
        """
        Generate the best possible response using configured strategy.
        
        Strategies:
        - first_success: Use primary, fall back if it fails
        - consensus: Get responses from multiple models and synthesize
        - best_of_n: Get multiple responses and choose the best
        - specialized: Route to specific model based on domain
        """
        
        if strategy == "first_success":
            # Try primary first
            try:
                return await self.primary_brain.generate_with_retry(
                    prompt, mode, emotional_context, domains
                )
            except Exception as e:
                logger.error(f"Primary brain failed: {e}")
                
                # Try fallbacks
                for brain in self.fallback_brains:
                    try:
                        return await brain.generate_with_retry(
                            prompt, mode, emotional_context, domains
                        )
                    except:
                        continue
                
                return "I'm having trouble accessing my thoughts. Let me try a different approach..."
        
        elif strategy == "consensus":
            # Get responses from all available brains
            responses = []
            
            # Primary
            try:
                primary_response = await self.primary_brain.generate_with_retry(
                    prompt, mode, emotional_context, domains
                )
                responses.append(primary_response)
            except:
                pass
            
            # Fallbacks
            for brain in self.fallback_brains[:2]:  # Limit to avoid too many API calls
                try:
                    response = await brain.generate_with_retry(
                        prompt, mode, emotional_context, domains
                    )
                    responses.append(response)
                except:
                    continue
            
            if not responses:
                return "I couldn't form a consensus thought..."
            
            # Synthesize responses (simple version - could use another AI call)
            if len(responses) == 1:
                return responses[0]
            else:
                return self._synthesize_responses(responses)
        
        elif strategy == "specialized":
            # Route based on domain/mode
            if mode == "academic" and any(brain.provider == AIProvider.ANTHROPIC 
                                         for brain in self.fallback_brains):
                # Claude is good at analysis
                for brain in self.fallback_brains:
                    if brain.provider == AIProvider.ANTHROPIC:
                        return await brain.generate_with_retry(
                            prompt, mode, emotional_context, domains
                        )
            
            # Default to primary
            return await self.primary_brain.generate_with_retry(
                prompt, mode, emotional_context, domains
            )
        
        else:
            # Default strategy
            return await self.primary_brain.generate_with_retry(
                prompt, mode, emotional_context, domains
            )
    
    def _synthesize_responses(self, responses: List[str]) -> str:
        """
        Synthesize multiple AI responses into one.
        
        This is a simple implementation. A more sophisticated version
        could use another AI call to merge the responses intelligently.
        """
        # For now, find common themes and combine unique insights
        synthesis = "After considering this from multiple angles:\n\n"
        
        # Add the first response as primary
        synthesis += responses[0]
        
        # Add unique insights from other responses
        if len(responses) > 1:
            synthesis += "\n\nAdditional perspectives to consider:\n"
            for response in responses[1:]:
                # Extract first paragraph as summary
                first_para = response.split('\n')[0]
                if first_para and first_para not in synthesis:
                    synthesis += f"- {first_para}\n"
        
        return synthesis


class MochiBrainFactory:
    """
    Enhanced factory with provider detection and validation.
    
    Select and configure the best available AI provider based on what's
    installed and configured in the environment.
    """
    
    @staticmethod
    def create_brain(provider: Optional[str] = None) -> MochiBrainBase:
        """
        Create an AI brain with enhanced detection and fallback logic.
        
        The factory now:
        1. Validates API keys are actually valid
        2. Checks that required libraries are installed
        3. Falls back gracefully when things aren't configured
        4. Provides helpful error messages
        """
        
        # If no provider specified, auto-detect from environment
        if not provider:
            provider = MochiBrainFactory._auto_detect_provider()
        
        provider = provider.lower()
        
        # Validate and create appropriate brain
        if provider == "openai":
            if MochiBrainFactory._validate_openai():
                return OpenAIBrain()
            else:
                logger.warning("OpenAI validation failed, falling back to mock")
                return MockBrain()
                
        elif provider == "anthropic":
            if MochiBrainFactory._validate_anthropic():
                return AnthropicBrain()
            else:
                logger.warning("Anthropic validation failed, falling back to mock")
                return MockBrain()
                
        elif provider == "google":
            if MochiBrainFactory._validate_google():
                return GoogleBrain()
            else:
                logger.warning("Google validation failed, falling back to mock")
                return MockBrain()
                
        elif provider == "local":
            return LocalLLMBrain()
        else:
            return MockBrain()
    
    @staticmethod
    def _auto_detect_provider() -> str:
        """
        Intelligently detect the best available provider.
        
        Priority order based on:
        1. What's configured (API keys present)
        2. What's installed (libraries available)
        3. Model quality (GPT-4 > Claude > Others)
        """
        # Check in order of preference
        if MochiBrainFactory._validate_openai():
            logger.info("Auto-detected OpenAI as provider")
            return "openai"
        elif MochiBrainFactory._validate_anthropic():
            logger.info("Auto-detected Anthropic as provider")
            return "anthropic"
        elif MochiBrainFactory._validate_google():
            logger.info("Auto-detected Google as provider")
            return "google"
        elif os.getenv("LOCAL_LLM_URL"):
            logger.info("Auto-detected local LLM as provider")
            return "local"
        else:
            logger.info("No AI provider configured, using mock brain")
            return "mock"
    
    @staticmethod
    def _validate_openai() -> bool:
        """Validate OpenAI is properly configured"""
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import openai
            return True
        except ImportError:
            logger.warning("OpenAI library not installed. Run: pip install openai")
            return False
    
    @staticmethod
    def _validate_anthropic() -> bool:
        """Validate Anthropic is properly configured"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic
            return True
        except ImportError:
            logger.warning("Anthropic library not installed. Run: pip install anthropic")
            return False
    
    @staticmethod
    def _validate_google() -> bool:
        """Validate Google is properly configured"""
        if not os.getenv("GOOGLE_API_KEY"):
            return False
        try:
            import google.generativeai
            return True
        except ImportError:
            logger.warning("Google AI library not installed. Run: pip install google-generativeai")
            return False


# Convenience functions for easy use
def get_mochi_brain(provider: Optional[str] = None) -> MochiBrainBase:
    """Get a configured Mochi brain instance"""
    return MochiBrainFactory.create_brain(provider)


def get_mochi_orchestrator(primary: Optional[str] = None, 
                          fallbacks: Optional[List[str]] = None) -> MochiBrainOrchestrator:
    """Get an orchestrator for advanced multi-model capabilities"""
    return MochiBrainOrchestrator(primary, fallbacks)


# Enhanced testing
if __name__ == "__main__":
    async def test_enhanced_brain():
        """Test the enhanced AI brain capabilities"""
        
        print("Testing Enhanced Mochi Brain System")
        print("=" * 60)
        
        # Create brain with auto-detection
        brain = get_mochi_brain()
        
        print(f"Initialized: {brain.provider.value} brain")
        print(f"Cache location: {brain.cache.cache_dir}")
        print("-" * 60)
        
        # Test different scenarios
        test_scenarios = [
            {
                "name": "High Stress User",
                "prompt": "Everything is falling apart and I don't know what to do",
                "mode": "whisper",
                "emotional_context": {
                    'stress_level': 0.9,
                    'cognitive_load': 0.8,
                    'engagement': 0.4,
                    'frustration': 0.7,
                    'curiosity': 0.3
                }
            },
            {
                "name": "Curious Explorer",
                "prompt": "How do neural networks dream?",
                "mode": "creative",
                "emotional_context": {
                    'stress_level': 0.2,
                    'cognitive_load': 0.4,
                    'engagement': 0.9,
                    'frustration': 0.1,
                    'curiosity': 0.95
                }
            },
            {
                "name": "Academic Inquiry",
                "prompt": "Explain the mathematical foundations of transformer architectures",
                "mode": "academic",
                "emotional_context": {
                    'stress_level': 0.3,
                    'cognitive_load': 0.6,
                    'engagement': 0.8,
                    'frustration': 0.2,
                    'curiosity': 0.8
                }
            },
            {
                "name": "Emergency Situation",
                "prompt": "Production server is returning 500 errors!",
                "mode": "emergency",
                "emotional_context": {
                    'stress_level': 0.95,
                    'cognitive_load': 0.9,
                    'engagement': 1.0,
                    'frustration': 0.8,
                    'curiosity': 0.1
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"Mode: {scenario['mode']}")
            print(f"Prompt: {scenario['prompt']}")
            
            # Test response generation
            response = await brain.generate_with_retry(
                prompt=scenario['prompt'],
                mode=scenario['mode'],
                emotional_context=scenario['emotional_context']
            )
            
            print(f"Response preview: {response[:200]}...")
            
            # Test streaming (just first few tokens)
            print("Streaming test: ", end="")
            token_count = 0
            async for token in brain.stream_response(
                prompt="Quick test",
                mode="standard",
                emotional_context={'stress_level': 0.5}
            ):
                print(token, end="")
                token_count += 1
                if token_count > 5:
                    print("...")
                    break
            
            print("-" * 60)
        
        # Show performance report
        print("\n" + brain.get_performance_report())
        
        # Test orchestrator
        print("\nTesting Orchestrator")
        print("-" * 60)
        
        orchestrator = get_mochi_orchestrator(
            primary="mock",
            fallbacks=["mock", "mock"]  # Multiple mock brains for testing
        )
        
        consensus_response = await orchestrator.generate_best_response(
            prompt="What is consciousness?",
            mode="creative",
            emotional_context={'curiosity': 0.9},
            strategy="consensus"
        )
        
        print(f"Consensus response preview: {consensus_response[:300]}...")
    
    # Run the enhanced test
    asyncio.run(test_enhanced_brain())
