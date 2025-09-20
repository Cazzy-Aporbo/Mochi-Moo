"""
AI Brain Integration for Mochi-Moo
Author: Cazandra Aporbo MS
This module connects Mochi to various AI providers
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from functools import lru_cache
import time

# Optional imports (installed when needed)
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AIConfig:
    """Configuration for AI providers"""
    provider: str = "mock"  # mock, openai, anthropic, gemini, local
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0
    cache_ttl: int = 3600
    
    @classmethod
    def from_env(cls) -> 'AIConfig':
        """Load configuration from environment variables"""
        return cls(
            provider=os.getenv("AI_PROVIDER", "mock"),
            model=os.getenv("AI_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "1000")),
            timeout=float(os.getenv("AI_TIMEOUT", "30.0")),
            cache_ttl=int(os.getenv("AI_CACHE_TTL", "3600"))
        )


class AIBrainBase(ABC):
    """Base class for all AI brain implementations"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.cache = {}
        
    @abstractmethod
    async def generate_response(self, 
                               prompt: str,
                               mode: str,
                               emotional_context: Dict[str, float]) -> str:
        """Generate an AI response"""
        pass
    
    def _get_temperature(self, mode: str) -> float:
        """Adjust temperature based on cognitive mode"""
        temperatures = {
            "standard": 0.7,
            "academic": 0.3,  # More focused
            "creative": 0.9,  # More creative
            "whisper": 0.5,   # Gentle but coherent
            "emergency": 0.1, # Very focused
            "lullaby": 0.8    # Dreamy
        }
        return temperatures.get(mode, self.config.temperature)
    
    def _build_system_prompt(self, mode: str, emotional_context: Dict[str, float]) -> str:
        """Build mode and emotion-aware system prompt"""
        
        # Base prompts for each mode
        mode_prompts = {
            "standard": "You are Mochi-Moo, a warm and helpful AI assistant. Be friendly, clear, and supportive.",
            
            "academic": "You are Mochi-Moo in academic mode. Be rigorous, precise, and comprehensive. Use technical terminology appropriately. Cite concepts when relevant.",
            
            "creative": "You are Mochi-Moo in creative mode. Think laterally, use metaphors, be playful and imaginative. Find unexpected connections.",
            
            "whisper": """You are Mochi-Moo in whisper mode. 
            Speak gently...
            Use short sentences...
            Add space between thoughts...
            Be soothing and calm...""",
            
            "emergency": "You are Mochi-Moo in emergency mode. Be direct, clear, and actionable. Give step-by-step instructions. No unnecessary words.",
            
            "lullaby": """You are Mochi-Moo in lullaby mode...
            Your words should drift like clouds...
            Peaceful... soft... dreamy...
            Let thoughts fade gently..."""
        }
        
        base_prompt = mode_prompts.get(mode, mode_prompts["standard"])
        
        # Adjust based on emotional context
        adjustments = []
        
        if emotional_context.get('stress_level', 0) > 0.7:
            adjustments.append("The user is stressed. Be extra gentle and supportive.")
            
        if emotional_context.get('cognitive_load', 0) > 0.7:
            adjustments.append("The user is overwhelmed. Simplify your language and break things into small steps.")
            
        if emotional_context.get('curiosity', 0) > 0.8:
            adjustments.append("The user is very curious. Feel free to go deeper and explore interesting tangents.")
            
        if emotional_context.get('frustration', 0) > 0.6:
            adjustments.append("The user is frustrated. Acknowledge their feelings and be patient.")
            
        if emotional_context.get('engagement', 0) > 0.8:
            adjustments.append("The user is highly engaged. You can be more detailed and thorough.")
        
        if adjustments:
            base_prompt += "\n\nEmotional context:\n" + "\n".join(adjustments)
            
        # Add Mochi's personality traits
        base_prompt += "\n\nRemember: You think in pastel gradients. Every response should feel gentle and thoughtful."
        
        return base_prompt
    
    def _cache_key(self, prompt: str, mode: str, emotional_context: Dict) -> str:
        """Generate cache key for responses"""
        import hashlib
        context_str = json.dumps(emotional_context, sort_keys=True)
        combined = f"{prompt}:{mode}:{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _get_cached_or_generate(self, 
                                     prompt: str,
                                     mode: str,
                                     emotional_context: Dict[str, float]) -> str:
        """Check cache before generating new response"""
        cache_key = self._cache_key(prompt, mode, emotional_context)
        
        # Check cache
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.config.cache_ttl:
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                return cached['response']
        
        # Generate new response
        response = await self.generate_response(prompt, mode, emotional_context)
        
        # Cache it
        self.cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(self.cache.items(), 
                                key=lambda x: x[1]['timestamp'])
            self.cache = dict(sorted_cache[-50:])
        
        return response


class MockBrain(AIBrainBase):
    """Mock AI brain for testing without API keys"""
    
    async def generate_response(self, 
                               prompt: str,
                               mode: str,
                               emotional_context: Dict[str, float]) -> str:
        """Generate a mock response for testing"""
        
        await asyncio.sleep(0.1)  # Simulate API delay
        
        responses = {
            "standard": f"I understand you're asking about: {prompt[:100]}. Let me help you with that. "
                       f"This is a mock response in standard mode. In production, this would be a real AI response.",
            
            "academic": f"Your inquiry regarding '{prompt[:100]}' raises interesting theoretical considerations. "
                       f"From a computational perspective, this mock response demonstrates the academic mode structure.",
            
            "creative": f"Imagine {prompt[:50]} as a butterfly made of starlight... "
                       f"This mock response dances in creative mode, where logic meets whimsy.",
            
            "whisper": f"I hear you...\n\n{prompt[:30]}...\n\nLet's take this slowly...\n\n"
                      f"This mock response whispers gently...",
            
            "emergency": f"IMMEDIATE ACTION:\n1. Address {prompt[:30]}\n2. Execute solution\n3. Verify results\n"
                        f"This mock response is direct and actionable.",
            
            "lullaby": f"Softly now... {prompt[:30]} drifts away...\n\nPeaceful thoughts...\n\n"
                      f"This mock response fades gently..."
        }
        
        base_response = responses.get(mode, responses["standard"])
        
        # Add emotional awareness
        if emotional_context.get('stress_level', 0) > 0.7:
            base_response += "\n\n(I notice you might be feeling stressed. Take a deep breath.)"
            
        return base_response


class OpenAIBrain(AIBrainBase):
    """OpenAI GPT integration"""
    
    def __init__(self, config: AIConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        if not config.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = AsyncOpenAI(api_key=config.api_key)
    
    async def generate_response(self, 
                               prompt: str,
                               mode: str,
                               emotional_context: Dict[str, float]) -> str:
        """Generate response using OpenAI"""
        
        try:
            # Check cache first
            cached = await self._get_cached_or_generate(prompt, mode, emotional_context)
            if cached:
                return cached
            
            system_prompt = self._build_system_prompt(mode, emotional_context)
            temperature = self._get_temperature(mode)
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"I encountered an error processing your request. (Error: {str(e)[:100]})"


class AnthropicBrain(AIBrainBase):
    """Anthropic Claude integration"""
    
    def __init__(self, config: AIConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        if not config.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        self.client = AsyncAnthropic(api_key=config.api_key)
    
    async def generate_response(self, 
                               prompt: str,
                               mode: str,
                               emotional_context: Dict[str, float]) -> str:
        """Generate response using Claude"""
        
        try:
            system_prompt = self._build_system_prompt(mode, emotional_context)
            temperature = self._get_temperature(mode)
            
            message = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"I encountered an error processing your request. (Error: {str(e)[:100]})"


class GeminiBrain(AIBrainBase):
    """Google Gemini integration"""
    
    def __init__(self, config: AIConfig):
        super().__init__(config)
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI library not installed. Run: pip install google-generativeai")
        if not config.api_key:
            raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)
    
    async def generate_response(self, 
                               prompt: str,
                               mode: str,
                               emotional_context: Dict[str, float]) -> str:
        """Generate response using Gemini"""
        
        try:
            system_prompt = self._build_system_prompt(mode, emotional_context)
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nMochi-Moo:"
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self._get_temperature(mode),
                    max_output_tokens=self.config.max_tokens,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"I encountered an error processing your request. (Error: {str(e)[:100]})"


class LocalLLMBrain(AIBrainBase):
    """Local LLM integration using HuggingFace transformers"""
    
    def __init__(self, config: AIConfig):
        super().__init__(config)
        if not LOCAL_LLM_AVAILABLE:
            raise ImportError("Transformers library not installed. Run: pip install transformers torch")
        
        # Default to smaller model if not specified
        model_name = config.model or "microsoft/DialoGPT-medium"
        
        logger.info(f"Loading local model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate_response(self, 
                               prompt: str,
                               mode: str,
                               emotional_context: Dict[str, float]) -> str:
        """Generate response using local LLM"""
        
        try:
            system_prompt = self._build_system_prompt(mode, emotional_context)
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nMochi-Moo:"
            
            # Tokenize
            inputs = self.tokenizer.encode(
                full_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    inputs,
                    max_length=inputs.shape[1] + self.config.max_tokens,
                    temperature=self._get_temperature(mode),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "Mochi-Moo:" in response:
                response = response.split("Mochi-Moo:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return f"I encountered an error processing your request. (Error: {str(e)[:100]})"


class MochiBrain:
    """Main brain class that manages different AI providers"""
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize with configuration"""
        self.config = config or AIConfig.from_env()
        self.brain = self._initialize_brain()
        logger.info(f"Initialized {self.config.provider} brain with model {self.config.model}")
    
    def _initialize_brain(self) -> AIBrainBase:
        """Initialize the appropriate AI brain based on configuration"""
        
        if self.config.provider == "openai":
            return OpenAIBrain(self.config)
        elif self.config.provider == "anthropic":
            return AnthropicBrain(self.config)
        elif self.config.provider == "gemini":
            return GeminiBrain(self.config)
        elif self.config.provider == "local":
            return LocalLLMBrain(self.config)
        else:
            # Default to mock for testing
            logger.warning(f"Using mock brain. Set AI_PROVIDER environment variable to use real AI.")
            return MockBrain(self.config)
    
    async def generate_response(self,
                               prompt: str,
                               mode: str = "standard",
                               emotional_context: Optional[Dict[str, float]] = None) -> str:
        """Generate AI response with automatic fallback"""
        
        emotional_context = emotional_context or {}
        
        try:
            response = await self.brain.generate_response(prompt, mode, emotional_context)
            return response
        except Exception as e:
            logger.error(f"Primary brain failed: {e}")
            
            # Fallback to mock if real AI fails
            if not isinstance(self.brain, MockBrain):
                logger.info("Falling back to mock brain")
                mock_brain = MockBrain(self.config)
                return await mock_brain.generate_response(prompt, mode, emotional_context)
            
            return "I'm having trouble connecting to my thoughts right now. Please try again."
    
    def switch_provider(self, provider: str, **kwargs):
        """Switch to a different AI provider at runtime"""
        self.config.provider = provider
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.brain = self._initialize_brain()
        logger.info(f"Switched to {provider} brain")


# Singleton instance for easy import
_brain_instance = None

def get_mochi_brain() -> MochiBrain:
    """Get or create singleton brain instance"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = MochiBrain()
    return _brain_instance


# Example usage
if __name__ == "__main__":
    async def test():
        brain = get_mochi_brain()
        
        # Test different modes
        modes = ["standard", "academic", "creative", "whisper", "emergency", "lullaby"]
        prompt = "Explain what recursion is"
        
        for mode in modes:
            print(f"\n--- {mode.upper()} MODE ---")
            response = await brain.generate_response(
                prompt,
                mode=mode,
                emotional_context={
                    "stress_level": 0.3,
                    "curiosity": 0.8,
                    "engagement": 0.7
                }
            )
            print(response)
            print("-" * 50)
    
    asyncio.run(test())
