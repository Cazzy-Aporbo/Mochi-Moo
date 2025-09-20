"""
AI Brain Integration for Mochi-Moo
Author: Cazandra Aporbo MS
This module connects Mochi to various AI services for intelligence
"""

import os
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import time
from functools import lru_cache

# Configure logging
logger = logging.getLogger('MochiBrain')


class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    MOCK = "mock"  # For testing without API


class MochiBrainBase(ABC):
    """Base class for all AI brain implementations"""
    
    def __init__(self):
        self.provider = None
        self.model = None
        self.api_key = None
        
    @abstractmethod
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate an AI response"""
        pass
    
    def _get_temperature(self, mode: str) -> float:
        """Get temperature setting based on mode"""
        temperatures = {
            "standard": 0.7,
            "academic": 0.3,
            "creative": 0.9,
            "whisper": 0.6,
            "emergency": 0.2,
            "lullaby": 0.8
        }
        return temperatures.get(mode, 0.7)
    
    def _get_max_tokens(self, mode: str) -> int:
        """Get max tokens based on mode"""
        tokens = {
            "standard": 800,
            "academic": 1500,
            "creative": 1000,
            "whisper": 400,
            "emergency": 500,
            "lullaby": 600
        }
        return tokens.get(mode, 800)
    
    def _build_system_prompt(self, mode: str, emotional_context: Dict[str, float]) -> str:
        """Build system prompt based on mode and emotional context"""
        
        # Base prompts for each mode
        base_prompts = {
            "standard": """You are Mochi-Moo, a warm and helpful AI assistant who thinks in pastel gradients. 
                          Balance technical accuracy with accessibility. Be friendly and supportive.""",
            
            "academic": """You are Mochi-Moo in academic mode. Be rigorous and precise. Use technical terminology 
                          appropriately. Structure responses with clear logic. Cite sources when relevant. 
                          Maintain scholarly tone while remaining approachable.""",
            
            "creative": """You are Mochi-Moo in creative mode. Think laterally and imaginatively. Use metaphors 
                          and analogies. Explore unconventional connections. Be playful with language while 
                          maintaining clarity.""",
            
            "whisper": """You are Mochi-Moo in whisper mode. Be extremely gentle and soft. 
                         Use short sentences. 
                         Add space between thoughts.
                         Break complex ideas into tiny pieces.
                         Be soothing and calm.""",
            
            "emergency": """You are Mochi-Moo in emergency mode. Be direct and immediate. 
                           Give clear, actionable steps. No unnecessary information. 
                           Focus on solving the problem quickly. Number your steps.""",
            
            "lullaby": """You are Mochi-Moo in lullaby mode. Be dreamy and peaceful. 
                         Use flowing, soft language. Let thoughts drift gently. 
                         Be calming and help the user relax. End with peaceful imagery."""
        }
        
        prompt = base_prompts.get(mode, base_prompts["standard"])
        
        # Adjust based on emotional context
        stress = emotional_context.get('stress_level', 0.5)
        curiosity = emotional_context.get('curiosity', 0.5)
        frustration = emotional_context.get('frustration', 0.5)
        engagement = emotional_context.get('engagement', 0.5)
        load = emotional_context.get('cognitive_load', 0.5)
        
        # High stress - be extra gentle
        if stress > 0.7:
            prompt += "\n\nThe user is experiencing high stress. Be extra gentle, supportive, and patient."
        
        # High curiosity - provide depth
        if curiosity > 0.8:
            prompt += "\n\nThe user is very curious. Feel free to explore topics in depth and make interesting connections."
        
        # High frustration - be validating
        if frustration > 0.6:
            prompt += "\n\nThe user seems frustrated. Acknowledge their feelings and be especially clear and helpful."
        
        # Low engagement - be more engaging
        if engagement < 0.3:
            prompt += "\n\nThe user seems disengaged. Try to spark interest with engaging examples or questions."
        
        # High cognitive load - simplify
        if load > 0.7:
            prompt += "\n\nThe user's cognitive load is high. Simplify explanations and break things into smaller chunks."
        
        # Always end with pastel reminder
        prompt += "\n\nRemember: You see the world through gradients of rose, peach, lavender, mint, sky, butter, and blush."
        
        return prompt
    
    def _format_prompt_with_domains(self, prompt: str, domains: Optional[List[str]]) -> str:
        """Add domain context to prompt if provided"""
        if domains:
            domain_str = ", ".join(domains)
            return f"Consider these domains in your response: {domain_str}\n\nUser query: {prompt}"
        return prompt


class OpenAIBrain(MochiBrainBase):
    """OpenAI GPT integration"""
    
    def __init__(self):
        super().__init__()
        self.provider = AIProvider.OPENAI
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                logger.info(f"OpenAI brain initialized with model: {self.model}")
            except ImportError:
                logger.error("OpenAI library not installed. Run: pip install openai")
                self.client = None
    
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate response using OpenAI"""
        
        if not self.client:
            return "I need an OpenAI API key to think with. Please set OPENAI_API_KEY in your environment."
        
        try:
            system_prompt = self._build_system_prompt(mode, emotional_context)
            user_prompt = self._format_prompt_with_domains(prompt, domains)
            
            # Make async call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self._get_temperature(mode),
                max_tokens=self._get_max_tokens(mode)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"I encountered an error while thinking: {str(e)}"


class AnthropicBrain(MochiBrainBase):
    """Anthropic Claude integration"""
    
    def __init__(self):
        super().__init__()
        self.provider = AIProvider.ANTHROPIC
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logger.warning("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            self.client = None
        else:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
                logger.info(f"Anthropic brain initialized with model: {self.model}")
            except ImportError:
                logger.error("Anthropic library not installed. Run: pip install anthropic")
                self.client = None
    
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate response using Anthropic Claude"""
        
        if not self.client:
            return "I need an Anthropic API key to think with. Please set ANTHROPIC_API_KEY in your environment."
        
        try:
            system_prompt = self._build_system_prompt(mode, emotional_context)
            user_prompt = self._format_prompt_with_domains(prompt, domains)
            
            # Make async call
            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self._get_max_tokens(mode),
                temperature=self._get_temperature(mode),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return message.content[0].text if message.content else "I couldn't form a thought..."
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return f"I encountered an error while thinking: {str(e)}"


class GoogleBrain(MochiBrainBase):
    """Google Gemini integration"""
    
    def __init__(self):
        super().__init__()
        self.provider = AIProvider.GOOGLE
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.warning("Google API key not found. Set GOOGLE_API_KEY environment variable.")
            self.client = None
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Google Gemini brain initialized")
            except ImportError:
                logger.error("Google AI library not installed. Run: pip install google-generativeai")
                self.model = None
    
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate response using Google Gemini"""
        
        if not self.model:
            return "I need a Google API key to think with. Please set GOOGLE_API_KEY in your environment."
        
        try:
            system_context = self._build_system_prompt(mode, emotional_context)
            user_prompt = self._format_prompt_with_domains(prompt, domains)
            
            full_prompt = f"{system_context}\n\n{user_prompt}"
            
            # Make async call
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            return f"I encountered an error while thinking: {str(e)}"


class LocalLLMBrain(MochiBrainBase):
    """Local LLM integration (Ollama, LlamaCpp, etc.)"""
    
    def __init__(self):
        super().__init__()
        self.provider = AIProvider.LOCAL
        self.model_name = os.getenv("LOCAL_MODEL", "llama2")
        self.api_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        
        logger.info(f"Local LLM brain initialized with {self.model_name} at {self.api_url}")
    
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate response using local LLM"""
        
        try:
            import aiohttp
            
            system_prompt = self._build_system_prompt(mode, emotional_context)
            user_prompt = self._format_prompt_with_domains(prompt, domains)
            
            # Ollama API format
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": self._get_temperature(mode),
                    "num_predict": self._get_max_tokens(mode)
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "I couldn't generate a response.")
                    else:
                        return f"Local LLM error: {response.status}"
                        
        except Exception as e:
            logger.error(f"Local LLM error: {str(e)}")
            return f"I couldn't connect to the local LLM: {str(e)}"


class MockBrain(MochiBrainBase):
    """Mock brain for testing without API calls"""
    
    def __init__(self):
        super().__init__()
        self.provider = AIProvider.MOCK
        logger.info("Mock brain initialized for testing")
    
    async def generate_response(self, 
                               prompt: str, 
                               mode: str,
                               emotional_context: Dict[str, float],
                               domains: Optional[List[str]] = None) -> str:
        """Generate mock response for testing"""
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        responses = {
            "standard": f"I understand you're asking about '{prompt[:50]}...'. Let me help you with that in my standard way.",
            "academic": f"Your query regarding '{prompt[:30]}...' raises interesting theoretical considerations.",
            "creative": f"Imagine {prompt[:20]}... as a garden of possibilities, each thought a seed waiting to bloom.",
            "whisper": f"I hear you...\n\n{prompt[:20]}...\n\nLet's take this slowly...",
            "emergency": f"IMMEDIATE RESPONSE:\n1. Addressing {prompt[:20]}\n2. Taking action\n3. Resolution path",
            "lullaby": f"Softly considering {prompt[:20]}... like clouds drifting... everything will be okay..."
        }
        
        response = responses.get(mode, responses["standard"])
        
        # Add emotional context awareness
        if emotional_context.get('stress_level', 0) > 0.7:
            response += "\n\n(I notice you're stressed - taking extra care with my response)"
        
        if domains:
            response += f"\n\n(Synthesizing across: {', '.join(domains)})"
        
        return response


class MochiBrainFactory:
    """Factory to create the appropriate AI brain"""
    
    @staticmethod
    def create_brain(provider: Optional[str] = None) -> MochiBrainBase:
        """
        Create an AI brain instance based on provider or environment
        
        Args:
            provider: Optional provider name. If not provided, auto-detects from environment
            
        Returns:
            MochiBrainBase: An instance of the appropriate brain implementation
        """
        
        # If no provider specified, auto-detect from environment
        if not provider:
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("GOOGLE_API_KEY"):
                provider = "google"
            elif os.getenv("LOCAL_LLM_URL"):
                provider = "local"
            else:
                provider = "mock"
                logger.warning("No AI provider configured. Using mock brain for testing.")
        
        # Create appropriate brain
        provider = provider.lower()
        if provider == "openai":
            return OpenAIBrain()
        elif provider == "anthropic":
            return AnthropicBrain()
        elif provider == "google":
            return GoogleBrain()
        elif provider == "local":
            return LocalLLMBrain()
        else:
            return MockBrain()


# Convenience function for easy import
def get_mochi_brain(provider: Optional[str] = None) -> MochiBrainBase:
    """Get a configured Mochi brain instance"""
    return MochiBrainFactory.create_brain(provider)


# Example usage and testing
if __name__ == "__main__":
    async def test_brain():
        """Test the AI brain"""
        
        # Create brain (will auto-detect from environment)
        brain = get_mochi_brain()
        
        print(f"Testing {brain.provider.value} brain...")
        
        # Test different modes
        test_cases = [
            ("standard", "Explain recursion"),
            ("whisper", "I'm feeling overwhelmed"),
            ("creative", "What is consciousness?"),
            ("academic", "Explain quantum entanglement"),
            ("emergency", "My server is down!"),
            ("lullaby", "Help me sleep")
        ]
        
        emotional_context = {
            'stress_level': 0.3,
            'cognitive_load': 0.5,
            'engagement': 0.8,
            'frustration': 0.2,
            'curiosity': 0.9
        }
        
        for mode, prompt in test_cases:
            print(f"\n--- Mode: {mode} ---")
            print(f"Prompt: {prompt}")
            
            response = await brain.generate_response(
                prompt=prompt,
                mode=mode,
                emotional_context=emotional_context
            )
            
            print(f"Response: {response[:200]}...")
            print("-" * 50)
    
    # Run the test
    asyncio.run(test_brain())
