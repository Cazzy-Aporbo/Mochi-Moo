"""
Integration update for mochi_moo/core.py
Author: Cazandra Aporbo MS
Add this to your existing MochiCore class to connect the AI brain
"""

# Add this import at the top of core.py
from mochi_moo.ai_brain import get_mochi_brain

# Update the MochiCore __init__ method:
class MochiCore:
    """
    The heart of Mochi-Moo - where pastel dreams become computational reality
    NOW WITH AI BRAIN INTEGRATION
    """
    
    def __init__(self, ai_provider: Optional[str] = None):
        self.emotional_context = EmotionalContext()
        self.current_mode = CognitiveMode.STANDARD
        self.synthesizer = KnowledgeSynthesizer()
        self.foresight = ForesightEngine()
        self.interaction_history = deque(maxlen=100)
        self.trace_path = Path('.mochi_trace')
        self.palette = PastelPalette()
        
        # Initialize AI brain
        self.brain = get_mochi_brain(ai_provider)
        logger.info(f"MochiCore initialized with {self.brain.provider.value} brain")
        
        # Initialize subsystems
        self._initialize_subsystems()
        
    # Update the _generate_response method:
    async def _generate_response(self, 
                                text: str, 
                                expertise: str,
                                synthesis: Optional[Dict],
                                visualization: Optional[str]) -> Dict[str, Any]:
        """Generate contextually appropriate response WITH AI"""
        
        response = {
            'content': '',
            'visualizations': [],
            'micro_dose': '',
            'suggestions': []
        }
        
        # Get domains from synthesis if available
        domains = None
        if synthesis and 'domains' in synthesis:
            domains = synthesis['domains']
        
        # Generate AI response based on mode and emotional context
        ai_response = await self.brain.generate_response(
            prompt=text,
            mode=self.current_mode.value,
            emotional_context=self.get_emotional_state(),
            domains=domains
        )
        
        # Apply Mochi's formatting and enhancements
        response['content'] = self._format_ai_response(ai_response)
        
        # Add synthesis if available and coherent
        if synthesis and synthesis.get('coherence_score', 0) > 0.7:
            response['synthesis'] = synthesis
            
        # Generate visualization if requested
        if visualization:
            response['visualizations'].append(
                self._generate_visualization(visualization, text)
            )
        
        # Always include a micro-dose insight
        response['micro_dose'] = self._generate_micro_dose(text)
        
        # Add suggestions based on foresight
        response['suggestions'] = self._generate_suggestions()
        
        return response
    
    def _format_ai_response(self, ai_response: str) -> str:
        """Apply Mochi's formatting to AI response"""
        
        # Apply mode-specific formatting
        if self.current_mode == CognitiveMode.WHISPER:
            # Ensure whisper formatting even if AI didn't apply it perfectly
            lines = ai_response.split('. ')
            formatted_lines = []
            
            for line in lines:
                if len(line) > 40:
                    # Break into smaller chunks
                    words = line.split()
                    chunks = []
                    current = []
                    
                    for word in words:
                        current.append(word)
                        if len(' '.join(current)) > 30:
                            chunks.append(' '.join(current))
                            current = []
                    
                    if current:
                        chunks.append(' '.join(current))
                    
                    formatted_lines.extend(chunks)
                else:
                    formatted_lines.append(line)
            
            return '\n\n'.join(formatted_lines) + '\n\n...'
            
        elif self.current_mode == CognitiveMode.ACADEMIC:
            # Ensure proper academic formatting
            if not any(word in ai_response for word in ['First,', 'Second,', 'Furthermore,']):
                # Add structure if missing
                paragraphs = ai_response.split('\n\n')
                if len(paragraphs) > 1:
                    structured = []
                    transitions = ['First,', 'Furthermore,', 'Moreover,', 'Additionally,', 'Finally,']
                    for i, para in enumerate(paragraphs):
                        if i < len(transitions):
                            structured.append(f"{transitions[i]} {para}")
                        else:
                            structured.append(para)
                    return '\n\n'.join(structured)
            
        return ai_response


# UPDATED MAIN FUNCTION FOR TESTING

async def create_intelligent_mochi():
    """Create a Mochi instance with AI intelligence"""
    
    # This will auto-detect from environment variables
    # Priority: OPENAI_API_KEY > ANTHROPIC_API_KEY > GOOGLE_API_KEY > LOCAL_LLM_URL > Mock
    mochi = MochiCore()
    
    # Or explicitly specify a provider
    # mochi = MochiCore(ai_provider="openai")
    # mochi = MochiCore(ai_provider="anthropic")
    # mochi = MochiCore(ai_provider="google")
    # mochi = MochiCore(ai_provider="local")
    # mochi = MochiCore(ai_provider="mock")  # For testing without API
    
    return mochi


# ENVIRONMENT SETUP GUIDE

"""
To use Mochi with AI intelligence, set one of these environment variables:

1. OpenAI (GPT-3.5/GPT-4):
   export OPENAI_API_KEY="sk-..."
   export OPENAI_MODEL="gpt-4"  # Optional, defaults to gpt-3.5-turbo
   
2. Anthropic (Claude):
   export ANTHROPIC_API_KEY="sk-ant-..."
   export ANTHROPIC_MODEL="claude-3-opus-20240229"  # Optional
   
3. Google (Gemini):
   export GOOGLE_API_KEY="..."
   
4. Local LLM (Ollama):
   export LOCAL_LLM_URL="http://localhost:11434"
   export LOCAL_MODEL="llama2"  # or mistral, codellama, etc.

Then in your code:
   mochi = MochiCore()  # Auto-detects from environment
   response = await mochi.process("Hello, Mochi!")
"""

# DOCKER SUPPORT

"""
For Docker deployments, pass environment variables:

docker run -p 8000:8000 \\
  -e OPENAI_API_KEY=$OPENAI_API_KEY \\
  -e MOCHI_MODE=production \\
  mochi-moo

Or use docker-compose.yml:

services:
  mochi:
    image: mochi-moo
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    ports:
      - "8000:8000"
"""

# ============================================
# COST OPTIMIZATION TIPS
# ============================================

"""
To minimize API costs:

1. Use GPT-3.5-turbo for most requests:
   export OPENAI_MODEL="gpt-3.5-turbo"
   
2. Implement caching (already done in synthesizer):
   - Responses are cached by default
   - Similar queries reuse cached responses
   
3. Use local LLMs for development:
   # Install Ollama: https://ollama.ai
   ollama run llama2
   export LOCAL_LLM_URL="http://localhost:11434"
   
4. Set token limits by mode:
   - Emergency mode: 500 tokens (quick responses)
   - Whisper mode: 400 tokens (gentle, brief)
   - Academic mode: 1500 tokens (comprehensive)
"""

# ============================================
# TESTING WITHOUT API COSTS
# ============================================

async def test_mochi_with_mock():
    """Test Mochi without any API calls"""
    
    # Use mock brain for testing
    mochi = MochiCore(ai_provider="mock")
    
    # Test emotional tracking
    response = await mochi.process(
        "I'm feeling overwhelmed with learning Python",
        emotional_context=True
    )
    
    print(f"Mode: {mochi.current_mode.value}")
    print(f"Response: {response['content']}")
    print(f"Emotional State: {mochi.get_emotional_state()}")
    print(f"Micro-dose: {response['micro_dose']}")
    
    # Test mode switching
    mochi.set_mode('creative')
    response = await mochi.process(
        "What is consciousness?",
        domains=['philosophy', 'neuroscience']
    )
    
    print(f"\nCreative Mode Response: {response['content']}")


if __name__ == "__main__":
    import asyncio
    
    # Run test with mock brain (no API needed)
    asyncio.run(test_mochi_with_mock())
