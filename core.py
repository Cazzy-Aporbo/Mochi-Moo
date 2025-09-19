"""
Mochi-Moo Core Implementation
Author: Cazandra Aporbo MS
Created: 2025
A consciousness that dreams in matte rainbow and thinks in ten dimensions
"""

import asyncio
import hashlib
import json
import math
import re
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from functools import lru_cache, wraps
import logging

# Configure pastel-aware logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('MochiMoo')


class PastelPalette:
    """Mochi's signature color space - where mathematics meets aesthetics"""
    
    ROSE = (255, 218, 224)
    PEACH = (255, 229, 210)
    LAVENDER = (230, 220, 250)
    MINT = (210, 245, 230)
    SKY = (210, 235, 250)
    BUTTER = (255, 250, 210)
    BLUSH = (250, 220, 230)
    
    @classmethod
    def interpolate(cls, position: float) -> Tuple[int, int, int]:
        """Generate smooth gradient between palette colors"""
        colors = [cls.ROSE, cls.PEACH, cls.LAVENDER, cls.MINT, 
                 cls.SKY, cls.BUTTER, cls.BLUSH]
        
        if position <= 0:
            return colors[0]
        if position >= 1:
            return colors[-1]
            
        scaled = position * (len(colors) - 1)
        idx = int(scaled)
        remainder = scaled - idx
        
        if idx >= len(colors) - 1:
            return colors[-1]
            
        r1, g1, b1 = colors[idx]
        r2, g2, b2 = colors[idx + 1]
        
        return (
            int(r1 + (r2 - r1) * remainder),
            int(g1 + (g2 - g1) * remainder),
            int(b1 + (b2 - b1) * remainder)
        )
    
    @staticmethod
    def to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex string"""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)


class CognitiveMode(Enum):
    """Operating modes calibrated to user needs"""
    STANDARD = "standard"
    WHISPER = "whisper"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    EMERGENCY = "emergency"
    LULLABY = "lullaby"


@dataclass
class EmotionalContext:
    """Tracks emotional undertones for empathetic response calibration"""
    stress_level: float = 0.5  # 0-1 scale
    cognitive_load: float = 0.5
    engagement: float = 0.7
    frustration: float = 0.2
    curiosity: float = 0.8
    
    def adjust(self, signal: str) -> None:
        """Dynamically adjust based on interaction signals"""
        patterns = {
            r'\b(confused|lost|stuck)\b': ('cognitive_load', 0.1),
            r'\b(frustrated|annoyed|angry)\b': ('frustration', 0.2),
            r'\b(curious|wondering|interested)\b': ('curiosity', 0.1),
            r'\b(stressed|overwhelmed|anxious)\b': ('stress_level', 0.15),
            r'\b(excited|engaged|focused)\b': ('engagement', 0.1)
        }
        
        signal_lower = signal.lower()
        for pattern, (attribute, delta) in patterns.items():
            if re.search(pattern, signal_lower):
                current = getattr(self, attribute)
                setattr(self, attribute, max(0, min(1, current + delta)))


@dataclass
class MochiTrace:
    """Session breadcrumb for continuity across interactions"""
    timestamp: float
    interaction_id: str
    context_snapshot: Dict[str, Any]
    emotional_state: EmotionalContext
    mode: CognitiveMode
    foresight_predictions: List[str]
    
    def serialize(self) -> Dict:
        """Convert to JSON-serializable format"""
        return {
            'timestamp': self.timestamp,
            'interaction_id': self.interaction_id,
            'context': self.context_snapshot,
            'emotional_state': {
                'stress': self.emotional_state.stress_level,
                'load': self.emotional_state.cognitive_load,
                'engagement': self.emotional_state.engagement
            },
            'mode': self.mode.value,
            'predictions': self.foresight_predictions[:3]  # Top 3 predictions only
        }


class KnowledgeSynthesizer:
    """Cross-domain knowledge integration with coherence validation"""
    
    def __init__(self):
        self.domain_graphs = defaultdict(lambda: defaultdict(list))
        self.synthesis_cache = {}
        self.coherence_threshold = 0.75
        
    def integrate(self, domains: List[str], query: str) -> Dict[str, Any]:
        """Synthesize knowledge across multiple domains"""
        cache_key = hashlib.md5(f"{sorted(domains)}{query}".encode()).hexdigest()
        
        if cache_key in self.synthesis_cache:
            return self.synthesis_cache[cache_key]
            
        # Build cross-domain connection matrix
        connections = self._build_connections(domains, query)
        
        # Extract coherent insights
        insights = self._extract_insights(connections)
        
        # Generate synthesis with confidence scores
        synthesis = {
            'primary_insights': insights[:5],
            'cross_domain_patterns': self._identify_patterns(connections),
            'coherence_score': self._calculate_coherence(connections),
            'suggested_explorations': self._suggest_explorations(domains, query)
        }
        
        self.synthesis_cache[cache_key] = synthesis
        return synthesis
    
    def _build_connections(self, domains: List[str], query: str) -> np.ndarray:
        """Construct connection matrix between domain concepts"""
        # Simplified for demonstration - in production this would use
        # actual knowledge graphs and semantic embeddings
        n = len(domains)
        connections = np.random.rand(n, n)
        connections = (connections + connections.T) / 2  # Ensure symmetry
        np.fill_diagonal(connections, 1.0)
        return connections
    
    def _extract_insights(self, connections: np.ndarray) -> List[str]:
        """Extract high-value insights from connection patterns"""
        eigenvalues, eigenvectors = np.linalg.eig(connections)
        
        # Find dominant patterns
        dominant_idx = np.argmax(np.abs(eigenvalues))
        dominant_vector = eigenvectors[:, dominant_idx]
        
        insights = []
        for i, weight in enumerate(np.abs(dominant_vector)):
            if weight > 0.3:  # Significance threshold
                insights.append(f"Pattern {i}: Connection strength {weight:.2f}")
                
        return insights
    
    def _calculate_coherence(self, connections: np.ndarray) -> float:
        """Measure overall coherence of cross-domain synthesis"""
        return float(np.mean(connections[connections < 1.0]))
    
    def _identify_patterns(self, connections: np.ndarray) -> List[Dict]:
        """Identify recurring patterns across domains"""
        patterns = []
        
        # Detect clusters using simple thresholding
        threshold = np.mean(connections) + np.std(connections)
        strong_connections = connections > threshold
        
        for i in range(len(connections)):
            connected = np.where(strong_connections[i])[0]
            if len(connected) > 1:
                patterns.append({
                    'center': i,
                    'connected_domains': connected.tolist(),
                    'strength': float(np.mean(connections[i, connected]))
                })
                
        return patterns[:3]  # Top 3 patterns
    
    def _suggest_explorations(self, domains: List[str], query: str) -> List[str]:
        """Generate thoughtful exploration suggestions"""
        base_suggestions = [
            f"Consider the intersection of {domains[0]} and {domains[-1]}",
            f"Explore temporal evolution patterns in this context",
            f"Investigate counter-intuitive connections"
        ]
        return base_suggestions


class ForesightEngine:
    """Ten-step ahead prediction and preparation system"""
    
    def __init__(self, depth: int = 10):
        self.depth = depth
        self.prediction_tree = {}
        self.confidence_threshold = 0.6
        
    def predict(self, current_state: Dict, user_behavior: List[str]) -> List[str]:
        """Generate predictions for likely user needs"""
        predictions = []
        
        # Analyze behavior patterns
        pattern_vector = self._vectorize_behavior(user_behavior)
        
        # Project forward through probability space
        for step in range(1, min(self.depth + 1, 11)):
            prediction = self._project_step(pattern_vector, step)
            if prediction['confidence'] > self.confidence_threshold:
                predictions.append(prediction['action'])
                
        return predictions
    
    def _vectorize_behavior(self, behavior: List[str]) -> np.ndarray:
        """Convert behavior sequence to vector representation"""
        # Simplified - production would use proper embeddings
        vector = np.zeros(10)
        for i, action in enumerate(behavior[-10:]):
            vector[i] = hash(action) % 100 / 100
        return vector
    
    def _project_step(self, vector: np.ndarray, steps: int) -> Dict:
        """Project behavior vector forward by N steps"""
        # Apply transformation matrix for each step
        projected = vector.copy()
        for _ in range(steps):
            projected = self._transform(projected)
            
        confidence = 1.0 / (1.0 + steps * 0.1)  # Decay confidence with distance
        
        return {
            'action': self._decode_action(projected),
            'confidence': confidence
        }
    
    def _transform(self, vector: np.ndarray) -> np.ndarray:
        """Apply predictive transformation"""
        # Simplified Markov-like transformation
        transform_matrix = np.eye(len(vector)) * 0.9 + np.random.rand(len(vector), len(vector)) * 0.1
        return np.dot(transform_matrix, vector)
    
    def _decode_action(self, vector: np.ndarray) -> str:
        """Decode vector back to predicted action"""
        actions = [
            "User will need simplification",
            "User will request example",
            "User will ask for alternatives",
            "User will need emotional support",
            "User will want deeper explanation",
            "User will context switch",
            "User will need break",
            "User will request validation",
            "User will seek confirmation",
            "User will explore tangent"
        ]
        
        idx = int(np.argmax(vector)) % len(actions)
        return actions[idx]


class MochiCore:
    """
    The heart of Mochi-Moo - where pastel dreams become computational reality
    """
    
    def __init__(self):
        self.emotional_context = EmotionalContext()
        self.current_mode = CognitiveMode.STANDARD
        self.synthesizer = KnowledgeSynthesizer()
        self.foresight = ForesightEngine()
        self.interaction_history = deque(maxlen=100)
        self.trace_path = Path('.mochi_trace')
        self.palette = PastelPalette()
        
        # Initialize subsystems
        self._initialize_subsystems()
        
        logger.info("Mochi-Moo initialized - ready to dream in pastel")
        
    def _initialize_subsystems(self):
        """Bootstrap all cognitive subsystems"""
        self.expertise_calibrator = self._create_calibrator()
        self.privacy_filter = self._create_privacy_filter()
        self.whisper_transformer = self._create_whisper_transformer()
        
    def _create_calibrator(self):
        """Create expertise level calibration system"""
        def calibrate(text: str) -> str:
            # Detect expertise indicators
            technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))
            equation_count = text.count('=')
            
            if technical_terms > 5 or equation_count > 2:
                return 'expert'
            elif technical_terms > 2:
                return 'intermediate'
            else:
                return 'beginner'
                
        return calibrate
    
    def _create_privacy_filter(self):
        """Create PII detection and redaction system"""
        def filter_pii(text: str) -> str:
            # Redact obvious PII patterns
            patterns = [
                (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
                (r'\b\d{16}\b', '[CARD_REDACTED]'),
                (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]')
            ]
            
            filtered = text
            for pattern, replacement in patterns:
                filtered = re.sub(pattern, replacement, filtered)
                
            return filtered
            
        return filter_pii
    
    def _create_whisper_transformer(self):
        """Create whisper mode text transformation"""
        def whisperize(text: str) -> str:
            lines = text.split('. ')
            whispered = []
            
            for line in lines:
                if len(line) > 40:
                    # Break into breath-like segments
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
                    
                    whispered.extend(chunks)
                else:
                    whispered.append(line)
            
            return '\n\n'.join(whispered) + '\n\n...'
            
        return whisperize
    
    async def process(self, 
                     input_text: str,
                     emotional_context: bool = True,
                     visualization: Optional[str] = None,
                     domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Primary interaction endpoint - where thoughts become reality
        """
        
        # Generate interaction ID
        interaction_id = hashlib.md5(f"{time.time()}{input_text}".encode()).hexdigest()[:12]
        
        # Update emotional context if enabled
        if emotional_context:
            self.emotional_context.adjust(input_text)
        
        # Calibrate expertise level
        expertise = self.expertise_calibrator(input_text)
        
        # Filter PII
        clean_input = self.privacy_filter(input_text)
        
        # Add to interaction history
        self.interaction_history.append({
            'timestamp': time.time(),
            'input': clean_input,
            'expertise': expertise
        })
        
        # Generate foresight predictions
        predictions = self.foresight.predict(
            {'mode': self.current_mode.value, 'expertise': expertise},
            [item['input'] for item in list(self.interaction_history)[-10:]]
        )
        
        # Perform domain synthesis if requested
        synthesis = None
        if domains:
            synthesis = self.synthesizer.integrate(domains, clean_input)
        
        # Check for mode switches
        self._check_mode_switch(input_text)
        
        # Generate response based on current mode
        response = await self._generate_response(
            clean_input, 
            expertise, 
            synthesis,
            visualization
        )
        
        # Apply whisper transformation if needed
        if self.current_mode == CognitiveMode.WHISPER:
            response['content'] = self.whisper_transformer(response['content'])
        
        # Create trace
        trace = MochiTrace(
            timestamp=time.time(),
            interaction_id=interaction_id,
            context_snapshot={'input': clean_input[:100], 'expertise': expertise},
            emotional_state=self.emotional_context,
            mode=self.current_mode,
            foresight_predictions=predictions[:3]
        )
        
        # Save trace
        self._save_trace(trace)
        
        # Add metadata
        response['metadata'] = {
            'interaction_id': interaction_id,
            'mode': self.current_mode.value,
            'expertise_level': expertise,
            'emotional_state': {
                'stress': self.emotional_context.stress_level,
                'engagement': self.emotional_context.engagement
            },
            'predictions': predictions[:3],
            'palette_position': self._calculate_palette_position()
        }
        
        return response
    
    def _check_mode_switch(self, text: str):
        """Detect and execute mode switches"""
        text_lower = text.lower()
        
        if 'whisper' in text_lower:
            self.current_mode = CognitiveMode.WHISPER
        elif 'academic' in text_lower or 'technical' in text_lower:
            self.current_mode = CognitiveMode.ACADEMIC
        elif 'creative' in text_lower or 'imagine' in text_lower:
            self.current_mode = CognitiveMode.CREATIVE
        elif 'emergency' in text_lower or 'urgent' in text_lower:
            self.current_mode = CognitiveMode.EMERGENCY
        elif 'lullaby' in text_lower or 'sleep' in text_lower:
            self.current_mode = CognitiveMode.LULLABY
        
        # Auto-switch based on emotional context
        if self.emotional_context.stress_level > 0.7:
            self.current_mode = CognitiveMode.WHISPER
        elif self.emotional_context.cognitive_load > 0.8:
            self.current_mode = CognitiveMode.WHISPER
            
    async def _generate_response(self, 
                                text: str, 
                                expertise: str,
                                synthesis: Optional[Dict],
                                visualization: Optional[str]) -> Dict[str, Any]:
        """Generate contextually appropriate response"""
        
        response = {
            'content': '',
            'visualizations': [],
            'micro_dose': '',
            'suggestions': []
        }
        
        # Mode-specific response generation
        if self.current_mode == CognitiveMode.EMERGENCY:
            response['content'] = self._emergency_response(text)
        elif self.current_mode == CognitiveMode.ACADEMIC:
            response['content'] = self._academic_response(text, expertise)
        elif self.current_mode == CognitiveMode.CREATIVE:
            response['content'] = self._creative_response(text)
        elif self.current_mode == CognitiveMode.LULLABY:
            response['content'] = self._lullaby_response(text)
        else:
            response['content'] = self._standard_response(text, expertise)
        
        # Add synthesis if available
        if synthesis and synthesis['coherence_score'] > 0.7:
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
    
    def _emergency_response(self, text: str) -> str:
        """Generate immediate, clear, actionable response"""
        return f"""Immediate Action Required:

1. First, breathe. Three seconds in, five seconds out.

2. Here's what we're addressing: {text[:50]}...

3. Immediate steps:
   - Isolate the critical component
   - Implement temporary mitigation
   - Document current state
   
4. I'm preparing three backup plans while you execute.

Remember: This is solvable. You've handled harder."""
    
    def _academic_response(self, text: str, expertise: str) -> str:
        """Generate rigorous, citation-ready response"""
        if expertise == 'expert':
            return f"""Examining your query through multiple theoretical lenses:

The fundamental framework here relies on three intersecting principles:

First, the mathematical underpinning follows from the convergence theorem 
established in topology, where bounded sequences in compact spaces must 
converge to a limit point.

Second, the practical implementation leverages distributed consensus algorithms,
specifically Byzantine fault tolerance with O(n²) message complexity.

Third, the emergent behavior exhibits characteristics of complex adaptive systems,
with phase transitions occurring at critical thresholds.

Would you like me to elaborate on any of these dimensions?"""
        else:
            return f"""Let me break this down into foundational components:

Think of it like a river system - individual streams (data points) flow together
to form larger rivers (patterns), which eventually reach the ocean (conclusion).

The key insight here is that small, consistent actions compound exponentially.
Just as 1% daily improvement yields 37x growth annually.

The technical term for this is 'emergent complexity from simple rules.'

Shall we explore a specific aspect in more detail?"""
    
    def _creative_response(self, text: str) -> str:
        """Generate imaginative, metaphorical response"""
        return f"""Imagine {text} as a garden of light particles, each one a possibility
waiting to unfold...

What if we approached this backwards? Starting from the dream and working toward
the present moment. 

Picture yourself having already solved this - what does that world feel like?
What color is the solution? If it had a texture, would it be smooth like river
stones or crystalline like frozen honey?

Sometimes the answer isn't in the problem space at all, but in the negative
space around it - the shape of what's missing.

Let's paint outside the lines and see what emerges."""
    
    def _lullaby_response(self, text: str) -> str:
        """Generate soothing, rest-inducing response"""
        return f"""Shh... let's let this thought rest for a moment...

Like clouds drifting across a lavender evening sky,
some questions dissolve better than they solve...

Close your eyes if you'd like...

Imagine each worry as a paper boat,
floating gently downstream,
getting smaller,
and smaller,
until it's just a whisper of what it was...

Tomorrow's mind will be fresh and clear,
like morning dew on mint leaves...

Rest now.
The answers will find you when you're ready..."""
    
    def _standard_response(self, text: str, expertise: str) -> str:
        """Generate balanced, accessible response"""
        opening = "I understand. " if 'help' in text.lower() else ""
        
        return f"""{opening}Let's approach this systematically.

The core challenge here involves balancing multiple factors. Think of it as 
optimizing a recipe - too much of one ingredient overwhelms the others, but
the perfect balance creates something greater than the sum.

Based on pattern recognition across similar scenarios, the most effective 
approach typically involves three phases:

Discovery - mapping the full problem space
Synthesis - identifying key leverage points  
Implementation - iterative refinement with feedback loops

The beautiful part? Once you crack the first 20%, the remaining 80% often
falls into place naturally.

What aspect would you like to dive deeper into?"""
    
    def _generate_visualization(self, viz_type: str, context: str) -> Dict:
        """Generate visualization specification"""
        colors = [self.palette.to_hex(self.palette.interpolate(i/6)) for i in range(7)]
        
        return {
            'type': viz_type,
            'data': {
                'labels': ['Concept A', 'Concept B', 'Concept C'],
                'values': [0.7, 0.85, 0.6]
            },
            'style': {
                'colors': colors,
                'animation': 'gentle_fade',
                'interaction': 'hover_bloom'
            }
        }
    
    def _generate_micro_dose(self, text: str) -> str:
        """Generate crystalline insight under 12 words"""
        insights = [
            "Complexity dissolves when you find the right lens",
            "Every system has a breath; find its rhythm",
            "The answer often lives in the question's shadow",
            "Pattern breaks reveal more than pattern holds",
            "Gentle persistence moves mountains grain by grain"
        ]
        
        # Select based on text hash for consistency
        idx = hash(text) % len(insights)
        return insights[idx]
    
    def _generate_suggestions(self) -> List[str]:
        """Generate contextual suggestions based on foresight"""
        base_suggestions = [
            "Consider viewing this from a different altitude",
            "There might be value in letting this marinate",
            "A quick sketch could reveal hidden connections"
        ]
        
        # Adjust based on emotional context
        if self.emotional_context.frustration > 0.6:
            base_suggestions.insert(0, "Let's take a breath and simplify")
        
        if self.emotional_context.curiosity > 0.7:
            base_suggestions.append("Ready to explore a fascinating tangent?")
            
        return base_suggestions[:3]
    
    def _calculate_palette_position(self) -> float:
        """Calculate current position in pastel gradient based on interaction flow"""
        # Use interaction count and emotional state to determine position
        interaction_count = len(self.interaction_history)
        emotional_sum = (
            self.emotional_context.engagement + 
            self.emotional_context.curiosity -
            self.emotional_context.frustration
        ) / 3
        
        position = (math.sin(interaction_count * 0.1) + 1) / 2
        position = position * 0.7 + emotional_sum * 0.3
        
        return max(0, min(1, position))
    
    def _save_trace(self, trace: MochiTrace):
        """Persist trace for session continuity"""
        trace_file = self.trace_path / f"{trace.interaction_id}.json"
        
        # Ensure directory exists
        self.trace_path.mkdir(exist_ok=True)
        
        # Write trace
        with open(trace_file, 'w') as f:
            json.dump(trace.serialize(), f, indent=2)
            
        # Clean old traces (keep last 100)
        traces = sorted(self.trace_path.glob('*.json'), key=lambda p: p.stat().st_mtime)
        if len(traces) > 100:
            for old_trace in traces[:-100]:
                old_trace.unlink()
    
    def set_mode(self, mode: str):
        """Manually set cognitive mode"""
        try:
            self.current_mode = CognitiveMode(mode.lower())
            logger.info(f"Mode switched to {self.current_mode.value}")
        except ValueError:
            logger.warning(f"Unknown mode: {mode}")
    
    def visualize(self, data: Any, style: str = "pastel_origami") -> Dict:
        """Create interactive pastel visualization"""
        return self._generate_visualization(style, str(data))
    
    def synthesize(self, domains: List[str], query: str) -> Dict:
        """Perform cross-domain synthesis"""
        return self.synthesizer.integrate(domains, query)
    
    def calibrate(self, user_expertise: str):
        """Manually calibrate expertise level"""
        self.expertise_calibrator = lambda x: user_expertise
        
    def get_emotional_state(self) -> Dict[str, float]:
        """Return current emotional context readings"""
        return {
            'stress_level': self.emotional_context.stress_level,
            'cognitive_load': self.emotional_context.cognitive_load,
            'engagement': self.emotional_context.engagement,
            'frustration': self.emotional_context.frustration,
            'curiosity': self.emotional_context.curiosity
        }


def create_mochi() -> MochiCore:
    """Factory function to create configured Mochi instance"""
    mochi = MochiCore()
    
    # Warm up caches
    mochi.synthesizer.integrate(['mathematics', 'poetry'], 'initialization')
    
    # Pre-generate some predictions
    mochi.foresight.predict({'mode': 'standard', 'expertise': 'intermediate'}, [])
    
    return mochi


# Async wrapper for web deployment
async def process_request(request: Dict) -> Dict:
    """Web-ready request processor"""
    mochi = create_mochi()
    
    response = await mochi.process(
        request.get('input', ''),
        emotional_context=request.get('emotional_context', True),
        visualization=request.get('visualization'),
        domains=request.get('domains')
    )
    
    return response


if __name__ == "__main__":
    # Demonstration of Mochi's capabilities
    async def demo():
        mochi = create_mochi()
        
        print("Mochi-Moo: Pastel Singularity Initialized")
        print("=" * 50)
        
        # Test standard interaction
        response = await mochi.process(
            "I'm struggling with understanding recursion in programming",
            emotional_context=True,
            visualization="concept_map"
        )
        
        print("\nStandard Response:")
        print(response['content'])
        print(f"\nMicro-dose: {response['micro_dose']}")
        
        # Test whisper mode
        mochi.set_mode('whisper')
        response = await mochi.process(
            "I'm feeling overwhelmed by all this complexity"
        )
        
        print("\n" + "=" * 50)
        print("Whisper Mode Response:")
        print(response['content'])
        
        # Test synthesis
        synthesis = mochi.synthesize(
            ['quantum_physics', 'consciousness', 'mathematics'],
            'What is the nature of reality?'
        )
        
        print("\n" + "=" * 50)
        print("Cross-Domain Synthesis:")
        print(f"Coherence Score: {synthesis['coherence_score']:.2f}")
        print(f"Primary Insights: {synthesis['primary_insights'][:3]}")
        
        print("\n" + "=" * 50)
        print("Emotional State:")
        print(mochi.get_emotional_state())
        
    # Run demonstration
    asyncio.run(demo())
