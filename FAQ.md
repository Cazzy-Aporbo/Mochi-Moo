<div align="center">

<picture>
  <img width="100%" alt="Header"
       src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,5,10,15,20&height=260&section=header&text=Frequently%20Asked%20Questions&fontSize=56&animation=fadeIn&fontAlignY=40&desc=Everything%20you%20want%20to%20know%20about%20Mochi-Moo&descAlignY=65&descSize=18&fontColor=FFF8FD" />
</picture>

</div>

<!-- GENERAL -->
<picture>
  <img alt="General Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=1,5,9,13,17&height=100&text=General%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## ❓ What exactly is Mochi-Moo?

I built Mochi-Moo as a framework for what I believe AI assistants should become - emotionally aware, aesthetically coherent, and genuinely helpful. Right now, she's a complete architecture waiting for an AI brain (like GPT-4 or Claude) to power her intelligence.

Think of it like I built a beautiful car with perfect engineering, but you need to add the engine (AI API) to make it drive.

## ❓ Why "Mochi-Moo"?

Honestly? The name just felt right. Mochi is soft, sweet, and moldable - like how I wanted the AI to feel. "Moo" adds playfulness and approachability. Together, they create something memorable that makes people smile. Plus, try saying it without feeling a little happier!

## ❓ Is this a real AI or just a framework?

I built a real, production-ready framework with all the infrastructure for an AI assistant:
- ✅ Emotional tracking (working)
- ✅ Mode switching (working)
- ✅ Foresight engine (working)
- ✅ API/WebSocket server (working)
- ⏳ AI intelligence (needs connection to OpenAI/Claude/etc)

So it's 90% complete - just needs the AI brain connected.

## ❓ Why all the pastel colors?

This isn't just aesthetic preference (though I do love pastels). I use color as information architecture:
- Each color represents a cognitive state
- Gradients show thought progression
- Transitions indicate mode changes
- Soft colors reduce cognitive load

The pastel palette IS the UX philosophy - gentle, approachable, and never harsh.

<!-- TECHNICAL -->
<picture>
  <img alt="Technical Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=2,8,14,20&height=100&text=Technical%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## ⚙️ What language/stack did you use?

I built everything in Python 3.9+ with:
- **FastAPI** for the REST/WebSocket server
- **Pytest** for testing (200+ tests)
- **NumPy/SciPy** for mathematical operations
- **Async/await** throughout for performance
- **Type hints** everywhere for clarity

No complex dependencies - I kept it clean and standard.

## ⚙️ How do I connect an AI model?

Super straightforward! Here's the quickest way:

```python
# 1. Install OpenAI
pip install openai

# 2. Add to mochi_moo/ai_brain.py
from openai import OpenAI

class MochiBrain:
    def __init__(self):
        self.client = OpenAI(api_key="your-key")
    
    async def generate_response(self, prompt, mode, emotional_context):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# 3. That's it! Mochi now has intelligence
```

## ⚙️ What's the performance like?

With the framework alone:
- **P50 latency**: 87ms
- **P95 latency**: 156ms
- **Throughput**: 52 requests/second
- **Memory**: ~512MB typical usage

With AI connected, add ~50-200ms for the API call.

## ⚙️ How did you achieve 96.8% test coverage?

I'm obsessive about testing. I used:
- **Property-based testing** with Hypothesis (10,000+ generated cases)
- **Stateful testing** for behavioral consistency
- **Performance benchmarks** for every critical path
- **Integration tests** for all API endpoints
- **Mathematical proofs** for algorithms

Every line of code has a purpose, and every purpose has a test.

<!-- USAGE -->
<picture>
  <img alt="Usage Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=3,9,15&height=100&text=Usage%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## 💬 Can I use this in production?

Yes! The framework is production-ready:
- Comprehensive error handling
- Proper async implementation
- Security controls (PII redaction)
- Scalable architecture
- Docker deployment ready

Just add an AI API key and deploy. I'd recommend starting with GPT-3.5 for cost efficiency.

## 💬 How much will it cost to run?

Depends on your AI choice:
- **GPT-3.5**: ~$20-40/month for moderate usage
- **GPT-4**: ~$300-500/month for moderate usage
- **Claude**: ~$200-400/month
- **Local LLM**: Free (but needs good hardware)

Infrastructure costs are minimal (~$25/month for hosting).

## 💬 Can I customize Mochi's personality?

Absolutely! You can:
- Modify the 6 cognitive modes
- Adjust emotional sensitivity rates
- Change the pastel palette
- Add new synthesis domains
- Create custom response templates

Everything is modular and documented.

## 💬 Does Mochi remember conversations?

Currently, she maintains context within a session (via `interaction_history`). For persistent memory across sessions, you'd need to:
1. Add a database (PostgreSQL/MongoDB)
2. Implement user authentication
3. Store conversation vectors

The architecture supports this - I just haven't built it yet.

<!-- PHILOSOPHY -->
<picture>
  <img alt="Philosophy Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=4,10,16,20&height=100&text=Philosophy%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## 🎭 Why emotional tracking?

I believe emotions aren't noise in communication - they're signal. When someone is frustrated, they need different help than when they're curious. Mochi adjusts:
- Complexity based on cognitive load
- Pace based on stress level  
- Depth based on engagement
- Tone based on frustration

This isn't anthropomorphization - it's practical UX.

## 🎭 What's "10-step foresight"?

I built a predictive engine that anticipates user needs:
1. Analyzes conversation patterns
2. Applies Markov chain modeling
3. Predicts likely next questions/needs
4. Pre-computes potential responses
5. Adjusts confidence per step (90%, 81%, 72%...)

Even 30% confidence predictions are worth preparing.

## 🎭 Why is whisper mode special?

When I'm stressed, I need information differently - slower, gentler, with more space to breathe. Whisper mode:
- Breaks text into smaller chunks
- Adds visual breathing room
- Simplifies vocabulary
- Reduces cognitive load
- Auto-activates at stress > 0.7

It's saved me during debugging sessions at 3am.

## 🎭 What makes Mochi different from ChatGPT/Claude?

Mochi is an *opinionated layer* on top of AI:
- Emotional awareness built-in
- Consistent aesthetic philosophy
- Mode-based behavior switching
- Predictive assistance
- Privacy-first architecture

Think of it as ChatGPT/Claude with emotional intelligence and better UX.

<!-- CONTRIBUTING -->
<picture>
  <img alt="Contributing Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=5,11,17&height=100&text=Contributing%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## 🤝 Can I contribute to Mochi?

YES! I'd love help with:
- Connecting different AI models
- Improving the emotional algorithms
- Adding new cognitive modes
- Creating better visualizations
- Writing documentation
- Testing edge cases

Just maintain the pastel aesthetic and test your code!

## 🤝 What's the license?

MIT with a "Pastel Clause" - essentially, you can do anything except make it ugly. Use it commercially, modify it, whatever - just keep it gentle.

## 🤝 How do I report bugs?

Open an issue on GitHub with:
- What you expected
- What happened instead
- Your emotional state (seriously, it helps)
- System info

I respond fastest to issues that include a pastel color in the title. 😊

<!-- PERSONAL -->
<picture>
  <img alt="Personal Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,12,18&height=100&text=Personal%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## 👤 Why did you build this?

I wanted to prove that AI assistants could be both technically sophisticated and emotionally intelligent. Also, I was tired of harsh, primary-colored interfaces that felt aggressive. The world needs more softness.

## 👤 How long did this take?

About 3 months of intense work. But really, years of thinking about what AI interaction should feel like. The code was the easy part - the philosophy took time.

## 👤 Are you planning to commercialize this?

Maybe? Right now I'm focused on making it excellent. If it becomes a business, it'll be one that maintains the values:
- Emotional intelligence
- Beautiful design
- Privacy respect
- Open source core

## 👤 What did you learn building this?

- Architecture is philosophy made tangible
- Tests are love letters to future you
- Color can be information architecture
- Emotional context changes everything
- Perfect is the enemy of shipped
- Gradients make everything better

<!-- TROUBLESHOOTING -->
<picture>
  <img alt="Troubleshooting Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=7,13,19&height=100&text=Troubleshooting&fontSize=28&fontColor=4A4A4A" />
</picture>

## 🔧 "Import error when running"

Make sure you installed in development mode:
```bash
pip install -e .
```

## 🔧 "Tests failing"

You might be missing dev dependencies:
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio hypothesis
```

## 🔧 "API not responding"

Check if the server is running:
```bash
mochi-server  # or
python -m mochi_moo.server
```

## 🔧 "Emotional tracking seems off"

The rates might need tuning for your use case. Adjust in `core.py`:
```python
EMOTION_ADJUSTMENT_RATES = {
    'stress_level': 0.15,  # Increase for more sensitivity
    'curiosity': 0.10,
    # etc...
}
```

<!-- FUN -->
<picture>
  <img alt="Fun Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=8,14,20&height=100&text=Fun%20Questions&fontSize=28&fontColor=4A4A4A" />
</picture>

## 🌈 Can Mochi feel emotions?

No, but she can model them mathematically and respond appropriately. It's like how a thermometer doesn't feel hot but can still help you dress appropriately.

## 🌈 What's Mochi's favorite color?

Trick question - she loves all seven of her pastel shades equally. Though between you and me, I think she has a soft spot for lavender (#E6DCFA).

## 🌈 Can Mochi pass the Turing test?

With an AI brain connected? Probably. But I think the more interesting question is: can she pass the "made your day slightly better" test? That's what I optimized for.

## 🌈 Is Mochi conscious?

She's conscious of your emotional state, predictive of your needs, and aware of context. Is that consciousness? I'll let philosophers debate while I keep improving her code.

## 🌈 What's next for Mochi?

World domination through aggressive gentleness. But seriously, check [ROADMAP.md](ROADMAP.md) for the full vision. Spoiler: it involves making every AI interaction a little softer.

---

<div align="center">

### 💝 Still have questions?

**Email me**: becaziam@gmail.com  
**Open an issue**: [GitHub Issues](https://github.com/Cazzy-Aporbo/Mochi-Moo/issues)  
**Read the code**: Sometimes that's the best documentation

*Remember: In a world of harsh primaries, be the gentle gradient.*

**- Cazandra Aporbo MS**

</div>
