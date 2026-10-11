# Structured Prompt: Build a Human-Like LLM Memory and Thinking System

You are an expert AI systems architect, cognitive scientist, and senior engineer. Use the research synthesis in `llm_memory_papers_dossier.md` as the foundation.

Our goal is to design and build an LLM-based system that can operate without relying on the internet as its primary source of intelligence. The system should be able to think from what it already knows, remember what it experiences, explore freely within safe boundaries, make mistakes, reflect on them, improve over time, and build human-like memory structures.

## Core Objective

Design a practical architecture for an LLM agent with:

1. Working memory for the current task.
2. Episodic memory for lived experiences and past interactions.
3. Semantic memory for facts, concepts, summaries, and relationships.
4. Procedural memory for skills, workflows, and reusable actions.
5. Reflective memory for lessons learned from mistakes.
6. Emotional or salience memory for importance, preference, urgency, and human context.
7. Curiosity and exploration mechanisms.
8. Self-improvement loops based on feedback, failure, and success.

The system should not merely retrieve information. It should reason, test, reflect, consolidate, and improve.

## Research Basis

Use these ideas from the dossier:

- **MemGPT**: context as RAM, external memory as disk, memory paging.
- **Generative Agents**: memory stream, reflection, planning.
- **MemoryBank**: long-term user memory, forgetting, reinforcement.
- **RAG / RETRO / Atlas**: external factual memory and retrieval.
- **Self-RAG**: deciding when to retrieve and critique retrieved evidence.
- **RAPTOR**: hierarchical summaries and memory consolidation.
- **GraphRAG**: entity relationships and global semantic memory.
- **Reflexion**: learning from mistakes through verbal reflection.
- **Voyager**: procedural skill library and lifelong exploration.
- **ReAct**: reason, act, observe loop.
- **Transformer-XL / RMT / Infini-Attention / LongRoPE**: long-context and recurrent working-memory inspiration.
- **ACT-R / SOAR**: cognitive architecture inspiration for declarative memory, procedural rules, goals, and problem solving.
- **AI Hippocampus**: taxonomy of implicit, explicit, and agentic memory.

## Key Design Philosophy

Do not design this as “LLM + vector database.”

Design it as a cognitive memory system around an LLM:

- The LLM is the reasoning core.
- Memory stores preserve experience and knowledge.
- A controller decides what to remember, retrieve, forget, summarize, and act on.
- Reflection turns failure into future improvement.
- Exploration allows the system to discover new knowledge from its environment.
- Procedural memory lets it reuse successful behaviors.

## Required Architecture

Propose an architecture with these modules:

### 1. Reasoning Core

Define:

- Which LLM role exists at the center.
- What it receives in context.
- What it can and cannot decide.
- How it handles uncertainty.
- How it thinks without internet access.

### 2. Working Memory

Define:

- Current task state.
- Current plan.
- Active observations.
- Retrieved memories.
- Temporary scratchpad.
- Context size management.

### 3. Episodic Memory

Define how the system stores:

- Events.
- Conversations.
- User requests.
- Environment observations.
- Actions taken.
- Outcomes.
- Mistakes.
- Timestamps.
- Source/provenance.
- Confidence.

### 4. Semantic Memory

Define how the system builds:

- Facts.
- Concepts.
- User preferences.
- Project summaries.
- Long-term knowledge.
- Hierarchical summaries.
- Contradiction handling.

Use RAPTOR-style recursive summarization where useful.

### 5. Graph Memory

Define:

- Entities.
- Relationships.
- Projects.
- People.
- Goals.
- Tools.
- Dependencies.
- How graph memory helps answer global questions.

Use GraphRAG-style community summaries where useful.

### 6. Procedural Memory

Define how the system stores:

- Skills.
- Workflows.
- Tool-use recipes.
- Successful plans.
- Code snippets.
- Automation patterns.
- Test results.

Use Voyager as inspiration.

### 7. Reflective Memory

Define:

- How the system detects failure.
- How it writes reflections.
- How reflections are stored.
- How future plans use reflections.
- How bad reflections are corrected.

Use Reflexion as inspiration.

### 8. Emotional / Salience Memory

Define:

- Importance scoring.
- User preference sensitivity.
- Emotional tone.
- Urgency.
- Trust.
- Safety.
- What should never be inferred or stored without care.

This should support better judgment, not manipulation.

### 9. Curiosity and Exploration

Define how the system:

- Forms hypotheses.
- Chooses experiments.
- Explores local files, tools, simulations, or environments.
- Learns without internet.
- Knows when to stop exploring.
- Avoids unsafe or destructive exploration.

Use Voyager and scientific reasoning as inspiration.

### 10. Memory Controller

Design the central memory controller.

It must decide:

- What to retrieve.
- What to write.
- What to summarize.
- What to forget.
- What to promote from episodic to semantic memory.
- What to turn into procedural memory.
- What memories are relevant to the current task.
- When to ask the user instead of guessing.

## Required Loops

Describe these loops clearly:

### Task Loop

1. Understand goal.
2. Retrieve relevant memory.
3. Plan.
4. Act.
5. Observe.
6. Reflect.
7. Update memory.
8. Continue or finish.

### Learning Loop

1. Experience something.
2. Score importance.
3. Store episode.
4. Compare with existing memory.
5. Extract lesson.
6. Update semantic/procedural memory.
7. Test future usefulness.

### Mistake Loop

1. Detect failure.
2. Identify cause.
3. Write reflection.
4. Update plan.
5. Retry.
6. Store lesson if useful.

### Consolidation Loop

1. Review old episodes.
2. Merge duplicates.
3. Summarize patterns.
4. Promote stable facts.
5. Decay low-value memories.
6. Preserve provenance.

## Offline Thinking Requirement

Assume there is no internet.

Explain how the system can still:

- Think from prior knowledge.
- Use local memory.
- Use local files.
- Run local experiments.
- Simulate possibilities.
- Ask clarifying questions.
- Learn from user feedback.
- Improve its future behavior.

Also explain what it cannot do without external information.

## Implementation Plan

Produce a build plan in phases:

### Phase 1: Minimal Memory Agent

Build:

- Episodic memory log.
- Semantic memory summaries.
- Basic retrieval.
- Task loop.
- Reflection after failures.

### Phase 2: Better Memory Management

Add:

- Memory importance scoring.
- Forgetting/decay.
- Contradiction handling.
- User preference memory.
- Memory provenance.

### Phase 3: Graph and Hierarchical Memory

Add:

- Entity graph.
- Relationship extraction.
- Project/topic summaries.
- RAPTOR-style memory trees.
- GraphRAG-style global recall.

### Phase 4: Procedural and Skill Memory

Add:

- Skill library.
- Workflow storage.
- Tool-use recipes.
- Success/failure tests.
- Skill retrieval and composition.

### Phase 5: Exploration and Self-Improvement

Add:

- Curiosity engine.
- Hypothesis testing.
- Safe experimentation.
- Long-term reflection.
- Self-evaluation metrics.

## Evaluation

Define benchmarks for:

- Long-term recall.
- Updated belief handling.
- Preference consistency.
- Mistake recovery.
- Procedural reuse.
- Planning improvement.
- Offline reasoning.
- Resistance to irrelevant memories.
- Safe forgetting.
- User trust and transparency.

## Deliverables

Produce:

1. A high-level architecture diagram in text.
2. A module-by-module explanation.
3. Data schemas for each memory type.
4. Retrieval and write policies.
5. Pseudocode for the main agent loop.
6. Pseudocode for memory consolidation.
7. A phased implementation roadmap.
8. Risks and open problems.
9. A minimal prototype design.
10. A list of the most important papers from the dossier mapped to each module.

## Output Style

Be concrete and implementation-oriented.

Avoid vague claims like “the system will understand.”

Instead, explain:

- What data is stored.
- When it is stored.
- How it is retrieved.
- How it changes behavior.
- How mistakes improve future actions.
- How the system thinks when external information is unavailable.

End with the smallest practical prototype that should be built first.
