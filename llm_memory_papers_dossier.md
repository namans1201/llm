# LLM Memory, Long Context, RAG, and Agentic Systems: Reading Dossier

Date: 2026-06-04

## Executive Takeaway

The papers converge on one core lesson: "more context" is not the same as "memory." Human-like AI memory needs several interacting systems:

1. **Working memory**: the active context used for the current task.
2. **Episodic memory**: timestamped experiences, conversations, observations, and outcomes.
3. **Semantic memory**: distilled facts, concepts, relationships, summaries, and knowledge graphs.
4. **Procedural memory**: reusable skills, tools, programs, plans, and policies.
5. **Affective memory**: preferences, emotional salience, interpersonal context, and value-weighted associations.
6. **Reflective/meta-memory**: lessons, self-critiques, retrieval policies, uncertainty estimates, and "what to remember/forget" rules.

The strongest practical architecture is a hybrid:

- A Transformer or long-context backbone for immediate reasoning.
- A MemGPT-like memory controller that pages relevant information into working context.
- RAG/Atlas/RETRO-style retrieval for factual grounding.
- RAPTOR/GraphRAG-style hierarchical and relational semantic memory.
- Generative Agents/MemoryBank-style episodic memory, reflection, importance scoring, and forgetting/reinforcement.
- Voyager-style procedural skill memory.
- Reflexion/Self-RAG-style self-critique over retrieval and actions.
- Infini-Attention/Transformer-XL/RMT/LongRoPE-style architectural extensions where streaming or very long raw context is required.

The practical conclusion: build memory as a managed system, not just a vector database.

## Citation and Source Notes

- `2006.11430`, listed as "Graph Memory Networks," is not a graph memory paper. It resolves to **Learning Minimax Estimators via Online Learning**. I exclude it from the memory synthesis except as a citation mismatch.
- "The Abstraction Fallacy" appears as a DeepMind/PhilPapers-hosted consciousness preprint, not an arXiv paper. The public abstract was available; the PhilPapers PDF was blocked by a browser challenge.
- "Nexus," "AI Hippocampus," "ELF," "Delta-Mem," and "Self-Distilled Agentic Reinforcement Learning" all resolve to 2026 arXiv records.
- ACT-R and SOAR are cognitive architecture projects rather than single arXiv papers.

## Taxonomy

### 1. Transformer and Parametric Memory

**Attention Is All You Need** is the root architecture. It replaces recurrence with self-attention, giving every token direct access to every other token inside a fixed window. This is powerful working memory, but not persistent memory. The model's long-term knowledge lives in parameters, while in-session reasoning lives in the context window.

**ELF: Embedded Language Flows** is not primarily a memory paper, but it matters because it explores a non-autoregressive, continuous embedding-space generation paradigm. If language models move toward continuous diffusion/flow generation, future memory systems may retrieve and condition over dense latent states instead of only discrete text chunks.

**Delta-Mem** directly augments a frozen full-attention model with a compact online associative memory state. It is interesting because it sits between pure architectural memory and external retrieval: a tiny learned online state produces low-rank corrections to attention. Its promise is efficient long-running adaptation without expanding the whole context window.

### 2. Long-Context and Infinite-Context Mechanisms

These papers extend the model's ability to process long sequences. They help with working memory, but do not fully solve long-term memory.

**Transformer-XL** introduces segment-level recurrence and relative positional encoding. It caches hidden states across segments, reducing context fragmentation and enabling dependencies beyond the fixed segment.

**Recurrent Memory Transformer** adds special memory tokens that are passed between segments. This gives the model a learned recurrent state without changing the base Transformer much. It is closer to a neural working-memory register than to a retrievable autobiographical memory.

**Ring Attention** distributes blockwise attention across devices so sequences can scale to millions of tokens. This is an infrastructure answer to long context.

**LongRoPE** extends RoPE-based models to very long position ranges, reportedly up to 2M tokens, using non-uniform positional interpolation and progressive extension. It addresses position extrapolation, not memory selection.

**Infini-Attention** combines local masked attention with compressive long-term memory inside the attention layer, enabling streaming-style processing with bounded memory. It is one of the most memory-relevant long-context architectures because it compresses prior context rather than merely expanding the window.

Design lesson: long context is useful for dense, contiguous evidence, but memory systems still need salience, consolidation, forgetting, retrieval, and provenance.

### 3. Retrieval and Semantic Memory

**RAG** formalizes the modern parametric plus non-parametric memory pattern. A generator is paired with a dense index, retrieving passages from Wikipedia and conditioning generation on them. RAG improves factuality, specificity, and updateability compared with parametric-only generation.

**RETRO** scales retrieval-augmented language modeling to trillions of tokens. It retrieves chunks based on local similarity and uses cross-attention over retrieved neighbors. It shows retrieval can reduce the need to store all knowledge in parameters.

**Atlas** shows that retrieval-augmented models can perform strongly in few-shot knowledge-intensive tasks. Its important contribution is not just retrieval, but pretraining a retriever-generator system and showing the document index can be updated.

**Self-RAG** teaches the model to decide when to retrieve, critique retrieved passages, and critique its own output using reflection tokens. This is a major step from passive retrieval to controlled retrieval.

**RAPTOR** builds a recursive tree of summaries over documents. It supports retrieval at multiple abstraction levels, from local chunks to high-level summaries. This is highly relevant to semantic memory consolidation.

**GraphRAG** builds an entity graph and community summaries, then answers global questions by summarizing across graph communities. It addresses a failure mode of ordinary RAG: questions about the whole corpus, not just a few matching chunks.

**LONGMEM** augments LMs with long-term memory retrieval and cached historical representations. Its main place in the landscape is external memory for "infinite context approximation" without stuffing everything into the prompt.

Design lesson: vector retrieval handles local semantic similarity; hierarchical summaries handle abstraction; graphs handle entities, relationships, and global sensemaking; self-reflection handles retrieval control.

### 4. Agentic Memory

**ReAct** interleaves reasoning traces and actions. It is not a long-term memory system by itself, but it establishes the action loop: think, act, observe, repeat. Memory systems attach naturally to this loop.

**Reflexion** adds verbal reinforcement. Agents reflect on trial feedback, store the reflection in episodic memory, and use it in later attempts. It converts scalar/environment feedback into natural-language learning.

**Generative Agents** stores an agent's observations in a memory stream, retrieves memories by recency, relevance, and importance, periodically reflects to form higher-level beliefs, and plans future behavior. This is one of the most important human-like memory architectures in the list.

**MemoryBank** targets long-term conversational companionship. It maintains user memories, updates them over time, models personality, and uses Ebbinghaus-inspired forgetting/reinforcement. It is especially relevant for personal AI assistants.

**MemGPT** treats the LLM context window like RAM and external storage like disk. It gives the model functions to page memory in/out and edit persistent stores. This is one of the clearest practical blueprints for virtual context management.

**Voyager** implements procedural memory through a growing library of executable skills. The agent explores, writes code for skills, stores successful skills, retrieves and composes them later. This is memory as reusable competence, not just remembered text.

**AutoGen** focuses on multi-agent conversation frameworks. Its memory relevance is orchestration: roles, tool use, dialogue state, and collaborative problem solving across agents.

**Nexus** applies agentic decomposition to time-series forecasting. Its memory lesson is that agents can divide a problem into specialized roles: macro temporal patterns, micro fluctuations, external context, and final synthesis.

**SDAR** addresses post-training for multi-turn agents. It keeps RL as the optimization backbone while using gated self-distillation for token-level guidance. Its relevance is that agent memory/action systems need stable training signals across long horizons.

Design lesson: agent memory is not only recall. It is the substrate for planning, self-improvement, skill reuse, personalization, and multi-step environmental interaction.

### 5. Human-Inspired, Emotional, and Cognitive Memory

**The AI Hippocampus** is a survey that organizes memory in LLMs and multimodal LLMs into implicit, explicit, and agentic memory. It is useful as the umbrella map for the whole field.

**Emotional Chatting Machine** uses emotion categories, internal emotion state, and an external emotion vocabulary to generate emotionally consistent dialogue. It predates modern LLM assistants but highlights that affect can be represented as state, conditioning, and vocabulary bias.

**ACT-R** and **SOAR** matter because they treat cognition as interacting memory systems and production rules. They point toward architectures where declarative memory, procedural memory, goals, and perception/action loops are explicit.

**What the F*ck Is AGI?** frames intelligence as adaptation under constraints, emphasizing search, approximation, hybrid systems, and sample/energy efficiency. For memory systems, the implication is that AGI requires adaptive memory, not static recall.

**The Abstraction Fallacy** is about consciousness and computation rather than LLM memory. Its useful caution is conceptual: behavioral simulation, functional organization, and physical instantiation are not the same thing. It does not change the engineering case for memory, but it warns against overclaiming human equivalence.

## Paper-by-Paper Notes

### Attention Is All You Need

- Introduces the Transformer: self-attention plus feed-forward layers, no recurrence or convolution.
- Memory role: fixed-window working memory.
- Limitation: quadratic attention cost and no persistence beyond context or parameters.
- Relevance: every later paper either extends, retrieves around, or controls this architecture.

### Transformer-XL

- Adds segment-level recurrence and relative positional encodings.
- Solves context fragmentation from fixed-length segments.
- Memory type: cached hidden-state recurrence.
- Best seen as extended working memory, not durable episodic memory.

### Recurrent Memory Transformer

- Adds learned memory tokens passed between segments.
- Lets the model learn what to store across segments.
- More flexible than raw hidden-state caching.
- Still limited by memory token capacity and training distribution.

### LongRoPE

- Extends positional encoding to very long context windows.
- Key problem solved: position extrapolation and catastrophic values at unseen positions.
- Useful for long documents, logs, and codebases.
- Does not solve which information should be remembered after the context is gone.

### Ring Attention

- Makes very long attention feasible by distributing blockwise computation across devices.
- Key contribution is systems-level scaling.
- Useful when raw sequence continuity matters.
- Expensive compared with retrieval for most assistant memory.

### Infini-Attention

- Adds compressive memory into attention with bounded compute.
- Combines local attention and long-term linear attention.
- Strong candidate for streaming agents and long-running sessions.
- The compressed state is efficient but less interpretable than external memory.

### RAG

- Combines parametric generation with a non-parametric vector index.
- Enables updateable knowledge and provenance.
- Best for factual, local, source-grounded recall.
- Weakness: naive retrieval may retrieve irrelevant passages or miss global context.

### RETRO

- Uses retrieval from a huge token database during language modeling.
- Demonstrates retrieval as an alternative to simply scaling parameters.
- Strong for factual associations and language modeling.
- Requires large infrastructure and carefully chunked retrieval.

### Atlas

- Retrieval-augmented few-shot model for knowledge-intensive tasks.
- Shows smaller retrieval-augmented models can beat much larger parametric models on some QA settings.
- Important for systems where knowledge changes and indexes can be updated.

### Self-RAG

- Trains the model to retrieve on demand and critique retrieved evidence and its own generations.
- Adds controllability through reflection tokens.
- Important because retrieval should be a learned decision, not a fixed always-on operation.

### RAPTOR

- Recursively clusters and summarizes text into a tree.
- Retrieves across abstraction levels.
- Excellent fit for semantic memory consolidation: raw episodes become summaries, summaries become themes.

### GraphRAG

- Builds entity graphs and community summaries from source text.
- Better than ordinary RAG for global questions over a corpus.
- Good fit for semantic memory involving people, places, projects, dependencies, and relationships.

### LONGMEM

- Adds long-term memory retrieval to language models.
- Useful framing: external memory as approximation to infinite context.
- Less directly agentic than MemGPT or Generative Agents, but important for model-level memory augmentation.

### MemGPT

- Treats context as RAM and external memory as disk.
- Gives the model explicit tools to page, edit, and search memory.
- Separates archival memory from active context.
- One of the best practical patterns for a personal assistant with persistent memory.

### Generative Agents

- Memory stream stores observations in natural language.
- Retrieval uses recency, relevance, and importance.
- Reflection synthesizes higher-level beliefs from raw memories.
- Planning turns memories/reflections into future behavior.
- Especially strong for episodic, social, and narrative continuity.

### MemoryBank

- Maintains long-term user memory in conversation.
- Uses forgetting/reinforcement inspired by the Ebbinghaus forgetting curve.
- Tracks user personality/preferences.
- Strong fit for companion/assistant personalization, though evaluation is less rigorous than core retrieval benchmarks.

### Reflexion

- Stores lessons from failures as natural-language reflections.
- Treats reflection as a semantic gradient.
- Useful for agents that repeat tasks, use tools, code, or interact with environments.
- Limitation: depends on quality of self-evaluation or external feedback.

### ReAct

- Interleaves reasoning and acting.
- Reduces hallucination by grounding reasoning in observations.
- Provides the loop into which memory retrieval and writing should be inserted.

### Voyager

- Stores executable skills as procedural memory.
- Uses curriculum, environment feedback, and self-verification.
- Demonstrates compositional reuse and reduced catastrophic forgetting.
- Key insight: mature agents need skill memory, not just factual memory.

### AutoGen

- Multi-agent conversation framework.
- Memory relevance is coordination: role state, dialogue history, tool traces, and cross-agent communication.
- Useful when one agent should not carry every specialization itself.

### Nexus

- Multi-agent framework for time-series forecasting.
- Decomposes forecasting into specialized stages and integrates contextual information.
- Shows how agentic systems can combine numerical memory, textual context, and role-specific reasoning.

### SDAR

- Stabilizes multi-turn agent training by gating self-distillation while keeping RL primary.
- Relevant to memory because long-horizon agents need stable credit assignment over actions, observations, and retrieved context.

### Delta-Mem

- Adds a compact online associative memory state to a frozen backbone.
- Uses delta-rule updates and attention corrections.
- Promising for efficient online adaptation.
- Open question: interpretability and long-term persistence compared with explicit external stores.

### AI Hippocampus

- Survey of memory in LLMs/MLLMs: implicit, explicit, and agentic.
- Useful organizing frame for the whole field.
- Highlights open challenges: capacity, factual consistency, multimodal alignment, benchmarks, and interoperability.

### Emotional Chatting Machine

- Models emotional expression with emotion embeddings, internal emotion state, and external emotion vocabulary.
- Useful ancestor for affective memory.
- Modern version would track emotional salience and user affect over time, not merely condition response style.

### ELF

- Continuous embedding-space flow model for language.
- Not a memory architecture, but relevant to future memory representations.
- Suggests future retrieval may happen over latent trajectories or continuous semantic states.

### What the F*ck Is AGI?

- Argues AGI should be understood through adaptation, search, approximation, and hybrid meta-approaches.
- Memory relevance: an intelligent system must adapt under compute, memory, and energy constraints.
- Supports the case for memory selection, compression, and procedural reuse.

### The Abstraction Fallacy

- Philosophical argument distinguishing simulation from instantiation.
- Not an engineering memory paper.
- Useful caution against conflating human-like memory behavior with human consciousness.

### ACT-R

- Cognitive architecture with declarative memory, procedural rules, goals, and modules.
- Strong conceptual model for separating facts from skills.
- Suggests memory retrieval should have activation, latency, recency, frequency, and utility dynamics.

### SOAR

- Cognitive architecture centered on problem spaces, operators, production rules, and chunking.
- Relevant to agentic planning and skill acquisition.
- Suggests reflective problem solving can compile into faster future behavior.

## Relationships and Tradeoffs

### Long Context vs External Memory

Long context keeps information directly visible but is expensive and often underused. External memory is cheap, updateable, and inspectable, but retrieval can fail. The best systems use both:

- Long context for dense, current task evidence.
- External memory for persistent facts, episodes, and skills.
- A controller to decide what enters context.

### Vector Memory vs Graph Memory

Vector memory is good for semantic similarity. Graph memory is good for relationships and global questions. They should coexist:

- Vector search: "Find memories like this."
- Graph traversal: "How is A related to B?"
- Community summaries: "What is this project/person/topic broadly about?"

### Episodic Memory vs Semantic Memory

Episodic memory preserves what happened. Semantic memory preserves what was learned.

Raw episode: "On June 3, user said they prefer concise final answers."

Semantic consolidation: "User prefers concise final answers unless asking for technical detail."

Both are needed. The semantic statement is useful, but the raw episode provides provenance and reversibility.

### Reflection vs Retrieval

Retrieval recalls evidence. Reflection changes the future policy. Reflexion, Generative Agents, Self-RAG, and Voyager all show that agents improve when they convert experience into lessons, plans, critiques, or skills.

### Procedural Memory vs Declarative Memory

Most RAG systems remember text. Voyager shows that agents also need remembered procedures. For a general assistant, procedural memory includes:

- Reusable workflows.
- Tool usage recipes.
- User-specific automations.
- Code snippets or scripts.
- Successful plans from prior tasks.

### Forgetting

Forgetting is not a defect; it is a requirement. MemoryBank's Ebbinghaus-inspired mechanism points in the right direction. A production system should forget or compress:

- Redundant episodes.
- Low-importance transients.
- Superseded preferences.
- Failed hypotheses.
- Sensitive data beyond retention policy.

## Blueprint for a Human-Like LLM Memory System

### 1. Memory Stores

**Working Context**

- Current conversation and task state.
- Retrieved memories.
- Active tool outputs.
- Current plan.

**Episodic Store**

- Timestamped interactions, observations, decisions, outcomes.
- Rich metadata: user, project, source, emotional tone, importance, privacy, confidence.
- Append-only by default, with later correction records.

**Semantic Store**

- Consolidated facts, preferences, relationships, summaries.
- Hierarchical summaries inspired by RAPTOR.
- Entity graph inspired by GraphRAG.

**Procedural Store**

- Skills, workflows, tool recipes, code modules, plans that worked.
- Versioned, tested, and tagged by applicability.
- Voyager-style retrieval by task similarity.

**Affective Store**

- Emotional salience, user preferences, sensitive topics, interpersonal context.
- Should be conservative and privacy-aware.
- Used to shape response appropriateness, not to manipulate.

**Reflective Store**

- Lessons learned from success/failure.
- Self-critiques.
- Retrieval mistakes.
- Planning improvements.

### 2. Memory Write Pipeline

1. Observe event.
2. Decide whether it is worth storing.
3. Classify type: episodic, semantic, procedural, affective, reflective.
4. Score importance, confidence, sensitivity, expected reuse.
5. Store raw episode if needed.
6. Update semantic summaries or graph if stable.
7. Trigger reflection after important events or repeated patterns.
8. Schedule decay, review, or consolidation.

### 3. Memory Retrieval Pipeline

1. Interpret task and user intent.
2. Decide retrieval need, Self-RAG style.
3. Query multiple stores:
   - vector episodic search,
   - semantic summaries,
   - graph relationships,
   - procedural skill search,
   - user preference records.
4. Rank by relevance, recency, importance, confidence, source quality, and privacy.
5. Compress retrieved items before inserting into context.
6. Cite or preserve provenance internally.
7. Critique retrieved evidence before using it.

### 4. Consolidation

Use scheduled and event-triggered consolidation:

- After each task: save outcome and useful lessons.
- Daily/weekly: summarize episodes into semantic facts.
- After repeated memories: promote stable preferences.
- After contradictions: keep both claims and ask/resolve when needed.
- After skill success: update procedural memory.

### 5. Planning Loop

A strong agent loop should look like:

1. Understand goal.
2. Retrieve relevant memory.
3. Plan.
4. Act.
5. Observe result.
6. Reflect on errors or surprises.
7. Update memory.
8. Continue or finish.

This is ReAct plus MemGPT plus Reflexion.

### 6. Evaluation

Evaluate memory systems on:

- Factual recall over long intervals.
- Preference consistency.
- Ability to update beliefs when facts change.
- Resistance to irrelevant retrieved context.
- Source/provenance accuracy.
- Task improvement from procedural memory.
- Privacy and deletion compliance.
- Emotional appropriateness without over-personalization.
- Cross-session planning continuity.
- Graceful forgetting/compression.

## Recommended Learning Path

1. Attention Is All You Need
2. Transformer-XL
3. RAG
4. RETRO
5. Atlas
6. ReAct
7. Reflexion
8. Generative Agents
9. MemGPT
10. MemoryBank
11. LONGMEM
12. Voyager
13. Self-RAG
14. RAPTOR
15. GraphRAG
16. Infini-Attention
17. LongRoPE and Ring Attention
18. Delta-Mem
19. AI Hippocampus
20. ACT-R and SOAR
21. Emotional Chatting Machine
22. Nexus and SDAR
23. AGI and Abstraction Fallacy papers for framing/caveats

## Highest-Value Papers for Building a Real System

If the goal is a practical AI assistant with human-like long-term memory, prioritize:

1. **MemGPT**: memory hierarchy and virtual context management.
2. **Generative Agents**: memory stream, reflection, planning.
3. **MemoryBank**: long-term conversational personalization and forgetting.
4. **RAG / Atlas / RETRO**: factual external memory foundations.
5. **Self-RAG**: deciding when to retrieve and how to critique evidence.
6. **RAPTOR**: hierarchical semantic consolidation.
7. **GraphRAG**: relationship/global memory.
8. **Reflexion**: learning from experience through verbal memory.
9. **Voyager**: procedural skill library.
10. **Infini-Attention / Transformer-XL / RMT**: architectural support for streaming or long sequences.
11. **AI Hippocampus**: survey-level map and open problems.

## Open Problems

- **Memory selection**: deciding what should be stored without saving everything.
- **Memory truth maintenance**: updating, correcting, and contradicting old memories.
- **Temporal reasoning**: knowing when a memory was true and whether it still applies.
- **Memory privacy**: retention, deletion, consent, and sensitive inference.
- **Retrieval robustness**: avoiding overreliance on irrelevant memories.
- **Evaluation**: lack of realistic long-term benchmarks for personal assistants.
- **Multimodal memory**: integrating images, audio, documents, locations, and actions.
- **Procedural transfer**: deciding when a learned skill applies to a new task.
- **Emotional memory boundaries**: using affective context helpfully without manipulation.
- **Scalable consolidation**: turning millions of episodes into usable semantic memory.
- **Agent training**: credit assignment across long action trajectories and retrieved context.

## Bottom Line

The state of the art is moving from "LLM plus vector DB" toward managed cognitive systems. The most credible path to human-like AI memory is not a single model or paper. It is a layered architecture:

- Transformer working memory.
- Retrieval-grounded semantic memory.
- Episodic logs with reflection.
- Hierarchical and graph consolidation.
- Procedural skill libraries.
- Emotional salience and preference tracking.
- A controller that manages what to retrieve, write, forget, and act on.

MemGPT supplies the operating-system metaphor; Generative Agents supplies the episodic/reflection/planning loop; MemoryBank supplies personalization and forgetting; RAPTOR and GraphRAG supply semantic consolidation; Voyager supplies procedural memory; Self-RAG and Reflexion supply self-critique. Together, they form the best current blueprint for a human-like LLM memory architecture.
