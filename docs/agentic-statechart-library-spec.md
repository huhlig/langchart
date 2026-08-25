# Agentic Statechart Library

**Status:** Draft 0.5
**Purpose:** Architecture and behavioral specification
**Audience:** Runtime implementers, editor developers, integration developers, and workflow authors

---

## Changelog

| Version | Change |
|---------|--------|
| 0.1 | Initial draft |
| 0.2 | Resolved open questions: agent execution model (opaque actor), guard language (CEL), memory/LLM/MCP adapter traits specified, crate structure finalized, WASM constraints documented, domain-neutral core enforced, parallel event routing clarified, subworkflow ports defined, context resolver plugin model added |
| 0.3 | Resolved remaining open questions: `AgentActor` trait in `langchart-runtime`, CEL opt-in extension whitelist, RON-typed workflow data schema, stable parallel event ordering, `SecretsAdapter` trait with host-map default; renamed LLM adapter crate to `langchart-llm-generic` (OpenAI + Anthropic APIs); added `langchart-model-router` for model routing |
| 0.4 | Implementation complete: all tracks A–I implemented and tested. Trait signatures corrected against implementation (`LlmAdapter::complete` takes only `LlmRequest`; `ContextResolverStage::resolve` takes `run_id: &RunId` not `snapshot: &RunSnapshot`). §6.4 updated with `langchart-artifact-fs`. §11.2 context-stage signature corrected. §14.1 LLM adapter signature corrected. §17 updated to reflect the implemented editor (9 panels, `simulateWorkflow` WASM binding, `npm run build:full` pipeline). Removed `.github/workflows/security.yml` — project is intended for embedding, not standalone CI. |
| 0.5 | Housekeeping: §16.1 updated with reference to machine-readable `docs/workflow-schema.json` (JSON Schema Draft 2020-12). Clippy warnings resolved across `langchart-model`, `langchart-adapters`, `langchart-artifact-fs`, and `langchart-runtime`. Doc-test `ignore` fence in `langchart-context` changed to `text`. `langchart-artifact-fs/README.md` added. |

---

## 1. Abstract

The Agentic Statechart Library (`langchart`) is a Rust library for defining, validating, executing, observing, and
visually authoring long-running agentic workflows as hierarchical statecharts.

The system combines deterministic statechart semantics with bounded agent autonomy. A workflow defines which states may
become active, which events may cause transitions, what information and capabilities are available in each state, and
how agent-produced work becomes durable artifact changes.

Each agentic state provides an execution envelope containing:

- an agent or agent template;
- prompt and model policies;
- a scoped view of workflow and artifact information;
- an allowlisted set of MCP tools and other capabilities;
- resource, iteration, and time limits;
- typed inputs, outputs, and emitted events;
- transition, retry, approval, and failure behavior.

The system is organized into two major components:

1. **Core engine** — owns the statechart model, validation, execution semantics, persistence, adapter traits, and
   integration interfaces. This is the primary deliverable: a library intended for embedding.
2. **Visual editor** — an optional TypeScript component for embedding in a web-based host application (e.g., an
   Obsidian-like knowledge environment). The editor is never required for headless execution.

The engine is designed to be embedded as a library in a larger application rather than deployed as a standalone service.
Visual authoring, storage backends, user authentication, and UI framework choices are the host application's
responsibility.

---

## 2. Design Thesis

An agentic workflow is modeled as a hierarchical, event-driven statechart in which agents perform bounded activities
inside explicitly governed states.

The statechart controls:

- where execution is;
- which operations are currently permitted;
- which information is disclosed;
- which events are accepted;
- how success, failure, interruption, and recovery are handled;
- when proposed changes become authoritative.

The agent controls only the decisions delegated to it within that envelope.

This division provides deterministic orchestration around nondeterministic model behavior. The goal is to provide
a better foundation than LangChain/LangGraph by grounding agentic workflows in formal statechart semantics while
retaining pragmatic adapter abstractions for the messy real world of LLMs, MCP servers, and memory systems.

**Key design decisions:**

- **Agents are opaque actors.** The runtime starts an agent actor and waits for it to emit a declared output event.
  The agent drives its own internal loop (multi-turn ReAct, single-shot, chain-of-thought, etc.). The runtime enforces
  turn and tool-call budgets through the capability broker, not by owning the loop.
- **Guards use CEL.** The Common Expression Language provides deterministic, side-effect-free, serializable guard
  expressions that compile to WebAssembly and are evaluable in both the Rust runtime and the TypeScript editor.
- **All external integrations are adapters.** LLMs, MCP servers, memory systems, artifact stores, checkpoint stores,
  and event sinks are all Rust traits. No concrete integration is mandatory.
- **The core is domain-neutral.** Domain-specific concepts (manuscripts, codices, etc.) belong in reference workflows
  and host applications, not in core types.

---

## 3. Goals

The library MUST:

- Support atomic, compound, parallel, human-interaction, agentic, subworkflow, and final states.
- Support hierarchical workflows that can be collapsed and expanded in the visual editor.
- Provide deterministic transition semantics based on typed events and CEL guards.
- Support durable execution, checkpoints, suspension, resumption, cancellation, and replay.
- Restrict agent access to explicitly granted tools, resources, information, and operations.
- Treat prompts, context selection, model choice, and MCP exposure as versioned workflow configuration.
- Separate control state from workflow data, durable artifacts, context views, and agent scratch data.
- Support versioned artifacts and proposal-based modification rather than uncontrolled mutation.
- Make every model call, tool call, transition, proposal, approval, and artifact update observable.
- Allow workflows to be authored visually without making the visual layout authoritative execution data.
- Use a language-neutral interchange format with explicit schema versions.
- Permit headless execution without the visual editor.
- Compile the model/validation layer to WebAssembly for editor-side use without duplication of validation logic.
- Be embeddable as a library with no mandatory runtime service, database, or UI dependency.

The visual editor MUST:

- Present a simple top-level workflow while allowing compound states to be opened progressively.
- Prevent or clearly identify invalid statecharts before execution.
- Expose simple configuration first and advanced configuration on demand.
- Display live execution state, event flow, context composition, tool use, and artifact proposals.
- Preserve unknown fields when editing documents produced by a newer compatible schema version.
- Be independently optional — the runtime MUST function fully without it.

---

## 4. Non-Goals

The initial system is not intended to:

- Allow agents to arbitrarily rewrite the workflow topology during execution.
- Treat free-form agent prose as an implicit transition instruction.
- Provide an unrestricted conversational swarm in place of explicit orchestration.
- Make a specific LLM provider, MCP implementation, document store, or UI framework mandatory.
- Require every state to contain an agent.
- Store private model reasoning or depend on hidden chain-of-thought.
- Own user authentication, authorization, or multi-tenancy — these are host application concerns.
- Replace a domain-specific writing editor, source repository, or content-management system.
- Provide a standalone deployed service — this is an embeddable library.

Dynamic work SHOULD initially be represented as bounded task data executed by a worker state or child workflow,
rather than runtime mutation of the statechart itself.

---

## 5. Terminology

### 5.1 Statechart

A hierarchical graph of states and event-driven transitions. A statechart may contain nested compound states and
orthogonal parallel regions. The semantics are inspired by Harel statecharts and SCXML but are not required to be
SCXML-conformant.

### 5.2 State

A durable control location that may execute activities, accept events, expose capabilities, and transition to another
state.

### 5.3 Agentic State

A state whose primary activity starts an agent actor. The agent is an opaque async unit that executes its own
internal loop and emits exactly one declared output event when complete. The runtime provides the capability envelope
and context view; the agent decides how to use them.

### 5.4 Agent Actor

The executable unit started by an agentic state. An agent actor may internally perform multi-turn reasoning, tool
calls, retrieval, and memory operations. It is opaque to the runtime except through the capability broker and the
final emitted event.

### 5.5 Agent Definition

A reusable description of an agent's purpose, default instructions, model policy, context policy, tools, result
schema, and behavioral limits. Agent definitions are versioned and referenced by agentic states.

### 5.6 Capability Envelope

The effective set of tools, resources, operations, credentials, limits, and information available while a state is
active. Calculated by the `CapabilityBroker` from layered policies.

### 5.7 Event

A typed occurrence delivered to a workflow instance. Events may originate from states, agents, tools, timers,
humans, integrations, or the runtime itself.

### 5.8 Artifact

A durable, versioned work product managed by the host application through the `ArtifactStore` adapter. The core
library does not define artifact content formats.

### 5.9 Proposal

A structured request to create or modify an artifact. A proposal is not authoritative until validated and committed
by an authorized state or human decision.

### 5.10 Context View

An immutable, reproducible snapshot of the information disclosed to an agent invocation. Produced by the
`ContextResolver` adapter chain and recorded for observability.

### 5.11 Guard

A CEL expression evaluated deterministically against workflow data and event payload. Guards MUST be side-effect
free and MUST NOT invoke an LLM or any external system.

### 5.12 Port

A typed input or output channel on a workflow or subworkflow state. Ports define the data contract between a
caller and a subworkflow invocation.

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Host Application                  │
│  (Obsidian-like app, CLI tool, API server, etc.)    │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Visual Editor│    │     Host Adapters         │  │
│  │ (TypeScript) │    │  LLM, MCP, Memory,        │  │
│  │  (optional)  │    │  Artifacts, Checkpoints   │  │
│  └──────┬───────┘    └────────────┬─────────────┘  │
│         │                         │                 │
│  ┌──────▼─────────────────────────▼─────────────┐  │
│  │              langchart Engine                 │  │
│  │                                               │  │
│  │  ┌──────────────┐  ┌──────────────────────┐  │  │
│  │  │  Model Layer │  │   Runtime Layer       │  │  │
│  │  │  (WASM-safe) │  │   (async, durable)    │  │  │
│  │  └──────────────┘  └──────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 6.1 Model Layer (`langchart-model`)

The model layer owns:

- workflow, state, transition, event, policy, and adapter-trait type definitions;
- schema loading and normalization;
- structural and semantic validation;
- statechart compilation;
- capability calculation (static, pre-execution);
- graph inspection APIs;
- schema migration support;
- CEL guard compilation and static analysis.

The model layer MUST have no dependency on a specific model provider, MCP server, persistence engine, async runtime,
or graphical interface. It MUST compile to WebAssembly.

**WASM constraint:** The model layer MUST NOT use `std::fs`, `std::thread`, `std::net`, or any I/O. All external
data is passed in. This constraint applies from the first commit.

### 6.2 Runtime Layer (`langchart-runtime`)

The runtime layer owns:

- workflow instance creation and lifecycle;
- event processing and transition selection (run-to-completion macro-step);
- state entry, exit, and activity lifecycle;
- parallel-region coordination;
- checkpointing and recovery;
- timers, retries, and timeouts (durable);
- context resolution delegation;
- capability enforcement (runtime);
- proposal validation and artifact commit coordination;
- runtime event streaming and observability;
- cancellation propagation to active actors.

The runtime layer depends on the model layer and on adapter traits. It does NOT depend on any concrete adapter.

### 6.3 Adapter Traits (`langchart-adapters`)

A thin crate of pure trait definitions (no implementations). Every external integration is expressed as one of
these traits. See §14 for the full trait catalog.

### 6.4 Built-in Adapters

Optional crates providing concrete implementations:

| Crate | Provides |
|---|---|
| `langchart-llm-generic` | OpenAI + Anthropic public API `LlmAdapter` with model enumeration |
| `langchart-model-router` | Model routing: policy-driven dispatch to any registered `LlmAdapter` |
| `langchart-mcp-client` | MCP client `McpAdapter` over rmcp stdio transport |
| `langchart-memory-redb` | Embedded key-value memory adapter (redb v2) |
| `langchart-checkpoint-redb` | Embedded checkpoint store (redb v2) |
| `langchart-artifact-fs` | File-system `ArtifactStore` with atomic writes and versioned layout |

These are convenience crates. The engine works with any implementation of the adapter traits.

### 6.5 Visual Editor (`langchart-editor`)

A TypeScript component intended for embedding in a web-based host UI. Uses the WASM build of `langchart-model`
for validation so the editor and runtime apply identical structural rules.

The editor MUST NOT define execution behavior that cannot be represented in the canonical workflow document.
The editor is optional; the runtime is fully functional without it.

---

## 7. State Model

### 7.1 State Types

| Type | Purpose |
|---|---|
| `atomic` | Runs a deterministic activity or waits for an event. |
| `agentic` | Starts an agent actor within a capability and context envelope. |
| `compound` | Contains a nested statechart with an initial child state. |
| `parallel` | Activates two or more orthogonal regions concurrently. |
| `human` | Suspends until an authorized human supplies a decision or data. |
| `subworkflow` | Invokes a separately versioned workflow through typed ports. |
| `final` | Marks completion of a region or workflow. |

The library MAY add specialized convenience states, but these MUST compile into the core state types above.

### 7.2 State Lifecycle

A state may define:

- entry actions (side-effect-free data transformations);
- one primary activity (agent actor, subworkflow invocation, or deterministic function);
- event handlers and transitions;
- exit actions;
- timeout behavior;
- retry policy;
- compensation or recovery transitions.

When a state becomes active, the runtime MUST:

1. calculate its effective capability envelope from layered policies;
2. execute entry actions;
3. checkpoint if required by policy;
4. start the state's activity;
5. accept only events valid for the active configuration.

When leaving a state, the runtime MUST:

1. cancel or complete the active activity according to cancellation policy;
2. execute exit actions;
3. revoke state-scoped capabilities;
4. persist the resulting transition and any emitted events.

### 7.3 Compound States and Progressive Disclosure

A compound state is displayed as a single node at its parent's level in the visual editor. Opening it reveals its
nested statechart.

```
  ┌─────────────────────────────────────────┐
  │              Draft (compound)           │
  │                                         │
  │  [*] → AssembleContext → Write         │
  │         Write → SelfReview             │
  │         SelfReview → Write (revision)  │
  │         SelfReview → Propose (ok)      │
  │         Propose → [*]                  │
  └─────────────────────────────────────────┘
```

Compound states provide the primary mechanism for progressive visual disclosure and reusable workflow composition.

### 7.4 Parallel States

A parallel state activates each of its regions concurrently. Completion behavior MUST be explicit:

| Mode | Meaning |
|---|---|
| `all` | All regions must reach a final state. |
| `any` | First region to reach a final state wins. |
| `quorum(n)` | N regions must complete. |
| `guard(expr)` | CEL expression evaluated after each completion. |
| `manual` | Requires an explicit external termination event. |

**Isolation requirement:** Parallel agent regions MUST receive immutable artifact versions or isolated working
branches. They MUST NOT concurrently mutate the same authoritative artifact.

**Event routing in parallel states:** Events are broadcast to all active regions. Each region processes the event
independently according to its own active configuration. A transition in one region does not automatically trigger
a transition in a sibling region.

**Unhandled events:** An event that has no enabled transition in a region is silently discarded within that region.
It is NOT an error for a parallel region to not handle a broadcast event.

### 7.5 History

Compound and parallel states SHOULD support shallow and deep history. History records which child configuration was
active when the parent was exited, allowing resumption without restarting from the initial state.

History is control-state history only. It does not replace artifact versioning or workflow checkpoints.

### 7.6 Subworkflow Ports

A subworkflow state invokes a separately versioned workflow. Ports define the data contract:

```yaml
id: run_review
type: subworkflow
workflow_ref: content-review@2.1
ports:
  input:
    draft_version: ${workflow.current_draft_version}
    review_scope: full
  output:
    on_completed:
      issues: ${event.payload.issues}
      approved: ${event.payload.approved}
on:
  subworkflow.completed:
    target: consolidate_issues
    guard: ${event.payload.approved == false}
  subworkflow.approved:
    target: commit_artifact
  subworkflow.failed:
    target: recovery
```

Input ports map workflow data expressions to the child workflow's typed input schema. Output ports map the child's
final output event payload back to the parent's workflow data. Port schemas are validated statically.

---

## 8. Transitions, Events, and Guards

### 8.1 Event Envelope

Every runtime event MUST contain a stable envelope:

```json
{
  "event_id": "evt_01JXXX",
  "event_type": "review.completed",
  "workflow_id": "content-review",
  "workflow_version": "2.1.0",
  "run_id": "run_01JXXX",
  "state_id": "review.continuity",
  "timestamp": "2026-07-19T17:00:00Z",
  "correlation_id": "inv_01JXXX",
  "causation_id": "evt_00JXXX",
  "payload": {},
  "schema": "schemas/review-completed@1"
}
```

Events MUST be validated against their declared payload schema before they participate in transition selection.
Invalid events are rejected with an observable error record; they do not silently fail.

**Event channels:** External events (from humans, integrations, timers) and internal events (from agent actors)
flow through the same validated event channel. The event envelope's `correlation_id` and `causation_id` fields
distinguish the causal chain.

### 8.2 Transition Selection

A transition defines:

- triggering event type (required);
- CEL guard expression (optional; absence means always-true);
- target state;
- optional transition actions (side-effect-free data transformations);
- priority (integer, lower = higher priority) where ambiguity is possible.

**Guard evaluation:**

Guards are CEL expressions evaluated against a context object containing:
- `event`: the triggering event envelope and payload;
- `workflow`: current workflow data;
- `state`: state-local data.

Guards MUST be deterministic and side-effect free. A guard MUST NOT invoke an LLM, MCP tool, or any I/O. This is
enforced at compile time by CEL's pure-expression semantics.

An agent may recommend an outcome, but the runtime transitions only on a validated event with a satisfied guard.

**Transition priority:** When multiple transitions are enabled by the same event, the transition with the lowest
priority number is selected. Ties are a validation error and MUST be reported by the validator.

### 8.3 Internal and External Transitions

| Type | Behavior |
|---|---|
| **internal** | Handles an event without exiting and re-entering the source state. Entry/exit actions do not run. |
| **external** | Executes normal exit and entry behavior. Default. |
| **local** | Remains within a compound state hierarchy; does not exit the compound state if the target is a descendant. |

### 8.4 Timers and Retries

Timers MUST be durable. A checkpoint save MUST include all pending timer state. A workflow restart MUST restore
timer state and not silently lose pending timeouts.

Retry policies support:

- `max_attempts`: maximum invocation attempts including the first;
- `delay`: initial delay before the first retry;
- `backoff`: `fixed`, `linear`, or `exponential`;
- `retryable_on`: event types or error classes that permit retry;
- `fallback_model`: alternative model profile for retry attempts;
- `on_exhausted`: target state when all attempts are consumed.

Retries MUST create distinct invocation records correlated to the original activity ID.

### 8.5 Unhandled Events

An event delivered to a state with no matching transition is:

1. propagated upward to the nearest ancestor state that handles it (SCXML-style bubbling);
2. if no ancestor handles it, emitted as an observable `event.unhandled` record;
3. NOT silently discarded at the workflow level.

An `event.unhandled` record at the workflow level MAY be configured as a workflow failure.

---

## 9. Agent Definitions and Agentic States

### 9.1 Agent Execution Model

An agent actor is an opaque async unit. The runtime:

1. constructs the capability envelope and context view;
2. calls `AgentActor::run(invocation, broker)` on the agent;
3. waits asynchronously for the actor to emit exactly one declared output event;
4. validates the emitted event against the declared schema;
5. delivers the event to the event channel.

The agent internally manages its own loop — multi-turn ReAct, single-shot, tool-augmented chain, or any other
pattern. The runtime does not own the loop. The runtime enforces limits through the capability broker:

- **turn limit**: the broker tracks model calls and rejects calls beyond `max_turns`;
- **tool call limit**: the broker tracks tool invocations and rejects calls beyond `max_tool_calls`;
- **timeout**: the runtime cancels the actor future when the wall-clock timeout is reached.

If an agent exceeds a limit, the capability broker returns a typed error. The agent SHOULD emit `activity.failed`
or an equivalent declared failure event. If the actor is cancelled (timeout), the runtime synthesizes an
`activity.cancelled` event.

This model allows existing agent frameworks (LangChain, custom ReAct loops, etc.) to be wrapped as actor
implementations with minimal adapter code.

The `AgentActor` trait is defined in **`langchart-runtime`** (not `langchart-adapters`) because the trait
receives a reference to `CapabilityBroker`, which is a runtime type. Actor implementors add `langchart-runtime`
as a dependency. See §14 for the `AgentActor` trait signature.

### 9.2 Reusable Agent Definition

```yaml
id: content_analyst
version: 1.0.0
description: Analyzes content and produces structured findings.

instructions:
  system: prompts/content-analyst.md

model_policy:
  profile: high_quality
  temperature: 0.2

default_context_policy:
  sources:
    - type: artifact
      selector: current_draft
    - type: memory
      query: ${workflow.topic}
      limit: 10

default_capabilities:
  tools:
    - content.read
    - knowledge.search

output_events:
  - analysis.completed
  - clarification.required
  - context.insufficient
  - activity.failed
```

### 9.3 Agentic State Configuration

An agentic state binds an agent definition to a specific workflow location:

```yaml
id: analyze_content
type: agentic
agent:
  ref: content_analyst@1

input:
  topic: ${workflow.current_topic}

prompt:
  task: Analyze the current content and return structured findings.

context:
  sources:
    - type: artifact
      selector: current_draft
    - type: memory
      query: ${workflow.current_topic}
      limit: 5
  token_budget: 24000

capabilities:
  mcp:
    content_store:
      allow:
        - read_section
        - propose_patch
    knowledge_base:
      allow:
        - search
        - read_entry

limits:
  max_turns: 12
  max_tool_calls: 20
  timeout: 10m

on:
  analysis.completed:
    target: review_findings
    guard: ${event.payload.confidence >= 0.7}
  clarification.required:
    target: await_human
  context.insufficient:
    target: expand_context
  activity.failed:
    target: recovery
```

State-level configuration MAY narrow agent defaults. Expanding permissions beyond the agent definition or parent
state MUST include an explicit `elevate: true` declaration that is flagged during static validation.

### 9.4 Agent Execution Contract

The runtime MUST provide the agent actor with:

- resolved instructions and prompt content (template evaluated against workflow data);
- the immutable context view (assembled by the context resolver chain);
- typed input data (resolved from the state's input expressions);
- only the tools permitted by the effective capability envelope (via the capability broker);
- output event schemas (for self-validation before emission);
- execution limits (turns, tool calls, timeout);
- run, state, invocation, and correlation identifiers.

The agent MUST emit a declared output event. Undeclared or schema-invalid output is an activity failure and MUST
NOT implicitly select a transition. Undeclared output generates an observable `activity.invalid_output` record.

---

## 10. Capability and MCP Model

### 10.1 Capability Resolution

Effective state capabilities are calculated by composing layers from outermost to innermost:

1. deployment policy (host application global policy);
2. workflow policy;
3. parent-state policy (inherited, never widened without `elevate: true`);
4. agent definition defaults;
5. state-specific restrictions or elevations;
6. run-specific authorization (e.g., human-granted temporary permission).

The effective set is the **intersection** of allowed capabilities at each layer, except where an explicit
authorized elevation is declared.

### 10.2 MCP Exposure

The runtime exposes a virtual MCP surface to the agent actor rather than raw MCP server access.

Restrictions may include:

- permitted tool names (allowlist);
- resource URI patterns (regex or glob);
- operation classes: `read`, `propose`, `commit`, `publish`, `delete`;
- argument schema constraints (CEL-validated before forwarding);
- result field redaction;
- per-invocation call budgets;
- state-scoped credentials (injected by the broker, not visible to the agent);
- human confirmation requirements (broker pauses, requests confirmation event, then proceeds).

Capability access MUST be revoked when the state exits or the invocation is cancelled.

### 10.3 Capability Inheritance

Nested states inherit restrictions from their ancestors. A child state MAY narrow inherited permissions without
special authorization. A child state MAY NOT silently widen them.

Any widening MUST declare `elevate: true` at the state level. The validator MUST flag all elevation declarations.
The editor SHOULD visually highlight elevated states.

### 10.4 The Capability Broker as Security Kernel

The `CapabilityBroker` is the system's security kernel. It is the only enforcement point between agent actors and
external systems. Correctness here is more critical than in any other component.

Implementation requirements:

- The broker MUST be a separate crate (`langchart-adapters`) with no dependency on runtime internals.
- Every `invoke` call MUST be logged as an observable event before the call is forwarded.
- The broker MUST be tested with property-based tests covering policy edge cases.
- Capability checks MUST NOT be skippable by agent actors — there is no "trusted" fast path.

---

## 11. Information and Context Model

### 11.1 Information Categories

| Category | Description |
|---|---|
| Control state | The active statechart configuration. |
| Workflow data | Structured variables and state outputs for the current run. |
| Artifact state | Durable, versioned work products (managed by `ArtifactStore`). |
| Context view | Immutable information assembled for one invocation. |
| Agent scratch data | Ephemeral notes and intermediate tool results (not persisted). |
| Long-term memory | Explicitly stored cross-run knowledge (managed by `MemoryAdapter`). |

### 11.2 Context Resolver Chain

Context resolution is a pipeline, not a single function. The `ContextResolver` adapter is a composable chain of
resolver stages:

```
ContextResolverChain
  │
  ├─ ArtifactResolver     — loads specific artifact versions
  ├─ MemoryResolver       — queries the MemoryAdapter
  ├─ WorkflowDataResolver — injects workflow variables
  ├─ TruncationResolver   — enforces token budgets
  └─ RecordingResolver    — records the final view for observability
```

Each resolver stage implements `ContextResolverStage`:

```rust
#[async_trait]
pub trait ContextResolverStage: Send + Sync {
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError>;
}
```

The final `ContextView` records every included source, artifact version, selection rationale, transformation,
exclusion, token count, and the assembled content. This record enables inspectable context and approximate replay.

### 11.3 Memory Adapter

Long-term memory is accessed through the `MemoryAdapter` trait. The adapter abstracts over key-value stores, vector
databases, and hybrid retrieval systems:

```rust
#[async_trait]
pub trait MemoryAdapter: Send + Sync {
    async fn store(&self, record: MemoryRecord) -> Result<MemoryId, MemoryError>;
    async fn search(&self, query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError>;
    async fn get(&self, id: MemoryId) -> Result<Option<MemoryRecord>, MemoryError>;
    async fn delete(&self, id: MemoryId) -> Result<(), MemoryError>;
}
```

Memory scope is determined by the `MemoryRecord`'s scope field:

| Scope | Visibility |
|---|---|
| `run` | Current workflow run only |
| `workflow` | All runs of a workflow ID |
| `agent` | All uses of an agent definition ID |
| `global` | Host application-wide (subject to policy) |

An agent actor MAY write to memory only if `memory.write` is in its capability envelope. Reading memory during
context resolution does not require capability elevation (it is part of the context policy, not agent-driven).

### 11.4 Context Expansion

An agent may emit `context.insufficient` with a structured request for additional information. The statechart
decides whether to expand context, route to a human, invoke another agent, or fail.

Agents MUST NOT autonomously escape their context boundary by calling undeclared retrieval tools.

---

## 12. Artifact and Proposal Model

### 12.1 Versioned Artifacts

Artifacts are managed by the host application through the `ArtifactStore` adapter. The core library defines the
protocol but not the storage format.

Artifacts MUST have stable identities and immutable versions. A workflow invocation reads specific artifact
versions. The library is content-format agnostic — artifacts may be Markdown, JSON, binary blobs, or any other
format the host application uses.

### 12.2 Proposal-Based Mutation

Agentic states SHOULD propose changes rather than directly mutate authoritative artifacts.

A proposal contains:

- target artifact ID and expected base version;
- structured patch or replacement content;
- rationale and supporting evidence;
- originating run ID, state ID, agent ID, and invocation ID;
- validation status;
- required approval class.

An authorized commit operation applies a proposal only when its base-version precondition remains valid. Version
conflicts produce an explicit `proposal.conflicted` event with enough information for a recovery path.

### 12.3 Parallel Proposal Consolidation

When parallel regions each produce proposals on the same artifact, a join state receives all proposals as a
structured collection. The join state's activity (agent or human) is responsible for consolidation:

1. **Non-overlapping patches:** The join agent may apply them sequentially to a common base.
2. **Overlapping patches:** The join agent must resolve the conflict and produce a single consolidated proposal.
3. **Irreconcilable conflicts:** The join agent emits a `conflict.unresolvable` event; the workflow transitions
   to a human decision state.

Consolidation does NOT automatically commit. A separate authorized commit operation is always required.

---

## 13. Execution Semantics

### 13.1 Run-to-Completion Macro-Step

The runtime uses run-to-completion (RTC) semantics for event processing:

1. Dequeue one external or internal event.
2. Validate the event envelope and payload.
3. Determine enabled transitions from the active configuration.
4. Select transitions deterministically (by priority, then source depth).
5. Exit affected states from inner to outer.
6. Execute transition actions (synchronous, side-effect-free).
7. Enter target states from outer to inner.
8. Start eligible activities (agent actors, subworkflow invocations).
9. Persist the new configuration and emitted events.
10. Publish observable runtime updates.

**RTC and long-running activities:** RTC applies to the transition cycle, not to the agent actor's execution.
Agent actors run concurrently with the event loop. While an agentic state is active, the runtime continues
processing other events (e.g., parallel region events, timeout events, cancellation events). An incoming
cancellation event for an active agentic state WILL cancel the actor without waiting for it to complete.

### 13.2 Idempotency and Outbox

External side effects (MCP tool calls, artifact commits) MUST use an outbox or equivalent mechanism:

- Each external call is recorded with a stable idempotency key before being forwarded.
- On checkpoint recovery, recorded-but-not-confirmed calls are retried using the same key.
- Tool adapters SHOULD accept idempotency keys when the underlying system supports them.
- The runtime MUST NOT re-execute already-confirmed external calls on recovery.

Long-running activities MUST have stable invocation IDs that survive checkpoint and recovery.

---

## 14. Adapter Traits

All external integrations are defined as async traits in `langchart-adapters`. The runtime depends only on
these traits. Concrete implementations are provided by built-in or host-application adapter crates.

### 14.1 LLM Adapter

```rust
#[async_trait]
pub trait LlmAdapter: Send + Sync {
    async fn complete(
        &self,
        request: LlmRequest,
    ) -> Result<LlmResponse, LlmError>;
}
```

`LlmRequest` includes: model policy, conversation history (with system prompt as the first message), and tool
definitions. `LlmResponse` includes: content, tool calls, usage tokens, finish reason, and model identifier.

Agent actors call the LLM through `CapabilityBroker::call_llm`, which enforces turn limits, logs the call, and
forwards to the `LlmAdapter`. The capability envelope is checked at the broker layer, not inside the adapter.

### 14.2 MCP Adapter

```rust
#[async_trait]
pub trait McpAdapter: Send + Sync {
    async fn call_tool(
        &self,
        server_id: &ServerId,
        tool_name: &ToolName,
        arguments: serde_json::Value,
        idempotency_key: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, McpError>;

    async fn list_tools(&self, server_id: &ServerId) -> Result<Vec<ToolDefinition>, McpError>;

    async fn read_resource(
        &self,
        server_id: &ServerId,
        uri: &ResourceUri,
    ) -> Result<ResourceContent, McpError>;
}
```

The `CapabilityBroker` wraps `McpAdapter` calls with policy enforcement before forwarding.

### 14.3 Memory Adapter

```rust
#[async_trait]
pub trait MemoryAdapter: Send + Sync {
    async fn store(&self, record: MemoryRecord) -> Result<MemoryId, MemoryError>;
    async fn search(&self, query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError>;
    async fn get(&self, id: MemoryId) -> Result<Option<MemoryRecord>, MemoryError>;
    async fn delete(&self, id: MemoryId) -> Result<(), MemoryError>;
}
```

`MemoryQuery` supports keyword, semantic (vector), and structured filter modes. Implementations choose which
modes they support and return `MemoryError::Unsupported` for others.

### 14.4 Artifact Store Adapter

```rust
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn read(
        &self,
        id: &ArtifactId,
        version: Option<&ArtifactVersion>,
    ) -> Result<ArtifactContent, ArtifactError>;

    async fn propose(
        &self,
        proposal: ArtifactProposal,
    ) -> Result<ProposalId, ArtifactError>;

    async fn commit(
        &self,
        proposal_id: &ProposalId,
        expected_base: &ArtifactVersion,
    ) -> Result<ArtifactVersion, ArtifactError>;

    async fn list_proposals(
        &self,
        artifact_id: &ArtifactId,
    ) -> Result<Vec<ProposalSummary>, ArtifactError>;
}
```

### 14.5 Checkpoint Store Adapter

```rust
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn save(&self, snapshot: &RunSnapshot) -> Result<CheckpointId, CheckpointError>;
    async fn load(&self, run_id: &RunId) -> Result<Option<RunSnapshot>, CheckpointError>;
    async fn latest(&self, run_id: &RunId) -> Result<Option<CheckpointId>, CheckpointError>;
}
```

### 14.6 Event Sink Adapter

```rust
#[async_trait]
pub trait EventSink: Send + Sync {
    async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError>;
}

pub trait EventSource: Send + Sync {
    fn subscribe(&self, run_id: &RunId) -> Box<dyn Stream<Item = RuntimeEvent> + Send>;
}
```

### 14.7 Secrets Adapter

```rust
/// A reference to a named secret declared in the workflow document.
/// Secrets are never embedded as values in workflow documents.
pub struct SecretRef(pub String);

/// An opaque resolved secret value. Never logged, serialized to checkpoints,
/// or included in observable event records.
pub struct SecretValue(pub String);

#[async_trait]
pub trait SecretsAdapter: Send + Sync {
    /// Resolve a named secret reference to its current value.
    async fn get(&self, secret_ref: &SecretRef) -> Result<SecretValue, SecretsError>;
}
```

The built-in `HostMapSecretsAdapter` implements `SecretsAdapter` over a `HashMap<String, SecretValue>` provided
by the host application at run start. Host applications may replace it with a vault, OS keychain, or cloud
secrets manager by implementing the trait.

### 14.8 Agent Actor Trait

`AgentActor` is defined in **`langchart-runtime`** because it requires a reference to `CapabilityBroker`:

```rust
#[async_trait]
pub trait AgentActor: Send + Sync {
    /// Execute the agent's internal loop using the provided invocation context.
    /// Returns exactly one declared output event when the agent completes,
    /// or an `activity.failed` / `activity.cancelled` event on error.
    async fn run(
        &self,
        invocation: AgentInvocation,
        broker: Arc<CapabilityBroker>,
    ) -> Result<EventEnvelope, AgentError>;
}

pub struct AgentInvocation {
    pub run_id: RunId,
    pub state_id: StateId,
    pub invocation_id: InvocationId,
    pub instructions: ResolvedInstructions,
    pub context_view: ContextView,
    pub input: ron::Value,
    pub output_schemas: Vec<EventSchema>,
    pub limits: ExecutionLimits,
}
```

### 14.9 Capability Broker

The `CapabilityBroker` is not a simple adapter — it is a core runtime component that holds references to the
LLM, MCP, memory, and secrets adapters and enforces policy on every call:

```rust
pub struct CapabilityBroker {
    llm: Arc<dyn LlmAdapter>,
    mcp: Arc<dyn McpAdapter>,
    memory: Arc<dyn MemoryAdapter>,
    secrets: Arc<dyn SecretsAdapter>,
    event_sink: Arc<dyn EventSink>,
}

impl CapabilityBroker {
    pub async fn call_llm(
        &self,
        invocation_id: &InvocationId,
        envelope: &CapabilityEnvelope,
        request: LlmRequest,
    ) -> Result<LlmResponse, BrokerError>;

    pub async fn call_tool(
        &self,
        invocation_id: &InvocationId,
        envelope: &CapabilityEnvelope,
        server_id: &ServerId,
        tool_name: &ToolName,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, BrokerError>;

    pub async fn memory_search(
        &self,
        invocation_id: &InvocationId,
        envelope: &CapabilityEnvelope,
        query: MemoryQuery,
    ) -> Result<Vec<MemoryResult>, BrokerError>;
}
```

Every call through the broker is logged to the event sink before forwarding. The broker checks the envelope
before every forwarded call. There is no bypass path.

The broker also resolves secret references from `SecretsAdapter` and injects them into tool call arguments or
model requests as needed. Resolved secret values are used in-flight only and are NEVER written to the event
sink, checkpoints, or any persistent record.

---

## 15. Core Library Interfaces

```text
WorkflowRepository
  load(workflow_id, version) -> workflow_document
  store(workflow_document)

WorkflowValidator
  validate(workflow_document) -> Vec<Diagnostic>
  compile(workflow_document) -> Result<CompiledWorkflow, Vec<Diagnostic>>

RuntimeEngine
  start(compiled_workflow, input, adapters) -> run_id
  send(run_id, event)
  suspend(run_id)
  resume(run_id)
  cancel(run_id)
  inspect(run_id) -> RunSnapshot

ContextResolverChain
  add_stage(stage: impl ContextResolverStage)
  resolve(policy, snapshot) -> ContextView

CapabilityBroker
  (see §14.7)

ArtifactStore, CheckpointStore, EventSink, EventSource
  (see §14.4–14.6)
```

---

## 16. Canonical Workflow Document

### 16.1 Format

JSON is the canonical machine representation. YAML is supported as a human-authored representation. Both MUST
conform to the same versioned JSON Schema.

**Machine-readable schema:** [`docs/workflow-schema.json`](workflow-schema.json) — a JSON Schema Draft 2020-12
document covering all fields of `WorkflowDocument` and its nested types. The `$id` is
`https://langchart.dev/schema/workflow/1.0.0`.

The workflow document contains:

- schema version;
- workflow identity and semantic version;
- typed input and output ports;
- referenced agent definitions (inline or by reference);
- state hierarchy;
- transitions and CEL guards;
- context and capability policies;
- artifact bindings;
- editor metadata in a non-semantic `_editor` namespace.

Visual positions, colors, collapsed state, and viewport settings MUST NOT affect execution semantics. They live
exclusively in `_editor`.

### 16.2 Identity

Every state, transition, region, port, and reusable definition MUST have a stable ID independent of its display
name. Renaming a state must not break stored run history or editor references.

IDs are opaque strings. The RECOMMENDED format is a short stable slug (e.g., `draft_scene`) rather than a UUID,
to support human-readable workflow documents.

### 16.3 Versioning

Running workflow instances MUST remain pinned to the workflow version with which they started unless an explicit
migration is performed.

Schema migrations and live-run migrations are separate concerns. The initial release MUST fail safely (refusing to
load an incompatible version) rather than silently switching definitions.

### 16.4 Field Preservation

The document loader MUST preserve unknown fields in a `_unknown` namespace when loading a document with a newer
minor schema version. This allows forward-compatible authoring without data loss.

---

## 17. Visual Editor Specification

### 17.1 Technology and Embedding

The editor is a TypeScript component (`editor/`) designed for embedding in a web-based host application. It uses
the WASM build of `langchart-model` (`langchart-wasm`) for validation, so the editor and runtime apply identical
structural rules.

**Build:** `npm run build:full` in `editor/` — runs `wasm-pack build` then the Vite TypeScript bundle. See
`editor/package.json` for all available scripts (`wasm:build`, `wasm:build:release`, `build`, `build:full`,
`build:full:release`). Until the WASM package is built, all functions fall back to stubs that return empty
results.

The editor MUST be independently optional. Starting the runtime MUST NOT require the editor to be present or
initialized.

### 17.2 Implemented Panels

The editor implements 9 panels, each a lightweight vanilla-DOM component subscribing to `EditorState`:

| Panel | File | Status |
|---|---|---|
| Workflow canvas | `workflow-canvas.ts` | SVG grid with state nodes, cubic-Bézier transition edges, active/selected/initial highlighting, click-to-select |
| State inspector | `state-inspector.ts` | Three-tab: Basic, Advanced, Source (raw JSON fragment) |
| Problems panel | `problems-panel.ts` | Live diagnostics + reachability; click a diagnostic to select its state |
| Run inspector | `run-inspector.ts` | Snapshot display with colour-coded event log (lifecycle/state/error/budget) |
| Simulation panel | `secondary-panels.ts` | Actor-scripted synchronous model simulation via `simulateWorkflow` WASM binding |
| Context inspector | `secondary-panels.ts` | Placeholder — populated by host from live run events |
| Capability inspector | `secondary-panels.ts` | Static capability summary for selected state |
| Artifact review | `secondary-panels.ts` | Placeholder — populated by host from `ArtifactStore` events |
| Trace timeline | `secondary-panels.ts` | Placeholder — populated by host from live `RuntimeEvent` stream |

### 17.3 WASM Bindings (`langchart-wasm`)

Ten browser-facing functions (all accept/return JSON strings):

| Function | Description |
|---|---|
| `schema_version()` | Returns the current schema version string |
| `validateWorkflow(json)` | Returns `Diagnostic[]` |
| `compileWorkflow(json)` | Returns `{ ok, errors[] }` |
| `listStateIds(json)` | Returns `string[]` of all state IDs |
| `inspectState(json, id)` | Returns `StateInspection \| null` |
| `getGuardErrors(json)` | Returns `GuardError[]` for invalid CEL expressions |
| `workflowSummary(json)` | Returns `WorkflowSummary` with state counts and agent IDs |
| `listTransitions(json)` | Returns `TransitionEdge[]` for graph rendering |
| `reachabilityAnalysis(json)` | Returns `{ reachable[], unreachable[] }` |
| `simulateWorkflow(json, script)` | Runs a synchronous actor-scripted simulation; returns `{ status, final_state, steps[], error }` |

### 17.4 Editor State and Events

`EditorState` is the central in-memory model. It emits typed `EditorEvent`s that panels subscribe to:

- `workflow-changed` — document reloaded
- `state-selected` — state selected in canvas
- `diagnostics-updated` — validation results refreshed
- `run-snapshot-updated` — host pushed a new `RunSnapshot`
- `run-event-appended` — host pushed a `RunEventEntry` (lifecycle / state / error / budget / other)

Host applications integrate by calling:
- `state.loadJson(json)` — load a workflow document
- `state.updateRunSnapshot(snap)` — push a live run snapshot
- `state.appendRunEvent(entry)` — push a typed event log entry

### 17.5 Validation

The editor validates continuously via `wasm.validateWorkflow` and `wasm.reachabilityAnalysis` on every document
change. Unreachable states produce `W100` synthetic diagnostics in the Problems panel.

### 17.6 Simulation Panel

The simulation panel exposes the `simulateWorkflow` WASM binding without requiring a Tokio runtime. It:
- maps each state ID to a scripted actor (which event to emit on entry);
- accepts an inject list of initial events;
- drives the compiled workflow's `state_index` synchronously;
- reports `completed` / `stuck` / `running` (budget-exhausted) with a step-by-step trace table.

Guards are not evaluated in this mode (no live `WorkflowData`). Only guardless transitions are followed.

---

## 18. Observability and Replay

### 18.1 Observable Records

Every significant runtime action MUST produce an observable `RuntimeEvent` record:

| Category | Events |
|---|---|
| Run lifecycle | `run.started`, `run.suspended`, `run.resumed`, `run.completed`, `run.failed`, `run.cancelled` |
| State lifecycle | `state.entered`, `state.exited` |
| Transitions | `transition.selected`, `transition.guard_evaluated` |
| Activities | `activity.started`, `activity.completed`, `activity.failed`, `activity.cancelled`, `activity.retried` |
| Model calls | `llm.request`, `llm.response` (with tokens, latency, cost) |
| Tool calls | `tool.request`, `tool.response`, `tool.rejected` (policy rejection) |
| Memory | `memory.stored`, `memory.searched` |
| Context | `context.resolved` (with source list and token counts) |
| Proposals | `proposal.created`, `proposal.accepted`, `proposal.rejected`, `proposal.committed`, `proposal.conflicted` |
| Checkpoints | `checkpoint.saved` |
| Budgets | `budget.warning`, `budget.exhausted` |
| Human | `human.input_requested`, `human.input_received` |
| Errors | `event.unhandled`, `activity.invalid_output` |

### 18.2 Activity Records

Each activity record SHOULD include:

- resolved workflow and agent versions;
- model and provider configuration;
- prompt template versions;
- context-view identity (hash of the assembled context);
- permitted capability envelope;
- model response and structured event;
- tool requests and results (subject to redaction policy);
- token usage, latency, and cost;
- retries and errors;
- artifact versions read and proposals produced.

### 18.3 Replay Modes

| Mode | Description |
|---|---|
| **Trace replay** | Display previously recorded execution in the editor timeline. |
| **Deterministic replay** | Reuse recorded external results (model responses, tool results) for exact reproduction. Note: context views may differ if retrieval systems have changed. |
| **Re-execution** | Call models and tools again from a checkpoint. Output will differ due to LLM nondeterminism. |
| **Fork** | Create a new run from a prior checkpoint with modified workflow data or configuration. |

Re-execution is NOT expected to produce identical model output. The system MUST NOT claim deterministic replay
through LLM re-execution.

---

## 19. Safety and Governance

The runtime MUST enforce capabilities independently of the editor, the agent prompt, and the agent actor's
claimed identity.

Required:

- least-privilege capability envelopes (intersection of all policy layers);
- separate `read`, `propose`, `commit`, `publish`, and `delete` authority classes;
- human approval gates for configurable operation classes;
- secret isolation — state-scoped credentials are injected by the broker and NEVER disclosed to agent actors;
- argument validation (CEL) for tool calls before forwarding;
- resource URI restrictions (allowlist patterns);
- output schema validation for every emitted agent event;
- configurable data retention and redaction for observable records;
- per-state and per-run cost limits (token budget exhaustion is a broker-enforced event);
- audit records for capability changes and artifact commits;
- cancellation propagation — cancelling a run MUST cancel all active actor futures.

**Prompt instructions are behavioral guidance, not a security boundary.** All enforcement is in the broker.

---

## 20. Testing Strategy

### 20.1 Static Tests

Workflow validation tests SHOULD cover topology, schemas, hierarchy, CEL guard compilation, capability
calculation, and schema migration.

### 20.2 Model-Free Runtime Tests

A `ScriptedAgentActor` implementation permits deterministic testing of transitions, retries, parallel regions,
timers, suspension, recovery, and artifact conflicts without invoking an LLM. The scripted actor emits a
pre-configured event after an optional simulated delay.

### 20.3 Agent Evaluation Tests

Agentic states SHOULD support fixture-based evaluations:

- input workflow data;
- artifact snapshot;
- expected context inclusions and exclusions;
- permitted tools;
- expected output event type;
- output quality rubric;
- maximum cost and latency thresholds.

### 20.4 Simulation

The editor SHOULD allow authors to inject mock events, step through transitions, inspect CEL guard evaluation,
and simulate failure paths using the `ScriptedAgentActor` backend.

### 20.5 Capability Broker Property Tests

The capability broker MUST have property-based tests (e.g., using `proptest`) covering:

- policy intersection correctness;
- elevation detection;
- budget enforcement;
- credential isolation.

---

## 21. Crate Structure

```
langchart/                        (workspace root)
│
├─ crates/
│   ├─ langchart-model/           Core types, schema, validation, CEL guards
│   │                             WASM-compatible. No I/O, no async runtime.
│   │
│   ├─ langchart-adapters/        Adapter traits only (no implementations)
│   │                             LlmAdapter, McpAdapter, MemoryAdapter,
│   │                             ArtifactStore, CheckpointStore,
│   │                             EventSink, EventSource
│   │                             Depends on: langchart-model
│   │
│   ├─ langchart-runtime/         Async execution engine
│   │                             CapabilityBroker, RuntimeEngine,
│   │                             event loop, checkpointing, timers
│   │                             Depends on: langchart-model, langchart-adapters
│   │
│   ├─ langchart-context/         ContextResolverChain and built-in stages
│   │                             Depends on: langchart-model, langchart-adapters
│   │
│   ├─ langchart-wasm/            WASM build of langchart-model
│   │                             wasm-bindgen bindings for the editor
│   │                             Depends on: langchart-model
│   │
│   ├─ langchart/                 Convenience re-export crate (the public API)
│   │                             Depends on: all above
│   │
│   └─ adapters/                  Optional concrete adapter implementations
│       ├─ langchart-llm-generic/    OpenAI + Anthropic APIs, model enumeration
│       ├─ langchart-model-router/   Policy-driven dispatch to LlmAdapter instances
│       ├─ langchart-mcp-client/
│       ├─ langchart-memory-redb/
│       └─ langchart-checkpoint-redb/
│
├─ editor/                        TypeScript visual editor (optional)
│
├─ examples/                      Reference workflows (domain-neutral)
│   └─ content-pipeline/          Multi-stage content analysis reference workflow
│
└─ docs/
    └─ agentic-statechart-library-spec.md
```

**Dependency rules:**

- `langchart-model` MUST have zero runtime dependencies and compile to WASM.
- `langchart-adapters` depends only on `langchart-model` and `async-trait`.
- `langchart-runtime` MUST NOT depend on any concrete adapter crate.
- `langchart-model-router` depends on `langchart-adapters` only; it does NOT depend on `langchart-runtime`.
- Concrete adapter crates MUST NOT depend on `langchart-runtime`.
- The root `langchart` crate re-exports the public API and is the only crate users add to their `Cargo.toml`.

---

## 22. Reference Workflow

The first end-to-end reference workflow is a generic content pipeline (domain-neutral):

```
Brief
  → Plan (agentic)
  → Human Plan Approval (human)
  → Draft (compound)
      → AssembleContext (atomic)
      → Write (agentic)
      → SelfReview (agentic)
      → [loop back to Write on revision_required]
      → ProposeArtifact (atomic)
  → Parallel Review (parallel, completion: all)
      ├─ StructureReview (agentic)
      ├─ ConsistencyReview (agentic)
      ├─ ToneReview (agentic)
      └─ FactualReview (agentic)
  → ConsolidateIssues (agentic)
  → Revise (agentic)
  → Human Change Approval (human)
  → CommitArtifactVersion (atomic)
  → [*]
```

This workflow exercises: hierarchy, parallel regions, agentic states, human states, context policies, MCP
restrictions, structured events, proposals, artifact versioning, and runtime visualization.

The host application (Obsidian-like environment) maps its own domain concepts (notes, knowledge base, etc.)
to this workflow's artifact and tool abstractions.

---

## 23. Delivery Phases

### Phase 1: Canonical Model

- [ ] Cargo workspace scaffolding
- [ ] `langchart-model`: workflow, state, transition, event types
- [ ] `langchart-adapters`: all adapter traits
- [ ] Atomic, compound, agentic, human, and final state types
- [ ] CEL guard compilation and static analysis
- [ ] JSON/YAML workflow document loading and schema validation
- [ ] Static capability calculation
- [ ] In-memory `ScriptedAgentActor` for testing

### Phase 2: Durable Runtime

- [ ] `langchart-runtime`: event loop, RTC macro-step
- [ ] `CapabilityBroker` with policy enforcement and event logging
- [ ] `langchart-context`: `ContextResolverChain` with built-in stages
- [ ] Checkpoint save and load
- [ ] Durable timers and retry policies
- [ ] Suspension, resumption, and cancellation
- [ ] `langchart-checkpoint-redb`: embedded checkpoint store

### Phase 3: Agent Adapters

- [ ] `langchart-llm-generic`: OpenAI + Anthropic LLM adapter with model enumeration
- [ ] `langchart-model-router`: model routing with policy-driven dispatch
- [ ] `langchart-mcp-client`: MCP client adapter
- [ ] `langchart-memory-redb`: embedded memory adapter
- [ ] Full end-to-end reference workflow execution (headless)

### Phase 4: Parallel and Subworkflow

- [ ] Parallel state with all completion modes
- [ ] Parallel proposal consolidation protocol
- [ ] Subworkflow ports and invocation
- [ ] History (shallow and deep)

### Phase 5: Visual Editor

- [ ] `langchart-wasm`: WASM build of model layer
- [ ] TypeScript editor scaffold
- [ ] Hierarchical canvas with compound state drill-down
- [ ] State and transition inspectors
- [ ] Continuous validation panel
- [ ] Import, export, and version save
- [ ] Runtime execution overlay (live trace)

### Phase 6: Evaluation and Production Hardening

- [ ] Test fixtures and simulation mode
- [ ] Trace replay and run forks
- [ ] Property-based capability broker tests
- [ ] Cost and token budget enforcement
- [ ] Governance, quota, and redaction configuration
- [ ] Host application embedding guide

---

## 24. Open Questions (Resolved)

| # | Question | Decision |
|---|---|---|
| 1 | SCXML, XState, or custom? | Custom agentic profile. Borrows ideas from both; not bound to either. Formal compliance is not a goal. |
| 2 | Guard expression language? | CEL (Common Expression Language). Deterministic, serializable, WASM-compatible, Rust-native via `cel-interpreter`. |
| 3 | Prompt template versioning? | Prompt templates are referenced by path and version in the agent definition. The workflow document pins the agent definition version. Host application manages prompt file storage. |
| 4 | Agent definitions inline or external? | Both. Inline definitions for simple cases; external versioned definitions for reuse. The validator handles both. |
| 5 | Artifact patch formats? | Defined by the `ArtifactStore` adapter. The core library is content-format agnostic. |
| 6 | Context resolution determinism? | Context views are recorded at invocation time. Deterministic replay reuses the recorded view. Re-execution acknowledges approximate fidelity when retrieval systems change. |
| 7 | Subworkflow version pinning? | Exact version pinning with explicit migration. Compatible-range pinning is deferred to a later phase. |
| 8 | Live-run migration guarantees? | Phase 1 fails safely on version mismatch. Live-run migration is deferred. |
| 9 | Distributed activities? | Deferred. Phase 1–3 use local async execution. |
| 10 | WASM scope? | `langchart-model` compiles to WASM. `langchart-runtime` does not (requires async runtime). |

---

## 25. Resolved Design Decisions (formerly Open Questions)

| # | Decision | Resolution |
|---|---|---|
| 1 | `AgentActor` trait location | Defined in **`langchart-runtime`**. The trait requires a reference to the `CapabilityBroker`, which lives in runtime. Placing it in adapters would create a circular dependency or require an awkward re-export. Actor implementors depend on `langchart-runtime` directly. |
| 2 | CEL extension functions | **Opt-in whitelist.** A `CelExtensions` registry in `langchart-model` holds named pure functions approved for use in guards. The whitelist is enforced at CEL compilation time. Side-effectful or I/O functions are permanently excluded. The initial whitelist is: `version_gte`, `version_lte`, `contains_all`, `contains_any`, `is_empty`. |
| 3 | Workflow data schema | **Typed via RON.** Workflow data fields are declared with explicit types in the workflow document using RON (Rusty Object Notation) syntax for type signatures. RON integrates naturally with Rust's type system, supports enums and structs, and round-trips cleanly without JSON's stringly-typed limitations. The declared schema is used for static CEL guard type-checking and runtime deserialization. |
| 4 | Event ordering in parallel regions | **Stable (deterministic).** Events emitted from parallel regions are ordered by region declaration order, then by emission timestamp (ULID monotonic). This guarantees deterministic replay and makes test assertions stable. |
| 5 | Secret injection mechanism | **`SecretsAdapter` trait** in `langchart-adapters`, with a built-in `HostMapSecretsAdapter` (a `HashMap<SecretRef, SecretValue>`) as the default. The trait allows host applications to delegate to a vault, AWS Secrets Manager, OS keychain, or any other secrets backend. Secrets are never serialized into checkpoints or event records. |

---

## 26. Initial Acceptance Criteria

The first usable release is complete when a user can:

1. Define reusable agents with prompts, context policies, typed outcomes, and MCP restrictions.
2. Construct a hierarchical agentic statechart as a JSON or YAML document.
3. Validate the workflow and receive actionable CEL and schema diagnostics.
4. Start a durable workflow run programmatically (headless, no editor required).
5. Observe active states, agent invocations, events, and tool calls via the event sink.
6. Suspend for a human decision and resume without losing run state.
7. Run multiple agents in parallel against immutable artifact versions.
8. Receive structured proposals and commit an authorized artifact version.
9. Inspect which prompts, context sources, tools, models, and artifact versions contributed to each invocation.
10. Embed the engine in a Rust application by implementing the adapter traits and calling `RuntimeEngine::start`.

---

## 27. Summary

`langchart` treats agent workflows as governed state machines rather than informal chains of model calls.

The defining structural choices are:

- **Statechart hierarchy** provides progressive disclosure both in the visual editor and in execution scope.
- **Agent actors are opaque** — the runtime governs the envelope, the agent governs its internal loop.
- **CEL guards** ensure deterministic, serializable, WASM-compatible transition selection.
- **Three universal adapter traits** (LLM, MCP, Memory) plus artifact, checkpoint, and event adapters give
  the engine a complete integration surface without mandating any specific implementation.
- **The capability broker is the security kernel** — all enforcement passes through it, nothing bypasses it.
- **The model layer is WASM-safe** — the same validation logic runs in the Rust runtime and the TypeScript editor.
- **The engine is a library** — designed for embedding in an Obsidian-like host application, not as a standalone
  service.

The goal is to provide a better foundation than LangChain/LangGraph: explicit control flow, governed information
disclosure, least-privilege capability management, and durable execution — while remaining as easy to integrate
as adding a Cargo dependency and implementing a few adapter traits.
