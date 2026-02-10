# eCommerce Intent Reasoning Engine — Technical Specification

**Version:** 0.1.0-draft
**Author:** Blake / Orderloop
**Date:** February 2026
**Status:** Design Phase

---

## 1. Executive Summary

This document specifies the architecture, data models, processing pipeline, and deployment strategy for an eCommerce intent reasoning engine. The system ingests multi-channel customer signals (chat, email, forms, webhooks, click behavior), classifies intent, performs multi-step reasoning over compound or ambiguous requests, and resolves intents into executable action plans against commerce platforms.

The engine is purpose-built for post-purchase eCommerce — the domain where Orderloop competes with AfterShip and Narvar — but is designed to extend into pre-purchase and full-lifecycle coverage.

**Core differentiator:** This is not a classifier. It is a *reasoning engine* — it decomposes compound intents, resolves conflicts between competing customer goals, enriches decisions with order/product context, and outputs structured action plans that downstream systems can execute.

---

## 2. Problem Statement

Existing eCommerce intent systems are shallow classifiers. They map "where is my order" to a WISMO label and call it a day. They fail when:

- A customer says "I got the wrong size and I need the replacement by Friday" (compound intent: return + exchange + shipping constraint)
- A customer submits a contact form saying "this is broken" with an attached photo (multi-modal input requiring damage assessment + return initiation)
- Context matters: a VIP customer with 50 orders gets different routing than a first-time buyer flagged for potential fraud
- Intent shifts mid-conversation: starts as a product question, becomes a complaint, escalates to a retention risk

The reasoning engine solves this by treating intent as a *graph of actions with constraints*, not a flat label.

---

## 3. Intent Taxonomy

### 3.1 Primary Intent Categories

| Category | Code | Description | Typical Volume |
|---|---|---|---|
| Order Status | `ORDER_STATUS` | Tracking, delivery estimates, shipment updates | 30-40% |
| Returns & Exchanges | `RETURN_EXCHANGE` | Initiate return, exchange, refund requests | 15-25% |
| Order Modification | `ORDER_MODIFY` | Cancel, change address, update items pre-ship | 10-15% |
| Product Inquiry | `PRODUCT_INQUIRY` | Sizing, compatibility, availability, comparison | 10-15% |
| Complaint & Escalation | `COMPLAINT` | Damaged, wrong item, poor experience, demand escalation | 8-12% |
| Account & Billing | `ACCOUNT_BILLING` | Payment issues, subscription, address management | 5-8% |
| Discovery | `DISCOVERY` | Recommendations, deals, restock alerts | 3-5% |
| Meta | `META` | Talk to human, frustrated with bot, general confusion | 3-5% |

### 3.2 Intent Hierarchy

Each primary category decomposes into specific intents. The engine supports a three-level hierarchy:

```
Category → Intent → Sub-Intent

RETURN_EXCHANGE
├── RETURN_INITIATE
│   ├── RETURN_DEFECTIVE
│   ├── RETURN_WRONG_ITEM
│   ├── RETURN_NOT_AS_DESCRIBED
│   ├── RETURN_CHANGED_MIND
│   └── RETURN_SIZE_FIT
├── EXCHANGE_REQUEST
│   ├── EXCHANGE_SIZE
│   ├── EXCHANGE_COLOR
│   └── EXCHANGE_DIFFERENT_PRODUCT
├── REFUND_STATUS
│   ├── REFUND_CHECK
│   ├── REFUND_EXPEDITE
│   └── REFUND_DISPUTE
└── RETURN_STATUS
    ├── RETURN_LABEL_REQUEST
    ├── RETURN_TRACKING
    └── RETURN_RECEIVED_CONFIRMATION

ORDER_STATUS
├── WISMO (Where Is My Order)
│   ├── WISMO_GENERAL
│   ├── WISMO_DELAYED
│   └── WISMO_LOST
├── DELIVERY_ESTIMATE
├── SHIPMENT_UPDATE
└── PROOF_OF_DELIVERY

ORDER_MODIFY
├── CANCEL_ORDER
│   ├── CANCEL_FULL
│   └── CANCEL_PARTIAL
├── CHANGE_ADDRESS
├── CHANGE_ITEMS
└── ADD_INSTRUCTIONS

COMPLAINT
├── DAMAGED_ITEM
├── WRONG_ITEM_RECEIVED
├── MISSING_ITEM
├── QUALITY_ISSUE
├── SHIPPING_COMPLAINT
└── ESCALATION_DEMAND

ACCOUNT_BILLING
├── PAYMENT_FAILED
├── PAYMENT_METHOD_UPDATE
├── SUBSCRIPTION_MANAGE
├── ADDRESS_UPDATE
└── ACCOUNT_ACCESS

PRODUCT_INQUIRY
├── SIZING_FIT
├── COMPATIBILITY
├── AVAILABILITY_RESTOCK
├── COMPARISON
└── PRODUCT_DETAILS

DISCOVERY
├── RECOMMENDATION
├── DEALS_PROMOTIONS
└── RESTOCK_ALERT

META
├── HUMAN_HANDOFF
├── FRUSTRATION_DETECTED
└── UNCLEAR_INTENT
```

### 3.3 Compound Intents

The reasoning engine uniquely handles compound intents — requests that span multiple categories or require multi-step resolution:

| Compound Pattern | Decomposition | Priority Logic |
|---|---|---|
| Wrong size + need it by date | `RETURN_SIZE_FIT` + `EXCHANGE_SIZE` + time constraint | Exchange initiated first, return label concurrent |
| Cancel + reorder different | `CANCEL_FULL` + implicit `ORDER_CREATE` | Cancel only if reorder is viable |
| Damaged + want refund + complaint | `DAMAGED_ITEM` + `REFUND_EXPEDITE` + `ESCALATION_DEMAND` | Complaint routing first, refund auto-approved |
| Product question + availability | `PRODUCT_DETAILS` + `AVAILABILITY_RESTOCK` | Answer product question, then check stock |

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT CHANNELS                           │
│  Chat │ Email │ Forms │ Webhooks │ SMS │ Social │ Click Stream  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                                │
│  Channel adapters → Normalization → Unified IntentRequest model  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EXTRACTION LAYER                               │
│  Entity extraction │ Sentiment analysis │ Semantic embeddings    │
│  Attachment processing │ Session context aggregation             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MATCHING LAYER (Fast Path)                    │
│  Embedding similarity against intent catalog                     │
│  Confidence threshold → if high, skip to Resolution              │
│  Handles ~70-80% of requests                                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │ (low confidence or compound signal)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    REASONING LAYER (Deep Path)                   │
│  LLM-powered decomposition │ Context enrichment from order DB   │
│  Constraint resolution │ Action plan generation                  │
│  Conflict detection │ Policy engine evaluation                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RESOLUTION LAYER                               │
│  Intent → Action mapping │ Platform API orchestration            │
│  Response generation │ Handoff routing │ Audit logging           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    OUTPUT CHANNELS                                │
│  Chat response │ Email reply │ Webhook trigger │ Agent dashboard │
│  Ticket creation │ Platform API calls (Shopify, WC, etc.)       │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Breakdown

```
src/
├── intent_engine/
│   ├── __init__.py
│   ├── config.py                  # Engine configuration, feature flags
│   ├── engine.py                  # Main orchestrator / entry point
│   │
│   ├── models/                    # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── intent.py              # Intent taxonomy models
│   │   ├── request.py             # Unified IntentRequest
│   │   ├── entity.py              # Extracted entities (order IDs, SKUs, dates)
│   │   ├── context.py             # Session/customer/order context
│   │   ├── action.py              # Action plan models
│   │   └── response.py            # Engine output models
│   │
│   ├── ingestion/                 # Channel adapters
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract adapter interface
│   │   ├── chat.py                # Live chat / messaging
│   │   ├── email.py               # Email parsing (subject, body, attachments)
│   │   ├── form.py                # Contact/return forms
│   │   ├── webhook.py             # Platform webhooks (Shopify, etc.)
│   │   ├── sms.py                 # SMS/MMS
│   │   └── clickstream.py         # Behavioral signals (page visits, cart actions)
│   │
│   ├── extractors/                # Feature extraction
│   │   ├── __init__.py
│   │   ├── entity_extractor.py    # NER for eCommerce entities
│   │   ├── sentiment.py           # Sentiment + urgency scoring
│   │   ├── embedding.py           # Semantic embedding generation
│   │   ├── attachment.py          # Image/file analysis
│   │   └── session.py             # Session context aggregation
│   │
│   ├── matchers/                  # Fast path — embedding-based matching
│   │   ├── __init__.py
│   │   ├── catalog.py             # Intent catalog with example embeddings
│   │   ├── similarity.py          # Cosine similarity + threshold logic
│   │   └── compound_detector.py   # Detect multi-intent signals
│   │
│   ├── reasoners/                 # Deep path — LLM-powered reasoning
│   │   ├── __init__.py
│   │   ├── decomposer.py          # Break compound intents into atomic actions
│   │   ├── context_enricher.py    # Pull order/customer data into reasoning
│   │   ├── constraint_solver.py   # Resolve time, policy, inventory constraints
│   │   ├── conflict_resolver.py   # Handle contradictory intents
│   │   └── policy_engine.py       # Business rules (return windows, VIP tiers)
│   │
│   ├── resolvers/                 # Intent → Action execution
│   │   ├── __init__.py
│   │   ├── action_mapper.py       # Map intents to platform API calls
│   │   ├── response_generator.py  # Generate customer-facing responses
│   │   ├── handoff.py             # Human agent escalation logic
│   │   └── audit.py               # Decision logging and explainability
│   │
│   ├── integrations/              # Platform connectors
│   │   ├── __init__.py
│   │   ├── shopify.py
│   │   ├── woocommerce.py
│   │   ├── bigcommerce.py
│   │   ├── magento.py
│   │   └── generic_api.py         # Configurable REST connector
│   │
│   ├── storage/                   # Persistence
│   │   ├── __init__.py
│   │   ├── vector_store.py        # Embedding storage (Qdrant, Pinecone, pgvector)
│   │   ├── intent_catalog.py      # Intent definitions + training examples
│   │   └── conversation.py        # Conversation history for multi-turn
│   │
│   └── api/                       # Service layer
│       ├── __init__.py
│       ├── server.py              # FastAPI app
│       ├── routes.py              # API endpoints
│       ├── middleware.py           # Auth, rate limiting, tenant isolation
│       └── websocket.py           # Real-time streaming for chat
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                  # Example inputs per channel, golden outputs
│
├── evals/                         # Evaluation harness
│   ├── datasets/                  # Labeled intent datasets
│   ├── metrics.py                 # Precision, recall, F1 per intent
│   └── runner.py                  # Batch evaluation runner
│
├── scripts/
│   ├── seed_catalog.py            # Populate intent catalog with examples
│   └── generate_embeddings.py     # Pre-compute embeddings for catalog
│
├── pyproject.toml
└── README.md
```

---

## 5. Data Models

### 5.1 Unified Intent Request

Every input channel normalizes into this common structure:

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional

class InputChannel(str, Enum):
    CHAT = "chat"
    EMAIL = "email"
    FORM = "form"
    WEBHOOK = "webhook"
    SMS = "sms"
    SOCIAL = "social"
    CLICKSTREAM = "clickstream"

class Attachment(BaseModel):
    url: str
    mime_type: str
    filename: Optional[str] = None
    analysis: Optional[dict] = None  # populated by attachment extractor

class IntentRequest(BaseModel):
    """Unified input model — every channel adapter produces this."""
    request_id: str = Field(description="Unique request identifier")
    tenant_id: str = Field(description="Merchant/store identifier")
    channel: InputChannel
    timestamp: datetime

    # Raw input
    raw_text: str = Field(description="Primary text content")
    raw_metadata: dict = Field(
        default_factory=dict,
        description="Channel-specific metadata (email headers, form fields, etc.)"
    )
    attachments: list[Attachment] = Field(default_factory=list)

    # Conversation context
    conversation_id: Optional[str] = None
    message_index: int = 0  # position in multi-turn conversation
    previous_intents: list[str] = Field(
        default_factory=list,
        description="Intents resolved earlier in this conversation"
    )

    # Customer context (populated by enrichment)
    customer_id: Optional[str] = None
    customer_tier: Optional[str] = None  # VIP, standard, new, flagged
    order_ids: list[str] = Field(
        default_factory=list,
        description="Order IDs mentioned or inferred"
    )

    # Behavioral signals
    page_context: Optional[str] = None  # URL/page the customer was on
    session_actions: list[str] = Field(
        default_factory=list,
        description="Recent clickstream actions"
    )
```

### 5.2 Extracted Entities

```python
class EntityType(str, Enum):
    ORDER_ID = "order_id"
    PRODUCT_SKU = "product_sku"
    PRODUCT_NAME = "product_name"
    TRACKING_NUMBER = "tracking_number"
    DATE = "date"
    DEADLINE = "deadline"           # "by Friday", "within 2 days"
    MONEY_AMOUNT = "money_amount"
    SIZE = "size"
    COLOR = "color"
    QUANTITY = "quantity"
    ADDRESS = "address"
    PERSON_NAME = "person_name"
    REASON = "reason"              # return reason, complaint reason

class ExtractedEntity(BaseModel):
    entity_type: EntityType
    value: str
    raw_span: str = Field(description="Original text span")
    start_pos: int
    end_pos: int
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    frustration_score: float = Field(ge=0.0, le=1.0)
    embedding: list[float] = Field(description="Semantic embedding vector")
```

### 5.3 Resolved Intent

```python
class IntentConfidence(str, Enum):
    HIGH = "high"       # > 0.85 — fast path, auto-resolve
    MEDIUM = "medium"   # 0.60 - 0.85 — reasoning path
    LOW = "low"         # < 0.60 — needs clarification or human handoff

class ResolvedIntent(BaseModel):
    """Single atomic intent with confidence."""
    category: str
    intent: str
    sub_intent: Optional[str] = None
    confidence: float
    confidence_tier: IntentConfidence
    evidence: list[str] = Field(
        description="Text spans or signals that support this classification"
    )

class Constraint(BaseModel):
    """A constraint on how an intent should be fulfilled."""
    constraint_type: str      # "deadline", "preference", "policy", "inventory"
    description: str
    value: str
    hard: bool = True         # hard constraint = must satisfy; soft = prefer

class ActionStep(BaseModel):
    """Single step in an action plan."""
    step_id: str
    action: str               # "initiate_return", "check_inventory", "send_label"
    target_system: str        # "shopify", "shipping_provider", "email"
    parameters: dict
    depends_on: list[str] = Field(
        default_factory=list,
        description="step_ids that must complete first"
    )
    estimated_duration_ms: Optional[int] = None
    fallback_action: Optional[str] = None

class ReasoningResult(BaseModel):
    """Complete output of the reasoning engine."""
    request_id: str
    resolved_intents: list[ResolvedIntent]
    is_compound: bool
    entities: list[ExtractedEntity]
    constraints: list[Constraint]
    action_plan: list[ActionStep]
    customer_response: str          # suggested response text
    internal_notes: str             # for agent dashboard / audit
    confidence_summary: float       # overall confidence
    requires_human: bool
    human_handoff_reason: Optional[str] = None
    reasoning_trace: list[str] = Field(
        description="Step-by-step reasoning log for explainability"
    )
    processing_time_ms: int
    path_taken: str                 # "fast_path" or "reasoning_path"
```

---

## 6. Processing Pipeline — Detailed

### 6.1 Ingestion Layer

Each channel adapter implements a common interface:

```python
from abc import ABC, abstractmethod

class ChannelAdapter(ABC):
    @abstractmethod
    async def normalize(self, raw_input: dict) -> IntentRequest:
        """Convert channel-specific payload to IntentRequest."""
        ...

    @abstractmethod
    def validate(self, raw_input: dict) -> bool:
        """Validate channel-specific payload structure."""
        ...
```

**Channel-specific normalization rules:**

| Channel | `raw_text` source | Key `raw_metadata` fields | Notes |
|---|---|---|---|
| Chat | Message body | `session_id`, `agent_id`, `widget_context` | Multi-turn: include last 5 messages in context |
| Email | Subject + body (stripped signatures/quotes) | `from`, `to`, `subject`, `thread_id`, `headers` | Parse reply chains, extract most recent message |
| Form | Concatenated field values | All form fields as key-value pairs | Map common field names to entity types |
| Webhook | Event description / payload body | `event_type`, `platform`, `resource_id` | Platform-initiated (e.g., Shopify order update triggers) |
| SMS | Message body | `phone_number`, `carrier` | Short text, high ambiguity — lean on reasoning path |
| Clickstream | Generated description of behavior pattern | `page_urls`, `actions`, `timing`, `cart_state` | Implicit intent: "viewed return policy 3x" → likely return |

### 6.2 Extraction Layer

Runs in parallel where possible:

**Entity Extraction** (`entity_extractor.py`)
- Uses spaCy with a custom eCommerce NER model for order IDs, SKUs, tracking numbers
- Regex patterns for structured identifiers (order ID formats vary per platform)
- Date/deadline extraction via dateparser with relative date support ("by Friday", "within 48 hours")
- Falls back to LLM extraction for ambiguous entities

**Sentiment Analysis** (`sentiment.py`)
- Lightweight classifier (fine-tuned distilbert or similar) for three scores: sentiment (-1 to 1), urgency (0 to 1), frustration (0 to 1)
- Frustration score directly influences routing: > 0.7 flags for priority handling
- Urgency score affects constraint generation: high urgency → tighter deadlines on action plan

**Semantic Embedding** (`embedding.py`)
- sentence-transformers model (e.g., `all-MiniLM-L6-v2` for speed or `e5-large-v2` for accuracy)
- Generates a single embedding vector for the normalized text
- Used by the matching layer for similarity search against intent catalog

**Attachment Processing** (`attachment.py`)
- Images: Vision model (Claude or similar) to describe damage, extract text from screenshots
- Documents: PDF/text extraction, then treat as additional raw_text
- Populates `Attachment.analysis` dict with structured findings

**Session Context** (`session.py`)
- Aggregates behavioral signals from the current session
- Detects patterns: repeated page visits (return policy, tracking page), cart abandonment, search terms
- Generates implicit intent signals that supplement explicit text

### 6.3 Matching Layer (Fast Path)

The fast path handles the 70-80% of requests that are unambiguous single intents.

**Intent Catalog** (`catalog.py`)
- Each intent in the taxonomy has 20-50 example utterances with pre-computed embeddings
- Examples span phrasing variations, formality levels, and languages (if multi-language support)
- Catalog is stored in a vector database (pgvector for simplicity, Qdrant for scale)

**Similarity Matching** (`similarity.py`)

```python
class MatchResult(BaseModel):
    intent: str
    similarity: float
    matched_example: str

async def match(embedding: list[float], top_k: int = 5) -> list[MatchResult]:
    """
    Cosine similarity search against intent catalog.
    Returns top_k matches ranked by similarity.
    """
    ...
```

Decision logic:

```
top_match = results[0]

if top_match.similarity >= 0.85:
    # HIGH confidence — fast path
    # Check: is there a second match within 0.10 of top?
    if results[1].similarity >= top_match.similarity - 0.10:
        # Ambiguous — route to reasoning path
        → REASONING PATH (with both candidates as hints)
    else:
        → FAST PATH RESOLUTION

elif top_match.similarity >= 0.60:
    # MEDIUM confidence — reasoning path
    → REASONING PATH (with top matches as hints)

else:
    # LOW confidence — likely unclear intent
    → REASONING PATH (open decomposition, may result in clarification request)
```

**Compound Intent Detection** (`compound_detector.py`)
- Checks for linguistic signals: "and", "also", "but", "then", multiple sentences with different verbs
- If the top-2 matches are in *different* categories (e.g., RETURN + ORDER_STATUS), flag as compound
- Sentence segmentation + per-sentence matching to identify individual intents within compound requests
- Any compound signal routes to the reasoning path regardless of confidence

### 6.4 Reasoning Layer (Deep Path)

This is the core differentiator. The reasoning layer uses LLM calls with structured output to perform multi-step reasoning.

**Decomposer** (`decomposer.py`)

Takes the request + extraction results + match hints and produces atomic intents:

```python
DECOMPOSITION_PROMPT = """
You are an eCommerce intent reasoning engine. Given a customer message
and context, decompose the request into atomic intents.

## Customer Message
{raw_text}

## Extracted Entities
{entities}

## Match Hints (from embedding similarity)
{match_hints}

## Customer Context
- Customer tier: {customer_tier}
- Order history: {order_summary}
- Previous intents in this conversation: {previous_intents}

## Instructions
1. Identify each distinct intent in the message
2. For compound requests, list each atomic intent separately
3. Identify constraints (deadlines, preferences, conditions)
4. Flag any contradictions or ambiguities
5. If intent is genuinely unclear, output UNCLEAR with clarification questions

Respond with the following JSON structure:
{output_schema}
"""
```

The LLM call uses function calling / structured output (Pydantic model) to ensure parseable results. Model: Claude Sonnet for cost-efficiency at this volume, Claude Opus for complex cases flagged by the compound detector.

**Context Enricher** (`context_enricher.py`)
- Pulls real-time data from the commerce platform:
  - Order details (status, items, shipping, tracking)
  - Customer profile (tier, lifetime value, previous interactions)
  - Product details (in stock, variants, return eligibility)
  - Policy data (return window, warranty status)
- Injects this context into the reasoning prompt so the LLM can make informed decisions
- Example: Customer says "I want to return this" — enricher finds the order is 45 days old, return window is 30 days. This becomes a constraint the solver must handle.

**Constraint Solver** (`constraint_solver.py`)
- Evaluates feasibility of each action in the plan:
  - Time constraints: Can the exchange arrive by the customer's deadline?
  - Policy constraints: Is the item eligible for return? Is the customer within the return window?
  - Inventory constraints: Is the exchange item in stock?
- Marks constraints as `satisfied`, `violable` (soft constraint, can proceed with caveat), or `blocked` (hard constraint, cannot proceed)
- When blocked, generates alternative action plans

**Conflict Resolver** (`conflict_resolver.py`)
- Handles contradictory intents: "I want a refund but also an exchange" (can't have both for same item)
- Priority rules:
  1. Explicit customer preference takes precedence
  2. Higher-value resolution preferred (exchange > refund for merchant)
  3. When ambiguous, ask for clarification
- Outputs a single coherent action plan or a clarification request

**Policy Engine** (`policy_engine.py`)
- Rule-based system (not LLM) for business logic that must be deterministic:
  - Return window validation
  - Refund amount calculation
  - Auto-approval thresholds (e.g., refund < $50 for VIP customers → auto-approve)
  - Escalation triggers (e.g., 3rd complaint in 30 days → auto-escalate)
- Configured per tenant via JSON/YAML policy files
- Policy evaluation results feed into the LLM reasoning as hard constraints

### 6.5 Resolution Layer

**Action Mapper** (`action_mapper.py`)
- Maps each `ResolvedIntent` to concrete API calls against the merchant's platform
- Maintains a registry of action handlers per platform:

```python
ACTION_REGISTRY = {
    "shopify": {
        "initiate_return": ShopifyReturnHandler,
        "cancel_order": ShopifyCancelHandler,
        "check_tracking": ShopifyTrackingHandler,
        "update_address": ShopifyAddressHandler,
        ...
    },
    "woocommerce": { ... },
    "generic": { ... },
}
```

- Executes the action plan steps respecting dependency ordering (`depends_on` field)
- Handles partial failures: if step 2 of 4 fails, determines if remaining steps are still viable

**Response Generator** (`response_generator.py`)
- Generates customer-facing text based on the resolution outcome
- Tone-matched to the customer's sentiment (frustrated customer gets more empathetic response)
- Includes actionable information: tracking links, return labels, refund timelines
- Template-based for common cases (fast, consistent), LLM-generated for complex/unusual cases

**Handoff Router** (`handoff.py`)
- Determines when and how to escalate to a human agent
- Escalation triggers:
  - `requires_human = True` from reasoning layer
  - Frustration score > 0.8
  - Customer explicitly requests human
  - Policy violation that requires manager approval
  - 3+ failed resolution attempts in conversation
- Packages full context (intent analysis, reasoning trace, customer history) for the receiving agent
- Supports warm handoff (context transfer) and cold handoff (ticket creation)

**Audit Logger** (`audit.py`)
- Logs every decision with full reasoning trace
- Stores: input, extracted entities, matched intents, reasoning steps, action plan, outcome
- Enables post-hoc analysis: "Why did the engine do X for this customer?"
- Powers the evaluation pipeline and continuous improvement feedback loop

---

## 7. API Design

### 7.1 Core Endpoints

```
POST /v1/intent/resolve
  Body: IntentRequest (or channel-specific payload with channel adapter)
  Response: ReasoningResult
  Description: Primary endpoint. Synchronous resolution.

POST /v1/intent/resolve/stream
  Body: IntentRequest
  Response: SSE stream of reasoning steps + final ReasoningResult
  Description: Streaming variant for chat UIs showing "thinking" state.

POST /v1/intent/batch
  Body: list[IntentRequest]
  Response: list[ReasoningResult]
  Description: Batch processing for email queues.

GET /v1/intent/catalog
  Response: Full intent taxonomy with example counts
  Description: Inspect current intent catalog.

POST /v1/intent/catalog/train
  Body: { intent: str, examples: list[str] }
  Description: Add training examples to intent catalog.

GET /v1/intent/audit/{request_id}
  Response: Full audit trail for a specific request.

POST /v1/webhook/ingest/{channel}
  Body: Channel-specific payload
  Description: Webhook receiver for platform events.
```

### 7.2 WebSocket API (for chat integration)

```
WS /v1/ws/{conversation_id}

Client → Server:
  { "type": "message", "text": "...", "metadata": {...} }

Server → Client:
  { "type": "thinking", "step": "Extracting entities..." }
  { "type": "thinking", "step": "Matching intent..." }
  { "type": "result", "data": ReasoningResult }
  { "type": "clarification", "question": "...", "options": [...] }
```

### 7.3 Authentication & Multi-tenancy

- API key per tenant (merchant), passed via `Authorization: Bearer <key>` header
- Tenant isolation at every layer: intent catalogs, policies, platform credentials, conversation history
- Rate limiting per tenant with configurable tiers

---

## 8. Technology Stack

### 8.1 Core Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.12+ | Ecosystem dominance for ML/NLP, Pydantic native |
| Package Manager | uv | Fast, modern, lockfile support |
| Web Framework | FastAPI | Async native, Pydantic integration, WebSocket support |
| Data Validation | Pydantic v2 | Schema enforcement everywhere, serialization |
| Task Queue | Celery + Redis | Async processing for batch, email, webhook ingestion |
| Database | PostgreSQL 16+ | Primary store, pgvector extension for embeddings |
| Cache | Redis | Session state, conversation history, rate limiting |
| Vector Store | pgvector (start) → Qdrant (scale) | Embedding similarity search |

### 8.2 ML/NLP Stack

| Component | Technology | Rationale |
|---|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Fast, good enough for intent matching |
| NER | spaCy + custom eCommerce model | Entity extraction, dependency parsing |
| Sentiment | Fine-tuned distilbert | Lightweight, fast inference |
| LLM (reasoning) | Claude API (Sonnet default, Opus for complex) | Structured output, function calling, reasoning quality |
| LLM (response gen) | Claude API (Haiku for templates, Sonnet for complex) | Cost-efficient at scale |

### 8.3 Infrastructure

| Component | Technology | Rationale |
|---|---|---|
| Containerization | Docker | Standard deployment |
| Orchestration | Kubernetes or Fly.io | Depending on scale needs |
| CI/CD | GitHub Actions | Standard, well-integrated |
| Monitoring | Prometheus + Grafana | Metrics, alerting |
| Logging | Structured JSON → Loki or Datadog | Searchable, traceable |
| Tracing | OpenTelemetry | Request tracing across layers |

---

## 9. Performance Requirements

| Metric | Target | Notes |
|---|---|---|
| Fast path latency (p50) | < 150ms | Embedding lookup + similarity match |
| Fast path latency (p99) | < 300ms | Including entity extraction |
| Reasoning path latency (p50) | < 2s | LLM call dominates |
| Reasoning path latency (p99) | < 5s | Complex compound with context enrichment |
| Throughput | 500 req/s per tenant | Horizontal scaling via K8s |
| Embedding search | < 20ms for 100k vectors | pgvector with HNSW index |
| Availability | 99.9% | Multi-region with failover |
| Intent accuracy (fast path) | > 92% F1 | Measured against labeled eval set |
| Intent accuracy (reasoning path) | > 88% F1 | Harder cases by definition |
| Compound detection recall | > 85% | Must catch multi-intent requests |

---

## 10. Evaluation Framework

### 10.1 Evaluation Datasets

Build labeled datasets across three tiers:

**Tier 1 — Golden set (500+ examples, manually labeled)**
- 50+ examples per primary intent category
- 100+ compound intent examples
- Stratified by channel, customer tier, complexity
- Used for regression testing — every release must maintain or improve scores

**Tier 2 — Silver set (2,000+ examples, semi-automated)**
- Production conversations with human-verified labels
- Continuously growing as agents confirm/correct engine decisions
- Used for training and evaluation

**Tier 3 — Synthetic set (10,000+ examples, LLM-generated)**
- Generated using Claude with persona prompts (angry customer, confused customer, VIP, etc.)
- Covers edge cases and long-tail intents
- Used for stress testing and coverage analysis

### 10.2 Metrics

| Metric | Description | Target |
|---|---|---|
| Precision per intent | Correct classifications / total predicted | > 90% |
| Recall per intent | Correct classifications / total actual | > 88% |
| F1 per intent | Harmonic mean of P and R | > 89% |
| Compound detection rate | Compound intents correctly identified | > 85% |
| Decomposition accuracy | All atomic intents extracted from compound | > 80% |
| Constraint extraction | Correct constraints identified | > 85% |
| Action plan viability | Plans that execute without errors | > 95% |
| Clarification rate | % of requests requiring clarification | < 8% |
| Handoff rate | % escalated to human | < 15% |
| Customer satisfaction (CSAT) | Post-interaction rating | > 4.2/5 |

### 10.3 Evaluation Pipeline

```
1. Load labeled dataset
2. Run each example through the engine
3. Compare: predicted intents vs. labels
4. Score: per-intent P/R/F1, macro averages, compound accuracy
5. Analyze failures: confusion matrix, false positive/negative breakdown
6. Regression check: compare against previous release scores
7. Generate report with recommendations
```

Runs automatically on every PR to main. Blocking: any intent F1 drops > 2% from baseline.

---

## 11. Security & Privacy

### 11.1 Data Handling

- All PII (names, emails, addresses, phone numbers) encrypted at rest (AES-256) and in transit (TLS 1.3)
- PII stripped from logs and audit trails (replaced with tenant-scoped pseudonymous IDs)
- LLM calls: PII is sent to the model for reasoning but not stored by the LLM provider (use Anthropic's zero-retention API option)
- Tenant data isolation: separate database schemas or row-level security

### 11.2 Access Control

- API keys scoped per tenant with configurable permissions
- Internal admin API with role-based access (read-only analyst, admin, superadmin)
- Audit log for all admin actions

### 11.3 Compliance

- GDPR: Data deletion API per customer, consent tracking, right to explanation (reasoning trace)
- CCPA: Same data deletion + opt-out of data sale (N/A since we don't sell data)
- SOC 2 Type II: Target for production readiness

---

## 12. Deployment Strategy

### 12.1 Phase 1 — MVP (Weeks 1-6)

**Goal:** Single-channel (chat) intent resolution for order lifecycle intents.

Scope:
- IntentRequest model + chat adapter
- Entity extraction (regex-based for order IDs, spaCy for general NER)
- Embedding-based fast path matching with 8 core intents
- Simple LLM reasoning for compound intents (no context enrichment yet)
- Shopify integration only
- FastAPI service with basic auth
- Golden eval set (200 examples)

**Definition of done:** > 85% F1 on order lifecycle intents, < 200ms fast path latency.

### 12.2 Phase 2 — Multi-Channel + Context (Weeks 7-12)

**Goal:** Email + form support. Full context enrichment.

Scope:
- Email and form channel adapters
- Context enricher pulling live order data from Shopify
- Policy engine with configurable return windows and auto-approval thresholds
- Sentiment-based routing (frustration → priority queue)
- Full intent taxonomy (all categories)
- Eval set expanded to 500+ examples
- Agent dashboard integration (reasoning trace visible to human agents)

### 12.3 Phase 3 — Production Hardening (Weeks 13-18)

**Goal:** Multi-tenant, production-grade.

Scope:
- Multi-tenant isolation
- WooCommerce + BigCommerce integrations
- WebSocket API for real-time chat
- Batch processing for email queues
- Monitoring, alerting, tracing (OpenTelemetry)
- SOC 2 readiness
- Synthetic eval set (10k examples)
- Performance optimization (caching, connection pooling, model optimization)

### 12.4 Phase 4 — Intelligence (Weeks 19+)

**Goal:** Continuous learning, advanced reasoning.

Scope:
- Feedback loop: agent corrections → retrain intent catalog
- A/B testing framework for reasoning prompts
- Multi-language support
- Clickstream integration (implicit intent detection)
- Predictive intent: anticipate needs before customer asks (e.g., delayed shipment → proactive WISMO)
- Custom fine-tuned embedding model trained on eCommerce intent data

---

## 13. Open Questions

| # | Question | Impact | Decision Needed By |
|---|---|---|---|
| 1 | pgvector vs. dedicated vector DB (Qdrant) from day one? | Infra complexity vs. scale ceiling | Phase 1 start |
| 2 | Fine-tune embedding model on eCommerce data or use general-purpose? | Accuracy vs. time investment | Phase 2 |
| 3 | How much context to include in LLM reasoning calls? (cost vs. accuracy) | Operating cost at scale | Phase 1 |
| 4 | Support real-time streaming of reasoning steps to chat UI? | UX quality vs. implementation complexity | Phase 2 |
| 5 | Build sentiment model or use off-the-shelf? | Accuracy on eCommerce text vs. dev time | Phase 1 |
| 6 | Single LLM provider (Anthropic) or multi-provider with fallback? | Reliability vs. complexity | Phase 3 |
| 7 | How to handle multilingual input? | Market scope | Phase 4 |
| 8 | Proactive intent (predict before ask) — how aggressive? | Customer experience vs. false positives | Phase 4 |

---

## 14. Success Criteria

The engine is successful when:

1. **Accuracy:** > 90% F1 across all intent categories on the golden eval set
2. **Speed:** < 200ms p50 for fast path, < 3s p50 for reasoning path
3. **Coverage:** < 10% of requests require human handoff for intent-related reasons (vs. policy/authority reasons)
4. **Adoption:** 3+ merchants running production traffic through the engine
5. **Economics:** Cost per resolution < $0.02 for fast path, < $0.15 for reasoning path (dominated by LLM costs)
6. **Improvement:** Measurable week-over-week accuracy improvement from feedback loop

---

*This specification is a living document. Update version number and date with each revision.*
