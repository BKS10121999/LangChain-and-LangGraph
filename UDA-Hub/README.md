# UDA-Hub: Multi-Agent Customer Support System

A multi-agent customer support system for **CultPass** — a cultural experiences subscription service. Built with LangGraph, this project implements intelligent ticket classification, knowledge-based resolution, and automated escalation using a supervisor-pattern orchestration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Message                              │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
                    ┌──────────────────┐
                    │   Memory Agent   │  ← Retrieves customer history
                    └────────┬─────────┘    & long-term preferences
                             ▼
                    ┌──────────────────┐
               ┌───►│   Supervisor     │◄──────────────────┐
               │    └────────┬─────────┘                    │
               │             │ (routes by state)            │
               │    ┌────────┼──────────┐                   │
               │    ▼        ▼          ▼                   │
        ┌────────────┐ ┌──────────┐ ┌────────────┐         │
        │ Classifier │ │ Resolver │ │ Escalation │         │
        └─────┬──────┘ └────┬─────┘ └─────┬──────┘         │
              │              │             │                 │
              └──────────────┴─────────────┴─────────────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │   END    │
                              └──────────┘
```

### Agent Roles

| Agent | Module | Responsibility |
|-------|--------|----------------|
| **Memory** | `agentic/agents/memory_agent.py` | Matches returning customers, retrieves interaction history, stores preferences |
| **Classifier** | `agentic/agents/classifier.py` | Categorizes tickets (billing, technical, account, subscription, reservation, general) and assigns urgency (low/medium/high/critical) |
| **Resolver** | `agentic/agents/resolver.py` | Searches knowledge base, invokes account/ticket tools, generates responses with confidence scoring |
| **Escalation** | `agentic/agents/escalation.py` | Creates escalation packages for critical/unresolvable issues with priority assignment |
| **Supervisor** | `agentic/workflow.py` | Routes between agents based on classification, urgency, and resolution confidence |

### Tools

| Tool | Module | Purpose |
|------|--------|---------|
| `search_knowledge_base` | `agentic/tools/knowledge_tools.py` | Keyword-based article retrieval with relevance scoring |
| `lookup_user` | `agentic/tools/account_tools.py` | Look up CultPass user by email |
| `get_subscription_details` | `agentic/tools/account_tools.py` | Retrieve subscription tier, status, and quota |
| `check_user_reservations` | `agentic/tools/account_tools.py` | List user's experience reservations |
| `update_ticket_status` | `agentic/tools/ticket_tools.py` | Update ticket status (open → resolved/escalated) |
| `log_ticket_message` | `agentic/tools/ticket_tools.py` | Persist conversation messages to ticket history |
| `get_ticket_history` | `agentic/tools/ticket_tools.py` | Retrieve full conversation history for a ticket |
| `retrieve_customer_history` | `agentic/agents/memory_agent.py` | Long-term memory retrieval for personalization |
| `store_customer_preference` | `agentic/agents/memory_agent.py` | Persist customer preferences across sessions |

## Getting Started

### Dependencies

```
langchain-core
langgraph
sqlalchemy
python-dotenv
```

Full list in `requirements.txt`.

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Database Setup

Run the notebooks in order:

1. **`01_external_db_setup.ipynb`** — Creates `data/external/cultpass.db` with users, experiences, subscriptions, and reservations for all 6 users.
2. **`02_core_db_setup.ipynb`** — Creates `data/core/udahub.db` with accounts, 15 knowledge base articles, users, tickets, and message history.
3. **`03_agentic_app.ipynb`** — Runs the multi-agent workflow with interactive chat and state inspection.

## Project Structure

```
UDA-Hub/
├── agentic/
│   ├── workflow.py              # LangGraph StateGraph orchestrator
│   ├── agents/
│   │   ├── classifier.py       # Ticket classification agent
│   │   ├── resolver.py         # Knowledge-based resolution agent
│   │   ├── escalation.py       # Escalation handling agent
│   │   └── memory_agent.py     # Long-term memory agent
│   └── tools/
│       ├── account_tools.py    # CultPass DB account tools
│       ├── ticket_tools.py     # UDA-Hub ticket management tools
│       └── knowledge_tools.py  # Knowledge base retrieval (RAG)
├── data/
│   ├── external/               # CultPass source data (JSONL + SQLite)
│   ├── core/                   # UDA-Hub application database
│   └── models/                 # SQLAlchemy ORM models
├── tests/
│   └── test_agents.py          # Agent unit tests
├── 01_external_db_setup.ipynb  # External DB initialization
├── 02_core_db_setup.ipynb      # Core DB + knowledge base setup
├── 03_agentic_app.ipynb        # Interactive agent demo
├── utils.py                    # Shared utilities (chat_interface, DB helpers)
└── setup_db.py                 # Programmatic DB setup script
```

## Testing

```bash
pytest tests/ -v
```

### Test Coverage

- **Classifier**: Verifies correct category and urgency assignment for different message types
- **Resolver**: Tests knowledge retrieval and confidence scoring
- **Escalation**: Validates escalation package structure
- **Workflow**: End-to-end routing from classification through resolution/escalation

## Memory & State Management

- **Short-term (session)**: `MemorySaver` checkpointer maintains state within a `thread_id` session — messages, tool usage, and routing decisions are inspectable via `orchestrator.get_state_history()`
- **Long-term (cross-session)**: Customer interactions are persisted to the UDA-Hub database. Returning customers are recognized and their history informs resolution.
- **Preferences**: Stored as system messages in the ticket history, retrieved on subsequent contacts.

## Built With

* [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration framework
* [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework (tools, messages)
* [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
* [SQLite](https://www.sqlite.org/) - Embedded database engine

## License

[License](../LICENSE.md)
