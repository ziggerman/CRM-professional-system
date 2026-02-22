# AEL CRM — AI-Powered Sales Command Center

A professional, high-performance Lead Management system built with **FastAPI**, **Aiogram (Telegram Bot)**, **Celery**, and **OpenAI**. Designed to bridge the gap between cold lead generation and hot sales conversion using AI advisory and robust pipeline security.

---

## 🏗 System Architecture & How it Works

The system follows a clean, modular architecture designed for scalability and maintainability:

### 📱 Telegram Bot (The "Frontend")
- **Asynchronous UI**: Built with `aiogram 3.x`, featuring a rich, emoji-driven interface.
- **State Management**: Uses FSM (Finite State Machine) for complex flows like `Search`, `Lead Creation`, and `Adding Notes`.
- **Role-Aware**: Dynamic UI rendering based on User Roles (Agent, Manager, Admin).

### ⚙️ FastAPI Backend (The "Engine")
- **Service Layer Pattern**: All business logic is encapsulated in `app/services`, keeping routers thin and focused on validation.
- **Repository Pattern**: Data access is isolated, supporting complex filtering and relationships via SQLAlchemy.
- **API Security**: Rigid RBAC (Role-Based Access Control) enforced via custom FastAPI dependencies.

### 🤖 AI Layer (The "Advisor")
- **OpenAI Integration**: Uses `gpt-4o-mini` for lead scoring and qualitative recommendation.
- **Cache Integrity**: Implements Redis caching with deterministic keys (SHA-256) to prevent redundant AI calls.
- **Stale Protection**: Automatically detects if a lead's data has changed significantly since the last analysis.

### 📦 Infrastructure
- **PostgreSQL/SQLite**: Relational storage for leads, sales, history, and user roles.
- **Redis**: Acts as the Celery broker and high-performance result/AI cache.
- **Celery**: Handles intensive background tasks like CSV generation and stale lead notifications.
- **WebSockets**: Provides low-latency dashboard reload signals to active clients.

---

## 🧠 AI Integration: Where, Why & What

AI in AEL CRM is designed as an **advisory layer**, not an autonomous decision-maker.

### Where and Why?
- **Lead Scoring**: AI analyzes lead activity and metadata to assign a "Warmth Score" (0.0 - 1.0). This helps humans prioritize high-value leads.
- **Recommendations**: It provides a textual reason for the score, suggesting whether to "transfer to sales", "keep nurturing", or "discard as low quality".
- **Objective Analysis**: Unlike humans, AI is not biased by conversation fatigue; it evaluates lead features against historical "winning" patterns.

### What Data is "AI"?
The following fields are strictly AI-generated:
- `ai_score`: A float value representing conversion probability.
- `ai_recommendation`: A structured suggestion (`transfer` / `nurture` / `discard`).
- `ai_reason`: The logic behind the recommendation.
- `last_ai_analysis_at`: Timestamp to track data freshness.

### AI Input Features (Isolation)
To maintain privacy and focus, the AI only sees **anonymous metadata**:
- `Source`: (Scanner vs Partner vs Manual)
- `Stage`: (Pipeline positioning)
- `Message Count`: (Engagement proxy)
- `Domain Presence`: (Qualification requirement)
- `Lead Age`: (Urgency signal)

---

## ⚖️ Human-in-the-Loop: Decisions

We maintain a strict boundary where **Humans make all state-changing decisions**:

| Decision | Human Role | AI Role |
|----------|------------|---------|
| **Stage Transitions** | Agent / Manager | None |
| **Lead Assignment** | Manager / Admin | None |
| **Final Transfer to Sales** | Manager | Provides advisory score |
| **Marking as "Lost"** | Agent / Manager | None |
| **File Attachments** | Agent (Manual Upload) | None |
| **CSV Export** | Admin | None |

---

## 🛠 What would be complicated in a Real Project? (Production Readiness)

While this MVP is robust, moving to a global enterprise scale would involve:

1.  **AI Feedback Loops (RLHF)**: Implementing a system where AI "learns" from actual closed sales. If AI recommended a lead that eventually "lost" in sales, the model should be fine-tuned.
2.  **Row-Level Security (RLS)**: True Multi-tenancy. Ensuring that Company A can never see Company B's leads even at the database driver level.
3.  **Advanced Vector Search**: Moving beyond `ILIKE` to semantic search using OpenAI Embeddings in a Vector DB (Pinecone/Milvus) for deep conversation analysis.
4.  **Complex Permissions**: Moving from internal User IDs to a full-blown Auth0/Cognito integration with OAuth2 scopes.
5.  **Circuit Breakers**: Protecting against OpenAI or Redis downtime using patterns that allow the CRM to function in "Offline/Safe Mode".
6.  **Observability Stack**: Integration with Datadog/Sentry for real-time alerting on API latency or Bot crash rates.
7.  **Data Sovereignty**: Implementing S3 storage with presigned URLs and encryption at rest for sensitive file attachments.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
cp .env.example .env
# Required: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, API_SECRET_TOKEN
```

### 2. Launch with Docker (Recommended)
```bash
docker-compose up -d --build
```

### 3. Manual Startup
```bash
# Apply Migrations
alembic upgrade head

# Run Celery Worker
celery -A app.celery.config worker --loglevel=info

# Run FastAPI
uvicorn main:app --reload
```

---
*Developed for Ascend Edge Ltd — CRM Modernization Initiative.*
