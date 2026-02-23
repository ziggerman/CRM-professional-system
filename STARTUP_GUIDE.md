# 🚀 STARTUP GUIDE - CRM Lead Management System

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up -d
```

This will start:
- PostgreSQL (port 5432)
- Redis (port 6379)
- API Server (port 8000)
- Celery Worker
- Celery Beat

### Option 2: Local Development

#### 1. Start Redis
```bash
# Install Redis if needed
brew install redis  # macOS
# or
sudo apt install redis-server  # Ubuntu

# Start Redis
redis-server
```

#### 2. Start API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. Start Telegram Bot (in new terminal)
```bash
python run_bot.py
```

---

## Environment Variables (.env)

### Required
```
# Database
DATABASE_URL=sqlite+aiosqlite:///./crm.db

# Telegram
TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here
TELEGRAM_ADMIN_IDS=[123456789]

# OpenAI (for AI features)
OPENAI_API_KEY=your-openai-api-key-hereOPENAI_MODEL=gpt-4o-mini
```

### Optional (for Voice Recognition)
```
# HuggingFace - FREE voice transcription
HUGGINGFACE_TOKEN=hf_...
```

---

## Testing Checklist

After starting, verify:

- [ ] Bot responds to /start
- [ ] Main menu displays correctly
- [ ] Can create a new lead
- [ ] AI draft lead flow shows buttons: Зберегти / Редагувати / Змінити питання
- [ ] Can view leads list
- [ ] Can edit lead stage
- [ ] Copilot works (🤖 Copilot button)
  - [ ] Voice messages are transcribed
  - [ ] Text commands also work in Copilot mode
- [ ] AI analysis shows next-step navigation buttons
  - [ ] Contacted / Qualify / Transfer
  - [ ] Додати нотатку / Наступне питання / Картка ліда- [ ] Sales pipeline accessible
- [ ] Statistics show correctly

---

## Copilot Features 🤖

When Copilot mode is active, users can use both text and voice:
| Command | Action |
|---------|--------|
| "додай ліда" | Create new lead |
| "покажи ліди" | Show leads list |
| "статистика" | Show statistics |
| "знайди [name]" | Search for lead |
| Voice message | Transcribe & process |

### Interactive Buttons

1) **Add Lead draft actions**
- ✅ Зберегти
- ✏️ Редагувати
- ❓ Змінити питання

2) **AI Analysis next steps**
- 📞 Contacted
- ✅ Qualify
- 🚀 Transfer
- 📝 Додати нотатку
- ➡️ Наступне питання
- 📄 Картка ліда

### Clarification Logic
- If intent confidence is low, bot asks to rephrase (no risky action execution).
- If required fields are missing, bot asks targeted follow-up questions:
  - create: name/phone/email
  - analyze: lead_id
  - note: lead_id + content
2. Add HUGGINGFACE_TOKEN to .env for free transcription
3. Or ensure OPENAI_API_KEY is set for paid Whisper

### Database errors?
1. Run migrations: `alembic upgrade head`
2. Or recreate: delete crm.db and restart

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Telegram Bot   │────▶│   API Server    │
│  (run_bot.py)  │     │  (FastAPI)      │
└─────────────────┘     └────────┬────────┘
                                  │
                         ┌────────┴────────┐
                         │                 │
                    ┌────▼────┐     ┌────▼────┐
                    │   AI    │     │  Database│
                    │ Service │     │ (SQLite) │
                    └─────────┘     └──────────┘
```

---

## Default Admin ID

Your Telegram ID for admin access: `585761464`
