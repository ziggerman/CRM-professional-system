# CRM Lead Management System - Інструкція

## Зміст
1. [Встановлення](#встановлення)
2. [Запуск](#запуск)
3. [API Endpoints](#api-endpoints)
4. [Робота з лідами](#робота-з-лідами)
5. [AI Аналіз](#ai-аналіз)
6. [Transfer до продажів](#transfer-до-продажів)
7. [Telegram Bot](#telegram-bot)
8. [Тестування](#тестування)

---

## Встановлення

### 1. Клонування репозиторію
```bash
git clone <repo-url>
cd TZ---AEL
```

### 2. Створення віртуального середовища
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# або
venv\Scripts\activate     # Windows
```

### 3. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 4. Налаштування .env
```bash
cp .env.example .env
```

Відредагуй `.env`:
```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./crm.db

# Redis (для кешування AI)
REDIS_URL=redis://localhost:6379

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_here

# App
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true

# AI Settings
MIN_TRANSFER_SCORE=0.6
AI_CACHE_TTL=3600
AI_ANALYSIS_STALE_DAYS=7
```

### 5. Ініціалізація бази даних
```bash
python3 -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

---

## Запуск

### Варіант 1: Локально (без Docker)

```bash
# Термінал 1: Запуск API
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Термінал 2: Запуск Telegram бота (опціонально)
python3 -m app.bot.handlers
```

### Варіант 2: Docker

```bash
# Збірка та запуск
docker-compose up -d

# Перегляд логів
docker-compose logs -f
```

### Перевірка роботи

```bash
# Health check
curl http://localhost:8000/health

# API документація
open http://localhost:8000/docs
```

---

## API Endpoints

### Leads

| Method | Endpoint | Опис |
|--------|----------|------|
| POST | `/api/v1/leads` | Створити ліда |
| GET | `/api/v1/leads` | Список лідів |
| GET | `/api/v1/leads/{id}` | Отримати ліда |
| PATCH | `/api/v1/leads/{id}/stage` | Змінити етап |
| POST | `/api/v1/leads/{id}/messages` | Оновити активність |
| POST | `/api/v1/leads/{id}/analyze` | AI аналіз |
| POST | `/api/v1/leads/{id}/transfer` | Передати в продажі |

### Sales

| Method | Endpoint | Опис |
|--------|----------|------|
| GET | `/api/v1/sales` | Список продажів |
| GET | `/api/v1/sales/{id}` | Отримати sale |
| PATCH | `/api/v1/sales/{id}/stage` | Змінити етап |

### Dashboard

| Method | Endpoint | Опис |
|--------|----------|------|
| GET | `/api/v1/dashboard` | Загальна статистика |
| GET | `/api/v1/dashboard/leads-by-stage` | Ліди по етапах |
| GET | `/api/v1/dashboard/sales-by-stage` | Продажі по етапах |
| GET | `/api/v1/dashboard/conversion-funnel` | Воронка конверсії |

---

## Робота з лідами

### 1. Створення ліда

```bash
curl -X POST http://localhost:8000/api/v1/leads \
  -H "Content-Type: application/json" \
  -d '{
    "source": "partner",
    "business_domain": "first"
  }'
```

**Відповідь:**
```json
{
  "id": 1,
  "source": "partner",
  "stage": "new",
  "business_domain": "first",
  "message_count": 0,
  "ai_score": null,
  ...
}
```

### 2. Просування по етапах

Етапи лідів: `new` → `contacted` → `qualified` → `transferred` / `lost`

```bash
# Неможливо пропускати етапи!
# Правильно:
curl -X PATCH http://localhost:8000/api/v1/leads/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "contacted"}'

# Помилка (пропуск етапу):
curl -X PATCH http://localhost:8000/api/v1/leads/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "qualified"}'
# Помилка: "Cannot transition from 'new' to 'qualified'"
```

### 3. Оновлення активності

```bash
curl -X POST http://localhost:8000/api/v1/leads/1/messages \
  -H "Content-Type: application/json" \
  -d '{"increment": 5}'
```

---

## AI Аналіз

### Запуск аналізу

```bash
curl -X POST http://localhost:8000/api/v1/leads/1/analyze
```

**Відповідь:**
```json
{
  "score": 0.75,
  "recommendation": "transfer_to_sales",
  "reason": "High activity, partner source, clear business domain"
}
```

### Примусовий переаналіз

```bash
curl -X POST "http://localhost:8000/api/v1/leads/1/analyze?force=true"
```

### Що аналізує AI

| Параметр | Опис |
|----------|------|
| source | Джерело (scanner/partner/manual) |
| stage | Поточний етап |
| message_count | Кількість повідомлень |
| has_business_domain | Чи вказаний домен |
| business_domain | Домен (first/second/third) |
| days_since_created | Днів від створення |

### Рекомендації AI

| Score | Рекомендація |
|-------|--------------|
| 0.0-0.3 | `lost` - низька якість |
| 0.3-0.6 | `continue_nurturing` - потрібно ще роботи |
| 0.6-1.0 | `transfer_to_sales` - готовий до продажів |

---

## Transfer до продажів

### Умови transfer (hard gates)

1. ✅ Лід ще не transferred
2. ✅ AI score >= 0.6
3. ✅ Business domain вказаний

### Transfer

```bash
curl -X POST "http://localhost:8000/api/v1/leads/1/transfer?amount=10000"
```

**Відповідь:**
```json
{
  "id": 1,
  "lead_id": 1,
  "stage": "new",
  "amount": 10000,
  "created_at": "2026-02-21T16:55:38.144323",
  ...
}
```

### Просування продажу

Етапи продажів: `new` → `kyc` → `agreement` → `paid` / `lost`

```bash
curl -X PATCH http://localhost:8000/api/v1/sales/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "kyc"}'
```

---

## Telegram Bot

### Запуск

```bash
python3 -m app.bot.handlers
```

### Команди

| Команда | Опис |
|---------|------|
| /start | Головне меню |
| /leads | Список лідів |
| /sales | Список продажів |
| /stats | Статистика |

### Приклад роботи

```
User: /start
Bot: 👋 Вітаю! Оберіть дію:
       📋 Leads
       💰 Sales
       📊 Статистика

User: 📋 Leads
Bot: [Кнопки з лідами]
```

---

## Тестування

### Приклад повного циклу

```bash
# 1. Створити ліда
curl -X POST http://localhost:8000/api/v1/leads \
  -H "Content-Type: application/json" \
  -d '{"source": "partner", "business_domain": "first"}'

# 2. Пройти етапи: new → contacted → qualified
curl -X PATCH http://localhost:8000/api/v1/leads/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "contacted"}'

curl -X PATCH http://localhost:8000/api/v1/leads/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "qualified"}'

# 3. Запустити AI аналіз
curl -X POST http://localhost:8000/api/v1/leads/1/analyze

# 4. Перевірити score (має бути >= 0.6)
# Якщо score < 0.6 - продовжуємо роботу з лідом

# 5. Transfer до продажів
curl -X POST "http://localhost:8000/api/v1/leads/1/transfer?amount=50000"

# 6. Просування продажу
curl -X PATCH http://localhost:8000/api/v1/sales/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "kyc"}'

curl -X PATCH http://localhost:8000/api/v1/sales/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "agreement"}'

curl -X PATCH http://localhost:8000/api/v1/sales/1/stage \
  -H "Content-Type: application/json" \
  -d '{"stage": "paid"}'
```

---

## Структура проєкту

```
TZ---AEL/
├── app/
│   ├── api/v1/          # API endpoints
│   │   ├── leads.py     # Ліди
│   │   ├── sales.py     # Продажі
│   │   ├── dashboard.py # Статистика
│   │   └── automation.py # Автоматизація
│   ├── core/            # Конфігурація
│   │   ├── config.py    # Налаштування
│   │   └── database.py  # База даних
│   ├── models/          # SQLAlchemy моделі
│   ├── repositories/    # Data access layer
│   ├── schemas/         # Pydantic схеми
│   ├── services/        # Бізнес-логіка
│   │   ├── lead_service.py
│   │   └── transfer_service.py
│   └── ai/              # AI сервіс
│       ├── ai_service.py
│       └── prompts.py
├── main.py              # Точка входу
├── docker-compose.yml   # Docker
└── requirements.txt     # Залежності
```

---

## Troubleshooting

### Помилка: "AI service unavailable"
- Перевір `OPENAI_API_KEY` в `.env`

### Помилка: "Database locked"
- Зупини всі процеси перед перезапуском

### Помилка: "Cannot transition stage"
- Перевір послідовність етапів

### Telegram conflict error
```bash
# Видали webhook
curl -s "https://api.telegram.org/bot<TOKEN>/deleteWebhook"

# Зачекай 30 секунд
# Перезапусти бота
```

---

## Контакти

Розробник: [Ваше ім'я]
Email: [Ваш email]
