# AEL CRM — Повний план покращень та розвитку системи

> **Контекст**: Документ фіксує всі виявлені проблеми поточної реалізації, пропоновані покращення та roadmap розвитку системи від MVP до production-ready CRM. Розділи впорядковані за пріоритетом.

---

## ✅ Оновлення документації (2026-02-23)

Нижче зафіксовано, що вже реалізовано в Copilot і відображено в актуальній документації (`README.md`, `STARTUP_GUIDE.md`):

### Реалізовано в боті
- Інтерактивний флоу створення ліда після AI-розпізнавання:
  - `✅ Зберегти`
  - `✏️ Редагувати`
  - `❓ Змінити питання`
- Інтерактивний флоу після AI-аналізу ліда:
  - `📞 Contacted`
  - `✅ Qualify`
  - `🚀 Transfer`
  - `📝 Додати нотатку`
  - `➡️ Наступне питання`
  - `📄 Картка ліда`
- Логіка уточнення наміру (safe-guard): при низькій впевненості NLU бот просить переформулювати запит.
- Slot-filling логіка для обов'язкових даних:
  - create: мінімум name/phone/email
  - analyze: обов'язково `lead_id`
  - note: `lead_id` + `content`

### Що це дає бізнесу
- Менше помилкових записів у спільну базу CRM.
- Контрольований human-in-the-loop перед збереженням/зміною даних.
- Швидка навігація наступних кроків по лідові прямо з кнопок.

---

## Зміст

1. [Виявлені проблеми в поточному коді](#1-виявлені-проблеми-в-поточному-коді)
2. [Критичні виправлення](#2-критичні-виправлення)
3. [Покращення бізнес-логіки](#3-покращення-бізнес-логіки)
4. [Покращення AI-шару](#4-покращення-ai-шару)
5. [Покращення API та DX](#5-покращення-api-та-dx)
6. [Покращення бази даних та репозиторіїв](#6-покращення-бази-даних-та-репозиторіїв)
7. [Безпека та RBAC](#7-безпека-та-rbac)
8. [Спостережуваність та логування](#8-спостережуваність-та-логування)
9. [Інфраструктура та DevOps](#9-інфраструктура-та-devops)
10. [Тестування](#10-тестування)
11. [Telegram Bot — покращення UX](#11-telegram-bot--покращення-ux)
12. [Dashboard — React](#12-dashboard--react)
13. [Production-roadmap](#13-production-roadmap)

---

## 1. Виявлені проблеми в поточному коді

### 1.1 Розбіжність констант між модулями

**Файл**: `app/ai/prompts.py` (рядок 50–51 та 55)

```python
# Поточний стан — НЕВІРНО:
VALID_LEAD_SOURCES = frozenset({'WEB', 'REFERRAL', 'SOCIAL', 'MANUAL'})
VALID_LEAD_STAGES = frozenset({'NEW', 'CONTACTED', 'QUALIFIED', 'NEGOTIATION', 'CLOSED'})
```

**Проблема**: У `app/models/lead.py` визначені `LeadSource.SCANNER`, `LeadSource.PARTNER`, `LeadSource.REGISTRATION` тощо, а в `prompts.py` ці значення відсутні як валідні. Якщо лід зі джерелом `SCANNER` потрапляє на AI-аналіз, валідація впаде з `ValueError`. Аналогічно, `VALID_LEAD_STAGES` містить `NEGOTIATION/CLOSED` замість реальних `CONTACTED/QUALIFIED/TRANSFERRED/LOST`.

**Виправлення**:
```python
# Потрібно синхронізувати з models/lead.py:
VALID_LEAD_SOURCES = frozenset({
    'WEB', 'REFERRAL', 'SOCIAL', 'MANUAL', 'SCANNER', 'PARTNER',
    'REGISTRATION', 'CALLBACK', 'LEAD_MAGNET', 'MESSAGE'
})
VALID_LEAD_STAGES = frozenset({'NEW', 'CONTACTED', 'QUALIFIED', 'TRANSFERRED', 'LOST'})
```

---

### 1.2 Дублювання імпорту залежності

**Файл**: `app/api/v1/leads.py` (рядок 12 і 27)

```python
from app.core.deps import get_lead_service, get_transfer_service  # рядок 12
# ...
from app.core.deps import get_lead_service, get_transfer_service, get_history_repo  # рядок 27
```

Один і той самий символ імпортується двічі. Перший імпорт є зайвим і може вводити в оману.

---

### 1.3 Синхронний запис файлу в async-ендпоїнті

**Файл**: `app/api/v1/leads.py` (рядок 498–499)

```python
# Поточний — БЛОКУЄ event loop:
with open(file_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)
```

**Виправлення** — використовувати `aiofiles`:
```python
import aiofiles
async with aiofiles.open(file_path, "wb") as buffer:
    content = await file.read()
    await buffer.write(content)
```

---

### 1.4 Неправильний imort `LeadHistory` всередині методу

**Файл**: `app/services/lead_service.py` (рядки 118, 209)

```python
# В середині методів:
from app.models.history import LeadHistory
```

Такі внутрішні імпорти — антипаттерн. Вони виконуються при кожному виклику методу. `LeadHistory` слід імпортувати на рівні модуля.

---

### 1.5 Notes pagination — неефективне завантаження всіх нотаток

**Файл**: `app/api/v1/leads.py` (рядок 279–286)

```python
notes = lead.notes          # Завантажує ВСІ нотатки в пам'ять
total = len(notes)
start = (page - 1) * page_size
page_notes = notes[start:end]  # Потім робить Python-slice
```

При 10 000 нотаток — завантажуємо все в RAM. Потрібен SQL-рівень пагінації через `OFFSET/LIMIT`.

---

### 1.6 `datetime.now()` без timezone в attachment endpoint

**Файл**: `app/api/v1/leads.py` (рядок 495)

```python
safe_name = f"{lead_id}_{int(datetime.now().timestamp())}_{file.filename}"
```

Використовується naive datetime. Потрібно `datetime.now(UTC)`.

---

### 1.7 Ненадійна логіка `user.current_leads` 

**Файл**: `app/api/v1/leads.py` (методи assign/unassign/reassign)

`current_leads` оновлюється вручну (`+= 1`, `-= 1`). При паралельних запитах або помилках це призведе до race condition та некоректного лічильника. Потрібно або `SELECT COUNT(*)` у реальному часі, або оптимістичне блокування.

---

## 2. Критичні виправлення

### 2.1 Синхронізація констант AI-валідатора з моделями

- [ ] Оновити `VALID_LEAD_SOURCES` у `prompts.py`, щоб відображав усі `LeadSource` enum-значення
- [ ] Оновити `VALID_LEAD_STAGES` у `prompts.py` на `ColdStage` enum-значення
- [ ] Додати unit-тест, що перевіряє: кожен `LeadSource` проходить `_validate_lead_features`

### 2.2 Async файловий I/O

- [ ] Додати `aiofiles` до `requirements.txt`
- [ ] Переписати upload endpoint на async I/O
- [ ] Додати обмеження розміру файлу (наразі відсутнє — DoS вектор)
- [ ] Додати whitelist MIME-типів

### 2.3 Виправлення imports

- [ ] Прибрати дублікат імпорту в `leads.py`
- [ ] Перенести всі `from app.models.history import LeadHistory` у top-level imports сервісів

---

## 3. Покращення бізнес-логіки

### 3.1 Розширення Stage Machine

**Поточний стан**: Лід може деградувати в `LOST` з будь-якого етапу, але не може повернутись назад з нетермінального етапу.

**Пропозиція**: Ввести концепцію `stage_back` для `CONTACTED → NEW` у разі втрати контакту (повернення на cold outreach). Це відповідає реальній CRM-практиці.

```python
# Нова константа:
REVERSIBLE_STAGE_TRANSITIONS = {
    ColdStage.CONTACTED: ColdStage.NEW,     # Lost contact
    ColdStage.QUALIFIED: ColdStage.CONTACTED,  # Lost qualification
}
```

Метод в `LeadService`:
```python
async def rollback_stage(self, lead: Lead, reason: str) -> Lead:
    """Дозволяє повернення на один крок назад з обов'язковим reason."""
```

---

### 3.2 Lead Scoring — поле `quality_tier`

Додати денормалізоване поле `quality_tier: Mapped[str | None]`, що автоматично оновлюється при збереженні AI score:
- `HOT` (score ≥ 0.8)
- `WARM` (score 0.6–0.8)
- `COLD` (score 0.3–0.6)
- `DEAD` (score < 0.3)

Це дає змогу фільтрувати без float-порівнянь у DB і відображати теплову карту в дашборді.

---

### 3.3 Lead Aging — автоматичне попередження

Нова поле `last_activity_at: Mapped[datetime]` (оновлюється при кожному `increment_messages` або `transition_stage`). Celery задача `check_stale_leads` повинна:

1. Шукати ліди, де `last_activity_at < NOW() - N days` і stage ∈ `{CONTACTED, QUALIFIED}`
2. Відправляти сповіщення assigned_to менеджеру
3. Логувати `NURTURE` запис у history (вже реалізовано `nurture_lead`, потрібно підключити до Celery)

---

### 3.4 Transfer — валідація стану ліда

**Поточний стан**: Transfer перевіряє `ai_score >= 0.6` і `business_domain IS NOT NULL`.

**Пропозиція** — додати третю умову: лід повинен бути на стадії `QUALIFIED`. Трансфер з `CONTACTED` не має сенсу в реальній CRM-практиці.

```python
# У TransferService:
if lead.stage not in (ColdStage.QUALIFIED,):
    raise TransferError(
        f"Lead must be in QUALIFIED stage before transfer. Current: {lead.stage.value}"
    )
```

---

### 3.5 Дублікат-контроль лідів

Додати унікальну перевірку при створенні ліда за `email` та `phone`. Якщо вже існує, повертати `409 Conflict` з лінком на існуючий лід.

```python
class DuplicateLeadError(Exception):
    def __init__(self, existing_id: int):
        self.existing_id = existing_id
```

---

### 3.6 Lead Score History

Зараз AI score перезаписується при кожному аналізі без збереження попереднього значення. Ввести окрему таблицю `lead_score_history`:

```sql
CREATE TABLE lead_score_history (
    id          SERIAL PRIMARY KEY,
    lead_id     INT NOT NULL REFERENCES leads(id),
    score       FLOAT NOT NULL,
    recommendation VARCHAR(64),
    reason      TEXT,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Це дає можливість будувати трендовий графік score над часом.

---

### 3.7 Sales Pipeline — додаткові бізнес-правила

**Поточний стан**: Sale може переходити в `LOST` з будь-якого нетермінального стану.

**Пропозиція**:
- `amount` є обов'язковим перед переходом у стан `AGREEMENT`
- При `PAID` — встановлювати `closed_at` timestamp і розраховувати `duration_days`
- При `LOST` — вимагати `lost_reason: str` (обов'язково)

```python
# Нові поля в Sale model:
lost_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
duration_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
```

---

## 4. Покращення AI-шару

### 4.1 Structured Output замість prompt engineering

**Поточний стан**: AI промпт вимагає конкретного JSON формату через текст. Якщо GPT відхиляється, парсинг ламається.

**Покращення** — використовувати OpenAI Structured Outputs (JSON Schema mode):

```python
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "lead_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "recommendation": {"type": "string", "enum": ["transfer_to_sales", "continue_nurturing", "lost"]},
                    "reason": {"type": "string"}
                },
                "required": ["score", "recommendation", "reason"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    messages=[...]
)
```

Це гарантує валідний JSON без необхідності try/except на парсинг.

---

### 4.2 AI Score Confidence Interval

Замість одного скалярного `score`, просити AI надавати:
- `score_mean: float` — центральна оцінка
- `score_confidence: float` — рівень впевненості (0.0–1.0)

Відображати в UI: `0.72 ± 0.08` замість просто `0.72`.

---

### 4.3 Fallback AI без OpenAI (Rule-Based Scoring)

При недоступності OpenAI API активувати rule-based fallback:

```python
def rule_based_score(lead: Lead) -> AIAnalysisResult:
    score = 0.0
    reasons = []

    # Source weight
    source_weights = {
        LeadSource.REFERRAL: 0.35, LeadSource.PARTNER: 0.30,
        LeadSource.WEB: 0.25, LeadSource.SCANNER: 0.20,
        LeadSource.SOCIAL: 0.15, LeadSource.MANUAL: 0.10,
    }
    score += source_weights.get(lead.source, 0.10)

    # Activity weight
    if lead.message_count >= 10:
        score += 0.25
        reasons.append("high activity")
    elif lead.message_count >= 5:
        score += 0.15
    elif lead.message_count >= 2:
        score += 0.08

    # Domain weight
    if lead.business_domain:
        score += 0.25
        reasons.append("domain defined")

    # Stage weight
    stage_weights = {
        ColdStage.QUALIFIED: 0.20, ColdStage.CONTACTED: 0.10, ColdStage.NEW: 0.0
    }
    score += stage_weights.get(lead.stage, 0.0)

    score = min(score, 1.0)
    recommendation = (
        "transfer_to_sales" if score >= 0.6
        else "continue_nurturing" if score >= 0.3
        else "lost"
    )
    return AIAnalysisResult(
        score=score,
        recommendation=recommendation,
        reason=f"Rule-based: {', '.join(reasons) or 'low signals'}. [AI OFFLINE]"
    )
```

---

### 4.4 AI Input Feature Engineering

Розширити фічі, що подаються в AI, без порушення privacy:

| Текуча фіча | Нова фіча | Чому |
|---|---|---|
| `message_count` | `message_velocity` (count / days) | Швидкість росту важливіша за абсолютне число |
| - | `has_phone AND has_email` | Повнота контактних даних |
| - | `has_company AND has_position` | B2B qualification signal |
| - | `notes_count` | Залученість менеджера |
| `days_since_created` | `days_since_last_activity` | Актуальніша свіжість |

---

### 4.5 AI Rate Limiting та Cost Control

Додати жорсткий ліміт на кількість AI-запитів:

```python
# Redis-based token bucket:
async def check_ai_rate_limit(lead_id: int) -> bool:
    key = f"ai:rate:{datetime.now(UTC).strftime('%Y-%m-%d')}"
    count = await redis.incr(key)
    await redis.expire(key, 86400)
    return count <= settings.AI_DAILY_LIMIT  # Default: 500
```

---

### 4.6 AI Audit Log

Кожен виклик AI фіксувати у таблиці `ai_audit_log`:

```sql
CREATE TABLE ai_audit_log (
    id          SERIAL PRIMARY KEY,
    lead_id     INT REFERENCES leads(id),
    model       VARCHAR(64),
    input_hash  CHAR(64),          -- SHA-256 input для дедуплікації
    score       FLOAT,
    recommendation VARCHAR(64),
    latency_ms  INT,
    tokens_used INT,
    cost_usd    FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

Цей лог дає змогу:
- Аналізувати cost per lead
- Виявляти аномалії в AI response
- Реалізувати RLHF feedback loop

---

## 5. Покращення API та DX

### 5.1 Стандартизований формат помилок

**Поточний стан**: Помилки повертаються як голий рядок у `detail`.

**Пропозиція** — RFC 7807 (Problem Details):

```json
{
  "type": "https://crm.example.com/errors/lead-stage-invalid",
  "title": "Invalid Stage Transition",
  "status": 400,
  "detail": "Cannot transition from 'NEW' to 'QUALIFIED'. Expected: 'CONTACTED'.",
  "instance": "/api/v1/leads/42/stage",
  "lead_id": 42,
  "current_stage": "NEW",
  "requested_stage": "QUALIFIED"
}
```

Це дозволяє клієнтам (бот, дашборд) реагувати на конкретні типи помилок без парсингу текстових рядків.

---

### 5.2 API Versioning Strategy

Поточна версія `v1` існує один вид. Рекомендація:
- Заморозити `v1` interface (breaking changes → `v2`)
- Додати `Deprecation` заголовок для застарілих ендпоїнтів
- Версіонувати через URL (`/api/v2/`) а не через header (простіше для debug)

---

### 5.3 Cursor-based Pagination

**Поточний стан**: Offset-based pagination (`page`, `page_size`).

**Проблема**: При 100 000 лідів `OFFSET 99000 LIMIT 50` — повний scan таблиці.

**Рішення** — Cursor-based:
```
GET /api/v1/leads?cursor=eyJpZCI6IDEwMH0&limit=50
Response: { "items": [...], "next_cursor": "eyJpZCI6IDE1MH0" }
```

Реалізується через `WHERE id > :last_id ORDER BY id LIMIT :limit`.

---

### 5.4 Bulk Operations API

Додати endpoint для пакетних операцій:

```
POST /api/v1/leads/bulk/stage      — масовий перехід стадії
POST /api/v1/leads/bulk/assign     — масове призначення менеджера
POST /api/v1/leads/bulk/analyze    — запуск AI аналізу для масиву лідів
DELETE /api/v1/leads/bulk          — масове видалення (Admin only)
```

---

### 5.5 Search/Filter розширення

Поточна фільтрація підтримує лише точний збіг. Додати:

```
GET /api/v1/leads?q=Nikolas              # Full-text search по name, email, company
GET /api/v1/leads?ai_score_min=0.6       # Пошук за AI score range
GET /api/v1/leads?ai_score_max=0.9
GET /api/v1/leads?created_after=2026-01-01
GET /api/v1/leads?stage=NEW,CONTACTED    # Multi-value filter
GET /api/v1/leads?sort=ai_score:desc     # Сортування
```

---

### 5.6 ETag та Conditional Requests

Для зменшення навантаження на API додати `ETag`:

```
GET /api/v1/leads/42
Response: ETag: "abc123", Cache-Control: max-age=30

GET /api/v1/leads/42
If-None-Match: "abc123"
Response: 304 Not Modified  ← без body
```

---

## 6. Покращення бази даних та репозиторіїв

### 6.1 Missing Database Indexes

Аналіз поточних запитів показує відсутні індекси:

```sql
-- Необхідні нові індекси:
CREATE INDEX idx_leads_ai_score ON leads(ai_score) WHERE ai_score IS NOT NULL;
CREATE INDEX idx_leads_stage_created ON leads(stage, created_at DESC);
CREATE INDEX idx_leads_assigned_stage ON leads(assigned_to_id, stage);
CREATE INDEX idx_leads_source_domain ON leads(source, business_domain);
CREATE INDEX idx_sales_stage_lead ON sales(stage, lead_id);
CREATE INDEX idx_history_lead_created ON lead_history(lead_id, created_at DESC);
CREATE INDEX idx_notes_lead_created ON lead_notes(lead_id, created_at DESC);
```

---

### 6.2 Soft Delete замість Hard Delete

**Поточний стан**: `DELETE /api/v1/leads/{id}` виконує `DELETE FROM leads`.

**Проблема**: Втрата всієї audit trail, history, notes, attachments.

**Рішення** — Soft delete:
```python
# Додати до Lead model:
is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
deleted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
```

Всі GET-запити автоматично фільтрують `WHERE is_deleted = FALSE` через Session Event або Repository base.

---

### 6.3 Repository Pattern — Base Class

Поточні репозиторії, очевидно, мають дублювання CRUD-коду. Ввести:

```python
class BaseRepository(Generic[T]):
    model: type[T]
    
    async def get_by_id(self, id: int) -> T | None: ...
    async def create(self, entity: T) -> T: ...
    async def save(self, entity: T) -> T: ...
    async def delete(self, entity: T) -> None: ...
    async def count(self, **filters) -> int: ...
```

---

### 6.4 Database Transaction Management

**Поточний стан**: Session flush/commit відбувається у різних місцях (сервіс і репозиторій).

**Рекомендація**: Unit of Work pattern — транзакція починається на рівні HTTP request і комітиться тільки при успішному завершенні handler. Rollback при будь-якому exception.

```python
# В middleware або dependency:
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        async with session.begin():
            yield session
        # AUTO COMMIT or ROLLBACK
```

---

### 6.5 Connection Pooling та Timeout

Додати до DB конфігурації:
```python
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,          # Перевірка зв'язку перед видачею
    pool_recycle=3600,           # Скидати з'єднання кожну годину
    connect_args={"command_timeout": 30},  # Query timeout
)
```

---

## 7. Безпека та RBAC

### 7.1 JWT Authentication замість Static Token

**Поточний стан**: `verify_api_token` перевіряє один статичний `API_SECRET_TOKEN` з `.env`.

**Проблема**: 
- Один токен для всіх — неможливо відкликати доступ для конкретного користувача
- Немає прив'язки токена до конкретного юзера/ролі
- Токен не має TTL

**Рішення** — JWT:
```
POST /api/v1/auth/login  →  { "access_token": "...", "expires_in": 3600 }
POST /api/v1/auth/refresh  →  новий access token
POST /api/v1/auth/logout  →  інвалідація через Redis blacklist
```

Payload JWT:
```json
{
  "sub": "user_id",
  "role": "manager",
  "exp": 1740000000,
  "jti": "unique-token-id"  // для blacklist
}
```

---

### 7.2 RBAC Matrix

Чіткіше визначити матрицю прав:

| Дія | Agent | Manager | Admin |
|-----|-------|---------|-------|
| Переглядати своїх лідів | ✅ | ✅ | ✅ |
| Переглядати всіх лідів | ❌ | ✅ | ✅ |
| Створювати лідів | ✅ | ✅ | ✅ |
| Призначати лідів | ❌ | ✅ | ✅ |
| Переходити стадії | ✅ | ✅ | ✅ |
| Запускати AI аналіз | ✅ | ✅ | ✅ |
| Передавати в продажі | ❌ | ✅ | ✅ |
| Видаляти лідів | ❌ | ❌ | ✅ |
| Переглядати дашборд | ❌ | ✅ | ✅ |
| Управляти користувачами | ❌ | ❌ | ✅ |
| Експорт CSV | ❌ | ✅ | ✅ |

---

### 7.3 Input Sanitization

Поточні поля `intent`, `pain_points`, `notes.content` не проходять XSS/injection sanitization. Для текстових полів додати:

```python
import bleach

def sanitize_text(value: str, max_length: int = 1024) -> str:
    cleaned = bleach.clean(value, tags=[], strip=True)
    return cleaned[:max_length].strip()
```

---

### 7.4 File Upload Security

```python
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif",
    "application/pdf", "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"File type {file.content_type} is not allowed")
    
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File exceeds 10 MB limit")
    
    # Magic bytes validation (not just MIME header):
    import magic
    detected = magic.from_buffer(content[:2048], mime=True)
    if detected not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, "File content does not match declared type")
```

---

### 7.5 Rate Limiting per User

Поточний `rate_limit.py` обмежує по IP. Потрібно додати per-user rate limiting після JWT integraiton:

```
Lead Creation:      10/hour per user
AI Analysis:        20/day per user
Stage Transitions:  100/hour per user
File Uploads:       5/hour per user
```

---

## 8. Спостережуваність та логування

### 8.1 Structured Logging (JSON)

**Поточний стан**: Логи у plain text формат.

**Рекомендація** — `structlog` з JSON форматом:

```python
import structlog

log = structlog.get_logger()

log.info(
    "lead_stage_transition",
    lead_id=lead.id,
    old_stage=current.value,
    new_stage=new_stage.value,
    changed_by=changed_by,
    duration_ms=elapsed,
)
```

Вихід:
```json
{"event": "lead_stage_transition", "lead_id": 42, "old_stage": "NEW", "new_stage": "CONTACTED", "changed_by": "manager@example.com", "duration_ms": 12}
```

---

### 8.2 Distributed Tracing

Додати `opentelemetry-sdk` + Jaeger для трасування запитів від API до DB до AI:

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument()
```

---

### 8.3 Business Metrics (Prometheus)

Додати `prometheus-fastapi-instrumentator` + кастомні метрики:

```python
from prometheus_client import Counter, Histogram, Gauge

LEADS_CREATED = Counter("crm_leads_created_total", "Total leads created", ["source"])
LEADS_BY_STAGE = Gauge("crm_leads_by_stage", "Leads count per stage", ["stage"])
AI_LATENCY = Histogram("crm_ai_analysis_duration_seconds", "AI analysis latency")
TRANSFER_RATE = Counter("crm_transfers_total", "Lead to sales transfers", ["result"])
```

Dashboard у Grafana з алертами: conversion rate < X%, AI error rate > Y%.

---

### 8.4 Health Check — розширений

**Поточний стан**: `/health` повертає `{"status": "ok"}`.

**Розширення** — компонентний health check:

```json
{
  "status": "healthy",
  "components": {
    "database": {"status": "healthy", "latency_ms": 3},
    "redis": {"status": "healthy", "latency_ms": 1},
    "openai": {"status": "healthy", "latency_ms": 342},
    "celery": {"status": "healthy", "active_workers": 2}
  },
  "uptime_seconds": 86400,
  "version": "1.2.0"
}
```

---

## 9. Інфраструктура та DevOps

### 9.1 Environment Configuration

**Поточний стан**: Всі налаштування в одному `.env`.

**Рекомендація** — Staged environments:

```
.env.development  — SQLite, debug=true, mock AI
.env.staging      — PostgreSQL, rate limits lower
.env.production   — PostgreSQL, Redis Sentinel, strict security
```

---

### 9.2 Docker — Production Hardening

Поточний `Dockerfile` (базовий). Production-ready варіант:

```dockerfile
FROM python:3.12-slim AS builder
# ... встановлення залежностей

FROM python:3.12-slim AS runtime
RUN groupadd -r crm && useradd -r -g crm crm  # Non-root user
USER crm

# Обмеження ресурсів у docker-compose:
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 512M
```

---

### 9.3 Alembic Migration Strategy

- [ ] Заборонити `alembic downgrade` в production CI/CD
- [ ] Додати pre-migration backup hook
- [ ] Ввести naming convention для міграцій: `{YYYYMMDD}_{hhmm}_{description}`
- [ ] Додати `include_schemas=True` для multi-schema підтримки

---

### 9.4 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
jobs:
  lint:    ruff check . && mypy app/
  test:    pytest tests/ --cov=app --cov-report=xml
  security: bandit -r app/ && safety check
  build:   docker build --target runtime .
  migrate: alembic upgrade head (staging only)
  deploy:  (production manual gate)
```

---

### 9.5 Secrets Management

Замінити `.env` файл на:
- **Development**: `.env` (OK)
- **Staging/Production**: HashiCorp Vault або AWS Secrets Manager

```python
# app/core/config.py:
from pydantic_settings import BaseSettings, SecretsSettingsSource

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets",  # Docker secrets mount
        env_file=".env",
    )
```

---

## 10. Тестування

### 10.1 Тестова стратегія (піраміда)

```
         E2E Tests (5%)
        /               \
    Integration (25%)
   /                     \
Unit Tests (70%)
```

**Поточний стан**: `test_business_logic.py`, `verify_logic.py`, `deep_test_crm.py` — але вони, здається, не є pytest-based suite.

---

### 10.2 Unit Tests — critical path

```python
# tests/unit/test_lead_service.py
class TestStageTransition:
    @pytest.mark.asyncio
    async def test_sequential_transition_ok(self):
        """NEW → CONTACTED should succeed"""
        
    async def test_skip_stage_raises(self):
        """NEW → QUALIFIED should raise LeadStageError"""
        
    async def test_terminal_stage_locked(self):
        """TRANSFERRED → anything should raise LeadStageError"""
        
    async def test_lost_allowed_from_any_stage(self):
        """Any stage → LOST should succeed"""

class TestTransfer:
    async def test_transfer_requires_ai_score(self): ...
    async def test_transfer_requires_domain(self): ...
    async def test_transfer_creates_sale(self): ...
    async def test_transfer_marks_lead_transferred(self): ...

class TestAIValidation:
    def test_all_lead_sources_are_valid_ai_input(self):
        """Every LeadSource enum value must pass _validate_lead_features"""
        for source in LeadSource:
            features = {
                "source": source.value,
                "stage": "NEW",
                "message_count": 0,
                "days_since_created": 1,
            }
            # Should NOT raise:
            _validate_lead_features(features)
```

---

### 10.3 Integration Tests — API

```python
# tests/integration/test_leads_api.py
class TestLeadLifecycle:
    async def test_full_pipeline_happy_path(self, client, mock_openai):
        """
        POST /leads → PATCH /stage (contacted) → PATCH /stage (qualified)
        → POST /analyze → POST /transfer → PATCH /sales/{id}/stage
        """
        
    async def test_parallel_stage_transitions_are_safe(self, client):
        """Два одночасних запити на зміну стадії — повинен пройти тільки один"""
```

---

### 10.4 AI Mock Strategy

```python
@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("app.ai.ai_service.openai_client.chat.completions.create",
        return_value=MockResponse(json_content={
            "score": 0.75,
            "recommendation": "transfer_to_sales",
            "reason": "Mock: high quality lead"
        })
    )

@pytest.fixture
def mock_openai_error(mocker):
    mocker.patch(
        "app.ai.ai_service.openai_client.chat.completions.create",
        side_effect=openai.APIConnectionError("Mocked connection error")
    )
```

---

### 10.5 Property-Based Testing

Для stage machine — `hypothesis`:

```python
from hypothesis import given, strategies as st

@given(
    current=st.sampled_from(list(ColdStage)),
    target=st.sampled_from(list(ColdStage))
)
def test_stage_transition_invariants(current, target):
    """
    Invariant: якщо current in TERMINAL, transition завжди raises.
    Invariant: якщо target == LOST, завжди дозволено.
    Invariant: якщо target не next і не LOST, завжди raises.
    """
```

---

## 11. Telegram Bot — покращення UX

### 11.1 Keyboard Navigation Flow

**Поточний стан**: Бот має базові команди `/leads`, `/sales`, `/stats`.

**Покращення**:
- Додати `Back` кнопку на кожен екран (FSM state stack)
- Breadcrumbs у заголовках: `🏠 > 📋 Leads > #42 > 🤖 AI Results`
- Підтвердження перед незворотніми діями: `"Ви впевнені, що хочете передати в продажі?"` → `✅ Так` / `❌ Скасувати`

---

### 11.2 Швидкі дії (Quick Actions)

При перегляді ліда — Inline кнопки:
```
[ 📞 Contacted ] [ 🤖 AI Analyze ] [ 📝 Add Note ]
[ ➡️ Qualified ] [ ❌ Mark Lost  ] [ 👤 Reassign ]
```

---

### 11.3 Smart Notifications

Celery + Bot сповіщення:
- `"⚠️ Lead #42 без активності вже 7 днів"`
- `"🔥 AI score ліда #15 виріс до 0.82 — час передати в продажі!"`
- `"🎉 Продаж #8 перейшов у PAID!"`

---

### 11.4 Telegram Bot → Mini App

Для більш складного UI використовувати Telegram Web App (Mini App):
- Повноцінна форма створення ліда з валідацією в реальному часі
- Drag-and-drop Kanban board лідів
- Графіки AI score trends

---

## 12. Dashboard — React

### 12.1 Kanban Board

Перетягувати картки лідів між колонками → автоматичний PATCH stage endpoint.

```
[NEW] → [CONTACTED] → [QUALIFIED] → [TRANSFERRED]
  ↓           ↓            ↓
[LOST]     [LOST]       [LOST]
```

Обмеження: WIP (Work in Progress) ліміти на колонках.

---

### 12.2 AI Score Heatmap

Таблиця лідів з кольоровим кодуванням по AI score:
- 🔴 < 0.3 (Dead)
- 🟡 0.3–0.6 (Cold)
- 🟢 0.6–0.8 (Warm)
- 🔥 > 0.8 (Hot)

---

### 12.3 Conversion Funnel Drill-Down

Клік на воронці → відкривається список конкретних лідів на цьому етапі, з фільтрацією та сортуванням.

---

### 12.4 Manager Performance Dashboard

```
  Manager    | Assigned | Converted | Lost | Avg Score | Avg Time-to-Close
  ---------  | -------- | --------- | ---- | --------- | -----------------
  Nikolas    |    23    |    12     |   4  |   0.71    |   8.3 days
  Alex       |    18    |     9     |   6  |   0.64    |   11.2 days
```

---

### 12.5 Real-time Dashboard via WebSocket

**Поточний стан**: WebSocket реалізований (`ws.py`, `manager.broadcast`).

**Покращення**:
- Розбити на топіки: `ws://host/ws/dashboard`, `ws://host/ws/leads/{id}`
- Додати reconnection logic на фронті (exponential backoff)
- Heartbeat ping/pong кожні 30 секунд

---

## 13. Production-roadmap

### Phase 1 — Stability (Тиждень 1–2)
Виправити всі виявлені bugs з розділу 1 та 2. Покрити unit-тестами весь critical path.

### Phase 2 — Observability (Тиждень 3)
Structured logging, Prometheus metrics, extended health check.

### Phase 3 — Security (Тиждень 4)
JWT authentication, per-user rate limiting, input sanitization, file security.

### Phase 4 — AI Enhancement (Тиждень 5–6)
Structured outputs, fallback scoring, AI audit log, cost tracking.

### Phase 5 — Scale (Тиждень 7–8)
Cursor pagination, bulk APIs, DB indexes, soft delete, connection pooling.

### Phase 6 — Product (Тиждень 9+)
Kanban board, mini app, manager analytics, lead score trends, RLHF feedback loop.

---

## Підсумок

| Категорія | Кількість елементів | Пріоритет |
|-----------|---------------------|-----------|
| Критичні баги | 7 | 🔴 Критичний |
| Бізнес-логіка | 7 розширень | 🟠 Високий |
| AI-шар | 6 покращень | 🟠 Високий |
| API/DX | 6 покращень | 🟡 Середній |
| База даних | 5 покращень | 🟡 Середній |
| Безпека | 5 покращень | 🟠 Високий |
| Спостережуваність | 4 покращення | 🟡 Середній |
| Інфраструктура | 5 покращень | 🟢 Низький |
| Тестування | 5 блоків | 🟠 Високий |
| Bot UX | 4 покращення | 🟢 Низький |
| Dashboard | 5 покращень | 🟢 Низький |

---

*Документ підготовлено на основі аудиту кодової бази AEL CRM v1.0 — 2026-02-22.*
