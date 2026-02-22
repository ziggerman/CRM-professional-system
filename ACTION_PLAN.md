# AEL CRM — Покроковий план дій з AI-промптами

> Кожен крок містить: **пріоритет**, **файли**, **умову завершення** та **готовий промпт** для AI-асистента.
> Виконуй кроки у порядку фаз. Не переходь до наступної фази без завершення попередньої.

---

## ФАЗА 1 — STABILITY (Тижні 1–2) 🔴

### Крок 1.1 — Синхронізація констант AI-валідатора

**Пріоритет**: 🔴 Критичний | **Файли**: `app/ai/prompts.py`, `app/models/lead.py`
**Готово коли**: `_validate_lead_features` не кидає помилку для жодного значення `LeadSource` або `ColdStage`

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python Engineer. В проєкті FastAPI CRM є баг: константи валідатора AI не збігаються з enum-моделями.
>
> ФАЙЛ 1 — app/models/lead.py:
> LeadSource enum: WEB, REFERRAL, SOCIAL, MANUAL, SCANNER, PARTNER, REGISTRATION, CALLBACK, LEAD_MAGNET, MESSAGE
> ColdStage enum: NEW, CONTACTED, QUALIFIED, TRANSFERRED, LOST
>
> ФАЙЛ 2 — app/ai/prompts.py (ПОТОЧНИЙ — НЕВІРНИЙ):
> VALID_LEAD_SOURCES = frozenset({'WEB', 'REFERRAL', 'SOCIAL', 'MANUAL'})
> VALID_LEAD_STAGES = frozenset({'NEW', 'CONTACTED', 'QUALIFIED', 'NEGOTIATION', 'CLOSED'})
>
> ЗАВДАННЯ:
> 1. Замінити VALID_LEAD_SOURCES та VALID_LEAD_STAGES в prompts.py щоб вони точно відображали всі значення з enum
> 2. Імпортувати LeadSource та ColdStage безпосередньо і будувати frozenset динамічно: frozenset({e.value for e in LeadSource})
> 3. Написати pytest-тест tests/unit/test_ai_prompts.py з класом TestAIValidation що перевіряє кожен LeadSource проходить _validate_lead_features без виключення
>
> Поверни повні файли з усіма змінами.
> ```

---

### Крок 1.2 — Async файловий I/O у upload endpoint

**Пріоритет**: 🔴 Критичний | **Файли**: `app/api/v1/leads.py`, `requirements.txt`
**Готово коли**: upload endpoint не блокує event loop; aiofiles в requirements

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python/FastAPI Engineer. В async FastAPI endpoint є синхронний блокуючий файловий I/O.
>
> ПОТОЧНИЙ КОД (app/api/v1/leads.py, ~рядок 495–510):
> with open(file_path, "wb") as buffer:
>     shutil.copyfileobj(file.file, buffer)
> safe_name = f"{lead_id}_{int(datetime.now().timestamp())}_{file.filename}"
>
> ЗАВДАННЯ:
> 1. Замінити синхронний запис на aiofiles: async with aiofiles.open(...) as f: await f.write(content)
> 2. Виправити datetime.now() на datetime.now(UTC)
> 3. Додати валідацію: MAX_FILE_SIZE = 10MB, якщо перевищено — HTTPException 413
> 4. Додати ALLOWED_MIME_TYPES = {"image/jpeg","image/png","image/gif","application/pdf","text/plain"} і перевірку file.content_type
> 5. Додати aiofiles до requirements.txt
>
> Зберігай async-природу всього endpoint. Поверни повний оновлений блок upload_attachment.
> ```

---

### Крок 1.3 — Виправлення imports та антипаттернів

**Пріоритет**: 🔴 Критичний | **Файли**: `app/api/v1/leads.py`, `app/services/lead_service.py`
**Готово коли**: жодного дублікату імпорту, LeadHistory імпортується на рівні модуля

---

> **ПРОМПТ**
>
> ```
> Ти Python code reviewer. Виправ два антипаттерни у CRM-проєкті:
>
> ПРОБЛЕМА 1 — app/api/v1/leads.py:
> рядок 12: from app.core.deps import get_lead_service, get_transfer_service
> рядок 27: from app.core.deps import get_lead_service, get_transfer_service, get_history_repo
> → Перший імпорт є зайвим. Залиш тільки другий (повний).
>
> ПРОБЛЕМА 2 — app/services/lead_service.py, методи transition_stage та nurture_lead:
> всередині тіла методів: from app.models.history import LeadHistory
> → Це виконується при кожному виклику. Перенеси імпорт на рівень модуля (top of file).
>
> ЗАВДАННЯ: Поверни diff-патч для обох файлів. Більше нічого не змінюй.
> ```

---

### Крок 1.4 — Виправлення race condition у current_leads

**Пріоритет**: 🔴 Критичний | **Файли**: `app/api/v1/leads.py`
**Готово коли**: assign/unassign/reassign не мають ручного `+= 1`

---

> **ПРОМПТ**
>
> ```
> Ти Senior Backend Engineer. В FastAPI CRM є race condition: поле user.current_leads оновлюється вручну (+= 1, -= 1) в трьох endpoints: assign_lead, unassign_lead, reassign_lead.
>
> ЗАВДАННЯ: Замінити ручний лічильник на реальний COUNT запит до бази даних.
> Після кожної зміни assigned_to_id виконувати:
>
> from sqlalchemy import func, select as sa_select
> count_result = await session.execute(
>     sa_select(func.count()).where(Lead.assigned_to_id == user_id, Lead.stage.notin_([ColdStage.LOST, ColdStage.TRANSFERRED]))
> )
> user.current_leads = count_result.scalar()
>
> Застосуй цей підхід для всіх трьох endpoints. Перевір що capacity check використовує fresh count перед призначенням. Поверни оновлені endpoint-функції.
> ```

---

### Крок 1.5 — Notes pagination на рівні SQL

**Пріоритет**: 🟠 Високий | **Файли**: `app/api/v1/leads.py`
**Готово коли**: list_lead_notes не завантажує всі нотатки в пам'ять

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI/SQLAlchemy Engineer. Endpoint list_lead_notes завантажує ВСІ нотатки через lead.notes (SQLAlchemy relationship), а потім робить Python slice. При 10 000 нотаток — це OOM.
>
> ПОТОЧНИЙ КОД:
> notes = lead.notes
> total = len(notes)
> start = (page - 1) * page_size
> page_notes = notes[start:end]
>
> ЗАВДАННЯ: Замінити на SQL-рівневу пагінацію:
> - SELECT COUNT(*) WHERE lead_id = :id
> - SELECT * WHERE lead_id = :id ORDER BY created_at DESC LIMIT :limit OFFSET :offset
>
> Використовуй async session із sqlalchemy.select. Прибери завантаження lead.notes повністю з цього endpoint. Поверни оновлену функцію list_lead_notes.
> ```

---

### Крок 1.6 — Transfer: додати перевірку стадії QUALIFIED

**Пріоритет**: 🟠 Високий | **Файли**: `app/services/transfer_service.py`
**Готово коли**: transfer_to_sales кидає TransferError якщо stage != QUALIFIED

---

> **ПРОМПТ**
>
> ```
> Ти Senior Backend Engineer. В CRM-системі TransferService.transfer_to_sales перевіряє ai_score >= 0.6 та business_domain, але не перевіряє стадію ліда.
>
> ЗАВДАННЯ: Додати третю перевірку перед існуючими:
> if lead.stage not in (ColdStage.QUALIFIED,):
>     raise TransferError(f"Lead must be in QUALIFIED stage before transfer. Current: {lead.stage.value}")
>
> Також: написати два pytest-тести:
> - test_transfer_from_contacted_raises: ліду зі stage=CONTACTED відмовляє
> - test_transfer_from_qualified_succeeds: ліду зі stage=QUALIFIED і score=0.7 і domain — дозволяє
>
> Поверни оновлений transfer_service.py та тести.
> ```

---

### Крок 1.7 — Дублікат-контроль лідів при створенні

**Пріоритет**: 🟠 Високий | **Файли**: `app/services/lead_service.py`, `app/api/v1/leads.py`
**Готово коли**: POST /leads повертає 409 якщо email або phone вже існує

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI Engineer. При створенні ліда в CRM немає перевірки на дублікати. Два менеджери можуть створити одного й того ж клієнта.
>
> ЗАВДАННЯ:
> 1. В LeadCreate schema email і phone є опціональними — залиш так
> 2. В LeadService.create_lead перед збереженням: якщо email або phone вказані — шукати існуючий лід:
>    SELECT id FROM leads WHERE (email = :email OR phone = :phone) AND is_deleted = FALSE LIMIT 1
> 3. Якщо знайдено — raise DuplicateLeadError(existing_id=found_id)
> 4. DuplicateLeadError — новий exception клас в lead_service.py
> 5. В API endpoint create_lead — перехоплювати DuplicateLeadError і повертати:
>    HTTPException(409, detail={"message": "Duplicate lead", "existing_lead_id": e.existing_id})
>
> Поверни оновлені файли.
> ```

---

## ФАЗА 2 — OBSERVABILITY (Тиждень 3) 🟡

### Крок 2.1 — Structured JSON Logging

**Пріоритет**: 🟡 Середній | **Файли**: `main.py`, `app/services/lead_service.py`
**Готово коли**: логи виводяться у JSON форматі зі структурованими полями

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python Engineer. CRM-система використовує стандартний logging у plain text. Потрібно перейти на structlog з JSON-форматом.
>
> ЗАВДАННЯ:
> 1. Додати structlog до requirements.txt
> 2. В main.py налаштувати structlog: structlog.configure(processors=[structlog.processors.JSONRenderer()])
> 3. В lead_service.py замінити print/logging на structlog:
>    log = structlog.get_logger()
>    log.info("lead_stage_transition", lead_id=lead.id, old_stage=old.value, new_stage=new.value, changed_by=changed_by)
>    log.info("lead_created", lead_id=lead.id, source=lead.source.value)
>    log.warning("ai_analysis_cached", lead_id=lead.id, analyzed_at=str(lead.ai_analyzed_at))
> 4. В ai_service.py логувати: log.info("ai_analysis_complete", lead_id=..., score=..., latency_ms=...)
>
> Забезпечити щоб у DEBUG режимі була human-readable консоль, у PRODUCTION — JSON.
> ```

---

### Крок 2.2 — Розширений Health Check

**Пріоритет**: 🟡 Середній | **Файли**: `app/api/health.py`
**Готово коли**: /health повертає статус кожного компонента з latency_ms

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI Engineer. Поточний /health endpoint повертає {"status": "ok"}. Потрібен компонентний health check.
>
> ЗАВДАННЯ: Переписати app/api/health.py щоб перевіряв кожен компонент паралельно (asyncio.gather):
>
> Компоненти:
> - database: SELECT 1 через asyncpg, вимір latency_ms
> - redis: await redis.ping(), вимір latency_ms
> - openai: перевіряти OPENAI_API_KEY не порожній (не робити реальний запит — дорого)
> - celery: перевіряти через celery.control.inspect().active_queues()
>
> Відповідь формату:
> {"status": "healthy"|"degraded"|"unhealthy", "components": {...}, "uptime_seconds": N, "version": "1.0.0"}
>
> Якщо хоч один компонент unhealthy — загальний статус degraded. HTTP 200 завжди (клієнт сам дивиться на статус).
> ```

---

### Крок 2.3 — Prometheus Metrics

**Пріоритет**: 🟡 Середній | **Файли**: `main.py`, нові файли `app/core/metrics.py`
**Готово коли**: /metrics endpoint повертає prometheus-сумісний формат

---

> **ПРОМПТ**
>
> ```
> Ти Senior Backend Engineer. Додай Prometheus метрики до FastAPI CRM.
>
> ЗАВДАННЯ:
> 1. Додати до requirements.txt: prometheus-fastapi-instrumentator, prometheus-client
> 2. Створити app/core/metrics.py з кастомними метриками:
>    LEADS_CREATED = Counter("crm_leads_created_total", "...", ["source"])
>    LEADS_BY_STAGE = Gauge("crm_leads_by_stage", "...", ["stage"])
>    AI_LATENCY = Histogram("crm_ai_duration_seconds", "...", buckets=[0.1, 0.5, 1, 2, 5, 10])
>    TRANSFERS_TOTAL = Counter("crm_transfers_total", "...", ["result"])  # result: success|failed
> 3. В main.py підключити Instrumentator().instrument(app).expose(app)
> 4. У LeadService.create_lead: LEADS_CREATED.labels(source=lead.source.value).inc()
> 5. У TransferService: TRANSFERS_TOTAL.labels(result="success").inc()
> 6. У AIService: використовувати AI_LATENCY.time() context manager
>
> Поверни всі змінені файли.
> ```

---

## ФАЗА 3 — SECURITY (Тиждень 4) 🟠

### Крок 3.1 — JWT Authentication

**Пріоритет**: 🟠 Високий | **Файли**: `app/core/security.py`, нові: `app/api/v1/auth.py`
**Готово коли**: POST /auth/login повертає JWT; всі endpoints перевіряють Bearer token

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI Security Engineer. CRM використовує один статичний API_SECRET_TOKEN. Потрібно замінити на JWT.
>
> СТЕК: FastAPI + SQLAlchemy async + python-jose або PyJWT + passlib[bcrypt]
>
> ЗАВДАННЯ:
> 1. Додати до requirements.txt: python-jose[cryptography], passlib[bcrypt]
> 2. Оновити User model: додати hashed_password: Mapped[str]
> 3. Створити app/api/v1/auth.py з endpoints:
>    - POST /auth/login: приймає {username, password}, повертає {access_token, token_type, expires_in}
>    - POST /auth/refresh: по refresh_token повертає новий access_token
>    - POST /auth/logout: додає jti в Redis blacklist на час TTL токена
> 4. JWT payload: {sub: user_id, role: "agent"|"manager"|"admin", exp: ..., jti: uuid}
> 5. Оновити app/core/security.py: verify_api_token → verify_jwt_token що декодує Bearer token, перевіряє blacklist в Redis, повертає current_user
> 6. Залишити старий API_SECRET_TOKEN як fallback для зворотної сумісності (з deprecation warning у log)
>
> Поверни всі нові та змінені файли з коментарями.
> ```

---

### Крок 3.2 — Input Sanitization

**Пріоритет**: 🟠 Високий | **Файли**: `app/schemas/lead.py`, нові: `app/core/sanitization.py`
**Готово коли**: всі текстові поля проходять bleach.clean перед збереженням

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python Security Engineer. CRM зберігає текстові поля (intent, pain_points, notes, company) без sanitization — XSS/injection ризик.
>
> ЗАВДАННЯ:
> 1. Додати bleach до requirements.txt
> 2. Створити app/core/sanitization.py:
>    def sanitize_text(value: str | None, max_length: int = 1024) -> str | None:
>        if not value: return value
>        return bleach.clean(value.strip(), tags=[], strip=True)[:max_length]
> 3. Створити Pydantic validator що використовує sanitize_text:
>    @field_validator("intent", "pain_points", "company", "position", mode="before")
>    def sanitize_fields(cls, v): return sanitize_text(v)
> 4. Застосувати validator до LeadCreate та LeadUpdate schemas
> 5. Застосувати до NoteCreate.content з max_length=4096
> 6. Написати тест: переконатись що <script>alert(1)</script> очищується до порожнього рядка
>
> Поверни все оновлені файли.
> ```

---

### Крок 3.3 — Per-User Rate Limiting

**Пріоритет**: 🟠 Високий | **Файли**: `app/api/rate_limit.py`
**Готово коли**: rate limit застосовується до user_id після JWT, а не тільки до IP

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI Engineer. Поточний rate limiter в app/api/rate_limit.py обмежує по IP. Після JWT інтеграції потрібен per-user rate limiting через Redis.
>
> ЛІМІТИ:
> - Lead Creation: 10/hour
> - AI Analysis: 20/day
> - Stage Transitions: 100/hour
> - File Uploads: 5/hour
>
> ЗАВДАННЯ:
> 1. Створити FastAPI dependency: check_rate_limit(action: str, user_id: int, redis: Redis)
> 2. Ключ Redis: f"rate:{action}:{user_id}:{window}" де window = current hour або day
> 3. Використати INCR + EXPIRE атомарно: lua script або pipeline
> 4. При перевищенні: HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(seconds_until_reset)})
> 5. Додати як Depends до: create_lead, analyze_lead, update_stage, upload_attachment
> 6. Зберегти IP-based fallback якщо user_id відсутній (anonymous requests)
>
> Поверни оновлений rate_limit.py та зміни в leads.py.
> ```

---

## ФАЗА 4 — AI ENHANCEMENT (Тижні 5–6) 🟠

### Крок 4.1 — OpenAI Structured Outputs

**Пріоритет**: 🟠 Високий | **Файли**: `app/ai/ai_service.py`, `app/ai/prompts.py`
**Готово коли**: AI відповідь гарантовано в JSON без try/except на парсинг

---

> **ПРОМПТ**
>
> ```
> Ти Senior AI Engineer. CRM використовує OpenAI з text prompt що вимагає JSON відповіді — але якщо GPT відхиляється, парсинг ламається.
>
> ЗАВДАННЯ: Мігрувати на OpenAI Structured Outputs (response_format з json_schema):
>
> 1. В ai_service.py змінити виклик на:
>    response = await client.chat.completions.create(
>        model="gpt-4o-mini",
>        response_format={
>            "type": "json_schema",
>            "json_schema": {
>                "name": "lead_analysis",
>                "schema": {
>                    "type": "object",
>                    "properties": {
>                        "score": {"type": "number", "minimum": 0, "maximum": 1},
>                        "recommendation": {"type": "string", "enum": ["transfer_to_sales", "continue_nurturing", "lost"]},
>                        "reason": {"type": "string", "minLength": 10}
>                    },
>                    "required": ["score", "recommendation", "reason"],
>                    "additionalProperties": False
>                },
>                "strict": True
>            }
>        },
>        messages=[...]
>    )
> 2. Прибрати необхідність ручного json.loads — використовувати response.choices[0].message.parsed
> 3. Оновити LEAD_ANALYSIS_SYSTEM_PROMPT: прибрати інструкції про JSON формат (вони тепер в schema)
> 4. Зберегти існуючий parse_lead_analysis_response як fallback для старих кешованих відповідей
>
> Поверни оновлений ai_service.py.
> ```

---

### Крок 4.2 — Rule-Based Fallback Scoring

**Пріоритет**: 🟠 Високий | **Файли**: `app/ai/ai_service.py`, новий: `app/ai/fallback_scorer.py`
**Готово коли**: при OpenAI недоступності аналіз повертає rule-based результат з міткою [AI OFFLINE]

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python Engineer. CRM AI-аналіз падає з AIServiceError коли OpenAI недоступний, замість того щоб використовувати fallback.
>
> ЗАВДАННЯ: Створити app/ai/fallback_scorer.py з функцією rule_based_score(lead: Lead) -> AIAnalysisResult:
>
> Вагові коефіцієнти:
> - Source: REFERRAL=0.35, PARTNER=0.30, WEB=0.25, SCANNER=0.20, SOCIAL=0.15, MANUAL=0.10
> - message_count >= 10: +0.25; >= 5: +0.15; >= 2: +0.08; else: +0.0
> - business_domain is not None: +0.25
> - Stage QUALIFIED: +0.20; CONTACTED: +0.10; NEW: +0.0
> - Нормалізувати score до [0.0, 1.0] через min(score, 1.0)
> - Recommendation: >= 0.6 → transfer_to_sales; >= 0.3 → continue_nurturing; else → lost
> - reason повинен містити "[AI OFFLINE - RULE BASED]"
>
> В ai_service.py: у блоці except (APIConnectionError, RateLimitError, openai.APIStatusError):
>   log.warning("openai_unavailable_using_fallback", ...)
>   return rule_based_score(lead)
>
> Поверни обидва файли.
> ```

---

### Крок 4.3 — AI Audit Log

**Пріоритет**: 🟠 Високий | **Файли**: нові: `app/models/ai_audit.py`, міграція
**Готово коли**: кожен AI виклик зберігається в ai_audit_log з latency та token count

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI/SQLAlchemy Engineer. Потрібно логувати кожен виклик AI в БД для cost tracking та аудиту.
>
> ЗАВДАННЯ:
> 1. Створити app/models/ai_audit.py — SQLAlchemy модель AIAuditLog:
>    - id, lead_id (FK nullable), model (str), input_hash (SHA-256 str),
>      score (float), recommendation (str), reason (text),
>      latency_ms (int), tokens_used (int), cost_usd (float nullable),
>      is_fallback (bool default False), created_at (timestamptz)
> 2. Додати до app/models/__init__.py
> 3. Згенерувати Alembic міграцію: alembic revision --autogenerate -m "add_ai_audit_log"
> 4. В ai_service.py після кожного AI виклику:
>    - Рахувати latency: time.perf_counter()
>    - input_hash = hashlib.sha256(prompt.encode()).hexdigest()
>    - tokens_used = response.usage.total_tokens
>    - cost_usd = tokens_used * 0.00000015  # gpt-4o-mini pricing
>    - Зберігати запис асинхронно (не блокувати основний flow)
> 5. Додати endpoint GET /api/v1/admin/ai-stats що повертає: total_calls, total_cost_usd, avg_latency_ms, fallback_rate за останні 30 днів
>
> Поверни всі нові файли.
> ```

---

### Крок 4.4 — Розширені AI Feature Engineering + Rate Limiting

**Пріоритет**: 🟡 Середній | **Файли**: `app/ai/prompts.py`, `app/ai/ai_service.py`
**Готово коли**: AI отримує 8 фіч замість 5; є Redis денний ліміт

---

> **ПРОМПТ**
>
> ```
> Ти Senior AI/ML Engineer. Розшир фічі що подаються в AI при аналізі ліда, та додай денний rate limit.
>
> НОВІ ФІЧІ ДЛЯ AI (app/ai/prompts.py, функція build_lead_analysis_prompt):
> 1. message_velocity = message_count / max(days_since_created, 1)  # messages per day
> 2. contact_completeness = bool(lead.phone) and bool(lead.email)  # True/False
> 3. b2b_qualification = bool(lead.company) and bool(lead.position)  # True/False
> 4. days_since_last_activity = обраховується з last_activity_at якщо є, інакше days_since_created
> 5. Оновити LeadFeatures TypedDict щоб включав нові поля
> 6. Оновити LEAD_ANALYSIS_SYSTEM_PROMPT — описати нові фічі у Key Scoring Factors
>
> RATE LIMITING (app/ai/ai_service.py):
> async def check_daily_ai_limit(redis_client) -> None:
>     key = f"ai:daily:{date.today()}"
>     count = await redis_client.incr(key)
>     if count == 1: await redis_client.expire(key, 86400)
>     if count > settings.AI_DAILY_LIMIT:  # default 500
>         raise AIServiceError("Daily AI analysis limit reached")
>
> Поверни оновлені файли.
> ```

---

## ФАЗА 5 — SCALE (Тижні 7–8) 🟡

### Крок 5.1 — Database Indexes

**Пріоритет**: 🟡 Середній | **Файли**: нова Alembic міграція
**Готово коли**: всі 7 індексів додані через міграцію

---

> **ПРОМПТ**
>
> ```
> Ти Senior Database Engineer. Додай оптимізаційні індекси до PostgreSQL таблиць CRM через Alembic.
>
> ЗАВДАННЯ: Створити Alembic міграцію що додає ці індекси (використовуй op.create_index):
>
> 1. idx_leads_ai_score: ON leads(ai_score) WHERE ai_score IS NOT NULL  — partial index
> 2. idx_leads_stage_created: ON leads(stage, created_at DESC)  — composite
> 3. idx_leads_assigned_stage: ON leads(assigned_to_id, stage)  — composite
> 4. idx_leads_source_domain: ON leads(source, business_domain)  — composite
> 5. idx_sales_stage_lead: ON sales(stage, lead_id)
> 6. idx_history_lead_created: ON lead_history(lead_id, created_at DESC)
> 7. idx_notes_lead_created: ON lead_notes(lead_id, created_at DESC)
>
> Також: покажи EXPLAIN ANALYZE для типового запиту "всі ліди менеджера у stage=QUALIFIED відсортовані по created_at" до і після індексу.
>
> Поверни файл міграції та SQL команди для верифікації.
> ```

---

### Крок 5.2 — Soft Delete

**Пріоритет**: 🟡 Середній | **Файли**: `app/models/lead.py`, `app/repositories/lead_repo.py`, міграція
**Готово коли**: DELETE не видаляє рядок; GET автоматично фільтрує is_deleted=False

---

> **ПРОМПТ**
>
> ```
> Ти Senior SQLAlchemy Engineer. Заміни hard delete на soft delete для Lead моделі.
>
> ЗАВДАННЯ:
> 1. Додати до Lead model в app/models/lead.py:
>    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
>    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
>    deleted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
> 2. Alembic міграція з ALTER TABLE leads ADD COLUMN ...
> 3. В LeadRepository: всі SELECT запити автоматично додають WHERE is_deleted = FALSE
>    Використати SQLAlchemy event: @event.listens_for(Session, "do_orm_execute") або filterext у BaseQuery
> 4. LeadRepository.delete(lead) → встановлює is_deleted=True, deleted_at=now(), зберігає
> 5. Новий endpoint: POST /api/v1/leads/{id}/restore (Admin only) → встановлює is_deleted=False
> 6. Новий endpoint: GET /api/v1/admin/leads/deleted (Admin only) → список видалених лідів
>
> Поверни всі змінені файли.
> ```

---

### Крок 5.3 — Cursor-Based Pagination

**Пріоритет**: 🟡 Середній | **Файли**: `app/api/v1/leads.py`, `app/repositories/lead_repo.py`
**Готово коли**: GET /leads підтримує ?cursor= параметр

---

> **ПРОМПТ**
>
> ```
> Ти Senior Backend Engineer. CRM використовує offset pagination — при великих таблицях це повний scan. Додай cursor-based пагінацію.
>
> ПІДХІД: keyset pagination на основі id (стабільний, не залежить від сортування):
> - cursor — base64-encoded JSON: {"id": last_seen_id}
> - Запит: WHERE id > :last_id ORDER BY id ASC LIMIT :limit
>
> ЗАВДАННЯ:
> 1. В app/schemas/lead.py додати CursorPage модель з полями: items, next_cursor, has_next, limit
> 2. В LeadRepository додати метод get_page_by_cursor(cursor_id: int | None, limit: int, filters: dict)
> 3. В GET /api/v1/leads додати опціональний query param cursor: str | None
>    - якщо cursor переданий — декодувати і використовувати keyset
>    - якщо ні — стандартна page/page_size поведінка (зворотна сумісність)
> 4. next_cursor: base64.b64encode(json.dumps({"id": items[-1].id}).encode()).decode()
> 5. Документувати в docstring: "cursor pagination is preferred for large datasets"
>
> Поверни оновлені файли.
> ```

---

### Крок 5.4 — Bulk Operations API

**Пріоритет**: 🟡 Середній | **Файли**: новий `app/api/v1/bulk.py`
**Готово коли**: 4 bulk endpoints працюють і обробляють до 100 items за раз

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI Engineer. Додай bulk operations API для масових дій над лідами.
>
> ЗАВДАННЯ: Створити app/api/v1/bulk.py з 4 endpoints:
>
> 1. POST /api/v1/leads/bulk/stage
>    Body: {lead_ids: list[int], stage: ColdStage}
>    Logic: для кожного ID — transition_stage якщо валідно, збирати results: {id, status: "ok"|"error", message}
>    Max 100 IDs за раз, інакше 422
>
> 2. POST /api/v1/leads/bulk/assign
>    Body: {lead_ids: list[int], user_id: int}
>    Logic: перевірити user exists, capacity; призначити, повернути results
>
> 3. POST /api/v1/leads/bulk/analyze
>    Body: {lead_ids: list[int]}
>    Logic: запустити Celery task analyze_leads_batch(lead_ids) асинхронно
>    Повернути {task_id: str, status: "queued"}
>
> 4. DELETE /api/v1/leads/bulk (Admin only)
>    Body: {lead_ids: list[int], confirm: bool}
>    Logic: якщо confirm != True → 400; soft delete all
>
> Всі endpoints повертають {processed: N, succeeded: N, failed: N, results: [...]}
>
> Поверни bulk.py та зміни в main.py (підключення router).
> ```

---

## ФАЗА 6 — BUSINESS LOGIC (Тиждень 9) 🟠

### Крок 6.1 — Lead Score History Table

**Пріоритет**: 🟠 Високий | **Файли**: нові моделі та міграція
**Готово коли**: кожен AI аналіз зберігає score в окремій таблиці; є endpoint для тренду

---

> **ПРОМПТ**
>
> ```
> Ти Senior SQLAlchemy Engineer. Зараз AI score перезаписується в leads.ai_score при кожному аналізі — немає history. Потрібна окрема таблиця.
>
> ЗАВДАННЯ:
> 1. Створити app/models/score_history.py — модель LeadScoreHistory:
>    id, lead_id (FK → leads.id ON DELETE CASCADE), score (Float),
>    recommendation (String 64), reason (Text), analyzed_by (String: "openai"|"fallback"),
>    analyzed_at (DateTime tz-aware)
> 2. Relationship: Lead.score_history → list[LeadScoreHistory]
> 3. Alembic міграція
> 4. В LeadService.save_ai_analysis: після save_lead — додатково INSERT в lead_score_history
> 5. Новий endpoint: GET /api/v1/leads/{id}/score-history
>    Повертає список {score, recommendation, analyzed_by, analyzed_at} відсортований по analyzed_at
> 6. В GET /api/v1/leads/{id}: додати поле score_trend: "rising"|"falling"|"stable"|null
>    (порівнює останні 2 записи з score_history)
>
> Поверни всі нові та змінені файли.
> ```

---

### Крок 6.2 — Sales Pipeline — нові бізнес-правила

**Пріоритет**: 🟠 Високий | **Файли**: `app/models/sale.py`, `app/services/transfer_service.py`, міграція
**Готово коли**: AGREEMENT вимагає amount; PAID встановлює closed_at; LOST вимагає reason

---

> **ПРОМПТ**
>
> ```
> Ти Senior Backend Engineer. Потрібно посилити бізнес-правила Sales pipeline в CRM.
>
> ПОТОЧНА МОДЕЛЬ Sale має: id, lead_id, stage, amount (nullable), created_at
>
> ЗАВДАННЯ:
> 1. Додати до Sale model: lost_reason (String 512 nullable), closed_at (DateTime tz nullable), duration_days (Integer nullable)
> 2. Alembic міграція для нових полів
> 3. В TransferService або SaleService.transition_sale_stage (або аналог) додати:
>    - При переході в AGREEMENT: if not sale.amount: raise SaleStageError("Amount required before AGREEMENT stage")
>    - При переході в PAID: sale.closed_at = datetime.now(UTC); sale.duration_days = (sale.closed_at - sale.created_at).days
>    - При переході в LOST: if not data.lost_reason: raise SaleStageError("lost_reason is required")
>                           sale.lost_reason = data.lost_reason
> 4. Оновити PATCH /api/v1/sales/{id}/stage schema: додати опціональне поле lost_reason
> 5. Написати 3 тести для нових правил
>
> Поверни всі змінені файли.
> ```

---

### Крок 6.3 — Stage Rollback (повернення ліда на крок назад)

**Пріоритет**: 🟡 Середній | **Файли**: `app/models/lead.py`, `app/services/lead_service.py`, `app/api/v1/leads.py`
**Готово коли**: POST /leads/{id}/stage-rollback з обов'язковим reason працює

---

> **ПРОМПТ**
>
> ```
> Ти Senior FastAPI Engineer. В CRM stage machine лід може тільки рухатись вперед або в LOST. Потрібно додати можливість повернення на один крок назад.
>
> БІЗНЕС-ЛОГІКА:
> REVERSIBLE_STAGE_TRANSITIONS = {
>     ColdStage.CONTACTED: ColdStage.NEW,      # втрата контакту
>     ColdStage.QUALIFIED: ColdStage.CONTACTED,  # втрата кваліфікації
> }
> NEW та TRANSFERRED та LOST — не можуть rollback
>
> ЗАВДАННЯ:
> 1. Додати REVERSIBLE_STAGE_TRANSITIONS константу в app/models/lead.py
> 2. В LeadService додати метод rollback_stage(lead, reason: str, changed_by: str):
>    - Перевіряє що stage є в REVERSIBLE_STAGE_TRANSITIONS
>    - Логує в LeadHistory з reason обов'язково
>    - Повертає оновлений ліда
> 3. Новий endpoint: POST /api/v1/leads/{id}/stage-rollback
>    Body: {reason: str (мінімум 10 символів)}
>    Response: LeadResponse
>    Auth: тільки Manager або Admin
> 4. Написати тести: QUALIFIED→CONTACTED OK; CONTACTED→NEW OK; NEW→rollback = error; TRANSFERRED→rollback = error
>
> Поверни всі нові та змінені файли.
> ```

---

### Крок 6.4 — quality_tier поле на ліді

**Пріоритет**: 🟡 Середній | **Файли**: `app/models/lead.py`, `app/services/lead_service.py`, міграція
**Готово коли**: quality_tier автоматично оновлюється при збереженні AI score

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python Engineer. Додай денормалізоване поле quality_tier до Lead для швидкої фільтрації без float порівнянь.
>
> ТИРИ (визначаються по ai_score):
> HOT: score >= 0.8
> WARM: 0.6 <= score < 0.8
> COLD: 0.3 <= score < 0.6
> DEAD: score < 0.3
> None: якщо ai_score ще не визначений
>
> ЗАВДАННЯ:
> 1. Додати QualityTier(str, Enum): HOT, WARM, COLD, DEAD — до app/models/lead.py
> 2. Додати до Lead: quality_tier: Mapped[QualityTier | None] = mapped_column(SAEnum(QualityTier), nullable=True)
> 3. Alembic міграція
> 4. Додати статичний метод: Lead.score_to_tier(score: float) -> QualityTier
> 5. В LeadService.save_ai_analysis: lead.quality_tier = Lead.score_to_tier(result.score)
> 6. Додати quality_tier як фільтр в GET /api/v1/leads: ?quality_tier=HOT,WARM
> 7. LeadRepository.get_all: якщо quality_tier фільтр — WHERE quality_tier IN (...)
>
> Поверни всі змінені файли.
> ```

---

## ФАЗА 7 — TESTING (Тиждень 10) 🟠

### Крок 7.1 — Pytest Suite Setup + Unit Tests

**Пріоритет**: 🟠 Високий | **Файли**: нові `tests/` директорія
**Готово коли**: `pytest tests/unit/` проходить з 100% success; coverage > 80%

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python Test Engineer. В CRM немає pytest-based тестового suite. Потрібно створити з нуля.
>
> ЗАВДАННЯ: Створити повну тестову інфраструктуру:
>
> 1. tests/conftest.py:
>    - Fixtures: async_session (SQLite in-memory), mock_openai (mocker.patch), sample_lead, sample_sale
>    - pytest.ini або pyproject.toml: asyncio_mode = "auto"
>
> 2. tests/unit/test_lead_service.py — клас TestStageTransition:
>    - test_new_to_contacted_ok
>    - test_skip_stage_raises_LeadStageError (NEW → QUALIFIED)
>    - test_terminal_transferred_locked
>    - test_terminal_lost_locked
>    - test_any_stage_to_lost_allowed
>    - test_rollback_qualified_to_contacted_ok
>    - test_rollback_new_raises
>
> 3. tests/unit/test_transfer_service.py:
>    - test_transfer_requires_qualified_stage
>    - test_transfer_requires_ai_score_above_threshold
>    - test_transfer_requires_business_domain
>    - test_successful_transfer_creates_sale
>
> 4. tests/unit/test_ai_prompts.py:
>    - test_all_lead_sources_pass_validation
>    - test_all_cold_stages_pass_validation
>    - test_invalid_source_raises_ValueError
>
> 5. requirements-dev.txt: pytest, pytest-asyncio, pytest-mock, pytest-cov, httpx, faker
>
> Поверни всі файли з повними реалізаціями тестів.
> ```

---

### Крок 7.2 — Integration Tests (API)

**Пріоритет**: 🟠 Високий | **Файли**: `tests/integration/`
**Готово коли**: full pipeline happy path тест проходить end-to-end

---

> **ПРОМПТ**
>
> ```
> Ти Senior Test Engineer. Створи integration тести для AEL CRM API використовуючи httpx.AsyncClient та FastAPI testclient.
>
> ЗАВДАННЯ:
>
> 1. tests/integration/conftest.py:
>    - async_client fixture: AsyncClient(app=app, base_url="http://test")
>    - auth_headers fixture: отримує JWT token через POST /auth/login з тестовим юзером
>    - seed_db fixture: створює тестові дані в SQLite in-memory
>
> 2. tests/integration/test_lead_lifecycle.py:
>    def test_full_happy_path():
>      POST /leads → 201, отримуємо id
>      PATCH /leads/{id}/stage {stage: CONTACTED} → 200
>      PATCH /leads/{id}/stage {stage: QUALIFIED} → 200
>      POST /leads/{id}/analyze (з mock_openai score=0.75) → 200
>      POST /leads/{id}/transfer → 201, sale_id отримуємо
>      PATCH /sales/{sale_id}/stage {stage: KYC} → 200
>      PATCH /sales/{sale_id}/stage {stage: AGREEMENT, amount: 50000} → 200
>      PATCH /sales/{sale_id}/stage {stage: PAID} → 200
>      Весь цикл має пройти без помилок
>
>    def test_skip_stage_returns_400():
>      POST /leads → PATCH stage QUALIFIED (skip) → 400 detail contains "Expected"
>
>    def test_transfer_without_score_returns_400()
>    def test_transfer_without_domain_returns_400()
>    def test_duplicate_lead_returns_409()
>
> Поверни всі файли.
> ```

---

## ФАЗА 8 — DASHBOARD & BOT (Тиждень 11+) 🟢

### Крок 8.1 — React Kanban Board

**Пріоритет**: 🟢 Низький | **Файли**: `dashboard/src/`
**Готово коли**: drag-and-drop між колонками викликає PATCH /stage API

---

> **ПРОМПТ**
>
> ```
> Ти Senior React/TypeScript Engineer. Додай Kanban Board до CRM dashboard для управління лідами drag-and-drop.
>
> СТЕК: React 18, TypeScript, @dnd-kit/core (або react-beautiful-dnd), axios, TailwindCSS або CSS modules.
>
> ЗАВДАННЯ:
> 1. Компонент KanbanBoard.tsx з колонками: NEW | CONTACTED | QUALIFIED | TRANSFERRED | LOST
> 2. Кожна картка LeadCard.tsx: показує #id, source, ai_score badge (кольоровий), message_count
> 3. AI score badge: score >= 0.8 → 🔥 red; 0.6-0.8 → 🟢 green; 0.3-0.6 → 🟡 yellow; <0.3 → ⚫ gray
> 4. Drag end handler: визначає новий stage, PATCH /api/v1/leads/{id}/stage
> 5. Оптимістичне оновлення UI (рухаємо картку миттєво), rollback при HTTP error
> 6. WIP limits: QUALIFIED максимум 20 лідів — показувати попередження якщо перевищено
> 7. WebSocket: підписуватись на ws://host/ws/dashboard, при події lead_updated — рефетч колонки
>
> Поверни KanbanBoard.tsx, LeadCard.tsx та необхідні типи.
> ```

---

### Крок 8.2 — Manager Performance Dashboard

**Пріоритет**: 🟢 Низький | **Файли**: новий endpoint + React компонент
**Готово коли**: /dashboard/managers API та React таблиця з метриками

---

> **ПРОМПТ**
>
> ```
> Ти Senior Fullstack Engineer. Додай Manager Performance Dashboard до CRM.
>
> BACKEND (FastAPI):
> Новий endpoint: GET /api/v1/dashboard/managers
> SQL aggregation query:
> SELECT
>   u.id, u.full_name,
>   COUNT(l.id) as assigned_count,
>   COUNT(CASE WHEN l.stage = 'TRANSFERRED' THEN 1 END) as converted_count,
>   COUNT(CASE WHEN l.stage = 'LOST' THEN 1 END) as lost_count,
>   AVG(l.ai_score) as avg_ai_score,
>   AVG(EXTRACT(EPOCH FROM (s.created_at - l.created_at))/86400) as avg_days_to_close
> FROM users u
> LEFT JOIN leads l ON l.assigned_to_id = u.id
> LEFT JOIN sales s ON s.lead_id = l.id AND s.stage = 'PAID'
> GROUP BY u.id, u.full_name
> ORDER BY converted_count DESC
>
> FRONTEND (React):
> Компонент ManagerPerformanceTable.tsx:
> - Таблиця: Manager | Assigned | Converted | Lost | Conversion% | Avg Score | Avg Days to Close
> - Конверсія = converted/assigned * 100
> - Сортування по кожній колонці
> - Клік на Manager → фільтрує основний список лідів по assigned_to_id
>
> Поверни backend endpoint та React компонент.
> ```

---

### Крок 8.3 — Telegram Bot: Quick Actions + Smart Notifications

**Пріоритет**: 🟢 Низький | **Файли**: `app/bot/`, `app/celery/`
**Готово коли**: при перегляді ліда є inline кнопки; stale leads отримують notification

---

> **ПРОМПТ**
>
> ```
> Ти Senior Python/Aiogram Engineer. Покращи Telegram Bot CRM двома функціями.
>
> ФУНКЦІЯ 1 — Quick Actions при перегляді ліда:
> Замість текстового меню — InlineKeyboardMarkup з кнопками:
> Ряд 1: [📞 Contacted] [🤖 AI Analyze] [📝 Add Note]
> Ряд 2: [➡️ Qualified ] [❌ Mark Lost ]  [👤 Reassign ]
> Ряд 3: [💸 Transfer  ] [📎 Attach     ] [⬅️ Back     ]
>
> Для кнопок Stage transition — додати підтвердження:
> "Перевести в QUALIFIED? Цю дію можна скасувати через rollback."
> → [✅ Підтвердити] [❌ Скасувати]
>
> ФУНКЦІЯ 2 — Smart Notifications (Celery task):
> Задача: celery beat кожні 6 годин запускає check_and_notify_stale_leads
> - Шукає ліди: stage IN (CONTACTED, QUALIFIED) AND last_activity_at < NOW() - 7 days AND assigned_to_id IS NOT NULL
> - Для кожного: відправляє повідомлення менеджеру (via bot.send_message):
>   "⚠️ Lead #{id} ({source}) без активності вже {days} днів. AI score: {score:.2f}"
>   + кнопки: [📞 Contacted] [❌ Lost]
> - Запис в history: NURTURE запис
> - Максимум 10 нотифікацій на один run щоб не флудити
>
> Поверни змінені handler та celery task.
> ```

---

## Чеклист прогресу

| Фаза | Крок | Задача | Статус |
|------|------|--------|--------|
| 1 | 1.1 | Синхронізація AI констант | ⬜ |
| 1 | 1.2 | Async file I/O | ⬜ |
| 1 | 1.3 | Виправлення imports | ⬜ |
| 1 | 1.4 | Race condition current_leads | ⬜ |
| 1 | 1.5 | Notes SQL pagination | ⬜ |
| 1 | 1.6 | Transfer QUALIFIED validation | ⬜ |
| 1 | 1.7 | Duplicate lead detection | ⬜ |
| 2 | 2.1 | Structured logging | ⬜ |
| 2 | 2.2 | Extended health check | ⬜ |
| 2 | 2.3 | Prometheus metrics | ⬜ |
| 3 | 3.1 | JWT Authentication | ⬜ |
| 3 | 3.2 | Input sanitization | ⬜ |
| 3 | 3.3 | Per-user rate limiting | ⬜ |
| 4 | 4.1 | OpenAI Structured Outputs | ⬜ |
| 4 | 4.2 | Fallback rule-based scorer | ⬜ |
| 4 | 4.3 | AI Audit Log | ⬜ |
| 4 | 4.4 | AI Feature Engineering + Rate Limit | ⬜ |
| 5 | 5.1 | Database indexes | ⬜ |
| 5 | 5.2 | Soft delete | ⬜ |
| 5 | 5.3 | Cursor pagination | ⬜ |
| 5 | 5.4 | Bulk operations API | ⬜ |
| 6 | 6.1 | Lead Score History | ⬜ |
| 6 | 6.2 | Sales pipeline rules | ⬜ |
| 6 | 6.3 | Stage rollback | ⬜ |
| 6 | 6.4 | quality_tier field | ⬜ |
| 7 | 7.1 | Unit tests suite | ⬜ |
| 7 | 7.2 | Integration tests | ⬜ |
| 8 | 8.1 | React Kanban Board | ⬜ |
| 8 | 8.2 | Manager Performance Dashboard | ⬜ |
| 8 | 8.3 | Bot Quick Actions + Notifications | ⬜ |

---

*Кожен промпт вставляй напряму в чат з AI-асистентом. Перед кожним кроком — переконайся що попередній тест пройшов.*
