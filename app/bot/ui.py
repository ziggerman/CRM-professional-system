"""
Centralized UI layer — emoji maps, message formatters, and visual helpers.
All bot messages flow through here for a consistent professional look.
"""
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Stage Metadata
# ─────────────────────────────────────────────────────────────
STAGE_META = {
    "NEW":         {"emoji": "🆕", "label": "New Lead",      "short": "NEW",   "order": 0, "desc": "Лише доданий в систему. Потребує первинного контакту."},
    "CONTACTED":   {"emoji": "📞", "label": "Contacted",     "short": "CONT",  "order": 1, "desc": "Перший контакт здійснено (дзвінок, повідомлення). Очікуємо відповідь."},
    "QUALIFIED":   {"emoji": "✅", "label": "Qualified",     "short": "QUAL",  "order": 2, "desc": "Лід пройшов кваліфікацію: є бюджет, потреба та інтерес."},
    "TRANSFERRED": {"emoji": "🚀", "label": "Transferred",   "short": "TRANS", "order": 3, "desc": "Ліда передано у відділ продажів для закриття угоди."},
    "LOST":        {"emoji": "❌", "label": "Lost",          "short": "LOST",  "order": 4, "desc": "Лід відмовився або перестав виходити на зв'язок."},
}

SALE_STAGE_META = {
    "NEW":       {"emoji": "🆕", "label": "New Deal",  "order": 0, "desc": "Нова угода створена відділом продажів."},
    "KYC":       {"emoji": "📋", "label": "KYC",       "order": 1, "desc": "Процес верифікації клієнта та збору документів."},
    "AGREEMENT": {"emoji": "📝", "label": "Agreement", "order": 2, "desc": "Узгодження умов договору та підписання."},
    "PAID":      {"emoji": "💰", "label": "Paid",      "order": 3, "desc": "Угоду успішно закрито! Оплату отримано."},
    "LOST":      {"emoji": "❌", "label": "Lost Deal", "order": 4, "desc": "Угода зірвалась. Причину див. у нотатках."},
}

SOURCE_META = {
    # ТЗ: scanner / partner / manual
    "SCANNER":      {"emoji": "🔍", "label": "Scanner",     "desc": "Лід знайдений через сканер або парсинг."},
    "PARTNER":      {"emoji": "🤝", "label": "Partner",     "desc": "Лід від партнера або реферальної програми."},
    "MANUAL":       {"emoji": "✏️", "label": "Manual",      "desc": "Додано менеджером вручну під час холодного обдзвону."},
}

DOMAIN_META = {
    # ТЗ: first / second / third
    "FIRST":      {"emoji": "1️⃣", "label": "First",      "desc": "Перша категорія бізнесу."},
    "SECOND":     {"emoji": "2️⃣", "label": "Second",     "desc": "Друга категорія бізнесу."},
    "THIRD":      {"emoji": "3️⃣", "label": "Third",      "desc": "Третя категорія бізнесу."},
}


# ─────────────────────────────────────────────────────────────
# Visual Helpers
# ─────────────────────────────────────────────────────────────

def pipeline_bar_lead(stage: str) -> str:
    """Render a 5-step pipeline progress bar for lead stages."""
    stages = ["NEW", "CONTACTED", "QUALIFIED", "TRANSFERRED", "LOST"]
    emojis = ["🆕", "📞", "✅", "🚀", "❌"]

    if stage == "LOST":
        # Lost is always final, highlight in red position
        dots = ["◉", "◉", "◉", "◉", "✖"]
        bar = " → ".join(dots)
        return f"<code>{bar}</code>"

    order = STAGE_META.get(stage, {}).get("order", 0)
    parts = []
    for i, s in enumerate(stages[:-1]):  # exclude 'lost'
        if i < order:
            parts.append("◉")  # completed
        elif i == order:
            parts.append("●")  # current
        else:
            parts.append("○")  # future

    bar = " → ".join(parts)
    return f"<code>{bar}</code>"


def ai_score_bar(score: Optional[float]) -> str:
    """Render a 10-block AI score bar."""
    if score is None:
        return "<i>Not analyzed yet</i>"
    filled = round(score * 10)
    bar = "▓" * filled + "░" * (10 - filled)
    pct = round(score * 100)
    icon = "🔥" if pct >= 80 else "💡" if pct >= 50 else "❄️"
    return f"{icon} <code>{bar}</code> {pct}%"


def pipeline_bar_sale(stage: str) -> str:
    """Render a 5-step pipeline progress bar for sale stages."""
    stages = ["NEW", "KYC", "AGREEMENT", "PAID", "LOST"]
    
    if stage == "LOST":
        dots = ["◉", "◉", "◉", "◉", "✖"]
        bar = " → ".join(dots)
        return f"<code>{bar}</code>"

    order = SALE_STAGE_META.get(stage, {}).get("order", 0)
    parts = []
    for i, s in enumerate(stages[:-1]):  # exclude 'lost'
        if i < order:
            parts.append("◉")  # completed
        elif i == order:
            parts.append("●")  # current
        else:
            parts.append("○")  # future

    bar = " → ".join(parts)
    return f"<code>{bar}</code>"


def fmt_stage(stage: Optional[str]) -> str:
    if not stage:
        return "—"
    m = STAGE_META.get(stage, {})
    return f"{m.get('emoji', '❓')} {m.get('label', stage.title())}"


def fmt_source(source: Optional[str]) -> str:
    if not source:
        return "—"
    m = SOURCE_META.get(source, {})
    return f"{m.get('emoji', '•')} {m.get('label', source.title())}"


def fmt_domain(domain: Optional[str]) -> str:
    if not domain:
        return "—"
    m = DOMAIN_META.get(domain, {})
    return f"{m.get('emoji', '•')} {m.get('label', domain.title())}"


def fmt_sale_stage(stage: Optional[str]) -> str:
    if not stage:
        return "—"
    m = SALE_STAGE_META.get(stage, {})
    return f"{m.get('emoji', '❓')} {m.get('label', stage.title())}"


def fmt_date(dt_str: Optional[str]) -> str:
    """Format ISO datetime string to readable format."""
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        return dt_str[:10] if len(dt_str) >= 10 else dt_str


def fmt_amount(amount_cents: Optional[int]) -> str:
    """Format cents to dollar display."""
    if amount_cents is None:
        return "—"
    return f"${amount_cents / 100:,.2f}"


# ─────────────────────────────────────────────────────────────
# Message Formatters
# ─────────────────────────────────────────────────────────────
def format_lead_creation_step(step: str, title: str, description: str, hint: str = None) -> str:
    """Format a step in the multi-stage lead capture flow."""
    text = (
        f"<b>{title}</b>\n"
        f"📊 Step <code>{step}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{description}\n\n"
    )
    if hint:
        text += f"💡 <i>{hint}</i>"
    else:
        text += "<i>Use the buttons below to choose an option or skip.</i>"
    return text

def format_welcome(name: str, is_admin: bool = False) -> str:
    role_line = "👑 <b>Admin Access</b> — full control enabled" if is_admin else "📩 Contact your admin for full access."
    return (
        f"👋 Hello, <b>{name}</b>!\n\n"
        f"<b>⚡ AEL CRM</b> — Your AI-Powered Sales Command Center\n\n"
        f"<b>What you can do:</b>\n"
        f"├─ 📋 Manage leads across the full pipeline\n"
        f"├─ 🤖 Run AI analysis on any lead\n"
        f"├─ 📊 View real-time stats & dashboard\n"
        f"├─ ⚡ Quick actions at your fingertips\n"
        f"└─ 🔍 Search leads by any attribute\n\n"
        f"{role_line}"
    )


def format_lead_card(lead: dict, show_pipeline: bool = True) -> str:
    """Rich lead detail card with all fields."""
    # New fields
    full_name = lead.get("full_name")
    email = lead.get("email")
    phone = lead.get("phone")
    intent = lead.get("intent")
    company = lead.get("company")
    position = lead.get("position")
    budget = lead.get("budget")
    budget = lead.get("budget")
    pain = lead.get("pain_points")

    lead_id = lead.get("id", "?")
    stage = lead.get("stage", "NEW")
    source = lead.get("source")
    domain = lead.get("business_domain")
    assigned = lead.get("assigned_to_id")
    msgs = lead.get("message_count", 0)
    notes_count = lead.get("notes_count", 0)
    ai_score = lead.get("ai_score")
    ai_rec = lead.get("ai_recommendation")
    ai_reason = lead.get("ai_reason")
    created = lead.get("created_at")

    stage_info = STAGE_META.get(stage, {"emoji": "❓", "label": stage})

    text = (
        f"📄 <b>Lead #{lead_id}</b>  {stage_info['emoji']} <b>{stage_info['label']}</b>\n"
    )

    if show_pipeline:
        text += f"{pipeline_bar_lead(stage)}\n\n"
    else:
        text += "\n"

    text += (
        f"👤 <b>Name:</b>  <b>{full_name or '—'}</b>\n"
        f"📧 <b>Email:</b>  {email or '—'}\n"
        f"📞 <b>Phone:</b>  {phone or '—'}\n\n"
        f"📥 <b>Source:</b>  {fmt_source(source)}\n"
        f"🏢 <b>Domain:</b>  {fmt_domain(domain)}\n"
        f"🎯 <b>Intent:</b>  {intent or '—'}\n"
        f"👤 <b>Assigned:</b>  {'#' + str(assigned) if assigned else '—'}\n"
        f"📨 <b>Messages:</b>  {msgs}   📝 <b>Notes:</b> {notes_count}\n\n"
    )

    if company or position or budget or pain:
        text += "🏢 <b>B2B Qualification</b>\n"
        if company: text += f"├─ Company:  {company}\n"
        if position: text += f"├─ Position: {position}\n"
        if budget: text += f"├─ Budget:   {budget}\n"
        if pain: text += f"└─ Pain:     <i>{pain[:100]}{'...' if len(pain)>100 else ''}</i>\n"
        text += "\n"

    if ai_score is not None or ai_rec:
        text += f"<b>🤖 AI Score:</b>\n{ai_score_bar(ai_score)}\n"
        if ai_rec:
            text += f"<b>💡 Recommendation:</b>  <i>{ai_rec}</i>\n"
        if ai_reason:
            text += f"<b>📋 Reason:</b>  <i>{ai_reason}</i>\n"
        text += "\n"

    text += f"<b>🗓 Created:</b>  {fmt_date(created)}"

    return text


def format_lead_row(lead: dict) -> str:
    """Compact one-line lead summary for list view."""
    lead_id = lead.get("id", "?")
    stage = lead.get("stage", "NEW")
    domain = lead.get("business_domain") or "—"
    source = lead.get("source", "?")
    ai_sc = lead.get("ai_score")

    stage_emoji = STAGE_META.get(stage, {}).get("emoji", "❓")
    src_emoji = SOURCE_META.get(source, {}).get("emoji", "•")
    domain_emoji = DOMAIN_META.get(domain, {}).get("emoji", "•") if domain else "•"
    score_str = f"  🤖{round(ai_sc * 100)}%" if ai_sc is not None else ""

    return f"#{lead_id} {stage_emoji} {domain_emoji}{score_str}  {src_emoji}"


def format_leads_list(leads: list, title: str, page: int = 0, total_pages: int = 1) -> str:
    """Paginated leads list header."""
    if not leads:
        return (
            f"📋 <b>{title}</b>\n\n"
            f"<i>No leads found in this category.</i>\n\n"
            f"Try a different filter or add a new lead."
        )

    count = len(leads)
    page_info = f"  ·  Page {page + 1}/{total_pages}" if total_pages > 1 else ""
    header = f"📋 <b>{title}</b>  <i>({count} leads{page_info})</i>\n\n"
    header += "<i>Tap a lead to view details:</i>"
    return header


def format_dashboard(stats: dict) -> str:
    """Visual dashboard with tree layout."""
    leads = stats.get("leads", {})
    sales = stats.get("sales", {})

    total_leads = leads.get("total", 0)
    total_transferred = leads.get("transferred", 0)
    conv = round(total_transferred / total_leads * 100, 1) if total_leads > 0 else 0

    total_sales = sales.get("total", 0)
    total_paid = sales.get("paid", 0)
    total_rev = stats.get("total_revenue", 0)
    avg_deal = stats.get("avg_deal_amount", 0)
    sales_conv = round(total_paid / total_sales * 100, 1) if total_sales > 0 else 0

    conv_bar_count = round(conv / 10)
    conv_bar = "▓" * conv_bar_count + "░" * (10 - conv_bar_count)

    now = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    return (
        f"📊 <b>DASHBOARD</b>  <i>({now})</i>\n\n"
        f"<b>📈 LEADS PIPELINE</b>\n"
        f"├─ Total:        <b>{total_leads}</b>\n"
        f"├─ 🆕 New:       {leads.get('new', 0)}\n"
        f"├─ 📞 Contacted: {leads.get('contacted', 0)}\n"
        f"├─ ✅ Qualified: {leads.get('qualified', 0)}\n"
        f"├─ 🚀 Transferred: {leads.get('transferred', 0)}\n"
        f"└─ ❌ Lost:      {leads.get('lost', 0)}\n\n"
        f"<b>💰 SALES PIPELINE</b>\n"
        f"├─ Total:        <b>{total_sales}</b>\n"
        f"├─ 📋 KYC:       {sales.get('kyc', 0)}\n"
        f"├─ 📝 Agreement: {sales.get('agreement', 0)}\n"
        f"├─ 💰 Paid:      {sales.get('paid', 0)}\n"
        f"└─ ❌ Lost:      {sales.get('lost', 0)}\n\n"
        f"<b>📊 KEY METRICS</b>\n"
        f"├─ Lead→Sale:   <code>{conv_bar}</code> {conv}%\n"
        f"├─ Sale→Paid:   {sales_conv}%\n"
        f"├─ Avg Deal:    {fmt_amount(int(avg_deal * 100) if avg_deal else None)}\n"
        f"└─ Revenue:     {fmt_amount(int(total_rev * 100) if total_rev else None)}"
    )


def format_stats_simple(leads: list) -> str:
    """Quick stats from a raw leads list (no API stats endpoint needed)."""
    total = len(leads)
    if total == 0:
        return "📊 <b>STATS</b>\n\n<i>No leads yet. Add your first lead!</i>"

    counts = {}
    for lead in leads:
        s = lead.get("stage", "new")
        counts[s] = counts.get(s, 0) + 1

    transferred = counts.get("transferred", 0)
    lost = counts.get("lost", 0)
    conv = round(transferred / total * 100, 1) if total > 0 else 0

    conv_filled = round(conv / 10)
    conv_bar = "▓" * conv_filled + "░" * (10 - conv_filled)

    return (
        f"📊 <b>STATS</b>\n\n"
        f"<b>Total Leads:</b>  {total}\n\n"
        f"├─ 🆕 New:        {counts.get('new', 0)}\n"
        f"├─ 📞 Contacted:  {counts.get('contacted', 0)}\n"
        f"├─ ✅ Qualified:  {counts.get('qualified', 0)}\n"
        f"├─ 🚀 Transferred:{transferred}\n"
        f"└─ ❌ Lost:       {lost}\n\n"
        f"<b>Conversion Rate</b>\n"
        f"<code>{conv_bar}</code> {conv}%"
    )


def format_sale_card(sale: dict, lead: dict = None) -> str:
    """Rich sale card."""
    sale_id = sale.get("id", "?")
    lead_id = sale.get("lead_id", "?")
    stage = sale.get("stage", "NEW")
    amount = sale.get("amount")
    notes = sale.get("notes")
    created = sale.get("created_at")

    stage_info = SALE_STAGE_META.get(stage, {"emoji": "❓", "label": stage})

    stages_order = ["NEW", "KYC", "AGREEMENT", "PAID"]
    if stage not in ["LOST"]:
        order = SALE_STAGE_META.get(stage, {}).get("order", 0)
        parts = []
        for i, s in enumerate(stages_order):
            if i < order:
                parts.append("◉")
            elif i == order:
                parts.append("●")
            else:
                parts.append("○")
        pipeline = "<code>" + " → ".join(parts) + "</code>\n\n"
    else:
        pipeline = "<code>◉ → ◉ → ◉ → ✖</code>\n\n"

    text = (
        f"💼 <b>Sale #{sale_id}</b>  {stage_info['emoji']} <b>{stage_info['label']}</b>\n"
        f"{pipeline}"
        f"<b>🔗 Lead:</b>  #{lead_id}\n"
        f"<b>💵 Amount:</b>  {fmt_amount(amount)}\n"
        f"<b>🗓 Created:</b>  {fmt_date(created)}\n"
    )

    if notes:
        text += f"\n<b>📝 Notes:</b>\n<i>{notes}</i>"

    return text


def format_delete_confirm(lead_id) -> str:
    return (
        f"⚠️ <b>DELETE LEAD #{lead_id}</b>\n\n"
        f"This action is <b>permanent</b> and cannot be undone.\n"
        f"All notes associated with this lead will also be deleted.\n\n"
        f"Are you sure?"
    )




def format_lead_confirm_card(data: dict) -> str:
    """Final summary card before creation."""
    name = data.get("full_name") or "—"
    email = data.get("email") or "—"
    phone = data.get("phone") or "—"
    src = data.get("source", "manual")
    dom = data.get("business_domain")
    intent = data.get("intent") or "—"
    
    # B2B
    company = data.get("company") or "—"
    pos = data.get("position") or "—"
    budget = data.get("budget") or "—"
    pain = data.get("pain_points") or "—"
    
    return (
        f"🏁 <b>LEAD SUMMARY</b>\n"
        "<i>Please review before saving.</i>\n\n"
        f"👤 <b>Contact</b>\n"
        f"├ Name: {name}\n"
        f"├ Email: {email}\n"
        f"├ Phone: {phone}\n"
        f"└ Source: {fmt_source(src)}\n\n"
        f"🎯 <b>Intent & Domain</b>\n"
        f"├ Domain: {fmt_domain(dom)}\n"
        f"└ Intent: {intent}\n\n"
        f"🏢 <b>Qualification (B2B)</b>\n"
        f"├ Company: {company}\n"
        f"├ Position: {pos}\n"
        f"├ Budget: {budget}\n"
        f"└ Pain: {pain[:100]}{'...' if len(pain)>100 else ''}"
    )


def format_notes_menu(lead_id: int, notes_count: int) -> str:
    """Header for the notes management menu."""
    return (
        f"📝 <b>NOTES MANAGEMENT</b>  —  Lead #{lead_id}\n\n"
        f"Total notes: <b>{notes_count}</b>\n\n"
        "Select an action below:"
    )


def format_single_note(lead_id: int, note: dict, index: int, total: int) -> str:
    """Format one note for viewing."""
    author = note.get("author_name") or f"User {note.get('created_by')}" or "System"
    note_type = (note.get("note_type") or note.get("category") or "general").lower()
    is_pinned = bool(note.get("is_pinned", False))
    type_meta = {
        "general": ("📋", "General"),
        "contact": ("📞", "Contact"),
        "email": ("📧", "Email"),
        "meeting": ("💼", "Meeting"),
        "problem": ("⚠️", "Problem"),
        "success": ("✅", "Success"),
        "task": ("🧩", "Task"),
        "objection": ("🛑", "Objection"),
        "comment": ("💬", "Comment"),
        "system": ("⚙️", "System"),
        "ai": ("🤖", "AI"),
    }
    type_emoji, type_label = type_meta.get(note_type, ("📝", note_type.upper()))
    pin_line = "📌 <b>Pinned:</b> Yes\n" if is_pinned else ""
    date_str = note.get("created_at", "")
    if date_str:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_str = dt.strftime("%d.%m.%Y %H:%M")
        except:
            pass
            
    return (
        f"👁 <b>VIEWING NOTE {index + 1}/{total}</b>\n"
        f"Lead: <b>#{lead_id}</b>\n"
        f"Type: {type_emoji} <b>{type_label}</b>\n"
        f"{pin_line}"
        f"Date: <i>{date_str}</i>\n"
        f"By: <b>{author}</b>\n\n"
        f"📝 <i>\"{note.get('content')}\"</i>"
    )


def format_note_prompt(lead_id: int) -> str:
    """Prompt for typing a new note."""
    return (
        f"📝 <b>ADD NEW NOTE</b>  —  Lead #{lead_id}\n\n"
        "Please type or record your note below.\n"
        "<i>Max 500 characters. Files and photos are also accepted.</i>"
    )


def format_note_confirm(lead_id: int, text: str) -> str:
    """Preview note content before save/edit/discard actions."""
    safe_text = text or ""
    if len(safe_text) > 500:
        safe_text = safe_text[:500] + "..."
    return (
        f"📝 <b>CONFIRM NOTE</b>  —  Lead #{lead_id}\n\n"
        "Перевірте текст перед збереженням:\n\n"
        f"<i>{safe_text}</i>\n\n"
        "Оберіть дію нижче:"
    )


def format_advanced_stats(data: dict) -> str:
    """Format the deep analytical report."""
    total = data.get("total_leads", 0)
    cov = data.get("coverage", {})
    intents = data.get("intents", {})
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

    text = (
        f"📊 <b>ADVANCED ANALYTICS</b>\n"
        f"<i>({now})</i>\n\n"
        f"👥 Total Leads: <b>{total}</b>\n\n"
        f"📑 <b>DATA QUALITY</b>\n"
        f"├ Email Capture:  {cov.get('email') or 0}%\n"
        f"├ Phone Capture:  {cov.get('phone') or 0}%\n"
        f"├ B2B Company:    {cov.get('b2b_company') or 0}%\n"
        f"├ B2B Budget:     {cov.get('b2b_budget') or 0}%\n"
        f"└ B2B Pain:       {cov.get('b2b_pain') or 0}%\n\n"
    )

    if intents:
        text += "🎯 <b>INTENT DISTRIBUTION</b>\n"
        for label, count in intents.items():
            pct = round(count / total * 100, 1) if total > 0 else 0
            text += f"├ {label}: <b>{count}</b> ({pct}%)\n"
        text = text[:-1] # Remove last newline/separator
        
    return text


def format_sale_card(sale: dict) -> str:
    """Format a detailed view of a single sale."""
    sale_id = sale.get("id", "?")
    stage = sale.get("stage", "NEW")
    amount = sale.get("amount")
    notes = sale.get("notes") or "<i>No notes</i>"
    
    lead = sale.get("lead", {})
    lead_id = lead.get("id", "?")
    lead_name = lead.get("full_name") or "Unnamed"
    
    amount_str = f"<b>{amount / 100:.2f} USD</b>" if amount is not None else "<i>Not set</i>"
    
    text = (
        f"💰 <b>SALE DETAILS</b>  —  #{sale_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"<b>Stage:</b> {fmt_sale_stage(stage)}\n"
        f"{pipeline_bar_sale(stage)}\n\n"
        f"👤 <b>Client:</b> {lead_name} (#{lead_id})\n"
        f"💵 <b>Deal Amount:</b> {amount_str}\n\n"
        f"📝 <b>Sales Notes:</b>\n{notes}\n"
    )
    return text


def format_sales_list(sales: list, title: str, page: int, total_pages: int) -> str:
    """Format header for a list of sales."""
    count = len(sales)
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    
    text = (
        f"💰 <b>{title.upper()}</b>\n"
        f"<i>Updated: {now} (Pg {page+1}/{total_pages})</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    if not sales:
        text += "<i>No sales found.</i>"
    return text


def format_intent_stats(stats: dict) -> str:
    """Format intent distribution with emojis."""
    return format_advanced_stats(stats) # Reuse the core logic if similar


def format_settings(user_info: dict = None) -> str:
    if user_info:
        name = user_info.get("full_name", "Unknown")
        role = user_info.get("role", "manager").title()
        active = "✅ Active" if user_info.get("is_active") else "⛔ Inactive"
        current = user_info.get("current_leads", 0)
        max_l = user_info.get("max_leads", 50)
        return (
            f"⚙️ <b>SETTINGS</b>\n\n"
            f"<b>👤 Profile</b>\n"
            f"├─ Name: {name}\n"
            f"├─ Role: {role}\n"
            f"├─ Status: {active}\n"
            f"└─ Leads: {current}/{max_l}\n\n"
            f"<b>Configure your preferences below:</b>"
        )
    return (
        f"⚙️ <b>SETTINGS</b>\n\n"
        f"<b>Configure your preferences:</b>"
    )


def format_error(message: str, context: str = None) -> str:
    ctx = f"\n<code>{context}</code>" if context else ""
    return (
        f"❌ <b>Error</b>\n\n"
        f"{message}{ctx}\n\n"
        f"<i>Please try again or contact support.</i>"
    )


def format_success(message: str) -> str:
    return f"✅ <b>Success</b>\n\n{message}"


def format_loading(message: str = "Loading...") -> str:
    return f"⏳ <i>{message}</i>"


def format_search_prompt() -> str:
    return (
        f"🔍 <b>SEARCH LEADS</b>\n\n"
        f"Введіть ваш запит для пошуку:\n\n"
        f"<b>Доступні фільтри:</b>\n"
        f"├─ ID Ліда (напр. <code>42</code>)\n"
        f"├─ Сфера (<code>retail</code>, <code>finance</code>, <code>tech</code>)\n"
        f"├─ Джерело (<code>web</code>, <code>referral</code>, <code>social</code>)\n"
        f"└─ Стадія (<code>new</code>, <code>contacted</code>, та ін.)\n\n"
        f"<i>Натисніть Скасувати, щоб повернутися.</i>"
    )


def format_help() -> str:
    return (
        f"❓ <b>HELP & COMMANDS</b>\n\n"
        f"<b>Commands:</b>\n"
        f"├─ /start — Restart the bot\n"
        f"├─ /menu — Main menu\n"
        f"└─ /help — This help page\n\n"
        f"<b>Navigation Tips:</b>\n"
        f"├─ Use <b>📋 Leads</b> to browse by filter\n"
        f"├─ Tap a lead to open its detail card\n"
        f"├─ Use <b>⚡ Quick</b> for fast actions\n"
        f"├─ <b>📊 Stats</b> shows live pipeline stats\n"
        f"└─ <b>🤖 AI Analyze</b> scores any lead\n\n"
        f"<b>Lead Stages:</b>\n"
        f"🆕 New → 📞 Contacted → ✅ Qualified → 🚀 Transferred\n"
        f"                                              ↓\n"
        f"                                         ❌ Lost\n\n"
        f"<b>Support:</b>  Contact @admin"
    )
