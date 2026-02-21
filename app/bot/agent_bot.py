"""
Enhanced AI Agent Bot - Full interactive experience with AI agents.
"""
import logging
import asyncio
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties

from app.bot.config import bot_settings

logger = logging.getLogger(__name__)

# Create router
router = Router()

# ──────────────────────────────────────────────
# API Helper Functions
# ──────────────────────────────────────────────

async def fetch_api(endpoint: str) -> dict:
    """Fetch data from API."""
    import aiohttp
    base_url = "http://localhost:8000"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}{endpoint}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"Status {resp.status}"}
    except Exception as e:
        logger.error(f"API error: {e}")
        return {"error": str(e)}


async def post_api(endpoint: str, data: dict = None) -> dict:
    """Post data to API."""
    import aiohttp
    base_url = "http://localhost:8000"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}{endpoint}", json=data or {}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                return {"error": f"Status {resp.status}"}
    except Exception as e:
        logger.error(f"API error: {e}")
        return {"error": str(e)}


# ──────────────────────────────────────────────
# State Management
# ──────────────────────────────────────────────

class AgentState:
    """States for AI agent conversation."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    CHATTING = "chatting"
    RECOMMENDING = "recommending"


# ──────────────────────────────────────────────
# Rich Message Helpers
# ──────────────────────────────────────────────

def get_enhanced_main_menu() -> InlineKeyboardMarkup:
    """Enhanced main menu with AI agent."""
    builder = InlineKeyboardBuilder()
    
    # Row 1: Dashboard & My Leads
    builder.add(InlineKeyboardButton(text="📊 Dashboard", callback_data="menu:dashboard"))
    builder.add(InlineKeyboardButton(text="🔔 My Leads", callback_data="menu:myleads"))
    
    # Row 2: New Lead & Search
    builder.add(InlineKeyboardButton(text="➕ New Lead", callback_data="menu:newlead"))
    builder.add(InlineKeyboardButton(text="🔍 Search", callback_data="menu:search"))
    
    # Row 3: AI Agent & Automation
    builder.add(InlineKeyboardButton(text="🤖 AI Agent", callback_data="menu:aiagent"))
    builder.add(InlineKeyboardButton(text="⚙️ Automation", callback_data="menu:automation"))
    
    # Row 4: Settings & Help
    builder.add(InlineKeyboardButton(text="⚙️ Settings", callback_data="menu:settings"))
    builder.add(InlineKeyboardButton(text="❓ Help", callback_data="menu:help"))
    
    builder.adjust(2, 2, 2, 2)
    return builder.as_markup()


def get_back_to_main() -> InlineKeyboardMarkup:
    """Back to main menu button."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🏠 Main Menu", callback_data="menu:main"))
    return builder.as_markup()


def get_lead_detail_actions(lead_id: int) -> InlineKeyboardMarkup:
    """Lead detail action buttons."""
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="📞 Contact", callback_data=f"lead:contact:{lead_id}"))
    builder.add(InlineKeyboardButton(text="🤖 AI Analyze", callback_data=f"lead:analyze:{lead_id}"))
    builder.add(InlineKeyboardButton(text="✓ Qualify", callback_data=f"lead:qualify:{lead_id}"))
    builder.add(InlineKeyboardButton(text="➡️ Transfer", callback_data=f"lead:transfer:{lead_id}"))
    builder.add(InlineKeyboardButton(text="📝 Note", callback_data=f"lead:note:{lead_id}"))
    builder.add(InlineKeyboardButton(text="❌ Lost", callback_data=f"lead:lost:{lead_id}"))
    
    builder.adjust(2, 2, 2)
    return builder.as_markup()


def get_ai_agent_menu() -> InlineKeyboardMarkup:
    """AI Agent submenu."""
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="🔍 Analyze Lead", callback_data="ai:analyze"))
    builder.add(InlineKeyboardButton(text="💡 Get Recommendations", callback_data="ai:recommend"))
    builder.add(InlineKeyboardButton(text="📈 Predict Conversion", callback_data="ai:predict"))
    builder.add(InlineKeyboardButton(text="🎯 Auto-Qualify", callback_data="ai:autofill"))
    builder.add(InlineKeyboardButton(text="💬 Chat with AI", callback_data="ai:chat"))
    builder.add(InlineKeyboardButton(text="🏠 Back", callback_data="menu:main"))
    
    builder.adjust(2, 2, 2)
    return builder.as_markup()


def get_filter_keyboard() -> InlineKeyboardMarkup:
    """Lead filter options."""
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="🆕 All", callback_data="filter:all"))
    builder.add(InlineKeyboardButton(text="🆕 New", callback_data="filter:new"))
    builder.add(InlineKeyboardButton(text="📞 Contacted", callback_data="filter:contacted"))
    builder.add(InlineKeyboardButton(text="✓ Qualified", callback_data="filter:qualified"))
    builder.add(InlineKeyboardButton(text="➡️ Transferred", callback_data="filter:transferred"))
    builder.add(InlineKeyboardButton(text="❌ Lost", callback_data="filter:lost"))
    builder.add(InlineKeyboardButton(text="🔄 Clear Filter", callback_data="filter:clear"))
    
    builder.adjust(3, 3, 1)
    return builder.as_markup()


# ──────────────────────────────────────────────
# Message Formatters
# ──────────────────────────────────────────────

def format_dashboard(stats: dict) -> str:
    """Format dashboard message."""
    leads = stats.get("leads", {})
    sales = stats.get("sales", {})
    
    return f"""
📊 <b>DASHBOARD</b>

<b>📈 LEADS</b>
├─ Total: {leads.get('total', 0)}
├─ 🆕 New: {leads.get('new', 0)}
├─ 📞 Contacted: {leads.get('contacted', 0)}
├─ ✓ Qualified: {leads.get('qualified', 0)}
├─ ➡️ Transferred: {leads.get('transferred', 0)}
└─ ❌ Lost: {leads.get('lost', 0)}

<b>💰 SALES</b>
├─ Total: {sales.get('total', 0)}
├─ 📋 KYC: {sales.get('kyc', 0)}
├─ 📝 Agreement: {sales.get('agreement', 0)}
├─ 💵 Paid: {sales.get('paid', 0)}
└─ ❌ Lost: {sales.get('lost', 0)}

<b>📊 METRICS</b>
├─ Conversion: {stats.get('conversion_rate', 0)}%
├─ Avg Deal: ${stats.get('avg_deal_amount', 0):.0f}
└─ Revenue: ${stats.get('total_revenue', 0):,}
"""


def format_lead_card(lead: dict) -> str:
    """Format single lead card."""
    status_emoji = {
        "new": "🆕",
        "contacted": "📞",
        "qualified": "✓",
        "transferred": "➡️",
        "lost": "❌"
    }
    
    emoji = status_emoji.get(lead.get("stage", "new"), "❓")
    domain = lead.get("business_domain", "N/A")
    source = lead.get("source", "N/A")
    score = lead.get("ai_score", "N/A")
    
    return f"""
<b>Lead #{lead.get('id')}</b> {emoji}

🏢 Domain: <b>{domain}</b>
📥 Source: {source}
🤖 AI Score: {score}
📅 Created: {lead.get('created_at', 'N/A')[:10]}
"""


def format_leads_list(leads: list, title: str = "LEADS") -> str:
    """Format leads list."""
    if not leads:
        return f"📋 No leads found."
    
    text = f"📋 <b>{title}</b>\n\n"
    
    for lead in leads[:10]:  # Show first 10
        text += format_lead_card(lead)
        text += "\n" + "─" * 20 + "\n\n"
    
    return text


# ──────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    """Handle /start command with welcome flow."""
    user = message.from_user
    user_id = user.id
    is_admin = user_id in bot_settings.TELEGRAM_ADMIN_IDS
    
    welcome_text = f"""
👋 <b>Welcome to CRM Bot!</b>

<b>Your AI-Powered Sales Assistant</b>

I'm here to help you manage leads and sales with the power of AI.

<b>Features:</b>
✓ AI-powered lead analysis
✓ Smart recommendations
✓ Automated workflows
✓ Real-time dashboard
✓ Instant notifications

{"🎉 You have admin access!" if is_admin else "📩 Contact admin for full access."}

<i>Loading your dashboard...</i>
"""
    
    await message.answer(welcome_text, reply_markup=get_enhanced_main_menu())
    
    # Auto-fetch dashboard
    try:
        stats = await fetch_api("/api/v1/dashboard")
        if "error" not in stats:
            await message.answer(format_dashboard(stats))
    except:
        pass


@router.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext):
    """Main menu."""
    await message.answer(
        "📋 <b>MAIN MENU</b>\n\nSelect an option:",
        reply_markup=get_enhanced_main_menu()
    )


# ──────────────────────────────────────────────
# Menu Callbacks
# ──────────────────────────────────────────────

@router.callback_query(F.data == "menu:main")
async def cb_main_menu(callback: CallbackQuery, state: FSMContext):
    """Main menu callback."""
    await callback.message.edit_text(
        "📋 <b>MAIN MENU</b>\n\nSelect an option:",
        reply_markup=get_enhanced_main_menu()
    )
    await callback.answer()


@router.callback_query(F.data == "menu:dashboard")
async def cb_dashboard(callback: CallbackQuery, state: FSMContext):
    """Dashboard callback."""
    await callback.message.edit_text("📊 Loading dashboard...")
    
    stats = await fetch_api("/api/v1/dashboard")
    
    if "error" in stats:
        await callback.message.edit_text(
            f"❌ Error loading dashboard: {stats['error']}",
            reply_markup=get_back_to_main()
        )
    else:
        await callback.message.edit_text(
            format_dashboard(stats),
            reply_markup=get_back_to_main()
        )
    
    await callback.answer()


@router.callback_query(F.data == "menu:myleads")
async def cb_my_leads(callback: CallbackQuery, state: FSMContext):
    """My leads callback."""
    await callback.message.edit_text(
        "🔔 <b>MY LEADS</b>\n\nFilter leads:",
        reply_markup=get_filter_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data.startswith("filter:"))
async def cb_filter_leads(callback: CallbackQuery, state: FSMContext):
    """Filter leads callback."""
    filter_type = callback.data.split(":")[1]
    
    if filter_type == "clear":
        stage = None
        title = "ALL LEADS"
    else:
        stage = filter_type
        title = f"{filter_type.upper()} LEADS"
    
    await callback.message.edit_text(f"🔍 Loading {title}...")
    
    endpoint = f"/api/v1/leads?stage={stage}" if stage else "/api/v1/leads"
    leads = await fetch_api(endpoint)
    
    if "error" in leads:
        await callback.message.edit_text(f"❌ Error: {leads['error']}")
    elif isinstance(leads, dict):
        items = leads.get("items", [])
        await callback.message.edit_text(
            format_leads_list(items, title),
            reply_markup=get_back_to_main()
        )
    else:
        await callback.message.edit_text(
            format_leads_list([], title),
            reply_markup=get_back_to_main()
        )
    
    await callback.answer()


@router.callback_query(F.data == "menu:newlead")
async def cb_new_lead(callback: CallbackQuery, state: FSMContext):
    """New lead callback."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🔍 Scanner", callback_data="newlead:scanner"))
    builder.add(InlineKeyboardButton(text="🤝 Partner", callback_data="newlead:partner"))
    builder.add(InlineKeyboardButton(text="✏️ Manual", callback_data="newlead:manual"))
    builder.add(InlineKeyboardButton(text="🏠 Back", callback_data="menu:main"))
    builder.adjust(3, 1)
    
    await callback.message.edit_text(
        "➕ <b>CREATE NEW LEAD</b>\n\nSelect source:",
        reply_markup=builder.as_markup()
    )
    await callback.answer()


@router.callback_query(F.data.startswith("newlead:"))
async def cb_create_lead(callback: CallbackQuery, state: FSMContext):
    """Create lead callback."""
    source = callback.data.split(":")[1]
    
    # In real implementation, would create via API
    await callback.message.edit_text(
        f"✅ Creating lead from <b>{source}</b>...\n\n"
        f"This would open a form to complete lead details.",
        reply_markup=get_back_to_main()
    )
    await callback.answer()


@router.callback_query(F.data == "menu:search")
async def cb_search(callback: CallbackQuery, state: FSMContext):
    """Search callback."""
    await callback.message.edit_text(
        "🔍 <b>SEARCH</b>\n\nEnter search query:\n\n"
        "You can search by:\n"
        "• Lead ID\n"
        "• Domain\n"
        "• Source\n"
        "• Stage",
        reply_markup=get_back_to_main()
    )
    await callback.answer()


@router.callback_query(F.data == "menu:aiagent")
async def cb_ai_agent(callback: CallbackQuery, state: FSMContext):
    """AI Agent menu."""
    await callback.message.edit_text(
        "🤖 <b>AI AGENT</b>\n\nYour AI-powered assistant:\n\n"
        "• Analyze leads with AI\n"
        "• Get smart recommendations\n"
        "• Predict conversion rates\n"
        "• Auto-qualify leads\n"
        "• Chat with AI",
        reply_markup=get_ai_agent_menu()
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ai:"))
async def cb_ai_action(callback: CallbackQuery, state: FSMContext):
    """AI action callbacks."""
    action = callback.data.split(":")[1]
    
    if action == "analyze":
        await callback.message.edit_text(
            "🔍 <b>AI ANALYZE</b>\n\n"
            "Enter lead ID to analyze:",
            reply_markup=get_back_to_main()
        )
    elif action == "recommend":
        await callback.message.edit_text(
            "💡 <b>RECOMMENDATIONS</b>\n\n"
            "Loading AI recommendations...",
            reply_markup=get_back_to_main()
        )
        
        # Fetch leads for recommendation
        leads = await fetch_api("/api/v1/leads?stage=qualified")
        if isinstance(leads, dict):
            items = leads.get("items", [])[:5]
            if items:
                text = "💡 <b>AI RECOMMENDATIONS</b>\n\n"
                for lead in items:
                    text += f"• Lead #{lead.get('id')}: {lead.get('business_domain')}\n"
                await callback.message.edit_text(text, reply_markup=get_ai_agent_menu())
            else:
                await callback.message.edit_text(
                    "💡 No qualified leads for recommendations.",
                    reply_markup=get_ai_agent_menu()
                )
    elif action == "predict":
        await callback.message.edit_text(
            "📈 <b>CONVERSION PREDICTION</b>\n\n"
            "Loading prediction models...",
            reply_markup=get_back_to_main()
        )
    elif action == "autofill":
        await callback.message.edit_text(
            "🎯 <b>AUTO-QUALIFY</b>\n\n"
            "AI will automatically qualify leads based on:\n"
            "• Engagement score\n"
            "• Business domain\n"
            "• Source quality\n"
            "• Historical data",
            reply_markup=get_back_to_main()
        )
    elif action == "chat":
        await callback.message.edit_text(
            "💬 <b>AI CHAT</b>\n\n"
            "Start chatting with AI assistant!\n\n"
            "Ask anything about:\n"
            "• Your leads\n"
            "• Sales pipeline\n"
            "• Recommendations\n"
            "• Best practices",
            reply_markup=get_back_to_main()
        )
    
    await callback.answer()


@router.callback_query(F.data == "menu:automation")
async def cb_automation(callback: CallbackQuery, state: FSMContext):
    """Automation menu."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🔄 Auto-Assign", callback_data="auto:assign"))
    builder.add(InlineKeyboardButton(text="🔔 Follow-up", callback_data="auto:followup"))
    builder.add(InlineKeyboardButton(text="📊 Process Stale", callback_data="auto:stale"))
    builder.add(InlineKeyboardButton(text="📈 Daily Stats", callback_data="auto:stats"))
    builder.add(InlineKeyboardButton(text="🏠 Back", callback_data="menu:main"))
    builder.adjust(2, 2, 1)
    
    await callback.message.edit_text(
        "⚙️ <b>AUTOMATION</b>\n\nConfigure automated workflows:",
        reply_markup=builder.as_markup()
    )
    await callback.answer()


@router.callback_query(F.data.startswith("auto:"))
async def cb_auto_action(callback: CallbackQuery, state: FSMContext):
    """Automation action."""
    action = callback.data.split(":")[1]
    
    if action == "assign":
        result = await post_api("/api/v1/automation/process-stale", {})
        await callback.message.edit_text(
            f"🔄 <b>AUTO-ASSIGN</b>\n\n"
            f"Processed: {result.get('total_stale', 0)} leads\n"
            f"Reassigned: {result.get('reassigned', 0)}",
            reply_markup=get_back_to_main()
        )
    elif action == "followup":
        leads = await fetch_api("/api/v1/automation/followup?days=3")
        await callback.message.edit_text(
            f"🔔 <b>FOLLOW-UP</b>\n\n"
            f"Leads needing follow-up: {leads.get('total', 0)}",
            reply_markup=get_back_to_main()
        )
    elif action == "stale":
        leads = await fetch_api("/api/v1/automation/stale?days=7")
        await callback.message.edit_text(
            f"📊 <b>STALE LEADS</b>\n\n"
            f"Stale leads (>7 days): {leads.get('total_stale', 0)}",
            reply_markup=get_back_to_main()
        )
    elif action == "stats":
        stats = await fetch_api("/api/v1/dashboard")
        if "error" not in stats:
            await callback.message.edit_text(
                format_dashboard(stats),
                reply_markup=get_back_to_main()
            )
    
    await callback.answer()


@router.callback_query(F.data == "menu:settings")
async def cb_settings(callback: CallbackQuery, state: FSMContext):
    """Settings callback."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="🔔 Notifications", callback_data="settings:notif"))
    builder.add(InlineKeyboardButton(text="🎯 AI Settings", callback_data="settings:ai"))
    builder.add(InlineKeyboardButton(text="👤 Profile", callback_data="settings:profile"))
    builder.add(InlineKeyboardButton(text="🏠 Back", callback_data="menu:main"))
    builder.adjust(2, 1, 1)
    
    await callback.message.edit_text(
        "⚙️ <b>SETTINGS</b>\n\nConfigure your preferences:",
        reply_markup=builder.as_markup()
    )
    await callback.answer()


@router.callback_query(F.data == "menu:help")
async def cb_help(callback: CallbackQuery, state: FSMContext):
    """Help callback."""
    help_text = """
❓ <b>HELP</b>

<b>Commands:</b>
/start - Start bot
/menu - Main menu
/help - This help

<b>Tips:</b>
• Use inline buttons for quick actions
• AI Agent provides smart recommendations
• Dashboard shows real-time stats
• Filter leads by stage

<b>Support:</b>
Contact @admin for help
"""
    await callback.message.edit_text(
        help_text,
        reply_markup=get_back_to_main()
    )
    await callback.answer()


# ──────────────────────────────────────────────
# Lead Action Callbacks
# ──────────────────────────────────────────────

@router.callback_query(F.data.startswith("lead:"))
async def cb_lead_action(callback: CallbackQuery, state: FSMContext):
    """Handle lead actions."""
    parts = callback.data.split(":")
    action = parts[1] if len(parts) > 1 else None
    lead_id = parts[2] if len(parts) > 2 else None
    
    if action == "analyze" and lead_id:
        await callback.message.edit_text(
            f"🤖 <b>AI ANALYSIS</b>\n\n"
            f"Analyzing lead #{lead_id}...",
            reply_markup=get_back_to_main()
        )
        
        result = await post_api(f"/api/v1/leads/{lead_id}/analyze", {})
        
        if "error" not in result:
            score = result.get("score", 0)
            recommendation = result.get("recommendation", "N/A")
            reason = result.get("reason", "No reason provided")
            
            await callback.message.edit_text(
                f"🤖 <b>AI ANALYSIS RESULT</b>\n\n"
                f"Lead #{lead_id}\n\n"
                f"📊 Score: {score:.2f}\n"
                f"💡 Recommendation: {recommendation}\n"
                f"📝 {reason}",
                reply_markup=get_lead_detail_actions(lead_id)
            )
        else:
            await callback.message.edit_text(
                f"❌ Analysis failed: {result.get('error')}",
                reply_markup=get_back_to_main()
            )
    
    elif action == "contact":
        await callback.message.edit_text(
            f"📞 Contact lead #{lead_id}\n\n"
            f"This will mark lead as contacted.",
            reply_markup=get_back_to_main()
        )
    
    elif action == "qualify":
        await callback.message.edit_text(
            f"✓ Qualifying lead #{lead_id}...",
            reply_markup=get_back_to_main()
        )
    
    elif action == "transfer":
        await callback.message.edit_text(
            f"➡️ Transfer lead #{lead_id}\n\n"
            f"Enter sale amount:",
            reply_markup=get_back_to_main()
        )
    
    elif action == "lost":
        await callback.message.edit_text(
            f"❌ Mark lead #{lead_id} as lost?\n\n"
            f"This action cannot be undone.",
            reply_markup=get_back_to_main()
        )
    
    await callback.answer()


# ──────────────────────────────────────────────
# Voice, Photo, Document Handlers
# ──────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext):
    """Handle voice messages with AI."""
    duration = message.voice.duration
    
    await message.answer(
        f"🎤 <b>Voice Received</b>\n\n"
        f"Duration: {duration}s\n\n"
        f"🎯 <b>AI Processing:</b>\n"
        f"Transcribing and analyzing...",
        reply_markup=get_ai_agent_menu()
    )


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    """Handle photo uploads."""
    photo = message.photo[-1]
    
    await message.answer(
        f"📷 <b>Photo Received</b>\n\n"
        f"Resolution: {photo.width}x{photo.height}\n\n"
        f"Analyze with AI?",
        reply_markup=get_ai_agent_menu()
    )


@router.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    """Handle document uploads."""
    doc = message.document
    size_mb = doc.file_size / (1024 * 1024)
    
    await message.answer(
        f"📄 <b>Document Received</b>\n\n"
        f"File: {doc.file_name}\n"
        f"Size: {size_mb:.2f} MB\n\n"
        f"Process document?",
        reply_markup=get_ai_agent_menu()
    )


# ──────────────────────────────────────────────
# Bot Initialization
# ──────────────────────────────────────────────

def get_agent_bot() -> Bot:
    """Get configured bot."""
    return Bot(
        token=bot_settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML")
    )


def get_agent_dispatcher() -> Dispatcher:
    """Get configured dispatcher."""
    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def start_agent_bot():
    """Start the enhanced agent bot."""
    bot = get_agent_bot()
    dp = get_agent_dispatcher()
    
    logger.info("Starting AI Agent Bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(start_agent_bot())
