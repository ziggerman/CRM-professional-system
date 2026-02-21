"""
Telegram Bot keyboard layouts - With My Leads category.
"""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder


# Main menu
def get_main_menu_keyboard() -> ReplyKeyboardMarkup:
    from aiogram.types import KeyboardButton
    
    builder = ReplyKeyboardBuilder()
    
    builder.add(KeyboardButton(text="Leads"))
    builder.add(KeyboardButton(text="NewLead"))
    builder.add(KeyboardButton(text="Stats"))
    builder.add(KeyboardButton(text="Quick"))
    builder.add(KeyboardButton(text="Settings"))
    
    builder.adjust(2, 2, 1)
    return builder.as_markup(resize_keyboard=True, is_persistent=True)


def get_start_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Leads", callback_data="goto_leads"))
    builder.add(InlineKeyboardButton(text="NewLead", callback_data="goto_newlead"))
    builder.add(InlineKeyboardButton(text="Stats", callback_data="goto_stats"))
    builder.add(InlineKeyboardButton(text="Quick", callback_data="goto_quick"))
    builder.add(InlineKeyboardButton(text="Settings", callback_data="goto_settings"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_start"))
    
    builder.adjust(2, 2, 1, 1)
    return builder.as_markup()


def get_menu_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Leads", callback_data="goto_leads"))
    builder.add(InlineKeyboardButton(text="NewLead", callback_data="goto_newlead"))
    builder.add(InlineKeyboardButton(text="Stats", callback_data="goto_stats"))
    builder.add(InlineKeyboardButton(text="Quick", callback_data="goto_quick"))
    builder.add(InlineKeyboardButton(text="Settings", callback_data="goto_settings"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_start"))
    
    builder.adjust(2, 2, 1, 1)
    return builder.as_markup()


# ===== LEADS CATEGORIES =====
def get_leads_category_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="My Leads", callback_data="filter_myleads"))
    builder.add(InlineKeyboardButton(text="By Stage >", callback_data="cat_stage"))
    builder.add(InlineKeyboardButton(text="By Source >", callback_data="cat_source"))
    builder.add(InlineKeyboardButton(text="By Domain >", callback_data="cat_domain"))
    builder.add(InlineKeyboardButton(text="All Leads", callback_data="filter_all"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_menu"))
    
    builder.adjust(1, 1, 1, 1, 1, 1)
    return builder.as_markup()


def get_stage_subcategories_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="New", callback_data="filter_new"))
    builder.add(InlineKeyboardButton(text="Contacted", callback_data="filter_contacted"))
    builder.add(InlineKeyboardButton(text="Qualified", callback_data="filter_qualified"))
    builder.add(InlineKeyboardButton(text="Transferred", callback_data="filter_transferred"))
    builder.add(InlineKeyboardButton(text="Lost", callback_data="filter_lost"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_leads"))
    
    builder.adjust(1, 1, 1, 1)
    return builder.as_markup()


def get_source_subcategories_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Scanner", callback_data="filter_scanner"))
    builder.add(InlineKeyboardButton(text="Partner", callback_data="filter_partner"))
    builder.add(InlineKeyboardButton(text="Manual", callback_data="filter_manual"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_leads"))
    
    builder.adjust(1, 1, 1)
    return builder.as_markup()


def get_domain_subcategories_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="First", callback_data="filter_first"))
    builder.add(InlineKeyboardButton(text="Second", callback_data="filter_second"))
    builder.add(InlineKeyboardButton(text="Third", callback_data="filter_third"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_leads"))
    
    builder.adjust(1, 1, 1)
    return builder.as_markup()


# Source selection for new lead
def get_source_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Scanner", callback_data="src_scanner"))
    builder.add(InlineKeyboardButton(text="Partner", callback_data="src_partner"))
    builder.add(InlineKeyboardButton(text="Manual", callback_data="src_manual"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_menu"))
    
    builder.adjust(3, 1)
    return builder.as_markup()


# Domain selection for new lead
def get_domain_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="First", callback_data="dmn_first"))
    builder.add(InlineKeyboardButton(text="Second", callback_data="dmn_second"))
    builder.add(InlineKeyboardButton(text="Third", callback_data="dmn_third"))
    builder.add(InlineKeyboardButton(text="Skip", callback_data="dmn_skip"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_newlead"))
    
    builder.adjust(2, 2, 1)
    return builder.as_markup()


# ===== LEAD DETAIL VIEW =====
def get_lead_detail_keyboard(lead_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="[Edit] Stage >", callback_data=f"led{lead_id}_stage"))
    builder.add(InlineKeyboardButton(text="[Edit] Source >", callback_data=f"led{lead_id}_src"))
    builder.add(InlineKeyboardButton(text="[Edit] Domain >", callback_data=f"led{lead_id}_dom"))
    builder.row()
    
    builder.add(InlineKeyboardButton(text="Contacted", callback_data=f"lac{lead_id}_c"))
    builder.add(InlineKeyboardButton(text="Qualified", callback_data=f"lac{lead_id}_q"))
    builder.add(InlineKeyboardButton(text="Transfer", callback_data=f"lac{lead_id}_t"))
    builder.add(InlineKeyboardButton(text="Lost", callback_data=f"lac{lead_id}_l"))
    builder.row()
    
    builder.add(InlineKeyboardButton(text="AI Analyze", callback_data=f"lac{lead_id}_a"))
    builder.add(InlineKeyboardButton(text="Add Note", callback_data=f"led{lead_id}_note"))
    builder.add(InlineKeyboardButton(text="Delete", callback_data=f"led{lead_id}_del"))
    builder.row()
    
    builder.add(InlineKeyboardButton(text="< Back to List", callback_data="goto_leads"))
    
    builder.adjust(3, 4, 3, 1)
    return builder.as_markup()


# Edit Stage keyboard
def get_edit_stage_keyboard(lead_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="New", callback_data=f"eds{lead_id}_new"))
    builder.add(InlineKeyboardButton(text="Contacted", callback_data=f"eds{lead_id}_contacted"))
    builder.add(InlineKeyboardButton(text="Qualified", callback_data=f"eds{lead_id}_qualified"))
    builder.add(InlineKeyboardButton(text="Transferred", callback_data=f"eds{lead_id}_transferred"))
    builder.add(InlineKeyboardButton(text="Lost", callback_data=f"eds{lead_id}_lost"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data=f"lvw{lead_id}"))
    
    builder.adjust(1, 1, 1, 1)
    return builder.as_markup()


# Edit Source keyboard
def get_edit_source_keyboard(lead_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Scanner", callback_data=f"edsrc{lead_id}_scanner"))
    builder.add(InlineKeyboardButton(text="Partner", callback_data=f"edsrc{lead_id}_partner"))
    builder.add(InlineKeyboardButton(text="Manual", callback_data=f"edsrc{lead_id}_manual"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data=f"lvw{lead_id}"))
    
    builder.adjust(3, 1)
    return builder.as_markup()


# Edit Domain keyboard
def get_edit_domain_keyboard(lead_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="First", callback_data=f"eddom{lead_id}_first"))
    builder.add(InlineKeyboardButton(text="Second", callback_data=f"eddom{lead_id}_second"))
    builder.add(InlineKeyboardButton(text="Third", callback_data=f"eddom{lead_id}_third"))
    builder.add(InlineKeyboardButton(text="Remove", callback_data=f"eddom{lead_id}_none"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data=f"lvw{lead_id}"))
    
    builder.adjust(2, 2, 1)
    return builder.as_markup()


# Lead list keyboard
def get_lead_list_keyboard(leads: list, page: int = 0, total_pages: int = 1) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    for lead in leads:
        status = {"new": "New", "contacted": "Cont", "qualified": "Qual", "transferred": "Trans", "lost": "Lost"}.get(lead["stage"], "?")
        builder.add(InlineKeyboardButton(text=f"#{lead['id']} {status}", callback_data=f"lvw{lead['id']}"))
    
    if page > 0 or page < total_pages - 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="< Prev", callback_data=f"pg{page-1}"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="Next >", callback_data=f"pg{page+1}"))
        builder.row(*nav)
    
    builder.add(InlineKeyboardButton(text="< Back to Categories", callback_data="goto_leads"))
    
    builder.adjust(1)
    return builder.as_markup()


# Quick actions keyboard
def get_quick_actions_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Dashboard", callback_data="quick_dashboard"))
    builder.add(InlineKeyboardButton(text="My Leads", callback_data="quick_myleads"))
    builder.add(InlineKeyboardButton(text="Add Lead", callback_data="quick_addlead"))
    builder.add(InlineKeyboardButton(text="Refresh", callback_data="quick_refresh"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_menu"))
    
    builder.adjust(2, 2, 1)
    return builder.as_markup()


# Settings keyboard
def get_settings_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="Leads", callback_data="goto_leads"))
    builder.add(InlineKeyboardButton(text="NewLead", callback_data="goto_newlead"))
    builder.add(InlineKeyboardButton(text="Stats", callback_data="goto_stats"))
    builder.add(InlineKeyboardButton(text="Quick", callback_data="goto_quick"))
    builder.add(InlineKeyboardButton(text="Settings", callback_data="goto_settings"))
    builder.add(InlineKeyboardButton(text="< Back", callback_data="goto_menu"))
    
    builder.adjust(2, 2, 1, 1)
    return builder.as_markup()


# Back keyboard
def get_back_keyboard(callback_data: str = "mnu_back") -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="< Back", callback_data=callback_data))
    return builder.as_markup()


# Confirm delete keyboard
def get_confirm_delete_keyboard(lead_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    
    builder.add(InlineKeyboardButton(text="YES, Delete", callback_data=f"cfddel{lead_id}_y"))
    builder.add(InlineKeyboardButton(text="Cancel", callback_data=f"lvw{lead_id}"))
    
    builder.adjust(2)
    return builder.as_markup()
