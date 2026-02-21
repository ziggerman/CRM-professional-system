"""
Telegram Bot FSM (Finite State Machine) states.
"""
from aiogram.fsm.state import State, StatesGroup


class LeadCreationState(StatesGroup):
    """States for lead creation flow."""
    waiting_for_source = State()
    waiting_for_domain = State()
    waiting_for_telegram_id = State()
    confirm = State()


class LeadManagementState(StatesGroup):
    """States for lead management."""
    viewing_leads = State()
    viewing_single_lead = State()
    confirming_action = State()


class SaleManagementState(StatesGroup):
    """States for sale management."""
    viewing_sales = State()
    viewing_single_sale = State()
    updating_notes = State()
    confirming_action = State()


class AdminState(StatesGroup):
    """States for admin actions."""
    main_menu = State()
    settings = State()
    statistics = State()


# State storage key prefixes
LEAD_CREATION_PREFIX = "lead_create"
LEAD_VIEW_PREFIX = "lead_view"
SALE_VIEW_PREFIX = "sale_view"
