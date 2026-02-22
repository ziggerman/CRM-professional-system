"""
Telegram Bot FSM (Finite State Machine) states.
"""
from aiogram.fsm.state import State, StatesGroup


class LeadCreationState(StatesGroup):
    """States for the new lead creation flow."""
    waiting_for_source = State()
    waiting_for_name = State()
    waiting_for_email = State()
    waiting_for_phone = State()
    waiting_for_username = State()
    waiting_for_domain = State()
    waiting_for_intent = State()
    waiting_for_qualification = State()
    confirm = State()


class AddNoteState(StatesGroup):
    """States for the add-note flow (triggered from lead detail)."""
    waiting_for_text = State()     # user types note text
    waiting_for_confirm = State()  # user confirms or edits


class SearchState(StatesGroup):
    """States for the lead search flow."""
    waiting_for_query = State()  # user types search query


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
