from aiogram.fsm.state import State, StatesGroup

class UserStates(StatesGroup):
    waiting_symbol = State()
    waiting_interval = State()
    waiting_horizon = State()