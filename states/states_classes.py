from aiogram.dispatcher.filters.state import StatesGroup, State

class ChangeStates(StatesGroup):
    type = State()

class CategoryStates(StatesGroup):
    type = State()

class FeedbackType(StatesGroup):
    on = State()