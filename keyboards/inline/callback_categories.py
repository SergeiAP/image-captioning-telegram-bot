from data.config import handlers_categories
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from emoji import emojize

from keyboards.inline.callback_datas import categories_choice_callback


inline_keyboard = []
inline_row = []

# Create bottoms for categories with categories_in_row
for key, value in handlers_categories['types'].items():
    inline_row.append(
        InlineKeyboardButton(emojize(f"{key}{value['smile']}", use_aliases=True),
                             callback_data=categories_choice_callback.new(type=key, smile=value['smile'].strip(':')))
    )
    if len(inline_row) == handlers_categories['categories_in_row']:
        inline_keyboard.append(inline_row)
        inline_row = []

markup_categories_choice = InlineKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True,
                                                 inline_keyboard=inline_keyboard)
