from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from emoji import emojize

from keyboards.inline.callback_datas import caption_num_callback, caption_non_num_callback

markup_caption_num_choice = InlineKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True,
    inline_keyboard=[
        [
            InlineKeyboardButton(emojize(':keycap_1::exclamation_mark:', use_aliases=True),
                                 callback_data=caption_num_callback.new(caption_num=1)),
            InlineKeyboardButton(emojize(':keycap_2:', use_aliases=True),
                                 callback_data=caption_num_callback.new(caption_num=2)),
            InlineKeyboardButton(emojize(':keycap_3:', use_aliases=True),
                                 callback_data=caption_num_callback.new(caption_num=3))
        ],
        [
            InlineKeyboardButton(emojize('Type your own number(<:keycap_10:)', use_aliases=True),
                                 callback_data=caption_non_num_callback.new(answer="type"))
        ],
        [
            InlineKeyboardButton(emojize('Cancel', use_aliases=True),
                                 callback_data=caption_non_num_callback.new(answer="cancel"))
        ],

    ]
)
