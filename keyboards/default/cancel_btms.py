from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

cancel_caption_type = ReplyKeyboardMarkup(
                    keyboard=[
                        [
                            KeyboardButton(text="Cancel"),
                        ]
                    ],
                    resize_keyboard=True
                )