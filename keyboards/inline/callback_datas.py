from aiogram.utils.callback_data import CallbackData

caption_num_callback = CallbackData("caption_num_choice", "caption_num") # "caption_num_choice:caption_num"
caption_non_num_callback = CallbackData("caption_non_num_callback", "answer")
categories_choice_callback = CallbackData("category", "type", "smile")