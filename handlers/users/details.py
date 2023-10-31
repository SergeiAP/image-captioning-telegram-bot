from loader import dp
from aiogram.dispatcher.filters import Command
from aiogram.types import Message
from emoji import emojize

@dp.message_handler(Command("details"))
async def show_num_caption_choice(message: Message):
    await message.answer(text=emojize("-*Basic algorithms*: inceptionV3+LSTM/Attention (selected in config) pretrained on MS COCO dataset by author\n"
                                      "-*Telegram bot*: based on aiogram library. Docker volume is attached for logs, config changes and feedbacks\n"
                                      "-*Deploy*: on server in docker containers with standalone chrome browser in it\n\n"
                                      "Author: @ParshinSA"))