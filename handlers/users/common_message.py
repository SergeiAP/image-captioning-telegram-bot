from aiogram.dispatcher import FSMContext

from loader import dp
from aiogram import types
from emoji import emojize

@dp.message_handler()
async def common_answer(message: types.Message):
    await message.answer_sticker('CAACAgIAAxkBAAIHaV8n_0_l7hg34h-_62rzA9nCRPyvAAL4CgACLw_wBpl8s8VK3QNlGgQ')
    await message.reply(emojize('I am not ready for conversations :pensive_face:\n'
                                'Use /help to discover my strength! :flexed_biceps:', use_aliases=True))