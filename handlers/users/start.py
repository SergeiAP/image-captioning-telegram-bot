from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import CommandStart

from loader import dp
from emoji import emojize



@dp.message_handler(CommandStart())
async def bot_start(message: types.Message, state: FSMContext):
    await state.update_data(category='random')
    await message.answer(emojize(f'Hello, _{message.from_user.full_name}_:waving_hand:\n'
                                 'Bot:robot_face: for describing photo for you!\n\n'
                                 '*Just send your photo*:winking_face:'))
    await message.answer_sticker(r'CAACAgIAAxkBAAIHlF8oDN-GX168WZgpsmRqsB5fwrnoAALeCgACLw_wBnZb4ZREhmHKGgQ')
    await message.answer(emojize('Use /example to see descriptions of some prepared photos'
                                 '...or use /random:game_die: to take photo randomly\n\n'
                                 '*To change some defaults:love-you_gesture:*\n'
                                 '/change - Change number descriptions per photo:pencil:\n'
                                 '/category - Choose category for random photo:pushpin:\n\n'
                                 '*Some information commands:nerd_face:*\n'
                                 '/help - Get help:rescue_worker’s_helmet:\n'
                                 '/details - Some app technical details:gear:\n'
                                 '/feedback - Type your feedback:folded_hands:'
                                 ))