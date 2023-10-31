from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandHelp

from loader import dp
from emoji import emojize
from utils.misc import rate_limit


@rate_limit(5, 'help')
@dp.message_handler(CommandHelp())
async def bot_help(message: types.Message):
    text = [
        emojize('*Command list*:clipboard:'),
        emojize('/start - Start a dialog:rocket:'),
        emojize('/change - Change number descriptions per photo:pencil:'),
        '\t_Random related_ ',
        emojize('/random - Describe random image:game_die:'),
        emojize('/example - Describe predef image:person_juggling:'),
        emojize('/category - Choose category for random photo:pushpin:'),
        '\t_Utility functions_',
        emojize('/help - Get help:rescue_worker’s_helmet:'),
        emojize('/details - Some app technical details:gear:'),
        emojize('/feedback - Type your feedback:hugging_face:')
    ]
    await message.answer('\n'.join(text))
