import logging
import os
import io
import random

from loader import dp
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.types import Message
from emoji import emojize

from data.config import image_captioning, handlers_categories

from handlers.users.caption_generator import caption_generate


@dp.message_handler(Command("example"))
async def show_num_caption_choice(message: Message, state: FSMContext):
    await message.answer(text=emojize("Generating description...\n"
                                      "Please, waiting:bomb:"))
    data = await state.get_data()
    search_term = data.get("category") # Choose random category
    if search_term is None:
        search_term = 'random'
    if search_term == 'random':
        search_term = random.sample(list(handlers_categories['types'].keys())[:-1], 1)[-1]
        await state.update_data(category=search_term)

    folder = __file__[:-len('/handlers/users/example.py')] + image_captioning['paths']['PICS_EXAMPLES'] \
             + search_term + '/'
    picts_names = [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
    pict_path = folder + random.sample(picts_names, 1)[-1] # TODO: improve to have many pictures
    with open(pict_path, 'rb') as f:
        image = io.BytesIO(f.read())

    await caption_generate([image], data, message, state)
    logging.info(f"SUCCESS - sent photos for category = {search_term} in example mode")