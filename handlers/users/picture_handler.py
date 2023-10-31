import logging

from aiogram.dispatcher import FSMContext

from loader import dp, bot
from aiogram import types
from emoji import emojize
from .random_click import RandomGenerator
from aiogram.dispatcher.filters import Command
from aiogram.types import Message
from handlers.errors.exceptions import NoPhotoInMessage
from aiogram.utils.exceptions import BadRequest
from .caption_generator import caption_generate


@dp.message_handler(content_types=['document', 'photo'])
async def receive_docs_photo(message: types.Message, state: FSMContext):
    data = await state.get_data()
    try:
        if message.content_type == 'document':
            photo_id = message.document.file_id
        elif message.content_type == 'photo':
            photo_id = message.photo[-1].file_id
        else:
            raise NoPhotoInMessage
        photo_telergam = await bot.get_file(photo_id)
        file_photo = await bot.download_file(photo_telergam.file_path) # TODO: handle several photos
        await message.reply(emojize('Photo is received:OK_hand:\nPlease, waiting:bomb:', use_aliases=True))
        logging.info(f"Picture was converted successfully")
        await caption_generate([file_photo], data, message, state)
        logging.info(f"SUCCESS - sent photos in custom mode")
    except OSError as e:
        logging.warning(f"OSError: {e}")
        await message.reply(emojize("There is no photo :slightly_frowning_face:\n"
                                        "Please, try again or use 'sent media' :camera:!",
                                        use_aliases=True))
    except NoPhotoInMessage as e:
        logging.warning(f"NoPhotoInMessage: {e}")
        await message.reply(emojize('There is no photo :slightly_frowning_face:\n'
                                    'Please, try again!',
                                    use_aliases=True))
    except BadRequest as e:
        logging.warning(f"BadRequest: {e}")
        await message.reply(emojize("File is too big :face_with_open_mouth:\n"
                                        "Please, try again!",
                                        use_aliases=True))

@dp.message_handler(Command("random"))
async def show_num_caption_choice(message: Message, state: FSMContext):
    await message.answer(text=emojize("Generating description...\n"
                                      "Please, waiting:bomb:"))
    data = await state.get_data()
    search_term = data.get("category") # Choose random category
    if search_term is None:
        search_term = 'random'
        await state.update_data(category=search_term)
    search_term, images = RandomGenerator.search_and_download(search_term, target_path=False, number_images=1) # TODO: add number images and possible to save AND add async
    await caption_generate(images, data, message, state)
    logging.info(f"SUCCESS - sent photos for category = {search_term} in random mode")