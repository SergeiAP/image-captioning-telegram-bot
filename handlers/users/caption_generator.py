from loader import bot, ImageCaptioning
from aiogram.types import Message, InputMediaPhoto
from aiogram.dispatcher import FSMContext
from emoji import emojize
from operator import itemgetter

async def caption_generate(images, data, message: Message, state: FSMContext):
    num_captions = data.get("num_captions")
    if num_captions is None:
        num_captions = 1
        await state.update_data(num_captions=num_captions)
    captions = ImageCaptioning(img=images, num_captions=num_captions)
    # Every time you want to re-read the byte stream, remember to point it back to the beginning
    for image in images:
        image.seek(0)
    # Sort by probability
    captions = sorted(captions, key=itemgetter(1), reverse=True)
    captions = [f'{i} description: _{caption[0]}_ \n  probability: _{caption[1]}_' for i, caption in enumerate(captions, start=1)]
    captions = '\n'.join(captions)

    if len(images) == 1:
        await bot.send_photo(message.from_user.id, images[-1],
                             caption=emojize(f'Booom:collision:\n'
                                             f'{captions}'))
    else: # TODO: add multiple generator
        media = [InputMediaPhoto(image, caption) for image, caption in (images, captions)]
        await bot.send_message(message.from_user.id, emojize(f'Booom:collision:\n'
                                                             f'Click on each photo to reach captions'))
        await bot.send_media_group(message.from_user.id, media)