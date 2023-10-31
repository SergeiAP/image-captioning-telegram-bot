import logging

from aiogram import types
from aiogram.utils.exceptions import MessageCantBeEdited
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, Text
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove

from keyboards.inline.callback_datas import categories_choice_callback
from keyboards.inline.callback_categories import markup_categories_choice
from loader import dp, bot
from states.states_classes import CategoryStates
from emoji import emojize

@dp.message_handler(Command("category"))
async def show_categories_choice(message: Message, state: FSMContext):
    rpl_message = await message.answer(text=emojize("Choose one of the category:star-struck:\n"),
                         reply_markup=markup_categories_choice)
    # Save to edit message them
    await state.update_data(chat_id=rpl_message.chat.id, message_id=rpl_message.message_id)
    await CategoryStates.type.set()


@dp.callback_query_handler(categories_choice_callback.filter(), state=CategoryStates.type)
async def ask_category(call: CallbackQuery, callback_data: dict, state: FSMContext):
    await state.reset_state(with_data=False)
    await call.answer(cache_time=30)
    category, smile = callback_data.get("type"), callback_data.get("smile")
    await state.update_data(category=category)
    await call.message.answer(emojize(f"You choose '{category}:{smile}:' category:thumbs_up:"))
    await call.message.edit_reply_markup()
    logging.info(f"call = {call.data}, category = {category}")

@dp.message_handler(content_types=[types.ContentType.ANY], state=CategoryStates.type)
async def state_default(message: Message, state: FSMContext):
    await message.answer(emojize('You choose default settings - random category to be shown:thumbs_up:'),
                         reply_markup = ReplyKeyboardRemove())
    category = "random"
    logging.info(f"default, category = random")
    # remove bottoms
    data = await state.get_data()
    try:
        await bot.edit_message_reply_markup(message_id=data.get("message_id"), chat_id=data.get("chat_id"))
    except Exception as e:
        logging.warning(f"Error while editing previous message: {e}")
    await state.update_data(message_id=None, chat_id=None)
