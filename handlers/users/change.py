import logging

from aiogram import types
from aiogram.utils.exceptions import MessageCantBeEdited
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, Text
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove

from keyboards.inline.callback_datas import caption_num_callback, caption_non_num_callback
from keyboards.inline.caption_num_choice_btms import markup_caption_num_choice
from keyboards.default.cancel_btms import cancel_caption_type
from loader import dp
from states.states_classes import ChangeStates
from emoji import emojize


async def cancel_caption_num(message: Message, state: FSMContext):
    await message.answer(emojize('You choose default settings - 1 description to be shown:thumbs_up:'),
                         reply_markup = ReplyKeyboardRemove())
    await state.update_data(num_captions=1)
    logging.info(f"default, num_captions = {1}")
    # remove bottoms
    try:
        await message.edit_reply_markup()
    except MessageCantBeEdited:
        pass
    await state.reset_state(with_data=False)

@dp.message_handler(Command("change"))
async def show_num_caption_choice(message: Message):
    await message.answer(text="Choose one of the description number choice\n"
                              "Or click and type your own",
                         reply_markup=markup_caption_num_choice)


@dp.callback_query_handler(caption_num_callback.filter())
async def ask_caption_num(call: CallbackQuery, callback_data: dict, state: FSMContext):
    # remove clocks after pushing bottom + cache_time to not get updates by bot
    pref = ''
    await call.answer(cache_time=30)
    num_captions = int(callback_data.get("caption_num"))
    await state.update_data(num_captions=num_captions)
    if num_captions != 1:
        pref = 's'
    await call.message.answer(emojize(f'You choose {num_captions} description{pref} to be shown:thumbs_up:'))
    await call.message.edit_reply_markup()
    logging.info(f"call = {call.data}, num_captions = {num_captions}")

@dp.callback_query_handler(caption_non_num_callback.filter(answer="type"))
async def type_caption_num(call: CallbackQuery):
    # remove clocks after pushing bottom + cache_time to not get updates by bot
    await call.message.edit_reply_markup()
    await call.answer("Type only your description number <10", show_alert=True)
    await ChangeStates.type.set()

@dp.message_handler(Text(contains="cancel", ignore_case=True), state=ChangeStates.type)
async def cancel_caption_num_msg(message: Message, state: FSMContext):
    await cancel_caption_num(message, state)

@dp.message_handler(state=ChangeStates.type)
async def handle_type_caption_num(message: types.Message, state: FSMContext):
    num_captions = message.text.strip()
    if num_captions.isdigit():
        num_captions = int(num_captions)
        if (num_captions < 10) and (num_captions > 0):
            await message.answer(emojize(f'You choose {num_captions} description(s) to be shown:thumbs_up:'),
                                 reply_markup = ReplyKeyboardRemove())
            await state.update_data(num_captions=num_captions)
            logging.info(f"type, num_captions = {num_captions}")
            # save data
            await state.reset_state(with_data=False)
        else:
            await message.answer(emojize("Please, type digit between 0 and 10 :angry_face:"),
                                 reply_markup=cancel_caption_type)
            await ChangeStates.type.set()
    else:
        await message.answer(emojize("Please, type digit only :angry_face:"),
                             reply_markup=cancel_caption_type)
        await ChangeStates.type.set()

@dp.callback_query_handler(caption_non_num_callback.filter(answer="cancel"))
async def cancel_caption_num_call(call: CallbackQuery, state: FSMContext):
    message = call.message
    await cancel_caption_num(message, state)