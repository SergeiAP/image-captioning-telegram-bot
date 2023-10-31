import logging

from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, Text
from aiogram.types import Message, ReplyKeyboardRemove

from keyboards.default.cancel_btms import cancel_caption_type
from states.states_classes import FeedbackType
from loader import dp
import csv
from data.config import path_to_feedback
from emoji import emojize


async def cancel_feedback(message: Message, state: FSMContext):
    await message.answer(emojize('Feedback type canceled:pensive_face:\n'
                                 'But there is hope you will come back to it!'),
                         reply_markup = ReplyKeyboardRemove())
    logging.info(f"Feedback of user (id: {message.from_user.id}, name: {message.from_user.full_name}) canceled")
    await state.reset_state(with_data=False)

@dp.message_handler(Command("feedback"))
async def handle_and_save_feedback(message: Message, state: FSMContext):
    await message.answer(text=emojize("Please, type you feedback:speech_balloon:"), reply_markup=cancel_caption_type)
    await FeedbackType.on.set()

@dp.message_handler(Text(contains="cancel", ignore_case=True), state=FeedbackType.on)
async def cancel_feedback_msg(message: Message, state: FSMContext):
    await cancel_feedback(message, state)

@dp.message_handler(state=FeedbackType.on)
async def cancel_feedback_msg(message: Message, state: FSMContext):
    feedback = message.text.strip()

    with open(path_to_feedback, 'a') as fa:
        writer = csv.writer(fa, delimiter=',')
        writer.writerow([message.date, message.from_user.id, message.from_user.full_name, feedback])

    await message.answer(emojize(f'Thanks _{message.from_user.full_name}_ for your feedback!:thumbs_up:'),
                         reply_markup=ReplyKeyboardRemove())
    logging.info(f"Feedback of user (id: {message.from_user.id}, name: {message.from_user.full_name}) saved")
    await state.reset_state(with_data=False)