from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import logging

from data import config
from functional.frame_capturing_pipeline import FrameCapturingPipeline


# Bot - all requires to Telegram
bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.MARKDOWN, proxy=config.proxy)
# Required for Infinite State Machine
storage = MemoryStorage()
# do - for operate with updates
dp = Dispatcher(bot, storage=storage)
# Upload algorithm
ImageCaptioning = FrameCapturingPipeline()
