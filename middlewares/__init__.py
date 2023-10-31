from aiogram import Dispatcher

from .throttling import ThrottlingMiddleware

"""
Module between Update sender and Handlers to filter info, which will be send to Handler
"""

def setup(dp: Dispatcher):
    dp.middleware.setup(ThrottlingMiddleware())
