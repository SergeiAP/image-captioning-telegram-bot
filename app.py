from datetime import datetime
import logging

import data.config as cfg
from loader import bot, storage
import os
import csv

async def on_startup(dp):
    import filters
    import middlewares
    filters.setup(dp)
    middlewares.setup(dp)

    from utils.notify_admins import on_startup_notify
    await on_startup_notify(dp)


async def on_shutdown(dp):
    await bot.close()
    await storage.close()


if __name__ == '__main__':
    logging.info('#' * 20 + ' Starting @ ' + str(datetime.utcnow()) + f' in {cfg.ENVIRONMENT} mode ' + '#' * 20)
    from aiogram import executor
    # import dp, which was modified by decorators in all handlers
    from handlers import dp

    if not os.path.exists(cfg.path_to_feedback):
        with open(cfg.path_to_feedback, 'w', newline='') as fw:
            writer = csv.writer(fw, delimiter=',')
            writer.writerow(['date', 'user_id', 'user_full_name', 'feedback'])

    executor.start_polling(dp, on_startup=on_startup, on_shutdown=on_shutdown, skip_updates=True)
