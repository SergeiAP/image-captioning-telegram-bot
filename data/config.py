import logging
import logging.config
import traceback
import sys
import os
from pathlib import Path

from dotenv import load_dotenv


# ========================== general dependencies config ==========================
load_dotenv() # download params from .env

APP_BASE_DIR = Path(__file__).resolve().parent
BOT_TOKEN = str(os.getenv("BOT_TOKEN"))


admins = [
          os.getenv("admin_id_1")
]
proxy = None # None or http://144.209.144.206
ip = os.getenv("ip")

aiogram_redis = {
    'host': ip,
}

redis = {
    'address': (ip, 6379),
    'encoding': 'utf8'
}

# ========================== handlers random config ==========================
handlers_random = {
    'SLEEP_BETWEEN_INTERACTIONS': 0.01,
    'DRIVER_CHOICE': 'REMOTE', # REMOTE-for docker, PATH_AUTO-for local, PATH-for local with path to Chrome
    'DRIVER_REMOTE': "http://selenium:4444/wd/hub",
    'GOOGLE_API': # For google API - disabled
        {
            'GCS_DEVELOPER_KEY': str(os.getenv("GCS_DEVELOPER_KEY")),
            'GCS_CX': str(os.getenv("GCS_CX"))
        },
    'DRIVER_PATH': None
}


# ========================== handlers categories config ==========================
# For categories - bottom: [smile, search tag]
handlers_categories = {
    'types' : {
        'transport': {'smile': ':automobile:', 'search_tag': 'photo of transport'},
        'animals': {'smile': ':pig_face:', 'search_tag': 'animals'},
        'food': {'smile': ':pot_of_food:', 'search_tag': 'food photo'},
        'fruits': {'smile': ':peach:', 'search_tag': 'fruits photos'},
        'furniture': {'smile': ':chair:', 'search_tag': 'furniture photo'},
        'appliances': {'smile': ':radio:', 'search_tag': 'household appliance in room photo'},
        'sport': {'smile': ':person_running:', 'search_tag': 'sport in action photo'},
        'ultimate': {'smile': ':flying_disc:', 'search_tag': 'ultimate fresbee in action photo'},
        'random': {'smile': ':game_die:', 'search_tag': 'random'}
    },
    'categories_in_row': 3,
}


# ========================== feedback config ==========================
# Path to save feedback
path_to_feedback = './data/feedback/feedbacks.csv'


# ========================== neural network config ==========================

image_captioning  = {
    'general':
        {
            'MIN_WORD_FREQ': 3,
            'MAX_LEN': 18,
            'MODEL_TYPE': '2_layers_attn', # '1_layer_simple_LSTM', '2_layer_simple_LSTM', '2_layers_attn
            'SEED': 42
         },
    'paths':
        {
            'VOCAB_PATH': r'/data/vocabulary.pkl',
            'WEIGHTS_PATH': r'/data/',
            'INCEPTION_WEIGHTS_PATH': r'/data/inception_v3_google.pth',
            'PICS_EXAMPLES': r'/examples/'
         },
    'model_parameters':
        {
            '1_layer_simple_LSTM':
                {
                    'EMB_DIM': 128,
                    'HIDDEN_SIZE': 256,
                    'DROPOUT': 0.5,
                    'BIDIRECTIONAL': False,
                    'NUM_LAYERS': 1,
                    'CNN_FEATURE_SIZE': 2048
                },
            '2_layer_simple_LSTM':
                {
                    'EMB_DIM': 256,
                    'HIDDEN_SIZE': 512,
                    'DROPOUT': 0.5,
                    'BIDIRECTIONAL': False,
                    'NUM_LAYERS': 2,
                    'CNN_FEATURE_SIZE': 2048
                },
            '2_layers_attn':
                {
                    'EMB_DIM': 128,
                    'HIDDEN_SIZE': 256,
                    'DROPOUT': 0.5,
                    'BIDIRECTIONAL': False,
                    'NUM_LAYERS': 2,
                    'CNN_FEATURE_SIZE': 2048,
                    'ATTENTION_METHOD': 'concat',
                    'SRC_LEN': 64, # number of filters
                    'KERNEL_SIZE': 8,
                    'IN_CHANNELS': 1
                }
         },
    'caption_generator':
        {
            'TOP_K': 3
        }
}


# ========================== log config ==========================
ENVIRONMENT = 'INFO'
PATH_TO_LOGS = APP_BASE_DIR / 'logs' / 'app.log'
PATH_TO_LOGS.parent.mkdir(parents=True, exist_ok=True)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console_logging': {
            'format': '{levelname} {asctime} {filename:s}[LINE:{lineno:d}] {message}',
            'style': '{',
        },
        'verbose_logging': {
            'format': '{levelname} {asctime} {filename:s}[LINE:{lineno:d}] {process:d} {thread:d} {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console_logging',
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler', # log rotating
            'formatter': 'verbose_logging',
            'filename': PATH_TO_LOGS,
            'when': 'midnight',
            'interval': 1,
            'backupCount': 7,
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG' if ENVIRONMENT == 'DEBUG' else 'INFO',
    },
}


logging.config.dictConfig(LOGGING_CONFIG)

# redefine exception hook for logging unhandled exceptions (out of try-except loop)
def __exception_hook(*exc_info):
    msg = "".join(traceback.format_exception(*exc_info))
    logging.error("Unhandled exception: %s", msg)


sys.excepthook = __exception_hook