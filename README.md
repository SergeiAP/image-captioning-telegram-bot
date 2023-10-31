# Details
The project from 2020 year
* __Basic algorithms__: __inceptionV3+LSTM/Attention__ (selected in config) pretrained on MS COCO dataset by [author-see google colab](https://drive.google.com/drive/folders/15bWFHG6TTabkFSUTgZmAOOAFEIYENTpt?usp=sharing).
* __Telegram bot__: based on __aiogram library__ (async). __Docker volume is attached for logs, config changes and feedbacks.__
* __Deploy__: on server in __docker containers with standalone chrome browser__ in it
# Commands    
__Command list__ :clipboard:
* /start - Start a dialog :rocket:
* /change - Change number descriptions per photo :pencil:    
__Random related__
* /random - Describe random image :game_die:
* /example - Describe predef image🤹
* /category - Choose category for random photo :pushpin:    
__Utility functions__
* help - Get help⛑
* details - Some app technical details⚙
* feedback - Type your feedback🤗
# Some bot operation examples
See videos [here in Google.Drive](https://drive.google.com/drive/folders/1M5g6UVQ-JAQf4TxNbmukhwilOzbcOsyZ?usp=sharing).    
# Before deploying
It is required change two files in __data__ folder: __handler random__ in __config.py__ depending on where bot is going to be deployed and change __.env__ file. And change imports in __handlers->users->random_click.py__ corresponding to chosen __handler random__ (it was made so to not import extra libraries).   
Also it is required to download models from the folder in [Google.Drive](https://drive.google.com/drive/folders/1N44O-Rt6Gio8R-pdF-y5dCOQTKqVlH5n) to `./functional/data/`
