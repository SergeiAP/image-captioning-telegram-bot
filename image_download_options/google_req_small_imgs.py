# from loader import dp, bot
# from aiogram import types
# from aiogram.dispatcher.filters import Command
# from aiogram.types import Message
#
# from selenium import webdriver
# from bs4 import BeautifulSoup
# import urllib.request
# from webdriver_manager.chrome import ChromeDriverManager
# import os
# import time

async def downloadimages(download='dogs'):
    # Using Chrome to access web
    site = 'https://www.google.com/search?tbm=isch&q=' + download
    driver = webdriver.Chrome(ChromeDriverManager().install())

    # passing site url
    driver.get(site)

    i=0

    while i < 7:
        # for scrolling page
        driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")

        try:
            # for clicking show more results button
            driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[5]/input").click()
        except Exception as e:
            pass
        i += 1

    # parsing
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # closing web browser
    driver.close()

    # scraping image urls with the help of image tag and class used for images
    img_tags = soup.find_all("img", class_="rg_i")

    count = 0
    for i in img_tags:
        # print(i['src'])
        try:
            # passing image urls one by one and downloading
            urllib.request.urlretrieve(i['src'], str(count) + ".jpg")
            count += 1
            print("Number of images downloaded = " + str(count), end='\r')
        except Exception as e:
            pass

@dp.message_handler(Command("random"))
async def show_num_caption_choice(message: Message):
    await message.answer(text="Generating description...\n"
                              "please, waiting...")
    await downloadimages()
