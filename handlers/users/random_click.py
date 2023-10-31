from handlers.errors.exceptions import RandomImageLimit
from data.config import handlers_random, handlers_categories

import logging
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities # for Docker
# from webdriver_manager.chrome import ChromeDriverManager for local

import io
from PIL import Image
import requests
import hashlib
import os
import time
import random


class RandomImage:

    def __init__(self, sleep_between_interactions, driver_choice):

        self.sleep_between_interactions = sleep_between_interactions
        self.driver_choice = driver_choice

    def load_page(self, driver_choice):
        if driver_choice == 'REMOTE':  # For docker
            return webdriver.Remote(handlers_random['DRIVER_REMOTE'], DesiredCapabilities.CHROME)
        if driver_choice == 'PATH_AUTO': # For local
            return webdriver.Chrome(executable_path=ChromeDriverManager().install())
        if driver_choice == 'PATH':
            return webdriver.Chrome(executable_path=handlers_random['DRIVER_PATH'])

    def scroll_to_end(self, driver):
        # to scroll to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(self.sleep_between_interactions)

    def fetch_image_urls(self, query: str, max_links_to_fetch: int, webdriver: webdriver):
        """
        Function to get urls for all pictures
        :param query:
        :param max_links_to_fetch: max 100
        :param wd:
        :param sleep_between_interactions:
        :return:
        """

        # build the google query
        search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

        webdriver.get(search_url.format(q=query))

        image_urls = set()
        # 1 scroll = 100 images. If you want more - scroll more
        self.scroll_to_end(webdriver)

        # get all image thumbnail results
        thumbnail_results = webdriver.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        # |Image choose limit is 100 (could be extended - see google_selenium.py in image download options)
        if number_results < max_links_to_fetch:
            logging.warning(f"There are chosen more images rhan possible: {number_results} < {max_links_to_fetch}")
            raise RandomImageLimit

        logging.debug(f"Found: {number_results} search results. Extracting links from {0}:{number_results}")
        thumbnail_results = random.sample(thumbnail_results, number_results)
        for img in thumbnail_results:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(self.sleep_between_interactions)
            except Exception:
                continue

            # extract image urls with normal size
            actual_images = webdriver.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images[:max_links_to_fetch]:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))


            if len(image_urls) >= max_links_to_fetch:
                logging.debug(f"Found: {len(image_urls)} image links, done!")
                break

        return image_urls

    def persist_image(self, folder_path, url: str):
        """
        Function to open or save images
        :param folder_path:
        :param url:
        :param is_save:
        :return:
        """
        try:
            image_content = requests.get(url).content

        except Exception as e:
            logging.error(f"ERROR - Could not download {url} - {e}")

        try:
            image = io.BytesIO(image_content)
        except Exception as e:
            logging.error(f"ERROR - Could not open {url} - {e}")

        if folder_path:
            try:
                image = Image.open(image).convert('RGB')
                # to show image
                # image.show()
                # hashlib provides hashing functions
                file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
                with open(file_path, 'wb') as f:
                    image.save(f, "JPEG", quality=85)
                logging.info(f"SUCCESS - saved {url} - as {file_path}")

            except Exception as e:
                logging.error(f"ERROR - Could not save {url} - {e}") \

        return image


    def search_and_download(self, search_term: str, target_path='./images', number_images=5):
        """
        Accumulate two mentioned functions
        :param search_term:
        :param target_path:
        :param number_images:
        :return:
        """
        # load the page
        if search_term == 'random':
            search_term = random.sample(list(handlers_categories['types'].keys())[:-1], 1)[-1]
        search_term = handlers_categories['types'][search_term]['search_tag']

        webdriver = self.load_page(self.driver_choice)

        if target_path:
            target_folder = os.path.join(target_path, '_'.join(search_term.lower().split(' ')))

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
        else:
            target_folder = False

        with webdriver:
            res = self.fetch_image_urls(search_term, number_images, webdriver)

        # list for store images
        images = []
        for elem in res:
            image = self.persist_image(folder_path=target_folder, url=elem)
            images.append(image)

        return search_term, images

sleep_between_interactions = handlers_random['SLEEP_BETWEEN_INTERACTIONS']
driver_choice = handlers_random['DRIVER_CHOICE']
RandomGenerator = RandomImage(sleep_between_interactions, driver_choice)