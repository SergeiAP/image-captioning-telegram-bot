# import logging
# import time
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# import os
# import io
# from PIL import Image
# import requests
# import hashlib
# import random

SLEEP_BETWEEN_INTERACTIONS = 0.1
SlEEP_FOR_LOOKING_MORE = 1
DRIVER_PATH = ChromeDriverManager().install()

DRIVER = webdriver.Chrome(executable_path=DRIVER_PATH)

def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: int = SLEEP_BETWEEN_INTERACTIONS):
    """
    Function to get urls for all pictures
    :param query:
    :param max_links_to_fetch:
    :param wd:
    :param sleep_between_interactions:
    :return:
    """
    def scroll_to_end(wd):
        # to scroll to the bottom of the page
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        # 1 scroll = 100 images. If you want more - scroll more
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        logging.info(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls with normal size
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                logging.info(f"Found: {len(image_urls)} image links, done!")
                break
            else:
                logging.info("Found:", len(image_urls), "image links, looking for more ...")
                time.sleep(SlEEP_FOR_LOOKING_MORE)
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path, url: str):
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
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        # to show image
        image.show()
    except Exception as e:
        logging.error(f"ERROR - Could not open {url} - {e}")

    if folder_path:
        try:
            # hashlib provides hashing functions
            file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=85)
            logging.info(f"SUCCESS - saved {url} - as {file_path}")

        except Exception as e:
            logging.error(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_term: str, target_path='./images', number_images=5, driver=DRIVER):
    """
    Accumulate two mentioned functions
    :param search_term: 
    :param target_path: 
    :param number_images: 
    :return: 
    """
    if target_path:
        target_folder = os.path.join(target_path, '_'.join(search_term.lower().split(' ')))

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
    else:
        target_folder = False

    with driver as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=SLEEP_BETWEEN_INTERACTIONS)

    for elem in res:
        persist_image(folder_path=target_folder, url=elem,)

search_term = 'dog'
search_and_download(search_term, target_path=False, number_images=1, driver=DRIVER)