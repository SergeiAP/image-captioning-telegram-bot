# from icrawler.builtin import BingImageCrawler

bing_crawler = BingImageCrawler(downloader_threads=1)
bing_crawler.crawl(keyword='cat', filters=None, offset=10, max_num=1)