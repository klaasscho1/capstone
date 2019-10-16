import os
import time
from urllib import quote

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import WebDriverException

class URLNotFoundException(Exception):
    pass

class NexisDriver:
    QUERY_URL = "https://advance.lexis.com/search/?pdsearchterms=%s"
    LOGIN_URL = "http"

    def __init__(self, download_dir_name="downloads", chrome_driver_path="./chromedriver"):
        self.download_dir = os.path.join(download_dir_name)
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        chrome_options = webdriver.ChromeOptions()
        preferences = {"download.default_directory": self.download_dir}
        chrome_options.add_experimental_option("prefs", preferences)

        self.driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=chrome_options)



    def get_query_article_urls(self, query_string):
        safe_query_string = quote(query_string)
        request_url = self.QUERY_URL % safe_query_string

        self.request(request_url)

    def request(self, url):
        print "Requesting %s" % url
        got_url = False
        for j in range(10):
            try:
                self.driver.get(url)

                got_url = True
                break
            except WebDriverException, e:
                print "Waiting for Chrome to find url"
                time.sleep(2)

        if not got_url:
            print 'Request to %s failed.' % url
            raise URLNotFoundException


