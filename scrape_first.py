# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:31:39 2018

@author: chongshu
"""


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
import re
import numpy as np
import os
import time
import zipfile
import pandas as pd
from xml.etree.cElementTree import XML
import sys
import json
from selenium.common.exceptions import NoSuchElementException

searchTerms = r'Test'
url = r'http://databases.uba.uva.nl/db/831'
username = "<<UvANet ID>>"
password = "<<UvANet Password>>"
root = r'~/capstone-article-data'
path_to_chromedriver = r'./media_frames_corpus/chromedriver'
download_folder = r'./download'
dead_time = 500

def check_exists_by_xpath(driver, xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True

def strip(str):
    whitelist = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return ''.join(filter(whitelist.__contains__, str))

def enable_download_in_headless_chrome(driver, download_dir):
    # add missing support for chrome "send_command"  to selenium webdriver
    driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')

    params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
    command_result = driver.execute("send_command", params)

def download_file(url = url, searchTerms = searchTerms, username = username, \
                  dead_time = dead_time, path_to_chromedriver=path_to_chromedriver, \
                  download_folder = download_folder):
    while True:
        chromeOptions = webdriver.ChromeOptions()
        prefs = {"download.default_directory" : download_folder}
        chromeOptions.add_experimental_option("prefs", prefs)
        browser = webdriver.Chrome(executable_path = path_to_chromedriver, chrome_options=chromeOptions)
        try:
            browser.set_window_size(1500, 1000)

            browser.get(url)

            # Login if necessary
            if check_exists_by_xpath(browser, '//*[@id="username"]'):
                print("Login necessary, filling in credentials and proceeding")
                browser.find_element_by_xpath('//*[@id="username"]').send_keys(username)
                browser.find_element_by_xpath('//*[@id="password"]').send_keys(password)
                browser.find_element_by_name('submit').click()

            # Get Page Info
            print("Filling in dummy search term")
            browser.find_element_by_xpath('//*[@id="searchTerms"]').send_keys(searchTerms)
            print("Dismissing signin tooltip")
            browser.find_element_by_xpath('//*[@class="primary signintooltip_dismiss"]').click()
            print("Pressing dummy search")
            browser.find_element_by_xpath('//*[@id="mainSearch"]').click()

            with open('mfc_article_data.json') as article_data_json:
                article_data = json.load(article_data_json)

            results = []

            count = 0

            for article in article_data:
                count += 1
                if os.path.exists('results/'+ article["id"] + '.json'):
                    continue

                # Searching for article Title
                art_start_time = time.time()
                result = {}
                result["article"] = article

                print("Article "+str(count)+"/"+str(len(article_data)))
                print("Filling in article title: %s" % article["title"])
                browser.find_element_by_xpath('//*[@id="searchTerms"]').clear()
                browser.find_element_by_xpath('//*[@id="searchTerms"]').send_keys(article["title"])
                print("Pressing search")
                browser.find_element_by_xpath('//*[@id="mainSearch"]').click()

                if check_exists_by_xpath(browser, '//*[@data-id="sr0"]'):
                    print("Result found")
                    result["found"] = True
                else:
                    print("NO result found for this search term, continuing")
                    result["found"] = False
                    results.append(result)
                    try:
                        # Get a file object with write permission.
                        file_object = open('./results/%s.json' % article["id"], 'w')

                        # Save dict data into the JSON file.
                        json.dump(result, file_object)

                        print("Success!")
                    except FileNotFoundError:
                        print("Results file not found. ")
                    continue

                print("Checking first title")

                title_label = browser.find_element_by_xpath('//*[@data-id="sr0"]/div/h2/a')
                first_title = title_label.get_attribute("innerHTML")

                if strip(article["title"]).lower() in strip(first_title).lower():
                    print("Title: MATCH")
                    result["title_match"] = True
                else:
                    print("Title: NO MATCH")
                    print(first_title)
                    print("=/=")
                    print(article["title"])
                    result["title_match"] = False
                    result["title_result"] = first_title

    # =============================================================================
    # Wait the first checkbox to be clickable
    # =============================================================================
                print("Pressing first checkbox")
                time.sleep(2)
                start_time = time.time()
                while True:
                    if time.time() - start_time > dead_time:
                        raise Exception()
                    try:
                        if browser.find_element_by_xpath('//*[@data-id="sr0"]/label/input').get_attribute('checked') != 'true':
                            browser.find_element_by_xpath('//*[@data-id="sr0"]/label/input').click()
                        break
                    except WebDriverException:
                        pass

                print("Waiting for download button, then pressing")

                try:
                    start_time = time.time()
                    WebDriverWait(browser, dead_time).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="select more"]')))
                    browser.find_element_by_xpath('//*[@data-action="downloadopt"]').click()
                except WebDriverException:
                    pass

                start_time = time.time()
                print("Selecting preferences")
                while True:
                    if time.time() - start_time > dead_time:
                        raise Exception()

                    try:
                        browser.find_element_by_xpath('//*[@id="DocumentsOnly"]').click()
                        browser.find_element_by_xpath('//*[@id="IncludeAttachments"]').click()
                        break
                    except WebDriverException:
                        pass
                browser.find_element_by_xpath('//*[@id="Pdf"]').click()
                browser.find_element_by_xpath('//*[@id="SeparateFiles"]').click()
                browser.find_element_by_xpath('//*[@id="FileName"]').clear()
                browser.find_element_by_xpath('//*[@id="FileName"]').send_keys(article["id"])
    # =============================================================================
    #     After downloading Close the pop up window. LexisNexis only allows 5 cocurrent windows
    # =============================================================================
                print("After download finishes, closing popup")
                before = browser.window_handles[0]
                browser.find_element_by_xpath('//*[@class="button primary"]').click()
                start_time = time.time()
                while True:
                    if time.time() - start_time > dead_time:
                        raise Exception()
                    try:
                        after = browser.window_handles[1]
                        break
                    except:
                        pass
                browser.switch_to.window(after)
                start_time = time.time()
                while True:
                    if time.time() - start_time > dead_time:
                        raise Exception()
                    if  browser.find_elements_by_link_text(article["id"]):
                        break
                browser.close()
                browser.switch_to.window(before)
                result["downloaded"] = True
                print('Downloaded article (' +article["id"]+ '): ' + article["title"])
                print("Duration: %i sec" % (time.time() - art_start_time))
                try:
                    # Get a file object with write permission.
                    file_object = open('./results/%s.json' % article["id"], 'w')

                    # Save dict data into the JSON file.
                    json.dump(result, file_object)

                    print("Success!")
                except FileNotFoundError:
                    print("Results file not found. ")

                results.append(result)

            try:
                file_object = open('./results.json', 'w')
                json.dump(results, file_object)

                print("Success!")
            except FileNotFoundError:
                print("Could not save results. Dump:")
                print(results)
        except:
            print("Scraper crashed, restarting")
            browser.quit()
            continue


def unzip(download_folder=download_folder):
    if not os.path.exists(download_folder + '\\' + 'unzipped'):
        os.makedirs(download_folder + '\\' + 'unzipped')
    for filename in os.listdir(download_folder):
        if not filename.endswith('.ZIP'):
            continue
        zip_ref = zipfile.ZipFile(download_folder +  '\\' + filename, 'r')
        zip_ref.extractall(download_folder + '\\' + 'unzipped')
        if len(zip_ref.namelist()) != 11:
            print('missing document at ' + filename)
        zip_ref.close()
        print('unzipping ' + filename)


download_file()
