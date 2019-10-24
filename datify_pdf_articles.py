import os
import json
import zipfile
from tika import parser
import pprint
import re

pp = pprint.PrettyPrinter(indent=4)

ARTICLE_PATH_ROOT = '/Users/klaasschoenmaker/capstone-article-data/unzipped_articles/'
DATA_PATH_ROOT = '/Users/klaasschoenmaker/capstone-article-data/article-contents/'
page_count_pattern = re.compile("^Page [0-9]+ of [0-9]+$")

articnt = 0

for root, dirs, _ in os.walk(ARTICLE_PATH_ROOT):
    for dir_name in dirs:
        article_dir = root + dir_name + '/'
        article_id = dir_name

        for article_root, _, files in os.walk(article_dir):
            for file_name in files:
                articnt += 1
                print("Article " + str(articnt) + ": "+file_name)
                if "deliverynotification" in file_name:
                    print("Not an article, skipping..")
                    continue

                article_pdf_path = article_root + file_name

                raw = parser.from_file(article_pdf_path)

                title = raw["metadata"]["title"]

                content = raw["content"]
                content_lines = content.splitlines()

                # Remove empty lines
                #content_lines = [l for l in content_lines if l.strip()]

                # Extract body lines
                #line_no = 0
                body_lines = []
                body_started = False
                body_ended = False
                for line in content_lines:
                    #print(str(line_no)+":"+line)
                    if not body_ended:
                        if not body_started:
                            if line.strip() == "Body":
                                body_started = True
                        else:
                            if line.strip() != "Classification":
                                body_lines.append(line)
                            else:
                                body_ended = True
                    else:
                        break
                    #line_no += 1

                # Filter body lines (e.g. page headers)
                filtered_body_lines = []
                reading_page_header = False
                reading_url_list = False
                header_whitespace_lines = 0
                for line in body_lines:
                    if page_count_pattern.match(line):
                        reading_page_header = True
                        continue

                    if line.startswith("https://"):
                        reading_url_list = True
                        continue

                    if not reading_page_header and not reading_url_list:
                        filtered_body_lines.append(line)
                    else:
                        if reading_page_header:
                            if line.strip() == "":
                                header_whitespace_lines += 1
                                if header_whitespace_lines >= 3:
                                    reading_page_header = False
                                    header_whitespace_lines = 0
                            else:
                                header_whitespace_lines = 0
                        elif reading_url_list:
                            if line.strip() == "":
                                reading_url_list = False

                article_data = {
                    "title": title,
                    "body": os.linesep.join(filtered_body_lines)
                }

                article_data_file_path = DATA_PATH_ROOT + article_id + ".json"

                with open(article_data_file_path, 'w') as outfile:
                    json.dump(article_data, outfile)
