import os
import json
import zipfile

STATUS_VALID = "VALID"

RESULTS_PATH = '/Users/klaasschoenmaker/capstone-article-data/results/'
ARTICLE_CONTENTS_PATH = '/Users/klaasschoenmaker/capstone-article-data/article-contents/'
DATA_PATH = '/Users/klaasschoenmaker/capstone-article-data/data.json'
MFC_ANN_PATH = '/Users/klaasschoenmaker/capstone-article-data/media_frames_corpus/annotations/immigration.json'

with open(MFC_ANN_PATH) as json_file:
    mfc_annotations = json.load(json_file)

unified_data = []

for _, _, files in os.walk(RESULTS_PATH):
    for result_file_path in files:
        unified_article_data = {}
        with open(RESULTS_PATH + result_file_path) as result_json:
            try:
                result_data = json.load(result_json)
                unified_article_data["scrape_result"] = result_data
                if result_data["found"]:
                    article_data_path = ARTICLE_CONTENTS_PATH + result_data["article"]["id"] + '.json'

                    # Check if article has corresponding data
                    if os.path.exists(article_data_path):
                        if result_data["title_match"]:
                            unified_article_data["status"] = "VALID"
                            unified_article_data["mfc_annotation"] = mfc_annotations[result_data["article"]["id"]]
                            with open(ARTICLE_CONTENTS_PATH+result_data["article"]["id"]+".json") as json_file:
                                article_content = json.load(json_file)

                            unified_article_data["content"] = article_content
                        else:
                            unified_article_data["status"] = "INVALID_titlenomatch"
                    else:
                        unified_article_data["status"] = "INVALID_nodata"

                else:
                    unified_article_data["status"] = "INVALID_nosearchresult"
            except ValueError as e:
                unified_article_data["status"] = "INVALID_corruptedresult"
                print("Could not parse result file "+result_file_path+":"+str(e))
            unified_data.append(unified_article_data)

with open(DATA_PATH, 'w') as outfile:
    json.dump(unified_data, outfile)

import pyperclip
pyperclip.copy(json.dumps(unified_data[:5]))
