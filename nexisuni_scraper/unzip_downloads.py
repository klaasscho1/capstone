import os
import json
import zipfile

RESULTS_PATH = '/Users/klaasschoenmaker/capstone-article-data/results/'
DOWNLOADS_PATH = '/Users/klaasschoenmaker/Downloads/'

def unzip_in_dir(zip_path, name):
    zip_dir = os.path.dirname(os.path.realpath(zip_path))
    dir_path = "/Users/klaasschoenmaker/capstone-article-data/unzipped_articles/" + name

    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(dir_path)
        zip_file.close()

    return dir_path

for _, _, files in os.walk(RESULTS_PATH):
    for result_file_path in files:
        with open(RESULTS_PATH + result_file_path) as result_json:
            try:
                result_data = json.load(result_json)
                if result_data["found"]:
                    download_zip_path = DOWNLOADS_PATH + result_data["article"]["id"] + '.ZIP'
                    if os.path.exists(download_zip_path):
                        tmp_dir_path = unzip_in_dir(download_zip_path, result_data["article"]["id"])

            except ValueError as e:
                print("Could not parse result file "+result_file_path+": "+str(e))
