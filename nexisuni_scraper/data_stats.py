import os
import json

results_path = '/Users/klaasschoenmaker/capstone-article-data/results/'
downloads_path = '/Users/klaasschoenmaker/Downloads/'

stats = {}

files = []
# r=root, d=directories, f = files

for _, _, files in os.walk(results_path):
    cnt = 0
    result_cnt = len(files)
    not_found_cnt = 0
    downloaded_cnt = 0
    title_no_match_cnt = 0
    valid_result_cnt = 0
    nonvalid_result_cnt = 0
    imparsible_result_cnt = 0
    for result_file_path in files:
        with open(results_path+result_file_path) as result_json:
            try:
                result_data = json.load(result_json)
                if result_data["found"]:
                    download_zip_path = downloads_path + result_data["article"]["id"] + '.ZIP'

                    # Check if article has corresponding download
                    if os.path.exists(download_zip_path):
                        downloaded_cnt += 1
                        if result_data["title_match"]:
                            valid_result_cnt += 1
                        else:
                            title_no_match_cnt += 1

                else:
                    not_found_cnt += 1
            except ValueError as e:
                print("Could not parse result file "+result_file_path+":"+str(e))
                imparsible_result_cnt += 1
    nonvalid_result_cnt = result_cnt - valid_result_cnt
    stats["No. of results"] = result_cnt
    stats["No. of downloaded articles"] = downloaded_cnt
    stats["No. of mismatched titles"] = title_no_match_cnt
    stats["No. of articles not found"] = not_found_cnt
    stats["No. of valid results"] = valid_result_cnt
    stats["No. of invalid results"] = nonvalid_result_cnt
    stats["No. of corrupted results"] = imparsible_result_cnt

for name in stats:
    print(name+": "+str(stats[name]))
