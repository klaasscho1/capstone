import json

with open('media_frames_corpus/annotations/immigration.json') as annotations_json:
    article_annotations = json.load(annotations_json)

article_data_list = []

for article_id in article_annotations:
    article_data = {}
    article_data["id"] = article_id
    article_data["title"] = article_annotations[article_id]["title"]
    article_data_list.append(article_data)

try:
    # Get a file object with write permission.
    file_object = open('./mfc_article_data.json', 'w')

    # Save dict data into the JSON file.
    json.dump(article_data_list, file_object)

    print("Success!")
except FileNotFoundError:
    print(file_path + " not found. ")
