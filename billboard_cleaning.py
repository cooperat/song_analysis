import pandas
import json
import os

with open('./data/billboard_extract.json') as f:
    data = json.load(f)
    for index, element in enumerate(data):
        for key, value in element.items():
            if value == []:
                data[index][key] = 'null'
            elif type(value) == list:
                data[index][key] = str(value)[2:-2]
            elif type(value) == str:
                data[index][key] = value.lower()
    with open("./data/billboard_clean.json", "w") as f:
        json.dump(data, f)