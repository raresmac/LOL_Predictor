import json

def get_db(championRole='withRole'):
    with open(championRole + '.txt', 'r') as filehandle:
        json_map = json.load(filehandle)
    X = [a["X"] for a in json_map]
    y = [a["y"] for a in json_map]
    return X, y
