import preprocessor as lolp
import json
import time

def bigger_db(championRole='withRole'):
    X = []
    y = []
    for p in range(8, 15):
        for j in range(5):
            text = str(p)
            new_x, new_y = lolp.preprocess(text, offset=j * 500, champion_type=championRole)
            X += new_x
            y += new_y
        print(str(p))
        time.sleep(1)
        for i in range(1, 26):
            text = str(p) + '.' + str(i)
            for j in range(50):
                new_x, new_y = lolp.preprocess(text, offset=j*500, champion_type=championRole)
                if len(new_x) == 0:
                    break
                X += new_x
                y += new_y
            print(text)
            time.sleep(1)
    return X, y

def write_db(championRole='withRole'):
    X, y = bigger_db(championRole)
    new_list = [{'X': x_item, 'y': y_item} for x_item, y_item in zip(X, y)]
    with open(championRole + '.txt', 'w') as filehandle:
        json.dump(new_list, filehandle)
    return X, y

# write_db()
# write_db('withoutRole')
# write_db('top')
# write_db('jungle')
# write_db('mid')
write_db('bot')
# write_db('support')
