import database_ops as dbops
import sklearn.neural_network as nn
from sklearn.model_selection import train_test_split

def get_champions():
    aux_champions = dbops.response_retrieve(tables="Champions=CH", fields="CH.Name")
    champions = []
    for champ in aux_champions:
        champions.append(champ['Name'])
    return champions
champions = get_champions()

def champ_to_int_withRole(name, team, role):
    global champions
    if name == 'Nunu':
        name = 'Nunu &amp; Willump'
    try:
        pos = champions.index(name)
    except ValueError:
        pos = -1
    if pos == -1:
        return -1
    else:
        return 10 * pos + 5 * team + role

def champ_to_int_withoutRole(name, team):
    global champions
    if name == 'Nunu':
        name = 'Nunu &amp; Willump'
    try:
        pos = champions.index(name)
    except ValueError:
        pos = -1
    if pos == -1:
        return -1
    else:
        return 2 * pos + team

def int_to_champ(value):
    global champions
    name = value // 10
    team = (value // 10) // 5
    role = value % 5
    return champions[name], team, role

def preprocess(patch, offset=0, champion_type='withRole'):
    response = dbops.response_retrieve(
        tables="ScoreboardGames=SG",
        fields="SG.Team1, SG.Team2, SG.Team1Picks, SG.Team2Picks, SG.Winner",
        where="SG.Patch = '" + patch + "'",
        order_by="SG.DateTime_UTC DESC",
        offset=offset
    )
    data = dbops.data_retrieve(response)

    X = []
    y = []
    for row in data:
        new_values = []
        if champion_type == 'top':
            val = champ_to_int_withRole(row[0][0], 0, 0)
            new_values.append(val)
            val = champ_to_int_withRole(row[0][0], 1, 0)
            new_values.append(val)
        elif champion_type == 'jungle':
            val = champ_to_int_withRole(row[0][0], 0, 1)
            new_values.append(val)
            val = champ_to_int_withRole(row[0][0], 1, 1)
            new_values.append(val)
        elif champion_type == 'mid':
            val = champ_to_int_withRole(row[0][0], 0, 2)
            new_values.append(val)
            val = champ_to_int_withRole(row[0][0], 1, 2)
            new_values.append(val)
        elif champion_type == 'bot':
            val = champ_to_int_withRole(row[0][0], 0, 3)
            new_values.append(val)
            val = champ_to_int_withRole(row[0][0], 1, 3)
            new_values.append(val)
        elif champion_type == 'support':
            val = champ_to_int_withRole(row[0][0], 0, 4)
            new_values.append(val)
            val = champ_to_int_withRole(row[0][0], 1, 4)
            new_values.append(val)
        else:
            for i in range(5):
                if champion_type == 'withRole':
                    val = champ_to_int_withRole(row[0][i], 0, i)
                else:
                    val = champ_to_int_withoutRole(row[0][i], 0)
                if val == -1:
                    continue
                new_values.append(val)
            for i in range(5):
                if champion_type == 'withRole':
                    val = champ_to_int_withRole(row[1][i], 1, i)
                else:
                    val = champ_to_int_withoutRole(row[0][i], 0)
                if val == -1:
                    continue
                new_values.append(val)
        X.append(new_values)
        y.append(int(row[2]) - 1)

    return X, y
