from mwrogue.esports_client import EsportsClient
site = EsportsClient("lol")

def response_retrieve(tables, fields, where="", order_by="", limit=500, offset=0):
    response = site.cargo_client.query(
        tables=tables,
        fields=fields,
        where=where,
        order_by=order_by,
        limit=limit,
        offset=offset
    )
    return response

def data_retrieve(response):
    data = []
    for single in response:
        team1Picks = single['Team1Picks'].split(',')
        team2Picks = single['Team2Picks'].split(',')
        winner = single['Winner']
        data.append([team1Picks, team2Picks, winner])
    return data
