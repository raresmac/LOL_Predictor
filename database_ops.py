"""
Legacy compatibility wrapper for data fetching operations.
"""

from data_fetcher import EsportsDataFetcher

_default_fetcher = EsportsDataFetcher()


def response_retrieve(tables: str, fields: str, where: str = "", order_by: str = "", limit: int = 500, offset: int = 0):
    """Legacy query function delegating to EsportsDataFetcher HTTP API."""
    params = {
        "action": "cargoquery",
        "tables": tables,
        "fields": fields,
        "where": where,
        "order_by": order_by,
        "limit": str(limit),
        "offset": str(offset),
        "format": "json"
    }
    try:
        res = _default_fetcher.session.get(_default_fetcher.API_URL, params=params, timeout=_default_fetcher.timeout)
        res.raise_for_status()
        data = res.json()
        cargo_results = data.get("cargoquery", [])
        return [item["title"] for item in cargo_results if "title" in item]
    except Exception:
        return []


def data_retrieve(response):
    """Legacy data parsing function."""
    parsed_records = []
    for record in response:
        parsed = EsportsDataFetcher.parse_match_record(record)
        parsed_records.append([parsed["team1_picks"], parsed["team2_picks"], parsed["winner"] + 1])
    return parsed_records
