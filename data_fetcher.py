"""
Data fetcher module for retrieving League of Legends esports statistics via direct HTTP Cargo API.
"""

from typing import List, Dict, Any, Optional
import requests


class EsportsDataFetcher:
    """Manages queries to the Leaguepedia Cargo database via direct REST API requests."""

    API_URL = "https://lol.fandom.com/api.php"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        })

    def fetch_champion_names(self) -> List[str]:
        """Retrieves all registered champion names from Leaguepedia Cargo DB."""
        params = {
            "action": "cargoquery",
            "tables": "Champions=CH",
            "fields": "CH.Name",
            "limit": "500",
            "format": "json"
        }
        try:
            res = self.session.get(self.API_URL, params=params, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            cargo_results = data.get("cargoquery", [])
            champions = [item["title"]["Name"] for item in cargo_results if "title" in item and "Name" in item["title"]]
            if champions:
                return champions
        except Exception:
            pass

        # Offline / Fallback champion list if network or API is unavailable
        return [
            "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Amumu", "Anivia", "Annie", "Aphelios", "Ashe",
            "Aurelion Sol", "Azir", "Bard", "Bel'Veth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn",
            "Camille", "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko",
            "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", "Garen",
            "Gnar", "Gragas", "Graves", "Gwen", "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern",
            "Janna", "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx", "K'Sante", "Kai'Sa", "Kalista", "Karma",
            "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled",
            "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite",
            "Malzahar", "Maokai", "Master Yi", "Milio", "Miss Fortune", "Mordekaiser", "Morgana", "Naafiri",
            "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu & Willump", "Olaf",
            "Orianna", "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus", "Rek'Sai",
            "Rell", "Renata Glasc", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira", "Sejuani",
            "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner",
            "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Tahm Kench", "Taliyah", "Talon", "Taric",
            "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr", "Urgot",
            "Varus", "Vayne", "Veigar", "Vel'Koz", "Vex", "Vi", "Viego", "Viktor", "Vladimir", "Volibear",
            "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", "Yuumi", "Zac",
            "Zed", "Zeri", "Ziggs", "Zilean", "Zoe", "Zyra"
        ]

    def fetch_scoreboard_games(
        self,
        patch: str,
        limit: int = 500,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieves match records for a specified game patch using HTTP requests."""
        params = {
            "action": "cargoquery",
            "tables": "ScoreboardGames",
            "fields": "Team1, Team2, Team1Picks, Team2Picks, Winner",
            "where": f"Patch = '{patch}'",
            "limit": str(limit),
            "offset": str(offset),
            "format": "json"
        }
        try:
            res = self.session.get(self.API_URL, params=params, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            cargo_results = data.get("cargoquery", [])
            return [item["title"] for item in cargo_results if "title" in item]
        except Exception:
            return []

    @staticmethod
    def parse_match_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Parses a single raw API match record into structured pick lists and winner indicator."""
        team1_picks = [p.strip() for p in record.get("Team1Picks", "").split(",") if p.strip()]
        team2_picks = [p.strip() for p in record.get("Team2Picks", "").split(",") if p.strip()]
        
        try:
            winner = int(record.get("Winner", 1)) - 1
        except (ValueError, TypeError):
            winner = 0

        return {
            "team1_picks": team1_picks,
            "team2_picks": team2_picks,
            "winner": winner
        }
