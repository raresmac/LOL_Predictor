"""
Legacy entrypoint delegating to main.py match prediction pipeline.
"""

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline("bot.txt")
