"""
train_model.py
Pulls game + goalie data from Supabase, retrains the Dixon-Coles model,
and saves nhl_skellam_model.json to the repo root.
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Allow importing nhl_model from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nhl_model import NHLPoissonModel, MODEL_PATH

load_dotenv()

DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ['DB_PORT']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']

def main():
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    engine = create_engine(conn_str)
    print("Connected to Supabase")

    print("Loading game data...")
    query_games = """
        SELECT
            "gameId", season, team, "opposingTeam",
            "goalsFor", "goalsAgainst", home_or_away, "gameDate",
            "xGoalsFor", "xGoalsAgainst", "corsiPercentage",
            "shotsOnGoalFor", "shotsOnGoalAgainst",
            "scoreAdjustedShotsAttemptsFor", "scoreAdjustedShotsAttemptsAgainst",
            "highDangerxGoalsFor", "highDangerxGoalsAgainst"
        FROM "NHL_Gamelog"
        WHERE situation = 'all'
        AND position = 'Team Level'
        AND "playoffGame" = '0'
        AND season IN (2024, 2025)
    """
    games_df = pd.read_sql(query_games, engine)
    prior_season   = games_df[games_df['season'] == 2024].copy()
    current_season = games_df[games_df['season'] == 2025].copy()
    print(f"  Prior season   (2024-25): {len(prior_season['gameId'].unique())} games")
    print(f"  Current season (2025-26): {len(current_season['gameId'].unique())} games")

    print("Loading goalie data...")
    query_goalies = """
        SELECT name, team, season, situation, games_played, "xGoals", goals
        FROM "Goalie_stats"
        WHERE season = 2025
    """
    goalie_df = pd.read_sql(query_goalies, engine)
    print(f"  Goalies loaded: {len(goalie_df[goalie_df['situation'] == 'all'])}")

    print("\nTraining model...")
    model = NHLPoissonModel(prior_season, current_season, goalie_df)
    model.save(MODEL_PATH)
    print(f"Done — model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
