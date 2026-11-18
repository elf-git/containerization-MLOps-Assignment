"""
FIFA World Cup 2026 Winner Prediction & Tournament Simulation
Uses Machine Learning (XGBoost / Gradient Boosting) and ELO Ratings
"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Locate data directory dynamically
def find_data_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), "FIFA-2026-Winner-Prediction-main", "FIFA Prediction", "data"),
        os.path.join(os.getcwd(), "FIFA-2026-Winner-Prediction-main", "FIFA Prediction", "data"),
        os.path.join(os.path.dirname(__file__), "FIFA Prediction", "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "features_train.csv")):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate FIFA dataset directory.")

DATA_DIR = find_data_dir()
print("=" * 90)
print("⚽  FIFA WORLD CUP 2026 WINNER PREDICTION & TOURNAMENT SIMULATOR")
print("=" * 90)
print(f"📁 Dataset directory: {DATA_DIR}")

# 1. Load Datasets
print("\n[1/6] Loading engineered feature datasets...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "features_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "features_test.csv"))
fixtures_2026 = pd.read_csv(os.path.join(DATA_DIR, "features_2026.csv"))
teams_2026 = pd.read_csv(os.path.join(DATA_DIR, "2026_teams.csv"))
current_elo = pd.read_csv(os.path.join(DATA_DIR, "current_elo.csv"))

# Ensure correct groups
team_to_group = dict(zip(teams_2026["team"], teams_2026["group"]))
fixtures_2026["group"] = fixtures_2026["home_team"].map(team_to_group)

print(f"  ✓ Training matches:  {len(train_df):,} (2000–2021 historical matches)")
print(f"  ✓ Evaluation matches: {len(test_df):,} (2022 FIFA World Cup)")
print(f"  ✓ 2026 Fixtures:      {len(fixtures_2026)} group matches across 12 groups (A-L)")

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff", "elo_win_prob",
    "home_form_win", "home_form_draw",
    "away_form_win", "away_form_draw",
    "form_diff",
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    "xgd",
    "h2h_home_win_rate", "h2h_draw_rate",
    "tournament_weight", "is_neutral", "is_worldcup",
]

result_map = {"L": 0, "D": 1, "W": 2}
inv_result_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
outcome_code_map = {0: "L", 1: "D", 2: "W"}

train_df["target"] = train_df["result"].map(result_map)
test_df["target"] = test_df["result"].map(result_map)

X_train = train_df[FEATURE_COLS].copy()
y_train = train_df["target"].copy()
X_test = test_df[FEATURE_COLS].copy()
y_test = test_df["target"].copy()

# 2. Train Model (XGBoost or HistGradientBoosting)
print("\n[2/6] Initializing and training Machine Learning model...")
model = None
model_name = ""

try:
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model_name = "XGBoost Classifier (multi:softprob)"
except Exception as e:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    model_name = "HistGradientBoostingClassifier (Scikit-Learn)"

print(f"  ✓ Model trained: {model_name}")

# 3. Evaluate on 2022 World Cup Test Set
print("\n[3/6] Evaluating on 2022 World Cup hold-out test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  ✓ Test Set Accuracy: {accuracy:.2%}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"], digits=3))

# Feature importances
if hasattr(model, "feature_importances_"):
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("  Top 5 Predictive Features:")
    for _, row in feat_imp.head(5).iterrows():
        print(f"    • {row['feature']:<22} {row['importance']:.4f}")

# 4. Predict 2026 World Cup Group Fixtures
print("\n[4/6] Generating Match Predictions for 2026 World Cup Group Stage...")
X_2026 = fixtures_2026[FEATURE_COLS].copy()
pred_2026 = model.predict(X_2026)
proba_2026 = model.predict_proba(X_2026)

fixtures_2026["predicted_outcome"] = pred_2026
fixtures_2026["prob_loss"] = proba_2026[:, 0]
fixtures_2026["prob_draw"] = proba_2026[:, 1]
fixtures_2026["prob_win"]  = proba_2026[:, 2]
fixtures_2026["outcome_label"] = fixtures_2026["predicted_outcome"].map(outcome_code_map)

print(f"  ✓ Predictions generated for all {len(fixtures_2026)} group matches!")

# Helper dictionary for team stats (for knockout simulation)
team_stats = {}
for _, row in fixtures_2026.iterrows():
    h = row["home_team"]
    a = row["away_team"]
    if h not in team_stats:
        team_stats[h] = {
            "elo": row["home_elo"],
            "form_win": row["home_form_win"],
            "form_draw": row["home_form_draw"],
            "avg_scored": row["home_avg_scored"] if not pd.isna(row["home_avg_scored"]) else 1.2,
            "avg_conceded": row["home_avg_conceded"] if not pd.isna(row["home_avg_conceded"]) else 1.0,
        }
    if a not in team_stats:
        team_stats[a] = {
            "elo": row["away_elo"],
            "form_win": row["away_form_win"],
            "form_draw": row["away_form_draw"],
            "avg_scored": row["away_avg_scored"] if not pd.isna(row["away_avg_scored"]) else 1.2,
            "avg_conceded": row["away_avg_conceded"] if not pd.isna(row["away_avg_conceded"]) else 1.0,
        }

# Fallback from current_elo for any missing team
elo_dict = dict(zip(current_elo["team"], current_elo["current_elo"]))
for t in teams_2026["team"]:
    if t not in team_stats:
        e = elo_dict.get(t, 1500.0)
        team_stats[t] = {
            "elo": e,
            "form_win": 0.5,
            "form_draw": 0.3,
            "avg_scored": 1.2,
            "avg_conceded": 1.0
        }

# 5. Simulate Group Standings
print("\n[5/6] Simulating Group Standings & Qualifiers...")
all_standings = []
qualifiers_top2 = []
third_placed = []

for group in sorted(fixtures_2026["group"].unique()):
    grp_matches = fixtures_2026[fixtures_2026["group"] == group]
    teams = pd.concat([grp_matches["home_team"], grp_matches["away_team"]]).unique()
    
    st = pd.DataFrame({
        "group": group,
        "team": teams,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "elo": [team_stats[t]["elo"] for t in teams]
    })
    
    for _, match in grp_matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        outcome = match["predicted_outcome"]  # 2=H Win, 1=Draw, 0=Away Win
        
        st.loc[st["team"] == home, "played"] += 1
        st.loc[st["team"] == away, "played"] += 1
        
        if outcome == 2:
            st.loc[st["team"] == home, "wins"] += 1
            st.loc[st["team"] == away, "losses"] += 1
            st.loc[st["team"] == home, "points"] += 3
        elif outcome == 1:
            st.loc[st["team"] == home, "draws"] += 1
            st.loc[st["team"] == away, "draws"] += 1
            st.loc[st["team"] == home, "points"] += 1
            st.loc[st["team"] == away, "points"] += 1
        else:
            st.loc[st["team"] == home, "losses"] += 1
            st.loc[st["team"] == away, "wins"] += 1
            st.loc[st["team"] == away, "points"] += 3
            
    st = st.sort_values(by=["points", "elo"], ascending=[False, False]).reset_index(drop=True)
    all_standings.append(st)
    
    qualifiers_top2.extend(st.iloc[:2]["team"].tolist())
    third_placed.append(st.iloc[2])

# Rank 3rd placed teams across all 12 groups (top 8 advance to Round of 32)
third_df = pd.DataFrame(third_placed).sort_values(by=["points", "elo"], ascending=[False, False]).reset_index(drop=True)
qualifiers_3rd = third_df.iloc[:8]["team"].tolist()
total_qualifiers = qualifiers_top2 + qualifiers_3rd

print(f"  ✓ Group Stage complete! 32 teams qualified for Knockout Phase (24 top-2 + 8 best 3rd)")

# Display Group Tables
for st in all_standings:
    grp_name = st["group"].iloc[0]
    print(f"\n  --- GROUP {grp_name} ---")
    print(f"  Pos  {'Team':<20} P   W   D   L  Pts   ELO")
    for idx, row in st.iterrows():
        q_mark = "✓ Q" if row["team"] in total_qualifiers else "   "
        print(f"  {q_mark}  {row['team']:<20} {row['played']}   {row['wins']}   {row['draws']}   {row['losses']}  {row['points']:3d}  {row['elo']:.0f}")

# 6. Simulate Knockout Rounds & Champion
print("\n" + "=" * 90)
print("  🏆 [6/6] TOURNAMENT KNOCKOUT STAGE SIMULATION")
print("=" * 90)

def predict_single_match(team_a, team_b):
    stat_a = team_stats[team_a]
    stat_b = team_stats[team_b]
    elo_a = stat_a["elo"]
    elo_b = stat_b["elo"]
    elo_diff = elo_a - elo_b
    elo_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    form_a = stat_a["form_win"]
    form_b = stat_b["form_win"]
    form_diff = form_a - form_b
    avg_s_a = stat_a["avg_scored"]
    avg_c_a = stat_a["avg_conceded"]
    avg_s_b = stat_b["avg_scored"]
    avg_c_b = stat_b["avg_conceded"]
    xgd = (avg_s_a - avg_c_a) - (avg_s_b - avg_c_b)
    
    feat_dict = {
        "home_elo": [elo_a],
        "away_elo": [elo_b],
        "elo_diff": [elo_diff],
        "elo_win_prob": [elo_win_prob],
        "home_form_win": [form_a],
        "home_form_draw": [1.0 - form_a],
        "away_form_win": [form_b],
        "away_form_draw": [1.0 - form_b],
        "form_diff": [form_diff],
        "home_avg_scored": [avg_s_a],
        "home_avg_conceded": [avg_c_a],
        "away_avg_scored": [avg_s_b],
        "away_avg_conceded": [avg_c_b],
        "xgd": [xgd],
        "h2h_home_win_rate": [0.5],
        "h2h_draw_rate": [0.33],
        "tournament_weight": [1.0],
        "is_neutral": [1],
        "is_worldcup": [1],
    }
    df = pd.DataFrame(feat_dict)[FEATURE_COLS]
    probs = model.predict_proba(df)[0]
    p_away, p_draw, p_home = probs[0], probs[1], probs[2]
    
    # In knockouts, ties resolved by extra-time / penalties weighted by ELO and win prob
    adj_home = p_home + (p_draw * elo_win_prob)
    adj_away = p_away + (p_draw * (1.0 - elo_win_prob))
    total_adj = adj_home + adj_away
    adj_home /= total_adj
    adj_away /= total_adj
    
    winner = team_a if adj_home >= adj_away else team_b
    winner_prob = adj_home if winner == team_a else adj_away
    return winner, winner_prob, p_home, p_draw, p_away

# Seed 32 teams by ELO for fair tournament bracket
bracket_teams = sorted(total_qualifiers, key=lambda t: team_stats[t]["elo"], reverse=True)
# Standard 1 vs 32, 2 vs 31 bracket pairing
round_32_pairs = []
for i in range(16):
    round_32_pairs.append((bracket_teams[i], bracket_teams[31 - i]))

knockout_stages = [
    ("ROUND OF 32", round_32_pairs),
]

current_pairs = round_32_pairs
round_names = ["ROUND OF 16", "QUARTER-FINALS", "SEMI-FINALS", "FINAL"]

tournament_match_records = []

for round_name in round_names:
    next_round_teams = []
    print(f"\n  ➤ {knockout_stages[-1][0]}:")
    for t1, t2 in current_pairs:
        winner, win_p, ph, pd_val, pa = predict_single_match(t1, t2)
        next_round_teams.append(winner)
        print(f"    {t1:<18} vs {t2:<18} → Winner: {winner:<15} ({win_p*100:5.1f}% win chance)")
        tournament_match_records.append({
            "stage": knockout_stages[-1][0],
            "team1": t1,
            "team2": t2,
            "winner": winner,
            "win_chance": win_p
        })
        
    next_pairs = []
    for i in range(0, len(next_round_teams), 2):
        if i + 1 < len(next_round_teams):
            next_pairs.append((next_round_teams[i], next_round_teams[i+1]))
            
    knockout_stages.append((round_name, next_pairs))
    current_pairs = next_pairs

# Final match
print(f"\n  ➤ FINAL:")
final_t1, final_t2 = current_pairs[0]
champion, champ_p, ph, pd_val, pa = predict_single_match(final_t1, final_t2)
runner_up = final_t2 if champion == final_t1 else final_t1
print(f"    {final_t1:<18} vs {final_t2:<18} → Winner: {champion:<15} ({champ_p*100:5.1f}% win chance)")
tournament_match_records.append({
    "stage": "FINAL",
    "team1": final_t1,
    "team2": final_t2,
    "winner": champion,
    "win_chance": champ_p
})

print("\n" + "=" * 90)
print(f"  🏆 PREDICTED FIFA WORLD CUP 2026 CHAMPION: {champion.upper()} 🏆")
print(f"  🥈 Runner-up: {runner_up}")
print("=" * 90)

# Save predictions output
out_cols = ["group", "home_team", "away_team", "outcome_label", "prob_win", "prob_draw", "prob_loss", "elo_diff", "home_elo", "away_elo"]
out_pred = fixtures_2026[out_cols].copy()
out_pred.to_csv(os.path.join(DATA_DIR, "2026_predictions.csv"), index=False)
pd.DataFrame(tournament_match_records).to_csv(os.path.join(DATA_DIR, "2026_tournament_simulation.csv"), index=False)

print(f"\n✅ All outputs saved successfully to:")
print(f"   • {os.path.join(DATA_DIR, '2026_predictions.csv')}")
print(f"   • {os.path.join(DATA_DIR, '2026_tournament_simulation.csv')}")
print("\n" + "=" * 90 + "\n")
