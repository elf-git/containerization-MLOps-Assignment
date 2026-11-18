import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with Project Group & Team Members
with st.sidebar:
    st.header("👥 Project Team")
    st.markdown("""
**FIFA World Cup 2026 Prediction System**

| Team Member | ID |
|---|:---:|
| **Anurag Chowdhury** | `2548512` |
| **Elvis Vidit Fernandiz** | `2548520` |
| **Abhinash K Prabhu** | `2548528` |
| **Sriram K** | `2548557` |
""")
    st.divider()
    st.markdown("### ⚙️ Engine Info")
    st.markdown("""
- **Algorithm:** XGBoost (`multi:softprob`)
- **Evaluation:** 2022 World Cup Test Set
- **Runtime:** Dockerized Python 3.10
""")

# Dynamic data directory discovery
@st.cache_resource
def get_data_dir():
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

DATA_DIR = get_data_dir()

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

@st.cache_data
def load_all_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "features_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "features_test.csv"))
    fixtures_2026 = pd.read_csv(os.path.join(DATA_DIR, "features_2026.csv"))
    teams_2026 = pd.read_csv(os.path.join(DATA_DIR, "2026_teams.csv"))
    current_elo = pd.read_csv(os.path.join(DATA_DIR, "current_elo.csv"))
    
    # Fix group mapping if needed
    t_map = dict(zip(teams_2026["team"], teams_2026["group"]))
    fixtures_2026["group"] = fixtures_2026["home_team"].map(t_map)
    
    return train_df, test_df, fixtures_2026, teams_2026, current_elo

@st.cache_resource
def train_prediction_model(train_df, test_df):
    train_df["target"] = train_df["result"].map(result_map)
    test_df["target"] = test_df["result"].map(result_map)
    
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["target"].copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["target"].copy()
    
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        m_name = "XGBoost Classifier"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        clf.fit(X_train, y_train)
        m_name = "HistGradientBoosting Classifier"
        
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    return clf, m_name, acc

# Load data and model
train_df, test_df, fixtures_2026, teams_2026, current_elo = load_all_data()
model, model_name, test_acc = train_prediction_model(train_df, test_df)

# Build team statistics lookup
team_stats = {}
for _, row in fixtures_2026.iterrows():
    h, a = row["home_team"], row["away_team"]
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

# Generate 2026 match predictions
X_2026 = fixtures_2026[FEATURE_COLS].copy()
proba_2026 = model.predict_proba(X_2026)
pred_2026 = model.predict(X_2026)
fixtures_2026["predicted_outcome"] = pred_2026
fixtures_2026["prob_loss"] = proba_2026[:, 0]
fixtures_2026["prob_draw"] = proba_2026[:, 1]
fixtures_2026["prob_win"]  = proba_2026[:, 2]
outcome_code_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
fixtures_2026["outcome_label"] = fixtures_2026["predicted_outcome"].map(outcome_code_map)

# Simulation function for single match with memoization
_match_cache = {}

def predict_match(t1, t2):
    cache_key = (t1, t2)
    if cache_key in _match_cache:
        return _match_cache[cache_key]
        
    s1, s2 = team_stats[t1], team_stats[t2]
    e1, e2 = s1["elo"], s2["elo"]
    e_diff = e1 - e2
    e_prob = 1.0 / (1.0 + 10.0 ** (-e_diff / 400.0))
    f1, f2 = s1["form_win"], s2["form_win"]
    f_diff = f1 - f2
    as1, ac1 = s1["avg_scored"], s1["avg_conceded"]
    as2, ac2 = s2["avg_scored"], s2["avg_conceded"]
    xgd = (as1 - ac1) - (as2 - ac2)
    
    row = {
        "home_elo": [e1], "away_elo": [e2], "elo_diff": [e_diff], "elo_win_prob": [e_prob],
        "home_form_win": [f1], "home_form_draw": [1.0 - f1], "away_form_win": [f2], "away_form_draw": [1.0 - f2],
        "form_diff": [f_diff], "home_avg_scored": [as1], "home_avg_conceded": [ac1],
        "away_avg_scored": [as2], "away_avg_conceded": [ac2], "xgd": [xgd],
        "h2h_home_win_rate": [0.5], "h2h_draw_rate": [0.33], "tournament_weight": [1.0],
        "is_neutral": [1], "is_worldcup": [1]
    }
    probs = model.predict_proba(pd.DataFrame(row)[FEATURE_COLS])[0]
    p_away, p_draw, p_home = probs[0], probs[1], probs[2]
    
    # Tie-break extra-time/penalty adjustment for knockout
    adj_home = p_home + (p_draw * e_prob)
    adj_away = p_away + (p_draw * (1.0 - e_prob))
    tot = adj_home + adj_away
    adj_home /= tot
    adj_away /= tot
    
    winner = t1 if adj_home >= adj_away else t2
    winner_prob = adj_home if winner == t1 else adj_away
    res = (winner, winner_prob, p_home, p_draw, p_away)
    _match_cache[cache_key] = res
    return res

# UI Layout
st.title("⚽ FIFA World Cup 2026 Winner Prediction")
st.markdown("Predict match outcomes, simulate group stages, and forecast the 2026 World Cup champion using Machine Learning and dynamic ELO ratings.")

# Summary Metrics Bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Teams", "48 Teams")
with col2:
    st.metric("Group Matches", "72 Matches")
with col3:
    st.metric("ML Model", model_name.split()[0])
with col4:
    st.metric("2022 WC Accuracy", f"{test_acc:.1%}")
with col5:
    top_elo_team = max(team_stats.items(), key=lambda x: x[1]["elo"])[0]
    top_elo_val = team_stats[top_elo_team]["elo"]
    st.metric("#1 Ranked Team", top_elo_team, f"ELO {top_elo_val:.0f}")

st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Tournament Simulation & Winner",
    "📊 Group Stage Standings",
    "⚔️ Head-to-Head Match Predictor",
    "📈 Model Performance & ELO Ratings"
])

# TAB 1: Tournament Simulation
with tab1:
    st.header("🏆 FIFA World Cup 2026 Knockout Bracket Simulation")
    st.markdown("Simulating all 12 groups, qualifying 32 teams (top 2 per group + 8 best 3rd placed), and playing out the knockout bracket to crown the champion.")
    
    # Run Group simulation
    qualifiers_top2 = []
    third_placed = []
    
    for grp in sorted(fixtures_2026["group"].unique()):
        g_matches = fixtures_2026[fixtures_2026["group"] == grp]
        teams = pd.concat([g_matches["home_team"], g_matches["away_team"]]).unique()
        st_df = pd.DataFrame({"team": teams, "points": 0, "elo": [team_stats[t]["elo"] for t in teams]})
        for _, m in g_matches.iterrows():
            out = m["predicted_outcome"]
            if out == 2:
                st_df.loc[st_df["team"] == m["home_team"], "points"] += 3
            elif out == 1:
                st_df.loc[st_df["team"] == m["home_team"], "points"] += 1
                st_df.loc[st_df["team"] == m["away_team"], "points"] += 1
            else:
                st_df.loc[st_df["team"] == m["away_team"], "points"] += 3
        st_df = st_df.sort_values(by=["points", "elo"], ascending=[False, False]).reset_index(drop=True)
        qualifiers_top2.extend(st_df.iloc[:2]["team"].tolist())
        third_placed.append(st_df.iloc[2])
        
    third_df = pd.DataFrame(third_placed).sort_values(by=["points", "elo"], ascending=[False, False]).reset_index(drop=True)
    qualifiers_3rd = third_df.iloc[:8]["team"].tolist()
    total_32 = qualifiers_top2 + qualifiers_3rd
    
    # Seed by ELO
    bracket_teams = sorted(total_32, key=lambda t: team_stats[t]["elo"], reverse=True)
    r32_pairs = [(bracket_teams[i], bracket_teams[31 - i]) for i in range(16)]
    
    # Play rounds
    stages = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]
    current_pairs = r32_pairs
    round_results = {}
    
    for stage in stages:
        winners = []
        matches_info = []
        for t1, t2 in current_pairs:
            win, win_p, ph, pd_val, pa = predict_match(t1, t2)
            winners.append(win)
            matches_info.append((t1, t2, win, win_p, ph, pd_val, pa))
        round_results[stage] = matches_info
        
        # Pair up next round
        next_pairs = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                next_pairs.append((winners[i], winners[i+1]))
        current_pairs = next_pairs

    champion = round_results["Final"][0][2]
    champ_match = round_results["Final"][0]
    runner_up = champ_match[0] if champion == champ_match[1] else champ_match[1]
    
    # Champion Banner
    st.success(f"### 🏆 Predicted Champion: **{champion.upper()}** (Defeated {runner_up} in the Final with {champ_match[3]*100:.1f}% win probability)")
    
    # Interactive Stage Viewer
    stage_choice = st.radio("Select Knockout Stage to View:", stages, horizontal=True)
    matches_to_show = round_results[stage_choice]
    
    cols = st.columns(2)
    for idx, (t1, t2, win, win_p, ph, pd_val, pa) in enumerate(matches_to_show):
        with cols[idx % 2]:
            st.markdown(f"""
            <div style="padding:12px; border-radius:8px; border:1px solid #444; margin-bottom:12px; background-color: rgba(255,255,255,0.03);">
                <div style="display:flex; justify-content:space-between; font-weight:bold; font-size:16px;">
                    <span>{'🏅 ' + t1 if win == t1 else t1}</span>
                    <span style="color:#888;">vs</span>
                    <span>{'🏅 ' + t2 if win == t2 else t2}</span>
                </div>
                <div style="font-size:13px; color:#aaa; margin-top:4px;">
                    ELO: {t1} ({team_stats[t1]['elo']:.0f}) vs {t2} ({team_stats[t2]['elo']:.0f})
                </div>
                <div style="margin-top:6px; font-size:14px; color: #4CAF50; font-weight:bold;">
                    Winner: {win} ({win_p*100:.1f}% win chance)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()
    
    # Monte Carlo Forecasting
    st.subheader("🎲 Monte Carlo Tournament Simulator (Title Probability %)")
    num_sims = st.slider("Select number of tournament simulations:", min_value=100, max_value=2000, value=500, step=100)
    
    if st.button("🚀 Run Monte Carlo Simulation"):
        with st.spinner(f"Simulating {num_sims} World Cups..."):
            titles = {t: 0 for t in total_32}
            np.random.seed(42)
            for _ in range(num_sims):
                current_pairs = list(r32_pairs)
                # 4 rounds to get to Final (R32 -> R16 -> QF -> SF)
                for _ in range(4):
                    winners = []
                    for t1, t2 in current_pairs:
                        _, win_p, _, _, _ = predict_match(t1, t2)
                        winner = t1 if np.random.rand() < win_p else t2
                        winners.append(winner)
                    current_pairs = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
                
                # Final match
                t1, t2 = current_pairs[0]
                _, win_p, _, _, _ = predict_match(t1, t2)
                champion = t1 if np.random.rand() < win_p else t2
                titles[champion] += 1
                
            sim_df = pd.DataFrame([
                {"Team": t, "ELO": round(team_stats[t]["elo"], 1), "Championships": count, "Win Rate (%)": round((count / num_sims) * 100, 1)}
                for t, count in titles.items() if count > 0
            ]).sort_values("Championships", ascending=False).reset_index(drop=True)
            
            top_winner = sim_df.iloc[0]
            st.success(f"Simulation completed! Most frequent Champion: **{top_winner['Team']}** ({top_winner['Win Rate (%)']}% win rate across {num_sims} simulations)")
            st.bar_chart(sim_df.set_index("Team")["Win Rate (%)"])
            st.dataframe(sim_df, use_container_width=True)

# TAB 2: Group Stage Standings
with tab2:
    st.header("📊 Group Stage Predictions & Tables")
    selected_group = st.selectbox("Select World Cup Group:", sorted(fixtures_2026["group"].unique()))
    
    grp_matches = fixtures_2026[fixtures_2026["group"] == selected_group].copy()
    
    # Compute table
    teams = pd.concat([grp_matches["home_team"], grp_matches["away_team"]]).unique()
    standings = pd.DataFrame({
        "Team": teams,
        "P": 0, "W": 0, "D": 0, "L": 0, "Pts": 0,
        "ELO": [team_stats[t]["elo"] for t in teams]
    })
    
    for _, match in grp_matches.iterrows():
        h, a = match["home_team"], match["away_team"]
        out = match["predicted_outcome"]
        standings.loc[standings["Team"] == h, "P"] += 1
        standings.loc[standings["Team"] == a, "P"] += 1
        if out == 2:
            standings.loc[standings["Team"] == h, "W"] += 1
            standings.loc[standings["Team"] == a, "L"] += 1
            standings.loc[standings["Team"] == h, "Pts"] += 3
        elif out == 1:
            standings.loc[standings["Team"] == h, "D"] += 1
            standings.loc[standings["Team"] == a, "D"] += 1
            standings.loc[standings["Team"] == h, "Pts"] += 1
            standings.loc[standings["Team"] == a, "Pts"] += 1
        else:
            standings.loc[standings["Team"] == h, "L"] += 1
            standings.loc[standings["Team"] == a, "W"] += 1
            standings.loc[standings["Team"] == a, "Pts"] += 3
            
    standings = standings.sort_values(by=["Pts", "ELO"], ascending=[False, False]).reset_index(drop=True)
    standings["Status"] = ["✓ Qualified (Top 2)" if i < 2 else ("Potential 3rd Qualifier" if i == 2 else "Eliminated") for i in range(len(standings))]
    
    st.subheader(f"Group {selected_group} Standings")
    st.dataframe(standings, use_container_width=True)
    
    st.subheader(f"Group {selected_group} Match Predictions")
    disp_cols = ["home_team", "away_team", "outcome_label", "prob_win", "prob_draw", "prob_loss", "elo_diff"]
    m_disp = grp_matches[disp_cols].copy()
    m_disp.columns = ["Home Team", "Away Team", "Prediction", "Home Win %", "Draw %", "Away Win %", "ELO Diff"]
    m_disp["Home Win %"] = (m_disp["Home Win %"] * 100).map("{:.1f}%".format)
    m_disp["Draw %"] = (m_disp["Draw %"] * 100).map("{:.1f}%".format)
    m_disp["Away Win %"] = (m_disp["Away Win %"] * 100).map("{:.1f}%".format)
    m_disp["ELO Diff"] = m_disp["ELO Diff"].map("{:+.0f}".format)
    st.dataframe(m_disp, use_container_width=True)

# TAB 3: Head to Head Predictor
with tab3:
    st.header("⚔️ Custom Head-to-Head Match Simulator")
    st.markdown("Select any two international football teams to run live ML predictions.")
    
    all_teams = sorted(list(team_stats.keys()))
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Select Team A (Home/Neutral):", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
    with col_b:
        team_b = st.selectbox("Select Team B (Away/Neutral):", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)
        
    if team_a == team_b:
        st.warning("Please choose two different teams.")
    else:
        win, win_p, ph, pd_val, pa = predict_match(team_a, team_b)
        
        st.subheader("Match Outcome Probability Breakdown")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.metric(f"{team_a} Win", f"{ph*100:.1f}%", f"ELO: {team_stats[team_a]['elo']:.0f}")
        with p_col2:
            st.metric("Draw", f"{pd_val*100:.1f}%")
        with p_col3:
            st.metric(f"{team_b} Win", f"{pa*100:.1f}%", f"ELO: {team_stats[team_b]['elo']:.0f}")
            
        st.progress(float(ph), text=f"{team_a} Win: {ph*100:.1f}% | Draw: {pd_val*100:.1f}% | {team_b} Win: {pa*100:.1f}%")
        
        st.info(f"🏆 Predicted Knockout Winner (including extra time/penalties): **{win}** ({win_p*100:.1f}% probability)")

# TAB 4: Model Performance & ELO
with tab4:
    st.header("📈 Model Evaluation & Global ELO Rankings")
    
    st.subheader("2022 World Cup Evaluation Metrics")
    st.write(f"The model was evaluated against the 2022 World Cup hold-out test set with **{test_acc:.2%} test accuracy**.")
    
    # ELO Table
    st.subheader("FIFA 2026 Contenders ELO Leaderboard")
    elo_df = pd.DataFrame([
        {"Team": t, "ELO Rating": s["elo"], "Recent Form (Win Rate)": f"{s['form_win']*100:.1f}%", "Avg Goals Scored": f"{s['avg_scored']:.2f}", "Avg Goals Conceded": f"{s['avg_conceded']:.2f}"}
        for t, s in team_stats.items()
    ]).sort_values("ELO Rating", ascending=False).reset_index(drop=True)
    
    st.dataframe(elo_df, use_container_width=True)

st.markdown("---")
st.caption("FIFA 2026 Winner Prediction System | XGBoost & Scikit-Learn | ELO Dynamics")

