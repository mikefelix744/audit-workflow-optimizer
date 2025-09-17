
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from pathlib import Path
from sklearn.linear_model import LinearRegression


import logging
logging.basicConfig(level=logging.INFO)


# --- Config Section: File paths, skills, mappings ---
ENGAGEMENTS_FILE = os.getenv("ENGAGEMENTS_FILE", "engagements_sample.csv")
STAFF_FILE = os.getenv("STAFF_FILE", "staff_sample.csv")
REQUIRED_SKILLS = ["Audit"]
SIZE_MAP = {"Small":0,"Medium":1,"Large":2}
COMP_MAP = {"Low":0,"Medium":1,"High":2}
LEVEL_SCORE = {"Junior":1,"Associate":2,"Senior":3,"Manager":4}

ROOT = Path(__file__).resolve().parent
try:
    eng = pd.read_csv(ROOT/ENGAGEMENTS_FILE)
except Exception as e:
    logging.error(f"Failed to load {ENGAGEMENTS_FILE}: {e}")
    raise
try:
    staff = pd.read_csv(ROOT/STAFF_FILE)
except Exception as e:
    logging.error(f"Failed to load {STAFF_FILE}: {e}")
    raise

def audit_optimizer(industry, size, complexity, prev_issues, add_delay=0):
    """
    Estimate audit hours and recommend staff team.
    Args:
        industry (str): Industry name.
        size (str): Size category (Small, Medium, Large).
        complexity (str): Complexity level (Low, Medium, High).
        prev_issues (int): Number of previous issues.
        add_delay (int): Additional weeks to add to timeline.
    Returns:
        dict: Results with estimated hours, weeks, staff ranking, and suggested team.
    Raises:
        ValueError: If input values are invalid.
    """
    """
    Estimate audit hours and recommend staff team.
    Args:
        industry (str): Industry name.
        size (str): Size category (Small, Medium, Large).
        complexity (str): Complexity level (Low, Medium, High).
        prev_issues (int): Number of previous issues.
        add_delay (int): Additional weeks to add to timeline.
    Returns:
        dict: Results with estimated hours, weeks, staff ranking, and suggested team.
    Raises:
        ValueError: If input values are invalid.
    """
    # --- Input validation ---
    logging.info("Validating input parameters...")
    if industry not in eng.client_industry.unique():
        logging.error(f"Unknown industry: {industry}")
        raise ValueError(f"Unknown industry: {industry}")
    if size not in SIZE_MAP:
        logging.error(f"Unknown size: {size}")
        raise ValueError(f"Unknown size: {size}")
    if complexity not in COMP_MAP:
        logging.error(f"Unknown complexity: {complexity}")
        raise ValueError(f"Unknown complexity: {complexity}")
    if not isinstance(prev_issues, int) or prev_issues < 0:
        logging.error("prev_issues must be a non-negative integer")
        raise ValueError("prev_issues must be a non-negative integer")
    if not isinstance(add_delay, int) or add_delay < 0:
        logging.error("add_delay must be a non-negative integer")
        raise ValueError("add_delay must be a non-negative integer")
    # --- Encode features
    size_map = {"Small":0,"Medium":1,"Large":2}
    comp_map = {"Low":0,"Medium":1,"High":2}
    ind_map = {v:i for i,v in enumerate(sorted(eng.client_industry.unique()))}
    
    # --- Build historical data ---
    logging.info("Building historical data for regression model...")
    ind_map = {v:i for i,v in enumerate(sorted(eng.client_industry.unique()))}
    X = []
    y = eng.hours_spent.values
    for _, row in eng.iterrows():
        X.append([ind_map[row.client_industry], SIZE_MAP[row.client_size], COMP_MAP[row.complexity], row.prev_issues])
    X = np.array(X, dtype=float)

    # --- Fit regression using scikit-learn ---
    logging.info("Fitting regression model...")
    model = LinearRegression()
    model.fit(X, y)

    # --- Predict hours for new engagement ---
    xnew = np.array([[ind_map[industry], SIZE_MAP[size], COMP_MAP[complexity], prev_issues]])
    est_hours = float(model.predict(xnew)[0])
    logging.info(f"Estimated hours: {est_hours}")
    
    # --- Timeline suggestion ---
    recommended_weeks = max(1, int(np.round(est_hours / 120)))  # 4 staff * 30h/week = 120h
    if add_delay > 0:
        recommended_weeks += add_delay
    logging.info(f"Recommended weeks: {recommended_weeks}")
    
    # --- Staff matching heuristic ---
    logging.info("Scoring and ranking staff...")
    staff_df = staff.copy()
    # Vectorized scoring for performance
    def skill_score_vec(skills):
        return sum(1 for s in REQUIRED_SKILLS if s in [x.strip() for x in skills.split(",")])
    staff_df["skill_score"] = staff_df["skills"].apply(skill_score_vec)
    staff_df["level_score"] = staff_df["level"].map(LEVEL_SCORE).fillna(1)
    staff_df["avail_score"] = staff_df["available_hours_per_week"] / 40
    staff_df["score"] = staff_df["skill_score"]*2 + staff_df["level_score"] + staff_df["avail_score"]
    staff_df = staff_df.sort_values("score", ascending=False)
    suggested_team = staff_df.head(4)[["staff_id","name","level","skills"]]
    logging.info(f"Suggested team: {suggested_team.to_dict(orient='records')}")
    
    # --- Package results ---
    result = {
        "estimated_hours": round(est_hours, 2),
        "recommended_weeks": recommended_weeks,
        "staff_ranking": staff_df[["staff_id","name","level","skills","score"]].to_dict(orient="records"),
        "suggested_team": suggested_team.to_dict(orient="records"),
        "scenario_note": "Re-check staff availability if delays extend beyond baseline plan."
    }
    return result
# Example run
if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Audit Workflow Optimizer CLI")
    parser.add_argument("industry", type=str, help="Client industry")
    parser.add_argument("size", type=str, choices=list(SIZE_MAP.keys()), help="Client size")
    parser.add_argument("complexity", type=str, choices=list(COMP_MAP.keys()), help="Engagement complexity")
    parser.add_argument("prev_issues", type=int, help="Number of previous issues")
    parser.add_argument("--add-delay", type=int, default=0, help="Additional weeks to add to timeline")
    args = parser.parse_args()
    try:
        output = audit_optimizer(args.industry, args.size, args.complexity, args.prev_issues, add_delay=args.add_delay)
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Error: {e}")
