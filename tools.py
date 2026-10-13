"""Prediction helpers and chat logic shared by the app."""
import difflib
import json
import re
import time
from datetime import date, timedelta

import pandas as pd

from data import BASE, LEAGUES, _download, load_league
from model import MatchModel

CODE_TO_LEAGUE = {code: name for name, (kind, code) in LEAGUES.items() if kind == "main"}
MODEL_TTL = 6 * 3600
FIXTURE_TTL = 3600
_models = {}
_fixtures = {}


# ---------- models and name matching ----------

def get_model(league):
    hit = _models.get(league)
    if hit and time.time() - hit[0] < MODEL_TTL:
        return hit[1], None
    df = load_league(league)
    if df.empty:
        return None, f"Could not download results for {league}."
    try:
        model = MatchModel(df)
    except ValueError as e:
        return None, str(e)
    _models[league] = (time.time(), model)
    return model, None


def find_league(text):
    if not text:
        return None
    low = {k.lower(): k for k in LEAGUES}
    t = text.lower().strip()
    if t in low:
        return low[t]
    for k, v in low.items():
        if t in k:
            return v
    m = difflib.get_close_matches(t, list(low), n=1, cutoff=0.4)
    return low[m[0]] if m else None


def find_team(model, name):
    t = name.lower().strip()
    low = {x.lower(): x for x in model.teams}
    if t in low:
        return low[t]
    subs = [k for k in low if t in k or k in t]
    if subs:
        return low[min(subs, key=len)]
    m = difflib.get_close_matches(t, list(low), n=1, cutoff=0.6)
    return low[m[0]] if m else None


def locate(home, away, hint=None):
    """Find the league that contains both teams. Tries the hint, then loaded leagues, then the rest."""
    order = []
    h = find_league(hint)
    if h:
        order.append(h)
    order += [x for x in _models if x not in order]
    order += [x for x in LEAGUES if x not in order]
    for league in order:
        model, err = get_model(league)
        if err:
            continue
        th, ta = find_team(model, home), find_team(model, away)
        if th and ta and th != ta:
            return league, model, th, ta
    return None


# ---------- tools ----------

def predict_match(home, away, league=None, odds_home=None, odds_draw=None, odds_away=None):
    found = locate(home, away, league)
    if not found:
        return {"error": "Could not find both teams in the covered leagues. Check the spelling."}
    league, model, h, a = found
    p = model.predict(h, a)
    out = {
        "league": league, "home": h, "away": a,
        "home_win": round(p["home"], 3), "draw": round(p["draw"], 3), "away_win": round(p["away"], 3),
        "home_or_draw": round(p["home"] + p["draw"], 3),
        "away_or_draw": round(p["away"] + p["draw"], 3),
        "home_or_away": round(p["home"] + p["away"], 3),
        "expected_goals": [round(p["xg_home"], 2), round(p["xg_away"], 2)],
        "over_2_5": round(p["over_2_5"], 3), "both_teams_score": round(p["btts"], 3),
        "likely_scores": [f"{x}-{y} ({pr:.0%})" for x, y, pr in p["scorelines"][:3]],
        "fair_odds": {"home": round(1 / p["home"], 2), "draw": round(1 / p["draw"], 2),
                      "away": round(1 / p["away"], 2)},
    }
    given = {"home": odds_home, "draw": odds_draw, "away": odds_away}
    probs = {"home": p["home"], "draw": p["draw"], "away": p["away"]}
    edges = {k: round(probs[k] - 1 / v, 3) for k, v in given.items() if v and v > 1}
    if edges:
        out["edge_vs_your_odds"] = edges
    return out


def load_fixtures():
    hit = _fixtures.get("fx")
    if hit and time.time() - hit[0] < FIXTURE_TTL:
        return hit[1]
    df = _download(f"{BASE}/fixtures.csv")
    if df is None or "Date" not in df.columns:
        return None
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    _fixtures["fx"] = (time.time(), df)
    return df


def _odds(row, side):
    for c in (f"Avg{side}", f"B365{side}", f"PS{side}", f"Max{side}"):
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v) and v > 1:
            return float(v)
    return None


def rank_picks(day=None, top_n=8):
    fx = load_fixtures()
    if fx is None:
        return {"error": "Could not download the fixtures list."}
    try:
        target = pd.Timestamp(day).normalize() if day else pd.Timestamp(date.today())
    except Exception:
        return {"error": f"Could not read the date: {day}"}
    todays = fx[fx["Date"].dt.normalize() == target]
    if todays.empty:
        later = fx[fx["Date"] >= target]["Date"].dt.normalize().unique()
        return {"message": f"No fixtures in the covered leagues on {target:%Y-%m-%d}.",
                "next_dates_with_fixtures": sorted(f"{d:%Y-%m-%d}" for d in later)[:3]}

    rows, skipped = [], 0
    for div, grp in todays.groupby("Div"):
        league = CODE_TO_LEAGUE.get(div)
        model, err = (get_model(league) if league else (None, "unsupported"))
        if err:
            skipped += len(grp)
            continue
        for _, r in grp.iterrows():
            h, a = find_team(model, str(r["HomeTeam"])), find_team(model, str(r["AwayTeam"]))
            if not h or not a or h == a:
                skipped += 1
                continue
            p = model.predict(h, a)
            sides = {"Home win": p["home"], "Draw": p["draw"], "Away win": p["away"]}
            lean, lean_p = max(sides.items(), key=lambda kv: kv[1])
            dc = {"Home or draw": p["home"] + p["draw"], "Away or draw": p["away"] + p["draw"]}
            safe, safe_p = max(dc.items(), key=lambda kv: kv[1])
            odds = {"Home win": _odds(r, "H"), "Draw": _odds(r, "D"), "Away win": _odds(r, "A")}
            edges = {k: sides[k] - 1 / v for k, v in odds.items() if v}
            entry = {
                "match": f"{h} vs {a}", "league": league, "time": str(r.get("Time", "")),
                "lean": lean, "lean_prob": round(lean_p, 3), "lean_odds": odds.get(lean),
                "safest_double_chance": safe, "safest_prob": round(safe_p, 3),
            }
            if edges:
                best = max(edges, key=edges.get)
                entry["value_pick"] = best
                entry["value_edge"] = round(edges[best], 3)
                entry["value_odds"] = odds[best]
            rows.append(entry)

    by_conf = sorted(rows, key=lambda e: e["lean_prob"], reverse=True)[:top_n]
    by_value = sorted([e for e in rows if e.get("value_edge", 0) > 0],
                      key=lambda e: e["value_edge"], reverse=True)[:top_n]
    return {"date": f"{target:%Y-%m-%d}", "matches_analysed": len(rows),
            "skipped_unrecognised": skipped,
            "highest_confidence": by_conf, "best_value_vs_odds": by_value}


# ---------- plain-text formatting (used without an API key) ----------

def format_match(res):
    if "error" in res:
        return res["error"]
    xg = res["expected_goals"]
    return (
        f"**{res['home']} vs {res['away']}** ({res['league']})\n\n"
        f"Home {res['home_win']:.0%}, draw {res['draw']:.0%}, away {res['away_win']:.0%}.\n\n"
        f"Safer options: home or draw {res['home_or_draw']:.0%}, away or draw {res['away_or_draw']:.0%}.\n\n"
        f"Expected goals {xg[0]} to {xg[1]}. Over 2.5 goals {res['over_2_5']:.0%}. "
        f"Both teams score {res['both_teams_score']:.0%}.\n\n"
        f"Likely scores: {', '.join(res['likely_scores'])}."
    )


def format_picks(res):
    if "error" in res:
        return res["error"]
    if "message" in res:
        later = ", ".join(res.get("next_dates_with_fixtures", []))
        return res["message"] + (f" Next dates with fixtures: {later}." if later else "")
    lines = [f"**Picks for {res['date']}** ({res['matches_analysed']} matches analysed)", ""]
    for e in res["highest_confidence"]:
        lines.append(f"- {e['match']}: {e['lean']} {e['lean_prob']:.0%}. "
                     f"Safer: {e['safest_double_chance']} {e['safest_prob']:.0%}.")
    if res["best_value_vs_odds"]:
        lines += ["", "**Best value against bookmaker odds**"]
        for e in res["best_value_vs_odds"][:5]:
            lines.append(f"- {e['match']}: {e['value_pick']} at {e['value_odds']} "
                         f"(model edge {e['value_edge']:+.0%})")
    lines += ["", "The model only knows goals scored and conceded. Check team news first."]
    return "\n".join(lines)


def simple_reply(text, default_league=None):
    m = re.search(r"(.+?)\s+(?:vs\.?|v\.?|versus|against)\s+(.+)", text, re.I)
    if m:
        home, away = m.group(1).strip(" ?!."), m.group(2).strip(" ?!.")
        return format_match(predict_match(home, away, default_league))
    if re.search(r"tonight|today|tomorrow|pick|best|safe", text, re.I):
        day = date.today() + timedelta(days=1) if "tomorrow" in text.lower() else None
        return format_picks(rank_picks(day))
    return "Try a match like **Arsenal vs Chelsea**, or ask for **best picks tonight**."


# ---------- Claude chat ----------

SYSTEM = (
    "You are a football match analyst inside a prediction app. Always call the tools for numbers. "
    "Never invent probabilities, odds, injuries, lineups or fixtures. "
    "The model only uses goals scored and conceded from recent results, so say that briefly when giving picks. "
    "For 'safest' picks, prefer high-probability outcomes and double chance, and say that no pick is certain. "
    "If a tool returns an error or no fixtures, say so plainly and suggest what to try. "
    "Replies must be short and easy to read on a phone: lead with the answer, then 3 to 6 short lines. "
    "Use plain language. Remind the user to bet responsibly only when you give picks."
)

TOOLS = [
    {
        "name": "predict_match",
        "description": "Model probabilities for one match (1X2, double chance, goals). Finds the league automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home": {"type": "string", "description": "Home team name"},
                "away": {"type": "string", "description": "Away team name"},
                "league": {"type": "string", "description": "Optional league hint, e.g. Championship"},
                "odds_home": {"type": "number"}, "odds_draw": {"type": "number"},
                "odds_away": {"type": "number"},
            },
            "required": ["home", "away"],
        },
    },
    {
        "name": "rank_picks",
        "description": "Analyse every fixture on a date in the covered leagues and rank by confidence and by value against bookmaker odds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD. Omit for today."},
                "top_n": {"type": "integer", "description": "How many matches to return (default 8)."},
            },
        },
    },
]


def run_tool(name, args):
    try:
        if name == "predict_match":
            return predict_match(**args)
        if name == "rank_picks":
            return rank_picks(args.get("date"), int(args.get("top_n") or 8))
        return {"error": f"Unknown tool {name}"}
    except Exception as e:
        return {"error": f"Tool failed: {e}"}


def chat_with_claude(client, model_name, history):
    msgs = [dict(m) for m in history[-12:]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    system = SYSTEM + f" Today is {date.today():%A %d %B %Y}."
    for _ in range(6):
        resp = client.messages.create(model=model_name, max_tokens=1200, system=system,
                                      tools=TOOLS, messages=msgs)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return text or "I couldn't put an answer together. Try rephrasing."
        msgs.append({"role": "assistant", "content": resp.content})
        results = [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": json.dumps(run_tool(b.name, dict(b.input)), default=str)}
            for b in resp.content if b.type == "tool_use"
        ]
        msgs.append({"role": "user", "content": results})
    return "That took too many steps. Try a simpler question."
