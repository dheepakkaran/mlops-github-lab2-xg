import csv
import json
import math
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
TOURNAMENTS = [("FIFA World Cup", "2022"), ("UEFA Euro", "2024")]
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shots.csv")


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_ids(competitions, comp_name, season_name):
    for c in competitions:
        if c["competition_name"] == comp_name and c["season_name"] == season_name:
            return c["competition_id"], c["season_id"]
    available = sorted({f'{c["competition_name"]} {c["season_name"]}' for c in competitions})
    raise ValueError(f"{comp_name} {season_name} not found. Available: {available}")


def shot_row(ev, match_id, tournament):
    shot = ev["shot"]
    x, y = ev["location"][0], ev["location"][1]
    dx, dy = 120 - x, abs(40 - y)
    distance = math.hypot(dx, dy)
    angle = math.atan2(8 * dx, dx ** 2 + dy ** 2 - 16)
    return {
        "match_id": match_id,
        "tournament": tournament,
        "minute": ev["minute"],
        "x": x,
        "y": y,
        "distance": round(distance, 3),
        "angle": round(angle, 4),
        "is_header": int(shot["body_part"]["name"] == "Head"),
        "first_time": int(shot.get("first_time", False)),
        "under_pressure": int(ev.get("under_pressure", False)),
        "one_on_one": int(shot.get("one_on_one", False)),
        "from_free_kick": int(shot["type"]["name"] == "Free Kick"),
        "from_counter": int(ev["play_pattern"]["name"] == "From Counter"),
        "from_set_piece": int(ev["play_pattern"]["name"] in ("From Corner", "From Free Kick", "From Throw In")),
        "statsbomb_xg": shot.get("statsbomb_xg"),
        "is_goal": int(shot["outcome"]["name"] == "Goal"),
    }


def main():
    competitions = get_json(f"{BASE}/competitions.json")
    rows = []
    for comp_name, season_name in TOURNAMENTS:
        comp_id, season_id = find_ids(competitions, comp_name, season_name)
        matches = get_json(f"{BASE}/matches/{comp_id}/{season_id}.json")
        print(f"{comp_name} {season_name}: {len(matches)} matches")
        for i, m in enumerate(matches, 1):
            events = get_json(f"{BASE}/events/{m['match_id']}.json")
            for ev in events:
                if ev["type"]["name"] != "Shot" or ev["period"] == 5:
                    continue
                if ev["shot"]["type"]["name"] == "Penalty":
                    continue
                rows.append(shot_row(ev, m["match_id"], f"{comp_name} {season_name}"))
            print(f"  {i}/{len(matches)} matches, {len(rows)} shots so far", end="\r", flush=True)
        print()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    goals = sum(r["is_goal"] for r in rows)
    print(f"Saved {len(rows)} shots ({goals} goals, {goals / len(rows):.1%}) -> data/shots.csv")


if __name__ == "__main__":
    main()
