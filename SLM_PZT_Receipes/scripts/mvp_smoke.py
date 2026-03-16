#!/usr/bin/env python3
import os
import sys
import time
import json
import httpx


def main():
    api = os.environ.get("API_BASE", "http://localhost:8000")
    text = (
        "I can cook:\nNo-Bake Cookies\nIngredients: sugar, butter, oats\nInstructions: mix and chill"
    )
    k = 3
    print(f"POST /resolve k={k}")
    r = httpx.post(f"{api}/resolve", json={"text": text, "k": k}, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    print(json.dumps(data, indent=2)[:400])
    if not data.get("candidates"):
        print("No candidates returned.")
        return 1
    rid = data["candidates"][0]["recipe_id"]

    print(f"GET /transfer/from/{rid}?k=5&mode=few_switches")
    r2 = httpx.get(f"{api}/transfer/from/{rid}", params={"k": 5, "mode": "few_switches"}, timeout=60.0)
    r2.raise_for_status()
    data2 = r2.json()
    print(json.dumps(data2, indent=2)[:400])
    if not data2.get("results"):
        print("No few-switches transfers; trying closest...")
        r2 = httpx.get(f"{api}/transfer/from/{rid}", params={"k": 5, "mode": "closest"}, timeout=60.0)
        r2.raise_for_status()
        data2 = r2.json()
        print(json.dumps(data2, indent=2)[:400])
    if not data2.get("results"):
        print("No transfers found.")
        return 0
    tid = data2["results"][0].get("target_id") or data2["results"][0].get("source_id")
    if not tid:
        print("No pair to explain.")
        return 0
    print("POST /explain")
    r3 = httpx.post(f"{api}/explain", json={"source_id": rid, "target_id": tid, "style": "paragraph"}, timeout=60.0)
    r3.raise_for_status()
    print(json.dumps(r3.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

