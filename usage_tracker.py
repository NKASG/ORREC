import requests
import os
from dotenv import load_dotenv

load_dotenv()

USAGE_URL = "https://fhgsufjolykcuqcanywl.supabase.co/functions/v1/track-usage"
AGENT_ID = os.getenv("AGENT_ID")
API_KEY = os.getenv("USAGE_TRACKER_KEY")

def track_usage(prompt, engine, tokens_used, cost):
    try:
        payload = {
            "agentId": AGENT_ID,
            "prompt": prompt,
            "engine": engine,
            "tokensUsed": tokens_used,
            "cost": cost
        }

        headers = {
            "Content-Type": "application/json",
            "x-agent-key": API_KEY
        }

        r = requests.post(USAGE_URL, json=payload, headers=headers, timeout=5)

        if r.status_code == 200:
            data = r.json()
            if data.get("success"):
                print("✅ Usage tracked")
            else:
                print("⚠️ Tracking failed:", data)
        else:
            print("⚠️ Tracking server error:", r.text)

    except Exception as e:
        print("⚠️ Usage tracking error:", e)

