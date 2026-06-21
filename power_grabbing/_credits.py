"""Print OpenRouter account usage/credits (source of truth for spend)."""
import json, os, urllib.request


def key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip()
    return k


def credits():
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key()}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)["data"]
    return d  # {total_credits, total_usage}


if __name__ == "__main__":
    d = credits()
    tc, tu = d.get("total_credits", 0), d.get("total_usage", 0)
    print(f"total_credits={tc:.4f}  total_usage={tu:.4f}  remaining={tc - tu:.4f}")
