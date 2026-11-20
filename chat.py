"""Chat with the fine-tuned model as Kaushik, talking to any friend.

    uv run python chat.py --friend "Avantika"
    uv run python chat.py --friend "Swapnil Agarwal" --url http://localhost:8080/v1   # llama-server

Works with any OpenAI-compatible server: Ollama (default) or llama.cpp's llama-server.
Commands: /friend <name> switches person and clears history, /reset clears history, /quit exits.
"""
import argparse
import json
import urllib.request

SYSTEM = "You are Kaushik Dey, chatting with {friend} on Instagram. Reply the way Kaushik would."
MAX_TURNS = 10  # matches the context window used in training


def complete(url, model, messages, temperature):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "top_p": 0.9}).encode()
    req = urllib.request.Request(f"{url}/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friend", required=True, help="who Kaushik is talking to, as their Instagram display name")
    ap.add_argument("--model", default="kaydee")
    ap.add_argument("--url", default="http://localhost:11434/v1")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    friend, history = args.friend, []
    print(f"Chatting as Kaushik with {friend}. /friend <name>, /reset, /quit")
    while True:
        try:
            text = input(f"{friend}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset" or text.startswith("/friend "):
            if text.startswith("/friend "):
                friend = text[len("/friend "):].strip() or friend
            history = []
            print(f"(now chatting with {friend}, history cleared)")
            continue

        history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": SYSTEM.format(friend=friend)}] + history[-MAX_TURNS:]
        reply = complete(args.url, args.model, messages, args.temperature)
        history.append({"role": "assistant", "content": reply})
        for line in reply.splitlines():
            print(f"Kaushik> {line}")


if __name__ == "__main__":
    main()
