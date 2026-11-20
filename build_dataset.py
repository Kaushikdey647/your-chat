"""Build prompt/completion SFT JSONL from the Instagram inbox export.

Every sample's completion is one Kaushik Dey turn (consecutive messages merged);
the prompt is the preceding turns of the same session, always ending on the other person.
"""
import argparse
import json
import random
import re
import statistics
from pathlib import Path

ME = "Kaushik Dey"
INBOX = Path("data/messages/inbox")
OUT_DIR = Path("data")
SESSION_GAP_MS = 6 * 60 * 60 * 1000
MAX_CTX_TURNS = 10
VAL_FRAC = 0.05
SEED = 0

NOISE = re.compile(
    r"^Reacted .* to your message|^Liked a message|sent an attachment\.?$"
    r"|^(\S+ )?(started|missed) (a|an) (video|audio) (call|chat)"
    r"|changed the theme to|set (the|your|his|her) nickname|set the quick reaction",
    re.DOTALL,
)
QUIET = re.compile(r"^.*wasn't notified about this message because they're in quiet mode\.?$", re.MULTILINE)
SYSTEM = "You are Kaushik Dey, chatting with {other} on Instagram. Reply the way Kaushik would."


def fix(s):
    # Instagram exports UTF-8 bytes as latin-1 code points.
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def clean(m):
    if m.get("is_unsent") or "call_duration" in m or not m.get("content"):
        return None
    text = QUIET.sub("", fix(m["content"])).strip()
    return None if not text or NOISE.search(text) else text


def sessions(msgs):
    """msgs: [(ts, sender, text)] sorted by ts -> list of sessions."""
    out, prev = [], None
    for ts, sender, text in msgs:
        if prev is None or ts - prev > SESSION_GAP_MS:
            out.append([])
        out[-1].append((sender, text))
        prev = ts
    return out


def to_turns(session):
    turns = []
    for sender, text in session:
        role = "assistant" if sender == ME else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + text
        else:
            turns.append({"role": role, "content": text})
    return turns


def make_samples(turns, system):
    return [
        {
            "prompt": [{"role": "system", "content": system}] + turns[max(0, i - MAX_CTX_TURNS):i],
            "completion": [turns[i]],
        }
        for i in range(1, len(turns))
        if turns[i]["role"] == "assistant"
    ]


def load_thread(thread_dir):
    """-> (other_name, [turns per session]) or None if the thread is skipped."""
    files = sorted(thread_dir.glob("message_*.json"))
    if not files:
        return None
    data = [json.loads(f.read_text()) for f in files]
    names = [fix(p["name"]) for p in data[0]["participants"]]
    if len(names) != 2 or ME not in names or len(set(names)) != 2:
        return None
    other = next(n for n in names if n != ME)
    msgs = []
    for d in data:
        for m in d["messages"]:
            text = clean(m)
            if text:
                msgs.append((m["timestamp_ms"], fix(m["sender_name"]), text))
    msgs.sort(key=lambda x: x[0])
    return other, [to_turns(sess) for sess in sessions(msgs)]


def thread_samples(thread_dir):
    loaded = load_thread(thread_dir)
    if not loaded:
        return []
    other, sess_turns = loaded
    system = SYSTEM.format(other=other)
    return [s for turns in sess_turns for s in make_samples(turns, system)]


def preview(n):
    """Print n cleaned thread sessions and n model input/output samples."""
    rng = random.Random()
    threads = [p for p in INBOX.iterdir() if p.is_dir()]
    rng.shuffle(threads)
    loaded = [(d, t) for d in threads if (t := load_thread(d)) and any(len(s) > 3 for s in t[1])][:n]

    print("=" * 30, "MESSAGE THREADS (cleaned, merged turns)", "=" * 30)
    for d, (other, sess_turns) in loaded:
        sess = rng.choice([s for s in sess_turns if len(s) > 3])
        print(f"\n--- {d.name}: {len(sess_turns)} sessions, showing one with {len(sess)} turns ---")
        for t in sess[:12]:
            who = ME if t["role"] == "assistant" else other
            print(f"[{who}] {t['content']!r}")

    print("\n" + "=" * 30, "MODEL INPUT -> OUTPUT", "=" * 30)
    for d, _ in loaded:
        s = rng.choice(thread_samples(d))
        print(f"\n--- {d.name} ---\nINPUT:")
        for t in s["prompt"]:
            print(f"  {t['role']:>9}: {t['content']!r}")
        print(f"OUTPUT:\n  assistant: {s['completion'][0]['content']!r}")


def _selftest():
    pattern = "KSKSSKKSKSKSKKSKSK"
    session = [("Kaushik Dey" if c == "K" else "Swapnil", f"{c}{i}") for i, c in enumerate(pattern)]
    turns = to_turns(session)
    shape = " ".join("K" * t["content"].count("K") or "S" * t["content"].count("S") for t in turns)
    assert shape == "K S K SS KK S K S K S KK S K S K", shape
    samples = make_samples(turns, "sys")
    assert all(s["prompt"][-1]["role"] == "user" for s in samples)
    assert all(s["completion"][0]["role"] == "assistant" for s in samples)
    # KS -> K2, KSKSS -> K5\nK6, KSKSSKKS -> K8
    assert [s["completion"][0]["content"] for s in samples[:3]] == ["K2", "K5\nK6", "K8"]
    assert len(samples) == 7
    ts = [(0, ME, "a"), (1, "x", "b"), (1 + SESSION_GAP_MS + 1, "x", "c")]
    assert len(sessions(ts)) == 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, metavar="N", help="print N sample threads and model I/O, don't write files")
    args = ap.parse_args()
    _selftest()
    print("selftest passed")
    if args.preview:
        return preview(args.preview)
    threads = sorted(p for p in INBOX.iterdir() if p.is_dir())
    random.Random(SEED).shuffle(threads)
    n_val = max(1, int(len(threads) * VAL_FRAC))
    splits = {"val": threads[:n_val], "train": threads[n_val:]}

    for name, dirs in splits.items():
        n_threads, lens = 0, []
        with open(OUT_DIR / f"{name}.jsonl", "w") as f:
            for d in dirs:
                samples = thread_samples(d)
                n_threads += bool(samples)
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    lens.append(sum(len(t["content"]) for t in s["prompt"]))
        p95 = statistics.quantiles(lens, n=20)[-1] if len(lens) > 1 else 0
        print(f"{name}: {n_threads} threads, {len(lens)} samples, "
              f"prompt chars median={statistics.median(lens) if lens else 0:.0f} p95={p95:.0f}")


if __name__ == "__main__":
    main()
