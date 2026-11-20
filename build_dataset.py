"""Build prompt/completion SFT JSONL from an Instagram DM export.

Every sample's completion is one of your turns (consecutive messages merged);
the prompt is the preceding turns of the same session, always ending on the other person.
"""
import argparse
import collections
import json
import random
import re
import statistics
from pathlib import Path

DATA_DIR = Path("data")
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
SYSTEM = "You are {me}, chatting with {other} on Instagram. Reply the way {first} would."


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


def to_turns(session, me):
    turns = []
    for sender, text in session:
        role = "assistant" if sender == me else "user"
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


def find_inbox(root):
    """Instagram exports put DMs at messages/inbox or your_instagram_activity/messages/inbox."""
    hits = sorted(root.glob("**/messages/inbox"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"no messages/inbox folder under {root}/; unzip your Instagram export (JSON format) there")
    return hits[0]


def participants(thread_dir):
    f = thread_dir / "message_1.json"
    return [fix(p["name"]) for p in json.loads(f.read_text())["participants"]] if f.is_file() else []


def detect_me(inbox):
    """The account owner is the one participant present in (nearly) every thread."""
    counts = collections.Counter(n for d in inbox.iterdir() if d.is_dir() for n in set(participants(d)))
    if not counts:
        raise SystemExit(f"no threads with message_1.json in {inbox}")
    return counts.most_common(1)[0][0]


def load_thread(thread_dir, me):
    """-> (other_name, [turns per session]) or None if the thread is skipped."""
    files = sorted(thread_dir.glob("message_*.json"))
    if not files:
        return None
    data = [json.loads(f.read_text()) for f in files]
    names = [fix(p["name"]) for p in data[0]["participants"]]
    if len(names) != 2 or me not in names or len(set(names)) != 2:
        return None
    other = next(n for n in names if n != me)
    msgs = []
    for d in data:
        for m in d["messages"]:
            text = clean(m)
            if text:
                msgs.append((m["timestamp_ms"], fix(m["sender_name"]), text))
    msgs.sort(key=lambda x: x[0])
    return other, [to_turns(sess, me) for sess in sessions(msgs)]


def thread_samples(thread_dir, me):
    loaded = load_thread(thread_dir, me)
    if not loaded:
        return []
    other, sess_turns = loaded
    system = SYSTEM.format(me=me, other=other, first=me.split()[0])
    return [s for turns in sess_turns for s in make_samples(turns, system)]


def preview(inbox, me, n):
    """Print n cleaned thread sessions and n model input/output samples."""
    rng = random.Random()
    threads = [p for p in inbox.iterdir() if p.is_dir()]
    rng.shuffle(threads)
    loaded = [(d, t) for d in threads if (t := load_thread(d, me)) and any(len(s) > 3 for s in t[1])][:n]

    print("=" * 30, "MESSAGE THREADS (cleaned, merged turns)", "=" * 30)
    for d, (other, sess_turns) in loaded:
        sess = rng.choice([s for s in sess_turns if len(s) > 3])
        print(f"\n--- {d.name}: {len(sess_turns)} sessions, showing one with {len(sess)} turns ---")
        for t in sess[:12]:
            who = me if t["role"] == "assistant" else other
            print(f"[{who}] {t['content']!r}")

    print("\n" + "=" * 30, "MODEL INPUT -> OUTPUT", "=" * 30)
    for d, _ in loaded:
        s = rng.choice(thread_samples(d, me))
        print(f"\n--- {d.name} ---\nINPUT:")
        for t in s["prompt"]:
            print(f"  {t['role']:>9}: {t['content']!r}")
        print(f"OUTPUT:\n  assistant: {s['completion'][0]['content']!r}")


def _selftest():
    pattern = "KSKSSKKSKSKSKKSKSK"
    session = [("Me" if c == "K" else "Swapnil", f"{c}{i}") for i, c in enumerate(pattern)]
    turns = to_turns(session, "Me")
    shape = " ".join("K" * t["content"].count("K") or "S" * t["content"].count("S") for t in turns)
    assert shape == "K S K SS KK S K S K S KK S K S K", shape
    samples = make_samples(turns, "sys")
    assert all(s["prompt"][-1]["role"] == "user" for s in samples)
    assert all(s["completion"][0]["role"] == "assistant" for s in samples)
    # KS -> K2, KSKSS -> K5\nK6, KSKSSKKS -> K8
    assert [s["completion"][0]["content"] for s in samples[:3]] == ["K2", "K5\nK6", "K8"]
    assert len(samples) == 7
    ts = [(0, "Me", "a"), (1, "x", "b"), (1 + SESSION_GAP_MS + 1, "x", "c")]
    assert len(sessions(ts)) == 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--me", help="your Instagram display name exactly as in the export (default: auto-detect)")
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="folder containing the unzipped export (default: data)")
    ap.add_argument("--preview", type=int, metavar="N", help="print N sample threads and model I/O, don't write files")
    args = ap.parse_args()
    _selftest()
    inbox = find_inbox(args.data)
    me = fix(args.me) if args.me else detect_me(inbox)
    print(f"inbox: {inbox}\nme: {me}" + ("" if args.me else "  (auto-detected; pass --me to override)"))
    if args.preview:
        return preview(inbox, me, args.preview)
    threads = sorted(p for p in inbox.iterdir() if p.is_dir())
    random.Random(SEED).shuffle(threads)
    n_val = max(1, int(len(threads) * VAL_FRAC))
    splits = {"val": threads[:n_val], "train": threads[n_val:]}

    for name, dirs in splits.items():
        n_threads, lens = 0, []
        with open(args.data / f"{name}.jsonl", "w") as f:
            for d in dirs:
                samples = thread_samples(d, me)
                n_threads += bool(samples)
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    lens.append(sum(len(t["content"]) for t in s["prompt"]))
        p95 = statistics.quantiles(lens, n=20)[-1] if len(lens) > 1 else 0
        print(f"{name}: {n_threads} threads, {len(lens)} samples, "
              f"prompt chars median={statistics.median(lens) if lens else 0:.0f} p95={p95:.0f}")


if __name__ == "__main__":
    main()
