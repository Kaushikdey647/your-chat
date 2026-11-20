# your chat

Turn your Instagram DM export into a chat fine-tuning dataset that teaches a model to reply the way **you** do.

One script, standard library only. The output is prompt/completion JSONL in chat format, ready for TRL's `SFTTrainer`, Unsloth, or anything else that reads conversational data.

## Usage

1. Download your data from Instagram: *Accounts Center → Your information and permissions → Download your information*. Choose **JSON** format; it must include **Messages**.
2. Unzip the export into `data/`. The script finds `messages/inbox` anywhere under it, which covers both the older `messages/inbox` layout and the newer `your_instagram_activity/messages/inbox` layout.
3. Preview, then build:

```bash
python build_dataset.py --preview 3   # show a few cleaned threads and samples, writes nothing
python build_dataset.py               # writes data/train.jsonl and data/val.jsonl
```

Your display name is auto-detected as the participant present in the most threads. If that picks the wrong person, pass it explicitly: `--me "Your Name"`. Use `--data path/to/export` if the export isn't in `data/`.

## What it produces

One line per reply you sent:

```json
{"prompt": [{"role": "system", "content": "You are Jane Doe, chatting with Alex on Instagram. Reply the way Jane would."},
            {"role": "user", "content": "free tonight?"}],
 "completion": [{"role": "assistant", "content": "yesss\nwhere tho"}]}
```

- The **prompt** is a system line naming who you're talking to, plus up to 10 preceding turns, always ending on the other person's message.
- The **completion** is your reply. Consecutive messages you sent are merged into one turn, joined with newlines, so the model learns your multi-message bursts.
- Train on the completion only, so the model learns your side and not your friends'.

For example, if you (`Y`) and a friend (`F`) talk as `Y F Y F F Y Y F`, the samples are `YF → Y`, `YFYFF → YY` and so on: every reply of yours, given what came before it.

## How to use it

- For my use case I took a Qwen 2.5 7B 4bit quantized and created a LORA Adapter on top of it
- [Sample Notebook](https://www.kaggle.com/code/kaushikdey2026/kaydee647-kaydee-qwen2-5-7b-lora-fine-tune)

## How the data is cleaned

- **Encoding:** fixes Instagram's double-encoded text, so emojis and non-English characters come out right.
- **Threads:** keeps only one-to-one chats. Group chats and chats with yourself are skipped.
- **Noise:** drops anything that isn't typed text, including reactions, likes, "sent an attachment" stubs, media-only messages, unsent messages, call events, theme and nickname changes, and "quiet mode" notices.
- **Sessions:** splits each thread wherever messages are more than 6 hours apart, so a reply's context never comes from an unrelated old conversation.
- **Train/validation split:** done by thread (5% held out), so no conversation leaks from training into validation.

The tunable constants (`SESSION_GAP_MS`, `MAX_CTX_TURNS`, `VAL_FRAC`, `SEED`) are at the top of `build_dataset.py`.

## Privacy

Your export and the generated datasets contain other people's private messages. `data/` is git-ignored. Keep it that way, keep any models trained on it private, and don't share them.
