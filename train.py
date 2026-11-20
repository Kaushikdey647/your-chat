"""LoRA fine-tune Qwen2.5-3B-Instruct on data/{train,val}.jsonl. Needs a CUDA GPU."""
from unsloth import FastLanguageModel  # must be imported before transformers/trl

import json

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/Qwen2.5-3B-Instruct"
MAX_LEN = 2048
OUT = "outputs/lora"

model, tokenizer = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAX_LEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=0,
)

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=SFTConfig(
        output_dir="outputs",
        max_length=MAX_LEN,
        completion_only_loss=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        bf16=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=20,
        save_steps=500,
        report_to="none",
        seed=0,
    ),
)
trainer.train()
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

FastLanguageModel.for_inference(model)
for row in ds["validation"].select(range(2)):
    inputs = tokenizer.apply_chat_template(
        row["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8)
    print(json.dumps(row["prompt"][-1], ensure_ascii=False))
    print("model :", tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print("actual:", row["completion"][0]["content"], "\n")
