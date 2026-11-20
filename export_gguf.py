"""Merge the trained LoRA into the base model, convert to GGUF, quantize, and upload to a private HF repo.

Runs on CPU (Mac is fine). Needs:  brew install llama.cpp   (for llama-quantize)
Auth for upload:                  hf auth login   (or HF_TOKEN env var)

    uv run python export_gguf.py --adapter outputs/lora
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA_CPP = Path.home() / ".cache" / "llama.cpp"
QUANTS = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0"}
SYSTEM = "You are Kaushik Dey, chatting with Swapnil Agarwal on Instagram. Reply the way Kaushik would."

MODELFILE = '''FROM ./{gguf}

TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""

SYSTEM "{system}"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
'''


def merge(adapter, base, out):
    # QLoRA adapters were trained on a 4-bit base; merging into the full-precision base is the standard path.
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)


def to_gguf(merged, out_dir, name, quants):
    if not (LLAMA_CPP / "convert_hf_to_gguf.py").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)], check=True)
    quantize = shutil.which("llama-quantize")
    if not quantize:
        sys.exit("llama-quantize not found: brew install llama.cpp")

    f16 = out_dir / f"{name}-F16.gguf"
    subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(merged),
                    "--outfile", str(f16), "--outtype", "f16"], check=True)
    files = []
    for q in quants:
        dst = out_dir / f"{name}-{QUANTS[q]}.gguf"
        subprocess.run([quantize, str(f16), str(dst), QUANTS[q]], check=True)
        files.append(dst)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/lora", help="folder from model.save_pretrained()")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--name", default="kaydee-qwen2.5-3b")
    ap.add_argument("--quants", nargs="+", default=["q4_k_m", "q8_0"], choices=sorted(QUANTS))
    ap.add_argument("--out", default="outputs/gguf")
    ap.add_argument("--repo", help="HF repo id; default <your-user>/<name>-gguf")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="upload GGUFs already in --out without rebuilding")
    args = ap.parse_args()

    adapter = Path(args.adapter).resolve()
    if not (adapter / "adapter_config.json").is_file():
        sys.exit(f"{adapter} has no adapter_config.json; unzip lora.zip from Kaggle there first")
    out_dir = Path(args.out).resolve()
    merged = out_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_build:
        files = [out_dir / f"{args.name}-{QUANTS[q]}.gguf" for q in args.quants]
        if missing := [f.name for f in files if not f.is_file()]:
            sys.exit(f"missing {missing}; run without --skip-build first")
    else:
        print(f"merging {adapter} into {args.base}")
        merge(adapter, args.base, merged)
        files = to_gguf(merged, out_dir, args.name, args.quants)
    (out_dir / "Modelfile").write_text(MODELFILE.format(gguf=files[0].name, system=SYSTEM))
    for f in files:
        print(f"{f}  {f.stat().st_size / 1e9:.2f} GB")

    if args.no_upload:
        return
    api = HfApi()
    repo = args.repo or f"{api.whoami()['name']}/{args.name}-gguf"
    api.create_repo(repo, private=True, exist_ok=True)  # trained on private chats: never public
    for f in [*files, out_dir / "Modelfile"]:
        print("uploading", f.name)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo)
    print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
