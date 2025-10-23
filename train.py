import os, torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

ds = load_dataset("yahma/alpaca-cleaned", split="train[:1%]")

model_name = "unsloth/tinyllama"
max_seq_len = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_len,
    dtype=None,
)

def format_example(e):
    inst = e.get("instruction","")
    inp  = e.get("input","")
    out  = e.get("output","")
    prefix = f"### Instruction:\n{inst}\n"
    if inp:
        prefix += f"\n### Input:\n{inp}\n"
    return {"text": prefix + "\n### Response:\n" + out}

ds = ds.map(format_example, remove_columns=ds.column_names)

cfg = SFTConfig(
    output_dir="/outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=60,
    logging_steps=5,
    num_train_epochs=1,
    max_seq_length=max_seq_len,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tokenizer)
print(">>> Starting training …")
print(trainer.train())
