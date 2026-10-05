import random
import torch
from google.colab import userdata
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from cpqtools import find_beta, compute_qhat, prediction_sets, normalize_answer


model_id = "Qwen/Qwen3-8B"

BATCH_SIZE = 1
MAX_NEW_TOKENS = 10

token = ""
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=token,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    token=token,
)

model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split="validation",
)
ds = dataset.select_columns(["question", "answer"]).shuffle(seed=42)

Dcal1 = ds.select(range(0, 100))
Dcal2 = ds.select(range(1000, 1100))
Dval = ds.select(range(2000, 2100))

B = 15
alpha = 0.3

template = (
    "Answer this trivia question with only the answer."
    "No explanation. Provide the shortest answer you can, without restating any part of the question."
    "Question: "
)

beta_star = find_beta(B, Dcal1, model, tokenizer, template)
q_hat = compute_qhat(B, alpha, beta_star, Dcal2, tokenizer, model, template)

num_examples = len(Dval)
avg_setsize = 0
ee_count = 0
coverage = 0

for row in Dval:

    question, answer = row["question"], row["answer"]

    pred = prediction_sets(question, beta_star, q_hat, tokenizer, model, template, B)

    has_ee = "EE" in pred
    seen_preds = [normalize_answer(x) for x in pred if x != "EE"]


    avg_setsize += len(seen_preds)
    if has_ee:
        ee_count += 1

    is_covered = has_ee or any(a in seen_preds for a in answer["normalized_aliases"])
    if is_covered:
        coverage += 1

    if random.random() < 0.005:
        print(seen_preds, "EE" if has_ee else "", answer["aliases"][0])

print(
    f"Set size: {avg_setsize/num_examples} "
    f"EE fraction: {ee_count/num_examples} "
    f"Coverage: {coverage/num_examples}"
    f"q_hat": {q_hat}
)
