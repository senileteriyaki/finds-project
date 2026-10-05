import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
"""
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
"""

from cpqtools import find_beta, compute_qhat, prediction_sets, normalize_answer

def runCPQ(Dval, Dcal2, tokenizer, model, template, B, beta_star, alpha):

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

    return (avg_setsize/num_examples, ee_count/num_examples, coverage/num_examples)


model_id = "Qwen/Qwen3-8B"

BATCH_SIZE = 1
MAX_NEW_TOKENS = 10

model = None #don't need since we are using cached samples
tokenizer = None 
"""
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
"""

dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split="validation",
)
ds = dataset.select_columns(["question", "answer"]).shuffle(seed=42)

Dcal1 = ds.select(range(0, 100))
Dcal2 = ds.select(range(1000, 1100))
Dval = ds.select(range(2000, 2100))

B = 20
alpha = 0.3

template = (
    "Answer this trivia question with only the answer."
    "No explanation. Provide the shortest answer you can, without restating any part of the question."
    "Question: "
)

beta_star = find_beta(B, Dcal1, None, None, template)

alpha_trials = np.arange(0.05, 0.40, 0.05).tolist()
eefrac = []
coverage = []
setsize = []

for alpha in alpha_trials:
    res = runCPQ(Dval, Dcal2, tokenizer, model, template, B, beta_star, alpha)
    setsize.append(res[0])
    eefrac.append(res[1])
    coverage.append(res[2])

plots = [
    (setsize, "Set Size", "alpha vs Set Size", "x_vs_setsize.png"),
    (eefrac, "EE Fraction", "alpha vs EE Fraction", "x_vs_eefrac.png"),
    (coverage, "Coverage", "alpha vs Coverage", "x_vs_coverage.png"),
]

for y, ylabel, title, filename in plots:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(alpha_trials, y, s=30, alpha=0.8)

    ax.set_xlabel("Alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)