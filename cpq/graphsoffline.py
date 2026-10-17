import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from cpqtools import find_beta, compute_qhat, prediction_sets, normalize_answer
from clustering import is_equivalent

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

        is_covered = False
        for a in answer["normalized_aliases"]:
            for s in seen_preds:
                if is_equivalent(question, a, s, tokenizer, model):
                    is_covered = True
                    break
        is_covered = is_covered or has_ee
        if is_covered:
            coverage += 1

        if random.random() < 0.02:
            print(alpha, ": ", seen_preds, "EE" if has_ee else "", answer["aliases"][0])

    return (avg_setsize/num_examples, ee_count/num_examples, coverage/num_examples, q_hat)


model_id = "Qwen/Qwen3-8B"

BATCH_SIZE = 1
MAX_NEW_TOKENS = 10

model = None #don't need since we are using cached samples
tokenizer = None 
token = "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"
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
Dcal2 = ds.select(range(1000, 1200))
Dval = ds.select(range(2000, 2200))

B = 15
alpha = 0.3

template = (
    "Answer this trivia question with only the answer."
    "No explanation. Provide the shortest answer you can, without restating any part of the question."
    "Question: "
)

beta_star = find_beta(B, Dcal1, None, None, template)
print("beta", beta_star)

alpha_trials = np.arange(0.05, 0.75, 0.05).tolist()
one_minus_alpha = [1 - a for a in alpha_trials]
eefrac = []
coverage = []
setsize = []
q_hat = []
for alpha in alpha_trials:
    res = runCPQ(Dval, Dcal2, tokenizer, model, template, B, beta_star, alpha)
    setsize.append(res[0])
    eefrac.append(res[1])
    coverage.append(res[2])
    q_hat.append(res[3])
plots = [
    (setsize, "Set Size", "1 - alpha vs Set Size", "graphs/HighMin1malpha_EE_vs_setsizeV2.png"),
    (eefrac, "EE Fraction", "1 - alpha vs EE Fraction", "graphs/HighMin1malpha_EE_vs_eefracV2.png"),
    (coverage, "Coverage", "1 - alpha vs Coverage", "graphs/HighMin1malpha_EE_vs_coverageV2.png"),
]
print(q_hat)

for y, ylabel, title, filename in plots:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(one_minus_alpha, y, marker="o", linestyle="-",  alpha=0.8)

    # Target coverage line (y = x): coverage should sit at or above it
    if ylabel == "Coverage":
        ax.plot(one_minus_alpha, one_minus_alpha, "k--", alpha=0.5, label="Target (1 - alpha)")
        ax.legend()

    ax.set_xlabel("1 - alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
