import numpy as np
from clustering import Cluster, is_equivalent
import re
import string

def sample(question, tokenizer, model, template):
    messages = [
        {
            "role": "user",
            "content": template + question
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()



def query(question, beta, tokenizer, model, template, max_t):
    clusters = Cluster(question)
    t = 0
    while t < max_t:
        s = sample(question, tokenizer, model, template)
        clusters.add(question, s, tokenizer, model)
        t += 1
        if (t>=2 and clusters.delta_hat() > beta):
            break

    return clusters, t

def find_beta(B, data, model, tokenizer, template): #finish this sometime
    return 0.01


def match_gold_to_cluster(clusters, question, aliases, tokenizer, model):
    for c in clusters.clusters:
        for alias in aliases:
            if is_equivalent(question, alias, c[0], tokenizer, model):
                return len(c)
    return None #represents EE

def compute_qhat(B, alpha, beta_star, data, tokenizer, model, template):
    scores = []
    for row in data:
        question = row["quetion"]
        answer = row["answer"]
        gold_aliases = answer["aliases"]
        clusters, t = query(question, beta_star, tokenizer, model, template, 2*B)
        r = match_gold_to_cluster(clusters, question, gold_aliases, tokenizer, model)
        if (r is None):
            scores.append(2 - clusters.theta_hat())
        else:
            scores.append(1 - clusters.omega_hat(r))
    
    scores.append(float("inf"))
    quantile = 1 - alpha
    return np.quantile(scores, quantile)

def prediction_sets(question, beta_star, qhat, tokenizer, model, template, B):
    clusters, t = query(question, beta_star, tokenizer, model, template, 2*B)
    pred = []
    for c in clusters.clusters:
        r = len(c)
        score = 1 - clusters.omega_hat(r)
        if score <= qhat:
            pred.append(c[0])
    
    EEscore = 2 - clusters.theta_hat()
    if EEscore <= qhat:
        pred.append("EE")
    
    return pred



def normalize_answer(s: str) -> str:
    """Exact normalization used by the official TriviaQA evaluation."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(
            handle_punc(
                lower(
                    replace_underscore(s)
                )
            )
        )
    ).strip()
