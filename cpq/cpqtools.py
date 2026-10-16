import numpy as np
from clustering import Cluster, FrequencyCluster, is_equivalent, betterCluster
import cpqcache
import re
import string


def sample(question, tokenizer, model, template, sample_num):

    saved = cpqcache.get_sample(template, question, sample_num)
    if saved is not None:
        return saved

    if model is None:
        raise RuntimeError("No saved sample, No model.")

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
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    cpqcache.put_sample(template, question, sample_num, answer)
    return answer


def query(question, beta, tokenizer, model, template, max_t, min_t=15): ##FREQUENCY BASED version
    clusters = Cluster(question)
    t = 0
    while t < max_t:
        s = sample(question, tokenizer, model, template, sample_num=t)
        clusters.add(question, s, tokenizer, model)
        t += 1
        if (t >= min_t and clusters.delta_hat() > beta):
            break

    return clusters, t


def find_beta(B, data, model, tokenizer, template, min_t=8):  # finish this sometime
    all_delta_hats = []
    for row in data:
        delta_hats = []
        question = row["question"]
        clusters = Cluster(question)

        for t in range(2*B):
            s = sample(question, tokenizer, model, template, sample_num=t)
            clusters.add(question, s, tokenizer, model)
            if t + 1 >= min_t:
                delta_hats.append(clusters.delta_hat())
        all_delta_hats.append(delta_hats)
    
    def avg_t(beta):
        sum = 0
        for delta_hats in all_delta_hats:
            T = 2*B
            for i in range(0, len(delta_hats)):
                if delta_hats[i] > beta:
                    T = (i + min_t) #because we started range at min_t
                    break
            sum += T
        return sum / len(all_delta_hats)
    
        
    betas = -np.logspace(-3, -1, 30)
    possible_betas = [b for b in betas if avg_t(b) <= B]
    print("Average T for each beta: ", {b: avg_t(b) for b in betas})
    print("Possible betas: ", possible_betas)
    if possible_betas:
        return max(possible_betas)
    else:
        print("No suitable beta found. Returning default value of -0.1.")
        return -0.1 #default value 



def match_gold_to_cluster(clusters, question, aliases, tokenizer, model):
    for c in clusters.clusters:
        for alias in aliases:
            if is_equivalent(question, alias, c[0], tokenizer, model):
                return len(c)
    return None  # represents EE


def compute_qhat(B, alpha, beta_star, data, tokenizer, model, template):
    scores = []
    for row in data:
        question = row["question"]
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
    return np.quantile(scores, quantile, method="higher")


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
        return ["EE"]

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
