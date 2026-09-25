import numpy as np
from clustering import Cluster, is_equivalent

def sample(question, tokenizer, model, template):
    inputs = tokenizer(template + question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs, skip_special_tokens=True).strip()



def query(question, beta, tokenizer, model, template, max_t):
    clusters = Cluster(question)
    t = 0
    while t < max_t:
        sample = sample(question, tokenizer, model, template)
        clusters.add(sample, tokenizer, model)
        t += 1
        if (t>=2 and clusters.delta_hat() > beta):
            break

    return clusters, t

def find_beta(B, D, model, tokenizer, template): #finish this sometime
    return 0.01


def match_gold_to_cluster(clusters, question, aliases, tokenizer, model):
    for c in clusters.clusters:
        for alias in aliases:
            if is_equivalent(question, alias, c[0], tokenizer, model):
                return len(c)
    return None #represents EE

def compute_qhat(data, beta_star, tokenizer, model, template, B, alpha):
    scores = []
    n = len()
    for question, gold_aliases in data:
        clusters, t = query(question, beta_star, tokenizer, model, template, 2*B)
        r = match_gold_to_cluster(clusters, question, gold_aliases, tokenizer, model)
        if (r is None):
            scores.append(2 - clusters.theta_hat())
        else:
            scores.append(1 - clusters.omega_hat(r))
    
    scores.append(float("inf"))
    n = len(scores)
    quantile = np.ceil( (n + 1) * (1 - alpha))/n
    return np.quantile(scores, quantile)

def prediction_sets(question, beta_star, tokenizer, model, template, alpha, qhat, B):
    clusters, t = query(question, beta_star, tokenizer, model, template, 2*B)
    pred = []
    for c in clusters:
        r = len(clusters)
        score = 1 - clusters.omega_hat(r)
        if score <= qhat:
            pred.append[c[0]]
    
    EEscore = 2 - clusters.theta_hat
    if EEscore <= qhat:
        pred.append("EE")
    
    return pred










        
