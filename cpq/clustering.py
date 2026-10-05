from collections import Counter
import numpy as np
import cpqcache


def is_equivalent(question, ansA, ansB, tokenizer, model):
    prompt = f""" Classify whether answer A expresses the same answer as answer B to the question. Do not add outside knowledge.
    Question: {question}
    A: {ansA}
    B: {ansB}
    Output exactly: MATCH, NO_MATCH. Output NOTHING else. 
    """

    cached = cpqcache.get_judgment(prompt)

    if cached is not None:
        return cached

    if model is None:
        raise ValueError("Model is None, nothing saved.")
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2)
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(generated, skip_special_tokens=True)

    label = out.strip().upper()
    print(prompt)
    print(label)

    result = label == "MATCH"
    cpqcache.put_judgment(prompt, result)
    return result


class Cluster:
    def __init__(self, question):
        self.question = question
        self.clusters = []
        self.num = 0

    def add(self, question, new_ans, tokenizer, model):
        self.num += 1
        for c in self.clusters:
            if is_equivalent(question, new_ans, c[0], tokenizer, model):
                c.append(new_ans)
                return c
        self.clusters.append([new_ans])

    def N(self, r):
        sizes = [len(x) for x in self.clusters if len(x) == r]
        return len(sizes)

    def theta_hat(self):
        return self.N(1)/self.num

    def delta_hat(self):
        return -2*self.N(2)/(self.num**2)

    def omega_hat(self, r):
        return ((r + 1)/self.num) * (self.N(r + 1)/self.N(r))

class betterCluster:
    def __init__(self, question):
        self.question = question
        self.clusters = []
        self.num = 0
        self.SGT = None

    def add(self, question, new_ans, tokenizer, model):
        self.hasSGT = None #invalid after adding
        self.num += 1
        for c in self.clusters:
            if is_equivalent(question, new_ans, c[0], tokenizer, model):
                c.append(new_ans)
                return c
        self.clusters.append([new_ans])

    def N(self, r):
        sizes = [len(x) for x in self.clusters if len(x) == r]
        return len(sizes)

    def theta_hat(self):
        return self.N(1)/self.num

    def delta_hat(self):
        return -2*self.N(2)/(self.num**2)
    
    def omega_hat(self, r):
        if self.SGT is None:
            Ni = Counter(len(c) for c in self.clusters)
            self.sgt = self.simple_good_turing(Ni, self.num)

        return self.sgt[r]
                                          
    
    def simple_good_turing(freq_counts, n, min_distinct=3, renormalize=True):
        rs = sorted(freq_counts)
        mle = {r: r / n for r in rs}

        # Need enough distinct frequencies for the log-log regression to mean anything
        if len(rs) < min_distinct:
            return mle

        k = len(rs)

        # 1. Average N_r over the gaps between nonzero frequencies (Z_r)
        Z = np.empty(k)
        for i, r in enumerate(rs):
            q = rs[i - 1] if i > 0 else 0
            t = rs[i + 1] if i < k - 1 else 2 * r - q
            Z[i] = freq_counts[r] / (0.5 * (t - q))

        # 2. Fit log Z_r = a + b log r
        b, a = np.polyfit(np.log(rs), np.log(Z), 1)
        if b > -1:                      # fit is invalid per Gale & Sampson
            return mle

        def S(r):                       # smoothed N_r
            return np.exp(a + b * np.log(r))

        # 3. Turing estimate while it differs significantly from the smoothed
        #    (LGT) estimate; once they agree, or N_{r+1} is missing, use LGT.
        r_star = {}
        use_lgt = False
        for r in rs:
            lgt = (r + 1) * S(r + 1) / S(r)
            if not use_lgt and (r + 1) in freq_counts:
                Nr, Nr1 = freq_counts[r], freq_counts[r + 1]
                turing = (r + 1) * Nr1 / Nr
                sd = np.sqrt((r + 1) ** 2 * (Nr1 / Nr**2) * (1 + Nr1 / Nr))
                if abs(turing - lgt) <= 1.96 * sd:
                    use_lgt = True
                    r_star[r] = lgt
                else:
                    r_star[r] = turing
            else:
                use_lgt = True
                r_star[r] = lgt

        p = {r: r_star[r] / n for r in rs}

        # 4. Renormalize so the seen mass equals 1 - N_1/n (the missing mass
        #    that theta_hat already reports)
        if renormalize:
            seen = sum(freq_counts[r] * p[r] for r in rs)
            target = 1 - freq_counts.get(1, 0) / n
            if seen > 0:
                p = {r: v * target / seen for r, v in p.items()}

        return p