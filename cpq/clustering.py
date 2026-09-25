
from collections import Counter


def is_equivalent(ansA, ansB, tokenizer, model):
    prompt = f""" Classify whether answer A expresses the same answer as answer B.  Do not add outside knowledge.
    A: {ansA} \\
    B: {ansB}
    Output exactly: MATCH, NO_MATCH
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=4)
    out = tokenizer.decode(outputs, skip_special_tokens=True)

    return ("".join(out.lower().split()) == "match")

class Cluster:
    def __init__(self, question):
        self.question = question
        self.clusters = []
        self.num = 0
    
    def add(self, new_ans, tokenizer, model):
        for c in self.clusters:
            self.num += 1
            if is_equivalent(new_ans, c[0], tokenizer, model):
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
    
    def omega_hat(self,r):
        return ((r + 1)/self.num) * (1 + 1/self.N(r))


    

    