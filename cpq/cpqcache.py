"""
Disk cache for the two expensive model calls in CPQ: drawing samples and
judging whether two answers match.

Everything is stored as append-only JSONL, so a run that crashes halfway still
keeps every result it produced, and nothing is ever rewritten.

  samples.jsonl    one line per sampled answer. The n-th line for a given
                   (model, template, question) is that question's n-th sample.
  judgments.jsonl  one line per MATCH / NO_MATCH verdict.

Settings (all optional):
  CPQ_CACHE_DIR=path   where to keep the files (default: ./cpq_results)
  CPQ_CACHE=0          turn the cache off entirely (no reads, no writes)
  configure(...)       the same settings from Python, plus model_tag
"""
import hashlib
import json
import os
from collections import Counter

_cfg = {
    "dir": os.environ.get("CPQ_CACHE_DIR", "cpq_results"),
    "enabled": os.environ.get("CPQ_CACHE", "1") != "0",
    "model_tag": None,
}
_samples = None     # key -> [answer, answer, ...] in draw order
_judgments = None   # key -> bool
STATS = Counter()   # hit/miss counts, handy for checking the cache is working


def configure(cache_dir=None, enabled=None, model_tag=None):
    """Change settings. model_tag replaces the automatic model fingerprint, which
    lets you run entirely from saved results with model=None."""
    global _samples, _judgments
    if cache_dir is not None:
        _cfg["dir"] = cache_dir
    if enabled is not None:
        _cfg["enabled"] = enabled
    if model_tag is not None:
        _cfg["model_tag"] = model_tag
    _samples = _judgments = None  # force a reload from the (possibly new) directory


def _model_id(model):
    """Fingerprint of everything about the model that changes its outputs."""
    if _cfg["model_tag"] is not None:
        return _cfg["model_tag"]
    if model is None:
        raise RuntimeError(
            "model is None and no model_tag is set, so the cache key can't be built. "
            "Call cpq_cache.configure(model_tag='...') to run from saved results "
            "without a model."
        )
    name = (
        getattr(model, "name_or_path", None)
        or getattr(getattr(model, "config", None), "_name_or_path", None)
        or type(model).__name__
    )
    gc = getattr(model, "generation_config", None)
    gen = {k: getattr(gc, k, None) for k in ("do_sample", "temperature", "top_p", "top_k")}
    return f"{name}|{json.dumps(gen, sort_keys=True)}"


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read(filename):
    path = os.path.join(_cfg["dir"], filename)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # last line may be cut off if a run was killed mid-write


def _append(filename, record):
    os.makedirs(_cfg["dir"], exist_ok=True)
    with open(os.path.join(_cfg["dir"], filename), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _samples_db():
    global _samples
    if _samples is None:
        _samples = {}
        for r in _read("samples.jsonl"):
            _samples.setdefault(r["key"], []).append(r["value"])
    return _samples


def _judgments_db():
    global _judgments
    if _judgments is None:
        _judgments = {r["key"]: r["value"] for r in _read("judgments.jsonl")}
    return _judgments


# ---- samples -----------------------------------------------------------------

def get_sample(model, template, question, idx):
    """The idx-th saved sample for this question, or None if we don't have it yet."""
    if not _cfg["enabled"]:
        return None
    saved = _samples_db().get(_key("sample", _model_id(model), template, question), [])
    if idx < len(saved):
        STATS["sample_hit"] += 1
        return saved[idx]
    STATS["sample_miss"] += 1
    return None


def put_sample(model, template, question, idx, answer):
    if not _cfg["enabled"]:
        return
    key = _key("sample", _model_id(model), template, question)
    saved = _samples_db().setdefault(key, [])
    if idx != len(saved):
        return  # only ever extend the list in order
    saved.append(answer)
    _append("samples.jsonl", {"key": key, "idx": idx, "value": answer, "question": question})


# ---- judgments ---------------------------------------------------------------

def get_judgment(model, prompt):
    """True/False if this exact prompt was judged before, else None."""
    if not _cfg["enabled"]:
        return None
    hit = _judgments_db().get(_key("judge", _model_id(model), prompt))
    STATS["judge_hit" if hit is not None else "judge_miss"] += 1
    return hit


def put_judgment(model, prompt, is_match):
    if not _cfg["enabled"]:
        return
    key = _key("judge", _model_id(model), prompt)
    _judgments_db()[key] = is_match
    _append("judgments.jsonl", {"key": key, "value": is_match, "prompt": prompt})