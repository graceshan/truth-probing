"""Model loading and activation extraction. Runs on Colab GPU (or MPS locally)."""

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

HF_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TL_MODEL_NAME = "qwen2.5-1.5b"


def load_model(device: str | None = None) -> HookedTransformer:
    """Load base (non-instruct) Qwen2.5-1.5B via HF handoff into TransformerLens.

    from_pretrained_no_processing takes the HF weights as-is (no LayerNorm
    folding etc.), so resid_post matches the original model exactly.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    return HookedTransformer.from_pretrained_no_processing(
        TL_MODEL_NAME, hf_model=hf_model, dtype=torch.float16, device=device
    )


def statement_logprob(model: HookedTransformer, statement: str) -> float:
    """Total log-probability the model assigns to `statement`'s tokens (teacher-forced)."""
    tokens = model.to_tokens(statement)
    with torch.no_grad():
        logits = model(tokens)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    tgt = tokens[0, 1:]  # predict token i+1 from position i
    lp = logprobs[0, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp.sum().item()


def extract_acts(model: HookedTransformer, texts: list[str]) -> np.ndarray:
    """Resid_post at the last token, every layer, for each text.

    Returns array of shape [len(texts), n_layers, d_model].
    """
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    acts = np.zeros((len(texts), n_layers, d_model), dtype=np.float32)
    with torch.no_grad():
        for i, t in enumerate(tqdm(texts)):
            # One statement at a time (not batched) to keep peak memory low
            # on the T4; caching every resid_post is otherwise expensive.
            _, cache = model.run_with_cache(t, names_filter=lambda n: "resid_post" in n)
            for L in range(n_layers):
                acts[i, L] = cache["resid_post", L][0, -1, :].float().cpu().numpy()
            del cache  # avoid accumulating cached activations across statements
    return acts
