import torch
import torch.nn.functional as F
from cog import BasePredictor, Input
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from tiktoken import get_encoding

from model_architectures.alibi_arch import alibi_GPT
from model_architectures.fire_arch import fire_GPT
from model_architectures.kerple_arch import kerple_GPT
from model_architectures.learnedPE_arch import learned_pe_GPT
from model_architectures.rope_arch import rope_GPT
from model_architectures.sinusoidal_arch import sinusoidal_GPT


REPO_ID = "thillsss/848k-models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_MAP = {
    "ALIBI": (alibi_GPT, "final_alibi_model.pth"),
    "FIRE": (fire_GPT, "final_fire_model.pth"),
    "Kerple": (kerple_GPT, "final_kerple_model.pth"),
    "Learned PE": (learned_pe_GPT, "final_learned_pe_model.pth"),
    "RoPE": (rope_GPT, "final_rope_model.pth"),
    "Sinusoidal": (sinusoidal_GPT, "final_sinusoidal_model.pth"),
}


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


def _clean_state_dict(sd):
    # Strip common prefixes introduced by DDP or Torch.compile tracing.
    return {
        k.replace("_orig_mod.", "").replace("module._orig_mod.", "").replace("module.", ""): v
        for k, v in sd.items()
    }


class Predictor(BasePredictor):
    def setup(self):
        self.tokenizer = get_encoding("gpt2")
        self.model_cache = {}

    def _load_model(self, model_name: str):
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        model_class, model_filename = MODEL_MAP[model_name]
        model_path = hf_hub_download(repo_id=REPO_ID, filename=model_filename)

        config = GPTConfig(vocab_size=50304)
        model = model_class(config).to(DEVICE)

        state_dict = torch.load(model_path, map_location=DEVICE)
        state_dict = _clean_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model.eval()

        self.model_cache[model_name] = model
        return model

    def predict(
        self,
        prompt: str = Input(description="Full conversation context to continue."),
        model_name: str = Input(choices=list(MODEL_MAP.keys()), default="Sinusoidal"),
        max_new_tokens: int = Input(description="Number of tokens to generate.", default=50, ge=1, le=200),
        temperature: float = Input(description="Softmax temperature.", default=1.0, ge=0.1, le=2.0),
    ) -> str:
        model = self._load_model(model_name)

        tokens = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(tokens)[0][:, -1, :]
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
                if next_token.item() == self.tokenizer.eot_token:
                    break

        return self.tokenizer.decode(tokens[0].tolist())
