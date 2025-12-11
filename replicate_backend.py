import os
from typing import Dict

import replicate


# Mapping from UI model name to Replicate model version.
# Configure per model via environment variables, falling back to a shared default.
MODEL_VERSION_MAP: Dict[str, str] = {
    "ALIBI": os.getenv("REPLICATE_ALIBI_VERSION"),
    "FIRE": os.getenv("REPLICATE_FIRE_VERSION"),
    "Kerple": os.getenv("REPLICATE_KERPLE_VERSION"),
    "Learned PE": os.getenv("REPLICATE_LEARNED_PE_VERSION"),
    "RoPE": os.getenv("REPLICATE_ROPE_VERSION"),
    "Sinusoidal": os.getenv("REPLICATE_SINUSOIDAL_VERSION"),
}

DEFAULT_VERSION = os.getenv("REPLICATE_MODEL_VERSION")


def _get_model_version(model_name: str) -> str:
    model_version = MODEL_VERSION_MAP.get(model_name) or DEFAULT_VERSION
    if not model_version:
        raise RuntimeError(
            f"Replicate model version not configured for '{model_name}'. "
            "Set REPLICATE_MODEL_VERSION to a shared version or "
            f"REPLICATE_{model_name.upper().replace(' ', '_')}_VERSION for per-model overrides."
        )
    return model_version


def call_replicate(prompt: str, model_name: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
    """Call a Replicate-hosted predictor. Assumes the predictor accepts
    prompt/model_name/max_new_tokens/temperature inputs."""
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN is not set.")

    version = _get_model_version(model_name)
    client = replicate.Client(api_token=api_token)

    output = client.run(
        version,
        input={
            "prompt": prompt,
            "model_name": model_name,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    )

    if isinstance(output, str):
        return output

    # The Replicate SDK can return a generator/iterable for streaming models.
    try:
        return "".join(output)
    except TypeError:
        return str(output)
