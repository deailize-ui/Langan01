
import gc
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


APP_DATA = Path.home() / ".langan"
MODEL_DIR = APP_DATA / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "logic": {
        "repo": "tensorblock/SmolLM2-360M-Instruct-GGUF",
        "file": "SmolLM2-360M-Instruct-Q5_0.gguf",
        "context": 2048,
    },
    "planning": {
        "repo": "tensorblock/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "context": 2048,
    },
}


class LanganAI:
    def __init__(self):
        self.model = None
        self.active_role = None

    def download_model(self, role):
        if role not in MODELS:
            raise ValueError("Unknown model role: " + role)

        settings = MODELS[role]

        return hf_hub_download(
            repo_id=settings["repo"],
            filename=settings["file"],
            local_dir=str(MODEL_DIR),
        )

    def load_model(self, role):
        if self.active_role == role and self.model is not None:
            return

        self.unload_model()

        settings = MODELS[role]
        model_path = self.download_model(role)

        self.model = Llama(
            model_path=model_path,
            n_ctx=settings["context"],
            n_threads=2,
            n_batch=128,
            use_mmap=True,
            verbose=False,
        )

        self.active_role = role

    def unload_model(self):
        if self.model is not None:
            self.model.close()
            self.model = None
            self.active_role = None
            gc.collect()

    def generate(self, message, role="logic", system_prompt=None):
        self.load_model(role)

        if not system_prompt:
            system_prompt = (
                "You are Langan, a helpful AI assistant. "
                "Be clear, accurate, thoughtful and honest. "
                "Do not invent facts."
            )

        result = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=512,
            temperature=0.7,
        )

        return result["choices"][0]["message"]["content"].strip()
