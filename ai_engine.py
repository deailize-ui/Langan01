
import gc
import shutil
from pathlib import Path

from llama_cpp import Llama

APP_DATA = Path.home() / ".langan"
MODEL_DIR = APP_DATA / "models"
OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"

APP_DATA.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "logic": {
        "source": "sha256-c524b6c7dd36a91c5930ebc4dfee07858a1e424ee9fcb3c9ee3ddc89b23714f5",
        "file": "SmolLM2-360M-Instruct-Q5_0.gguf",
        "context": 2048,
    },
    "planning": {
        "source": "sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515",
        "file": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "context": 2048,
    },
}


class LanganAI:
    def __init__(self):
        self.model = None
        self.active_role = None

    def get_model_path(self, role):
        if role not in MODELS:
            raise ValueError(f"Unknown model role: {role}")

        settings = MODELS[role]
        destination = MODEL_DIR / settings["file"]
        source = OLLAMA_BLOBS / settings["source"]

        if destination.exists():
            with destination.open("rb") as f:
                if f.read(4) == b"GGUF":
                    return str(destination)

            destination.unlink()

        if not source.exists():
            raise FileNotFoundError(
                f"Model file not found: {source}\n"
                "Make sure Ollama and the model are installed "
                "for this Windows user."
            )

        with source.open("rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(
                    f"The source file is not a valid GGUF model: {source}"
                )

        shutil.copy2(source, destination)

        return str(destination)

    def load_model(self, role):
        if self.active_role == role and self.model is not None:
            return

        self.unload_model()

        settings = MODELS[role]
        model_path = self.get_model_path(role)

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
