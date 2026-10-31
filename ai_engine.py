
import gc
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from llama_cpp import Llama


APP_DATA = Path.home() / ".langan"
MODEL_DIR = APP_DATA / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# The local Ollama model blobs you already have.
# Langan can also use copies placed in ~/.langan/models.
LOCAL_MODELS = {
    "logic": {
        "filename": "smol.gguf",
        "ollama_blob": (
            Path.home()
            / ".ollama"
            / "models"
            / "blobs"
            / "sha256-c524b6c7dd36a91c5930ebc4dfee07858a1e424ee9fcb3c9ee3ddc89b23714f5"
        ),
        "context": 2048,
    },
    "planning": {
        "filename": "qwen.gguf",
        "ollama_blob": (
            Path.home()
            / ".ollama"
            / "models"
            / "blobs"
            / "sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515"
        ),
        "context": 2048,
    },
}


class LanganAI:
    def __init__(self):
        self.model = None
        self.active_role = None

    def model_path(self, role):
        settings = LOCAL_MODELS[role]

        # Preferred location for standalone Langan models.
        local_copy = MODEL_DIR / settings["filename"]

        if local_copy.exists():
            return local_copy

        # Development fallback: use the existing Ollama model file.
        ollama_file = settings["ollama_blob"]

        if ollama_file.exists():
            return ollama_file

        raise FileNotFoundError(
            f"The {role} model was not found.\n\n"
            f"Expected either:\n{local_copy}\n\n"
            f"Or:\n{ollama_file}"
        )

    def unload_model(self):
        if self.model is not None:
            self.model.close()
            self.model = None
            self.active_role = None
            gc.collect()

    def load_model(self, role):
        if role not in LOCAL_MODELS:
            raise ValueError(f"Unknown model role: {role}")

        if self.active_role == role and self.model is not None:
            return

        self.unload_model()

        path = self.model_path(role)
        settings = LOCAL_MODELS[role]

        self.model = Llama(
            model_path=str(path),
            n_ctx=settings["context"],
            n_threads=2,
            n_batch=64,
            use_mmap=True,
            verbose=False,
        )

        self.active_role = role

    def ask_model(
        self,
        role,
        system_prompt,
        user_message,
        max_tokens=512,
    ):
        self.load_model(role)

        result = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )

        return (
            result["choices"][0]["message"]["content"].strip()
        )

    def search_web(self, query, config):
        api_key = config.get("tavily_key", "").strip()

        if not api_key:
            return "No Tavily API key is configured."

        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
        }).encode("utf-8")

        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=25,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

            results = []

            answer = data.get("answer")
            if answer:
                results.append(
                    "SEARCH SUMMARY:\n" + answer
                )

            for item in data.get("results", []):
                results.append(
                    "TITLE: " + item.get("title", "")
                    + "\nURL: " + item.get("url", "")
                    + "\nCONTENT: " + item.get("content", "")
                )

            return "\n\n".join(results) or "No useful results found."

        except Exception as error:
            return f"Web search failed: {error}"

    def fetch_website(self, url):
        if not url.startswith(("http://", "https://")):
            return "The configured website URL is invalid."

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "LanganAI/1.0",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=15,
            ) as response:
                content = response.read(30000).decode(
                    "utf-8",
                    errors="replace",
                )

            # Basic removal of HTML tags.
            content = re.sub(
                r"<script.*?</script>|<style.*?</style>",
                " ",
                content,
                flags=re.I | re.S,
            )
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content)

            return content[:12000]

        except Exception as error:
            return f"Website retrieval failed: {error}"

    def research(self, request_text, config):
        method = config.get("web_engine", "Tavily")

        if method == "Website URL":
            url = config.get("website_url", "").strip()
            if url:
                return self.fetch_website(url)

            return "No website URL has been configured."

        return self.search_web(request_text, config)

    def extract_action(self, text):
        match = re.search(
            r"ENGINE_ACTION\s*:\s*([A-Z_]+)",
            text.upper(),
        )

        if match:
            return match.group(1)

        return ""

    def explain_result(self, question, verdict, config):
        system_prompt = (
            "You are Langan, the explanation and communication "
            "assistant. Explain the Logic Engine's result clearly. "
            "Preserve its verdict. Do not change its conclusion, "
            "invent evidence, or present assumptions as facts."
        )

        evidence_prompt = config.get(
            "evidence_prompt",
            "",
        )

        user_message = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"LOGIC ENGINE RESULT:\n{verdict}\n\n"
            f"EXPLANATION GUIDANCE:\n{evidence_prompt}\n\n"
            "Explain the result in clear, natural language. "
            "Mention important uncertainties and limitations."
        )

        return self.ask_model(
            "planning",
            system_prompt,
            user_message,
            max_tokens=512,
        )

    def generate_conversation(
        self,
        question,
        config,
        progress_callback=None,
    ):
        def report(message):
            if progress_callback:
                progress_callback(message)

        # STAGE 1: Qwen prepares the problem.
        report("Qwen is preparing the problem...")

        planning_prompt = (
            "You are the Langan Planning Engine powered by Qwen. "
            "Prepare the user's problem for independent logical "
            "evaluation by Smol.\n\n"
            "Follow these instructions:\n"
            + config.get("premise_prompt", "")
            + "\n\nSeparate facts, definitions, assumptions, "
            "unknowns, and proposed conclusions. Do not invent "
            "facts or pretend you have verified information. "
            "Identify information that may need research. "
            "Return a structured problem submission."
        )

        plan = self.ask_model(
            "planning",
            planning_prompt,
            question,
            max_tokens=512,
        )

        # STAGE 2: Smol evaluates the premises.
        report("Smol is evaluating the premises...")

        logic_prompt = config.get("logic_prompt", "").strip()

        if not logic_prompt:
            logic_prompt = (
                "You are Langan's Logic Engine. Evaluate the "
                "premises carefully. Identify missing information, "
                "invalid inferences, contradictions, and uncertainty. "
                "Never invent facts. If more information is needed, "
                "return ENGINE_ACTION: REQUEST_PREMISES. "
                "Otherwise return ENGINE_ACTION: FINAL_RESULT "
                "and a verdict: VALID, INVALID, CONDITIONAL, "
                "INSUFFICIENT, or CONTRADICTORY."
            )

        evaluation = self.ask_model(
            "logic",
            logic_prompt,
            (
                f"ORIGINAL QUESTION:\n{question}\n\n"
                f"QWEN PROBLEM SUBMISSION:\n{plan}\n\n"
                "Evaluate this submission. Return a clear verdict "
                "or request specific missing information."
            ),
            max_tokens=512,
        )

        # STAGE 3: Allow a limited research and re-evaluation cycle.
        max_research_rounds = 2

        for round_number in range(max_research_rounds):
            action = self.extract_action(evaluation)

            if action != "REQUEST_PREMISES":
                break

            report(
                f"Qwen is researching missing information "
                f"(round {round_number + 1})..."
            )

            research_results = self.research(
                evaluation,
                config,
            )

            report("Smol is checking the new information...")

            # Release Smol before Qwen is loaded again.
            self.unload_model()

            followup_prompt = (
                "You are the Langan Planning Engine. The Logic "
                "Engine has requested additional information. "
                "Review the request and the research results. "
                "Separate sourced information from assumptions. "
                "Do not claim that a source proves something it "
                "does not establish. Return an updated problem "
                "submission for the Logic Engine."
            )

            updated_plan = self.ask_model(
                "planning",
                followup_prompt,
                (
                    f"ORIGINAL QUESTION:\n{question}\n\n"
                    f"PREVIOUS PLAN:\n{plan}\n\n"
                    f"LOGIC ENGINE REQUEST:\n{evaluation}\n\n"
                    f"RESEARCH RESULTS:\n{research_results}"
                ),
                max_tokens=512,
            )

            report("Smol is re-evaluating the updated premises...")

            self.unload_model()

            evaluation = self.ask_model(
                "logic",
                logic_prompt,
                (
                    f"ORIGINAL QUESTION:\n{question}\n\n"
                    f"UPDATED QWEN SUBMISSION:\n{updated_plan}\n\n"
                    f"PREVIOUS LOGIC ENGINE RESPONSE:\n{evaluation}\n\n"
                    "Re-evaluate the problem. Return a final verdict "
                    "or request more information."
                ),
                max_tokens=512,
            )

            plan = updated_plan

        # STAGE 4: Qwen explains Smol's result.
        report("Qwen is preparing the final explanation...")

        self.unload_model()

        final_answer = self.explain_result(
            question,
            evaluation,
            config,
        )

        self.unload_model()

        return final_answer

    def close(self):
        self.unload_model()
