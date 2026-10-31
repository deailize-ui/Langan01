import gc
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from llama_cpp import Llama


APP_DATA = Path.home() / ".langan"
MODEL_DIR = APP_DATA / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Default model labels shown in Langan's Admin pages map to these files.
# Place the GGUF files in %USERPROFILE%\.langan\models\
DEFAULT_MODEL_FILES = {
    "logic": {
        "label": "SmolLM2-360M-Instruct-Q5_0",
        "filename": "smol.gguf",
        "ollama_blob": (
            Path.home()
            / ".ollama"
            / "models"
            / "blobs"
            / "sha256-c524b6c7dd36a91c5930ebc4dfee07858a1e424ee9fcb3c9ee3ddc89b23714f5"
        ),
        "context": 4096,
        "config_key": "smol_model",
    },
    "planning": {
        "label": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "filename": "qwen.gguf",
        "ollama_blob": (
            Path.home()
            / ".ollama"
            / "models"
            / "blobs"
            / "sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515"
        ),
        "context": 4096,
        "config_key": "planning_model",
    },
}

ALLOWED_VERDICTS = (
    "VALID",
    "INVALID",
    "CONDITIONAL",
    "INSUFFICIENT",
    "CONTRADICTORY",
)


class LanganAI:
    def __init__(self):
        self.model = None
        self.active_role = None
        self.current_config = {}

    # --------------------------------------------------
    # MODEL MANAGEMENT
    # --------------------------------------------------

    def configured_model_path(self, role, config=None):
        """Resolve an Admin-selected model file or the default model."""
        if role not in DEFAULT_MODEL_FILES:
            raise ValueError(f"Unknown model role: {role}")

        settings = DEFAULT_MODEL_FILES[role]
        config = config if isinstance(config, dict) else self.current_config
        configured_value = str(
            config.get(settings["config_key"], settings["label"])
        ).strip()

        default_names = {
            settings["label"].casefold(),
            settings["filename"].casefold(),
        }

        if not configured_value or configured_value.casefold() in default_names:
            selected_path = MODEL_DIR / settings["filename"]
            allow_ollama_fallback = True
        else:
            candidate = Path(configured_value).expanduser()
            if candidate.is_absolute():
                selected_path = candidate
            else:
                selected_path = MODEL_DIR / candidate
            allow_ollama_fallback = False

        if selected_path.is_file():
            return selected_path

        if allow_ollama_fallback and settings["ollama_blob"].is_file():
            return settings["ollama_blob"]

        raise FileNotFoundError(
            f"The {role} model could not be found.\n\n"
            f"Admin model setting: {configured_value}\n"
            f"Checked: {selected_path}\n\n"
            "For the default setup, place the model file here:\n"
            f"{MODEL_DIR / settings['filename']}\n\n"
            "For a custom model, enter its filename (inside the models "
            "folder) or its full file path in Admin."
        )

    def unload_model(self):
        if self.model is not None:
            model_to_close = self.model
            self.model = None
            self.active_role = None
            try:
                model_to_close.close()
            finally:
                del model_to_close
                gc.collect()
        else:
            self.active_role = None
            gc.collect()

    def load_model(self, role, config=None):
        if role not in DEFAULT_MODEL_FILES:
            raise ValueError(f"Unknown model role: {role}")

        path = self.configured_model_path(role, config)

        if (
            self.active_role == role
            and self.model is not None
            and getattr(self, "active_model_path", None) == str(path)
        ):
            return

        self.unload_model()

        settings = DEFAULT_MODEL_FILES[role]
        self.model = Llama(
            model_path=str(path),
            n_ctx=settings["context"],
            n_threads=4,
            n_batch=64,
            use_mmap=True,
            verbose=False,
        )

        self.active_role = role
        self.active_model_path = str(path)

    def ask_model(
        self,
        role,
        system_prompt,
        user_message,
        max_tokens=256,
        config=None,
    ):
        self.load_model(role, config)

        result = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )

        return result["choices"][0]["message"]["content"].strip()

    # --------------------------------------------------
    # QWEN WEB ENGINE
    # --------------------------------------------------

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
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            answer = data.get("answer")

            if answer:
                results.append("SEARCH SUMMARY:\n" + answer)

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
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "The configured website URL is invalid."

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LanganAI/1.0"},
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                content = response.read(30000).decode(
                    "utf-8", errors="replace"
                )

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

    # --------------------------------------------------
    # LOGIC ENGINE RESPONSE HANDLING
    # --------------------------------------------------

    def extract_action(self, text):
        match = re.search(
            r"ENGINE_ACTION\s*:\s*([A-Z_]+)",
            text.upper(),
        )

        return match.group(1) if match else ""

    def extract_verdict(self, text):
        pattern = (
            r"\b(?:FINAL\s+)?VERDICT\s*:\s*"
            r"(VALID|INVALID|CONDITIONAL|"
            r"INSUFFICIENT|CONTRADICTORY)\b"
        )

        match = re.search(pattern, text.upper())

        if match:
            return match.group(1)

        return "INSUFFICIENT"

    # --------------------------------------------------
    # QWEN FINAL EXPLANATION
    # --------------------------------------------------

    def explain_result(self, question, verdict_text, config):
        verdict = self.extract_verdict(verdict_text)

        system_prompt = (
            "You are Langan's explanation assistant, powered by Qwen.\n\n"
            "Your only responsibility is to explain the Logic Engine's "
            "result. You MUST preserve the exact verdict supplied by "
            "Smol. You are not allowed to make a new logical decision, "
            "change, upgrade, downgrade or reverse the verdict, invent "
            "supporting evidence, or present assumptions as established "
            "facts. If the reasoning is unclear, say so."
        )

        evidence_prompt = config.get("evidence_prompt", "")

        user_message = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"SMOL'S COMPLETE RESPONSE:\n{verdict_text}\n\n"
            f"LOCKED SMOL VERDICT: {verdict}\n\n"
            f"EXPLANATION GUIDANCE:\n{evidence_prompt}\n\n"
            "Explain Smol's reasoning in clear language. Preserve the "
            "locked verdict exactly. Mention uncertainty and limitations. "
            "Do not introduce a new logical conclusion."
        )

        explanation = self.ask_model(
            "planning",
            system_prompt,
            user_message,
            max_tokens=160,
            config=config,
        )

        return f"LANGAN VERDICT: {verdict}\n\n{explanation}"

    # --------------------------------------------------
    # MAIN QWEN / SMOL WORKFLOW
    # --------------------------------------------------

    def generate_conversation(
        self,
        question,
        config,
        progress_callback=None,
    ):
        config = config if isinstance(config, dict) else {}
        self.current_config = config

        def report(message):
            if progress_callback:
                progress_callback(message)

        administrator_rules = config.get("logic_prompt", "").strip()

        logic_prompt = (
            "You are Langan's Logic Engine, powered by Smol.\n\n"
            "Apply the administrator's Logic Engine rules below. "
            "Do not invent premises or facts. Treat research material "
            "as unverified until its support is assessed. Qwen does "
            "not decide your verdict.\n\n"
            "ADMINISTRATOR'S LOGIC RULES:\n"
            + (
                administrator_rules
                if administrator_rules
                else "Evaluate premises carefully. Identify missing "
                     "information, invalid inferences, contradictions "
                     "and uncertainty."
            )
            + "\n\nRESPONSE PROTOCOL:\n"
            "If essential information is missing, begin your response "
            "with ENGINE_ACTION: REQUEST_PREMISES and specify what is "
            "missing.\n"
            "If you can reach a result, begin with "
            "ENGINE_ACTION: FINAL_RESULT and include exactly one line "
            "in this format:\n"
            "VERDICT: VALID, INVALID, CONDITIONAL, INSUFFICIENT, "
            "or CONTRADICTORY.\n"
            "Use only one of the five allowed verdict labels."
        )

        try:
            # STAGE 1: QWEN PREPARES THE PROBLEM.
            report("Qwen is preparing the problem...")

            planning_prompt = (
                "You are the Langan Planning Engine powered by Qwen.\n\n"
                "Prepare the user's problem for independent evaluation "
                "by Smol. Follow these administrator instructions:\n"
                + config.get("premise_prompt", "")
                + "\n\nSeparate facts, definitions, assumptions, "
                  "unknowns and proposed conclusions. Do not invent "
                  "facts. Do not claim information is verified unless "
                  "reliable evidence supports that claim. Identify "
                  "information that may need research. Return a "
                  "structured problem submission."
            )

            plan = self.ask_model(
                "planning",
                planning_prompt,
                question,
                max_tokens=160,
                config=config,
            )

            # STAGE 2: SMOL EVALUATES.
            report("Smol is evaluating the premises...")

            evaluation = self.ask_model(
                "logic",
                logic_prompt,
                (
                    f"ORIGINAL QUESTION:\n{question}\n\n"
                    f"QWEN PROBLEM SUBMISSION:\n{plan}\n\n"
                    "Apply your logic rules. Return a verdict or "
                    "request specific missing premises."
                ),
                max_tokens=192,
                config=config,
            )

            # STAGE 3: ONE RESEARCH ROUND IF REQUIRED.
            final_evaluation_reached = False
            max_research_rounds = 1

            for round_number in range(max_research_rounds):

                if self.extract_action(evaluation) != "REQUEST_PREMISES":
                    final_evaluation_reached = True
                    break

                report(
                    f"Qwen is researching missing information "
                    f"(round {round_number + 1})..."
                )

                research_results = self.research(evaluation, config)

                self.unload_model()

                report("Qwen is preparing updated premises...")

                followup_prompt = (
                    "You are Langan's Planning Engine. The Logic Engine "
                    "has requested additional information. Review the "
                    "request and research material. Separate sourced "
                    "statements, assumptions and unresolved questions. "
                    "Research may be incomplete, inaccurate or irrelevant. "
                    "Do not treat it as automatically true. Do not invent "
                    "sources or facts. Do not make the logical decision. "
                    "Return an updated problem submission for Smol."
                )

                updated_plan = self.ask_model(
                    "planning",
                    followup_prompt,
                    (
                        f"ORIGINAL QUESTION:\n{question}\n\n"
                        f"PREVIOUS PLAN:\n{plan}\n\n"
                        f"SMOL'S REQUEST:\n{evaluation}\n\n"
                        f"UNVERIFIED RESEARCH MATERIAL:\n{research_results}"
                    ),
                    max_tokens=160,
                    config=config,
                )

                self.unload_model()

                report("Smol is re-evaluating the premises...")

                previous_evaluation = evaluation

                evaluation = self.ask_model(
                    "logic",
                    logic_prompt,
                    (
                        f"ORIGINAL QUESTION:\n{question}\n\n"
                        f"UPDATED QWEN SUBMISSION:\n{updated_plan}\n\n"
                        f"PREVIOUS LOGIC RESPONSE:\n"
                        f"{previous_evaluation}\n\n"
                        "Independently apply your logic rules to the "
                        "updated premises. Do not accept claims merely "
                        "because Qwen included them. Return a final "
                        "verdict or request additional premises."
                    ),
                    max_tokens=192,
                    config=config,
                )

                plan = updated_plan

            # If Smol still requests premises after the research round,
            # report INSUFFICIENT rather than guessing.
            if (
                not final_evaluation_reached
                and self.extract_action(evaluation) == "REQUEST_PREMISES"
            ):
                evaluation = (
                    "ENGINE_ACTION: FINAL_RESULT\n"
                    "VERDICT: INSUFFICIENT\n\n"
                    "The Logic Engine still requires additional "
                    "information after the permitted research round. "
                    "A reliable determination cannot yet be made.\n\n"
                    "Last Logic Engine response:\n"
                    + evaluation
                )

            # A missing verdict is also safely converted to INSUFFICIENT.
            verdict = self.extract_verdict(evaluation)

            if verdict == "INSUFFICIENT" and not re.search(
                r"\bVERDICT\s*:\s*INSUFFICIENT\b",
                evaluation,
                re.I,
            ):
                evaluation = (
                    "ENGINE_ACTION: FINAL_RESULT\n"
                    "VERDICT: INSUFFICIENT\n\n"
                    "Smol did not provide a recognized verdict. "
                    "No stronger conclusion has been assumed.\n\n"
                    + evaluation
                )

            # STAGE 4: QWEN EXPLAINS WITHOUT OWNING THE VERDICT.
            report("Qwen is preparing the final explanation...")

            self.unload_model()

            return self.explain_result(question, evaluation, config)

        finally:
            self.unload_model()

    def close(self):
        self.unload_model()
