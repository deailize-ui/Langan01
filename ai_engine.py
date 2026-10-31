
import gc
import json
import re
import urllib.request
from pathlib import Path

from llama_cpp import Llama


APP_DATA = Path.home() / ".langan"
MODEL_DIR = APP_DATA / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Smol is exclusively the Logic Engine.
# Qwen handles all planning, research and explanations.
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

    # --------------------------------------------------
    # MODEL MANAGEMENT
    # --------------------------------------------------

    def model_path(self, role):
        settings = LOCAL_MODELS[role]

        local_copy = MODEL_DIR / settings["filename"]

        if local_copy.exists():
            return local_copy

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
            try:
                self.model.close()
            finally:
                self.model = None
                self.active_role = None
                gc.collect()

    def load_model(self, role):
        if role not in LOCAL_MODELS:
            raise ValueError(
                f"Unknown model role: {role}"
            )

        if (
            self.active_role == role
            and self.model is not None
        ):
            return

        self.unload_model()

        settings = LOCAL_MODELS[role]
        path = self.model_path(role)

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
            temperature=0.2,
        )

        return (
            result["choices"][0]["message"]["content"].strip()
        )

    # --------------------------------------------------
    # QWEN WEB ENGINE
    # --------------------------------------------------

    def search_web(self, query, config):
        api_key = config.get(
            "tavily_key",
            "",
        ).strip()

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

            return (
                "\n\n".join(results)
                or "No useful results found."
            )

        except Exception as error:
            return f"Web search failed: {error}"

    def fetch_website(self, url):

        if not url.startswith(
            ("http://", "https://")
        ):
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

                content = response.read(
                    30000
                ).decode(
                    "utf-8",
                    errors="replace",
                )

            content = re.sub(
                r"<script.*?</script>|<style.*?</style>",
                " ",
                content,
                flags=re.I | re.S,
            )

            content = re.sub(
                r"<[^>]+>",
                " ",
                content,
            )

            content = re.sub(
                r"\s+",
                " ",
                content,
            )

            return content[:12000]

        except Exception as error:
            return f"Website retrieval failed: {error}"

    def research(self, request_text, config):

        method = config.get(
            "web_engine",
            "Tavily",
        )

        if method == "Website URL":

            url = config.get(
                "website_url",
                "",
            ).strip()

            if url:
                return self.fetch_website(url)

            return "No website URL has been configured."

        return self.search_web(
            request_text,
            config,
        )

    # --------------------------------------------------
    # LOGIC ENGINE RESPONSE HANDLING
    # --------------------------------------------------

    def extract_action(self, text):

        match = re.search(
            r"ENGINE_ACTION\s*:\s*([A-Z_]+)",
            text.upper(),
        )

        if match:
            return match.group(1)

        return ""

    def extract_verdict(self, text):

        pattern = (
            r"\b(?:FINAL\s+)?VERDICT\s*:\s*"
            r"(VALID|INVALID|CONDITIONAL|"
            r"INSUFFICIENT|CONTRADICTORY)\b"
        )

        match = re.search(
            pattern,
            text.upper(),
        )

        if match:
            return match.group(1)

        # A missing verdict must never be assumed VALID.
        return "INSUFFICIENT"

    # --------------------------------------------------
    # QWEN FINAL EXPLANATION
    # --------------------------------------------------

    def explain_result(
        self,
        question,
        verdict_text,
        config,
    ):

        verdict = self.extract_verdict(
            verdict_text
        )

        system_prompt = (
            "You are Langan's explanation assistant, "
            "powered by Qwen.\n\n"
            "Your only responsibility is to explain "
            "the Logic Engine's result.\n\n"
            "You MUST preserve the exact verdict supplied "
            "by Smol.\n\n"
            "You are NOT allowed to:\n"
            "- Make a new logical decision.\n"
            "- Change the verdict.\n"
            "- Upgrade or downgrade the verdict.\n"
            "- Reverse the conclusion.\n"
            "- Invent supporting evidence.\n"
            "- Present assumptions as established facts.\n\n"
            "If the reasoning is unclear, explain that "
            "the reasoning is unclear."
        )

        evidence_prompt = config.get(
            "evidence_prompt",
            "",
        )

        user_message = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"SMOL'S COMPLETE RESPONSE:\n{verdict_text}\n\n"
            f"LOCKED SMOL VERDICT: {verdict}\n\n"
            f"EXPLANATION GUIDANCE:\n{evidence_prompt}\n\n"
            "Explain Smol's reasoning in clear language. "
            "Preserve the locked verdict exactly. "
            "Mention uncertainty and limitations. "
            "Do not introduce a new logical conclusion."
        )

        explanation = self.ask_model(
            "planning",
            system_prompt,
            user_message,
            max_tokens=512,
        )

        # The displayed verdict comes from Smol,
        # not from Qwen's generated explanation.
        return (
            f"LANGAN VERDICT: {verdict}\n\n"
            f"{explanation}"
        )

    # --------------------------------------------------
    # MAIN QWEN / SMOL WORKFLOW
    # --------------------------------------------------

    def generate_conversation(
        self,
        question,
        config,
        progress_callback=None,
    ):

        def report(message):
            if progress_callback:
                progress_callback(message)

        try:

            # STAGE 1
            # QWEN PREPARES THE PROBLEM.

            report(
                "Qwen is preparing the problem..."
            )

            planning_prompt = (
                "You are the Langan Planning Engine "
                "powered by Qwen.\n\n"
                "Prepare the user's problem for "
                "independent evaluation by Smol.\n\n"
                "Follow these instructions:\n"
                + config.get(
                    "premise_prompt",
                    "",
                )
                + "\n\n"
                "Separate facts, definitions, assumptions, "
                "unknowns, and proposed conclusions.\n"
                "Do not invent facts.\n"
                "Do not claim that information is verified "
                "unless reliable evidence supports that claim.\n"
                "Identify information that may need research.\n"
                "Return a structured problem submission."
            )

            plan = self.ask_model(
                "planning",
                planning_prompt,
                question,
                max_tokens=512,
            )

            # STAGE 2
            # SMOL APPLIES THE LOGIC RULES.

            report(
                "Smol is evaluating the premises..."
            )

            logic_prompt = config.get(
                "logic_prompt",
                "",
            ).strip()

            if not logic_prompt:

                logic_prompt = (
                    "You are Langan's Logic Engine, "
                    "powered by Smol.\n\n"
                    "Your responsibility is logical "
                    "evaluation only.\n\n"
                    "Evaluate the premises carefully.\n"
                    "Identify missing information, invalid "
                    "inferences, contradictions, and uncertainty.\n"
                    "Never invent facts.\n"
                    "Never allow Qwen to decide your verdict.\n\n"
                    "If more information is needed, return:\n"
                    "ENGINE_ACTION: REQUEST_PREMISES\n\n"
                    "Otherwise return:\n"
                    "ENGINE_ACTION: FINAL_RESULT\n"
                    "VERDICT: VALID, INVALID, CONDITIONAL, "
                    "INSUFFICIENT, or CONTRADICTORY."
                )

            evaluation = self.ask_model(
                "logic",
                logic_prompt,
                (
                    f"ORIGINAL QUESTION:\n{question}\n\n"
                    f"QWEN PROBLEM SUBMISSION:\n{plan}\n\n"
                    "Apply your logic rules to the submission. "
                    "Return a verdict or request specific "
                    "missing premises."
                ),
                max_tokens=512,
            )

            # STAGE 3
            # QWEN RESEARCHES.
            # SMOL RE-EVALUATES.
            # MAXIMUM TWO RESEARCH ROUNDS.

            max_research_rounds = 2

            final_evaluation_reached = False

            for round_number in range(
                max_research_rounds
            ):

                action = self.extract_action(
                    evaluation
                )

                if action != "REQUEST_PREMISES":
                    final_evaluation_reached = True
                    break

                report(
                    f"Qwen is researching missing "
                    f"information (round "
                    f"{round_number + 1})..."
                )

                research_results = self.research(
                    evaluation,
                    config,
                )

                # Smol is unloaded before Qwen is loaded.
                self.unload_model()

                report(
                    "Qwen is preparing updated premises..."
                )

                followup_prompt = (
                    "You are Langan's Planning Engine.\n\n"
                    "The Logic Engine has requested "
                    "additional information.\n\n"
                    "Review the request and research material.\n"
                    "Separate sourced statements, assumptions, "
                    "and unresolved questions.\n"
                    "Research material may be incomplete, "
                    "inaccurate, or irrelevant.\n"
                    "Do not treat it as automatically true.\n"
                    "Do not invent sources or facts.\n"
                    "Do not make the logical decision.\n"
                    "Return an updated problem submission "
                    "for Smol."
                )

                updated_plan = self.ask_model(
                    "planning",
                    followup_prompt,
                    (
                        f"ORIGINAL QUESTION:\n{question}\n\n"
                        f"PREVIOUS PLAN:\n{plan}\n\n"
                        f"SMOL'S REQUEST:\n{evaluation}\n\n"
                        f"UNVERIFIED RESEARCH MATERIAL:\n"
                        f"{research_results}"
                    ),
                    max_tokens=512,
                )

                # Qwen is unloaded before Smol is loaded.
                self.unload_model()

                report(
                    "Smol is re-evaluating the premises..."
                )

                evaluation = self.ask_model(
                    "logic",
                    logic_prompt,
                    (
                        f"ORIGINAL QUESTION:\n{question}\n\n"
                        f"UPDATED QWEN SUBMISSION:\n"
                        f"{updated_plan}\n\n"
                        f"PREVIOUS LOGIC RESPONSE:\n"
                        f"{evaluation}\n\n"
                        "Independently apply your logic rules "
                        "to the updated premises. Do not accept "
                        "claims merely because Qwen included them. "
                        "Return a final verdict or request "
                        "additional premises."
                    ),
                    max_tokens=512,
                )

                plan = updated_plan

            # If Smol still requests information after
            # the allowed research rounds, report INSUFFICIENT.

            if not final_evaluation_reached:

                if (
                    self.extract_action(evaluation)
                    == "REQUEST_PREMISES"
                ):

                    evaluation = (
                        "ENGINE_ACTION: FINAL_RESULT\n"
                        "VERDICT: INSUFFICIENT\n\n"
                        "The Logic Engine still requires "
                        "additional information after the "
                        "permitted research rounds.\n"
                        "A reliable determination cannot "
                        "yet be made.\n\n"
                        "Last Logic Engine response:\n"
                        + evaluation
                    )

            # STAGE 4
            # QWEN EXPLAINS SMOL'S VERDICT.
            # QWEN DOES NOT MAKE A NEW DECISION.

            report(
                "Qwen is preparing the final explanation..."
            )

            self.unload_model()

            final_answer = self.explain_result(
                question,
                evaluation,
                config,
            )

            return final_answer

        finally:
            # Always release model memory.
            self.unload_model()

    def close(self):
        self.unload_model()
