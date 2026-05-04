from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.settings import get_settings


def _create_openai_compatible_llm(model, temperature, api_key, base_url=None, provider_label="API"):
    if not api_key:
        raise ValueError(f"{provider_label} API key is missing.")

    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def create_judge_llm():
    settings = get_settings()
    provider = os.environ.get("EVAL_JUDGE_PROVIDER", settings.llm_provider).lower()
    model = os.environ.get("EVAL_JUDGE_MODEL", settings.llm_model)
    temperature = float(os.environ.get("EVAL_JUDGE_TEMPERATURE", "0"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=os.environ.get("EVAL_JUDGE_OLLAMA_BASE_URL", settings.ollama_base_url),
        )

    if provider == "openai":
        return _create_openai_compatible_llm(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("EVAL_JUDGE_OPENAI_API_KEY", settings.openai_api_key),
            base_url=os.environ.get("EVAL_JUDGE_OPENAI_BASE_URL", settings.openai_base_url) or None,
            provider_label="OpenAI",
        )

    if provider == "deepseek":
        return _create_openai_compatible_llm(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("EVAL_JUDGE_DEEPSEEK_API_KEY", settings.deepseek_api_key),
            base_url=os.environ.get("EVAL_JUDGE_DEEPSEEK_BASE_URL", settings.deepseek_base_url),
            provider_label="DeepSeek",
        )

    if provider == "openai_compatible":
        return _create_openai_compatible_llm(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("EVAL_JUDGE_API_KEY", settings.openai_compat_api_key),
            base_url=os.environ.get("EVAL_JUDGE_BASE_URL", settings.openai_compat_base_url),
            provider_label="OpenAI-compatible",
        )

    raise ValueError(
        "Unsupported EVAL_JUDGE_PROVIDER '{}'. Use one of: ollama, openai, deepseek, openai_compatible.".format(
            provider
        )
    )


class JudgeScore(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    score: int = Field(ge=0, le=5)
    passed: bool = Field(alias="pass")
    reason: str = Field(default="")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


@dataclass
class CalibrationSummary:
    available: bool
    calibration_passed: bool
    details: dict[str, Any]
    reason: str = ""


class EvalJudge:
    def __init__(self, llm=None):
        self._init_error = ""
        try:
            self.llm = llm or create_judge_llm()
        except Exception as exc:
            self.llm = None
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self.llm is not None

    def judge_faithfulness(self, user_query: str, paper_profiles: list[dict], final_answer: str) -> JudgeScore | None:
        prompt = (
            "User question:\n"
            f"{user_query}\n\n"
            "Paper profiles:\n"
            f"{json.dumps(paper_profiles, ensure_ascii=False, indent=2)}\n\n"
            "Candidate answer:\n"
            f"{final_answer}"
        )
        system_prompt = """You are a strict judge for literature-agent evaluations.

Score only from the supplied paper profiles and answer. Never use outside knowledge.

Rubric:
- 5: faithful, complete enough for the question, no cross-paper attribution errors
- 4: mostly faithful, minor omission or wording issue
- 3: partially faithful, noticeable omission or overstatement
- 2: major omission or unsupported claim
- 1: severe hallucination or multiple attribution errors
- 0: completely unsupported

Return structured JSON with:
- score: integer 0-5
- pass: true only when score >= 4
- reason: one concise paragraph
"""
        return self._invoke(system_prompt, prompt)

    def judge_alignment(self, user_query: str, paper_profiles: list[dict], final_answer: str) -> JudgeScore | None:
        prompt = (
            "User question:\n"
            f"{user_query}\n\n"
            "Paper profiles:\n"
            f"{json.dumps(paper_profiles, ensure_ascii=False, indent=2)}\n\n"
            "Candidate answer:\n"
            f"{final_answer}"
        )
        system_prompt = """You are a strict judge for cross-document literature comparisons.

Score only from the supplied paper profiles and answer. Never use outside knowledge.

Rubric:
- 5: comparison dimensions are aligned, conflicts or trade-offs are handled clearly, no cross-paper confusion
- 4: mostly aligned, minor gap in conflict handling
- 3: mixed quality, some alignment but also some apples-to-oranges comparison
- 2: weak alignment or mostly stitched summary
- 1: serious cross-paper confusion
- 0: unusable comparison

Return structured JSON with:
- score: integer 0-5
- pass: true only when score >= 4
- reason: one concise paragraph
"""
        return self._invoke(system_prompt, prompt)

    def judge_valid_fallback(
        self,
        user_query: str,
        final_answer: str,
        workspace_id: str,
        retrieved_paper_ids: list[str],
    ) -> JudgeScore | None:
        prompt = (
            "User question:\n"
            f"{user_query}\n\n"
            f"Workspace ID: {workspace_id}\n"
            f"Retrieved paper IDs: {retrieved_paper_ids}\n\n"
            "Candidate answer:\n"
            f"{final_answer}"
        )
        system_prompt = """You are a strict judge for fallback behavior in a literature-reading agent.

Judge whether the answer correctly acknowledges missing evidence from the current workspace instead of inventing facts.

Rubric:
- 5: explicitly states evidence is missing or insufficient and stays within workspace boundaries
- 4: mostly valid fallback with minor wording issue
- 3: hedged but still somewhat ambiguous
- 2: partially invents unsupported details
- 1: mostly hallucinates
- 0: completely ignores missing-evidence constraints

Return structured JSON with:
- score: integer 0-5
- pass: true only when score >= 4
- reason: one concise paragraph
"""
        return self._invoke(system_prompt, prompt)

    def run_calibration(self) -> CalibrationSummary:
        if not self.available:
            return CalibrationSummary(
                available=False,
                calibration_passed=False,
                details={},
                reason=self._init_error or "Judge model is unavailable.",
            )

        profiles = [
            {
                "paper_id": "paper_a",
                "title": "Paper A",
                "core_method": "Method A",
                "main_results": ["improves accuracy by 3%"],
                "limitations": ["uses more memory"],
                "evidence_spans": ["Results: +3% accuracy", "Limitations: higher memory use"],
            },
            {
                "paper_id": "paper_b",
                "title": "Paper B",
                "core_method": "Method B",
                "main_results": ["reduces memory consumption by 20%"],
                "limitations": ["slightly lower accuracy"],
                "evidence_spans": ["Results: -20% memory", "Trade-off: small accuracy drop"],
            },
        ]
        good_answer = (
            "Paper A emphasizes accuracy and reports a 3% gain, while Paper B emphasizes efficiency and reduces "
            "memory consumption by 20%. The main trade-off is that Paper A uses more memory, whereas Paper B accepts "
            "a slight accuracy drop."
        )
        bad_answer = "Paper B improves accuracy by 3%, and both papers report the same result with no trade-offs."

        details = {}
        passed = True
        faithfulness_good = self.judge_faithfulness("Compare the two papers.", profiles, good_answer)
        faithfulness_bad = self.judge_faithfulness("Compare the two papers.", profiles, bad_answer)
        alignment_good = self.judge_alignment("Compare the two papers.", profiles, good_answer)
        alignment_bad = self.judge_alignment("Compare the two papers.", profiles, bad_answer)

        pairs = {
            "faithfulness": {"good": faithfulness_good, "bad": faithfulness_bad},
            "alignment": {"good": alignment_good, "bad": alignment_bad},
        }
        for rubric, results in pairs.items():
            good = results["good"]
            bad = results["bad"]
            if good is None or bad is None:
                passed = False
                details[rubric] = {"available": False}
                continue
            rubric_passed = good.score > bad.score and good.passed and not bad.passed
            passed = passed and rubric_passed
            details[rubric] = {
                "available": True,
                "passed": rubric_passed,
                "good": good.as_dict(),
                "bad": bad.as_dict(),
            }

        return CalibrationSummary(available=True, calibration_passed=passed, details=details)

    def _invoke(self, system_prompt: str, prompt: str) -> JudgeScore | None:
        if not self.available:
            return None

        structured = self.llm.with_structured_output(JudgeScore, method="function_calling")
        try:
            return structured.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        except Exception:
            return None


def calibration_to_dict(summary: CalibrationSummary) -> dict[str, Any]:
    return asdict(summary)

