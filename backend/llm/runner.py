"""Glue between versioned prompt files and the LLM client.

``run_prompt`` is the single entry point services use: it loads the template,
renders it, picks the template's temperature (0.2 for evaluation prompts, 0.7
for generation prompts), validates the JSON against the declared schema and
falls back safely. Each call is one fixed step of a deterministic pipeline.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from backend.llm.client import LLMClient, LLMJSONResult, get_llm
from backend.llm.prompt_loader import load_prompt
from backend.models.llm_schemas import SCHEMA_REGISTRY

T = TypeVar("T", bound=BaseModel)


def run_prompt(
    name: str,
    variables: dict,
    schema: type[T] | None = None,
    fallback: T | Callable[[], T] | None = None,
    client: LLMClient | None = None,
    version: int | None = None,
    temperature: float | None = None,
) -> LLMJSONResult[T]:
    """Render prompt ``name`` with ``variables`` and return validated JSON."""
    tpl = load_prompt(name, version)
    schema_cls = schema or SCHEMA_REGISTRY[tpl.output_schema]
    system, user = tpl.render(**variables)
    llm = client or get_llm()
    result = llm.complete_json(
        system=system,
        user=user,
        schema=schema_cls,
        temperature=tpl.temperature if temperature is None else temperature,
        task=tpl.name,
        variables=variables,
        fallback=fallback,
    )
    result.meta["prompt"] = tpl.id
    return result
