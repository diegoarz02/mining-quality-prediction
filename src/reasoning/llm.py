"""Punto único donde se configuran el cliente LLM y sus credenciales.

El token se lee del archivo .env vía python-dotenv. Si no existe, make_llm devuelve None
y el grafo cae a la síntesis determinística por reglas, de modo que el pipeline corre con
o sin credenciales.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from src.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
# gpt-4o-mini: gpt-4.1-mini quedó saturado en el tier gratuito (429 sostenido durante
# días con tokens distintos), mientras el resto del catálogo respondía con normalidad.
MODEL_NAME = "openai/gpt-4o-mini"


def make_llm():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return None
    from langchain_openai import ChatOpenAI

    # El tier gratuito puede tardar ~60 s en responder un 429; con un timeout corto el
    # cliente lo reporta como timeout y el backoff no llega a reintentar.
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=token,
        base_url=GITHUB_MODELS_BASE_URL,
        temperature=0,
        timeout=90,
        max_retries=0,
    )


def call_with_backoff(llm, messages, retries: int = 3):
    """Invoca el LLM con reintentos exponenciales ante errores 429 del tier gratuito."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - solo tratamos especial el rate limit
            if ("429" in str(exc) or "rate" in str(exc).lower()) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("LLM no disponible tras los reintentos")
