"""Verificación amigable del entorno antes de importar dependencias pesadas.

En máquinas con varios Python instalados es fácil lanzar la API o el pipeline con un
intérprete que no tiene las librerías del proyecto. En lugar de un traceback crudo, se
explica en español qué falta y cómo activar el venv correcto.
"""
from __future__ import annotations

import importlib.util
import sys


def ensure_dependencies(modules: list[str]) -> None:
    """Termina con un mensaje accionable si falta alguna dependencia crítica."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return
    raise SystemExit(
        "\n[Error de entorno] Faltan dependencias: " + ", ".join(missing) + ".\n"
        f"Este proceso está corriendo con: {sys.executable}\n"
        "Lo más probable es que el venv del proyecto no esté activado. Solución:\n"
        "  1. Activar el venv:  C:\\venvs\\minsur\\Scripts\\activate\n"
        "  2. Instalar dependencias si faltan:  pip install -r requirements.txt\n"
        "  3. Lanzar siempre con el python del venv, por ejemplo:\n"
        "       python -m uvicorn api.main:app\n"
        "       python scripts/run_pipeline.py\n"
        "  (usar 'python -m uvicorn', no 'uvicorn' a secas: el ejecutable global puede\n"
        "   pertenecer a otro intérprete de Python sin estas librerías)\n"
    )
