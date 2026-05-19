"""
Service-Schicht: Dinge, die App/UI orchestration betreffen (Training, Laden, State).
"""

from .pipeline import init_session_state, load_pipeline_run, train_pipeline_run

__all__ = [
    "init_session_state",
    "train_pipeline_run",
    "load_pipeline_run",
]

