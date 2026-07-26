"""Noisy Aer backend — re-exports from run_noise for shot-based simulation."""

from prototype.run_noise import (  # noqa: F401
    NoiseCase,
    generate_cases,
    build_noise_model,
    run_shot_sequence,
)

__all__ = [
    "NoiseCase",
    "generate_cases",
    "build_noise_model",
    "run_shot_sequence",
]
