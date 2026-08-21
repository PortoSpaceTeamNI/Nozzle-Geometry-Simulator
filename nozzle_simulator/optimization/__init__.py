"""Maintained contour-optimization API used by the main desktop window."""

from .contour_genetic_algorithm import (
    OptimizationResult,
    OptimizationSettings,
    evaluate_contour,
    evaluate_moc_contour,
    optimize_geometry,
)

__all__ = [
    "OptimizationResult",
    "OptimizationSettings",
    "evaluate_contour",
    "evaluate_moc_contour",
    "optimize_geometry",
]
