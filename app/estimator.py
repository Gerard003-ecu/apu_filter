"""
[FACADE] Módulo Estimator (Legacy)
Redirige al nuevo Microservicio Táctico (SemanticEstimator).
Mantenido para compatibilidad con tests y referencias heredadas.
"""

from .semantic_estimator import (
    SearchArtifacts,
    calculate_estimate,
    _find_best_keyword_match,
    _find_best_semantic_match,
    _calculate_historical_average,
    _calculate_rendimiento_from_detail,
    DerivationDetails,
    MatchCandidate,
    DataQualityMetrics,
    MatchMode,
    TipoAPU,
    TipoInsumo,
    validate_search_artifacts,
    assess_data_quality,
    validate_dataframe_columns,
    safe_float_conversion,
    safe_int_conversion,
    get_safe_column_value,
    _calculate_match_score
)

# Re-exportar símbolos
__all__ = [
    "SearchArtifacts",
    "calculate_estimate",
    "DerivationDetails",
    "MatchCandidate",
    "MatchMode",
    "TipoAPU",
    "TipoInsumo",
]
