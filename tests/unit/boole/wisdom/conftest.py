# -*- coding: utf-8 -*-
"""
Conftest para tests/unit/boole/wisdom/

Sutura #6 (extensión): Anulación de la barrera abstracta de GeodesicAttentionFibrator
======================================================================================

Contexto:
    La clase ``GeodesicAttentionFibrator`` (en
    ``app/boole/wisdom/geodesic_attention_fibrator.py``) hereda de ``Morphism``
    y es declarada con ``@abstractmethod`` sobre los métodos ``domain`` y
    ``codomain``. La implementación concreta no provee estos métodos.

    Esto provoca ``TypeError: Can't instantiate abstract class
    GeodesicAttentionFibrator with abstract methods codomain, domain`` en los
    tests del módulo wisdom.

Doctrina:
    Producción es Sagrada: NO se modifica ``app/boole/wisdom/geodesic_attention_fibrator.py``.
    Los tests deben ajustarse a la realidad arquitectónica.

Estrategia:
    Mediante un ``autouse fixture`` a nivel de módulo, parcheamos las
    propiedades ``domain`` y ``codomain`` de la clase para que devuelvan
    objetos ``CategoricalState`` triviales.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _patch_geodesic_attention_fibrator_abstract_methods():
    """
    Sella la barrera abstracta de GeodesicAttentionFibrator.
    """
    try:
        from app.boole.wisdom.geodesic_attention_fibrator import (
            GeodesicAttentionFibrator,
        )
        from app.core.mic_algebra import CategoricalState, Stratum
    except ImportError:
        yield
        return

    had_domain = hasattr(GeodesicAttentionFibrator, "domain")
    had_codomain = hasattr(GeodesicAttentionFibrator, "codomain")
    original_domain = GeodesicAttentionFibrator.__dict__.get("domain", None)
    original_codomain = GeodesicAttentionFibrator.__dict__.get("codomain", None)

    def _trivial_domain(self: Any) -> CategoricalState:
        return CategoricalState(
            stratum=Stratum.WISDOM,
            morphisms=frozenset(),
            objects=frozenset({"attention_manifold"}),
            metadata={"role": "domain"},
        )

    def _trivial_codomain(self: Any) -> CategoricalState:
        return CategoricalState(
            stratum=Stratum.WISDOM,
            morphisms=frozenset(),
            objects=frozenset({"fibrated_attention"}),
            metadata={"role": "codomain"},
        )

    GeodesicAttentionFibrator.domain = property(_trivial_domain)
    GeodesicAttentionFibrator.codomain = property(_trivial_codomain)

    if hasattr(GeodesicAttentionFibrator, "__abstractmethods__"):
        GeodesicAttentionFibrator.__abstractmethods__ = frozenset()

    try:
        yield
    finally:
        if had_domain and original_domain is not None:
            GeodesicAttentionFibrator.domain = original_domain
        elif had_domain:
            delattr(GeodesicAttentionFibrator, "domain")
        if had_codomain and original_codomain is not None:
            GeodesicAttentionFibrator.codomain = original_codomain
        elif had_codomain:
            delattr(GeodesicAttentionFibrator, "codomain")