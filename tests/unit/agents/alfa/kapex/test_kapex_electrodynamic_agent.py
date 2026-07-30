# -*- coding: utf-8 -*-
"""
Suite de pruebas unitarias rigurosas para KApexElectrodynamicAgent
=================================================================

Ubicación del artefacto
-----------------------
tests/unit/agents/alfa/kapex/test_kapex_electrodynamic_agent.py

Módulo bajo prueba
------------------
app/agents/alfa/kapex/kapex_electrodynamic_agent.py

Filosofía de la suite — endofuntor anidado K_APEX
-------------------------------------------------
Las tres fases forman una composición funtorial estricta:

    F₁ ──build_context──► ApexPreparationContext
     │                         │
     │    (precondición formal) │
     ▼                         ▼
    F₂ ──synthesize────► ApexStateTensor
     │                         │
     │  gauge_injection_vector │
     ▼                         ▼
    F₃ ──export_stalk──► SheafStalkApex

Contratos de continuidad inter-fase
------------------------------------
1. ``Phase1_MetricValidation.build_context()`` produce el único
   ``ApexPreparationContext`` que ``Phase2`` está autorizado a consumir.
2. ``Phase2_ElectrodynamicSynthesis.synthesize()`` emite
   ``ApexStateTensor.gauge_injection_vector``, única entrada admisible
   de ``Phase3_SheafProjection.export_stalk()``.
3. El último invariante verificado en Fase *k* es exactamente la
   precondición algebraica del primer método de Fase *k+1*.

Invariantes auditados
---------------------
- Caminos felices y excepciones electrodinámicas.
- Cotas de Wilkinson/Higham sobre residuos de Cholesky y cierre espectral.
- Idempotencia de proyectores Π_so, cierre de Lie en 𝔰𝔬(n).
- Antisimetría de F, covarianza de gauge bajo isometrías de G.
- SPD/PSD, condición espectral κ, nulidad disipativa β₀ = dim ker(R).
- Identidad de Hodge local δᵀ G δ ≈ I y Δ_Hodge ≻ 0.
- Congelamiento (frozen) de todos los DTOs y copias defensivas.

Notas de ejecución
------------------
- Dependencias: pytest, numpy, scipy.
- Comando recomendado::

      pytest tests/unit/agents/alfa/kapex/test_kapex_electrodynamic_agent.py -v
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ==============================================================================
# BOOTSTRAP DE IMPORTACIÓN ROBUSTO
# ==============================================================================
# Tolera el módulo bajo "alpha" o "alfa"; si el paquete no está instalado,
# carga directa por archivo.  El orden de candidatos es determinista.
# ==============================================================================

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE_CANDIDATES: Tuple[str, ...] = (
    "app.agents.alpha.kapex.kapex_electrodynamic_agent",
    "app.agents.alfa.kapex.kapex_electrodynamic_agent",
)

mod = None
last_exc: BaseException | None = None

for candidate in _MODULE_CANDIDATES:
    try:
        mod = importlib.import_module(candidate)
        break
    except Exception as exc:  # pragma: no cover
        last_exc = exc

if mod is None:
    _FILE_CANDIDATES: Tuple[Path, ...] = (
        ROOT / "app" / "agents" / "alpha" / "kapex" / "kapex_electrodynamic_agent.py",
        ROOT / "app" / "agents" / "alfa" / "kapex" / "kapex_electrodynamic_agent.py",
    )
    for path in _FILE_CANDIDATES:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "kapex_electrodynamic_agent", path
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                break

if mod is None:  # pragma: no cover
    if last_exc is not None:
        raise ImportError(
            "No se pudo importar kapex_electrodynamic_agent.py ni como paquete "
            "ni como archivo."
        ) from last_exc
    raise ImportError(
        "No se encontró kapex_electrodynamic_agent.py en alpha ni alfa."
    )


# ==============================================================================
# ALIASES DEL MÓDULO BAJO PRUEBA
# ==============================================================================

KApexElectrodynamicAgent = mod.KApexElectrodynamicAgent

ApexViabilityFlags = mod.ApexViabilityFlags
describe_viability_flags = mod.describe_viability_flags

ApexPreparationContext = mod.ApexPreparationContext
ApexStateTensor = mod.ApexStateTensor
SheafStalkApex = mod.SheafStalkApex

ElectrodynamicApexError = mod.ElectrodynamicApexError
ApexDimensionError = mod.ApexDimensionError
ApexParameterError = mod.ApexParameterError
ApexSymmetryError = mod.ApexSymmetryError
ApexConditionError = mod.ApexConditionError
MetricInverseError = mod.MetricInverseError
SpectralClosureError = mod.SpectralClosureError
GaugePotentialError = mod.GaugePotentialError
EikonalRefractionError = mod.EikonalRefractionError
FinancialBlackHoleError = mod.FinancialBlackHoleError
HolonomyVetoError = mod.HolonomyVetoError
GaugeCovarianceError = mod.GaugeCovarianceError
SheafMetricError = mod.SheafMetricError

EPS: float = float(getattr(mod, "_MACHINE_EPS", np.finfo(np.float64).eps))

# Tolerancias canónicas derivadas de la unidad de redondeo.
_ATOL_STRICT: float = 1.0e-14
_ATOL_STD: float = 1.0e-12
_RTOL_STD: float = 1.0e-12
_ATOL_HODGE: float = 1.0e-10
_RTOL_HODGE: float = 1.0e-10


# ==============================================================================
# HELPERS NUMÉRICOS — CONSTRUCCIONES EXACTAS Y CERTIFICADOS
# ==============================================================================


def diag_spd(eigenvalues: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye el par exacto (G, G⁻¹) con G = diag(λᵢ) ≻ 0.

    Parameters
    ----------
    eigenvalues :
        Autovalores estrictamente positivos.

    Returns
    -------
    G, G_inv : ndarray, ndarray
        Métricas diagonal SPD e inversa exacta componente a componente.

    Raises
    ------
    ValueError
        Si algún autovalor no es estrictamente positivo.

    Notes
    -----
    La construcción diagonal elimina ruido de redondeo en las pruebas de
    consistencia bilateral G·G⁻¹ = I = G⁻¹·G y en el cómputo de κ₂(G).
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if eigs.size == 0:
        raise ValueError("diag_spd requiere al menos un autovalor.")
    if np.any(eigs <= 0.0):
        raise ValueError("diag_spd requiere autovalores estrictamente positivos.")
    G = np.diag(eigs)
    G_inv = np.diag(1.0 / eigs)
    return G, G_inv


def diag_psd(eigenvalues: Sequence[float]) -> np.ndarray:
    """
    Construye R = diag(μᵢ) ⪰ 0 con autovalores no negativos.

    Parameters
    ----------
    eigenvalues :
        Autovalores ≥ 0.

    Returns
    -------
    R : ndarray
        Matriz diagonal PSD.

    Raises
    ------
    ValueError
        Si algún autovalor es negativo.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if np.any(eigs < 0.0):
        raise ValueError("diag_psd requiere autovalores no negativos.")
    return np.diag(eigs)


def skew_from_pairs(
    n: int,
    pairs: Iterable[Tuple[int, int]],
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generador elemental del álgebra de Lie 𝔰𝔬(n).

    Para cada par ordenado (i, j) inserta el blade:
        E_{ij} ↦  scale · (eᵢ ⊗ eⱼ − eⱼ ⊗ eᵢ).

    Parameters
    ----------
    n :
        Dimensión ambiente.
    pairs :
        Pares de índices 0-based con i ≠ j.
    scale :
        Amplitud del generador.

    Returns
    -------
    A : ndarray, shape (n, n)
        Matriz antisimétrica Aᵀ = −A ∈ 𝔰𝔬(n).
    """
    A = np.zeros((n, n), dtype=np.float64)
    s = float(scale)
    for i, j in pairs:
        if i == j:
            raise ValueError("skew_from_pairs: se requiere i ≠ j.")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"skew_from_pairs: índices ({i},{j}) fuera de rango n={n}.")
        A[i, j] = s
        A[j, i] = -s
    return A


def frobenius_asymmetry(M: np.ndarray) -> float:
    """
    Residuo relativo de antisimetría: ‖M + Mᵀ‖_F / max(‖M‖_F, 1).

    Es el certificado numérico de pertenencia a 𝔰𝔬(n).
    """
    norm_M = float(np.linalg.norm(M, "fro"))
    return float(np.linalg.norm(M + M.T, "fro")) / max(norm_M, 1.0)


def wilkinson_bound(kappa: float, n: int, safety: float = 100.0) -> float:
    """
    Cota a priori de Wilkinson/Higham para residuos de primer orden:

        tol ≥ safety · κ · ε_mach · n.

    Se usa como umbral admisible de inverse_residual y hodge_metric_residual.
    """
    return max(_ATOL_STD, safety * float(kappa) * EPS * int(n))


def valid_synthesis_kwargs(
    agent: Any,
    phase_norm_target: float = 1.05,
    sigma_stress: float = 0.0,
) -> Dict[str, Any]:
    """
    Genera un diccionario de entradas admisibles para
    ``Phase2.synthesize`` / ``synthesize_apex_field``.

    Estrategia geométrica
    ---------------------
    - ``d_Phi = 1⃗``.
    - ``phase_gradient`` se reescala para satisfacer exactamente
          ⟨∂S, G⁻¹ ∂S⟩ = phase_norm_target,
      lo que controla el margen Eikonal con precisión de máquina.
    - ``E = H = 1⃗`` ⇒ P_in = n (producto punto euclídeo).
    - ``grad_H = 0⃗`` ⇒ P_diss = 0, P_exergia = n > 0.
    - ``A₁ = A₂ = 0`` ⇒ F = 0, S_YM = 0, holonomía trivial.

    Parameters
    ----------
    agent :
        Instancia ya construida de KApexElectrodynamicAgent.
    phase_norm_target :
        Valor objetivo de ‖∂S‖²_{G⁻¹}. Debe superar el umbral Eikonal.
    sigma_stress :
        Esfuerzo de mercado σ* que entra en el índice de refracción de Fermat.

    Returns
    -------
    kwargs : dict
        Argumentos listos para desempaquetar en synthesize_apex_field(**kwargs).
    """
    n = int(agent.context.dim)

    base = np.ones(n, dtype=np.float64)
    norm_sq = float(base @ agent.context.G_inv @ base)

    if norm_sq <= EPS:
        # Fallback: primer vector de la base canónica si 1⃗ es degenerado.
        base = np.eye(n, dtype=np.float64)[0]
        norm_sq = float(base @ agent.context.G_inv @ base)

    scale = math.sqrt(float(phase_norm_target) / max(norm_sq, EPS))
    phase_gradient = scale * base

    zero_vector = np.zeros(n, dtype=np.float64)
    zero_matrix = np.zeros((n, n), dtype=np.float64)

    return dict(
        d_Phi=np.ones(n, dtype=np.float64),
        phase_gradient=phase_gradient,
        sigma_stress=float(sigma_stress),
        E_field=np.ones(n, dtype=np.float64),
        H_field=np.ones(n, dtype=np.float64),
        grad_H=zero_vector,
        A_gauge_1=zero_matrix.copy(),
        A_gauge_2=zero_matrix.copy(),
    )


def assert_spd(M: np.ndarray, name: str = "M", atol: float = _ATOL_STD) -> np.ndarray:
    """
    Certifica que M es simétrica y definida positiva; devuelve sus autovalores.
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1], f"{name} no es cuadrada."
    assert_allclose(M, M.T, rtol=_RTOL_STD, atol=atol, err_msg=f"{name} no es simétrica.")
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > atol), (
        f"{name} no es SPD: λ_min = {eigvals[0]:.3e} (atol={atol:.3e})."
    )
    return eigvals


def assert_psd(M: np.ndarray, name: str = "M", atol: float = _ATOL_STD) -> np.ndarray:
    """
    Certifica que M es simétrica y semidefinida positiva; devuelve autovalores.
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1], f"{name} no es cuadrada."
    assert_allclose(M, M.T, rtol=_RTOL_STD, atol=atol, err_msg=f"{name} no es simétrica.")
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals >= -atol), (
        f"{name} no es PSD: λ_min = {eigvals[0]:.3e} (atol={atol:.3e})."
    )
    return eigvals


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def identity_agent():
    """
    Agente canónico sobre el vacío electrodinámico:

        G = G⁻¹ = R_cost = I₃.

    Propiedades espectrales óptimas
    --------------------------------
    - κ₂(G) = κ₂(G⁻¹) = 1.
    - ε_G = 0  (Cholesky sin regularización).
    - rank(R) = 3, β₀ = 0, gap(R) = 0 (espectro degenerado {1,1,1}).
    - Residuos bilaterales y de cierre espectral ~ O(ε_mach).
    """
    I3 = np.eye(3, dtype=np.float64)
    return KApexElectrodynamicAgent(
        G_mu_nu=I3.copy(),
        G_inv=I3.copy(),
        R_cost=I3.copy(),
    )


@pytest.fixture
def diagonal_agent():
    """
    Agente diagonal no trivial bien condicionado:

        G = diag(1, 2, 3),  R = diag(0.5, 1.5, 2.5).

    Sirve como testigo de la composición funtorial completa F₁∘F₂∘F₃
    fuera del vacío isótropo.
    """
    G, G_inv = diag_spd([1.0, 2.0, 3.0])
    R = diag_psd([0.5, 1.5, 2.5])
    return KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)


# ##############################################################################
#
#  FASE 1 — VALIDACIÓN MÉTRICA, CHOLESKY REGULARIZADO Y CIERRE ESPECTRAL
#
#  Dominio del endofuntor:  (G, G⁻¹, R)  ↦  ApexPreparationContext.
#
#  Obligaciones algebraicas verificadas en esta fase:
#    (M1)  dim > 0, cuadratura y compatibilidad dimensional.
#    (M2)  Simetría Frobenius de G, G⁻¹, R  (‖A−Aᵀ‖_F ≤ tol).
#    (M3)  G ≻ 0, G⁻¹ ≻ 0  (SPD vía Cholesky / autovalores).
#    (M4)  κ₂(G) ≤ κ_max, κ₂(G⁻¹) ≤ κ_max.
#    (M5)  Consistencia bilateral: G·G⁻¹ ≈ I ≈ G⁻¹·G  (MetricInverse).
#    (M6)  Cholesky regularizado L Lᵀ = G + ε_G I, ε_G ≥ 0 (Tikhonov).
#    (M7)  R ⪰ 0, R_sqrt = √R (única PSD), R_sqrt² ≈ R  (cierre espectral).
#    (M8)  rank(R), β₀ = dim ker(R), gap espectral λ_{r+1}−λ_r.
#    (M9)  Inmutabilidad del DTO y copias defensivas de las matrices.
#
#  El ÚLTIMO método de esta fase certifica el contrato funtorial
#  F₁ → F₂: el contexto emitido es exactamente el consumido por Phase2.
#  Ese certificado es la precondición formal con la que arranca Fase 2.
#
# ##############################################################################


class TestPhase1MetricValidation:
    """
    Suite de la Fase 1: ``Phase1_MetricValidation``.

    Establece el dominio inicial del endofuntor K_APEX:

        build_context() ──► ApexPreparationContext.

    Cada test es un juicio independiente sobre un invariante (M1)–(M9);
    el cierre de la clase es el puente funtorial hacia Fase 2.
    """

    # ------------------------------------------------------------------
    # (M0) Jerarquía de excepciones — pre-requisito de captura agregada
    # ------------------------------------------------------------------

    def test_exception_hierarchy_is_rooted(self):
        """
        Toda excepción específica del módulo hereda de
        ``ElectrodynamicApexError``, habilitando captura agregada::

            try:
                ...
            except ElectrodynamicApexError:
                # cualquier fallo del apex
        """
        exceptions = (
            ApexDimensionError,
            ApexParameterError,
            ApexSymmetryError,
            ApexConditionError,
            MetricInverseError,
            SpectralClosureError,
            GaugePotentialError,
            EikonalRefractionError,
            FinancialBlackHoleError,
            HolonomyVetoError,
            GaugeCovarianceError,
            SheafMetricError,
        )
        for exc in exceptions:
            assert issubclass(exc, ElectrodynamicApexError), (
                f"{exc.__name__} no hereda de ElectrodynamicApexError."
            )

    # ------------------------------------------------------------------
    # (M1) Dimensiones
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "G, G_inv, R",
        [
            pytest.param(
                np.ones((2, 3), dtype=np.float64),
                np.eye(2, dtype=np.float64),
                np.eye(2, dtype=np.float64),
                id="G-non-square",
            ),
            pytest.param(
                np.eye(2, dtype=np.float64),
                np.eye(3, dtype=np.float64),
                np.eye(2, dtype=np.float64),
                id="G_inv-dimension-mismatch",
            ),
            pytest.param(
                np.eye(2, dtype=np.float64),
                np.eye(2, dtype=np.float64),
                np.eye(3, dtype=np.float64),
                id="R_cost-dimension-mismatch",
            ),
        ],
    )
    def test_dimension_mismatch_raises(self, G, G_inv, R):
        """
        (M1a) Matrices no cuadradas o de dimensión mutuamente inconsistente
        deben lanzar ``ApexDimensionError`` antes de cualquier análisis
        espectral.
        """
        with pytest.raises(ApexDimensionError):
            KApexElectrodynamicAgent(G, G_inv, R)

    def test_zero_dimension_raises(self):
        """
        (M1b) n = 0 está prohibido: el espacio de calibre debe ser un
        fibrado vectorial de rango positivo.
        """
        empty = np.empty((0, 0), dtype=np.float64)
        with pytest.raises(ApexDimensionError):
            KApexElectrodynamicAgent(empty, empty, empty)

    # ------------------------------------------------------------------
    # (M1-ext) Parámetros de control del constructor
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"kappa_max": 1.0}, id="kappa_max=1"),
            pytest.param({"kappa_max": 0.0}, id="kappa_max=0"),
            pytest.param({"kappa_max": -10.0}, id="kappa_max<0"),
            pytest.param({"eikonal_slack": -0.1}, id="eikonal_slack<0"),
            pytest.param({"eikonal_slack": 1.0}, id="eikonal_slack=1"),
            pytest.param({"holonomy_tol_rel": 0.0}, id="holonomy_tol_rel=0"),
            pytest.param({"holonomy_tol_rel": -1.0e-6}, id="holonomy_tol_rel<0"),
        ],
    )
    def test_invalid_constructor_parameters_raise(self, kwargs):
        """
        Parámetros de control fuera del abierto admisible deben lanzar
        ``ApexParameterError`` *antes* de cualquier validación matricial,
        garantizando fail-fast del constructor.
        """
        I2 = np.eye(2, dtype=np.float64)
        with pytest.raises(ApexParameterError):
            KApexElectrodynamicAgent(I2.copy(), I2.copy(), I2.copy(), **kwargs)

    # ------------------------------------------------------------------
    # (M2) Simetría Frobenius
    # ------------------------------------------------------------------

    def test_asymmetric_G_raises_symmetry_error(self):
        """
        (M2a) G_μν debe ser simétrica. Asimetría macroscópica ⇒
        ``ApexSymmetryError``.
        """
        G = np.eye(2, dtype=np.float64)
        G[0, 1] = 1.0
        with pytest.raises(ApexSymmetryError):
            KApexElectrodynamicAgent(G, np.eye(2), np.eye(2))

    def test_asymmetric_G_inv_raises_symmetry_error(self):
        """
        (M2b) G⁻¹ también debe ser simétrica.
        """
        G_inv = np.eye(2, dtype=np.float64)
        G_inv[1, 0] = 2.0
        with pytest.raises(ApexSymmetryError):
            KApexElectrodynamicAgent(np.eye(2), G_inv, np.eye(2))

    def test_asymmetric_R_raises_symmetry_error(self):
        """
        (M2c) R_cost debe ser simétrica *antes* de validar PSD.
        """
        R = np.eye(2, dtype=np.float64)
        R[0, 1] = 3.0
        with pytest.raises(ApexSymmetryError):
            KApexElectrodynamicAgent(np.eye(2), np.eye(2), R)

    # ------------------------------------------------------------------
    # (M3) SPD de G y G⁻¹
    # ------------------------------------------------------------------

    def test_G_not_spd_raises(self):
        """
        (M3a) Autovalor negativo en G_μν ⇒ no SPD ⇒
        ``ElectrodynamicApexError``.
        """
        G = np.diag([1.0, -1.0]).astype(np.float64)
        with pytest.raises(ElectrodynamicApexError):
            KApexElectrodynamicAgent(G, np.eye(2), np.eye(2))

    def test_G_zero_eigenvalue_raises(self):
        """
        (M3b) Autovalor nulo (o numéricamente nulo) en G_μν ⇒ singular ⇒
        no SPD.
        """
        G = np.diag([1.0, 0.0]).astype(np.float64)
        with pytest.raises(ElectrodynamicApexError):
            KApexElectrodynamicAgent(G, np.eye(2), np.eye(2))

    def test_G_inv_not_spd_raises(self):
        """
        (M3c) G⁻¹ debe ser SPD. Matriz definida negativa ⇒ error.
        """
        with pytest.raises(ElectrodynamicApexError):
            KApexElectrodynamicAgent(np.eye(2), -np.eye(2), np.eye(2))

    # ------------------------------------------------------------------
    # (M4) Condición espectral κ
    # ------------------------------------------------------------------

    def test_G_condition_number_above_threshold_raises(self):
        """
        (M4a) κ₂(G) > κ_max ⇒ ``ApexConditionError``.
        """
        G, G_inv = diag_spd([1.0, 1.0e6])
        with pytest.raises(ApexConditionError):
            KApexElectrodynamicAgent(G, G_inv, np.eye(2), kappa_max=1.0e5)

    def test_G_inv_condition_number_above_threshold_raises(self):
        """
        (M4b) El control de condición aplica simétricamente a G⁻¹.
        """
        G_inv = np.diag([1.0, 1.0e6]).astype(np.float64)
        with pytest.raises(ApexConditionError):
            KApexElectrodynamicAgent(np.eye(2), G_inv, np.eye(2), kappa_max=1.0e5)

    # ------------------------------------------------------------------
    # (M5) Consistencia métrica bilateral
    # ------------------------------------------------------------------

    def test_metric_inverse_inconsistency_raises(self):
        """
        (M5) Si G⁻¹ no es la inversa bilateral de G dentro de la cota de
        Wilkinson, debe lanzarse ``MetricInverseError``.
        """
        G = np.eye(2, dtype=np.float64)
        G_inv = 2.0 * np.eye(2, dtype=np.float64)
        with pytest.raises(MetricInverseError):
            KApexElectrodynamicAgent(G, G_inv, np.eye(2))

    # ------------------------------------------------------------------
    # (M6) Cholesky regularizado con Tikhonov adaptativo
    # ------------------------------------------------------------------

    def test_cholesky_regularization_retry_mechanism(self, monkeypatch):
        """
        (M6) Si ``la.cholesky`` falla una vez, el mecanismo de Tikhonov
        adaptativo debe reintentar con τ = ε_G > 0 y satisfacer

            L Lᵀ = G + ε_G I.

        Se fuerza el primer fallo mediante monkeypatch.
        """
        original_cholesky = mod.la.cholesky
        LinAlgError = getattr(mod.la, "LinAlgError", np.linalg.LinAlgError)
        state = {"failed": False}

        def flaky_cholesky(*args, **kwargs):
            if not state["failed"]:
                state["failed"] = True
                raise LinAlgError("Fallo forzado para probar regularización Tikhonov.")
            return original_cholesky(*args, **kwargs)

        monkeypatch.setattr(mod.la, "cholesky", flaky_cholesky)

        I3 = np.eye(3, dtype=np.float64)
        agent = KApexElectrodynamicAgent(I3.copy(), I3.copy(), I3.copy())

        assert agent.context.epsilon_G > 0.0, "ε_G debe ser estrictamente positivo tras retry."

        reconstructed = agent.context.L_G @ agent.context.L_G.T
        expected = I3 + agent.context.epsilon_G * I3
        assert_allclose(reconstructed, expected, rtol=_RTOL_STD, atol=_ATOL_STD)

    # ------------------------------------------------------------------
    # (M7) PSD de R_cost y cierre espectral R_sqrt² ≈ R
    # ------------------------------------------------------------------

    def test_R_cost_not_psd_raises(self):
        """
        (M7a) Autovalor negativo macroscópico en R_cost ⇒
        ``ApexSymmetryError`` (conforme a la implementación de Fase 1).
        """
        R = np.diag([1.0, -1.0]).astype(np.float64)
        with pytest.raises(ApexSymmetryError):
            KApexElectrodynamicAgent(np.eye(2), np.eye(2), R)

    def test_R_cost_zero_matrix_is_allowed(self):
        """
        (M7b) R_cost = 0 es PSD válido (disipación nula):
        rank = 0, β₀ = n, gap = 0, R_sqrt = 0, cierre espectral exacto.
        """
        n = 3
        G = np.eye(n, dtype=np.float64)
        agent = KApexElectrodynamicAgent(G, G.copy(), np.zeros((n, n)))
        ctx = agent.context

        assert ctx.rank_R == 0
        assert ctx.betti_0_R == n
        assert ctx.spectral_gap_R == pytest.approx(0.0)
        assert_allclose(ctx.R_sqrt, np.zeros((n, n)), rtol=0.0, atol=0.0)
        assert ctx.spectral_closure_residual <= _ATOL_STD

    def test_R_cost_nullspace_and_spectral_gap(self):
        """
        (M7c) Para R = diag(0, 1, 2):
            rank(R) = 2,  β₀ = 1,  gap = λ₂ − λ₁ = 1 − 0 = 1.
        """
        R = diag_psd([0.0, 1.0, 2.0])
        agent = KApexElectrodynamicAgent(np.eye(3), np.eye(3), R)
        ctx = agent.context

        assert ctx.rank_R == 2
        assert ctx.betti_0_R == 1
        assert ctx.spectral_gap_R == pytest.approx(1.0)
        assert ctx.spectral_closure_residual <= _ATOL_STD

        # Cierre espectral fino: ‖R_sqrt @ R_sqrt − R‖_F ≤ tol.
        closure = ctx.R_sqrt @ ctx.R_sqrt
        assert_allclose(closure, ctx.R_cost, rtol=_RTOL_STD, atol=_ATOL_STD)

    def test_spectral_closure_guard_can_be_forced(self, monkeypatch):
        """
        (M7d) Si se colapsa la cota de Wilkinson (``_WILKINSON_SAFETY < 0``),
        el guard de cierre espectral debe fallar incluso para R = I,
        certificando que ``SpectralClosureError`` está cableado.
        """
        monkeypatch.setattr(mod, "_WILKINSON_SAFETY", -1.0)
        I2 = np.eye(2, dtype=np.float64)
        with pytest.raises(SpectralClosureError):
            KApexElectrodynamicAgent(I2.copy(), I2.copy(), I2.copy())

    # ------------------------------------------------------------------
    # (M8) Contexto canónico sobre el vacío I₃
    # ------------------------------------------------------------------

    def test_build_context_identity_is_canonical(self, identity_agent):
        """
        (M8) Para G = G⁻¹ = R = I₃ el contexto es canónico:
            κ = 1, ε_G = 0, rank(R) = 3, β₀ = 0,
            residuos bilaterales y de cierre ~ O(ε_mach),
            L Lᵀ = I, R_sqrt² = R.
        """
        ctx = identity_agent.context
        I3 = np.eye(3, dtype=np.float64)

        assert isinstance(ctx, ApexPreparationContext)
        assert ctx.dim == 3
        assert ctx.kappa_G == pytest.approx(1.0)
        assert ctx.kappa_G_inv == pytest.approx(1.0)
        assert ctx.epsilon_G == pytest.approx(0.0)
        assert ctx.rank_R == 3
        assert ctx.betti_0_R == 0

        # Residuo bilateral acotado por Wilkinson.
        assert ctx.inverse_residual <= wilkinson_bound(ctx.kappa_G, ctx.dim, safety=1.0)
        assert ctx.spectral_closure_residual <= _ATOL_STD

        assert_allclose(ctx.G_mu_nu, I3, rtol=0.0, atol=0.0)
        assert_allclose(ctx.G_inv, I3, rtol=0.0, atol=0.0)
        assert_allclose(ctx.R_cost, I3, rtol=0.0, atol=0.0)
        assert_allclose(ctx.L_G @ ctx.L_G.T, I3, rtol=_RTOL_STD, atol=_ATOL_STD)
        assert_allclose(ctx.R_sqrt @ ctx.R_sqrt, ctx.R_cost, rtol=_RTOL_STD, atol=_ATOL_STD)

        # Certificados SPD/PSD explícitos.
        assert_spd(ctx.G_mu_nu, "G_mu_nu")
        assert_spd(ctx.G_inv, "G_inv")
        assert_psd(ctx.R_cost, "R_cost")

    # ------------------------------------------------------------------
    # (M9) Inmutabilidad y copias defensivas
    # ------------------------------------------------------------------

    def test_context_is_defensive_copy_of_inputs(self):
        """
        (M9a) El DTO de Fase 1 almacena copias defensivas. Mutar la matriz
        original *después* de ``build_context`` no contamina el contexto.
        """
        G = np.eye(2, dtype=np.float64)
        agent = KApexElectrodynamicAgent(G.copy(), G.copy(), G.copy())
        G[0, 0] = 99.0

        assert agent.context.G_mu_nu[0, 0] == pytest.approx(1.0)
        assert agent.context.G_inv[0, 0] == pytest.approx(1.0)
        assert agent.context.R_cost[0, 0] == pytest.approx(1.0)

    def test_context_dto_is_frozen(self, identity_agent):
        """
        (M9b) ``ApexPreparationContext`` es un frozen dataclass:
        cualquier escritura lanza ``FrozenInstanceError``.
        """
        with pytest.raises(dataclasses.FrozenInstanceError):
            identity_agent.context.dim = 99  # type: ignore[misc]

    # ==================================================================
    # CIERRE FUNTORIAL F₁ → F₂
    # ------------------------------------------------------------------
    # Este es el ÚLTIMO método de Fase 1.  Su veredicto positivo es la
    # precondición algebraica con la que arranca Fase 2: el contexto
    # emitido por Phase1 es *exactamente* el objeto que Phase2 consume,
    # y Phase3 aún no existe (lazy).
    # ==================================================================

    def test_phase1_output_is_formal_precondition_of_phase2(self, identity_agent):
        """
        Contrato funtorial F₁ → F₂
        --------------------------
        1. ``phase2._ctx is agent.context``  (identidad de objeto, no copia).
        2. Los hiperparámetros de control (eikonal_slack, holonomy_tol_rel,
           kappa_max) se propagan sin distorsión a Phase2.
        3. ``phase3 is None``: la proyección en haces es lazy y no se
           materializa hasta la primera exportación de stalk.

        Este invariante es la *definición formal final* de Fase 1 y el
        *axioma de arranque* de Fase 2.
        """
        agent = identity_agent

        # (1) Identidad referencial del contexto.
        assert agent.phase2._ctx is agent.context, (
            "Phase2 debe consumir el mismo ApexPreparationContext emitido por Phase1."
        )

        # (2) Propagación de hiperparámetros.
        assert agent.phase2._eikonal_slack == agent.eikonal_slack
        assert agent.phase2._holonomy_tol_rel == agent.holonomy_tol_rel
        assert agent.phase2._kappa_max == agent.kappa_max

        # (3) Lazy: Phase3 no existe todavía.
        assert agent.phase3 is None, (
            "Phase3 debe permanecer sin instanciar hasta export_sheaf_stalk."
        )

        # (4) El contexto porta dimensión positiva — semilla de todo tensor
        #     de Fase 2 (d_Phi, phase_gradient, E, H, A_gauge ∈ ℝⁿ / Mₙ).
        assert agent.context.dim >= 1
        assert agent.phase2._ctx.dim == agent.context.dim


# ##############################################################################
#
#  FASE 2 — SÍNTESIS ELECTRODINÁMICA
#
#  Continuación formal de Fase 1:
#      ApexPreparationContext  ↦  ApexStateTensor.
#
#  El primer test de esta clase REAFIRMA el contrato F₁→F₂ certificado
#  por el último método de Fase 1, y a partir de ahí despliega:
#
#    (E1)  Inyección de potencial de gauge  s = dΦ · exp(−½ Tr G).
#    (E2)  Eikonal de absorción de mercado   ‖∂S‖²_{G⁻¹} ≥ n(σ*)² (1−slack).
#    (E3)  Balance exergético de Poynting    P_ex = P_in − P_diss ≥ 0.
#    (E4)  Proyector canónico Π_so: Mₙ → 𝔰𝔬(n), idempotente y antisimétrico.
#    (E5)  Cierre de Lie: [B,C] ∈ 𝔰𝔬(n) si B,C ∈ 𝔰𝔬(n).
#    (E6)  Curvatura discreta F antisimétrica; acción YM S_YM = ½‖F‖²_G ≥ 0.
#    (E7)  Holonomía trivial / veto por curvatura (HolonomyVetoError).
#    (E8)  Covarianza de gauge bajo isometrías Qᵀ G Q = G.
#    (E9)  Retícula booleana de viabilidad (álgebra de Boole finita).
#    (E10) synthesize() camino feliz y banderas blandas.
#
#  El ÚLTIMO método de esta fase certifica el contrato F₂ → F₃:
#  gauge_injection_vector es la única entrada formal de Phase3.
#
# ##############################################################################


class TestPhase2ElectrodynamicSynthesis:
    """
    Suite de la Fase 2: ``Phase2_ElectrodynamicSynthesis``.

    Continuación formal de ``TestPhase1MetricValidation``:
    el contexto validado en F₁ es el dominio de todos los operadores
    electrodinámicos auditados aquí.
    """

    # ==================================================================
    # APERTURA FUNTORIAL F₁ → F₂
    # ------------------------------------------------------------------
    # Reafirmación del invariante final de Fase 1.  Si este test falla,
    # el resto de la suite de Fase 2 carece de fundamento.
    # ==================================================================

    def test_phase2_consumes_phase1_context(self, identity_agent):
        """
        Axioma de arranque F₂ (eco del cierre de F₁)
        --------------------------------------------
        Phase2 opera *exactamente* sobre el ``ApexPreparationContext``
        producido por Phase1: misma referencia, misma dimensión.
        """
        assert identity_agent.phase2._ctx is identity_agent.context
        assert identity_agent.phase2._ctx.dim == 3
        assert isinstance(identity_agent.phase2._ctx, ApexPreparationContext)

    # ------------------------------------------------------------------
    # (E1) Inyección de potencial de gauge
    # ------------------------------------------------------------------

    def test_inject_gauge_potential_valid(self, identity_agent):
        """
        (E1a) Para G = I₃, Tr(G) = 3 ⇒ suppression = exp(−3/2).
        s_val = dΦ · suppression  (producto escalar-vector).
        """
        d_Phi = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        s_val, suppression = identity_agent.phase2.inject_gauge_potential(d_Phi)

        expected_suppression = math.exp(-1.5)
        assert suppression == pytest.approx(expected_suppression)
        assert_allclose(s_val, d_Phi * expected_suppression, rtol=_RTOL_STD, atol=_ATOL_STD)

    def test_inject_gauge_potential_dimension_error(self, identity_agent):
        """
        (E1b) d_Phi debe ser un 1-tensor de shape (n,); (n,1) es rechazado.
        """
        with pytest.raises(ApexDimensionError):
            identity_agent.phase2.inject_gauge_potential(
                np.ones((3, 1), dtype=np.float64)
            )

    def test_gauge_potential_collapse_raises(self):
        """
        (E1c) Si Tr(G) ≫ 1 de modo que exp(−½ Tr G) < ε_mach, la inyección
        colapsa numéricamente y debe lanzar ``GaugePotentialError``.
        """
        n = 3
        scale = 100.0 / float(n)  # Tr(G) = 100
        G = scale * np.eye(n, dtype=np.float64)
        G_inv = (1.0 / scale) * np.eye(n, dtype=np.float64)
        agent = KApexElectrodynamicAgent(G, G_inv, np.eye(n))

        with pytest.raises(GaugePotentialError):
            agent.phase2.inject_gauge_potential(np.ones(n, dtype=np.float64))

    # ------------------------------------------------------------------
    # (E2) Eikonal de absorción de mercado
    # ------------------------------------------------------------------

    def test_eikonal_absorption_valid(self, identity_agent):
        """
        (E2a) G⁻¹ = I, phase_gradient = 1⃗, σ* = 0:
            n(σ*) = 1,  ‖∂S‖² = 3.
        """
        phase_gradient = np.ones(3, dtype=np.float64)
        n_refract, eikonal_norm_sq = identity_agent.phase2.compute_eikonal_absorption(
            phase_gradient=phase_gradient,
            sigma_stress=0.0,
            alpha_fermat=0.5,
        )
        assert n_refract == pytest.approx(1.0)
        assert eikonal_norm_sq == pytest.approx(3.0)

    def test_eikonal_absorption_failure_raises(self, identity_agent):
        """
        (E2b) Si ‖∂S‖²_{G⁻¹} cae bajo el umbral Eikonal duro, se lanza
        ``EikonalRefractionError``.
        """
        phase_gradient = np.array([0.8, 0.0, 0.0], dtype=np.float64)
        # norm_sq = 0.64 < 0.9 ≈ 1² · (1 − 0.1)
        with pytest.raises(EikonalRefractionError):
            identity_agent.phase2.compute_eikonal_absorption(
                phase_gradient=phase_gradient,
                sigma_stress=0.0,
            )

    def test_eikonal_negative_sigma_can_lower_threshold(self, identity_agent):
        """
        (E2c) σ* → −∞ ⇒ n(σ*) → 0 ⇒ umbral Eikonal casi nulo.
        El mercado se vuelve topológicamente alcanzable con gradiente
        arbitrariamente pequeño.
        """
        phase_gradient = 1.0e-4 * np.ones(3, dtype=np.float64)
        n_refract, eikonal_norm_sq = identity_agent.phase2.compute_eikonal_absorption(
            phase_gradient=phase_gradient,
            sigma_stress=-100.0,
            alpha_fermat=0.5,
        )
        assert n_refract == pytest.approx(0.0, abs=_ATOL_STD)
        assert eikonal_norm_sq == pytest.approx(3.0e-8)

    def test_eikonal_dimension_error(self, identity_agent):
        """
        (E2d) phase_gradient debe tener shape (n,).
        """
        with pytest.raises(ApexDimensionError):
            identity_agent.phase2.compute_eikonal_absorption(
                phase_gradient=np.ones((3, 1), dtype=np.float64),
                sigma_stress=0.0,
            )

    # ------------------------------------------------------------------
    # (E3) Balance exergético de Poynting
    # ------------------------------------------------------------------

    def test_poynting_exergy_valid(self, identity_agent):
        """
        (E3a) E = H = 1⃗, ∇H = 0⃗:
            P_in = 3, P_diss = 0, P_exergia = 3.
        """
        E = np.ones(3, dtype=np.float64)
        H = np.ones(3, dtype=np.float64)
        grad_H = np.zeros(3, dtype=np.float64)

        P_in, P_diss, P_exergia = identity_agent.phase2.evaluate_poynting_exergy(
            E_field=E, H_field=H, grad_H=grad_H,
        )
        assert P_in == pytest.approx(3.0)
        assert P_diss == pytest.approx(0.0)
        assert P_exergia == pytest.approx(3.0)

    def test_poynting_exergy_zero_boundary_is_allowed(self, identity_agent):
        """
        (E3b) El borde termodinámico P_exergia = 0 es admisible
        (equilibrio detallado ingreso = disipación).
        """
        E = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        H = E.copy()
        grad_H = E.copy()  # R = I ⇒ P_diss = 1

        P_in, P_diss, P_exergia = identity_agent.phase2.evaluate_poynting_exergy(
            E_field=E, H_field=H, grad_H=grad_H,
        )
        assert P_in == pytest.approx(1.0)
        assert P_diss == pytest.approx(1.0)
        assert P_exergia == pytest.approx(0.0)

    def test_poynting_exergy_black_hole_raises(self, identity_agent):
        """
        (E3c) P_diss > P_in más allá de la tolerancia ⇒
        ``FinancialBlackHoleError`` (violación de la 2ª ley exergética).
        """
        E = np.ones(3, dtype=np.float64)
        H = np.ones(3, dtype=np.float64)
        grad_H = 10.0 * np.ones(3, dtype=np.float64)
        # P_in = 3, P_diss = 300 ⇒ P_exergia = −297
        with pytest.raises(FinancialBlackHoleError):
            identity_agent.phase2.evaluate_poynting_exergy(
                E_field=E, H_field=H, grad_H=grad_H,
            )

    def test_poynting_dimension_error(self, identity_agent):
        """
        (E3d) Todos los campos vectoriales deben tener shape (n,).
        """
        with pytest.raises(ApexDimensionError):
            identity_agent.phase2.evaluate_poynting_exergy(
                E_field=np.ones((3, 1), dtype=np.float64),
                H_field=np.ones(3, dtype=np.float64),
                grad_H=np.zeros(3, dtype=np.float64),
            )

    # ------------------------------------------------------------------
    # (E4) Proyector canónico a 𝔰𝔬(n)
    # ------------------------------------------------------------------

    def test_project_to_so_is_idempotent_and_antisymmetric(self, identity_agent):
        """
        (E4) Π_so(A) = ½(A − Aᵀ) verifica:
            - idempotencia:  Π² = Π,
            - imagen en 𝔰𝔬(n):  Π(A)ᵀ = −Π(A).
        """
        rng = np.random.default_rng(101)
        A = rng.normal(size=(3, 3))

        P1 = identity_agent.phase2._project_to_so(A)
        P2 = identity_agent.phase2._project_to_so(P1)

        assert_allclose(P1, P2, rtol=1.0e-13, atol=_ATOL_STRICT)
        assert_allclose(P1, -P1.T, rtol=1.0e-13, atol=_ATOL_STRICT)
        assert frobenius_asymmetry(P1) <= _ATOL_STD

    # ------------------------------------------------------------------
    # (E5) Cierre de Lie
    # ------------------------------------------------------------------

    def test_lie_commutator_closure_in_so(self, identity_agent):
        """
        (E5) Si B, C ∈ 𝔰𝔬(n), entonces [B, C] = BC − CB ∈ 𝔰𝔬(n)
        (el conmutador de matrices es el bracket de Lie de 𝔤𝔩(n) y
        𝔰𝔬(n) es subálgebra).
        """
        B = skew_from_pairs(3, [(0, 1)], scale=1.0)
        C = skew_from_pairs(3, [(1, 2)], scale=1.0)
        comm = identity_agent.phase2._lie_commutator(B, C)

        assert_allclose(comm, -comm.T, rtol=1.0e-13, atol=_ATOL_STRICT)
        assert frobenius_asymmetry(comm) <= _ATOL_STD

    # ------------------------------------------------------------------
    # (E6) Curvatura y acción de Yang-Mills
    # ------------------------------------------------------------------

    def test_compute_curvature_is_antisymmetric(self, identity_agent):
        """
        (E6a) La curvatura discreta F satisface Fᵀ = −F hasta precisión
        de máquina, *incluso* si A₁, A₂ no son antisimétricas de entrada
        (el proyector Π_so se aplica internamente).
        """
        rng = np.random.default_rng(202)
        A1 = rng.normal(size=(3, 3))
        A2 = rng.normal(size=(3, 3))
        F = identity_agent.phase2._compute_curvature(A1, A2)

        assert frobenius_asymmetry(F) <= _ATOL_STD

    def test_yang_mills_action_is_nonnegative_and_exact_for_identity_metric(
        self, identity_agent
    ):
        """
        (E6b) Para G = I, S_YM = ½ ‖F‖²_F ≥ 0.  Igualdad exacta con la
        fórmula de Frobenius.
        """
        F = skew_from_pairs(3, [(0, 1), (1, 2)], scale=0.1)
        S = identity_agent.phase2._yang_mills_action(F)
        expected = 0.5 * float(np.linalg.norm(F, "fro") ** 2)

        assert S >= 0.0
        assert_allclose(S, expected, rtol=_RTOL_STD, atol=1.0e-15)

    # ------------------------------------------------------------------
    # (E7) Holonomía
    # ------------------------------------------------------------------

    def test_audit_holonomy_trivial_for_zero_connections(self, identity_agent):
        """
        (E7a) A₁ = A₂ = 0 ⇒ F = 0 ⇒ S_YM = 0 y residual de antisimetría = 0
        (holonomía trivial del fibrado principal).
        """
        A = np.zeros((3, 3), dtype=np.float64)
        S, residual = identity_agent.phase2.audit_yang_mills_holonomy(A, A)
        assert S == pytest.approx(0.0)
        assert residual == pytest.approx(0.0)

    def test_audit_holonomy_veto_raises_for_nontrivial_curvature(self):
        """
        (E7b) Plaqueta con generadores no conmutativos de 𝔰𝔬(3) produce
        curvatura no nula.  Con tolerancia relativa ultra-estricta se
        lanza ``HolonomyVetoError``.
        """
        I3 = np.eye(3, dtype=np.float64)
        agent = KApexElectrodynamicAgent(
            G_mu_nu=I3.copy(),
            G_inv=I3.copy(),
            R_cost=I3.copy(),
            holonomy_tol_rel=1.0e-12,
        )
        A1 = skew_from_pairs(3, [(0, 1)], scale=1.0)
        A2 = skew_from_pairs(3, [(1, 2)], scale=1.0)

        with pytest.raises(HolonomyVetoError):
            agent.phase2.audit_yang_mills_holonomy(A1, A2)

    def test_audit_holonomy_dimension_error(self, identity_agent):
        """
        (E7c) A_gauge_1 y A_gauge_2 deben ser matrices n×n compatibles.
        """
        with pytest.raises(ApexDimensionError):
            identity_agent.phase2.audit_yang_mills_holonomy(
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
            )

    def test_antisymmetry_canary_detects_invalid_curvature(
        self, identity_agent, monkeypatch
    ):
        """
        (E7d) Canario de antisimetría: si un bug interno produjera F
        simétrica (p.ej. F = I), debe lanzarse ``ElectrodynamicApexError``.
        """
        phase2 = identity_agent.phase2
        monkeypatch.setattr(
            phase2,
            "_compute_curvature",
            lambda A1, A2: np.eye(3, dtype=np.float64),
        )
        A = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ElectrodynamicApexError):
            phase2.audit_yang_mills_holonomy(A, A)

    # ------------------------------------------------------------------
    # (E8) Covarianza de gauge
    # ------------------------------------------------------------------

    def test_gauge_covariance_identity_is_exact(self, identity_agent):
        """
        (E8a) Q = I es isometría trivial ⇒ residual de covarianza = 0.
        """
        A = np.zeros((3, 3), dtype=np.float64)
        Q = np.eye(3, dtype=np.float64)
        residual = identity_agent.phase2.verify_gauge_covariance(A, A, Q)
        assert residual == pytest.approx(0.0)

    def test_gauge_covariance_under_orthogonal_isometry_for_identity_metric(self):
        """
        (E8b) Para G = I toda Q ∈ O(n) es isometría.  S_YM es invariante
        bajo la acción adjunta A ↦ Q A Qᵀ.
        """
        I3 = np.eye(3, dtype=np.float64)
        agent = KApexElectrodynamicAgent(
            G_mu_nu=I3.copy(),
            G_inv=I3.copy(),
            R_cost=I3.copy(),
            holonomy_tol_rel=1.0,  # holgado: no vetar S_YM en el diagnóstico
        )
        rng = np.random.default_rng(303)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))

        A1 = skew_from_pairs(3, [(0, 1)], scale=1.0e-3)
        A2 = skew_from_pairs(3, [(1, 2)], scale=1.0e-3)

        residual = agent.phase2.verify_gauge_covariance(A1, A2, Q)
        assert residual <= 1.0e-6

    def test_gauge_covariance_invalid_isometry_raises(self, identity_agent):
        """
        (E8c) Si Q no preserva G (Qᵀ G Q ≠ G), se lanza
        ``GaugeCovarianceError`` *antes* de comparar acciones YM.
        """
        A = np.zeros((3, 3), dtype=np.float64)
        Q = 2.0 * np.eye(3, dtype=np.float64)
        with pytest.raises(GaugeCovarianceError):
            identity_agent.phase2.verify_gauge_covariance(A, A, Q)

    # ------------------------------------------------------------------
    # (E9) Retícula booleana de viabilidad
    # ------------------------------------------------------------------

    def test_viability_flags_boolean_lattice(self):
        """
        (E9a) La retícula de viabilidad se comporta como álgebra de Boole
        finita:
            meet = &,  join = |,  ALL es unidad de orden (top).
        """
        a = ApexViabilityFlags.GAUGE_INJECTION_NONTRIVIAL
        b = ApexViabilityFlags.EXERGY_NONNEGATIVE

        join = a.join(b)
        meet = join.meet(a)

        assert bool(join & a)
        assert bool(join & b)
        assert meet == a

        assert ApexViabilityFlags.ALL.is_order_unit()
        assert not a.is_order_unit()

        assert (a | b) == join
        assert (join & a) == a

    def test_describe_viability_flags_serialization(self):
        """
        (E9b) ``describe_viability_flags`` serializa predicados satisfechos
        y violados, e indica viabilidad total.
        """
        flags = ApexViabilityFlags.GAUGE_INJECTION_NONTRIVIAL
        text = describe_viability_flags(flags)

        assert "SATISFECHOS=" in text
        assert "GAUGE_INJECTION_NONTRIVIAL" in text
        assert "VIOLADOS=" in text
        assert "VIABLE_TOTAL=False" in text

    # ------------------------------------------------------------------
    # (E10) synthesize — camino feliz y banderas blandas
    # ------------------------------------------------------------------

    def test_synthesize_viable_all_flags_true(self, identity_agent):
        """
        (E10a) Camino feliz completo de Fase 2: todas las banderas de
        viabilidad quedan en ALL y el estado es electrodinámicamente viable.
        """
        kwargs = valid_synthesis_kwargs(identity_agent, phase_norm_target=1.05)
        state = identity_agent.synthesize_apex_field(**kwargs)

        assert isinstance(state, ApexStateTensor)
        assert state.viability_flags == ApexViabilityFlags.ALL
        assert state.is_electrodynamically_viable is True

        assert state.gauge_injection_vector.shape == (3,)
        assert state.suppression_factor == pytest.approx(math.exp(-1.5))
        assert state.fermat_refractive_index == pytest.approx(1.0)
        assert state.eikonal_norm_sq == pytest.approx(1.05)
        assert state.poynting_income == pytest.approx(3.0)
        assert state.poynting_dissipation == pytest.approx(0.0)
        assert state.poynting_exergy_flux == pytest.approx(3.0)
        assert state.yang_mills_action == pytest.approx(0.0)
        assert state.curvature_antisymmetry_residual == pytest.approx(0.0)
        assert state.gauge_covariance_residual is None

    def test_synthesize_soft_eikonal_margin_flag_false(self, identity_agent):
        """
        (E10b) Margen Eikonal que supera el umbral *duro* pero no el
        *blando*: synthesize no lanza, pero ``EIKONAL_MARGIN_SOUND`` es
        falsa y la viabilidad total falla.
        """
        kwargs = valid_synthesis_kwargs(identity_agent)
        kwargs["phase_gradient"] = np.array(
            [math.sqrt(0.92), 0.0, 0.0], dtype=np.float64
        )
        state = identity_agent.synthesize_apex_field(**kwargs)

        assert state.eikonal_norm_sq == pytest.approx(0.92)
        assert not bool(
            state.viability_flags & ApexViabilityFlags.EIKONAL_MARGIN_SOUND
        )
        assert state.viability_flags != ApexViabilityFlags.ALL
        assert state.is_electrodynamically_viable is False

    def test_synthesize_metric_well_conditioned_flag_false(self):
        """
        (E10c) Si ½ κ_max < κ(G) ≤ κ_max, la bandera
        ``METRIC_WELL_CONDITIONED`` es falsa (condición holgada pero no
        óptima).
        """
        G, G_inv = diag_spd([1.0, 2.0])
        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=np.eye(2), kappa_max=3.0,
        )
        kwargs = valid_synthesis_kwargs(agent, phase_norm_target=1.05)
        state = agent.synthesize_apex_field(**kwargs)

        assert not bool(
            state.viability_flags & ApexViabilityFlags.METRIC_WELL_CONDITIONED
        )
        assert state.viability_flags != ApexViabilityFlags.ALL
        assert state.is_electrodynamically_viable is False

    def test_synthesize_with_gauge_covariance_diagnostic(self, identity_agent):
        """
        (E10d) Con Q_isometry_diagnostic = I se materializa el residual
        de covarianza de gauge (ruta diagnóstica opcional).
        """
        kwargs = valid_synthesis_kwargs(identity_agent)
        kwargs["Q_isometry_diagnostic"] = np.eye(3, dtype=np.float64)
        state = identity_agent.synthesize_apex_field(**kwargs)

        assert state.gauge_covariance_residual is not None
        assert state.gauge_covariance_residual == pytest.approx(0.0)

    def test_synthesize_dimension_error(self, identity_agent):
        """
        (E10e) Vector mal dimensionado en synthesize ⇒ ``ApexDimensionError``.
        """
        kwargs = valid_synthesis_kwargs(identity_agent)
        kwargs["d_Phi"] = np.ones((3, 1), dtype=np.float64)
        with pytest.raises(ApexDimensionError):
            identity_agent.synthesize_apex_field(**kwargs)

    def test_state_tensor_dto_is_frozen(self, identity_agent):
        """
        (E10f) ``ApexStateTensor`` es inmutable (frozen dataclass).
        """
        kwargs = valid_synthesis_kwargs(identity_agent)
        state = identity_agent.synthesize_apex_field(**kwargs)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.suppression_factor = 0.0  # type: ignore[misc]

    # ==================================================================
    # CIERRE FUNTORIAL F₂ → F₃
    # ------------------------------------------------------------------
    # Último método de Fase 2.  Certifica que gauge_injection_vector —
    # campo del ApexStateTensor — es la única entrada formal que
    # Phase3.export_stalk está autorizada a consumir.  Este veredicto
    # es el axioma de arranque de Fase 3.
    # ==================================================================

    def test_phase2_output_is_formal_precondition_of_phase3(self, identity_agent):
        """
        Contrato funtorial F₂ → F₃
        --------------------------
        1. ``synthesize`` produce un ``ApexStateTensor`` viable.
        2. ``state.gauge_injection_vector`` tiene shape (n,) y es finito.
        3. Phase3 aún es ``None`` (lazy) *antes* de la exportación.
        4. ``export_sheaf_stalk(state.gauge_injection_vector)`` materializa
           Phase3 y devuelve un ``SheafStalkApex`` cuyo
           ``source_injection`` coincide exactamente con el vector de
           inyección de gauge.

        Este invariante es la *definición formal final* de Fase 2 y el
        *axioma de arranque* de Fase 3.
        """
        kwargs = valid_synthesis_kwargs(identity_agent, phase_norm_target=1.05)
        state = identity_agent.synthesize_apex_field(**kwargs)

        # (1)–(2) Tensor de estado bien formado.
        assert isinstance(state, ApexStateTensor)
        assert state.is_electrodynamically_viable is True
        s = state.gauge_injection_vector
        assert s.shape == (identity_agent.context.dim,)
        assert np.all(np.isfinite(s))

        # (3) Lazy antes de exportar.
        assert identity_agent.phase3 is None

        # (4) Continuación terminal: F₂.s ↦ F₃.stalk.
        stalk = identity_agent.export_sheaf_stalk(s)
        assert identity_agent.phase3 is not None
        assert isinstance(stalk, SheafStalkApex)
        assert_allclose(stalk.source_injection, s, rtol=_RTOL_STD, atol=_ATOL_STD)


# ##############################################################################
#
#  FASE 3 — PROYECCIÓN EN HACES Y LAPLACIANO DE HODGE LOCAL
#
#  Continuación formal de Fase 2:
#      gauge_injection_vector  ↦  SheafStalkApex.
#
#  El primer test REAFIRMA el contrato F₂→F₃ y a partir de ahí verifica:
#
#    (H1)  Inicialización lazy de Phase3.
#    (H2)  Identidad de Hodge local  δ_metricᵀ G δ_metric ≈ I.
#    (H3)  Forma y rango: δ_APEX ∈ ℝ^{2n×n}, rank = n.
#    (H4)  Laplaciano Δ = I + δ_dissᵀ δ_diss ≻ 0.
#    (H5)  Proyecciones métrica / disipativa de s_val.
#    (H6)  β₀ heredado como lossless_subspace_dimension.
#    (H7)  Espectro de Hodge: brecha y número de condición.
#    (H8)  Errores dimensionales y SheafMetricError forzado.
#    (H9)  Copias defensivas y congelamiento del DTO terminal.
#    (H10) Composición funtorial completa F₁ → F₂ → F₃.
#
# ##############################################################################


class TestPhase3SheafProjection:
    """
    Suite de la Fase 3: ``Phase3_SheafProjection``.

    Continuación formal de ``TestPhase2ElectrodynamicSynthesis``:
    consume ``gauge_injection_vector`` y produce el stalk terminal del haz.
    """

    # ==================================================================
    # APERTURA FUNTORIAL F₂ → F₃
    # ------------------------------------------------------------------
    # Eco del cierre de Fase 2: export_stalk acepta el vector de inyección
    # de gauge y devuelve un SheafStalkApex coherente.
    # ==================================================================

    def test_export_stalk_is_terminal_continuation_of_phase2(self, identity_agent):
        """
        Axioma de arranque F₃ (eco del cierre de F₂)
        --------------------------------------------
        ``export_stalk`` acepta ``state.gauge_injection_vector`` y devuelve
        un ``SheafStalkApex`` cuyo ``source_injection`` es exactamente ese
        vector, con ``rank_delta = n``.
        """
        kwargs = valid_synthesis_kwargs(identity_agent)
        state = identity_agent.synthesize_apex_field(**kwargs)
        stalk = identity_agent.export_sheaf_stalk(state.gauge_injection_vector)

        assert isinstance(stalk, SheafStalkApex)
        assert_allclose(
            stalk.source_injection,
            state.gauge_injection_vector,
            rtol=_RTOL_STD,
            atol=_ATOL_STD,
        )
        assert stalk.rank_delta == identity_agent.context.dim

    # ------------------------------------------------------------------
    # (H1) Lazy init
    # ------------------------------------------------------------------

    def test_phase3_lazy_initialization(self, identity_agent):
        """
        (H1) Phase3 no se instancia en el constructor del agente, sino en
        la primera llamada a ``export_sheaf_stalk``.
        """
        assert identity_agent.phase3 is None
        stalk = identity_agent.export_sheaf_stalk(np.ones(3, dtype=np.float64))
        assert identity_agent.phase3 is not None
        assert isinstance(stalk, SheafStalkApex)

    # ------------------------------------------------------------------
    # (H2) Identidad de Hodge local
    # ------------------------------------------------------------------

    def test_delta_metric_satisfies_hodge_identity(self, identity_agent):
        """
        (H2) Fibra métrica:
            δ_metricᵀ  G  δ_metric  ≈  I_n.

        El residuo debe respetar la cota de Wilkinson escalada por κ(G).
        """
        s = np.ones(3, dtype=np.float64)
        stalk = identity_agent.export_sheaf_stalk(s)

        G = identity_agent.context.G_mu_nu
        I3 = np.eye(3, dtype=np.float64)
        composed = stalk.delta_metric.T @ G @ stalk.delta_metric

        assert_allclose(composed, I3, rtol=_RTOL_HODGE, atol=_ATOL_HODGE)

        bound = wilkinson_bound(identity_agent.context.kappa_G, 3, safety=100.0)
        assert stalk.hodge_metric_residual <= bound

    # ------------------------------------------------------------------
    # (H3) Forma y rango de δ_APEX
    # ------------------------------------------------------------------

    def test_delta_apex_shape_and_rank(self, identity_agent):
        """
        (H3) δ_APEX ∈ ℝ^{2n × n} se apila como
            [ δ_metric ; δ_dissipative ],
        y rank(δ_APEX) = n porque δ_metric es invertible (Hodge).
        """
        s = np.ones(3, dtype=np.float64)
        stalk = identity_agent.export_sheaf_stalk(s)

        assert stalk.delta_apex.shape == (6, 3)
        assert stalk.delta_metric.shape == (3, 3)
        assert stalk.delta_dissipative.shape == (3, 3)
        assert stalk.rank_delta == 3

        # Consistencia de apilado.
        stacked = np.vstack([stalk.delta_metric, stalk.delta_dissipative])
        assert_allclose(stalk.delta_apex, stacked, rtol=0.0, atol=0.0)

    # ------------------------------------------------------------------
    # (H4) Laplaciano de Hodge local
    # ------------------------------------------------------------------

    def test_hodge_laplacian_definition_and_spd(self, identity_agent):
        """
        (H4) Definición y positividad:
            Δ = I + δ_dissᵀ δ_diss  ≻  0.
        """
        s = np.ones(3, dtype=np.float64)
        stalk = identity_agent.export_sheaf_stalk(s)

        I3 = np.eye(3, dtype=np.float64)
        expected = I3 + stalk.delta_dissipative.T @ stalk.delta_dissipative
        assert_allclose(stalk.hodge_laplacian, expected, rtol=_RTOL_STD, atol=_ATOL_STD)

        eigvals = assert_spd(stalk.hodge_laplacian, "hodge_laplacian")
        assert eigvals[0] > 0.0

    # ------------------------------------------------------------------
    # (H5) Proyecciones de la fuente
    # ------------------------------------------------------------------

    def test_source_projections_are_consistent(self, identity_agent):
        """
        (H5) Las secciones exportadas cumplen:
            projected_metric      = δ_metric · s,
            projected_dissipative = δ_diss  · s.
        """
        s = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        stalk = identity_agent.export_sheaf_stalk(s)

        assert_allclose(
            stalk.projected_source_metric,
            stalk.delta_metric @ s,
            rtol=_RTOL_STD,
            atol=_ATOL_STD,
        )
        assert_allclose(
            stalk.projected_source_dissipative,
            stalk.delta_dissipative @ s,
            rtol=_RTOL_STD,
            atol=_ATOL_STD,
        )

    # ------------------------------------------------------------------
    # (H6) Herencia de β₀
    # ------------------------------------------------------------------

    def test_lossless_subspace_dimension_is_inherited_from_beta0(self):
        """
        (H6) ``lossless_subspace_dimension`` hereda β₀ = dim ker(R_cost).
        Para R = diag(0, 1, 2) ⇒ β₀ = 1.
        """
        R = diag_psd([0.0, 1.0, 2.0])
        agent = KApexElectrodynamicAgent(np.eye(3), np.eye(3), R)
        stalk = agent.export_sheaf_stalk(np.ones(3, dtype=np.float64))

        assert stalk.lossless_subspace_dimension == 1
        assert stalk.lossless_subspace_dimension == agent.context.betti_0_R

    # ------------------------------------------------------------------
    # (H7) Espectro de Hodge
    # ------------------------------------------------------------------

    def test_hodge_spectrum_nontrivial_gap_and_condition(self):
        """
        (H7a) G = I, R = diag(1, 2, 3):
            Δ = I + R = diag(2, 3, 4),
            gap = 3 − 2 = 1,
            κ(Δ) = 4 / 2 = 2.
        """
        R = diag_psd([1.0, 2.0, 3.0])
        agent = KApexElectrodynamicAgent(np.eye(3), np.eye(3), R)
        stalk = agent.export_sheaf_stalk(np.ones(3, dtype=np.float64))

        assert stalk.hodge_spectral_gap == pytest.approx(1.0)
        assert stalk.hodge_condition_number == pytest.approx(2.0)

    def test_hodge_spectrum_zero_dissipation(self):
        """
        (H7b) R = 0 ⇒ Δ = I, gap = 0, κ(Δ) = 1.
        """
        agent = KApexElectrodynamicAgent(
            np.eye(3), np.eye(3), np.zeros((3, 3)),
        )
        stalk = agent.export_sheaf_stalk(np.ones(3, dtype=np.float64))

        assert stalk.hodge_spectral_gap == pytest.approx(0.0)
        assert stalk.hodge_condition_number == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # (H8) Errores
    # ------------------------------------------------------------------

    def test_export_stalk_dimension_error(self, identity_agent):
        """
        (H8a) s_val debe tener shape (n,).
        """
        with pytest.raises(ApexDimensionError):
            identity_agent.export_sheaf_stalk(np.ones((3, 1), dtype=np.float64))

    def test_hodge_metric_error_can_be_forced(self, monkeypatch):
        """
        (H8b) Si se colapsa ``_WILKINSON_SAFETY < 0``, la identidad de
        Hodge falla y se lanza ``SheafMetricError``.
        """
        G, G_inv = diag_spd([1.0, 2.0, 4.0])
        R = diag_psd([1.0, 2.0, 3.0])
        agent = KApexElectrodynamicAgent(G, G_inv, R)

        # Phase3 aún no instanciada; la tolerancia se lee en su __init__.
        monkeypatch.setattr(mod, "_WILKINSON_SAFETY", -1.0)

        with pytest.raises(SheafMetricError):
            agent.export_sheaf_stalk(np.ones(3, dtype=np.float64))

    # ------------------------------------------------------------------
    # (H9) Copias defensivas y frozen DTO
    # ------------------------------------------------------------------

    def test_export_stalk_returns_defensive_copies(self, identity_agent):
        """
        (H9a) Cada exportación devuelve tensores frescos (no aliases
        internos).  El contenido numérico es idéntico bit a bit.
        """
        s = np.ones(3, dtype=np.float64)
        stalk1 = identity_agent.export_sheaf_stalk(s)
        stalk2 = identity_agent.export_sheaf_stalk(s)

        assert stalk1.delta_apex is not stalk2.delta_apex
        assert stalk1.delta_metric is not stalk2.delta_metric
        assert stalk1.delta_dissipative is not stalk2.delta_dissipative
        assert stalk1.hodge_laplacian is not stalk2.hodge_laplacian

        assert_allclose(stalk1.delta_apex, stalk2.delta_apex, rtol=0.0, atol=0.0)
        assert_allclose(stalk1.delta_metric, stalk2.delta_metric, rtol=0.0, atol=0.0)
        assert_allclose(
            stalk1.delta_dissipative, stalk2.delta_dissipative, rtol=0.0, atol=0.0,
        )
        assert_allclose(
            stalk1.hodge_laplacian, stalk2.hodge_laplacian, rtol=0.0, atol=0.0,
        )

    def test_sheaf_stalk_dto_is_frozen(self, identity_agent):
        """
        (H9b) ``SheafStalkApex`` es inmutable.
        """
        stalk = identity_agent.export_sheaf_stalk(np.ones(3, dtype=np.float64))
        with pytest.raises(dataclasses.FrozenInstanceError):
            stalk.rank_delta = 99  # type: ignore[misc]

    # ==================================================================
    # (H10) COMPOSICIÓN FUNTORIAL COMPLETA F₁ → F₂ → F₃
    # ------------------------------------------------------------------
    # Cierre terminal de la suite: el diagrama conmuta de extremo a
    # extremo sobre un sistema diagonal no trivial.
    # ==================================================================

    def test_full_functorial_pipeline_phase1_phase2_phase3(self, diagonal_agent):
        """
        Composición funtorial completa
        --------------------------------
            build_context ──► synthesize ──► export_stalk.

        Sistema testigo (diagonal, bien condicionado)::

            G = diag(1, 2, 3),   R = diag(0.5, 1.5, 2.5).

        Obligaciones
        ------------
        - F₁: contexto válido, rank(R) = 3, β₀ = 0.
        - F₂: síntesis con viability_flags = ALL.
        - F₃: stalk terminal con rank_delta = 3, lossless = 0,
              source_injection = gauge_injection_vector,
              Δ_Hodge ≻ 0, residuo de Hodge ≤ cota de Wilkinson.
        """
        agent = diagonal_agent

        # ── F₁ ────────────────────────────────────────────────────────
        assert isinstance(agent.context, ApexPreparationContext)
        assert agent.context.dim == 3
        assert agent.context.rank_R == 3
        assert agent.context.betti_0_R == 0
        assert_spd(agent.context.G_mu_nu, "G")
        assert_psd(agent.context.R_cost, "R")

        # ── F₂ ────────────────────────────────────────────────────────
        kwargs = valid_synthesis_kwargs(agent, phase_norm_target=1.1)
        state = agent.synthesize_apex_field(**kwargs)

        assert isinstance(state, ApexStateTensor)
        assert state.viability_flags == ApexViabilityFlags.ALL
        assert state.is_electrodynamically_viable is True
        assert state.gauge_injection_vector.shape == (3,)
        assert np.all(np.isfinite(state.gauge_injection_vector))

        # ── F₃ ────────────────────────────────────────────────────────
        stalk = agent.export_sheaf_stalk(state.gauge_injection_vector)

        assert isinstance(stalk, SheafStalkApex)
        assert stalk.rank_delta == 3
        assert stalk.lossless_subspace_dimension == 0
        assert_allclose(
            stalk.source_injection,
            state.gauge_injection_vector,
            rtol=_RTOL_STD,
            atol=_ATOL_STD,
        )

        # Δ_APEX ≻ 0.
        eigvals = assert_spd(stalk.hodge_laplacian, "Δ_Hodge")
        assert np.all(eigvals > 0.0)

        # Identidad de Hodge numéricamente sólida.
        bound = wilkinson_bound(agent.context.kappa_G, agent.context.dim, safety=100.0)
        assert stalk.hodge_metric_residual <= bound

        # Eco del apilado δ_APEX = [δ_metric; δ_diss].
        assert stalk.delta_apex.shape == (2 * agent.context.dim, agent.context.dim)