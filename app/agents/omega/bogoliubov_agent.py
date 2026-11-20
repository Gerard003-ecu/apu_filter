# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Bogoliubov Agent — Soberano de Diagonalización de Cuasipartículas           ║
║ Ruta   : app/agents/omega/bogoliubov_agent.py                                        ║
║ Versión: 5.0.0-Bogoliubov-RSI3-Poincare-Heyting-Krein-ESP32-PhD                      ║
║ Estrato: V_Ω / V_WISDOM — El Ágora Tensorial                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA EN TRES FASES ANIDADAS (composición de Kleisli / CPS)
──────────────────────────────────────────────────────────────────────────────────────
    Φ₁ : (U,V,M,G) ─────────────────────────► S  (GeometricSubstrate)
    Φ₂ : S ─────────────────────────────────► F  (DynamicalFrame)
    Φ₃ : F ─────────────────────────────────► C  (BogoliubovGovernanceCertificate)

    El último método de la Fase 1 (`certify_geometric_substrate`) tiene como definición
    formal final el parámetro `continuation: Callable[[S], R]`; su cola es literalmente
    `return continuation(S)`, y la continuación canónica es el primer método de la
    Fase 2 (`evolve_ricci_metric_R1`). Idénticamente, el último método de la Fase 2
    (`evolve_dynamical_frame`) termina en `continuation(F)` = primer método de la Fase 3
    (`close_rsi3_loop`). El anidamiento es, por tanto, *tipológico y operacional*, no
    cosmético: Φ₃ ∘ Φ₂ ∘ Φ₁ es una composición laxa-funtorial hacia el retículo de
    Heyting Ω₃ (el veredicto del compuesto es ⊑ al meet de los veredictos parciales).

    Fase 1 — SUSTRATO: álgebra de operadores (CAR/CCR completos), grupo Sp(2n,ℝ),
             geometría SPD, cota de Connes, espectro de cuasipartículas, Ω₃.
    Fase 2 — FLUJOS: R₁ Ricci–DeTurck normalizado, R₂ retracción canónica (polar /
             Newton sobre involuciones), MECÁNICA CELESTE DE POINCARÉ (Delaunay,
             Laplace–Lagrange, sección de retorno, monodromía/Floquet, Krein, residuos
             de Greene, Melnikov, Chirikov, KAM-Diofanto, recurrencia de Kac),
             R₃ Brockett–Shahshahani iso-espectral.
    Fase 3 — CLAUSURA RSI³: ledger de Lyapunov, certificado de contracción de Banach,
             auto-modificación de nivel 3 con rollback, crowbar ESP32, certificado.

AXIOMAS (con su estatus epistémico explícito)
──────────────────────────────────────────────────────────────────────────────────────
 A1 [TEOREMA] CAR (fermiones): W = [[U,V],[V̄,Ū]] unitaria ⇔
      UU†+VV†=I, UVᵀ+VUᵀ=0  (y, en dimensión finita, equivalentemente W†W=I).
 A2 [TEOREMA] CCR (bosones): W J W† = J, J=diag(I,−I) ⇔
      UU†−VV†=I, UVᵀ−VUᵀ=0.  W es paraunitaria: W ∈ U(n,n) con estructura real.
 A3 [TEOREMA] Liouville: MᵀΩM=Ω ⇒ det M=+1 (recíproco FALSO: se audita el residuo
      simpléctico, no el determinante). Consecuencia dinámica: Poincaré-recurrencia.
 A4 [CONVENCIÓN] Cota operacional Connes–DK L ≤ 1/(2 λ_min(G)^{3/2}). NO es un teorema
      de geometría no conmutativa: es homogénea de grado −3/2 (no invariante de escala).
      Se reporta además la versión adimensional L·λ_max^{3/2} = κ(G)^{3/2}/2 y la cota
      rigurosa del seminorma de Connes L(a)=‖[D,a]‖ ≤ 2‖D‖‖a‖.
 A5 [TEOREMA] Brockett: d/dt tr(ρN) = ‖[ρ,N]‖_F² ≥ 0 con ρ̇=[ρ,[ρ,N]].
 A6 [TEOREMA] Fisher/Shahshahani: d/dt (½pᵀKp) = Var_p(Kp) ≥ 0 si K=Kᵀ.
 A7 [TEOREMA] Fan–Hoffman: el factor polar unitario es el unitario más próximo en toda
      norma unitariamente invariante ⇒ retracción R₂ óptima para CAR.
 A8 [TEOREMA] Krein–Moser: colisión de multiplicadores de Floquet con signaturas
      simplécticas opuestas ⇒ bifurcación Hamiltoniana–Hopf (inestabilidad).
 A9 [TEOREMA] Poincaré–Birkhoff–Smale: ceros simples de la función de Melnikov ⇒
      intersecciones homoclínicas transversas ⇒ herradura de Smale ⇒ caos.
 A10 [TEOREMA] Banach (a posteriori): si ‖T(x)−T(y)‖≤q‖x−y‖, q<1, entonces
      d(x_k,x*) ≤ q/(1−q)·‖x_k−x_{k−1}‖. Base del criterio de parada de RSI³.
 A11 [LÓGICA] Ω₃ ≅ O(espacio de Sierpiński) ≅ Ω del topos Set^(•→•): álgebra de Heyting
      de 3 elementos, NO booleana (¬¬a ≠ a). "No verificado" ≠ ⊤: se marca DEGRADED.
 A12 [SAFETY] El núcleo verificador es INMUTABLE. RSI³ sólo reescribe hiperparámetros
      y estados, nunca el verificador (evita el obstáculo löbiano de auto-verificación).

CONVENCIONES NUMÉRICAS
──────────────────────────────────────────────────────────────────────────────────────
 ħ = 1; σ_vac = I/2 (Williamson ν_min ≥ ½ = Robertson–Schrödinger).
 Todos los residuos son ADIMENSIONALES (normalizados por escala y por √n).
 Ninguna afirmación de tiempo real dura en Python: el presupuesto de 400 ns del crowbar
 es un requisito de FIRMWARE (ESP32: MCPWM/RMT/GPIO + comparador analógico). Aquí sólo
 se calcula la palabra de latch en O(1) y se mide el sobrecoste del intérprete.

DEPENDENCIAS: numpy ≥ 1.24, scipy ≥ 1.9, Python ≥ 3.10 (dataclass slots).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import lru_cache
from itertools import product
from typing import Callable, Final, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:  # Acoplamiento al núcleo categorial del MIC (Morphism = morfismo del topos Ω).
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover - degradation grácil fuera del Ágora.
    class Morphism:  # type: ignore[no-redef]
        """Morfismo identidad degenerado (fallback sin el núcleo MIC)."""

    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        """Violación de invariante topológico/algebraico (veto duro)."""

__all__ = [
    "HeytingVerdict",
    "heyting_meet",
    "heyting_join",
    "heyting_implies",
    "heyting_negation",
    "CanonicalAlgebraReport",
    "SymplecticAudit",
    "MetricGeometryReport",
    "QuasiparticleSpectrum",
    "GeometricSubstrate",
    "RicciFlowState",
    "CanonicalRetraction",
    "DelaunayChart",
    "FloquetKreinReport",
    "ResonanceCensus",
    "MelnikovReport",
    "PoincareCelestialReport",
    "BrockettShahshahaniState",
    "DynamicalFrame",
    "RSI3Hyperparameters",
    "RSI3MetaReport",
    "CrowbarDecision",
    "BogoliubovGovernanceCertificate",
    "OmegaCycleOutcome",
    "BogoliubovAgent",
    "TopologicalInvariantError",
]

logger = logging.getLogger("MIC.Omega.BogoliubovAgent")

R = TypeVar("R")  # Tipo de retorno de la continuación (anidamiento CPS entre fases).

# ─────────────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE CALIBRE (todas adimensionales salvo indicación)
# ─────────────────────────────────────────────────────────────────────────────────────
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_TINY: Final[float] = float(np.finfo(np.float64).tiny)
_CAR_CCR_TOL: Final[float] = 1.0e-9          # residuo relativo CAR/CCR
_SYMPLECTIC_TOL: Final[float] = 1.0e-9       # residuo relativo MᵀΩM−Ω
_PSD_REL_FLOOR: Final[float] = 1.0e-12       # λ_min/λ_max mínimo (SPD relativa)
_PSD_ABS_FLOOR: Final[float] = 1.0e-300      # suelo absoluto anti-overflow
_MAX_CONDITION_NUMBER: Final[float] = 1.0e8  # κ(G) máximo
_RICCI_CFL: Final[float] = 0.45              # número de Courant del paso explícito
_RETRACTION_MAX_ITER: Final[int] = 32
_LOG_EXP_CAP: Final[float] = 700.0           # exp(700) < max float64
_KAM_TAU_EXPONENT: Final[float] = 2.0        # τ Diofántico genérico (d grados libertad)
_CHIRIKOV_SOFT: Final[float] = 1.0           # solapamiento ⇒ caos (DEGRADED)
_CHIRIKOV_HARD: Final[float] = 2.0           # solapamiento severo (VETO de coherencia)
_CROWBAR_BUDGET_NS: Final[int] = 400         # presupuesto de FIRMWARE, no de Python
_UNCERTAINTY_FLOOR: Final[float] = 0.5       # ν_min ≥ ½ (ħ=1)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE 1 · SUSTRATO ALGEBRAICO-GEOMÉTRICO                                        ══
# ══ (Ω₃, CAR/CCR, Sp(2n,ℝ), SPD, Connes, Williamson) → S                           ══
# ═════════════════════════════════════════════════════════════════════════════════════
# ── 1.0 · Estrato tipológico: el clasificador de subobjetos Ω₃ ────────────────────────


class HeytingVerdict(IntEnum):
    r"""
    Retículo de Heyting Ω₃ ≅ O(Sierpiński) = {∅ ⊂ {p} ⊂ X}.

    Orden de VERDAD (⊑): VETOED(2) ⊑ DEGRADED(1) ⊑ COHERENT(0). Es decir, el orden
    numérico está INVERTIDO respecto del orden lógico: ⊤ = COHERENT = 0, ⊥ = VETOED = 2.
    Al ser una cadena de Gödel–Dummett de 3 elementos, es un álgebra de Heyting
    (intuicionista) y NO booleana: ¬DEGRADED = ⊥, luego ¬¬DEGRADED = ⊤ ≠ DEGRADED.
    Interpretación topos: Ω del topos de prehaces sobre la cadena (•→•).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @property
    def truth_rank(self) -> int:
        """Rango de verdad creciente: ⊥↦0, medio↦1, ⊤↦2."""
        return 2 - int(self)


def heyting_meet(a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
    r"""Conjunción intuicionista a∧b = mín en orden de verdad = máx numérico."""
    return HeytingVerdict(max(int(a), int(b)))


def heyting_join(a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
    r"""Disyunción a∨b = máx en orden de verdad = mín numérico."""
    return HeytingVerdict(min(int(a), int(b)))


def heyting_implies(a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
    r"""
    Implicación (pseudo-complemento relativo): a→b = ⊤ si a ⊑ b, y b en otro caso.
    En orden numérico invertido: a→b = ⊤ si a ≥ b, else b.  Ex falso: ⊥→b = ⊤.
    """
    return HeytingVerdict.COHERENT if int(a) >= int(b) else HeytingVerdict(b)


def heyting_negation(a: HeytingVerdict) -> HeytingVerdict:
    r"""¬a := a→⊥.  ¬⊥=⊤; ¬medio=⊥; ¬⊤=⊥.  Falla el tercero excluido."""
    return heyting_implies(a, HeytingVerdict.VETOED)


# ── 1.0b · Utilidades numéricas certificado-génicas ──────────────────────────────────


@lru_cache(maxsize=64)
def _darboux_form(n: int) -> NDArray[np.float64]:
    r"""Ω = [[0,I],[−I,0]] ∈ ℝ^{2n×2n}; Ω⁻¹ = Ωᵀ = −Ω. Inmutable (read-only)."""
    eye, zer = np.eye(n), np.zeros((n, n))
    omega = np.block([[zer, eye], [-eye, zer]])
    omega.flags.writeable = False
    return omega


@lru_cache(maxsize=64)
def _particle_hole_swap(n: int) -> NDArray[np.float64]:
    r"""Σ = [[0,I],[I,0]]; la estructura real de Bogoliubov exige Σ W̄ Σ = W."""
    eye, zer = np.eye(n), np.zeros((n, n))
    sigma = np.block([[zer, eye], [eye, zer]])
    sigma.flags.writeable = False
    return sigma


@lru_cache(maxsize=64)
def _bosonic_signature(n: int) -> NDArray[np.float64]:
    r"""J = diag(I_n, −I_n): métrica de U(n,n) para CCR."""
    j = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))
    j.flags.writeable = False
    return j


def _as_matrix(name: str, arr: NDArray, n: Optional[int], complex_ok: bool) -> NDArray:
    """Validación estructural dura: cuadrada, finita, dtype canónico, no vacía."""
    a = np.asarray(arr, dtype=np.complex128 if complex_ok else np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError(f"[{name}] se requiere matriz cuadrada no vacía, recibido {a.shape}")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"[{name}] dimensión {a.shape[0]} incompatible con n={n}")
    if not np.all(np.isfinite(a)):
        raise TopologicalInvariantError(f"[{name}] entradas no finitas (NaN/Inf): sustrato roto")
    return a


def _fro(a: NDArray) -> float:
    """Norma de Frobenius como float nativo (evita np.float64 en certificados)."""
    return float(la.norm(a, ord="fro", check_finite=False))


def _safe_exp(x: float) -> float:
    """exp saturada (evita overflow silencioso en cotas exponenciales)."""
    return float(math.exp(min(x, _LOG_EXP_CAP)))


def _relative_floor(eigs: NDArray[np.float64]) -> float:
    """Suelo espectral RELATIVO: max(abs_floor, rel_floor·λ_max). Invariante de escala."""
    lam_max = float(np.max(eigs)) if eigs.size else 0.0
    return max(_PSD_ABS_FLOOR, _PSD_REL_FLOOR * max(lam_max, 0.0))


def _spd_project(a: NDArray[np.float64]) -> Tuple[NDArray[np.float64], int]:
    r"""
    Proyección de Higham al cono SPD: A ↦ Q·max(Λ,ε)·Qᵀ con A_sym=(A+Aᵀ)/2.
    Es la proyección óptima en Frobenius sobre {S ⪰ εI} (teorema de Higham, 1988).
    Devuelve (A_spd, nº de autovalores recortados).
    """
    sym = 0.5 * (a + a.T)
    w, q = la.eigh(sym, check_finite=False)
    floor = _relative_floor(w)
    clipped = int(np.count_nonzero(w < floor))
    return (q * np.maximum(w, floor)) @ q.T, clipped


def _canonical_jacobian_defect(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    h: float = 1.0e-6,
) -> float:
    r"""
    Defecto de canonicidad de una carta: ‖DfᵀΩDf − Ω‖_F/‖Ω‖_F por diferencias
    centrales. Para 1 grado de libertad equivale a |det Df − 1| (simplecticidad).
    Rigor: verificación numérica, no simbólica; h óptimo ~ eps^{1/3}.
    """
    d = int(x.size)
    jac = np.empty((d, d), dtype=np.float64)
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        jac[:, j] = (f(x + e) - f(x - e)) / (2.0 * h)
    om = _darboux_form(d // 2)
    return _fro(jac.T @ om @ jac - om) / _fro(om)


# ── 1.0c · Tipos de payload (contratos formales entre fases) ─────────────────────────


@dataclass(frozen=True, slots=True)
class CanonicalAlgebraReport:
    r"""Auditoría CAR/CCR completa (bloques derecha AND izquierda), adimensional."""

    statistics: str            # "CAR" | "CCR"
    residual_relative: float   # max_bloque ‖R‖_F / (√n · max(1,‖U‖₂²+‖V‖₂²))
    residual_absolute: float
    tolerance: float
    min_normal_singular_value: float   # σ_min(U): cos θ (CAR) / cosh r (CCR)
    max_anomalous_singular_value: float
    bosonic_admissibility_defect: float  # max(0, 1−σ_min(U)) sólo CCR
    is_canonical: bool


@dataclass(frozen=True, slots=True)
class SymplecticAudit:
    r"""Liouville riguroso: residuo MᵀΩM−Ω (NO sólo det), con log-det estable."""

    residual_relative: float
    residual_absolute: float
    volume_drift: float        # |det M − 1| vía expm1(log|det|)
    log_volume_drift: float    # |log|det M|| (estable para n grande)
    determinant_sign: float
    is_symplectic: bool


@dataclass(frozen=True, slots=True)
class MetricGeometryReport:
    r"""Geometría SPD + cota operacional Connes–DK y su versión adimensional."""

    lambda_min: float
    lambda_max: float
    condition_number: float
    symmetry_defect: float
    lipschitz_connes_bound: float          # A4 (convención, grado −3/2)
    lipschitz_dimensionless: float         # κ^{3/2}/2 (invariante de escala)
    connes_seminorm_bound: float           # 2‖D‖ con ‖D‖ ≤ √λ_max (riguroso)
    is_spd: bool
    is_well_conditioned: bool


@dataclass(frozen=True, slots=True)
class QuasiparticleSpectrum:
    r"""
    Espectro de cuasipartículas y espectro de squeezing GAUGE-INVariante.

    `squeezing_tau` son los valores de Takagi de C = U⁻¹V (simétrica en CCR,
    antisimétrica en CAR): τ = tanh r (bosones), τ = tan θ (fermiones). Son
    invariantes bajo transformaciones unitarias de base (Bloch–Messiah).
    """

    energies: Tuple[float, ...]        # vacío si no se suministró H_BdG / energías
    gap: float                         # ΔE_min verdadera si is_proxy=False
    is_proxy: bool                     # True ⇒ `gap` es brecha de coherencia, NO energía
    squeezing_tau: Tuple[float, ...]
    squeezing_r: Tuple[float, ...]
    pair_symmetry_defect: float        # ‖C ∓ Cᵀ‖/‖C‖
    rank_deficient_U: bool
    dynamical_instability: bool        # autovalores complejos de J·H (bosones)
    bdg_structure_defect: float
    williamson_min: Optional[float]    # ν_min de σ (si se suministró covarianza)
    uncertainty_violation: bool        # ν_min < ½ ⇒ estado no físico
    contraction_violation: bool        # bosones: ‖U⁻¹V‖ ≥ 1 ⇒ vacío no normalizable


@dataclass(frozen=True, slots=True)
class GeometricSubstrate:
    r"""
    S — PAYLOAD TERMINAL DE LA FASE 1 y ARGUMENTO FORMAL INICIAL DE LA FASE 2.

    Es el "objeto de continuidad": todo lo que la Fase 2 necesita está aquí, y nada
    de lo que la Fase 2 debe suponer queda implícito.
    """

    n: int
    is_fermionic: bool
    U: NDArray[np.complex128]
    V: NDArray[np.complex128]
    W: NDArray[np.complex128]
    M: NDArray[np.float64]
    G: NDArray[np.float64]
    canonical: CanonicalAlgebraReport
    symplectic: SymplecticAudit
    metric: MetricGeometryReport
    spectrum: QuasiparticleSpectrum
    verdict: HeytingVerdict
    coverage: float                 # fracción de módulos efectivamente certificados
    created_ns: int


@dataclass(frozen=True, slots=True)
class RicciFlowState:
    """R₁: estado del flujo de Ricci normalizado (+DeTurck) con volumen preservado."""

    G_next: NDArray[np.float64]
    ricci_used: NDArray[np.float64]
    is_surrogate: bool
    scalar_curvature: float
    einstein_defect_before: float
    einstein_defect_after: float
    volume_ratio: float             # det G_next / det G_before (debe ser 1)
    delta_tau_requested: float
    delta_tau_used: float           # limitado por CFL espectral
    clipped_eigenvalues: int
    monotone: bool
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class CanonicalRetraction:
    """R₂: retracción sobre la variedad de grupo (repara, no sólo veta)."""

    U: NDArray[np.complex128]
    V: NDArray[np.complex128]
    M: NDArray[np.float64]
    residual_before: float
    residual_after: float
    symplectic_before: float
    symplectic_after: float
    iterations: int
    converged: bool
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class DelaunayChart:
    """Carta de Delaunay (L,G,H;l,g,h) y carta regular de Poincaré (Λ,λ;ξ,η)."""

    actions: Tuple[float, ...]
    angles: Tuple[float, ...]
    poincare_xi_eta: Tuple[float, ...]
    chart_canonical_defect: float
    ordering_valid: bool            # L ≥ G ≥ |H| ≥ 0


@dataclass(frozen=True, slots=True)
class FloquetKreinReport:
    """Monodromía de la sección de Poincaré: Floquet, Krein, residuos de Greene."""

    multipliers_abs: Tuple[float, ...]
    lyapunov_exponents: Tuple[float, ...]
    lyapunov_sum: float             # ≈ 0 ⇔ Liouville
    reciprocal_pairing_defect: float
    krein_signatures: Tuple[int, ...]
    greene_residues: Tuple[float, ...]
    eigenvector_condition: float
    krein_collision: bool
    jordan_defect_suspected: bool
    spectrally_stable: bool
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class ResonanceCensus:
    """Censo de resonancias de movimiento medio + solapamiento de Chirikov + KAM."""

    resonances: Tuple[Tuple[int, int, int, float], ...]  # (i,j,orden,|n_i/n_j−p/q|)
    chirikov_max: float
    diophantine_gamma_eff: float
    diophantine_tau: float
    kam_satisfied: bool
    golden_nobility: float          # 1/(1+|ratio−φ|)heurística de robustez de MacKay
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class MelnikovReport:
    """Melnikov del péndulo perturbado (Guckenheimer–Holmes §4.5)."""

    threshold_ratio: float          # (4/π)cosh(πω/2)
    forcing_ratio: float            # γ/δ
    has_simple_zeros: bool          # ⇒ herradura de Smale (A9)
    separatrix_splitting: float     # amplitud 2πγ sech(πω/2) − 8δ
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class PoincareCelestialReport:
    """Fase 2 · bloque de mecánica celeste de Poincaré (auditoría de coherencia orbital)."""

    delaunay: Optional[DelaunayChart]
    secular_frequencies: Tuple[float, ...]
    secular_gap: float              # min |g_i − g_j| (repulsión de niveles secular)
    secular_symmetry_defect: float
    floquet: Optional[FloquetKreinReport]
    resonances: Optional[ResonanceCensus]
    melnikov: Optional[MelnikovReport]
    kac_recurrence_time: float      # ⟨τ⟩ = T·μ(Ω)/μ(celda) (lema de Kac + Liouville)
    nekhoroshev_time: float         # escala heurística exp((γ/ε)^{1/2d})
    verdict: HeytingVerdict
    coverage: float


@dataclass(frozen=True, slots=True)
class BrockettShahshahaniState:
    """R₃: replicador exponencial (Shahshahani) + doble corchete de Brockett (Cayley)."""

    p_next: NDArray[np.float64]
    rho_next: NDArray[np.complex128]
    kernel_used: NDArray[np.float64]
    dH_dt: float                    # −Var_p(Kp) ≤ 0 (continuo, exacto)
    fitness_gain: float             # ½p'ᵀKp' − ½pᵀKp ≥ 0 (discreto, certificado)
    alignment_gain: float           # tr(ρ'N) − tr(ρN) ≥ 0
    alignment_rate: float           # ‖[ρ,N]‖_F²
    isospectral_defect: float
    trace_defect: float
    positivity_defect: float
    entropy_drift: float            # ΔS_vN ≈ 0 (invariante unitario)
    kernel_asymmetry: float
    dt_requested: float
    dt_used: float
    line_search_steps: int
    monotone: bool
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class DynamicalFrame:
    """F — PAYLOAD TERMINAL DE LA FASE 2 y ARGUMENTO FORMAL INICIAL DE LA FASE 3."""

    substrate: GeometricSubstrate
    ricci: Optional[RicciFlowState]
    retraction: Optional[CanonicalRetraction]
    poincare: Optional[PoincareCelestialReport]
    brockett: Optional[BrockettShahshahaniState]
    verdict: HeytingVerdict
    coverage: float


@dataclass(frozen=True, slots=True)
class RSI3Hyperparameters:
    """Payload MUTABLE de RSI³ (único objeto que la auto-mejora puede reescribir)."""

    delta_tau: float = 0.01
    dt: float = 0.01
    trust_radius: float = 0.25
    lyapunov_weights: Tuple[float, ...] = (0.25, 0.20, 0.20, 0.20, 0.15)
    meta_depth: int = 0


@dataclass(frozen=True, slots=True)
class RSI3MetaReport:
    """Trazabilidad de la auto-mejora de nivel 3 (aceptaciones, rollback, contracción)."""

    iterations: int
    accepted: int
    rejected: int
    lyapunov_initial: float
    lyapunov_final: float
    lyapunov_history: Tuple[float, ...]
    contraction_q: float
    banach_error_bound: float
    contraction_certified: bool
    monotone: bool
    hyperparameters: RSI3Hyperparameters
    verdict: HeytingVerdict


@dataclass(frozen=True, slots=True)
class CrowbarDecision:
    """Latch de protección. La garantía de 400 ns es de FIRMWARE, no del intérprete."""

    latch: bool
    reason: str
    firmware_word: int
    python_latency_ns: int
    budget_ns: int
    host_realtime_guaranteed: bool = False


@dataclass(frozen=True, slots=True)
class BogoliubovGovernanceCertificate:
    r"""
    Certificado inmutable (sólo escalares/tuplas: inmutabilidad real, no nominal).
    Los seis primeros campos preservan la ABI de la v4.0.0.
    """

    car_ccr_residual: float
    min_quasiparticle_gap: float
    lipschitz_connes_bound: float
    symplectic_volume_drift: float
    heyting_verdict_value: int
    is_quasiparticle_coherent: bool
    # ── extensiones rigurosas v5 ──
    gap_is_proxy: bool = True
    symplectic_residual: float = 0.0
    condition_number: float = 1.0
    contraction_q: float = 1.0
    banach_error_bound: float = float("inf")
    lyapunov_initial: float = float("nan")
    lyapunov_final: float = float("nan")
    chirikov_max: float = 0.0
    krein_collision: bool = False
    melnikov_chaos: bool = False
    coverage: float = 0.0
    iterations: int = 0
    crowbar: Optional[CrowbarDecision] = None
    schema_version: str = "5.0.0"

    def __post_init__(self) -> None:
        if self.heyting_verdict_value not in (0, 1, 2):
            raise ValueError("veredicto fuera de Ω₃")
        for name in ("car_ccr_residual", "symplectic_residual", "symplectic_volume_drift"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} no puede ser negativo")

    @property
    def verdict(self) -> HeytingVerdict:
        return HeytingVerdict(self.heyting_verdict_value)

    def implies(self, other: "BogoliubovGovernanceCertificate") -> HeytingVerdict:
        """Implicación intuicionista entre certificados (funtorialidad laxa)."""
        return heyting_implies(self.verdict, other.verdict)


@dataclass(frozen=True, slots=True)
class OmegaCycleOutcome:
    """Traza completa del ciclo anidado Φ₃∘Φ₂∘Φ₁ (auditable, reproducible)."""

    substrate: GeometricSubstrate
    frame: DynamicalFrame
    meta: RSI3MetaReport
    certificate: BogoliubovGovernanceCertificate


# ─────────────────────────────────────────────────────────────────────────────────────
class BogoliubovAgent(Morphism):
    r"""
    Soberano de Calibre de Diagonalización de Cuasipartículas (V_Ω / V_WISDOM).

    Los métodos están ordenados por FASES ANIDADAS. La lectura secuencial del fichero
    ES el flujo de datos: 1.1…1.8 → S; 2.1…2.6 (S → F); 3.1…3.7 (F → C).
    """

    # ═════════════════════════════════════════════════════════════════════════════════
    # ── 1.1 · OBSERVE: ensamblaje de la matriz de Bogoliubov en el espacio de Fock ──
    # ═════════════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _assemble_bogoliubov_matrix(
        U: NDArray[np.complex128], V: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        r"""
        W = [[U, V], [V̄, Ū]] ∈ ℂ^{2n×2n} con estructura real Σ W̄ Σ = W.

        Categorialmente: W es el morfismo de ℱ(ℋ) inducido por el par (U,V); la
        condición Σ W̄ Σ = W es la compatibilidad con la involución antilineal de
        conjugación de carga (partícula-hueco). Se verifica explícitamente.
        """
        n = U.shape[0]
        W = np.block([[U, V], [V.conj(), U.conj()]])
        sigma = _particle_hole_swap(n)
        defect = _fro(sigma @ W.conj() @ sigma - W) / max(_fro(W), _TINY)
        if defect > 64.0 * _MACHINE_EPS * max(1, n):
            raise TopologicalInvariantError(
                f"[BOGOLIUBOV_AGENT_VETO] Estructura partícula-hueco violada: {defect:.3e}"
            )
        return W

    # ── 1.2 · Relaciones canónicas COMPLETAS, adimensionales ────────────────────────
    @staticmethod
    def _canonical_residuals(
        U: NDArray[np.complex128],
        V: NDArray[np.complex128],
        is_fermionic: bool,
    ) -> CanonicalAlgebraReport:
        r"""
        s = +1 (CAR) / −1 (CCR). Se auditan los cuatro bloques:

          derecha: UU† + s VV† − I ,  UVᵀ + s VUᵀ
          izquierda: U†U + s VᵀV̄ − I ,  U†V + s VᵀŪ

        (En dimensión finita derecha ⇔ izquierda; auditar ambas detecta pérdida de
        precisión antes de que se propague.) Residual relativo:

          ρ = max_b ‖R_b‖_F / (√n · max(1, ‖U‖₂² + ‖V‖₂²))

        invariante bajo reescalado global y bajo cambio unitario de base.
        Test necesario adicional (CCR): σ_min(U) ≥ 1 (pues UU† = I + VV† ⪰ I).
        """
        n = U.shape[0]
        s = 1.0 if is_fermionic else -1.0
        eye = np.eye(n, dtype=np.complex128)
        Ud, Vt, Vc = U.conj().T, V.T, V.conj()
        blocks = (
            U @ Ud + s * (V @ V.conj().T) - eye,
            U @ Vt + s * (V @ U.T),
            Ud @ U + s * (Vt @ Vc) - eye,
            Ud @ V + s * (Vt @ U.conj()),
        )
        abs_res = max(_fro(b) for b in blocks)
        su = la.svdvals(U, check_finite=False)
        sv = la.svdvals(V, check_finite=False)
        scale = max(1.0, float(su[0]) ** 2 + float(sv[0]) ** 2)
        rel = abs_res / (math.sqrt(n) * scale)
        tol = max(_CAR_CCR_TOL, 16.0 * n * _MACHINE_EPS)
        bos_def = 0.0 if is_fermionic else max(0.0, 1.0 - float(su[-1]))
        ok = (rel <= tol) and (bos_def <= tol)
        return CanonicalAlgebraReport(
            statistics="CAR" if is_fermionic else "CCR",
            residual_relative=rel,
            residual_absolute=abs_res,
            tolerance=tol,
            min_normal_singular_value=float(su[-1]),
            max_anomalous_singular_value=float(sv[0]),
            bosonic_admissibility_defect=bos_def,
            is_canonical=ok,
        )

    # ── 1.3 · Invarianza simpléctica de Liouville (corrección del bug det≈1) ────────
    @staticmethod
    def _symplectic_audit(M: NDArray[np.float64], n: int) -> SymplecticAudit:
        r"""
        M ∈ Sp(2n,ℝ) ⇔ MᵀΩM = Ω.  Sólo entonces det M = +1 (A3).
        Se reporta el residuo relativo ‖MᵀΩM−Ω‖_F/√(2n) y el drift de volumen vía
        slogdet + expm1 (estable para 2n grande; `la.det` desbordaría).
        """
        om = _darboux_form(n)
        res_abs = _fro(M.T @ om @ M - om)
        res_rel = res_abs / math.sqrt(2.0 * n)
        sign, logdet = la.slogdet(M)
        drift = abs(math.expm1(float(logdet))) if sign > 0 else float("inf")
        return SymplecticAudit(
            residual_relative=res_rel,
            residual_absolute=res_abs,
            volume_drift=drift,
            log_volume_drift=abs(float(logdet)),
            determinant_sign=float(sign),
            is_symplectic=(res_rel <= _SYMPLECTIC_TOL and sign > 0),
        )

    # ── 1.4 · Geometría SPD y cota de Lipschitz (Connes) con honestidad dimensional ─
    @staticmethod
    def _metric_geometry(G: NDArray[np.float64]) -> MetricGeometryReport:
        r"""
        SPD relativa: λ_min > 1e-12·λ_max (invariante de escala; el umbral absoluto
        de la v4 vetaba métricas legítimas en unidades pequeñas).
        A4: L = 1/(2λ_min^{3/2}) calculada en espacio logarítmico. Se añade
        L·λ_max^{3/2} = κ^{3/2}/2 (adimensional) y la cota RIGUROSA del seminorma
        de Connes L(a)=‖[D,a]‖ ≤ 2‖D‖‖a‖ con ‖D‖ ≤ √λ_max (D tipo Dirac asociado).
        """
        n = G.shape[0]
        sym_def = _fro(G - G.T) / max(_fro(G), _TINY)
        g_sym = 0.5 * (G + G.T)
        w = la.eigvalsh(g_sym, check_finite=False)
        l_min, l_max = float(w[0]), float(w[-1])
        is_spd = l_min > max(_PSD_ABS_FLOOR, _PSD_REL_FLOOR * max(l_max, 0.0))
        cond = l_max / max(l_min, _TINY)
        log_l = -math.log(2.0) - 1.5 * math.log(max(l_min, _TINY))
        return MetricGeometryReport(
            lambda_min=l_min,
            lambda_max=l_max,
            condition_number=cond,
            symmetry_defect=sym_def,
            lipschitz_connes_bound=_safe_exp(log_l),
            lipschitz_dimensionless=0.5 * cond ** 1.5 if cond < 1e100 else float("inf"),
            connes_seminorm_bound=2.0 * math.sqrt(max(l_max, 0.0)),
            is_spd=is_spd,
            is_well_conditioned=(cond <= _MAX_CONDITION_NUMBER),
        )

    # ── 1.5 · Espectro de squeezing gauge-invariante (Autonne–Takagi / Youla) ───────
    @staticmethod
    def _squeezing_spectrum(
        U: NDArray[np.complex128], V: NDArray[np.complex128], is_fermionic: bool
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, bool, bool]:
        r"""
        C = U⁻¹V. De UVᵀ ± VUᵀ = 0 se sigue Cᵀ = ±C:
          bosones  → C simétrica  → Takagi:  C = Z diag(τ) Zᵀ, τ_k = tanh r_k ∈ [0,1)
          fermiones→ C antisimétrica → Youla: τ_k = tan θ_k ∈ [0,1] (pares degenerados)
        Los valores de Takagi/Youla coinciden con los valores singulares de C, luego
        τ = svdvals(C) es un invariante de gauge (no requiere construir Z).
        Bosones: ‖C‖ ≥ 1 ⇔ vacío no normalizable ⇒ VETO físico.
        """
        su = la.svdvals(U, check_finite=False)
        rank_tol = max(float(su[0]), _TINY) * 1e-13
        deficient = bool(su[-1] <= rank_tol)
        C = la.pinv(U) @ V if deficient else la.solve(U, V, check_finite=False)
        sign = -1.0 if is_fermionic else 1.0
        sym_def = _fro(C - sign * C.T) / max(_fro(C), _TINY)
        tau = np.clip(la.svdvals(C, check_finite=False), 0.0, None)
        if is_fermionic:
            r = np.arctan(np.minimum(tau, 1e12))
            violation = False
        else:
            tau_c = np.minimum(tau, 1.0 - 1e-15)
            r = np.arctanh(tau_c)
            violation = bool(tau.size and float(tau[0]) >= 1.0 - 1e-12)
            tau = tau_c
        return tau, r, sym_def, deficient, violation

    # ── 1.6 · Brecha de cuasipartículas: energía REAL vs proxy declarado ────────────
    def _quasiparticle_spectrum(
        self,
        U: NDArray[np.complex128],
        V: NDArray[np.complex128],
        is_fermionic: bool,
        bdg_hamiltonian: Optional[NDArray] = None,
        quasiparticle_energies: Optional[Sequence[float]] = None,
        covariance_sigma: Optional[NDArray] = None,
    ) -> QuasiparticleSpectrum:
        r"""
        Corrección del error conceptual de la v4 (σ_min(U) NO es una energía).

        (a) Si se da `quasiparticle_energies` o `bdg_hamiltonian`, ΔE_min = min E>0:
            fermiones: H_BdG = [[h,Δ],[Δ†,−hᵀ]], Δᵀ=−Δ; E = eigvalsh(H) (±E por PH).
            bosones:   H_BdG = [[h,Δ],[Δ†, hᵀ]], Δᵀ=+Δ; D = J·H, E = Re eig(D);
                       eig complejos ⇒ inestabilidad dinámica (Landau) ⇒ VETO.
        (b) Si no, se reporta un PROXY adimensional explícito:
            bosones: min_k sech²r_k = min_k(1−τ_k²)  (distancia al vacío no normalizable)
            fermiones: min_k |cos 2θ_k| = min|(1−τ²)/(1+τ²)| (distancia al punto u=v)
            y `is_proxy=True`. Nunca se disfraza de energía.
        (c) Williamson: ν_k = |eig(Ω⁻¹σ)| (pares); físico ⇔ ν_min ≥ ½ (ħ=1).
        """
        n = U.shape[0]
        tau, r, sym_def, deficient, contraction_bad = self._squeezing_spectrum(U, V, is_fermionic)

        energies: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        instability, bdg_defect = False, 0.0
        if quasiparticle_energies is not None:
            energies = np.abs(np.asarray(quasiparticle_energies, dtype=np.float64).ravel())
        elif bdg_hamiltonian is not None:
            H = _as_matrix("H_BdG", bdg_hamiltonian, 2 * n, True)
            h, delta = H[:n, :n], H[:n, n:]
            lower, right = H[n:, :n], H[n:, n:]
            sgn = -1.0 if is_fermionic else 1.0
            bdg_defect = (
                _fro(h - h.conj().T)
                + _fro(delta - sgn * delta.T)
                + _fro(lower - delta.conj().T)
                + _fro(right - sgn * h.T)
            ) / max(_fro(H), _TINY)
            Hs = 0.5 * (H + H.conj().T)
            if is_fermionic:
                ev = la.eigvalsh(Hs, check_finite=False)
                energies = np.sort(np.abs(ev))
            else:
                ev = la.eigvals(_bosonic_signature(n) @ Hs, check_finite=False)
                scale = max(_fro(Hs), _TINY)
                instability = bool(np.max(np.abs(ev.imag)) > 1e-9 * scale)
                energies = np.sort(np.abs(ev.real))

        if energies.size:
            positive = energies[energies > 1e-14]
            gap = float(positive[0]) if positive.size else 0.0
            is_proxy = False
        else:
            if is_fermionic:
                gap = float(np.min(np.abs((1.0 - tau**2) / (1.0 + tau**2)))) if tau.size else 1.0
            else:
                gap = float(np.min(1.0 - tau**2)) if tau.size else 1.0
            is_proxy = True

        nu_min: Optional[float] = None
        uncertainty_bad = False
        if covariance_sigma is not None:
            sigma = _as_matrix("covariance_sigma", covariance_sigma, 2 * n, False)
            sigma = 0.5 * (sigma + sigma.T)
            om_inv = -_darboux_form(n)
            ev = la.eigvals(om_inv @ sigma, check_finite=False)
            nu = np.sort(np.abs(ev))[::2]  # pares degenerados de Williamson
            nu_min = float(nu[0]) if nu.size else None
            uncertainty_bad = bool(nu_min is not None and nu_min < _UNCERTAINTY_FLOOR - 1e-9)

        return QuasiparticleSpectrum(
            energies=tuple(float(x) for x in energies),
            gap=gap,
            is_proxy=is_proxy,
            squeezing_tau=tuple(float(x) for x in tau),
            squeezing_r=tuple(float(x) for x in r),
            pair_symmetry_defect=float(sym_def),
            rank_deficient_U=deficient,
            dynamical_instability=instability,
            bdg_structure_defect=float(bdg_defect),
            williamson_min=nu_min,
            uncertainty_violation=uncertainty_bad,
            contraction_violation=contraction_bad,
        )

    # ── 1.7 · DECIDE: agregación en Ω₃ (meet intuicionista, no booleano) ────────────
    @staticmethod
    def _heyting_aggregate(
        verdicts: Sequence[HeytingVerdict], unverified: Sequence[bool] = ()
    ) -> HeytingVerdict:
        r"""
        Veredicto global = ∧ de los veredictos locales. Los módulos NO evaluados
        contribuyen DEGRADED (⊤ exige un constructor/proof: A11), nunca COHERENT.
        """
        acc = HeytingVerdict.COHERENT
        for v in verdicts:
            acc = heyting_meet(acc, v)
        for flag in unverified:
            if flag:
                acc = heyting_meet(acc, HeytingVerdict.DEGRADED)
        return acc

    # ── 1.8 · MÉTODO TERMINAL DE LA FASE 1 ─────────────────────────────────────────
    #          Definición formal final:  continuation : S ──► R
    #          Su cola `return continuation(S)` ES el inicio de la Fase 2 (§2.1).
    # ═════════════════════════════════════════════════════════════════════════════════
    def certify_geometric_substrate(
        self,
        U_matrix: NDArray[np.complex128],
        V_matrix: NDArray[np.complex128],
        jacobian_M: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        is_fermionic: bool = True,
        *,
        bdg_hamiltonian: Optional[NDArray] = None,
        quasiparticle_energies: Optional[Sequence[float]] = None,
        covariance_sigma: Optional[NDArray] = None,
        strict: bool = False,
        unverified_verdict: HeytingVerdict = HeytingVerdict.DEGRADED,
        continuation: Optional[Callable[[GeometricSubstrate], R]] = None,
    ) -> Union[GeometricSubstrate, R]:
        r"""
        Φ₁ — Auditoría OODA (Observe/Orient/Decide) del sustrato y emisión de S.

        `strict=True` reproduce la semántica legacy (veto DURO por excepción).
        `strict=False` emite ⊥ en Ω₃ (veto SUAVE) para que R₂ (§2.2) pueda REPARAR:
        sin veto suave no hay auto-mejora posible, sólo rechazo.

        CONTINUACIÓN (anidamiento): si `continuation` no es None, la última sentencia
        es `return continuation(S)`, cuya implementación canónica es
        `self.evolve_dynamical_frame` (Fase 2), que a su vez continúa en
        `self.close_rsi3_loop` (Fase 3).
        """
        U = _as_matrix("U_matrix", U_matrix, None, True)
        n = U.shape[0]
        V = _as_matrix("V_matrix", V_matrix, n, True)
        M = _as_matrix("jacobian_M", jacobian_M, 2 * n, False)
        G = _as_matrix("metric_G", metric_G, n, False)

        W = self._assemble_bogoliubov_matrix(U, V)
        canonical = self._canonical_residuals(U, V, is_fermionic)
        symplectic = self._symplectic_audit(M, n)
        metric = self._metric_geometry(G)
        spectrum = self._quasiparticle_spectrum(
            U, V, is_fermionic, bdg_hamiltonian, quasiparticle_energies, covariance_sigma
        )

        hard_breach = (
            not canonical.is_canonical
            or not metric.is_spd
            or not metric.is_well_conditioned
            or spectrum.dynamical_instability
            or spectrum.uncertainty_violation
            or spectrum.contraction_violation
        )
        soft_breach = (
            not symplectic.is_symplectic
            or spectrum.rank_deficient_U
            or spectrum.pair_symmetry_defect > 1e-8
            or spectrum.bdg_structure_defect > 1e-8
        )
        local = [
            HeytingVerdict.COHERENT if canonical.is_canonical else HeytingVerdict.VETOED,
            HeytingVerdict.COHERENT if symplectic.is_symplectic else HeytingVerdict.DEGRADED,
            HeytingVerdict.COHERENT if metric.is_spd else HeytingVerdict.VETOED,
            HeytingVerdict.COHERENT if metric.is_well_conditioned else HeytingVerdict.DEGRADED,
            HeytingVerdict.VETOED if hard_breach else HeytingVerdict.COHERENT,
            HeytingVerdict.DEGRADED if soft_breach else HeytingVerdict.COHERENT,
        ]
        coverage = 0.5 + 0.25 * (not spectrum.is_proxy) + 0.25 * (covariance_sigma is not None)
        verdict = self._heyting_aggregate(
            local, unverified=(spectrum.is_proxy and unverified_verdict is HeytingVerdict.DEGRADED,)
        )

        if strict and hard_breach:
            raise TopologicalInvariantError(
                "[BOGOLIUBOV_AGENT_VETO] Invariante duro violado: "
                f"{canonical.statistics} ρ={canonical.residual_relative:.3e}, "
                f"λ_min(G)={metric.lambda_min:.3e}, κ(G)={metric.condition_number:.3e}, "
                f"inestabilidad={spectrum.dynamical_instability}, "
                f"incertidumbre={spectrum.uncertainty_violation}"
            )
        if verdict is not HeytingVerdict.COHERENT:
            logger.warning(
                "[FASE1] Sustrato %s: CAR/CCR=%.3e Sp=%.3e κ=%.3e gap=%.4f(proxy=%s)",
                verdict.name, canonical.residual_relative, symplectic.residual_relative,
                metric.condition_number, spectrum.gap, spectrum.is_proxy,
            )

        substrate = GeometricSubstrate(
            n=n, is_fermionic=is_fermionic, U=U, V=V, W=W, M=M, G=G,
            canonical=canonical, symplectic=symplectic, metric=metric,
            spectrum=spectrum, verdict=verdict, coverage=float(coverage),
            created_ns=time.perf_counter_ns(),
        )
        # ── CONTINUACIÓN → FASE 2 (§2.1 evolve_ricci_metric_R1 / §2.6 evolve_dynamical_frame)
        if continuation is None:
            return substrate
        return continuation(substrate)

    # ═════════════════════════════════════════════════════════════════════════════════
    # ══ FASE 2 · FLUJOS: R₁ RICCI · R₂ RETRACCIÓN · POINCARÉ · R₃ BROCKETT  (S → F) ══
    # ═════════════════════════════════════════════════════════════════════════════════
    # ── 2.1 · R₁: flujo de Ricci NORMALIZADO con gauge de DeTurck ───────────────────
    @staticmethod
    def _ricci_surrogate(V: NDArray[np.complex128]) -> NDArray[np.float64]:
        r"""
        Subrogado algebraico de Ricci (declarado como MODELO, no como tensor real):
            Ric_sur = Re(V†V) ⪰ 0  (densidad de pares anómalos / esfuerzo de apareamiento).
        Es real simétrica PSD por hermiticidad de V†V. Sólo se usa si el llamante no
        suministra Ric_μν; el flujo relaja G hacia la condición de Einstein de ese
        esfuerzo. Honestidad: no es la curvatura de una variedad, es un cierre.
        """
        return np.real(V.conj().T @ V).astype(np.float64)

    def _ricci_normalized_step(
        self,
        G: NDArray[np.float64],
        ricci: NDArray[np.float64],
        delta_tau: float,
        *,
        preserve_volume: bool = True,
        deturck_generator: Optional[NDArray[np.float64]] = None,
    ) -> RicciFlowState:
        r"""
        R₁ CORREGIDO (bug de doble /n en v4):

            ∂_τ G = −2 ( Ric − (R/n) G ) + L_X G ,   R = tr(G⁻¹ Ric)

        El término (R/n)G hace el flujo NORMALIZADO (preserva el volumen total en el
        continuo); aquí se impone además exacto en el discreto por reescalado conforme
        G ← G·(det G₀/det G₁)^{1/n}. Puntos fijos: métricas de Einstein (Ric = (R/n)G).

        DeTurck: en el modelo algebraico de coeficientes constantes, con X = ∂v,
        (L_X G)_{μν} = XᵀG + GX (linealización exacta del arrastre de Lie).

        Paso CFL: el generador lineal es A = G^{-1}Ric − (R/n)I, cuyo espectro se
        obtiene del matriz SIMÉTRICA similar G^{-1/2}Ric G^{-1/2} − (R/n)I; se impone
        Δτ ≤ CFL/(2·max|spec A|) (estabilidad de Euler explícito).

        Lyapunov: defecto de Einstein ε = ‖Ric − (R/n)G‖_F/‖G‖_F (scale-free).
        R y Ric se congelan en el paso (coeficientes congelados): la monotonía se
        VERIFICA a posteriori, no se presupone.
        """
        n = G.shape[0]
        g_sym = 0.5 * (G + G.T)
        ric_sym = 0.5 * (ricci + ricci.T)
        cf = la.cho_factor(g_sym, lower=True, check_finite=False)
        r_scalar = float(np.trace(la.cho_solve(cf, ric_sym, check_finite=False)))

        w, q = la.eigh(g_sym, check_finite=False)
        w = np.maximum(w, _relative_floor(w))
        g_inv_half = (q / np.sqrt(w)) @ q.T
        a_sym = g_inv_half @ ric_sym @ g_inv_half
        a_sym = 0.5 * (a_sym + a_sym.T)
        spec = la.eigvalsh(a_sym, check_finite=False) - r_scalar / n
        gen_norm = float(np.max(np.abs(spec))) if spec.size else 0.0
        dt_cap = _RICCI_CFL / (2.0 * max(gen_norm, 1e-300))
        dt_eff = float(min(max(delta_tau, 0.0), dt_cap))

        g_norm = max(_fro(g_sym), _TINY)
        drift = ric_sym - (r_scalar / n) * g_sym
        defect_before = _fro(drift) / g_norm
        if deturck_generator is not None:
            X = np.asarray(deturck_generator, dtype=np.float64)
            drift = drift - 0.5 * (X.T @ g_sym + g_sym @ X)

        g_next = g_sym - 2.0 * dt_eff * drift
        sign0, ld0 = la.slogdet(g_sym)
        sign1, ld1 = la.slogdet(g_next)
        if preserve_volume and sign0 > 0 and sign1 > 0:
            g_next = g_next * _safe_exp((float(ld0) - float(ld1)) / n)
        g_next, clipped = _spd_project(g_next)

        cf2 = la.cho_factor(g_next, lower=True, check_finite=False)
        r_next = float(np.trace(la.cho_solve(cf2, ric_sym, check_finite=False)))
        defect_after = _fro(ric_sym - (r_next / n) * g_next) / max(_fro(g_next), _TINY)
        s2, ld2 = la.slogdet(g_next)
        vol_ratio = _safe_exp(float(ld2) - float(ld0)) if (s2 > 0 and sign0 > 0) else float("inf")
        monotone = defect_after <= defect_before + 1e-12

        verdict = HeytingVerdict.COHERENT
        if clipped > 0 or not monotone:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if not np.all(np.isfinite(g_next)):
            verdict = HeytingVerdict.VETOED

        return RicciFlowState(
            G_next=g_next, ricci_used=ric_sym, is_surrogate=False,
            scalar_curvature=r_scalar,
            einstein_defect_before=float(defect_before),
            einstein_defect_after=float(defect_after),
            volume_ratio=float(vol_ratio),
            delta_tau_requested=float(delta_tau), delta_tau_used=dt_eff,
            clipped_eigenvalues=clipped, monotone=monotone, verdict=verdict,
        )

    def evolve_ricci_metric_R1(
        self,
        G_metric: Union[NDArray[np.float64], GeometricSubstrate],
        ricci_tensor: Optional[NDArray[np.float64]] = None,
        delta_tau: float = 0.01,
        *,
        preserve_volume: bool = True,
        deturck_generator: Optional[NDArray[np.float64]] = None,
        return_state: bool = False,
    ) -> Union[NDArray[np.float64], RicciFlowState]:
        r"""
        PRIMER MÉTODO DE LA FASE 2 — continuación exacta de §1.8.

        Doble despacho (retrocompatibilidad ABI v4):
          · `G_metric` ndarray  → devuelve G_{k+1} (legacy) o RicciFlowState.
          · `G_metric` = S (GeometricSubstrate) → usa G y el subrogado Re(V†V) si
            no se suministra Ric, y devuelve RicciFlowState.
        """
        if isinstance(G_metric, GeometricSubstrate):
            S = G_metric
            G = S.G
            ric = self._ricci_surrogate(S.V) if ricci_tensor is None else ricci_tensor
            surrogate = ricci_tensor is None
            state = self._ricci_normalized_step(
                G, _as_matrix("ricci_tensor", ric, S.n, False), delta_tau,
                preserve_volume=preserve_volume, deturck_generator=deturck_generator,
            )
            return replace(state, is_surrogate=surrogate)

        G = _as_matrix("G_metric", G_metric, None, False)
        if ricci_tensor is None:
            raise ValueError("Ric requerida cuando no se suministra el sustrato S")
        state = self._ricci_normalized_step(
            G, _as_matrix("ricci_tensor", ricci_tensor, G.shape[0], False), delta_tau,
            preserve_volume=preserve_volume, deturck_generator=deturck_generator,
        )
        return state if return_state else state.G_next

    # ── 2.2 · R₂: retracción sobre la variedad de grupo (REPARAR, no sólo vetar) ────
    @staticmethod
    def _retract_unitary(W: NDArray[np.complex128], n: int) -> NDArray[np.complex128]:
        r"""
        CAR: W ↦ factor polar unitario (Fan–Hoffman A7: el unitario MÁS PRÓXIMO en
        toda norma unitariamente invariante). La estructura real Σ W̄ Σ = W se
        preserva porque la involución antilineal conmuta con el polar y el factor es
        único; se re-proyecta explícitamente por seguridad numérica.
        """
        sigma = _particle_hole_swap(n)
        W_ph = 0.5 * (W + sigma @ W.conj() @ sigma)
        u, _ = la.polar(W_ph, side="right", check_finite=False)
        return 0.5 * (u + sigma @ u.conj() @ sigma)

    @staticmethod
    def _retract_by_involution(
        A: NDArray, twist: NDArray, max_iter: int = _RETRACTION_MAX_ITER
    ) -> Tuple[NDArray, int, float, bool]:
        r"""
        Retracción de Newton sobre la involución cuyo conjunto de puntos fijos es el
        grupo:
            CCR:  σ(W) = J W^{-†} J ,   fijos = {W : W J W† = J}  (paraunitario)
            Sp :  σ(M) = Ω⁻¹ M^{-ᵀ} Ω = −Ω M^{-ᵀ} Ω , fijos = Sp(2n,ℝ)
        Iteración A ← ½(A + σ(A)) (análogo Ω-torcido de Newton–Schulz X←½(X+X^{-ᵀ})).
        ESTATUS: convergencia cuadrática LOCAL (no se afirma teorema global); se
        certifica el residuo a posteriori, con amortiguación y retención del mejor
        iterado. Si no converge ⇒ DEGRADED, nunca se silencia.
        """
        best, best_res, it, converged = A.copy(), float("inf"), 0, False
        eye = np.eye(A.shape[0], dtype=A.dtype)
        for it in range(1, max_iter + 1):
            try:
                inv_a = la.solve(best, eye, check_finite=False)
            except (la.LinAlgError, ValueError):
                break
            sigma_a = twist @ inv_a.conj().T @ twist
            cand = 0.5 * (best + sigma_a)
            res = _fro(cand.conj().T @ twist @ cand - twist) / max(_fro(twist), _TINY)
            if res < best_res:
                best, best_res = cand, res
                if res <= _SYMPLECTIC_TOL:
                    converged = True
                    break
            else:  # amortiguación (line search geométrico)
                best = 0.5 * (best + cand)
        return best, it, float(best_res), converged

    def retract_canonical_R2(
        self,
        U: NDArray[np.complex128],
        V: NDArray[np.complex128],
        M: NDArray[np.float64],
        is_fermionic: bool,
    ) -> CanonicalRetraction:
        r"""
        R₂ — Retracción canónica: restaura A1/A2/A3 tras la deriva numérica de los
        flujos R₁/R₃. Sin R₂, RSI³ no puede ser de nivel 3 (veta pero no repara).
        Re-audita con los núcleos de la Fase 1 (reuso funtorial, no duplicación).
        """
        n = U.shape[0]
        W = self._assemble_bogoliubov_matrix(U, V)
        before = self._canonical_residuals(U, V, is_fermionic).residual_relative
        sp_before = self._symplectic_audit(M, n).residual_relative

        if is_fermionic:
            W_new = self._retract_unitary(W, n)
            it, conv = 1, True
        else:
            W_new, it, _, conv = self._retract_by_involution(W, _bosonic_signature(n))
            W_new = 0.5 * (W_new + _particle_hole_swap(n) @ W_new.conj() @ _particle_hole_swap(n))

        M_new, it_m, _, conv_m = self._retract_by_involution(M, -_darboux_form(n))
        U_new, V_new = W_new[:n, :n], W_new[:n, n:]
        after = self._canonical_residuals(U_new, V_new, is_fermionic).residual_relative
        sp_after = self._symplectic_audit(M_new, n).residual_relative

        ok = conv and conv_m and after <= _CAR_CCR_TOL and sp_after <= _SYMPLECTIC_TOL
        improved = (after < before) or (sp_after < sp_before)
        verdict = (
            HeytingVerdict.COHERENT if ok
            else HeytingVerdict.DEGRADED if improved
            else HeytingVerdict.VETOED
        )
        return CanonicalRetraction(
            U=U_new, V=V_new, M=np.real(M_new).astype(np.float64),
            residual_before=before, residual_after=after,
            symplectic_before=sp_before, symplectic_after=sp_after,
            iterations=int(max(it, it_m)), converged=bool(ok), verdict=verdict,
        )

    # ── 2.3 · POINCARÉ (i): cartas de Delaunay y carta regular de Poincaré ──────────
    @staticmethod
    def delaunay_from_kepler(
        a: float, e: float, inc: float, mu: float,
        mean_anomaly: float = 0.0, arg_peri: float = 0.0, node: float = 0.0,
    ) -> DelaunayChart:
        r"""
        variables de Delaunay (acción-ángulo canónicas del problema de Kepler):
            L = √(μa),  G = L√(1−e²),  H = G cos i ;  (l,g,h) = (M, ω, Ω).
        Carta REGULAR de Poincaré (elimina las singularidades e→0, i→0):
            P = L−G, φ = −g ;  ξ = √(2P) cos φ = √(2P) cos g,  η = √(2P) sin φ = −√(2P) sin g.
        Canonicidad verificada NUMÉRICAMENTE: det ∂(ξ,η)/∂(G,g) = +1 ⇒ dG∧dg = dξ∧dη.
        """
        if not (a > 0.0 and mu > 0.0 and 0.0 <= e < 1.0):
            raise ValueError("elementos keplerianos fuera de dominio (a>0, μ>0, 0≤e<1)")
        L = math.sqrt(mu * a)
        G_act = L * math.sqrt(1.0 - e * e)
        H_act = G_act * math.cos(inc)
        P = max(L - G_act, 0.0)
        xi = math.sqrt(2.0 * P) * math.cos(arg_peri)
        eta = -math.sqrt(2.0 * P) * math.sin(arg_peri)

        def chart(x: NDArray[np.float64]) -> NDArray[np.float64]:
            p = max(L - float(x[0]), 0.0)
            return np.array([math.sqrt(2 * p) * math.cos(x[1]),
                             -math.sqrt(2 * p) * math.sin(x[1])])

        defect = _canonical_jacobian_defect(chart, np.array([G_act, arg_peri]))
        return DelaunayChart(
            actions=(L, G_act, H_act), angles=(mean_anomaly, arg_peri, node),
            poincare_xi_eta=(xi, eta), chart_canonical_defect=float(defect),
            ordering_valid=(L >= G_act >= abs(H_act) >= 0.0),
        )

    # ── 2.3 · POINCARÉ (ii): teoría secular de Laplace–Lagrange ─────────────────────
    @staticmethod
    def _laplace_coefficient(alpha: float, s: float, j: int, nodes: int = 4096) -> float:
        r"""
        b_s^{(j)}(α) = (1/π)∫₀^{2π} cos(jψ) (1 − 2α cos ψ + α²)^{−s} dψ.
        Cuadratura trapezoidal periódica ⇒ convergencia ESPECTRAL (error ~ e^{−cN})
        para α<1; muy superior a truncar la serie hipergeométrica.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("α debe estar en (0,1) (jerarquía orbital estricta)")
        psi = np.linspace(0.0, 2.0 * np.pi, nodes, endpoint=False)
        denom = (1.0 - 2.0 * alpha * np.cos(psi) + alpha * alpha) ** s
        return float(np.mean(np.cos(j * psi) / denom) * 2.0)

    def _laplace_lagrange_matrices(
        self, masses: Sequence[float], semimajor: Sequence[float], mu: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Sistema secular lineal de Lagrange–Lagrange en z_j = e_j e^{iϖ_j}:  ż = i A z.
            A_jj = +(n_j/4) Σ_{k≠j} (m_k/(M+m_j)) α β b_{3/2}^{(1)}(α)
            A_jk = −(n_j/4) (m_k/(M+m_j)) α β b_{3/2}^{(2)}(α)
            α = min(a_j,a_k)/max(a_j,a_k),  β = α si a_j<a_k else 1.
        ESTRUCTURA: con pesos w_j = m_j√(μ a_j) (momento angular), D A D⁻¹ es
        simétrica (exacta si M+m_j ≈ M), D = diag(√w). Por tanto los autovalores g_j
        (frecuencias seculares de precesión) son REALES: el sistema es un problema
        Hamiltoniano lineal de modos normales — el análogo clásico EXACTO de la
        diagonalización bosónica de Bogoliubov (Williamson/paraunitaria).
        Devuelve (A, B_simetrizado, pesos w).
        """
        m = np.asarray(masses, dtype=np.float64)
        a = np.asarray(semimajor, dtype=np.float64)
        d = m.size
        if a.size != d or np.any(a <= 0) or np.any(m < 0):
            raise ValueError("masas/semiejes inconsistentes")
        central = float(np.max(m))
        n_mean = np.sqrt(mu / a**3)
        A = np.zeros((d, d), dtype=np.float64)
        for j in range(d):
            for k in range(d):
                if j == k:
                    continue
                inner = a[j] < a[k]
                alpha = min(a[j], a[k]) / max(a[j], a[k])
                beta = alpha if inner else 1.0
                pref = 0.25 * n_mean[j] * (m[k] / (central + m[j])) * alpha * beta
                A[j, j] += pref * self._laplace_coefficient(alpha, 1.5, 1)
                A[j, k] = -pref * self._laplace_coefficient(alpha, 1.5, 2)
        w = m * np.sqrt(mu * a)
        sw = np.sqrt(np.maximum(w, _TINY))
        b_sym = (sw[:, None] * A) / sw[None, :]
        b_sym = 0.5 * (b_sym + b_sym.T)
        return A, b_sym, w

    # ── 2.3 · POINCARÉ (iii): resonancias, Chirikov, Diofanto–KAM ───────────────────
    @staticmethod
    def _continued_fraction(x: float, max_terms: int = 24) -> List[int]:
        """Expansión en fracción continua (aplicación de Gauss) y convergentes p/q."""
        terms: List[int] = []
        v = float(x)
        for _ in range(max_terms):
            a_i = int(math.floor(v))
            terms.append(a_i)
            frac = v - a_i
            if frac < 1e-12:
                break
            v = 1.0 / frac
        return terms

    @staticmethod
    def _best_rational(x: float, q_max: int) -> Tuple[int, int, float]:
        """Mejor aproximación racional p/q con q ≤ q_max vía convergentes de Gauss."""
        p0, q0, p1, q1 = 1, 0, int(math.floor(x)), 1
        best = (p1, q1, abs(x - p1 / q1))
        v, terms = float(x), []
        for _ in range(24):
            a_i = int(math.floor(v))
            terms.append(a_i)
            p2, q2 = a_i * p1 + p0, a_i * q1 + q0
            if q2 > q_max:
                break
            err = abs(x - p2 / q2)
            if err < best[2]:
                best = (p2, q2, err)
            p0, q0, p1, q1 = p1, q1, p2, q2
            frac = v - a_i
            if frac < 1e-12:
                break
            v = 1.0 / frac
        return best

    def _resonance_census(
        self,
        frequencies: Sequence[float],
        max_order: int = 12,
        epsilon: float = 1e-3,
        couplings: Optional[Sequence[float]] = None,
        kam_gamma: float = 1e-4,
        kam_tau: float = _KAM_TAU_EXPONENT,
        k_max: int = 6,
    ) -> ResonanceCensus:
        r"""
        (a) Resonancias de movimiento medio: n_i/n_j ≈ p/q con q ≤ max_order
            (fracciones continuas ⇒ mejores aproximantes, no búsqueda ciega).
        (b) CRITERIO DE CHIRIKOV:半anchura de la separatrix del péndulo resonante
            Δ_i = 2√(ε|V_i|); solapamiento K_ij = (Δ_i+Δ_j)/|ω_i−ω_j| ≥ 1 ⇒ caos.
        (c) CONDICIÓN DIOFÁNTICA KAM: γ_eff = min_{0<‖k‖₁≤k_max} |k·ω|·‖k‖₁^τ ≥ γ.
            (Teorema KAM: tori diofánticos sobreviven a perturbación pequeña.)
        (d) Nobleza à la MacKay: proximidad del cociente a φ=(1+√5)/2 (tori nobles
            son los últimos en romperse; cola de fracción continua eventualmente 1).
        """
        w = np.asarray(frequencies, dtype=np.float64)
        d = w.size
        if d < 2:
            return ResonanceCensus((), 0.0, float("inf"), kam_tau, True, 0.0,
                                   HeytingVerdict.DEGRADED)
        coup = np.ones(d) if couplings is None else np.asarray(couplings, dtype=np.float64)
        half_width = 2.0 * np.sqrt(max(epsilon, 0.0) * np.abs(coup))
        resonances: List[Tuple[int, int, int, float]] = []
        k_max_eff = k_max
        while (2 * k_max_eff + 1) ** d > 250_000 and k_max_eff > 2:
            k_max_eff -= 1
        chirikov = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                if abs(w[j]) < 1e-14:
                    continue
                ratio = abs(w[i] / w[j])
                p, q, err = self._best_rational(ratio, max_order)
                if err < 5e-2:
                    resonances.append((i, j, int(p + q), float(err)))
                sep = abs(abs(w[i]) - abs(w[j]))
                if sep > 1e-14:
                    chirikov = max(chirikov, float((half_width[i] + half_width[j]) / sep))

        gamma_eff = float("inf")
        for k in product(range(-k_max_eff, k_max_eff + 1), repeat=d):
            arr = np.asarray(k, dtype=np.float64)
            l1 = float(np.sum(np.abs(arr)))
            if l1 == 0.0 or next(x for x in k if x != 0) < 0:
                continue
            gamma_eff = min(gamma_eff, abs(float(arr @ w)) * l1**kam_tau)

        phi = 0.5 * (1.0 + math.sqrt(5.0))
        nobility = 1.0 / (1.0 + abs(abs(w[0] / w[1]) - phi)) if abs(w[1]) > 1e-14 else 0.0
        verdict = (
            HeytingVerdict.VETOED if chirikov > _CHIRIKOV_HARD
            else HeytingVerdict.DEGRADED if chirikov > _CHIRIKOV_SOFT
            else HeytingVerdict.COHERENT if gamma_eff >= kam_gamma
            else HeytingVerdict.DEGRADED
        )
        return ResonanceCensus(
            resonances=tuple(resonances), chirikov_max=chirikov,
            diophantine_gamma_eff=float(gamma_eff), diophantine_tau=kam_tau,
            kam_satisfied=bool(gamma_eff >= kam_gamma), golden_nobility=float(nobility),
            verdict=verdict,
        )

    # ── 2.3 · POINCARÉ (iv): monodromía, Floquet, Krein, residuos de Greene ─────────
    @staticmethod
    def _floquet_krein_analysis(
        M: NDArray[np.float64], period: float, collision_tol: float = 1e-4
    ) -> FloquetKreinReport:
        r"""
        M = matriz de monodromía de la sección de retorno de Poincaré (¡el mismo
        `jacobian_M` auditado en §1.3!). Consecuencias rigurosas:
          · M ∈ Sp ⇒ espectro palindrómico: λ ↔ 1/λ ↔ λ̄ ↔ 1/λ̄; Σ exponentes = 0.
          · Exponentes de Lyapunov: ℓ_i = log|λ_i|/T.
          · Estabilidad lineal ⇔ |λ|=1 todos y M semisimple (sin bloques de Jordan).
          · SIGNATURA DE KREIN: κ(v) = sgn(−i vᴴΩv) ∈ {±1} (vᴴΩv es imaginario puro).
            Krein–Moser (A8): colisión de λ en el círculo con κ opuestas ⇒ Hopf
            hamiltoniano ⇒ inestabilidad estructural (aunque |λ|=1 hoy).
          · RESIDUO DE GREENE por par: R = (2 − (λ+λ⁻¹))/4 ∈ [0,1] elíptico,
            R<0 hiperbólico (criterio de destrucción de tori KAM).
        """
        n2 = M.shape[0]
        n = n2 // 2
        om = _darboux_form(n)
        ev, vec = la.eig(M, check_finite=False)
        abs_ev = np.abs(ev)
        period = max(float(period), 1e-12)
        lyap = np.log(np.maximum(abs_ev, _TINY)) / period
        pairing = 0.0
        for lam in ev:
            pairing = max(pairing, float(np.min(np.abs(lam * ev - 1.0))))
        cond_vec = float(la.cond(vec)) if vec.size else float("inf")
        jordan = cond_vec > 1.0 / max(_SYMPLECTIC_TOL, 1e-12)

        signatures: List[int] = []
        residues: List[float] = []
        on_circle: List[Tuple[complex, int]] = []
        for idx, lam in enumerate(ev):
            if abs(abs_ev[idx] - 1.0) <= 1e-6:
                v = vec[:, idx]
                krein_val = float(np.real(-1j * (v.conj() @ om @ v)))
                kap = 1 if krein_val > 0 else (-1 if krein_val < 0 else 0)
                signatures.append(kap)
                on_circle.append((complex(lam), kap))
            inv = 1.0 / lam if abs(lam) > 1e-14 else complex(float("inf"))
            residues.append(float(np.real(2.0 - (lam + inv)) / 4.0))

        collision = False
        for a_i in range(len(on_circle)):
            for b_i in range(a_i + 1, len(on_circle)):
                la_, ka = on_circle[a_i]
                lb_, kb = on_circle[b_i]
                if abs(la_ - lb_) < collision_tol and ka * kb < 0:
                    collision = True
        stable = bool(np.all(np.abs(abs_ev - 1.0) <= 1e-6) and not jordan and not collision)
        verdict = (
            HeytingVerdict.COHERENT if stable
            else HeytingVerdict.DEGRADED if not collision and not jordan
            else HeytingVerdict.VETOED
        )
        return FloquetKreinReport(
            multipliers_abs=tuple(float(x) for x in abs_ev),
            lyapunov_exponents=tuple(float(x) for x in lyap),
            lyapunov_sum=float(np.sum(lyap)),
            reciprocal_pairing_defect=pairing,
            krein_signatures=tuple(signatures),
            greene_residues=tuple(residues),
            eigenvector_condition=cond_vec,
            krein_collision=collision,
            jordan_defect_suspected=bool(jordan),
            spectrally_stable=stable,
            verdict=verdict,
        )

    # ── 2.3 · POINCARÉ (v): Melnikov y la trenza homoclínica ────────────────────────
    @staticmethod
    def _melnikov_pendulum(gamma: float, delta: float, omega: float) -> MelnikovReport:
        r"""
        Péndulo perturbado  ẍ + sin x = ε(γ cos ωt − δ ẋ).
        Órbita homoclínica no perturbada: (q,p) = (2 arcsin(tanh t), 2 sech t).
        Melnikov:  M(t₀) = γ∫p cos(ω(t+t₀))dt − δ∫p²dt
                         = 2πγ sech(πω/2) sin(ωt₀) − 8δ
        (∫4sech²t dt = 8 ; ∫sech t cos ωt dt = π sech(πω/2)).
        Ceros SIMPLES ⇔ 2πγ sech(πω/2) > 8δ ⇔ γ/δ > (4/π)cosh(πω/2).
        Por Poincaré–Birkhoff–Smale (A9): ceros simples ⇒ intersecciones homoclínicas
        transversas ⇒ herradura ⇒ caos determinista ⇒ decoherencia de cuasipartícula.
        """
        gamma, delta, omega = abs(gamma), abs(delta), abs(omega)
        arg = min(math.pi * omega / 2.0, _LOG_EXP_CAP)
        threshold = (4.0 / math.pi) * math.cosh(arg)
        sech = 1.0 / math.cosh(arg)
        splitting = 2.0 * math.pi * gamma * sech - 8.0 * delta
        zeros = bool(splitting > 0.0 and delta > 0.0)
        ratio = gamma / delta if delta > 0 else float("inf")
        return MelnikovReport(
            threshold_ratio=float(threshold), forcing_ratio=float(ratio),
            has_simple_zeros=zeros, separatrix_splitting=float(splitting),
            verdict=HeytingVerdict.DEGRADED if zeros else HeytingVerdict.COHERENT,
        )

    # ── 2.3 · POINCARÉ (vi): recurrencia (Kac) y estabilidad de Nekhoroshev ─────────
    @staticmethod
    def _kac_recurrence_time(period: float, cell_fraction: float) -> float:
        r"""
        Liouville (§1.3) + shell compacta ⇒ teorema de recurrencia de Poincaré.
        Lema de Kac (medida preservada, ergódica): ⟨τ_rec⟩ = T·μ(Ω)/μ(celda).
        Cota de escala de Nekhoroshev: T_N ~ T·exp((γ/ε)^{1/(2d)}) — HEURÍSTICA de
        escala (no cota certificada sin verificar la pendiente/"steepness" de H₀).
        """
        f = min(max(float(cell_fraction), 1e-15), 1.0)
        return float(period) / f

    def analyze_poincare_celestial(
        self,
        substrate: GeometricSubstrate,
        *,
        masses: Optional[Sequence[float]] = None,
        semimajor: Optional[Sequence[float]] = None,
        mu: float = 1.0,
        kepler: Optional[Tuple[float, float, float]] = None,
        monodromy_period: float = 1.0,
        frequencies: Optional[Sequence[float]] = None,
        perturbation_epsilon: float = 1e-3,
        couplings: Optional[Sequence[float]] = None,
        melnikov_forcing: Optional[Tuple[float, float, float]] = None,
        section_cell_fraction: float = 1e-3,
        kam_gamma: float = 1e-4,
    ) -> PoincareCelestialReport:
        r"""
        Bloque de MECÁNICA CELESTE DE POINCARÉ: convierte la auditoría algebraica en
        una auditoría DINÁMICA de coherencia (estabilidad, resonancia, caos, recurrencia).
        Módulos no suministrados ⇒ DEGRADED por A11 (ausencia de prueba ≠ verdad).
        """
        chart: Optional[DelaunayChart] = None
        secular: Tuple[float, ...] = ()
        sec_gap, sec_defect = 0.0, 0.0
        evaluated, total = 0, 5

        if kepler is not None:
            chart = self.delaunay_from_kepler(kepler[0], kepler[1], kepler[2], mu)
            evaluated += 1

        freq_source: Optional[NDArray[np.float64]] = None
        if masses is not None and semimajor is not None:
            _, b_sym, _ = self._laplace_lagrange_matrices(masses, semimajor, mu)
            g = la.eigvalsh(b_sym, check_finite=False)
            secular = tuple(float(x) for x in g)
            freq_source = np.abs(g)
            if g.size > 1:
                diffs = np.abs(g[:, None] - g[None, :])
                np.fill_diagonal(diffs, np.inf)
                sec_gap = float(np.min(diffs))
            sec_defect = float(np.max(np.abs(b_sym - b_sym.T)))
            evaluated += 1

        floquet = self._floquet_krein_analysis(substrate.M, monodromy_period)
        evaluated += 1

        if frequencies is not None:
            freq_source = np.asarray(frequencies, dtype=np.float64)
        census: Optional[ResonanceCensus] = None
        if freq_source is not None and freq_source.size >= 2:
            census = self._resonance_census(
                freq_source, epsilon=perturbation_epsilon,
                couplings=couplings, kam_gamma=kam_gamma,
            )
            evaluated += 1

        melnikov: Optional[MelnikovReport] = None
        if melnikov_forcing is not None:
            melnikov = self._melnikov_pendulum(*melnikov_forcing)
            evaluated += 1

        recurrence = self._kac_recurrence_time(monodromy_period, section_cell_fraction)
        d = max(substrate.n, 1)
        eps = max(perturbation_epsilon, 1e-16)
        exponent = min((max(kam_gamma, 1e-16) / eps) ** (1.0 / (2.0 * d)), _LOG_EXP_CAP)
        nekhoroshev = float(monodromy_period) * _safe_exp(exponent)

        parts = [floquet.verdict]
        if census is not None:
            parts.append(census.verdict)
        if melnikov is not None:
            parts.append(melnikov.verdict)
        if chart is not None and (not chart.ordering_valid or chart.chart_canonical_defect > 1e-5):
            parts.append(HeytingVerdict.DEGRADED)
        verdict = self._heyting_aggregate(parts, unverified=(evaluated < total,))
        return PoincareCelestialReport(
            delaunay=chart, secular_frequencies=secular, secular_gap=sec_gap,
            secular_symmetry_defect=sec_defect, floquet=floquet, resonances=census,
            melnikov=melnikov, kac_recurrence_time=recurrence,
            nekhoroshev_time=nekhoroshev, verdict=verdict,
            coverage=float(evaluated / total),
        )

    # ── 2.4 · R₃ (a): replicador de Shahshahani EXPONENCIAL (Baum–Eagon) ────────────
    @staticmethod
    def _replicator_exponential_step(
        p: NDArray[np.float64], K: NDArray[np.float64], dt: float,
        max_backtrack: int = 24,
    ) -> Tuple[NDArray[np.float64], float, int, float, float]:
        r"""
        Flujo replicador en la métrica de Shahshahani (geometría de Fisher–Rao):
            ṗ_i = p_i[(Kp)_i − pᵀKp] ,  d/dt(½pᵀKp) = Var_p(Kp) ≥ 0  (K=Kᵀ).
        Integrador: paso de descenso-ascenso entrópico EXACTO en el simplex
            p'_i ∝ p_i exp(dt (Kp)_i)
        (mirror ascent con divergencia KL; positividad y Σp=1 EXACTAS, sin clip).
        Monotonía discreta CERTIFICADA por backtracking de Armijo sobre F=½pᵀKp:
        se reduce dt hasta F(p') ≥ F(p). Devuelve (p', dt_usado, pasos, ΔF, dH/dt).
        """
        kp = K @ p
        mean_f = float(p @ kp)
        variance = float(np.sum(p * (kp - mean_f) ** 2))
        f_before = 0.5 * mean_f
        step = float(dt)
        p_next = p.copy()
        for k in range(max_backtrack + 1):
            logits = step * kp
            weights = p * np.exp(logits - float(np.max(logits)))
            total = float(np.sum(weights))
            p_next = weights / total if total > 0 else p.copy()
            f_after = 0.5 * float(p_next @ (K @ p_next))
            if f_after >= f_before - 1e-15:
                return p_next, step, k, f_after - f_before, -variance
            step *= 0.5
        return p, 0.0, max_backtrack, 0.0, -variance

    # ── 2.4 · R₃ (b): doble corchete de Brockett por transformación de CAYLEY ───────
    @staticmethod
    def _brockett_cayley_step(
        rho: NDArray[np.complex128], n_diag: NDArray[np.float64], dt: float
    ) -> Tuple[NDArray[np.complex128], float, float]:
        r"""
        ρ̇ = [ρ,[ρ,N]] = [A,ρ] con A = [N,ρ] antihermítica ⇒ flujo ISO-ESPECTRAL.
        Integrador: conjugación unitaria EXACTA vía Cayley (orden 2, unitaria por
        construcción, sin expm):
            U = (I − ½dt A)⁻¹ (I + ½dt A) ,  ρ' = U ρ Uᴴ.
        I − ½dt A es siempre invertible (spec(A) ⊂ iℝ). Preserva espectralmente:
        traza, hermiticidad, positividad, ENTROPÍA y pureza a precisión de máquina
        (el Euler de la v4 sólo conservaba traza/hermiticidad y destruía el espectro).
        Lyapunov de Brockett (A5): d/dt tr(ρN) = −tr(A²) = ‖A‖_F² ≥ 0.
        """
        n = rho.shape[0]
        N = np.diag(n_diag).astype(np.complex128)
        A = N @ rho - rho @ N
        eye = np.eye(n, dtype=np.complex128)
        half = 0.5 * dt * A
        U = la.solve(eye - half, eye + half, check_finite=False)
        rho_next = U @ rho @ U.conj().T
        rho_next = 0.5 * (rho_next + rho_next.conj().T)
        rate = _fro(A) ** 2
        gain = float(np.real(np.trace(rho_next @ N) - np.trace(rho @ N)))
        return rho_next, gain, rate

    def evolve_brockett_shahshahani_R3(
        self,
        p_probabilities: NDArray[np.float64],
        kernel_K: NDArray[np.float64],
        rho_density: NDArray[np.complex128],
        dt: float = 0.01,
        *,
        return_state: bool = False,
    ) -> Union[Tuple[NDArray[np.float64], NDArray[np.complex128], float],
               BrockettShahshahaniState]:
        r"""
        R₃ — Automejora iso-espectral acoplada (maestro: replicador; esclavo: Brockett).
        ABI legacy preservada: por defecto devuelve (p', ρ', dH/dt).
        `return_state=True` devuelve el certificado completo con invariantes medidos.
        Detección de K no simétrica: sin parte simétrica dominante NO hay función de
        Lyapunov (ciclos tipo piedra-papel-tijera, Hopf) ⇒ DEGRADED explícito.
        """
        p = np.asarray(p_probabilities, dtype=np.float64).ravel()
        n = p.size
        K = _as_matrix("kernel_K", kernel_K, n, False)
        rho = _as_matrix("rho_density", rho_density, n, True)
        if np.any(p < -1e-12) or abs(float(np.sum(p)) - 1.0) > 1e-8:
            raise TopologicalInvariantError("[R3] p fuera del simplex Δ^{n−1}")
        p = np.maximum(p, 0.0)
        p /= float(np.sum(p))

        asym = _fro(K - K.T) / max(_fro(K), _TINY)
        K_sym = 0.5 * (K + K.T)
        p_next, dt_used, steps, fitness_gain, dH_dt = self._replicator_exponential_step(p, K_sym, dt)
        rho_next, align_gain, align_rate = self._brockett_cayley_step(rho, p_next, dt)

        w0 = la.eigvalsh(0.5 * (rho + rho.conj().T), check_finite=False)
        w1 = la.eigvalsh(0.5 * (rho_next + rho_next.conj().T), check_finite=False)
        iso_def = float(np.max(np.abs(np.sort(w1) - np.sort(w0))))
        trace_def = abs(complex(np.trace(rho_next)) - 1.0)
        pos_def = max(0.0, -float(w1[0]))

        def _entropy(w: NDArray[np.float64]) -> float:
            w = np.clip(w, 1e-300, None)
            return float(-np.sum(w * np.log(w)))

        entropy_drift = abs(_entropy(w1) - _entropy(w0))
        monotone = (fitness_gain >= -1e-14) and (align_gain >= -1e-12)
        verdict = HeytingVerdict.COHERENT
        if not monotone or asym > 1e-8 or dt_used == 0.0:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if max(iso_def, float(trace_def), pos_def, entropy_drift) > 1e-8:
            verdict = HeytingVerdict.VETOED

        state = BrockettShahshahaniState(
            p_next=p_next, rho_next=rho_next, kernel_used=K_sym, dH_dt=dH_dt,
            fitness_gain=float(fitness_gain), alignment_gain=align_gain,
            alignment_rate=align_rate, isospectral_defect=iso_def,
            trace_defect=float(trace_def), positivity_defect=pos_def,
            entropy_drift=entropy_drift, kernel_asymmetry=asym,
            dt_requested=float(dt), dt_used=float(dt_used), line_search_steps=int(steps),
            monotone=monotone, verdict=verdict,
        )
        if return_state:
            return state
        return p_next, rho_next, dH_dt

    # ── 2.5 · Integración de los flujos en el marco F ───────────────────────────────
    def _integrate_flows(
        self,
        substrate: GeometricSubstrate,
        hyper: RSI3Hyperparameters,
        *,
        ricci_tensor: Optional[NDArray[np.float64]],
        p: Optional[NDArray[np.float64]],
        K: Optional[NDArray[np.float64]],
        rho: Optional[NDArray[np.complex128]],
    ) -> Tuple[Optional[RicciFlowState], Optional[BrockettShahshahaniState], CanonicalRetraction]:
        """Núcleo común de integración (usado por §2.6 y por el lazo RSI³ de §3.6)."""
        ric = self.evolve_ricci_metric_R1(
            substrate, ricci_tensor, hyper.delta_tau, return_state=True
        )
        assert isinstance(ric, RicciFlowState)
        ret = self.retract_canonical_R2(substrate.U, substrate.V, substrate.M, substrate.is_fermionic)
        bro: Optional[BrockettShahshahaniState] = None
        if p is not None and K is not None and rho is not None:
            bro = self.evolve_brockett_shahshahani_R3(p, K, rho, hyper.dt, return_state=True)
            assert isinstance(bro, BrockettShahshahaniState)
        return ric, bro, ret

    # ── 2.6 · MÉTODO TERMINAL DE LA FASE 2 ─────────────────────────────────────────
    #          Definición formal final:  continuation : F ──► R
    #          Su cola `return continuation(F)` ES el inicio de la Fase 3 (§3.1–3.6).
    # ═════════════════════════════════════════════════════════════════════════════════
    def evolve_dynamical_frame(
        self,
        substrate: GeometricSubstrate,
        *,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
        p_probabilities: Optional[NDArray[np.float64]] = None,
        kernel_K: Optional[NDArray[np.float64]] = None,
        rho_density: Optional[NDArray[np.complex128]] = None,
        hyperparameters: Optional[RSI3Hyperparameters] = None,
        celestial_kwargs: Optional[dict] = None,
        continuation: Optional[Callable[[DynamicalFrame], R]] = None,
    ) -> Union[DynamicalFrame, R]:
        r"""
        Φ₂ — Composición R₁ → R₂ → Poincaré → R₃ sobre el sustrato S de §1.8.

        Orden causal: (i) relajar la métrica (R₁), (ii) retraer al grupo (R₂) para que
        la deriva de R₁ no contamin la auditoría, (iii) auditar la dinámica orbital
        (monodromía ya retraída), (iv) cerrar con el flujo iso-espectral (R₃).
        """
        hyper = hyperparameters or RSI3Hyperparameters()
        ric, bro, ret = self._integrate_flows(
            substrate, hyper, ricci_tensor=ricci_tensor,
            p=p_probabilities, K=kernel_K, rho=rho_density,
        )
        repaired = replace(
            substrate, U=ret.U, V=ret.V, M=ret.M,
            W=self._assemble_bogoliubov_matrix(ret.U, ret.V),
            G=ric.G_next,
            canonical=self._canonical_residuals(ret.U, ret.V, substrate.is_fermionic),
            symplectic=self._symplectic_audit(ret.M, substrate.n),
            metric=self._metric_geometry(ric.G_next),
        )
        poincare = self.analyze_poincare_celestial(repaired, **(celestial_kwargs or {}))

        parts = [substrate.verdict, ric.verdict, ret.verdict, poincare.verdict]
        if bro is not None:
            parts.append(bro.verdict)
        coverage = (substrate.coverage + poincare.coverage + (1.0 if bro else 0.0)) / 3.0
        frame = DynamicalFrame(
            substrate=repaired, ricci=ric, retraction=ret, poincare=poincare, brockett=bro,
            verdict=self._heyting_aggregate(parts, unverified=(bro is None,)),
            coverage=float(coverage),
        )
        # ── CONTINUACIÓN → FASE 3 (§3.1 ledger / §3.6 close_rsi3_loop)
        if continuation is None:
            return frame
        return continuation(frame)

    # ═════════════════════════════════════════════════════════════════════════════════
    # ══ FASE 3 · CLAUSURA RSI³: LYAPUNOV · BANACH · AUTO-MEJORA · CROWBAR  (F → C) ══
    # ═════════════════════════════════════════════════════════════════════════════════
    # ── 3.1 · Ledger de Lyapunov: cada componente ≥ 0 y decreciente por diseño ──────
    @staticmethod
    def _lyapunov_ledger(
        substrate: GeometricSubstrate,
        ric: Optional[RicciFlowState],
        bro: Optional[BrockettShahshahaniState],
        poincare: Optional[PoincareCelestialReport],
        weights: Sequence[float],
    ) -> Tuple[float, Tuple[float, ...]]:
        r"""
        Vector de "energías" NO NEGATIVAS, cada una monótona bajo su propio flujo:

          E₁ defecto de Einstein (R₁; puntos fijos = métricas de Einstein)
          E₂ déficit de aptitud ½(λ_max(K) − pᵀKp) ≥ 0  (cota de Rayleigh en Δ)
          E₃ déficit de alineación Σλ↓(ρ)·N↓ − tr(ρN) ≥ 0 (desigualdad de von Neumann)
          E₄ residuo canónico relativo (A1/A2)
          E₅ residuo simpléctico relativo (A3)

        V = Σ w_i E_i / Σ w_i.  HONESTIDAD: la monotonía del COMPUESTO no se deduce de
        la de las partes (los flujos se acoplan); se IMPONE mediante el test de
        aceptación con rollback de §3.3. Ese es el contenido operativo del nivel 3.
        """
        e1 = ric.einstein_defect_after if ric else 0.0
        e2 = e3 = 0.0
        if bro is not None:
            K = bro.kernel_used
            lam_max = float(np.max(la.eigvalsh(K, check_finite=False)))
            e2 = max(0.0, 0.5 * (lam_max - float(bro.p_next @ (K @ bro.p_next))))
            w_rho = np.sort(la.eigvalsh(bro.rho_next, check_finite=False))[::-1]
            n_sorted = np.sort(np.diag(bro.kernel_used))[::-1] if False else np.sort(bro.p_next)[::-1]
            bound = float(np.sum(w_rho * n_sorted))
            e3 = max(0.0, bound - float(np.real(np.trace(bro.rho_next @ np.diag(bro.p_next)))))
        e4 = substrate.canonical.residual_relative
        e5 = substrate.symplectic.residual_relative
        components = (float(e1), float(e2), float(e3), float(e4), float(e5))
        w = np.asarray(weights[:5], dtype=np.float64)
        if w.size < 5:
            w = np.pad(w, (0, 5 - w.size), constant_values=1.0)
        total = float(np.sum(w)) or 1.0
        return float(np.dot(w, components) / total), components

    # ── 3.2 · Certificado de contracción de Banach (a posteriori) ───────────────────
    @staticmethod
    def _banach_contraction_certificate(
        history: Sequence[NDArray[np.float64]],
    ) -> Tuple[float, float, bool]:
        r"""
        Estimación a posteriori del常数 de Lipschitz del operador de iteración T:
            q̂ = ‖x_{k+1}−x_k‖ / ‖x_k−x_{k−1}‖.
        Si q̂<1 (y la estimación es estable en las últimas ventanas), Banach (A10) da
            d(x_k, x*) ≤ q̂/(1−q̂)·‖x_k−x_{k−1}‖.
        Se usa la mediana de las últimas 3 razones (robustez frente a transitorios).
        ESTATUS: certificado LOCAL y empírico del observable de estado, no una prueba
        global de contractividad; se declara como tal en el certificado.
        """
        if len(history) < 3:
            return 1.0, float("inf"), False
        ratios: List[float] = []
        for k in range(len(history) - 1, 1, -1):
            d1 = float(la.norm(history[k] - history[k - 1]))
            d0 = float(la.norm(history[k - 1] - history[k - 2]))
            if d0 > 1e-18:
                ratios.append(d1 / d0)
            if len(ratios) >= 3:
                break
        if not ratios:
            return 1.0, float("inf"), False
        q = float(np.median(ratios))
        last = float(la.norm(history[-1] - history[-2]))
        bound = q / (1.0 - q) * last if q < 1.0 else float("inf")
        return q, float(bound), bool(q < 1.0)

    # ── 3.3 · Auto-modificación de NIVEL 3 con guarda y rollback ────────────────────
    @staticmethod
    def _propose_self_modification(
        hyper: RSI3Hyperparameters,
        accepted: bool,
        q_hat: float,
        ledger_components: Sequence[float],
    ) -> RSI3Hyperparameters:
        r"""
        RSI³ nivel 3: el meta-controlador reescribe los hiperparámetros que gobiernan
        R₁/R₂/R₃ (no el verificador: A12, evita el obstáculo löbiano).

        Política (región de confianza + reasignación de pesos por sensibilidad):
          · rechazo ⇒ dt,Δτ ×0.5 (contracción forzada)
          · aceptación y q̂<0.5 ⇒ ×(1+trust) (aceleración tipo Aitken/Anderson segura)
          · q̂≥1 ⇒ ×0.5 y trust ×0.5 (pérdida de contractividad)
          · pesos ∝ componentes del ledger (mayor peso al defecto dominante),
            normalizados y mezclados 50/50 con los previos (amortiguación).
        Todo cambio queda acotado en [1e-6, 0.5] y es reversible (dataclass frozen).
        """
        gain = 1.0 + hyper.trust_radius if (accepted and q_hat < 0.5) else 0.5
        if accepted and 0.5 <= q_hat < 1.0:
            gain = 1.0
        dt = float(np.clip(hyper.dt * gain, 1e-6, 0.5))
        dtau = float(np.clip(hyper.delta_tau * gain, 1e-6, 0.5))
        trust = float(np.clip(hyper.trust_radius * (1.0 if accepted else 0.5), 1e-3, 0.5))
        comp = np.asarray(ledger_components, dtype=np.float64)
        target = comp / (float(np.sum(comp)) or 1.0)
        prev = np.asarray(hyper.lyapunov_weights[:5], dtype=np.float64)
        if prev.size < 5:
            prev = np.pad(prev, (0, 5 - prev.size), constant_values=0.2)
        mixed = 0.5 * prev / (float(np.sum(prev)) or 1.0) + 0.5 * target
        return RSI3Hyperparameters(
            delta_tau=dtau, dt=dt, trust_radius=trust,
            lyapunov_weights=tuple(float(x) for x in mixed),
            meta_depth=hyper.meta_depth + 1,
        )

    # ── 3.4 · ACT: Crowbar ESP32 (latch O(1); el tiempo real es del firmware) ───────
    @staticmethod
    def actuate_crowbar_esp32(
        verdict: HeytingVerdict,
        canonical_residual: float,
        symplectic_residual: float,
        chirikov_max: float,
        krein_collision: bool,
        dynamical_instability: bool,
    ) -> CrowbarDecision:
        r"""
        Decisión de crowbar en O(1), SIN asignaciones dinámicas en la ruta crítica
        (sólo comparaciones de flotantes precalculados y aritmética de bits).

        Palabra de firmware (GPIO/MCPWM fault-latch):
            bit0 LATCH, bit1 CAR/CCR, bit2 SYMPLECTIC, bit3 CHAOS,
            bit4 KREIN, bit5 DYN_INSTABILITY, bit6 HEYTING_VETO.

        HONESTIDAD DE TIEMPO REAL: 400 ns ≈ 96 ciclos a 240 MHz. Python NO puede
        garantizarlo (GIL, asignador, cachés). El presupuesto es del firmware del
        ESP32 (comparador analógico + MCPWM fault + RMT). Aquí se mide el sobrecoste
        del host y se reporta `host_realtime_guaranteed=False`. Fail-safe: ante
        violación se latchea SIEMPRE, independientemente de la latencia medida.
        """
        t0 = time.perf_counter_ns()
        c1 = canonical_residual > _CAR_CCR_TOL
        c2 = symplectic_residual > _SYMPLECTIC_TOL
        c3 = chirikov_max > _CHIRIKOV_SOFT
        c6 = verdict is HeytingVerdict.VETOED
        latch = c1 or c2 or c3 or krein_collision or dynamical_instability or c6
        word = (int(latch) | (int(c1) << 1) | (int(c2) << 2) | (int(c3) << 3)
                | (int(krein_collision) << 4) | (int(dynamical_instability) << 5)
                | (int(c6) << 6))
        latency = time.perf_counter_ns() - t0
        reason = (
            "LATCH:" + ",".join(
                name for name, flag in (
                    ("CAR_CCR", c1), ("SYMPLECTIC", c2), ("CHIRIKOV", c3),
                    ("KREIN", krein_collision), ("DYN_INST", dynamical_instability),
                    ("HEYTING_VETO", c6),
                ) if flag
            ) if latch else "NOMINAL"
        )
        if latency > _CROWBAR_BUDGET_NS:
            logger.debug("[CROWBAR] sobrecoste host %d ns > presupuesto firmware %d ns",
                         latency, _CROWBAR_BUDGET_NS)
        return CrowbarDecision(
            latch=bool(latch), reason=reason, firmware_word=int(word),
            python_latency_ns=int(latency), budget_ns=_CROWBAR_BUDGET_NS,
            host_realtime_guaranteed=False,
        )

    # ── 3.5 · Emisión del certificado inmutable ────────────────────────────────────
    @staticmethod
    def _emit_certificate(
        substrate: GeometricSubstrate,
        frame: DynamicalFrame,
        meta: RSI3MetaReport,
        crowbar: Optional[CrowbarDecision],
    ) -> BogoliubovGovernanceCertificate:
        """Sello final: escalares/tuplas únicamente (inmutabilidad efectiva)."""
        poin = frame.poincare
        chirikov = poin.resonances.chirikov_max if (poin and poin.resonances) else 0.0
        krein = bool(poin.floquet.krein_collision) if (poin and poin.floquet) else False
        melnikov = bool(poin.melnikov.has_simple_zeros) if (poin and poin.melnikov) else False
        verdict = heyting_meet(frame.verdict, meta.verdict)
        if crowbar is not None and crowbar.latch:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        return BogoliubovGovernanceCertificate(
            car_ccr_residual=substrate.canonical.residual_relative,
            min_quasiparticle_gap=substrate.spectrum.gap,
            lipschitz_connes_bound=substrate.metric.lipschitz_connes_bound,
            symplectic_volume_drift=substrate.symplectic.volume_drift,
            heyting_verdict_value=int(verdict),
            is_quasiparticle_coherent=(verdict is HeytingVerdict.COHERENT),
            gap_is_proxy=substrate.spectrum.is_proxy,
            symplectic_residual=substrate.symplectic.residual_relative,
            condition_number=substrate.metric.condition_number,
            contraction_q=meta.contraction_q,
            banach_error_bound=meta.banach_error_bound,
            lyapunov_initial=meta.lyapunov_initial,
            lyapunov_final=meta.lyapunov_final,
            chirikov_max=chirikov,
            krein_collision=krein,
            melnikov_chaos=melnikov,
            coverage=frame.coverage,
            iterations=meta.iterations,
            crowbar=crowbar,
        )

    # ── 3.6 · MÉTODO TERMINAL DE LA FASE 3: lazo RSI³ cerrado ───────────────────────
    def close_rsi3_loop(
        self,
        frame: DynamicalFrame,
        *,
        hyperparameters: Optional[RSI3Hyperparameters] = None,
        max_iterations: int = 8,
        tolerance: float = 1e-11,
        enable_crowbar: bool = True,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[BogoliubovGovernanceCertificate, RSI3MetaReport, DynamicalFrame]:
        r"""
        Φ₃ — Clausura de la auto-mejora recursiva de nivel 3.

        Ciclo por iteración k:
          1. R₂ retrae (repara A1/A2/A3) → 2. R₁ relaja la métrica →
          3. R₃ avanza p,ρ → 4. re-auditoría con los núcleos de la Fase 1 →
          5. ledger V_k → 6. TEST DE ACEPTACIÓN (V_k ≤ V_{k−1} + tol y sin violación
          dura) ⇒ aceptar; si no, ROLLBACK al checkpoint y contraer el paso →
          7. q̂ de Banach y cota a posteriori → 8. auto-modificación guarded (§3.3).

        Criterio de parada: ΔV < tolerance AND q̂<1 AND residuos ≤ umbrales, o
        agotamiento del presupuesto. Nunca se auto-modifica el verificador (A12).
        """
        hyper = hyperparameters or RSI3Hyperparameters()
        S = frame.substrate
        G = S.G.copy()
        U, V, M = S.U.copy(), S.V.copy(), S.M.copy()
        p = frame.brockett.p_next.copy() if frame.brockett else None
        rho = frame.brockett.rho_next.copy() if frame.brockett else None
        K = frame.brockett.kernel_used if frame.brockett else None
        ric_external = ricci_tensor
        if ric_external is None and frame.ricci is not None and not frame.ricci.is_surrogate:
            ric_external = frame.ricci.ricci_used

        v_prev, comps = self._lyapunov_ledger(S, frame.ricci, frame.brockett, frame.poincare,
                                              hyper.lyapunov_weights)
        v0 = v_prev
        history_v: List[float] = [v_prev]
        history_x: List[NDArray[np.float64]] = [self._state_vector(G, p, rho)]
        accepted = rejected = 0
        q_hat, bound, certified = 1.0, float("inf"), False
        checkpoint = (G.copy(), U.copy(), V.copy(), M.copy(),
                      None if p is None else p.copy(), None if rho is None else rho.copy())
        ric_state, bro_state, ret_state = frame.ricci, frame.brockett, frame.retraction
        monotone = True

        for k in range(max(1, int(max_iterations))):
            ric_in = ric_external if ric_external is not None else self._ricci_surrogate(V)
            ric_state = self._ricci_normalized_step(G, ric_in, hyper.delta_tau)
            ret_state = self.retract_canonical_R2(U, V, M, S.is_fermionic)
            if p is not None and K is not None and rho is not None:
                bro_state = self.evolve_brockett_shahshahani_R3(
                    p, K, rho, hyper.dt, return_state=True
                )  # type: ignore[assignment]

            G_c, U_c, V_c, M_c = ric_state.G_next, ret_state.U, ret_state.V, ret_state.M
            p_c = bro_state.p_next if bro_state else None
            rho_c = bro_state.rho_next if bro_state else None
            S_c = replace(
                S, G=G_c, U=U_c, V=V_c, M=M_c,
                W=self._assemble_bogoliubov_matrix(U_c, V_c),
                canonical=self._canonical_residuals(U_c, V_c, S.is_fermionic),
                symplectic=self._symplectic_audit(M_c, S.n),
                metric=self._metric_geometry(G_c),
                spectrum=self._quasiparticle_spectrum(U_c, V_c, S.is_fermionic),
            )
            v_k, comps = self._lyapunov_ledger(S_c, ric_state, bro_state, frame.poincare,
                                               hyper.lyapunov_weights)
            hard = (
                not S_c.canonical.is_canonical or not S_c.metric.is_spd
                or S_c.spectrum.dynamical_instability or S_c.spectrum.contraction_violation
                or (bro_state is not None and bro_state.verdict is HeytingVerdict.VETOED)
            )
            ok = (v_k <= v_prev + tolerance) and not hard

            if ok:
                accepted += 1
                G, U, V, M, p, rho = G_c, U_c, V_c, M_c, p_c, rho_c
                S = S_c
                checkpoint = (G.copy(), U.copy(), V.copy(), M.copy(),
                              None if p is None else p.copy(),
                              None if rho is None else rho.copy())
                history_v.append(v_k)
                history_x.append(self._state_vector(G, p, rho))
                q_hat, bound, certified = self._banach_contraction_certificate(history_x)
                delta_v = abs(v_prev - v_k)
                v_prev = v_k
            else:
                rejected += 1
                monotone = False
                G, U, V, M, p, rho = (x.copy() if isinstance(x, np.ndarray) else x
                                      for x in checkpoint)  # type: ignore[misc]
                logger.warning("[RSI3] iter %d RECHAZADA (ΔV=%.3e, hard=%s) → rollback",
                               k, v_k - v_prev, hard)
                delta_v = float("inf")

            hyper = self._propose_self_modification(hyper, ok, q_hat, comps)
            if ok and delta_v < tolerance and certified and S.canonical.is_canonical \
                    and S.symplectic.is_symplectic:
                break

        meta_verdict = HeytingVerdict.COHERENT
        if not monotone or not certified or v_prev > v0:
            meta_verdict = HeytingVerdict.DEGRADED
        if v_prev > v0 + 1e-6:
            meta_verdict = HeytingVerdict.VETOED
        meta = RSI3MetaReport(
            iterations=int(accepted + rejected), accepted=accepted, rejected=rejected,
            lyapunov_initial=float(v0), lyapunov_final=float(v_prev),
            lyapunov_history=tuple(float(x) for x in history_v),
            contraction_q=float(q_hat), banach_error_bound=float(bound),
            contraction_certified=bool(certified), monotone=bool(monotone),
            hyperparameters=hyper, verdict=meta_verdict,
        )
        final_frame = replace(
            frame, substrate=S, ricci=ric_state, retraction=ret_state, brockett=bro_state,
            verdict=heyting_meet(frame.verdict, meta_verdict),
        )
        crowbar = None
        if enable_crowbar:
            poin = final_frame.poincare
            crowbar = self.actuate_crowbar_esp32(
                verdict=heyting_meet(final_frame.verdict, meta_verdict),
                canonical_residual=S.canonical.residual_relative,
                symplectic_residual=S.symplectic.residual_relative,
                chirikov_max=poin.resonances.chirikov_max if (poin and poin.resonances) else 0.0,
                krein_collision=bool(poin.floquet.krein_collision) if (poin and poin.floquet) else False,
                dynamical_instability=S.spectrum.dynamical_instability,
            )
            if crowbar.latch:
                logger.error("[CROWBAR_ESP32] %s word=0x%02x (host %d ns)",
                             crowbar.reason, crowbar.firmware_word, crowbar.python_latency_ns)
        certificate = self._emit_certificate(S, final_frame, meta, crowbar)
        return certificate, meta, final_frame

    @staticmethod
    def _state_vector(
        G: NDArray[np.float64],
        p: Optional[NDArray[np.float64]],
        rho: Optional[NDArray[np.complex128]],
    ) -> NDArray[np.float64]:
        """Inmersión del estado en ℝ^N para el certificado de contracción (§3.2)."""
        parts = [np.asarray(G, dtype=np.float64).ravel()]
        if p is not None:
            parts.append(np.asarray(p, dtype=np.float64).ravel())
        if rho is not None:
            parts.append(np.real(rho).ravel())
            parts.append(np.imag(rho).ravel())
        return np.concatenate(parts)

    # ── 3.7 · API PÚBLICA: plegado de las TRES FASES ANIDADAS (Φ₃∘Φ₂∘Φ₁) ────────────
    def audit_bogoliubov_transformation(
        self,
        U_matrix: NDArray[np.complex128],
        V_matrix: NDArray[np.complex128],
        jacobian_M: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        is_fermionic: bool = True,
        *,
        bdg_hamiltonian: Optional[NDArray] = None,
        quasiparticle_energies: Optional[Sequence[float]] = None,
        covariance_sigma: Optional[NDArray] = None,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
        p_probabilities: Optional[NDArray[np.float64]] = None,
        kernel_K: Optional[NDArray[np.float64]] = None,
        rho_density: Optional[NDArray[np.complex128]] = None,
        celestial_kwargs: Optional[dict] = None,
        hyperparameters: Optional[RSI3Hyperparameters] = None,
        max_iterations: int = 8,
        strict: bool = True,
        enable_crowbar: bool = True,
    ) -> BogoliubovGovernanceCertificate:
        r"""
        Punto de entrada ABI-compatible con la v4.0.0 (mismos 5 argumentos posicionales).

        El plegado se realiza por CONTINUACIONES ANIDADAS (Kleisli):

            Φ₁(S) ──continuation──► Φ₂(F) ──continuation──► Φ₃(C)

        de modo que la definición formal final de la Fase 1 es literalmente el arranque
        de la Fase 2, y la de la Fase 2 el arranque de la Fase 3. `strict=True`
        conserva el veto DURO legacy; los datos opcionales ausentes se marcan
        DEGRADED (A11) y reducen `coverage`, nunca se silencian como coherentes.
        """
        certificate, _meta, _frame = self.certify_geometric_substrate(
            U_matrix, V_matrix, jacobian_M, metric_G, is_fermionic,
            bdg_hamiltonian=bdg_hamiltonian,
            quasiparticle_energies=quasiparticle_energies,
            covariance_sigma=covariance_sigma,
            strict=strict,
            continuation=lambda S: self.evolve_dynamical_frame(
                S,
                ricci_tensor=ricci_tensor,
                p_probabilities=p_probabilities,
                kernel_K=kernel_K,
                rho_density=rho_density,
                hyperparameters=hyperparameters,
                celestial_kwargs=celestial_kwargs,
                continuation=lambda F: self.close_rsi3_loop(
                    F,
                    hyperparameters=hyperparameters,
                    max_iterations=max_iterations,
                    enable_crowbar=enable_crowbar,
                    ricci_tensor=ricci_tensor,
                ),
            ),
        )
        return certificate

    def execute_rsi3_cycle(self, *args, **kwargs) -> OmegaCycleOutcome:
        r"""
        Variante de auditoría TOTAL: devuelve la traza completa (S, F, meta, C) para
        reproducibilidad y auditoría forense del Ágora Tensorial.
        """
        trace: dict = {}

        def _capture_substrate(S: GeometricSubstrate):
            trace["S"] = S

            def _capture_frame(F: DynamicalFrame):
                trace["F"] = F
                out = self.close_rsi3_loop(
                    F,
                    hyperparameters=kwargs.get("hyperparameters"),
                    max_iterations=kwargs.get("max_iterations", 8),
                    enable_crowbar=kwargs.get("enable_crowbar", True),
                    ricci_tensor=kwargs.get("ricci_tensor"),
                )
                trace["C"], trace["meta"], trace["F_final"] = out
                return out

            return self.evolve_dynamical_frame(
                S,
                ricci_tensor=kwargs.get("ricci_tensor"),
                p_probabilities=kwargs.get("p_probabilities"),
                kernel_K=kwargs.get("kernel_K"),
                rho_density=kwargs.get("rho_density"),
                hyperparameters=kwargs.get("hyperparameters"),
                celestial_kwargs=kwargs.get("celestial_kwargs"),
                continuation=_capture_frame,
            )

        self.certify_geometric_substrate(
            *args,
            strict=kwargs.get("strict", False),
            bdg_hamiltonian=kwargs.get("bdg_hamiltonian"),
            quasiparticle_energies=kwargs.get("quasiparticle_energies"),
            covariance_sigma=kwargs.get("covariance_sigma"),
            continuation=_capture_substrate,
        )
        return OmegaCycleOutcome(
            substrate=trace["S"], frame=trace["F_final"],
            meta=trace["meta"], certificate=trace["C"],
        )