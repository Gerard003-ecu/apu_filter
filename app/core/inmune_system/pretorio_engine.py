# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pretorio Engine (Caballería Suprema de Cálculo Epistémico)          ║
║ Ruta   : app/core/inmune_system/pretorio_engine.py                           ║
║ Versión: 5.0.0-Nested-Poincare-Cartan-Christoffel-CZ-Birkhoff-KAM-Cech-      ║
║          deRham-Brouwer-Heyting-Ultrafilter-PhD                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Este motor supremo ejecuta el escrutinio final e independiente de la coherencia
ciber-física. Evalúa la hipercohomología del bicomplejo de Čech–de Rham, purga
las corrientes asimétricas en la base de la MAC mediante simetrización de
Weyl–Toeplitz y colapsa los veredictos parciales en un ultrafiltro booleano
binario. Sobre esta armazón, teje la Mécanique Céleste de Henri Poincaré en tres
fases anidadas, donde el objeto terminal de una es la continuación formal de la
siguiente.

FASE I — Geometría de la fase (Banach + Weyl + Maupertuis–Jacobi + Poincaré–Cartan):
  1. Sumación compensada Kahan / KBN / Klein en el álgebra de Banach (ℝ, +, ·).
  2. Simetrización de Weyl–Toeplitz Π_Herm(M) = (M + M†)/2.
  3. Estado densidad más próximo (Higham: Herm + PSD + Tr = 1).
  4. Métrica conforme de Maupertuis–Jacobi g̃ = 2(H₀ − V) g = n(q)² g.
  5. Región de Hill: H₀ − V(q) > 0 (curva de velocidad cero ∂D_H).
  6. Símbolos de Christoffel conformes de Koszul–Levi-Civita y torsión de Koszul.
  7. Marea geodésica de Jacobi ‖∇V‖² / (H₀ − V).
  8. 1-forma de Poincaré–Cartan λ = p dq − H dt y circulación discreta.
  9. 2-forma canónica de Liouville Ω (Darboux) y defecto de pullback Φ*ω − ω.
 10. Gérmen del bicomplejo Čech–de Rham y su Laplaciano de Hodge Δ_D.
  ⇒ Objeto terminal 𝒢_I = _PretorioCelestialGerm  (inicial de Fase II).

FASE II — Certificación (Hipercohomología + Brouwer + KAM + Melnikov + Birkhoff):
 11. Continuación formal: lift_from_celestial_germ(𝒢_I).
 12. Hipercohomología del bicomplejo: D² = δ² + d² + {δ, d} ≡ 0.
 13. Brouwer en 𝒟(ℋ): ρ = f(ρ) con Weyl–Toeplitz.
 14. Pequeños divisores de Poincaré–KAM: |⟨k, ω⟩| ≥ γ/|k|^τ, τ > n − 1.
 15. Ecuación homológica de Poincaré–Lindstedt: i⟨k,ω⟩ χ_k = (H₁)_k.
 16. Resonancias de Arnol'd y absorción ultramétrica de Novikov.
 17. Función de Melnikov M(t₀) = ∫ {H₀, H₁}(γ⁰(t − t₀)) dt.
 18. Twist map de Poincaré–Birkhoff + teorema geométrico último.
 19. Mapa de retorno P: Σ → Σ con Floquet, Lyapunov, Conley–Zehnder y ρ.
  ⇒ Objeto terminal 𝒢_II = _UltrafilterCelestialGerm  (inicial de Fase III).

FASE III — Colapso (Heyting H₃ + Ultrafiltro booleano + Crowbar ESP32):
 20. Continuación formal: collapse_from_ultrafilter_celestial_germ(𝒢_II).
 21. Valuación de Heyting Ω₃ (orden de permiso): VETOED ≤ DEGRADED ≤ COHERENT.
 22. Dual de severidad: COHERENT ≺ DEGRADED ≺ VETOED (átomo generador).
 23. Ultrafiltro 𝒰 : H₃ⁿ → 2 = {VIABLE, RECHAZAR} gobernado por el MEET.
 24. Colapso a actuación en silicio real (< 400 ns) via GPIO14 / BT151.

ÁLGEBRA DE HEYTING — DUALIDAD PERMISO / SEVERIDAD:
  Permiso  (Gödel): VETOED=0.0 ≤ DEGRADED=0.5 ≤ COHERENT=1.0
                    meet = min = peor permiso = colapso de seguridad.
  Severidad:        COHERENT=0 ≺ DEGRADED=1 ≺ VETOED=2
                    max = átomo generador del filtro.
  El join (supremo de verdad) es diagnóstico; el meet gobierna el interlock.
  Corrección v5: un único VETOED basta para RECHAZAR (el join lo ocultaría).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.PretorioEngine")

__version__: Final[str] = (
    "5.0.0-Nested-Poincare-Cartan-Christoffel-CZ-Birkhoff-KAM-Cech-"
    "deRham-Brouwer-Heyting-Ultrafilter-PhD"
)

# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA (WILKINSON & HIGHAM)
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-12
_HIGHAM_TIKHONOV_REG: Final[float] = 1e-15
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_BROUWER_VETO_HS: Final[float] = 1e-6
_BROUWER_DEGRADED_HS: Final[float] = 1e-9
_BROUWER_VETO_TRACE: Final[float] = 1e-9
_BROUWER_DEGRADED_TRACE: Final[float] = 1e-12
_HYPER_VETO_SCALE: Final[float] = 100.0
_PSD_NEG_TOL: Final[float] = 1e-10

# Constantes celestes de Poincaré
_KAM_TAU_FLOOR: Final[float] = 1.0
_KAM_GAMMA_FLOOR: Final[float] = 1e-12
_KAM_DIVISOR_FLOOR: Final[float] = 1e-9
_MELNIKOV_QUAD_NODES: Final[int] = 513
_MELNIKOV_T_INF: Final[float] = 25.0
_FLOQUET_PARABOLIC_BAND: Final[float] = 1e-6
_LYAPUNOV_CLIP: Final[float] = 700.0
_HILL_MARGIN_FLOOR: Final[float] = 1e-12
_BIRKHOFF_AREA_DRIFT_MAX: Final[float] = 1e-12
_BIRKHOFF_TWIST_FLOOR: Final[float] = 1e-9
_LOG_EXP_CLIP: Final[float] = 700.0
_CSMD_STEP: Final[float] = 1e-8
_CARTAN_DEFECT_TOL: Final[float] = 1e-8
_CHRISTOFFEL_BLOWUP: Final[float] = 1.0e3
_JACOBI_TIDAL_VETO: Final[float] = 1.0e2
_HOMOLOGICAL_KAM_CEILING: Final[float] = 1.0e8
_SECTION_TRANSVERSALITY_FLOOR: Final[float] = 1.0e-12
_UNIT_CIRCLE_ATOL: Final[float] = 1.0e-8
_CZ_DEGENERATE_ATOL: Final[float] = 1.0e-8
_KOSZUL_TORSION_TOL: Final[float] = 1.0e-12
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_COND_WARN: Final[float] = 1.0e12

# Dualidad Heyting: permiso (Gödel) vs severidad (átomo del filtro)
_HEYTING_SEVERITY: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_CANONICAL_VERDICTS: Final[Tuple[str, ...]] = ("COHERENT", "DEGRADED", "VETOED")
_HEYTING_ORDER: Final[Dict[str, int]] = _HEYTING_SEVERITY  # alias retrocompatible


# =============================================================================
# RETÍCULO DE HEYTING (álgebra de Gödel G₃) — orden de PERMISO
# =============================================================================
class HeytingVerdict(Enum):
    r"""
    Retículo de Heyting lineal de tres valores (Gödel G₃).

    Orden de permiso / coherencia:
        VETOED ≤ DEGRADED ≤ COHERENT

    En este orden:
      • meet = min = ínfimo de permiso  = PEOR CASO de seguridad (colapso).
      • join = max = supremo de verdad  = MEJOR CASO diagnóstico.
    El ultrafiltro de Fase III DEBE gobernarse por el meet, jamás por el join:
    join(VETOED, COHERENT) = COHERENT ocultaría un veto.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(self.value, other.value))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Implicación de Gödel: a → b = 1 si a ≤ b, else b."""
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @property
    def godel_value(self) -> float:
        return {0: 0.0, 1: 0.5, 2: 1.0}[self.value]

    @property
    def severity(self) -> int:
        """Dual de severidad: COHERENT=0 ≺ DEGRADED=1 ≺ VETOED=2."""
        return 2 - self.value

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[str(token)]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc

    @classmethod
    def meet_all(cls, *verdicts: "HeytingVerdict") -> "HeytingVerdict":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc

    @classmethod
    def join_all(cls, *verdicts: "HeytingVerdict") -> "HeytingVerdict":
        acc = cls.VETOED
        for v in verdicts:
            acc = acc.join(v)
        return acc


# =============================================================================
# REPORTES Y CERTIFICADOS INMUTABLES
# =============================================================================
@dataclass(frozen=True, slots=True)
class PretorioSpectrumReport:
    r"""
    Reporte del espectro del Twist Map de Poincaré–Birkhoff sobre la variedad anular.

    Invariantes de Mécanique Céleste de Henri Poincaré:
      1. Area Drift: Δ_Area = |det M − 1|.
      2. Opposite Twist: θ'_a − θ > 0 > θ'_b − θ.
      3. Fixed Points: |Fix(f)| = |{λ_k ∈ Spec(M) : |λ_k| = 1}| ≥ 2.
      4. Validez: opposite_twist ∧ area_drift ≤ 10^{-12} ∧ |Fix| ≥ 2.
    """

    area_drift: float
    has_opposite_twist: bool
    fixed_points_count: int
    is_spectrum_valid: bool


@dataclass(frozen=True, slots=True)
class MaupertuisStepReport:
    r"""
    Informe de un paso Störmer–Verlet sobre la métrica conforme de Maupertuis–Jacobi.

    Transporta n(q), densidad de acción, drift de Liouville y Hamiltonian.
    Compatible con el contrato consumido por ImperialGuardsCenturions.
    """

    refractive_index_n: float
    maupertuis_action_density: float
    volume_drift_det: float
    is_symplectic_coherent: bool
    hamiltonian_energy: float
    hill_margin: float
    christoffel_strength: float = 0.0
    jacobi_tidal_norm: float = 0.0
    cartan_circulation: float = 0.0
    koszul_torsion: float = 0.0
    x_next: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class _MaupertuisJacobiGerm:
    r"""
    Gérmen de la métrica conforme de Maupertuis–Jacobi.

    Para H(q, p) = H₀ con H₀ − V(q) > 0,
        g̃_{jk}(q) = 2(H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q),
    y el principio variacional abreviado
        S_M[γ] = ∫ √(2(H₀ − V) g_{jk} q̇^j q̇^k) dτ = ∫ d s̃.
    """

    conformal_factor: float
    refractive_index: float
    hill_margin: float
    is_in_hill_region: bool
    min_eigenvalue: float
    is_positive_definite: bool
    christoffel_strength: float = 0.0
    jacobi_tidal_norm: float = 0.0
    koszul_torsion: float = 0.0


@dataclass(frozen=True, slots=True)
class _PoincareCartanGerm:
    r"""
    Gérmen de la 1-forma de Poincaré–Cartan.

    λ = p dq − H dt,  dλ = ω − dH ∧ dt.
    Invariante integral absoluto de Poincaré (É. Cartan, 1922).
    """

    lambda_vector: np.ndarray
    hamiltonian_value: float
    dim: int
    two_n: int
    symplectic_skew_residual: float
    cartan_lagrangian: float = 0.0
    darboux_ok: bool = True


@dataclass(frozen=True, slots=True)
class _HypercohomologyGerm:
    r"""
    Gérmen del bicomplejo Čech–de Rham (objeto clásico de Fase I).

    Par (δ, d) con factibilidad δ², d², δd + dδ, y Laplaciano de Hodge
    Δ_D = D*D + DD* del diferencial total D = δ + d.
    """

    d1: np.ndarray
    d2: np.ndarray
    d1_square: bool
    d2_square: bool
    composition_lr: bool
    composition_rl: bool
    hodge_laplacian: Optional[np.ndarray]
    reg_floor: float
    fro_d1: float
    fro_d2: float


@dataclass(frozen=True, slots=True)
class _PretorioCelestialGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN CELESTE (objeto terminal de Fase I, inicial de Fase II).
    ═══════════════════════════════════════════════════════════════════════════
    Compone la geometría metrológica de Fase I:
      • Maupertuis–Jacobi (g̃, n(q), región de Hill, Christoffel, marea).
      • Poincaré–Cartan (λ = p dq − H dt).
      • Darboux (Ω, dim) y defecto de pullback.
      • Hipercohomología (δ, d) y Laplaciano de Hodge.
      • Piso de regularización de Wilkinson.
    """

    two_n: int
    n: int
    omega: np.ndarray
    maupertuis_germ: _MaupertuisJacobiGerm
    poincare_cartan_germ: _PoincareCartanGerm
    hyper_germ: _HypercohomologyGerm
    reg_floor: float
    darboux_residual: float = 0.0
    almost_complex_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class _KAMAudit:
    r"""
    Auditoría KAM de pequeños divisores de Poincaré.

    Verifica |⟨k, ω⟩| ≥ γ / |k|^τ y la absorción ultramétrica T-ádica en Λ_Nov:
        W_Nov(k, ω) = exp(−T_val / (ε_floor + |⟨k, ω⟩|)).
    Adjunta el residual homológico de Lindstedt χ ∼ 1/|⟨k,ω⟩| y el defecto
    de pullback de Cartan ‖Mᵀ Ω M − Ω‖_F.
    """

    min_divisor: float
    tau: float
    gamma: float
    resonance_gap: float
    is_diophantine: bool
    novikov_weight: float
    maurercartan_residual: float
    volume_drift: float
    is_kam_stable: bool
    engine_ok: bool = True
    homological_residual: float = 0.0
    cartan_pullback_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class _MelnikovAudit:
    r"""
    Auditoría de la función de Melnikov.

    M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt.
    Cero simple ⇒ fractura homoclínica (Poincaré 1890, Melnikov 1963).
    """

    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    engine_ok: bool = True


@dataclass(frozen=True, slots=True)
class _ReturnMapAudit:
    r"""
    Auditoría del mapa de retorno de Poincaré P: Σ → Σ
    (Floquet + Lyapunov + Conley–Zehnder + ρ + Birkhoff + Cartan).
    """

    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    max_lyapunov: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    engine_ok: bool = True
    floquet_parabolic: float = 0.0
    conley_zehnder_index: int = 0
    cz_nondegenerate: bool = True
    cz_distance_to_one: float = 1.0
    rotation_number: float = 0.0
    birkhoff_twist: float = 0.0
    elliptic_multiplicity: int = 0
    section_transversality: float = 1.0
    cartan_pullback_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class _PoincareBirkhoffAudit:
    r"""
    Auditoría del twist map de Poincaré–Birkhoff.

    Verifica giro opuesto, conservación de área, |Spec(M) ∩ S¹| ≥ 2
    y el invariante integral absoluto Φ*ω = ω.
    """

    area_drift: float
    has_opposite_twist: bool
    fixed_points_count: int
    is_spectrum_valid: bool
    engine_ok: bool = True
    cartan_pullback_defect: float = 0.0
    rotation_number: float = 0.0
    birkhoff_twist: float = 0.0


@dataclass(frozen=True, slots=True)
class _UltrafilterCelestialGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN CELESTE DE ULTRAFILTRO (objeto terminal de Fase II, inicial Fase III).
    ═══════════════════════════════════════════════════════════════════════════
    Empaqueta los veredictos locales H₃ (hipercohomología, Brouwer, KAM,
    Melnikov, Poincaré–Birkhoff, mapa de retorno) con sus certificados y la
    valuación de Gödel, listos para el colapso booleano del Pretorio.
    """

    hyper: "_HypercohomologyResult"
    brouwer: "_BrouwerResult"
    kam: Optional[_KAMAudit]
    melnikov: Optional[_MelnikovAudit]
    birkhoff: Optional[_PoincareBirkhoffAudit]
    return_map: Optional[_ReturnMapAudit]
    heyting_verdicts: Tuple[str, ...]
    godel_values: np.ndarray
    two_n: int
    heyting_meet: str = "COHERENT"
    heyting_join: str = "VETOED"
    symplectic_cartan_defect: float = 0.0
    conley_zehnder_index: int = 0
    rotation_number: float = 0.0


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE I — NÚCLEO DE BANACH, WEYL–TOEPLITZ, MAUPERTUIS–JACOBI,            ║
# ║          POINCARÉ–CARTAN, CHRISTOFFEL, JACOBI, HODGE                     ║
# ║                                                                          ║
# ║ Objetos: sumas compensadas, proyección hermítica, estado densidad,       ║
# ║          métrica conforme, región de Hill, 1-forma de Cartan,            ║
# ║          conexión de Koszul, marea de Jacobi, 2-forma de Liouville.      ║
# ║                                                                          ║
# ║ Morfismo terminal (I.10): synthesize_pretorio_celestial_germ             ║
# ║     ↦ 𝒢_I = _PretorioCelestialGerm (inicial de Fase II).                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
class _NumericalCore:
    r"""
    Fase I. Álgebra numérica de precisión metrológica.

    Topos lineal subyacente: sumación compensada en (ℝ, +, ·), proyección de
    Weyl–Toeplitz al cono hermítico y proyección de Higham al simplejo 𝒟(ℋ).
    Sobre esta base teje Maupertuis–Jacobi y Poincaré–Cartan.
    """

    # ── I.1 Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """Sumación compensada de Kahan."""
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            if not np.isfinite(x):
                raise ValueError("kahan_sum: se detectó un no-finito.")
            y = float(x) - c
            t = total + y
            c = (t - total) - y
            total = t
        return float(total)

    @staticmethod
    def kahan_babuska_neumaier_sum(arr: np.ndarray) -> float:
        """Sumación de Kahan–Babuška–Neumaier (KBN)."""
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                raise ValueError("kahan_babuska_neumaier_sum: no-finito.")
            t = total + xf
            if abs(total) >= abs(xf):
                c += (total - t) + xf
            else:
                c += (xf - t) + total
            total = t
        return float(total + c)

    kahan_babuska_neumann_sum = kahan_babuska_neumaier_sum

    @staticmethod
    def klein_sum(arr: np.ndarray) -> float:
        """Sumación doblemente compensada de Klein (error O(u²) relativo)."""
        s = 0.0
        cs = 0.0
        ccs = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                raise ValueError("klein_sum: no-finito.")
            t = s + xf
            if abs(s) >= abs(xf):
                c = (s - t) + xf
            else:
                c = (xf - t) + s
            s = t
            t = cs + c
            if abs(cs) >= abs(c):
                cc = (cs - t) + c
            else:
                cc = (c - t) + cs
            cs = t
            ccs += cc
        return float(s + cs + ccs)

    # ── I.2 Normas, validación y Weyl–Toeplitz ───────────────────────────
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt / Frobenius ‖A‖_F."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

    @staticmethod
    def euclidean_norm(vec: np.ndarray) -> float:
        """Norma euclídea ‖v‖₂ con acumulación KBN."""
        v = np.asarray(vec, dtype=np.float64).ravel()
        if v.size == 0:
            return 0.0
        return float(np.sqrt(max(_NumericalCore.kahan_babuska_neumaier_sum(v * v), 0.0)))

    @staticmethod
    def cstar_norm(matrix: np.ndarray) -> float:
        """Norma C* = ‖A‖₂ (radio espectral de A†A)^{1/2}."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, 2))

    @staticmethod
    def banach_condition_number(matrix: np.ndarray) -> float:
        svals = la.svdvals(np.asarray(matrix))
        smax = float(svals[0]) if svals.size else 0.0
        smin = float(svals[-1]) if svals.size else 0.0
        if smin <= _WILKINSON_DEFLATION_FLOOR * max(smax, 1.0):
            return float("inf")
        return smax / smin

    @staticmethod
    def assert_finite(name: str, array: np.ndarray) -> None:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contiene entradas no finitas.")

    @staticmethod
    def assert_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(matrix)
        if a.ndim == 1:
            side = int(np.sqrt(a.size))
            if side * side != a.size:
                raise ValueError(f"{name} plana no es un cuadrado perfecto.")
            a = a.reshape(side, side)
        if a.ndim != 2:
            raise ValueError(f"{name} debe ser de rango 2; recibido {a.shape}.")
        _NumericalCore.assert_finite(name, a)
        return a

    @staticmethod
    def assert_square(name: str, matrix: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        a = _NumericalCore.assert_matrix(name, matrix)
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; recibido {a.shape}.")
        if dim is not None and a.shape[0] != dim:
            raise ValueError(f"{name} debe ser {dim}×{dim}; recibido {a.shape}.")
        return a

    @staticmethod
    def assert_vec(name: str, vec: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        v = np.asarray(vec).reshape(-1)
        if dim is not None and v.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; recibido {v.size}.")
        _NumericalCore.assert_finite(name, v)
        return v

    @staticmethod
    def weyl_toeplitz_symmetrization(matrix: np.ndarray) -> np.ndarray:
        r"""Proyección de Weyl–Toeplitz / Higham al cono hermítico: (M + M†)/2."""
        a = _NumericalCore.assert_square("weyl_toeplitz_symmetrization", matrix)
        return 0.5 * (a + a.T.conj())

    @staticmethod
    def skew_symmetrize(matrix: np.ndarray) -> np.ndarray:
        a = _NumericalCore.assert_square("skew_symmetrize", matrix)
        return 0.5 * (a - a.T)

    @staticmethod
    def compensated_trace(matrix: np.ndarray) -> float:
        """Traza real por KBN sobre la diagonal (Re Tr A)."""
        a = _NumericalCore.assert_square("compensated_trace", matrix)
        return _NumericalCore.kahan_babuska_neumaier_sum(np.real(np.diag(a)))

    @staticmethod
    def compensated_complex_trace(matrix: np.ndarray) -> complex:
        """Traza compleja: KBN(Re diag) + i KBN(Im diag)."""
        a = _NumericalCore.assert_square("compensated_complex_trace", matrix)
        diag = np.diag(a)
        re = _NumericalCore.kahan_babuska_neumaier_sum(np.real(diag))
        im = _NumericalCore.kahan_babuska_neumaier_sum(np.imag(diag))
        return complex(re, im)

    @staticmethod
    def hermitian_residual(matrix: np.ndarray) -> float:
        """‖A − A†‖_F (cero sii A es hermítica)."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a - a.T.conj())

    @staticmethod
    def skew_residual(matrix: np.ndarray) -> float:
        """‖A + Aᵀ‖_F (cero sii A es antisimétrica real)."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a + a.T)

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        r"""Piso de deflación adaptativo: ε_W = max(‖A‖_F · ε_mach · 10, ε_W)."""
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro_norm = _NumericalCore.frobenius_norm(matrix)
        return float(
            max(
                fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
                _WILKINSON_DEFLATION_FLOOR,
            )
        )

    @staticmethod
    def relative_residual(num: float, den: float, abs_floor: float = _MACHINE_EPS) -> float:
        """Residuo mixto |num| / max(|den|, floor)."""
        return float(abs(num) / max(abs(den), abs_floor))

    @staticmethod
    def higham_nearest_density(
        matrix: np.ndarray,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Estado densidad más próximo en ‖·‖_F (Higham + renormalización)."""
        herm = _NumericalCore.weyl_toeplitz_symmetrization(matrix)
        evals, evecs = la.eigh(herm)
        evals = np.maximum(np.real(evals), 0.0)
        evals[evals < float(floor)] = 0.0
        tr = _NumericalCore.kahan_babuska_neumaier_sum(evals)
        if tr > _MACHINE_EPS:
            evals = evals / tr
        else:
            evals = np.zeros_like(evals)
            evals[-1] = 1.0
        rho = evecs @ (evals[:, None] * evecs.T.conj())
        rho = _NumericalCore.weyl_toeplitz_symmetrization(rho)
        return rho, evals

    @staticmethod
    def compose_if_able(left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
        """Producto left @ right si las dimensiones encajan; si no, None."""
        if np.asarray(left).shape[1] != np.asarray(right).shape[0]:
            return None
        return np.asarray(left) @ np.asarray(right)

    # ── I.3 Gradiente CSMD y corchete de Poisson ─────────────────────────
    @staticmethod
    def compute_gradient_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Gradiente CSMD:
            ∇_k H(x) = Im[H(x + j·h·e_k)] / h + O(h²),
        eludiendo cancelaciones sustractivas en la mantisa de la FPU.
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        if not np.isfinite(h) or h == 0.0:
            raise ValueError("El paso CSMD h debe ser finito y no nulo.")
        h = float(abs(h))
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        holomorphic = False
        try:
            probe = func(xv.astype(np.complex128))
            holomorphic = np.isfinite(np.real(probe)) or np.isfinite(np.imag(probe))
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False
        if holomorphic:
            for i in range(dim):
                xp = xv.astype(np.complex128)
                xp[i] += 1j * h
                try:
                    val = func(xp)
                    imag = float(np.imag(val))
                except Exception:
                    holomorphic = False
                    break
                if not np.isfinite(imag):
                    holomorphic = False
                    break
                grad[i] = imag / h
        if not holomorphic:
            for i in range(dim):
                xp = xv.copy()
                xm = xv.copy()
                xp[i] += h
                xm[i] -= h
                try:
                    fp = float(np.real(func(xp)))
                    fm = float(np.real(func(xm)))
                except Exception:
                    fp = fm = 0.0
                grad[i] = (fp - fm) / (2.0 * h)
        return grad

    @staticmethod
    def poisson_bracket(
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        r"""
        Corchete de Poisson {H₀, H₁}(x) = (∇H₀)ᵀ Ω ∇H₁ en T*Q.

        Integrando de Melnikov: si {H₀, H₁} ≠ 0, ε H₁ rompe las integrales
        primeras de H₀ y puede producir caos homoclínico.
        """
        xv = _NumericalCore.assert_vec("x", x)
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("El corchete de Poisson exige dim par (Darboux).")
        grad0 = _NumericalCore.compute_gradient_csmd(hamiltonian_0, xv, h)
        grad1 = _NumericalCore.compute_gradient_csmd(hamiltonian_1, xv, h)
        return float(grad0 @ omega @ grad1)

    # ── I.4 Métrica conforme de Maupertuis–Jacobi ────────────────────────
    @staticmethod
    def compute_maupertuis_jacobi_conformal_metric(
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: np.ndarray,
    ) -> Tuple[np.ndarray, _MaupertuisJacobiGerm]:
        r"""
        Métrica conforme de Maupertuis–Jacobi:
            g̃_{jk}(q) = 2 (H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q),
        n(q) = √(2(H₀ − V(q))). Región de Hill: H₀ − V(q) > 0.
        """
        g = _NumericalCore.assert_square("base_metric_g", base_metric_g)
        h0 = float(hamiltonian_energy_H0)
        v = float(potential_energy_V)
        if not (np.isfinite(h0) and np.isfinite(v)):
            raise ValueError("H₀ y V deben ser finitos.")
        free_energy = 2.0 * (h0 - v)
        in_hill = bool(free_energy > _HILL_MARGIN_FLOOR)
        phi = max(free_energy, _WILKINSON_DEFLATION_FLOOR)
        refractive = float(np.sqrt(phi))
        gt = phi * g
        try:
            evals = la.eigvalsh(_NumericalCore.weyl_toeplitz_symmetrization(gt))
            min_eig = float(np.min(np.real(evals))) if evals.size else 0.0
        except la.LinAlgError:
            min_eig = 0.0
        germ = _MaupertuisJacobiGerm(
            conformal_factor=float(phi),
            refractive_index=refractive,
            hill_margin=float(h0 - v),
            is_in_hill_region=in_hill,
            min_eigenvalue=min_eig,
            is_positive_definite=bool(min_eig > _WILKINSON_DEFLATION_FLOOR),
        )
        return gt, germ

    @staticmethod
    def compute_hill_region_margin(
        potential_V: float,
        total_energy_H0: float,
    ) -> float:
        """Margen de Hill: H₀ − V(q)."""
        return float(total_energy_H0 - potential_V)

    @staticmethod
    def compute_christoffel_conformal_symbols(
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> np.ndarray:
        r"""
        Símbolos de Christoffel conformes de Koszul–Levi-Civita.

        Para g̃ = e^{2φ} g con φ = ln n = ½ ln(2(H₀ − V)):
            Γ̃^i_{jk} = Γ^i_{jk} + δ^i_j ∂_k φ + δ^i_k ∂_j φ − g_{jk} g^{il} ∂_l φ.
        ∇φ = −∇V / (2(H₀ − V)). Para g = I, Γ_base = 0.
        """
        grad_V = np.asarray(grad_V, dtype=np.float64).ravel()
        g = _NumericalCore.assert_square("g_base_metric", g_base_metric)
        n_dim = grad_V.size
        if g.shape[0] != n_dim:
            raise ValueError(f"g_base_metric debe ser {n_dim}×{n_dim}.")
        headroom = 2.0 * (total_energy_H0 - potential_V)
        if headroom <= _HILL_MARGIN_FLOOR:
            raise ValueError(
                "[PRETORIO_VETO] Cero energía cinética: invasión de pozo de potencial."
            )
        grad_phi = -grad_V / (headroom + _WILKINSON_DEFLATION_FLOOR)
        g_inv = la.inv(g)
        christoffel = np.zeros((n_dim, n_dim, n_dim), dtype=np.float64)
        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_dim):
                    term1 = (1.0 if i == j else 0.0) * grad_phi[k]
                    term2 = (1.0 if i == k else 0.0) * grad_phi[j]
                    term3 = g[j, k] * float(
                        _NumericalCore.kahan_babuska_neumaier_sum(g_inv[i, :] * grad_phi)
                    )
                    christoffel[i, j, k] = term1 + term2 - term3
        return christoffel

    # ── I.5 1-forma de Poincaré–Cartan ───────────────────────────────────
    @staticmethod
    def compute_poincare_cartan_lambda(
        x: np.ndarray,
        hamiltonian_value: float = 0.0,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, _PoincareCartanGerm]:
        r"""
        1-forma de Poincaré–Cartan λ = p dq − H dt evaluada en x ∈ T*Q.

        Retorna el vector de contacto (p, −H) ∈ ℝ^{n+1} y el gérmen con el
        residuo antisimétrico de Ω. dλ = ω − dH ∧ dt.
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("λ de Poincaré–Cartan exige dim par (T*Q).")
        n = dim // 2
        p = xv[n:]
        qdot = xv[n:]  # analogía euclídea q̇ ∼ p
        h_val = float(hamiltonian_value) if np.isfinite(hamiltonian_value) else 0.0
        lam = np.concatenate([p, np.array([-h_val], dtype=np.float64)])
        if omega is None:
            omega = np.zeros((dim, dim), dtype=np.float64)
        skew_res = _NumericalCore.skew_residual(omega) if omega.size else 0.0
        cartan_L = float(_NumericalCore.kahan_babuska_neumaier_sum(p * qdot) - h_val)
        germ = _PoincareCartanGerm(
            lambda_vector=lam,
            hamiltonian_value=h_val,
            dim=dim,
            two_n=dim,
            symplectic_skew_residual=float(skew_res),
            cartan_lagrangian=cartan_L,
            darboux_ok=bool(skew_res <= _STRUCTURE_ATOL * max(_NumericalCore.frobenius_norm(omega), 1.0)),
        )
        return lam, germ

    # ── I.6 Integrador simpléctico Störmer–Verlet ────────────────────────
    @staticmethod
    def stormer_verlet_step(
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Paso Störmer–Verlet sobre H(q, p) = ½|p|²/m + V(q):
            p_{n+½} = p_n − (dt/2) ∇V(q_n)
            q_{n+1} = q_n + dt · mass_inv · p_{n+½}
            p_{n+1} = p_{n+½} − (dt/2) ∇V(q_{n+1})
        Preserva la 2-forma de Liouville Ω (mapa simpléctico).
        """
        qn = np.asarray(q, dtype=np.float64)
        pn = np.asarray(p, dtype=np.float64)
        p_half = pn - 0.5 * dt * np.asarray(grad_v(qn), dtype=np.float64)
        q_next = qn + dt * mass_inv * p_half
        p_next = p_half - 0.5 * dt * np.asarray(grad_v(q_next), dtype=np.float64)
        return q_next, p_next

    # ── I.7 Generación de la 2-forma canónica ────────────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        """2-forma canónica de Liouville Ω ∈ ℝ^{dim×dim}, dim = 2n par."""
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio simpléctico dim={dim} debe ser par y positiva."
            )
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def darboux_residuals(omega: np.ndarray) -> Tuple[float, float]:
        r"""
        Residuos de Darboux / casi-complejidad:
            skew = ‖Ω + Ωᵀ‖_F,   ac = ‖Ω² + I‖_F.
        Para la forma canónica ambos son 0 (Ωᵀ = −Ω, Ω² = −I).
        """
        om = np.asarray(omega, dtype=np.float64)
        if om.ndim != 2 or om.shape[0] != om.shape[1]:
            return float("inf"), float("inf")
        skew = _NumericalCore.frobenius_norm(om + om.T)
        ac = _NumericalCore.frobenius_norm(om @ om + np.eye(om.shape[0]))
        return float(skew), float(ac)

    # ── I.8 Gérmen del bicomplejo Čech–de Rham ───────────────────────────
    @staticmethod
    def synthesize_hypercohomology_germ(
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
    ) -> _HypercohomologyGerm:
        r"""Ensambla 𝒢 = (δ, d, fact(δ², d², δd, dδ), Δ_D, ε_W)."""
        d1 = _NumericalCore.assert_matrix("cech_boundary_d1", cech_boundary_d1)
        d2 = _NumericalCore.assert_matrix("derham_boundary_d2", derham_boundary_d2)
        d1 = np.asarray(d1, dtype=np.complex128)
        d2 = np.asarray(d2, dtype=np.complex128)
        d1_self = d1.shape[0] == d1.shape[1]
        d2_self = d2.shape[0] == d2.shape[1]
        composition_lr = d1.shape[1] == d2.shape[0]
        composition_rl = d2.shape[1] == d1.shape[0]
        floor = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        floor = max(
            floor,
            _NumericalCore.wilkinson_deflation_floor(d1),
            _NumericalCore.wilkinson_deflation_floor(d2),
        )
        hodge: Optional[np.ndarray] = None
        if d1_self and d2_self and d1.shape == d2.shape:
            total = d1 + d2
            adj = total.T.conj()
            hodge = adj @ total + total @ adj
            hodge = _NumericalCore.weyl_toeplitz_symmetrization(hodge)
        return _HypercohomologyGerm(
            d1=d1,
            d2=d2,
            d1_square=bool(d1_self),
            d2_square=bool(d2_self),
            composition_lr=bool(composition_lr),
            composition_rl=bool(composition_rl),
            hodge_laplacian=hodge,
            reg_floor=float(floor),
            fro_d1=_NumericalCore.frobenius_norm(d1),
            fro_d2=_NumericalCore.frobenius_norm(d2),
        )


# =============================================================================
# NÚCLEO CELESTE DE POINCARÉ (Cartan, Christoffel, CZ, Birkhoff, homológica)
# =============================================================================
class _PoincareCelestialKernel:
    r"""
    Operaciones de la Mécanique Céleste de Poincaré usadas por las Fases I–II.

    Implementa, con cotas de Wilkinson y sumas de Neumaier/KBN:
      • escisión Darboux (q, p) de T*Q,
      • λ(X) = p·q̇ − H y circulación discreta,
      • defecto de pullback simpléctico Φ*ω − ω,
      • intensidad ‖Γ̃‖_F y torsión de Koszul,
      • marea de Jacobi de la métrica de Maupertuis,
      • índice de Conley–Zehnder (Robbin–Salamon),
      • número de rotación de Poincaré y twist de Birkhoff,
      • residual homológico de Lindstedt–Poincaré,
      • transversalidad de la sección |det(M − I)|.
    """

    @staticmethod
    def split_qp(x: np.ndarray, two_n: int) -> Tuple[np.ndarray, np.ndarray]:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != two_n or two_n % 2 != 0:
            raise ValueError(f"x debe vivir en R^{two_n} con two_n par.")
        half = two_n // 2
        return vec[:half].copy(), vec[half:].copy()

    @staticmethod
    def poincare_cartan_on_field(
        p: np.ndarray,
        qdot: np.ndarray,
        hamiltonian: float,
    ) -> float:
        r"""λ(X) = p_i q̇^i − H."""
        p_v = np.asarray(p, dtype=np.float64).reshape(-1)
        qd = np.asarray(qdot, dtype=np.float64).reshape(-1)
        if p_v.size != qd.size:
            raise ValueError("p y q̇ deben tener la misma dimensión de Q.")
        pairing = _NumericalCore.kahan_babuska_neumaier_sum(p_v * qd)
        return float(pairing - float(hamiltonian))

    @staticmethod
    def poincare_cartan_circulation_step(
        q0: np.ndarray,
        p0: np.ndarray,
        q1: np.ndarray,
        p1: np.ndarray,
        hamiltonian: float,
        dt: float,
    ) -> float:
        r"""Circulación discreta ∫_γ λ ≈ p̄·Δq − H Δt (punto medio)."""
        p_mid = 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))
        dq = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
        return float(
            _NumericalCore.kahan_babuska_neumaier_sum(p_mid * dq)
            - float(hamiltonian) * float(dt)
        )

    @staticmethod
    def symplectic_pullback_defect(
        monodromy_M: np.ndarray,
        canonical_omega: np.ndarray,
    ) -> float:
        r"""
        Defecto del invariante integral absoluto de Poincaré:
            δ = ‖Mᵀ Ω M − Ω‖_F .
        Vale 0 sii M ∈ Sp(2n, ℝ) (en aritmética exacta).
        """
        M = np.asarray(monodromy_M, dtype=np.float64)
        Om = np.asarray(canonical_omega, dtype=np.float64)
        if M.shape != Om.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
            return float("inf")
        residual = M.T @ Om @ M - Om
        return _NumericalCore.frobenius_norm(residual)

    @staticmethod
    def koszul_torsion(christoffel: np.ndarray) -> float:
        r"""Torsión de Koszul T^i_{jk} = Γ^i_{jk} − Γ^i_{kj}. Levi-Civita ⇒ T ≡ 0."""
        G = np.asarray(christoffel, dtype=np.float64)
        if G.ndim != 3 or G.shape[0] != G.shape[1] or G.shape[1] != G.shape[2]:
            return float("inf")
        return _NumericalCore.frobenius_norm((G - np.swapaxes(G, 1, 2)).reshape(G.shape[0], -1))

    @classmethod
    def christoffel_conformal_strength(
        cls,
        refractive_index: float,
        grad_V: np.ndarray,
        hill_margin: float,
    ) -> float:
        r"""
        Intensidad ‖Γ̃‖_F de la conexión conforme euclídea.
        ∇ln n = −∇V / (2(H₀ − V)). Blow-up cerca de ∂D_H.
        """
        n_idx = float(refractive_index)
        margin = float(hill_margin)
        if (
            (not np.isfinite(n_idx))
            or n_idx <= _HILL_MARGIN_FLOOR
            or margin <= _HILL_MARGIN_FLOOR
        ):
            return float("inf")
        gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
        dln = -gV / (2.0 * margin)
        if not np.all(np.isfinite(dln)):
            return float("inf")
        d = int(dln.size)
        gamma = np.zeros((d, d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    term = 0.0
                    if i == j:
                        term += dln[k]
                    if i == k:
                        term += dln[j]
                    if j == k:
                        term -= dln[i]
                    gamma[i, j, k] = term
        return _NumericalCore.frobenius_norm(gamma.reshape(d, -1))

    @staticmethod
    def jacobi_tidal_norm(grad_V: np.ndarray, hill_margin: float) -> float:
        r"""
        Proxy de marea geodésica de Jacobi sobre Maupertuis–Jacobi.
        Curvatura seccional óptica ~ ‖∇V‖² / (H₀ − V)²; usamos ‖∇V‖² / (H₀ − V).
        """
        gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
        if gV.size == 0 or (not np.all(np.isfinite(gV))):
            return float("inf")
        num = float(_NumericalCore.kahan_babuska_neumaier_sum(gV * gV))
        den = max(abs(float(hill_margin)), _HILL_MARGIN_FLOOR)
        return float(num / den)

    @staticmethod
    def conley_zehnder_index(
        monodromy_M: np.ndarray,
        atol: float = _CZ_DEGENERATE_ATOL,
    ) -> Tuple[int, bool, float]:
        r"""
        Índice de Conley–Zehnder del camino recto γ(t) = exp(t Log M), t∈[0,1],
        convención de Robbin–Salamon / Long:
            i_CZ(M) ≈ n + (1/π) ∑_i arg(λ_i),
        n = dim Q. Degeneración ⇔ 1 ∈ spec(M).
        Retorna (indice, es_no_degenerado, dist(spec, {1})).
        """
        M = np.asarray(monodromy_M)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2 != 0:
            return 0, False, 0.0
        n_half = M.shape[0] // 2
        try:
            eigs = la.eigvals(M)
        except (np.linalg.LinAlgError, ValueError):
            return 0, False, 0.0
        dist_one = float(np.min(np.abs(eigs - 1.0))) if eigs.size else 0.0
        nondeg = bool(dist_one > atol)
        args = np.angle(eigs)
        raw = float(n_half) + float(_NumericalCore.kahan_babuska_neumaier_sum(args)) / float(np.pi)
        if not np.isfinite(raw):
            return 0, False, dist_one
        return int(np.rint(raw)), nondeg, dist_one

    @staticmethod
    def rotation_number_and_birkhoff_twist(
        floquet_multipliers: np.ndarray,
        unit_atol: float = _UNIT_CIRCLE_ATOL,
    ) -> Tuple[float, float, int]:
        r"""
        Número de rotación de Poincaré y twist de Poincaré–Birkhoff.

        Para λ = e^{iθ}: ρ = mean |θ| / 2π ∈ [0, 1/2], twist = max ρ − min ρ.
        El teorema geométrico último exige twist ≠ 0 y preservación de área.
        """
        floq = np.asarray(floquet_multipliers, dtype=np.complex128).reshape(-1)
        if floq.size == 0:
            return float("nan"), 0.0, 0
        on_circle = floq[np.abs(np.abs(floq) - 1.0) <= unit_atol]
        if on_circle.size == 0:
            return float("nan"), 0.0, 0
        rhos = np.abs(np.angle(on_circle)) / (2.0 * np.pi)
        rho_mean = float(np.mean(rhos))
        twist = float(np.max(rhos) - np.min(rhos)) if rhos.size else 0.0
        return rho_mean, twist, int(on_circle.size)

    @staticmethod
    def homological_lindstedt_residual(min_divisor: float) -> float:
        r"""Residual de i⟨k,ω⟩ χ_k = (H₁)_k  ⇒  |χ_k| ∼ 1 / |⟨k,ω⟩|."""
        if (not np.isfinite(min_divisor)) or min_divisor <= _MACHINE_EPS:
            return float("inf")
        return float(min(1.0 / abs(min_divisor), _HOMOLOGICAL_KAM_CEILING))

    @staticmethod
    def section_transversality(monodromy_M: np.ndarray) -> float:
        r"""Transversalidad de Σ: |det(M − I)|. Cero ⇒ mapa de retorno no definido."""
        M = np.asarray(monodromy_M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            return 0.0
        try:
            return float(abs(np.linalg.det(M - np.eye(M.shape[0]))))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def classify_floquet(
        magnitudes: np.ndarray,
        band: float = _FLOQUET_PARABOLIC_BAND,
    ) -> Tuple[bool, bool, bool, float]:
        r"""
        Clasificación mutuamente excluyente (v5):
          • elíptico  : max_i ||λ_i| − 1| ≤ band,
          • parabólico: no elíptico y min_i ||λ_i| − 1| ≤ band,
          • hiperbólico: min_i ||λ_i| − 1| > band.
        En v4 is_elliptic e is_hyperbolic podían ser True a la vez.
        """
        mag = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
        if mag.size == 0:
            return False, False, False, float("inf")
        dev = np.abs(mag - 1.0)
        floq_par = float(np.max(dev))
        is_elliptic = bool(floq_par <= band)
        is_parabolic = bool((not is_elliptic) and (float(np.min(dev)) <= band))
        is_hyperbolic = bool((not is_elliptic) and (not is_parabolic))
        return is_hyperbolic, is_elliptic, is_parabolic, floq_par


class _NumericalCoreCelestial(_NumericalCore):
    r"""
    Extensión de Fase I: paso Maupertuis–Verlet y ensamblaje del gérmen 𝒢_I.

    Se define como subclase para preservar el topos lineal de `_NumericalCore`
    y añadir el morfismo terminal I.10 sin romper la API estática.
    """

    # ── I.9 Paso simpléctico de Maupertuis–Jacobi ────────────────────────
    @classmethod
    def integrate_symplectic_maupertuis_step(
        cls,
        x_state: np.ndarray,
        dt_step: float,
        g_base_metric: np.ndarray,
        potential_V: float,
        grad_V: np.ndarray,
        total_energy_H0: float,
        mass_inv: float = 1.0,
    ) -> MaupertuisStepReport:
        r"""
        Un paso Störmer–Verlet acoplado a la geometría conforme:
          • n(q), g̃, región de Hill, Christoffel, marea de Jacobi,
          • circulación de Poincaré–Cartan,
          • drift de Liouville |det DΦ − 1| por linealización FD.
        """
        xv = cls.assert_vec("x_state", np.asarray(x_state, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("x_state debe tener dimensión par (T*Q).")
        g = cls.assert_square("g_base_metric", g_base_metric)
        gV = cls.assert_vec("grad_V", np.asarray(grad_V, dtype=np.float64))
        half = dim // 2
        if g.shape[0] != half or gV.size != half:
            raise ValueError("g y ∇V deben vivir en Q (dim = n = two_n/2).")
        dt = float(dt_step)
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("dt_step debe ser finito y no nulo.")

        q, p = _PoincareCelestialKernel.split_qp(xv, dim)
        gt, mau = cls.compute_maupertuis_jacobi_conformal_metric(
            total_energy_H0, potential_V, g
        )
        n_idx = float(mau.refractive_index)
        hill = float(mau.hill_margin)

        gV_const = np.asarray(gV, dtype=np.float64)

        def _grad_fn(_q: np.ndarray) -> np.ndarray:
            return gV_const

        q1, p1 = cls.stormer_verlet_step(q, p, _grad_fn, float(mass_inv), dt)
        x_next = np.concatenate([q1, p1])

        # Linealización FD del mapa Φ: (q,p) ↦ (q',p') para det DΦ
        two_n = dim
        jac = np.zeros((two_n, two_n), dtype=np.float64)
        h_fd = max(np.sqrt(_MACHINE_EPS), abs(dt) * 1.0e-6)
        for k in range(two_n):
            xp = xv.copy()
            xm = xv.copy()
            xp[k] += h_fd
            xm[k] -= h_fd
            qp, pp = _PoincareCelestialKernel.split_qp(xp, two_n)
            qm, pm = _PoincareCelestialKernel.split_qp(xm, two_n)
            qpp, ppp = cls.stormer_verlet_step(qp, pp, _grad_fn, float(mass_inv), dt)
            qmm, pmm = cls.stormer_verlet_step(qm, pm, _grad_fn, float(mass_inv), dt)
            fp = np.concatenate([qpp, ppp])
            fm = np.concatenate([qmm, pmm])
            jac[:, k] = (fp - fm) / (2.0 * h_fd)
        try:
            det_j = float(np.real(np.linalg.det(jac)))
        except (np.linalg.LinAlgError, ValueError):
            det_j = float("nan")
        volume_drift = float(abs(det_j - 1.0)) if np.isfinite(det_j) else float("inf")
        is_symp = bool(np.isfinite(volume_drift) and volume_drift <= _WILKINSON_DRIFT_LIMIT)

        qdot = p * float(mass_inv)
        action = float(n_idx * cls.euclidean_norm(qdot))
        H_step = 0.5 * float(mass_inv) * float(cls.kahan_babuska_neumaier_sum(p * p)) + float(
            potential_V
        )
        try:
            Gamma = cls.compute_christoffel_conformal_symbols(
                gV, potential_V, total_energy_H0, g
            )
            torsion = _PoincareCelestialKernel.koszul_torsion(Gamma)
            strength = _NumericalCore.frobenius_norm(Gamma.reshape(Gamma.shape[0], -1))
        except ValueError:
            torsion = float("inf")
            strength = float("inf")
        tidal = _PoincareCelestialKernel.jacobi_tidal_norm(gV, hill)
        circ = _PoincareCelestialKernel.poincare_cartan_circulation_step(
            q, p, q1, p1, H_step, dt
        )
        return MaupertuisStepReport(
            refractive_index_n=n_idx,
            maupertuis_action_density=action,
            volume_drift_det=volume_drift,
            is_symplectic_coherent=is_symp,
            hamiltonian_energy=float(H_step),
            hill_margin=hill,
            christoffel_strength=float(strength),
            jacobi_tidal_norm=float(tidal),
            cartan_circulation=float(circ),
            koszul_torsion=float(torsion),
            x_next=x_next,
        )

    # ── I.10 MORFISMO TERMINAL Φ_I: gérmen celeste de Poincaré ───────────
    @staticmethod
    def synthesize_pretorio_celestial_germ(
        dimension_two_n: int,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        base_metric_g: Optional[np.ndarray] = None,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        grad_V: Optional[np.ndarray] = None,
    ) -> _PretorioCelestialGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE I ≅ OBJETO INICIAL DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Ensambla 𝒢_I = (Ω, g̃, λ, H₀, 𝒢_hyper, ε_W, Darboux) que la Fase II
        recibe para instanciar KAM, Melnikov, Poincaré–Birkhoff y retorno.

        Firma de continuación (Fase II):
            lift_from_celestial_germ(germ: _PretorioCelestialGerm)
                -> _PoincareCelestialAuditor
        """
        dim = int(dimension_two_n)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dimension_two_n debe ser par y positivo; recibido {dim}.")
        n = dim // 2
        omega = _NumericalCore.generate_canonical_symplectic_form(dim)
        skew_r, ac_r = _NumericalCore.darboux_residuals(omega)
        if base_metric_g is None:
            base_metric_g = np.eye(n, dtype=np.float64)
        gt, maupertuis_germ = _NumericalCore.compute_maupertuis_jacobi_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, base_metric_g
        )
        if grad_V is not None:
            gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
            strength = _PoincareCelestialKernel.christoffel_conformal_strength(
                maupertuis_germ.refractive_index, gV, maupertuis_germ.hill_margin
            )
            tidal = _PoincareCelestialKernel.jacobi_tidal_norm(
                gV, maupertuis_germ.hill_margin
            )
            try:
                Gamma = _NumericalCore.compute_christoffel_conformal_symbols(
                    gV, potential_energy_V, hamiltonian_energy_H0, base_metric_g
                )
                torsion = _PoincareCelestialKernel.koszul_torsion(Gamma)
            except ValueError:
                torsion = float("inf")
            maupertuis_germ = _MaupertuisJacobiGerm(
                conformal_factor=maupertuis_germ.conformal_factor,
                refractive_index=maupertuis_germ.refractive_index,
                hill_margin=maupertuis_germ.hill_margin,
                is_in_hill_region=maupertuis_germ.is_in_hill_region,
                min_eigenvalue=maupertuis_germ.min_eigenvalue,
                is_positive_definite=maupertuis_germ.is_positive_definite,
                christoffel_strength=float(strength),
                jacobi_tidal_norm=float(tidal),
                koszul_torsion=float(torsion),
            )
        x0 = np.zeros(dim, dtype=np.float64)
        lam, cartan_germ = _NumericalCore.compute_poincare_cartan_lambda(
            x0, hamiltonian_value=potential_energy_V, omega=omega
        )
        zero = np.zeros((n, n), dtype=np.complex128)
        hyper_germ = _NumericalCore.synthesize_hypercohomology_germ(
            zero, zero, regularizer=regularizer
        )
        return _PretorioCelestialGerm(
            two_n=dim,
            n=n,
            omega=omega,
            maupertuis_germ=maupertuis_germ,
            poincare_cartan_germ=cartan_germ,
            hyper_germ=hyper_germ,
            reg_floor=float(max(regularizer, _HIGHAM_TIKHONOV_REG)),
            darboux_residual=float(skew_r),
            almost_complex_residual=float(ac_r),
        )


# Re-export estático para que Φ_I viva en el núcleo numérico
_NumericalCore.synthesize_pretorio_celestial_germ = (  # type: ignore[attr-defined]
    staticmethod(_NumericalCoreCelestial.synthesize_pretorio_celestial_germ)
)
_NumericalCore.integrate_symplectic_maupertuis_step = (  # type: ignore[attr-defined]
    classmethod(_NumericalCoreCelestial.integrate_symplectic_maupertuis_step.__func__)  # type: ignore[attr-defined]
)


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE II — HIPERCOHOMOLOGÍA, BROUWER, KAM, MELNIKOV, BIRKHOFF Y RETORNO  ║
# ║                                                                          ║
# ║ Continuación formal de I.10: el primer morfismo consume exactamente      ║
# ║ 𝒢_I = _PretorioCelestialGerm.                                            ║
# ║                                                                          ║
# ║ Morfismo terminal (II.9): induce_ultrafilter_celestial_germ              ║
# ║     ↦ 𝒢_II = _UltrafilterCelestialGerm (inicial de Fase III).           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True)
class _HypercohomologyResult:
    """Resultado certificado de la nilpotencia D² ≡ 0."""

    residual: float
    wilkinson_limit: float
    verdict: str
    d1_squared_residual: float
    d2_squared_residual: float
    anticommutator_residual: float
    commutator_residual: float
    total_d2_residual: float
    relative_residual: float
    betti_0: int
    hodge_kernel_mass: float
    shapes_compatible: bool


@dataclass(frozen=True)
class _BrouwerResult:
    """Resultado certificado del punto fijo de Brouwer en 𝒟(ℋ)."""

    brouwer_residual: float
    trace_residual: float
    verdict: str
    hs_relative: float
    min_eigenvalue: float
    purity: float
    hermiticity_residual: float
    positivity_ok: bool
    lipschitz_hint: float
    projected_residual: float


@dataclass(frozen=True)
class _UltrafilterGerm:
    """Gérmen de Heyting (objeto terminal de la Fase II clásica)."""

    heyting_verdicts: Tuple[str, ...]
    godel_values: np.ndarray
    hyper_verdict: str
    brouwer_verdict: str
    hyper_residual: float
    brouwer_residual: float
    source: str


class _HypercohomologyChecker:
    """Fase II. Nilpotencia del bicomplejo Čech–de Rham."""

    def __init__(
        self,
        germ: Optional[_HypercohomologyGerm] = None,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
    ) -> None:
        self._germ = germ
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)

    @property
    def germ(self) -> Optional[_HypercohomologyGerm]:
        return self._germ

    def _resolve(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> _HypercohomologyGerm:
        germ = _NumericalCore.synthesize_hypercohomology_germ(
            cech_boundary_d1, derham_boundary_d2, regularizer=self._reg
        )
        self._germ = germ
        return germ

    @staticmethod
    def _verdict(residual: float, limit: float) -> str:
        if not np.isfinite(residual):
            return "VETOED"
        if residual > limit * _HYPER_VETO_SCALE:
            return "VETOED"
        if residual > limit:
            return "DEGRADED"
        return "COHERENT"

    def verify(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> _HypercohomologyResult:
        """Certifica D² ≡ 0."""
        try:
            germ = self._resolve(cech_boundary_d1, derham_boundary_d2)
        except ValueError as exc:
            logger.error("Dimensiones / datos inválidos en hipercohomología: %s", exc)
            return _HypercohomologyResult(
                residual=float("inf"),
                wilkinson_limit=0.0,
                verdict="VETOED",
                d1_squared_residual=float("inf"),
                d2_squared_residual=float("inf"),
                anticommutator_residual=float("inf"),
                commutator_residual=float("inf"),
                total_d2_residual=float("inf"),
                relative_residual=float("inf"),
                betti_0=0,
                hodge_kernel_mass=0.0,
                shapes_compatible=False,
            )
        d1, d2 = germ.d1, germ.d2
        norm_factor = float(germ.fro_d1 * germ.fro_d2)
        wilkinson_limit = max(
            norm_factor * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
            germ.reg_floor,
            _WILKINSON_DEFLATION_FLOOR,
        )
        if not germ.composition_lr:
            logger.error(
                "Dimensiones incompatibles en hipercohomología: %s @ %s",
                d1.shape,
                d2.shape,
            )
            return _HypercohomologyResult(
                residual=float("inf"),
                wilkinson_limit=float(wilkinson_limit),
                verdict="VETOED",
                d1_squared_residual=float("nan"),
                d2_squared_residual=float("nan"),
                anticommutator_residual=float("nan"),
                commutator_residual=float("nan"),
                total_d2_residual=float("nan"),
                relative_residual=float("inf"),
                betti_0=0,
                hodge_kernel_mass=0.0,
                shapes_compatible=False,
            )
        composition = d1 @ d2
        residual = _NumericalCore.frobenius_norm(composition)
        rel = _NumericalCore.relative_residual(residual, norm_factor, germ.reg_floor)
        d1_sq = _NumericalCore.frobenius_norm(d1 @ d1) if germ.d1_square else float("nan")
        d2_sq = _NumericalCore.frobenius_norm(d2 @ d2) if germ.d2_square else float("nan")
        if germ.composition_lr and germ.composition_rl:
            lr = d1 @ d2
            rl = d2 @ d1
            if lr.shape == rl.shape:
                anticomm = _NumericalCore.frobenius_norm(lr + rl)
                comm = _NumericalCore.frobenius_norm(lr - rl)
            else:
                anticomm = comm = float("nan")
        else:
            anticomm = comm = float("nan")
        if germ.d1_square and germ.d2_square and d1.shape == d2.shape:
            total = d1 + d2
            total_d2 = _NumericalCore.frobenius_norm(total @ total)
        else:
            total_d2 = float("nan")
        betti_0 = 0
        kernel_mass = 0.0
        if germ.hodge_laplacian is not None:
            evals = np.real(la.eigvalsh(germ.hodge_laplacian))
            ker_tol = max(germ.reg_floor, _WILKINSON_DEFLATION_FLOOR * max(evals.size, 1))
            mask = evals <= ker_tol
            betti_0 = int(np.sum(mask))
            if betti_0:
                kernel_mass = _NumericalCore.kahan_babuska_neumaier_sum(
                    np.clip(evals[mask], 0.0, None)
                )
        return _HypercohomologyResult(
            residual=float(residual),
            wilkinson_limit=float(wilkinson_limit),
            verdict=self._verdict(residual, wilkinson_limit),
            d1_squared_residual=float(d1_sq),
            d2_squared_residual=float(d2_sq),
            anticommutator_residual=float(anticomm),
            commutator_residual=float(comm),
            total_d2_residual=float(total_d2),
            relative_residual=float(rel),
            betti_0=int(betti_0),
            hodge_kernel_mass=float(kernel_mass),
            shapes_compatible=True,
        )


class _BrouwerChecker:
    """Fase II. Punto fijo de Brouwer en 𝒟(ℋ)."""

    def __init__(self, regularizer: float = _HIGHAM_TIKHONOV_REG) -> None:
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)

    @staticmethod
    def _verdict(hs: float, tr: float) -> str:
        if (not np.isfinite(hs)) or (not np.isfinite(tr)):
            return "VETOED"
        if tr > _BROUWER_VETO_TRACE or hs > _BROUWER_VETO_HS:
            return "VETOED"
        if tr > _BROUWER_DEGRADED_TRACE or hs > _BROUWER_DEGRADED_HS:
            return "DEGRADED"
        return "COHERENT"

    def verify(
        self,
        rho_current: np.ndarray,
        rho_transformed: np.ndarray,
    ) -> _BrouwerResult:
        """Certifica ρ = f(ρ) con Weyl–Toeplitz."""
        try:
            raw_1 = _NumericalCore.assert_square("rho_current", rho_current)
            raw_2 = _NumericalCore.assert_square("rho_transformed", rho_transformed)
            if raw_1.shape != raw_2.shape:
                raise ValueError(
                    f"ρ y T(ρ) deben compartir dimensión; {raw_1.shape} vs {raw_2.shape}."
                )
        except ValueError as exc:
            logger.error("Fallo en verificación de Brouwer: %s", exc)
            return _BrouwerResult(
                brouwer_residual=float("inf"),
                trace_residual=float("inf"),
                verdict="VETOED",
                hs_relative=float("inf"),
                min_eigenvalue=float("nan"),
                purity=float("nan"),
                hermiticity_residual=float("inf"),
                positivity_ok=False,
                lipschitz_hint=float("inf"),
                projected_residual=float("inf"),
            )
        rho_1 = _NumericalCore.weyl_toeplitz_symmetrization(raw_1)
        rho_2 = _NumericalCore.weyl_toeplitz_symmetrization(raw_2)
        trace_val = _NumericalCore.compensated_trace(rho_1)
        trace_residual = float(abs(trace_val - 1.0))
        brouwer_residual = _NumericalCore.frobenius_norm(rho_1 - rho_2)
        herm = max(
            _NumericalCore.hermitian_residual(raw_1),
            _NumericalCore.hermitian_residual(raw_2),
        )
        evals_1 = np.real(la.eigvalsh(rho_1))
        min_ev = float(np.min(evals_1)) if evals_1.size else 0.0
        positivity_ok = bool(min_ev >= -max(self._reg, _PSD_NEG_TOL))
        purity = _NumericalCore.kahan_babuska_neumaier_sum(evals_1 * evals_1)
        proj_1, _ = _NumericalCore.higham_nearest_density(raw_1, floor=self._reg)
        proj_2, _ = _NumericalCore.higham_nearest_density(raw_2, floor=self._reg)
        projected = _NumericalCore.frobenius_norm(proj_1 - proj_2)
        scale = max(_NumericalCore.frobenius_norm(rho_1), 1.0)
        hs_rel = float(brouwer_residual / scale)
        lipschitz = float(brouwer_residual / np.sqrt(2.0))
        verdict = self._verdict(brouwer_residual, trace_residual)
        if not positivity_ok and verdict == "COHERENT":
            verdict = "DEGRADED"
        return _BrouwerResult(
            brouwer_residual=float(brouwer_residual),
            trace_residual=float(trace_residual),
            verdict=verdict,
            hs_relative=hs_rel,
            min_eigenvalue=min_ev,
            purity=float(purity),
            hermiticity_residual=float(herm),
            positivity_ok=positivity_ok,
            lipschitz_hint=lipschitz,
            projected_residual=float(projected),
        )

    def induce_ultrafilter_germ(
        self,
        hyper: Optional[_HypercohomologyResult] = None,
        brouwer: Optional[_BrouwerResult] = None,
        extra_verdicts: Optional[Sequence[str]] = None,
    ) -> _UltrafilterGerm:
        """Empaqueta veredictos locales en el gérmen H₃ⁿ."""
        verdicts: List[str] = []
        hyper_v = "COHERENT"
        brouwer_v = "COHERENT"
        hyper_r = 0.0
        brouwer_r = 0.0
        if hyper is not None:
            hyper_v = str(hyper.verdict)
            hyper_r = float(hyper.residual)
            verdicts.append(hyper_v)
        if brouwer is not None:
            brouwer_v = str(brouwer.verdict)
            brouwer_r = float(brouwer.brouwer_residual)
            verdicts.append(brouwer_v)
        if extra_verdicts:
            verdicts.extend(str(v) for v in extra_verdicts)
        if not verdicts:
            verdicts = ["COHERENT"]
        canon = tuple(v if v in _HEYTING_SEVERITY else "VETOED" for v in verdicts)
        godel = np.array([_HEYTING_GODEL[v] for v in canon], dtype=np.float64)
        source = "+".join(
            s
            for s, flag in (
                ("hyper", hyper is not None),
                ("brouwer", brouwer is not None),
                ("extra", bool(extra_verdicts)),
            )
            if flag
        ) or "unit"
        return _UltrafilterGerm(
            heyting_verdicts=canon,
            godel_values=godel,
            hyper_verdict=hyper_v if hyper_v in _HEYTING_SEVERITY else "VETOED",
            brouwer_verdict=brouwer_v if brouwer_v in _HEYTING_SEVERITY else "VETOED",
            hyper_residual=hyper_r,
            brouwer_residual=brouwer_r,
            source=source,
        )


class _PoincareCelestialAuditor:
    r"""
    Fase II (continuación celeste). Auditor de mecánica celeste de Poincaré.

    CONTINUACIÓN FORMAL DE I.10:
        lift_from_celestial_germ(germ: _PretorioCelestialGerm) es el primer
        morfismo de esta fase y consume el objeto terminal de la Fase I.

    Expone:
      • Pequeños divisores de Poincaré–KAM con diofantinidad γ/|k|^τ.
      • Residual homológico de Lindstedt y pullback de Cartan.
      • Resonancias de Arnol'd (retícula k·ω ≈ 0).
      • Función de Melnikov M(t₀) = ∫ {H₀, H₁}(γ⁰(t − t₀)) dt.
      • Twist map de Poincaré–Birkhoff (giro opuesto + área + Φ*ω).
      • Mapa de retorno P: Σ → Σ (Floquet + Lyapunov + CZ + ρ).

    Morfismo terminal (II.9): induce_ultrafilter_celestial_germ → 𝒢_II.
    """

    def __init__(self, germ: _PretorioCelestialGerm) -> None:
        self._germ = germ
        self._kernel = _PoincareCelestialKernel

    @property
    def germ(self) -> _PretorioCelestialGerm:
        return self._germ

    @property
    def kernel(self) -> type[_PoincareCelestialKernel]:
        return self._kernel

    # ── II.0 Continuación de I.10: lifting del gérmen celeste ────────────
    @classmethod
    def lift_from_celestial_germ(
        cls, germ: _PretorioCelestialGerm
    ) -> "_PoincareCelestialAuditor":
        r"""
        Primer morfismo de la Fase II. Continúa I.10:
            synthesize_pretorio_celestial_germ(...) -> 𝒢_I
            lift_from_celestial_germ(𝒢_I)           -> auditor de Fase II
        """
        if not isinstance(germ, _PretorioCelestialGerm):
            raise TypeError("lift_from_celestial_germ exige _PretorioCelestialGerm.")
        return cls(germ)

    # ── II.1 Pequeños divisores de Poincaré–KAM ──────────────────────────
    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        canonical_J: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
        novikov_valuation_T: float = 1.0,
    ) -> _KAMAudit:
        r"""
        Espectro de pequeños divisores de Poincaré–KAM con absorción
        ultramétrica T-ádica en Λ_Nov, residual homológico de Lindstedt
        y defecto de pullback de Cartan.
        """
        try:
            omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega)
            wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
            jac_m = _NumericalCore.assert_square("jacobian_M", jacobian_M)
            canon = np.asarray(canonical_J, dtype=np.float64)
            if wave_k.ndim == 1:
                divisors = np.abs(np.dot(wave_k, omega))
                k_norms = np.abs(wave_k)
                min_divisor = float(divisors)
                argmin_idx: Optional[int] = 0
                resonance_gap = float(divisors)
            else:
                divisors = np.abs(wave_k @ omega)
                k_norms = np.linalg.norm(wave_k, axis=1)
                if divisors.size == 0:
                    min_divisor = 1.0
                    argmin_idx = None
                    resonance_gap = 1.0
                else:
                    argmin_idx = int(np.argmin(divisors))
                    min_divisor = float(divisors[argmin_idx])
                    resonance_gap = min_divisor
            novikov_weight = float(
                np.exp(
                    -np.clip(
                        novikov_valuation_T / (_WILKINSON_DRIFT_LIMIT + min_divisor),
                        0.0,
                        _LOG_EXP_CLIP,
                    )
                )
            )
            mc_residual = float(abs(min_divisor * novikov_weight))
            if argmin_idx is not None and wave_k.ndim > 1 and wave_k.size > 0:
                k_norm_min = float(max(k_norms[argmin_idx], 1.0))
                is_diophantine = bool(min_divisor * (k_norm_min ** tau) >= gamma)
            else:
                is_diophantine = bool(min_divisor >= _WILKINSON_DRIFT_LIMIT)
            det_M = float(np.real(la.det(jac_m))) if jac_m.size > 0 else 0.0
            volume_drift = float(abs(det_M - 1.0))
            cartan = self._kernel.symplectic_pullback_defect(jac_m, canon)
            homo = self._kernel.homological_lindstedt_residual(min_divisor)
            is_kam_stable = bool(
                (min_divisor >= _KAM_DIVISOR_FLOOR)
                and (volume_drift <= _WILKINSON_DRIFT_LIMIT)
                and is_diophantine
                and (not np.isfinite(cartan) or cartan <= _CARTAN_DEFECT_TOL)
            )
            return _KAMAudit(
                min_divisor=min_divisor,
                tau=float(tau),
                gamma=float(gamma),
                resonance_gap=float(resonance_gap),
                is_diophantine=is_diophantine,
                novikov_weight=novikov_weight,
                maurercartan_residual=mc_residual,
                volume_drift=volume_drift,
                is_kam_stable=is_kam_stable,
                engine_ok=True,
                homological_residual=float(homo),
                cartan_pullback_defect=float(cartan),
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_small_divisors_spectrum: %s", exc)
            return _KAMAudit(
                min_divisor=float("inf"),
                tau=float(tau),
                gamma=float(gamma),
                resonance_gap=float("inf"),
                is_diophantine=False,
                novikov_weight=0.0,
                maurercartan_residual=float("inf"),
                volume_drift=float("inf"),
                is_kam_stable=False,
                engine_ok=False,
                homological_residual=float("inf"),
                cartan_pullback_defect=float("inf"),
            )

    # ── II.2 Retícula de resonancias de Arnol'd ──────────────────────────
    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: np.ndarray,
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> np.ndarray:
        r"""
        Retícula de resonancias de Arnol'd:
            R_ε(ω) = { k ∈ ℤ^n \ {0} : |k|₁ ≤ max_order, |⟨k, ω⟩| < tol }.
        """
        omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega)
        n = omega.size
        if n == 0 or max_order < 1:
            return np.zeros((0, max(n, 1)), dtype=np.int64)
        ranges = [np.arange(-max_order, max_order + 1) for _ in range(n)]
        grid = np.meshgrid(*ranges, indexing="ij")
        k_all = np.stack([g.ravel() for g in grid], axis=1).astype(np.int64)
        norms = np.abs(k_all).sum(axis=1)
        keep = (norms > 0) & (norms <= max_order)
        if not np.any(keep):
            return np.zeros((0, n), dtype=np.int64)
        k_cand = k_all[keep]
        inner = np.abs(k_cand @ omega)
        return k_cand[inner < tol]

    # ── II.3 Función de Melnikov ─────────────────────────────────────────
    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = _MELNIKOV_T_INF,
        n_quad: int = _MELNIKOV_QUAD_NODES,
    ) -> _MelnikovAudit:
        r"""
        Función de Melnikov para H = H₀ + ε H₁ sobre γ⁰(t) de H₀:
            M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt.
        Gauss–Legendre en [−t_inf, +t_inf]. Cero simple ⇒ fractura homoclínica.
        """
        try:
            t0s = _NumericalCore.assert_vec("t0_grid", t0_grid)
            if t0s.size == 0:
                raise ValueError("t0_grid no puede estar vacío.")
            n_quad = int(max(16, n_quad))
            nodes, weights = np.polynomial.legendre.leggauss(n_quad)
            t_nodes = t_inf * nodes
            w_nodes = t_inf * weights
            omega = self._germ.omega
            melnikov_vals = np.zeros(t0s.size, dtype=np.float64)
            for i, t0 in enumerate(t0s):
                acc = 0.0
                comp = 0.0
                for t_shift, w in zip(t_nodes, w_nodes):
                    try:
                        x = np.asarray(homoclinic_flow(float(t_shift - t0)), dtype=np.float64)
                    except Exception:
                        continue
                    if x.size != omega.shape[0]:
                        continue
                    try:
                        pb = _NumericalCore.poisson_bracket(
                            hamiltonian_0, hamiltonian_1, x, omega
                        )
                    except (TypeError, ValueError, FloatingPointError):
                        pb = 0.0
                    if not np.isfinite(pb):
                        pb = 0.0
                    y = pb * float(w) - comp
                    t = acc + y
                    comp = (t - acc) - y
                    acc = t
                melnikov_vals[i] = acc
            idx_min = int(np.argmin(np.abs(melnikov_vals)))
            m_val = float(melnikov_vals[idx_min])
            if t0s.size >= 2 and 0 < idx_min < t0s.size - 1:
                dm = (melnikov_vals[idx_min + 1] - melnikov_vals[idx_min - 1]) / (
                    t0s[idx_min + 1] - t0s[idx_min - 1]
                )
            elif t0s.size >= 2:
                dm = (melnikov_vals[-1] - melnikov_vals[0]) / max(
                    t0s[-1] - t0s[0], _MACHINE_EPS
                )
            else:
                dm = 0.0
            is_simple = bool(
                abs(m_val) < _WILKINSON_DRIFT_LIMIT and abs(dm) > _WILKINSON_DRIFT_LIMIT
            )
            try:
                x_star = np.asarray(homoclinic_flow(float(t0s[idx_min])), dtype=np.float64)
                grad_h0 = _NumericalCore.compute_gradient_csmd(hamiltonian_0, x_star)
                grad_norm = _NumericalCore.euclidean_norm(grad_h0)
            except Exception:
                grad_norm = 0.0
            splitting = float(abs(m_val) / max(grad_norm, _WILKINSON_DRIFT_LIMIT))
            return _MelnikovAudit(
                melnikov_value=m_val,
                melnikov_derivative=float(dm),
                is_simple_zero=is_simple,
                homoclinic_splitting=splitting,
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en compute_melnikov_function: %s", exc)
            return _MelnikovAudit(
                melnikov_value=float("nan"),
                melnikov_derivative=float("nan"),
                is_simple_zero=False,
                homoclinic_splitting=float("nan"),
                engine_ok=False,
            )

    # ── II.4 Twist map de Poincaré–Birkhoff ──────────────────────────────
    def compute_poincare_birkhoff_twist_spectrum(
        self,
        deliberation_matrix_M: np.ndarray,
        contractor_twist_angle: float,
        auditor_twist_angle: float,
    ) -> _PoincareBirkhoffAudit:
        r"""
        Espectro del Twist Map de Poincaré–Birkhoff y residuo simpléctico.

        Axiomas:
          1. Twist: θ'_a − θ > 0 > θ'_b − θ ⟹ product < 0.
          2. Área de Liouville: |det M − 1| ≤ ε.
          3. Puntos fijos: |Spec(M) ∩ S¹| ≥ 2 (arco iris de Birkhoff).
          4. Invariante integral: Φ*ω = ω.
        """
        try:
            M = _NumericalCore.assert_square("deliberation_matrix_M", deliberation_matrix_M)
            det_M = float(np.real(la.det(M)))
            area_drift = abs(det_M - 1.0)
            has_opposite_twist = bool(
                (contractor_twist_angle * auditor_twist_angle) < -_BIRKHOFF_TWIST_FLOOR
            )
            eigenvalues = la.eigvals(M)
            unit_circle_fixed_points = int(
                np.sum(np.isclose(np.abs(eigenvalues), 1.0, atol=1e-9))
            )
            cartan = self._kernel.symplectic_pullback_defect(M, self._germ.omega)
            rho, twist, _n_ell = self._kernel.rotation_number_and_birkhoff_twist(
                eigenvalues
            )
            is_valid = bool(
                has_opposite_twist
                and (area_drift <= _BIRKHOFF_AREA_DRIFT_MAX)
                and (unit_circle_fixed_points >= 2)
                and (not np.isfinite(cartan) or cartan <= _CARTAN_DEFECT_TOL)
            )
            return _PoincareBirkhoffAudit(
                area_drift=float(area_drift),
                has_opposite_twist=has_opposite_twist,
                fixed_points_count=unit_circle_fixed_points,
                is_spectrum_valid=is_valid,
                engine_ok=True,
                cartan_pullback_defect=float(cartan),
                rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
                birkhoff_twist=float(twist),
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_birkhoff_twist_spectrum: %s", exc)
            return _PoincareBirkhoffAudit(
                area_drift=float("inf"),
                has_opposite_twist=False,
                fixed_points_count=0,
                is_spectrum_valid=False,
                engine_ok=False,
                cartan_pullback_defect=float("inf"),
                rotation_number=float("nan"),
                birkhoff_twist=0.0,
            )

    # ── II.5 Mapa de retorno de Poincaré ─────────────────────────────────
    def compute_poincare_return_map(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
    ) -> _ReturnMapAudit:
        r"""
        Mapa de retorno P: Σ → Σ con Floquet, Lyapunov, Conley–Zehnder,
        número de rotación, twist de Birkhoff, transversalidad y Cartan.

        Clasificación v5 mutuamente excluyente: elíptico / parabólico / hiperbólico.
        """
        try:
            M = _NumericalCore.assert_square("jacobian_M", jacobian_M)
            ev = la.eigvals(M)
            magnitudes = np.abs(ev)
            lyap = np.log(np.maximum(magnitudes, _MACHINE_EPS)) / max(
                abs(period_T), _MACHINE_EPS
            )
            lyap = np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)
            max_lyap = float(np.max(lyap)) if lyap.size else 0.0
            is_hyp, is_ell, is_par, floq_par = self._kernel.classify_floquet(magnitudes)
            cz_idx, cz_nd, cz_dist = self._kernel.conley_zehnder_index(M)
            rho, twist, n_ell = self._kernel.rotation_number_and_birkhoff_twist(ev)
            trans = self._kernel.section_transversality(M)
            cartan = self._kernel.symplectic_pullback_defect(M, self._germ.omega)
            return _ReturnMapAudit(
                floquet_multipliers=ev,
                lyapunov_spectrum=lyap,
                max_lyapunov=max_lyap,
                is_hyperbolic=is_hyp,
                is_elliptic=is_ell,
                is_parabolic=is_par,
                trace_M=float(np.trace(M)),
                det_M=float(np.real(la.det(M))),
                engine_ok=True,
                floquet_parabolic=float(floq_par),
                conley_zehnder_index=int(cz_idx),
                cz_nondegenerate=bool(cz_nd),
                cz_distance_to_one=float(cz_dist),
                rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
                birkhoff_twist=float(twist),
                elliptic_multiplicity=int(n_ell),
                section_transversality=float(trans),
                cartan_pullback_defect=float(cartan),
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_return_map: %s", exc)
            return _ReturnMapAudit(
                floquet_multipliers=np.array([], dtype=np.complex128),
                lyapunov_spectrum=np.array([], dtype=np.float64),
                max_lyapunov=float("inf"),
                is_hyperbolic=False,
                is_elliptic=False,
                is_parabolic=False,
                trace_M=float("nan"),
                det_M=float("nan"),
                engine_ok=False,
                floquet_parabolic=float("inf"),
                conley_zehnder_index=0,
                cz_nondegenerate=False,
                cz_distance_to_one=0.0,
                rotation_number=float("nan"),
                birkhoff_twist=0.0,
                elliptic_multiplicity=0,
                section_transversality=0.0,
                cartan_pullback_defect=float("inf"),
            )

    @staticmethod
    def _token_from_heyting(token: str) -> HeytingVerdict:
        try:
            return HeytingVerdict.from_token(token)
        except ValueError:
            return HeytingVerdict.VETOED

    # ── II.9 MORFISMO TERMINAL Φ_II: gérmen de ultrafiltro celeste ────────
    def induce_ultrafilter_celestial_germ(
        self,
        hyper: _HypercohomologyResult,
        brouwer: _BrouwerResult,
        kam: Optional[_KAMAudit] = None,
        melnikov: Optional[_MelnikovAudit] = None,
        birkhoff: Optional[_PoincareBirkhoffAudit] = None,
        return_map: Optional[_ReturnMapAudit] = None,
        extra_verdicts: Optional[Sequence[str]] = None,
    ) -> _UltrafilterCelestialGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE II ≅ OBJETO INICIAL DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Compone 𝒢_I + veredictos (hipercohomología, Brouwer, KAM, Melnikov,
        Birkhoff, retorno) en un único `_UltrafilterCelestialGerm`.

        Firma de continuación (Fase III):
            collapse_from_ultrafilter_celestial_germ(germ: _UltrafilterCelestialGerm)
                -> _UltrafilterResult
        """
        verdicts: List[str] = [str(hyper.verdict), str(brouwer.verdict)]
        if kam is not None:
            if not kam.engine_ok:
                verdicts.append("VETOED")
            elif (
                np.isfinite(kam.cartan_pullback_defect)
                and kam.cartan_pullback_defect > _CARTAN_DEFECT_TOL
            ):
                verdicts.append("VETOED")
            elif kam.is_kam_stable:
                verdicts.append("COHERENT")
            elif kam.min_divisor > _WILKINSON_DRIFT_LIMIT and kam.is_diophantine:
                verdicts.append("DEGRADED")
            else:
                verdicts.append("VETOED")
        if melnikov is not None:
            if not melnikov.engine_ok:
                verdicts.append("VETOED")
            elif melnikov.is_simple_zero:
                verdicts.append("VETOED")
            elif np.isfinite(melnikov.melnikov_value) and abs(
                melnikov.melnikov_value
            ) > _WILKINSON_DRIFT_LIMIT:
                verdicts.append("COHERENT")
            else:
                verdicts.append("DEGRADED")
        if birkhoff is not None:
            if not birkhoff.engine_ok:
                verdicts.append("VETOED")
            else:
                verdicts.append("COHERENT" if birkhoff.is_spectrum_valid else "DEGRADED")
        if return_map is not None:
            if not return_map.engine_ok:
                verdicts.append("VETOED")
            elif (
                np.isfinite(return_map.cartan_pullback_defect)
                and return_map.cartan_pullback_defect > _CARTAN_DEFECT_TOL
            ):
                verdicts.append("VETOED")
            elif return_map.is_elliptic and return_map.cz_nondegenerate:
                verdicts.append("COHERENT")
            elif return_map.is_parabolic or (
                return_map.is_elliptic and not return_map.cz_nondegenerate
            ):
                verdicts.append("DEGRADED")
            else:
                verdicts.append("VETOED")
        if extra_verdicts:
            verdicts.extend(str(v) for v in extra_verdicts)
        canon = tuple(v if v in _HEYTING_SEVERITY else "VETOED" for v in verdicts)
        godel = np.array([_HEYTING_GODEL[v] for v in canon], dtype=np.float64)
        hv = [self._token_from_heyting(v) for v in canon]
        meet = HeytingVerdict.meet_all(*hv) if hv else HeytingVerdict.COHERENT
        join = HeytingVerdict.join_all(*hv) if hv else HeytingVerdict.VETOED
        cartan_defect = 0.0
        cz_idx = 0
        rho = float("nan")
        if return_map is not None:
            cartan_defect = float(return_map.cartan_pullback_defect)
            cz_idx = int(return_map.conley_zehnder_index)
            rho = float(return_map.rotation_number)
        if kam is not None and np.isfinite(kam.cartan_pullback_defect):
            cartan_defect = max(cartan_defect, float(kam.cartan_pullback_defect))
        if birkhoff is not None and np.isfinite(birkhoff.cartan_pullback_defect):
            cartan_defect = max(cartan_defect, float(birkhoff.cartan_pullback_defect))
        return _UltrafilterCelestialGerm(
            hyper=hyper,
            brouwer=brouwer,
            kam=kam,
            melnikov=melnikov,
            birkhoff=birkhoff,
            return_map=return_map,
            heyting_verdicts=canon,
            godel_values=godel,
            two_n=int(self._germ.two_n),
            heyting_meet=meet.name,
            heyting_join=join.name,
            symplectic_cartan_defect=float(cartan_defect),
            conley_zehnder_index=int(cz_idx),
            rotation_number=float(rho),
        )


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE III — ÁLGEBRA DE HEYTING H₃, ULTRAFILTRO PRIMO Y COLAPSO           ║
# ║                                                                          ║
# ║ Continuación formal de II.9: el primer morfismo consume exactamente      ║
# ║ 𝒢_II = _UltrafilterCelestialGerm.                                        ║
# ║                                                                          ║
# ║ Morfismo terminal (III.2): evaluate / collapse_from_ultrafilter_…        ║
# ║     ↦ 𝒰 : H₃ⁿ → 2 = {VIABLE, RECHAZAR}, gobernado por el MEET.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True)
class _UltrafilterResult:
    """Colapso certificado del ultrafiltro booleano sobre H₃ⁿ."""

    consensus: str
    interlock_fired: bool
    n_votes: int
    n_coherent: int
    n_degraded: int
    n_vetoed: int
    godel_meet: float
    godel_join: float
    lukasiewicz_mean: float
    majority_margin: float
    filter_is_prime: bool
    generating_atom: str
    heyting_meet: str = "COHERENT"
    heyting_join: str = "VETOED"
    authorization_residue: str = "COHERENT"
    symplectic_cartan_defect: float = 0.0
    conley_zehnder_index: int = 0
    rotation_number: float = 0.0


class _UltrafilterEvaluator:
    r"""
    Fase III. Colapso 𝒰 : H₃ⁿ → 2 = {VIABLE, RECHAZAR}.

    CONTINUACIÓN FORMAL DE II.9:
        collapse_from_ultrafilter_celestial_germ(𝒢_II) es el primer morfismo
        de esta fase y consume el objeto terminal de la Fase II.

    El álgebra de Heyting de tres valores (orden de PERMISO)
        H₃ = {VETOED ≤ DEGRADED ≤ COHERENT}
    clasifica los veredictos locales. El filtro se genera por el MEET:
        x ∈ 𝒰  ⇔  meet_i(x_i) = VETOED
                   ∨  (#{i : x_i ⪯ DEGRADED} > n/2  ∧  ningún COHERENT-puro).
    Imagen: clasificador de subobjetos 2 = {VIABLE, RECHAZAR}.

    Corrección v5: un único VETOED dispara RECHAZAR. El join es diagnóstico.
    """

    def __init__(self, germ: Optional[_UltrafilterGerm] = None) -> None:
        self._germ = germ
        self._celestial_germ: Optional[_UltrafilterCelestialGerm] = None

    def attach_celestial_germ(self, germ: _UltrafilterCelestialGerm) -> None:
        """Ancla el gérmen celeste para el colapso (continuación de Fase II)."""
        self._celestial_germ = germ

    # ── III.1 Continuación de II.9 ───────────────────────────────────────
    def collapse_from_ultrafilter_celestial_germ(
        self, germ: _UltrafilterCelestialGerm
    ) -> _UltrafilterResult:
        r"""
        Primer morfismo de la Fase III. Continúa II.9:
            induce_ultrafilter_celestial_germ(...) -> 𝒢_II
            collapse_from_ultrafilter_celestial_germ(𝒢_II) -> acta 𝒰
        """
        self.attach_celestial_germ(germ)
        return self.evaluate([])

    @staticmethod
    def _canonicalize(heyting_verdicts: Sequence[str]) -> Tuple[str, ...]:
        return tuple(v if v in _HEYTING_SEVERITY else "VETOED" for v in heyting_verdicts)

    def _resolve(self, heyting_verdicts: Sequence[str]) -> Tuple[str, ...]:
        if not heyting_verdicts:
            if self._celestial_germ is not None:
                return self._celestial_germ.heyting_verdicts
            if self._germ is not None:
                return self._germ.heyting_verdicts
        return self._canonicalize(heyting_verdicts)

    def evaluate(self, heyting_verdicts: List[str]) -> _UltrafilterResult:
        """Aplica el ultrafiltro booleano no trivial 𝒰, gobernado por el MEET."""
        votes = self._resolve(list(heyting_verdicts) if heyting_verdicts is not None else [])
        total = len(votes)
        cartan = 0.0
        cz_idx = 0
        rho = float("nan")
        if self._celestial_germ is not None:
            cartan = float(self._celestial_germ.symplectic_cartan_defect)
            cz_idx = int(self._celestial_germ.conley_zehnder_index)
            rho = float(self._celestial_germ.rotation_number)
        if total == 0:
            return _UltrafilterResult(
                consensus="VIABLE",
                interlock_fired=False,
                n_votes=0,
                n_coherent=0,
                n_degraded=0,
                n_vetoed=0,
                godel_meet=1.0,
                godel_join=0.0,
                lukasiewicz_mean=1.0,
                majority_margin=0.0,
                filter_is_prime=True,
                generating_atom="COHERENT",
                heyting_meet="COHERENT",
                heyting_join="VETOED",
                authorization_residue="COHERENT",
                symplectic_cartan_defect=cartan,
                conley_zehnder_index=cz_idx,
                rotation_number=rho,
            )
        numeric = [_HEYTING_SEVERITY[v] for v in votes]
        n_veto = int(numeric.count(2))
        n_deg = int(numeric.count(1))
        n_coh = int(numeric.count(0))
        max_severity = max(numeric)
        generating = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}[max_severity]
        hv = [HeytingVerdict.from_token(v) for v in votes]
        meet = HeytingVerdict.meet_all(*hv)
        join = HeytingVerdict.join_all(*hv)
        residue = meet.implies(HeytingVerdict.COHERENT)
        godel = np.array([_HEYTING_GODEL[v] for v in votes], dtype=np.float64)
        godel_meet = float(np.min(godel))
        godel_join = float(np.max(godel))
        luk = float(_NumericalCore.kahan_babuska_neumaier_sum(godel) / total)
        n_non_coherent = n_veto + n_deg
        majority_margin = float(n_non_coherent - (total / 2.0))
        # MEET gobierna: cualquier VETOED ∈ 𝒰. Mayoría degradada también.
        in_filter = (meet is HeytingVerdict.VETOED) or (n_deg > (total // 2))
        filter_is_prime = True
        if in_filter:
            consensus = "RECHAZAR"
            interlock = True
            logger.critical(
                "¡COLAPSO DE ULTRAFILTRO BOOLEANO EN EL PRETORIO! "
                "Meet H₃=%s Join H₃=%s Votos=%s. Consenso: RECHAZAR. "
                "CartanDefect=%.3e CZ=%d ρ=%.6f. Interlock ACTIVADO.",
                meet.name,
                join.name,
                list(votes),
                cartan,
                cz_idx,
                rho,
            )
        else:
            consensus = "VIABLE"
            interlock = False
        return _UltrafilterResult(
            consensus=consensus,
            interlock_fired=bool(interlock),
            n_votes=total,
            n_coherent=n_coh,
            n_degraded=n_deg,
            n_vetoed=n_veto,
            godel_meet=godel_meet,
            godel_join=godel_join,
            lukasiewicz_mean=luk,
            majority_margin=majority_margin,
            filter_is_prime=filter_is_prime,
            generating_atom=generating,
            heyting_meet=meet.name,
            heyting_join=join.name,
            authorization_residue=residue.name,
            symplectic_cartan_defect=cartan,
            conley_zehnder_index=cz_idx,
            rotation_number=rho,
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class PretorioEngine:
    r"""
    Motor matemático de rango supremo. Provee cálculo espectral, homotópico,
    celeste y categorial al Pretorio Agéntico (`pretorio_agent.py`).

    Compone las tres fases anidadas (cada morfismo terminal es el objeto
    inicial del siguiente):
        Φ_I   : Banach + Weyl + Maupertuis + Cartan + Hodge          ⟶  𝒢_I
        Φ_II  : lift_from_celestial_germ(𝒢_I) + KAM + Melnikov + CZ  ⟶  𝒢_II
        Φ_III : collapse_from_ultrafilter_celestial_germ(𝒢_II)       ⟶  acta 𝒰
    """

    def __init__(
        self,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        dimension_two_n: int = 4,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        novikov_valuation_T: float = 1.0,
    ) -> None:
        self._reg: Final[float] = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        self._two_n: Final[int] = int(dimension_two_n) if int(dimension_two_n) % 2 == 0 else 4
        self._H0: Final[float] = float(hamiltonian_energy_H0)
        self._V: Final[float] = float(potential_energy_V)
        self._novikov_T: Final[float] = float(novikov_valuation_T)

        zero = np.zeros((1, 1), dtype=np.complex128)
        self._hyper_germ: _HypercohomologyGerm = (
            _NumericalCore.synthesize_hypercohomology_germ(zero, zero, regularizer=self._reg)
        )
        self._hyper_cohomology_checker = _HypercohomologyChecker(
            germ=self._hyper_germ, regularizer=self._reg
        )
        self._brouwer_checker = _BrouwerChecker(regularizer=self._reg)

        self._celestial_germ: _PretorioCelestialGerm = (
            _NumericalCoreCelestial.synthesize_pretorio_celestial_germ(
                dimension_two_n=self._two_n,
                hamiltonian_energy_H0=self._H0,
                potential_energy_V=self._V,
                regularizer=self._reg,
            )
        )
        self._celestial_auditor = _PoincareCelestialAuditor.lift_from_celestial_germ(
            self._celestial_germ
        )
        self._ultra_germ: _UltrafilterGerm = self._brouwer_checker.induce_ultrafilter_germ()
        self._ultrafilter_evaluator = _UltrafilterEvaluator(germ=self._ultra_germ)
        self._last_celestial_ultra: Optional[_UltrafilterCelestialGerm] = None

    @property
    def celestial_germ(self) -> _PretorioCelestialGerm:
        """Gérmen celeste de Fase I vigente (objeto terminal)."""
        return self._celestial_germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE I — Banach, Weyl–Toeplitz, Maupertuis–Jacobi, Hill, Cartan
    # ══════════════════════════════════════════════════════════════════════
    def kahan_compensated_trace(self, matrix: np.ndarray) -> float:
        return _NumericalCore.compensated_trace(matrix)

    def kahan_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def weyl_toeplitz_symmetrization(self, density_matrix: np.ndarray) -> np.ndarray:
        return _NumericalCore.weyl_toeplitz_symmetrization(density_matrix)

    def higham_nearest_density(self, matrix: np.ndarray) -> np.ndarray:
        rho, _evals = _NumericalCore.higham_nearest_density(matrix, floor=self._reg)
        return rho

    def compute_maupertuis_jacobi_conformal_metric(
        self,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: np.ndarray,
    ) -> Tuple[np.ndarray, _MaupertuisJacobiGerm]:
        return _NumericalCore.compute_maupertuis_jacobi_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, base_metric_g
        )

    def compute_hill_region_margin(
        self,
        potential_V: float,
        total_energy_H0: float,
    ) -> float:
        return _NumericalCore.compute_hill_region_margin(potential_V, total_energy_H0)

    def compute_christoffel_conformal_symbols(
        self,
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> np.ndarray:
        return _NumericalCore.compute_christoffel_conformal_symbols(
            grad_V, potential_V, total_energy_H0, g_base_metric
        )

    def compute_poincare_cartan_lambda(
        self,
        x: np.ndarray,
        hamiltonian_value: float = 0.0,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, _PoincareCartanGerm]:
        return _NumericalCore.compute_poincare_cartan_lambda(x, hamiltonian_value, omega)

    def compute_poisson_bracket(
        self,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: Optional[np.ndarray] = None,
    ) -> float:
        if omega is None:
            omega = self._celestial_germ.omega
        return _NumericalCore.poisson_bracket(hamiltonian_0, hamiltonian_1, x, omega)

    def stormer_verlet_step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _NumericalCore.stormer_verlet_step(q, p, grad_v, mass_inv, dt)

    def integrate_symplectic_maupertuis_step(
        self,
        x_state: np.ndarray,
        dt_step: float,
        g_base_metric: np.ndarray,
        potential_V: float,
        grad_V: np.ndarray,
        total_energy_H0: float,
        mass_inv: float = 1.0,
    ) -> MaupertuisStepReport:
        r"""Paso Störmer–Verlet + geometría conforme (contrato Imperial Guards)."""
        return _NumericalCoreCelestial.integrate_symplectic_maupertuis_step(
            x_state=x_state,
            dt_step=dt_step,
            g_base_metric=g_base_metric,
            potential_V=potential_V,
            grad_V=grad_V,
            total_energy_H0=total_energy_H0,
            mass_inv=mass_inv,
        )

    def synthesize_hypercohomology_germ(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> _HypercohomologyGerm:
        germ = _NumericalCore.synthesize_hypercohomology_germ(
            cech_boundary_d1, derham_boundary_d2, regularizer=self._reg
        )
        self._hyper_germ = germ
        self._hyper_cohomology_checker = _HypercohomologyChecker(
            germ=germ, regularizer=self._reg
        )
        return germ

    def synthesize_pretorio_celestial_germ(
        self,
        dimension_two_n: Optional[int] = None,
        hamiltonian_energy_H0: Optional[float] = None,
        potential_energy_V: Optional[float] = None,
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
    ) -> _PretorioCelestialGerm:
        r"""Réplica pública del morfismo I.10: sintetiza 𝒢_I (inicial de Fase II)."""
        germ = _NumericalCoreCelestial.synthesize_pretorio_celestial_germ(
            dimension_two_n=dimension_two_n if dimension_two_n is not None else self._two_n,
            hamiltonian_energy_H0=(
                hamiltonian_energy_H0 if hamiltonian_energy_H0 is not None else self._H0
            ),
            potential_energy_V=(
                potential_energy_V if potential_energy_V is not None else self._V
            ),
            base_metric_g=base_metric_g,
            regularizer=self._reg,
            grad_V=grad_V,
        )
        self._celestial_germ = germ
        self._celestial_auditor = _PoincareCelestialAuditor.lift_from_celestial_germ(germ)
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE II — Hipercohomología, Brouwer, KAM, Melnikov, Birkhoff, CZ
    # (continúa desde synthesize_pretorio_celestial_germ → 𝒢_I)
    # ══════════════════════════════════════════════════════════════════════
    def lift_from_celestial_germ(
        self, germ: Optional[_PretorioCelestialGerm] = None
    ) -> _PoincareCelestialAuditor:
        r"""Réplica pública del morfismo II.0: 𝒢_I ↦ auditor de Fase II."""
        g = germ if germ is not None else self._celestial_germ
        auditor = _PoincareCelestialAuditor.lift_from_celestial_germ(g)
        self._celestial_auditor = auditor
        self._celestial_germ = g
        return auditor

    def verify_cech_derham_hypercohomology(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> Tuple[float, str]:
        result = self.verify_cech_derham_hypercohomology_certified(
            cech_boundary_d1, derham_boundary_d2
        )
        return result.residual, result.verdict

    def verify_cech_derham_hypercohomology_certified(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> _HypercohomologyResult:
        result = self._hyper_cohomology_checker.verify(
            cech_boundary_d1, derham_boundary_d2
        )
        if self._hyper_cohomology_checker.germ is not None:
            self._hyper_germ = self._hyper_cohomology_checker.germ
        return result

    def verify_brouwer_fixed_point(
        self,
        rho_current: np.ndarray,
        rho_transformed: np.ndarray,
    ) -> Tuple[float, float, str]:
        result = self._brouwer_checker.verify(rho_current, rho_transformed)
        return result.brouwer_residual, result.trace_residual, result.verdict

    def verify_brouwer_fixed_point_certified(
        self,
        rho_current: np.ndarray,
        rho_transformed: np.ndarray,
    ) -> _BrouwerResult:
        return self._brouwer_checker.verify(rho_current, rho_transformed)

    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        canonical_J: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> _KAMAudit:
        return self._celestial_auditor.compute_poincare_small_divisors_spectrum(
            frequency_vector_omega,
            wave_vectors_k,
            jacobian_M,
            canonical_J,
            tau=tau,
            gamma=gamma,
            novikov_valuation_T=self._novikov_T,
        )

    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: np.ndarray,
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> np.ndarray:
        return self._celestial_auditor.compute_arnold_resonance_lattice(
            frequency_vector_omega, max_order=max_order, tol=tol
        )

    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = _MELNIKOV_T_INF,
    ) -> _MelnikovAudit:
        return self._celestial_auditor.compute_melnikov_function(
            homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf=t_inf
        )

    def compute_poincare_birkhoff_twist_spectrum(
        self,
        deliberation_matrix_M: np.ndarray,
        contractor_twist_angle: float,
        auditor_twist_angle: float,
    ) -> PretorioSpectrumReport:
        audit = self._celestial_auditor.compute_poincare_birkhoff_twist_spectrum(
            deliberation_matrix_M, contractor_twist_angle, auditor_twist_angle
        )
        return PretorioSpectrumReport(
            area_drift=audit.area_drift,
            has_opposite_twist=audit.has_opposite_twist,
            fixed_points_count=audit.fixed_points_count,
            is_spectrum_valid=audit.is_spectrum_valid,
        )

    def compute_poincare_birkhoff_twist_spectrum_certified(
        self,
        deliberation_matrix_M: np.ndarray,
        contractor_twist_angle: float,
        auditor_twist_angle: float,
    ) -> _PoincareBirkhoffAudit:
        return self._celestial_auditor.compute_poincare_birkhoff_twist_spectrum(
            deliberation_matrix_M, contractor_twist_angle, auditor_twist_angle
        )

    def compute_poincare_return_map(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
    ) -> _ReturnMapAudit:
        return self._celestial_auditor.compute_poincare_return_map(
            jacobian_M, period_T=period_T
        )

    def induce_ultrafilter_germ(
        self,
        heyting_verdicts: Optional[Sequence[str]] = None,
        hyper: Optional[_HypercohomologyResult] = None,
        brouwer: Optional[_BrouwerResult] = None,
    ) -> _UltrafilterGerm:
        germ = self._brouwer_checker.induce_ultrafilter_germ(
            hyper=hyper, brouwer=brouwer, extra_verdicts=heyting_verdicts
        )
        self._ultra_germ = germ
        self._ultrafilter_evaluator = _UltrafilterEvaluator(germ=germ)
        return germ

    def induce_ultrafilter_celestial_germ(
        self,
        hyper: _HypercohomologyResult,
        brouwer: _BrouwerResult,
        kam: Optional[_KAMAudit] = None,
        melnikov: Optional[_MelnikovAudit] = None,
        birkhoff: Optional[_PoincareBirkhoffAudit] = None,
        return_map: Optional[_ReturnMapAudit] = None,
        extra_verdicts: Optional[Sequence[str]] = None,
    ) -> _UltrafilterCelestialGerm:
        r"""Réplica pública del morfismo II.9: 𝒢_I + votos ↦ 𝒢_II (inicial Fase III)."""
        germ = self._celestial_auditor.induce_ultrafilter_celestial_germ(
            hyper=hyper,
            brouwer=brouwer,
            kam=kam,
            melnikov=melnikov,
            birkhoff=birkhoff,
            return_map=return_map,
            extra_verdicts=extra_verdicts,
        )
        self._last_celestial_ultra = germ
        self._ultrafilter_evaluator.attach_celestial_germ(germ)
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE III — Ultrafiltro booleano y colapso a silicio
    # (continúa desde induce_ultrafilter_celestial_germ → 𝒢_II)
    # ══════════════════════════════════════════════════════════════════════
    def collapse_from_ultrafilter_celestial_germ(
        self, germ: Optional[_UltrafilterCelestialGerm] = None
    ) -> _UltrafilterResult:
        r"""Réplica pública del morfismo III.1: 𝒢_II ↦ acta 𝒰 (MEET gobierna)."""
        g = germ if germ is not None else self._last_celestial_ultra
        if g is None:
            raise ValueError(
                "No hay gérmen celeste de ultrafiltro: invoque "
                "induce_ultrafilter_celestial_germ."
            )
        return self._ultrafilter_evaluator.collapse_from_ultrafilter_celestial_germ(g)

    def evaluate_ultrafilter_consensus(
        self,
        heyting_verdicts: List[str],
    ) -> Tuple[str, bool]:
        result = self._ultrafilter_evaluator.evaluate(heyting_verdicts)
        return result.consensus, result.interlock_fired

    def evaluate_ultrafilter_consensus_certified(
        self,
        heyting_verdicts: List[str],
    ) -> _UltrafilterResult:
        return self._ultrafilter_evaluator.evaluate(heyting_verdicts)

    def evaluate_celestial_ultrafilter_consensus(self) -> _UltrafilterResult:
        r"""
        Colapso Φ_III sobre los votos celestes previamente inducidos.
        Requiere `induce_ultrafilter_celestial_germ`.
        """
        return self._ultrafilter_evaluator.evaluate([])

    def execute_celestial_cycle(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
        rho_current: np.ndarray,
        rho_transformed: np.ndarray,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        contractor_twist_angle: float = 1.0,
        auditor_twist_angle: float = -1.0,
        period_T: float = 1.0,
    ) -> _UltrafilterResult:
        r"""
        Ciclo completo Φ_III ∘ Φ_II ∘ Φ_I: hipercohomología + Brouwer + canales
        celestes, colapsando por el MEET de Heyting al ultrafiltro 𝒰.
        """
        hyper = self.verify_cech_derham_hypercohomology_certified(
            cech_boundary_d1, derham_boundary_d2
        )
        brouwer = self.verify_brouwer_fixed_point_certified(rho_current, rho_transformed)
        kam = None
        if frequency_vector_omega is not None and wave_vectors_k is not None:
            jac = jacobian_M if jacobian_M is not None else np.eye(self._two_n)
            canon = (
                canonical_J if canonical_J is not None else self._celestial_germ.omega
            )
            kam = self.compute_poincare_small_divisors_spectrum(
                frequency_vector_omega, wave_vectors_k, jac, canon
            )
        melnikov = None
        if (
            homoclinic_flow is not None
            and hamiltonian_0 is not None
            and hamiltonian_1 is not None
            and t0_grid is not None
        ):
            melnikov = self.compute_melnikov_function(
                homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid
            )
        birkhoff = None
        return_map = None
        if jacobian_M is not None:
            birkhoff = self.compute_poincare_birkhoff_twist_spectrum_certified(
                jacobian_M, contractor_twist_angle, auditor_twist_angle
            )
            return_map = self.compute_poincare_return_map(jacobian_M, period_T=period_T)
        germ = self.induce_ultrafilter_celestial_germ(
            hyper=hyper,
            brouwer=brouwer,
            kam=kam,
            melnikov=melnikov,
            birkhoff=birkhoff,
            return_map=return_map,
        )
        return self.collapse_from_ultrafilter_celestial_germ(germ)


__all__ = [
    "PretorioEngine",
    "PretorioSpectrumReport",
    "MaupertuisStepReport",
    "HeytingVerdict",
]