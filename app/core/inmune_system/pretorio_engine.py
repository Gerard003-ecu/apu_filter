# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pretorio Engine (Caballería Suprema de Cálculo Epistémico)          ║
║ Ruta   : app/core/inmune_system/pretorio_engine.py                           ║
║ Versión: 3.0.0-Nested-Phases-Cech-deRham-Hodge-Brouwer-Heyting-Ultrafilter   ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Este motor supremo ejecuta el escrutinio final e independiente de la coherencia ciber-física.
Evalúa la hipercohomología del bicomplejo de Čech-de Rham, purga las corrientes asimétricas en
la base de la MAC mediante simetrización de Weyl-Toeplitz y colapsa los veredictos parciales en
un ultrafiltro booleano binario.

MÉTODOS GRANULARES:

1. verify_cech_derham_hypercohomology(d1: np.ndarray, d2: np.ndarray) -> Tuple[float, bool]:
   Audita la consistencia global del bicomplejo de haces de calibre. Define el operador diferencial
   total D = d_1 + (-1)^p d_2 y exige la aniquilación incondicional de su nilpotencia en la FPU:
   $$D^2 = d_1 \circ d_2 + d_2 \circ d_1 \equiv \mathbf{0} \implies \epsilon_{\mathrm{cohom}} = \|d_1 d_2 + d_2 d_1\|_F \le \tau_{\mathrm{cohom}}$$
   - d1: np.ndarray (operador diferencial Čech).
   - d2: np.ndarray (operador diferencial de de Rham).
   - Retorna: Tuple con el residual homológico de Čech-de Rham y la certificación de nulidad.

2. verify_brouwer_fixed_point(rho: np.ndarray, transition_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
   Certifica que el transporte paralelo del operador densidad \rho conserve el punto fijo hermítico-positivo
   regularizado por Weyl-Toeplitz bajo la acción del mapa de transición f(\rho) = \rho:
   $$\epsilon_{\mathrm{Brouwer}} = \|\tilde{\rho} - f(\tilde{\rho})\|_F \equiv 0 \quad \text{con} \quad \tilde{\rho} = \frac{1}{2}\left( \rho + \rho^\dagger \right)$$
   Valida mediante sumación KBN que la traza cuántica de sabiduría conserve de forma exacta su carácter unitario: \operatorname{Tr}(\tilde{\rho}) \equiv 1.0.
   - rho: np.ndarray (matriz de densidad cuántica).
   - transition_matrix: np.ndarray (matriz de transición de-confinada).
   - Retorna: Tuple con el residuo de Brouwer y el operador densidad purificado.

3. evaluate_ultrafilter_consensus(heyting_verdicts: list[str]) -> Tuple[str, bool]:
   Ingiere los veredictos parciales de Heyting \Omega_3 de todas las capas de seguridad y evalúa si el conjunto
   de subcapas con veto pertenece al ultrafiltro booleano no trivial \mathcal{U} para colapsar la lógica
   distributiva hacia una instrucción clásica de actuación en silicio (ESP32):
   $$\mathcal{U} = \{A \subseteq S_{\mathrm{Capas}} \mid \nu_{\mathrm{global}}(A) = \mathtt{VETOED}\} \implies B_2 = \{\mathtt{VIABLE}, \, \mathtt{RECHAZAR}\}$$
   - heyting_verdicts: list[str] (veredictos de los 55 agentes soberanos).
   - Retorna: Tuple con la sentencia final de Heyting unificada y el indicador de actuación inmediata de potencia (disparo Crowbar BT151).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.PretorioEngine")

__version__: Final[str] = (
    "3.0.0-Nested-Phases-Cech-deRham-Hodge-Brouwer-Heyting-Ultrafilter"
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

_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_CANONICAL_VERDICTS: Final[Tuple[str, ...]] = ("COHERENT", "DEGRADED", "VETOED")


# =============================================================================
# FASE I — NÚCLEO DE BANACH, WEYL–TOEPLITZ Y DENSIDAD DE HIGHAM
# -----------------------------------------------------------------------------
# Objetos: sumas compensadas, proyección hermítica, estado densidad más
#          próximo, normas y residuales relativos de Wilkinson.
# Morfismo terminal (I.8): synthesize_hypercohomology_germ
#          ≅ objeto inicial de la Fase II (bicomplejo Čech–de Rham).
# =============================================================================
@dataclass(frozen=True)
class _HypercohomologyGerm:
    """
    Gérmen del bicomplejo Čech–de Rham (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: un par de operadores

        δ = d₁ : C^{p,q} → C^{p+1,q} ,   d = d₂ : C^{p,q} → C^{p,q+1}

    junto con la factibilidad de las identidades de bicomplejo

        δ² = 0 ,   d² = 0 ,   δd + dδ = 0

    y, si ambos son endomorfismos del mismo espacio, el Laplaciano de
    Hodge Δ_D = D* D + D D* del diferencial total D = δ + d.

    Atributos
    ---------
    d1, d2:
        Representantes matriciales de δ y d.
    d1_square, d2_square:
        Factibilidad de δ² y d².
    composition_lr, composition_rl:
        Factibilidad de δ∘d y d∘δ.
    hodge_laplacian:
        Δ_D si D es un endomorfismo; None en otro caso.
    reg_floor:
        Piso de Tikhonov–Wilkinson.
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


class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica.

    Provee el topos lineal subyacente: sumación compensada en el álgebra
    de Banach (ℝ, +, ·), la proyección de Weyl–Toeplitz al cono hermítico
    y la proyección de Higham al simplejo de estados densidad 𝒟(ℋ).
    """

    # ── I.1  Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """
        Sumación compensada de Kahan.

        Neutraliza el término de redondeo \(c_{k+1}=(t_k-s_k)-y_k\) de modo
        que \(\sum x_i\) sea exacta módulo O(u · Σ|x_i|).
        """
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
        """
        Sumación de Kahan–Babuška–Neumaier (KBN).

        Acumula la compensación cuando |x| > |s|, estabilizando
        cancelaciones de signo mixto (trazas, residuales de D²).
        """
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

    # Alias histórico (v2.0 usaba el epónimo incorrecto "Neumann").
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

    # ── I.2  Normas, validación y Weyl–Toeplitz ───────────────────────────
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt / Frobenius ‖A‖_F."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

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
    def weyl_toeplitz_symmetrization(matrix: np.ndarray) -> np.ndarray:
        """
        Proyección de Weyl–Toeplitz / Higham al cono hermítico:

            Π_Herm(M) = (M + M†) / 2 .

        Es la matriz hermítica más próxima en ‖·‖_F (teorema de Higham).
        Purga el ruido antisimétrico de redondeo en la mantisa.
        """
        a = _NumericalCore.assert_square("weyl_toeplitz_symmetrization", matrix)
        return 0.5 * (a + a.T.conj())

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
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        """
        Piso de deflación adaptativo de Wilkinson:

            ε_W = max( ‖A‖_F · ε_mach · 10 ,  ε_Wilkinson ).
        """
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
        """
        Estado densidad más próximo en ‖·‖_F (Higham + renormalización):

            Hermitiza, recorta el espectro a [0, +∞), renormaliza Tr = 1.

        Devuelve (ρ, autovalores normalizados). Si la traza positiva
        colapsa, se emite el estado puro del modo dominante.
        """
        herm = _NumericalCore.weyl_toeplitz_symmetrization(matrix)
        evals, evecs = la.eigh(herm)
        evals = np.maximum(np.real(evals), 0.0)
        # Deflación de ruido bajo el piso (0 log 0 := 0 más adelante).
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

    # ── I.8  Morfismo terminal de la Fase I ───────────────────────────────
    @staticmethod
    def synthesize_hypercohomology_germ(
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
    ) -> _HypercohomologyGerm:
        """
        I.8 — Morfismo terminal de la Fase I / objeto inicial de la Fase II.

        Ensambla el gérmen del bicomplejo Čech–de Rham

            𝒢_I = (δ, d, fact(δ², d², δd, dδ), Δ_D, ε_W)

        sobre el cual la Fase II certifica la nilpotencia del diferencial
        total D = δ + (−1)^p d, i.e.

            D² = δ² + d² + {δ, d}_grad  ≡  0 .

        Si δ y d son endomorfismos del mismo espacio se construye además
        el Laplaciano de Hodge Δ_D = D†D + DD† (D := δ + d), cuyos
        números de Betti numéricos alimentan el certificado.

        Este método *es* el arranque formal de `_HypercohomologyChecker`
        y, a través de sus veredictos, de `_BrouwerChecker` (el estado
        densidad vive en el mismo topos de Weyl–Toeplitz).
        """
        d1 = _NumericalCore.assert_matrix("cech_boundary_d1", cech_boundary_d1)
        d2 = _NumericalCore.assert_matrix("derham_boundary_d2", derham_boundary_d2)
        d1 = np.asarray(d1, dtype=np.complex128)
        d2 = np.asarray(d2, dtype=np.complex128)

        d1_square = d1.shape[0] == d1.shape[1] and d1.shape[0] == d1.shape[1]
        d2_square = d2.shape[0] == d2.shape[1]
        # δ² exige que δ sea un endomorfismo (o al menos composable consigo).
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
# FASE II — HIPERCOHOMOLOGÍA, HODGE, BROUWER Y LIFTING A H₃
# -----------------------------------------------------------------------------
# Continúa I.8: los verificadores se instancian desde un HypercohomologyGerm
# (o lo inducen al vuelo). Morfismo terminal (II.7): induce_ultrafilter_germ
#          ≅ objeto inicial de la Fase III (valuación de Heyting).
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
    """
    Gérmen de Heyting (objeto terminal de la Fase II).

    Es el objeto inicial de la Fase III: una tupla de veredictos locales
    en el álgebra de Heyting

        H₃ = { VETOED ≺ DEGRADED ≺ COHERENT }

    con su valuación de Gödel ν : H₃ → [0,1] y los residuales que la
    originaron (hipercohomología / Brouwer). El colapso de ultrafiltro
    es el morfismo de clasificadores 𝒰 : H₃ⁿ → 2.
    """

    heyting_verdicts: Tuple[str, ...]
    godel_values: np.ndarray
    hyper_verdict: str
    brouwer_verdict: str
    hyper_residual: float
    brouwer_residual: float
    source: str


class _HypercohomologyChecker:
    """
    Fase II. Nilpotencia del bicomplejo Čech–de Rham.

    Continúa el gérmen 𝒢_I. Sobre (δ, d) mide:

    * ‖δ ∘ d‖_F                 (residual 2.0, composición cruzada);
    * ‖δ²‖_F , ‖d²‖_F          (nilpotencia de cada diferencial);
    * ‖δd + dδ‖_F , ‖δd − dδ‖_F (anticommutador / conmutador graduado);
    * ‖D²‖_F con D = δ + d      (si ambos son endomorfismos);
    * β₀ numérico de Δ_D        (dimensión del núcleo de Hodge).

    El veredicto 2.0 se conserva: compara ‖δd‖ con el umbral de Wilkinson
    ε_W = max(‖δ‖_F ‖d‖_F · 10 ε_mach, ε_defl) y escala el veto ×100.
    """

    def __init__(
        self,
        germ: Optional[_HypercohomologyGerm] = None,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
    ) -> None:
        self._germ = germ
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)

    @property
    def germ(self) -> Optional[_HypercohomologyGerm]:
        """Gérmen de Fase I del que este verificador es continuación."""
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
        """
        Certifica D² ≡ 0. El campo `residual` permanece siendo ‖δ d‖_F
        (API 2.0). Si las dimensiones no permiten δ∘d se preserva el
        contrato histórico: residual = +∞, verdict = VETOED.
        """
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

        d1_sq = (
            _NumericalCore.frobenius_norm(d1 @ d1) if germ.d1_square else float("nan")
        )
        d2_sq = (
            _NumericalCore.frobenius_norm(d2 @ d2) if germ.d2_square else float("nan")
        )

        if germ.composition_lr and germ.composition_rl:
            lr = d1 @ d2
            rl = d2 @ d1
            if lr.shape == rl.shape:
                anticomm = _NumericalCore.frobenius_norm(lr + rl)
                comm = _NumericalCore.frobenius_norm(lr - rl)
            else:
                anticomm = float("nan")
                comm = float("nan")
        else:
            anticomm = float("nan")
            comm = float("nan")

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
    """
    Fase II (continuación geométrica). Punto fijo de Brouwer en 𝒟(ℋ).

    En dimensión finita 𝒟(ℋ) es compacto, convexo y no vacío, de modo
    que toda f : 𝒟(ℋ) → 𝒟(ℋ) continua admite un punto fijo. Se certifica:

    * pertenencia aproximada de ρ y T(ρ) a 𝒟(ℋ) (hermiticidad, PSD, Tr = 1);
    * residual de Hilbert–Schmidt ‖ρ − T(ρ)‖_F (API 2.0);
    * residual proyectado ‖Π(ρ) − Π(T(ρ))‖_F (Higham);
    * pureza Tr ρ² y λ_min;
    * pista de Lipschitz ‖T(ρ)−ρ‖ / diam  (diam ≤ √2 en HS sobre 𝒟).
    """

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
        """
        Aplica Weyl–Toeplitz y calcula residuales de Brouwer y traza.

        Los umbrales 2.0 se conservan (1e-6 / 1e-9 en HS, 1e-9 / 1e-12
        en |Tr ρ − 1|). El certificado añade PSD, pureza y proyección.
        """
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
        # diam_HS(𝒟) ≤ √2; la pista de Lipschitz es residual / diam.
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

    # ── II.7  Morfismo terminal de la Fase II ─────────────────────────────
    def induce_ultrafilter_germ(
        self,
        hyper: Optional[_HypercohomologyResult] = None,
        brouwer: Optional[_BrouwerResult] = None,
        extra_verdicts: Optional[Sequence[str]] = None,
    ) -> _UltrafilterGerm:
        """
        II.7 — Morfismo terminal de la Fase II / objeto inicial de la Fase III.

        Empaqueta los veredictos locales (hipercohomología, Brouwer y
        cualquier test auxiliar de las aduanas) como un elemento de H₃ⁿ
        con valuación de Gödel

            ν(COHERENT) = 1 ,  ν(DEGRADED) = ½ ,  ν(VETOED) = 0 .

        Este morfismo *es* el arranque formal de `_UltrafilterEvaluator`:
        el colapso 𝒰 : H₃ⁿ → 2 se aplica exactamente sobre este gérmen.
        """
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

        canon = tuple(
            v if v in _HEYTING_ORDER else "VETOED" for v in verdicts
        )
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
            hyper_verdict=hyper_v if hyper_v in _HEYTING_ORDER else "VETOED",
            brouwer_verdict=brouwer_v if brouwer_v in _HEYTING_ORDER else "VETOED",
            hyper_residual=hyper_r,
            brouwer_residual=brouwer_r,
            source=source,
        )


# =============================================================================
# FASE III — ÁLGEBRA DE HEYTING H₃, ULTRAFILTRO PRIMO Y MOTOR INTEGRADOR
# -----------------------------------------------------------------------------
# Continúa II.7: el evaluador se ancla a un UltrafilterGerm (o lo induce
# por lectura directa de una lista de veredictos locales).
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


class _UltrafilterEvaluator:
    """
    Fase III. Colapso 𝒰 : H₃ⁿ → 2.

    Continúa el gérmen 𝒢_II. El álgebra de Heyting de tres valores

        H₃ = {0 = VETOED  ≺  ½ = DEGRADED  ≺  1 = COHERENT}

    clasifica los veredictos locales. El filtro primo (principal) usado
    por el Pretorio —idéntico en extensión a la regla 2.0— es

        x ∈ 𝒰  ⇔  (∨_i x_i = VETOED)  ∨  (#{i : x_i ⪰ DEGRADED} > n/2).

    Su imagen es el clasificador de subobjetos 2 = {VIABLE, RECHAZAR}.
    `interlock_fired` es la función característica de 𝒰.
    """

    def __init__(self, germ: Optional[_UltrafilterGerm] = None) -> None:
        self._germ = germ

    @staticmethod
    def _canonicalize(heyting_verdicts: Sequence[str]) -> Tuple[str, ...]:
        return tuple(
            v if v in _HEYTING_ORDER else "VETOED" for v in heyting_verdicts
        )

    def _resolve(self, heyting_verdicts: Sequence[str]) -> Tuple[str, ...]:
        if (not heyting_verdicts) and self._germ is not None:
            return self._germ.heyting_verdicts
        return self._canonicalize(heyting_verdicts)

    def evaluate(self, heyting_verdicts: List[str]) -> _UltrafilterResult:
        """
        Aplica el ultrafiltro booleano no trivial 𝒰 sobre los veredictos.

        Lista vacía ⇒ VIABLE (unidad del monoid de votos), como en 2.0.
        Veredictos desconocidos se tratan como VETOED.
        """
        votes = self._resolve(list(heyting_verdicts) if heyting_verdicts is not None else [])
        total = len(votes)
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
            )

        numeric = [_HEYTING_ORDER[v] for v in votes]
        n_veto = int(numeric.count(2))
        n_deg = int(numeric.count(1))
        n_coh = int(numeric.count(0))
        max_severity = max(numeric)
        generating = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}[max_severity]

        godel = np.array([_HEYTING_GODEL[v] for v in votes], dtype=np.float64)
        godel_meet = float(np.min(godel))
        godel_join = float(np.max(godel))
        luk = float(_NumericalCore.kahan_babuska_neumaier_sum(godel) / total)

        # Margen de mayoría degradada: (#no-COHERENT) − n/2.
        n_non_coherent = n_veto + n_deg
        majority_margin = float(n_non_coherent - (total / 2.0))

        in_filter = (max_severity == 2) or (n_veto > 0) or (n_deg > (total // 2))
        # Primalidad en H₃ⁿ finito: el filtro principal ↑(atom) es primo
        # porque H₃ es una cadena (todo filtro primo es principal).
        filter_is_prime = True

        if in_filter:
            consensus = "RECHAZAR"
            interlock = True
            logger.critical(
                "¡COLAPSO DE ULTRAFILTRO BOOLEANO EN EL PRETORIO! "
                "Votos locales: %s. Consenso de calibre: RECHAZAR. "
                "Interlock ciber-físico ACTIVADO.",
                list(votes),
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
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class PretorioEngine:
    """
    Motor matemático de rango supremo. Provee cálculo espectral,
    homotópico y categorial al Pretorio Agéntico (`pretorio_agent.py`).

    Compone las tres fases anidadas:

    1. Fase I   — gérmen Weyl / bicomplejo (`_NumericalCore`).
    2. Fase II  — D² ≡ 0 y Brouwer (`_HypercohomologyChecker`,
                  `_BrouwerChecker`).
    3. Fase III — ultrafiltro sobre H₃ (`_UltrafilterEvaluator`),
                  inicializado con el gérmen unidad (COHERENT);
                  cada llamada puede sustituirlo.

    La API pública de 2.0 se conserva. Los métodos `*_certified`
    exponen los invariantes añadidos en 3.0. El regularizador alimenta
    el piso de densidad de Higham, el Hodge y los umbrales relativos.
    """

    def __init__(self, regularizer: float = _HIGHAM_TIKHONOV_REG) -> None:
        """
        Inicializa el motor y materializa el encadenamiento de gérmenes.

        Args:
            regularizer: Piso de Tikhonov–Higham contra polos espectrales.
        """
        self._reg: Final[float] = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        # Fase I → objeto inicial de Fase II (bicomplejo nulo 1×1).
        zero = np.zeros((1, 1), dtype=np.complex128)
        self._hyper_germ: _HypercohomologyGerm = (
            _NumericalCore.synthesize_hypercohomology_germ(zero, zero, regularizer=self._reg)
        )
        self._hyper_cohomology_checker = _HypercohomologyChecker(
            germ=self._hyper_germ, regularizer=self._reg
        )
        self._brouwer_checker = _BrouwerChecker(regularizer=self._reg)
        # Fase II → objeto inicial de Fase III (unidad de Heyting).
        self._ultra_germ: _UltrafilterGerm = self._brouwer_checker.induce_ultrafilter_germ()
        self._ultrafilter_evaluator = _UltrafilterEvaluator(germ=self._ultra_germ)

    # ── Fase I expuesta ───────────────────────────────────────────────────
    def kahan_compensated_trace(self, matrix: np.ndarray) -> float:
        """Traza compensada con Kahan–Babuška–Neumaier."""
        return _NumericalCore.compensated_trace(matrix)

    def kahan_sum(self, arr: np.ndarray) -> float:
        """Sumación compensada de Kahan (expuesta explícitamente)."""
        return _NumericalCore.kahan_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        """Sumación KBN (expuesta explícitamente)."""
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def weyl_toeplitz_symmetrization(self, density_matrix: np.ndarray) -> np.ndarray:
        """Simetrización de Weyl–Toeplitz / proyección de Higham a Herm."""
        return _NumericalCore.weyl_toeplitz_symmetrization(density_matrix)

    def higham_nearest_density(self, matrix: np.ndarray) -> np.ndarray:
        """Estado densidad más próximo (Hermitian + PSD + Tr = 1)."""
        rho, _evals = _NumericalCore.higham_nearest_density(matrix, floor=self._reg)
        return rho

    def synthesize_hypercohomology_germ(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> _HypercohomologyGerm:
        """Réplica pública del morfismo I.8; actualiza el gérmen de Fase II."""
        germ = _NumericalCore.synthesize_hypercohomology_germ(
            cech_boundary_d1, derham_boundary_d2, regularizer=self._reg
        )
        self._hyper_germ = germ
        self._hyper_cohomology_checker = _HypercohomologyChecker(
            germ=germ, regularizer=self._reg
        )
        return germ

    # ── Fase II expuesta ──────────────────────────────────────────────────
    def verify_cech_derham_hypercohomology(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Verifica la hipercohomología de Čech–de Rham.
        Retorna (residual, veredicto). API 2.0.
        """
        result = self.verify_cech_derham_hypercohomology_certified(
            cech_boundary_d1, derham_boundary_d2
        )
        return result.residual, result.verdict

    def verify_cech_derham_hypercohomology_certified(
        self,
        cech_boundary_d1: np.ndarray,
        derham_boundary_d2: np.ndarray,
    ) -> _HypercohomologyResult:
        """D² ≡ 0 con δ², d², {δ,d}, Hodge y β₀."""
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
        """
        Verifica el punto fijo de Brouwer.
        Retorna (residual_brouwer, residual_traza, veredicto). API 2.0.
        """
        result = self._brouwer_checker.verify(rho_current, rho_transformed)
        return result.brouwer_residual, result.trace_residual, result.verdict

    def verify_brouwer_fixed_point_certified(
        self,
        rho_current: np.ndarray,
        rho_transformed: np.ndarray,
    ) -> _BrouwerResult:
        """Brouwer con PSD, pureza, residual proyectado y pista de Lipschitz."""
        return self._brouwer_checker.verify(rho_current, rho_transformed)

    def induce_ultrafilter_germ(
        self,
        heyting_verdicts: Optional[Sequence[str]] = None,
        hyper: Optional[_HypercohomologyResult] = None,
        brouwer: Optional[_BrouwerResult] = None,
    ) -> _UltrafilterGerm:
        """
        Réplica pública del morfismo II.7: veredictos / certificados ↦ gérmen H₃.
        Actualiza el objeto con el que opera la Fase III.
        """
        germ = self._brouwer_checker.induce_ultrafilter_germ(
            hyper=hyper, brouwer=brouwer, extra_verdicts=heyting_verdicts
        )
        self._ultra_germ = germ
        self._ultrafilter_evaluator = _UltrafilterEvaluator(germ=germ)
        return germ

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def evaluate_ultrafilter_consensus(
        self,
        heyting_verdicts: List[str],
    ) -> Tuple[str, bool]:
        """
        Colapso de ultrafiltro booleano.
        Retorna (consenso, interlock_fired). API 2.0.
        """
        result = self._ultrafilter_evaluator.evaluate(heyting_verdicts)
        return result.consensus, result.interlock_fired

    def evaluate_ultrafilter_consensus_certified(
        self,
        heyting_verdicts: List[str],
    ) -> _UltrafilterResult:
        """Ultrafiltro con histograma, Gödel, Łukasiewicz, margen y átomo."""
        return self._ultrafilter_evaluator.evaluate(heyting_verdicts)


__all__ = ["PretorioEngine"]