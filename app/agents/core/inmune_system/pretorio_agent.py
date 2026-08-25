# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pretorio Agent (El Pretorio Agéntico — Comandante Supremo)          ║
║ Ruta   : app/agents/core/inmune_system/pretorio_agent.py                     ║
║ Versión: 3.0.0-Doctoral-Nested-Hodge-Brouwer-TMR-Ultrafilter-Heyting         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y DE GOBERNANZA:
────────────────────────────────────────────────────────────────────────────────
Ejerce el mando absoluto e independiente en el penthouse de la pirámide de control 
אDIKΩαWΓ. Realiza monitoreo pasivo en RAM sin introducir latencias en el ciclo OODA 
ordinario, evaluando la consistencia mediante tres pilares:

1. Hipercohomología de Čech-de Rham:
   Audita la consistencia global del bicomplejo de haces de calibre, exigiendo la 
   aniquilación de la nilpotencia diferencial total $D = d_1 + (-1)^p d_2$:
   $$D^2 = d_1 \circ d_2 + d_2 \circ d_1 \equiv \mathbf{0}$$

2. Consistencia Geodésica de Punto Fijo de Brouwer:
   Certifica que el transporte paralelo del operador densidad cuántica $\rho$ de la 
   MAC conserve el punto fijo regularizado por Weyl-Toeplitz bajo el mapeo de 
   transición $f$:
   $$f(\rho) = \rho \quad \implies \quad \|\rho - f(\rho)\|_F \equiv 0$$

3. Colapso de Ultrafiltro Booleano ($\mathcal{U}$):
   Ingiere los veredictos parciales de Heyting ($\Omega_3$) de todos los estratos 
   inferiores y evalúa si el conjunto de subcapas con veto pertenece al ultrafiltro 
   booleano no trivial:
   $$\mathcal{U} = \{A \subseteq S_{\mathrm{Capas}} \mid \nu_{\mathrm{global}}(A) = \mathtt{VETOED}\}$$
   Esta reducción monoidal colapsa la lógica intuicionista trivalente en una 
   instrucción clásica binaria dura de actuación por hardware (RECHAZAR \equiv \top) 
   enviada al ESP32.

INVARIANTES DE CATEGORÍA:
────────────────────────────────────────────────────────────────────────────────
- Monotonicidad estricta en el Poset de filtración covariante de-confinado.
- Preservación de la estructura convexa y compacta del espacio de operadores densidad.
- Unicidad y reflexividad del ultrafiltro no principal booleano de veto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Agents.PretorioAgent")

# =============================================================================
# CONSTANTES UNIVERSALES DE PRECISIÓN METROLÓGICA Y LÍMITES DE WILKINSON
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT_LIMIT: Final[float] = 1.0e-12
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_MIN_SINGULAR_VALUE_FLOOR: Final[float] = 1.0e-12
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_HYPERCOHOMOLOGY_THRESHOLD_DEFAULT: Final[float] = 1.0e-9
_HYPER_DEGRADATION_FACTOR: Final[float] = 1.0e-2
_ACYCLIC_SOFT_VETO: Final[float] = 0.5
_ACYCLIC_SOFT_DEGRADE: Final[float] = 0.05
_BROUWER_HARD_TOLERANCE: Final[float] = 1.0e-7
_BROUWER_DEGRADED_TOLERANCE: Final[float] = 1.0e-9
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_NS: Final[float] = 3.0
_CROWBAR_T_MIN_NS: Final[float] = 385.0
_CROWBAR_T_MAX_NS: Final[float] = 415.0

_LAYER_WEIGHTS: Final[Dict[str, float]] = {
    "capa_1_guards": 1.0,
    "capa_2_centurions": 1.5,
    "capa_3_tesserarios": 2.0,
    "capa_4_pretorio_hyper": 1.8,
    "capa_4_pretorio_brouwer": 1.8,
}
_TMR_LAYERS: Final[Tuple[str, ...]] = (
    "capa_1_guards",
    "capa_2_centurions",
    "capa_3_tesserarios",
)
_SUPERVISOR_LAYERS: Final[Tuple[str, ...]] = (
    "capa_4_pretorio_hyper",
    "capa_4_pretorio_brouwer",
)
_CRITICAL_ATOM: Final[str] = "capa_3_tesserarios"


# #############################################################################
#                                                                             #
#  FASE I                                                                     #
#  NÚCLEO ESPECTRAL · HODGE–DE RHAM · SIMPLEX CUÁNTICO · BROUWER              #
#                                                                             #
#  Objetos: complejo de co-cadenas (d^k), operador de densidad ρ,             #
#           mapa de transición T.                                             #
#  Morfismos: certificación d^{k+1}∘d^k=0, laplaciano de Hodge Δ^k,           #
#             números de Betti numéricos, proyección al simplex de estados,   #
#             residual de Brouwer y testigo de contracción de Banach.         #
#  Cierre formal: assemble_pretorio_jet  →  _PretorioJet                      #
#                 (dominio de todos los métodos de la Fase II).               #
#                                                                             #
# #############################################################################


class HeytingVerdict(Enum):
    """
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

    Orden de verdad (permiso / coherencia):
        VETOED ≤ DEGRADED ≤ COHERENT

    Operaciones:
        a ∧ b = min(a, b) ,   a ∨ b = max(a, b)
        a → b = ⊤ si a ≤ b, else b
        ¬a    = a → ⊥
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(self.value, other.value))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    def booleanize_closed(self) -> "HeytingVerdict":
        """Núcleo cerrado: todo lo que no es ⊤ colapsa a ⊥."""
        return self if self is HeytingVerdict.COHERENT else HeytingVerdict.VETOED

    def booleanize_open(self) -> "HeytingVerdict":
        """Doble negación ¬¬: DEGRADED ↦ COHERENT (Booleanización estándar)."""
        return self.negate().negate()

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[token]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc

    @classmethod
    def from_token_or_bottom(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[token]
        except KeyError:
            return cls.VETOED


@dataclass(frozen=True, slots=True)
class _HodgeSpectrum:
    """Espectro de Hodge–de Rham de un complejo de co-cadenas (sin veredicto)."""

    max_nilpotency: float
    nilpotency_residuals: Tuple[float, ...]
    relative_nilpotency: Tuple[float, ...]
    betti_numbers: Tuple[int, ...]
    soft_betti: Tuple[float, ...]
    hodge_gaps: Tuple[float, ...]
    hyper_obstruction: float
    complex_valid: bool
    space_dims: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BrouwerSpectrum:
    """Testigos de Brouwer / Banach sobre el simplex de densidades (sin veredicto)."""

    residual: float
    simplex_residual: float
    residual_unnormalized: float
    trace_defect: float
    positivity_defect: float
    hermiticity_defect: float
    isometry_defect: float
    lipschitz: float
    banach_contraction: bool


@dataclass(frozen=True, slots=True)
class _PretorioJet:
    """
    1-jet pretoreano inmutable. Cierre formal de la Fase I y objeto inicial
    de la Fase II (toda aduana del Pretorio factoriza por este tipo).

    Contiene espectros *brutos*. La decisión metrológica es un morfismo
    de la Fase II.
    """

    n: int
    hodge: _HodgeSpectrum
    brouwer: _BrouwerSpectrum
    input_fault: Optional[str]


class _PretorioSpectralCore:
    r"""
    Núcleo de cómputo espectral y homológico.

    Opera en la categoría de complejos de co-cadenas de dimensión finita
    sobre \(\mathbb{C}\) y en el simplex de estados

    .. math::

        \mathcal{S}_n=\{\rho=\rho^\dagger\succeq 0,\;\mathrm{Tr}\,\rho=1\},

    que es compacto y convexo en el afinado hermitiano de traza unidad
    (hipótesis de Brouwer).
    """

    # ------------------------------------------------------------------
    # I.1  Formas de Banach, Wilkinson y proyección al simplex
    # ------------------------------------------------------------------

    @staticmethod
    def frobenius(array: np.ndarray) -> float:
        return float(la.norm(np.asarray(array), "fro"))

    @staticmethod
    def hermitize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.conj().T)

    @staticmethod
    def wilkinson_floor(scale: float, dim: int) -> float:
        return max(
            abs(scale) * _MACHINE_EPS * max(dim, 1) * _WILKINSON_DEFLATION_SCALE,
            _MIN_SINGULAR_VALUE_FLOOR,
        )

    @classmethod
    def numerical_rank(cls, matrix: np.ndarray) -> int:
        arr = np.asarray(matrix)
        if arr.size == 0:
            return 0
        svals = la.svd(arr, compute_uv=False)
        if svals.size == 0:
            return 0
        floor = cls.wilkinson_floor(float(svals[0]), max(arr.shape))
        return int(np.sum(svals > floor))

    @classmethod
    def regularize_density(
        cls,
        rho: np.ndarray,
        dimension_n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Proyección de Higham al simplex cuántico \(\mathcal{S}_n\).

        Retorna ``(rho_reg, evals_norm)``.
        """
        herm = cls.hermitize(np.asarray(rho))
        evals, evecs = la.eigh(herm)
        evals_clipped = np.maximum(np.real(evals), _WILKINSON_DRIFT_LIMIT)
        trace = float(np.sum(evals_clipped))
        if trace <= _MACHINE_EPS:
            eye = np.eye(dimension_n, dtype=np.complex128)
            return eye / dimension_n, np.full(dimension_n, 1.0 / dimension_n)
        evals_norm = evals_clipped / trace
        rho_reg = evecs @ np.diag(evals_norm) @ evecs.conj().T
        return cls.hermitize(rho_reg), evals_norm

    @staticmethod
    def _as_matrix(name: str, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz; ndim={arr.ndim}.")
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/Inf.")
        return arr

    # ------------------------------------------------------------------
    # I.2  Complejo de co-cadenas, nilpotencia y Hodge
    # ------------------------------------------------------------------

    @classmethod
    def _space_dims(cls, differentials: Sequence[np.ndarray]) -> Tuple[int, ...]:
        if not differentials:
            return tuple()
        dims = [int(differentials[0].shape[1])]
        for d_k in differentials:
            dims.append(int(d_k.shape[0]))
        return tuple(dims)

    @classmethod
    def _validate_complex(
        cls,
        differentials: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], bool, Optional[str]]:
        mats: List[np.ndarray] = []
        for i, raw in enumerate(differentials):
            try:
                mats.append(cls._as_matrix(f"d^{i}", raw))
            except ValueError as exc:
                return [], False, str(exc)
        for i in range(len(mats) - 1):
            if mats[i + 1].shape[1] != mats[i].shape[0]:
                return mats, False, (
                    f"d^{i+1}∘d^{i} incompatible: "
                    f"{mats[i + 1].shape} ∘ {mats[i].shape}."
                )
        return mats, True, None

    @classmethod
    def nilpotency_residuals(
        cls,
        differentials: Sequence[np.ndarray],
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        r"""
        Residuales de la axioma de complejo
        :math:`d^{k+1}\circ d^k=0`:

        .. math::

            \varepsilon_k=\|d^{k+1}d^k\|_F,\qquad
            \varepsilon_k^{\mathrm{rel}}
              =\varepsilon_k\big/\bigl(\|d^{k+1}\|_F\|d^k\|_F\bigr).
        """
        abs_res: List[float] = []
        rel_res: List[float] = []
        for i in range(len(differentials) - 1):
            d_k = differentials[i]
            d_k1 = differentials[i + 1]
            product = d_k1 @ d_k
            residual = cls.frobenius(product)
            denom = cls.frobenius(d_k1) * cls.frobenius(d_k)
            if denom > _MIN_SINGULAR_VALUE_FLOOR:
                relative = residual / denom
            else:
                relative = 0.0 if residual <= _MIN_SINGULAR_VALUE_FLOOR else float("inf")
            abs_res.append(residual)
            rel_res.append(float(relative))
        return tuple(abs_res), tuple(rel_res)

    @classmethod
    def hodge_laplacian(
        cls,
        differentials: Sequence[np.ndarray],
        space_dims: Sequence[int],
        degree: int,
    ) -> np.ndarray:
        r"""
        Laplaciano de Hodge en grado :math:`k` (métrica euclídea):

        .. math::

            \Delta^k
              = d^{k-1}(d^{k-1})^\dagger
              + (d^k)^\dagger d^k.
        """
        dim_k = int(space_dims[degree])
        delta = np.zeros((dim_k, dim_k), dtype=np.complex128)
        if degree >= 1:
            d_prev = np.asarray(differentials[degree - 1])
            delta += d_prev @ d_prev.conj().T
        if degree <= len(differentials) - 1:
            d_k = np.asarray(differentials[degree])
            delta += d_k.conj().T @ d_k
        return cls.hermitize(delta)

    @classmethod
    def hodge_invariants(
        cls,
        differentials: Sequence[np.ndarray],
        space_dims: Sequence[int],
    ) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]]:
        r"""
        Números de Betti numéricos, Betti suaves y gaps espectrales.

        Por Hodge en dimensión finita,
        :math:`\dim H^k=\dim\ker\Delta^k=\#\{\lambda_i(\Delta^k)\le\varepsilon\}`.

        Betti suave: :math:`\sum_i\exp(-\lambda_i/\varepsilon)` (masa armónica).
        """
        betti: List[int] = []
        soft: List[float] = []
        gaps: List[float] = []
        for k, dim_k in enumerate(space_dims):
            if dim_k <= 0:
                betti.append(0)
                soft.append(0.0)
                gaps.append(0.0)
                continue
            delta = cls.hodge_laplacian(differentials, space_dims, k)
            evals = np.real(la.eigh(delta, eigvals_only=True))
            scale = float(np.max(np.abs(evals))) if evals.size else 0.0
            floor = cls.wilkinson_floor(scale, dim_k)
            kernel = evals <= floor
            betti.append(int(np.sum(kernel)))
            soft.append(float(np.sum(np.exp(-np.maximum(evals, 0.0) / floor))))
            positive = evals[evals > floor]
            gaps.append(float(np.min(positive)) if positive.size else 0.0)
        return tuple(betti), tuple(soft), tuple(gaps)

    @classmethod
    def compute_hodge_spectrum(
        cls,
        cochain_complex_matrices: Sequence[np.ndarray],
    ) -> _HodgeSpectrum:
        r"""
        Espectro completo del complejo (o del Tot Čech–de Rham ya ensamblado).

        La obstrucción de hipercohomología positiva es

        .. math::

            \mathrm{obs}
              =\max_k\varepsilon_k
               +\sum_{k>0}\beta_k^{\mathrm{suave}}.
        """
        empty = _HodgeSpectrum(
            max_nilpotency=0.0,
            nilpotency_residuals=tuple(),
            relative_nilpotency=tuple(),
            betti_numbers=tuple(),
            soft_betti=tuple(),
            hodge_gaps=tuple(),
            hyper_obstruction=0.0,
            complex_valid=True,
            space_dims=tuple(),
        )
        if len(cochain_complex_matrices) == 0:
            return empty

        mats, valid, fault = cls._validate_complex(cochain_complex_matrices)
        if not mats:
            return _HodgeSpectrum(
                max_nilpotency=float("inf"),
                nilpotency_residuals=(float("inf"),),
                relative_nilpotency=(float("inf"),),
                betti_numbers=tuple(),
                soft_betti=tuple(),
                hodge_gaps=tuple(),
                hyper_obstruction=float("inf"),
                complex_valid=False,
                space_dims=tuple(),
            )

        if not valid:
            logger.error("Complejo de co-cadenas inválido: %s", fault)
            return _HodgeSpectrum(
                max_nilpotency=float("inf"),
                nilpotency_residuals=(float("inf"),),
                relative_nilpotency=(float("inf"),),
                betti_numbers=tuple(),
                soft_betti=tuple(),
                hodge_gaps=tuple(),
                hyper_obstruction=float("inf"),
                complex_valid=False,
                space_dims=cls._space_dims(mats) if mats else tuple(),
            )

        abs_res, rel_res = cls.nilpotency_residuals(mats)
        max_nil = max(abs_res) if abs_res else 0.0
        dims = cls._space_dims(mats)
        betti, soft, gaps = cls.hodge_invariants(mats, dims)
        positive_mass = float(sum(soft[1:])) if len(soft) > 1 else 0.0
        obstruction = float(max_nil) + positive_mass
        structure_ok = all(
            (not np.isfinite(r)) or r <= _STRUCTURE_ATOL for r in abs_res
        )
        return _HodgeSpectrum(
            max_nilpotency=float(max_nil),
            nilpotency_residuals=abs_res,
            relative_nilpotency=rel_res,
            betti_numbers=betti,
            soft_betti=soft,
            hodge_gaps=gaps,
            hyper_obstruction=float(obstruction),
            complex_valid=bool(structure_ok),
            space_dims=dims,
        )

    # ------------------------------------------------------------------
    # I.3  Brouwer sobre el simplex de densidades
    # ------------------------------------------------------------------

    @classmethod
    def compute_brouwer_spectrum(
        cls,
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        dimension_n: int,
    ) -> _BrouwerSpectrum:
        r"""
        Testigos del endomorfismo

        .. math::

            f:\mathcal{S}_n\to\mathcal{S}_n,\qquad
            f(\rho)=\frac{T\rho T^\dagger}{\mathrm{Tr}(T\rho T^\dagger)}.

        \(\mathcal{S}_n\) es compacto convexo ⇒ Brouwer garantiza un punto
        fijo; aquí se audita si el *estado actual* lo es.

        El residual histórico (compatibilidad 2.0)

        .. math::

            \|T\rho T^\dagger-\rho\|_F+|\mathrm{Tr}(T\rho T^\dagger)-1|

        se conserva. El residual geométrico es
        \(\|f(\rho)-\rho\|_F\). El testigo de Banach es
        \(\mathrm{Lip}(g)=\|T\|_2^2<1\) para \(g(\rho)=T\rho T^\dagger\).
        """
        inf = _BrouwerSpectrum(
            residual=float("inf"),
            simplex_residual=float("inf"),
            residual_unnormalized=float("inf"),
            trace_defect=float("inf"),
            positivity_defect=float("inf"),
            hermiticity_defect=float("inf"),
            isometry_defect=float("inf"),
            lipschitz=float("inf"),
            banach_contraction=False,
        )
        try:
            raw = np.asarray(density_matrix)
            if raw.shape != (dimension_n, dimension_n):
                raise ValueError(
                    f"density_matrix de forma {raw.shape}, esperada "
                    f"({dimension_n}, {dimension_n})."
                )
            if not np.all(np.isfinite(raw)):
                raise ValueError("density_matrix contiene NaN/Inf.")
            t_map = cls._as_matrix("transition_map_matrix", transition_map_matrix)
            if t_map.shape != (dimension_n, dimension_n):
                raise ValueError(
                    f"transition_map_matrix de forma {t_map.shape}, esperada "
                    f"({dimension_n}, {dimension_n})."
                )
        except ValueError as exc:
            logger.error("Espectro de Brouwer inválido: %s", exc)
            return inf

        herm_def = cls.frobenius(raw - raw.conj().T)
        raw_evals = np.real(la.eigh(cls.hermitize(raw), eigvals_only=True))
        pos_def = float(np.sum(np.clip(-raw_evals, 0.0, None)))

        rho_reg, _ = cls.regularize_density(raw, dimension_n)
        sigma = cls.hermitize(t_map @ rho_reg @ t_map.conj().T)
        trace_val = float(np.real(np.trace(sigma)))
        trace_defect = abs(trace_val - 1.0)
        residual_unnorm = cls.frobenius(sigma - rho_reg)
        residual = residual_unnorm + trace_defect

        if trace_val > _MACHINE_EPS:
            simplex_residual = cls.frobenius(sigma / trace_val - rho_reg)
        else:
            simplex_residual = float("inf")

        eye = np.eye(dimension_n, dtype=t_map.dtype)
        iso_def = cls.frobenius(t_map.conj().T @ t_map - eye)
        lipschitz = float(la.norm(t_map, 2)) ** 2
        banach = bool(np.isfinite(lipschitz) and lipschitz < 1.0 - 1.0e-12)

        return _BrouwerSpectrum(
            residual=float(residual),
            simplex_residual=float(simplex_residual),
            residual_unnormalized=float(residual_unnorm),
            trace_defect=float(trace_defect),
            positivity_defect=float(pos_def),
            hermiticity_defect=float(herm_def),
            isometry_defect=float(iso_def),
            lipschitz=float(lipschitz),
            banach_contraction=banach,
        )

    # ------------------------------------------------------------------
    # I.4  Álgebra de decisión (Heyting / mediana / Booleanización)
    #      Las políticas de Capa 4 se instancian en la Fase II.
    # ------------------------------------------------------------------

    @staticmethod
    def heyting_meet_tokens(tokens: Sequence[str]) -> HeytingVerdict:
        acc = HeytingVerdict.COHERENT
        for token in tokens:
            acc = acc.meet(HeytingVerdict.from_token_or_bottom(token))
        return acc

    @staticmethod
    def heyting_lower_median(tokens: Sequence[str]) -> HeytingVerdict:
        """
        Mediana inferior en G₃ (TMR valorado en retículo).

        Para \(n=3\) es la mediana ordinaria. Para \(n\) par elige el
        testigo más restrictivo (no el más permisivo).
        """
        if not tokens:
            return HeytingVerdict.COHERENT
        ordered = sorted(
            HeytingVerdict.from_token_or_bottom(token).value for token in tokens
        )
        return HeytingVerdict(ordered[(len(ordered) - 1) // 2])

    @staticmethod
    def weighted_social_choice(
        layer_verdicts: Dict[str, str],
    ) -> Tuple[HeytingVerdict, Dict[str, float], Tuple[str, ...]]:
        """
        Ancilla de elección social (NO es un ultrafiltro).

        Empates se rompen hacia VETOED. Veredictos anómalos se leen como
        ⊥ y duplican el peso de la capa (amplificador de anomalía 2.0).
        """
        masses = {verdict.name: 0.0 for verdict in HeytingVerdict}
        anomalies: List[str] = []
        for layer, token in layer_verdicts.items():
            weight = float(_LAYER_WEIGHTS.get(layer, 1.0))
            if token not in HeytingVerdict.__members__:
                verdict = HeytingVerdict.VETOED
                weight *= 2.0
                anomalies.append(layer)
            else:
                verdict = HeytingVerdict[token]
            masses[verdict.name] += weight
        winner = max(
            list(HeytingVerdict),
            key=lambda verdict: (masses[verdict.name], -verdict.value),
        )
        return winner, masses, tuple(anomalies)

    @classmethod
    def compute_all_pretorio_metrics(
        cls,
        cochain_complex_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        dimension_n: int,
    ) -> Tuple[float, float, float]:
        """
        Proyección 2.0 del 1-jet: (max_nilpotency, residual_Brouwer, defecto_traza).
        Conservada como testigo de compatibilidad; el objeto formal es el jet.
        """
        hodge = cls.compute_hodge_spectrum(cochain_complex_matrices)
        brouwer = cls.compute_brouwer_spectrum(
            density_matrix, transition_map_matrix, dimension_n
        )
        return hodge.max_nilpotency, brouwer.residual, brouwer.trace_defect

    # ------------------------------------------------------------------
    # I.ω  ÚLTIMO MORFISMO DE LA FASE I
    #      Codominio ≡ dominio de PretorioAgent (Fase II)
    # ------------------------------------------------------------------

    @classmethod
    def assemble_pretorio_jet(
        cls,
        dimension_n: int,
        cochain_complex_matrices: Sequence[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
    ) -> _PretorioJet:
        r"""
        Cierre formal de la Fase I / unidad de la adjunción con la Fase II.

        Congela el espectro de Hodge del complejo de calibre y los testigos
        de Brouwer del estado MAC en un 1-jet inmutable. No decide
        veredictos: la decisión es un morfismo de la Fase II sobre este objeto.
        """
        faults: List[str] = []
        if dimension_n <= 0:
            faults.append(f"dimension_n={dimension_n} no es positiva")

        try:
            hodge = cls.compute_hodge_spectrum(cochain_complex_matrices)
            if not hodge.complex_valid and hodge.max_nilpotency == float("inf"):
                faults.append("hodge:complejo_invalido")
        except (ValueError, np.linalg.LinAlgError) as exc:
            faults.append(f"hodge:{exc}")
            hodge = _HodgeSpectrum(
                max_nilpotency=float("inf"),
                nilpotency_residuals=(float("inf"),),
                relative_nilpotency=(float("inf"),),
                betti_numbers=tuple(),
                soft_betti=tuple(),
                hodge_gaps=tuple(),
                hyper_obstruction=float("inf"),
                complex_valid=False,
                space_dims=tuple(),
            )

        try:
            brouwer = cls.compute_brouwer_spectrum(
                density_matrix, transition_map_matrix, dimension_n
            )
            if not np.isfinite(brouwer.residual):
                faults.append("brouwer:residual_no_finito")
        except (ValueError, np.linalg.LinAlgError) as exc:
            faults.append(f"brouwer:{exc}")
            brouwer = _BrouwerSpectrum(
                residual=float("inf"),
                simplex_residual=float("inf"),
                residual_unnormalized=float("inf"),
                trace_defect=float("inf"),
                positivity_defect=float("inf"),
                hermiticity_defect=float("inf"),
                isometry_defect=float("inf"),
                lipschitz=float("inf"),
                banach_contraction=False,
            )

        return _PretorioJet(
            n=int(dimension_n),
            hodge=hodge,
            brouwer=brouwer,
            input_fault=("|".join(faults) if faults else None),
        )


# #############################################################################
#                                                                             #
#  FASE II                                                                    #
#  PRETORIO · ADUANAS DE HIPERCOHOMOLOGÍA / BROUWER / TMR+ULTRAFILTRO         #
#                                                                             #
#  Continuación directa del último morfismo de la Fase I:                     #
#      _PretorioJet  ↦  PretorioAgent                                         #
#                                                                             #
#  Cierre formal: compile_pretorio_edict  →  _PretorioEdict                   #
#                 (dominio de la Cámara de Coherencia, Fase III).             #
#                                                                             #
# #############################################################################


@dataclass(frozen=True, slots=True)
class _HypercohomologyResult:
    """Resultado de la aduana de hipercohomología (con veredicto)."""

    max_residual: float
    hyper_obstruction: float
    betti_numbers: Tuple[int, ...]
    soft_betti: Tuple[float, ...]
    hodge_gaps: Tuple[float, ...]
    nilpotency_residuals: Tuple[float, ...]
    complex_valid: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class _BrouwerResult:
    """Resultado de la aduana de Brouwer (con veredicto)."""

    residual: float
    simplex_residual: float
    trace_defect: float
    positivity_defect: float
    hermiticity_defect: float
    isometry_defect: float
    lipschitz: float
    banach_contraction: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class _UltrafilterResult:
    """
    Colapso de decisión de Capa 4.

    `global_verdict` es el ínfimo de (mediana TMR, átomo crítico, supervisor).
    `principal_atom` nombra el generador del ultrafiltro principal que forzó
    el interlock, o None si el interlock no dispara.
    `weighted_ancilla` es elección social, no un ultrafiltro.
    """

    global_verdict: str
    heyting_meet: str
    tmr_median: str
    supervisor_meet: str
    boolean_closed: str
    principal_atom: Optional[str]
    hardware_interlock: bool
    weighted_ancilla: str
    vote_masses: Dict[str, float]
    anomalies: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PretorioEdict:
    """
    Edicto pretoreano. Cierre de la Fase II y objeto inicial de la Fase III
    (la Cámara no recompute espectros: consume este morfismo ya certificado).
    """

    jet: _PretorioJet
    hyper: _HypercohomologyResult
    brouwer: _BrouwerResult
    ultrafilter: _UltrafilterResult
    extended_verdicts: Dict[str, str]


class PretorioAgent:
    """
    El Pretorio Agéntico (Capa 4 — Comandante Supremo de Seguridad).

    Consume el `_PretorioJet` de la Fase I y evalúa aciclicidad de calibre,
    consistencia de punto fijo y el colapso TMR+ultrafiltro sobre el retículo
    de Heyting de las capas inferiores.
    """

    def __init__(
        self,
        dimension_n: int,
        hypercohomology_threshold: float = _HYPERCOHOMOLOGY_THRESHOLD_DEFAULT,
        safety_margin: float = 1.0,
        require_positive_vanishing: bool = False,
    ) -> None:
        """
        Inicio formal de la Fase II.

        `require_positive_vanishing=True` endurece la aduana de hipercohomología
        hasta exigir \(\mathbb{H}^{k>0}=0\) (clases virtuales nulas). Por
        defecto se *reporta* la masa armónica y se degrada, sin vetar, para
        no romper el contrato 2.0 (que sólo exigía \(d^2=0\)).
        """
        if dimension_n <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        if hypercohomology_threshold <= 0.0:
            raise ValueError("hypercohomology_threshold debe ser estrictamente positivo.")
        if safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser estrictamente positivo.")

        self._n: Final[int] = int(dimension_n)
        self._threshold: Final[float] = float(hypercohomology_threshold)
        self._safety_margin: Final[float] = float(safety_margin)
        self._require_positive_vanishing: Final[bool] = bool(require_positive_vanishing)

    def ingest_observables(
        self,
        cochain_complex_matrices: Sequence[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
    ) -> _PretorioJet:
        """Reenvía las observables del ciclo al último morfismo de la Fase I."""
        return _PretorioSpectralCore.assemble_pretorio_jet(
            dimension_n=self._n,
            cochain_complex_matrices=cochain_complex_matrices,
            density_matrix=density_matrix,
            transition_map_matrix=transition_map_matrix,
        )

    def _metric_verdict(
        self,
        metric: float,
        hard: float,
        degraded: float,
    ) -> str:
        hard_tol = float(hard) * self._safety_margin
        deg_tol = float(degraded) * self._safety_margin
        if (not np.isfinite(metric)) or metric > hard_tol:
            return "VETOED"
        if metric > deg_tol:
            return "DEGRADED"
        return "COHERENT"

    def _hyper_from_spectrum(self, spectrum: _HodgeSpectrum) -> _HypercohomologyResult:
        if (not spectrum.complex_valid) or (not np.isfinite(spectrum.max_nilpotency)):
            nil_verdict = "VETOED"
        else:
            nil_verdict = self._metric_verdict(
                spectrum.max_nilpotency,
                self._threshold,
                self._threshold * _HYPER_DEGRADATION_FACTOR,
            )

        positive_mass = (
            float(sum(spectrum.soft_betti[1:])) if len(spectrum.soft_betti) > 1 else 0.0
        )
        if not np.isfinite(positive_mass):
            acyc_verdict = "VETOED"
        elif positive_mass > _ACYCLIC_SOFT_VETO:
            acyc_verdict = "VETOED" if self._require_positive_vanishing else "DEGRADED"
        elif positive_mass > _ACYCLIC_SOFT_DEGRADE:
            acyc_verdict = "DEGRADED"
        else:
            acyc_verdict = "COHERENT"

        verdict = (
            HeytingVerdict.from_token(nil_verdict)
            .meet(HeytingVerdict.from_token(acyc_verdict))
            .name
        )
        return _HypercohomologyResult(
            max_residual=float(spectrum.max_nilpotency),
            hyper_obstruction=float(spectrum.hyper_obstruction),
            betti_numbers=spectrum.betti_numbers,
            soft_betti=spectrum.soft_betti,
            hodge_gaps=spectrum.hodge_gaps,
            nilpotency_residuals=spectrum.nilpotency_residuals,
            complex_valid=spectrum.complex_valid,
            verdict=verdict,
        )

    def _brouwer_from_spectrum(self, spectrum: _BrouwerSpectrum) -> _BrouwerResult:
        residual_verdict = self._metric_verdict(
            spectrum.residual,
            _BROUWER_HARD_TOLERANCE,
            _BROUWER_DEGRADED_TOLERANCE,
        )
        if (not np.isfinite(spectrum.trace_defect)) or (
            spectrum.trace_defect > _WILKINSON_DRIFT_LIMIT * self._safety_margin
        ):
            residual_verdict = "VETOED"
        if (not np.isfinite(spectrum.positivity_defect)) or (
            spectrum.positivity_defect > _BROUWER_HARD_TOLERANCE * self._safety_margin
        ):
            residual_verdict = "VETOED"
        if not np.isfinite(spectrum.simplex_residual):
            residual_verdict = "VETOED"

        return _BrouwerResult(
            residual=float(spectrum.residual),
            simplex_residual=float(spectrum.simplex_residual),
            trace_defect=float(spectrum.trace_defect),
            positivity_defect=float(spectrum.positivity_defect),
            hermiticity_defect=float(spectrum.hermiticity_defect),
            isometry_defect=float(spectrum.isometry_defect),
            lipschitz=float(spectrum.lipschitz),
            banach_contraction=bool(spectrum.banach_contraction),
            verdict=residual_verdict,
        )

    def audit_calibre_hypercohomology(
        self,
        cochain_complex_matrices: List[np.ndarray],
        jet: Optional[_PretorioJet] = None,
    ) -> _HypercohomologyResult:
        r"""
        [PRETORIO — HIPERCOHOMOLOGÍA DE ČECH–DE RHAM / HODGE]

        Evalúa el complejo (o el Tot ya ensamblado) y exige, en la medida
        numérica de Wilkinson,

        .. math::

            d^{k+1}\circ d^k=0,
            \qquad
            \mathbb{H}^{k>0}(\mathcal{U};\mathcal{F}^\bullet)
              \cong\ker\Delta^{k>0}\approx 0

        (lo segundo sólo veta si `require_positive_vanishing`; si no, degrada).
        """
        if jet is not None:
            if jet.n != self._n:
                logger.error("Jet de dimensión %d, pretoreo n=%d.", jet.n, self._n)
            return self._hyper_from_spectrum(jet.hodge)
        if len(cochain_complex_matrices) < 2:
            spectrum = _PretorioSpectralCore.compute_hodge_spectrum(
                cochain_complex_matrices
            )
            if len(cochain_complex_matrices) < 2 and spectrum.complex_valid:
                return self._hyper_from_spectrum(spectrum)
        spectrum = _PretorioSpectralCore.compute_hodge_spectrum(cochain_complex_matrices)
        return self._hyper_from_spectrum(spectrum)

    def verify_brouwer_fixed_point_consistency(
        self,
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        jet: Optional[_PretorioJet] = None,
    ) -> _BrouwerResult:
        r"""
        [PRETORIO — PUNTO FIJO DE BROUWER SOBRE \(\mathcal{S}_n\)]

        Audita \(f(\rho)=\rho\) para
        \(f(\rho)=T\rho T^\dagger/\mathrm{Tr}(T\rho T^\dagger)\).
        Un defecto de traza por encima del límite de Wilkinson, pérdida de
        positividad o un residual por encima de \(\tau_{\mathrm{Brouwer}}\)
        se leen como reparametrización espuria de la geodésica semántica.
        """
        if jet is not None:
            return self._brouwer_from_spectrum(jet.brouwer)
        spectrum = _PretorioSpectralCore.compute_brouwer_spectrum(
            density_matrix, transition_map_matrix, self._n
        )
        return self._brouwer_from_spectrum(spectrum)

    def evaluate_global_boolean_ultrafilter(
        self,
        layer_verdicts: Dict[str, str],
    ) -> _UltrafilterResult:
        r"""
        [PRETORIO — TMR + ULTRAFILTRO PRINCIPAL + ÍNFIMO DE HEYTING]

        Tres morfismos, explícitamente separados:

        1. \(\nu_\wedge=\bigwedge_\ell\nu_\ell\) en G₃.
        2. \(\nu_{\mathrm{TMR}}=\mathrm{mediana}(\nu_1,\nu_2,\nu_3)\).
        3. Ultrafiltro principal \(\mathcal{U}_\tau\) generado por el átomo
           crítico \(\tau=\) Tesserarios: el interlock dispara si
           \(\{\tau\}\in\mathcal{U}_\tau\) con \(\nu_\tau=\bot\), o si el
           supervisor (autoauditorías de Capa 4) es \(\bot\), o si el TMR
           es \(\bot\).

        El voto ponderado se emite sólo como ancilla (social choice).
        """
        core = _PretorioSpectralCore
        tokens = tuple(layer_verdicts.values())
        heyting = core.heyting_meet_tokens(tokens)

        tmr_present = tuple(
            layer_verdicts[layer] for layer in _TMR_LAYERS if layer in layer_verdicts
        )
        tmr = core.heyting_lower_median(tmr_present)

        supervisor_present = tuple(
            layer_verdicts[layer]
            for layer in _SUPERVISOR_LAYERS
            if layer in layer_verdicts
        )
        supervisor = core.heyting_meet_tokens(supervisor_present)

        critical: Optional[HeytingVerdict] = None
        if _CRITICAL_ATOM in layer_verdicts:
            critical = HeytingVerdict.from_token_or_bottom(
                layer_verdicts[_CRITICAL_ATOM]
            )

        global_h = tmr.meet(supervisor)
        if critical is not None:
            global_h = global_h.meet(critical)

        boolean_closed = global_h.booleanize_closed()

        principal_atom: Optional[str] = None
        if critical is HeytingVerdict.VETOED:
            principal_atom = _CRITICAL_ATOM
        elif supervisor is HeytingVerdict.VETOED:
            for layer in _SUPERVISOR_LAYERS:
                if layer_verdicts.get(layer) == "VETOED":
                    principal_atom = layer
                    break
        elif tmr is HeytingVerdict.VETOED:
            principal_atom = "tmr_majority"

        interlock = boolean_closed is HeytingVerdict.VETOED
        weighted, masses, anomalies = core.weighted_social_choice(layer_verdicts)

        return _UltrafilterResult(
            global_verdict=global_h.name,
            heyting_meet=heyting.name,
            tmr_median=tmr.name,
            supervisor_meet=supervisor.name,
            boolean_closed=boolean_closed.name,
            principal_atom=principal_atom,
            hardware_interlock=bool(interlock),
            weighted_ancilla=weighted.name,
            vote_masses=dict(masses),
            anomalies=anomalies,
        )

    # ------------------------------------------------------------------
    # II.ω  ÚLTIMO MORFISMO DE LA FASE II
    #       Codominio ≡ dominio de PretorioCoherenceChamber (Fase III)
    # ------------------------------------------------------------------

    def compile_pretorio_edict(
        self,
        jet: _PretorioJet,
        layer_verdicts: Dict[str, str],
    ) -> _PretorioEdict:
        """
        Cierre formal de la Fase II / unidad de la adjunción con la Fase III.

        Pega las aduanas de hipercohomología y Brouwer sobre el 1-jet,
        enriquece el vector de capas con los autoveredictos pretoreanos y
        colapsa el ultrafiltro. Si el jet declara ``input_fault``, las
        aduanas correspondientes ya viajan como VETOED (residuales infinitos).
        """
        if jet.n != self._n:
            logger.error("Jet de dimensión %d incompatible con n=%d.", jet.n, self._n)
        if jet.input_fault:
            logger.error("Jet ensamblado con fallos de entrada: %s", jet.input_fault)

        hyper = self._hyper_from_spectrum(jet.hodge)
        brouwer = self._brouwer_from_spectrum(jet.brouwer)

        extended = dict(layer_verdicts)
        extended["capa_4_pretorio_hyper"] = hyper.verdict
        extended["capa_4_pretorio_brouwer"] = brouwer.verdict

        ultra = self.evaluate_global_boolean_ultrafilter(extended)
        return _PretorioEdict(
            jet=jet,
            hyper=hyper,
            brouwer=brouwer,
            ultrafilter=ultra,
            extended_verdicts=extended,
        )


# #############################################################################
#                                                                             #
#  FASE III                                                                   #
#  CÁMARA DE COHERENCIA · OODA · CROWBAR BT151                                #
#                                                                             #
#  Continuación directa del último morfismo de la Fase II:                    #
#      _PretorioEdict  ↦  PretorioCoherenceChamber                            #
#                                                                             #
# #############################################################################


@dataclass
class _ThyristorCrowbar:
    r"""
    Modelo lumped del tiristor BT151 como bypass de silicio.

    Física de circuito (actuador de seguridad de Capa 4, no un exploit):
      • disparo de puerta → enganche mientras \(I_A>I_H\);
      • latencia de puerta \(\sim 400\,\mathrm{ns}\) con jitter térmico
        \(\delta t\sim\mathcal{N}(0,\sigma^2)\).
    Una vez enganchado permanece conductor durante el ciclo OODA.
    """

    latched: bool = False
    last_latency_ns: float = 0.0

    def fire(self, rng: np.random.Generator) -> float:
        jitter = float(rng.normal(0.0, _CROWBAR_JITTER_NS))
        latency = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_T_MIN_NS,
                _CROWBAR_T_MAX_NS,
            )
        )
        self.latched = True
        self.last_latency_ns = latency
        return latency

    def reset(self) -> None:
        self.latched = False
        self.last_latency_ns = 0.0


class PretorioCoherenceChamber:
    """
    Cámara de Coherencia de la Capa 4.

    Consume el `_PretorioEdict` (cierre de la Fase II) y dispara el crowbar
    si el ultrafiltro principal / TMR / supervisor colapsa a ⊥.
    """

    def __init__(
        self,
        agent: PretorioAgent,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.agent = agent
        self._crowbar = _ThyristorCrowbar()
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    @classmethod
    def assemble_from_spectral_seed(
        cls,
        dimension_n: int,
        hypercohomology_threshold: float = _HYPERCOHOMOLOGY_THRESHOLD_DEFAULT,
        safety_margin: float = 1.0,
        require_positive_vanishing: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> "PretorioCoherenceChamber":
        """
        Composición functorial Fase I → Fase II → Fase III.

        El agente (II.0) queda listo para consumir jets de `assemble_pretorio_jet`
        (I.ω); la cámara es el objeto terminal del topos de seguridad pretoreano.
        """
        agent = PretorioAgent(
            dimension_n=dimension_n,
            hypercohomology_threshold=hypercohomology_threshold,
            safety_margin=safety_margin,
            require_positive_vanishing=require_positive_vanishing,
        )
        return cls(agent, rng=rng)

    def fuse_and_actuate(self, edict: _PretorioEdict) -> Dict[str, Any]:
        """
        Ciclo OODA unificado de Capa 4 sobre un edicto ya compilado.

        1. Observar  — espectros del 1-jet (Fase I.ω).
        2. Orientar  — aduanas de Hodge y Brouwer (Fase II).
        3. Decidir   — TMR ∧ 𝒰_τ ∧ supervisor (Fase II.ω).
        4. Actuar    — latch del BT151 si el colapso es ⊥.
        """
        ultra = edict.ultrafilter
        latency_ns = 0.0
        interlock_fired = False
        if ultra.hardware_interlock:
            latency_ns = self._crowbar.fire(self._rng)
            interlock_fired = True
            logger.critical(
                "[EL PRETORIO AGÉNTICO — VETO SUPREMO] Colapso del ultrafiltro "
                "detectado (átomo=%s, TMR=%s, ⋀=%s). Bypass de potencia BT151 "
                "[GPIO14] despachado en %.2f ns via ISR en IRAM. Obra real "
                "paralizada incondicionalmente. ε_ℍ=%.3e  ε_B=%.3e  β=%s",
                ultra.principal_atom,
                ultra.tmr_median,
                ultra.heyting_meet,
                latency_ns,
                edict.hyper.max_residual,
                edict.brouwer.residual,
                edict.hyper.betti_numbers,
            )

        global_h = HeytingVerdict.from_token(ultra.global_verdict)
        return {
            "pretorio_global_verdict": ultra.global_verdict,
            "heyting_value": global_h.value,
            "heyting_meet": ultra.heyting_meet,
            "tmr_median": ultra.tmr_median,
            "supervisor_meet": ultra.supervisor_meet,
            "boolean_closed": ultra.boolean_closed,
            "ultrafilter_principal_atom": ultra.principal_atom,
            "weighted_vote_ancilla": ultra.weighted_ancilla,
            "weighted_vote_masses": dict(ultra.vote_masses),
            "ultrafilter_anomalies": list(ultra.anomalies),
            "hypercohomology_max_residual": edict.hyper.max_residual,
            "hypercohomology_obstruction": edict.hyper.hyper_obstruction,
            "hypercohomology_betti": list(edict.hyper.betti_numbers),
            "hypercohomology_soft_betti": list(edict.hyper.soft_betti),
            "hypercohomology_hodge_gaps": list(edict.hyper.hodge_gaps),
            "hypercohomology_nilpotency": list(edict.hyper.nilpotency_residuals),
            "hypercohomology_complex_valid": edict.hyper.complex_valid,
            "hypercohomology_verdict": edict.hyper.verdict,
            "brouwer_fixed_point_residual": edict.brouwer.residual,
            "brouwer_simplex_residual": edict.brouwer.simplex_residual,
            "brouwer_trace_defect": edict.brouwer.trace_defect,
            "brouwer_positivity_defect": edict.brouwer.positivity_defect,
            "brouwer_hermiticity_defect": edict.brouwer.hermiticity_defect,
            "brouwer_isometry_defect": edict.brouwer.isometry_defect,
            "brouwer_lipschitz": edict.brouwer.lipschitz,
            "brouwer_banach_contraction": edict.brouwer.banach_contraction,
            "brouwer_verdict": edict.brouwer.verdict,
            "hardware_interlock_fired": interlock_fired,
            "hardware_crowbar_latched": self._crowbar.latched,
            "actuation_latency_ns": latency_ns,
            "input_fault": edict.jet.input_fault,
            "audited_layers_audit_trail": dict(edict.extended_verdicts),
        }

    def process_supervision_cycle(
        self,
        cochain_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map: np.ndarray,
        layer_verdicts: Dict[str, str],
    ) -> Dict[str, Any]:
        """Atajo I.ω → II.ω → III sobre observables crudas de un ciclo."""
        jet = self.agent.ingest_observables(
            cochain_matrices, density_matrix, transition_map
        )
        edict = self.agent.compile_pretorio_edict(jet, layer_verdicts)
        return self.fuse_and_actuate(edict)


def execute_pretorio_supervision_cycle(
    agent: PretorioAgent,
    cochain_matrices: List[np.ndarray],
    density_matrix: np.ndarray,
    transition_map: np.ndarray,
    layer_verdicts: Dict[str, str],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Fachada de compatibilidad: delega en la Cámara de la Fase III."""
    chamber = PretorioCoherenceChamber(agent, rng=rng)
    return chamber.process_supervision_cycle(
        cochain_matrices, density_matrix, transition_map, layer_verdicts
    )


def _bind_agent_cycle() -> None:
    def _cycle(
        self: PretorioAgent,
        cochain_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map: np.ndarray,
        layer_verdicts: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Orquesta el ciclo de supervisión omnipresente del Pretorio.

        Composición \(I.\omega\to II.\omega\to III\): ensambla el 1-jet,
        compila el edicto y actúa. Detona el crowbar ante colapso a ⊥.
        """
        return execute_pretorio_supervision_cycle(
            self,
            cochain_matrices,
            density_matrix,
            transition_map,
            layer_verdicts,
        )

    PretorioAgent.execute_pretorio_supervision_cycle = _cycle  # type: ignore[attr-defined]


_bind_agent_cycle()


__all__ = [
    "HeytingVerdict",
    "PretorioAgent",
    "PretorioCoherenceChamber",
    "execute_pretorio_supervision_cycle",
]