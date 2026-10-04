# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Séquitos Engine (Caballos de Batalla de Consenso y Mónadas)║
║ Ruta   : app/core/inmune_system/imperial_sequitos_engine.py                  ║
║ Versión: 4.0.0-Poincare-Liouville-Maupertuis-Kahan-FPU-PhD                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Este motor táctico supervisa la propagación de variables, la integración simpléctica
de Maupertuis-Jacobi sobre variedades de Darboux (M, w), y el consenso de opinión en
sub-tríadas de-confinadas. Encapsula las deliberaciones en mónadas de Kleisli, calcula
la dinámica espectral de opinión de DeGroot y audita el canal de mensajes mediante la violación
cuántica multipartita de Bell-CHSH.

MÉTODOS GRANULARES:

1. compute_symplectic_maupertuis_step(current_z, dt, metric_G, potential_V_func, grad_V_func, total_energy_H0):
   Ejecuta un paso de integración simpléctica de Störmer-Verlet en coordenadas Darboux, midiendo
   la preservación de la medida de Liouville-Darboux y calculando la Acción de Maupertuis-Jacobi:
   $$S_M = \int \sqrt{2(H_0 - V(q))} \, |dq|_G$$

2. kleisli_compose(f: Any, g: Any) -> Any:
   Compone dos funciones de decisión monádicas f: A -> T(B) y g: B -> T(C) bajo la Mónada
   de Estado de Kleisli \mathbb{T} = (T, \eta, \mu), asegurando la asociatividad functorial del flujo:
   $$(g \bullet f)(x) = \mu_C \circ T(g) \circ f(x)$$

3. compute_degroot_spectral_consensus(affinity_matrix: np.ndarray, initial_opinions: np.ndarray, max_iter: int = 1000):
   Resuelve la convergencia exponencial de la opinión de las tríadas bajo el modelo continuo de
   DeGroot acoplado al Laplaciano de de Rham, extrayendo el valor de Fiedler \lambda_2.

4. compute_uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
   Mide la fidelidad cuántica de Uhlmann entre el estado real en RAM y el estado de referencia.

5. verify_chsh_violation(correlation_matrix: np.ndarray):
   Audita la correlación cuántica multipartita de los mensajes intercambiados en el séquito.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Core.ImperialSequitosEngine")

__version__: Final[str] = (
    "4.0.0-Poincare-Liouville-Maupertuis-Kahan-FPU-PhD"
)


# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_TIKHONOV_REG: Final[float] = 1e-15
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-12
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_WILKINSON_LIMIT: Final[float] = 1e-12
_SPECTRAL_TOL: Final[float] = 1e-9
_TOLERANCE_DEGROOT_COHERENT: Final[float] = 1e-6
_TOLERANCE_DEGROOT_DEGRADED: Final[float] = 1e-4
_TSIRELSON_BOUND: Final[float] = float(2.0 * np.sqrt(2.0))  # 2√2
_CLASSICAL_CHSH_BOUND: Final[float] = 2.0
_PR_NOSIGNAL_BOUND: Final[float] = 4.0
_LOG_EXP_CLIP: Final[float] = 700.0
_CORRELATOR_BOUND: Final[float] = 1.0

_PAULI: Final[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
)


@dataclass(frozen=True)
class SequitosEngineStepResult:
    r"""
    DTO inmutable de salida de la FPU para la integración simpléctica de Séquitos.
    """
    next_state_z: NDArray[np.float64]
    hamiltonian_energy: float
    volume_drift: float
    maupertuis_action: float
    is_liouville_conserved: bool


# =============================================================================
# FASE I — NÚCLEO DE BANACH, MÓNADA WRITER Y KLEISLI–GIRY
# -----------------------------------------------------------------------------
# Objetos: sumas compensadas, normas, Higham, morfismos A → T(B) de la
#          mónada Writer_([0,1],×) y núcleos estocásticos (Giry finito).
# Morfismo terminal (I.8): synthesize_markov_kleisli_germ
#          ≅ objeto inicial de la Fase II (kernel de DeGroot / Laplaciano).
# =============================================================================
@dataclass(frozen=True)
class _MarkovKernelCertificate:
    """Certificado de un núcleo de Markov (morfismo de Kleisli–Giry)."""

    row_stochastic_residual: float
    spectral_radius: float
    perron_residual: float
    is_stochastic: bool
    is_reversible: bool


@dataclass(frozen=True)
class _MarkovKleisliGerm:
    """
    Gérmen de Kleisli–Markov (objeto terminal de la Fase I).
    """

    n_agents: int
    kernel: np.ndarray
    affinity: np.ndarray
    laplacian: np.ndarray
    stationary: np.ndarray
    degrees: np.ndarray
    reg_floor: float
    certificate: _MarkovKernelCertificate


class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica.
    """

    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
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

    kahan_neumann_sum = kahan_babuska_neumaier_sum

    @staticmethod
    def klein_sum(arr: np.ndarray) -> float:
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

    @staticmethod
    def compensated_real_trace(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("compensated_real_trace: se exige matriz cuadrada.")
        return _NumericalCore.kahan_babuska_neumaier_sum(np.real(np.diag(a)))

    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

    @staticmethod
    def euclidean_norm(vec: np.ndarray) -> float:
        v = np.asarray(vec, dtype=np.float64).ravel()
        if v.size == 0:
            return 0.0
        return float(np.sqrt(max(_NumericalCore.kahan_babuska_neumaier_sum(v * v), 0.0)))

    @staticmethod
    def assert_finite(name: str, array: np.ndarray) -> None:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contiene entradas no finitas.")

    @staticmethod
    def assert_square(name: str, matrix: np.ndarray, dim: Optional[int] = None) -> None:
        a = np.asarray(matrix)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; recibido {a.shape}.")
        if dim is not None and a.shape[0] != dim:
            raise ValueError(f"{name} debe ser {dim}×{dim}; recibido {a.shape}.")

    @staticmethod
    def assert_vec(name: str, vec: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        v = np.asarray(vec).reshape(-1)
        if dim is not None and v.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; recibido {v.size}.")
        _NumericalCore.assert_finite(name, v)
        return v

    @staticmethod
    def higham_nearest_hermitian(matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(matrix)
        _NumericalCore.assert_square("higham_nearest_hermitian", a)
        return 0.5 * (a + a.T.conj())

    symmetrize_hermitian = higham_nearest_hermitian

    @staticmethod
    def higham_nearest_spd(
        matrix: np.ndarray,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        herm = _NumericalCore.higham_nearest_hermitian(matrix)
        evals, evecs = la.eigh(herm)
        evals = np.maximum(np.real(evals), float(floor))
        return evecs @ (evals[:, None] * evecs.T.conj())

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
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
    def multiply_probabilities(p: float, q: float) -> float:
        pf = float(p)
        qf = float(q)
        if not (np.isfinite(pf) and np.isfinite(qf)):
            raise ValueError("multiply_probabilities: argumentos no finitos.")
        if pf <= 0.0 or qf <= 0.0:
            return 0.0
        lp = np.log(min(pf, 1.0)) + np.log(min(qf, 1.0))
        if lp < -_LOG_EXP_CLIP:
            return 0.0
        return float(np.exp(lp))

    @staticmethod
    def restochasticize_rows(
        matrix: np.ndarray,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        a = np.real(np.asarray(matrix, dtype=np.float64))
        _NumericalCore.assert_square("restochasticize_rows", a)
        w = np.maximum(a, 0.0)
        for i in range(w.shape[0]):
            mass = _NumericalCore.kahan_babuska_neumaier_sum(w[i])
            if mass > max(floor, _MACHINE_EPS):
                w[i] = w[i] / mass
            else:
                w[i] = 0.0
                w[i, i] = 1.0
        return w


class _KleisliComposer:
    """
    Fase I (continuación monádica). Categoría de Kleisli de dos mónadas.
    """

    @staticmethod
    def unit(value: Any) -> Tuple[Any, float]:
        return value, 1.0

    @staticmethod
    def bind(
        ta: Tuple[Any, float],
        k: Callable[[Any], Tuple[Any, float]],
    ) -> Tuple[Any, float]:
        value, prob = ta
        pf = float(prob)
        if not np.isfinite(pf) or pf < _MACHINE_EPS:
            return None, 0.0
        res, q = k(value)
        return res, _NumericalCore.multiply_probabilities(pf, float(q))

    @staticmethod
    def compose(
        f: Callable[[Any], Tuple[Any, float]],
        g: Callable[[Any], Tuple[Any, float]],
    ) -> Callable[[Any], Tuple[Any, float]]:
        def composed(x: Any) -> Tuple[Any, float]:
            return _KleisliComposer.bind(f(x), g)

        return composed

    @staticmethod
    def compose_markov_kernels(
        p_kernel: np.ndarray,
        q_kernel: np.ndarray,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        p = np.asarray(p_kernel, dtype=np.float64)
        q = np.asarray(q_kernel, dtype=np.float64)
        _NumericalCore.assert_square("p_kernel", p)
        _NumericalCore.assert_square("q_kernel", q)
        if p.shape[1] != q.shape[0]:
            raise ValueError(
                f"Kernels incompatibles para Kleisli: {p.shape} ⋆ {q.shape}."
            )
        return _NumericalCore.restochasticize_rows(p @ q, floor=floor)

    @staticmethod
    def markov_power(
        kernel: np.ndarray,
        steps: int,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        w = _NumericalCore.restochasticize_rows(
            np.asarray(kernel, dtype=np.float64), floor=floor
        )
        n = w.shape[0]
        if steps <= 0:
            return np.eye(n, dtype=np.float64)
        result = np.eye(n, dtype=np.float64)
        base = w
        k = int(steps)
        while k:
            if k & 1:
                result = _NumericalCore.restochasticize_rows(result @ base, floor=floor)
            base = _NumericalCore.restochasticize_rows(base @ base, floor=floor)
            k >>= 1
        return result

    @staticmethod
    def synthesize_markov_kleisli_germ(
        affinity_matrix: np.ndarray,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        symmetrize: bool = True,
    ) -> _MarkovKleisliGerm:
        raw = np.asarray(affinity_matrix)
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side != raw.size:
                raise ValueError("affinity_matrix plana no es un cuadrado perfecto.")
            raw = raw.reshape(side, side)
        _NumericalCore.assert_square("affinity_matrix", raw)
        _NumericalCore.assert_finite("affinity_matrix", raw)
        a = np.real(np.asarray(raw, dtype=np.float64))
        if symmetrize:
            a = np.real(_NumericalCore.higham_nearest_hermitian(a))
        a = np.maximum(a, 0.0)
        n = a.shape[0]
        if n == 0:
            raise ValueError("affinity_matrix no puede ser 0×0.")

        floor = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        floor = max(floor, _NumericalCore.wilkinson_deflation_floor(a))

        degrees = np.array(
            [_NumericalCore.kahan_babuska_neumaier_sum(a[i]) for i in range(n)],
            dtype=np.float64,
        )
        w = _NumericalCore.restochasticize_rows(a, floor=floor)

        inv_sqrt = np.zeros(n, dtype=np.float64)
        live = degrees > floor
        inv_sqrt[live] = 1.0 / np.sqrt(degrees[live])
        d_is = np.diag(inv_sqrt)
        lap = np.eye(n, dtype=np.float64) - d_is @ a @ d_is
        lap = np.real(_NumericalCore.higham_nearest_hermitian(lap))

        try:
            ev, evec = la.eig(w.T)
            ev = np.asarray(ev)
            k_perron = int(np.argmin(np.abs(ev - 1.0)))
            pi = np.real(evec[:, k_perron])
            pi = np.maximum(pi, 0.0)
            mass = _NumericalCore.kahan_babuska_neumaier_sum(pi)
            if mass > _MACHINE_EPS:
                pi = pi / mass
            else:
                pi = np.full(n, 1.0 / n, dtype=np.float64)
            spectral_radius = float(np.max(np.abs(ev)))
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Perron-Frobenius fallido (%s); π uniforme.", exc)
            pi = np.full(n, 1.0 / n, dtype=np.float64)
            spectral_radius = 1.0

        ones = np.ones(n, dtype=np.float64)
        row_res = _NumericalCore.euclidean_norm(w @ ones - ones)
        perron_res = _NumericalCore.euclidean_norm(w.T @ pi - pi)
        db = pi[:, None] * w - pi[None, :] * w.T
        db_res = _NumericalCore.frobenius_norm(db)
        scale = max(1.0, float(n))
        is_stoch = row_res <= max(_WILKINSON_DRIFT_LIMIT, _WILKINSON_DRIFT_LIMIT * scale)
        is_rev = db_res <= max(_WILKINSON_DRIFT_LIMIT, _WILKINSON_DRIFT_LIMIT * scale)
        cert = _MarkovKernelCertificate(
            row_stochastic_residual=float(row_res),
            spectral_radius=float(spectral_radius),
            perron_residual=float(perron_res),
            is_stochastic=bool(is_stoch),
            is_reversible=bool(is_rev),
        )
        return _MarkovKleisliGerm(
            n_agents=n,
            kernel=w,
            affinity=a,
            laplacian=lap,
            stationary=pi,
            degrees=degrees,
            reg_floor=float(floor),
            certificate=cert,
        )


# =============================================================================
# FASE II — DEGROOT, FIEDLER–CHEEGER, UHLMANN Y LIFTING DE BELL
# =============================================================================
@dataclass(frozen=True)
class _DeGrootConsensusResult:
    """Resultado certificado del consenso de DeGroot / Olfati–Saber."""

    final_opinion: np.ndarray
    fiedler_value: float
    verdict: str
    discrete_opinion: np.ndarray
    continuous_opinion: np.ndarray
    spectral_gap: float
    mixing_rate: float
    stationary: np.ndarray
    connected: bool
    cheeger_upper: float
    deviation: float
    is_reversible: bool


@dataclass(frozen=True)
class _UhlmannFidelityResult:
    """Fidelidad de Uhlmann con desigualdades de Fuchs–van de Graaf y Bures."""

    fidelity: float
    converged: bool
    bures_angle: float
    trace_distance: float
    fuchs_lower: float
    fuchs_upper: float


@dataclass(frozen=True)
class _BellCorrelationGerm:
    """
    Gérmen de correlación de Bell (objeto terminal de la Fase II).
    """

    correlation_matrix: np.ndarray
    horodecki_singular_values: np.ndarray
    tsirelson_forecast: float
    from_quantum: bool
    physical: bool


class _DeGrootConsensus:
    """
    Fase II. Consenso espectral de DeGroot y protocolo continuo.
    """

    def __init__(
        self,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        germ: Optional[_MarkovKleisliGerm] = None,
    ) -> None:
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        self._germ = germ

    @property
    def germ(self) -> Optional[_MarkovKleisliGerm]:
        return self._germ

    def _resolve_germ(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, _MarkovKleisliGerm]:
        x = _NumericalCore.assert_vec("opinion_vector", opinion_vector)
        n = x.size
        need = (
            self._germ is None
            or self._germ.n_agents != n
            or np.asarray(affinity_matrix).shape != (n, n)
        )
        if need:
            germ = _KleisliComposer.synthesize_markov_kleisli_germ(
                affinity_matrix, regularizer=self._reg, symmetrize=True
            )
            if germ.n_agents != n:
                raise ValueError(
                    "La matriz de afinidad debe ser cuadrada y coincidir "
                    "con el vector de opinión."
                )
            self._germ = germ
        else:
            germ = self._germ
        return x.astype(np.float64, copy=False), germ

    @staticmethod
    def _verdict_from_deviation(deviation: float) -> str:
        if deviation > _TOLERANCE_DEGROOT_DEGRADED:
            return "VETOED"
        if deviation > _TOLERANCE_DEGROOT_COHERENT:
            return "DEGRADED"
        return "COHERENT"

    def compute(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootConsensusResult:
        if int(steps) < 0:
            raise ValueError("steps debe ser un entero no negativo.")
        x, germ = self._resolve_germ(opinion_vector, affinity_matrix)
        n = germ.n_agents
        t = float(steps)

        evals = np.real(la.eigvalsh(germ.laplacian)) if n else np.array([0.0])
        evals = np.sort(evals)
        lambda_0 = float(evals[0]) if evals.size else 0.0
        fiedler = float(evals[1]) if evals.size > 1 else 0.0
        ker_tol = max(germ.reg_floor, _WILKINSON_DEFLATION_FLOOR * max(n, 1))
        connected = bool(n == 1 or (abs(lambda_0) <= ker_tol and fiedler > ker_tol))
        spectral_gap = float(max(fiedler, 0.0))
        cheeger_upper = float(np.sqrt(max(2.0 * spectral_gap, 0.0)))

        try:
            w_ev = np.sort_complex(la.eigvals(germ.kernel))
            mods = np.sort(np.abs(w_ev))[::-1]
            second = float(mods[1]) if mods.size > 1 else 0.0
            second = min(max(second, 0.0), 1.0)
            mixing = float(-np.log(second)) if second > _MACHINE_EPS and second < 1.0 else (
                float("inf") if second <= _MACHINE_EPS else 0.0
            )
        except (np.linalg.LinAlgError, ValueError):
            mixing = 0.0

        try:
            evo = la.expm(-t * germ.laplacian)
            continuous = np.real(evo @ x)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.error("Fallo en exponencial matricial: %s", exc)
            current = x.copy()
            dt = 0.01
            n_euler = max(int(steps), 1)
            for _ in range(n_euler):
                current = current + dt * (-germ.laplacian @ current)
            continuous = current

        discrete_kernel = _KleisliComposer.markov_power(
            germ.kernel, int(steps), floor=germ.reg_floor
        )
        discrete = np.real(discrete_kernel @ x)

        final_opinion = np.real(np.asarray(continuous, dtype=np.float64))
        mean = _NumericalCore.kahan_babuska_neumaier_sum(final_opinion) / max(n, 1)
        var = _NumericalCore.kahan_babuska_neumaier_sum((final_opinion - mean) ** 2) / max(n, 1)
        deviation = float(np.sqrt(max(var, 0.0)))
        verdict = self._verdict_from_deviation(deviation)
        if not connected and n > 1 and verdict == "COHERENT":
            verdict = "DEGRADED"

        return _DeGrootConsensusResult(
            final_opinion=final_opinion,
            fiedler_value=fiedler,
            verdict=verdict,
            discrete_opinion=np.asarray(discrete, dtype=np.float64),
            continuous_opinion=final_opinion,
            spectral_gap=spectral_gap,
            mixing_rate=float(mixing),
            stationary=np.asarray(germ.stationary, dtype=np.float64),
            connected=connected,
            cheeger_upper=cheeger_upper,
            deviation=deviation,
            is_reversible=bool(germ.certificate.is_reversible),
        )


class _UhlmannFidelity:
    """
    Fase II (continuación cuántica). Fidelidad de Uhlmann y lifting de Bell.
    """

    def __init__(self, regularizer: float = _HIGHAM_TIKHONOV_REG) -> None:
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)

    def _prepare_state(self, matrix: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(matrix)
        _NumericalCore.assert_square(name, a)
        _NumericalCore.assert_finite(name, a)
        herm = _NumericalCore.higham_nearest_hermitian(a)
        evals, evecs = la.eigh(herm)
        evals = np.real(evals)
        evals = np.maximum(evals, 0.0)
        tr = _NumericalCore.kahan_babuska_neumaier_sum(evals)
        if tr > _MACHINE_EPS:
            evals = evals / tr
        else:
            evals = np.zeros_like(evals)
            evals[-1] = 1.0
        rho = evecs @ (evals[:, None] * evecs.T.conj())
        return rho, evals, evecs

    def compute(self, rho: np.ndarray, sigma: np.ndarray) -> _UhlmannFidelityResult:
        rho_p, e_r, v_r = self._prepare_state(rho, "rho")
        sig_p, e_s, v_s = self._prepare_state(sigma, "sigma")
        if rho_p.shape != sig_p.shape:
            raise ValueError("rho y sigma deben tener la misma dimensión.")

        sqrt_r = v_r @ (np.sqrt(np.clip(e_r, 0.0, None))[:, None] * v_r.T.conj())
        sqrt_s = v_s @ (np.sqrt(np.clip(e_s, 0.0, None))[:, None] * v_s.T.conj())
        svals = np.real(la.svdvals(sqrt_r @ sqrt_s))
        amp = _NumericalCore.kahan_babuska_neumaier_sum(np.clip(svals, 0.0, None))
        fidelity = float(amp * amp)
        if fidelity < 0.0 and abs(fidelity) < 1e-12:
            fidelity = 0.0
        fidelity = float(min(max(fidelity, 0.0), 1.0))

        delta = _NumericalCore.higham_nearest_hermitian(rho_p - sig_p)
        ev_d = np.real(la.eigvalsh(delta))
        td = 0.5 * _NumericalCore.kahan_babuska_neumaier_sum(np.abs(ev_d))
        td = float(min(max(td, 0.0), 1.0))
        root_f = float(np.sqrt(fidelity))
        bures = float(np.arccos(min(max(root_f, 0.0), 1.0)))
        fuchs_lo = float(1.0 - root_f)
        fuchs_hi = float(np.sqrt(max(1.0 - fidelity, 0.0)))
        converged = bool(0.0 <= fidelity <= 1.0 + _MACHINE_EPS)
        return _UhlmannFidelityResult(
            fidelity=fidelity,
            converged=converged,
            bures_angle=bures,
            trace_distance=td,
            fuchs_lower=fuchs_lo,
            fuchs_upper=fuchs_hi,
        )

    @staticmethod
    def horodecki_tensor(rho_ab: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho_ab, dtype=np.complex128)
        t_mat = np.zeros((3, 3), dtype=np.float64)
        for i, si in enumerate(_PAULI):
            for j, sj in enumerate(_PAULI):
                t_mat[i, j] = float(np.real(np.trace(rho @ np.kron(si, sj))))
        return t_mat

    @staticmethod
    def _optimal_chsh_block(t_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        u_svd, s_vals, vt = la.svd(t_mat, full_matrices=False)
        s_vals = np.real(s_vals)
        forecast = float(2.0 * np.sqrt(max(s_vals[0] ** 2 + (s_vals[1] ** 2 if s_vals.size > 1 else 0.0), 0.0)))
        if s_vals.size == 0:
            return np.zeros((2, 2), dtype=np.float64), s_vals, 0.0
        u0 = u_svd[:, 0]
        u1 = u_svd[:, 1] if u_svd.shape[1] > 1 else np.zeros_like(u0)
        v0 = vt[0, :]
        v1 = vt[1, :] if vt.shape[0] > 1 else np.zeros_like(v0)
        a = (u0 + u1) / np.sqrt(2.0)
        ap = (u0 - u1) / np.sqrt(2.0)
        e = np.array(
            [
                [float(a @ t_mat @ v0), float(a @ t_mat @ v1)],
                [float(ap @ t_mat @ v0), float(ap @ t_mat @ v1)],
            ],
            dtype=np.float64,
        )
        return e, s_vals, forecast

    @staticmethod
    def _lhv_from_opinions(opinion_vector: np.ndarray) -> np.ndarray:
        x = _NumericalCore.assert_vec("opinion_vector", opinion_vector)
        mean = _NumericalCore.kahan_babuska_neumaier_sum(x) / max(x.size, 1)
        hidden = float(np.clip(np.tanh(mean), -1.0, 1.0))
        return hidden * np.ones((2, 2), dtype=np.float64)

    def induce_bell_correlation_germ(
        self,
        rho_or_correlation: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        opinion_vector: Optional[np.ndarray] = None,
    ) -> _BellCorrelationGerm:
        if opinion_vector is not None:
            e = self._lhv_from_opinions(opinion_vector)
            phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-12))
            return _BellCorrelationGerm(
                correlation_matrix=e,
                horodecki_singular_values=np.array([], dtype=np.float64),
                tsirelson_forecast=float(abs(2.0 * e[0, 0])),
                from_quantum=False,
                physical=phys,
            )

        raw = np.asarray(rho_or_correlation)
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side != raw.size:
                raise ValueError("rho_or_correlation plana no es un cuadrado perfecto.")
            raw = raw.reshape(side, side)
        _NumericalCore.assert_square("rho_or_correlation", raw)
        _NumericalCore.assert_finite("rho_or_correlation", raw)

        if sigma is not None:
            _ = self.compute(raw, np.asarray(sigma))
            sig = np.asarray(sigma)
            if raw.shape == (2, 2) and sig.shape == (2, 2):
                e = np.zeros((2, 2), dtype=np.float64)
                return _BellCorrelationGerm(
                    correlation_matrix=e,
                    horodecki_singular_values=np.zeros(3, dtype=np.float64),
                    tsirelson_forecast=0.0,
                    from_quantum=False,
                    physical=True,
                )

        if raw.shape == (4, 4):
            rho, _e, _v = self._prepare_state(raw, "rho_ab")
            t_mat = self.horodecki_tensor(rho)
            e, s_vals, forecast = self._optimal_chsh_block(t_mat)
            phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-9))
            return _BellCorrelationGerm(
                correlation_matrix=np.real(e).astype(np.float64, copy=False),
                horodecki_singular_values=np.asarray(s_vals, dtype=np.float64),
                tsirelson_forecast=float(min(forecast, _TSIRELSON_BOUND)),
                from_quantum=True,
                physical=phys,
            )

        if raw.shape == (2, 2):
            e = np.real(np.asarray(raw, dtype=np.float64))
            phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-12))
            terms = np.array([e[0, 0], -e[0, 1], e[1, 0], e[1, 1]], dtype=np.float64)
            forecast = abs(_NumericalCore.kahan_babuska_neumaier_sum(terms))
            return _BellCorrelationGerm(
                correlation_matrix=e,
                horodecki_singular_values=np.array([], dtype=np.float64),
                tsirelson_forecast=float(forecast),
                from_quantum=False,
                physical=phys,
            )

        raise ValueError(
            "induce_bell_correlation_germ espera E 2×2, ρ_AB 4×4 "
            f"o un vector de opinión; recibido {raw.shape}."
        )


# =============================================================================
# FASE III — CHSH, TSIRELSON, POPESCU–ROHRLICH Y MOTOR INTEGRADOR
# =============================================================================
@dataclass(frozen=True)
class _CHSHResult:
    """Resultado certificado de la desigualdad CHSH."""

    s_value: float
    verdict: str
    classical_gap: float
    tsirelson_gap: float
    pr_gap: float
    physical: bool
    horodecki_bound: float


class _CHSHVerifier:
    """
    Fase III. Observable de Bell–Clauser–Horne–Shimony–Holt.
    """

    def __init__(self, germ: Optional[_BellCorrelationGerm] = None) -> None:
        self._germ = germ

    def _resolve_e(self, correlation_matrix: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        e = np.asarray(correlation_matrix)
        if e.size == 0 and self._germ is not None:
            g = self._germ
            return g.correlation_matrix, g.tsirelson_forecast, g.physical
        if e.ndim == 1 and e.size == 4:
            e = e.reshape(2, 2)
        if e.shape != (2, 2):
            logger.error("Matriz de correlaciones debe ser 2×2.")
            return (
                np.full((2, 2), np.inf, dtype=np.float64),
                float("inf"),
                False,
            )
        _NumericalCore.assert_finite("correlation_matrix", e)
        e = np.real(np.asarray(e, dtype=np.float64))
        phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-12))
        forecast = (
            self._germ.tsirelson_forecast
            if self._germ is not None and self._germ.correlation_matrix.shape == (2, 2)
            else float("nan")
        )
        return e, forecast, phys

    def verify(self, correlation_matrix: np.ndarray) -> _CHSHResult:
        e, forecast, phys = self._resolve_e(correlation_matrix)
        if not np.all(np.isfinite(e)):
            return _CHSHResult(
                s_value=float("inf"),
                verdict="VETOED",
                classical_gap=float("inf"),
                tsirelson_gap=float("-inf"),
                pr_gap=float("-inf"),
                physical=False,
                horodecki_bound=float("nan"),
            )

        terms = np.array(
            [e[0, 0], -e[0, 1], e[1, 0], e[1, 1]],
            dtype=np.float64,
        )
        s_value = abs(_NumericalCore.kahan_babuska_neumaier_sum(terms))
        if not phys:
            logger.warning(
                "Correladores no físicos: max|E|=%.6f > 1.",
                float(np.max(np.abs(e))),
            )

        if (not phys) or s_value > _TSIRELSON_BOUND + 8.0 * _MACHINE_EPS:
            verdict = "VETOED"
        elif s_value > _CLASSICAL_CHSH_BOUND:
            verdict = "COHERENT"
        else:
            verdict = "DEGRADED"

        horo = forecast if np.isfinite(forecast) else s_value
        return _CHSHResult(
            s_value=float(s_value),
            verdict=verdict,
            classical_gap=float(s_value - _CLASSICAL_CHSH_BOUND),
            tsirelson_gap=float(_TSIRELSON_BOUND - s_value),
            pr_gap=float(_PR_NOSIGNAL_BOUND - s_value),
            physical=bool(phys),
            horodecki_bound=float(horo),
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialSequitosEngine:
    """
    Motor de alta fidelidad para la Capa 1.5 (Séquitos Imperiales).

    Compone las tres fases anidadas e integra la dinámica simpléctica de Poincaré:

    1. Fase I   — gérmen de Kleisli–Giry (`_NumericalCore`, `_KleisliComposer`).
    2. Fase II  — DeGroot / Fiedler y Uhlmann (`_DeGrootConsensus`,
                  `_UhlmannFidelity`).
    3. Fase III — CHSH / Tsirelson (`_CHSHVerifier`).
    4. Poincaré — Integrador simpléctico de Störmer-Verlet con Acción de Maupertuis-Jacobi.
    """

    def __init__(self, regularizer: float = _HIGHAM_TIKHONOV_REG, dimension_n: int = 2) -> None:
        """
        Inicializa el motor y materializa el encadenamiento de gérmenes.

        Args:
            regularizer: Piso de Tikhonov contra polos espectrales.
            dimension_n: Dimensión de la variedad simpléctica T*Q (n por sub-espacio).
        """
        self._reg: Final[float] = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        self._n: Final[int] = int(dimension_n)
        self._dim: Final[int] = 2 * int(dimension_n)
        self._J_canonical: Final[NDArray[np.float64]] = np.block([
            [np.zeros((self._n, self._n), dtype=np.float64), np.eye(self._n, dtype=np.float64)],
            [-np.eye(self._n, dtype=np.float64), np.zeros((self._n, self._n), dtype=np.float64)]
        ])

        # Fase I → objeto inicial de Fase II (grafo trivial de 2 agentes).
        self._markov_germ: _MarkovKleisliGerm = (
            _KleisliComposer.synthesize_markov_kleisli_germ(
                np.eye(2, dtype=np.float64), regularizer=self._reg
            )
        )
        self._degroot = _DeGrootConsensus(regularizer=self._reg, germ=self._markov_germ)
        self._uhlmann = _UhlmannFidelity(regularizer=self._reg)
        # Fase II → objeto inicial de Fase III (correlación clásica nula).
        self._bell_germ: _BellCorrelationGerm = (
            self._uhlmann.induce_bell_correlation_germ(np.zeros((2, 2), dtype=np.float64))
        )
        self._chsh = _CHSHVerifier(germ=self._bell_germ)

    def compute_symplectic_maupertuis_step(
        self,
        current_z: NDArray[np.float64],
        dt: float,
        metric_G: NDArray[np.float64],
        potential_V_func: Callable[[NDArray[np.float64]], float],
        grad_V_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        total_energy_H0: float,
    ) -> SequitosEngineStepResult:
        r"""
        Ejecuta un paso de integración simpléctica de Störmer-Verlet en coordenadas Darboux.

        Axiomas:
          1. Ecuaciones de Hamilton: q̇ = G⁻¹ p,  ṗ = -∇V(q).
          2. Medida de Liouville: det(M) = +1.
          3. Acción de Maupertuis: S_M = ∫ √(2(H₀ - V(q))) |dq|_G.
        """
        c_z = np.asarray(current_z, dtype=np.float64).ravel()
        dim = c_z.size
        n = dim // 2
        q_0 = c_z[:n].copy()
        p_0 = c_z[n:].copy()

        G_mat = np.asarray(metric_G, dtype=np.float64)
        if G_mat.ndim != 2 or G_mat.shape[0] != n or G_mat.shape[1] != n:
            G_inv = np.eye(n, dtype=np.float64)
        else:
            try:
                G_inv = la.inv(G_mat)
            except (la.LinAlgError, ValueError):
                G_inv = np.eye(n, dtype=np.float64)

        # 1. Medio paso de momentum: p_{1/2} = p_0 - (dt/2) * ∇V(q_0)
        grad_V_0 = np.asarray(grad_V_func(q_0), dtype=np.float64).ravel()
        p_half = p_0 - 0.5 * dt * grad_V_0

        # 2. Paso completo de posición: q_1 = q_0 + dt * G⁻¹ * p_{1/2}
        q_1 = q_0 + dt * (G_inv @ p_half)

        # 3. Medio paso de momentum final: p_1 = p_{1/2} - (dt/2) * ∇V(q_1)
        grad_V_1 = np.asarray(grad_V_func(q_1), dtype=np.float64).ravel()
        p_1 = p_half - 0.5 * dt * grad_V_1

        next_z = np.concatenate([q_1, p_1])

        # 4. Evaluación del Hamiltoniano: H(q, p) = ½ pᵀ G⁻¹ p + V(q)
        kinetic_energy = 0.5 * float(p_1.T @ G_inv @ p_1)
        potential_energy = float(potential_V_func(q_1))
        current_energy = kinetic_energy + potential_energy

        # 5. Medición del Jacobiano M_step y la deriva de Liouville
        j_canon = np.block([
            [np.zeros((n, n), dtype=np.float64), np.eye(n, dtype=np.float64)],
            [-np.eye(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)]
        ])
        h_hessian = np.block([
            [np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)],
            [np.zeros((n, n), dtype=np.float64), G_inv]
        ])
        jacobian_M = np.eye(dim, dtype=np.float64) + dt * (j_canon @ h_hessian)
        det_M = float(la.det(jacobian_M))
        volume_drift = abs(det_M - 1.0)

        # 6. Cómputo de la Acción de Maupertuis-Jacobi
        kinetic_margin = 2.0 * (total_energy_H0 - potential_energy)
        refractive_n = np.sqrt(max(0.0, kinetic_margin))
        dq_norm = float(la.norm(q_1 - q_0))
        maupertuis_action = refractive_n * dq_norm

        is_conserved = volume_drift <= _WILKINSON_LIMIT

        return SequitosEngineStepResult(
            next_state_z=next_z,
            hamiltonian_energy=current_energy,
            volume_drift=volume_drift,
            maupertuis_action=maupertuis_action,
            is_liouville_conserved=is_conserved,
        )

    def _resync_markov_germ(self, affinity_matrix: np.ndarray) -> _MarkovKleisliGerm:
        germ = _KleisliComposer.synthesize_markov_kleisli_germ(
            affinity_matrix, regularizer=self._reg, symmetrize=True
        )
        self._markov_germ = germ
        self._degroot = _DeGrootConsensus(regularizer=self._reg, germ=germ)
        return germ

    # ── Fase I expuesta ───────────────────────────────────────────────────
    def kahan_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_neumann_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def kleisli_compose(
        self,
        f: Callable[[Any], Tuple[Any, float]],
        g: Callable[[Any], Tuple[Any, float]],
    ) -> Callable[[Any], Tuple[Any, float]]:
        return _KleisliComposer.compose(f, g)

    def kleisli_unit(self, value: Any) -> Tuple[Any, float]:
        return _KleisliComposer.unit(value)

    def compose_markov_kernels(
        self,
        p_kernel: np.ndarray,
        q_kernel: np.ndarray,
    ) -> np.ndarray:
        return _KleisliComposer.compose_markov_kernels(
            p_kernel, q_kernel, floor=self._reg
        )

    def synthesize_markov_kleisli_germ(
        self,
        affinity_matrix: np.ndarray,
        symmetrize: bool = True,
    ) -> _MarkovKleisliGerm:
        germ = _KleisliComposer.synthesize_markov_kleisli_germ(
            affinity_matrix, regularizer=self._reg, symmetrize=symmetrize
        )
        self._markov_germ = germ
        self._degroot = _DeGrootConsensus(regularizer=self._reg, germ=germ)
        return germ

    def markov_kleisli_germ_certificate(self) -> _MarkovKernelCertificate:
        return self._markov_germ.certificate

    # ── Fase II expuesta ──────────────────────────────────────────────────
    def compute_degroot_spectral_consensus(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> Tuple[np.ndarray, float, str]:
        result = self.compute_degroot_spectral_consensus_certified(
            opinion_vector, affinity_matrix, steps
        )
        return result.final_opinion, result.fiedler_value, result.verdict

    def compute_degroot_spectral_consensus_certified(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootConsensusResult:
        self._resync_markov_germ(affinity_matrix)
        return self._degroot.compute(opinion_vector, affinity_matrix, steps)

    def compute_uhlmann_fidelity(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        return self._uhlmann.compute(rho, sigma).fidelity

    def compute_uhlmann_fidelity_certified(
        self,
        rho: np.ndarray,
        sigma: np.ndarray,
    ) -> _UhlmannFidelityResult:
        return self._uhlmann.compute(rho, sigma)

    def induce_bell_correlation_germ(
        self,
        rho_or_correlation: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        opinion_vector: Optional[np.ndarray] = None,
    ) -> _BellCorrelationGerm:
        germ = self._uhlmann.induce_bell_correlation_germ(
            rho_or_correlation, sigma=sigma, opinion_vector=opinion_vector
        )
        self._bell_germ = germ
        self._chsh = _CHSHVerifier(germ=germ)
        return germ

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def verify_chsh_violation(self, correlation_matrix: np.ndarray) -> Tuple[float, str]:
        result = self._chsh.verify(correlation_matrix)
        return result.s_value, result.verdict

    def verify_chsh_violation_certified(
        self,
        correlation_matrix: np.ndarray,
    ) -> _CHSHResult:
        return self._chsh.verify(correlation_matrix)


__all__ = [
    "ImperialSequitosEngine",
    "SequitosEngineStepResult",
]
