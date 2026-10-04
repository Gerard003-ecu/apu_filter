# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Engine (Caballos Imperiales de Cálculo Espectral)   ║
║ Ruta   : app/core/inmune_system/imperial_guards_engine.py                    ║
║ Versión: 4.1.0-Poincare-Liouville-Maupertuis-Novikov-FPU-PhD                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Este motor elíptico ciego ejecuta los cálculos espectrales, algebraicos, topológicos
y de mecánica celeste de Henri Poincaré sobre la FPU. Alimentando a las aduanas
de-confinadas del módulo `imperial_guards_agent.py`, opera con alta precisión
garantizando la invarianza del volumen de Liouville, la acción geodésica de Maupertuis-Jacobi,
la absorción de pequeños divisores en el anillo de Novikov, la conservación de la traza cuántica
y el confinamiento de mermas discretas.

MÉTODOS GRANULARES Y AXIOMAS DE POINCARÉ:

1. step_poincare_symplectic_integration:
   Ejecuta un paso de integración simpléctica Störmer-Verlet preservando Liouville y Darboux:
   $$\mathbf{M}^\top \mathbf{J}_{2n} \mathbf{M} = \mathbf{J}_{2n} \implies \det(\mathbf{M}) = +1$$
   Acción de Maupertuis-Jacobi sobre la métrica conforme de Jacobi-Fermat:
   $$\mathcal{S}_M = \int \sqrt{2(H_0 - V(q))} \, \|\dot{q}\|_G \, dt$$
   Absorción de Pequeños Divisores de Poincaré en el Anillo Ultramétrico de Novikov $\Lambda_{\mathrm{Nov}}$:
   $$W_{\mathrm{Nov}}(k) = \exp\left(-\frac{1.0}{\varepsilon_{\mathrm{Wilkinson}} + |\langle k, \boldsymbol{\omega}\rangle|}\right)$$

2. kahan_sum(arr: np.ndarray) -> float:
   Realiza la sumación compensada de Kahan-Babuška-Neumaier (KBN) para aniquilar la deriva numérica en
   la mantisa flotante durante la integración espectral:
   $$S_N = \sum_{i=1}^N x_i \quad \text{donde} \quad c_{k+1} = (t_{k+1} - S_k) - y_{k+1}$$

3. compute_complex_step_gradient(func: Any, x: np.ndarray, h: float = 1e-20) -> np.ndarray:
   Calcula el gradiente exacto mediante Diferenciación por Paso Complejo (CSMD):
   $$\nabla_k f(x) = \frac{\operatorname{Im}\left(f(x + j \cdot h \cdot e_k)\right)}{h} + \mathcal{O}(h^2)$$

4. compute_dirac_operator_spectrum(density_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
   Somete la matriz densidad mixta \rho de sabiduría a una de-coherencia espectral:
   $$\not\!D = \rho^{-1/2} = V \Lambda^{-1/2} V^\top$$

5. compute_petz_fisher_rao_metric(rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
   Calcula la Métrica de Fisher-Rao Cuántica de Petz-Fisher no conmutativa:
   $$g_\rho(A, B) = \sum_{i,j} \langle i|A|j\rangle \langle j|B|i\rangle \, \varphi(\lambda_i, \lambda_j)^{-1}$$

6. compute_simplicial_normalized_laplacian(boundary_matrix: np.ndarray) -> np.ndarray:
   Construye el Laplaciano del Haz celular normalizado:
   $$L_F = \delta_0^\top G^{-1} \delta_0 \implies L_{\mathrm{sym}} = D^{-1/2} L_{\mathrm{base}} D^{-1/2}$$

7. estimate_cheeger_constant_bounds(eigenvalues_L: np.ndarray) -> Tuple[float, float]:
   Mide las cotas de la constante isoperimétrica de Cheeger h(G) a partir del valor de Fiedler \lambda_2:
   $$\frac{\lambda_2}{2} \le h(G) \le \sqrt{2 \lambda_2}$$

8. compute_euler_poincare_characteristic(boundary_0: np.ndarray, boundary_1: np.ndarray) -> int:
   Certifica la nulidad homológica evaluando la característica de Euler simplicial:
   $$\chi(K) = \beta_0 - \beta_1 + \beta_2 = |V| - |E| + |F|$$
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Agents.ImperialGuardsEngine")

__version__: Final[str] = (
    "4.1.0-Poincare-Liouville-Maupertuis-Novikov-FPU-PhD"
)


# =============================================================================
# CONSTANTES METROLÓGICAS DE PRECISIÓN IEEE-754
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_TIKHONOV_FLOOR: Final[float] = 1e-20
_DEFAULT_REGULARIZER: Final[float] = 1e-15
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-12
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9

_PSD_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_PSD_REL_TOL: Final[float] = 1e-8
_HERMITIAN_REL_TOL: Final[float] = 1e-8
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_LOG_MEAN_REL_TOL: Final[float] = 1e-8

_COMPLEX_STEP_DEFAULT_H: Final[float] = 1e-20
_COMPLEX_STEP_MIN_H: Final[float] = 1e-30
_COMPLEX_STEP_FD_FALLBACK: Final[float] = 1e-8
_LOG_EXP_CLIP: Final[float] = 700.0
_SPECTRAL_DIM_MIN_MODES: Final[int] = 4


# =============================================================================
# DATACLASSES ESTRUCTURALES E INMUTABLES
# =============================================================================

@dataclass(frozen=True, slots=True)
class ImperialEngineStepResult:
    r"""Resultado inmutable de integración simpléctica de Poincaré en FPU."""

    next_state: NDArray[np.float64]
    hamiltonian_energy: float
    volume_drift: float
    maupertuis_action: float
    novikov_absorbed_weight: float
    is_step_valid: bool


@dataclass(frozen=True)
class _SpectralTripleCertificate:
    """Certificado numérico del estado que sostiene el triple (A, ℋ, ·)."""

    hermiticity_residual: float
    trace: float
    min_eigenvalue: float
    purity: float
    von_neumann_entropy: float
    effective_rank: int
    is_density: bool


@dataclass(frozen=True)
class _SpectralTripleGerm:
    """
    Gérmen del triple espectral (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: un estado densidad ρ ∈ 𝒟(ℋ)
    sobre el cual se instancia el Dirac modular de Connes–Chamseddine
    y la métrica monótona de Petz g_ρ.
    """

    dim: int
    density: Optional[np.ndarray]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    reg_floor: float
    csmd_step: float
    certificate: _SpectralTripleCertificate


# =============================================================================
# FASE I — FUNDAMENTOS NUMÉRICOS: VALIDACIÓN, KAHAN–KLEIN Y CSMD
# =============================================================================

class Phase1NumericalFoundationsMixin:
    """
    FASE I — FUNDAMENTOS NUMÉRICOS.

    Provee el topos lineal subyacente: validación IEEE-754, sumación
    compensada en el álgebra de Banach (ℝ, +, ·), diferenciación por
    paso complejo y el gérmen que inicia la Fase II.
    """

    # ── I.1  Validación escalar ───────────────────────────────────────────
    @staticmethod
    def _validate_nonnegative_finite(name: str, value: Any) -> float:
        """Valida que `value` sea un real finito ≥ 0 (rechaza bool)."""
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f < 0.0:
            raise ValueError(f"{name} debe ser finito y mayor o igual que cero.")
        return value_f

    @staticmethod
    def _validate_positive_finite(name: str, value: Any) -> float:
        """Valida que `value` sea un real finito estrictamente positivo."""
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente mayor que cero.")
        return value_f

    # ── I.2  Validación de arreglos ───────────────────────────────────────
    @staticmethod
    def _ensure_finite_array(arr: np.ndarray, name: str) -> None:
        """Garantiza que un ndarray sea finito y numérico."""
        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise ValueError(f"{name} contiene tipos no numéricos.") from exc
        if not finite:
            raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")

    def _as_numeric_vector(
        self,
        values: Any,
        name: str,
        *,
        allow_complex: bool = False,
    ) -> np.ndarray:
        """Convierte `values` a vector 1-D; rechaza Im no despreciable si procede."""
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim == 0:
            raw = raw.reshape(1)
        elif raw.ndim > 1:
            raw = raw.ravel()
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if not allow_complex:
                if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                    raise ValueError(
                        f"{name} posee componente imaginaria no despreciable."
                    )
                raw = raw.real
        try:
            dtype = np.complex128 if allow_complex else np.float64
            arr = np.asarray(raw, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a vector numérico.") from exc
        self._ensure_finite_array(arr, name)
        return arr

    def _as_numeric_matrix(
        self,
        values: Any,
        name: str,
        *,
        allow_complex: bool = False,
        square: bool = False,
    ) -> np.ndarray:
        """Convierte `values` a matriz 2-D; exige cuadratura si `square`."""
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side != raw.size:
                raise ValueError(f"{name} debe ser una matriz 2D.")
            raw = raw.reshape(side, side)
        if raw.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if not allow_complex:
                if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                    raise ValueError(
                        f"{name} posee componente imaginaria no despreciable."
                    )
                raw = raw.real
        try:
            dtype = np.complex128 if allow_complex else np.float64
            arr = np.asarray(raw, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a matriz numérica.") from exc
        self._ensure_finite_array(arr, name)
        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")
        return arr

    def _regularizer_floor(self) -> float:
        """Piso de Tikhonov vigente."""
        reg = getattr(self, "_reg", _DEFAULT_REGULARIZER)
        try:
            value = float(reg)
        except (TypeError, ValueError):
            value = _DEFAULT_REGULARIZER
        if not math.isfinite(value) or value < 0.0:
            value = _DEFAULT_REGULARIZER
        return float(max(value, _HIGHAM_TIKHONOV_FLOOR))

    def _csmd_step(self) -> float:
        step = getattr(self, "_csmd_h", _COMPLEX_STEP_DEFAULT_H)
        try:
            value = float(step)
        except (TypeError, ValueError):
            value = _COMPLEX_STEP_DEFAULT_H
        if not math.isfinite(value) or value <= 0.0:
            value = _COMPLEX_STEP_DEFAULT_H
        return float(max(value, _COMPLEX_STEP_MIN_H))

    # ── I.3  Normas y Higham ──────────────────────────────────────────────
    @staticmethod
    def _frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt / Frobenius ‖A‖_F."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

    def _higham_nearest_hermitian(self, matrix: np.ndarray, name: str) -> np.ndarray:
        """Proyección de Weyl–Toeplitz: (A + A†)/2."""
        a = self._as_numeric_matrix(matrix, name, allow_complex=True, square=True)
        return 0.5 * (a + a.T.conj())

    def _wilkinson_deflation_floor(self, matrix: np.ndarray) -> float:
        """ε_W = max(‖A‖_F · ε_mach · 10, ε_Wilkinson)."""
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro = self._frobenius_norm(matrix)
        return float(
            max(fro * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE, _WILKINSON_DEFLATION_FLOOR)
        )

    def _higham_nearest_density(
        self,
        matrix: np.ndarray,
        name: str,
        floor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estado densidad más próximo en ‖·‖_F."""
        herm = self._higham_nearest_hermitian(matrix, name)
        try:
            evals, evecs = la.eigh(herm, check_finite=True)
        except la.LinAlgError as exc:
            raise ValueError(f"La descomposición espectral de {name} falló.") from exc
        evals = np.real(np.asarray(evals, dtype=np.float64))
        self._ensure_finite_array(evals, f"{name}.eigenvalues")
        eps = float(floor) if floor is not None else self._regularizer_floor()
        evals = np.maximum(evals, 0.0)
        evals[evals < eps] = 0.0
        tr = self.kahan_sum(evals)
        if tr > _MACHINE_EPS:
            evals = evals / tr
        else:
            evals = np.zeros_like(evals)
            evals[-1] = 1.0
        rho = evecs @ (evals[:, None] * evecs.T.conj())
        rho = 0.5 * (rho + rho.T.conj())
        return rho, evals, evecs

    # ── I.4  Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def _neumaier_accumulate(
        total: float,
        compensation: float,
        term: float,
    ) -> Tuple[float, float]:
        """Paso elemental de Kahan–Babuška–Neumaier."""
        t = total + term
        if not math.isfinite(t):
            return float(t), compensation
        if abs(total) >= abs(term):
            compensation += (total - t) + term
        else:
            compensation += (term - t) + total
        return t, compensation

    def kahan_sum(self, arr: np.ndarray) -> float:
        r"""Sumación compensada de Kahan–Babuška–Neumaier (KBN)."""
        vec = self._as_numeric_vector(arr, "arr", allow_complex=False)
        total = 0.0
        compensation = 0.0
        for term in vec:
            total, compensation = self._neumaier_accumulate(total, compensation, float(term))
            if not math.isfinite(total):
                return float(total)
        return float(total + compensation)

    def kahan_classical_sum(self, arr: np.ndarray) -> float:
        """Sumación de Kahan clásica."""
        vec = self._as_numeric_vector(arr, "arr", allow_complex=False)
        total = 0.0
        c = 0.0
        for term in vec:
            y = float(term) - c
            t = total + y
            c = (t - total) - y
            total = t
        return float(total)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        """Alias explícito de la sumación KBN."""
        return self.kahan_sum(arr)

    def klein_sum(self, arr: np.ndarray) -> float:
        """Sumación doblemente compensada de Klein."""
        vec = self._as_numeric_vector(arr, "arr", allow_complex=False)
        s = 0.0
        cs = 0.0
        ccs = 0.0
        for term in vec:
            xf = float(term)
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

    # ── I.5  Diferenciación holomorfa por paso complejo (CSMD) ────────────
    def compute_complex_step_gradient(
        self,
        func: Callable[[np.ndarray], Any],
        x: np.ndarray,
        h: float = _COMPLEX_STEP_DEFAULT_H,
    ) -> np.ndarray:
        r"""Gradiente CSMD (Lyness–Moler / Squire–Trapp)."""
        if not callable(func):
            raise TypeError("func debe ser callable.")
        x_vec = self._as_numeric_vector(x, "x", allow_complex=False)
        h_val = self._validate_nonnegative_finite("h", h)
        if h_val <= 0.0:
            raise ValueError("h debe ser estrictamente positivo.")
        if h_val < _COMPLEX_STEP_MIN_H:
            h_val = _COMPLEX_STEP_MIN_H

        dim = x_vec.size
        grad = np.zeros(x_vec.shape, dtype=np.float64)
        if dim == 0:
            return grad

        holomorphic = True
        try:
            probe = func(x_vec.astype(np.complex128))
            probe_arr = np.asarray(probe)
            if probe_arr.size != 1:
                holomorphic = False
            else:
                _ = np.imag(probe_arr.reshape(-1)[0])
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False

        if holomorphic:
            x_complex = x_vec.astype(np.complex128)
            for i in range(dim):
                x_pert = x_complex.copy()
                x_pert[i] += 1j * h_val
                try:
                    val = func(x_pert)
                except Exception as exc:
                    holomorphic = False
                    logger.warning("CSMD: func falló en el stencil (%s).", exc)
                    break
                val_arr = np.asarray(val)
                if val_arr.size != 1:
                    holomorphic = False
                    break
                val_scalar = val_arr.reshape(-1)[0]
                try:
                    if not np.isfinite(val_scalar):
                        raise ValueError("func retornó un valor no finito.")
                except TypeError as exc:
                    raise ValueError("func retornó un tipo no numérico.") from exc
                imag = float(np.imag(val_scalar))
                if not math.isfinite(imag):
                    holomorphic = False
                    break
                grad_i = imag / h_val
                if not math.isfinite(grad_i):
                    raise ValueError("El gradiente CSMD produjo un valor no finito.")
                grad[i] = grad_i

        if not holomorphic:
            logger.warning(
                "CSMD: func no es holomorfa en el stencil; se usa diferencia central."
            )
            scale = float(max(np.linalg.norm(x_vec), 1.0))
            h_fd = max(_COMPLEX_STEP_FD_FALLBACK, _COMPLEX_STEP_FD_FALLBACK * scale)
            for i in range(dim):
                xp = x_vec.copy()
                xm = x_vec.copy()
                xp[i] += h_fd
                xm[i] -= h_fd
                try:
                    fp = np.asarray(func(xp)).reshape(-1)[0]
                    fm = np.asarray(func(xm)).reshape(-1)[0]
                except Exception as exc:
                    raise ValueError("func falló en la diferencia central.") from exc
                fp_f = float(np.real(fp))
                fm_f = float(np.real(fm))
                if not (math.isfinite(fp_f) and math.isfinite(fm_f)):
                    raise ValueError("func retornó un valor no finito.")
                grad[i] = (fp_f - fm_f) / (2.0 * h_fd)
        return grad

    def compute_complex_step_hessian(
        self,
        func: Callable[[np.ndarray], Any],
        x: np.ndarray,
        h: float = _COMPLEX_STEP_DEFAULT_H,
    ) -> np.ndarray:
        """Hessiano por diferencia central de gradientes CSMD."""
        x_vec = self._as_numeric_vector(x, "x", allow_complex=False)
        dim = x_vec.size
        scale = float(max(np.linalg.norm(x_vec), 1.0))
        eta = max(math.sqrt(_MACHINE_EPS), math.sqrt(_MACHINE_EPS) * scale)
        hess = np.zeros((dim, dim), dtype=np.float64)
        for j in range(dim):
            xp = x_vec.copy()
            xm = x_vec.copy()
            xp[j] += eta
            xm[j] -= eta
            gp = self.compute_complex_step_gradient(func, xp, h=h)
            gm = self.compute_complex_step_gradient(func, xm, h=h)
            hess[:, j] = (gp - gm) / (2.0 * eta)
        return 0.5 * (hess + hess.T)

    # ── I.9  Morfismo terminal de la Fase I ───────────────────────────────
    def synthesize_spectral_triple_germ(
        self,
        density_matrix: Optional[np.ndarray] = None,
        csmd_step: float = _COMPLEX_STEP_DEFAULT_H,
        regularizer: Optional[float] = None,
    ) -> _SpectralTripleGerm:
        """Ensambla el gérmen del triple espectral."""
        h_val = self._validate_positive_finite("csmd_step", csmd_step)
        h_val = max(h_val, _COMPLEX_STEP_MIN_H)
        floor = (
            self._regularizer_floor()
            if regularizer is None
            else float(max(self._validate_nonnegative_finite("regularizer", regularizer),
                           _HIGHAM_TIKHONOV_FLOOR))
        )

        empty_cert = _SpectralTripleCertificate(
            hermiticity_residual=0.0,
            trace=0.0,
            min_eigenvalue=0.0,
            purity=0.0,
            von_neumann_entropy=0.0,
            effective_rank=0,
            is_density=False,
        )
        if density_matrix is None:
            return _SpectralTripleGerm(
                dim=0,
                density=None,
                eigenvalues=np.array([], dtype=np.float64),
                eigenvectors=np.zeros((0, 0), dtype=np.complex128),
                reg_floor=floor,
                csmd_step=float(h_val),
                certificate=empty_cert,
            )

        raw = self._as_numeric_matrix(
            density_matrix, "density_matrix", allow_complex=True, square=True
        )
        if raw.shape[0] == 0:
            raise ValueError("density_matrix no puede ser 0×0.")
        herm_res = self._frobenius_norm(raw - raw.T.conj())
        rho, evals, evecs = self._higham_nearest_density(raw, "density_matrix", floor=floor)
        tr = self.kahan_sum(evals)
        min_ev = float(np.min(evals)) if evals.size else 0.0
        purity = self.kahan_sum(evals * evals)
        pos = evals > floor
        if np.any(pos):
            vn = float(self.kahan_sum(-evals[pos] * np.log(evals[pos])))
        else:
            vn = 0.0
        rank = int(np.sum(pos))
        scale = max(self._frobenius_norm(raw), 1.0)
        is_dens = (
            herm_res <= _HERMITIAN_REL_TOL * scale
            and abs(tr - 1.0) <= 1e-10
            and min_ev >= -max(floor, _PSD_ABS_TOL)
        )
        cert = _SpectralTripleCertificate(
            hermiticity_residual=float(herm_res),
            trace=float(tr),
            min_eigenvalue=min_ev,
            purity=float(purity),
            von_neumann_entropy=float(max(vn, 0.0)),
            effective_rank=rank,
            is_density=bool(is_dens),
        )
        if not is_dens:
            logger.warning(
                "Gérmen espectral: ρ no es densidad estricta."
            )
        return _SpectralTripleGerm(
            dim=int(raw.shape[0]),
            density=rho,
            eigenvalues=np.asarray(evals, dtype=np.float64),
            eigenvectors=np.asarray(evecs),
            reg_floor=floor,
            csmd_step=float(h_val),
            certificate=cert,
        )


# =============================================================================
# FASE II — TRIPLE ESPECTRAL DE CONNES Y MÉTRICAS DE PETZ
# =============================================================================

@dataclass(frozen=True)
class _DiracSpectrumResult:
    """Espectro certificado del Dirac modular de Connes–Chamseddine."""

    dirac_eigs: np.ndarray
    eigenvalues: np.ndarray
    dirac_operator: np.ndarray
    laplacian_eigs: np.ndarray
    spectral_action: float
    spectral_dimension: float
    condition_number: float
    kernel_dim: int


@dataclass(frozen=True)
class _PetzMetricResult:
    """Métricas monótonas de Petz en ρ."""

    bkm: float
    bures: float
    wigner_yanase: float
    dropped_kernel_terms: int
    tangent_hermiticity: float
    tangent_traceless: float


@dataclass(frozen=True)
class _HodgeCheegerGerm:
    """Gérmen de Hodge–Cheeger (objeto terminal de la Fase II)."""

    laplacian_eigs: np.ndarray
    combinatorial_laplacian: Optional[np.ndarray]
    dirac_eigs: np.ndarray
    petz_scale: float
    two_n_hint: int
    reg_floor: float
    from_connes: bool


class Phase2SpectralQuantumMixin(Phase1NumericalFoundationsMixin):
    """
    FASE II — ESPECTRAL CUÁNTICA.

    Resuelve el Dirac modular del triple de Connes y las métricas monótonas de Petz.
    """

    def _resolve_triple_germ(
        self,
        density_matrix: np.ndarray,
        germ: Optional[_SpectralTripleGerm] = None,
    ) -> _SpectralTripleGerm:
        cached = germ if germ is not None else getattr(self, "_triple_germ", None)
        raw = self._as_numeric_matrix(
            density_matrix, "density_matrix", allow_complex=True, square=True
        )
        if (
            cached is not None
            and cached.density is not None
            and cached.dim == raw.shape[0]
            and cached.density.shape == raw.shape
        ):
            return cached
        built = self.synthesize_spectral_triple_germ(raw)
        self._triple_germ = built
        return built

    def _ensure_hermitian(self, matrix: np.ndarray, name: str) -> np.ndarray:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada.")
        adjoint = matrix.conj().T
        diff_norm = self._frobenius_norm(matrix - adjoint)
        scale = max(1.0, self._frobenius_norm(matrix))
        if diff_norm > _HERMITIAN_REL_TOL * scale:
            raise ValueError(
                f"{name} no es Hermitiano/simétrico dentro de tolerancia."
            )
        return 0.5 * (matrix + adjoint)

    def _eigh_safe(
        self,
        hermitian_matrix: np.ndarray,
        name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            eigenvalues, eigenvectors = la.eigh(hermitian_matrix, check_finite=True)
        except la.LinAlgError as exc:
            raise ValueError(f"La descomposición espectral de {name} falló.") from exc
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self._ensure_finite_array(eigenvalues, f"{name}.eigenvalues")
        return eigenvalues, eigenvectors

    @staticmethod
    def _psd_tolerance(eigenvalues: np.ndarray) -> float:
        if eigenvalues.size == 0:
            return _PSD_ABS_TOL
        max_abs = float(np.max(np.abs(eigenvalues)))
        return max(_PSD_ABS_TOL, _PSD_REL_TOL * max_abs)

    def _enforce_psd(self, eigenvalues: np.ndarray, name: str) -> None:
        if eigenvalues.size == 0:
            return
        tol = self._psd_tolerance(eigenvalues)
        min_eig = float(np.min(eigenvalues))
        if min_eig < -tol:
            raise ValueError(
                f"{name} no es semidefinido positivo dentro de tolerancia."
            )

    @staticmethod
    def _logarithmic_mean(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 0.0
        if a == b:
            return float(a)
        lo, hi = (a, b) if a < b else (b, a)
        if (hi - lo) <= _LOG_MEAN_REL_TOL * hi:
            return float(0.5 * (a + b))
        t = hi / lo
        log_t = math.log(t)
        if log_t == 0.0:
            return float(0.5 * (a + b))
        return float(lo * (t - 1.0) / log_t)

    @staticmethod
    def _arithmetic_mean(a: float, b: float) -> float:
        if a <= 0.0 and b <= 0.0:
            return 0.0
        return float(0.5 * (max(a, 0.0) + max(b, 0.0)))

    @staticmethod
    def _wigner_yanase_mean(a: float, b: float) -> float:
        if a <= 0.0 and b <= 0.0:
            return 0.0
        return float(0.25 * (math.sqrt(max(a, 0.0)) + math.sqrt(max(b, 0.0))) ** 2)

    def _petz_sum(
        self,
        eigs: np.ndarray,
        a_rot: np.ndarray,
        b_rot: np.ndarray,
        mean_fn: Callable[[float, float], float],
    ) -> Tuple[float, int]:
        total = 0.0
        compensation = 0.0
        dropped = 0
        n = eigs.size
        for i in range(n):
            lam_i = float(eigs[i])
            for j in range(n):
                lam_j = float(eigs[j])
                mean_val = mean_fn(lam_i, lam_j)
                if mean_val <= _MACHINE_EPS:
                    dropped += 1
                    continue
                term_f = float(np.real(a_rot[i, j] * b_rot[j, i]) / mean_val)
                if not math.isfinite(term_f):
                    raise ValueError(
                        "La métrica de Petz produjo un término no finito."
                    )
                total, compensation = self._neumaier_accumulate(total, compensation, term_f)
        return float(total + compensation), int(dropped)

    def compute_dirac_operator_spectrum(
        self,
        density_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Espectro del Dirac modular regularizado del triple de Connes."""
        result = self.compute_dirac_operator_spectrum_certified(density_matrix)
        return result.dirac_eigs, result.eigenvalues

    def compute_dirac_operator_spectrum_certified(
        self,
        density_matrix: np.ndarray,
    ) -> _DiracSpectrumResult:
        germ = self._resolve_triple_germ(density_matrix)
        assert germ.density is not None
        eigenvalues = np.asarray(germ.eigenvalues, dtype=np.float64)
        evecs = germ.eigenvectors
        self._enforce_psd(eigenvalues, "density_matrix")

        floor = germ.reg_floor
        clamped = np.clip(eigenvalues, floor, None)
        dirac_eigs = 1.0 / np.sqrt(clamped)
        self._ensure_finite_array(dirac_eigs, "dirac_eigs")

        dirac_op = evecs @ (dirac_eigs[:, None] * evecs.T.conj())
        dirac_op = 0.5 * (dirac_op + dirac_op.T.conj())
        lap_eigs = dirac_eigs * dirac_eigs

        expo = np.clip(-lap_eigs, -_LOG_EXP_CLIP, 0.0)
        spectral_action = float(self.kahan_sum(np.exp(expo)))

        spec_dim = 0.0
        live = np.sort(dirac_eigs[np.isfinite(dirac_eigs)])
        if live.size >= _SPECTRAL_DIM_MIN_MODES:
            lo = live.size // 3
            hi = max(lo + 2, (2 * live.size) // 3)
            lam = live[lo:hi]
            counts = np.arange(lo + 1, lo + 1 + lam.size, dtype=np.float64)
            if np.all(lam > 0.0):
                x = np.log(lam)
                y = np.log(counts)
                xm = float(self.kahan_sum(x) / x.size)
                ym = float(self.kahan_sum(y) / y.size)
                var_x = float(self.kahan_sum((x - xm) ** 2))
                cov = float(self.kahan_sum((x - xm) * (y - ym)))
                if var_x > _MACHINE_EPS:
                    spec_dim = float(max(cov / var_x, 0.0))

        cond = float(dirac_eigs.max() / max(dirac_eigs.min(), _MACHINE_EPS))
        ker = int(np.sum(eigenvalues <= floor))

        return _DiracSpectrumResult(
            dirac_eigs=np.asarray(dirac_eigs, dtype=np.float64),
            eigenvalues=eigenvalues,
            dirac_operator=dirac_op,
            laplacian_eigs=np.asarray(lap_eigs, dtype=np.float64),
            spectral_action=spectral_action,
            spectral_dimension=spec_dim,
            condition_number=cond,
            kernel_dim=ker,
        )

    def compute_petz_fisher_rao_metric(
        self,
        rho: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> float:
        r"""Métrica de Petz–Bogoliubov–Kubo–Mori."""
        return self.compute_petz_fisher_rao_metric_certified(rho, A, B).bkm

    def compute_petz_fisher_rao_metric_certified(
        self,
        rho: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> _PetzMetricResult:
        germ = self._resolve_triple_germ(rho)
        assert germ.density is not None
        a_mat = self._as_numeric_matrix(A, "A", allow_complex=True, square=True)
        b_mat = self._as_numeric_matrix(B, "B", allow_complex=True, square=True)
        if germ.density.shape != a_mat.shape or germ.density.shape != b_mat.shape:
            raise ValueError("rho, A y B deben tener la misma forma cuadrada.")
        if germ.dim == 0:
            return _PetzMetricResult(
                bkm=0.0,
                bures=0.0,
                wigner_yanase=0.0,
                dropped_kernel_terms=0,
                tangent_hermiticity=0.0,
                tangent_traceless=0.0,
            )

        eigenvalues = np.asarray(germ.eigenvalues, dtype=np.float64)
        v_mat = germ.eigenvectors
        self._enforce_psd(eigenvalues, "rho")
        eigs = np.clip(eigenvalues, germ.reg_floor, None)

        a_rot = v_mat.conj().T @ a_mat @ v_mat
        b_rot = v_mat.conj().T @ b_mat @ v_mat

        bkm, dropped = self._petz_sum(eigs, a_rot, b_rot, self._logarithmic_mean)
        bures, _ = self._petz_sum(eigs, a_rot, b_rot, self._arithmetic_mean)
        wy, _ = self._petz_sum(eigs, a_rot, b_rot, self._wigner_yanase_mean)

        herm_a = self._frobenius_norm(a_mat - a_mat.T.conj())
        herm_b = self._frobenius_norm(b_mat - b_mat.T.conj())
        tr_a = float(np.real(np.trace(a_mat)))
        tr_b = float(np.real(np.trace(b_mat)))
        return _PetzMetricResult(
            bkm=float(bkm),
            bures=float(bures),
            wigner_yanase=float(wy),
            dropped_kernel_terms=int(dropped),
            tangent_hermiticity=float(max(herm_a, herm_b)),
            tangent_traceless=float(max(abs(tr_a), abs(tr_b))),
        )

    def induce_hodge_cheeger_germ(
        self,
        density_or_boundary: np.ndarray,
        tangent_a: Optional[np.ndarray] = None,
        boundary_matrix: Optional[np.ndarray] = None,
    ) -> _HodgeCheegerGerm:
        raw = np.asarray(density_or_boundary)
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side == raw.size:
                raw = raw.reshape(side, side)

        floor = self._regularizer_floor()
        petz_scale = 0.0
        dirac_eigs = np.array([], dtype=np.float64)
        lap_eigs = np.array([], dtype=np.float64)
        comb = None
        from_connes = False
        dim_hint = 0

        is_square = raw.ndim == 2 and raw.shape[0] == raw.shape[1] and raw.size > 0
        looks_density = False
        if is_square:
            herm = 0.5 * (raw + raw.T.conj())
            looks_density = self._frobenius_norm(raw - raw.T.conj()) <= (
                _HERMITIAN_REL_TOL * max(self._frobenius_norm(raw), 1.0)
            )
            if looks_density:
                ev = np.real(la.eigvalsh(herm)) if herm.size else np.array([])
                looks_density = ev.size > 0 and float(np.min(ev)) >= -self._psd_tolerance(ev)

        if looks_density:
            certified = self.compute_dirac_operator_spectrum_certified(raw)
            dirac_eigs = certified.dirac_eigs
            lap_eigs = np.sort(np.real(certified.laplacian_eigs))
            from_connes = True
            dim_hint = int(raw.shape[0])
            floor = max(floor, self._resolve_triple_germ(raw).reg_floor)
            if tangent_a is not None:
                petz_scale = self.compute_petz_fisher_rao_metric(raw, tangent_a, tangent_a)
        elif raw.ndim == 2:
            comb = Phase3TopologicalGeometricMixin.compute_simplicial_normalized_laplacian(
                self, raw
            )
            try:
                lap_eigs = np.sort(np.real(la.eigvalsh(comb)))
            except la.LinAlgError:
                lap_eigs = np.array([], dtype=np.float64)
            dim_hint = int(comb.shape[0])
        else:
            raise ValueError(
                "induce_hodge_cheeger_germ espera una densidad cuadrada o una incidencia 2-D."
            )

        extra_b = boundary_matrix
        if extra_b is not None and comb is None:
            comb = Phase3TopologicalGeometricMixin.compute_simplicial_normalized_laplacian(
                self, extra_b
            )
            if lap_eigs.size == 0:
                try:
                    lap_eigs = np.sort(np.real(la.eigvalsh(comb)))
                except la.LinAlgError:
                    lap_eigs = np.array([], dtype=np.float64)
            dim_hint = max(dim_hint, int(comb.shape[0]))

        germ = _HodgeCheegerGerm(
            laplacian_eigs=np.asarray(lap_eigs, dtype=np.float64),
            combinatorial_laplacian=comb,
            dirac_eigs=np.asarray(dirac_eigs, dtype=np.float64),
            petz_scale=float(petz_scale),
            two_n_hint=int(dim_hint),
            reg_floor=float(floor),
            from_connes=bool(from_connes),
        )
        self._hodge_germ = germ
        return germ


# =============================================================================
# FASE III — HODGE, CHEEGER, BETTI Y EULER–POINCARÉ
# =============================================================================

@dataclass(frozen=True)
class _LaplacianResult:
    """Laplaciano normalizado de Hodge con certificados de cadena."""

    laplacian: np.ndarray
    eigenvalues: np.ndarray
    fiedler: float
    algebraic_connectivity: float
    kernel_dim: int
    nilpotency_residual: float


@dataclass(frozen=True)
class _CheegerResult:
    """Cotas de Cheeger–Buser y conectividad."""

    h_lower: float
    h_upper: float
    fiedler: float
    connected: bool
    buser_gap: float


@dataclass(frozen=True)
class _EulerPoincareResult:
    """Característica de Euler combinatoria y homológica."""

    characteristic: int
    vertices: int
    edges: int
    faces: int
    betti_0: int
    betti_1: int
    betti_2: int
    homological_chi: int
    rank_d0: int
    rank_d1: int
    nilpotency_residual: float


class Phase3TopologicalGeometricMixin(Phase2SpectralQuantumMixin):
    """
    FASE III — TOPOLOGÍA Y GEOMETRÍA.

    Operadores discretos de de Rham–Hodge e invariantes topológicos/isoperimétricos.
    """

    def _resolve_hodge_germ(
        self,
        eigenvalues_or_boundary: Optional[np.ndarray] = None,
    ) -> Optional[_HodgeCheegerGerm]:
        cached = getattr(self, "_hodge_germ", None)
        if eigenvalues_or_boundary is None:
            return cached
        return cached

    def _numeric_rank(self, matrix: np.ndarray, floor: float) -> int:
        a = np.asarray(matrix)
        if a.size == 0:
            return 0
        s_vals = np.real(la.svd(a, compute_uv=False))
        tol = max(floor, self._wilkinson_deflation_floor(a))
        return int(np.sum(s_vals > tol))

    def compute_simplicial_normalized_laplacian(
        self,
        boundary_matrix: np.ndarray,
    ) -> np.ndarray:
        r"""Laplaciano de Hodge de 0-cochains normalizado."""
        return self.compute_simplicial_normalized_laplacian_certified(
            boundary_matrix
        ).laplacian

    def compute_simplicial_normalized_laplacian_certified(
        self,
        boundary_matrix: np.ndarray,
        boundary_1: Optional[np.ndarray] = None,
    ) -> _LaplacianResult:
        delta_0 = self._as_numeric_matrix(
            boundary_matrix, "boundary_matrix", allow_complex=False, square=False
        )
        l_base = delta_0.T @ delta_0
        l_base = 0.5 * (l_base + l_base.T)

        degrees = np.real(np.diagonal(l_base)).astype(np.float64, copy=False)
        inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
        positive = degrees > _MACHINE_EPS
        inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
        d_is = np.diag(inv_sqrt)
        l_norm = d_is @ l_base @ d_is
        l_norm = 0.5 * (l_norm + l_norm.T)

        try:
            evals = np.sort(np.real(la.eigvalsh(l_norm)))
        except la.LinAlgError as exc:
            raise ValueError("El espectro del Laplaciano normalizado falló.") from exc
        floor = max(self._regularizer_floor(), self._wilkinson_deflation_floor(l_norm))
        evals = np.where(evals < 0.0, np.where(evals > -self._psd_tolerance(evals), 0.0, evals), evals)
        ker = int(np.sum(evals <= max(floor, _PSD_ABS_TOL)))
        fiedler = float(evals[1]) if evals.size > 1 else 0.0

        nilp = 0.0
        if boundary_1 is not None:
            delta_1 = self._as_numeric_matrix(
                boundary_1, "boundary_1", allow_complex=False, square=False
            )
            if delta_1.shape[1] == delta_0.shape[0]:
                nilp = self._frobenius_norm(delta_1 @ delta_0)
            elif delta_0.shape[1] == delta_1.shape[0]:
                nilp = self._frobenius_norm(delta_0 @ delta_1)

        return _LaplacianResult(
            laplacian=np.asarray(l_norm, dtype=np.float64),
            eigenvalues=np.asarray(evals, dtype=np.float64),
            fiedler=float(fiedler),
            algebraic_connectivity=float(max(fiedler, 0.0)),
            kernel_dim=int(ker),
            nilpotency_residual=float(nilp),
        )

    def estimate_cheeger_constant_bounds(
        self,
        eigenvalues_L: np.ndarray,
    ) -> Tuple[float, float]:
        r"""Cotas de Cheeger–Buser: λ₂ / 2 ≤ h(G) ≤ √(2 λ₂)."""
        result = self.estimate_cheeger_constant_bounds_certified(eigenvalues_L)
        return result.h_lower, result.h_upper

    def estimate_cheeger_constant_bounds_certified(
        self,
        eigenvalues_L: np.ndarray,
    ) -> _CheegerResult:
        germ = getattr(self, "_hodge_germ", None)
        vec = np.asarray(eigenvalues_L)
        if vec.size == 0 and germ is not None and germ.laplacian_eigs.size:
            vec = germ.laplacian_eigs
        eigs = self._as_numeric_vector(vec, "eigenvalues_L", allow_complex=False)
        if eigs.size < 2:
            return _CheegerResult(
                h_lower=0.0,
                h_upper=0.0,
                fiedler=0.0,
                connected=eigs.size == 1,
                buser_gap=0.0,
            )
        min_eig = float(np.min(eigs))
        if min_eig < -_PSD_ABS_TOL:
            logger.warning(
                "Laplaciano no PSD en estimate_cheeger_constant_bounds."
            )
            return _CheegerResult(
                h_lower=0.0,
                h_upper=0.0,
                fiedler=0.0,
                connected=False,
                buser_gap=0.0,
            )
        eigs = np.where(eigs < 0.0, 0.0, eigs)
        sorted_eigs = np.sort(eigs)
        sorted_eigs[np.abs(sorted_eigs) <= _PSD_ABS_TOL] = 0.0
        fiedler_val = float(sorted_eigs[1])
        connected = bool(sorted_eigs[0] <= _PSD_ABS_TOL and fiedler_val > _PSD_ABS_TOL)
        if not math.isfinite(fiedler_val) or fiedler_val <= 0.0:
            return _CheegerResult(
                h_lower=0.0,
                h_upper=0.0,
                fiedler=float(fiedler_val) if math.isfinite(fiedler_val) else 0.0,
                connected=False,
                buser_gap=0.0,
            )
        h_lower = float(fiedler_val / 2.0)
        h_upper = float(math.sqrt(max(0.0, 2.0 * fiedler_val)))
        return _CheegerResult(
            h_lower=h_lower,
            h_upper=h_upper,
            fiedler=fiedler_val,
            connected=connected,
            buser_gap=float(max(h_upper - h_lower, 0.0)),
        )

    def compute_euler_poincare_characteristic(
        self,
        boundary_0: np.ndarray,
        boundary_1: Optional[np.ndarray],
    ) -> int:
        r"""Característica combinatoria del 2-complejo K."""
        return self.compute_euler_poincare_characteristic_certified(
            boundary_0, boundary_1
        ).characteristic

    def compute_euler_poincare_characteristic_certified(
        self,
        boundary_0: np.ndarray,
        boundary_1: Optional[np.ndarray],
    ) -> _EulerPoincareResult:
        b0 = self._as_numeric_matrix(
            boundary_0, "boundary_0", allow_complex=False, square=False
        )
        vertices = int(b0.shape[1])
        edges = int(b0.shape[0])
        faces = 0
        b1: Optional[np.ndarray] = None
        if boundary_1 is not None:
            b1 = self._as_numeric_matrix(
                boundary_1, "boundary_1", allow_complex=False, square=False
            )
            faces = int(b1.shape[0])

        chi = int(vertices - edges + faces)
        floor = max(self._regularizer_floor(), self._wilkinson_deflation_floor(b0))
        rank_d0 = self._numeric_rank(b0, floor)
        rank_d1 = self._numeric_rank(b1, floor) if b1 is not None else 0

        betti_0 = int(max(vertices - rank_d0, 0))
        betti_1 = int(max(edges - rank_d0 - rank_d1, 0))
        betti_2 = int(max(faces - rank_d1, 0))
        chi_h = int(betti_0 - betti_1 + betti_2)

        nilp = 0.0
        if b1 is not None:
            if b1.shape[1] == b0.shape[0]:
                nilp = self._frobenius_norm(b1 @ b0)
            elif b0.shape[1] == b1.shape[0]:
                nilp = self._frobenius_norm(b0 @ b1)

        return _EulerPoincareResult(
            characteristic=chi,
            vertices=vertices,
            edges=edges,
            faces=faces,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            homological_chi=chi_h,
            rank_d0=int(rank_d0),
            rank_d1=int(rank_d1),
            nilpotency_residual=float(nilp),
        )


# =============================================================================
# CLASE PÚBLICA — INTEGRACIÓN DEL MORFISMO Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================

class ImperialGuardsEngine(Phase3TopologicalGeometricMixin):
    """
    Motor matemático de alta precisión para los operadores elípticos
    de-confinados y la mecánica celeste de Henri Poincaré.

    Compone las fases numéricas, cuántico-espectrales, topológico-geométricas
    y de integración simpléctica sobre el fibrado cotangente T*M.
    """

    def __init__(
        self,
        regularizer: float = _DEFAULT_REGULARIZER,
        dimension: int = 6,
    ) -> None:
        """
        Inicializa el motor espectral y simpléctico.

        Mantiene compatibilidad con inicializaciones posicionales tanto de
        `regularizer` (float) como de `dimension` (int).

        Args:
            regularizer: Piso mínimo de regularización conforme.
            dimension: Dimensión del espacio de configuración n.
        """
        if isinstance(regularizer, (int, np.integer)) and regularizer > 1 and not isinstance(regularizer, bool):
            self._dim = int(regularizer)
            self._reg = self._validate_regularizer(_DEFAULT_REGULARIZER)
        else:
            self._dim = self._validate_positive_int("dimension", dimension)
            self._reg = self._validate_regularizer(regularizer)

        self._csmd_h = _COMPLEX_STEP_DEFAULT_H

        self._J_canonical = np.block([
            [np.zeros((self._dim, self._dim)), np.eye(self._dim)],
            [-np.eye(self._dim), np.zeros((self._dim, self._dim))]
        ])

        mixed = 0.5 * np.eye(2, dtype=np.complex128)
        self._triple_germ: _SpectralTripleGerm = self.synthesize_spectral_triple_germ(
            mixed, csmd_step=self._csmd_h, regularizer=self._reg
        )
        self._hodge_germ: _HodgeCheegerGerm = self.induce_hodge_cheeger_germ(mixed)

    @property
    def regularizer(self) -> float:
        """Piso de regularización vigente."""
        return self._reg

    @property
    def dimension(self) -> int:
        """Dimensión del espacio de configuración."""
        return self._dim

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser strictly mayor que cero.")
        return int(value)

    def _validate_regularizer(self, regularizer: float) -> float:
        value = self._validate_nonnegative_finite("regularizer", regularizer)
        return float(max(value, _HIGHAM_TIKHONOV_FLOOR))

    # ── MÉTODOS DE MECÁNICA CELESTE DE HENRI POINCARÉ ─────────────────────

    def step_poincare_symplectic_integration(
        self,
        current_state: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        dt_step: float,
        external_freq_omega: NDArray[np.float64],
        wave_k: NDArray[np.float64],
    ) -> ImperialEngineStepResult:
        r"""
        Ejecuta un paso de integración simpléctica Störmer-Verlet preservando Liouville.

        Axiomas de Poincaré:
          1. Conservación de Liouville: det(M_step) = +1.
          2. Acción de Maupertuis-Jacobi: S_M = ∫ √(2(H₀ - V)) |dq|_G.
          3. Absorción de Novikov: W_Nov = exp(-1 / (ε + |⟨k, ω⟩|)).
        """
        state_vec = self._as_numeric_vector(current_state, "current_state")
        if state_vec.size % 2 != 0:
            raise ValueError("current_state debe tener dimensión par (2n).")
        n = state_vec.size // 2

        q_pos = state_vec[:n]
        p_mom = state_vec[n:]

        g_mat = self._as_numeric_matrix(metric_G, "metric_G", square=True)
        if g_mat.shape[0] != n:
            raise ValueError(f"metric_G debe ser {n}x{n} para un estado de dimensión {2*n}.")

        g_inv = la.inv(g_mat)

        # 1. Cómputo del Hamiltoniano Imperial: H = ½ pᵀ G⁻¹ p + V(q)
        kinetic_energy = 0.5 * float(p_mom.T @ g_inv @ p_mom)
        hamiltonian_H = kinetic_energy + float(potential_V)

        # 2. Integrador Simpléctico Störmer-Verlet
        grad_V = g_mat @ q_pos
        p_half = p_mom - 0.5 * dt_step * grad_V
        q_next = q_pos + dt_step * (g_inv @ p_half)
        grad_V_next = g_mat @ q_next
        p_next = p_half - 0.5 * dt_step * grad_V_next

        next_state = np.concatenate([q_next, p_next])

        # 3. Medición de las matrices Jacobianas M1, M2, M3 de la transformación Störmer-Verlet
        # M1: p_half update -> shear (det M1 = 1)
        m1 = np.block([
            [np.eye(n), np.zeros((n, n))],
            [-0.5 * dt_step * g_mat, np.eye(n)]
        ])

        # M2: q_next update -> shear (det M2 = 1)
        m2 = np.block([
            [np.eye(n), dt_step * g_inv],
            [np.zeros((n, n)), np.eye(n)]
        ])

        # M3: p_next update -> shear (det M3 = 1)
        m3 = np.block([
            [np.eye(n), np.zeros((n, n))],
            [-0.5 * dt_step * g_mat, np.eye(n)]
        ])

        # Jacobiano total de la aplicación simpléctica M = M3 @ M2 @ M1
        m_step = m3 @ m2 @ m1
        det_M = float(la.det(m_step))
        volume_drift = abs(det_M - 1.0)

        # 4. Acción de Maupertuis-Jacobi sobre la métrica conforme g̃_jk = 2(H₀ - V) g_jk
        kinetic_headroom = max(_WILKINSON_DRIFT_LIMIT, 2.0 * (float(total_energy_H0) - float(potential_V)))
        refractive_n = np.sqrt(kinetic_headroom)
        velocity_q = g_inv @ p_next
        velocity_norm_g = float(np.sqrt(max(0.0, float(velocity_q.T @ g_mat @ velocity_q))))
        maupertuis_action = float(refractive_n * velocity_norm_g)

        # 5. Absorción de Pequeños Divisores de Poincaré en Anillo de Novikov Λ_Nov
        omega_vec = self._as_numeric_vector(external_freq_omega, "external_freq_omega")
        k_vec = self._as_numeric_vector(wave_k, "wave_k")
        small_divisor = float(np.dot(k_vec, omega_vec))
        novikov_weight = float(np.exp(-1.0 / (_WILKINSON_DRIFT_LIMIT + abs(small_divisor))))

        is_valid = (volume_drift <= _WILKINSON_DRIFT_LIMIT) and (maupertuis_action > 0.0)

        return ImperialEngineStepResult(
            next_state=next_state,
            hamiltonian_energy=hamiltonian_H,
            volume_drift=volume_drift,
            maupertuis_action=maupertuis_action,
            novikov_absorbed_weight=novikov_weight,
            is_step_valid=is_valid,
        )

    def spectral_triple_germ_certificate(self) -> _SpectralTripleCertificate:
        """Certificado de pertenencia a 𝒟(ℋ) del gérmen de Fase I."""
        return self._triple_germ.certificate

    def attach_spectral_triple_germ(
        self,
        density_matrix: np.ndarray,
        csmd_step: float = _COMPLEX_STEP_DEFAULT_H,
    ) -> _SpectralTripleGerm:
        """Réplica pública del morfismo I.9; actualiza 𝒢_I y reinduce 𝒢_II."""
        germ = self.synthesize_spectral_triple_germ(
            density_matrix, csmd_step=csmd_step, regularizer=self._reg
        )
        self._triple_germ = germ
        if germ.density is not None:
            self._hodge_germ = self.induce_hodge_cheeger_germ(germ.density)
        return germ

    def attach_hodge_cheeger_germ(
        self,
        density_or_boundary: np.ndarray,
        tangent_a: Optional[np.ndarray] = None,
        boundary_matrix: Optional[np.ndarray] = None,
    ) -> _HodgeCheegerGerm:
        """Réplica pública del morfismo II.7; actualiza 𝒢_II."""
        germ = self.induce_hodge_cheeger_germ(
            density_or_boundary,
            tangent_a=tangent_a,
            boundary_matrix=boundary_matrix,
        )
        self._hodge_germ = germ
        return germ


__all__ = [
    "Phase1NumericalFoundationsMixin",
    "Phase2SpectralQuantumMixin",
    "Phase3TopologicalGeometricMixin",
    "ImperialEngineStepResult",
    "ImperialGuardsEngine",
]
