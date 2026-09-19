
from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Scalar Momentum Satellite Engine (Motor de Transferencia de Momentum)║
║ Ruta   : app/core/immune_system/scalar_momentum_satellite_engine.py          ║
║ Versión: 1.1.0-Doctoral-Lie-CSMD-StressEnergy-KBN-FPU-Secure                 ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:                                  ║
║ Este módulo implementa el motor de cálculo ciego en la FPU para la          ║
║ transferencia de cantidad de movimiento (momentum covariante $p_\mu$) sobre  ║
║ un campo escalar $\phi \in C^\infty(\mathcal{M})$ en el Cinturón Orbital de  ║
║ Frontera ($\partial \mathcal{M} \neq \varnothing$).                          ║
║                                                                              ║
║ Evalúa la derivada de Lie $\mathcal{L}_v \phi = p_\mu G^{\mu\nu} \partial_\nu \phi$  ║
║ mediante Diferenciación por Paso Complejo (CSMD) holomorfa, construye el    ║
║ Tensor de Energía-Impulso $T_{\mu\nu}[\phi, p]$ con sumación compensada de   ║
║ Kahan-Babuška-Neumaier (KBN), y audita la pasividad termodinámica $P_{\mathrm{diss}} \ge 0$.║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA FPU):                   ║
║   Fase 1  Observe  : Ingesta, CSMD de $d\phi$, Cholesky de $G$, $\kappa_2(G)$║
║   Fase 2  Orient   : Derivada de Lie $\mathcal{L}_v \phi$, Tensor $T_{\mu\nu}$, Pasividad  ║
║   Fase 3  Act      : Telemetría FPU, Sello Inmutable SHA-256                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Final, Optional, Tuple, Dict, Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Physics.ScalarMomentumSatelliteEngine")

# Constantes de precisión metrológica e inercia de máquina
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CSMD_STEP: Final[float] = 1.0e-20
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_WILKINSON_FLOOR: Final[float] = 1.0e-15


class ScalarMomentumEngineError(Exception):
    """Excepción raíz para errores del motor de momentum escalar."""
    pass


class DimensionMismatchError(ScalarMomentumEngineError):
    """Excepción lanzada ante inconsistencia de dimensiones tensoriales."""
    pass


class MetricIndefinitenessError(ScalarMomentumEngineError):
    """Excepción lanzada si el tensor métrico G no es estrictamente SPD."""
    pass


class CSMDHolomorphyError(ScalarMomentumEngineError):
    """Excepción lanzada ante pérdida de holomorfía en la diferenciación por paso complejo."""
    pass


@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    r"""Caché espectral inmutable del tensor métrico $G_{\mu\nu}$."""
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    condition_number: float
    dimension: int


@dataclass(frozen=True, slots=True)
class ScalarMomentumKernel:
    r"""Expediente inmutable de la Fase 1 (Observe)."""
    phi_scalar: float
    gradient_dphi: NDArray[np.float64]
    momentum_p: NDArray[np.float64]
    metric_cache: BaseMetricCache
    sha256_seal: str


@dataclass(frozen=True, slots=True)
class MomentumTransferReport:
    r"""Expediente inmutable de la Fase 2 (Orient)."""
    lie_derivative_transfer: float
    stress_energy_tensor: NDArray[np.float64]
    stress_energy_trace: float
    dissipated_power: float
    is_passivity_satisfied: bool


@dataclass(frozen=True, slots=True)
class ScalarMomentumEngineState:
    r"""Certificado global e inmutable emitido por la Fase 3 (Act) en la FPU."""
    kernel: ScalarMomentumKernel
    transfer_report: MomentumTransferReport
    fpu_execution_time_ms: float
    cryptographic_seal: str


class KahanNeumaierSum:
    r"""
    Sumación compensada de Kahan-Babuška-Neumaier (KBN).
    Neutraliza el ruido de redondeo de Wilkinson acumulando el residuo en la mantisa IEEE-754.
    """

    @staticmethod
    def sum(arr: NDArray[np.float64]) -> float:
        r"""Calcula la suma compensada de un vector real de dimensión $N$."""
        s = 0.0
        c = 0.0
        for x in arr:
            t = s + x
            if abs(s) >= abs(x):
                c += (s - t) + x
            else:
                c += (x - t) + s
            s = t
        return float(s + c)


class Phase1_ScalarMomentumObserver:
    r"""
    FASE 1 — Observe: Ingesta de tensores, cálculo de $d\phi$ mediante CSMD holomorfo,
    inversión estable por Cholesky y metrología del tensor métrico de fondo $G_{\mu\nu}$.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        self._tol: Final[float] = float(tolerance)

    def _extract_scalar_gradient(
        self,
        phi_func: Any,
        x_point: NDArray[np.float64],
        step: float = _CSMD_STEP
    ) -> NDArray[np.float64]:
        r"""
        Calcula el gradiente $d\phi = \partial_\mu \phi \, dx^\mu$ mediante Diferenciación
        por Paso Complejo (CSMD) holomorfa sobre la FPU:
        
        $$\partial_\mu \phi = \frac{\operatorname{Im}\left(\phi(x + j \cdot h \cdot e_\mu)\right)}{h} + \mathcal{O}(h^2)$$
        """
        dim = x_point.shape[0]
        dphi = np.zeros(dim, dtype=np.float64)
        j_unit = 1j

        for k in range(dim):
            x_complex = np.array(x_point, dtype=np.complex128)
            x_complex[k] += j_unit * step
            phi_val = phi_func(x_complex)
            
            if not isinstance(phi_val, (complex, np.complex128)):
                # Si la función no soporta paso complejo, usar diferencia finita equilibrada KBN
                x_plus = np.array(x_point, dtype=np.float64)
                x_minus = np.array(x_point, dtype=np.float64)
                h_real = 1.0e-8
                x_plus[k] += h_real
                x_minus[k] -= h_real
                dphi[k] = (phi_func(x_plus) - phi_func(x_minus)) / (2.0 * h_real)
            else:
                dphi[k] = float(np.imag(phi_val)) / step

        return dphi

    def _audit_metric_regularity(self, G_metric: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Somete al tensor métrico $G_{\mu\nu}$ a factorización Cholesky ($G = L L^\top \succ 0$),
        invierte de forma de-confinada y calcula el número de condición $\kappa_2(G)$.
        """
        dim = G_metric.shape[0]
        if G_metric.shape != (dim, dim):
            raise DimensionMismatchError(f"Tensor métrico $G$ debe ser cuadrado. Forma: {G_metric.shape}")

        # Verificación de simetría Frobenius
        sym_res = float(la.norm(G_metric - G_metric.T, 'fro'))
        if sym_res > 1.0e-10:
            G_metric = 0.5 * (G_metric + G_metric.T)

        try:
            L_cholesky = la.cholesky(G_metric, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(
                f"El tensor métrico de fondo $G$ no es strictly Definido Positivo (SPD): {exc}"
            ) from exc

        # Inversión estable por Cholesky
        L_inv = la.solve_triangular(L_cholesky, np.eye(dim, dtype=np.float64), lower=True)
        G_inv = L_inv.T @ L_inv

        # Número de condición espectral \kappa_2(G)
        eigvals = la.eigvalsh(G_metric)
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))

        if min_eig <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                f"Autovalor mínimo de $G$ colapsó por debajo del límite de Wilkinson: {min_eig:.6e}"
            )

        cond_num = max_eig / min_eig
        if cond_num > _CONDITION_NUMBER_MAX:
            logger.warning(
                "¡Advertencia de metrología!: Número de condición $\\kappa_2(G) = %.4e$ excede umbral de seguridad.",
                cond_num
            )

        return BaseMetricCache(
            g_base=G_metric.copy(),
            cholesky_factor=L_cholesky,
            g_inv=G_inv,
            condition_number=cond_num,
            dimension=dim
        )

    def observe_scalar_field(
        self,
        phi_func: Any,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> ScalarMomentumKernel:
        r"""
        Ejecuta la Fase 1: Ingesta, CSMD de gradiente, Cholesky de métrica y empaquetado del Kernel.
        """
        if x_point.ndim != 1 or momentum_p.ndim != 1:
            raise DimensionMismatchError("Punto de evaluación $x$ y Momentum $p$ deben ser vectores 1D.")

        if x_point.shape[0] != momentum_p.shape[0]:
            raise DimensionMismatchError(
                f"Dimensión de $x$ ({x_point.shape[0]}) no coincide con Momentum $p$ ({momentum_p.shape[0]})."
            )

        cache = self._audit_metric_regularity(G_metric)
        if cache.dimension != x_point.shape[0]:
            raise DimensionMismatchError(
                f"Dimensión de métrica $G$ ({cache.dimension}) no coincide con espacio de fase ({x_point.shape[0]})."
            )

        # Evaluar escalar \phi(x)
        phi_val = float(np.real(phi_func(x_point)))

        # Extraer gradiente d\phi via CSMD
        dphi = self._extract_scalar_gradient(phi_func, x_point)

        # Sello criptográfico SHA-256 de Fase 1
        sha = hashlib.sha256()
        sha.update(x_point.tobytes())
        sha.update(momentum_p.tobytes())
        sha.update(dphi.tobytes())
        sha.update(G_metric.tobytes())
        seal = sha.hexdigest()

        return ScalarMomentumKernel(
            phi_scalar=phi_val,
            gradient_dphi=dphi,
            momentum_p=momentum_p.copy(),
            metric_cache=cache,
            sha256_seal=seal
        )


class Phase2_CovariantTransferOrient(Phase1_ScalarMomentumObserver):
    r"""
    FASE 2 — Orient: Cálculo covariante de la derivada de Lie $\mathcal{L}_v \phi = p_\mu G^{\mu\nu} \partial_\nu \phi$,
    construcción del Tensor de Energía-Impulso $T_{\mu\nu}$ y verificación de pasividad $P_{\mathrm{diss}} \ge 0$.
    """

    def _compute_lie_momentum_transfer(
        self,
        p_mom: NDArray[np.float64],
        d_phi: NDArray[np.float64],
        G_inv: NDArray[np.float64]
    ) -> float:
        r"""
        Calcula la transferencia de momentum sobre el campo escalar a lo largo del flujo de velocidad $v^\mu = G^{\mu\nu} p_\nu$:
        
        $$\Xi_{\mathrm{transfer}} = \mathcal{L}_v \phi = p_\mu G^{\mu\nu} \partial_\nu \phi$$
        
        utilizando la sumación compensada de Kahan-Neumaier.
        """
        v_velocity = G_inv @ p_mom
        terms = v_velocity * d_phi
        transfer_val = KahanNeumaierSum.sum(terms)
        return transfer_val

    def _build_stress_energy_tensor(
        self,
        p_mom: NDArray[np.float64],
        d_phi: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        phi_val: float,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Construye el Tensor de Energía-Impulso de de Rham acoplado al momentum:
        
        $$T_{\mu\nu}[\phi, p] = \partial_\mu \phi \partial_\nu \phi - \frac{1}{2} G_{\mu\nu} \left( G^{\alpha\beta} \partial_\alpha \phi \partial_\beta \phi + m^2 \phi^2 \right) + \alpha \left( p_\mu \partial_\nu \phi + p_\nu \partial_\mu \phi \right)$$
        """
        dim = d_phi.shape[0]
        
        # Norm al cuadrado del gradiente: G^{\alpha\beta} \partial_\alpha \phi \partial_\beta \phi
        grad_norm_sq = KahanNeumaierSum.sum((G_inv @ d_phi) * d_phi)
        
        # Término cinético-potencial escalar: \frac{1}{2} ( \|d\phi\|^2_{G^{-1}} + m^2 \phi^2 )
        scalar_density = 0.5 * (grad_norm_sq + (mass_m ** 2) * (phi_val ** 2))
        
        # Producto exterior de gradiente: \partial_\mu \phi \partial_\nu \phi
        outer_dphi = np.outer(d_phi, d_phi)
        
        # Término de acoplamiento de momentum: p_\mu \partial_\nu \phi + p_\nu \partial_\mu \phi
        outer_coupling = np.outer(p_mom, d_phi) + np.outer(d_phi, p_mom)
        
        # Tensor completo T_{\mu\nu}
        T_tensor = outer_dphi - scalar_density * G_metric + coupling_alpha * outer_coupling
        
        # Traza contravariante: Tr(T) = G^{\mu\nu} T_{\mu\nu}
        trace_T = KahanNeumaierSum.sum(np.diag(G_inv @ T_tensor))
        
        return T_tensor, float(trace_T)

    def _verify_dissipation_passivity(
        self,
        transfer_val: float,
        d_phi: NDArray[np.float64],
        G_inv: NDArray[np.float64]
    ) -> Tuple[float, bool]:
        r"""
        Verifica la pasividad termodinámica: $P_{\mathrm{diss}} = \langle d\phi, G^{-1} d\phi \rangle \ge 0$.
        """
        p_diss = KahanNeumaierSum.sum((G_inv @ d_phi) * d_phi)
        is_passive = (p_diss >= -_WILKINSON_FLOOR)
        return float(p_diss), bool(is_passive)

    def orient_momentum_transfer(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0
    ) -> MomentumTransferReport:
        r"""
        Orquesta la Fase 2: Derivada de Lie, Tensor de Esfuerzos $T_{\mu\nu}$ y reporte de pasividad.
        """
        p_mom = kernel.momentum_p
        d_phi = kernel.gradient_dphi
        G_inv = kernel.metric_cache.g_inv
        G_metric = kernel.metric_cache.g_base
        phi_val = kernel.phi_scalar

        # 1. Transferencia de Lie \mathcal{L}_v \phi
        lie_transfer = self._compute_lie_momentum_transfer(p_mom, d_phi, G_inv)

        # 2. Tensor de Energía-Impulso T_{\mu\nu}
        T_tensor, trace_T = self._build_stress_energy_tensor(
            p_mom, d_phi, G_metric, G_inv, phi_val, coupling_alpha=coupling_alpha, mass_m=mass_m
        )

        # 3. Verificación de Pasividad
        p_diss, is_passive = self._verify_dissipation_passivity(lie_transfer, d_phi, G_inv)

        return MomentumTransferReport(
            lie_derivative_transfer=lie_transfer,
            stress_energy_tensor=T_tensor,
            stress_energy_trace=trace_T,
            dissipated_power=p_diss,
            is_passivity_satisfied=is_passive
        )


class Phase3_ScalarMomentumEngine(Phase2_CovariantTransferOrient):
    r"""
    FASE 3 — Act: Orquestador supremo en la FPU. Cómputo ciego, registro de tiempo de ejecución
    y sellado inmutable SHA-256 de la sesión del satélite de momentum.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        super().__init__(tolerance=tolerance)

    def execute_momentum_transfer_audit(
        self,
        phi_func: Any,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0
    ) -> ScalarMomentumEngineState:
        r"""
        Orquesta la ejecución completa de las tres fases anidadas en la FPU Secure.
        """
        t_start = time.perf_counter()

        # Fase 1: Observe
        kernel = self.observe_scalar_field(phi_func, x_point, momentum_p, G_metric)

        # Fase 2: Orient
        report = self.orient_momentum_transfer(kernel, coupling_alpha=coupling_alpha, mass_m=mass_m)

        t_elapsed_ms = float((time.perf_counter() - t_start) * 1000.0)

        # Sello inmutable de la sesión en la FPU
        sha = hashlib.sha256()
        sha.update(kernel.sha256_seal.encode("utf-8"))
        sha.update(report.stress_energy_tensor.tobytes())
        sha.update(f"{report.lie_derivative_transfer:.12e}".encode("utf-8"))
        sha.update(f"{report.dissipated_power:.12e}".encode("utf-8"))
        cryptographic_seal = sha.hexdigest()

        return ScalarMomentumEngineState(
            kernel=kernel,
            transfer_report=report,
            fpu_execution_time_ms=t_elapsed_ms,
            cryptographic_seal=cryptographic_seal
        )


class ScalarMomentumSatelliteEngine(Phase3_ScalarMomentumEngine):
    r"""
    Motor de Transferencia de Momentum Escalar de-confinado en la FPU (Cinturón Orbital).
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        super().__init__(tolerance=tolerance)


__all__ = [
    "ScalarMomentumSatelliteEngine",
    "BaseMetricCache",
    "ScalarMomentumKernel",
    "MomentumTransferReport",
    "ScalarMomentumEngineState",
    "ScalarMomentumEngineError",
    "DimensionMismatchError",
    "MetricIndefinitenessError",
    "CSMDHolomorphyError",
    "KahanNeumaierSum",
]
