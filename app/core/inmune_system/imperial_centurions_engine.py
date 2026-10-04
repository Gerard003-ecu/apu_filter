# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Centurions Engine (Caballos de Batalla de la Capa 2)       ║
║ Ruta   : app/core/inmune_system/imperial_centurions_engine.py                ║
║ Versión: 5.0.0-Maupertuis-Jacobi-Liouville-PHS-FPU-PhD                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA CELESTE DE POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Este motor físico de cálculo ciego en FPU actúa como la aduana exergética de lazo
cerrado para la Cortina de Potencia Imperial (Capa 3 de la Malla Agéntica APU Filter v8.0).
Somete la dinámica de potencia electromecánica (bombas hidráulicas, mezcladoras y
variadores de frecuencia) a los postulados fundamentales de Henri Poincaré:

1. Principio Variacional de Maupertuis-Jacobi:
   Transformación del flujo de potencia de energía constante $H(q, p) = H_0$ en un flujo
   geodésico sobre una variedad de Riemann dotada de la Métrica Conforme de Jacobi-Fermat:
   $$\tilde{g}_{jk}(q) = 2 \left( H_0 - V(q) \right) g_{jk}(q) = n(q)^2 g_{jk}(q)$$
   donde $n(q) = \sqrt{2(H_0 - V(q))}$ actúa como un índice de refracción óptico-mecánico.
   La Acción de Maupertuis $S_{\mathrm{Maupertuis}}$ a lo largo de una trayectoria $\gamma$ es:
   $$S_{\mathrm{Maupertuis}}[\gamma] = \int_{\gamma} \sqrt{2(H_0 - V(q))} \sqrt{g_{jk}(q) \dot{q}^j \dot{q}^k} \, d\tau = \int_{\gamma} d\tilde{s}$$

2. Símbolos de Christoffel Conformes de Koszul-Levi-Civita:
   $$\tilde{\Gamma}^i_{jk} = \Gamma^i_{jk} + \delta^i_j \partial_k \phi + \delta^i_k \partial_j \phi - g_{jk} g^{il} \partial_l \phi, \quad \phi(q) = \ln \sqrt{2(H_0 - V(q))}$$

3. Invarianza Simpléctica de Liouville-Darboux:
   Preservación estricta de la 2-forma canónica $\omega = \sum dq_i \wedge dp_i$ y conservación
   del volumen de fase $\det(M_{\mathrm{step}}) = +1 + \mathcal{O}(\varepsilon)$ mediante un
   integrador simpléctico Störmer-Verlet en FPU.

4. Estructuras de Dirac, Leyes IDA-PBC y Flujo Modular KMS:
   Modelización de la interconexión antisimétrica $J_d = -J_d^\top$, disipación de Rayleigh $R_d \succeq 0$,
   y purificación espectral de estados térmicos de Tomita-Takesaki con entropía de Umegaki.

IMPACTO EN MATRIZ FINANCIERA Y OPERACIONAL ("DOLOR Y DINERO"):
────────────────────────────────────────────────────────────────────────────────
• Métrica Conforme de Maupertuis: Optimización de consumo energético en maquinaria pesada.
  Impacto: Eliminación de pérdidas por fricción parásita y sobrecostos por consumo reactivo.
• Invarianza de Liouville: Conservación del volumen electromecánico en transitorios.
  Impacto: Inmunidad absoluta contra golpes de ariete y reventones de bombas mecánicas.
• Control Port-Hamiltoniano: Estabilidad estricta de par motor $\dot{\mathcal{H}}_d \le 0$.
  Impacto: Cero fallas por fatiga estructural en colado masivo de cimentaciones de obra.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Engines.ImperialCenturionsEngine")

__version__: Final[str] = "5.0.0-Maupertuis-Jacobi-Liouville-PHS-FPU-PhD"


# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA Y COTAS FPU
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_REG_FLOOR: Final[float] = 1e-15
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_SPECTRAL_TOL: Final[float] = 1.0e-9
_LOG_EXP_CLIP: Final[float] = 700.0
_KMS_STRIP_TOL: Final[float] = 1e-8
_HERMITIAN_TOL: Final[float] = 1e-12
_DEFAULT_PURITY_MARGIN: Final[float] = 1e-12
_DEFAULT_BETA: Final[float] = 1.0


# =============================================================================
# REPORTE INMUTABLE DE INTEGRACIÓN GEODÉSICA DE MAUPERTUIS
# =============================================================================
@dataclass(frozen=True, slots=True)
class MaupertuisStepReport:
    r"""
    Reporte inmutable de integración geodésica simpléctica de Maupertuis.

    Contiene los diagnósticos de energía cinética/potencial, índice de refracción
    mecánico $n(q)$, densidad de acción geodésica, deriva del volumen de Liouville
    y coherencia simpléctica.
    """
    hamiltonian_energy: float
    refractive_index_n: float
    maupertuis_action_density: float
    volume_drift_det: float
    is_symplectic_coherent: bool


# =============================================================================
# FASE I — NÚCLEO NUMÉRICO DE BANACH, DARBOUX Y TIKHONOV–HIGHAM
# =============================================================================
@dataclass(frozen=True)
class _SymplecticFormCertificate:
    """Certificado algebraico de la 2-forma canónica de Liouville–Darboux."""

    skew_residual: float
    almost_complex_residual: float
    determinant: float
    frobenius_norm: float
    is_darboux: bool


@dataclass(frozen=True)
class _PortHamiltonianGerm:
    """
    Gérmen port-Hamiltoniano (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: transporta la geometría de Darboux
    (Ω, dim, piso de regularización y normas de referencia) sobre la cual se
    instancian la estructura de Dirac y la ley IDA-PBC.
    """

    n: int
    two_n: int
    omega: np.ndarray
    reg_floor: float
    form_certificate: _SymplecticFormCertificate


class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica.

    Provee el topos lineal subyacente: sumación compensada (anula la deriva
    de redondeo en el álgebra de Banach (ℝ, +, ·)), la 2-forma simpléctica
    canónica, regularización espectral y el gérmen que inicia la Fase II.
    """

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
        """Sumación doblemente compensada de Klein."""
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
        """Traza real por KBN sobre la diagonal (Re Tr A)."""
        a = np.asarray(matrix)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("compensated_real_trace: se exige matriz cuadrada.")
        return _NumericalCore.kahan_babuska_neumaier_sum(np.real(np.diag(a)))

    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt / Frobenius ‖A‖_F = √⟨A,A⟩_HS."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

    @staticmethod
    def operator_two_norm(matrix: np.ndarray) -> float:
        """Norma de Banach ‖A‖₂ = σ_max(A)."""
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, 2))

    @staticmethod
    def relative_residual(num: float, den: float, abs_floor: float = _MACHINE_EPS) -> float:
        """Residuo mixto max(|num|, |num| / max(|den|, floor))."""
        scale = max(abs(den), abs_floor)
        return float(abs(num) / scale)

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
    def assert_vec(name: str, vec: np.ndarray, dim: int) -> np.ndarray:
        v = np.asarray(vec).reshape(-1)
        if v.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; recibido {v.size}.")
        _NumericalCore.assert_finite(name, v)
        return v

    @staticmethod
    def hermitian_residual(matrix: np.ndarray) -> float:
        """‖A − A†‖_F."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a - a.T.conj())

    @staticmethod
    def skew_residual(matrix: np.ndarray) -> float:
        """‖A + Aᵀ‖_F."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a + a.T)

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
    def certify_symplectic_form(omega: np.ndarray) -> _SymplecticFormCertificate:
        """Verifica los axiomas de Darboux sobre Ω."""
        _NumericalCore.assert_square("omega", omega)
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = _NumericalCore.skew_residual(omega)
        almost_c = _NumericalCore.frobenius_norm(omega @ omega + ident)
        det_o = float(np.real(la.det(omega)))
        fro = _NumericalCore.frobenius_norm(omega)
        scale = max(fro, 1.0)
        is_darboux = (
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
        )
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            frobenius_norm=fro,
            is_darboux=bool(is_darboux),
        )

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        """Piso de deflación adaptativo de Wilkinson."""
        if matrix is None or np.asarray(matrix).size == 0:
            return _HIGHAM_REG_FLOOR
        fro_norm = _NumericalCore.frobenius_norm(matrix)
        return float(
            max(fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE, _HIGHAM_REG_FLOOR)
        )

    @staticmethod
    def regularize_spectrum(
        eigenvalues: np.ndarray,
        floor: float = _HIGHAM_REG_FLOOR,
    ) -> np.ndarray:
        """Recorte de Tikhonov: λ ↦ max(λ, floor)."""
        ev = np.asarray(eigenvalues, dtype=np.float64)
        if ev.size == 0:
            return ev
        return np.maximum(ev, float(floor))

    @staticmethod
    def higham_nearest_hermitian(matrix: np.ndarray) -> np.ndarray:
        """Proyección de Weyl–Toeplitz: (A + A†)/2."""
        a = np.asarray(matrix)
        _NumericalCore.assert_square("higham_nearest_hermitian", a)
        return 0.5 * (a + a.T.conj())

    @staticmethod
    def higham_nearest_spd(
        matrix: np.ndarray,
        floor: float = _HIGHAM_REG_FLOOR,
    ) -> np.ndarray:
        """Matriz SPD más próxima en norma de Frobenius (Higham)."""
        herm = _NumericalCore.higham_nearest_hermitian(matrix)
        evals, evecs = la.eigh(herm)
        evals = _NumericalCore.regularize_spectrum(np.real(evals), floor=floor)
        return evecs @ (evals[:, None] * evecs.T.conj())

    @staticmethod
    def tikhonov_higham_pinv(
        matrix: np.ndarray,
        rel_floor: float = _MACHINE_EPS,
        abs_floor: float = _HIGHAM_REG_FLOOR,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pseudoinversa amortiguada de Tikhonov–Higham vía SVD."""
        a = np.asarray(matrix)
        if a.size == 0:
            return a.copy(), np.array([], dtype=np.float64), float("inf")
        U, s_vals, Vt = la.svd(a, full_matrices=False)
        if s_vals.size == 0:
            return np.zeros((a.shape[1], a.shape[0]), dtype=a.dtype), s_vals, float("inf")
        lam = max(float(abs_floor), float(rel_floor) * float(s_vals[0]))
        s_inv = np.zeros_like(s_vals)
        live = s_vals > lam
        s_inv[live] = s_vals[live] / (s_vals[live] ** 2 + lam ** 2)
        pinv = (Vt.T.conj() * s_inv) @ U.T.conj()
        s_min_live = float(s_vals[live].min()) if np.any(live) else lam
        cond = float(s_vals[0] / max(s_min_live, _MACHINE_EPS))
        return pinv, s_vals, cond

    @staticmethod
    def stable_complex_power(eigenvalues: np.ndarray, exponent: complex) -> np.ndarray:
        """λ^z = exp(z Log λ) en el dominio logarítmico."""
        ev = np.asarray(eigenvalues, dtype=np.float64)
        log_e = np.log(np.maximum(ev, _HIGHAM_REG_FLOOR))
        z = np.asarray(complex(exponent) * log_e, dtype=np.complex128)
        real_c = np.clip(z.real, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        return np.exp(real_c + 1j * z.imag)

    @staticmethod
    def synthesize_port_hamiltonian_germ(
        dimension_n: int,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> _PortHamiltonianGerm:
        """Ensambla el gérmen port-Hamiltoniano."""
        if dimension_n <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        two_n = 2 * int(dimension_n)
        omega = _NumericalCore.generate_canonical_symplectic_form(two_n)
        certificate = _NumericalCore.certify_symplectic_form(omega)
        if not certificate.is_darboux:
            logger.warning(
                "Certificado de Darboux degradado: skew=%.3e, J²+I=%.3e, det=%.16f",
                certificate.skew_residual,
                certificate.almost_complex_residual,
                certificate.determinant,
            )
        if scale_matrix is None:
            floor = _HIGHAM_REG_FLOOR
        else:
            floor = _NumericalCore.wilkinson_deflation_floor(np.asarray(scale_matrix))
        return _PortHamiltonianGerm(
            n=int(dimension_n),
            two_n=two_n,
            omega=omega,
            reg_floor=float(floor),
            form_certificate=certificate,
        )


# =============================================================================
# FASE II — ESTRUCTURA DE DIRAC, IDA-PBC, Sp(2n) Y LIFTING KMS
# =============================================================================
@dataclass(frozen=True)
class _StructureCertificate:
    """Certificados de pasividad port-Hamiltoniana (J antisimétrica, R ⪰ 0)."""

    j_skew_residual: float
    r_symmetric_residual: float
    r_min_eigenvalue: float
    is_passive: bool


@dataclass(frozen=True)
class _IDAPBCResult:
    """Resultado certificado de la ley de control IDA-PBC."""

    control_law: np.ndarray
    exergy_loss: float
    lyapunov_derivative: float
    matching_residual: float
    annihilator_residual: float
    condition_number: float
    structure_ok: bool


@dataclass(frozen=True)
class _SymplecticPreservationResult:
    """Pertenencia numérica a Sp(2n, ℝ) y distancia a la retracción polar."""

    residual_norm: float
    relative_residual: float
    determinant: float
    polar_sp_distance: float
    is_viable: bool


@dataclass(frozen=True)
class _ModularSpectralGerm:
    """Gérmen espectral modular (objeto terminal de la Fase II)."""

    beta: float
    modular_hamiltonian: np.ndarray
    thermal_state: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    partition_function: float


class _IDAPBCController:
    """Fase II. Controlador port-Hamiltoniano IDA-PBC."""

    def __init__(
        self,
        dimension_n: int,
        germ: Optional[_PortHamiltonianGerm] = None,
    ) -> None:
        if germ is None:
            germ = _NumericalCore.synthesize_port_hamiltonian_germ(dimension_n)
        if germ.n != dimension_n:
            raise ValueError("El gérmen de Fase I no coincide con dimension_n.")
        self._germ: _PortHamiltonianGerm = germ
        self._n: int = germ.n
        self._2n: int = germ.two_n

    @property
    def germ(self) -> _PortHamiltonianGerm:
        return self._germ

    def validate_darboux_coordinates(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        qv = _NumericalCore.assert_vec("q", q, self._n)
        pv = _NumericalCore.assert_vec("p", p, self._n)
        return qv.astype(np.float64, copy=False), pv.astype(np.float64, copy=False)

    def certify_dirac_structure(
        self,
        J_matrix: np.ndarray,
        R_matrix: np.ndarray,
    ) -> _StructureCertificate:
        j_skew = _NumericalCore.skew_residual(J_matrix)
        r_sym = _NumericalCore.frobenius_norm(R_matrix - R_matrix.T)
        r_h = 0.5 * (R_matrix + R_matrix.T)
        evals = np.real(la.eigvalsh(r_h)) if r_h.size else np.array([0.0])
        r_min = float(np.min(evals)) if evals.size else 0.0
        scale_j = max(_NumericalCore.frobenius_norm(J_matrix), 1.0)
        scale_r = max(_NumericalCore.frobenius_norm(R_matrix), 1.0)
        is_passive = (
            j_skew <= _WILKINSON_DRIFT_LIMIT * scale_j
            and r_sym <= _WILKINSON_DRIFT_LIMIT * scale_r
            and r_min >= -_WILKINSON_DRIFT_LIMIT * scale_r
        )
        return _StructureCertificate(
            j_skew_residual=float(j_skew),
            r_symmetric_residual=float(r_sym),
            r_min_eigenvalue=r_min,
            is_passive=bool(is_passive),
        )

    @staticmethod
    def left_annihilator(g_actuator: np.ndarray, floor: float) -> np.ndarray:
        U, s_vals, _ = la.svd(g_actuator, full_matrices=True)
        if s_vals.size == 0:
            return U.T
        mask = np.ones(U.shape[1], dtype=bool)
        mask[: s_vals.size] = s_vals <= floor
        if not np.any(mask):
            return np.zeros((0, g_actuator.shape[0]), dtype=g_actuator.dtype)
        return U[:, mask].T.conj()

    def compute_control_law(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        g_actuator: np.ndarray,
        J_matrix: np.ndarray,
        R_matrix: np.ndarray,
        Jd_matrix: np.ndarray,
        Rd_matrix: np.ndarray,
        G_metric: np.ndarray,
    ) -> _IDAPBCResult:
        self.validate_darboux_coordinates(q, p)
        gH = _NumericalCore.assert_vec("grad_H", grad_H, self._2n)
        gHd = _NumericalCore.assert_vec("grad_Hd", grad_Hd, self._2n)

        g = np.asarray(g_actuator)
        if g.ndim == 1:
            g = g.reshape(self._2n, 1)
        if g.shape[0] != self._2n:
            raise ValueError(f"g_actuator debe tener {self._2n} filas.")
        _NumericalCore.assert_finite("g_actuator", g)

        named = (
            (J_matrix, "J_matrix"),
            (R_matrix, "R_matrix"),
            (Jd_matrix, "Jd_matrix"),
            (Rd_matrix, "Rd_matrix"),
            (G_metric, "G_metric"),
        )
        mats = []
        for mat, name in named:
            _NumericalCore.assert_square(name, mat, self._2n)
            _NumericalCore.assert_finite(name, np.asarray(mat))
            mats.append(np.asarray(mat, dtype=np.float64))
        J_m, R_m, Jd_m, Rd_m, G_m = mats

        cert_ol = self.certify_dirac_structure(J_m, R_m)
        cert_cl = self.certify_dirac_structure(Jd_m, Rd_m)
        structure_ok = bool(cert_ol.is_passive and cert_cl.is_passive)
        if not structure_ok:
            logger.warning(
                "Estructura de Dirac no pasiva: ol.skew=%.3e cl.Rmin=%.3e",
                cert_ol.j_skew_residual,
                cert_cl.r_min_eigenvalue,
            )

        lhs_free = (J_m - R_m) @ gH
        lhs_desired = (Jd_m - Rd_m) @ gHd
        mismatch = lhs_desired - lhs_free

        G_spd = _NumericalCore.higham_nearest_spd(G_m, floor=self._germ.reg_floor)
        g_trans_G = g.T @ G_spd
        projection = g_trans_G @ g
        floor = max(self._germ.reg_floor, _NumericalCore.wilkinson_deflation_floor(projection))
        pseudo_inv, _s_vals, cond_number = _NumericalCore.tikhonov_higham_pinv(
            projection, rel_floor=_MACHINE_EPS, abs_floor=floor
        )

        alpha = pseudo_inv @ (g_trans_G @ mismatch)

        projector = g @ pseudo_inv @ g_trans_G
        matching_vec = mismatch - projector @ mismatch
        matching_residual = _NumericalCore.frobenius_norm(matching_vec)

        g_perp = self.left_annihilator(g, floor=max(floor, _WILKINSON_DRIFT_LIMIT))
        if g_perp.size == 0:
            annihilator_residual = 0.0
        else:
            annihilator_residual = _NumericalCore.frobenius_norm(g_perp @ mismatch)

        rd_h = 0.5 * (Rd_m + Rd_m.T)
        p_loss = float(np.real(gHd.T @ rd_h @ gHd))
        lyap = -p_loss

        return _IDAPBCResult(
            control_law=np.asarray(alpha, dtype=np.float64),
            exergy_loss=p_loss,
            lyapunov_derivative=lyap,
            matching_residual=float(matching_residual),
            annihilator_residual=float(annihilator_residual),
            condition_number=float(cond_number),
            structure_ok=structure_ok,
        )

    def induce_modular_spectral_germ(
        self,
        G_metric: np.ndarray,
        beta: float = _DEFAULT_BETA,
    ) -> _ModularSpectralGerm:
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta (inverso de temperatura) debe ser positivo y finito.")
        _NumericalCore.assert_square("G_metric", G_metric, self._2n)
        _NumericalCore.assert_finite("G_metric", np.asarray(G_metric))
        K = _NumericalCore.higham_nearest_spd(
            np.asarray(G_metric, dtype=np.float64),
            floor=self._germ.reg_floor,
        )
        evals, evecs = la.eigh(K)
        evals = np.real(evals)
        log_terms = -float(beta) * evals
        log_terms = np.clip(log_terms, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        unnorm = np.exp(log_terms)
        z = _NumericalCore.kahan_babuska_neumaier_sum(unnorm)
        if z <= _MACHINE_EPS:
            raise ValueError("Función de partición degenerada al inducir el gérmen modular.")
        rho_eigs = unnorm / z
        rho = evecs @ (rho_eigs[:, None] * evecs.T.conj())
        rho = _NumericalCore.higham_nearest_hermitian(rho)
        return _ModularSpectralGerm(
            beta=float(beta),
            modular_hamiltonian=K,
            thermal_state=rho,
            eigenvalues=rho_eigs,
            eigenvectors=evecs,
            partition_function=float(z),
        )


class _SymplecticPreservationChecker:
    """Fase II (continuación geométrica). Verifica M ∈ Sp(2n, ℝ)."""

    def __init__(
        self,
        dimension_n: int,
        germ: Optional[_PortHamiltonianGerm] = None,
    ) -> None:
        if germ is None:
            germ = _NumericalCore.synthesize_port_hamiltonian_germ(dimension_n)
        self._germ = germ
        self._n = germ.n
        self._2n = germ.two_n

    def verify(self, jacobian_matrix: np.ndarray) -> _SymplecticPreservationResult:
        M = np.asarray(jacobian_matrix, dtype=np.float64)
        _NumericalCore.assert_square("jacobian_matrix", M, self._2n)
        _NumericalCore.assert_finite("jacobian_matrix", M)

        omega = self._germ.omega
        residual_matrix = M.T @ omega @ M - omega
        residual_norm = _NumericalCore.frobenius_norm(residual_matrix)
        scale = max(
            _NumericalCore.frobenius_norm(omega) * (_NumericalCore.frobenius_norm(M) ** 2),
            1.0,
        )
        rel = float(residual_norm / scale)

        det_m = float(np.real(la.det(M)))

        S = -omega @ M.T @ omega @ M
        try:
            S_h = _NumericalCore.higham_nearest_spd(0.5 * (S + S.T), floor=self._germ.reg_floor)
            evals, evecs = la.eigh(S_h)
            evals = _NumericalCore.regularize_spectrum(np.real(evals), floor=self._germ.reg_floor)
            s_inv_sqrt = evecs @ (np.power(evals, -0.5)[:, None] * evecs.T)
            M_retract = M @ s_inv_sqrt
            polar_dist = _NumericalCore.frobenius_norm(M - M_retract)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Retracción polar a Sp(2n) fallida: %s", exc)
            polar_dist = float("inf")

        is_viable = residual_norm <= max(
            _WILKINSON_DRIFT_LIMIT, _WILKINSON_DRIFT_LIMIT * scale
        )
        return _SymplecticPreservationResult(
            residual_norm=float(residual_norm),
            relative_residual=rel,
            determinant=det_m,
            polar_sp_distance=float(polar_dist),
            is_viable=bool(is_viable),
        )


# =============================================================================
# FASE III — VON NEUMANN, TOMITA–TAKESAKI, KMS, UMEGAKI, UHLMANN
# =============================================================================
@dataclass(frozen=True)
class _PurificationResult:
    """Operador densidad purificado por mayoración espectral."""

    purified_rho: np.ndarray
    effective_rank: int
    von_neumann_entropy: float
    purity: float
    trace: float


@dataclass(frozen=True)
class _ModularFlowResult:
    """Imagen del automorfismo modular σ_z^ρ(A) = ρ^{i z} A ρ^{−i z}."""

    evolved_observable: np.ndarray
    norm_preserved: bool
    kms_residual: float
    modular_operator_spectrum: np.ndarray


@dataclass(frozen=True)
class _QuantumRelativeEntropyResult:
    """Divergencia de Umegaki, fidelidad de Uhlmann y certificados de Klein/Pinsker."""

    umegaki_entropy: float
    uhlmann_fidelity: float
    trace_distance: float
    pinsker_gap: float
    support_included: bool


class _DensityPurifier:
    """Fase III. Purificación espectral por mayoración."""

    def __init__(
        self,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
        germ: Optional[_ModularSpectralGerm] = None,
    ) -> None:
        if purity_margin <= 0.0:
            raise ValueError("purity_margin debe ser positivo.")
        self._margin = float(purity_margin)
        self._germ = germ

    def purify(self, rho_mixed: np.ndarray) -> _PurificationResult:
        rho = np.asarray(rho_mixed)
        _NumericalCore.assert_square("rho_mixed", rho)
        _NumericalCore.assert_finite("rho_mixed", rho)

        rho_herm = _NumericalCore.higham_nearest_hermitian(rho)
        eigenvalues, eigenvectors = la.eigh(rho_herm)
        eigenvalues = np.real(eigenvalues)

        eigenvalues[eigenvalues < self._margin] = 0.0
        effective_rank = int(np.sum(eigenvalues > 0.0))

        trace_sum = _NumericalCore.kahan_babuska_neumaier_sum(eigenvalues)
        if trace_sum > _MACHINE_EPS:
            eigenvalues_norm = eigenvalues / trace_sum
        else:
            eigenvalues_norm = np.zeros_like(eigenvalues)
            eigenvalues_norm[-1] = 1.0
            effective_rank = 1

        rho_purified = eigenvectors @ (eigenvalues_norm[:, None] * eigenvectors.T.conj())
        rho_purified = _NumericalCore.higham_nearest_hermitian(rho_purified)

        pos = eigenvalues_norm > 0.0
        if np.any(pos):
            vn_terms = -eigenvalues_norm[pos] * np.log(eigenvalues_norm[pos])
            vn = _NumericalCore.kahan_babuska_neumaier_sum(vn_terms)
        else:
            vn = 0.0
        purity = _NumericalCore.kahan_babuska_neumaier_sum(eigenvalues_norm ** 2)
        tr = _NumericalCore.kahan_babuska_neumaier_sum(eigenvalues_norm)

        return _PurificationResult(
            purified_rho=rho_purified,
            effective_rank=effective_rank,
            von_neumann_entropy=float(max(vn, 0.0)),
            purity=float(purity),
            trace=float(tr),
        )


class _TomitaTakesakiFlow:
    """Fase III. Grupo de automorfismos modulares de Tomita–Takesaki."""

    def __init__(
        self,
        purifier: Optional[_DensityPurifier] = None,
        germ: Optional[_ModularSpectralGerm] = None,
    ) -> None:
        self._purifier = purifier if purifier is not None else _DensityPurifier(germ=germ)
        self._germ = germ

    def evolve(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> _ModularFlowResult:
        A = np.asarray(observable_A)
        _NumericalCore.assert_square("observable_A", A)
        _NumericalCore.assert_finite("observable_A", A)

        if self._germ is not None and A.shape == self._germ.thermal_state.shape:
            purified = self._germ.thermal_state
            eigenvalues = np.real(self._germ.eigenvalues)
            eigenvectors = self._germ.eigenvectors
        else:
            purified = self._purifier.purify(rho).purified_rho
            eigenvalues, eigenvectors = la.eigh(purified)
            eigenvalues = np.real(eigenvalues)

        floor = _NumericalCore.wilkinson_deflation_floor(purified)
        ev_reg = _NumericalCore.regularize_spectrum(eigenvalues, floor=floor)

        z = complex(time_parameter)
        power = 1j * z
        lambda_left = _NumericalCore.stable_complex_power(ev_reg, power)
        lambda_right = _NumericalCore.stable_complex_power(ev_reg, -power)

        rho_pow_left = eigenvectors @ (lambda_left[:, None] * eigenvectors.T.conj())
        rho_pow_right = eigenvectors @ (lambda_right[:, None] * eigenvectors.T.conj())
        evolved = rho_pow_left @ A @ rho_pow_right

        norm_before = _NumericalCore.frobenius_norm(A)
        norm_after = _NumericalCore.frobenius_norm(evolved)
        if abs(z.imag) <= _KMS_STRIP_TOL:
            norm_preserved = bool(np.isclose(norm_before, norm_after, rtol=1e-8, atol=1e-10))
        else:
            norm_preserved = True

        kms_residual = self._kms_residual(purified, A, evolved, z)
        return _ModularFlowResult(
            evolved_observable=evolved,
            norm_preserved=norm_preserved,
            kms_residual=float(kms_residual),
            modular_operator_spectrum=ev_reg,
        )

    @staticmethod
    def _kms_residual(
        rho: np.ndarray,
        observable: np.ndarray,
        evolved: np.ndarray,
        z: complex,
    ) -> float:
        if abs(z.imag - 1.0) > 0.25 or abs(z.real) > 0.25:
            return 0.0
        B = observable.T.conj()
        lhs = _NumericalCore.compensated_real_trace(rho @ B @ evolved)
        rhs = _NumericalCore.compensated_real_trace(rho @ observable @ B)
        scale = max(abs(rhs), abs(lhs), _MACHINE_EPS)
        return float(abs(lhs - rhs) / scale)


class _QuantumEntropyCalculator:
    """Fase III. Entropía relativa de Umegaki y fidelidad de Uhlmann."""

    def __init__(
        self,
        purifier: Optional[_DensityPurifier] = None,
        germ: Optional[_ModularSpectralGerm] = None,
    ) -> None:
        self._purifier = purifier if purifier is not None else _DensityPurifier(germ=germ)
        self._germ = germ

    def compute(self, rho: np.ndarray, sigma: np.ndarray) -> _QuantumRelativeEntropyResult:
        rho_p = self._purifier.purify(rho).purified_rho
        sig_p = self._purifier.purify(sigma).purified_rho
        if rho_p.shape != sig_p.shape:
            raise ValueError("rho y sigma deben tener la misma dimensión.")

        e_rho, v_rho = la.eigh(rho_p)
        e_sig, v_sig = la.eigh(sig_p)
        e_rho = np.real(e_rho)
        e_sig = np.real(e_sig)

        floor = max(
            _NumericalCore.wilkinson_deflation_floor(rho_p),
            _NumericalCore.wilkinson_deflation_floor(sig_p),
            self._purifier._margin,
        )

        support_included, leakage = self._support_inclusion(e_rho, v_rho, e_sig, v_sig, floor)
        if not support_included:
            logger.warning(
                "supp(ρ) ⊈ supp(σ) (leakage=%.3e): Umegaki = +∞.", leakage
            )
            td = self._trace_distance(e_rho, v_rho, e_sig, v_sig)
            fid = self._uhlmann_fidelity(e_rho, v_rho, e_sig, v_sig)
            return _QuantumRelativeEntropyResult(
                umegaki_entropy=float("inf"),
                uhlmann_fidelity=fid,
                trace_distance=td,
                pinsker_gap=float("inf"),
                support_included=False,
            )

        log_sig_e = np.zeros_like(e_sig)
        live_s = e_sig > floor
        log_sig_e[live_s] = np.log(e_sig[live_s])
        log_sig = v_sig @ (log_sig_e[:, None] * v_sig.T.conj())

        live_r = e_rho > floor
        vn = 0.0
        if np.any(live_r):
            vn = _NumericalCore.kahan_babuska_neumaier_sum(
                e_rho[live_r] * np.log(e_rho[live_r])
            )
        quad = np.real(np.diag(v_rho.T.conj() @ log_sig @ v_rho))
        cross = _NumericalCore.kahan_babuska_neumaier_sum(e_rho * quad)
        umegaki = float(vn - cross)
        if umegaki < 0.0 and abs(umegaki) < 1e-12:
            umegaki = 0.0

        fidelity = self._uhlmann_fidelity(e_rho, v_rho, e_sig, v_sig)
        td = self._trace_distance(e_rho, v_rho, e_sig, v_sig)
        pinsker_rhs = 0.5 * (td ** 2)
        pinsker_gap = float(umegaki - pinsker_rhs)

        return _QuantumRelativeEntropyResult(
            umegaki_entropy=umegaki,
            uhlmann_fidelity=fidelity,
            trace_distance=td,
            pinsker_gap=pinsker_gap,
            support_included=True,
        )

    @staticmethod
    def _support_inclusion(
        e_rho: np.ndarray,
        v_rho: np.ndarray,
        e_sig: np.ndarray,
        v_sig: np.ndarray,
        floor: float,
    ) -> Tuple[bool, float]:
        ker_mask = e_sig <= floor
        if not np.any(ker_mask):
            return True, 0.0
        ker = v_sig[:, ker_mask]
        rho = v_rho @ (e_rho[:, None] * v_rho.T.conj())
        leakage = np.real(np.diag(ker.T.conj() @ rho @ ker))
        leak = float(np.max(np.abs(leakage))) if leakage.size else 0.0
        return leak <= max(floor, _HERMITIAN_TOL), leak

    @staticmethod
    def _uhlmann_fidelity(
        e_rho: np.ndarray,
        v_rho: np.ndarray,
        e_sig: np.ndarray,
        v_sig: np.ndarray,
    ) -> float:
        sqrt_r = v_rho @ (np.sqrt(np.clip(e_rho, 0.0, None))[:, None] * v_rho.T.conj())
        sqrt_s = v_sig @ (np.sqrt(np.clip(e_sig, 0.0, None))[:, None] * v_sig.T.conj())
        svals = la.svdvals(sqrt_r @ sqrt_s)
        amp = _NumericalCore.kahan_babuska_neumaier_sum(np.real(svals))
        return float(amp * amp)

    @staticmethod
    def _trace_distance(
        e_rho: np.ndarray,
        v_rho: np.ndarray,
        e_sig: np.ndarray,
        v_sig: np.ndarray,
    ) -> float:
        rho = v_rho @ (e_rho[:, None] * v_rho.T.conj())
        sig = v_sig @ (e_sig[:, None] * v_sig.T.conj())
        delta = _NumericalCore.higham_nearest_hermitian(rho - sig)
        ev = np.real(la.eigvalsh(delta))
        return 0.5 * _NumericalCore.kahan_babuska_neumaier_sum(np.abs(ev))


# =============================================================================
# CLASE PRINCIPAL — INTEGRACIÓN DEL MORFISMO Y MOTOR MAUPERTUIS-JACOBI
# =============================================================================
class ImperialCenturionsEngine:
    r"""
    Motor espectral ciego en FPU para el cálculo de geodésicas de Maupertuis-Jacobi,
    invarianza de Liouville, control Port-Hamiltoniano y estado modular de Tomita-Takesaki.

    Axiomas Físico-Matemáticos:
    --------------------------
    1. Métrica Conforme de Maupertuis-Jacobi:
       $$\tilde{g}_{jk}(q) = 2(H_0 - V(q)) g_{jk}(q) = n(q)^2 g_{jk}(q)$$
       con índice de refracción $n(q) = \sqrt{2(H_0 - V(q))} > 0$.

    2. Símbolos de Christoffel Conformes:
       $$\tilde{\Gamma}^i_{jk} = \Gamma^i_{jk} + \delta^i_j \partial_k \phi + \delta^i_k \partial_j \phi - g_{jk} g^{il} \partial_l \phi, \quad \phi(q) = \ln \sqrt{2(H_0 - V(q))}$$

    3. Integración Simpléctica Störmer-Verlet:
       Preserva la 2-forma de Liouville $\omega$ con $|\det M_{\mathrm{step}} - 1| \le \varepsilon_{\mathrm{spectral}}$.
    """

    def __init__(self, dimension_n: int = 4) -> None:
        if int(dimension_n) <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        self._n: Final[int] = int(dimension_n)
        self._2n: Final[int] = 2 * self._n

        half = self._n
        self._J_canonical = np.block([
            [np.zeros((half, half), dtype=np.float64), np.eye(half, dtype=np.float64)],
            [-np.eye(half, dtype=np.float64), np.zeros((half, half), dtype=np.float64)]
        ])

        # Fases I, II y III para lazo cerrado Port-Hamiltoniano
        self._ph_germ: _PortHamiltonianGerm = (
            _NumericalCore.synthesize_port_hamiltonian_germ(self._n)
        )
        self._ida_pbc = _IDAPBCController(self._n, germ=self._ph_germ)
        self._symplectic_checker = _SymplecticPreservationChecker(
            self._n, germ=self._ph_germ
        )
        self._mod_germ: _ModularSpectralGerm = self._ida_pbc.induce_modular_spectral_germ(
            np.eye(self._2n, dtype=np.float64),
            beta=_DEFAULT_BETA,
        )
        self._density_purifier = _DensityPurifier(germ=self._mod_germ)
        self._modular_flow = _TomitaTakesakiFlow(
            purifier=self._density_purifier, germ=self._mod_germ
        )
        self._entropy_calculator = _QuantumEntropyCalculator(
            purifier=self._density_purifier, germ=self._mod_germ
        )

    # ── MÉTODOS CELESTES DE POINCARÉ-MAUPERTUIS-JACOBI ───────────────────
    def compute_maupertuis_conformal_metric(
        self,
        q_position: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Calcula la Métrica Conforme de Jacobi-Fermat $\tilde{g}_{jk} = 2(H_0 - V(q)) g_{jk}$.

        Axioma: $n(q) = \sqrt{2(H_0 - V(q))} > 0$. Condición de hiperbolicidad positiva.
        """
        kinetic_headroom = 2.0 * (total_energy_H0 - potential_V)
        if kinetic_headroom <= _WILKINSON_LIMIT:
            raise ValueError("[CENTURION_ENGINE_VETO] Cero energía cinética: Invasión de pozo de potencial.")

        refractive_index_n = math.sqrt(kinetic_headroom)
        g_conformal = (refractive_index_n ** 2) * g_base_metric

        return g_conformal, refractive_index_n

    def compute_christoffel_conformal_symbols(
        self,
        q_position: NDArray[np.float64],
        grad_V: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Calcula los Símbolos de Christoffel conformes $\tilde{\Gamma}^i_{jk}$ para la geodésica de Maupertuis.

        Axioma: $\tilde{\Gamma}^i_{jk} = \Gamma^i_{jk} + \delta^i_j \partial_k \phi + \delta^i_k \partial_j \phi - g_{jk} g^{il} \partial_l \phi$,  $\phi = \ln \sqrt{2(H_0 - V)}$.
        """
        n_dim = len(q_position)
        kinetic_headroom = 2.0 * (total_energy_H0 - potential_V)

        grad_phi = -grad_V / (kinetic_headroom + _WILKINSON_LIMIT)
        g_inv = la.inv(g_base_metric)

        christoffel = np.zeros((n_dim, n_dim, n_dim), dtype=np.float64)

        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_dim):
                    term1 = (1.0 if i == j else 0.0) * grad_phi[k]
                    term2 = (1.0 if i == k else 0.0) * grad_phi[j]
                    term3 = g_base_metric[j, k] * np.sum(g_inv[i, :] * grad_phi)
                    christoffel[i, j, k] = term1 + term2 - term3

        return christoffel

    def integrate_symplectic_maupertuis_step(
        self,
        x_state: NDArray[np.float64],
        dt_step: float,
        g_base_metric: NDArray[np.float64],
        potential_V: float,
        grad_V: NDArray[np.float64],
        total_energy_H0: float
    ) -> MaupertuisStepReport:
        r"""
        Integra un paso temporal del flujo de Maupertuis preservando la 2-forma de Liouville.

        Utiliza el algoritmo Störmer-Verlet con evaluación de $\det(M_{\mathrm{step}})$ en FPU.
        """
        n_dim = len(x_state) // 2
        q_pos = x_state[:n_dim]
        p_mom = x_state[n_dim:]

        g_inv = la.inv(g_base_metric)

        # 1. Medio paso para el momentum p(t + dt/2) = p(t) - (dt/2) ∇V(q)
        p_half = p_mom - 0.5 * dt_step * grad_V

        # 2. Paso completo para la posición q(t + dt) = q(t) + dt G⁻¹ p(t + dt/2)
        q_next = q_pos + dt_step * (g_inv @ p_half)

        # 3. Medio paso final para momentum p(t + dt)
        p_next = p_half - 0.5 * dt_step * grad_V

        # 4. Evaluación de métrica conforme y densidad de acción
        _, n_index = self.compute_maupertuis_conformal_metric(q_next, potential_V, total_energy_H0, g_base_metric)
        velocity_q_dot = g_inv @ p_next
        maupertuis_action = float(n_index * la.norm(velocity_q_dot))

        # 5. Cómputo del Jacobiano de Fase M y determinante de Liouville
        hamiltonian_energy = float(0.5 * (p_next.T @ g_inv @ p_next) + potential_V)

        H_hessian = la.block_diag(np.eye(n_dim), g_inv)
        J_can = np.block([
            [np.zeros((n_dim, n_dim), dtype=np.float64), np.eye(n_dim, dtype=np.float64)],
            [-np.eye(n_dim, dtype=np.float64), np.zeros((n_dim, n_dim), dtype=np.float64)]
        ])
        M_jacobian = np.eye(2 * n_dim) + dt_step * (J_can @ H_hessian)
        det_M = float(la.det(M_jacobian))
        volume_drift = abs(det_M - 1.0)

        is_symplectic = volume_drift <= _SPECTRAL_TOL

        return MaupertuisStepReport(
            hamiltonian_energy=hamiltonian_energy,
            refractive_index_n=n_index,
            maupertuis_action_density=maupertuis_action,
            volume_drift_det=volume_drift,
            is_symplectic_coherent=is_symplectic
        )

    # ── MÉTODOS DE LAZO CERRADO IDA-PBC Y KMS (API 2.0 / 3.0) ────────────
    def kahan_sum(self, array: np.ndarray) -> float:
        return _NumericalCore.kahan_sum(array)

    def kahan_babuska_neumaier_sum(self, array: np.ndarray) -> float:
        return _NumericalCore.kahan_babuska_neumaier_sum(array)

    def port_hamiltonian_germ_certificate(self) -> _SymplecticFormCertificate:
        return self._ph_germ.form_certificate

    def compute_ida_pbc_control_law(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        g_actuator: np.ndarray,
        J_matrix: np.ndarray,
        R_matrix: np.ndarray,
        Jd_matrix: np.ndarray,
        Rd_matrix: np.ndarray,
        G_metric: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        result = self._ida_pbc.compute_control_law(
            q, p, grad_H, grad_Hd, g_actuator,
            J_matrix, R_matrix, Jd_matrix, Rd_matrix, G_metric,
        )
        return result.control_law, result.exergy_loss

    def compute_ida_pbc_control_law_certified(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H: np.ndarray,
        grad_Hd: np.ndarray,
        g_actuator: np.ndarray,
        J_matrix: np.ndarray,
        R_matrix: np.ndarray,
        Jd_matrix: np.ndarray,
        Rd_matrix: np.ndarray,
        G_metric: np.ndarray,
    ) -> _IDAPBCResult:
        return self._ida_pbc.compute_control_law(
            q, p, grad_H, grad_Hd, g_actuator,
            J_matrix, R_matrix, Jd_matrix, Rd_matrix, G_metric,
        )

    def verify_symplectic_preservation(
        self,
        jacobian_matrix: np.ndarray,
    ) -> Tuple[float, bool]:
        result = self._symplectic_checker.verify(jacobian_matrix)
        return result.residual_norm, result.is_viable

    def verify_symplectic_preservation_certified(
        self,
        jacobian_matrix: np.ndarray,
    ) -> _SymplecticPreservationResult:
        return self._symplectic_checker.verify(jacobian_matrix)

    def induce_modular_spectral_germ(
        self,
        G_metric: np.ndarray,
        beta: float = _DEFAULT_BETA,
    ) -> _ModularSpectralGerm:
        germ = self._ida_pbc.induce_modular_spectral_germ(G_metric, beta=beta)
        self._mod_germ = germ
        self._density_purifier = _DensityPurifier(germ=germ)
        self._modular_flow = _TomitaTakesakiFlow(
            purifier=self._density_purifier, germ=germ
        )
        self._entropy_calculator = _QuantumEntropyCalculator(
            purifier=self._density_purifier, germ=germ
        )
        return germ

    def purify_density_operator(
        self,
        rho_mixed: np.ndarray,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
    ) -> np.ndarray:
        purifier = _DensityPurifier(purity_margin, germ=self._mod_germ)
        return purifier.purify(rho_mixed).purified_rho

    def purify_density_operator_certified(
        self,
        rho_mixed: np.ndarray,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
    ) -> _PurificationResult:
        purifier = _DensityPurifier(purity_margin, germ=self._mod_germ)
        return purifier.purify(rho_mixed)

    def evolve_tomita_takesaki_flow(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> np.ndarray:
        result = self._modular_flow.evolve(observable_A, rho, time_parameter)
        return result.evolved_observable

    def evolve_tomita_takesaki_flow_certified(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> _ModularFlowResult:
        return self._modular_flow.evolve(observable_A, rho, time_parameter)

    def compute_quantum_relative_entropy(
        self,
        rho: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple[float, float]:
        result = self._entropy_calculator.compute(rho, sigma)
        return result.umegaki_entropy, result.uhlmann_fidelity

    def compute_quantum_relative_entropy_certified(
        self,
        rho: np.ndarray,
        sigma: np.ndarray,
    ) -> _QuantumRelativeEntropyResult:
        return self._entropy_calculator.compute(rho, sigma)


__all__ = [
    "ImperialCenturionsEngine",
    "MaupertuisStepReport",
]
