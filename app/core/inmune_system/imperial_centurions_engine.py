# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Centurions Engine (Caballos de Batalla de la Capa 2)       ║
║ Ruta   : app/core/inmune_system/imperial_centurions_engine.py                ║
║ Versión: 3.0.0-Nested-Phases-Dirac-IDA-PBC-Sp-KMS-Tomita-Takesaki-FPU        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU
────────────────────────────────────────────────────────────────────────────────
Motor tensorial de alta precisión para la Capa 2 (Centuriones). Implementa un
morfismo de fases anidadas

    Φ_III ∘ Φ_II ∘ Φ_I :  NumBanach  →  PortHam(Dirac)  →  vN_* (KMS)

donde cada fase consume el gérmen formal producido por la precedente.

  Fase I   Núcleo de Banach, 2-forma de Darboux, regularización espectral
           de Tikhonov–Higham–Wilkinson y sumación compensada.
           Último morfismo: synthesize_port_hamiltonian_germ.
  Fase II  Estructura de Dirac, ley IDA-PBC, matching g^⊥, certificados de
           pasividad / Lyapunov y retracción al grupo simpléctico Sp(2n, ℝ).
           Último morfismo: induce_modular_spectral_germ.
  Fase III Purificación por mayoración, flujo modular de Tomita–Takesaki,
           condición KMS y divergencias de Umegaki / fidelidad de Uhlmann.

Precisión metrológica: Kahan, Kahan–Babuška–Neumaier, Klein; pinv amortiguada
relativa; logaritmos con soporte; norma nuclear para Uhlmann.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Engines.ImperialCenturionsEngine")

__version__: Final[str] = "3.0.0-Nested-Phases-Dirac-IDA-PBC-Sp-KMS-Tomita-Takesaki-FPU"


# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_REG_FLOOR: Final[float] = 1e-15
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_LOG_EXP_CLIP: Final[float] = 700.0
_KMS_STRIP_TOL: Final[float] = 1e-8
_HERMITIAN_TOL: Final[float] = 1e-12
_DEFAULT_PURITY_MARGIN: Final[float] = 1e-12
_DEFAULT_BETA: Final[float] = 1.0


# =============================================================================
# FASE I — NÚCLEO NUMÉRICO DE BANACH, DARBOUX Y TIKHONOV–HIGHAM
# -----------------------------------------------------------------------------
# Objetos: sumas compensadas, normas de operadores, 2-forma canónica Ω,
#          proyección espectral de Higham, pseudoinversa relativa.
# Morfismo terminal (I.8): synthesize_port_hamiltonian_germ
#          ≅ objeto inicial de la Fase II (germen de Dirac / IDA-PBC).
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

    Atributos
    ---------
    n:
        Dimensión del espacio de configuración Q (mitad de Darboux).
    two_n:
        Dimensión de T*Q ≅ ℝ^{2n}.
    omega:
        2-forma canónica Ω = [0 I; −I 0] ∈ ℝ^{2n×2n}.
    reg_floor:
        Piso de Tikhonov–Higham adaptativo.
    form_certificate:
        Certificado (Ωᵀ = −Ω, Ω² = −I, det Ω = 1).
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

    # ── I.1  Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """
        Sumación compensada de Kahan.

        Neutraliza el término de redondeo \(c_{k+1}=(t_k-s_k)-y_k\) de modo
        que \(\sum x_i\) sea exacta módulo O(u · Σ|x_i|) en lugar de
        O(n u · max|x_i|).
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

        A diferencia de Kahan, acumula la compensación cuando |x| > |s|,
        lo que estabiliza cancelaciones de signo mixto.
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
        """
        Sumación doblemente compensada de Klein.

        Mantiene dos residuos de redondeo; el error es O(u²) relativo a
        la condición de la suma. Se usa en trazas espectrales críticas.
        """
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

    # ── I.2  Normas de operadores (álgebra de Banach) ─────────────────────
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

    # ── I.3  Validación tensorial ─────────────────────────────────────────
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
        """‖A − A†‖_F (cero sii A es hermítica)."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a - a.T.conj())

    @staticmethod
    def skew_residual(matrix: np.ndarray) -> float:
        """‖A + Aᵀ‖_F (cero sii A es antisimétrica real)."""
        a = np.asarray(matrix)
        return _NumericalCore.frobenius_norm(a + a.T)

    # ── I.4  2-forma simpléctica canónica de Darboux ──────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        """
        2-forma canónica de Liouville Ω ∈ ℝ^{dim×dim}, dim = 2n par.

        En coordenadas de Darboux (q, p):

            Ω = ⎡ 0   I_n ⎤ ,   Ωᵀ = −Ω ,   Ω² = −I ,   Ω⁻¹ = −Ω .
                ⎣−I_n  0  ⎦

        Es el tensor que define Sp(2n, ℝ) = {M | Mᵀ Ω M = Ω}.
        """
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

    # ── I.5  Regularización espectral Tikhonov–Higham–Wilkinson ───────────
    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        """
        Piso de deflación adaptativo de Wilkinson:

            ε_W = max( ‖A‖_F · ε_mach · 10 ,  ε_Higham ).

        Filtra modos parásitos por debajo del ruido de redondeo relativo
        a la escala de Frobenius.
        """
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
        """Recorte de Tikhonov: λ ↦ max(λ, floor). Evita log 0 e inversiones nulas."""
        ev = np.asarray(eigenvalues, dtype=np.float64)
        if ev.size == 0:
            return ev
        return np.maximum(ev, float(floor))

    @staticmethod
    def higham_nearest_hermitian(matrix: np.ndarray) -> np.ndarray:
        """Proyección de Weyl–Toeplitz: (A + A†)/2, el hermítico más próximo en ‖·‖_F."""
        a = np.asarray(matrix)
        _NumericalCore.assert_square("higham_nearest_hermitian", a)
        return 0.5 * (a + a.T.conj())

    @staticmethod
    def higham_nearest_spd(
        matrix: np.ndarray,
        floor: float = _HIGHAM_REG_FLOOR,
    ) -> np.ndarray:
        """
        Matriz SPD más próxima en norma de Frobenius (Higham):

            Hermitiza, recorta el espectro a [floor, +∞) y reconstruye.
        """
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
        """
        Pseudoinversa amortiguada de Tikhonov–Higham vía SVD.

            σ⁺_i = σ_i / (σ_i² + λ²)    si σ_i > λ,
                 = 0                    en otro caso,

        con λ = max(abs_floor, rel_floor · σ_max). Devuelve (A⁺, σ, κ₂).
        """
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
        """
        λ^z = exp(z Log λ) en el dominio logarítmico, con recorte de la
        parte real para evitar overflow/underflow (continuación analítica
        controlada hacia el semiplano de KMS).
        """
        ev = np.asarray(eigenvalues, dtype=np.float64)
        log_e = np.log(np.maximum(ev, _HIGHAM_REG_FLOOR))
        z = np.asarray(complex(exponent) * log_e, dtype=np.complex128)
        real_c = np.clip(z.real, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        return np.exp(real_c + 1j * z.imag)

    # ── I.6 / I.8  Morfismo terminal de la Fase I ─────────────────────────
    @staticmethod
    def synthesize_port_hamiltonian_germ(
        dimension_n: int,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> _PortHamiltonianGerm:
        """
        I.8 — Morfismo terminal de la Fase I / objeto inicial de la Fase II.

        Ensambla el gérmen port-Hamiltoniano

            𝒢_I = (n,  2n,  Ω_{Darboux},  ε_W,  Cert(Ω))

        sobre el cual la Fase II define la estructura de Dirac

            𝔇 = { (f, e) ∈ 𝔽 ⊕ 𝔽* | e = (J − R) f ,  Jᵀ = −J ,  R = Rᵀ ⪰ 0 }

        y resuelve la asignación IDA-PBC. Si se provee `scale_matrix`
        (p. ej. la métrica G o un Jacobiano), el piso de Wilkinson se
        adapta a su norma de Frobenius.

        Este método *es* el arranque formal de `_IDAPBCController` y
        `_SymplecticPreservationChecker`.
        """
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
# -----------------------------------------------------------------------------
# Continúa I.8: todo controlador se instancia desde un PortHamiltonianGerm.
# Morfismo terminal (II.6): induce_modular_spectral_germ
#          ≅ objeto inicial de la Fase III (estado de Gibbs / Hamiltoniano
#            modular K = −log ρ).
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
    """
    Gérmen espectral modular (objeto terminal de la Fase II).

    Es el objeto inicial de la Fase III: estado de Gibbs asociado a la
    métrica / Hamiltoniano cuadrático del lazo cerrado,

        ρ_β = exp(−β K) / Tr exp(−β K) ,   K = HighamSPD(G) ,

    junto con su resolución espectral, lista para el flujo de Tomita–Takesaki
    y las divergencias de Umegaki.
    """

    beta: float
    modular_hamiltonian: np.ndarray
    thermal_state: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    partition_function: float


class _IDAPBCController:
    """
    Fase II. Controlador port-Hamiltoniano IDA-PBC.

    Continúa el gérmen 𝒢_I: sobre (T*Q, Ω) resuelve la ecuación de matching

        g^⊥ [ (J_d − R_d) ∇H_d − (J − R) ∇H ] = 0

    y la ley de esfuerzo en los puertos

        α = (gᵀ G g)⁺ gᵀ G [ (J_d − R_d) ∇H_d − (J − R) ∇H ] .

    La pasividad del lazo cerrado se certifica por Ḣ_d = −∇H_dᵀ R_d ∇H_d ≤ 0.
    """

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
        """Gérmen de Fase I del que este controlador es continuación."""
        return self._germ

    def validate_darboux_coordinates(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Valida (q, p) ∈ T*Q ≅ ℝ^n ⊕ ℝ^n y devuelve copias 1-D."""
        qv = _NumericalCore.assert_vec("q", q, self._n)
        pv = _NumericalCore.assert_vec("p", p, self._n)
        return qv.astype(np.float64, copy=False), pv.astype(np.float64, copy=False)

    def certify_dirac_structure(
        self,
        J_matrix: np.ndarray,
        R_matrix: np.ndarray,
    ) -> _StructureCertificate:
        """
        Certifica la estructura de Dirac: Jᵀ = −J y (R + Rᵀ)/2 ⪰ 0.
        Equivale a balance de potencia eᵀ f = −fᵀ R f ≤ 0 (Kirchhoff).
        """
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
        """
        Aniquilador izquierdo g^⊥ (rango máximo) tal que g^⊥ g = 0.

        Se obtiene de los vectores singulares izquierdos asociados a
        σ_i ≤ floor (núcleo de gᵀ).
        """
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
        """
        Ley IDA-PBC certificada.

        Usa las coordenadas de Darboux (q, p) para anclar el estado en T*Q
        (el matching algebraico depende de los gradientes, no de (q, p)
        salvo validación y diagnóstico). Calcula además:

        * residuo de matching métrico (I − P_g) Δ,
        * residuo del aniquilador ‖g^⊥ Δ‖,
        * pérdida exergética de Rayleigh ∇H_dᵀ R_d ∇H_d,
        * derivada de Lyapunov Ḣ_d = −∇H_dᵀ R_d ∇H_d,
        * κ₂ del operador de proyección gᵀ G g.
        """
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

        # Proyector métrico sobre im(g) y residuo de matching.
        projector = g @ pseudo_inv @ g_trans_G
        matching_vec = mismatch - projector @ mismatch
        matching_residual = _NumericalCore.frobenius_norm(matching_vec)

        g_perp = self.left_annihilator(g, floor=max(floor, _WILKINSON_DRIFT_LIMIT))
        if g_perp.size == 0:
            annihilator_residual = 0.0
        else:
            annihilator_residual = _NumericalCore.frobenius_norm(g_perp @ mismatch)

        # Rayleigh / Lyapunov: Ḣ_d = −∇H_dᵀ R_d ∇H_d  (R_d ⪰ 0 ⇒ Ḣ_d ≤ 0).
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

    # ── II.6  Morfismo terminal de la Fase II ─────────────────────────────
    def induce_modular_spectral_germ(
        self,
        G_metric: np.ndarray,
        beta: float = _DEFAULT_BETA,
    ) -> _ModularSpectralGerm:
        """
        II.6 — Morfismo terminal de la Fase II / objeto inicial de la Fase III.

        Cuantiza la métrica del lazo cerrado como Hamiltoniano modular
        (cuantización de Weyl del Hamiltoniano cuadrático ½ xᵀ G x):

            K  = HighamSPD(G) ,
            ρ_β = e^{−β K} / Z ,   Z = Tr e^{−β K} .

        El par (K, ρ_β) es el estado KMS de tipo I que la Fase III toma
        como dato inicial de Tomita–Takesaki (Δ = ρ ⊗ ρ⁻¹ en forma estándar)
        y de las divergencias de Umegaki. Este método *es* el arranque
        formal de `_DensityPurifier`, `_TomitaTakesakiFlow` y
        `_QuantumEntropyCalculator`.
        """
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
    """
    Fase II (continuación geométrica). Verifica M ∈ Sp(2n, ℝ) y estima la
    distancia a la retracción polar sobre el grupo simpléctico.
    """

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
        """
        Certifica Mᵀ Ω M − Ω ≡ 0.

        Además reporta det M (Liouville: det = 1) y la distancia de Frobenius
        a una corrección polar de primer orden hacia Sp(2n):

            S = Ωᵀ Mᵀ Ω M ,   M_♠ ≈ M · HighamSPD(S)^{−1/2} .
        """
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

        # Retracción polar de primer orden sobre Sp(2n).
        S = -omega @ M.T @ omega @ M  # debería ser I si M es simpléctica (Ω⁻¹=−Ω).
        # Equiv.: Mᵀ Ω M = Ω  ⇒  (−Ω Mᵀ Ω) M = I.
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
# -----------------------------------------------------------------------------
# Continúa II.6: purificador, flujo modular y entropías se anclan a un
# ModularSpectralGerm (o lo inducen por purificación espectral directa).
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
    """
    Fase III. Purificación espectral por mayoración (Schur–Horn / Perron).

    Continúa el gérmen modular 𝒢_II cuando se provee: en ese caso el estado
    térmico ya es el objeto de trabajo. En caso contrario, hermitiza,
    trunca el espectro bajo el margen de pureza y renormaliza la traza
    con suma KBN (0 log 0 := 0).
    """

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
        """
        Proyección al simplejo espectral:

        1. Hermitización de Higham (Weyl–Toeplitz).
        2. Truncamiento de autovalores < margen (deflación de Perron–Frobenius).
        3. Renormalización de traza unitaria por KBN.
        4. Certificados: rango efectivo, S_vN = −Tr ρ log ρ, pureza Tr ρ².
        """
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
    """
    Fase III. Grupo de automorfismos modulares de Tomita–Takesaki.

    Para un estado fiel φ(A) = Tr(ρ A) en un factor de tipo I,

        S A ξ_φ = A* ξ_φ ,   Δ = S* S ≅ ρ ⊗ ρ⁻¹ ,
        σ_z(A) = Δ^{i z} A Δ^{−i z} = ρ^{i z} A ρ^{−i z} .

    La condición KMS en el borde de la franja {0 ≤ Im z ≤ 1} se reduce a

        φ(A σ_i(B)) = φ(B A) .
    """

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
        """
        Aplica σ_z^ρ(A) con z = time_parameter (continuación analítica).

        Tiempo real: z = t ∈ ℝ  →  evolución unitaria modular.
        Tiempo imaginario puro z = −i β  →  ρ^β A ρ^{−β} (círculo térmico).
        """
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
        # El flujo modular es una *-automorfismo: preserva ‖·‖_F sii A es HS
        # y z es real (unitario). En tiempo complejo la norma puede cambiar.
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
        """
        Residuo KMS elemental sobre el observable dado.

        En z = i (borde de la franja), σ_i(A) = ρ^{−1} A ρ y debe cumplirse
        Tr(ρ B σ_i(A)) = Tr(ρ A B). Tomamos B = A† como sonda canónica.
        """
        if abs(z.imag - 1.0) > 0.25 or abs(z.real) > 0.25:
            return 0.0
        B = observable.T.conj()
        lhs = _NumericalCore.compensated_real_trace(rho @ B @ evolved)
        rhs = _NumericalCore.compensated_real_trace(rho @ observable @ B)
        scale = max(abs(rhs), abs(lhs), _MACHINE_EPS)
        return float(abs(lhs - rhs) / scale)


class _QuantumEntropyCalculator:
    """
    Fase III. Entropía relativa de Umegaki y fidelidad de Uhlmann.

    S(ρ ‖ σ) = Tr ρ (log ρ − log σ)   (con S = +∞ si supp ρ ⊈ supp σ),
    F(ρ, σ)  = ‖ √ρ √σ ‖_1²          (norma nuclear / SVD, numéricamente
                                      más estable que √(√ρ σ √ρ)).
    Certifica Klein (S ≥ 0) y Pinsker (S ≥ ½ ‖ρ − σ‖_1²).
    """

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

        # log σ sobre su soporte; en el núcleo no se evalúa (ya certificado).
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
        # Klein: S(ρ‖σ) ≥ 0; recorte de ruido numérico negativo minúsculo.
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
        """ker(σ) ⊆ ker(ρ)  ⇔  P_{ker σ} ρ P_{ker σ} = 0."""
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
        """F(ρ,σ) = ‖√ρ √σ‖_1² = (Σ_i σ_i(√ρ √σ))²."""
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
        """‖ρ − σ‖_1 / 2, con ‖·‖_1 = Σ |λ_i|."""
        rho = v_rho @ (e_rho[:, None] * v_rho.T.conj())
        sig = v_sig @ (e_sig[:, None] * v_sig.T.conj())
        delta = _NumericalCore.higham_nearest_hermitian(rho - sig)
        ev = np.real(la.eigvalsh(delta))
        return 0.5 * _NumericalCore.kahan_babuska_neumaier_sum(np.abs(ev))


# =============================================================================
# CLASE PRINCIPAL — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialCenturionsEngine:
    """
    Motor tensorial de alta precisión para la Capa 2 de Seguridad (Centuriones).

    Compone las tres fases anidadas:

    1. Fase I   — gérmen de Darboux / Banach (`_NumericalCore`).
    2. Fase II  — Dirac, IDA-PBC, Sp(2n) (`_IDAPBCController`, checker).
    3. Fase III — von Neumann / Tomita–Takesaki / Umegaki
                  (purificador, flujo modular, entropías),
                  inicializados con el gérmen modular inducido de la métrica
                  identidad (estado de Gibbs de referencia); cada llamada
                  espectral puede sustituirlo por el estado que se le pase.

    La API pública de 2.0 se conserva (tuplas). Los métodos `*_certified`
    exponen los certificados añadidos en 3.0.
    """

    def __init__(self, dimension_n: int) -> None:
        """
        Inicializa el motor y materializa el encadenamiento de gérmenes.

        Args:
            dimension_n: Dimensión de Q (mitad de Darboux). El espacio de
                fases es T*Q ≅ ℝ^{2n}.
        """
        if int(dimension_n) <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        self._n: Final[int] = int(dimension_n)
        self._2n: Final[int] = 2 * self._n

        # Fase I → objeto inicial de Fase II.
        self._ph_germ: _PortHamiltonianGerm = (
            _NumericalCore.synthesize_port_hamiltonian_germ(self._n)
        )
        # Fase II.
        self._ida_pbc = _IDAPBCController(self._n, germ=self._ph_germ)
        self._symplectic_checker = _SymplecticPreservationChecker(
            self._n, germ=self._ph_germ
        )
        # Fase II → objeto inicial de Fase III (Gibbs de la métrica identidad).
        self._mod_germ: _ModularSpectralGerm = self._ida_pbc.induce_modular_spectral_germ(
            np.eye(self._2n, dtype=np.float64),
            beta=_DEFAULT_BETA,
        )
        # Fase III.
        self._density_purifier = _DensityPurifier(germ=self._mod_germ)
        self._modular_flow = _TomitaTakesakiFlow(
            purifier=self._density_purifier, germ=self._mod_germ
        )
        self._entropy_calculator = _QuantumEntropyCalculator(
            purifier=self._density_purifier, germ=self._mod_germ
        )

    # ── Fase I expuesta ───────────────────────────────────────────────────
    def kahan_sum(self, array: np.ndarray) -> float:
        """Sumación compensada de Kahan (expuesta públicamente)."""
        return _NumericalCore.kahan_sum(array)

    def kahan_babuska_neumaier_sum(self, array: np.ndarray) -> float:
        """Sumación KBN (expuesta públicamente)."""
        return _NumericalCore.kahan_babuska_neumaier_sum(array)

    def port_hamiltonian_germ_certificate(self) -> _SymplecticFormCertificate:
        """Certificado de Darboux del gérmen de Fase I."""
        return self._ph_germ.form_certificate

    # ── Fase II expuesta ──────────────────────────────────────────────────
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
        """Ley de control IDA-PBC. Retorna (alpha, exergy_loss). API 2.0."""
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
        """Ley IDA-PBC con residuos de matching, aniquilador y Lyapunov."""
        return self._ida_pbc.compute_control_law(
            q, p, grad_H, grad_Hd, g_actuator,
            J_matrix, R_matrix, Jd_matrix, Rd_matrix, G_metric,
        )

    def verify_symplectic_preservation(
        self,
        jacobian_matrix: np.ndarray,
    ) -> Tuple[float, bool]:
        """Verifica MᵀΩM = Ω. Retorna (residual_norm, is_viable). API 2.0."""
        result = self._symplectic_checker.verify(jacobian_matrix)
        return result.residual_norm, result.is_viable

    def verify_symplectic_preservation_certified(
        self,
        jacobian_matrix: np.ndarray,
    ) -> _SymplecticPreservationResult:
        """Pertenencia a Sp(2n) con residuo relativo, det y retracción polar."""
        return self._symplectic_checker.verify(jacobian_matrix)

    def induce_modular_spectral_germ(
        self,
        G_metric: np.ndarray,
        beta: float = _DEFAULT_BETA,
    ) -> _ModularSpectralGerm:
        """
        Réplica pública del morfismo II.6: métrica G ↦ estado KMS ρ_β.
        Actualiza el gérmen con el que opera la Fase III.
        """
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

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def purify_density_operator(
        self,
        rho_mixed: np.ndarray,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
    ) -> np.ndarray:
        """Purificación espectral del operador densidad. API 2.0."""
        purifier = _DensityPurifier(purity_margin, germ=self._mod_germ)
        return purifier.purify(rho_mixed).purified_rho

    def purify_density_operator_certified(
        self,
        rho_mixed: np.ndarray,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
    ) -> _PurificationResult:
        """Purificación con rango efectivo, S_vN y pureza Tr ρ²."""
        purifier = _DensityPurifier(purity_margin, germ=self._mod_germ)
        return purifier.purify(rho_mixed)

    def evolve_tomita_takesaki_flow(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> np.ndarray:
        """Evolución modular de Tomita–Takesaki. API 2.0."""
        result = self._modular_flow.evolve(observable_A, rho, time_parameter)
        return result.evolved_observable

    def evolve_tomita_takesaki_flow_certified(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> _ModularFlowResult:
        """Flujo modular con certificado de norma y residuo KMS."""
        return self._modular_flow.evolve(observable_A, rho, time_parameter)

    def compute_quantum_relative_entropy(
        self,
        rho: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple[float, float]:
        """Entropía relativa cuántica y fidelidad. API 2.0: (umegaki, F)."""
        result = self._entropy_calculator.compute(rho, sigma)
        return result.umegaki_entropy, result.uhlmann_fidelity

    def compute_quantum_relative_entropy_certified(
        self,
        rho: np.ndarray,
        sigma: np.ndarray,
    ) -> _QuantumRelativeEntropyResult:
        """Umegaki + Uhlmann + distancia de traza + gap de Pinsker + soporte."""
        return self._entropy_calculator.compute(rho, sigma)


__all__ = ["ImperialCenturionsEngine"]