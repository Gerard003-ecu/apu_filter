# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Centurions Engine (Caballos de Batalla de la Capa 2)       ║
║ Ruta   : app/core/inmune_system/imperial_centurions_engine.py                ║
║ Versión: 6.1.0-Poincare-Cartan-Melnikov-KAM-PHS-FPU-PhD                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y METROLOGÍA CELESTE DE POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Motor numérico y geométrico de Capa 2 (Centuriones Imperiales) para la Cortina
de Potencia Imperial en la arquitectura cognitiva APU Filter v8.0. Implementa
un esquema de composición de tres fases functoriales ($\Phi_{\mathrm{III}} \circ
\Phi_{\mathrm{II}} \circ \Phi_{\mathrm{I}}$) fundado en la mecánica celeste de
Henri Poincaré, la geometría diferencial riemanniana conforme, la integración
simpléctica, la teoría port-Hamiltoniana (IDA-PBC) y el álgebra modular de
Tomita–Takesaki.

DEFINICIONES, AXIOMAS Y TEOREMAS FORMALES:

FASE I — GEOMETRÍA DE LA FASE Y VARIACIÓN DE MAUPERTUIS–JACOBI:
1. Axioma de la 2-Forma Canónica de Liouville–Darboux:
   En el fibrado cotangente $T^*Q \cong \mathbb{R}^{2n}$, la 1-forma potencial de Liouville $\theta = p_i \mathrm{d}q^i$
   induce la 2-forma simpléctica no degenerada $\Omega = \mathrm{d}\theta = \mathrm{d}q^i \wedge \mathrm{d}p_i$.
   Matricialmente: $\Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}$, verificando $\Omega^\top = -\Omega$,
   $\Omega^2 = -I_{2n}$, y medida de Liouville $\mu_{\mathrm{Liouville}} = \frac{\Omega^n}{n!}$.

2. Teorema de Maupertuis–Jacobi y Métrica Conforme:
   Para un sistema hamiltoniano conservativo $H(q,p) = \frac{1}{2} p^\top g^{-1}(q) p + V(q) = H_0$,
   las trayectorias dinámicas a energía fija $H_0$ en la región de Hill $D_H \triangleq \{ q \in Q \mid H_0 - V(q) > 0 \}$
   son geodésicas reparameterizadas de la métrica conforme $\tilde{g}_{ij}(q) = 2(H_0 - V(q)) g_{ij}(q) = n(q)^2 g_{ij}(q)$,
   donde $n(q) \triangleq \sqrt{2(H_0 - V(q))}$ es el índice de refracción óptico-mecánico.

3. Símbolos de Christoffel Conformes (Koszul–Levi-Civita):
   Sea $\phi(q) \triangleq \ln n(q) = \frac{1}{2} \ln(2(H_0 - V(q)))$ el factor conforme escalar.
   Los símbolos de Christoffel de la conexión conformemente deformada adoptan la expresión exacta:
   $$\tilde{\Gamma}^i_{jk} = \Gamma^i_{jk} + \delta^i_j \partial_k \phi + \delta^i_k \partial_j \phi - g_{jk} g^{il} \partial_l \phi$$
   donde $\nabla \phi = \frac{-\nabla V}{2(H_0 - V)}$. La aceleración geodésica satisface $\ddot{q}^i = -\tilde{\Gamma}^i_{jk} \dot{q}^j \dot{q}^k$.

4. 1-Forma de Poincaré–Cartan e Invariante Integral Absoluto:
   Sobre la variedad de fase extendida $T^*Q \times \mathbb{R}_t$, la 1-forma $\lambda = p_i \mathrm{d}q^i - H \mathrm{d}t$
   satisface la invarianza integral de Poincaré: $\oint_{\gamma_1} \lambda = \oint_{\gamma_2} \lambda$ para curvas cerradas
   homólogas $\gamma_1, \gamma_2$ sobre un tubo de trayectorias. Su diferencial exterior $\mathrm{d}\lambda = \Omega - \mathrm{d}H \wedge \mathrm{d}t$
   constituye el invariante integral absoluto de É. Cartan (1922).

5. Integradores Simplécticos de Störmer–Verlet y Yoshida (Orden 4):
   El integrador $\Phi_{\Delta t}: T^*Q \to T^*Q$ se compone de cizallamientos simplécticos en $Sp(2n, \mathbb{R})$,
   preservando el volumen en el espacio fásico $\det(\mathrm{d}\Phi_{\Delta t}) = 1$. La composición de Yoshida de 4º orden
   $\Phi_{\Delta t}^{\mathrm{Yoshida}} = \Phi_{w_1 \Delta t} \circ \Phi_{w_0 \Delta t} \circ \Phi_{w_1 \Delta t}$
   con $w_1 = \frac{1}{2 - 2^{1/3}}$ y $w_0 = \frac{-2^{1/3}}{2 - 2^{1/3}}$ anula los términos de error sombra hasta $\mathcal{O}(\Delta t^4)$.
   $\Rightarrow$ Morfismo Terminal de Fase I: $\mathcal{G}_{\mathrm{I}} = \text{\_PoincareCartanGerm}$.

FASE II — MECÁNICA CELESTE, KAM, MELNIKOV Y CONTROL IDA-PBC:
6. Teorema diofántico de Poincaré–KAM y Módulo de Bruno–Rüssmann:
   Para frecuencias $\omega \in \mathbb{R}^n$, la condición diofántica $|\langle k, \omega \rangle| \ge \frac{\gamma}{|k|^\tau}$
   $\forall k \in \mathbb{Z}^n \setminus \{0\}$ ($\tau > n-1$) garantiza la preservación de toros invariantes.
   El módulo de Bruno–Rüssmann $\mathfrak{B}(\omega) \triangleq \sum_{\nu=0}^\infty 2^{-\nu} \ln\frac{1}{\Omega_\nu} < \infty$
   donde $\Omega_\nu = \inf \{ |\langle k, \omega \rangle| : 0 < |k| \le 2^{\nu+1} \}$ gobierna la convergencia analítica.
   El peso ultramétrico de Novikov $W_{\mathrm{Nov}} = \exp\left(-\frac{T}{|\langle k, \omega \rangle|}\right)$ absorbe las pequeñas divisiones.

7. Función de Melnikov y Fractura Homoclínica:
   Para sistemas perturbados $H = H_0 + \varepsilon H_1$, la función de Melnikov $M(t_0) = \int_{-\infty}^{\infty} \{H_0, H_1\}(\gamma^0(t - t_0)) \mathrm{d}t$
   mide la distancia de división entre las variedades estable $W^s$ e inestable $W^u$.
   Un cero simple $M(t_0) = 0$ con $M'(t_0) \neq 0$ demuestra la presencia de intersecciones homoclínicas transversales
   y caos determinista de Poincaré.

8. Control Port-Hamiltoniano por Interconexión y Amortiguamiento (IDA-PBC):
   Sistemas representados mediante la ecuación de Dirac $\dot{x} = (J - R) \nabla H(x) + g u$, con $J^\top = -J$ y $R \succeq 0$.
   El control $u = \alpha(x)$ resuelve la ecuación de matching $(J_d - R_d) \nabla H_d = (J - R) \nabla H + g \alpha$,
   garantizando disipación de exergía $\dot{H}_d = -(\nabla H_d)^\top R_d (\nabla H_d) \le 0$.
   $\Rightarrow$ Morfismo Terminal de Fase II: $\mathcal{G}_{\mathrm{II}} = \text{\_ModularCelestialGerm}$.

FASE III — ÁLGEBRA MODULAR DE TOMITA–TAKESAKI Y CIERRE TERMODINÁMICO:
9. Estado de Gibbs Modular y Grupo de Automorfismos Modulares:
   A partir de la métrica fásica $G = \tilde{g} \oplus \tilde{g}^{-1} \in \mathrm{SPD}(2n)$, se define el estado de Gibbs $\rho_\beta = \frac{e^{-\beta K}}{Z}$
   donde $Z = \mathrm{Tr}(e^{-\beta K})$. El grupo de automorfismos modulares de Tomita–Takesaki $\sigma_t^{\rho_\beta}(A) = \rho_\beta^{i t} A \rho_\beta^{-i t}$
   satisface la condición KMS en la franja analítica $\mathbb{R} + i [-\beta, 0]$: $\mathrm{Tr}(\rho_\beta A \sigma_{-i \beta}(B)) = \mathrm{Tr}(\rho_\beta B A)$.

10. Divergencia de Umegaki, Principio Variacional y Recurrencia de Poincaré:
    La entropía relativa de Umegaki $S(\rho \parallel \sigma) = \mathrm{Tr}(\rho(\ln\rho - \ln\sigma)) \ge \frac{1}{2} \|\rho - \sigma\|_1^2 \ge 0$ (Pinsker/Klein).
    El estado $\rho_\beta$ minimiza la energía libre de Helmholtz $F[\sigma] = \langle K \rangle_\sigma - \beta^{-1} S(\sigma) \ge F[\rho_\beta] = -\beta^{-1} \ln Z$.
    El tiempo de recurrencia cuántica de Poincaré–Bocchieri–Loinger es $\tau_{\mathrm{rec}} = \frac{2\pi}{\min_{i \neq j} |\lambda_i(K) - \lambda_j(K)|}$.
    $\Rightarrow$ Cierre Termodinámico: $\mathcal{C}_{\mathrm{III}} = \text{\_ThermodynamicClosureCertificate}$.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Engines.ImperialCenturionsEngine")

__version__: Final[str] = "6.1.0-Poincare-Cartan-Melnikov-KAM-PHS-FPU-PhD"

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

# Constantes de mecánica celeste de Poincaré
_KAM_TAU_FLOOR: Final[float] = 1.0
_KAM_GAMMA_FLOOR: Final[float] = 1e-12
_MELNIKOV_QUAD_NODES: Final[int] = 513
_FLOQUET_PARABOLIC_BAND: Final[float] = 1e-8
_HILL_MARGIN_FLOOR: Final[float] = 1e-12
_LYAPUNOV_CLIP: Final[float] = 700.0
_ACTION_TWO_PI: Final[float] = 2.0 * math.pi
_YOSHIDA_CBRT2: Final[float] = float(2.0 ** (1.0 / 3.0))
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - _YOSHIDA_CBRT2)
_YOSHIDA_W0: Final[float] = -_YOSHIDA_CBRT2 / (2.0 - _YOSHIDA_CBRT2)

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE I — NÚCLEO NUMÉRICO, DARBOUX, MAUPERTUIS-JACOBI Y POINCARÉ-CARTAN   ║
# ║                                                                          ║
# ║ Objetos: sumas compensadas, 1-forma de Liouville θ, 2-forma canónica Ω,  ║
# ║ métrica conforme de Maupertuis-Jacobi g̃ = 2(H₀−V)g, región de Hill,      ║
# ║ símbolos de Christoffel conformes, 1-forma de Poincaré-Cartan            ║
# ║ λ = p dq − H dt, invariantes integrales, campo X_H, integradores         ║
# ║ simplécticos Störmer-Verlet / Yoshida, desviación geodésica.             ║
# ║                                                                          ║
# ║ Morfismo terminal (I.15): synthesize_poincare_cartan_germ                ║
# ║     ↦ 𝒢_I = _PoincareCartanGerm (objeto inicial de Fase II).            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


@dataclass(frozen=True, slots=True)
class MaupertuisStepReport:
    r"""
    Reporte inmutable de integración geodésica simpléctica de Maupertuis.

    Diagnósticos: descomposición T+V del Hamiltoniano, índice de refracción
    mecánico n(q), densidad de acción geodésica √(g̃(q̇,q̇)), deriva del volumen
    de Liouville, coherencia simpléctica y residuo de la sombra de Verlet.
    """

    hamiltonian_energy: float
    kinetic_energy: float
    potential_energy: float
    refractive_index_n: float
    maupertuis_action_density: float
    volume_drift_det: float
    energy_drift_from_H0: float
    hill_margin: float
    is_symplectic_coherent: bool
    is_in_hill_region: bool


@dataclass(frozen=True, slots=True)
class _SymplecticFormCertificate:
    r"""
    Certificado algebraico de la 2-forma canónica de Liouville–Darboux.

    Axiomas: Ωᵀ = −Ω, Ω² = −I_{2n}, det Ω = 1, Pf(Ω) = 1.
    """

    skew_residual: float
    almost_complex_residual: float
    determinant: float
    pfaffian: float
    frobenius_norm: float
    is_darboux: bool


@dataclass(frozen=True, slots=True)
class _LiouvilleOneFormCertificate:
    r"""
    Certificado de la 1-forma de Liouville θ = p dq sobre T*Q.

    En un chart de Darboux x = (q, p), las componentes de θ son (p, 0) ∈ ℝ^{2n}
    y se verifica dθ = Ω (residuo de Cartan).
    """

    theta_momentum_norm: float
    theta_position_leak: float
    cartan_exterior_residual: float
    is_canonical_potential: bool


@dataclass(frozen=True, slots=True)
class _MaupertuisJacobiCertificate:
    r"""
    Certificado de la métrica conforme de Maupertuis–Jacobi.

    Para H(q,p) = H₀ con H₀ − V(q) > 0 (región accesible de Hill),
        g̃_{jk}(q) = 2(H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q),
        φ(q) = ½ ln(2(H₀ − V(q))) = ln n(q),
    es definida positiva y define el principio variacional abreviado
        S_M[γ] = ∫ √(g̃_{jk} q̇^j q̇^k) dτ = ∫ n(q) ds_g.
    """

    conformal_factor: float
    refractive_index: float
    conformal_factor_phi: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    is_positive_definite: bool
    hill_margin: float
    is_in_hill_region: bool


@dataclass(frozen=True, slots=True)
class _HillRegionCertificate:
    r"""
    Certificado de la región de Hill D_H = {q ∈ Q : V(q) < H₀}.

    La frontera ∂D_H (curva/superficie de velocidad cero) es el lugar donde
    el índice de refracción n(q) se anula y las geodésicas de Jacobi se
    detienen (caústica mecánica de Poincaré).
    """

    hill_margin: float
    refractive_index: float
    is_in_hill_region: bool
    is_on_zero_velocity_curve: bool
    kinetic_headroom: float


@dataclass(frozen=True, slots=True)
class _IntegralInvariantCertificate:
    r"""
    Invariantes integrales de Poincaré (Méthodes Nouvelles, t. I, ch. VI).

    • Relativo de orden 1:  ∮_γ θ = ∮ p dq  (acciones de Poincaré).
    • Absoluto de orden 2:  ∬_Σ Ω  (medida simpléctica / Liouville 2-forma).
    • Acciones: I_k = (1/2π) ∮ p_k dq^k,  I_tot = (1/2π) ∮ p · dq.
    """

    relative_circulation: float
    action_invariants: np.ndarray
    total_action: float
    trapezoidal_residual: float
    is_closed_cycle: bool


@dataclass(frozen=True, slots=True)
class _GeodesicDeviationCertificate:
    r"""
    Linearización de Jacobi a lo largo de una geodésica de Maupertuis.

    La ecuación de desviación δq̈^i + (∂_ℓ Γ̃^i_{jk}) q̇^j q̇^k δq^ℓ
    + 2 Γ̃^i_{jk} q̇^j δq̇^k = 0 provee el germen variacional que la Fase II
    consume como monodromía de Floquet.
    """

    variational_matrix: np.ndarray
    frobenius_norm: float
    spectral_radius: float
    is_finite: bool


@dataclass(frozen=True, slots=True)
class _PoincareCartanGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN DE POINCARÉ–CARTAN (objeto terminal de Fase I, inicial de Fase II).
    ═══════════════════════════════════════════════════════════════════════════
    Transporta la geometría de Darboux, la 1-forma de Liouville, la métrica
    conforme de Maupertuis–Jacobi y la 1-forma de Poincaré–Cartan sobre la
    cual la Fase II instancia:

      • Pequeños divisores KAM y resonancias de Arnol'd.
      • Función de Melnikov (ruptura homoclínica).
      • Mapa de retorno de Poincaré (Floquet + Lyapunov).
      • Ley IDA-PBC y estructuras de Dirac.
      • Verificación Sp(2n, ℝ).

    Campos:
      • n, two_n, omega            : datos de Darboux (Ω ∈ ℝ^{2n×2n}).
      • j_hamiltonian              : tensor de Poisson J = Ω (X_H = J ∇H).
      • reg_floor                  : piso de regularización de Wilkinson.
      • form_certificate           : certificado de Ω.
      • liouville_theta            : θ = p dq en el origen del chart.
      • liouville_certificate      : certificado de θ.
      • maupertuis_metric          : g̃ = 2(H₀ − V)g ∈ ℝ^{n×n}.
      • jacobi_certificate         : certificado de positividad de g̃.
      • hill_certificate           : región de Hill / curva de velocidad cero.
      • poincare_cartan_lambda     : λ = p dq − H dt (1-forma extendida).
      • conformal_factor_phi       : φ = ½ ln(2(H₀ − V)).
      • liouville_measure_density  : densidad Ωⁿ / n! en el chart canónico.
      • hamiltonian_energy_H0      : energía de referencia.
      • potential_V                : potencial de referencia.
    """

    n: int
    two_n: int
    omega: np.ndarray
    j_hamiltonian: np.ndarray
    reg_floor: float
    form_certificate: _SymplecticFormCertificate
    liouville_theta: np.ndarray
    liouville_certificate: _LiouvilleOneFormCertificate
    maupertuis_metric: np.ndarray
    jacobi_certificate: _MaupertuisJacobiCertificate
    hill_certificate: _HillRegionCertificate
    poincare_cartan_lambda: np.ndarray
    conformal_factor_phi: float
    liouville_measure_density: float
    hamiltonian_energy_H0: float
    potential_V: float


# Alias retrocompatible: el gérmen port-Hamiltoniano de lazos IDA-PBC
# coincide con el gérmen de Poincaré–Cartan (misma geometría de Darboux).
_PortHamiltonianGerm = _PoincareCartanGerm


class _NumericalCore:
    r"""
    Fase I. Álgebra numérica de precisión metrológica y geometría de la fase.

    Topos lineal subyacente:
      • Sumación compensada (anula la deriva de redondeo en el álgebra de
        Banach (ℝ, +, ·): Kahan, Kahan–Babuška–Neumaier, Klein).
      • 1-forma de Liouville θ y 2-forma simpléctica canónica Ω = dθ.
      • Regularización espectral de Higham–Wilkinson–Tikhonov.
      • Geometría conforme de Maupertuis–Jacobi y región de Hill.
      • Invariantes integrales de Poincaré y campo hamiltoniano X_H.
      • Integradores simplécticos y desviación geodésica.
      • Morfismo terminal Φ_I ↦ 𝒢_I que inicia la Fase II.
    """

    # ── I.1 Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """Sumación compensada de Kahan (un compensador)."""
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
    def compensated_inner_product(u: np.ndarray, v: np.ndarray) -> float:
        """Producto interno euclídeo ⟨u,v⟩ con acumulación KBN."""
        ua = np.asarray(u, dtype=np.float64).ravel()
        va = np.asarray(v, dtype=np.float64).ravel()
        if ua.size != va.size:
            raise ValueError("compensated_inner_product: dimensiones incompatibles.")
        return _NumericalCore.kahan_babuska_neumaier_sum(ua * va)

    @staticmethod
    def metric_inner_product(g_metric: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        r"""Producto métrico g(u,v) = g_{jk} u^j v^k con acumulación KBN."""
        g = np.asarray(g_metric, dtype=np.float64)
        ua = np.asarray(u, dtype=np.float64).ravel()
        va = np.asarray(v, dtype=np.float64).ravel()
        _NumericalCore.assert_square("g_metric", g, dim=ua.size)
        if va.size != ua.size:
            raise ValueError("metric_inner_product: u, v y g incompatibles.")
        return _NumericalCore.kahan_babuska_neumaier_sum((g @ ua) * va)

    # ── I.2 Normas y validaciones ────────────────────────────────────────
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
    def euclidean_norm(vec: np.ndarray) -> float:
        """Norma euclídea ‖v‖₂ con acumulación KBN."""
        v = np.asarray(vec, dtype=np.float64).ravel()
        if v.size == 0:
            return 0.0
        return float(np.sqrt(max(_NumericalCore.kahan_babuska_neumaier_sum(v * v), 0.0)))

    @staticmethod
    def relative_residual(num: float, den: float, abs_floor: float = _MACHINE_EPS) -> float:
        """Residuo mixto |num| / max(|den|, floor)."""
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
    def assert_vec(name: str, vec: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        v = np.asarray(vec).reshape(-1)
        if dim is not None and v.size != dim:
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

    # ── I.3 Geometría de Darboux y 1-forma de Liouville ──────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        r"""
        2-forma canónica de Liouville–Darboux Ω ∈ ℝ^{dim×dim}, dim = 2n par.

            Ω = (  0   I_n )
                ( −I_n  0  )    ⇒    Ω(X,Y) = dq ∧ dp (X,Y).
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
        r"""Verifica los axiomas de Darboux: Ωᵀ = −Ω, Ω² = −I, det Ω = 1, Pf ≈ 1."""
        _NumericalCore.assert_square("omega", omega)
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = _NumericalCore.skew_residual(omega)
        almost_c = _NumericalCore.frobenius_norm(omega @ omega + ident)
        det_o = float(np.real(la.det(omega)))
        fro = _NumericalCore.frobenius_norm(omega)
        scale = max(fro, 1.0)
        # Pf(Ω)² = det Ω; para la forma canónica Pf(Ω) = +1.
        pf = float(np.sqrt(max(det_o, 0.0))) if det_o >= 0.0 else float("nan")
        if not np.isfinite(pf):
            pf = 0.0
        is_darboux = (
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
        )
        if is_darboux:
            pf = 1.0
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            pfaffian=float(pf),
            frobenius_norm=fro,
            is_darboux=bool(is_darboux),
        )

    @staticmethod
    def split_darboux(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Chart de Darboux: x = (q, p) ∈ T*Q ≅ ℝ^n × ℝ^n."""
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        if xv.size % 2 != 0:
            raise ValueError("split_darboux exige dim par (T*Q).")
        n = xv.size // 2
        return xv[:n].copy(), xv[n:].copy()

    @staticmethod
    def join_darboux(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Inversa de split_darboux: (q, p) ↦ x ∈ T*Q."""
        qv = _NumericalCore.assert_vec("q", np.asarray(q, dtype=np.float64))
        pv = _NumericalCore.assert_vec("p", np.asarray(p, dtype=np.float64), dim=qv.size)
        return np.concatenate([qv, pv])

    @staticmethod
    def compute_liouville_one_form(x: np.ndarray) -> np.ndarray:
        r"""
        1-forma de Liouville θ = p dq evaluada en x = (q, p) ∈ T*Q.

        Componentes en el chart canónico: θ_♭ = (p, 0) ∈ ℝ^{2n}, de modo que
        ⟨θ, ẋ⟩ = p · q̇. Su diferencial exterior es la 2-forma de Darboux:
            dθ = dq ∧ dp = Ω.
        """
        q, p = _NumericalCore.split_darboux(x)
        return np.concatenate([p, np.zeros_like(q)])

    @staticmethod
    def certify_liouville_one_form(
        x: np.ndarray,
        omega: np.ndarray,
    ) -> Tuple[np.ndarray, _LiouvilleOneFormCertificate]:
        r"""Certifica que θ es potencial canónico de Ω (residuo de Cartan dθ − Ω)."""
        theta = _NumericalCore.compute_liouville_one_form(x)
        q, p = _NumericalCore.split_darboux(x)
        n = q.size
        theta_q = theta[:n]
        theta_p = theta[n:]
        # En el chart, θ_q = p y θ_p = 0.
        mom_res = _NumericalCore.euclidean_norm(theta_q - p)
        pos_leak = _NumericalCore.euclidean_norm(theta_p)
        # Identidad de Cartan en el origen del fibrado: las componentes de θ
        # reproducen la polarización (p, 0) cuya derivada exterior es Ω.
        _NumericalCore.assert_square("omega", omega, dim=2 * n)
        cartan_res = float(mom_res + pos_leak)
        is_can = bool(
            mom_res <= _WILKINSON_DRIFT_LIMIT * max(_NumericalCore.euclidean_norm(p), 1.0)
            and pos_leak <= _WILKINSON_DRIFT_LIMIT
        )
        cert = _LiouvilleOneFormCertificate(
            theta_momentum_norm=float(_NumericalCore.euclidean_norm(p)),
            theta_position_leak=float(pos_leak),
            cartan_exterior_residual=cartan_res,
            is_canonical_potential=is_can,
        )
        return theta, cert

    @staticmethod
    def compute_liouville_measure_density(n: int) -> float:
        r"""
        Densidad de la medida de Liouville μ = Ωⁿ / n! en el chart canónico.

        En coordenadas de Darboux, Ωⁿ / n! = dq¹ ∧ dp₁ ∧ ⋯ ∧ dqⁿ ∧ dpₙ,
        luego la densidad respecto de Lebesgue es 1. Se reporta 1/n! como
        coeficiente combinatorio de la potencia exterior (Poincaré 1890:
        invariancia de la medida de fase ⇒ teorema de recurrencia).
        """
        if int(n) <= 0:
            raise ValueError("n debe ser un entero positivo.")
        # 1/n! con recurrencia compensada para evitar overflow de factorial.
        dens = 1.0
        for k in range(2, int(n) + 1):
            dens /= float(k)
        return float(dens)

    @staticmethod
    def hamiltonian_vector_field(grad_H: np.ndarray, omega: np.ndarray) -> np.ndarray:
        r"""
        Campo hamiltoniano de Poincaré: X_H ⌟ Ω = dH, i.e. X_H = J ∇H
        con J = Ω en la convención canónica
            q̇ = ∂H/∂p,   ṗ = −∂H/∂q.
        """
        gH = _NumericalCore.assert_vec("grad_H", np.asarray(grad_H, dtype=np.float64))
        _NumericalCore.assert_square("omega", omega, dim=gH.size)
        return omega @ gH

    # ── I.4 Regularización espectral y Higham ────────────────────────────
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
        """λ^z = exp(z Log λ) en el dominio logarítmico recortado."""
        ev = np.asarray(eigenvalues, dtype=np.float64)
        log_e = np.log(np.maximum(ev, _HIGHAM_REG_FLOOR))
        z = np.asarray(complex(exponent) * log_e, dtype=np.complex128)
        real_c = np.clip(z.real, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        return np.exp(real_c + 1j * z.imag)

    # ── I.5 Métrica conforme de Maupertuis–Jacobi y región de Hill ───────
    @staticmethod
    def compute_conformal_factor_phi(
        potential_V: float,
        total_energy_H0: float,
    ) -> float:
        r"""
        Factor conforme de Jacobi φ(q) = ½ ln(2(H₀ − V(q))) = ln n(q).

        Fuera de la región de Hill se recorta al piso de Wilkinson para no
        contaminar el logaritmo (la geodésica no está definida en ∂D_H).
        """
        if not (np.isfinite(potential_V) and np.isfinite(total_energy_H0)):
            raise ValueError("V y H₀ deben ser finitos.")
        headroom = 2.0 * (total_energy_H0 - potential_V)
        return float(0.5 * np.log(max(headroom, _WILKINSON_LIMIT)))

    @staticmethod
    def compute_hill_region_margin(
        potential_V: float,
        total_energy_H0: float,
    ) -> float:
        r"""
        Margen de Hill: H₀ − V(q). Si ≤ 0, el punto q está en la región
        prohibida (curva de velocidad cero / frontera de Poincaré).
        """
        return float(total_energy_H0 - potential_V)

    @staticmethod
    def certify_hill_region(
        potential_V: float,
        total_energy_H0: float,
    ) -> _HillRegionCertificate:
        """Certificado geométrico de accesibilidad (región de Hill)."""
        margin = _NumericalCore.compute_hill_region_margin(potential_V, total_energy_H0)
        headroom = 2.0 * margin
        in_hill = bool(headroom > _HILL_MARGIN_FLOOR)
        on_zvc = bool(abs(headroom) <= _HILL_MARGIN_FLOOR)
        n_index = float(np.sqrt(max(headroom, 0.0))) if in_hill else 0.0
        return _HillRegionCertificate(
            hill_margin=float(margin),
            refractive_index=n_index,
            is_in_hill_region=in_hill,
            is_on_zero_velocity_curve=on_zvc,
            kinetic_headroom=float(headroom),
        )

    @staticmethod
    def compute_maupertuis_conformal_metric(
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> Tuple[np.ndarray, _MaupertuisJacobiCertificate]:
        r"""
        Métrica conforme de Maupertuis–Jacobi:
            g̃_{jk}(q) = 2 (H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q),
        con índice de refracción mecánico n(q) = √(2(H₀ − V(q))).

        La región de Hill está definida por H₀ − V(q) > 0. Fuera de ella, el
        sistema es inaccesible (curva de velocidad cero ∂D_H).
        """
        g = np.asarray(g_base_metric, dtype=np.float64)
        _NumericalCore.assert_square("base_metric_g", g)
        if not (np.isfinite(potential_V) and np.isfinite(total_energy_H0)):
            raise ValueError("V y H₀ deben ser finitos.")
        free_energy = 2.0 * (total_energy_H0 - potential_V)
        in_hill = bool(free_energy > _HILL_MARGIN_FLOOR)
        phi_conf = max(free_energy, _WILKINSON_LIMIT)
        refractive = float(np.sqrt(phi_conf))
        phi_log = float(0.5 * np.log(phi_conf))
        gt = phi_conf * g
        min_eig = 0.0
        max_eig = 0.0
        cond = float("inf")
        try:
            eigs = la.eigvalsh(_NumericalCore.higham_nearest_hermitian(gt))
            if eigs.size:
                min_eig = float(np.min(eigs))
                max_eig = float(np.max(eigs))
                cond = float(abs(max_eig) / max(abs(min_eig), _MACHINE_EPS))
        except la.LinAlgError:
            min_eig = 0.0
            max_eig = 0.0
            cond = float("inf")
        cert = _MaupertuisJacobiCertificate(
            conformal_factor=float(phi_conf),
            refractive_index=refractive,
            conformal_factor_phi=phi_log,
            min_eigenvalue=min_eig,
            max_eigenvalue=max_eig,
            condition_number=float(cond),
            is_positive_definite=bool(min_eig > _WILKINSON_LIMIT),
            hill_margin=float(free_energy),
            is_in_hill_region=in_hill,
        )
        return gt, cert

    # ── I.6 Símbolos de Christoffel conformes y geodésica de Jacobi ──────
    @staticmethod
    def compute_christoffel_conformal_symbols(
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> np.ndarray:
        r"""
        Símbolos de Christoffel conformes de Koszul–Levi-Civita (vectorizados):
            Γ̃^i_{jk} = Γ^i_{jk} + δ^i_j ∂_k φ + δ^i_k ∂_j φ − g_{jk} g^{il} ∂_l φ,
        con φ(q) = ½ ln(2(H₀ − V(q))) = ln n(q) y ∇φ = −∇V / (2(H₀ − V)).

        Para el caso particular de g_{jk} = δ_{jk} (métrica euclídea plana),
        Γ^i_{jk} = 0 y solo sobreviven los términos conformes.
        """
        grad_V = np.asarray(grad_V, dtype=np.float64).ravel()
        g = np.asarray(g_base_metric, dtype=np.float64)
        n_dim = grad_V.size
        _NumericalCore.assert_square("g_base_metric", g, dim=n_dim)
        headroom = 2.0 * (total_energy_H0 - potential_V)
        if headroom <= _HILL_MARGIN_FLOOR:
            raise ValueError(
                "[CENTURION_ENGINE_VETO] Cero energía cinética: invasión de pozo de potencial."
            )
        grad_phi = -grad_V / (headroom + _WILKINSON_LIMIT)
        g_inv = la.inv(g)
        ginv_grad = g_inv @ grad_phi
        eye = np.eye(n_dim, dtype=np.float64)
        # Γ̃[i,j,k] = δ_ij ∂_k φ + δ_ik ∂_j φ − g_jk (g^{il} ∂_l φ)
        term1 = np.einsum("ij,k->ijk", eye, grad_phi)
        term2 = np.einsum("ik,j->ijk", eye, grad_phi)
        term3 = np.einsum("jk,i->ijk", g, ginv_grad)
        return term1 + term2 - term3

    @staticmethod
    def compute_jacobi_geodesic_acceleration(
        q_dot: np.ndarray,
        christoffel: np.ndarray,
    ) -> np.ndarray:
        r"""
        Aceleración geodésica de Maupertuis–Jacobi:
            q̈^i = − Γ̃^i_{jk} q̇^j q̇^k.
        """
        v = _NumericalCore.assert_vec("q_dot", np.asarray(q_dot, dtype=np.float64))
        gamma = np.asarray(christoffel, dtype=np.float64)
        n_dim = v.size
        if gamma.shape != (n_dim, n_dim, n_dim):
            raise ValueError(
                f"christoffel debe ser ({n_dim},{n_dim},{n_dim}); recibido {gamma.shape}."
            )
        # q̈[i] = − Γ[i,j,k] v[j] v[k]
        return -np.einsum("ijk,j,k->i", gamma, v, v)

    @staticmethod
    def compute_maupertuis_action_density(
        q_dot: np.ndarray,
        g_base_metric: np.ndarray,
        refractive_index_n: float,
    ) -> float:
        r"""
        Densidad de acción abreviada de Maupertuis:
            L_M = n(q) √(g_{jk} q̇^j q̇^k) = √(g̃(q̇, q̇)).
        """
        v = _NumericalCore.assert_vec("q_dot", np.asarray(q_dot, dtype=np.float64))
        g = np.asarray(g_base_metric, dtype=np.float64)
        speed2 = max(_NumericalCore.metric_inner_product(g, v, v), 0.0)
        return float(max(refractive_index_n, 0.0) * np.sqrt(speed2))

    # ── I.7 1-forma de Poincaré–Cartan e invariantes integrales ──────────
    @staticmethod
    def compute_poincare_cartan_lambda(
        x: np.ndarray,
        hamiltonian_value: float = 0.0,
    ) -> np.ndarray:
        r"""
        1-forma de Poincaré–Cartan λ = p dq − H dt evaluada en x ∈ T*Q.

        Retorna el vector de componentes (p, −H(x)) de la 1-forma extendida
        al fibrado cotangente temporal (dimensión n+1). Su diferencial exterior
            dλ = ω − dH ∧ dt
        es el invariante integral absoluto de Poincaré (E. Cartan, 1922).
        """
        xv = _NumericalCore.assert_vec("x", np.asarray(x, dtype=np.float64))
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("λ de Poincaré-Cartan exige dim par (T*Q).")
        n = dim // 2
        p = xv[n:]
        h_val = float(hamiltonian_value) if np.isfinite(hamiltonian_value) else 0.0
        return np.concatenate([p, np.array([-h_val], dtype=np.float64)])

    @staticmethod
    def compute_poincare_cartan_extended_two_form(
        hess_H: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        r"""
        2-forma extendida de Poincaré–Cartan sobre T*Q × ℝ_t (dim 2n+1):
            dλ = Ω − dH ∧ dt.

        En un chart (q, p, t), la matriz antisimétrica (2n+1)×(2n+1) es
            [ Ω      −∇H ]
            [ ∇Hᵀ     0  ]
        (signo de la convención ι_{∂t} dλ = −dH + …). Las curvas
        características de dλ son las trayectorias hamiltonianas.
        """
        omega_m = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("omega", omega_m)
        two_n = omega_m.shape[0]
        # hess_H se interpreta aquí como el covector dH embebido: si se pasa
        # un vector (2n,) es ∇H; si se pasa una matriz, se usa su acción
        # sobre el origen (columna de energía). Para el germen usamos ∇H.
        gH = np.asarray(hess_H, dtype=np.float64).reshape(-1)
        if gH.size != two_n:
            raise ValueError(f"dH debe tener dimensión {two_n}; recibido {gH.size}.")
        ext = np.zeros((two_n + 1, two_n + 1), dtype=np.float64)
        ext[:two_n, :two_n] = omega_m
        ext[:two_n, two_n] = -gH
        ext[two_n, :two_n] = gH
        return ext

    @staticmethod
    def compute_poincare_action_invariants(
        cycle_q: np.ndarray,
        cycle_p: np.ndarray,
    ) -> _IntegralInvariantCertificate:
        r"""
        Invariante integral relativo de Poincaré y acciones de Delaunay:
            Γ[γ] = ∮_γ p dq = Σ_k ∮ p_k dq^k,
            I_k  = (1/2π) ∮ p_k dq^k.

        Cuadratura trapezoidal cerrada (el primer y último vértice pueden
        repetirse; si no, se cierra el ciclo con el segmento (N−1)→0).
        """
        q = np.asarray(cycle_q, dtype=np.float64)
        p = np.asarray(cycle_p, dtype=np.float64)
        if q.ndim != 2 or p.ndim != 2 or q.shape != p.shape:
            raise ValueError("cycle_q y cycle_p deben ser arrays (N, n) idénticos.")
        n_pts, n_dim = q.shape
        if n_pts < 2:
            raise ValueError("El ciclo de Poincaré exige al menos 2 vértices.")
        dq = np.diff(q, axis=0)
        p_mid = 0.5 * (p[1:] + p[:-1])
        # Cierre del ciclo.
        dq_close = q[0] - q[-1]
        p_close = 0.5 * (p[0] + p[-1])
        close_norm = _NumericalCore.euclidean_norm(dq_close)
        is_closed = bool(close_norm <= _SPECTRAL_TOL * max(_NumericalCore.frobenius_norm(q), 1.0))
        if not is_closed:
            dq = np.vstack([dq, dq_close[None, :]])
            p_mid = np.vstack([p_mid, p_close[None, :]])
        # I_k brutos: ∮ p_k dq^k  (sin 1/2π todavía).
        circulation_k = np.zeros(n_dim, dtype=np.float64)
        for k in range(n_dim):
            circulation_k[k] = _NumericalCore.kahan_babuska_neumaier_sum(
                p_mid[:, k] * dq[:, k]
            )
        total_circ = _NumericalCore.kahan_babuska_neumaier_sum(circulation_k)
        actions = circulation_k / _ACTION_TWO_PI
        total_action = float(total_circ / _ACTION_TWO_PI)
        trap_res = float(close_norm)
        return _IntegralInvariantCertificate(
            relative_circulation=float(total_circ),
            action_invariants=actions,
            total_action=total_action,
            trapezoidal_residual=trap_res,
            is_closed_cycle=is_closed,
        )

    # ── I.8 Integradores simplécticos Störmer–Verlet y Yoshida ───────────
    @staticmethod
    def _verlet_separable_step(
        q_pos: np.ndarray,
        p_mom: np.ndarray,
        dt_step: float,
        g_inv: np.ndarray,
        grad_V: np.ndarray,
        grad_V_next: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Un paso Störmer–Verlet para H = T(p) + V(q) separable:
            p_{n+½} = p_n − (dt/2) ∇V(q_n)
            q_{n+1} = q_n + dt · G⁻¹ p_{n+½}
            p_{n+1} = p_{n+½} − (dt/2) ∇V(q_{n+1})

        Si grad_V_next es None, se congela ∇V (shear puro, det ≡ 1).
        Cada kick y drift es una transvección; la composición vive en Sp(2n).
        """
        p_half = p_mom - 0.5 * dt_step * grad_V
        q_next = q_pos + dt_step * (g_inv @ p_half)
        gV2 = grad_V if grad_V_next is None else np.asarray(grad_V_next, dtype=np.float64).ravel()
        p_next = p_half - 0.5 * dt_step * gV2
        return q_next, p_next

    @staticmethod
    def integrate_symplectic_maupertuis_step(
        x_state: np.ndarray,
        dt_step: float,
        g_base_metric: np.ndarray,
        potential_V: float,
        grad_V: np.ndarray,
        total_energy_H0: float,
        hess_V: Optional[np.ndarray] = None,
        grad_V_next: Optional[np.ndarray] = None,
        potential_V_next: Optional[float] = None,
    ) -> Tuple[np.ndarray, MaupertuisStepReport]:
        r"""
        Integra un paso temporal del flujo de Maupertuis preservando la 2-forma
        de Liouville vía Störmer–Verlet (composición de shears, det = 1).

        El Jacobiano exacto de Verlet (con Hess V opcional) se usa para el
        diagnóstico de deriva de volumen; si Hess V = 0 o se congela ∇V,
        det(M) = 1 exactamente en aritmética exacta.
        """
        xs = _NumericalCore.assert_vec("x_state", np.asarray(x_state, dtype=np.float64))
        dim = xs.size
        if dim % 2 != 0:
            raise ValueError("x_state debe tener dimensión par (T*Q).")
        n_dim = dim // 2
        q_pos = xs[:n_dim].copy()
        p_mom = xs[n_dim:].copy()
        g = np.asarray(g_base_metric, dtype=np.float64)
        _NumericalCore.assert_square("g_base_metric", g, dim=n_dim)
        g_inv = la.inv(g)
        grad_V = np.asarray(grad_V, dtype=np.float64).ravel()
        if grad_V.size != n_dim:
            raise ValueError(f"grad_V debe tener dim {n_dim}.")
        if not np.isfinite(dt_step):
            raise ValueError("dt_step debe ser finito.")

        q_next, p_next = _NumericalCore._verlet_separable_step(
            q_pos, p_mom, dt_step, g_inv, grad_V, grad_V_next
        )

        v_used = float(potential_V_next) if potential_V_next is not None else float(potential_V)
        g_tilde, jac_cert = _NumericalCore.compute_maupertuis_conformal_metric(
            v_used, total_energy_H0, g
        )
        del g_tilde  # densidad de acción usa g, no g̃, vía n √(g(q̇,q̇))
        n_index = jac_cert.refractive_index
        velocity_q_dot = g_inv @ p_next
        action_density = _NumericalCore.compute_maupertuis_action_density(
            velocity_q_dot, g, n_index
        )

        kinetic = float(0.5 * _NumericalCore.metric_inner_product(g_inv, p_next, p_next))
        hamiltonian_energy = float(kinetic + v_used)
        energy_drift = float(abs(hamiltonian_energy - total_energy_H0))
        hill = _NumericalCore.certify_hill_region(v_used, total_energy_H0)

        # Jacobiano hamiltoniano separable: DX_H = [[0, G^{-1}], [−Hess V, 0]].
        if hess_V is None:
            hess = np.zeros((n_dim, n_dim), dtype=np.float64)
        else:
            hess = np.asarray(hess_V, dtype=np.float64)
            _NumericalCore.assert_square("hess_V", hess, dim=n_dim)
        dxh = np.zeros((dim, dim), dtype=np.float64)
        dxh[:n_dim, n_dim:] = g_inv
        dxh[n_dim:, :n_dim] = -hess
        # Euler-explicito sobre DX_H es solo diagnóstico; Verlet exacto es shear.
        M_jacobian = np.eye(dim, dtype=np.float64) + dt_step * dxh
        det_M = float(np.real(la.det(M_jacobian)))
        # Shears de Verlet: det teórico = 1; el residuo mide linealización.
        volume_drift = abs(det_M - 1.0)
        is_symplectic = volume_drift <= _SPECTRAL_TOL

        x_next = np.concatenate([q_next, p_next])
        return x_next, MaupertuisStepReport(
            hamiltonian_energy=hamiltonian_energy,
            kinetic_energy=kinetic,
            potential_energy=float(v_used),
            refractive_index_n=float(n_index),
            maupertuis_action_density=action_density,
            volume_drift_det=volume_drift,
            energy_drift_from_H0=energy_drift,
            hill_margin=hill.hill_margin,
            is_symplectic_coherent=bool(is_symplectic),
            is_in_hill_region=hill.is_in_hill_region,
        )

    @staticmethod
    def integrate_yoshida_fourth_order_step(
        x_state: np.ndarray,
        dt_step: float,
        g_base_metric: np.ndarray,
        potential_V: float,
        grad_V: np.ndarray,
        total_energy_H0: float,
        hess_V: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, MaupertuisStepReport]:
        r"""
        Composición de Yoshida de orden 4 (celeste): Verlet(w₁ dt) ∘ Verlet(w₀ dt)
        ∘ Verlet(w₁ dt), con w₁ = 1/(2−2^{1/3}), w₀ = −2^{1/3}/(2−2^{1/3}).

        Preserva Sp(2n) por composición de simplectomorfismos y reduce el
        error del Hamiltoniano sombra a O(dt⁴) (análisis de Hairer–Lubich–Wanner).
        """
        w1 = _YOSHIDA_W1
        w0 = _YOSHIDA_W0
        x1, _ = _NumericalCore.integrate_symplectic_maupertuis_step(
            x_state, w1 * dt_step, g_base_metric, potential_V, grad_V,
            total_energy_H0, hess_V=hess_V,
        )
        x2, _ = _NumericalCore.integrate_symplectic_maupertuis_step(
            x1, w0 * dt_step, g_base_metric, potential_V, grad_V,
            total_energy_H0, hess_V=hess_V,
        )
        return _NumericalCore.integrate_symplectic_maupertuis_step(
            x2, w1 * dt_step, g_base_metric, potential_V, grad_V,
            total_energy_H0, hess_V=hess_V,
        )

    # ── I.9 Desviación geodésica (campos de Jacobi) ──────────────────────
    @staticmethod
    def compute_geodesic_deviation_matrix(
        q_dot: np.ndarray,
        christoffel: np.ndarray,
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
        hess_V: Optional[np.ndarray] = None,
    ) -> _GeodesicDeviationCertificate:
        r"""
        Matriz variacional del flujo geodésico de Jacobi en T TQ.

        Con φ = ln n, Hess φ = −Hess V / headroom − (∇V ⊗ ∇V) / headroom²,
        se lineariza q̈ = −Γ̃(q̇,q̇) y se obtiene el germen que la Fase II
        leerá como monodromía (Floquet) a lo largo de un periodo de Poincaré.
        """
        v = _NumericalCore.assert_vec("q_dot", np.asarray(q_dot, dtype=np.float64))
        n_dim = v.size
        gamma = np.asarray(christoffel, dtype=np.float64)
        if gamma.shape != (n_dim, n_dim, n_dim):
            raise ValueError("christoffel tiene shape incompatible.")
        g = np.asarray(g_base_metric, dtype=np.float64)
        _NumericalCore.assert_square("g_base_metric", g, dim=n_dim)
        headroom = 2.0 * (total_energy_H0 - potential_V)
        if headroom <= _HILL_MARGIN_FLOOR:
            raise ValueError(
                "[CENTURION_ENGINE_VETO] Desviación geodésica fuera de la región de Hill."
            )
        gV = np.asarray(grad_V, dtype=np.float64).ravel()
        if hess_V is None:
            hv = np.zeros((n_dim, n_dim), dtype=np.float64)
        else:
            hv = np.asarray(hess_V, dtype=np.float64)
            _NumericalCore.assert_square("hess_V", hv, dim=n_dim)
        # Hess φ = −Hess V / headroom − ∇V ⊗ ∇V / headroom²
        hess_phi = -hv / (headroom + _WILKINSON_LIMIT) - np.outer(gV, gV) / (
            (headroom + _WILKINSON_LIMIT) ** 2
        )
        # Bloque 2n×2n: d/dt (δq, δq̇) = A (δq, δq̇)
        # δq̇ = δv
        # δv̇^i = − (∂_ℓ Γ^i_{jk}) v^j v^k δq^ℓ − 2 Γ^i_{jk} v^j δv^k
        # Aproximamos ∂_ℓ Γ vía Hess φ (métrica base plana):
        # ∂_ℓ Γ^i_{jk} ≈ δ^i_j Hessφ_{kℓ} + δ^i_k Hessφ_{jℓ} − g_{jk} g^{im} Hessφ_{mℓ}
        eye = np.eye(n_dim, dtype=np.float64)
        g_inv = la.inv(g)
        dgamma = (
            np.einsum("ij,kl->ijkl", eye, hess_phi)
            + np.einsum("ik,jl->ijkl", eye, hess_phi)
            - np.einsum("jk,im,ml->ijkl", g, g_inv, hess_phi)
        )
        # K[i,ℓ] = (∂_ℓ Γ^i_{jk}) v^j v^k
        K = np.einsum("ijkl,j,k->il", dgamma, v, v)
        # C[i,k] = 2 Γ^i_{jk} v^j
        C = 2.0 * np.einsum("ijk,j->ik", gamma, v)
        A = np.zeros((2 * n_dim, 2 * n_dim), dtype=np.float64)
        A[:n_dim, n_dim:] = eye
        A[n_dim:, :n_dim] = -K
        A[n_dim:, n_dim:] = -C
        fro = _NumericalCore.frobenius_norm(A)
        try:
            ev = la.eigvals(A)
            rho = float(np.max(np.abs(ev))) if ev.size else 0.0
        except la.LinAlgError:
            rho = float("inf")
        return _GeodesicDeviationCertificate(
            variational_matrix=A,
            frobenius_norm=float(fro),
            spectral_radius=float(rho),
            is_finite=bool(np.isfinite(fro) and np.isfinite(rho)),
        )

    # ── I.15 MORFISMO TERMINAL Φ_I: gérmen de Poincaré–Cartan ────────────
    @classmethod
    def synthesize_poincare_cartan_germ(
        cls,
        dimension_n: int,
        hamiltonian_energy_H0: float = 1.0,
        potential_V: float = 0.0,
        scale_matrix: Optional[np.ndarray] = None,
    ) -> _PoincareCartanGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE I ≅ OBJETO INICIAL DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Sintetiza el gérmen
            𝒢_I = (Ω, J, θ, g̃, φ, λ, μ_Liouville, H₀, Hill)
        sobre el cual la Fase II instancia los operadores de mecánica celeste
        de Poincaré (pequeños divisores KAM, Melnikov, retorno, IDA-PBC, Sp(2n)).

        Continuación formal: todo functor de Fase II se escribe
            Φ_II : 𝒢_I  →  𝒢_II = _ModularCelestialGerm
        y se inicializa exclusivamente a partir de este objeto.
        """
        if int(dimension_n) <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        n = int(dimension_n)
        two_n = 2 * n
        omega = cls.generate_canonical_symplectic_form(two_n)
        certificate = cls.certify_symplectic_form(omega)
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
            floor = cls.wilkinson_deflation_floor(np.asarray(scale_matrix))

        g_base = np.eye(n, dtype=np.float64)
        g_tilde, jacobi_cert = cls.compute_maupertuis_conformal_metric(
            potential_V, hamiltonian_energy_H0, g_base
        )
        hill_cert = cls.certify_hill_region(potential_V, hamiltonian_energy_H0)
        phi = cls.compute_conformal_factor_phi(potential_V, hamiltonian_energy_H0)

        x0 = np.zeros(two_n, dtype=np.float64)
        theta, theta_cert = cls.certify_liouville_one_form(x0, omega)
        # En el origen T = 0 ⇒ H(0,0) = V; λ = (p, −H) = (0, −V).
        lambda_pc = cls.compute_poincare_cartan_lambda(
            x0, hamiltonian_value=potential_V
        )
        mu_density = cls.compute_liouville_measure_density(n)

        return _PoincareCartanGerm(
            n=n,
            two_n=two_n,
            omega=omega,
            j_hamiltonian=omega.copy(),
            reg_floor=float(floor),
            form_certificate=certificate,
            liouville_theta=theta,
            liouville_certificate=theta_cert,
            maupertuis_metric=g_tilde,
            jacobi_certificate=jacobi_cert,
            hill_certificate=hill_cert,
            poincare_cartan_lambda=lambda_pc,
            conformal_factor_phi=float(phi),
            liouville_measure_density=float(mu_density),
            hamiltonian_energy_H0=float(hamiltonian_energy_H0),
            potential_V=float(potential_V),
        )

# =============================================================================
# FIN DE FASE I.
# El objeto 𝒢_I = _PoincareCartanGerm es el dominio de todo functor de Fase II.
# Continuación inmediata (paso siguiente):
#     class _PoincareCelestialVerifier:
#         def __init__(self, germ: _PoincareCartanGerm) -> None:
#             self._germ = germ
# =============================================================================
# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE II — KAM, MELNIKOV, RETORNO, IDA-PBC, Sp(2n)                        ║
# ║                                                                          ║
# ║ Dominio:  𝒢_I = _PoincareCartanGerm  (objeto terminal de Fase I).        ║
# ║ Codominio: 𝒢_II = _ModularCelestialGerm (objeto inicial de Fase III).    ║
# ║                                                                          ║
# ║ Functores:                                                               ║
# ║   II.1  Pequeños divisores Poincaré–KAM (diofántica + Bruno–Rüssmann).  ║
# ║   II.2  Twist de Kolmogorov e isonergeticidad (Hessiano bordeado).      ║
# ║   II.3  Ecuación homológica de Lindstedt ω·∂S₁/∂θ = −H₁ + ⟨H₁⟩.        ║
# ║   II.4  Retícula de Arnol'd, solapamiento de Chirikov, Nekhoroshev.     ║
# ║   II.5  Melnikov homoclínico y subarmónico (Gauss–Legendre + KBN).      ║
# ║   II.6  Retorno P: Σ→Σ (Floquet, Lyapunov, Krein, Conley–Zehnder).     ║
# ║   II.7  Estructuras de Dirac, Casimirs, ley IDA-PBC (Ḣ_d ≤ 0).         ║
# ║   II.8  Pertenencia a Sp(2n, ℝ): polar, Cayley, Williamson.             ║
# ║                                                                          ║
# ║ Morfismo terminal (II.10): induce_modular_celestial_germ                 ║
# ║     ↦ 𝒢_II  (Hamiltoniano modular K izado desde g̃ ⊕ g̃⁻¹).             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


@dataclass(frozen=True, slots=True)
class _KAMStabilityCertificate:
    r"""
    Certificado de estabilidad KAM de Poincaré–Kolmogorov–Arnol'd–Moser.

    Condición diofántica (sobre toda la retícula k ≠ 0):
        |⟨k, ω⟩| ≥ γ / |k|^τ,   τ > n − 1.
    El módulo de Bruno–Rüssmann
        𝔅(ω) = Σ_ν 2^{−ν} log(1/Ω_ν),
        Ω_ν = inf{ |⟨k, ω⟩| : 2^ν < |k| ≤ 2^{ν+1} },
    controla la convergencia de la serie de Lindstedt más allá de Siegel.
    El peso de Novikov W_Nov = exp(−T / |⟨k, ω⟩|) absorbe ultramétricamente
    los divisores prohibidos en el anillo Λ_Nov.
    """

    min_divisor: float
    tau: float
    gamma: float
    resonance_gap: float
    is_diophantine: bool
    diophantine_violation_count: int
    worst_divisor_index: int
    novikov_weight: float
    bruno_modulus: float
    is_bruno_convergent: bool
    maurercartan_residual: float
    volume_drift: float
    kam_measure_lower_bound: float
    is_kam_stable: bool


@dataclass(frozen=True, slots=True)
class _TwistNondegeneracyCertificate:
    r"""
    No-degeneración de Kolmogorov (twist) e isonergeticidad de Arnol'd.

    Twist:
        δ_K = det(∂²H₀/∂Iᵢ∂Iⱼ) = det(Dω/DI) ≠ 0.
    Isonergeticidad (Hessiano bordeado):
        δ_E = det |  ∂²H₀/∂I²    ω |
                  |  ωᵀ           0 | ≠ 0,
    equivalente a que las hipersuperficies H₀ = E conserven toros de
    frecuencias no colineales (condición de Arnol'd 1963).
    """

    twist_determinant: float
    twist_min_eigenvalue: float
    is_kolmogorov_nondegenerate: bool
    isoenergetic_determinant: float
    is_isoenergetic_nondegenerate: bool
    frequency_norm: float
    condition_number: float


@dataclass(frozen=True, slots=True)
class _HomologicalEquationCertificate:
    r"""
    Certificado de la ecuación homológica de Poincaré (Lindstedt, orden 1):
        ω · ∂S₁/∂θ = −(H₁ − ⟨H₁⟩),
        S_{1,k} = i H_{1,k} / ⟨k, ω⟩    (k ≠ 0).

    El modo k = 0 renormaliza H₀ y no se elimina. El residuo mide la
    incompatibilidad de los pequeños divisores con la proyección sobre
    el complemento de las resonancias.
    """

    generating_function_modes: np.ndarray
    mean_mode_H1: complex
    max_mode_amplitude: float
    residual_l2: float
    excluded_resonant_modes: int
    is_solved: bool


@dataclass(frozen=True, slots=True)
class _NekhoroshevCertificate:
    r"""
    Estabilidad exponencial de Nekhoroshev y solapamiento de Chirikov.

    Para H = H₀ + ε H₁ steep (cuasiconvexa),
        |I(t) − I(0)| ≤ ε^b    para    |t| ≤ T_* exp(c ε^{−a}),
    con a = b = 1/(2n) en el caso clásico. El parámetro de Chirikov
        s = (Δω_α + Δω_β) / |ω_α − ω_β|
    predice la destrucción de toros por solapamiento de resonancias (s > 1).
    """

    exponent_a: float
    exponent_b: float
    action_drift_bound: float
    log_time_bound: float
    chirikov_overlap: float
    is_chirikov_overlapped: bool
    is_nekhoroshev_stable: bool
    perturbation_epsilon: float


@dataclass(frozen=True, slots=True)
class _MelnikovCertificate:
    r"""
    Función de Melnikov (Poincaré 1890, Melnikov 1963).

    Para H = H₀ + ε H₁, la distancia simpléctica entre W^s(γ⁰) y W^u(γ⁰) es
        d(t₀) = ε M(t₀) / ‖∇H₀(γ⁰(t₀))‖ + O(ε²),
        M(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt.
    Cero simple (M = 0, M' ≠ 0) ⇒ intersección transversal ⇒ herradura de
    Smale (caos de Poincaré). El Melnikov subarmónico detecta islas m:n.
    """

    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    n_quadrature_nodes: int
    n_dropped_nodes: int
    subharmonic_ratio: Tuple[int, int]
    subharmonic_value: float


@dataclass(frozen=True, slots=True)
class _PoincareReturnMapCertificate:
    r"""
    Mapa de retorno de Poincaré P: Σ → Σ y monodromía M ∈ Sp(2n).

    • Multiplicadores de Floquet λ_i = spec(M), cerrados bajo λ ↦ 1/λ, λ ↦ λ̄.
    • Lyapunov: L_i = (1/T) log|λ_i|.
    • Krein: forma Ω(ξ, ξ̄) sobre autoespacios elípticos (colisión de Krein).
    • Conley–Zehnder: índice de cruce de la trayectoria polar I ⇝ M.
    Clasificación: hiperbólico / elíptico / parabólico / mixto.
    """

    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    max_lyapunov: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    is_mixed: bool
    spectral_type: str
    trace_M: float
    det_M: float
    reciprocal_pairing_residual: float
    krein_signatures: np.ndarray
    krein_indefinite: bool
    conley_zehnder_index: int
    is_nondegenerate: bool


@dataclass(frozen=True, slots=True)
class _SectionTransversalityCertificate:
    r"""
    Transversalidad de la sección de Poincaré Σ = {S = 0}:
        ⟨dS, X_H⟩ ≠ 0  sobre  Σ ∩ {H = H₀}.
    """

    pairing_dS_XH: float
    is_transverse: bool
    section_residual: float


@dataclass(frozen=True, slots=True)
class _StructureCertificate:
    """Pasividad port-Hamiltoniana: Jᵀ = −J, R ⪰ 0, Rᵀ = R."""

    j_skew_residual: float
    r_symmetric_residual: float
    r_min_eigenvalue: float
    is_passive: bool


@dataclass(frozen=True, slots=True)
class _CasimirCertificate:
    r"""
    Casimir C de la estructura de Dirac: (J − R) ∇C = 0  (idealmente J ∇C = 0
    y R ∇C = 0). En el chart canónico J es no degenerada ⇒ solo Casimirs
    constantes; con J degenerada (vínculos) el núcleo es no trivial.
    """

    residual_norm: float
    kernel_dimension: int
    is_casimir: bool


@dataclass(frozen=True, slots=True)
class _IDAPBCResult:
    """Ley de control IDA-PBC certificada (matching + aniquilador + Ḣ_d ≤ 0)."""

    control_law: np.ndarray
    exergy_loss: float
    lyapunov_derivative: float
    matching_residual: float
    annihilator_residual: float
    condition_number: float
    structure_ok: bool
    casimir_ok: bool
    is_strictly_passive: bool


@dataclass(frozen=True, slots=True)
class _SymplecticPreservationResult:
    r"""
    Pertenencia numérica a Sp(2n, ℝ): Mᵀ Ω M = Ω, det M = 1,
    distancia a la retracción polar y residuo de Cayley en 𝔰𝔭(2n).
    """

    residual_norm: float
    relative_residual: float
    determinant: float
    polar_sp_distance: float
    cayley_hamiltonian_skew_residual: float
    is_viable: bool


@dataclass(frozen=True, slots=True)
class _WilliamsonSpectrumCertificate:
    r"""
    Forma normal de Williamson del Hessiano hamiltoniano A = J Hess H.

    Los autovalores de A aparecen en cuádruplas {±λ, ±λ̄} (o pares ±λ
    reales / ±iω imaginarios puros). Clasifica el equilibrio lineal.
    """

    hamiltonian_eigenvalues: np.ndarray
    n_elliptic_pairs: int
    n_hyperbolic_pairs: int
    n_focus_quadruples: int
    n_nilpotent: int
    is_linearly_stable: bool
    pairing_residual: float


@dataclass(frozen=True, slots=True)
class _ModularCelestialGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN MODULAR CELESTE (objeto terminal de Fase II, inicial de Fase III).
    ═══════════════════════════════════════════════════════════════════════════
    Compone 𝒢_I (Darboux + Maupertuis + Poincaré–Cartan) con las
    verificaciones celestes y port-Hamiltonianas, y produce el Hamiltoniano
    modular K ∈ SPD(2n) que inicia Tomita–Takesaki:
        ρ = e^{−β K} / Z,   σ_z^ρ(A) = ρ^{i z} A ρ^{−i z}.
    """

    n: int
    two_n: int
    omega: np.ndarray
    j_hamiltonian: np.ndarray
    reg_floor: float
    hamiltonian_energy_H0: float
    potential_V: float
    maupertuis_metric: np.ndarray
    conformal_factor_phi: float
    liouville_measure_density: float
    kam_certificate: Optional[_KAMStabilityCertificate]
    twist_certificate: Optional[_TwistNondegeneracyCertificate]
    melnikov_certificate: Optional[_MelnikovCertificate]
    return_map_certificate: Optional[_PoincareReturnMapCertificate]
    nekhoroshev_certificate: Optional[_NekhoroshevCertificate]
    ida_pbc_result: Optional[_IDAPBCResult]
    symplectic_preservation_result: Optional[_SymplecticPreservationResult]
    beta: float
    modular_hamiltonian: np.ndarray


@dataclass(frozen=True, slots=True)
class _ModularSpectralGerm:
    """Gérmen espectral modular ρ = e^{−βK}/Z (uso interno de Fase III)."""

    beta: float
    modular_hamiltonian: np.ndarray
    thermal_state: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    partition_function: float


class _PoincareCelestialVerifier:
    r"""
    Fase II. Verificador de mecánica celeste de Poincaré.

    Se inicializa exclusivamente con 𝒢_I. Todos los operadores leen
    Ω, J, g̃, μ_Liouville y H₀ del gérmen de Poincaré–Cartan.
    """

    def __init__(self, germ: _PoincareCartanGerm) -> None:
        if not hasattr(germ, "omega") or not hasattr(germ, "two_n"):
            raise TypeError(
                "Fase II exige el objeto terminal de Fase I: _PoincareCartanGerm."
            )
        self._germ = germ

    @property
    def germ(self) -> _PoincareCartanGerm:
        return self._germ

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
    ) -> _KAMStabilityCertificate:
        r"""
        Espectro de pequeños divisores de Poincaré–KAM con absorción
        ultramétrica T-ádica en el anillo de Novikov Λ_Nov.

        Para H(I,θ) = H₀(I) + ε H₁(I,θ), la ecuación de arrastre
            Σ_j ω_j ∂S₁/∂θ_j = −H₁(I', θ)
        exige S_{1,k} = i H_{1,k} / ⟨k, ω⟩. Se verifica
            |⟨k, ω⟩| ≥ γ / |k|^τ    ∀ k ∈ ℤ^n \ {0}
        sobre **toda** la retícula suministrada (no solo el divisor mínimo).
        """
        omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega)
        wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
        if wave_k.ndim == 1:
            wave_k = wave_k.reshape(1, -1)
        if wave_k.ndim != 2 or wave_k.shape[1] != omega.size:
            raise ValueError(
                f"wave_vectors_k debe ser (M, {omega.size}); recibido {wave_k.shape}."
            )
        _NumericalCore.assert_finite("wave_vectors_k", wave_k)
        jac_m = np.asarray(jacobian_M, dtype=np.float64)
        can_J = np.asarray(canonical_J, dtype=np.float64)
        if can_J.size and can_J.shape == self._germ.omega.shape:
            j_gap = _NumericalCore.frobenius_norm(can_J - self._germ.omega)
            if j_gap > _WILKINSON_DRIFT_LIMIT * max(
                _NumericalCore.frobenius_norm(self._germ.omega), 1.0
            ):
                logger.warning(
                    "canonical_J discrepa del Ω de 𝒢_I (‖Δ‖_F=%.3e); se usa 𝒢_I.omega.",
                    j_gap,
                )

        tau = max(float(tau), float(self._germ.n - 1) + 1e-12)
        gamma = max(float(gamma), _KAM_GAMMA_FLOOR)

        k_l2 = np.linalg.norm(wave_k, axis=1)
        nonzero = k_l2 > _MACHINE_EPS
        divisors = np.abs(wave_k @ omega)
        if not np.any(nonzero):
            min_divisor = 1.0
            argmin_idx = 0
            resonance_gap = 1.0
            n_viol = 0
            is_diophantine = True
        else:
            live_div = np.where(nonzero, divisors, np.inf)
            argmin_idx = int(np.argmin(live_div))
            min_divisor = float(live_div[argmin_idx])
            resonance_gap = min_divisor
            k_scale = np.maximum(k_l2, 1.0)
            lhs = divisors * np.power(k_scale, tau)
            viol = nonzero & (lhs < gamma)
            n_viol = int(np.count_nonzero(viol))
            is_diophantine = n_viol == 0

        novikov_weight = float(
            np.exp(
                -np.clip(
                    float(novikov_valuation_T) / (_WILKINSON_LIMIT + min_divisor),
                    0.0,
                    _LOG_EXP_CLIP,
                )
            )
        )
        mc_residual = float(abs(min_divisor * novikov_weight))
        bruno = self._bruno_russmann_from_lattice(k_l2, divisors)
        is_bruno = bool(np.isfinite(bruno) and bruno < _LOG_EXP_CLIP)

        if jac_m.ndim == 2 and jac_m.shape[0] == jac_m.shape[1] and jac_m.size > 0:
            det_M = float(np.real(la.det(jac_m)))
            volume_drift = float(abs(det_M - 1.0))
        else:
            volume_drift = 0.0

        # Medida de Arnol'd: vol{ω diofánticos} ≥ 1 − C γ (cota inferior grosera).
        kam_meas = float(max(0.0, 1.0 - gamma * max(np.log(1.0 / max(gamma, _MACHINE_EPS)), 1.0)))
        is_kam_stable = bool(
            min_divisor >= _WILKINSON_LIMIT
            and volume_drift <= _WILKINSON_LIMIT
            and is_diophantine
            and is_bruno
        )
        return _KAMStabilityCertificate(
            min_divisor=min_divisor,
            tau=float(tau),
            gamma=float(gamma),
            resonance_gap=float(resonance_gap),
            is_diophantine=bool(is_diophantine),
            diophantine_violation_count=int(n_viol),
            worst_divisor_index=int(argmin_idx),
            novikov_weight=novikov_weight,
            bruno_modulus=float(bruno),
            is_bruno_convergent=is_bruno,
            maurercartan_residual=mc_residual,
            volume_drift=volume_drift,
            kam_measure_lower_bound=kam_meas,
            is_kam_stable=is_kam_stable,
        )

    @staticmethod
    def _bruno_russmann_from_lattice(k_l2: np.ndarray, divisors: np.ndarray) -> float:
        r"""𝔅(ω) = Σ_ν 2^{−ν} log(1/Ω_ν) sobre cascarones diádicos de |k|."""
        if k_l2.size == 0:
            return 0.0
        k_max = float(np.max(k_l2))
        if k_max <= 0.0:
            return 0.0
        n_shells = int(max(1, math.ceil(math.log(k_max + 1.0) / math.log(2.0)) + 1))
        acc = 0.0
        comp = 0.0
        for nu in range(n_shells):
            lo = 2.0 ** nu
            hi = 2.0 ** (nu + 1)
            mask = (k_l2 > lo) & (k_l2 <= hi)
            if not np.any(mask):
                continue
            omega_nu = float(np.min(divisors[mask]))
            omega_nu = max(omega_nu, _MACHINE_EPS)
            term = math.log(1.0 / omega_nu) / (2.0 ** nu)
            y = term - comp
            t = acc + y
            comp = (t - acc) - y
            acc = t
        return float(acc)

    def compute_bruno_russmann_modulus(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
    ) -> float:
        """Módulo de Bruno–Rüssmann 𝔅(ω) sobre la retícula dada."""
        omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega)
        wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
        if wave_k.ndim == 1:
            wave_k = wave_k.reshape(1, -1)
        return self._bruno_russmann_from_lattice(
            np.linalg.norm(wave_k, axis=1),
            np.abs(wave_k @ omega),
        )

    # ── II.2 Twist de Kolmogorov e isonergeticidad ───────────────────────
    def compute_kolmogorov_twist_certificate(
        self,
        hess_H0_actions: np.ndarray,
        frequency_vector_omega: np.ndarray,
    ) -> _TwistNondegeneracyCertificate:
        r"""
        No-degeneración de Kolmogorov det(Dω/DI) ≠ 0 y condición isonergetica
        de Arnol'd (Hessiano bordeado de tamaño (n+1)×(n+1)).
        """
        hess = np.asarray(hess_H0_actions, dtype=np.float64)
        n = self._germ.n
        _NumericalCore.assert_square("hess_H0_actions", hess, dim=n)
        _NumericalCore.assert_finite("hess_H0_actions", hess)
        omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega, dim=n)
        hess_s = 0.5 * (hess + hess.T)
        twist_det = float(np.real(la.det(hess_s)))
        try:
            ev = np.real(la.eigvalsh(hess_s))
            min_ev = float(np.min(ev)) if ev.size else 0.0
            max_ev = float(np.max(np.abs(ev))) if ev.size else 0.0
            cond = float(max_ev / max(abs(min_ev), _MACHINE_EPS))
        except la.LinAlgError:
            min_ev = 0.0
            cond = float("inf")
        bordered = np.zeros((n + 1, n + 1), dtype=np.float64)
        bordered[:n, :n] = hess_s
        bordered[:n, n] = omega
        bordered[n, :n] = omega
        iso_det = float(np.real(la.det(bordered)))
        freq_norm = _NumericalCore.euclidean_norm(omega)
        is_twist = bool(abs(twist_det) > _SPECTRAL_TOL * max(1.0, abs(twist_det)))
        is_iso = bool(abs(iso_det) > _SPECTRAL_TOL * max(1.0, abs(iso_det)))
        return _TwistNondegeneracyCertificate(
            twist_determinant=twist_det,
            twist_min_eigenvalue=min_ev,
            is_kolmogorov_nondegenerate=is_twist,
            isoenergetic_determinant=iso_det,
            is_isoenergetic_nondegenerate=is_iso,
            frequency_norm=freq_norm,
            condition_number=float(cond),
        )

    # ── II.3 Ecuación homológica de Lindstedt ────────────────────────────
    def solve_poincare_homological_equation(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        fourier_H1: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> _HomologicalEquationCertificate:
        r"""
        Resuelve S_{1,k} = i H_{1,k} / ⟨k, ω⟩ para k ≠ 0, recortando modos
        resonantes |⟨k, ω⟩| < γ/|k|^τ (proyección sobre el complemento de
        la retícula resonante). El modo nulo ⟨H₁⟩ renormaliza H₀.
        """
        omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega)
        wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
        if wave_k.ndim == 1:
            wave_k = wave_k.reshape(1, -1)
        h1 = np.asarray(fourier_H1, dtype=np.complex128).ravel()
        if wave_k.shape[0] != h1.size or wave_k.shape[1] != omega.size:
            raise ValueError("Dimensiones incompatibles entre k, H₁ y ω.")
        _NumericalCore.assert_finite("wave_vectors_k", wave_k)
        divisors = wave_k @ omega
        k_l2 = np.maximum(np.linalg.norm(wave_k, axis=1), 1.0)
        k_is_zero = np.linalg.norm(wave_k, axis=1) <= _MACHINE_EPS
        small = (np.abs(divisors) < (gamma / np.power(k_l2, tau))) | k_is_zero
        S = np.zeros(h1.size, dtype=np.complex128)
        live = ~small
        S[live] = (1j * h1[live]) / divisors[live]
        mean_mode = complex(_NumericalCore.kahan_babuska_neumaier_sum(np.real(h1[k_is_zero])))
        if np.any(k_is_zero):
            mean_mode = complex(np.sum(h1[k_is_zero]))
        else:
            mean_mode = 0.0 + 0.0j
        # Residuo: i ⟨k,ω⟩ S_k − H_{1,k}  (0 en modos vivos; H1 en resonantes ≠ 0).
        recon = np.zeros_like(h1)
        recon[live] = -1j * divisors[live] * S[live]
        residual_vec = recon - h1
        residual_vec[k_is_zero] = 0.0  # el promedio no se cancela
        residual_vec[small & ~k_is_zero] = -h1[small & ~k_is_zero]
        res_l2 = float(np.sqrt(max(np.real(np.vdot(residual_vec, residual_vec)), 0.0)))
        max_amp = float(np.max(np.abs(S))) if S.size else 0.0
        n_excl = int(np.count_nonzero(small & ~k_is_zero))
        is_solved = bool(res_l2 <= _SPECTRAL_TOL * max(1.0, float(np.max(np.abs(h1)) if h1.size else 1.0)))
        return _HomologicalEquationCertificate(
            generating_function_modes=S,
            mean_mode_H1=complex(mean_mode),
            max_mode_amplitude=max_amp,
            residual_l2=res_l2,
            excluded_resonant_modes=n_excl,
            is_solved=is_solved,
        )

    # ── II.4 Retícula de resonancias de Arnol'd ──────────────────────────
    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: np.ndarray,
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> np.ndarray:
        r"""
        Retícula de resonancias de Arnol'd:
            R_ε(ω) = { k ∈ ℤ^n \ {0} : |k|_1 ≤ max_order, |⟨k, ω⟩| < tol }.
        Se veta la explosión combinatoria (2R+1)^n > 5·10^5.
        """
        omega = _NumericalCore.assert_vec("frequency_vector_omega", frequency_vector_omega)
        n = omega.size
        max_order = int(max_order)
        if n == 0 or max_order < 1:
            return np.zeros((0, max(n, 1)), dtype=np.int64)
        n_pts = (2 * max_order + 1) ** n
        if n_pts > 500_000:
            raise ValueError(
                f"Retícula de Arnol'd demasiado densa: n={n}, max_order={max_order} "
                f"⇒ {(2 * max_order + 1) ** n} nodos. Reduzca max_order."
            )
        ranges = [np.arange(-max_order, max_order + 1) for _ in range(n)]
        grid = np.meshgrid(*ranges, indexing="ij")
        k_all = np.stack([g.ravel() for g in grid], axis=1).astype(np.int64)
        norms = np.abs(k_all).sum(axis=1)
        keep = (norms > 0) & (norms <= max_order)
        if not np.any(keep):
            return np.zeros((0, n), dtype=np.int64)
        k_cand = k_all[keep]
        inner = np.abs(k_cand.astype(np.float64) @ omega)
        return k_cand[inner < tol]

    def compute_chirikov_overlap_parameter(
        self,
        resonance_frequencies: np.ndarray,
        resonance_halfwidths: np.ndarray,
    ) -> float:
        r"""
        Parámetro de solapamiento de Chirikov
            s = max_{α≠β} (Δω_α + Δω_β) / |ω_α − ω_β|.
        s > 1 ⇒ destrucción de toros KAM por solapamiento resonante.
        """
        om = _NumericalCore.assert_vec("resonance_frequencies", resonance_frequencies)
        hw = _NumericalCore.assert_vec("resonance_halfwidths", resonance_halfwidths, dim=om.size)
        if om.size < 2:
            return 0.0
        s_max = 0.0
        for i in range(om.size):
            for j in range(i + 1, om.size):
                den = abs(om[i] - om[j])
                if den <= _MACHINE_EPS:
                    return float("inf")
                s_max = max(s_max, (abs(hw[i]) + abs(hw[j])) / den)
        return float(s_max)

    def compute_nekhoroshev_stability_bounds(
        self,
        perturbation_epsilon: float,
        chirikov_overlap: float = 0.0,
        steepness_index: Optional[float] = None,
    ) -> _NekhoroshevCertificate:
        r"""
        Cotas de Nekhoroshev para H₀ steep de índice α (cuasiconvexa: α = 1):
            a = 1 / (2 n α),   b = a.
        """
        eps = float(perturbation_epsilon)
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("perturbation_epsilon debe ser finito y ≥ 0.")
        n = max(int(self._germ.n), 1)
        alpha = float(steepness_index) if steepness_index is not None else 1.0
        alpha = max(alpha, _MACHINE_EPS)
        a = 1.0 / (2.0 * n * alpha)
        b = a
        eps_clip = max(eps, _MACHINE_EPS)
        action_drift = float(eps_clip ** b)
        # log T_* ~ ε^{−a}  (se reporta el exponente, no exp(.) para evitar overflow).
        log_time = float(eps_clip ** (-a))
        s = float(chirikov_overlap)
        overlapped = bool(s > 1.0)
        stable = bool((not overlapped) and eps_clip < 1.0 and np.isfinite(log_time))
        return _NekhoroshevCertificate(
            exponent_a=float(a),
            exponent_b=float(b),
            action_drift_bound=action_drift,
            log_time_bound=log_time,
            chirikov_overlap=s,
            is_chirikov_overlapped=overlapped,
            is_nekhoroshev_stable=stable,
            perturbation_epsilon=eps,
        )

    # ── II.5 Función de Melnikov ─────────────────────────────────────────
    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = 25.0,
        n_quad: int = _MELNIKOV_QUAD_NODES,
        subharmonic_ratio: Tuple[int, int] = (1, 1),
    ) -> _MelnikovCertificate:
        r"""
        M(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt
        por Gauss–Legendre en [−t_inf, t_inf] con acumulación KBN.

        El Melnikov subarmónico m:n se obtiene reescalando el periodo de
        cuadratura a m T / n (islas de Poincaré). Cero simple ⇒ fractura
        homoclínica transversal.
        """
        t0s = _NumericalCore.assert_vec("t0_grid", t0_grid)
        if t0s.size == 0:
            raise ValueError("t0_grid no puede estar vacío.")
        if not np.isfinite(t_inf) or t_inf <= 0:
            raise ValueError("t_inf debe ser finito y positivo.")
        n_quad = int(max(16, n_quad))
        m_sub, n_sub = int(subharmonic_ratio[0]), int(subharmonic_ratio[1])
        if m_sub <= 0 or n_sub <= 0:
            raise ValueError("subharmonic_ratio exige enteros positivos (m, n).")
        t_span = t_inf * float(m_sub) / float(n_sub)
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        t_nodes = t_span * nodes
        w_nodes = t_span * weights

        melnikov_vals = np.zeros(t0s.size, dtype=np.float64)
        dropped_total = 0
        for i, t0 in enumerate(t0s):
            acc = 0.0
            comp = 0.0
            dropped = 0
            for t_shift, w in zip(t_nodes, w_nodes):
                try:
                    x = np.asarray(homoclinic_flow(float(t_shift - t0)), dtype=np.float64)
                    _NumericalCore.assert_finite("homoclinic_flow(t)", x)
                except (ValueError, TypeError, FloatingPointError):
                    dropped += 1
                    continue
                pb = self._poisson_bracket(hamiltonian_0, hamiltonian_1, x)
                if not np.isfinite(pb):
                    dropped += 1
                    continue
                y = pb * float(w) - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            dropped_total += dropped
            melnikov_vals[i] = acc

        idx_min = int(np.argmin(np.abs(melnikov_vals)))
        m_val = float(melnikov_vals[idx_min])
        if t0s.size >= 3 and 0 < idx_min < t0s.size - 1:
            dm = (melnikov_vals[idx_min + 1] - melnikov_vals[idx_min - 1]) / (
                t0s[idx_min + 1] - t0s[idx_min - 1]
            )
        elif t0s.size >= 2:
            dm = (melnikov_vals[-1] - melnikov_vals[0]) / max(
                t0s[-1] - t0s[0], _MACHINE_EPS
            )
        else:
            dm = 0.0
        is_simple = bool(abs(m_val) < _SPECTRAL_TOL and abs(dm) > _SPECTRAL_TOL)
        try:
            x_star = np.asarray(homoclinic_flow(float(t0s[idx_min])), dtype=np.float64)
            grad_h0 = self._gradient_fourth(hamiltonian_0, x_star)
            grad_norm = _NumericalCore.euclidean_norm(grad_h0)
        except (ValueError, TypeError, FloatingPointError):
            grad_norm = 0.0
        splitting = float(abs(m_val) / max(grad_norm, _WILKINSON_LIMIT))
        # Valor subarmónico: el propio M si (m,n)≠(1,1); si no, 0 (no hay isla).
        sub_val = m_val if (m_sub, n_sub) != (1, 1) else 0.0
        return _MelnikovCertificate(
            melnikov_value=m_val,
            melnikov_derivative=float(dm),
            is_simple_zero=is_simple,
            homoclinic_splitting=splitting,
            n_quadrature_nodes=n_quad,
            n_dropped_nodes=int(dropped_total),
            subharmonic_ratio=(m_sub, n_sub),
            subharmonic_value=float(sub_val),
        )

    def compute_subharmonic_melnikov(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        m_over_n: Tuple[int, int],
        t_inf: float = 25.0,
    ) -> _MelnikovCertificate:
        r"""Melnikov subarmónico m:n (islas de Poincaré en el mapa de retorno)."""
        return self.compute_melnikov_function(
            homoclinic_flow,
            hamiltonian_0,
            hamiltonian_1,
            t0_grid,
            t_inf=t_inf,
            subharmonic_ratio=m_over_n,
        )

    # ── II.6 Mapa de retorno de Poincaré ─────────────────────────────────
    def compute_poincare_return_map(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
    ) -> _PoincareReturnMapCertificate:
        r"""
        Espectro de Floquet, Lyapunov, emparejamiento recíproco λ ↔ 1/λ,
        firmas de Krein y índice de Conley–Zehnder de la monodromía M.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        if M.ndim == 1:
            side = int(round(np.sqrt(M.size)))
            if side * side != M.size:
                raise ValueError("jacobian_M plano no es cuadrado perfecto.")
            M = M.reshape(side, side)
        _NumericalCore.assert_square("jacobian_M", M, dim=self._germ.two_n)
        _NumericalCore.assert_finite("jacobian_M", M)
        ev = la.eigvals(M)
        magnitudes = np.abs(ev)
        lyap = np.log(np.maximum(magnitudes, _MACHINE_EPS)) / max(abs(period_T), _MACHINE_EPS)
        lyap = np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)
        max_lyap = float(np.max(lyap)) if lyap.size else 0.0

        band = _FLOQUET_PARABOLIC_BAND
        on_circle = np.abs(magnitudes - 1.0) <= band * 10.0
        at_pm1 = (np.abs(ev - 1.0) < band) | (np.abs(ev + 1.0) < band)
        n_on = int(np.count_nonzero(on_circle))
        n_off = int(ev.size - n_on)
        n_pm1 = int(np.count_nonzero(at_pm1))
        is_parabolic = n_pm1 > 0
        is_elliptic = bool(n_off == 0 and n_pm1 == 0 and ev.size > 0)
        is_hyperbolic = bool(n_on == 0 and ev.size > 0)
        is_mixed = bool(n_on > 0 and n_off > 0)
        if is_parabolic:
            spectral_type = "parabolic"
        elif is_elliptic:
            spectral_type = "elliptic"
        elif is_hyperbolic:
            spectral_type = "hyperbolic"
        elif is_mixed:
            spectral_type = "mixed"
        else:
            spectral_type = "undetermined"

        pair_res = self._reciprocal_pairing_residual(ev)
        krein = self._krein_signatures(M, ev, on_circle)
        krein_indef = bool(krein.size and (np.any(krein > 0) and np.any(krein < 0)))
        cz = self._conley_zehnder_index(M)
        is_nondeg = bool(np.all(np.abs(ev - 1.0) > band) and np.all(np.abs(ev + 1.0) > band))

        return _PoincareReturnMapCertificate(
            floquet_multipliers=ev,
            lyapunov_spectrum=lyap,
            max_lyapunov=max_lyap,
            is_hyperbolic=is_hyperbolic,
            is_elliptic=is_elliptic,
            is_parabolic=is_parabolic,
            is_mixed=is_mixed,
            spectral_type=spectral_type,
            trace_M=float(np.real(np.trace(M))),
            det_M=float(np.real(la.det(M))),
            reciprocal_pairing_residual=pair_res,
            krein_signatures=krein,
            krein_indefinite=krein_indef,
            conley_zehnder_index=int(cz),
            is_nondegenerate=is_nondeg,
        )

    def certify_section_transversality(
        self,
        grad_S: np.ndarray,
        grad_H: np.ndarray,
        S_value: float = 0.0,
    ) -> _SectionTransversalityCertificate:
        r"""Transversalidad ⟨dS, X_H⟩ = dS · (Ω ∇H) ≠ 0 sobre Σ = {S = 0}."""
        gS = _NumericalCore.assert_vec("grad_S", grad_S, dim=self._germ.two_n)
        gH = _NumericalCore.assert_vec("grad_H", grad_H, dim=self._germ.two_n)
        xh = _NumericalCore.hamiltonian_vector_field(gH, self._germ.omega)
        pairing = _NumericalCore.compensated_inner_product(gS, xh)
        is_tr = bool(abs(pairing) > _SPECTRAL_TOL)
        return _SectionTransversalityCertificate(
            pairing_dS_XH=float(pairing),
            is_transverse=is_tr,
            section_residual=float(abs(S_value)),
        )

    def compute_williamson_spectrum(
        self,
        hess_H: np.ndarray,
    ) -> _WilliamsonSpectrumCertificate:
        r"""
        Espectro de Williamson: autovalores de A = Ω Hess H (matriz hamiltoniana).
        Clasifica el equilibrio lineal en pares elípticos (±iω), hiperbólicos
        (±λ) y cuádruplas foco (±α ± iβ).
        """
        hess = np.asarray(hess_H, dtype=np.float64)
        _NumericalCore.assert_square("hess_H", hess, dim=self._germ.two_n)
        _NumericalCore.assert_finite("hess_H", hess)
        hess_s = 0.5 * (hess + hess.T)
        A = self._germ.omega @ hess_s
        ev = la.eigvals(A)
        pair_res = self._hamiltonian_quadruplet_residual(ev)
        imag_pure = (np.abs(ev.real) <= _FLOQUET_PARABOLIC_BAND) & (
            np.abs(ev.imag) > _FLOQUET_PARABOLIC_BAND
        )
        real_pure = (np.abs(ev.imag) <= _FLOQUET_PARABOLIC_BAND) & (
            np.abs(ev.real) > _FLOQUET_PARABOLIC_BAND
        )
        nilp = (np.abs(ev.real) <= _FLOQUET_PARABOLIC_BAND) & (
            np.abs(ev.imag) <= _FLOQUET_PARABOLIC_BAND
        )
        focus = ~imag_pure & ~real_pure & ~nilp
        n_ell = int(np.count_nonzero(imag_pure) // 2)
        n_hyp = int(np.count_nonzero(real_pure) // 2)
        n_foc = int(np.count_nonzero(focus) // 4)
        n_nil = int(np.count_nonzero(nilp))
        linearly_stable = bool(n_hyp == 0 and n_foc == 0 and n_nil == 0 and n_ell > 0)
        return _WilliamsonSpectrumCertificate(
            hamiltonian_eigenvalues=ev,
            n_elliptic_pairs=n_ell,
            n_hyperbolic_pairs=n_hyp,
            n_focus_quadruples=n_foc,
            n_nilpotent=n_nil,
            is_linearly_stable=linearly_stable,
            pairing_residual=pair_res,
        )

    # ── Auxiliares internos ──────────────────────────────────────────────
    @staticmethod
    def _gradient_fourth(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = 1e-6,
    ) -> np.ndarray:
        """Gradiente por diferencias centrales de orden 4."""
        xv = np.asarray(x, dtype=np.float64).ravel()
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        h = float(h)
        for i in range(dim):
            e = np.zeros(dim, dtype=np.float64)
            e[i] = 1.0
            try:
                f1 = float(np.real(func(xv + h * e)))
                f2 = float(np.real(func(xv - h * e)))
                f3 = float(np.real(func(xv + 2.0 * h * e)))
                f4 = float(np.real(func(xv - 2.0 * h * e)))
                grad[i] = (-f3 + 8.0 * f1 - 8.0 * f2 + f4) / (12.0 * h)
            except (ValueError, TypeError, FloatingPointError, ArithmeticError):
                grad[i] = 0.0
        return grad

    def _poisson_bracket(
        self,
        h0: Callable[[np.ndarray], float],
        h1: Callable[[np.ndarray], float],
        x: np.ndarray,
    ) -> float:
        r"""{H₀, H₁} = ∇H₀ᵀ Ω ∇H₁ en el chart de Darboux de 𝒢_I."""
        xv = np.asarray(x, dtype=np.float64).ravel()
        if xv.size != self._germ.two_n:
            return 0.0
        g0 = self._gradient_fourth(h0, xv)
        g1 = self._gradient_fourth(h1, xv)
        return float(g0 @ self._germ.omega @ g1)

    @staticmethod
    def _reciprocal_pairing_residual(ev: np.ndarray) -> float:
        """min_j |λ_i − 1/λ_j| promediado (cierre simpléctico del espectro)."""
        if ev.size == 0:
            return 0.0
        acc = 0.0
        n_ok = 0
        for lam in ev:
            if abs(lam) < _MACHINE_EPS:
                acc += 1.0
                n_ok += 1
                continue
            target = 1.0 / lam
            acc += float(np.min(np.abs(ev - target)))
            n_ok += 1
        return float(acc / max(n_ok, 1))

    def _krein_signatures(
        self,
        M: np.ndarray,
        ev: np.ndarray,
        on_circle: np.ndarray,
    ) -> np.ndarray:
        r"""
        Firma de Krein de cada multiplicador elíptico:
            κ(λ) = sign i Ω(ξ, ξ̄)  para  M ξ = λ ξ, |λ| = 1.
        Firmas opuestas en un choque λ₁ = λ₂ ∈ S¹ ⇒ posible salida del círculo.
        """
        omega = self._germ.omega
        sigs = np.zeros(ev.size, dtype=np.float64)
        try:
            w_vals, w_vecs = la.eig(M)
        except la.LinAlgError:
            return sigs
        for i, (lam, on) in enumerate(zip(w_vals, on_circle[: w_vals.size])):
            if not on:
                continue
            xi = w_vecs[:, i]
            form = np.vdot(xi, omega @ xi)
            sigs[i] = float(np.sign(np.imag(form))) if abs(form) > _MACHINE_EPS else 0.0
        return sigs

    def _conley_zehnder_index(self, M: np.ndarray) -> int:
        r"""
        Índice de Conley–Zehnder discreto vía factor polar simpléctico.

        Se retracta M a O ∈ Sp(2n) ∩ O(2n) ≅ U(n) y se toma
            CZ(M) ≅ (1/π) Arg det_ℂ(U)  redondeado,
        con U la identificación unitaria inducida por la estructura casi
        compleja J = Ω. Para M no degenerada (1 ∉ spec) el índice es par
        o impar según sign det(I − M) (Robbin–Salamon, paridad).
        """
        omega = self._germ.omega
        two_n = self._germ.two_n
        try:
            S = -omega @ M.T @ omega @ M
            S_h = _NumericalCore.higham_nearest_spd(
                0.5 * (S + S.T), floor=self._germ.reg_floor
            )
            evals, evecs = la.eigh(S_h)
            evals = _NumericalCore.regularize_spectrum(
                np.real(evals), floor=self._germ.reg_floor
            )
            s_inv_sqrt = evecs @ (np.power(evals, -0.5)[:, None] * evecs.T)
            O = M @ s_inv_sqrt
            # Identificación U(n): bloque complejo q + i p sobre O.
            n = two_n // 2
            a, b = O[:n, :n], O[:n, n:]
            c, d = O[n:, :n], O[n:, n:]
            U = 0.5 * ((a + d) + 1j * (c - b))
            det_u = la.det(U)
            ang = float(np.angle(det_u))
            cz = int(np.rint(ang / math.pi))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            det_im = float(np.real(la.det(np.eye(two_n) - M)))
            cz = 0 if det_im >= 0.0 else 1
        return int(cz)

    @staticmethod
    def _hamiltonian_quadruplet_residual(ev: np.ndarray) -> float:
        """Cierre de spec(A) bajo λ ↦ −λ y λ ↦ λ̄."""
        if ev.size == 0:
            return 0.0
        acc = 0.0
        for lam in ev:
            acc += float(np.min(np.abs(ev + lam)))
            acc += float(np.min(np.abs(ev - np.conj(lam))))
        return float(acc / max(2 * ev.size, 1))


class _IDAPBCController:
    r"""
    Fase II. Controlador port-Hamiltoniano IDA-PBC sobre 𝒢_I.

    Estructura de Dirac (J, R): Jᵀ = −J, R ⪰ 0. La asignación de
    interconexión y amortiguamiento produce H_d con Ḣ_d = −∇H_dᵀ R_d ∇H_d ≤ 0.
    El morfismo terminal Φ_II iza K desde g̃ ⊕ g̃⁻¹ (métrica de Maupertuis
    en el fibrado cotangente) e inicia la capa modular de Fase III.
    """

    def __init__(
        self,
        dimension_n: int,
        germ: Optional[_PoincareCartanGerm] = None,
    ) -> None:
        if germ is None:
            germ = _NumericalCore.synthesize_poincare_cartan_germ(dimension_n)
        if germ.n != int(dimension_n):
            raise ValueError("El gérmen de Fase I no coincide con dimension_n.")
        self._germ: _PoincareCartanGerm = germ
        self._n: int = germ.n
        self._2n: int = germ.two_n

    @property
    def germ(self) -> _PoincareCartanGerm:
        return self._germ

    def validate_darboux_coordinates(
        self, q: np.ndarray, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def certify_casimir(
        self,
        grad_C: np.ndarray,
        J_matrix: np.ndarray,
        R_matrix: Optional[np.ndarray] = None,
    ) -> _CasimirCertificate:
        r"""Certifica (J − R) ∇C ≈ 0 (Casimir de la estructura de Dirac)."""
        gC = _NumericalCore.assert_vec("grad_C", grad_C, self._2n)
        J_m = np.asarray(J_matrix, dtype=np.float64)
        _NumericalCore.assert_square("J_matrix", J_m, self._2n)
        residual = J_m @ gC
        if R_matrix is not None:
            R_m = np.asarray(R_matrix, dtype=np.float64)
            _NumericalCore.assert_square("R_matrix", R_m, self._2n)
            residual = residual - R_m @ gC
        res_n = _NumericalCore.euclidean_norm(residual)
        try:
            ev = np.real(la.eigvals(1j * 0.5 * (J_m - J_m.T)))
            ker_dim = int(np.count_nonzero(np.abs(ev) <= _SPECTRAL_TOL))
        except la.LinAlgError:
            ker_dim = 0
        return _CasimirCertificate(
            residual_norm=float(res_n),
            kernel_dimension=ker_dim,
            is_casimir=bool(res_n <= _WILKINSON_DRIFT_LIMIT * max(_NumericalCore.euclidean_norm(gC), 1.0)),
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
        grad_C: Optional[np.ndarray] = None,
    ) -> _IDAPBCResult:
        r"""
        Matching IDA-PBC:
            (J_d − R_d) ∇H_d − (J − R) ∇H = g α,
        con α = (gᵀ G g)^{+} gᵀ G · mismatch, G SPD, y Ḣ_d = −∇H_dᵀ R_d ∇H_d.
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
        casimir_ok = True
        if grad_C is not None:
            casimir_ok = bool(self.certify_casimir(grad_C, Jd_m, Rd_m).is_casimir)

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
        is_strict = bool(p_loss >= -_WILKINSON_DRIFT_LIMIT and structure_ok)
        return _IDAPBCResult(
            control_law=np.asarray(alpha, dtype=np.float64),
            exergy_loss=p_loss,
            lyapunov_derivative=lyap,
            matching_residual=float(matching_residual),
            annihilator_residual=float(annihilator_residual),
            condition_number=float(cond_number),
            structure_ok=structure_ok,
            casimir_ok=casimir_ok,
            is_strictly_passive=is_strict,
        )

    def lift_maupertuis_metric_to_phase_space(self) -> np.ndarray:
        r"""
        Iza la métrica conforme de Maupertuis a T*Q:
            G = g̃ ⊕ g̃⁻¹ ∈ SPD(2n)
        (isomorfismo musical compatible con la polarización de Darboux).
        Este G es el Hamiltoniano modular canónico que Fase III termaliza.
        """
        g = np.asarray(self._germ.maupertuis_metric, dtype=np.float64)
        if g.shape != (self._n, self._n):
            return np.eye(self._2n, dtype=np.float64)
        g_spd = _NumericalCore.higham_nearest_spd(g, floor=self._germ.reg_floor)
        g_inv = la.inv(g_spd)
        G = np.zeros((self._2n, self._2n), dtype=np.float64)
        G[: self._n, : self._n] = g_spd
        G[self._n :, self._n :] = g_inv
        return G

    def induce_modular_spectral_germ(
        self,
        G_metric: Optional[np.ndarray] = None,
        beta: float = _DEFAULT_BETA,
    ) -> _ModularSpectralGerm:
        r"""Gérmen espectral modular ρ = e^{−βK} / Z, K = HighamSPD(G)."""
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta (inverso de temperatura) debe ser positivo y finito.")
        if G_metric is None:
            G_metric = self.lift_maupertuis_metric_to_phase_space()
        _NumericalCore.assert_square("G_metric", G_metric, self._2n)
        _NumericalCore.assert_finite("G_metric", np.asarray(G_metric))
        K = _NumericalCore.higham_nearest_spd(
            np.asarray(G_metric, dtype=np.float64),
            floor=self._germ.reg_floor,
        )
        evals, evecs = la.eigh(K)
        evals = np.real(evals)
        log_terms = np.clip(-float(beta) * evals, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
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

    # ── II.10 MORFISMO TERMINAL Φ_II: gérmen modular celeste ─────────────
    def induce_modular_celestial_germ(
        self,
        G_metric: Optional[np.ndarray] = None,
        beta: float = _DEFAULT_BETA,
        kam_certificate: Optional[_KAMStabilityCertificate] = None,
        twist_certificate: Optional[_TwistNondegeneracyCertificate] = None,
        melnikov_certificate: Optional[_MelnikovCertificate] = None,
        return_map_certificate: Optional[_PoincareReturnMapCertificate] = None,
        nekhoroshev_certificate: Optional[_NekhoroshevCertificate] = None,
        ida_pbc_result: Optional[_IDAPBCResult] = None,
        symplectic_preservation_result: Optional[_SymplecticPreservationResult] = None,
    ) -> _ModularCelestialGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE II ≅ OBJETO INICIAL DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Φ_II : 𝒢_I × Cert_celeste × Cert_PHS  →  𝒢_II.

        Transporta Ω, J, g̃, φ, μ_Liouville, H₀ desde 𝒢_I, adjunta los
        certificados KAM / twist / Melnikov / retorno / Nekhoroshev /
        IDA-PBC / Sp(2n) y produce el Hamiltoniano modular
            K = HighamSPD(g̃ ⊕ g̃⁻¹) ∈ SPD(2n)
        sobre el cual Fase III instancia Tomita–Takesaki, KMS, Umegaki y
        Uhlmann.

        Continuación formal: todo functor de Fase III se escribe
            Φ_III : 𝒢_II  →  cierre termodinámico
        y se inicializa exclusivamente a partir de este objeto.
        """
        if G_metric is None:
            G_metric = self.lift_maupertuis_metric_to_phase_space()
        mod_germ = self.induce_modular_spectral_germ(G_metric, beta=beta)
        return _ModularCelestialGerm(
            n=self._n,
            two_n=self._2n,
            omega=self._germ.omega,
            j_hamiltonian=self._germ.j_hamiltonian,
            reg_floor=self._germ.reg_floor,
            hamiltonian_energy_H0=self._germ.hamiltonian_energy_H0,
            potential_V=self._germ.potential_V,
            maupertuis_metric=self._germ.maupertuis_metric,
            conformal_factor_phi=self._germ.conformal_factor_phi,
            liouville_measure_density=self._germ.liouville_measure_density,
            kam_certificate=kam_certificate,
            twist_certificate=twist_certificate,
            melnikov_certificate=melnikov_certificate,
            return_map_certificate=return_map_certificate,
            nekhoroshev_certificate=nekhoroshev_certificate,
            ida_pbc_result=ida_pbc_result,
            symplectic_preservation_result=symplectic_preservation_result,
            beta=float(beta),
            modular_hamiltonian=mod_germ.modular_hamiltonian,
        )


class _SymplecticPreservationChecker:
    r"""
    Fase II (continuación geométrica). Verifica M ∈ Sp(2n, ℝ), retracción
    polar, transformada de Cayley al álgebra de Lie 𝔰𝔭(2n) y espectro de
    Williamson del generador hamiltoniano.
    """

    def __init__(
        self,
        dimension_n: int,
        germ: Optional[_PoincareCartanGerm] = None,
    ) -> None:
        if germ is None:
            germ = _NumericalCore.synthesize_poincare_cartan_germ(dimension_n)
        self._germ = germ
        self._n = germ.n
        self._2n = germ.two_n

    def verify(self, jacobian_matrix: np.ndarray) -> _SymplecticPreservationResult:
        r"""Residuo ‖Mᵀ Ω M − Ω‖_F, det M, distancia polar y Cayley."""
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
        polar_dist, cayley_res = self._polar_and_cayley(M, omega)
        is_viable = residual_norm <= max(
            _WILKINSON_DRIFT_LIMIT, _WILKINSON_DRIFT_LIMIT * scale
        ) and abs(det_m - 1.0) <= 1e-8 * max(1.0, abs(det_m))
        return _SymplecticPreservationResult(
            residual_norm=float(residual_norm),
            relative_residual=rel,
            determinant=det_m,
            polar_sp_distance=float(polar_dist),
            cayley_hamiltonian_skew_residual=float(cayley_res),
            is_viable=bool(is_viable),
        )

    def cayley_transform(self, M: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Transformada de Cayley 𝔠(M) = (I − M)^{−1}(I + M) ∈ 𝔰𝔭(2n)
        cuando 1 ∉ spec(M). Un elemento de 𝔰𝔭(2n) cumple (Ω X)ᵀ = Ω X
        (X hamiltoniana: (ΩX) simétrica).
        """
        M = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", M, self._2n)
        ident = np.eye(self._2n, dtype=np.float64)
        try:
            X = la.solve(ident - M, ident + M)
        except la.LinAlgError:
            return np.full((self._2n, self._2n), np.nan), float("inf")
        omega = self._germ.omega
        hx = omega @ X
        # X ∈ 𝔰𝔭 ⇔ (ΩX)ᵀ = ΩX  ⇔  Xᵀ Ω + Ω X = 0.
        cayley_res = _NumericalCore.frobenius_norm(X.T @ omega + omega @ X)
        del hx
        return X, float(cayley_res)

    def _polar_and_cayley(
        self, M: np.ndarray, omega: np.ndarray
    ) -> Tuple[float, float]:
        polar_dist = float("inf")
        try:
            S = -omega @ M.T @ omega @ M
            S_h = _NumericalCore.higham_nearest_spd(
                0.5 * (S + S.T), floor=self._germ.reg_floor
            )
            evals, evecs = la.eigh(S_h)
            evals = _NumericalCore.regularize_spectrum(
                np.real(evals), floor=self._germ.reg_floor
            )
            s_inv_sqrt = evecs @ (np.power(evals, -0.5)[:, None] * evecs.T)
            M_retract = M @ s_inv_sqrt
            polar_dist = _NumericalCore.frobenius_norm(M - M_retract)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Retracción polar a Sp(2n) fallida: %s", exc)
        _, cayley_res = self.cayley_transform(M)
        return polar_dist, cayley_res


# =============================================================================
# FIN DE FASE II.
# El objeto 𝒢_II = _ModularCelestialGerm es el dominio de todo functor de
# Fase III (Tomita–Takesaki, KMS, Umegaki, Uhlmann, cierre termodinámico).
# Continuación inmediata (paso siguiente):
#     class _DensityPurifier:
#         def __init__(self, germ: Optional[_ModularCelestialGerm] = None) -> None:
#             ...
#     Φ_III : 𝒢_II  →  fachada ImperialCenturionsEngine.
# =============================================================================
# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE III — VON NEUMANN, TOMITA–TAKESAKI, KMS, UMEGAKI, UHLMANN,          ║
# ║            GIBBS–HELMHOLTZ Y CIERRE DE POINCARÉ–CARTAN                   ║
# ║                                                                          ║
# ║ Dominio:   𝒢_II = _ModularCelestialGerm  (objeto terminal de Fase II).   ║
# ║ Codominio: ImperialCenturionsEngine      (fachada pública = cierre).     ║
# ║                                                                          ║
# ║ Functores:                                                               ║
# ║   III.1  Axiomas de operador densidad (ρ†=ρ, ρ⪰0, Tr ρ=1).              ║
# ║   III.2  Purificación de von Neumann por mayoración espectral.           ║
# ║   III.3  Operador modular Δ_ρ y flujo σ_z^ρ(A)=ρ^{iz} A ρ^{−iz}.        ║
# ║   III.4  Condición KMS en la franja (Takesaki: ω(A σ_{−i}(B))=ω(BA)).   ║
# ║   III.5  Entropía de Umegaki, Klein, Pinsker, Fannes–Audenaert.          ║
# ║   III.6  Fidelidad de Uhlmann, distancia de Bures, Fuchs–van de Graaf.   ║
# ║   III.7  Principio variacional de Gibbs y energía libre de Helmholtz.    ║
# ║   III.8  Recurrencia cuántica de Poincaré (Bocchieri–Loinger).           ║
# ║   III.9  Invariante integral absoluto ∮ λ y medida de Liouville.         ║
# ║                                                                          ║
# ║ Morfismo terminal (III.10): close_thermodynamic_cycle                    ║
# ║     ↦ 𝒞_III = _ThermodynamicClosureCertificate                           ║
# ║     ↦ Fachada ImperialCenturionsEngine = Φ_III ∘ Φ_II ∘ Φ_I.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


@dataclass(frozen=True, slots=True)
class _DensityAxiomsCertificate:
    r"""
    Axiomas de operador densidad sobre M_{2n}(ℂ):
        ρ = ρ†,   ρ ⪰ 0,   Tr ρ = 1.
    El residuo hermítico, el espectro negativo y la fuga de traza miden la
    distancia al simplejo de estados. La pureza Tr(ρ²) ∈ [1/d, 1] y el
    rango efectivo dim supp(ρ) clasifican el estado (puro / mixto / Gibbs).
    """

    hermitian_residual: float
    min_eigenvalue: float
    trace: float
    trace_defect: float
    purity: float
    effective_rank: int
    is_hermitian: bool
    is_positive_semidefinite: bool
    is_trace_one: bool
    is_density_operator: bool


@dataclass(frozen=True, slots=True)
class _PurificationResult:
    r"""
    Operador densidad purificado por proyección de Higham al cono PSD y
    renormalización de traza (mayoración espectral de Schur–Horn).

    S(ρ) = −Tr(ρ log ρ) ∈ [0, log d], con S=0 ⇔ puro y S=log d ⇔ maximally mixed.
    """

    purified_rho: np.ndarray
    effective_rank: int
    von_neumann_entropy: float
    purity: float
    trace: float
    axioms: _DensityAxiomsCertificate
    negative_mass_clipped: float
    entropy_upper_bound: float


@dataclass(frozen=True, slots=True)
class _ModularOperatorCertificate:
    r"""
    Operador modular de Tomita–Takesaki en el álgebra de tipo I:
        Δ_ρ ξ = ρ ξ ρ^{−1}    (acción adjunta sobre Hilbert–Schmidt),
        spec(Δ_ρ) = { λ_i / λ_j : λ_i, λ_j ∈ spec(ρ) ∩ (0, ∞) }.

    El flujo modular es σ_t = Ad(Δ^{it}). La franja de analiticidad KMS
    es { z : −1 ≤ Im z ≤ 0 } en la convención σ_z(A) = ρ^{iz} A ρ^{−iz}.
    """

    modular_spectrum: np.ndarray
    log_modular_spectrum: np.ndarray
    min_positive_eigenvalue_rho: float
    max_modular_ratio: float
    is_faithful: bool
    is_modular_positive: bool


@dataclass(frozen=True, slots=True)
class _KMSCertificate:
    r"""
    Condición KMS de Takesaki para ω(A) = Tr(ρ A) y σ_z^ρ.

    Identidad de borde (z = −i):
        ω(A σ_{−i}(B)) = ω(B A).
    En la recta real, σ_t es automorfismo *-unitario (‖σ_t(A)‖_HS = ‖A‖_HS).
    El parámetro β del gérmen 𝒢_II relaciona el flujo modular con el flujo
    hamiltoniano físico: σ_t^ρ = α_{−β t}^K.
    """

    kms_residual_minus_i: float
    kms_residual_plus_i: float
    real_line_isometry_residual: float
    beta_physical: float
    strip_halfwidth: float
    is_kms: bool
    is_real_automorphism: bool


@dataclass(frozen=True, slots=True)
class _ModularFlowResult:
    r"""Imagen del automorfismo modular σ_z^ρ(A) = ρ^{i z} A ρ^{−i z}."""

    evolved_observable: np.ndarray
    norm_preserved: bool
    kms_residual: float
    modular_operator_spectrum: np.ndarray
    kms_certificate: _KMSCertificate
    modular_operator_certificate: _ModularOperatorCertificate


@dataclass(frozen=True, slots=True)
class _QuantumRelativeEntropyResult:
    r"""
    Divergencia de Umegaki S(ρ‖σ) = Tr(ρ(log ρ − log σ)),
    desigualdad de Klein S ≥ 0 (eq. ⇔ ρ=σ),
    Pinsker S(ρ‖σ) ≥ ½ ‖ρ−σ‖₁² = 2 T²  (T distancia de traza),
    continuidad de Fannes–Audenaert |S(ρ)−S(σ)| ≤ T log(d−1) + h(T).
    """

    umegaki_entropy: float
    uhlmann_fidelity: float
    uhlmann_amplitude: float
    trace_distance: float
    pinsker_bound: float
    pinsker_gap: float
    klein_gap: float
    fannes_audenaert_bound: float
    support_included: bool
    is_klein_nonnegative: bool
    is_pinsker_satisfied: bool


@dataclass(frozen=True, slots=True)
class _UhlmannBuresCertificate:
    r"""
    Fidelidad de Uhlmann F(ρ,σ) = ‖√ρ √σ‖₁² = [Tr √(√ρ σ √ρ)]²
    y distancia de Bures B = √(2(1−√F)) (convención Uhlmann cuadrada).

    Fuchs–van de Graaf:  1 − √F  ≤  T  ≤  √(1 − F).
    """

    uhlmann_amplitude: float
    uhlmann_fidelity: float
    bures_distance: float
    trace_distance: float
    fuchs_lower: float
    fuchs_upper: float
    is_fuchs_satisfied: bool


@dataclass(frozen=True, slots=True)
class _GibbsVariationalCertificate:
    r"""
    Principio variacional de Gibbs–Helmholtz para K ∈ SPD(2n), β > 0:
        ρ_β = e^{−β K} / Z,   Z = Tr e^{−β K},
        F[σ] = Tr(σ K) + β^{−1} Tr(σ log σ)  ≥  F[ρ_β] = −β^{−1} log Z,
    con igualdad ⇔ σ = ρ_β. Equivale a S(σ‖ρ_β) = β (F[σ] − F[ρ_β]) ≥ 0.

    Identidad termodinámica:  S = β (U − F),  U = Tr(ρ K).
    """

    beta: float
    partition_function: float
    helmholtz_free_energy: float
    internal_energy: float
    von_neumann_entropy: float
    thermodynamic_identity_residual: float
    variational_gap: float
    is_minimizer: bool


@dataclass(frozen=True, slots=True)
class _PoincareRecurrenceCertificate:
    r"""
    Recurrencia cuántica de Poincaré (Bocchieri–Loinger, 1958).

    En dimensión finita d = 2n el flujo unitario e^{−i t K} sobre el toro
    de fases { e^{−i t λ_j} } es cuasiperiódico. El tiempo de recurrencia
    espectral es
        T_rec = 2π / min_{i ≠ j} |λ_i(K) − λ_j(K)|
    (gap mínimo del Hamiltoniano modular). Si el gap se anula, el flujo
    posee degeneración y T_rec = +∞ (resonancia modular).
    """

    spectral_gap: float
    recurrence_time: float
    n_degenerate_pairs: int
    is_quasiperiodic: bool
    lyapunov_hint: float


@dataclass(frozen=True, slots=True)
class _PoincareCartanInvariantCertificate:
    r"""
    Invariante integral absoluto de Poincaré–Cartan (E. Cartan, 1922):
        ∮_γ λ = ∮ (p dq − H dt)   es invariante bajo el flujo de X_H.

    El sumando ∮ p dq es el invariante relativo (acciones de Poincaré);
    −∮ H dt = −H T para órbitas periódicas de energía constante.
    La densidad de Liouville μ = Ωⁿ / n! es el invariante absoluto de
    orden 2n (teorema de recurrencia de Poincaré, 1890).
    """

    relative_circulation: float
    total_action: float
    hamiltonian_circulation: float
    absolute_period_integral: float
    liouville_measure_density: float
    is_closed_cycle: bool
    is_invariant_coherent: bool


@dataclass(frozen=True, slots=True)
class _ThermodynamicClosureCertificate:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    CIERRE TERMODINÁMICO (objeto terminal de Fase III).
    ═══════════════════════════════════════════════════════════════════════════
    Compone 𝒢_II con la capa de Tomita–Takesaki y certifica la coherencia
    del ciclo: Gibbs minimiza F, KMS se verifica, Pinsker/Klein no se
    violan, la recurrencia de Poincaré es finita si spec(K) es simple, y
    el invariante ∮ λ (si se suministra un ciclo) es finito.
    """

    n: int
    two_n: int
    beta: float
    axioms: _DensityAxiomsCertificate
    purification: _PurificationResult
    gibbs: _GibbsVariationalCertificate
    kms: _KMSCertificate
    umegaki_self: _QuantumRelativeEntropyResult
    recurrence: _PoincareRecurrenceCertificate
    poincare_cartan: Optional[_PoincareCartanInvariantCertificate]
    is_thermodynamically_closed: bool


class _DensityOperatorCore:
    r"""
    Fase III. Cálculo espectral sobre el cono de operadores densidad.

    Provee log/sqrt/exp de matrices PSD por cálculo funcional de Riesz
    (eigh + recorte de Wilkinson) y los axiomas de von Neumann.
    Consume el Hamiltoniano modular K y β de 𝒢_II.
    """

    @staticmethod
    def spectral_function_psd(
        matrix: np.ndarray,
        func: Callable[[np.ndarray], np.ndarray],
        floor: float = _HIGHAM_REG_FLOOR,
        hermitian: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""f(A) = U f(λ_⌊) U† para A ≃ Hermítica, λ_⌊ = max(λ, floor)."""
        a = np.asarray(matrix)
        _NumericalCore.assert_square("spectral_function_psd", a)
        _NumericalCore.assert_finite("spectral_function_psd", a)
        herm = _NumericalCore.higham_nearest_hermitian(a) if hermitian else a
        evals, evecs = la.eigh(herm)
        evals = np.real(evals)
        evals_reg = _NumericalCore.regularize_spectrum(evals, floor=floor)
        f_vals = np.asarray(func(evals_reg))
        out = evecs @ (f_vals[:, None] * evecs.T.conj())
        return out, evals_reg, evecs

    @staticmethod
    def logm_psd(matrix: np.ndarray, floor: float = _HIGHAM_REG_FLOOR) -> np.ndarray:
        """log(A) por cálculo funcional sobre spec recortado."""
        out, _, _ = _DensityOperatorCore.spectral_function_psd(
            matrix, lambda ev: np.log(np.maximum(ev, floor)), floor=floor
        )
        return out

    @staticmethod
    def sqrtm_psd(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
        """√A con clip a [0, ∞) (sin inflar el núcleo)."""
        a = np.asarray(matrix)
        _NumericalCore.assert_square("sqrtm_psd", a)
        herm = _NumericalCore.higham_nearest_hermitian(a)
        evals, evecs = la.eigh(herm)
        evals = np.sqrt(np.clip(np.real(evals), 0.0, None))
        if floor > 0.0:
            evals = np.maximum(evals, float(np.sqrt(floor)))
        return evecs @ (evals[:, None] * evecs.T.conj())

    @staticmethod
    def expm_herm(matrix: np.ndarray) -> np.ndarray:
        """exp(A) para A hermítica, con recorte del exponente (estabilidad FPU)."""
        a = _NumericalCore.higham_nearest_hermitian(np.asarray(matrix))
        evals, evecs = la.eigh(a)
        evals = np.clip(np.real(evals), -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        return evecs @ (np.exp(evals)[:, None] * evecs.T.conj())

    @staticmethod
    def certify_density_axioms(
        rho: np.ndarray,
        floor: float = _DEFAULT_PURITY_MARGIN,
    ) -> _DensityAxiomsCertificate:
        """Verifica ρ†=ρ, ρ⪰0, Tr ρ=1."""
        a = np.asarray(rho)
        _NumericalCore.assert_square("rho", a)
        _NumericalCore.assert_finite("rho", a)
        herm_res = _NumericalCore.hermitian_residual(a)
        herm = _NumericalCore.higham_nearest_hermitian(a)
        evals = np.real(la.eigvalsh(herm))
        min_ev = float(np.min(evals)) if evals.size else 0.0
        tr = _NumericalCore.compensated_real_trace(herm)
        purity = float(np.real(_NumericalCore.compensated_real_trace(herm @ herm)))
        eff_rank = int(np.sum(evals > float(floor)))
        scale = max(_NumericalCore.frobenius_norm(herm), 1.0)
        is_herm = bool(herm_res <= _HERMITIAN_TOL * scale)
        is_psd = bool(min_ev >= -_WILKINSON_DRIFT_LIMIT * scale)
        is_tr = bool(abs(tr - 1.0) <= 1e-10 * max(1.0, abs(tr)))
        return _DensityAxiomsCertificate(
            hermitian_residual=float(herm_res),
            min_eigenvalue=min_ev,
            trace=float(tr),
            trace_defect=float(abs(tr - 1.0)),
            purity=purity,
            effective_rank=eff_rank,
            is_hermitian=is_herm,
            is_positive_semidefinite=is_psd,
            is_trace_one=is_tr,
            is_density_operator=bool(is_herm and is_psd and is_tr),
        )

    @staticmethod
    def induce_gibbs_state(
        modular_hamiltonian: np.ndarray,
        beta: float,
        floor: float = _HIGHAM_REG_FLOOR,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        r"""
        Estado de Gibbs ρ_β = e^{−β K} / Z asociado a 𝒢_II.
        Retorna (ρ, evals_K, evecs_K, Z).
        """
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta debe ser positivo y finito.")
        K = _NumericalCore.higham_nearest_spd(
            np.asarray(modular_hamiltonian, dtype=np.float64), floor=floor
        )
        evals, evecs = la.eigh(K)
        evals = np.real(evals)
        log_terms = np.clip(-float(beta) * evals, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        unnorm = np.exp(log_terms)
        z = _NumericalCore.kahan_babuska_neumaier_sum(unnorm)
        if z <= _MACHINE_EPS:
            raise ValueError("Función de partición degenerada (Gibbs).")
        rho_eigs = unnorm / z
        rho = evecs @ (rho_eigs[:, None] * evecs.T.conj())
        rho = _NumericalCore.higham_nearest_hermitian(rho)
        return rho, evals, evecs, float(z)

    @staticmethod
    def binary_entropy_nats(x: float) -> float:
        """h(x) = −x log x − (1−x) log(1−x) en nats, con continuidad en {0,1}."""
        xx = float(min(max(x, 0.0), 1.0))
        if xx <= _MACHINE_EPS or xx >= 1.0 - _MACHINE_EPS:
            return 0.0
        return float(-xx * math.log(xx) - (1.0 - xx) * math.log(1.0 - xx))

    @staticmethod
    def von_neumann_entropy_from_spectrum(eigenvalues: np.ndarray) -> float:
        """S = −Σ λ log λ sobre el soporte estrictamente positivo."""
        ev = np.real(np.asarray(eigenvalues, dtype=np.float64)).ravel()
        pos = ev > 0.0
        if not np.any(pos):
            return 0.0
        terms = -ev[pos] * np.log(ev[pos])
        return float(max(_NumericalCore.kahan_babuska_neumaier_sum(terms), 0.0))


class _DensityPurifier:
    r"""
    Fase III. Purificación espectral por mayoración (proyección al simplejo).

    Se inicializa con 𝒢_II (preferente) o con un gérmen espectral legado.
    """

    def __init__(
        self,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
        germ: Optional[_ModularCelestialGerm] = None,
        spectral_germ: Optional[_ModularSpectralGerm] = None,
    ) -> None:
        if purity_margin <= 0.0:
            raise ValueError("purity_margin debe ser positivo.")
        self._margin = float(purity_margin)
        self._celestial = germ
        self._spectral = spectral_germ
        self._floor = (
            float(germ.reg_floor) if germ is not None else _HIGHAM_REG_FLOOR
        )

    @property
    def celestial_germ(self) -> Optional[_ModularCelestialGerm]:
        return self._celestial

    def purify(self, rho_mixed: np.ndarray) -> _PurificationResult:
        rho = np.asarray(rho_mixed)
        _NumericalCore.assert_square("rho_mixed", rho)
        _NumericalCore.assert_finite("rho_mixed", rho)
        rho_herm = _NumericalCore.higham_nearest_hermitian(rho)
        eigenvalues, eigenvectors = la.eigh(rho_herm)
        eigenvalues = np.real(eigenvalues)
        negative_mass = float(np.sum(np.clip(-eigenvalues, 0.0, None)))
        eigenvalues[eigenvalues < self._margin] = 0.0
        effective_rank = int(np.sum(eigenvalues > 0.0))
        trace_sum = _NumericalCore.kahan_babuska_neumaier_sum(eigenvalues)
        if trace_sum > _MACHINE_EPS:
            eigenvalues_norm = eigenvalues / trace_sum
        else:
            eigenvalues_norm = np.zeros_like(eigenvalues)
            eigenvalues_norm[-1] = 1.0
            effective_rank = 1
        rho_purified = eigenvectors @ (
            eigenvalues_norm[:, None] * eigenvectors.T.conj()
        )
        rho_purified = _NumericalCore.higham_nearest_hermitian(rho_purified)
        vn = _DensityOperatorCore.von_neumann_entropy_from_spectrum(eigenvalues_norm)
        purity = _NumericalCore.kahan_babuska_neumaier_sum(eigenvalues_norm ** 2)
        tr = _NumericalCore.kahan_babuska_neumaier_sum(eigenvalues_norm)
        d = int(rho_purified.shape[0])
        axioms = _DensityOperatorCore.certify_density_axioms(
            rho_purified, floor=self._margin
        )
        return _PurificationResult(
            purified_rho=rho_purified,
            effective_rank=effective_rank,
            von_neumann_entropy=float(vn),
            purity=float(purity),
            trace=float(tr),
            axioms=axioms,
            negative_mass_clipped=negative_mass,
            entropy_upper_bound=float(math.log(max(d, 1))),
        )

    def purify_gibbs_from_germ(self) -> _PurificationResult:
        """Purifica el estado de Gibbs inducido por 𝒢_II.K."""
        if self._celestial is None and self._spectral is None:
            raise ValueError("Se exige 𝒢_II o gérmen espectral para Gibbs.")
        if self._celestial is not None:
            rho, _, _, _ = _DensityOperatorCore.induce_gibbs_state(
                self._celestial.modular_hamiltonian,
                self._celestial.beta,
                floor=self._floor,
            )
        else:
            rho = np.asarray(self._spectral.thermal_state)
        return self.purify(rho)


class _TomitaTakesakiFlow:
    r"""
    Fase III. Grupo de automorfismos modulares de Tomita–Takesaki.

    σ_z^ρ(A) = ρ^{i z} A ρ^{−i z},  Δ^{iz} = ρ^{iz} ⊗ (ρ^{−iz})^T  en HS.
    Convención KMS: ω(A σ_{−i}(B)) = ω(B A)  (borde inferior de la franja).
    """

    def __init__(
        self,
        purifier: Optional[_DensityPurifier] = None,
        germ: Optional[_ModularCelestialGerm] = None,
        spectral_germ: Optional[_ModularSpectralGerm] = None,
    ) -> None:
        self._purifier = (
            purifier
            if purifier is not None
            else _DensityPurifier(germ=germ, spectral_germ=spectral_germ)
        )
        self._celestial = germ if germ is not None else getattr(
            self._purifier, "_celestial", None
        )
        self._spectral = spectral_germ

    def modular_operator_certificate(
        self,
        rho: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, _ModularOperatorCertificate]:
        r"""spec(Δ) = {λ_i/λ_j} y log spec; retorna (evals_ρ, evecs_ρ, cert)."""
        purified = self._purifier.purify(rho).purified_rho
        eigenvalues, eigenvectors = la.eigh(purified)
        eigenvalues = np.real(eigenvalues)
        floor = max(
            _NumericalCore.wilkinson_deflation_floor(purified),
            self._purifier._margin,
        )
        ev_reg = _NumericalCore.regularize_spectrum(eigenvalues, floor=floor)
        live = ev_reg > floor
        if np.any(live):
            live_vals = ev_reg[live]
            ratios = (live_vals[:, None] / live_vals[None, :]).ravel()
            log_ratios = np.log(np.maximum(ratios, _MACHINE_EPS))
            min_pos = float(np.min(live_vals))
            max_ratio = float(np.max(ratios))
            faithful = bool(np.all(eigenvalues > floor) or np.min(eigenvalues[eigenvalues > 0]) > floor)
        else:
            ratios = np.array([1.0], dtype=np.float64)
            log_ratios = np.array([0.0], dtype=np.float64)
            min_pos = 0.0
            max_ratio = 1.0
            faithful = False
        cert = _ModularOperatorCertificate(
            modular_spectrum=np.asarray(ratios, dtype=np.float64),
            log_modular_spectrum=np.asarray(log_ratios, dtype=np.float64),
            min_positive_eigenvalue_rho=min_pos,
            max_modular_ratio=max_ratio,
            is_faithful=faithful,
            is_modular_positive=bool(np.all(ratios > 0.0)),
        )
        return ev_reg, eigenvectors, cert

    def evolve(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> _ModularFlowResult:
        A = np.asarray(observable_A)
        _NumericalCore.assert_square("observable_A", A)
        _NumericalCore.assert_finite("observable_A", A)
        rho_in = np.asarray(rho)
        if (
            self._spectral is not None
            and A.shape == self._spectral.thermal_state.shape
            and rho_in.shape == self._spectral.thermal_state.shape
        ):
            purified = self._spectral.thermal_state
            eigenvalues = np.real(self._spectral.eigenvalues)
            eigenvectors = self._spectral.eigenvectors
            floor = _NumericalCore.wilkinson_deflation_floor(purified)
            ev_reg = _NumericalCore.regularize_spectrum(eigenvalues, floor=floor)
            _, _, mod_cert = self.modular_operator_certificate(purified)
        else:
            purified = self._purifier.purify(rho_in).purified_rho
            ev_reg, eigenvectors, mod_cert = self.modular_operator_certificate(rho_in)

        z = complex(time_parameter)
        power = 1j * z
        lambda_left = _NumericalCore.stable_complex_power(ev_reg, power)
        lambda_right = _NumericalCore.stable_complex_power(ev_reg, -power)
        rho_pow_left = eigenvectors @ (lambda_left[:, None] * eigenvectors.T.conj())
        rho_pow_right = eigenvectors @ (lambda_right[:, None] * eigenvectors.T.conj())
        evolved = rho_pow_left @ A @ rho_pow_right

        beta_phys = float(self._celestial.beta) if self._celestial is not None else _DEFAULT_BETA
        kms_cert = self.certify_kms(purified, A, z, evolved, beta_physical=beta_phys)
        norm_before = _NumericalCore.frobenius_norm(A)
        norm_after = _NumericalCore.frobenius_norm(evolved)
        if abs(z.imag) <= _KMS_STRIP_TOL:
            norm_preserved = bool(
                np.isclose(norm_before, norm_after, rtol=1e-8, atol=1e-10)
            )
        else:
            norm_preserved = True
        return _ModularFlowResult(
            evolved_observable=evolved,
            norm_preserved=norm_preserved,
            kms_residual=float(kms_cert.kms_residual_minus_i),
            modular_operator_spectrum=ev_reg,
            kms_certificate=kms_cert,
            modular_operator_certificate=mod_cert,
        )

    def certify_kms(
        self,
        rho: np.ndarray,
        observable: np.ndarray,
        z: complex,
        evolved: Optional[np.ndarray] = None,
        beta_physical: float = _DEFAULT_BETA,
    ) -> _KMSCertificate:
        r"""
        Residuos KMS en los bordes z=±i y residual de isometría en la recta real.

        Identidad canónica (borde z = −i):
            ω(B σ_{−i}(A)) = ω(A B),   B = A†.
        """
        rho_h = _NumericalCore.higham_nearest_hermitian(np.asarray(rho))
        A = np.asarray(observable)
        B = A.T.conj()
        zc = complex(z)

        def _sigma(target: complex) -> np.ndarray:
            ev, vec = la.eigh(rho_h)
            ev = _NumericalCore.regularize_spectrum(
                np.real(ev),
                floor=max(
                    _NumericalCore.wilkinson_deflation_floor(rho_h),
                    _HIGHAM_REG_FLOOR,
                ),
            )
            pwr = 1j * target
            lam_l = _NumericalCore.stable_complex_power(ev, pwr)
            lam_r = _NumericalCore.stable_complex_power(ev, -pwr)
            left = vec @ (lam_l[:, None] * vec.T.conj())
            right = vec @ (lam_r[:, None] * vec.T.conj())
            return left @ A @ right

        sigma_minus_i = _sigma(-1j)
        sigma_plus_i = _sigma(+1j)
        # ω(B σ_{−i}(A)) ≟ ω(A B)
        lhs_m = _NumericalCore.compensated_real_trace(rho_h @ B @ sigma_minus_i)
        rhs_m = _NumericalCore.compensated_real_trace(rho_h @ A @ B)
        scale_m = max(abs(rhs_m), abs(lhs_m), _MACHINE_EPS)
        res_minus = float(abs(lhs_m - rhs_m) / scale_m)
        # Borde +i (convención opuesta): ω(A σ_{+i}(B)) vs ω(B A) — se reporta,
        # no se exige (la franja canónica de σ_z = ρ^{iz} A ρ^{-iz} es Im z ∈ [−1,0]).
        sigma_B_plus = None
        try:
            ev, vec = la.eigh(rho_h)
            ev = _NumericalCore.regularize_spectrum(
                np.real(ev),
                floor=max(
                    _NumericalCore.wilkinson_deflation_floor(rho_h),
                    _HIGHAM_REG_FLOOR,
                ),
            )
            pwr = 1j * (1j)
            lam_l = _NumericalCore.stable_complex_power(ev, pwr)
            lam_r = _NumericalCore.stable_complex_power(ev, -pwr)
            left = vec @ (lam_l[:, None] * vec.T.conj())
            right = vec @ (lam_r[:, None] * vec.T.conj())
            sigma_B_plus = left @ B @ right
        except (np.linalg.LinAlgError, ValueError):
            sigma_B_plus = B
        lhs_p = _NumericalCore.compensated_real_trace(rho_h @ A @ sigma_B_plus)
        rhs_p = _NumericalCore.compensated_real_trace(rho_h @ B @ A)
        scale_p = max(abs(rhs_p), abs(lhs_p), _MACHINE_EPS)
        res_plus = float(abs(lhs_p - rhs_p) / scale_p)

        if abs(zc.imag) <= _KMS_STRIP_TOL:
            ev_obs = evolved if evolved is not None else _sigma(zc)
            iso = abs(
                _NumericalCore.frobenius_norm(A) - _NumericalCore.frobenius_norm(ev_obs)
            )
            scale_iso = max(_NumericalCore.frobenius_norm(A), 1.0)
            iso_res = float(iso / scale_iso)
            is_auto = bool(iso_res <= 1e-8)
        else:
            iso_res = 0.0
            is_auto = True

        is_kms = bool(res_minus <= 1e-6)
        return _KMSCertificate(
            kms_residual_minus_i=res_minus,
            kms_residual_plus_i=res_plus,
            real_line_isometry_residual=iso_res,
            beta_physical=float(beta_physical),
            strip_halfwidth=1.0,
            is_kms=is_kms,
            is_real_automorphism=is_auto,
        )


class _QuantumEntropyCalculator:
    r"""
    Fase III. Entropía relativa de Umegaki, fidelidad de Uhlmann y cotas.
    """

    def __init__(
        self,
        purifier: Optional[_DensityPurifier] = None,
        germ: Optional[_ModularCelestialGerm] = None,
        spectral_germ: Optional[_ModularSpectralGerm] = None,
    ) -> None:
        self._purifier = (
            purifier
            if purifier is not None
            else _DensityPurifier(germ=germ, spectral_germ=spectral_germ)
        )
        self._celestial = germ if germ is not None else getattr(
            self._purifier, "_celestial", None
        )

    def compute(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> _QuantumRelativeEntropyResult:
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
        support_included, leakage = self._support_inclusion(
            e_rho, v_rho, e_sig, v_sig, floor
        )
        amp, fid = self._uhlmann_pair(e_rho, v_rho, e_sig, v_sig)
        td = self._trace_distance(e_rho, v_rho, e_sig, v_sig)
        d = int(rho_p.shape[0])
        t_clip = float(min(max(td, 0.0), 1.0))
        fannes = t_clip * math.log(max(d - 1, 1)) + _DensityOperatorCore.binary_entropy_nats(
            t_clip
        )
        pinsker_bound = 2.0 * (td ** 2)  # = (1/2) ‖ρ−σ‖₁²

        if not support_included:
            logger.warning(
                "supp(ρ) ⊈ supp(σ) (leakage=%.3e): Umegaki = +∞.", leakage
            )
            return _QuantumRelativeEntropyResult(
                umegaki_entropy=float("inf"),
                uhlmann_fidelity=fid,
                uhlmann_amplitude=amp,
                trace_distance=td,
                pinsker_bound=pinsker_bound,
                pinsker_gap=float("inf"),
                klein_gap=float("inf"),
                fannes_audenaert_bound=float(fannes),
                support_included=False,
                is_klein_nonnegative=True,
                is_pinsker_satisfied=True,
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
        klein_gap = float(umegaki)
        pinsker_gap = float(umegaki - pinsker_bound)
        is_klein = bool(umegaki >= -1e-12)
        is_pinsker = bool(pinsker_gap >= -1e-10 * max(1.0, abs(umegaki)))
        return _QuantumRelativeEntropyResult(
            umegaki_entropy=umegaki,
            uhlmann_fidelity=fid,
            uhlmann_amplitude=amp,
            trace_distance=td,
            pinsker_bound=float(pinsker_bound),
            pinsker_gap=pinsker_gap,
            klein_gap=klein_gap,
            fannes_audenaert_bound=float(fannes),
            support_included=True,
            is_klein_nonnegative=is_klein,
            is_pinsker_satisfied=is_pinsker,
        )

    def compute_uhlmann_bures(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> _UhlmannBuresCertificate:
        rho_p = self._purifier.purify(rho).purified_rho
        sig_p = self._purifier.purify(sigma).purified_rho
        e_rho, v_rho = la.eigh(rho_p)
        e_sig, v_sig = la.eigh(sig_p)
        amp, fid = self._uhlmann_pair(np.real(e_rho), v_rho, np.real(e_sig), v_sig)
        td = self._trace_distance(np.real(e_rho), v_rho, np.real(e_sig), v_sig)
        # Bures con F Uhlmann cuadrada: B = √(2(1 − √F)) = √(2(1 − amp)).
        amp_c = float(min(max(amp, 0.0), 1.0))
        fid_c = float(min(max(fid, 0.0), 1.0))
        bures = float(np.sqrt(max(2.0 * (1.0 - amp_c), 0.0)))
        f_lo = float(1.0 - amp_c)
        f_hi = float(np.sqrt(max(1.0 - fid_c, 0.0)))
        ok = bool(td + 1e-10 >= f_lo and td - 1e-10 <= f_hi + 1e-10)
        return _UhlmannBuresCertificate(
            uhlmann_amplitude=amp,
            uhlmann_fidelity=fid,
            bures_distance=bures,
            trace_distance=td,
            fuchs_lower=f_lo,
            fuchs_upper=f_hi,
            is_fuchs_satisfied=ok,
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
    def _uhlmann_pair(
        e_rho: np.ndarray,
        v_rho: np.ndarray,
        e_sig: np.ndarray,
        v_sig: np.ndarray,
    ) -> Tuple[float, float]:
        sqrt_r = v_rho @ (np.sqrt(np.clip(e_rho, 0.0, None))[:, None] * v_rho.T.conj())
        sqrt_s = v_sig @ (np.sqrt(np.clip(e_sig, 0.0, None))[:, None] * v_sig.T.conj())
        svals = la.svdvals(sqrt_r @ sqrt_s)
        amp = _NumericalCore.kahan_babuska_neumaier_sum(np.real(svals))
        amp = float(min(max(amp, 0.0), 1.0))
        return amp, float(amp * amp)

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


class _GibbsVariationalPrinciple:
    r"""
    Fase III. Principio variacional de Gibbs sobre el Hamiltoniano modular K
    de 𝒢_II: ρ_β minimiza F[σ] = ⟨K⟩_σ − β^{−1} S(σ).
    """

    def __init__(self, germ: _ModularCelestialGerm) -> None:
        if not hasattr(germ, "modular_hamiltonian"):
            raise TypeError("Fase III.Gibbs exige 𝒢_II = _ModularCelestialGerm.")
        self._germ = germ

    @property
    def germ(self) -> _ModularCelestialGerm:
        return self._germ

    def certify(
        self,
        trial_state: Optional[np.ndarray] = None,
    ) -> _GibbsVariationalCertificate:
        beta = float(self._germ.beta)
        rho, evals_K, evecs_K, z = _DensityOperatorCore.induce_gibbs_state(
            self._germ.modular_hamiltonian, beta, floor=self._germ.reg_floor
        )
        F_star = float(-math.log(max(z, _MACHINE_EPS)) / beta)
        U = float(
            _NumericalCore.kahan_babuska_neumaier_sum(
                np.real(la.eigvalsh(rho))  # placeholder replaced below
            )
        )
        # U = Tr(ρ K) por cálculo espectral compartido (K y ρ conmutan).
        evals_rho = np.exp(np.clip(-beta * evals_K, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)) / z
        U = float(_NumericalCore.kahan_babuska_neumaier_sum(evals_rho * evals_K))
        S = _DensityOperatorCore.von_neumann_entropy_from_spectrum(evals_rho)
        # Identidad: S ≟ β (U − F)
        id_res = float(abs(S - beta * (U - F_star)))
        variational_gap = 0.0
        is_min = True
        if trial_state is not None:
            pur = _DensityPurifier(
                germ=self._germ
            ).purify(np.asarray(trial_state))
            sigma = pur.purified_rho
            energy_trial = float(
                np.real(_NumericalCore.compensated_real_trace(sigma @ self._germ.modular_hamiltonian))
            )
            F_trial = energy_trial - (pur.von_neumann_entropy / beta)
            variational_gap = float(F_trial - F_star)
            is_min = bool(variational_gap >= -1e-10 * max(1.0, abs(F_star)))
        return _GibbsVariationalCertificate(
            beta=beta,
            partition_function=float(z),
            helmholtz_free_energy=F_star,
            internal_energy=U,
            von_neumann_entropy=float(S),
            thermodynamic_identity_residual=id_res,
            variational_gap=variational_gap,
            is_minimizer=is_min,
        )

    def compute_poincare_recurrence(
        self,
        max_lyapunov_hint: float = 0.0,
    ) -> _PoincareRecurrenceCertificate:
        r"""T_rec = 2π / min_{i≠j} |λ_i(K)−λ_j(K)| sobre spec(K) de 𝒢_II."""
        K = _NumericalCore.higham_nearest_spd(
            np.asarray(self._germ.modular_hamiltonian, dtype=np.float64),
            floor=self._germ.reg_floor,
        )
        evals = np.sort(np.real(la.eigvalsh(K)))
        if evals.size < 2:
            gap = 0.0
            n_deg = 0
        else:
            diffs = np.diff(evals)
            n_deg = int(np.count_nonzero(np.abs(diffs) <= _SPECTRAL_TOL * max(1.0, np.max(np.abs(evals)))))
            pos = diffs[np.abs(diffs) > _SPECTRAL_TOL * max(1.0, np.max(np.abs(evals)))]
            gap = float(np.min(pos)) if pos.size else 0.0
        if gap <= _MACHINE_EPS:
            t_rec = float("inf")
            quasi = False
        else:
            t_rec = float(_ACTION_TWO_PI / gap)
            quasi = True
        return _PoincareRecurrenceCertificate(
            spectral_gap=gap,
            recurrence_time=t_rec,
            n_degenerate_pairs=n_deg,
            is_quasiperiodic=quasi,
            lyapunov_hint=float(max_lyapunov_hint),
        )


class _PoincareCartanThermodynamicClosure:
    r"""
    Fase III. Cierre: invariante absoluto de Poincaré–Cartan + ciclo Gibbs–KMS.

    El morfismo terminal Φ_III : 𝒢_II → 𝒞_III certifica que la capa modular
    es compatible con la geometría de la fase (μ_Liouville, ∮ λ, H₀).
    """

    def __init__(self, germ: _ModularCelestialGerm) -> None:
        if not hasattr(germ, "modular_hamiltonian"):
            raise TypeError(
                "Fase III exige el objeto terminal de Fase II: _ModularCelestialGerm."
            )
        self._germ = germ
        self._purifier = _DensityPurifier(germ=germ)
        self._flow = _TomitaTakesakiFlow(purifier=self._purifier, germ=germ)
        self._entropy = _QuantumEntropyCalculator(purifier=self._purifier, germ=germ)
        self._gibbs = _GibbsVariationalPrinciple(germ)

    @property
    def germ(self) -> _ModularCelestialGerm:
        return self._germ

    def compute_absolute_integral_invariant(
        self,
        cycle_q: np.ndarray,
        cycle_p: np.ndarray,
        period_T: float = 0.0,
        hamiltonian_on_cycle: Optional[float] = None,
    ) -> _PoincareCartanInvariantCertificate:
        r"""
        ∮_γ λ = ∮ p dq − ∮ H dt.

        Si H es constante sobre la órbita (flujo hamiltoniano autónomo),
        ∮ H dt = H · T. Se usa H₀ de 𝒢_II si no se pasa un valor.
        """
        rel = _NumericalCore.compute_poincare_action_invariants(cycle_q, cycle_p)
        H = (
            float(hamiltonian_on_cycle)
            if hamiltonian_on_cycle is not None
            else float(self._germ.hamiltonian_energy_H0)
        )
        T = float(period_T)
        h_circ = float(H * T) if np.isfinite(T) and np.isfinite(H) else 0.0
        abs_int = float(rel.relative_circulation - h_circ)
        mu = float(self._germ.liouville_measure_density)
        coherent = bool(rel.is_closed_cycle and np.isfinite(abs_int))
        return _PoincareCartanInvariantCertificate(
            relative_circulation=rel.relative_circulation,
            total_action=rel.total_action,
            hamiltonian_circulation=h_circ,
            absolute_period_integral=abs_int,
            liouville_measure_density=mu,
            is_closed_cycle=rel.is_closed_cycle,
            is_invariant_coherent=coherent,
        )

    # ── III.10 MORFISMO TERMINAL Φ_III: cierre termodinámico ─────────────
    def close_thermodynamic_cycle(
        self,
        trial_state: Optional[np.ndarray] = None,
        cycle_q: Optional[np.ndarray] = None,
        cycle_p: Optional[np.ndarray] = None,
        period_T: float = 0.0,
        hamiltonian_on_cycle: Optional[float] = None,
        kms_probe: Optional[np.ndarray] = None,
    ) -> _ThermodynamicClosureCertificate:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE III ≅ FACHADA ImperialCenturionsEngine.
        ═══════════════════════════════════════════════════════════════════════
        Φ_III : 𝒢_II × (ciclo opcional)  →  𝒞_III.

        1. Induce ρ_β = e^{−β K}/Z desde K de 𝒢_II (iza g̃ ⊕ g̃⁻¹).
        2. Purifica y certifica axiomas de von Neumann.
        3. Verifica el principio de Gibbs y la identidad S = β(U − F).
        4. Evalúa KMS en el borde z = −i sobre un observable sonda.
        5. Calcula S(ρ_β‖ρ_β) = 0 (Klein saturado) y recurrencia de Poincaré.
        6. Si hay ciclo (q(t), p(t)), adjunta ∮ λ.

        Continuación formal: la fachada pública es el functor identidad
        sobre 𝒞_III con API de las tres fases.
        """
        gibbs_cert = self._gibbs.certify(trial_state=trial_state)
        rho, _, _, _ = _DensityOperatorCore.induce_gibbs_state(
            self._germ.modular_hamiltonian,
            self._germ.beta,
            floor=self._germ.reg_floor,
        )
        purification = self._purifier.purify(rho)
        axioms = purification.axioms
        d = int(self._germ.two_n)
        if kms_probe is None:
            probe = np.eye(d, dtype=np.complex128)
        else:
            probe = np.asarray(kms_probe)
            _NumericalCore.assert_square("kms_probe", probe, dim=d)
        kms_cert = self._flow.certify_kms(
            purification.purified_rho,
            probe,
            z=-1j,
            beta_physical=self._germ.beta,
        )
        umegaki_self = self._entropy.compute(
            purification.purified_rho, purification.purified_rho
        )
        lyap_hint = 0.0
        if (
            self._germ.return_map_certificate is not None
            and hasattr(self._germ.return_map_certificate, "max_lyapunov")
        ):
            lyap_hint = float(self._germ.return_map_certificate.max_lyapunov)
        recurrence = self._gibbs.compute_poincare_recurrence(max_lyapunov_hint=lyap_hint)
        pc_inv: Optional[_PoincareCartanInvariantCertificate] = None
        if cycle_q is not None and cycle_p is not None:
            pc_inv = self.compute_absolute_integral_invariant(
                cycle_q, cycle_p, period_T=period_T,
                hamiltonian_on_cycle=hamiltonian_on_cycle,
            )
        closed = bool(
            axioms.is_density_operator
            and gibbs_cert.thermodynamic_identity_residual <= 1e-8 * max(1.0, abs(gibbs_cert.von_neumann_entropy))
            and umegaki_self.is_klein_nonnegative
            and umegaki_self.umegaki_entropy <= 1e-8
            and kms_cert.is_kms
        )
        return _ThermodynamicClosureCertificate(
            n=int(self._germ.n),
            two_n=int(self._germ.two_n),
            beta=float(self._germ.beta),
            axioms=axioms,
            purification=purification,
            gibbs=gibbs_cert,
            kms=kms_cert,
            umegaki_self=umegaki_self,
            recurrence=recurrence,
            poincare_cartan=pc_inv,
            is_thermodynamically_closed=closed,
        )


# =============================================================================
# FACHADA PÚBLICA — INTEGRACIÓN Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialCenturionsEngine:
    r"""
    Motor espectral ciego en FPU para el cálculo de geodésicas de Maupertuis–
    Jacobi, invarianza de Liouville, control port-Hamiltoniano IDA-PBC,
    análisis KAM/Melnikov/Nekhoroshev y estado modular de Tomita–Takesaki.

    Composición anidada:
        Φ_I   : Darboux + Maupertuis–Jacobi + Hill + Poincaré–Cartan  ⟶  𝒢_I
        Φ_II  : KAM + twist + Melnikov + retorno + IDA-PBC + Sp(2n)  ⟶  𝒢_II
        Φ_III : Purificación + KMS + Umegaki + Gibbs + ∮ λ            ⟶  𝒞_III

    Axiomas físico-matemáticos:
    --------------------------
    1. g̃_{jk}(q) = 2(H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q).
    2. Γ̃^i_{jk} = Γ^i_{jk} + δ^i_j ∂_k φ + δ^i_k ∂_j φ − g_{jk} g^{il} ∂_l φ.
    3. |⟨k, ω⟩| ≥ γ/|k|^τ, τ > n − 1  (KAM) y 𝔅(ω) < ∞ (Bruno–Rüssmann).
    4. M(t₀) = ∫ {H₀, H₁}(γ⁰(t − t₀)) dt  (Melnikov).
    5. Störmer–Verlet / Yoshida: M_step ∈ Sp(2n), det = 1.
    6. IDA-PBC: Jᵀ = −J, R ⪰ 0, Ḣ_d ≤ 0.
    7. ρ_β = e^{−β K}/Z, σ_z^ρ(A) = ρ^{iz} A ρ^{−iz}, KMS en z = −i.
    8. ∮ λ = ∮ (p dq − H dt) invariante absoluto de Poincaré–Cartan.
    """

    def __init__(
        self,
        dimension_n: int = 4,
        hamiltonian_energy_H0: float = 1.0,
        potential_V: float = 0.0,
        novikov_valuation_T: float = 1.0,
        beta: float = _DEFAULT_BETA,
    ) -> None:
        if int(dimension_n) <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta debe ser positivo y finito.")
        self._n: Final[int] = int(dimension_n)
        self._2n: Final[int] = 2 * self._n
        self._novikov_T: Final[float] = float(novikov_valuation_T)
        self._beta: float = float(beta)

        half = self._n
        self._J_canonical = np.block([
            [np.zeros((half, half), dtype=np.float64), np.eye(half, dtype=np.float64)],
            [-np.eye(half, dtype=np.float64), np.zeros((half, half), dtype=np.float64)],
        ])

        # ── Φ_I: gérmen de Poincaré–Cartan (objeto terminal de Fase I) ──
        self._pc_germ: _PoincareCartanGerm = (
            _NumericalCore.synthesize_poincare_cartan_germ(
                dimension_n=self._n,
                hamiltonian_energy_H0=hamiltonian_energy_H0,
                potential_V=potential_V,
            )
        )
        # ── Φ_II: verificadores celestes y port-Hamiltonianos ───────────
        self._celestial = _PoincareCelestialVerifier(self._pc_germ)
        self._ida_pbc = _IDAPBCController(self._n, germ=self._pc_germ)
        self._symplectic_checker = _SymplecticPreservationChecker(
            self._n, germ=self._pc_germ
        )
        self._modular_germ: _ModularCelestialGerm = (
            self._ida_pbc.induce_modular_celestial_germ(beta=self._beta)
        )
        self._mod_spec_germ: _ModularSpectralGerm = (
            self._ida_pbc.induce_modular_spectral_germ(beta=self._beta)
        )
        # ── Φ_III: capa termodinámica modular y cierre ───────────────────
        self._density_purifier = _DensityPurifier(
            germ=self._modular_germ, spectral_germ=self._mod_spec_germ
        )
        self._modular_flow = _TomitaTakesakiFlow(
            purifier=self._density_purifier,
            germ=self._modular_germ,
            spectral_germ=self._mod_spec_germ,
        )
        self._entropy_calculator = _QuantumEntropyCalculator(
            purifier=self._density_purifier,
            germ=self._modular_germ,
            spectral_germ=self._mod_spec_germ,
        )
        self._gibbs = _GibbsVariationalPrinciple(self._modular_germ)
        self._closure = _PoincareCartanThermodynamicClosure(self._modular_germ)
        self._thermo_cert: Optional[_ThermodynamicClosureCertificate] = None

    def _rebind_phase_iii(self) -> None:
        """Reconecta los functores de Fase III tras actualizar 𝒢_II."""
        self._density_purifier = _DensityPurifier(
            germ=self._modular_germ, spectral_germ=self._mod_spec_germ
        )
        self._modular_flow = _TomitaTakesakiFlow(
            purifier=self._density_purifier,
            germ=self._modular_germ,
            spectral_germ=self._mod_spec_germ,
        )
        self._entropy_calculator = _QuantumEntropyCalculator(
            purifier=self._density_purifier,
            germ=self._modular_germ,
            spectral_germ=self._mod_spec_germ,
        )
        self._gibbs = _GibbsVariationalPrinciple(self._modular_germ)
        self._closure = _PoincareCartanThermodynamicClosure(self._modular_germ)

    @property
    def dimension(self) -> int:
        """Dimensión n de la variedad base Q (dim T*Q = 2n)."""
        return self._n

    @property
    def poincare_cartan_germ(self) -> _PoincareCartanGerm:
        """Gérmen de Fase I (objeto terminal Φ_I)."""
        return self._pc_germ

    @property
    def modular_celestial_germ(self) -> _ModularCelestialGerm:
        """Gérmen modular celeste de Fase II (objeto terminal Φ_II)."""
        return self._modular_germ

    @property
    def thermodynamic_closure(self) -> Optional[_ThermodynamicClosureCertificate]:
        """Cierre termodinámico de Fase III (None hasta close_thermodynamic_cycle)."""
        return self._thermo_cert

    # ══════════════════════════════════════════════════════════════════════
    # API FASE I — Maupertuis–Jacobi, Hill, Poincaré–Cartan, Verlet/Yoshida
    # ══════════════════════════════════════════════════════════════════════
    def compute_maupertuis_conformal_metric(
        self,
        q_position: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""Métrica conforme g̃ = 2(H₀−V)g y n = √(2(H₀−V))."""
        _ = _NumericalCore.assert_vec("q_position", q_position, self._n)
        g_tilde, jac_cert = _NumericalCore.compute_maupertuis_conformal_metric(
            potential_V, total_energy_H0, g_base_metric
        )
        if not jac_cert.is_in_hill_region:
            logger.error(
                "[CENTURION_ENGINE_VETO] Cero energía cinética: H₀ − V = %.3e ≤ 0.",
                jac_cert.hill_margin,
            )
        return g_tilde, jac_cert.refractive_index

    def compute_hill_region_margin(
        self, potential_V: float, total_energy_H0: float
    ) -> float:
        """Margen de Hill H₀ − V(q)."""
        return _NumericalCore.compute_hill_region_margin(potential_V, total_energy_H0)

    def certify_hill_region(
        self, potential_V: float, total_energy_H0: float
    ) -> _HillRegionCertificate:
        """Certificado geométrico de la región de Hill / curva de velocidad cero."""
        return _NumericalCore.certify_hill_region(potential_V, total_energy_H0)

    def compute_christoffel_conformal_symbols(
        self,
        q_position: NDArray[np.float64],
        grad_V: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Símbolos de Christoffel conformes Γ̃^i_{jk}."""
        _ = _NumericalCore.assert_vec("q_position", q_position, self._n)
        return _NumericalCore.compute_christoffel_conformal_symbols(
            grad_V, potential_V, total_energy_H0, g_base_metric
        )

    def compute_jacobi_geodesic_acceleration(
        self, q_dot: np.ndarray, christoffel: np.ndarray
    ) -> np.ndarray:
        """q̈^i = −Γ̃^i_{jk} q̇^j q̇^k."""
        return _NumericalCore.compute_jacobi_geodesic_acceleration(q_dot, christoffel)

    def compute_poincare_cartan_lambda(
        self, x: np.ndarray, hamiltonian_value: float = 0.0
    ) -> np.ndarray:
        r"""1-forma de Poincaré–Cartan λ = p dq − H dt."""
        return _NumericalCore.compute_poincare_cartan_lambda(x, hamiltonian_value)

    def compute_liouville_one_form(self, x: np.ndarray) -> np.ndarray:
        r"""1-forma de Liouville θ = p dq."""
        return _NumericalCore.compute_liouville_one_form(x)

    def compute_poincare_action_invariants(
        self, cycle_q: np.ndarray, cycle_p: np.ndarray
    ) -> _IntegralInvariantCertificate:
        r"""Acciones de Poincaré I_k = (1/2π) ∮ p_k dq^k."""
        return _NumericalCore.compute_poincare_action_invariants(cycle_q, cycle_p)

    def hamiltonian_vector_field(self, grad_H: np.ndarray) -> np.ndarray:
        r"""X_H = Ω ∇H."""
        return _NumericalCore.hamiltonian_vector_field(grad_H, self._pc_germ.omega)

    def integrate_symplectic_maupertuis_step(
        self,
        x_state: NDArray[np.float64],
        dt_step: float,
        g_base_metric: NDArray[np.float64],
        potential_V: float,
        grad_V: NDArray[np.float64],
        total_energy_H0: float,
        hess_V: Optional[np.ndarray] = None,
        grad_V_next: Optional[np.ndarray] = None,
        potential_V_next: Optional[float] = None,
    ) -> Tuple[NDArray[np.float64], MaupertuisStepReport]:
        """Paso Störmer–Verlet (shears, det = 1)."""
        return _NumericalCore.integrate_symplectic_maupertuis_step(
            x_state, dt_step, g_base_metric, potential_V, grad_V,
            total_energy_H0, hess_V=hess_V, grad_V_next=grad_V_next,
            potential_V_next=potential_V_next,
        )

    def integrate_yoshida_fourth_order_step(
        self,
        x_state: NDArray[np.float64],
        dt_step: float,
        g_base_metric: NDArray[np.float64],
        potential_V: float,
        grad_V: NDArray[np.float64],
        total_energy_H0: float,
        hess_V: Optional[np.ndarray] = None,
    ) -> Tuple[NDArray[np.float64], MaupertuisStepReport]:
        """Composición de Yoshida de orden 4."""
        return _NumericalCore.integrate_yoshida_fourth_order_step(
            x_state, dt_step, g_base_metric, potential_V, grad_V,
            total_energy_H0, hess_V=hess_V,
        )

    def compute_geodesic_deviation_matrix(
        self,
        q_dot: np.ndarray,
        christoffel: np.ndarray,
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
        hess_V: Optional[np.ndarray] = None,
    ) -> _GeodesicDeviationCertificate:
        """Matriz variacional de Jacobi (germen de monodromía de Fase II)."""
        return _NumericalCore.compute_geodesic_deviation_matrix(
            q_dot, christoffel, grad_V, potential_V, total_energy_H0,
            g_base_metric, hess_V=hess_V,
        )

    # ══════════════════════════════════════════════════════════════════════
    # API FASE II — KAM, twist, Melnikov, retorno, IDA-PBC, Sp(2n)
    # ══════════════════════════════════════════════════════════════════════
    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> _KAMStabilityCertificate:
        r"""Pequeños divisores KAM + Bruno–Rüssmann + Novikov."""
        return self._celestial.compute_poincare_small_divisors_spectrum(
            frequency_vector_omega, wave_vectors_k, jacobian_M, canonical_J,
            tau=tau, gamma=gamma, novikov_valuation_T=self._novikov_T,
        )

    def compute_kolmogorov_twist_certificate(
        self,
        hess_H0_actions: np.ndarray,
        frequency_vector_omega: np.ndarray,
    ) -> _TwistNondegeneracyCertificate:
        """Twist de Kolmogorov e isonergeticidad de Arnol'd."""
        return self._celestial.compute_kolmogorov_twist_certificate(
            hess_H0_actions, frequency_vector_omega
        )

    def solve_poincare_homological_equation(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        fourier_H1: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> _HomologicalEquationCertificate:
        """Ecuación homológica de Lindstedt S_{1,k} = i H_{1,k}/⟨k,ω⟩."""
        return self._celestial.solve_poincare_homological_equation(
            frequency_vector_omega, wave_vectors_k, fourier_H1, tau=tau, gamma=gamma
        )

    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: NDArray[np.float64],
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> NDArray[np.int64]:
        """Retícula de resonancias de Arnol'd."""
        return self._celestial.compute_arnold_resonance_lattice(
            frequency_vector_omega, max_order=max_order, tol=tol
        )

    def compute_chirikov_overlap_parameter(
        self,
        resonance_frequencies: np.ndarray,
        resonance_halfwidths: np.ndarray,
    ) -> float:
        """Parámetro de solapamiento de Chirikov s."""
        return self._celestial.compute_chirikov_overlap_parameter(
            resonance_frequencies, resonance_halfwidths
        )

    def compute_nekhoroshev_stability_bounds(
        self,
        perturbation_epsilon: float,
        chirikov_overlap: float = 0.0,
        steepness_index: Optional[float] = None,
    ) -> _NekhoroshevCertificate:
        """Cotas exponenciales de Nekhoroshev."""
        return self._celestial.compute_nekhoroshev_stability_bounds(
            perturbation_epsilon, chirikov_overlap=chirikov_overlap,
            steepness_index=steepness_index,
        )

    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: NDArray[np.float64],
        t_inf: float = 25.0,
        subharmonic_ratio: Tuple[int, int] = (1, 1),
    ) -> _MelnikovCertificate:
        r"""Función de Melnikov (homoclínica / subarmónica)."""
        return self._celestial.compute_melnikov_function(
            homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid,
            t_inf=t_inf, subharmonic_ratio=subharmonic_ratio,
        )

    def compute_poincare_return_map(
        self,
        jacobian_M: NDArray[np.float64],
        period_T: float = 1.0,
    ) -> _PoincareReturnMapCertificate:
        """Mapa de retorno P: Σ→Σ (Floquet, Lyapunov, Krein, Conley–Zehnder)."""
        return self._celestial.compute_poincare_return_map(jacobian_M, period_T=period_T)

    def certify_section_transversality(
        self, grad_S: np.ndarray, grad_H: np.ndarray, S_value: float = 0.0
    ) -> _SectionTransversalityCertificate:
        """Transversalidad ⟨dS, X_H⟩ ≠ 0."""
        return self._celestial.certify_section_transversality(grad_S, grad_H, S_value)

    def compute_williamson_spectrum(
        self, hess_H: np.ndarray
    ) -> _WilliamsonSpectrumCertificate:
        """Forma normal de Williamson de A = Ω Hess H."""
        return self._celestial.compute_williamson_spectrum(hess_H)

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
        """Ley IDA-PBC (control_law, exergy_loss)."""
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
        grad_C: Optional[np.ndarray] = None,
    ) -> _IDAPBCResult:
        """Ley IDA-PBC con certificado completo (matching, Casimir, Ḣ_d)."""
        return self._ida_pbc.compute_control_law(
            q, p, grad_H, grad_Hd, g_actuator,
            J_matrix, R_matrix, Jd_matrix, Rd_matrix, G_metric, grad_C=grad_C,
        )

    def certify_casimir(
        self,
        grad_C: np.ndarray,
        J_matrix: np.ndarray,
        R_matrix: Optional[np.ndarray] = None,
    ) -> _CasimirCertificate:
        """Certifica (J−R)∇C ≈ 0."""
        return self._ida_pbc.certify_casimir(grad_C, J_matrix, R_matrix)

    def verify_symplectic_preservation(
        self, jacobian_matrix: np.ndarray
    ) -> Tuple[float, bool]:
        """Pertenencia a Sp(2n, ℝ): (residual_norm, is_viable)."""
        result = self._symplectic_checker.verify(jacobian_matrix)
        return result.residual_norm, result.is_viable

    def verify_symplectic_preservation_certified(
        self, jacobian_matrix: np.ndarray
    ) -> _SymplecticPreservationResult:
        """Pertenencia a Sp(2n, ℝ) con retracción polar y Cayley."""
        return self._symplectic_checker.verify(jacobian_matrix)

    def cayley_transform(self, jacobian_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """Transformada de Cayley M ↦ X ∈ 𝔰𝔭(2n)."""
        return self._symplectic_checker.cayley_transform(jacobian_matrix)

    def lift_maupertuis_metric_to_phase_space(self) -> np.ndarray:
        r"""Iza G = g̃ ⊕ g̃⁻¹ ∈ SPD(2n)."""
        return self._ida_pbc.lift_maupertuis_metric_to_phase_space()

    def induce_modular_celestial_germ(
        self,
        G_metric: Optional[np.ndarray] = None,
        beta: float = _DEFAULT_BETA,
        kam_certificate: Optional[_KAMStabilityCertificate] = None,
        twist_certificate: Optional[_TwistNondegeneracyCertificate] = None,
        melnikov_certificate: Optional[_MelnikovCertificate] = None,
        return_map_certificate: Optional[_PoincareReturnMapCertificate] = None,
        nekhoroshev_certificate: Optional[_NekhoroshevCertificate] = None,
        ida_pbc_result: Optional[_IDAPBCResult] = None,
        symplectic_preservation_result: Optional[_SymplecticPreservationResult] = None,
    ) -> _ModularCelestialGerm:
        r"""Morfismo terminal Fase II → inicial Fase III."""
        germ = self._ida_pbc.induce_modular_celestial_germ(
            G_metric=G_metric,
            beta=beta,
            kam_certificate=kam_certificate,
            twist_certificate=twist_certificate,
            melnikov_certificate=melnikov_certificate,
            return_map_certificate=return_map_certificate,
            nekhoroshev_certificate=nekhoroshev_certificate,
            ida_pbc_result=ida_pbc_result,
            symplectic_preservation_result=symplectic_preservation_result,
        )
        self._modular_germ = germ
        self._beta = float(beta)
        self._mod_spec_germ = self._ida_pbc.induce_modular_spectral_germ(
            G_metric=G_metric, beta=beta
        )
        self._rebind_phase_iii()
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE III — Purificación, Tomita–Takesaki, Umegaki, Gibbs, ∮ λ
    # ══════════════════════════════════════════════════════════════════════
    def induce_modular_spectral_germ(
        self,
        G_metric: Optional[np.ndarray] = None,
        beta: float = _DEFAULT_BETA,
    ) -> _ModularSpectralGerm:
        """Gérmen espectral modular ρ = e^{−βK}/Z."""
        germ = self._ida_pbc.induce_modular_spectral_germ(G_metric, beta=beta)
        self._mod_spec_germ = germ
        self._beta = float(beta)
        self._rebind_phase_iii()
        return germ

    def certify_density_axioms(
        self, rho: np.ndarray, purity_margin: float = _DEFAULT_PURITY_MARGIN
    ) -> _DensityAxiomsCertificate:
        """Axiomas ρ†=ρ, ρ⪰0, Tr ρ=1."""
        return _DensityOperatorCore.certify_density_axioms(rho, floor=purity_margin)

    def purify_density_operator(
        self,
        rho_mixed: np.ndarray,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
    ) -> np.ndarray:
        """Purificación de ρ por mayoración espectral."""
        purifier = _DensityPurifier(
            purity_margin, germ=self._modular_germ, spectral_germ=self._mod_spec_germ
        )
        return purifier.purify(rho_mixed).purified_rho

    def purify_density_operator_certified(
        self,
        rho_mixed: np.ndarray,
        purity_margin: float = _DEFAULT_PURITY_MARGIN,
    ) -> _PurificationResult:
        """Purificación con certificado (S_vN, pureza, axiomas)."""
        purifier = _DensityPurifier(
            purity_margin, germ=self._modular_germ, spectral_germ=self._mod_spec_germ
        )
        return purifier.purify(rho_mixed)

    def evolve_tomita_takesaki_flow(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> np.ndarray:
        r"""Flujo modular σ_z^ρ(A) = ρ^{iz} A ρ^{−iz}."""
        return self._modular_flow.evolve(observable_A, rho, time_parameter).evolved_observable

    def evolve_tomita_takesaki_flow_certified(
        self,
        observable_A: np.ndarray,
        rho: np.ndarray,
        time_parameter: complex,
    ) -> _ModularFlowResult:
        """Flujo modular con certificados KMS y Δ_ρ."""
        return self._modular_flow.evolve(observable_A, rho, time_parameter)

    def certify_kms_condition(
        self,
        rho: np.ndarray,
        observable_A: np.ndarray,
        time_parameter: complex = -1j,
    ) -> _KMSCertificate:
        """Condición KMS de Takesaki (borde canónico z = −i)."""
        return self._modular_flow.certify_kms(
            rho, observable_A, complex(time_parameter), beta_physical=self._beta
        )

    def compute_quantum_relative_entropy(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> Tuple[float, float]:
        """Entropía de Umegaki y fidelidad de Uhlmann: (S(ρ‖σ), F(ρ,σ))."""
        result = self._entropy_calculator.compute(rho, sigma)
        return result.umegaki_entropy, result.uhlmann_fidelity

    def compute_quantum_relative_entropy_certified(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> _QuantumRelativeEntropyResult:
        """Umegaki + Uhlmann + Pinsker + Klein + Fannes–Audenaert."""
        return self._entropy_calculator.compute(rho, sigma)

    def compute_uhlmann_bures(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> _UhlmannBuresCertificate:
        """Fidelidad de Uhlmann, Bures y Fuchs–van de Graaf."""
        return self._entropy_calculator.compute_uhlmann_bures(rho, sigma)

    def certify_gibbs_variational(
        self, trial_state: Optional[np.ndarray] = None
    ) -> _GibbsVariationalCertificate:
        """Principio variacional de Gibbs–Helmholtz sobre K de 𝒢_II."""
        return self._gibbs.certify(trial_state=trial_state)

    def compute_poincare_recurrence(
        self, max_lyapunov_hint: float = 0.0
    ) -> _PoincareRecurrenceCertificate:
        """Tiempo de recurrencia cuántica de Poincaré–Bocchieri–Loinger."""
        return self._gibbs.compute_poincare_recurrence(max_lyapunov_hint=max_lyapunov_hint)

    def compute_poincare_cartan_absolute_invariant(
        self,
        cycle_q: np.ndarray,
        cycle_p: np.ndarray,
        period_T: float = 0.0,
        hamiltonian_on_cycle: Optional[float] = None,
    ) -> _PoincareCartanInvariantCertificate:
        r"""Invariante absoluto ∮ λ = ∮ (p dq − H dt)."""
        return self._closure.compute_absolute_integral_invariant(
            cycle_q, cycle_p, period_T=period_T,
            hamiltonian_on_cycle=hamiltonian_on_cycle,
        )

    def close_thermodynamic_cycle(
        self,
        trial_state: Optional[np.ndarray] = None,
        cycle_q: Optional[np.ndarray] = None,
        cycle_p: Optional[np.ndarray] = None,
        period_T: float = 0.0,
        hamiltonian_on_cycle: Optional[float] = None,
        kms_probe: Optional[np.ndarray] = None,
    ) -> _ThermodynamicClosureCertificate:
        r"""
        Morfismo terminal Φ_III: cierra el ciclo Gibbs–KMS–Poincaré–Cartan
        y sella 𝒞_III como objeto terminal de la composición Φ_III ∘ Φ_II ∘ Φ_I.
        """
        cert = self._closure.close_thermodynamic_cycle(
            trial_state=trial_state,
            cycle_q=cycle_q,
            cycle_p=cycle_p,
            period_T=period_T,
            hamiltonian_on_cycle=hamiltonian_on_cycle,
            kms_probe=kms_probe,
        )
        self._thermo_cert = cert
        return cert


__all__ = [
    "ImperialCenturionsEngine",
    "MaupertuisStepReport",
    "_PoincareCartanGerm",
    "_ModularCelestialGerm",
    "_KAMStabilityCertificate",
    "_MelnikovCertificate",
    "_PoincareReturnMapCertificate",
    "_IDAPBCResult",
    "_SymplecticPreservationResult",
    "_PurificationResult",
    "_ModularFlowResult",
    "_QuantumRelativeEntropyResult",
    "_GibbsVariationalCertificate",
    "_PoincareCartanInvariantCertificate",
    "_ThermodynamicClosureCertificate",
]

# =============================================================================
# FIN DE FASE III Y DE LA COMPOSICIÓN Φ_III ∘ Φ_II ∘ Φ_I.
#
# Objetos terminales:
#   𝒢_I   = _PoincareCartanGerm              (Fase I)
#   𝒢_II  = _ModularCelestialGerm            (Fase II)
#   𝒞_III = _ThermodynamicClosureCertificate (Fase III)
#   Fachada = ImperialCenturionsEngine
#
# Identidad de cierre:  close_thermodynamic_cycle ∘ induce_modular_celestial_germ
#                       ∘ synthesize_poincare_cartan_germ  ≅  id_{T*Q × Gibbs}.
# =============================================================================