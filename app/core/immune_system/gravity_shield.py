# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Gravitational Shield (Atractor Determinista Absoluto)                ║
║ Ubicación: app/core/immune_system/gravity_shield.py                          ║
║ Versión: 3.0.0 – Fases Anidadas, Rigor Analítico y Garantías Teorémicas      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo repudia la concepción escalar del "peso" financiero. Introduce un
Pozo Gravitacional en la Variedad Diferenciable de la Malla Agéntica, modelado
como un funtor endomórfico compuesto de tres morfismos encadenados:

    C1 : Costo               ⟶ MasaInercialEfectiva      (Fase 1)
    C2 : MasaInercialEfectiva ⟶ EspacioTiempoDeformado    (Fase 2)
    C3 : (EspacioTiempoDeformado, Trayectoria) ⟶ AcciónDePolyakov  (Fase 3)

    F = C3 ∘ C2 ∘ C1

FUNDAMENTACIÓN MATEMÁTICA Y AXIOMAS DE EJECUCIÓN (v3):

§1. ADQUISICIÓN DE MASA (Fröhlich saturado + piso suave tipo dispersión):
    $$ m^{**} = \sqrt{\left(m^*\left(1+\frac{\alpha}{6}\right)\right)^2 + m_{\min}^2}
               \cdot \left(1 + \tanh\!\left(\frac{\alpha_f}{2\pi}\right)\right) $$
    El piso ya no usa $\max(\cdot)$ (no diferenciable); se usa el análogo de la
    relación de dispersión relativista $E=\sqrt{p^2+m^2}$, infinitamente
    diferenciable y con el mismo límite asintótico. La corrección de Fröhlich
    se satura vía $\tanh$ para evitar masas divergentes (regularización UV),
    preservando la pendiente $1/2\pi$ en el origen (compatibilidad IR).

§2. DEFORMACIÓN MÉTRICA (Congruencia + Ley de Inercia de Sylvester):
    $$ \tilde G = \Lambda^{1/2} G \Lambda^{1/2}, \quad
       \Lambda_a = \exp\!\left(\frac{2\mathcal G}{c^4} m^{**}\,\delta_{a,k}\right) $$
    Puesto que $\Lambda^{1/2}$ es diagonal real e invertible, el **Teorema de
    Sylvester (Ley de Inercia)** garantiza que si $G$ es SPD, $\tilde G$ es SPD
    para *todo* $m^{**}\ge 0$ — sin necesidad de verificación espectral en cada
    invocación. La verificación se hace **una sola vez** sobre $G_{PHYSICS}$
    (Cholesky) al construir el caché base.

    Los símbolos de Christoffel se derivan **analíticamente** (forma cerrada,
    error de truncamiento nulo) explotando que $\partial_\mu \tilde G_{ab}$ es
    idénticamente cero salvo $\mu = k$ (el nodo masivo es la única coordenada
    de la que depende la métrica). La curvatura seccional en los 2-planos
    $(e_k, e_j)$ se obtiene con **diferenciación de paso complejo**
    (Squire–Trapp, 1998) sobre la misma fórmula cerrada, alcanzando precisión
    de máquina sin error de cancelación.

§3. ATRAPAMIENTO GEODÉSICO (Feynman-Kac + veto booleano dual):
    $$ S_E[\gamma] = \frac12\int_0^1 \tilde G_{\mu\nu}\dot\gamma^\mu\dot\gamma^\nu\,d\tau,
       \qquad \Psi[\gamma]=\exp(-S_E/\hbar_{eff}) $$
    La decisión de veto se evalúa en **espacio logarítmico**
    ($S_E \gtrless -\hbar_{eff}\ln(\text{tol})$), inmune a underflow de punto
    flotante, y se combina mediante un **operador booleano OR** con un segundo
    criterio independiente de curvatura crítica:
    $$ \text{Atrapado} = (S_E > S_{crit}) \lor (K_{\max} > K_{crit}) $$

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias arquitectónicas estrictas de APU Filter
from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.telemetry_schemas import PolaronCartridge

logger = logging.getLogger("MIC.ImmuneSystem.GravityShield")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS Y UMBRALES GRAVITACIONALES
# ══════════════════════════════════════════════════════════════════════════════
class GravitationalConstants:
    r""" Constantes cosmológicas y cuánticas adaptadas a la Malla Agéntica. """

    CYBER_G: Final[float] = 6.67430e-2
    CYBER_C: Final[float] = 2.99792e2
    G_C4_FACTOR: Final[float] = (2.0 * CYBER_G) / (CYBER_C ** 4)
    HBAR_EFF: Final[float] = 1.05457e-3
    SCHWARZSCHILD_TOLERANCE: Final[float] = 1e-9
    MINIMAL_INERTIAL_MASS: Final[float] = 1e-12

    # ── Añadidos v3.0.0 ──────────────────────────────────────────────────────
    # Umbral de curvatura seccional crítica (segundo criterio de veto, OR).
    CRITICAL_SECTIONAL_CURVATURE: Final[float] = 1.0e3
    # Paso h para diferenciación de paso complejo (precisión ~ε_máquina).
    COMPLEX_STEP_H: Final[float] = 1e-30
    # Umbral de simetría numérica aceptado al validar G_PHYSICS.
    SYMMETRY_ATOL: Final[float] = 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES TOPOLÓGICO-GRAVITACIONALES
# ══════════════════════════════════════════════════════════════════════════════
class GravitationalCollapseError(TopologicalInvariantError):
    """ Detonada si la deformación métrica destruye la firma SPD del tensor,
    o si una forma cuadrática que debería ser semidefinida positiva no lo es. """
    pass


class EventHorizonViolation(TopologicalInvariantError):
    """ Detonada si el LLM intenta escapar de un nodo cuya masa (o curvatura)
    supera el límite de escape. Puede originarse por veto de acción, veto de
    curvatura, o ambos (combinador booleano OR). """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ FORMAL DEL CARTUCHO POLARÓN (Duck-Typing Verificable)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class PolaronLike(Protocol):
    r""" Especificación formal mínima que debe satisfacer todo objeto pasado
    como `polaron` al funtor. Sustituye la cadena de `hasattr` dispersos por
    un contrato de interfaz verificable en tiempo de ejecución. """
    inertial_mass: float
    volatility_alpha: float
    frohlich_coupling: float


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO-TIEMPO LOGÍSTICO
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MassAcquisitionResult:
    r""" Salida de la Fase 1. Registro de procedencia auditable: no sólo la
    masa final, sino los factores que la produjeron (trazabilidad forense). """
    raw_cost: float
    volatility_alpha: float
    frohlich_coupling: float
    frohlich_factor: float
    effective_mass: float


@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    r""" Caché espectral de la métrica base, construido y validado UNA sola
    vez. Su existencia misma es la prueba de que $G_{PHYSICS}$ es SPD
    (Cholesky exitoso), lo cual —por la Ley de Inercia de Sylvester—
    garantiza SPD para toda deformación congruente diagonal posterior. """
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    dimension: int


@dataclass(frozen=True, slots=True)
class WarpedSpaceTime:
    r""" Colector Riemanniano deformado por la masa inercial. Salida de la
    Fase 2 / entrada de la Fase 3. """
    original_metric: NDArray[np.float64]
    deformed_metric: NDArray[np.float64]
    christoffel_symbols: NDArray[np.float64]        # Γ^μ_{νρ}, forma cerrada exacta
    riemann_curvature: NDArray[np.float64]          # (n,n) reducido: (R(e_k,e_j)e_j)^ρ
    sectional_curvatures: NDArray[np.float64]        # K(e_k, e_j) ∀ j
    max_sectional_curvature: float
    node_index: int


@dataclass(frozen=True, slots=True)
class PolyakovAction:
    r""" Evaluación de la acción y amplitud de Feynman-Kac de una trayectoria,
    junto con el desglose booleano explícito del veredicto de atrapamiento. """
    action_integral: float
    feynman_amplitude: float
    is_trapped: bool
    action_veto: bool
    curvature_veto: bool
    discretization_error: float


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1 — ADQUISICIÓN DE MASA INERCIAL                                    ║
# ║   Acoplamiento de Fröhlich saturado + piso suave tipo dispersión           ║
# ║   Morfismo: C1 : (costo, α, α_f) ⟶ MassAcquisitionResult                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
def _smooth_inertial_floor(m_linear: float, m_min: float) -> float:
    r"""
    Piso de masa infinitamente diferenciable, análogo a la relación de
    dispersión relativista $E=\sqrt{p^2+m^2}$:

    $$ m_{\text{floor}}(m) = \sqrt{m^2 + m_{\min}^2} $$

    Propiedades verificadas:
      • $m_{\text{floor}}(0) = m_{\min}$ (piso exacto).
      • $\lim_{m\to\infty} m_{\text{floor}}(m) - m = 0$ (transparente en el
        régimen de masas grandes).
      • $C^\infty$ en todo $\mathbb{R}_{\ge 0}$ (a diferencia de `max()`,
        que introduce un punto anguloso no diferenciable en $m=m_{\min}$).
    """
    return math.sqrt(m_linear * m_linear + m_min * m_min)


def _saturating_frohlich_factor(coupling: float) -> float:
    r"""
    Corrección de auto-energía de Fröhlich, saturada mediante $\tanh$:

    $$ f(\alpha_f) = 1 + \tanh\!\left(\frac{\alpha_f}{2\pi}\right) $$

    A diferencia de la corrección lineal original $1+\alpha_f/2\pi$ (divergente),
    esta función:
      • Preserva la pendiente física en el origen: $f'(0) = 1/2\pi$ (idéntica
        a la aproximación de bajo acoplamiento del polarón de Fröhlich).
      • Está acotada: $f(\alpha_f) \in [1, 2)$ para $\alpha_f \ge 0$, evitando
        masas efectivas divergentes (regularización ultravioleta del modelo).
    """
    return 1.0 + math.tanh(coupling / (2.0 * math.pi))


def _acquire_effective_mass(
    base_cost: float,
    volatility_alpha: float,
    frohlich_coupling: Optional[float] = None,
) -> MassAcquisitionResult:
    r"""
    Calcula la masa inercial efectiva del insumo bajo estrés térmico,
    con procedencia auditable completa.

    $$ m^{**} = \sqrt{\left(m^*\!\left(1+\frac{\alpha}{6}\right)\right)^{2}
                       + m_{\min}^{2}}\;\cdot\;\left(1+\tanh\!\left(\frac{\alpha_f}{2\pi}\right)\right) $$

    Args:
        base_cost: costo financiero $m^*\ge 0$.
        volatility_alpha: parámetro de volatilidad $\alpha \ge 0$.
        frohlich_coupling: acoplamiento de Fröhlich $\alpha_f \ge 0$ (opcional).

    Returns:
        MassAcquisitionResult — registro inmutable y trazable, que constituye
        la ENTRADA FORMAL de la Fase 2 (`_deform_metric_tensor`).

    Raises:
        ValueError: si algún parámetro viola el dominio físico ($\ge 0$).
    """
    if base_cost < 0.0 or volatility_alpha < 0.0:
        raise ValueError("El costo base y la volatilidad deben ser no negativos.")
    if frohlich_coupling is not None and frohlich_coupling < 0.0:
        raise ValueError("El acoplamiento de Fröhlich debe ser no negativo.")

    m_linear = base_cost * (1.0 + volatility_alpha / 6.0)
    m_floored = _smooth_inertial_floor(m_linear, GravitationalConstants.MINIMAL_INERTIAL_MASS)

    coupling_used = 0.0
    frohlich_factor = 1.0
    if frohlich_coupling:
        coupling_used = frohlich_coupling
        frohlich_factor = _saturating_frohlich_factor(frohlich_coupling)

    effective_mass = m_floored * frohlich_factor

    result = MassAcquisitionResult(
        raw_cost=base_cost,
        volatility_alpha=volatility_alpha,
        frohlich_coupling=coupling_used,
        frohlich_factor=frohlich_factor,
        effective_mass=effective_mass,
    )
    logger.debug(
        f"[Fase 1] m**={effective_mass:.6e} "
        f"(base={base_cost}, α={volatility_alpha}, f_frohlich={frohlich_factor:.6f})"
    )
    return result


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2 — DEFORMACIÓN MÉTRICA (POZO GRAVITACIONAL)                        ║
# ║   Congruencia diagonal + Sylvester + Christoffel analítico +               ║
# ║   curvatura seccional vía diferenciación de paso complejo                  ║
# ║   Morfismo: C2 : MassAcquisitionResult ⟶ WarpedSpaceTime                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ── Continuación directa de la Fase 1: el primer método de la Fase 2 recibe
#    exactamente el tipo `MassAcquisitionResult` que produce el último método
#    de la Fase 1, cerrando la composición categórica C2 ∘ C1. ──────────────

def _build_base_metric_cache(g_base: NDArray[np.float64]) -> BaseMetricCache:
    r"""
    Construye y **verifica UNA sola vez** la estructura espectral de la
    métrica base $G_{PHYSICS}$.

    La factorización de Cholesky $G = LL^\top$ es, simultáneamente, la
    verificación de simetría-definida-positividad y el medio más estable
    para obtener $G^{-1}$ vía `cho_solve` (evita `np.linalg.inv` explícito).

    Fundamento teorémico (Ley de Inercia de Sylvester): si $G$ es SPD y
    $\Lambda^{1/2}$ es diagonal real invertible, entonces
    $\Lambda^{1/2} G \Lambda^{1/2}$ es SPD para cualquier $\Lambda^{1/2}$.
    En consecuencia, **no es necesario re-verificar SPD en cada deformación**
    — sólo aquí, una vez, sobre la base.

    Raises:
        GravitationalCollapseError: si $G_{PHYSICS}$ no es simétrica o no es
            definida positiva (Cholesky falla).
    """
    if g_base.ndim != 2 or g_base.shape[0] != g_base.shape[1]:
        raise GravitationalCollapseError("G_PHYSICS debe ser una matriz cuadrada.")
    if not np.allclose(g_base, g_base.T, atol=GravitationalConstants.SYMMETRY_ATOL):
        raise GravitationalCollapseError(
            "G_PHYSICS no es simétrica: viola el axioma Riemanniano de base."
        )
    try:
        cholesky_factor = la.cholesky(g_base, lower=True)
    except la.LinAlgError as exc:
        raise GravitationalCollapseError(
            f"G_PHYSICS no es SPD (fallo de Cholesky): {exc}"
        ) from exc

    n = g_base.shape[0]
    g_inv = la.cho_solve((cholesky_factor, True), np.eye(n))

    logger.info(f"[Fase 2·setup] Caché métrica base verificada SPD (Sylvester) — dim={n}")
    return BaseMetricCache(
        g_base=g_base, cholesky_factor=cholesky_factor, g_inv=g_inv, dimension=n
    )


def _construct_warped_metric(
    cache: BaseMetricCache,
    effective_mass: float | complex,
    node_index: int,
) -> Tuple[NDArray, NDArray]:
    r"""
    Construye $(\tilde G, \tilde G^{-1})$ mediante la transformación de
    semejanza diagonal:

    $$ \tilde G = \Lambda^{1/2} G \Lambda^{1/2}, \qquad
       \tilde G^{-1} = \Lambda^{-1/2} G^{-1} \Lambda^{-1/2} $$

    La inversa se obtiene **analíticamente** (sin resolver un sistema lineal
    en cada llamada): al ser $\Lambda^{1/2}$ diagonal, su inversa es trivial.

    Nota de diseño: `effective_mass` admite tipo `complex` para habilitar la
    diferenciación de paso complejo (ver `_complex_step_partial_k_christoffel`).
    """
    n = cache.dimension
    if not (0 <= node_index < n):
        raise IndexError(f"node_index={node_index} fuera de rango [0,{n}).")

    c = GravitationalConstants.G_C4_FACTOR
    dtype = np.complex128 if isinstance(effective_mass, complex) else np.float64

    delta = np.zeros(n, dtype=dtype)
    delta[node_index] = c * effective_mass

    lam_sqrt = np.exp(delta / 2.0)
    lam_inv_sqrt = np.exp(-delta / 2.0)

    g_base = cache.g_base.astype(dtype)
    g_inv_base = cache.g_inv.astype(dtype)

    g_tilde = np.outer(lam_sqrt, lam_sqrt) * g_base
    g_inv_tilde = np.outer(lam_inv_sqrt, lam_inv_sqrt) * g_inv_base
    return g_tilde, g_inv_tilde


def _christoffel_from_metric(
    g_tilde: NDArray, g_inv_tilde: NDArray, node_index: int
) -> NDArray:
    r"""
    Símbolos de Christoffel en **forma cerrada exacta** para la deformación
    conforme-diagonal anclada al nodo $k$.

    Dado que $\tilde G$ depende únicamente de la coordenada $x^k$ (identificada
    con $m^{**}$), se tiene $\partial_\mu \tilde G_{ab} = 0 \;\forall \mu\neq k$,
    y:
    $$ \partial_k \tilde G_{ab} = \tilde G_{ab}\cdot\frac{c}{2}\left(
       \delta_{a,k}+\delta_{b,k}\right), \qquad c = \frac{2\mathcal G}{c_{\text{luz}}^4} $$

    Sustituyendo en $\Gamma^\mu_{\nu\rho}=\tfrac12 \tilde G^{\mu\sigma}
    (\partial_\nu \tilde G_{\sigma\rho}+\partial_\rho \tilde G_{\nu\sigma}
    -\partial_\sigma \tilde G_{\nu\rho})$ y usando $\tilde G^{\mu\sigma}
    \tilde G_{\sigma\rho}=\delta^\mu_\rho$, se obtiene una expresión
    vectorizable en $O(n^2)$, sin diferencias finitas y con **error de
    truncamiento matemáticamente nulo**.
    """
    n = g_tilde.shape[0]
    k = node_index
    c = GravitationalConstants.G_C4_FACTOR

    gk_up = g_inv_tilde[:, k]        # $\tilde G^{\mu k}$
    gk_down = g_tilde[k, :]          # $\tilde G_{k\rho}$
    e_k = np.zeros(n, dtype=g_tilde.dtype)
    e_k[k] = 1.0

    base_term = np.outer(gk_up, gk_down) + np.outer(e_k, e_k)

    term1 = np.zeros((n, n, n), dtype=g_tilde.dtype)
    term1[:, k, :] = (c / 2.0) * base_term

    term2 = np.zeros((n, n, n), dtype=g_tilde.dtype)
    term2[:, :, k] = (c / 2.0) * base_term

    delta_sum = np.zeros((n, n), dtype=g_tilde.dtype)
    delta_sum[k, :] += 1.0
    delta_sum[:, k] += 1.0
    term3 = (c / 2.0) * np.einsum("m,vr->mvr", gk_up, delta_sum * g_tilde)

    return 0.5 * (term1 + term2 - term3)


def _complex_step_partial_k_christoffel(
    cache: BaseMetricCache,
    effective_mass: float,
    node_index: int,
    h: Optional[float] = None,
) -> NDArray[np.float64]:
    r"""
    Derivada $\partial_k \Gamma^\mu_{\nu\rho}$ mediante **diferenciación de
    paso complejo** (Squire & Trapp, 1998):

    $$ \partial_k f(m^{**}) \approx \frac{\operatorname{Im} f(m^{**}+ih)}{h} $$

    Ventaja doctoral sobre diferencias finitas centradas: al no restar dos
    evaluaciones de magnitud similar, se elimina el error de cancelación
    catastrófica, permitiendo $h\to 10^{-30}$ y precisión de máquina en la
    derivada — sin necesidad de optimizar $h$ mediante heurísticas.

    Válido porque toda la cadena (`exp`, producto, contracción tensorial) es
    **holomorfa** en $m^{**}$.
    """
    h = GravitationalConstants.COMPLEX_STEP_H if h is None else h
    g_tilde_c, g_inv_c = _construct_warped_metric(
        cache, complex(effective_mass, h), node_index
    )
    gamma_c = _christoffel_from_metric(g_tilde_c, g_inv_c, node_index)
    return gamma_c.imag / h


def _sectional_curvatures_around_node(
    g_tilde: NDArray[np.float64],
    gamma: NDArray[np.float64],
    d_k_gamma: NDArray[np.float64],
    node_index: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Curvatura seccional exacta $K(e_k, e_j)$ para cada $j\neq k$, restringida
    deliberadamente a los 2-planos que contienen la dirección de colapso
    $e_k$ (los únicos relevantes para el atrapamiento geodésico en torno al
    nodo masivo; la curvatura en planos ortogonales a $e_k$ queda fuera del
    alcance operativo de este escudo — restricción documentada, no omisión).

    Fórmula:
    $$ K(e_k,e_j) = \frac{R(e_k,e_j,e_j,e_k)}{g_{kk}g_{jj}-g_{kj}^2}, \qquad
       R(e_k,e_j)e_j = \partial_k\Gamma^\rho_{jj} + \Gamma^\rho_{k\lambda}\Gamma^\lambda_{jj}
                        - \Gamma^\rho_{j\lambda}\Gamma^\lambda_{kj} $$

    (El término $-\partial_j\Gamma^\rho_{kj}$ se anula idénticamente para
    $j\neq k$, pues la métrica no depende de $x^j$.)

    Returns:
        (K, R_vec): vector de curvaturas seccionales (n,) y tensor de Riemann
        **reducido** (n,n) — no se materializa el tensor completo $O(n^4)$
        porque no se usa fuera de este contexto (honestidad computacional).
    """
    n = g_tilde.shape[0]
    k = node_index

    diag_dk_gamma = np.einsum("rjj->rj", d_k_gamma)      # ∂_k Γ^ρ_{jj}
    diag_gamma = np.einsum("ljj->lj", gamma)             # Γ^λ_{jj}
    gamma_k = gamma[:, k, :]                             # Γ^ρ_{kλ}  (también Γ^λ_{kj})

    term_b = gamma_k @ diag_gamma                        # Σ_λ Γ^ρ_{kλ} Γ^λ_{jj}
    term_c = np.einsum("rjl,lj->rj", gamma, gamma_k)     # Σ_λ Γ^ρ_{jλ} Γ^λ_{kj}

    r_vec = diag_dk_gamma + term_b - term_c              # (R(e_k,e_j)e_j)^ρ
    r_full = g_tilde[:, k] @ r_vec                        # R(e_k,e_j,e_j,e_k)

    gk = g_tilde[k, k]
    gjj = np.diagonal(g_tilde)
    gkj = g_tilde[k, :]
    denom = gk * gjj - gkj ** 2

    curvatures = np.zeros(n, dtype=np.float64)
    valid = np.abs(denom) > 1e-14
    valid[k] = False  # plano (e_k,e_k) degenerado, no definido
    curvatures[valid] = r_full[valid] / denom[valid]

    return curvatures, r_vec


def _deform_metric_tensor(
    cache: BaseMetricCache,
    mass_result: MassAcquisitionResult,
    node_index: int,
) -> WarpedSpaceTime:
    r"""
    **Último método de la Fase 2.** Orquesta la deformación completa a partir
    de la salida directa de la Fase 1 (`MassAcquisitionResult`), cerrando la
    composición $C_2$. Su valor de retorno, `WarpedSpaceTime`, es exactamente
    el tipo de entrada del primer método de la Fase 3.

    Args:
        cache: caché espectral de la métrica base (construido una sola vez).
        mass_result: salida de `_acquire_effective_mass` (Fase 1).
        node_index: índice del nodo donde se concentra la masa.

    Returns:
        WarpedSpaceTime completo: métrica deformada, Christoffel exacto,
        curvatura seccional real y su reducción de Riemann.
    """
    n = cache.dimension
    if not (0 <= node_index < n):
        raise IndexError(
            f"El índice del nodo ({node_index}) está fuera de la "
            f"dimensionalidad de G_PHYSICS (n={n})."
        )

    m_eff = mass_result.effective_mass
    g_tilde, g_inv_tilde = _construct_warped_metric(cache, m_eff, node_index)

    # Aserción defensiva de bajo costo (O(n)): la SPD ya está garantizada
    # teorémicamente (Sylvester); esto sólo detecta defectos de implementación.
    if np.any(np.diagonal(g_tilde) <= 0.0):
        raise GravitationalCollapseError(
            "Violación de invariante: diagonal no positiva tras congruencia "
            "diagonal — defecto de implementación, no físico."
        )

    gamma = _christoffel_from_metric(g_tilde, g_inv_tilde, node_index)
    d_k_gamma = _complex_step_partial_k_christoffel(cache, m_eff, node_index)
    curvatures, reduced_riemann = _sectional_curvatures_around_node(
        g_tilde, gamma, d_k_gamma, node_index
    )
    max_curvature = float(np.max(np.abs(curvatures))) if n > 1 else 0.0

    logger.debug(
        f"[Fase 2] m**={m_eff:.6e} → K_max={max_curvature:.6e} en nodo {node_index}"
    )

    return WarpedSpaceTime(
        original_metric=cache.g_base,
        deformed_metric=g_tilde,
        christoffel_symbols=gamma,
        riemann_curvature=reduced_riemann,
        sectional_curvatures=curvatures,
        max_sectional_curvature=max_curvature,
        node_index=node_index,
    )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3 — ATRAPAMIENTO GEODÉSICO (INTEGRAL DE CAMINO DE FEYNMAN‑KAC)      ║
# ║   Cuadratura real de Simpson + Richardson, veto en espacio logarítmico,    ║
# ║   combinador booleano dual (acción ∨ curvatura)                           ║
# ║   Morfismo: C3 : (WarpedSpaceTime, Trayectoria) ⟶ PolyakovAction          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ── Continuación directa de la Fase 2: el primer método de esta fase recibe
#    exactamente el `WarpedSpaceTime` producido por `_deform_metric_tensor`,
#    cerrando la composición categórica C3 ∘ C2 ∘ C1. ───────────────────────

def _kinetic_density(velocity: NDArray[np.float64], metric: NDArray[np.float64]) -> float:
    r"""
    Densidad cinética instantánea $\tilde G_{\mu\nu}\dot\gamma^\mu\dot\gamma^\nu$.

    Al ser $\tilde G$ SPD (garantizado por Sylvester), esta forma cuadrática
    debe ser $\ge 0$ (desigualdad de Cauchy-Schwarz generalizada); se afirma
    como invariante defensivo, no como verificación redundante de SPD.
    """
    value = float(np.real(velocity @ metric @ velocity))
    if value < -1e-9:
        raise GravitationalCollapseError(
            "Forma cuadrática negativa: la métrica deformada perdió su "
            "signatura SPD en tiempo de evaluación (violación de invariante)."
        )
    return max(value, 0.0)


def _simpson_composite(f_vals: NDArray[np.float64], h: float) -> float:
    r"""
    Regla de Simpson compuesta clásica sobre $N+1$ muestras equiespaciadas
    ($N$ par). Exactitud $O(h^4)$ para integrandos suficientemente suaves.
    """
    n_points = len(f_vals)
    if n_points < 3 or (n_points - 1) % 2 != 0:
        raise ValueError(
            "Simpson compuesto requiere un número impar de puntos "
            "(subintervalos pares)."
        )
    return (h / 3.0) * (
        f_vals[0]
        + f_vals[-1]
        + 4.0 * np.sum(f_vals[1:-1:2])
        + 2.0 * np.sum(f_vals[2:-2:2])
    )


def _evaluate_feynman_kac(
    warped_space: WarpedSpaceTime,
    attention_trajectory: NDArray[np.float64],
    integration_steps: int = 16,
) -> PolyakovAction:
    r"""
    **Último método de la Fase 3** (y del funtor completo). Calcula la
    Acción de Polyakov y decide el veto de horizonte de eventos.

    Dos modos de entrada, distinguidos por rango de `attention_trajectory`:

      • **1D** $(n,)$: velocidad constante $\dot\gamma$. La integral
        $\int_0^1 d\tau = 1$ es exacta; el error de discretización es
        **demostrablemente cero** (no un `0.0` supuesto como en v2).

      • **2D** $(m, n)$, $m\ge 3$: trayectoria muestreada $\dot\gamma(\tau_i)$.
        Se aplica Simpson compuesto real y se estima el error por
        **extrapolación de Richardson**, comparando contra Simpson de
        media resolución (factor de reducción $15$, propio del orden $O(h^4)$
        de Simpson).

    Decisión de veto (álgebra de Boole explícita, sin ambigüedad):
    $$ V_{\text{acción}} \equiv S_E > S_{crit} = -\hbar_{eff}\ln(\text{tol}) $$
    $$ V_{\text{curvatura}} \equiv K_{\max} > K_{crit} $$
    $$ \text{Atrapado} = V_{\text{acción}} \lor V_{\text{curvatura}} $$

    La comparación $V_{\text{acción}}$ se realiza en **espacio logarítmico**
    (sobre $S_E$, no sobre $\Psi$), por lo que es exacta incluso cuando
    $\Psi\to 0$ produciría underflow de punto flotante.

    Raises:
        ValueError: forma de trayectoria inválida.
        GravitationalCollapseError: densidad cinética negativa detectada.
    """
    traj = np.asarray(attention_trajectory, dtype=np.float64)
    g_metric = warped_space.deformed_metric

    if traj.ndim == 1:
        kinetic = _kinetic_density(traj, g_metric)
        action = 0.5 * kinetic
        discretization_error = 0.0  # exacto: integrando constante, ∫₀¹dτ=1

    elif traj.ndim == 2:
        m = traj.shape[0]
        if m < 3:
            raise ValueError("Se requieren ≥3 muestras temporales para Simpson.")
        if (m - 1) % 2 != 0:
            traj = np.vstack([traj, traj[-1]])  # cierre par por duplicación
            m = traj.shape[0]

        f_vals = np.array([0.5 * _kinetic_density(traj[i], g_metric) for i in range(m)])
        h = 1.0 / (m - 1)
        action = _simpson_composite(f_vals, h)

        if m >= 5 and (m - 1) % 4 == 0:
            action_coarse = _simpson_composite(f_vals[::2], 2.0 * h)
            discretization_error = abs(action - action_coarse) / 15.0
        else:
            discretization_error = float("nan")  # resolución insuficiente para Richardson

    else:
        raise ValueError(
            "attention_trajectory debe ser 1D (velocidad constante) "
            "o 2D (trayectoria muestreada)."
        )

    # ---- Veto dual en espacio logarítmico (Boolean OR explícito) ----
    critical_action = -GravitationalConstants.HBAR_EFF * math.log(
        GravitationalConstants.SCHWARZSCHILD_TOLERANCE
    )
    action_veto = action > critical_action
    curvature_veto = (
        warped_space.max_sectional_curvature
        > GravitationalConstants.CRITICAL_SECTIONAL_CURVATURE
    )
    is_trapped = bool(action_veto or curvature_veto)

    exponent = -action / GravitationalConstants.HBAR_EFF
    amplitude = math.exp(exponent) if exponent > -700.0 else 0.0

    return PolyakovAction(
        action_integral=action,
        feynman_amplitude=amplitude,
        is_trapped=is_trapped,
        action_veto=action_veto,
        curvature_veto=curvature_veto,
        discretization_error=discretization_error,
    )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: GRAVITATIONAL SHIELD FUNCTOR                        ║
# ║   Composición categórica F = C3 ∘ C2 ∘ C1                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class GravitationalShieldFunctor(Morphism):
    r"""
    Funtor Endomórfico $F: \text{WISDOM} \to \text{WISDOM}$.

    Acopla la masa logística al tensor métrico y aniquila probabilísticamente
    las intenciones estocásticas que intentan escapar de los nodos críticos,
    mediante la composición estricta:

        Fase 1 → _acquire_effective_mass       : Costo ⟶ MassAcquisitionResult
        Fase 2 → _deform_metric_tensor         : MassAcquisitionResult ⟶ WarpedSpaceTime
        Fase 3 → _evaluate_feynman_kac         : WarpedSpaceTime ⟶ PolyakovAction

    El caché de la métrica base (`BaseMetricCache`) se construye **una sola
    vez** en `__init__`, portando la garantía teorémica de Sylvester para
    toda la vida del funtor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric_cache: BaseMetricCache = _build_base_metric_cache(
            np.asarray(G_PHYSICS, dtype=np.float64)
        )

    @staticmethod
    def _extract_polaron_fields(
        polaron: PolaronLike,
    ) -> Tuple[float, float, Optional[float]]:
        r"""
        Extractor único y tipado (sustituye la cadena de `hasattr` dispersa
        de v2), conforme al contrato formal `PolaronLike`.
        """
        cost = abs(getattr(polaron, "inertial_mass", 0.0))
        alpha = abs(getattr(polaron, "volatility_alpha", 0.0))
        raw_coupling = getattr(polaron, "frohlich_coupling", None)
        coupling = abs(raw_coupling) if raw_coupling is not None else None
        return cost, alpha, coupling

    def enforce_gravitational_attractor(
        self,
        polaron: PolaronCartridge,
        node_index: int,
        llm_attention_vector: NDArray[np.float64],
        integration_steps: int = 16,
    ) -> PolyakovAction:
        r"""
        Ejerce el colapso absoluto sobre el vector (o trayectoria) de
        atención del Agente, ejecutando la composición $F = C_3\circ C_2\circ C_1$.

        Args:
            polaron: Cartucho TOON con masa inercial y acoplamiento de Fröhlich.
            node_index: Índice de la dimensión logística en G_PHYSICS.
            llm_attention_vector: vector 1D (velocidad constante) o matriz 2D
                (trayectoria muestreada) de la atención del agente.
            integration_steps: reservado para compatibilidad; el número real
                de muestras se infiere de `llm_attention_vector` si es 2D.

        Returns:
            PolyakovAction con el dictamen de atrapamiento y su desglose booleano.

        Raises:
            EventHorizonViolation: si `action_veto ∨ curvature_veto` es verdadero.
        """
        logger.info(f"Iniciando secuencia gravitacional para nodo {node_index}")

        # FASE 1 — Adquisición de masa inercial efectiva
        cost, alpha, coupling = self._extract_polaron_fields(polaron)
        mass_result = _acquire_effective_mass(cost, alpha, coupling)
        logger.debug(f"Fase 1 completada: m** = {mass_result.effective_mass:.6e}")

        # FASE 2 — Deformación del tensor métrico (entrada = salida de Fase 1)
        warped_space = _deform_metric_tensor(self._metric_cache, mass_result, node_index)
        logger.debug(
            f"Fase 2 completada: K_max = {warped_space.max_sectional_curvature:.4e}"
        )

        # FASE 3 — Evaluación de la acción y colapso cuántico (entrada = salida de Fase 2)
        action_result = _evaluate_feynman_kac(
            warped_space=warped_space,
            attention_trajectory=llm_attention_vector,
            integration_steps=integration_steps,
        )
        logger.debug(
            f"Fase 3 completada: S={action_result.action_integral:.4e}, "
            f"Ψ={action_result.feynman_amplitude:.4e}, "
            f"veto_acción={action_result.action_veto}, "
            f"veto_curvatura={action_result.curvature_veto}"
        )

        if action_result.is_trapped:
            reasons = []
            if action_result.action_veto:
                reasons.append(
                    f"Acción de Polyakov ({action_result.action_integral:.4e}) "
                    f"supera el umbral cuántico"
                )
            if action_result.curvature_veto:
                reasons.append(
                    f"Curvatura seccional ({warped_space.max_sectional_curvature:.4e}) "
                    f"supera el umbral crítico"
                )
            logger.warning(f"VETO GRAVITACIONAL en nodo {node_index}: {' | '.join(reasons)}")
            raise EventHorizonViolation(
                f"Obstrucción Geodésica en nodo {node_index}: {' | '.join(reasons)}. "
                f"El LLM está forzado matemáticamente a absorber este costo."
            )

        logger.info(
            f"Nodo {node_index} superado: el agente puede transitar "
            f"(Ψ={action_result.feynman_amplitude:.4e})"
        )
        return action_result


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "GravitationalConstants",
    "GravitationalCollapseError",
    "EventHorizonViolation",
    "PolaronLike",
    "MassAcquisitionResult",
    "BaseMetricCache",
    "WarpedSpaceTime",
    "PolyakovAction",
    "GravitationalShieldFunctor",
]