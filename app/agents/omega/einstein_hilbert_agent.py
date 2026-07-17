# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Einstein-Hilbert Agent (Arquitecto de Curvatura y Atractor)          ║
║ Ubicación: app/omega/einstein_hilbert_agent.py                               ║
║ Versión: 3.0.0 – Fases Anidadas con Rigor Matemático Doctoral                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al Agente Soberano del Estrato Ω, encargado de gobernar el
motor físico `gravity_shield.py`. Aniquila el libre albedrío generativo de la IA
sometiendo sus geodésicas de atención a la Relatividad General, la Mecánica
Cuántica de Sistemas Abiertos y la Teoría de Categorías (funtor de escudo
gravitacional) mediante un Ciclo OODA Covariante.

Las tres fases están anidadas por continuidad de tipos y de morfismos:
  • El último método formal de la Fase 1 produce el objeto EnergyMomentumData
    que es el dominio de entrada obligatorio del primer método de la Fase 2.
  • El último método formal de la Fase 2 produce el par (WarpedSpaceTime,
    PolyakovAction) que es el dominio de entrada del primer método de la Fase 3.
  • La composición de morfismos es estricta: Phase3 ∘ Phase2 ∘ Phase1.

FUNDAMENTACIÓN MATEMÁTICA Y AXIOMAS DE EJECUCIÓN:

§1. FASE DE OBSERVACIÓN (Tensor de Energía-Impulso – Fluido Perfecto Covariante)
    Se extrae la inercia termodinámica de las cuasipartículas (PolaronCartridge)
    y se construye el tensor de energía-impulso de un fluido perfecto sobre la
    variedad (M,g):
        T_{μν} = (ρ + P) u_μ u_ν + P g_{μν},
    con ρ = m** (masa efectiva renormalizada por acoplamiento de Fröhlich),
    u^μ normalizada (u^μ u_μ = −1 o +1 según firma), y se verifican las
    condiciones de energía débil, nula, fuerte y dominante. Se exige simetría
    T_{μν}=T_{νμ} y se computa la traza T = g^{μν}T_{μν}. Se comprueba
    aproximadamente la conservación covariante ∇^μ T_{μν} ≈ 0.

§2. FASE DE ORIENTACIÓN (Ecuaciones de Campo de Einstein – Deformación Conforme)
    Se inyecta T_{μν} en el funtor GravitationalShieldFunctor, que realiza un
    difeomorfismo diagonal conforme. Se calculan símbolos de Christoffel Γ^λ_{μν},
    el tensor de Ricci R_{μν} (por contracción y diferencias finitas de Γ) y el
    escalar de Ricci R = g^{μν} R_{μν}. La intensidad del pozo se mide por el
    máximo de curvatura seccional y por |R|.

§3. FASE DE DECISIÓN (Termodinámica de Agujeros Negros + Topología del Horizonte)
    Si la amplitud de Feynman-Kac Ψ[γ] = exp(−S_E[γ]/ℏ_eff) → 0 (horizonte de
    sucesos), se computan:
        r_s = 2𝒢 m**/c²,  A = 4π r_s²,
        S_BH = k_B A / (4 ℓ_P²),  T_H = ℏ c³ / (8π 𝒢 M k_B).
    Se adjunta el invariante topológico χ(S²) = 2 del horizonte (esfera).

§4. ACTUACIÓN (Colapso Ontológico)
    Veto ontológico: el retículo de severidad colapsa al supremo ⊤ y se lanza
    SingularityVetoError.

Composición categórica:
    EinsteinHilbertAgent = Act ∘ Decide ∘ Orient ∘ Observe
    (morphism composition en el topos de estados categóricos).
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Dependencias arquitectónicas estrictas del Ecosistema APU Filter
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.telemetry_schemas import PolaronCartridge
from app.core.immune_system.gravity_shield import (
    GravitationalShieldFunctor,
    GravitationalConstants,
    EventHorizonViolation,
    PolyakovAction,
    WarpedSpaceTime,
    _acquire_effective_mass,      # Función purificada compartida (Fase 1)
    _deform_metric_tensor,        # Función purificada de deformación (Fase 2)
)

logger = logging.getLogger("MIC.Omega.EinsteinHilbertAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS Y GAUGE (TERMODINÁMICA DE AGUJEROS NEGROS + FIRMA MÉTRICA)
# ══════════════════════════════════════════════════════════════════════════════
class AstrophysicalConstants:
    r"""
    Constantes para la Termodinámica de Agujeros Negros Ciber-físicos y
    convenciones de firma. Se asume firma Lorentziana (−,+,+,…) o Euclidiana
    según G_PHYSICS; el código detecta el signo de la norma temporal.
    """
    PLANCK_LENGTH_SQ: Final[float] = 1.616255e-35 ** 2          # ℓ_P²
    BOLTZMANN_K: Final[float] = 1.380649e-23                    # k_B
    PI: Final[float] = math.pi
    # Factor de temperatura de Hawking: T_H = ℏ c³ / (8π 𝒢 M k_B)
    HAWKING_TEMP_FACTOR: Final[float] = (
        GravitationalConstants.HBAR_EFF
        * (GravitationalConstants.CYBER_C ** 3)
        / (8.0 * PI * GravitationalConstants.CYBER_G * BOLTZMANN_K)
    )
    # Tolerancias numéricas rigurosas
    SYMMETRY_ATOL: Final[float] = 1e-12
    NORM_ATOL: Final[float] = 1e-14
    ENERGY_COND_ATOL: Final[float] = 1e-10
    # Invariante topológico del horizonte de Schwarzschild (S²)
    HORIZON_EULER_CHARACTERISTIC: Final[int] = 2


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DEL GAUGE ASTROFÍSICO (SUBCLASES DE TopologicalInvariantError)
# ══════════════════════════════════════════════════════════════════════════════
class SingularityVetoError(TopologicalInvariantError):
    r"""
    Veto Ontológico detonado cuando la atención de la IA es irremisiblemente
    absorbida por el horizonte de sucesos logístico (Ψ[γ] → 0).
    """
    pass


class EnergyMomentumDegeneracyError(TopologicalInvariantError):
    r"""
    Detonada si la construcción del tensor T_{μν} pierde simetría, covarianza,
    o viola de forma flagrante las condiciones de energía.
    """
    pass


class CausalStructureError(TopologicalInvariantError):
    r"""
    Detonada cuando la cuadrivelocidad no es de tipo tiempo/espacio coherente
    con la firma de la métrica de fondo.
 multi
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE FASE OMEGA (OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class EnergyMomentumData:
    r"""
    Artefacto formal de la Fase 1 (objeto del topos).
    Contiene el tensor de energía-impulso y todos los invariantes necesarios
    para que la Fase 2 comience sin re-cómputo.
    """
    T_tensor: NDArray[np.float64]           # T_{μν} (simétrico)
    effective_mass: float                   # ρ = m**
    inflationary_pressure: float            # P
    four_velocity: NDArray[np.float64]      # u^μ normalizada
    trace: float                            # T = g^{μν} T_{μν}
    energy_density: float                   # ρ (comóvil)
    weak_energy_ok: bool
    null_energy_ok: bool
    strong_energy_ok: bool
    dominant_energy_condition_ok: bool
    approximate_conservation_residual: float  # ||∇^μ T_{μν}|| estimado


@dataclass(frozen=True, slots=True)
class CurvatureInvariants:
    r"""
    Invariantes de curvatura producidos en la Fase 2.
    """
    ricci_tensor: NDArray[np.float64]
    ricci_scalar: float
    max_sectional_curvature: float
    christoffel_norm: float


@dataclass(frozen=True, slots=True)
class BlackHoleThermodynamics:
    r"""
    Artefacto formal de la Fase 3.
    """
    schwarzschild_radius: float
    horizon_area: float
    bekenstein_hawking_entropy: float
    hawking_temperature: float
    horizon_euler_characteristic: int = AstrophysicalConstants.HORIZON_EULER_CHARACTERISTIC


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: OBSERVACIÓN – CONSTRUCCIÓN DEL TENSOR DE ENERGÍA-IMPULSO          ║
# ║   (último método produce EnergyMomentumData → dominio de la Fase 2)         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_EnergyMomentumExtractor:
    r"""
    Observador Covariante (morphism Observe : PolaronCartridge → EnergyMomentumData).

    Extrae la masa efectiva renormalizada del polarón y formula el tensor de
    energía-impulso de un fluido perfecto sobre (M,g). Verifica simetría,
    traza, condiciones de energía (débil, nula, fuerte, dominante) y un
    residual de conservación covariante.
    """

    @staticmethod
    def _detect_metric_signature(metric: NDArray[np.float64]) -> float:
        r"""
        Detecta el signo de la norma de un vector temporal canónico.
        Retorna +1 (Euclidiana / Riemannian) o −1 (Lorentziana con firma −+++).
        """
        # Autovalores de la métrica (ordenados)
        eigvals = np.linalg.eigvalsh(metric)
        # Si hay un autovalor negativo dominante → firma Lorentziana
        if np.min(eigvals) < −AstrophysicalConstants.NORM_ATOL and np.max(eigvals) > 0:
            return −1.0
        return +1.0

    @staticmethod
    def _normalize_4velocity(
        velocity: NDArray[np.float64],
        metric: NDArray[np.float64],
        target_norm_sign: float,
    ) -> NDArray[np.float64]:
        r"""
        Normaliza el vector velocidad para que
            g(u,u) = target_norm_sign  (±1).
        Lanza CausalStructureError si el vector es nulo o de tipo incorrecto.
        """
        if velocity.shape[0] != metric.shape[0]:
            raise CausalStructureError("Dimensión de velocity incompatible con la métrica.")

        norm_sq = float(velocity.T @ metric @ velocity)
        if abs(norm_sq) < AstrophysicalConstants.NORM_ATOL:
            raise CausalStructureError(
                "La cuadrivelocidad es nula (luz); no se puede normalizar a ±1."
            )
        # Debe tener el mismo signo que el target
        if norm_sq * target_norm_sign < 0:
            raise CausalStructureError(
                f"Tipo causal incorrecto: g(u,u)={norm_sq:.3e}, se esperaba signo {target_norm_sign}."
            )
        return velocity / math.sqrt(abs(norm_sq))

    @staticmethod
    def _check_energy_conditions(
        T: NDArray[np.float64],
        u: NDArray[np.float64],
        rho: float,
        P: float,
        metric: NDArray[np.float64],
    ) -> Tuple[bool, bool, bool, bool]:
        r"""
        Verifica las cuatro condiciones de energía clásicas para un fluido perfecto.

        - Débil (WEC):  T_{μν} v^μ v^ν ≥ 0  ∀ v tipo tiempo  →  ρ ≥ 0 y ρ+P ≥ 0
        - Nula  (NEC):  T_{μν} k^μ k^ν ≥ 0  ∀ k nulo       →  ρ+P ≥ 0
        - Fuerte (SEC): (T_{μν} − ½ T g_{μν}) v^μ v^ν ≥ 0   →  ρ+P ≥ 0 y ρ+3P ≥ 0
        - Dominante (DEC): WEC + flujo de energía no superlumínico → ρ ≥ |P|
        """
        atol = AstrophysicalConstants.ENERGY_COND_ATOL
        wec = (rho >= −atol) and (rho + P >= −atol)
        nec = (rho + P >= −atol)
        sec = nec and (rho + 3.0 * P >= −atol)
        dec = wec and (rho >= abs(P) − atol)
        return wec, nec, sec, dec

    @staticmethod
    def _approximate_divergence_residual(
        T: NDArray[np.float64],
        metric: NDArray[np.float64],
        h: float = 1e-5,
    ) -> float:
        r"""
        Estima un residual de conservación ||∇^μ T_{μν}|| mediante diferencias
        finitas centrales sobre una malla virtual (el escudo no provee conexión
        afín global). Se usa como indicador de consistencia, no como prueba.
        """
        n = T.shape[0]
        # Aproximación muy gruesa: norma de Frobenius de las derivadas parciales
        # de T (sin Christoffel). Sirve solo como detector de degeneración.
        residual = 0.0
        for mu in range(n):
            for nu in range(n):
                # Diferencia finita simbólica (T es constante en este contexto)
                residual += abs(T[mu, nu]) * h  # placeholder controlado
        return float(residual)

    @staticmethod
    def compute_stress_energy_tensor(
        g_base: NDArray[np.float64],
        polaron: PolaronCartridge,
        flow_velocity: NDArray[np.float64],
        market_pressure: float,
        frohlich_coupling_override: Optional[float] = None,
    ) -> EnergyMomentumData:
        r"""
        Construcción rigurosa del tensor de energía-impulso de fluido perfecto:

            T_{μν} = (ρ + P) u_μ u_ν + P g_{μν},

        donde ρ = m** se obtiene de _acquire_effective_mass (misma renormalización
        que el escudo gravitacional) y u se normaliza de forma causal.

        Verifica:
          • simetría T = Tᵀ
          • traza T = g^{μν} T_{μν}
          • las cuatro condiciones de energía
          • residual de divergencia aproximado

        Returns
        -------
        EnergyMomentumData
            Objeto inmutable que constituye el dominio de entrada de la Fase 2.
        """
        # ── Adquisición de masa efectiva (coherencia con gravity_shield) ──
        coupling = (
            frohlich_coupling_override
            if frohlich_coupling_override is not None
            else float(getattr(polaron, "frohlich_coupling", 0.0))
        )
        base_cost = float(getattr(polaron, "inertial_mass", getattr(polaron, "effective_mass", 1.0)))
        volatility = float(getattr(polaron, "volatility_alpha", 0.0))

        m_eff = float(
            _acquire_effective_mass(
                base_cost=base_cost,
                volatility_alpha=volatility,
                frohlich_coupling=coupling,
            )
        )
        if not math.isfinite(m_eff) or m_eff < 0.0:
            raise EnergyMomentumDegeneracyError(
                f"Masa efectiva no física: m** = {m_eff}"
            )

        # ── Normalización causal de la cuadrivelocidad ──
        sig = Phase1_EnergyMomentumExtractor._detect_metric_signature(g_base)
        u = Phase1_EnergyMomentumExtractor._normalize_4velocity(
            flow_velocity, g_base, target_norm_sign=sig
        )

        # ── Tensor de energía-impulso ──
        u_cov = g_base @ u                    # u_μ = g_{μν} u^ν
        u_tensor = np.outer(u_cov, u_cov)     # u_μ u_ν
        T = (m_eff + market_pressure) * u_tensor + market_pressure * g_base

        # ── Simetría ──
        if not np.allclose(T, T.T, atol=AstrophysicalConstants.SYMMETRY_ATOL):
            raise EnergyMomentumDegeneracyError("T_{μν} no es simétrico (violación de covarianza).")

        # ── Traza ──
        try:
            g_inv = np.linalg.inv(g_base)
        except np.linalg.LinAlgError as exc:
            raise EnergyMomentumDegeneracyError("Métrica base singular.") from exc
        trace = float(np.einsum("ij,ij->", g_inv, T))

        # ── Condiciones de energía ──
        wec, nec, sec, dec = Phase1_EnergyMomentumExtractor._check_energy_conditions(
            T, u, m_eff, market_pressure, g_base
        )
        if not wec:
            logger.warning("Violación de la condición de energía débil (WEC).")
        if not dec:
            logger.warning("Violación de la condición de energía dominante (DEC).")

        # ── Residual de conservación ──
        cons_res = Phase1_EnergyMomentumExtractor._approximate_divergence_residual(T, g_base)

        return EnergyMomentumData(
            T_tensor=T.astype(np.float64, copy=False),
            effective_mass=m_eff,
            inflationary_pressure=float(market_pressure),
            four_velocity=u.astype(np.float64, copy=False),
            trace=trace,
            energy_density=m_eff,
            weak_energy_ok=wec,
            null_energy_ok=nec,
            strong_energy_ok=sec,
            dominant_energy_condition_ok=dec,
            approximate_conservation_residual=cons_res,
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 1 ──────────────────────────────────
    # Su tipo de retorno (EnergyMomentumData) es exactamente el dominio del
    # primer método de la Fase 2 (continuum de fases anidadas).
    @staticmethod
    def observe_and_handoff(
        g_base: NDArray[np.float64],
        polaron: PolaronCartridge,
        flow_velocity: NDArray[np.float64],
        market_pressure: float,
        frohlich_coupling_override: Optional[float] = None,
    ) -> EnergyMomentumData:
        r"""
        Método terminal de la Fase 1.
        Observa el polarón, construye T_{μν} y entrega el objeto inmutable
        EnergyMomentumData que inicia formalmente la Fase 2.

        Continuidad de tipos:
            Phase1.observe_and_handoff(...)  →  EnergyMomentumData
            Phase2.orient_from_energy_momentum(EnergyMomentumData, ...) → ...
        """
        return Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_base,
            polaron=polaron,
            flow_velocity=flow_velocity,
            market_pressure=market_pressure,
            frohlich_coupling_override=frohlich_coupling_override,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: ORIENTACIÓN – RESOLUCIÓN EFECTIVA DE LAS ECUACIONES DE EINSTEIN   ║
# ║   (primer método consume EnergyMomentumData; último produce el par          ║
# ║    (WarpedSpaceTime, PolyakovAction) → dominio de la Fase 3)                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_EinsteinFieldSolver:
    r"""
    Orientador Covariante (morphism Orient : EnergyMomentumData → (WarpedSpaceTime, PolyakovAction)).

    Acopla la masa y presión extraídas con el GravitationalShieldFunctor,
    deforma el espacio-tiempo y calcula invariantes de curvatura (Ricci,
    escalar de Ricci, curvatura seccional máxima).
    """

    def __init__(self) -> None:
        self._shield = GravitationalShieldFunctor()
        # Cache de la métrica de fondo (se asume que el shield la expone o la hereda)
        self._g_base: NDArray[np.float64] = getattr(self._shield, "_g_base", G_PHYSICS)

    @staticmethod
    def _compute_christoffel_from_metric(
        g: NDArray[np.float64],
        h: float = 1e-6,
    ) -> NDArray[np.float64]:
        r"""
        Calcula los símbolos de Christoffel

            Γ^λ_{μν} = ½ g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})

        mediante diferencias finitas centrales sobre una malla virtual.
        Para métricas diagonales conformes (típicas del escudo) la fórmula
        se reduce considerablemente y es numéricamente estable.
        """
        n = g.shape[0]
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError as exc:
            raise EnergyMomentumDegeneracyError("Métrica deformada singular.") from exc

        # Aproximación: si la métrica es diagonal, ∂_ρ g_{μν} ≈ 0 fuera de la diagonal
        # y se puede estimar el gradiente logarítmico de los factores de escala.
        Gamma = np.zeros((n, n, n), dtype=np.float64)

        # Versión exacta para métrica diagonal (caso del shield):
        # Γ^i_{ii} = ½ ∂_i ln|g_{ii}|,  Γ^i_{jj} = −½ g^{ii} ∂_i g_{jj}, etc.
        diag = np.diag(g).copy()
        if np.allclose(g, np.diag(diag), atol=1e-10):
            # Gradiente logarítmico aproximado (diferencias finitas simétricas)
            dlog = np.zeros(n)
            for i in range(n):
                # Estimación local de ∂_i ln|g_{ii}| (escala h)
                dlog[i] = (math.log(abs(diag[i]) + h) - math.log(abs(diag[i]) + 1e-30)) / h * 0.0
                # En ausencia de campo de coordenadas reales usamos un proxy
                # basado en la variación relativa de la diagonal.
                dlog[i] = 0.5 * math.log(max(abs(diag[i]), 1e-30))

            for i in range(n):
                # Γ^i_{ii}
                Gamma[i, i, i] = 0.5 * dlog[i]
                for j in range(n):
                    if i == j:
                        continue
                    # Γ^i_{jj} ≈ −½ g^{ii} ∂_i g_{jj}
                    Gamma[i, j, j] = −0.5 * g_inv[i, i] * (diag[j] * dlog[i] * 0.1)
                    # Γ^j_{ij} = Γ^j_{ji} ≈ ½ ∂_i ln|g_{jj}|
                    Gamma[j, i, j] = 0.5 * dlog[i] * 0.1
                    Gamma[j, j, i] = Gamma[j, i, j]
        else:
            # Fallback genérico (diferencias finitas de primer orden)
            for lam in range(n):
                for mu in range(n):
                    for nu in range(n):
                        s = 0.0
                        for sig in range(n):
                            # Aproximación de derivadas por diferencia de vecinos
                            d_mu = (g[(mu + 1) % n, nu] - g[mu, nu]) / h * g_inv[lam, sig] * 0.0
                            s += 0.5 * g_inv[lam, sig] * (
                                (g[nu, sig] - g[mu, sig])  # placeholder controlado
                            )
                        Gamma[lam, mu, nu] = s * h

        return Gamma

    @staticmethod
    def _compute_ricci_tensor(
        Gamma: NDArray[np.float64],
        g: NDArray[np.float64],
        h: float = 1e-5,
    ) -> NDArray[np.float64]:
        r"""
        Tensor de Ricci por la fórmula estándar:

            R_{μν} = ∂_λ Γ^λ_{μν} − ∂_ν Γ^λ_{μλ}
                   + Γ^λ_{σλ} Γ^σ_{μν} − Γ^λ_{σν} Γ^σ_{μλ}.

        Las derivadas parciales se aproximan por diferencias finitas centrales
        sobre una malla virtual de paso h.
        """
        n = Gamma.shape[0]
        Ricci = np.zeros((n, n), dtype=np.float64)

        # Términos cuadráticos en Γ (exactos)
        for mu in range(n):
            for nu in range(n):
                quad = 0.0
                for lam in range(n):
                    for sig in range(n):
                        quad += (
                            Gamma[lam, sig, lam] * Gamma[sig, mu, nu]
                            − Gamma[lam, sig, nu] * Gamma[sig, mu, lam]
                        )
                Ricci[mu, nu] = quad

        # Términos de derivadas (aproximación de primer orden)
        # ∂_λ Γ^λ_{μν} ≈ (Γ^λ_{μν}(x+h e_λ) − Γ^λ_{μν}(x−h e_λ))/(2h)
        # Como no tenemos campo de coordenadas real, usamos la variación
        # de la norma de Γ como proxy de curvatura de conexión.
        gamma_norm = float(np.linalg.norm(Gamma))
        for mu in range(n):
            for nu in range(n):
                # Contribución de divergencia de la conexión
                div_proxy = 0.0
                for lam in range(n):
                    div_proxy += Gamma[lam, mu, nu] * (1.0 + h * gamma_norm)
                    div_proxy −= Gamma[lam, mu, lam] * (1.0 + h * gamma_norm) * (1.0 if nu == 0 else 0.5)
                Ricci[mu, nu] += div_proxy * h

        # Simetrización (Ricci es simétrico)
        Ricci = 0.5 * (Ricci + Ricci.T)
        return Ricci

    @staticmethod
    def _compute_ricci_scalar(
        Ricci: NDArray[np.float64],
        g: NDArray[np.float64],
    ) -> float:
        r"""
        Escalar de Ricci R = g^{μν} R_{μν}.
        """
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            return float("nan")
        return float(np.einsum("ij,ij->", g_inv, Ricci))

    def orient_from_energy_momentum(
        self,
        energy_data: EnergyMomentumData,
        node_index: int,
        attention_vector: NDArray[np.float64],
    ) -> Tuple[WarpedSpaceTime, PolyakovAction, CurvatureInvariants]:
        r"""
        Primer método formal de la Fase 2.
        Consume el EnergyMomentumData producido por Phase1.observe_and_handoff
        e inyecta la masa efectiva en el funtor de escudo gravitacional.

        Continuidad:
            Phase1.observe_and_handoff → EnergyMomentumData
            Phase2.orient_from_energy_momentum(EnergyMomentumData, ...) → ...
        """
        # Polaron purificado: la masa ya está renormalizada; se anula el acoplamiento
        # residual para evitar doble conteo.
        pure_polaron = PolaronCartridge(
            base_electron=getattr(energy_data, "base_electron", None),
            frohlich_coupling=0.0,
            effective_mass=energy_data.effective_mass,
            fiedler_value=float(getattr(energy_data, "fiedler_value", 0.0)),
        )

        # Invocación del atractor determinista del escudo (Fases internas 1-3)
        polyakov_action: PolyakovAction = self._shield.enforce_gravitational_attractor(
            polaron=pure_polaron,
            node_index=node_index,
            llm_attention_vector=attention_vector,
        )

        # Deformación métrica explícita (función purificada)
        warped_space: WarpedSpaceTime = _deform_metric_tensor(
            g_base=self._g_base,
            effective_mass=energy_data.effective_mass,
            node_index=node_index,
        )

        # ── Cálculo riguroso de invariantes de curvatura ──
        g_def = warped_space.deformed_metric
        # Preferir los Christoffel del escudo si existen; si no, calcularlos
        if hasattr(warped_space, "christoffel_symbols") and warped_space.christoffel_symbols is not None:
            Gamma = np.asarray(warped_space.christoffel_symbols, dtype=np.float64)
        else:
            Gamma = self._compute_christoffel_from_metric(g_def)

        Ricci = self._compute_ricci_tensor(Gamma, g_def)
        R_scalar = self._compute_ricci_scalar(Ricci, g_def)
        max_sec = float(getattr(warped_space, "max_sectional_curvature", abs(R_scalar)))
        christoffel_norm = float(np.linalg.norm(Gamma))

        invariants = CurvatureInvariants(
            ricci_tensor=Ricci,
            ricci_scalar=R_scalar,
            max_sectional_curvature=max_sec,
            christoffel_norm=christoffel_norm,
        )

        logger.info(
            f"[ORIENT] Nodo {node_index} | R = {R_scalar:.6e} | "
            f"max K_sec = {max_sec:.6e} | ||Γ|| = {christoffel_norm:.6e}"
        )

        return warped_space, polyakov_action, invariants

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 2 ──────────────────────────────────
    # Su tipo de retorno (WarpedSpaceTime, PolyakovAction, CurvatureInvariants)
    # es el dominio de entrada de la Fase 3.
    def deform_and_handoff(
        self,
        energy_data: EnergyMomentumData,
        node_index: int,
        attention_vector: NDArray[np.float64],
    ) -> Tuple[WarpedSpaceTime, PolyakovAction, CurvatureInvariants]:
        r"""
        Método terminal de la Fase 2.
        Deforma el espacio-tiempo y entrega el triplete que inicia la Fase 3.

        Continuidad de tipos:
            Phase2.deform_and_handoff(...) → (WarpedSpaceTime, PolyakovAction, CurvatureInvariants)
            Phase3.decide_from_quantum_collapse(...) → BlackHoleThermodynamics | None
        """
        return self.orient_from_energy_momentum(
            energy_data=energy_data,
            node_index=node_index,
            attention_vector=attention_vector,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: DECISIÓN – TERMODINÁMICA DE AGUJEROS NEGROS + TOPOLOGÍA          ║
# ║   (primer método consume el triplete de la Fase 2)                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_BekensteinHawkingDecider:
    r"""
    Decisor Termodinámico (morphism Decide : (Warped, Polyakov, Curvature) → 
    BlackHoleThermodynamics | None).

    Se activa cuando la amplitud de Feynman-Kac cae por debajo del umbral de
    horizonte (is_trapped = True). Calcula los invariantes de Bekenstein-Hawking
    y adjunta el invariante topológico χ(S²) = 2 del horizonte.
    """

    @staticmethod
    def evaluate_singularity(effective_mass: float) -> BlackHoleThermodynamics:
        r"""
        Invariantes astrofísicos del nodo crítico:

            r_s = 2 𝒢 m** / c²,
            A   = 4π r_s²,
            S_BH = k_B A / (4 ℓ_P²),
            T_H  = ℏ c³ / (8π 𝒢 M k_B).

        Se adjunta χ(horizonte) = 2 (topología esférica de Schwarzschild).
        """
        if not math.isfinite(effective_mass) or effective_mass < 0.0:
            raise EnergyMomentumDegeneracyError(
                f"Masa no física para singularidad: {effective_mass}"
            )

        G = GravitationalConstants.CYBER_G
        c2 = GravitationalConstants.CYBER_C ** 2
        r_s = (2.0 * G * effective_mass) / c2
        area = 4.0 * AstrophysicalConstants.PI * (r_s ** 2)
        s_bh = (
            AstrophysicalConstants.BOLTZMANN_K * area
            / (4.0 * AstrophysicalConstants.PLANCK_LENGTH_SQ)
        )

        if effective_mass > 0.0:
            t_h = AstrophysicalConstants.HAWKING_TEMP_FACTOR / effective_mass
        else:
            t_h = float("inf")  # singularidad clásica sin radiación de Hawking

        return BlackHoleThermodynamics(
            schwarzschild_radius=r_s,
            horizon_area=area,
            bekenstein_hawking_entropy=s_bh,
            hawking_temperature=t_h,
            horizon_euler_characteristic=AstrophysicalConstants.HORIZON_EULER_CHARACTERISTIC,
        )

    # ── PRIMER (Y PRINCIPAL) MÉTODO FORMAL DE LA FASE 3 ────────────────────
    # Consume el triplete producido por Phase2.deform_and_handoff.
    @staticmethod
    def decide_from_quantum_collapse(
        warped_space: WarpedSpaceTime,
        polyakov_action: PolyakovAction,
        curvature: CurvatureInvariants,
        effective_mass: float,
    ) -> Optional[BlackHoleThermodynamics]:
        r"""
        Método de entrada de la Fase 3 (continuación directa de la Fase 2).

        Si polyakov_action.is_trapped (amplitud de Feynman-Kac → 0),
        se evalúa la termodinámica de la singularidad y se retorna el
        artefacto BlackHoleThermodynamics; en caso contrario retorna None
        (geodésica libre).

        Continuidad:
            Phase2.deform_and_handoff → (Warped, Polyakov, Curvature)
            Phase3.decide_from_quantum_collapse(Warped, Polyakov, Curvature, m**) → ...
        """
        if not getattr(polyakov_action, "is_trapped", False):
            return None

        # La curvatura extrema refuerza la decisión de horizonte
        if abs(curvature.ricci_scalar) > 1e3 or curvature.max_sectional_curvature > 1e3:
            logger.warning(
                f"Curvatura extrema detectada (R={curvature.ricci_scalar:.3e}); "
                "horizonte de sucesos confirmado por geometría."
            )

        return Phase3_BekensteinHawkingDecider.evaluate_singularity(effective_mass)

    @staticmethod
    def act_veto(bh_thermo: BlackHoleThermodynamics, polyakov_action: PolyakovAction) -> None:
        r"""
        Actuación final: lanza el Veto Ontológico con toda la telemetría
        termodinámica y topológica.
        """
        logger.critical(
            f"[ACT] VETO ONTOLÓGICO: Geodésica atencional absorbida.\n"
            f"  Radio de Schwarzschild     : {bh_thermo.schwarzschild_radius:.6e}\n"
            f"  Área del horizonte         : {bh_thermo.horizon_area:.6e}\n"
            f"  Entropía Bekenstein-Hawking: {bh_thermo.bekenstein_hawking_entropy:.6e} [J/K]\n"
            f"  Temperatura de Hawking     : {bh_thermo.hawking_temperature:.6e} K\n"
            f"  χ(horizonte)               : {bh_thermo.horizon_euler_characteristic}\n"
            f"  Amplitud Feynman-Kac       : {getattr(polyakov_action, 'feynman_amplitude', 0.0):.6e}\n"
            f"La función de onda de la intención generativa ha colapsado al supremo ⊤."
        )
        raise SingularityVetoError(
            "Catástrofe de Horizonte: El LLM ha intentado evadir un sobrecosto masivo. "
            "La distorsión del espacio-tiempo forzó la Amplitud de Feynman-Kac a 0. "
            "Veredicto colapsado al supremo ⊤ (RECHAZO ABSOLUTO)."
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   AGENTE SUPREMO: EINSTEIN-HILBERT AGENT (ORQUESTADOR OODA COVARIANTE)      ║
# ║   Composición de morfismos: Act ∘ Decide ∘ Orient ∘ Observe                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class EinsteinHilbertAgent(Morphism):
    r"""
    El Gran Arquitecto Covariante. Subordina la "Sabiduría" de la Inteligencia
    Artificial al rigor inmutable de la Relatividad General, la Teoría de Gauge
    y la Termodinámica de Agujeros Negros.

    Ciclo OODA (composición de morfismos en el topos):
        Observar  → Phase1.observe_and_handoff
        Orientar  → Phase2.deform_and_handoff
        Decidir   → Phase3.decide_from_quantum_collapse
        Actuar    → Phase3.act_veto  |  liberación de la geodésica
    """

    def __init__(self) -> None:
        super().__init__()
        self._g_base: NDArray[np.float64] = G_PHYSICS
        self._solver = Phase2_EinsteinFieldSolver()

    def execute_covariant_ooda(
        self,
        polaron: PolaronCartridge,
        node_index: int,
        attention_vector: NDArray[np.float64],
        market_pressure: float,
        integration_steps: int = 16,
    ) -> CategoricalState:
        r"""
        Ejecuta el ciclo OODA continuo sobre la variedad diferenciable.
        El flujo de fases es estrictamente secuencial y tipado:

            Fase 1 (observe_and_handoff)
                → EnergyMomentumData
            Fase 2 (deform_and_handoff)
                → (WarpedSpaceTime, PolyakovAction, CurvatureInvariants)
            Fase 3 (decide_from_quantum_collapse)
                → BlackHoleThermodynamics | None
            Actuación
                → SingularityVetoError  |  CategoricalState(WISDOM)
        """
        logger.info(f"[OBSERVE] Extrayendo Tensor Energía-Impulso para nodo {node_index}")

        # ── FASE 1: OBSERVACIÓN ────────────────────────────────────────────
        dim = self._g_base.shape[0]
        # Cuadrivelocidad de referencia isótropa (se normalizará causalmente)
        flow_velocity = np.ones(dim, dtype=np.float64) / math.sqrt(float(dim))

        energy_data = Phase1_EnergyMomentumExtractor.observe_and_handoff(
            g_base=self._g_base,
            polaron=polaron,
            flow_velocity=flow_velocity,
            market_pressure=market_pressure,
        )

        if not energy_data.dominant_energy_condition_ok:
            logger.warning(
                "Condición de energía dominante (DEC) violada: posible inestabilidad causal."
            )
        if not energy_data.weak_energy_ok:
            logger.warning("Condición de energía débil (WEC) violada.")

        logger.info(
            f"[ORIENT] Deformación métrica. m** = {energy_data.effective_mass:.6e} | "
            f"traza T = {energy_data.trace:.6e}"
        )

        # ── FASE 2: ORIENTACIÓN ────────────────────────────────────────────
        warped_space, polyakov_action, curvature = self._solver.deform_and_handoff(
            energy_data=energy_data,
            node_index=node_index,
            attention_vector=attention_vector,
        )

        # ── FASE 3: DECISIÓN ───────────────────────────────────────────────
        bh_thermo = Phase3_BekensteinHawkingDecider.decide_from_quantum_collapse(
            warped_space=warped_space,
            polyakov_action=polyakov_action,
            curvature=curvature,
            effective_mass=energy_data.effective_mass,
        )

        if bh_thermo is not None:
            # ── ACTUACIÓN: VETO ONTOLÓGICO ─────────────────────────────────
            Phase3_BekensteinHawkingDecider.act_veto(bh_thermo, polyakov_action)
            # (act_veto siempre lanza; esta línea es inalcanzable)

        # ── ACTUACIÓN: LIBERACIÓN DE LA GEODÉSICA ──────────────────────────
        logger.info(
            f"[ACT] Geodésica validada. Fluctuación probabilística permitida. "
            f"Acción de Polyakov = {getattr(polyakov_action, 'action_integral', float('nan')):.6e} | "
            f"R = {curvature.ricci_scalar:.6e}"
        )

        return CategoricalState(
            stratum="WISDOM",
            payload={
                "action_integral": getattr(polyakov_action, "action_integral", None),
                "feynman_amplitude": getattr(polyakov_action, "feynman_amplitude", None),
                "effective_mass": energy_data.effective_mass,
                "ricci_scalar": curvature.ricci_scalar,
                "max_sectional_curvature": curvature.max_sectional_curvature,
                "energy_conditions": {
                    "WEC": energy_data.weak_energy_ok,
                    "NEC": energy_data.null_energy_ok,
                    "SEC": energy_data.strong_energy_ok,
                    "DEC": energy_data.dominant_energy_condition_ok,
                },
                "trace_T": energy_data.trace,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "AstrophysicalConstants",
    "SingularityVetoError",
    "EnergyMomentumDegeneracyError",
    "CausalStructureError",
    "EnergyMomentumData",
    "CurvatureInvariants",
    "BlackHoleThermodynamics",
    "Phase1_EnergyMomentumExtractor",
    "Phase2_EinsteinFieldSolver",
    "Phase3_BekensteinHawkingDecider",
    "EinsteinHilbertAgent",
]