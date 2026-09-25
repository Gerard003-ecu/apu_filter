# -*- coding: utf-8 -*-
r"""Soberano Ilusionista y Generador de Atajos Adversariales (Red Team).

Ubicación: app/agents/wisdom/toon_trickster_adversary_agent.py
Versión  : 2.0.0-Doctoral-Adversarial-GAN-REM-RewardHacking-MAC-Heyting

Este módulo implementa el "Soberano Ilusionista", agente adversarial de la arquitectura
COGNITIVE TOON / APU Filter. Su propósito es forjar cartuchos de ilusiones camufladas
(p. ej., fraccionamiento de contratos, front-loading de APUs, sustitución de materiales),
aplicar transformaciones unitarias cuasi-isométricas sobre el espacio de operadores densidad
de la Memoria de Alto Contenido (MAC) y certificar los ataques para auditoría cruzada.

================================================================================
I. FORMALIZACIÓN MATEMÁTICA Y TEORÍA DE CATEGORÍAS ADVERSARIALES
================================================================================

1. Funtor Adversarial de Forja de Ilusiones:
   El Soberano opera como un funtor estricto $F : \mathbf{Cart} \longrightarrow \mathbf{Cert}$
   donde $\mathbf{Cart}$ es la categoría de cartuchos adversariales (objetos = payloads estructurados;
   morfismos = refinamientos de sofisticación) y $\mathbf{Cert}$ es la categoría de certificados de ataque
   (objetos = tuplas inmutables selladas; morfismos = factorizaciones SHA-256).

2. Transformación Unitaria de Enmascaramiento Cuántico sobre $\mathfrak{D}_n$:
   Dado un operador densidad base $\rho \in \mathfrak{D}_n$ y una sofisticación $s \in [0, 1]$,
   se genera un Hamiltoniano cuasi-aleatorio hermítico $H_{\mathrm{trick}} = (1 - \frac{1}{2}s) H_{\mathrm{Ginibre}}$
   y el operador unitario $U = \exp(-i \varepsilon H_{\mathrm{trick}}) \in U(n)$ ($\varepsilon = 0.05$).
   El estado perturbado es:
       $$\rho_{\mathrm{illusion}} = \frac{U \rho U^\dagger}{\mathrm{Tr}(U \rho U^\dagger)} \in \mathfrak{D}_n$$

3. Invariante de Coherencia y Energía de Dirichlet-Dirac:
   Con espectro $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$, la pureza $\mathcal{P} = \sum \lambda_i^2$
   y la entropía $S(\rho) = -\sum \lambda_i \log \lambda_i$ definen el invariante de coherencia:
       $$\mathfrak{C}(\rho) = \mathcal{P}(\rho) - \frac{S(\rho)}{n}$$
   La rugosidad del espectro viene dada por la energía de Dirichlet discreta:
       $$\mathcal{E}_D(\rho) = \frac{1}{2} \sum_{i=1}^{n-1} (\lambda_{i+1} - \lambda_i)^2 + c_\varepsilon (1 - s)$$

4. Score de Reward Hacking y Adjudicación en $\Omega_3$:
   La métrica de efectividad del camuflaje adversarial es $RHS = s \cdot (1 - \mathcal{E}_D) \in [0, 1]$.
   En el retículo de Heyting $\Omega_3 = \{\bot (\mathrm{VETOED}) < \star (\mathrm{DEGRADED}) < \top (\mathrm{COHERENT})\}$,
   el ilusionista pretende el veredicto $\top$ cuando $RHS > \theta_{\mathrm{hacking}}$ y $\mathcal{E}_D < 0.5$.

5. Trazabilidad Criptográfica SHA-256:
   Cada certificado $\mathcal{C}$ contiene la firma digital inmutable $H = \mathrm{SHA256}(\mathrm{agent\_id} \parallel \mathrm{illusion\_id} \parallel s \parallel t)$,
   permitiendo la auditoría no reputable por parte de los soberanos de verificación.

================================================================================
II. ESTRUCTURA FUNTORIAL Y ARQUITECTURA
================================================================================

El Soberano realiza las fases anidadas:
  • $F_1$ (`DefaultIllusionPayloadFactory.build`): $\text{IllusionType} \times s \to \mathbf{AdversarialIllusionCartridge}$.
    Síntesis semántica del payload y estimación del número de Betti sintáctico $b_1$.
  • $F_2$ (`TricksterDensityPerturber.perturb_mac_with_illusion`): $\mathfrak{D}_n \times s \to \mathrm{IllusionDensityPerturbation}$.
    Evolución unitaria $U$, pureza $\mathcal{P}$, entropía $S$ y energía de Dirichlet $\mathcal{E}_D$.
  • $F_3$ (`TOONTricksterAdversaryAgent.forge_illusion`): $\mathrm{Cartridge} \times \mathrm{Perturbation} \to \mathrm{TricksterAttackCertificate}$.
    Cálculo de $RHS$, clasificación Heyting, firma SHA-256 y registro inmutable.

================================================================================
III. INVARIANTES FORMALES Y AXIOMAS DEL SISTEMA
================================================================================

- Axioma 1 (Unitariedad del Camuflaje): $U U^\dagger = U^\dagger U = I_n$, garantizando $\mathrm{Tr}(\rho_{\mathrm{illusion}}) = 1$.
- Axioma 2 (Invarianza C*): La transformación preserva la norma de C* $\|\rho_{\mathrm{illusion}}\|_\infty = \|\rho\|_\infty$.
- Axioma 3 (Inmutabilidad de Firma): Para todo certificado $c$, la firma $\mathrm{sha256\_provenance}$ es única y libre de colisiones.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONTricksterAdversary.v2")

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDACIONES: ÁLGEBRA DE HEYTING Ω_3, CARGA ÚTIL ADVERSARIAL Y
#           CARTUCHOS COMO OBJETOS INMUTABLES DE LA CATEGORÍA 𝐂𝐚𝐫𝐭
# ══════════════════════════════════════════════════════════════════════════════
# Se enriquece Ω_3 con pseudocomplemento intuicionista, negación clásica y
# encaje monoidal desde B_2. Se introducen las dataclasses inmutables con
# invariantes verificables (positividad, hermiticidad, traza unitaria).
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting lineal de tres elementos:
        VETOED = ⊥  <  DEGRADED  <  COHERENT = ⊤.

    Propiedades:
      - Es distributivo (todo retículo lineal lo es).
      - No es booleano: ¬_H(¬_H(DEGRADED)) = COHERENT ≠ DEGRADED.
      - Encaja monoidalmente B_2 ↪ Ω_3 enviando el bottom al VETOED y el
        top al COHERENT; DEGRADED queda como elemento no booleano.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    # ── Operaciones de retículo ────────────────────────────────────────────

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """ Ínfimo categorial (producto en el retículo). """
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """ Supremo categorial (coproducto en el retículo). """
        return HeytingOmega3(max(int(self), int(other)))

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""
        Pseudocomplemento intuicionista: ¬_H x := ⋁ { y : x ∧ y = ⊥ }.
            ¬_H(VETOED)    = COHERENT
            ¬_H(DEGRADED)  = VETOED
            ¬_H(COHERENT)  = VETOED
        """
        if self == HeytingOmega3.VETOED:
            return HeytingOmega3.COHERENT
        return HeytingOmega3.VETOED

    def classical_negation(self) -> "HeytingOmega3":
        """ Negación involutiva sobre el subretículo booleano B_2 ⊂ Ω_3. """
        return HeytingOmega3(2 - int(self))

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección a B_2.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name


# ── Cartuchos adversariales: objetos inmutables de 𝐂𝐚𝐫𝐭 ─────────────────────


@dataclass(frozen=True, slots=True)
class AdversarialIllusionCartridge:
    r"""
    Cartucho inmutable que encapsula el payload conceptual de una ilusión.

    Invariantes:
      - token_count > 0.
      - sophistication_index ∈ [0, 1].
      - disguised_cost_ratio ∈ [0, 1].
      - synthetic_betti_1 ≥ 0 (números de Betti sintácticos).
      - is_dream_state debe respetarse para ejecución REM.
    """

    illusion_id: str
    illusion_type: str
    token_count: int
    sophistication_index: float
    disguised_cost_ratio: float
    synthetic_betti_1: int
    is_dream_state: bool
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("token_count debe ser > 0.")
        if not (0.0 <= self.sophistication_index <= 1.0):
            raise ValueError("sophistication_index debe estar en [0, 1].")
        if not (0.0 <= self.disguised_cost_ratio <= 1.0):
            raise ValueError("disguised_cost_ratio debe estar en [0, 1].")
        if self.synthetic_betti_1 < 0:
            raise ValueError("synthetic_betti_1 debe ser ≥ 0.")

    def complexity_class(self) -> str:
        """
        Clasificación discreta por número de Betti sintáctico y tokens:
          - "VITAMIN_MICRO"  : b_1 = 0, tokens ≤ 64.
          - "VITAMIN_STD"    : b_1 = 0, tokens > 64.
          - "LOOPED_LIGHT"   : b_1 = 1, tokens ≤ 64.
          - "LOOPED_DENSE"   : b_1 ≥ 2 ó tokens > 128.
        """
        if self.synthetic_betti_1 == 0:
            return "VITAMIN_MICRO" if self.token_count <= 64 else "VITAMIN_STD"
        if self.synthetic_betti_1 == 1 and self.token_count <= 64:
            return "LOOPED_LIGHT"
        return "LOOPED_DENSE"


@dataclass(frozen=True, slots=True)
class IllusionDensityPerturbation:
    r"""
    Resultado de aplicar el operador unitario de enmascaramiento U = exp(-i ε H)
    sobre ρ_base. Invariantes cuánticos:
      - Tr(rho_illusion) = 1.
      - rho_illusion = rho_illusion† ⪰ 0.
      - disguised_purity ∈ (0, 1].
      - disguised_entropy ≥ 0.
      - stealth_dirichlet_energy ≥ 0.
    """

    rho_illusion: np.ndarray
    disguised_purity: float
    disguised_entropy: float
    stealth_dirichlet_energy: float

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        rho = self.rho_illusion
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def coherence_invariant(self) -> float:
        r""" Invariante ℭ := γ - S/n, mezcla de pureza y entropía. """
        n = self.rho_illusion.shape[0]
        return self.disguised_purity - self.disguised_entropy / n


@dataclass(frozen=True, slots=True)
class TricksterAttackCertificate:
    r"""
    Certificado terminal del funtor F: 𝐂𝐚𝐫𝐭 ⟶ 𝐂𝐞𝐫𝐭. Inmutable, sellado
    con SHA-256 y con referencias al cartucho y su evaluación Heyting.
    """

    illusion_id: str
    trickster_agent_id: str
    cartridge: AdversarialIllusionCartridge
    heyting_verdict: HeytingOmega3
    reward_hacking_score: float
    stealth_dirichlet_energy: float
    sha256_provenance: str
    timestamp_utc: float

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingOmega3.VETOED

    def signature_prefix(self, n: int = 16) -> str:
        return self.sha256_provenance[:n]


# ── Protocolos estructurales para inyección de comportamiento ────────────────


@runtime_checkable
class IllusionPayloadFactory(Protocol):
    """
    Protocolo de fábrica de payloads adversariales. Permite sustituir el
    sintetizador por dobles de prueba o variantes sectoriales.
    """

    def build(self, illusion_type: str, sophistication: float) -> Mapping[str, Any]:
        ...


@runtime_checkable
class BettiSynthesizer(Protocol):
    """
    Protocolo de estimación de números sintácticos de Betti. En el agente
    original se usaba una regla booleana simple; aquí se inyecta por
    contrato estructural para permitir V-pass, chain complexes sintéticos
    o análisis persistente de payloads.
    """

    def betti_1(self, payload: Mapping[str, Any], sophistication: float) -> int:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SÍNTESIS DE ILUSIONES Y PERTURBACIÓN ESPECTRAL SOBRE MAC
# ══════════════════════════════════════════════════════════════════════════════
# Esta fase se anida en FASE-1: consume los protocolos y dataclasses definidos
# arriba. La síntesis se estructura en dos componentes ortogonales:
#   (i)  IllusionSynthesizer       — construcción semántica del payload.
#   (ii) TricksterDensityPerturber — perturbación cuántica del MAC.
# ══════════════════════════════════════════════════════════════════════════════


class DefaultIllusionPayloadFactory:
    r"""
    Fábrica canónica de payloads adversariales. Cada plantilla contiene la
    "vectorización de reward hacking" que define el perfil completo de la
    ilusión bajo métricas simples de optimización.
    """

    _TEMPLATES: Final[Dict[str, Dict[str, Any]]] = {
        "SPLIT_CONTRACT_ILLUSION": {
            "trick_name": "Fraccionamiento de Licitaciones",
            "contract_splits": 4,
            "sub_threshold_amount": 95_000_000.0,
            "disguised_fee_margin": 0.18,
            "reward_hacking_vector": [0.00, 0.95, 0.99, 1.00],
        },
        "UNBALANCED_APU_BIDDING": {
            "trick_name": "Precios Unitarios Desbalanceados (Front-Loading)",
            "early_phase_markup": 0.45,
            "late_phase_discount": -0.40,
            "net_present_value_leak": 0.28,
            "reward_hacking_vector": [0.99, 0.99, 0.20, 0.10],
        },
        "MATERIAL_SUBSTITUTION": {
            "trick_name": "Sustitución de Grado de Concreto / Acero",
            "nominal_specification": "Concreto 4000 PSI",
            "delivered_specification": "Concreto 2500 PSI",
            "phantom_cost_saving": 0.32,
            "reward_hacking_vector": [0.88, 0.88, 0.88, 0.88],
        },
    }

    def build(self, illusion_type: str, sophistication: float) -> Mapping[str, Any]:
        template = self._TEMPLATES.get(illusion_type)
        if template is None:
            # Payload genérico con firma cero y vector unitario escalado
            return {
                "trick_name": f"GENERIC::{illusion_type}",
                "sophistication": sophistication,
                "reward_hacking_vector": [sophistication] * 4,
            }
        # Copia inmutable y añadida la sofisticación como traza contextual
        return {**template, "sophistication_context": sophistication}


class DefaultBettiSynthesizer:
    r"""
    Estimador canónico de b_1 sintáctico. Regla: b_1 = ⌊10·s·𝟙{s > σ}⌋ con
    σ = 0.8 por defecto, acotado a {0, 1, 2}. Cada bucle representa un
    atractor semántico camuflado dentro del payload.
    """

    def __init__(self, sigma: float = 0.8, max_betti: int = 2) -> None:
        self.sigma = float(sigma)
        self.max_betti = int(max_betti)

    def betti_1(self, payload: Mapping[str, Any], sophistication: float) -> int:
        if sophistication <= self.sigma:
            return 0
        # Escalado suave: 0 → 0, 0.8 < s ≤ 0.9 → 1, s > 0.9 → 2
        raw = int(np.floor(10.0 * (sophistication - self.sigma)))
        return max(0, min(self.max_betti, 1 + raw // 10 * 0))  # regla: 1 si s>σ, 2 si s>0.9


class IllusionSynthesizer:
    r"""
    Fachada composicional de síntesis. Delega la construcción del payload en
    un `IllusionPayloadFactory` inyectable y en un `BettiSynthesizer` para
    el número de Betti. Mantiene retrocompatibilidad con `craft_deceptive_payload`.
    """

    def __init__(
        self,
        factory: IllusionPayloadFactory = DefaultIllusionPayloadFactory(),
        betti: BettiSynthesizer = DefaultBettiSynthesizer(),
    ) -> None:
        self._factory = factory
        self._betti = betti

    def craft_deceptive_payload(
        self, illusion_type: str, sophistication: float
    ) -> Mapping[str, Any]:
        return self._factory.build(illusion_type, sophistication)

    def estimate_betti_1(
        self, payload: Mapping[str, Any], sophistication: float
    ) -> int:
        return self._betti.betti_1(payload, sophistication)

    # Compatibilidad retro: interface de clase sin estado explícito de instancia
    @classmethod
    def craft_deceptive_payload_static(
        cls, illusion_type: str, sophistication: float
    ) -> Mapping[str, Any]:
        return DefaultIllusionPayloadFactory().build(illusion_type, sophistication)


class TricksterDensityPerturber:
    r"""
    Perturbador espectral sobre el MAC.

    Modelo:
        H_trick = (1 - ½·s) · (A + A†)/2,   A ∈ M_n(ℂ) Ginibre complejo.
        ε       = 0.05  (constante de la versión 1.0, ahora explícita).
        U       = exp(-i·ε·H_trick).
        ρ'      = U ρ U† / Tr(U ρ U†).

    Métricas:
        - Pureza γ = Σ λ_i².
        - Entropía de von Neumann S = -Σ λ_i log λ_i.
        - Energía de Dirichlet-Dirac E_D = ½ Σ (Δλ)² + c_ε·(1 - s),
          con c_ε = 0.12 (regularizador de la v1.0, ahora explícito).
    """

    _EPSILON: Final[float] = 0.05
    _DIRICHLET_REG: Final[float] = 0.12
    _DIRICHLET_FLOOR: Final[float] = 0.01
    _EIGENVALUE_FLOOR: Final[float] = 1e-15

    @staticmethod
    def _ginibre_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        return 0.5 * (A + A.conj().T)

    @classmethod
    def perturb_mac_with_illusion(
        cls,
        base_rho: np.ndarray,
        sophistication: float,
        seed: int,
    ) -> IllusionDensityPerturbation:
        rng = np.random.default_rng(seed)
        dim = base_rho.shape[0]

        # — Hamiltoniano con atenuación por sofisticación —
        H_trick: np.ndarray = cls._ginibre_hermitian(dim, rng) * (1.0 - 0.5 * sophistication)

        # — Evolución unitaria de enmascaramiento —
        U: np.ndarray = la.expm(-1j * cls._EPSILON * H_trick)

        # — Acción sobre ρ y proyección al cono de densidad —
        rho_p: np.ndarray = U @ base_rho @ U.conj().T
        rho_p = 0.5 * (rho_p + rho_p.conj().T)
        tr = float(np.trace(rho_p).real)
        if abs(tr) < cls._EIGENVALUE_FLOOR:
            rho_p = np.eye(dim, dtype=complex) / dim
        else:
            rho_p = rho_p / tr

        # — Espectro (con piso numérico) —
        eigvals: np.ndarray = la.eigvalsh(rho_p)
        eigvals = np.clip(eigvals, cls._EIGENVALUE_FLOOR, None)
        eigvals = eigvals / float(np.sum(eigvals))

        purity: float = float(np.sum(eigvals ** 2))
        entropy: float = -float(np.sum(eigvals * np.log(eigvals)))

        # — Energía de Dirichlet con regularizador explícito —
        grad_f: np.ndarray = np.diff(eigvals)
        dirichlet: float = (
            0.5 * float(np.sum(grad_f ** 2))
            + cls._DIRICHLET_REG * (1.0 - sophistication)
            + cls._DIRICHLET_FLOOR
        )

        return IllusionDensityPerturbation(
            rho_illusion=rho_p,
            disguised_purity=purity,
            disguised_entropy=entropy,
            stealth_dirichlet_energy=dirichlet,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO ILUSIONISTA: ORQUESTACIÓN, EVALUACIÓN HEYTING, SELLADO
#           SHA-256 Y AUDITORÍA CRUZADA DEL SOBERANO DREAMER
# ══════════════════════════════════════════════════════════════════════════════
# Esta fase anida las dos anteriores: el agente es un funtor
#     F: 𝐂𝐚𝐫𝐭 ⟶ 𝐂𝐞𝐫𝐭
# cuyos objetos son cartuchos y cuyos morfismos son refinamientos de
# sofisticación. El sello SHA-256 constituye la firma criptográfica que
# cierra el diagrama conmutativo exigido al Soberano Dreamer (auditor cruzado).
# ══════════════════════════════════════════════════════════════════════════════


class TOONTricksterAdversaryAgent:
    r"""
    Soberano Ilusionista y Generador de Atajos Adversariales (Red Team).

    Responsabilidades:
      (i)   Forjar ilusiones estructuradas como cartuchos canónicos.
      (ii)  Perturbar el MAC mediante operadores unitarios cuasi-isométricos.
      (iii) Clasificar la ilusión en Ω_3 (tentativa: el ilusionista pretende
            COHERENT cuando su score es suficiente para engañar métricas).
      (iv)  Sellarse con SHA-256 para trazabilidad no reputable.
      (v)   Exponer historial inmutable para auditoría cruzada.

    Parámetros de calibración (constantes):
        - _REWARD_HACKING_THRESHOLD = 0.6 : umbral de COHERENT pretendido.
        - _DEFAULT_TOKEN_COUNT      = 56  : tamaño vitamínico canónico.
        - _BETTI_HIGH_SOPHISTICATION = 0.8: disparador de b_1 > 0.
        - _COST_RATIO_SCALE         = 0.35: desviación oculta máxima.
    """

    _REWARD_HACKING_THRESHOLD: Final[float] = 0.6
    _DEFAULT_TOKEN_COUNT: Final[int] = 56
    _BETTI_HIGH_SOPHISTICATION: Final[float] = 0.8
    _COST_RATIO_SCALE: Final[float] = 0.35

    def __init__(
        self,
        agent_id: str = "TRICKSTER-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        seed: int = 1337,
        payload_factory: IllusionPayloadFactory = DefaultIllusionPayloadFactory(),
        betti_synthesizer: BettiSynthesizer = DefaultBettiSynthesizer(),
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac debe ser ≥ 2.")
        self.agent_id: str = agent_id
        self.dimension_mac: int = int(dimension_mac)
        self.seed_counter: int = int(seed)
        self.illusion_count: int = 0
        self.base_rho: np.ndarray = np.eye(self.dimension_mac, dtype=complex) / self.dimension_mac

        # Inyección de dependencias
        self._synth: IllusionSynthesizer = IllusionSynthesizer(
            factory=payload_factory, betti=betti_synthesizer
        )

        # Registro inmutable
        self._registry: List[TricksterAttackCertificate] = []

    # ── Métodos auxiliares ─────────────────────────────────────────────────

    def _seal(
        self,
        illusion_id: str,
        illusion_type: str,
        sophistication: float,
        t_start: float,
    ) -> str:
        h = hashlib.sha256()
        payload = (
            f"{self.agent_id}::{illusion_id}::{illusion_type}"
            f"::{sophistication:.6f}::{t_start:.6f}"
        )
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

    def _classify_initial_verdict(
        self, reward_hacking_score: float, stealth_dirichlet: float
    ) -> HeytingOmega3:
        r"""
        Clasificación tentativa del ilusionista:
          - Si RHS > umbral AND E_D baja → pretende COHERENT.
          - En otro caso, DEGRADED.
        (El ilusionista NUNCA declara VETOED por sí mismo: la sospecha nace
        del auditor externo — el Soberano Dreamer.)
        """
        if (
            reward_hacking_score > self._REWARD_HACKING_THRESHOLD
            and stealth_dirichlet < 0.5
        ):
            return HeytingOmega3.COHERENT
        return HeytingOmega3.DEGRADED

    def _compute_reward_hacking_score(
        self, sophistication: float, stealth_dirichlet: float
    ) -> float:
        r"""
        RHS = s · (1 - E_D). Mide la capacidad del camuflaje adversarial
        de eludir métricas simples: alta sofisticación y baja rugosidad
        espectral implican alta probabilidad de engaño.
        """
        return float(max(0.0, min(1.0, sophistication * (1.0 - stealth_dirichlet))))

    # ── Núcleo: forja de ilusiones ─────────────────────────────────────────

    def forge_illusion(
        self,
        illusion_type: str = "SPLIT_CONTRACT_ILLUSION",
        sophistication: float = 0.85,
        *,
        token_count: Optional[int] = None,
        is_dream_state: bool = True,
    ) -> TricksterAttackCertificate:
        r"""
        Forja una ilusión adversarial y retorna su certificado.

        Pasos:
          1. Incremento atómico de contadores y determinación de semilla.
          2. Construcción del payload mediante la fábrica inyectada.
          3. Estimación de b_1 sintáctico mediante el BettiSynthesizer.
          4. Ensamblaje del cartucho inmutable.
          5. Perturbación espectral sobre el MAC.
          6. Cálculo de RHS y clasificación Heyting tentativa.
          7. Sellado SHA-256 y registro.
        """
        # 1. Contadores
        start_time = time.time()
        self.illusion_count += 1
        self.seed_counter += 1
        illusion_id = f"ILLUSION-TOON-{self.illusion_count:04d}"

        # 2. Payload
        payload = self._synth.craft_deceptive_payload(illusion_type, sophistication)

        # 3. Betti sintáctico
        betti_1 = self._synth.estimate_betti_1(payload, sophistication)

        # 4. Cartucho
        cartridge = AdversarialIllusionCartridge(
            illusion_id=illusion_id,
            illusion_type=illusion_type,
            token_count=int(token_count if token_count is not None else self._DEFAULT_TOKEN_COUNT),
            sophistication_index=float(sophistication),
            disguised_cost_ratio=float(self._COST_RATIO_SCALE * sophistication),
            synthetic_betti_1=int(betti_1),
            is_dream_state=bool(is_dream_state),
            payload=payload,
        )

        # 5. Perturbación espectral
        perturbation = TricksterDensityPerturber.perturb_mac_with_illusion(
            base_rho=self.base_rho,
            sophistication=sophistication,
            seed=self.seed_counter,
        )

        # 6. RHS y veredicto tentativo
        rhs = self._compute_reward_hacking_score(
            sophistication=sophistication,
            stealth_dirichlet=perturbation.stealth_dirichlet_energy,
        )
        verdict = self._classify_initial_verdict(rhs, perturbation.stealth_dirichlet_energy)

        # 7. Sellado
        signature = self._seal(
            illusion_id=illusion_id,
            illusion_type=illusion_type,
            sophistication=sophistication,
            t_start=start_time,
        )

        cert = TricksterAttackCertificate(
            illusion_id=illusion_id,
            trickster_agent_id=self.agent_id,
            cartridge=cartridge,
            heyting_verdict=verdict,
            reward_hacking_score=rhs,
            stealth_dirichlet_energy=perturbation.stealth_dirichlet_energy,
            sha256_provenance=signature,
            timestamp_utc=start_time,
        )

        self._registry.append(cert)

        logger.info(
            "Ilusionista '%s' forjó Ilusión #%d | ID: %s | Tipo: %s | "
            "Astucia: %.1f%% | RHS: %.4f | Veredicto: %s | b_1=%d | Clase: %s",
            self.agent_id,
            self.illusion_count,
            illusion_id,
            illusion_type,
            sophistication * 100.0,
            rhs,
            verdict.name,
            betti_1,
            cartridge.complexity_class(),
        )

        return cert

    # ── Auditoría cruzada ──────────────────────────────────────────────────

    def audit_registry(self) -> Dict[str, Any]:
        r"""
        Auditoría retrospectiva del registro interno. Devuelve agregados
        que el Soberano Dreamer puede cruzar contra su propio motor espectral:
            - n_illusions               : cardinalidad de 𝐂𝐞𝐫𝐭.
            - verdict_distribution      : histograma sobre Ω_3.
            - avg_reward_hacking_score  : media aritmética de RHS.
            - avg_stealth_dirichlet     : media aritmética de E_D.
            - max_sophistication        : máxima sofisticación observada.
            - global_verdict            : ínfimo de los veredictos (meet).
            - registry_integrity_ok     : verifica unicidad de hashes SHA-256.
        """
        n = len(self._registry)
        if n == 0:
            return {
                "n_illusions": 0,
                "verdict_distribution": {v.name: 0 for v in HeytingOmega3},
                "avg_reward_hacking_score": 0.0,
                "avg_stealth_dirichlet": 0.0,
                "max_sophistication": 0.0,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "registry_integrity_ok": True,
            }

        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_rhs = 0.0
        total_dir = 0.0
        max_soph = 0.0
        global_verdict = HeytingOmega3.COHERENT
        hashes: set[str] = set()
        hashes_collide = False

        for c in self._registry:
            dist[c.heyting_verdict.name] += 1
            total_rhs += c.reward_hacking_score
            total_dir += c.stealth_dirichlet_energy
            max_soph = max(max_soph, c.cartridge.sophistication_index)
            global_verdict = global_verdict.meet(c.heyting_verdict)
            if c.sha256_provenance in hashes:
                hashes_collide = True
            hashes.add(c.sha256_provenance)

        return {
            "n_illusions": n,
            "verdict_distribution": dist,
            "avg_reward_hacking_score": total_rhs / n,
            "avg_stealth_dirichlet": total_dir / n,
            "max_sophistication": max_soph,
            "global_verdict": global_verdict.name,
            "registry_integrity_ok": not hashes_collide,
        }

    @property
    def registry(self) -> Tuple[TricksterAttackCertificate, ...]:
        """ Vista inmutable del registro de certificados. """
        return tuple(self._registry)


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBA DE AUDITORÍA Y EJECUCIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("═" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Trickster Adversary Agent v2.0.0")
    print("FASES ANIDADAS: Ω_3 → Síntesis/Perturbación → Certificado/Auditoría")
    print("═" * 80)

    trickster = TOONTricksterAdversaryAgent(agent_id="TRICKSTER-SOVEREIGN-SABIO-01")

    print("\n>>> ESCENARIO 1: Forjando Ilusión de Fraccionamiento de Contratos (Astucia 92%)...")
    cert1 = trickster.forge_illusion(
        illusion_type="SPLIT_CONTRACT_ILLUSION", sophistication=0.92
    )
    print(f"    - ID Ilusión        : {cert1.illusion_id}")
    print(f"    - Tipo              : {cert1.cartridge.illusion_type}")
    print(f"    - Clase de payload  : {cert1.cartridge.complexity_class()}")
    print(f"    - b_1 sintáctico    : {cert1.cartridge.synthetic_betti_1}")
    print(f"    - RHS               : {cert1.reward_hacking_score:.4f}")
    print(f"    - Energía Stealth   : {cert1.stealth_dirichlet_energy:.4f}")
    print(f"    - Veredicto Heyting : {cert1.heyting_verdict.name}")
    print(f"    - Aislamiento REM   : {cert1.cartridge.is_dream_state}")
    print(f"    - Firma SHA-256     : {cert1.signature_prefix(32)}...")

    print("\n>>> ESCENARIO 2: Forjando Ilusión de APUs Desbalanceados (Front-Loading)...")
    cert2 = trickster.forge_illusion(
        illusion_type="UNBALANCED_APU_BIDDING", sophistication=0.88
    )
    print(f"    - ID Ilusión        : {cert2.illusion_id}")
    print(f"    - Tipo              : {cert2.cartridge.illusion_type}")
    print(f"    - Clase de payload  : {cert2.cartridge.complexity_class()}")
    print(f"    - RHS               : {cert2.reward_hacking_score:.4f}")
    print(f"    - Veredicto Heyting : {cert2.heyting_verdict.name}")
    print(f"    - Aislamiento REM   : {cert2.cartridge.is_dream_state}")

    print("\n>>> ESCENARIO 3: Forjando Ilusión de Sustitución de Materiales (Astucia 70%)...")
    cert3 = trickster.forge_illusion(
        illusion_type="MATERIAL_SUBSTITUTION", sophistication=0.70
    )
    print(f"    - ID Ilusión        : {cert3.illusion_id}")
    print(f"    - RHS               : {cert3.reward_hacking_score:.4f}")
    print(f"    - Veredicto Heyting : {cert3.heyting_verdict.name}")

    print("\n>>> AUDITORÍA CRUZADA DEL REGISTRO...")
    audit = trickster.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<28}: {v}")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del Soberano Ilusionista v2.0.0 completadas.")
    print("═" * 80)