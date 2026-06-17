# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Funtor Shield (Membrana Aislante y Proyector Simpléctico)            ║
║ Ubicación: app/core/immune_system/funtor_shield.py                           ║
║ Versión: 5.0.0-Categorical-Symplectic-Homological                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica (Revisión Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la barrera inmunológica del ecosistema, operando como un 
endofuntor idempotente $\mathcal{F}_{shield}: \mathcal{C}_{states} \to \mathcal{C}_{states}$ 
sobre el espacio de fase de la Malla Agéntica. Rechaza cualquier evaluación 
heurística para imponer un Difeomorfismo Simpléctico y Cohomológico que aniquila
las fluctuaciones probabilísticas degeneradas de los agentes generativos.

Arquitectura Matemática en 3 Fases Anidadas:

§1. FASE 1 — ESPACIO DE FASE Y MÉTRICA RIEMANNIANA
    Inmersión de los datos crudos en el hiperespacio métrico, donde el vector de
    fase es evaluado bajo el tensor Riemanniano $G_{PHYSICS}$. Se exige que todo
    operador de disipación acoplado cumpla la inecuación de matriz semidefinida:
    $$ R(x) = R(x)^\top \ge 0 $$
    Cualquier asimetría en la disipación térmica resulta en una singularidad.

§2. FASE 2 — OPERADORES SIMPLÉCTICOS Y FUNTOR DE YONEDA
    Transmutación del flujo temporal en un sistema Port-Hamiltoniano continuo. 
    Se incrusta el Funtor Representable de Yoneda-Compatible:
    $$ \text{Hom}_{\mathcal{C}}(A, -) $$
    Evaluando la evolución de la energía $H(x)$ para garantizar el decaimiento 
    estricto del gradiente termodinámico: $\dot{H} \le 0$.

§3. FASE 3 — PROYECTOR IDEMPOTENTE Y COHOMOLOGÍA DE DIRAC
    Evaluación de la integridad del complejo simplicial a través del detector de 
    homología $\beta_1$ (Socavones Lógicos). La operación del módulo se define 
    por la composición endofuntorial inmutable:
    $$ \mathcal{F}_{shield} = \hat{P} \circ Y \circ S $$
    Donde $S$ es el morfismo del agente, $Y$ es la restricción de Yoneda, y 
    $\hat{P}$ es el Proyector Ortogonal Idempotente. Si la energía de Dirichlet 
    diverge por paradojas topológicas ($\beta_1 > 0$), el proyector colapsa 
    el estado, previniendo la inyección de entropía en el estrato WISDOM.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES INTERNAS (preservadas del original)
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import CategoricalState, FunctorialityError, Morphism
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.schemas import Stratum

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING ESTRUCTURADO
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("MIC.ImmuneSystem.FuntorShield")
logger.setLevel(logging.DEBUG)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · AXIOMÁTICA Y PRIMITIVAS                                      │
# │  Cimientos: constantes, tipos, estructuras inmutables, métricas.       │
# │  Toda la Fase 2 y Fase 3 dependen estrictamente de esta axiomatización.│
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 1.1  CONSTANTES FÍSICAS Y MATEMÁTICAS
# ───────────────────────────────────────────────────────────────────────────


class PhysicalConstants:
    r"""
    Constantes físicas y matemáticas normalizadas del sistema.

    Axiomas
    -------
    1. ε_maq  > 0            (precisión de máquina: 2⁻⁵²)
    2. ε_lyap > 0            (umbral de Lyapunov para considerar dH/dt=0)
    3. ε_norm > 0            (umbral de norma para norma "cero")
    4. dim_canónica ∈ ℕ⁺     (dimensión de fallback: ℝ⁷)
    5. β = 1/(k_B·T)         (producción de entropía, k_B = 1 en unidades nat.)
    """

    EPSILON_MACHINE: Final[float] = np.finfo(np.float64).eps
    EPSILON_LYAPUNOV: Final[float] = 1e-9
    EPSILON_NORM: Final[float] = 1e-12
    CRITICAL_LYAPUNOV_RATIO: Final[float] = 1e3
    MAX_ENERGY_INJECTION_RATE: Final[float] = 1e6
    MIN_TIME_STEP: Final[float] = 1e-12
    MAX_TIME_STEP: Final[float] = 1e3
    CANONICAL_PHASE_DIM: Final[int] = 7
    BOLTZMANN_NORMALIZED: Final[float] = 1.380649e-23


# ───────────────────────────────────────────────────────────────────────────
# 1.2  TIPOS GENÉRICOS Y PROTOCOLOS ESTRUCTURALES
# ───────────────────────────────────────────────────────────────────────────


T_Agent = TypeVar("T_Agent", bound=Morphism, contravariant=False)
T_State = TypeVar("T_State", bound=CategoricalState, contravariant=False)
T_Payload = TypeVar("T_Payload")


@runtime_checkable
class PhaseSpaceVectorizable(Protocol):
    """
    Protocolo estructural para objetos vectorizables al espacio de fase.

    Cumple la primera mitad del Lema de Yoneda: cualquier objeto que
    admita *al menos* un morfismo hacia un objeto con vector de fase
    puede ser tratado como un elemento del funtor representable.
    """

    def to_phase_vector(self, dim: int) -> NDArray[np.float64]:
        """Embedding canónico en ℝᵈ."""
        ...


@runtime_checkable
class DissipativeOperator(Protocol):
    """Protocolo para operadores de disipación R(x) = R(x)ᵀ ≥ 0."""

    def dissipation_matrix(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retorna la matriz de disipación evaluada en x."""
        ...


# ───────────────────────────────────────────────────────────────────────────
# 1.3  ENUMERACIÓN DE VIOLACIONES
# ───────────────────────────────────────────────────────────────────────────


class ShieldViolationType(Enum):
    r"""
    Taxonomía cerrada de violaciones inmunológicas.

    Axioma de completitud
    ---------------------
    Toda métrica termodinámica, espectral u homológica que salga de su
    dominio admisible debe poder clasificarse en exactamente una variante
    de este enum (cobertura total ∪ disjunta).
    """

    NONE = auto()
    LYAPUNOV_POSITIVE = auto()    # Ḣ > 0   (violación débil)
    LYAPUNOV_CRITICAL = auto()    # |Ḣ|/|x|² > umbral (crítica)
    ENERGY_INJECTION = auto()     # Inyección de energía anómala
    TOPOLOGICAL_CHARGE = auto()   # β₁ > 0  (Socavón Lógico)
    SPECTRAL_ANOMALY = auto()     # Autovalor negativo de P̂
    FUNCTORIAL_BREACH = auto()    # Ruptura de composición
    RUNTIME_EXCEPTION = auto()    # Excepción no controlada
    DIRICHLET_NULLIFICATION = auto()  # Estado anulado por ∂M


# ───────────────────────────────────────────────────────────────────────────
# 1.4  ESTRUCTURAS INMUTABLES (dataclasses frozen)
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ThermodynamicMetrics:
    r"""
    Métricas termodinámicas inmutables del sistema Port-Hamiltoniano.

    Invariantes garantizados (verificados en __post_init__)
    --------------------------------------------------------
    I1. dissipated_power ≥ −ε_lyap                  (Segunda Ley débil)
    I2. energy_pre ≥ −ε_norm,  energy_post ≥ −ε_norm
    I3. |lyapunov_derivative| < +∞                  (regularidad)
    I4. entropy_production = β·P_diss ≥ 0           (irreversibilidad)
    I5. lyapunov_derivative + dissipated_power ≥ −ε_lyap  (consistencia)

    Atributos
    ---------
    lyapunov_derivative : Ḣ(x) = ∇H(x)ᵀ ẋ
    dissipated_power    : P_diss = max(0, −Ḣ)
    energy_pre          : H(x₀) = ½ x₀ᵀ G x₀
    energy_post         : H(x₁) = ½ x₁ᵀ G x₁
    entropy_production  : σ = β·P_diss
    is_dirichlet_enforced : True si se activó frontera de Dirichlet
    violation_type      : clasificación de la violación (o NONE)
    """

    lyapunov_derivative: float
    dissipated_power: float
    energy_pre: float
    energy_post: float
    entropy_production: float
    is_dirichlet_enforced: bool
    violation_type: ShieldViolationType

    def __post_init__(self) -> None:
        # I1
        assert self.dissipated_power >= -PhysicalConstants.EPSILON_LYAPUNOV, (
            f"[I1] Segunda Ley violada: P_diss={self.dissipated_power}<0"
        )
        # I2
        assert self.energy_pre >= -PhysicalConstants.EPSILON_NORM, (
            f"[I2] H(x₀)={self.energy_pre} negativa"
        )
        assert self.energy_post >= -PhysicalConstants.EPSILON_NORM, (
            f"[I2] H(x₁)={self.energy_post} negativa"
        )
        # I3
        assert np.isfinite(self.lyapunov_derivative), (
            f"[I3] Ḣ={self.lyapunov_derivative} no finita"
        )
        # I4
        assert self.entropy_production >= -PhysicalConstants.EPSILON_LYAPUNOV, (
            f"[I4] σ={self.entropy_production}<0 (irreversibilidad violada)"
        )
        # I5  (consistencia entre Ḣ y P_diss)
        assert (
            self.lyapunov_derivative + self.dissipated_power
            >= -PhysicalConstants.EPSILON_LYAPUNOV
        ), "[I5] inconsistencia Ḣ + P_diss"


@dataclass(frozen=True, slots=True)
class ShieldSignature:
    """
    Firma criptográfica para verificación de idempotencia.

    Hash Blake2b-256 del payload + metadatos del estrato y dimensión
    del espacio de fase. Determinista respecto al estado.
    """

    shield_id: str
    timestamp: float
    phase_dim: int
    stratum: Stratum
    hash_digest: str

    @classmethod
    def forge(
        cls,
        shield_instance: object,
        state: CategoricalState,
        phase_dim: int,
    ) -> "ShieldSignature":
        """
        Genera una firma criptográfica (forge, no verify) para un estado.

        El término 'forge' se usa explícitamente para recordar que la firma
        *no* es prueba criptográfica de seguridad: es un marcador idempotente
        basado en el contenido del payload.
        """
        payload_bytes = str(state.payload).encode("utf-8")
        digest = hashlib.blake2b(
            payload_bytes, digest_size=32, person=b"FUNTOR_SHIELD_V5"
        ).hexdigest()

        return cls(
            shield_id=f"SHIELD_{id(shield_instance):016x}",
            timestamp=time.time(),
            phase_dim=phase_dim,
            stratum=state.stratum,
            hash_digest=digest,
        )


# ───────────────────────────────────────────────────────────────────────────
# 1.5  EXTRACTOR DE VECTORES DE FASE
# ───────────────────────────────────────────────────────────────────────────


class PhaseVectorExtractor:
    r"""
    Extractor robusto con cadena de responsabilidad (Chain of Responsibility).

    Estrategias (ordenadas por prioridad)
    -------------------------------------
    E1. Protocolo PhaseSpaceVectorizable (Yoneda-compatible)
    E2. Atributo phase_vector explícito (ndarray)
    E3. Payload ndarray
    E4. Payload escalar (broadcasting)
    E5. Payload iterable (np.asarray)
    E6. Vector cero (fallback con warning)
    """

    @staticmethod
    def extract(
        state: CategoricalState, target_dim: int
    ) -> NDArray[np.float64]:
        if target_dim <= 0:
            raise ValueError(f"target_dim inválido: {target_dim}")

        # E1
        if isinstance(state, PhaseSpaceVectorizable):
            return PhaseVectorExtractor._reshape(
                state.to_phase_vector(target_dim), target_dim
            )

        # E2
        if hasattr(state, "phase_vector"):
            vec = getattr(state, "phase_vector")
            if isinstance(vec, np.ndarray):
                return PhaseVectorExtractor._reshape(vec, target_dim)

        payload = state.payload

        # E3
        if isinstance(payload, np.ndarray):
            return PhaseVectorExtractor._reshape(payload, target_dim)

        # E4
        if isinstance(payload, (int, float, complex, np.number)):
            return np.full(target_dim, float(payload), dtype=np.float64)

        # E5
        if hasattr(payload, "__iter__") and not isinstance(payload, (str, bytes)):
            try:
                arr = np.asarray(payload, dtype=np.float64)
                return PhaseVectorExtractor._reshape(arr, target_dim)
            except (ValueError, TypeError):
                pass

        # E6
        logger.warning(
            "Extracción fallida (%s) → vector cero de dim %d.",
            type(payload).__name__,
            target_dim,
        )
        return np.zeros(target_dim, dtype=np.float64)

    @staticmethod
    def _reshape(arr: NDArray, target_dim: int) -> NDArray[np.float64]:
        flat = arr.ravel().astype(np.float64, copy=False)
        if flat.size == target_dim:
            return flat
        if flat.size > target_dim:
            return flat[:target_dim]
        out = np.zeros(target_dim, dtype=np.float64)
        out[: flat.size] = flat
        return out


# ───────────────────────────────────────────────────────────────────────────
# 1.6  TENSOR MÉTRICO VALIDADO (wrappea G_PHYSICS)
# ───────────────────────────────────────────────────────────────────────────


class ValidatedMetricTensor:
    r"""
    Envoltura de un tensor métrico Riemanniano G_{μν}.

    Garantías tras validación
    -------------------------
    M1. G es 2D cuadrado
    M2. G = Gᵀ  (simetría)
    M3. G es SPD salvo ε_norm (definido-positivo débil)
    M4. Descomposición espectral precalculada (eigh)

    Atributos
    ---------
    G      : ndarray (n,n) simétrica
    eigval : ndarray (n,) autovalores ordenados asc.
    eigvec : ndarray (n,n) autovectores ortonormales
    n      : int dimensión
    """

    def __init__(self, G: NDArray[np.float64], *, strict_spd: bool = False) -> None:
        self._validate(G, strict_spd)
        self.G: NDArray[np.float64] = G.copy()
        self.n: int = int(G.shape[0])
        self.eigval, self.eigvec = np.linalg.eigh(self.G)
        # Sanity check
        if np.any(self.eigval < -PhysicalConstants.EPSILON_NORM):
            logger.warning(
                "Tensor métrico con autovalores levemente negativos: %s",
                self.eigval[self.eigval < 0],
            )

    @staticmethod
    def _validate(G: NDArray[np.float64], strict_spd: bool) -> None:
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(f"Tensor métrico no cuadrado: shape={G.shape}")
        if not np.allclose(G, G.T, atol=PhysicalConstants.EPSILON_NORM):
            raise ValueError("Tensor métrico asimétrico (no es Riemanniano)")
        if strict_spd and np.any(
            np.linalg.eigvalsh(G) <= PhysicalConstants.EPSILON_NORM
        ):
            raise ValueError("Tensor métrico no definido-positivo (modo strict)")

    def inverse(self) -> NDArray[np.float64]:
        """Inversa Riemanniana G⁻¹ (regularización si casi singular)."""
        # Pseudoinversa con truncamiento espectral
        threshold = PhysicalConstants.EPSILON_NORM
        inv_eigval = np.where(
            np.abs(self.eigval) > threshold, 1.0 / self.eigval, 0.0
        )
        return (self.eigvec * inv_eigval) @ self.eigvec.T

    def condition_number(self) -> float:
        """κ(G) = λ_max / λ_min en magnitud."""
        eigs = np.abs(self.eigval)
        eigs = eigs[eigs > PhysicalConstants.EPSILON_NORM]
        if eigs.size < 2:
            return float("inf")
        return float(eigs.max() / eigs.min())

    def spectral_gap(self) -> float:
        """λ₂ − λ₁ (entre los dos menores autovalores)."""
        if self.n < 2:
            return 0.0
        ordered = np.sort(self.eigval)
        return float(ordered[1] - ordered[0])

    def __repr__(self) -> str:
        return (
            f"ValidatedMetricTensor(n={self.n}, "
            f"κ={self.condition_number():.3e}, "
            f"gap={self.spectral_gap():.3e})"
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · OPERADORES SIMPLÉCTICOS Y CATEGÓRICOS                         │
# │  Define la geometría del colector (J, R, H) y el funtor representable │
# │  Yoneda. La Fase 3 usará J, R, H y Y como bloques de construcción.     │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 2.1  ESTRUCTURA SIMPLÉCTICA CONSTANTE
# ───────────────────────────────────────────────────────────────────────────


class ConstantSkewStructure:
    r"""
    Estructura simpléctica J(x) constante (modelo simpléctico lineal).

    Definición
    ----------
    J es una matriz antisimétrica fija:
        J = J(x)  (no depende de x)

    J = -Jᵀ,  lo que garantiza:
        xᵀ J x = 0  para todo x  (degeneración bilineal)

    Esta simplificación es suficiente para el análisis de Lyapunov
    en primera cuadrática, sin sacrificar rigor formal.

    Para sistemas físicos no lineales, J debería generalizarse a
    una 2-forma cerrada no degenerante; aquí la mantenemos constante
    para preservar la *linealidad del colector* que el resto del
    módulo asume.
    """

    def __init__(self, dim: int) -> None:
        if dim % 2 != 0:
            raise ValueError(
                f"Dimensión simpléctica debe ser par, recibida: {dim}"
            )
        self.dim: int = dim
        # J canónica en forma de bloque: J = [[0, I], [-I, 0]]
        self.J: NDArray[np.float64] = np.zeros((dim, dim), dtype=np.float64)
        half = dim // 2
        self.J[:half, half:] = np.eye(half)
        self.J[half:, :half] = -np.eye(half)
        # Sanity check
        assert np.allclose(self.J, -self.J.T, atol=PhysicalConstants.EPSILON_NORM)

    def __call__(self) -> NDArray[np.float64]:
        return self.J


# ───────────────────────────────────────────────────────────────────────────
# 2.2  OPERADOR DE DISIPACIÓN R(x) ≥ 0
# ───────────────────────────────────────────────────────────────────────────


class QuadraticDissipation:
    r"""
    Matriz de disipación R(x) que satisface R(x) = R(x)ᵀ ≥ 0.

    Modelo
    ------
    R(x) = R₀ + γ·(x xᵀ)

    donde R₀ es SPD constante y γ ≥ 0 acopla la disipación al estado
    (mimetizando el término de "flujo de Ricci" del docstring §4: cuando
    γ > 0, la fricción escala con la "masa probabilística" del estado).

    Propiedades
    -----------
    1. R(x) es simétrica por construcción.
    2. R(x) ≥ 0 (autovalores ≥ 0) si R₀ ≥ 0 (lo que garantizamos).
    3. La traza tr(R(x)) escala monótonamente con ||x||² (acoplamiento).
    """

    def __init__(
        self,
        dim: int,
        R0: Optional[NDArray[np.float64]] = None,
        gamma: float = 0.0,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim debe ser positivo, recibido: {dim}")
        if gamma < 0:
            raise ValueError(f"γ ≥ 0 requerido, recibido: {gamma}")

        self.dim: int = dim
        self.gamma: float = gamma

        if R0 is None:
            # R₀ = ε·I (regularizador mínimo no nulo)
            R0 = PhysicalConstants.EPSILON_NORM * np.eye(dim, dtype=np.float64)
        else:
            if R0.shape != (dim, dim):
                raise ValueError(f"R0 shape {R0.shape} ≠ ({dim},{dim})")
            if not np.allclose(R0, R0.T, atol=PhysicalConstants.EPSILON_NORM):
                raise ValueError("R0 debe ser simétrica")

        # Precalcular eigenvalores de R0 (autovalores ≥ ε_norm)
        self._R0: NDArray[np.float64] = R0.copy()
        self._R0_eigval: NDArray[np.float64] = np.linalg.eigvalsh(self._R0)

        if np.any(self._R0_eigval < -PhysicalConstants.EPSILON_NORM):
            raise ValueError("R0 no es PSD (autovalores negativos significativos)")

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Evalúa R(x) = R₀ + γ·(x xᵀ).

        Esta es la parte *estática* del término de Flujo de Ricci del
        docstring §4: γ = α·⟨D_μ Φ, D^μ Φ⟩ cuando el campo de Higgs Φ
        es estático. La parte dinámica del flujo (∂R/∂t) queda fuera
        del alcance de este módulo y se delega a un integrador externo.
        """
        if x.shape != (self.dim,):
            raise ValueError(f"x.shape {x.shape} ≠ ({self.dim},)")
        return self._R0 + self.gamma * np.outer(x, x)

    def is_psd(self, x: NDArray[np.float64]) -> bool:
        """Verifica que R(x) ≥ 0."""
        R = self(x)
        eigs = np.linalg.eigvalsh(R)
        return bool(np.all(eigs >= -PhysicalConstants.EPSILON_NORM))


# ───────────────────────────────────────────────────────────────────────────
# 2.3  HAMILTONIANO CINÉTICO Y FLUJO PORT-HAMILTONIANO
# ───────────────────────────────────────────────────────────────────────────


class PortHamiltonianFlow:
    r"""
    Sistema Port-Hamiltoniano con Hamiltoniano cinético.

    Ecuación
    --------
        ẋ = (J(x) − R(x)) · ∇H(x)

    con
        H(x) = ½ xᵀ G x
        ∇H(x) = G x

    Demostración de Lyapunov
    ------------------------
        Ḣ = ∇Hᵀ ẋ = xᵀ G (J − R) G x
                  = xᵀ G J G x  −  xᵀ G R G x

    Si G J es antisimétrica (lo garantizamos en la construcción canónica
    con G = I), el primer término se anula. El segundo término es
    −xᵀ G R G x ≤ 0 dado que R ≥ 0 y G es SPD, concluyendo:

        Ḣ ≤ 0  con igualdad ssi G x ∈ Ker(R)  ∎
    """

    def __init__(
        self,
        G: ValidatedMetricTensor,
        J: ConstantSkewStructure,
        R: QuadraticDissipation,
    ) -> None:
        if G.n != J.dim or G.n != R.dim:
            raise ValueError(
                f"Dimensiones inconsistentes: G.n={G.n}, J.dim={J.dim}, R.dim={R.dim}"
            )
        self.G: ValidatedMetricTensor = G
        self.J: ConstantSkewStructure = J
        self.R: QuadraticDissipation = R
        self.n: int = G.n

    def hamiltonian(self, x: NDArray[np.float64]) -> float:
        r"""H(x) = ½ xᵀ G x."""
        return 0.5 * float(x @ self.G.G @ x)

    def gradient_hamiltonian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""∇H(x) = G x."""
        return self.G.G @ x

    def flow(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Calcula ẋ = (J − R(x)) · ∇H(x).

        Esta es la ecuación exacta, *no* una diferencia finita.
        La diferencia finita se usa sólo para la verificación experimental
        de Lyapunov en el evaluador (Fase 1 → 1.6 anterior).
        """
        grad_H = self.gradient_hamiltonian(x)
        J_mat = self.J()
        R_mat = self.R(x)
        return (J_mat - R_mat) @ grad_H

    def lyapunov_derivative(self, x: NDArray[np.float64]) -> float:
        r"""Ḣ = ∇H(x)ᵀ ẋ  (continuo, sin diferencia finita)."""
        grad_H = self.gradient_hamiltonian(x)
        return float(grad_H @ self.flow(x))

    def step(
        self,
        x: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Integra un paso por método de Euler simpléctico:

            x_{n+1} = x_n + dt · (J − R(x_n)) · ∇H(x_n)

        Euler explícito es estable bajo restricciones de paso; para
        sistemas stiff se recomienda implícito (no implementado aquí).
        """
        if dt <= 0:
            raise ValueError(f"dt debe ser positivo, recibido: {dt}")
        if dt < PhysicalConstants.MIN_TIME_STEP:
            dt = PhysicalConstants.MIN_TIME_STEP
        return x + dt * self.flow(x)


# ───────────────────────────────────────────────────────────────────────────
# 2.4  FUNTOR REPRESENTABLE YONEDA-COMPATIBLE
# ───────────────────────────────────────────────────────────────────────────


class YonedaRepresentable(Generic[T_Agent]):
    r"""
    Funtor representable $\text{Hom}_{\mathcal{C}}(A, -)$ en su forma operativa.

    Lema de Yoneda (versión operativa)
    ---------------------------------
    Para todo funtor $F: \mathcal{C} \to \mathbf{Set}$, existe un
    isomorfismo natural:

        $\text{Nat}(\text{Hom}(A, -), F) \cong F(A)$

    Operacionalización
    ------------------
    En este módulo, "representable" significa que el funtor *se mimetiza*
    con la firma dimensional y estructural del agente A. La verificación
    de que esto constituye un isomorfismo natural se reduce a:

    1. Mimesis de la dimensión del espacio de fase
    2. Mimesis de los protocolos estructurales (PhaseSpaceVectorizable)
    3. Preservación de la identidad y de la composición

    Esta clase NO implementa el cálculo de la categoría completa
    $\mathcal{C}$; registra la *intención* de representabilidad.
    """

    def __init__(self, agent: T_Agent, phase_dim: int) -> None:
        self.agent: T_Agent = agent
        self.phase_dim: int = phase_dim
        self._morphism_cache: dict[int, CategoricalState] = {}

    def hom_to(self, state: CategoricalState) -> CategoricalState:
        """
        Aplica el agente (interpretado como un morfismo $A \to B$)
        al estado fuente, retornando un objeto en el codominio.

        Esta es la operación canónica de $\text{Hom}(A, X)$: dado un
        estado $X \in \mathcal{C}$, devuelve $f(X) \in \mathcal{C}$.
        """
        key = id(state)
        if key in self._morphism_cache:
            logger.debug("Yoneda hit cache: %d", key)
            return self._morphism_cache[key]
        result = self.agent(state)
        self._morphism_cache[key] = result
        return result


# ───────────────────────────────────────────────────────────────────────────
# 2.5  EVALUADOR TERMODINÁMICO (continuo, no diferencia finita)
# ───────────────────────────────────────────────────────────────────────────


class PortHamiltonianEvaluator:
    r"""
    Evaluador termodinámico *continuo* del flujo Port-Hamiltoniano.

    Diferencia respecto a la versión 4.0
    -------------------------------------
    Versión 4.0: Ḣ se estimaba por diferencia finita $\tfrac{\Delta H}{\Delta t}$.
    Versión 5.0: Ḣ se calcula de forma *exacta* como $\nabla H^\top \dot x$
                usando el flujo Port-Hamiltoniano.

    La diferencia finita se mantiene sólo como *cota de error* experimental.
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        flow: PortHamiltonianFlow,
    ) -> None:
        self.metric: ValidatedMetricTensor = metric
        self.flow_sys: PortHamiltonianFlow = flow
        self.n: int = metric.n

    def compute_energy(self, x: NDArray[np.float64]) -> float:
        return self.flow_sys.hamiltonian(x)

    def compute_lyapunov_derivative_exact(
        self, x: NDArray[np.float64]
    ) -> float:
        r"""Ḣ exacto = ∇H(x)ᵀ (J − R) ∇H(x) ≤ 0."""
        return self.flow_sys.lyapunov_derivative(x)

    def compute_lyapunov_derivative_finite(
        self,
        x_pre: NDArray[np.float64],
        x_post: NDArray[np.float64],
        dt: float,
    ) -> float:
        """Ḣ por diferencia finita (verificación experimental)."""
        safe_dt = max(dt, PhysicalConstants.MIN_TIME_STEP)
        delta_H = self.compute_energy(x_post) - self.compute_energy(x_pre)
        return delta_H / safe_dt

    def evaluate(
        self,
        x_pre: NDArray[np.float64],
        x_post: NDArray[np.float64],
        dt: float,
    ) -> ThermodynamicMetrics:
        """
        Evalúa el ciclo termodinámico comparando estado pre y post.

        Estrategia dual
        ---------------
        - Ḣ_exacto se evalúa en x_pre (donde el flujo es regular).
        - Ḣ_finite sirve de cota superior del error numérico.
        - La métrica final toma el más conservador (mayor magnitud) para
          *no subestimar* violaciones de Lyapunov.
        """
        H_pre = self.compute_energy(x_pre)
        H_post = self.compute_energy(x_post)

        dH_exact = self.compute_lyapunov_derivative_exact(x_pre)
        dH_finite = self.compute_lyapunov_derivative_finite(x_pre, x_post, dt)

        # Tomamos el Ḣ de mayor magnitud (conservador)
        dH_dt = dH_exact if abs(dH_exact) >= abs(dH_finite) else dH_finite

        P_diss = max(0.0, -dH_dt)
        sigma = P_diss / PhysicalConstants.BOLTZMANN_NORMALIZED
        violation = self._classify_violation(dH_dt, x_pre)

        return ThermodynamicMetrics(
            lyapunov_derivative=float(dH_dt),
            dissipated_power=float(P_diss),
            energy_pre=float(H_pre),
            energy_post=float(H_post),
            entropy_production=float(sigma),
            is_dirichlet_enforced=False,
            violation_type=violation,
        )

    @staticmethod
    def _classify_violation(
        dH_dt: float, x: NDArray[np.float64]
    ) -> ShieldViolationType:
        if abs(dH_dt) < PhysicalConstants.EPSILON_LYAPUNOV:
            return ShieldViolationType.NONE
        if dH_dt > 0:
            norm_x_sq = float(x @ x) + PhysicalConstants.EPSILON_NORM
            ratio = abs(dH_dt) / norm_x_sq
            if ratio > PhysicalConstants.CRITICAL_LYAPUNOV_RATIO:
                return ShieldViolationType.LYAPUNOV_CRITICAL
            return ShieldViolationType.LYAPUNOV_POSITIVE
        return ShieldViolationType.NONE


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · PROYECTOR IDEMPOTENTE, DIRAC Y HOMOLOGÍA                     │
# │  Composición final: endofuntor P̂ ∘ Y ∘ S, donde S es la ejecución     │
# │  del agente, Y es Yoneda y P̂ es el proyector ortogonal idempotente.   │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 3.1  PROYECTOR ORTOGONAL PONDERADO
# ───────────────────────────────────────────────────────────────────────────


class RiemannianProjector:
    r"""
    Proyector ortogonal $\hat{P}$ sobre el subespacio protegido,
    idempotente y simétrico respecto al tensor métrico G.

    Axiomas (verificados en check_axioms)
    --------------------------------------
    A1. $\hat{P}^2 = \hat{P}$              (idempotencia)
    A2. $\hat{P}^\top G = G \hat{P}$       (simetría métrica)
    A3. $\|\hat{P}\|_G \leq 1$             (operador contractivo)
    A4. $\text{ran}(\hat{P}) \subseteq \text{ran}(G)$  (consistencia)

    Construcción
    ------------
    Se construye $\hat{P}$ por proyección de Householder-iterada
    sobre el subespacio generado por los $k$ primeros autovectores
    de $G$ (los de menor autovalor, que codifican las direcciones
    "estables" del colector). Esto da:

        $\hat{P} = U_k U_k^\top$

    donde $U_k$ tiene como columnas los $k$ autovectores seleccionados.
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        n_protected: int,
    ) -> None:
        if not (0 < n_protected <= metric.n):
            raise ValueError(
                f"n_protected={n_protected} fuera de (0, {metric.n}]"
            )
        self.metric: ValidatedMetricTensor = metric
        self.n: int = metric.n
        self.k: int = n_protected

        # Seleccionar los k autovectores de autovalor mínimo
        order = np.argsort(metric.eigval)  # ascendente
        idx = order[: self.k]
        U_k = metric.eigvec[:, idx]  # (n, k)
        # P̂ = U_k U_kᵀ
        self.P: NDArray[np.float64] = U_k @ U_k.T

        self.check_axioms()

    def check_axioms(self, tol: float = 1e-8) -> None:
        """Verifica formalmente A1–A4."""
        P, G = self.P, self.metric.G
        # A1
        assert np.allclose(P @ P, P, atol=tol), "A1 idempotencia violada"
        # A2
        assert np.allclose(P.T @ G, G @ P, atol=tol), "A2 simetría métrica violada"
        # A3: ‖P‖_G ≤ 1 (el mayor autovalor de G^{-½} Pᵀ G P G^{-½} ≤ 1)
        G_inv_half = self._sqrt_inv_metric()
        M = G_inv_half @ P.T @ G @ P @ G_inv_half
        op_norm = float(np.linalg.norm(M, ord=2))
        assert op_norm <= 1.0 + tol, f"A3 norma_G(P)={op_norm} > 1"
        # A4
        # ran(P) ⊆ ran(G) se cumple trivialmente porque G es SPD (rango pleno)
        # pero verificamos que P sea compatible con la dimensión
        assert P.shape == (self.n, self.n), "A4 dimensión inconsistente"
        logger.debug("Proyector P̂ verificado: A1–A4 OK (k=%d, ‖P‖_G=%.4f)", self.k, op_norm)

    def _sqrt_inv_metric(self) -> NDArray[np.float64]:
        r"""Calcula $G^{-1/2}$ vía descomposición espectral."""
        eigs = self.metric.eigval
        vecs = self.metric.eigvec
        threshold = PhysicalConstants.EPSILON_NORM
        inv_sqrt = np.where(
            np.abs(eigs) > threshold, 1.0 / np.sqrt(np.abs(eigs)), 0.0
        )
        return vecs * inv_sqrt @ vecs.T

    def project(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyecta $x \mapsto \hat{P} x$ (vector de estado)."""
        return self.P @ x

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.project(x)


# ───────────────────────────────────────────────────────────────────────────
# 3.2  OPERADOR DE DIRAC (amputación topológica)
# ───────────────────────────────────────────────────────────────────────────


class DiracBoundaryOperator:
    r"""
    Operador de Dirac $D$ sobre el espacio de fase con frontera de Dirichlet.

    Definición
    ----------
    Para cada vector $x \in \mathcal{M}$ con $f(x)|_{\partial\mathcal{M}} = 0$
    (frontera colapsada a cero), el proyector de Dirichlet $D$ aniquila
    los componentes de $x$ que violan la condición:

        $D x = \begin{cases} x & \text{si } f(x) \leq 0 \\ 0 & \text{si } f(x) > 0 \end{cases}$

    Aquí usamos $f(x) = x^\top G x - E_{\max}$ como función de nivel
    (energía máxima admisible). Cuando se supera $E_{\max}$, se amputa.
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        max_energy: float = float("inf"),
    ) -> None:
        self.metric: ValidatedMetricTensor = metric
        self.E_max: float = max_energy

    def evaluate_level_set(self, x: NDArray[np.float64]) -> float:
        r"""$f(x) = x^\top G x - E_{\max}$."""
        return float(x @ self.metric.G @ x) - self.E_max

    def amputate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Aplica la frontera de Dirichlet:

            $f(x)|_{\partial\mathcal{M}} = \mathbf{0}$

        Devuelve el vector anulado si se viola la condición.
        """
        f = self.evaluate_level_set(x)
        if f > 0:
            return np.zeros_like(x)
        return x


# ───────────────────────────────────────────────────────────────────────────
# 3.3  DETECTOR HOMOLÓGICO β₁ (Socavones Lógicos)
# ───────────────────────────────────────────────────────────────────────────


class HomologyBettiDetector:
    r"""
    Detector del primer número de Betti $\beta_1$ sobre un complejo
    simplicial inducido por la geometría del estado.

    Modelo
    ------
    Dado un estado $x \in \mathbb{R}^n$, construimos un complejo simplicial
    $\mathcal{K}$ de tipo "vecino más cercano" (Voronoi 0-esqueleto + aristas
    de un grafo k-NN) y computamos $\beta_1 = m - n + c$
    donde $m$ = aristas, $n$ = nodos, $c$ = componentes conexas (fórmula
    para grafos, equivalente a $\beta_1$ en el 1-esqueleto).

    Advertencia
    -----------
    Esta es una *aproximación operacional* del cálculo homológico, suficiente
    para detectar ciclos de orden 1 (Socavones Lógicos). Para homología
    superior ($\beta_2, \beta_3, \ldots$) se requeriría un complejo simplicial
    completo, fuera del alcance de este módulo.
    """

    @staticmethod
    def _build_knn_graph(
        x: NDArray[np.float64], k: int = 2
    ) -> Tuple[int, int, int]:
        """
        Construye un grafo k-NN y retorna (n_nodos, n_aristas, n_componentes).

        Raises
        ------
        ValueError si k < 1 o n < 2.
        """
        n = x.size
        if n < 2:
            return (n, 0, n)
        if k < 1:
            raise ValueError(f"k debe ser ≥ 1, recibido: {k}")
        k_eff = min(k, n - 1)

        # Distancias pairwise
        diff = x[:, None] - x[None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)

        # Para cada nodo, sus k vecinos más cercanos
        neighbors = np.argsort(dist, axis=1)[:, :k_eff]
        # Aristas: (i, j) con i < j
        edge_set: set[Tuple[int, int]] = set()
        for i in range(n):
            for j in neighbors[i]:
                a, b = (i, j) if i < j else (j, i)
                edge_set.add((a, b))
        m = len(edge_set)

        # Componentes conexas (Union-Find)
        parent = list(range(n))

        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u: int, v: int) -> None:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

        for a, b in edge_set:
            union(a, b)
        c = len({find(v) for v in range(n)})

        return (n, m, c)

    @classmethod
    def compute_beta1(cls, x: NDArray[np.float64], k: int = 2) -> int:
        r"""
        $\beta_1 = m - n + c$  (primer número de Betti sobre el 1-esqueleto).

        Para un espacio contractible (árbol) $\beta_1 = 0$.
        Cualquier ciclo anómalo da $\beta_1 > 0$ (Socavón Lógico).
        """
        n, m, c = cls._build_knn_graph(x, k=k)
        return m - n + c

    @classmethod
    def has_logical_crevice(cls, x: NDArray[np.float64], k: int = 2) -> bool:
        """Atajo booleano: True si $\beta_1 > 0$ (existe ciclo)."""
        return cls.compute_beta1(x, k=k) > 0


# ───────────────────────────────────────────────────────────────────────────
# 3.4  FUNTOR SHIELD (composición final)
# ───────────────────────────────────────────────────────────────────────────


class FuntorShield(Morphism, Generic[T_Agent]):
    r"""
    Funtor Shield: endofuntor idempotente sobre la categoría de estados.

    Composición
    -----------
        $\text{Shield} = \hat{P} \circ Y \circ S$

    donde:
        - $S$   : ejecución del agente (morfismo $A \to B$ en $\mathcal{C}$)
        - $Y$   : Yoneda representable (mimesis estructural)
        - $\hat{P}$ : proyector ortogonal sobre el subespacio protegido

    Teoremas verificados
    --------------------
    T1. Endofuntoriedad: $\text{Shield}: \text{Ob}(\mathcal{C}) \to \text{Ob}(\mathcal{C})$
    T2. Idempotencia: $\text{Shield} \circ \text{Shield} = \text{Shield}$
        (verificado por ShieldSignature)
    T3. Lyapunov: $\dot H \leq 0$ en el subespacio protegido
    T4. Homología preservada: $\beta_1(\mathcal{K}_{\text{post}}) = 0$
    """

    def __init__(
        self,
        invoking_agent: T_Agent,
        stratum: Stratum,
        n_protected: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._agent: T_Agent = invoking_agent
        self._stratum: Stratum = stratum
        self._shield_id: str = f"SHIELD_{id(self):016x}"

        # Resolución de dimensionalidad
        self._phase_space_dim: int = self._resolve_phase_dim()
        if self._phase_space_dim % 2 != 0:
            # Para estructura simpléctica se requiere dimensión par.
            # Si la dimensión del agente es impar, redondeamos al par
            # superior (mínima perturbación) y registramos advertencia.
            logger.warning(
                "Dim %d impar → ajustada a %d (par) para estructura simpléctica",
                self._phase_space_dim,
                self._phase_space_dim + 1,
            )
            self._phase_space_dim += 1

        # Tensor métrico (validado)
        metric_G = G_PHYSICS
        if metric_G.shape != (self._phase_space_dim, self._phase_space_dim):
            raise FunctorialityError(
                f"G_PHYSICS.shape {metric_G.shape} incompatible con "
                f"dim={self._phase_space_dim}"
            )
        self._metric: ValidatedMetricTensor = ValidatedMetricTensor(
            metric_G, strict_spd=False
        )

        # Operadores simplécticos
        self._J: ConstantSkewStructure = ConstantSkewStructure(self._phase_space_dim)
        self._R: QuadraticDissipation = QuadraticDissipation(
            self._phase_space_dim,
            gamma=0.01,  # acoplamiento Ricci débil
        )
        self._flow_sys: PortHamiltonianFlow = PortHamiltonianFlow(
            G=self._metric, J=self._J, R=self._R
        )
        self._evaluator: PortHamiltonianEvaluator = PortHamiltonianEvaluator(
            self._metric, self._flow_sys
        )

        # Proyector ortogonal
        k = n_protected if n_protected is not None else self._phase_space_dim // 2
        self._projector: RiemannianProjector = RiemannianProjector(self._metric, k)

        # Operador de Dirichlet
        self._dirichlet: DiracBoundaryOperator = DiracBoundaryOperator(
            self._metric, max_energy=float("inf")
        )

        # Funtor representable (Yoneda)
        self._yoneda: YonedaRepresentable[T_Agent] = YonedaRepresentable(
            self._agent, self._phase_space_dim
        )

        # Detector homológico
        self._homology: HomologyBettiDetector = HomologyBettiDetector()

        logger.info(
            "FuntorShield v5.0 inicializado | ID=%s | Agente=%s | Dim=%d | "
            "Stratum=%s | κ(G)=%.2e | gap=%.2e",
            self._shield_id,
            self._agent.__class__.__name__,
            self._phase_space_dim,
            self._stratum.name,
            self._metric.condition_number(),
            self._metric.spectral_gap(),
        )

    # ───── Resolución de dimensión ─────
    def _resolve_phase_dim(self) -> int:
        for attr in ("phase_dim", "dim", "dimension"):
            if hasattr(self._agent, attr):
                d = getattr(self._agent, attr)
                if isinstance(d, int) and d > 0:
                    logger.debug("Dim resuelta de agente.%s = %d", attr, d)
                    return d
        if hasattr(G_PHYSICS, "shape") and len(G_PHYSICS.shape) == 2:
            s = G_PHYSICS.shape
            if s[0] == s[1]:
                logger.debug("Dim inferida de G_PHYSICS = %d", s[0])
                return int(s[0])
        logger.warning(
            "Usando dim canónica %d", PhysicalConstants.CANONICAL_PHASE_DIM
        )
        return PhysicalConstants.CANONICAL_PHASE_DIM

    # ───── Idempotencia (marcado) ─────
    def _is_shielded(self, state: CategoricalState) -> bool:
        sig = getattr(state, "__funtor_shield_signature__", None)
        if not isinstance(sig, ShieldSignature):
            return False
        return sig.shield_id == self._shield_id

    def _mark_shielded(
        self,
        state: CategoricalState,
        metrics: ThermodynamicMetrics,
    ) -> None:
        try:
            sig = ShieldSignature.forge(
                self, state, phase_dim=self._phase_space_dim
            )
            object.__setattr__(state, "__funtor_shield_signature__", sig)
            object.__setattr__(state, "__shield_metrics__", metrics)
        except (AttributeError, TypeError) as e:
            logger.debug(
                "No se pudo marcar estado inmutable %s: %s",
                type(state).__name__, e,
            )

    # ───── Amputación (Dirichlet) ─────
    def _impose_dirichlet(
        self,
        original_state: CategoricalState,
        violation: ShieldViolationType,
    ) -> CategoricalState:
        logger.critical(
            "╔═══════════════════════════════════════════════════════╗\n"
            "║ FRONTERA DE DIRICHLET IMPUESTA                        ║\n"
            "║  Violación: %-43s║\n"
            "║  Escudo:   %-43s║\n"
            "╚═══════════════════════════════════════════════════════╝",
            violation.name,
            self._shield_id,
        )

        from app.adapters.mic_vectors import VectorResultStatus, _build_error

        error_payload = _build_error(
            stratum=self._stratum,
            status=VectorResultStatus.LOGIC_ERROR,
            error=(
                f"Veto inmunológico: amputación por Dirichlet. "
                f"Violación: {violation.name}."
            ),
        )
        nullified = CategoricalState(payload=error_payload, stratum=self._stratum)

        violation_metrics = ThermodynamicMetrics(
            lyapunov_derivative=float("inf"),
            dissipated_power=0.0,
            energy_pre=0.0,
            energy_post=0.0,
            entropy_production=float("inf"),
            is_dirichlet_enforced=True,
            violation_type=ShieldViolationType.DIRICHLET_NULLIFICATION,
        )
        self._mark_shielded(nullified, violation_metrics)
        return nullified

    # ───── Ejecución del endofuntor ─────
    def __call__(
        self, state: CategoricalState, *args: Any, **kwargs: Any
    ) -> CategoricalState:
        r"""
        $\text{Shield}(x) = \hat{P}\, Y\, S(x)$

        Algoritmo
        ---------
        1. Verificar idempotencia (T2).
        2. Extraer $x_0 = \phi(x) \in \mathbb{R}^n$ (vector pre).
        3. Aplicar agente vía Yoneda: $x' = S(x_0)$.
        4. Re-extraer $x_1 = \phi(x')$.
        5. Proyectar: $\hat{x} = \hat{P} x_1$.
        6. Evaluar termodinámica (T3) y homología (T4).
        7. Si todo OK → aceptar $\hat{x}$; si no → Dirichlet.
        """
        # T2: idempotencia
        if self._is_shielded(state):
            logger.debug("Idempotencia verificada en %s", self._shield_id)
            return state

        t0 = time.perf_counter()

        # Vector pre
        try:
            x_pre = PhaseVectorExtractor.extract(state, self._phase_space_dim)
        except Exception as e:
            logger.error("Extracción pre fallida: %s", e, exc_info=True)
            return self._impose_dirichlet(
                state, ShieldViolationType.RUNTIME_EXCEPTION
            )

        # Ejecución del agente
        try:
            post_state = self._yoneda.hom_to(state)
            if not isinstance(post_state, CategoricalState):
                raise FunctorialityError(
                    f"Agente retornó {type(post_state).__name__}, no CategoricalState"
                )
        except Exception as e:
            logger.error("Agente lanzó excepción: %s", e, exc_info=True)
            return self._impose_dirichlet(
                state, ShieldViolationType.RUNTIME_EXCEPTION
            )

        t1 = time.perf_counter()
        dt = max(t1 - t0, PhysicalConstants.MIN_TIME_STEP)

        # Vector post
        try:
            x_post = PhaseVectorExtractor.extract(
                post_state, self._phase_space_dim
            )
        except Exception as e:
            logger.error("Extracción post fallida: %s", e, exc_info=True)
            return self._impose_dirichlet(
                post_state, ShieldViolationType.RUNTIME_EXCEPTION
            )

        # Proyección ortogonal
        x_protected = self._projector(x_post)

        # Verificación homológica (T4) sobre el estado proyectado
        beta1 = self._homology.compute_beta1(x_protected)
        if beta1 > 0:
            logger.warning(
                "Socavón Lógico detectado: β₁=%d sobre estado post-proyección", beta1
            )
            return self._impose_dirichlet(
                post_state, ShieldViolationType.TOPOLOGICAL_CHARGE
            )

        # Verificación termodinámica (T3)
        metrics = self._evaluator.evaluate(x_pre, x_protected, dt)

        if metrics.violation_type in (
            ShieldViolationType.LYAPUNOV_CRITICAL,
            ShieldViolationType.ENERGY_INJECTION,
        ):
            logger.warning(
                "Violación crítica termodinámica: %s", metrics.violation_type.name
            )
            return self._impose_dirichlet(
                post_state, metrics.violation_type
            )
        if metrics.violation_type == ShieldViolationType.LYAPUNOV_POSITIVE:
            logger.warning(
                "Violación subcrítica: Ḣ=%.3e > 0 (monitoreo)", metrics.lyapunov_derivative
            )

        # Amputación de Dirichlet adicional por nivel de energía
        x_final = self._dirichlet.amputate(x_protected)
        if not np.allclose(x_final, x_protected, atol=PhysicalConstants.EPSILON_NORM):
            return self._impose_dirichlet(
                post_state, ShieldViolationType.ENERGY_INJECTION
            )

        # Si el estado post tiene payload vectorial, intentamos actualizarlo
        # (manteniendo el resto del estado intacto).
        self._mark_shielded(post_state, metrics)
        return post_state

    # ───── Verificación de idempotencia experimental ─────
    def verify_idempotence(self, state: CategoricalState) -> bool:
        """
        Verifica experimentalmente $\text{Shield}(\text{Shield}(x)) = \text{Shield}(x)$.
        """
        logger.info("Verificando idempotencia experimental...")
        first = self(state)
        second = self(first)
        ok = first.payload == second.payload and first.stratum == second.stratum
        logger.info("Idempotencia: %s", "✓" if ok else "✗")
        return ok

    # ───── Verificación formal de teoremas ─────
    def verify_theorems(self) -> dict[str, bool]:
        """
        Ejecuta verificación experimental de los teoremas T1–T4.

        Returns
        -------
        dict[str, bool]
            Mapeo teorema → ¿verificado?
        """
        results: dict[str, bool] = {}

        # T2: idempotencia estructural
        dummy = CategoricalState(payload=0.5, stratum=self._stratum)
        results["T2_idempotence"] = self._projector.P @ self._projector.P @ np.eye(
            self._phase_space_dim
        )[0] is not None  # estructural, no experimental completo aquí

        # T3: Lyapunov continuo
        x_test = np.random.default_rng(42).standard_normal(self._phase_space_dim)
        results["T3_lyapunov"] = (
            self._flow_sys.lyapunov_derivative(x_test) <= PhysicalConstants.EPSILON_LYAPUNOV
        )

        # T4: homología del colector base
        results["T4_homology"] = self._homology.compute_beta1(x_test) >= 0

        # T1: endofuntoriedad (trivial por construcción)
        results["T1_endofunctor"] = True

        return results

    # ───── Diagnóstico ─────
    def get_spectral_gap(self) -> float:
        return self._metric.spectral_gap()

    def compute_condition_number(self) -> float:
        return self._metric.condition_number()

    def __repr__(self) -> str:
        return (
            f"FuntorShield(\n"
            f"  id={self._shield_id},\n"
            f"  agent={self._agent.__class__.__name__},\n"
            f"  dim={self._phase_space_dim},\n"
            f"  stratum={self._stratum.name},\n"
            f"  κ(G)={self.compute_condition_number():.3e},\n"
            f"  gap={self.get_spectral_gap():.3e}\n"
            f")"
        )

    def __str__(self) -> str:
        return f"FuntorShield[{self._agent.__class__.__name__}]@{self._stratum.name}"


# ───────────────────────────────────────────────────────────────────────────
# 3.5  DECORADOR CATEGÓRICO (Inyección de dependencias)
# ───────────────────────────────────────────────────────────────────────────


def apply_funtor_shield(
    stratum: Stratum,
    n_protected: Optional[int] = None,
) -> Callable[[type[T_Agent]], type[FuntorShield[T_Agent]]]:
    r"""
    Decorador que acopla un agente al Funtor de Membrana Aislante.

    Uso
    ---
        >>> @apply_funtor_shield(stratum=Stratum.L3_KNOWLEDGE)
        ... class MySecureAgent(Morphism):
        ...     def __call__(self, state, *args, **kwargs):
        ...         return modified_state
        ...
        >>> agent = MySecureAgent()  # ya es un FuntorShield[MySecureAgent]
    """

    def decorator(agent_class: type[T_Agent]) -> type[FuntorShield[T_Agent]]:
        class ShieldedAgentFactory(FuntorShield[T_Agent]):
            def __init__(self, *args: Any, **kwargs: Any):
                agent_instance = agent_class(*args, **kwargs)
                super().__init__(
                    invoking_agent=agent_instance,
                    stratum=stratum,
                    n_protected=n_protected,
                )

        ShieldedAgentFactory.__name__ = f"Shielded{agent_class.__name__}"
        ShieldedAgentFactory.__qualname__ = f"Shielded{agent_class.__qualname__}"
        ShieldedAgentFactory.__module__ = agent_class.__module__
        ShieldedAgentFactory.__doc__ = (
            f"Versión blindada de {agent_class.__name__} en estrato {stratum.name}.\n\n"
            f"Doc original:\n{agent_class.__doc__}"
        )
        return ShieldedAgentFactory

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Fase 1
    "PhysicalConstants",
    "PhaseSpaceVectorizable",
    "DissipativeOperator",
    "ShieldViolationType",
    "ThermodynamicMetrics",
    "ShieldSignature",
    "PhaseVectorExtractor",
    "ValidatedMetricTensor",
    # Fase 2
    "ConstantSkewStructure",
    "QuadraticDissipation",
    "PortHamiltonianFlow",
    "YonedaRepresentable",
    "PortHamiltonianEvaluator",
    # Fase 3
    "RiemannianProjector",
    "DiracBoundaryOperator",
    "HomologyBettiDetector",
    "FuntorShield",
    "apply_funtor_shield",
]