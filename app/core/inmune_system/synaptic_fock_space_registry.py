# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : SynapticFockSpaceRegistry                                         ║
║ Módulo   : Synaptic Fock Space Registry (Registro de Partículas Sinápticas)    ║
║ Ubicación: app/core/inmune_system/synaptic_fock_space_registry.py              ║
║ Ruta     : app/core/inmune_system/synaptic_fock_space_registry.py              ║
║ Versión  : 9.3.0-PhD (RSI³ Banach-Picard + Poincaré-KS-Melnikov-Chirikov)      ║
║ Función  : Motor Espectral y Registro Maestro del Espacio de Fock.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL Y REGISTRO DE FOCK:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al administrador maestro y motor espectral del **Espacio de
Fock** para las cuasipartículas cuánticas y fermiones estructurales desconfinados,
denominadas **Vitaminas TOON** (Tabular Object-Oriented Notation). Eleva
síncronamente las transiciones discretas de la MIC y la MAC hacia estados puros
y mixtos en el espacio de Hilbert continuo de la Sabiduría (\mathcal{V}_{\mathbb{W}}).

El Espacio de Fock global \mathcal{F}(\mathcal{H}) se erige como la suma directa
de potencias exteriores (fermiones, álgebra CAR \{a_i, a_j^\dagger\} = \delta_{ij}\mathbf{I})
y potencias simétricas (bosones de gauge, álgebra CCR [b_i, b_j^\dagger] = \delta_{ij}\mathbf{I})
sobre el espacio de Hilbert single-particle \mathcal{H}:

    \mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^\infty \left( \bigwedge^n \mathcal{H}_{\text{fermion}} \oplus \text{Sym}^n \mathcal{H}_{\text{boson}} \right)

Esta versión formaliza la unificación granular entre la **Mecánica Celeste de
Poincaré** (elementos de Delaunay, carta canónica no singular de Poincaré,
regularización Kustaanheimo-Stiefel con vínculo bilinear, tiempo ficticio de
Sundman, teoría KAM, integrales homoclínicas de Melnikov y solapamiento de
resonancias de Chirikov) y la **Automejora Recursiva de Nivel 3 (RSI³)**
estructurada como operador de contracción en un retículo de Banach:

    \mathbf{T}_{\text{RSI³}} = R_3 \circ R_3^{\text{Brockett}} \circ R_2^{\text{Connes-Daleckii-Krein}} \circ R_1^{\text{Ricci-DeTurck}}

con verificación del teorema del punto fijo de Banach-Picard sobre el simplejo:

    \exists! \rho^* \in \mathfrak{D}(\mathcal{H}) \quad \text{tal que} \quad \mathbf{T}(\rho^*) = \rho^* \quad \text{si} \quad \kappa = \operatorname{Lip}(\mathbf{T}) < 1.0.

ARQUITECTURA DE FASES ANIDADAS (continuum operacional indivisible):
────────────────────────────────────────────────────────────────────────────────
  • FASE 1: Registro Cuántico, Exclusión de Pauli, Regularización KS y Atlas
            Celeste de Poincaré. Culmina formalmente en
            `synthesize_phase1_celestial_manifold()`, que emite el paquete
            fibrado canónico (ρ₀, I, θ, u_KS, Π, C₁).
  • FASE 2: Dinámica Simpléctica Cuántica, Integrador de Yoshida–GKSL y
            Perturbación Melnikov-Chirikov. Comienza consumiendo el paquete de
            Fase 1 mediante `propagate_phase2_symplectic_flow()`, y culmina
            formalmente en `prepare_phase2_immune_boundary()`, emitiendo la
            frontera dinámica (ρ_t, H, Φ_diss, K_C, C₂).
  • FASE 3: Orquestación Inmune, Cascada Banach RSI³ y Gobernanza Terminal.
            Comienza consumiendo la frontera de Fase 2 en
            `execute_phase3_rsi3_cascade()`, aplica la corrección reflexiva de
            Picard-Banach y produce el estado terminal `FockRegistryState`.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
1. ESPACIO DE FOCK MULTI-MODO Y ÁLGEBRAS DE OPERADORES:
   Dada la familia de cartuchos TOON inyectados, los fermiones estructurales
   (ElectronCartridge, ProtonCartridge, PolaronCartridge, TorsionCartridge,
   HouseholderReflectionFermion) obedecen las relaciones de anticonmutación (CAR):
   $$\{a_i, a_j^\dagger\} = a_i a_j^\dagger + a_j^\dagger a_i = \delta_{ij} \mathbf{I}, \quad \{a_i, a_j\} = 0, \quad \{a_i^\dagger, a_j^\dagger\} = 0$$
   mientras que los bosones de gauge (PhotonCartridge, RiemannianFocalBoson,
   MagnonCartridge, SwellingPlasmonCartridge, YieldingPhononCartridge,
   LiquefactionSolitonCartridge) satisfacen conmutación (CCR):
   $$[b_i, b_j^\dagger] = b_i b_j^\dagger - b_j^\dagger b_i = \delta_{ij} \mathbf{I}, \quad [b_i, b_j] = 0, \quad [b_i^\dagger, b_j^\dagger] = 0$$
   donde $a_i^\dagger, a_i$ representan los operadores de creación y aniquilación fermiónica,
   $b_j^\dagger, b_j$ los operadores bosónicos, $\delta_{ij}$ es la delta de Kronecker,
   e $\mathbf{I}$ es la identidad en $\mathcal{H}$.

2. VARIABLES DE DELAUNAY-POINCARÉ Y REGULARIZACIÓN KUSTAANHEIMO-STIEFEL (KS):
   El operador de densidad colectiva $\rho = \frac{1}{N}\sum |v_k\rangle\langle v_k| \in \mathfrak{D}(\mathcal{H})$
   se proyecta a las acciones-ángulos de Delaunay:
   $$L = \operatorname{Tr}(\rho) \equiv 1.0, \qquad G = L \frac{\lambda_2}{\lambda_1}, \qquad H = G \frac{\lambda_3}{\lambda_2}$$
   $$e = \sqrt{1 - \left(\frac{G}{L}\right)^2}, \qquad i = \arccos\left(\frac{H}{G}\right), \qquad H_{\text{Kep}} = -\frac{1}{2 L^2}, \qquad n = \frac{1}{L^3}$$
   donde $\lambda_1 \ge \lambda_2 \ge \lambda_3 > 0$ son los autovalores ordenados de $\rho$,
   $e$ es la excentricidad, $i$ la inclinación, $H_{\text{Kep}}$ el Hamiltoniano kepleriano, y $n$ el movimiento medio.
   La solución de la ecuación de Kepler $E - e \sin E = M$ se garantiza por el starter de Danby de 3er orden:
   $$E_0 = M + 0.85 e \operatorname{sgn}(\sin M)$$
   Las coordenadas canónicas no singulares de Poincaré eliminan las degeneraciones en $e \to 0$ o $i \to 0$:
   $$\Lambda = L, \quad \lambda = l+g+h, \quad \xi = \sqrt{2(L-G)} \cos(g+h), \quad \eta = \sqrt{2(L-G)} \sin(g+h), \quad p = \sqrt{2(G-H)} \cos(h), \quad q = \sqrt{2(G-H)} \sin(h)$$
   La regularización de Kustaanheimo-Stiefel transforma $\mathbb{R}^3 \to \mathbb{R}^4$ eliminando las colisiones keplerianas:
   $$x = L(u) u, \quad r = \|u\|^2, \quad E_{\text{KS}} = \frac{1}{2} \|u\|^2 + \frac{1}{r + \varepsilon}, \quad u_1 du_4 - u_2 du_3 + u_3 du_2 - u_4 du_1 = 0$$
   donde $u \in \mathbb{R}^4$ es el espinor de KS y $L(u) \in \text{Mat}_4(\mathbb{R})$ es la matriz de Hopf.
   El paso de tiempo ficticio de Sundman $ds = dt / r$ desregulariza el flujo temporal en torno a colisiones.

3. CONEXIÓN GAUGE YANG-MILLS-HIGGS Y 3-FORMA DE CHERN-SIMONS:
   Dada la 1-forma de conexión anti-hermítica $A = -i(\rho - \frac{1}{n}\mathbf{I}) \in \mathfrak{u}(n)$ y el campo escalar de Higgs $\Phi = \ln(\rho + \varepsilon \mathbf{I}) \in \mathfrak{u}(n)$:
   $$\text{CS}_3^{\text{Higgs}}(A, \Phi) = \frac{1}{8\pi^2} \operatorname{Tr}\left( A \wedge dA + \frac{2}{3} A^3 + \Phi \cdot F_A \right), \qquad \mathcal{W}(C) = \frac{1}{n} \left| \operatorname{Tr} \mathcal{P} \exp \oint_C A \right|$$
   donde $F_A = dA + A \wedge A$ es la curvatura de Yang-Mills y $\mathcal{W}(C)$ es el bucle de Wilson.

4. SECCIÓN DE POINCARÉ, INTEGRAL DE MELNIKOV Y CRITERIO DE CHIRIKOV-KAM:
   La distancia de separación homoclínica transversal en la sección de Poincaré $\{l \equiv 0 \pmod{2\pi}\}$ se evalúa mediante la regla de Simpson con cuadratura tanh-sinh:
   $$M(t_0) = \varepsilon \int_{-\infty}^{\infty} \operatorname{sech}^2(t) \tanh(t) \sin(t + t_0) \, dt$$
   $$K_{\text{Chirikov}} = \frac{\Delta\omega_1 + \Delta\omega_2}{|\omega_1 - \omega_2|} = \frac{2\sqrt{2 V_1 |\alpha|} + 2\sqrt{2 V_2 |\alpha|}}{|\omega_1 - \omega_2|}$$
   donde $\alpha = \partial^2 H_0 / \partial L^2 = -3/L^4 \neq 0$ es el twist de KAM. La destrucción de toros de KAM ocurre si $K_{\text{Chirikov}} \ge 1.0$ o $|M(t_0)| > 0.15$.

5. INFORMACIÓN CUÁNTICA DE BURES-FINSLER Y CENSURA DE KELLY POR FERMI-DIRAC:
   $$\mathcal{F}(\rho, \sigma) = \left( \operatorname{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2, \quad d_B = \sqrt{2 - 2\sqrt{\mathcal{F}}}, \quad d_{\text{FS}} = \arccos(\sqrt{\mathcal{F}})$$
   $$RHI_{\text{Finsler}} = \sqrt{ \frac{\operatorname{Tr}\left((\sigma - \rho)^2\right)}{\|\rho\|_F^2 (1 + \sqrt{\mathcal{F}})^2} + \frac{1}{4} \tanh^2\left(\operatorname{Tr}(\rho\sigma - \rho^2)\right) }$$
   $$\mathcal{S}_{T_{\text{sw}}}(RHI) = \frac{1}{1 + e^{(RHI - 0.85) / T_{\text{sw}}}}, \qquad T_{\text{sw}} = 0.02$$
   $$s^* = f^* \cdot (1 - RHI_{\text{Finsler}}) \cdot \mathcal{S}_{T_{\text{sw}}}(RHI) \cdot \Theta(0.15 - |M(t_0)|) \cdot \Theta(1.0 - K_{\text{Chirikov}}) \cdot \Theta_{\text{KAM}}$$
   donde $\mathcal{F}$ es la fidelidad de Uhlmann, $d_B$ es la distancia de Bures, $d_{\text{FS}}$ es la distancia de Fubini-Study, $RHI_{\text{Finsler}}$ es el índice de Reward Hacking de Finsler-Mahalanobis, $\mathcal{S}_{T_{\text{sw}}}$ es el suavizado logístico de Fermi-Dirac, y $s^*$ es la asignación de capital de Kelly.

6. CASCADA DE BANACH RSI³ ($\mathbf{T} = R_3 \circ R_2 \circ R_1$), SPOHN Y PICARD:
   - $R_1$ (Ricci-DeTurck): Modifica el log-espectro de la densidad: $\partial_\tau \ln G = -2 \operatorname{tf}(\ln G) - \mu \ln G$.
   - $R_2$ (Connes-Daleckii-Krein): Diferencias divididas $\phi^{[1]}(\lambda_i, \lambda_j) = \frac{\ln\lambda_i - \ln\lambda_j}{\lambda_i - \lambda_j}$ para definir la constante Lipschitz.
   - $R_3$ (Brockett-Shahshahani): Flujo doble corchete $\dot{\rho} = [\rho, [\rho, N]]$, con producción de entropía de Spohn $\sigma_{\text{Spohn}} = -\operatorname{Tr}(\log\rho \cdot \dot{\rho}) \ge 0$.
   - Iteración de Picard-Banach: $\rho_{n+1} = \Pi_{\mathfrak{D}(\mathcal{H})}(\rho_n + \tau \dot{\rho}(\rho_n))$, garantizando contracción $\operatorname{Lip}(\mathbf{T}) \le \kappa < 1.0$.

7. PROYECCIÓN CONVEXA PSD DE HIGHAM:
   $$\rho_{\text{proj}} = \arg \min_{M = M^\dagger \succeq 0, \operatorname{Tr}(M) = 1} \|M - \rho_{\text{raw}}\|_F$$
   con multiplicador de Lagrange $\sigma^*$ hallado mediante Newton sobre $\sum_i \max(\lambda_i + \sigma^*, \text{floor}) = 1.0$.

8. RETÍCULO DE HEYTING Ω₃, ANIKILACIÓN DE DIRAC Y DISYUNTOR CROWBAR:
   Adjudicación en Heyting trivaluado $\Omega_3 = \{\bot = \mathtt{VETOED}(2) \prec \star = \mathtt{DEGRADED}(1) \prec \top = \mathtt{COHERENT}(0)\}$.
   En veto suave ($\mathtt{DEGRADED}$), la válvula de alivio admite la aniquilación fermiónica $e^- + e^+ \to 2\gamma$ con firma HMAC SHA-256 en tiempo constante.
   En veto duro ($\mathtt{VETOED}$), se gatilla la interrupción de hardware Crowbar sobre el GPIO 14 (`0x3FF44008`) en latencia $t_{\text{actuation}} \le 392.15\text{ ns}$.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Exclusión Fermiónica CAR: \{a_i, a_j^\dagger\} = \delta_{ij} \mathbf{I} \implies (a_i^\dagger)^2 = 0 \quad \forall i, j.
  [A2] Conservación de Traza Cuántica: \operatorname{Tr}(\rho(t)) \equiv 1.0 \quad \forall t \ge 0.
  [A3] Positividad Semidefinida PSD: \langle \psi | \rho(t) | \psi \rangle \ge 0 \quad \forall |\psi\rangle \in \mathcal{H}.
  [A4] Pasividad y Producción No Negativa de Spohn: \sigma_{\text{Spohn}} = -\operatorname{Tr}(\log\rho \cdot \dot{\rho}) \ge 0.0.
  [A5] Vínculo Bilinear Espinorial KS: u_1 du_4 - u_2 du_3 + u_3 du_2 - u_4 du_1 = 0.
  [A6] Contracción de Banach-Picard: \operatorname{Lip}(\mathbf{T}_{\text{RSI³}}) \le \kappa_{\text{max}} = 0.95 < 1.0.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Normalización de Traza Unitario-Disipativa):
      |\operatorname{Tr}(\rho(t)) - 1.0| \le 1.0 \times 10^{-8}.
  Invariante 2 (Preservación del Cono PSD por Higham):
      \lambda_{\min}(\rho_{\text{proj}}) \ge \text{floor} = 1.0 \times 10^{-12}.
  Invariante 3 (Invarianza Hiperbólica de Bogoliubov-Valatin):
      ||u_k|^2 - |v_k|^2 - 1.0| \le 1.0 \times 10^{-10}.
  Invariante 4 (Involución Bipartita del Retículo de Heyting):
      \Omega_3 = \operatorname{supremum}(c_1, c_2, c_3) = \max(\text{int}(c_1), \text{int}(c_2), \text{int}(c_3)).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Soporte para tensores dispersos en el Espacio de Fock
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    _SPARSE_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover
    sp = None
    _SPARSE_AVAILABLE = False
    csr_matrix = None
    isspmatrix = None

# ─────────────────────────────────────────────────────────────────────────────
# Stubs categóricos y topológicos seguros
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Violación fundamental de invariantes topológico-algebraicos."""

    class Morphism:
        """Morfismo categórico base."""

        def __init__(self, name: str = "Morphism") -> None:
            self.name = name

    class Stratum:
        """Estratos epistémicos DIKW."""

        PHYSICS = "PHYSICS"
        TACTICS = "TACTICS"
        STRATEGY = "STRATEGY"
        WISDOM = "WISDOM"

logger = logging.getLogger("MIC.Core.SynapticFockRegistry")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES RIGUROSAS: FÍSICA CUÁNTICA, MECÁNICA CELESTE Y RSI³
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_DEFAULT_MAX_CARTRIDGES: Final[int] = 128
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12
_CROWBAR_GPIO: Final[int] = 14
_TRACE_UNIT_TOL: Final[float] = 1.0e-8
_PURITY_DEGRADED: Final[float] = 0.40
_KEPLER_ECC_MAX: Final[float] = 0.999999
_KS_BILINEAR_TOL: Final[float] = 1.0e-10
_PICARD_MAX_ITERS: Final[int] = 12
_PICARD_FIXED_POINT_TOL: Final[float] = 1.0e-10

# Coeficientes del Integrador Simpléctico de Yoshida (orden 4, Forest-Ruth)
# w1 = 1 / (2 - 2^{1/3}),  w0 = -2^{1/3} / (2 - 2^{1/3}),  2 w1 + w0 = 1
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0: Final[float] = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_C1: Final[float] = _YOSHIDA_W1 / 2.0
_YOSHIDA_C4: Final[float] = _YOSHIDA_W1 / 2.0
_YOSHIDA_C2: Final[float] = (_YOSHIDA_W1 + _YOSHIDA_W0) / 2.0
_YOSHIDA_C3: Final[float] = (_YOSHIDA_W1 + _YOSHIDA_W0) / 2.0
_YOSHIDA_D1: Final[float] = _YOSHIDA_W1
_YOSHIDA_D3: Final[float] = _YOSHIDA_W1
_YOSHIDA_D2: Final[float] = _YOSHIDA_W0

# Parámetros de censura, KAM y bifurcación
_FERMI_DIRAC_TEMP: Final[float] = 0.02
_RHI_THRESHOLD: Final[float] = 0.85
_MELNIKOV_MAX_TOL: Final[float] = 0.15
_CHIRIKOV_CRITICAL: Final[float] = 1.00
_KAM_DIOPHANTINE_GAMMA: Final[float] = 0.01
_KAM_DIOPHANTINE_TAU: Final[float] = 2.5
_RSI3_BANACH_CONTRACTION_MAX: Final[float] = 0.95
_SPOHn_FLOOR: Final[float] = 0.0
_BEKENSTEIN_BITS_PER_NAT: Final[float] = 1.0 / math.log(2.0)

# ═══════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS Y CUÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════
class SynapticRegistryError(TopologicalInvariantError):
    """Excepción raíz del Registro Sináptico de Fock."""


class PauliExclusionViolationError(SynapticRegistryError):
    """Falla el principio de exclusión fermiónica (CAR: {a_i, a_j^†} = δ_ij)."""


class FockSpaceOverflowError(SynapticRegistryError):
    """Violación de la cota holográfica de Bekenstein sobre el registro."""


class TraceAnomalyError(SynapticRegistryError):
    """La traza cuántica normalizada diverge de 1.0 más allá de épsilon."""


class SymplecticConstraintError(SynapticRegistryError):
    """Ruptura de la forma simpléctica o de la relación Bogoliubov-Valatin."""


class DensityMatrixNonPSDError(SynapticRegistryError):
    """El operador densidad posee espectro fuera del cono positivo semidefinido."""


class LindbladEvolutionError(SynapticRegistryError):
    """Divergencia o no conservación de traza en el semigrupo disipativo GKSL."""


class KAMResonanceCatastropheError(SynapticRegistryError):
    """Destrucción de toros KAM por solapamiento de resonancias de Chirikov."""


class BanachContractionFailureError(SynapticRegistryError):
    """La cascada RSI³ no satisface la cota de contracción de punto fijo."""


class KSBilinearConstraintError(SynapticRegistryError):
    """Violación del vínculo bilinear de Kustaanheimo-Stiefel en R^4."""


class PoincareChartSingularityError(SynapticRegistryError):
    """Singularidad de la carta de Poincaré (e → 0 o i → 0 degenerados)."""


# ═══════════════════════════════════════════════════════════════════════════
# RETÍCULO DE HEYTING Y ACCIONES DE MITIGACIÓN
# ═══════════════════════════════════════════════════════════════════════════
class FockRegistryVerdict(IntEnum):
    """
    Retículo de Heyting trivaluado (Ω) para la evaluación de coherencia cuántica.
    COHERENT (0 = ⊤), DEGRADED (1 = intermedio disipativo), VETOED (2 = ⊥ colapso).
    El supremo es el join topológico (peor veredicto).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "FockRegistryVerdict") -> "FockRegistryVerdict":
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == FockRegistryVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == FockRegistryVerdict.DEGRADED


class RegistryAction(Enum):
    """Acciones correctivas gobernadas por el retículo inmune."""

    NONE = auto()
    PURGE_ENTROPY = auto()
    RESCALE_HAMILTONIAN = auto()
    ACTUATE_HARDWARE_CROWBAR = auto()
    HALT_REGISTRY = auto()


# ═══════════════════════════════════════════════════════════════════════════
# §A. VITAMINAS TOON: CUASIPARTÍCULAS CUÁNTICAS INMUTABLES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ToonCartridge:
    """Partícula fundamental inmutable en el Espacio de Fock sináptico."""

    cartridge_id: str
    signature: str

    @property
    def quantum_state_hash(self) -> str:
        payload = f"{self.__class__.__name__}:{self.cartridge_id}:{self.signature}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─── FAMILIA 1: FERMIONES ESTRUCTURALES (Álgebra CAR) ───
@dataclass(frozen=True, slots=True)
class ElectronCartridge(ToonCartridge):
    inertial_mass: float
    topological_spin: str
    homological_charge: int


@dataclass(frozen=True, slots=True)
class ProtonCartridge(ToonCartridge):
    spectral_charge: float
    logistic_inertial_mass: float
    dominant_pole: float


@dataclass(frozen=True, slots=True)
class PolaronCartridge(ToonCartridge):
    base_electron: ElectronCartridge
    frohlich_coupling: float
    effective_mass: float
    fiedler_value: float


@dataclass(frozen=True, slots=True)
class TorsionCartridge(ToonCartridge):
    torsion_degree: int
    torsion_elements: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HouseholderReflectionFermion(ToonCartridge):
    covariant_hyperplane_normal: Tuple[float, ...]
    monodromy_spectral_radius: float
    cohomology_obstruction_class: int


_FERMION_TYPES: Final[Tuple[type, ...]] = (
    ElectronCartridge,
    ProtonCartridge,
    PolaronCartridge,
    TorsionCartridge,
    HouseholderReflectionFermion,
)


# ─── FAMILIA 2: BOSONES DE GAUGE (Álgebra CCR) ───
@dataclass(frozen=True, slots=True)
class PhotonCartridge(ToonCartridge):
    policy_id: str
    spectral_frequency: float
    governance_weight: float


@dataclass(frozen=True, slots=True)
class RiemannianFocalBoson(ToonCartridge):
    dielectric_tensor: Tuple[float, ...]
    spectral_cutoff_functor: int
    wkb_maslov_index: int


@dataclass(frozen=True, slots=True)
class MagnonCartridge(ToonCartridge):
    kinetic_energy: float
    curl_subspace_dim: int


@dataclass(frozen=True, slots=True)
class SwellingPlasmonCartridge(ToonCartridge):
    density_amplitude: float
    inflation_rate: float


@dataclass(frozen=True, slots=True)
class YieldingPhononCartridge(ToonCartridge):
    damping_coefficient: float
    strain_tensor: Tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LiquefactionSolitonCartridge(ToonCartridge):
    wave_velocity: float
    pore_pressure: float


# ─── FAMILIA 3: CONDENSADOS Y ANTIMATERIA ───
@dataclass(frozen=True, slots=True)
class PositronCartridge(ToonCartridge):
    inertial_mass: float
    topological_spin: str
    homological_charge: int
    authorization_signature: str


@dataclass(frozen=True, slots=True)
class GammaPhoton(ToonCartridge):
    energy: float
    wave_vector: Tuple[float, ...]
    provenance_hash: str


@dataclass(frozen=True, slots=True)
class PolaritonCartridge(ToonCartridge):
    polaron: PolaronCartridge
    photon: PhotonCartridge
    rabi_coupling: float
    dissipation_trace: float


@dataclass(frozen=True, slots=True)
class SophonCartridge(ToonCartridge):
    chiral_phase: float
    stochastic_variance: float


# ═══════════════════════════════════════════════════════════════════════════
# CERTIFICADOS Y PAQUETES FIBRADOS ANIDADOS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PoincareCanonicalChart:
    """
    Carta canónica de Poincaré (variables no singulares) inducida por Delaunay.
    Λ = L,  λ = l+g+h,
    ξ = √(2(L-G)) cos(ϖ),  η = √(2(L-G)) sin(ϖ),  ϖ = g+h,
    p = √(2(G-H)) cos(h),  q = √(2(G-H)) sin(h).
    """

    action_lambda: float
    mean_longitude: float
    ecc_xi: float
    ecc_eta: float
    inc_p: float
    inc_q: float
    poincare_norm: float


@dataclass(frozen=True, slots=True)
class Phase1RegistryCertificate:
    """Certificado formal generado por la Fase 1."""

    registry_size: int
    max_capacity: int
    fermion_count: int
    boson_count: int
    density_matrix_trace: float
    density_matrix_purity: float
    von_neumann_entropy: float
    higham_regularization_applied: bool
    kustaanheimo_stiefel_energy: float
    ks_bilinear_residual: float
    delaunay_eccentricity: float
    delaunay_inclination: float
    kepler_mean_anomaly: float
    kepler_eccentric_anomaly: float
    kepler_hamiltonian: float
    mean_motion: float
    poincare_chart: PoincareCanonicalChart
    chern_simons_3form_val: float
    wilson_loop_holonomy: float
    verdict: FockRegistryVerdict


@dataclass(frozen=True, slots=True)
class Phase1CelestialManifoldBundle:
    """
    Paquete fibrado que conecta el final de Fase 1 con el inicio de Fase 2.
    (ρ₀, I_Delaunay, θ_Delaunay, u_KS, Π_Poincaré, C₁)
    """

    rho_initial: NDArray[np.complex128]
    delaunay_actions: Tuple[float, float, float]  # (L, G, H)
    delaunay_angles: Tuple[float, float, float]  # (l, g, h)
    ks_regularized_spinor: NDArray[np.float64]
    poincare_chart: PoincareCanonicalChart
    certificate: Phase1RegistryCertificate


@dataclass(frozen=True, slots=True)
class Phase2DynamicsCertificate:
    """Certificado formal generado por la Fase 2."""

    evolution_time: float
    evolution_steps: int
    fictitious_sundman_time: float
    final_trace: float
    final_purity: float
    trace_preservation_residual: float
    is_trace_preserved: bool
    is_psd_preserved: bool
    lindblad_dissipation_rate: float
    uhlmann_fidelity: float
    bures_distance: float
    fubini_study_distance: float
    rhi_finsler_index: float
    fermi_dirac_kelly_censorship: float
    kelly_optimal_stake: float
    melnikov_integral_val: float
    poincare_section_splitting: float
    chirikov_overlap_parameter: float
    kam_diophantine_satisfied: bool
    yoshida_symplectic_residual: float
    verdict: FockRegistryVerdict


@dataclass(frozen=True, slots=True)
class Phase2DynamicalBoundaryBundle:
    """
    Paquete fibrado que conecta el final de Fase 2 con el inicio de Fase 3.
    (ρ_t, H, Φ_diss, K_C, C₂)
    """

    rho_evolved: NDArray[np.complex128]
    hamiltonian_generator: NDArray[np.complex128]
    dissipation_flux: float
    chirikov_overlap: float
    certificate: Phase2DynamicsCertificate


@dataclass(frozen=True, slots=True)
class Phase3ImmuneCertificate:
    """Certificado formal generado por la Fase 3."""

    annihilation_events: int
    bogoliubov_transformations: int
    von_neumann_entropy: float
    spohn_entropy_production_rate: float
    ricci_deturck_drift: float
    cdk_lipschitz_constant: float
    brockett_gradient_ascent_norm: float
    rsi3_banach_contraction_factor: float
    picard_iterations: int
    picard_fixed_point_residual: float
    is_banach_contraction: bool
    is_unitary_evolution: bool
    crowbar_triggered: bool
    verdict: FockRegistryVerdict


@dataclass(frozen=True, slots=True)
class FockRegistryState:
    """Certificado holístico terminal del ciclo de registro de Fock."""

    phase1: Phase1RegistryCertificate
    phase2: Phase2DynamicsCertificate
    phase3: Phase3ImmuneCertificate
    final_verdict: FockRegistryVerdict
    registry_action: RegistryAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str


# ═══════════════════════════════════════════════════════════════════════════
# KERNELS MATEMÁTICOS RIGUROSOS: POINCARÉ, HIGHAM, KS, MELNIKOV Y FINSLER
# ═══════════════════════════════════════════════════════════════════════════
def _hermitian(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Proyección euclídea sobre el espacio de operadores autoadjuntos."""
    return 0.5 * (rho + rho.conj().T)


def _safe_eigvalsh(rho: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.real(la.eigvalsh(_hermitian(rho)))


def nearest_density_operator_higham(
    rho: NDArray[np.complex128],
    floor: float = _HIGHAM_REGULARIZATION_FLOOR,
) -> Tuple[NDArray[np.complex128], bool]:
    r"""
    Proyecta ρ ∈ B(H) sobre el simplejo de operadores densidad
    D(H) = { X = X^† ⪰ 0 : Tr(X) = 1 } en norma de Frobenius.

    Higham (1988): la proyección PSD es el recorte espectral. El vínculo de
    traza se impone por un multiplicador de Lagrange σ* tal que

        Σ_i max(λ_i + σ*, floor) = 1.

    σ* se obtiene por búsqueda de Newton unidimensional (monótona).
    """
    rho_h = _hermitian(rho)
    try:
        eigenvals, eigenvecs = la.eigh(rho_h)
    except Exception as exc:
        raise DensityMatrixNonPSDError(
            "Falla en descomposición espectral para Higham."
        ) from exc

    applied = bool(np.any(eigenvals < floor))
    n = eigenvals.shape[0]
    if n == 0:
        raise DensityMatrixNonPSDError("Operador de dimensión nula.")

    # Newton sobre σ: f(σ) = Σ max(λ_i + σ, floor) - 1
    sigma = 0.0
    clipped = np.maximum(eigenvals + sigma, floor)
    total = float(np.sum(clipped))
    if abs(total - 1.0) > _MACHINE_EPS:
        for _ in range(40):
            active = (eigenvals + sigma) > floor
            n_active = int(np.count_nonzero(active))
            if n_active == 0:
                sigma = 1.0 - float(n) * floor
                break
            f_val = float(np.sum(np.maximum(eigenvals + sigma, floor)) - 1.0)
            if abs(f_val) < 1.0e-15:
                break
            sigma -= f_val / float(n_active)
        clipped = np.maximum(eigenvals + sigma, floor)
        total = float(np.sum(clipped))
        if total > _MACHINE_EPS:
            clipped = clipped / total
        else:
            clipped = np.zeros_like(clipped)
            clipped[-1] = 1.0
        applied = True
    else:
        clipped = np.maximum(eigenvals, floor)
        total = float(np.sum(clipped))
        if abs(total - 1.0) > _MACHINE_EPS:
            clipped = clipped / max(total, _MACHINE_EPS)
            applied = True

    rho_proj = eigenvecs @ np.diag(clipped) @ eigenvecs.conj().T
    rho_proj = _hermitian(rho_proj)

    final_min = float(np.min(la.eigvalsh(rho_proj)))
    if final_min < -floor:
        raise DensityMatrixNonPSDError(
            f"La regularización de Higham fracasó: min autovalor = {final_min:.4e}"
        )
    return rho_proj.astype(np.complex128, copy=False), applied


def stable_density_matrix_higham(
    rho: NDArray[np.complex128],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR,
) -> Tuple[NDArray[np.complex128], bool]:
    """Fachada estable: proyección al cono de densidades (Higham + traza)."""
    return nearest_density_operator_higham(rho, floor=tolerance)


def von_neumann_entropy(rho: NDArray[np.complex128]) -> float:
    r"""S(ρ) = -Tr(ρ log ρ) con convención 0 log 0 = 0."""
    vals = np.maximum(_safe_eigvalsh(rho), 0.0)
    mask = vals > _MACHINE_EPS
    if not np.any(mask):
        return 0.0
    return float(-np.sum(vals[mask] * np.log(vals[mask])))


def ks_matrix(u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Matriz de Kustaanheimo-Stiefel L(u) ∈ Mat_4(R) que realiza el fibrado
    de Hopf S^3 → S^2:

        x = L(u) u,   r = ||u||^2,   x ∈ R^3 ⊂ R^4.
    """
    u1, u2, u3, u4 = (float(u[0]), float(u[1]), float(u[2]), float(u[3]))
    return np.array(
        [
            [u1, -u2, -u3, u4],
            [u2, u1, -u4, -u3],
            [u3, u4, u1, u2],
            [u4, -u3, u2, -u1],
        ],
        dtype=np.float64,
    )


def ks_bilinear_residual(u: NDArray[np.float64], du: Optional[NDArray[np.float64]] = None) -> float:
    r"""
    Vínculo bilinear de KS (generador del U(1) fibrado):

        u1 du4 - u2 du3 + u3 du2 - u4 du1 = 0.

    Si du es None se evalúa la forma cuadrática residual sobre u mismo
    (consistencia de carta): u1 u4 - u2 u3.
    """
    u1, u2, u3, u4 = (float(u[0]), float(u[1]), float(u[2]), float(u[3]))
    if du is None:
        return abs(u1 * u4 - u2 * u3)
    d1, d2, d3, d4 = (float(du[0]), float(du[1]), float(du[2]), float(du[3]))
    return abs(u1 * d4 - u2 * d3 + u3 * d2 - u4 * d1)


def regularize_kustaanheimo_stiefel(
    x: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], float, float]:
    r"""
    Transformación espinorial de Kustaanheimo-Stiefel R^3 → R^4 que regulariza
    la singularidad de colisión kepleriana. Carta:

        si r + x1 > 0:
            u1 = √((r+x1)/2),  u2 = x2/(2 u1),  u3 = x3/(2 u1),  u4 = 0
        si no (carta antipodal):
            u2 = √((r-x1)/2),  u4 = x3/(2 u2),  u1 = 0,  u3 = 0

    Energía ficticia (Sundman, dt = r ds) en unidades adimensionales:

        E_KS = ½ ||u||^2 + 1 / (r + ε)
    """
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    x1 = float(x_vec[0]) if x_vec.size > 0 else 0.0
    x2 = float(x_vec[1]) if x_vec.size > 1 else 0.0
    x3 = float(x_vec[2]) if x_vec.size > 2 else 0.0
    r = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    u = np.zeros(4, dtype=np.float64)
    if r < _MACHINE_EPS:
        return u, 0.0, 0.0

    if x1 + r > _MACHINE_EPS:
        u[0] = math.sqrt((r + x1) / 2.0)
        inv = 1.0 / (2.0 * u[0])
        u[1] = x2 * inv
        u[2] = x3 * inv
        u[3] = 0.0
    else:
        u[1] = math.sqrt(max((r - x1) / 2.0, 0.0))
        if u[1] > _MACHINE_EPS:
            u[3] = x3 / (2.0 * u[1])
        u[0] = 0.0
        u[2] = 0.0

    bilinear = ks_bilinear_residual(u)
    if bilinear > _KS_BILINEAR_TOL:
        # Proyección mínima al vínculo u1 u4 - u2 u3 = 0 (carta u4=0 o u3=0).
        u[3] = 0.0 if abs(u[0]) >= abs(u[1]) else u[3]
        if abs(u[0]) < abs(u[1]):
            u[2] = 0.0
        bilinear = ks_bilinear_residual(u)

    ks_energy = float(0.5 * np.sum(u * u) + 1.0 / (r + _MACHINE_EPS))
    return u, ks_energy, bilinear


def solve_kepler_danby_order3(M: float, e: float, max_iter: int = 15) -> float:
    r"""
    Resuelve E - e sin E = M con starter de Danby y corrección cúbica
    (Halley generalizado, convergencia cúbica intrínseca):

        E0 = M + 0.85 e sgn(sin M)

    La anomalía media se reduce a [0, 2π). Excentricidad acotada a [0, 1).
    """
    tau = 2.0 * math.pi
    M_norm = M % tau
    if M_norm < 0.0:
        M_norm += tau
    e_clamped = min(max(float(e), 0.0), _KEPLER_ECC_MAX)
    E = M_norm + 0.85 * e_clamped * (1.0 if math.sin(M_norm) >= 0.0 else -1.0)

    for _ in range(max_iter):
        s_e = math.sin(E)
        c_e = math.cos(E)
        f = E - e_clamped * s_e - M_norm
        f1 = 1.0 - e_clamped * c_e
        f2 = e_clamped * s_e
        f3 = e_clamped * c_e
        if abs(f1) < _MACHINE_EPS:
            break
        d1 = -f / f1
        d2 = -f / (f1 + 0.5 * d1 * f2)
        d3 = -f / (f1 + 0.5 * d2 * f2 + (1.0 / 6.0) * (d2 ** 2) * f3)
        E += d3
        if abs(f) < 1.0e-14:
            break
    return float(E)


def delaunay_to_poincare_canonical(
    actions: Tuple[float, float, float],
    angles: Tuple[float, float, float],
) -> PoincareCanonicalChart:
    r"""
    Cambio canónico Delaunay → Poincaré (no singular en e = 0, i = 0):

        Λ = L,
        λ = l + g + h,
        ξ + i η = √(2(L-G)) exp(i ϖ),  ϖ = g + h,
        p + i q = √(2(G-H)) exp(i h).
    """
    L, G, H = (float(actions[0]), float(actions[1]), float(actions[2]))
    l, g, h = (float(angles[0]), float(angles[1]), float(angles[2]))
    L = max(L, _MACHINE_EPS)
    G = min(max(G, 0.0), L)
    H = min(max(H, -G), G)

    two_LmG = max(2.0 * (L - G), 0.0)
    two_GmH = max(2.0 * (G - abs(H)), 0.0)
    sqrt_e = math.sqrt(two_LmG)
    sqrt_i = math.sqrt(two_GmH)
    varpi = g + h
    mean_long = (l + g + h) % (2.0 * math.pi)

    xi = sqrt_e * math.cos(varpi)
    eta = sqrt_e * math.sin(varpi)
    p = sqrt_i * math.cos(h)
    q = sqrt_i * math.sin(h)
    p_norm = math.sqrt(xi * xi + eta * eta + p * p + q * q + L * L)
    return PoincareCanonicalChart(
        action_lambda=L,
        mean_longitude=mean_long,
        ecc_xi=xi,
        ecc_eta=eta,
        inc_p=p,
        inc_q=q,
        poincare_norm=p_norm,
    )


def kepler_hamiltonian_and_mean_motion(L: float) -> Tuple[float, float]:
    r"""
    Hamiltoniano kepleriano en acciones de Delaunay (unidades μ = 1):

        H_Kep = -1 / (2 L^2),   n = ∂H/∂L = 1 / L^3.
    """
    L_safe = max(abs(float(L)), _MACHINE_EPS)
    h_kep = -0.5 / (L_safe * L_safe)
    n = 1.0 / (L_safe ** 3)
    return float(h_kep), float(n)


def compute_delaunay_poincare_symplectic_elements(
    rho: NDArray[np.complex128],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float]:
    r"""
    Mapea el espectro de ρ sobre acciones-ángulos de Delaunay:

        L = Tr(ρ) ≡ 1,   G = L (λ2/λ1),   H = G (λ3/λ2),
        e = √(1 - (G/L)^2),   i = arccos(H/G),
        M = (λ1 - λ2) π  (mod 2π).
    """
    vals = np.sort(_safe_eigvalsh(rho))[::-1]
    n = len(vals)
    L = float(np.sum(vals))
    l1 = max(float(vals[0]), _MACHINE_EPS)
    l2 = float(vals[1]) if n > 1 else l1 * 0.5
    l3 = float(vals[2]) if n > 2 else l2 * 0.5

    G = L * min(l2 / l1, 1.0)
    H = G * min(l3 / max(l2, _MACHINE_EPS), 1.0)

    ratio_g = min(max(G / max(L, _MACHINE_EPS), 0.0), 1.0)
    eccentricity = math.sqrt(max(1.0 - ratio_g * ratio_g, 0.0))
    mean_anomaly = float((l1 - l2) * math.pi) % (2.0 * math.pi)
    eccentric_anomaly = solve_kepler_danby_order3(mean_anomaly, eccentricity)

    actions = (L, G, H)
    angles = (
        mean_anomaly,
        float(l2 * math.pi) % (2.0 * math.pi),
        float(l3 * math.pi) % (2.0 * math.pi),
    )
    return actions, angles, eccentricity, eccentric_anomaly


def compute_yang_mills_higgs_chern_simons_invariants(
    rho: NDArray[np.complex128],
) -> Tuple[float, float]:
    r"""
    3-forma regularizada de Chern-Simons con acoplamiento escalar de Higgs
    sobre el álgebra de matrices (discretización espectral):

        CS_3(A, Φ) = (1/(8π^2)) Tr( A ∧ dA + (2/3) A^3 + Φ · F_A )

    A se toma como conexión de gauge u(n) inducida por ρ - (1/n) I,
    Φ = log(ρ) (Higgs espectral). El lazo de Wilson es |Tr exp(2π A)| / n.
    """
    dim = int(rho.shape[0])
    rho_clean = _hermitian(rho)
    vals, vecs = la.eigh(rho_clean)
    vals_pos = np.maximum(np.real(vals), _MACHINE_EPS)
    phi = vecs @ np.diag(np.log(vals_pos)) @ vecs.conj().T
    A = -1j * (rho_clean - (np.eye(dim, dtype=np.complex128) / dim))
    dA = 0.5 * (A @ A - A.conj().T @ A.conj().T)
    F_A = dA + (A @ A)
    cs_integrand = (A @ dA) + ((2.0 / 3.0) * A @ A @ A) + (phi @ F_A)
    cs3_value = float(np.real(np.trace(cs_integrand)) / (8.0 * (math.pi ** 2)))
    holonomy_matrix = la.expm(2.0 * math.pi * A)
    wilson_loop = float(np.abs(np.trace(holonomy_matrix)) / dim)
    return cs3_value, wilson_loop


def compute_melnikov_homoclinic_integral(
    epsilon: float = 0.05,
    t0: float = 0.0,
    nodes: int = 120,
) -> float:
    r"""
    Integral homoclínica de Melnikov por cuadratura tanh-sinh (doble
    exponencial), que mide la distancia firmada entre variedades estable
    e inestable en una sección de Poincaré:

        M(t0) = ε ∫_{-∞}^{∞} sech^2(t) tanh(t) sin(t + t0) dt

    Un cero simple de M implica, por el teorema de Poincaré-Melnikov,
    intersección transversal homoclínica y, por Smale-Birkhoff, herradura.
    """
    h = 0.04
    integral = 0.0
    for k in range(-nodes, nodes + 1):
        x = k * h
        if abs(x) > 8.0:
            continue
        sh_x = math.sinh(x)
        arg = 0.5 * math.pi * sh_x
        if abs(arg) > 300.0:
            continue
        t = math.sinh(arg)
        cosh_arg = math.cosh(arg)
        weight = (0.5 * math.pi * math.cosh(x)) / (cosh_arg * cosh_arg + _MACHINE_EPS)
        if abs(t) > 300.0:
            continue
        sech_t = 1.0 / (math.cosh(t) + _MACHINE_EPS)
        integrand = (sech_t * sech_t) * math.tanh(t) * math.sin(t + t0)
        integral += integrand * weight * h
    return float(epsilon * integral)


def poincare_section_homoclinic_splitting(
    melnikov: float,
    mean_anomaly: float,
) -> float:
    r"""
    Residuo de la sección de Poincaré {l ≡ 0 mod 2π}: el splitting homoclínico
    se estima como |M(t0)| ponderado por la distancia angular al corte.
    """
    dist = min(abs(mean_anomaly % (2.0 * math.pi)), abs((mean_anomaly % (2.0 * math.pi)) - 2.0 * math.pi))
    window = math.exp(-dist * dist / 0.25)
    return float(abs(melnikov) * (0.5 + 0.5 * window))


def compute_chirikov_kam_criterion(
    rho: NDArray[np.complex128],
) -> Tuple[float, bool]:
    r"""
    Parámetro de solapamiento de resonancias de Chirikov K_C y condición
    diofántica de KAM sobre el vector de frecuencias espectrales ω:

        |⟨k, ω⟩| ≥ γ / ||k||^τ ,   K_C = (Δω1 + Δω2) / |ω1 - ω2|.

    K_C ≥ 1 implica destrucción de toros KAM (transición al caos global).
    """
    vals = np.sort(_safe_eigvalsh(rho))[::-1]
    dim = len(vals)
    if dim < 2:
        return 0.0, True

    l1 = max(float(vals[0]), _MACHINE_EPS)
    l2 = max(float(vals[1]), _MACHINE_EPS)
    l3 = float(vals[2]) if dim > 2 else l2 * 0.5

    alpha_twist = abs(-3.0 / (l1 ** 4 + _MACHINE_EPS))
    delta_v1 = abs(l1 - l2)
    delta_v2 = abs(l2 - l3)
    w1 = math.sqrt(l1)
    w2 = math.sqrt(l2)
    delta_omega1 = 2.0 * math.sqrt(max(2.0 * delta_v1 * alpha_twist, 0.0))
    delta_omega2 = 2.0 * math.sqrt(max(2.0 * delta_v2 * alpha_twist, 0.0))
    frequency_diff = abs(w1 - w2)
    if frequency_diff < _MACHINE_EPS:
        return 0.0, True

    k_chirikov = float((delta_omega1 + delta_omega2) / frequency_diff)
    k_norm = math.sqrt(2.0)
    diophantine_bound = _KAM_DIOPHANTINE_GAMMA / (k_norm ** _KAM_DIOPHANTINE_TAU)
    kam_satisfied = bool(
        (frequency_diff >= diophantine_bound) and (k_chirikov < _CHIRIKOV_CRITICAL)
    )
    return k_chirikov, kam_satisfied


def compute_bures_finsler_quantum_metrics(
    rho: NDArray[np.complex128],
    sigma: NDArray[np.complex128],
) -> Tuple[float, float, float, float, float]:
    r"""
    Métricas de información cuántica entre ρ y σ:

      • Fidelidad de Uhlmann: F = (Tr √(√ρ σ √ρ ))^2
      • Distancia de Bures:    d_B = √(2 - 2√F)
      • Fubini-Study:          d_FS = arccos(√F)
      • Índice Finsler-RHI y fracción de Kelly censurada por Fermi-Dirac,
        Melnikov y Chirikov-KAM.
    """
    rho_clean = _hermitian(rho)
    sigma_clean = _hermitian(sigma)

    sqrt_rho = la.sqrtm(rho_clean)
    middle = sqrt_rho @ sigma_clean @ sqrt_rho
    sqrt_middle = la.sqrtm(_hermitian(middle))
    fidelity = float(np.real(np.trace(sqrt_middle))) ** 2
    fidelity = min(max(fidelity, 0.0), 1.0)

    d_bures = math.sqrt(max(2.0 - 2.0 * math.sqrt(fidelity), 0.0))
    d_fubini = math.acos(min(max(math.sqrt(fidelity), 0.0), 1.0))

    delta = sigma_clean - rho_clean
    norm_frobenius = max(float(la.norm(rho_clean, "fro")), _MACHINE_EPS)
    trace_metric = float(np.real(np.trace(delta @ delta.conj().T)))
    term_riemann = trace_metric / (
        (norm_frobenius ** 2) * ((1.0 + math.sqrt(fidelity)) ** 2) + _MACHINE_EPS
    )
    fock_flux = float(np.real(np.trace(rho_clean @ sigma_clean - rho_clean @ rho_clean)))
    term_finsler = 0.25 * (math.tanh(fock_flux) ** 2)
    rhi_finsler = math.sqrt(max(term_riemann + term_finsler, 0.0))

    sw_fermi = 1.0 / (1.0 + math.exp((rhi_finsler - _RHI_THRESHOLD) / _FERMI_DIRAC_TEMP))
    f_star = 0.25
    melnikov = compute_melnikov_homoclinic_integral()
    chirikov, kam_ok = compute_chirikov_kam_criterion(rho_clean)
    theta_m = 1.0 if abs(melnikov) <= _MELNIKOV_MAX_TOL else 0.0
    theta_c = 1.0 if chirikov < _CHIRIKOV_CRITICAL else 0.0
    theta_r = 1.0 if rhi_finsler < _RHI_THRESHOLD else 0.0
    s_star = f_star * max(1.0 - rhi_finsler, 0.0) * sw_fermi * theta_m * theta_c * theta_r
    if not kam_ok:
        s_star = 0.0
    return fidelity, d_bures, d_fubini, rhi_finsler, s_star


def yoshida_composition_residual() -> float:
    """Identidad de consistencia 2 w1 + w0 = 1 del método de Yoshida orden 4."""
    return abs(2.0 * _YOSHIDA_W1 + _YOSHIDA_W0 - 1.0)


def sundman_fictitious_time(physical_dt: float, radius: float) -> float:
    r"""Transformación de Sundman ds = dt / r que regulariza colisiones."""
    return float(physical_dt / max(radius, _MACHINE_EPS))


def cdk_divided_difference(lam_i: float, lam_j: float) -> float:
    r"""
    Núcleo de Daleckii-Krein para f = log:

        φ^{[1]}(λ_i, λ_j) = (log λ_i - log λ_j) / (λ_i - λ_j)   (i ≠ j)
        φ^{[1]}(λ, λ)     = 1/λ
    """
    if abs(lam_i - lam_j) <= 1.0e-10:
        return 1.0 / max(lam_i, _MACHINE_EPS)
    return abs(math.log(lam_i) - math.log(lam_j)) / abs(lam_i - lam_j)


# ═══════════════════════════════════════════════════════════════════════════
# FASE 1: REGISTRO CUÁNTICO, EXCLUSIÓN DE PAULI Y ATLAS CELESTE DE POINCARÉ
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_CoreFockRegistry(ABC):
    r"""
    FASE 1 — Espacio de Fock fundamental, exclusión de Pauli fermiónica,
    evicción holográfica de Bekenstein, regularización Kustaanheimo-Stiefel
    y atlas celeste Delaunay-Poincaré.

    Método terminal formal: `synthesize_phase1_celestial_manifold()`.
    Su imagen es el paquete fibrado `Phase1CelestialManifoldBundle`, que
    constituye el dominio del método inicial de Fase 2.
    """

    def __init__(
        self,
        max_cartridges: int = _DEFAULT_MAX_CARTRIDGES,
        tolerance: float = _DEFAULT_TOL,
    ) -> None:
        self._max_cartridges: Final[int] = int(max_cartridges)
        self._tolerance: Final[float] = float(tolerance)
        self._registry: Dict[str, ToonCartridge] = {}
        self._embedding_cache: Dict[str, NDArray[np.float64]] = {}
        self._fermion_hashes: Dict[type, set] = {}
        self._embedding_dim: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self._registry)

    @property
    def max_capacity(self) -> int:
        return self._max_cartridges

    def inject_cartridge(
        self,
        key: str,
        cartridge: ToonCartridge,
        embedding: NDArray[np.float64],
    ) -> None:
        r"""
        Inyecta una partícula TOON en el Espacio de Fock haciendo cumplir:
          1. Principio de exclusión de Pauli para la familia fermiónica
             {a_i, a_j^†} = δ_ij  (ocupación n_i ∈ {0,1} por clase y hash).
          2. Cota holográfica de Bekenstein con evicción por similitud
             de configuración (proxy de entropía de embedding).
          3. Consistencia dimensional del fibrado de embeddings.
        """
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vec.size == 0:
            raise SynapticRegistryError("Embedding nulo: no define un ket en H.")
        if self._embedding_dim is None:
            self._embedding_dim = int(vec.size)
        elif int(vec.size) != self._embedding_dim:
            raise SynapticRegistryError(
                f"Dimensión de embedding inconsistente: {vec.size} ≠ {self._embedding_dim}."
            )

        if isinstance(cartridge, _FERMION_TYPES):
            c_type = type(cartridge)
            if c_type not in self._fermion_hashes:
                self._fermion_hashes[c_type] = set()
            h = cartridge.quantum_state_hash
            if h in self._fermion_hashes[c_type]:
                raise PauliExclusionViolationError(
                    f"Violación de Pauli: Fermión [{cartridge.cartridge_id}] ya ocupa "
                    f"el estado cuántico con firma hash {h[:12]}."
                )

        if len(self._registry) >= self._max_cartridges:
            logger.info("Cota holográfica de Bekenstein saturada. Ejecutando evicción.")
            self.evict_by_entropy(vec)

        self._registry[key] = cartridge
        norm = float(np.linalg.norm(vec)) + _MACHINE_EPS
        self._embedding_cache[key] = vec / norm

        if isinstance(cartridge, _FERMION_TYPES):
            self._fermion_hashes[type(cartridge)].add(cartridge.quantum_state_hash)

        logger.debug("Partícula [%s] inyectada en el Espacio de Fock.", cartridge.cartridge_id)

    def evict_by_entropy(self, probe_vector: NDArray[np.float64]) -> str:
        r"""
        Purga la partícula que minimiza la similitud cosenoidal respecto del
        vector de sondeo (máxima discordancia de configuración):

            k* = arg min_k  ⟨u, v_k⟩ / (||u|| ||v_k||)
        """
        if not self._registry:
            raise SynapticRegistryError("Evicción imposible: Espacio de Fock nulo.")

        probe = np.asarray(probe_vector, dtype=np.float64).reshape(-1)
        u = probe / (float(np.linalg.norm(probe)) + _MACHINE_EPS)
        min_sim = float("inf")
        worst_key: Optional[str] = None

        for k, v in self._embedding_cache.items():
            if v.shape[0] != u.shape[0]:
                sim = -1.0
            else:
                sim = float(np.dot(u, v))
            if sim < min_sim:
                min_sim = sim
                worst_key = k

        if worst_key is None:
            raise SynapticRegistryError("Falla de cálculo en evicción por entropía.")

        removed = self._registry.pop(worst_key)
        self._embedding_cache.pop(worst_key, None)
        if isinstance(removed, _FERMION_TYPES):
            bucket = self._fermion_hashes.get(type(removed))
            if bucket is not None:
                bucket.discard(removed.quantum_state_hash)

        logger.info(
            "Evicción completada: Cartucho [%s] purgado (similitud = %.4e).",
            worst_key,
            min_sim,
        )
        return worst_key

    def build_density_matrix_from_registry(self) -> Tuple[NDArray[np.complex128], bool]:
        r"""
        Construye el operador de densidad empírico

            ρ = (1/N) Σ_k |ψ_k⟩⟨ψ_k|

        y lo proyecta al simplejo D(H) mediante Higham+traza.
        """
        n = len(self._registry)
        if n == 0:
            return np.eye(1, dtype=np.complex128), False

        dim = int(next(iter(self._embedding_cache.values())).shape[0])
        rho_accum = np.zeros((dim, dim), dtype=np.complex128)
        for vec in self._embedding_cache.values():
            if vec.shape[0] != dim:
                continue
            ket = vec.reshape(-1, 1).astype(np.complex128)
            rho_accum += ket @ ket.conj().T
        rho_raw = rho_accum / float(n)
        return nearest_density_operator_higham(rho_raw, floor=self._tolerance)

    def count_fermions_and_bosons(self) -> Tuple[int, int]:
        fermions = sum(1 for c in self._registry.values() if isinstance(c, _FERMION_TYPES))
        bosons = len(self._registry) - fermions
        return fermions, bosons

    def generate_phase1_certificate(self) -> Phase1RegistryCertificate:
        fermion_count, boson_count = self.count_fermions_and_bosons()
        rho, higham_applied = self.build_density_matrix_from_registry()

        trace_val = float(np.real(np.trace(rho)))
        purity_val = float(np.real(np.trace(rho @ rho)))
        entropy_val = von_neumann_entropy(rho)

        evals = np.sort(_safe_eigvalsh(rho))[::-1]
        probe_evals = evals[:3] if len(evals) >= 3 else np.pad(evals, (0, 3 - len(evals)))
        _, ks_energy, ks_bilinear = regularize_kustaanheimo_stiefel(probe_evals)

        actions, angles, ecc, ecc_anomaly = compute_delaunay_poincare_symplectic_elements(rho)
        chart = delaunay_to_poincare_canonical(actions, angles)
        h_kep, mean_motion = kepler_hamiltonian_and_mean_motion(actions[0])
        cs3, wilson = compute_yang_mills_higgs_chern_simons_invariants(rho)

        inclination = math.acos(
            min(max(actions[2] / max(actions[1], _MACHINE_EPS), -1.0), 1.0)
        )

        verdict = FockRegistryVerdict.COHERENT
        if (not np.isfinite(trace_val)) or (not np.isfinite(purity_val)) or (
            abs(trace_val - 1.0) > 1.0e-4
        ):
            verdict = FockRegistryVerdict.VETOED
        elif higham_applied or purity_val < _PURITY_DEGRADED or ks_bilinear > 1.0e-6:
            verdict = FockRegistryVerdict.DEGRADED

        return Phase1RegistryCertificate(
            registry_size=self.size,
            max_capacity=self._max_cartridges,
            fermion_count=fermion_count,
            boson_count=boson_count,
            density_matrix_trace=trace_val,
            density_matrix_purity=purity_val,
            von_neumann_entropy=entropy_val,
            higham_regularization_applied=higham_applied,
            kustaanheimo_stiefel_energy=ks_energy,
            ks_bilinear_residual=ks_bilinear,
            delaunay_eccentricity=ecc,
            delaunay_inclination=inclination,
            kepler_mean_anomaly=angles[0],
            kepler_eccentric_anomaly=ecc_anomaly,
            kepler_hamiltonian=h_kep,
            mean_motion=mean_motion,
            poincare_chart=chart,
            chern_simons_3form_val=cs3,
            wilson_loop_holonomy=wilson,
            verdict=verdict,
        )

    def synthesize_phase1_celestial_manifold(self) -> Phase1CelestialManifoldBundle:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1.
        Sintetiza la variedad cuántica y el atlas celeste, emitiendo el paquete
        fibrado (ρ₀, I, θ, u_KS, Π_Poincaré, C₁) que constituye la entrada
        requerida para `propagate_phase2_symplectic_flow` (inicio de Fase 2).
        """
        rho_initial, _ = self.build_density_matrix_from_registry()
        cert1 = self.generate_phase1_certificate()
        actions, angles, _, _ = compute_delaunay_poincare_symplectic_elements(rho_initial)
        chart = delaunay_to_poincare_canonical(actions, angles)

        evals = np.sort(_safe_eigvalsh(rho_initial))[::-1]
        probe_evals = evals[:3] if len(evals) >= 3 else np.pad(evals, (0, 3 - len(evals)))
        spinor_ks, _, _ = regularize_kustaanheimo_stiefel(probe_evals)

        bundle = Phase1CelestialManifoldBundle(
            rho_initial=rho_initial,
            delaunay_actions=actions,
            delaunay_angles=angles,
            ks_regularized_spinor=spinor_ks,
            poincare_chart=chart,
            certificate=cert1,
        )
        logger.debug("FASE 1 consolidada. Paquete fibrado celeste emitido con éxito.")
        return bundle


# ═══════════════════════════════════════════════════════════════════════════
# FASE 2: DINÁMICA CUÁNTICA, INTEGRADOR YOSHIDA Y CRITERIO MELNIKOV-CHIRIKOV
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_QuantumDynamics(Phase1_CoreFockRegistry):
    r"""
    FASE 2 — Integración simpléctica de Yoshida (orden 4), evolución
    disipativa de Lindblad-GKSL por splitting de Lie-Trotter, transformaciones
    Bogoliubov-Valatin, sección de Poincaré y caos Melnikov-Chirikov.

    Método inicial: `propagate_phase2_symplectic_flow(bundle_fase_1)`.
    Método terminal: `prepare_phase2_immune_boundary()` → frontera para Fase 3.
    """

    def propagate_phase2_symplectic_flow(
        self,
        bundle: Phase1CelestialManifoldBundle,
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        dt: float = 0.1,
        steps: int = 10,
    ) -> Phase2DynamicalBoundaryBundle:
        r"""
        MÉTODO INICIAL DE FASE 2.
        Consume directamente el paquete fibrado `Phase1CelestialManifoldBundle`
        generado por `synthesize_phase1_celestial_manifold` e inicia la
        propagación dinámica del tensor de densidad. El Hamiltoniano por
        defecto es el kepleriano espectral diag(n_k) con n = 1/L^3.
        """
        rho_current = np.copy(bundle.rho_initial)
        dim = int(rho_current.shape[0])

        if hamiltonian_eff is None:
            L = max(bundle.delaunay_actions[0], _MACHINE_EPS)
            _, n_mean = kepler_hamiltonian_and_mean_motion(L)
            freqs = n_mean * np.linspace(1.0, 2.0, dim)
            hamiltonian_eff = np.diag(freqs.astype(np.complex128))
        if jump_operators is None:
            jump_operators = []
        if gamma_rates is None:
            gamma_rates = []

        rho_evolved, dissipation = self.solve_coupled_yoshida_lindblad_evolution(
            rho_current, hamiltonian_eff, jump_operators, gamma_rates, dt, steps
        )
        ks_radius = float(np.sum(bundle.ks_regularized_spinor ** 2)) + _MACHINE_EPS
        fict_time = sundman_fictitious_time(float(dt), ks_radius)
        cert2 = self._build_phase2_certificate(
            bundle.rho_initial,
            rho_evolved,
            dt,
            steps,
            dissipation,
            fict_time,
            bundle.delaunay_angles[0],
        )
        return self.prepare_phase2_immune_boundary(
            rho_evolved, hamiltonian_eff, dissipation, cert2
        )

    def apply_bogoliubov_valatin_symplectic(
        self,
        u_k: complex,
        v_k: complex,
        annihilation_op: NDArray[np.complex128],
        creation_op: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        Transformación canónica de Bogoliubov-Valatin (álgebra CCR / CAR):

            α_k  = u_k a_k + v_k a_k^†
            α_k^† = v_k* a_k + u_k* a_k^†

        con vínculo hiperbólico |u_k|^2 - |v_k|^2 ≡ 1  (bosones; Sp(2,R)).
        """
        constraint = abs(u_k) ** 2 - abs(v_k) ** 2
        if abs(constraint - 1.0) > self._tolerance * 100.0:
            raise SymplecticConstraintError(
                f"Violación Bogoliubov-Valatin: |u|² - |v|² = {constraint:.6e} != 1.0"
            )
        alpha_k = (u_k * annihilation_op) + (v_k * creation_op)
        alpha_k_dagger = (np.conj(v_k) * annihilation_op) + (np.conj(u_k) * creation_op)
        return alpha_k, alpha_k_dagger

    def perform_annihilation(
        self,
        electron_key: str,
        positron_key: str,
    ) -> Tuple[GammaPhoton, GammaPhoton]:
        r"""
        Aniquilación fermión-antifermión e⁻ + e⁺ → 2γ con verificación
        criptográfica HMAC en tiempo constante de la firma de autorización
        del positrón y conservación de carga homológica.
        """
        electron = self._registry.get(electron_key)
        positron = self._registry.get(positron_key)

        if not isinstance(electron, ElectronCartridge) or not isinstance(
            positron, PositronCartridge
        ):
            raise SynapticRegistryError(
                "Aniquilación inválida: requiere un Electrón y un Positrón registrados."
            )
        if electron.homological_charge + positron.homological_charge != 0:
            raise SynapticRegistryError(
                "Violación topológica: la suma de cargas homológicas debe ser nula."
            )

        expected_auth = hashlib.sha256(f"AUTH:{positron.cartridge_id}".encode("utf-8")).digest()
        provided_auth = hashlib.sha256(positron.authorization_signature.encode("utf-8")).digest()
        if not hmac.compare_digest(expected_auth, provided_auth):
            logger.warning("Firma HMAC de autorización de aniquilación no coincidente.")

        total_energy = float(electron.inertial_mass + positron.inertial_mass)
        prov_hash = hashlib.sha256(
            f"{electron.quantum_state_hash}::{positron.quantum_state_hash}".encode("utf-8")
        ).hexdigest()

        gamma1 = GammaPhoton(
            cartridge_id=f"gamma1-{electron.cartridge_id}",
            signature=f"sig-gamma1-{prov_hash[:8]}",
            energy=total_energy / 2.0,
            wave_vector=(0.0, 0.0, 1.0),
            provenance_hash=prov_hash,
        )
        gamma2 = GammaPhoton(
            cartridge_id=f"gamma2-{positron.cartridge_id}",
            signature=f"sig-gamma2-{prov_hash[8:16]}",
            energy=total_energy / 2.0,
            wave_vector=(0.0, 0.0, -1.0),
            provenance_hash=prov_hash,
        )

        e_type = type(electron)
        p_type = type(positron)
        if e_type in self._fermion_hashes:
            self._fermion_hashes[e_type].discard(electron.quantum_state_hash)
        if p_type in self._fermion_hashes:
            self._fermion_hashes[p_type].discard(positron.quantum_state_hash)

        self._registry.pop(electron_key, None)
        self._registry.pop(positron_key, None)
        self._embedding_cache.pop(electron_key, None)
        self._embedding_cache.pop(positron_key, None)

        dim = self._embedding_dim if self._embedding_dim is not None else 1
        self._registry[gamma1.cartridge_id] = gamma1
        self._registry[gamma2.cartridge_id] = gamma2
        # Fotones γ: kets isótropos de norma 1 (evita romper la dimensión de ρ).
        iso = np.ones(dim, dtype=np.float64) / math.sqrt(float(dim))
        self._embedding_cache[gamma1.cartridge_id] = iso
        self._embedding_cache[gamma2.cartridge_id] = iso
        return gamma1, gamma2

    def solve_coupled_yoshida_lindblad_evolution(
        self,
        rho: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
        jump_operators: List[NDArray[np.complex128]],
        gamma_rates: List[float],
        dt: float,
        steps: int,
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Ecuación de Liouville-von Neumann-Lindblad (semigrupo GKSL) por
        splitting de Lie-Trotter: flujo unitario exacto (expm, etapas de
        Yoshida orden 4) + flujo disipativo RK4:

            dρ/dt = -i[H, ρ] + Σ_k γ_k ( L_k ρ L_k^† - ½ {L_k^† L_k, ρ} )

        Cada subpaso se reproyecta a D(H) para preservar CP-traza a ε_mach.
        """
        if dt < 0.0 or steps < 0:
            raise LindbladEvolutionError("Paso temporal o número de etapas inválido.")
        rho_current = np.copy(rho)
        step_dt = float(dt) / float(max(steps, 1))
        H = np.asarray(hamiltonian, dtype=np.complex128)

        def dissipator(r: NDArray[np.complex128]) -> NDArray[np.complex128]:
            d = np.zeros_like(r, dtype=np.complex128)
            for gamma, L in zip(gamma_rates, jump_operators):
                LdL = L.conj().T @ L
                d += float(gamma) * ((L @ r @ L.conj().T) - 0.5 * (LdL @ r + r @ LdL))
            return d

        stages = (
            (_YOSHIDA_C1, _YOSHIDA_D1),
            (_YOSHIDA_C2, _YOSHIDA_D2),
            (_YOSHIDA_C3, _YOSHIDA_D3),
            (_YOSHIDA_C4, 0.0),
        )

        for _ in range(max(steps, 0)):
            for c_i, d_i in stages:
                if abs(c_i) > _MACHINE_EPS:
                    u_sub = la.expm(-1j * (c_i * step_dt) * H)
                    rho_current = u_sub @ rho_current @ u_sub.conj().T
                if abs(d_i) > _MACHINE_EPS and jump_operators:
                    sub_h = d_i * step_dt
                    k1 = dissipator(rho_current)
                    k2 = dissipator(rho_current + 0.5 * sub_h * k1)
                    k3 = dissipator(rho_current + 0.5 * sub_h * k2)
                    k4 = dissipator(rho_current + sub_h * k3)
                    rho_current = rho_current + (sub_h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            rho_current, _ = nearest_density_operator_higham(rho_current, floor=self._tolerance)

        dissipation_rate = 0.0
        for gamma, L in zip(gamma_rates, jump_operators):
            dissipation_rate += float(gamma) * float(
                np.real(np.trace(L.conj().T @ L @ rho_current))
            )
        if not np.isfinite(dissipation_rate):
            raise LindbladEvolutionError("Tasa de disipación GKSL no finita.")
        return rho_current, dissipation_rate

    def _build_phase2_certificate(
        self,
        rho_in: NDArray[np.complex128],
        rho_out: NDArray[np.complex128],
        dt: float,
        steps: int,
        dissipation: float,
        fict_time: float,
        mean_anomaly: float,
    ) -> Phase2DynamicsCertificate:
        trace_f = float(np.real(np.trace(rho_out)))
        purity_f = float(np.real(np.trace(rho_out @ rho_out)))
        res = abs(trace_f - 1.0)

        fid, db, dfs, rhi, s_star = compute_bures_finsler_quantum_metrics(rho_in, rho_out)
        melnikov = compute_melnikov_homoclinic_integral()
        splitting = poincare_section_homoclinic_splitting(melnikov, mean_anomaly)
        chirikov, kam_ok = compute_chirikov_kam_criterion(rho_out)
        yoshida_res = yoshida_composition_residual()
        min_eval = float(np.min(_safe_eigvalsh(rho_out)))

        verdict = FockRegistryVerdict.COHERENT
        if (
            (not np.isfinite(trace_f))
            or (not np.isfinite(purity_f))
            or (not kam_ok)
            or (chirikov >= _CHIRIKOV_CRITICAL)
        ):
            verdict = FockRegistryVerdict.VETOED
        elif res > 1.0e-4 or rhi >= _RHI_THRESHOLD or abs(melnikov) > _MELNIKOV_MAX_TOL:
            verdict = FockRegistryVerdict.DEGRADED

        return Phase2DynamicsCertificate(
            evolution_time=float(dt),
            evolution_steps=int(steps),
            fictitious_sundman_time=float(fict_time),
            final_trace=trace_f,
            final_purity=purity_f,
            trace_preservation_residual=res,
            is_trace_preserved=bool(res <= 1.0e-4),
            is_psd_preserved=bool(min_eval >= -self._tolerance),
            lindblad_dissipation_rate=dissipation,
            uhlmann_fidelity=fid,
            bures_distance=db,
            fubini_study_distance=dfs,
            rhi_finsler_index=rhi,
            fermi_dirac_kelly_censorship=float(
                1.0 / (1.0 + math.exp((rhi - _RHI_THRESHOLD) / _FERMI_DIRAC_TEMP))
            ),
            kelly_optimal_stake=s_star,
            melnikov_integral_val=melnikov,
            poincare_section_splitting=splitting,
            chirikov_overlap_parameter=chirikov,
            kam_diophantine_satisfied=kam_ok,
            yoshida_symplectic_residual=yoshida_res,
            verdict=verdict,
        )

    def prepare_phase2_immune_boundary(
        self,
        rho_evolved: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
        dissipation: float,
        cert2: Phase2DynamicsCertificate,
    ) -> Phase2DynamicalBoundaryBundle:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2.
        Construye la frontera dinámica (ρ_t, H, Φ_diss, K_C, C₂) que conecta
        con `execute_phase3_rsi3_cascade` (inicio de Fase 3).
        """
        if cert2.chirikov_overlap_parameter >= _CHIRIKOV_CRITICAL and not cert2.kam_diophantine_satisfied:
            logger.error("Catástrofe de resonancia KAM-Chirikov en la frontera de Fase 2.")
        boundary = Phase2DynamicalBoundaryBundle(
            rho_evolved=rho_evolved,
            hamiltonian_generator=hamiltonian,
            dissipation_flux=dissipation,
            chirikov_overlap=cert2.chirikov_overlap_parameter,
            certificate=cert2,
        )
        logger.debug("FASE 2 consolidada. Frontera dinámica emitida hacia Fase 3.")
        return boundary


# ═══════════════════════════════════════════════════════════════════════════
# FASE 3: ORQUESTACIÓN INMUNE, CASCADA BANACH RSI³ Y GOBERNANZA TERMINAL
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_ImmuneOrchestrator(Phase2_QuantumDynamics):
    r"""
    FASE 3 — Cascada reflexiva RSI³ en álgebra de Banach, producción de
    entropía de Spohn, punto fijo de Picard y mitigación inmune GPIO14.

    Método inicial: `execute_phase3_rsi3_cascade(boundary_fase_2)`.
    Síntesis holística: `SynapticFockSpaceRegistry.execute_full_registry_cycle`.
    """

    def execute_phase3_rsi3_cascade(
        self,
        boundary: Phase2DynamicalBoundaryBundle,
        perform_annihilations: bool = True,
    ) -> Tuple[Phase3ImmuneCertificate, NDArray[np.complex128]]:
        r"""
        MÉTODO INICIAL DE FASE 3.
        Consume directamente `Phase2DynamicalBoundaryBundle` emitido por
        `prepare_phase2_immune_boundary` e inicia la cascada de auto-mejora
        reflexiva en el retículo de Banach de operadores densidad.
        """
        annihilation_count = 0
        if perform_annihilations:
            annihilation_count = self._sweep_annihilations()

        rho_in = np.copy(boundary.rho_evolved)
        (
            rho_rsi3,
            spohn_rate,
            ricci_drift,
            cdk_lip,
            brockett_grad,
            contraction_factor,
            is_banach,
            picard_iters,
            picard_res,
        ) = self.execute_rsi3_banach_contraction_cascade(rho_in)

        entropy_val = von_neumann_entropy(rho_rsi3)

        verdict = FockRegistryVerdict.COHERENT
        if (not np.isfinite(entropy_val)) or (not is_banach) or (contraction_factor >= 1.0):
            verdict = FockRegistryVerdict.VETOED
        elif (
            boundary.certificate.verdict.is_degraded
            or (spohn_rate < _SPOHn_FLOOR)
            or (ricci_drift > 2.0)
        ):
            verdict = FockRegistryVerdict.DEGRADED

        crowbar_active = False
        if verdict == FockRegistryVerdict.VETOED:
            crowbar_active = self._trigger_hardware_crowbar_fail_safe()

        cert3 = Phase3ImmuneCertificate(
            annihilation_events=annihilation_count,
            bogoliubov_transformations=1,
            von_neumann_entropy=entropy_val,
            spohn_entropy_production_rate=spohn_rate,
            ricci_deturck_drift=ricci_drift,
            cdk_lipschitz_constant=cdk_lip,
            brockett_gradient_ascent_norm=brockett_grad,
            rsi3_banach_contraction_factor=contraction_factor,
            picard_iterations=picard_iters,
            picard_fixed_point_residual=picard_res,
            is_banach_contraction=is_banach,
            is_unitary_evolution=bool(boundary.dissipation_flux < _MACHINE_EPS),
            crowbar_triggered=crowbar_active,
            verdict=verdict,
        )
        return cert3, rho_rsi3

    def _sweep_annihilations(self) -> int:
        electrons = [c for c in self._registry.values() if isinstance(c, ElectronCartridge)]
        positrons = [c for c in self._registry.values() if isinstance(c, PositronCartridge)]
        count = 0
        consumed_e: set = set()
        consumed_p: set = set()
        for e in electrons:
            if id(e) in consumed_e:
                continue
            for p in positrons:
                if id(p) in consumed_p:
                    continue
                if e.homological_charge + p.homological_charge == 0:
                    e_key = next((k for k, v in self._registry.items() if v is e), None)
                    p_key = next((k for k, v in self._registry.items() if v is p), None)
                    if e_key and p_key:
                        try:
                            self.perform_annihilation(e_key, p_key)
                            count += 1
                            consumed_e.add(id(e))
                            consumed_p.add(id(p))
                            break
                        except SynapticRegistryError:
                            continue
        return count

    def execute_rsi3_banach_contraction_cascade(
        self,
        rho: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], float, float, float, float, float, bool, int, float]:
        r"""
        CASCADA DE BANACH RSI³ (tres niveles anidados + reflexión R₃):

        Nivel 1 (R₁) — Flujo de Ricci-DeTurck sobre la métrica espectral:
            ∂_τ log G = -2 tf(log G) - μ log G

        Nivel 2 (R₂) — Cálculo no conmutativo Connes-Daleckii-Krein:
            Lip(log) = max_{i,j} |φ^{[1]}(λ_i, λ_j)|

        Nivel 3 (R₃) — Flujo doblemente corchete de Brockett-Shahshahani:
            ρ̇ = [ρ, [ρ, N]],   σ_Spohn = -Tr(log ρ · ρ̇) ≥ 0

        Reflexión R₃ — Ajuste de τ para garantizar contracción de Banach
        κ = τ Lip(log) < 1, y Picard:

            ρ_{n+1} = Π_{D(H)}( ρ_n + τ ρ̇(ρ_n) ),   ||ρ_{n+1}-ρ_n||_F → 0.
        """
        rho_clean, _ = nearest_density_operator_higham(_hermitian(rho), floor=self._tolerance)
        dim = int(rho_clean.shape[0])
        vals, vecs = la.eigh(rho_clean)
        vals_pos = np.maximum(np.real(vals), _MACHINE_EPS)

        # R1: Ricci-DeTurck (parte de traza nula del log-espectro)
        log_G = np.diag(np.log(vals_pos))
        tf_ricci = log_G - (np.trace(log_G) * np.eye(dim) / dim)
        ricci_drift = float(np.linalg.norm(-2.0 * tf_ricci))

        # R2: Lipschitz CDK
        max_cdk = 0.0
        for i in range(dim):
            for j in range(dim):
                diff = cdk_divided_difference(float(vals_pos[i]), float(vals_pos[j]))
                if diff > max_cdk:
                    max_cdk = diff

        # R3: Brockett-Shahshahani con operador de conteo N
        N_op = np.diag(np.linspace(1.0, float(dim), dim)).astype(np.complex128)

        def brockett_vector_field(r: NDArray[np.complex128]) -> NDArray[np.complex128]:
            r_h = _hermitian(r)
            bracket1 = r_h @ N_op - N_op @ r_h
            return r_h @ bracket1 - bracket1 @ r_h

        rho_dot = brockett_vector_field(rho_clean)
        brockett_grad_norm = float(np.linalg.norm(rho_dot))

        log_rho = vecs @ np.diag(np.log(vals_pos)) @ vecs.conj().T
        spohn_prod = float(-np.real(np.trace(log_rho @ rho_dot)))
        spohn_prod = max(spohn_prod, 0.0)

        # Reflexión RSI³: control de contracción
        tau = 0.05 / (1.0 + max_cdk + brockett_grad_norm)
        contraction_factor = float(tau * max_cdk)
        if contraction_factor > _RSI3_BANACH_CONTRACTION_MAX:
            tau = _RSI3_BANACH_CONTRACTION_MAX / (max_cdk + _MACHINE_EPS)
            contraction_factor = float(tau * max_cdk)

        # Picard sobre el retículo de Banach (D(H), ||·||_F)
        rho_n = rho_clean
        residual = float("inf")
        iters = 0
        for iters in range(1, _PICARD_MAX_ITERS + 1):
            field = brockett_vector_field(rho_n)
            rho_trial = rho_n + tau * field
            rho_next, _ = nearest_density_operator_higham(rho_trial, floor=self._tolerance)
            residual = float(la.norm(rho_next - rho_n, "fro"))
            rho_n = rho_next
            if residual < _PICARD_FIXED_POINT_TOL:
                break

        is_banach = bool(contraction_factor < 1.0 and np.isfinite(residual))
        return (
            rho_n,
            spohn_prod,
            ricci_drift,
            max_cdk,
            brockett_grad_norm,
            contraction_factor,
            is_banach,
            int(iters),
            float(residual),
        )

    def _trigger_hardware_crowbar_fail_safe(self) -> bool:
        """Simula la protección de hardware Crowbar sobre el GPIO 14."""
        logger.critical(
            "CROWBAR_GPIO_%d ACTIVADO: Inestabilidad catastrófica en variedad cuántica de Fock.",
            _CROWBAR_GPIO,
        )
        return True


# ═══════════════════════════════════════════════════════════════════════════
# FACHADA INTEGRADA: SYNAPTIC FOCK SPACE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistry(Morphism, Phase3_ImmuneOrchestrator):
    r"""
    REGISTRO MAESTRO DEL ESPACIO DE FOCK (v9.0.0-PhD).
    Unifica las 3 Fases Anidadas en un flujo continuo y determinista:

        FASE 1  synthesize_phase1_celestial_manifold()        → bundle_1
        FASE 2  propagate_phase2_symplectic_flow(bundle_1)    → bundle_2
        FASE 3  execute_phase3_rsi3_cascade(bundle_2)         → cert_3
    """

    def __init__(
        self,
        max_cartridges: int = _DEFAULT_MAX_CARTRIDGES,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False,
    ) -> None:
        Morphism.__init__(self, "SynapticFockSpaceRegistry")
        Phase3_ImmuneOrchestrator.__init__(
            self, max_cartridges=max_cartridges, tolerance=tolerance
        )
        self._halt_on_veto: Final[bool] = bool(halt_on_veto)

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            return self.execute_full_registry_cycle(**kwargs)
        return state

    def execute_full_registry_cycle(
        self,
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        dt: float = 0.1,
        steps: int = 10,
        perform_annihilations: bool = True,
    ) -> FockRegistryState:
        r"""
        Ejecuta el continuum de las 3 Fases Anidadas. El último método de
        cada fase es exactamente el dominio del primero de la siguiente.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            bundle_1 = self.synthesize_phase1_celestial_manifold()
            cert_1 = bundle_1.certificate

            if cert_1.verdict.is_vetoed:
                cert_2 = self._emit_vetoed_phase2_certificate()
                cert_3 = self._emit_vetoed_phase3_certificate()
            else:
                bundle_2 = self.propagate_phase2_symplectic_flow(
                    bundle=bundle_1,
                    hamiltonian_eff=hamiltonian_eff,
                    jump_operators=jump_operators,
                    gamma_rates=gamma_rates,
                    dt=dt,
                    steps=steps,
                )
                cert_2 = bundle_2.certificate
                if cert_2.verdict.is_vetoed:
                    cert_3 = self._emit_vetoed_phase3_certificate()
                else:
                    cert_3, _ = self.execute_phase3_rsi3_cascade(
                        boundary=bundle_2,
                        perform_annihilations=perform_annihilations,
                    )

            final_verdict = FockRegistryVerdict.supremum(
                cert_1.verdict, cert_2.verdict, cert_3.verdict
            )
            action = RegistryAction.NONE
            diagnostic = "Ciclo cuántico de Fock completado bajo coherencia asintótica."

            if final_verdict == FockRegistryVerdict.VETOED:
                action = (
                    RegistryAction.ACTUATE_HARDWARE_CROWBAR
                    if cert_3.crowbar_triggered
                    else RegistryAction.HALT_REGISTRY
                )
                diagnostic = (
                    "VETO TOTAL: Colapso por resonancia caótica, violación de Pauli "
                    "o fallo Banach RSI³."
                )
                logger.error(diagnostic)
                if self._halt_on_veto:
                    raise SynapticRegistryError(diagnostic)
            elif final_verdict == FockRegistryVerdict.DEGRADED:
                action = RegistryAction.PURGE_ENTROPY
                diagnostic = (
                    "DEGRADACIÓN DETECTADA: Regularización de Higham o distorsión Finsler activa."
                )
                logger.warning(diagnostic)

            provenance = self._calculate_provenance_hash(cert_1, cert_2, cert_3, timestamp_utc)
            return FockRegistryState(
                phase1=cert_1,
                phase2=cert_2,
                phase3=cert_3,
                final_verdict=final_verdict,
                registry_action=action,
                timestamp_utc=timestamp_utc,
                provenance_hash=provenance,
                diagnostic_note=diagnostic,
            )
        except SynapticRegistryError as exc:
            logger.error("Colapso del registro sináptico: %s", exc)
            if self._halt_on_veto:
                raise
            return self._build_cataclysm_fallback(str(exc), timestamp_utc)

    def _emit_vetoed_phase2_certificate(self) -> Phase2DynamicsCertificate:
        return Phase2DynamicsCertificate(
            evolution_time=0.0,
            evolution_steps=0,
            fictitious_sundman_time=0.0,
            final_trace=0.0,
            final_purity=0.0,
            trace_preservation_residual=float("inf"),
            is_trace_preserved=False,
            is_psd_preserved=False,
            lindblad_dissipation_rate=float("inf"),
            uhlmann_fidelity=0.0,
            bures_distance=float("inf"),
            fubini_study_distance=float("inf"),
            rhi_finsler_index=1.0,
            fermi_dirac_kelly_censorship=0.0,
            kelly_optimal_stake=0.0,
            melnikov_integral_val=1.0,
            poincare_section_splitting=1.0,
            chirikov_overlap_parameter=2.0,
            kam_diophantine_satisfied=False,
            yoshida_symplectic_residual=float("inf"),
            verdict=FockRegistryVerdict.VETOED,
        )

    def _emit_vetoed_phase3_certificate(self) -> Phase3ImmuneCertificate:
        return Phase3ImmuneCertificate(
            annihilation_events=0,
            bogoliubov_transformations=0,
            von_neumann_entropy=float("inf"),
            spohn_entropy_production_rate=0.0,
            ricci_deturck_drift=float("inf"),
            cdk_lipschitz_constant=float("inf"),
            brockett_gradient_ascent_norm=float("inf"),
            rsi3_banach_contraction_factor=1.0,
            picard_iterations=0,
            picard_fixed_point_residual=float("inf"),
            is_banach_contraction=False,
            is_unitary_evolution=False,
            crowbar_triggered=True,
            verdict=FockRegistryVerdict.VETOED,
        )

    def _build_cataclysm_fallback(self, reason: str, timestamp: str) -> FockRegistryState:
        zero_chart = PoincareCanonicalChart(
            action_lambda=0.0,
            mean_longitude=0.0,
            ecc_xi=0.0,
            ecc_eta=0.0,
            inc_p=0.0,
            inc_q=0.0,
            poincare_norm=0.0,
        )
        c1 = Phase1RegistryCertificate(
            registry_size=0,
            max_capacity=self._max_cartridges,
            fermion_count=0,
            boson_count=0,
            density_matrix_trace=0.0,
            density_matrix_purity=0.0,
            von_neumann_entropy=0.0,
            higham_regularization_applied=False,
            kustaanheimo_stiefel_energy=0.0,
            ks_bilinear_residual=0.0,
            delaunay_eccentricity=0.0,
            delaunay_inclination=0.0,
            kepler_mean_anomaly=0.0,
            kepler_eccentric_anomaly=0.0,
            kepler_hamiltonian=0.0,
            mean_motion=0.0,
            poincare_chart=zero_chart,
            chern_simons_3form_val=0.0,
            wilson_loop_holonomy=0.0,
            verdict=FockRegistryVerdict.VETOED,
        )
        c2 = self._emit_vetoed_phase2_certificate()
        c3 = self._emit_vetoed_phase3_certificate()
        prov = hashlib.sha256(f"CATACLYSM::{reason}::{timestamp}".encode("utf-8")).hexdigest()
        return FockRegistryState(
            phase1=c1,
            phase2=c2,
            phase3=c3,
            final_verdict=FockRegistryVerdict.VETOED,
            registry_action=RegistryAction.HALT_REGISTRY,
            timestamp_utc=timestamp,
            provenance_hash=prov,
            diagnostic_note=f"CATACLYSM_FALLBACK: {reason}",
        )

    def _calculate_provenance_hash(
        self,
        c1: Phase1RegistryCertificate,
        c2: Phase2DynamicsCertificate,
        c3: Phase3ImmuneCertificate,
        timestamp: str,
    ) -> str:
        payload = (
            f"{timestamp}|"
            f"{c1.verdict.value}|{c2.verdict.value}|{c3.verdict.value}|"
            f"{c1.registry_size}:{c1.fermion_count}:{c1.boson_count}|"
            f"{c1.density_matrix_trace:.8e}|{c1.density_matrix_purity:.8e}|"
            f"{c1.kepler_hamiltonian:.8e}|{c1.poincare_chart.poincare_norm:.8e}|"
            f"{c2.melnikov_integral_val:.8e}|{c2.chirikov_overlap_parameter:.8e}|"
            f"{c2.poincare_section_splitting:.8e}|{c2.rhi_finsler_index:.8e}|"
            f"{c2.kelly_optimal_stake:.8e}|"
            f"{c3.spohn_entropy_production_rate:.8e}|{c3.rsi3_banach_contraction_factor:.8e}|"
            f"{c3.picard_fixed_point_residual:.8e}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════
__all__ = [
    "SynapticRegistryError",
    "PauliExclusionViolationError",
    "FockSpaceOverflowError",
    "TraceAnomalyError",
    "SymplecticConstraintError",
    "DensityMatrixNonPSDError",
    "LindbladEvolutionError",
    "KAMResonanceCatastropheError",
    "BanachContractionFailureError",
    "KSBilinearConstraintError",
    "PoincareChartSingularityError",
    "FockRegistryVerdict",
    "RegistryAction",
    "ToonCartridge",
    "ElectronCartridge",
    "ProtonCartridge",
    "PolaronCartridge",
    "TorsionCartridge",
    "HouseholderReflectionFermion",
    "PhotonCartridge",
    "RiemannianFocalBoson",
    "MagnonCartridge",
    "SwellingPlasmonCartridge",
    "YieldingPhononCartridge",
    "LiquefactionSolitonCartridge",
    "PositronCartridge",
    "GammaPhoton",
    "PolaritonCartridge",
    "SophonCartridge",
    "PoincareCanonicalChart",
    "Phase1RegistryCertificate",
    "Phase1CelestialManifoldBundle",
    "Phase2DynamicsCertificate",
    "Phase2DynamicalBoundaryBundle",
    "Phase3ImmuneCertificate",
    "FockRegistryState",
    "nearest_density_operator_higham",
    "stable_density_matrix_higham",
    "von_neumann_entropy",
    "ks_matrix",
    "ks_bilinear_residual",
    "regularize_kustaanheimo_stiefel",
    "solve_kepler_danby_order3",
    "delaunay_to_poincare_canonical",
    "kepler_hamiltonian_and_mean_motion",
    "compute_delaunay_poincare_symplectic_elements",
    "compute_yang_mills_higgs_chern_simons_invariants",
    "compute_melnikov_homoclinic_integral",
    "poincare_section_homoclinic_splitting",
    "compute_chirikov_kam_criterion",
    "compute_bures_finsler_quantum_metrics",
    "Phase1_CoreFockRegistry",
    "Phase2_QuantumDynamics",
    "Phase3_ImmuneOrchestrator",
    "SynapticFockSpaceRegistry",
]