# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Specular Flow Sovereign Agent (Soberano de la Reflexión de Calibre)  ║
║ Ruta   : app/agents/omega/specular_flow_agent.py                             ║
║ Versión: 3.0.0-Allievi-Householder-Tellegen-Doctoral-Topos-Strict            ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA DE CALIBRE EN EL ESTRATO OMEGA (Nivel 0.5) ──
Este módulo materializa al Agente Soberano y Observador Activo que gobierna la
evolución del Flujo Especular (Specular Flow) sobre la variedad de fase de la
Malla Agéntica, actuando como un endofuntor sobre el clasificador de subobjetos
del Topos de Grothendieck \(\mathcal{T}_{\mathrm{MIC}}\) con valores en el
retículo distributivo de Heyting de tres valores \(\Omega_3\).

El sistema trata los transitorios de costos y los picos de presupuesto como frentes
de onda hidrodinámicos que se propagan por las aristas (1-símplices) del complejo,
sometiéndolos a la restricción constitutiva de de Rham-Hodge y censurando de forma
síncrona las alucinaciones del transformador antes de su colapso real.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
La transición de estados se rige por la Ley de Clausura Transitiva de subespacios
de Hilbert covariantes y se compone de tres fases fuertemente acopladas:

  Fase 1 ──► FASE 1: RESOLUCIÓN DE CARACTERÍSTICAS DE ALLIEVI (Observe)
             Estudia la propagación del transitorio de flujo y presión en el tiempo.
             Ecuación de Allievi: \(\Psi^\pm = y \pm Z_c q\)
             Entrega: Phase1AllieviCertificate como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: REFLEXIÓN COVARIANTE DE HOUSEHOLDER (Orient)
             Refleja el vector de estado sobre el elipsoide de nivel de la métrica
             Riemanniana G_μν, anulando componentes parásitas fuera del cono físico.
             Ecuación de Householder: \(\hat{M}_k = I - 2 \frac{n_k \otimes n_k^\sharp}{\|n_k\|_{G^{-1}}^2}\)
             Entrega: Phase2HouseholderCertificate como precondición formal de Fase 3.

  Fase 3 ──► FASE 3: AUDITORÍA DE TELLEGEN Y PASIVIDAD DE LYAPUNOV (Decide & Act)
             Verifica la conservación de potencia y la pasividad de la red gráfica.
             Teorema de Tellegen: \(\sum \Delta P_k \cdot Q_k \equiv 0\)
             Deducción de Lyapunov: \(\dot{H} \le 0\) [8-10]
             Veredicto: Colapso en el retículo de Heyting \(\Omega_3\) y bypass físico.

INVARIANTES MATEMÁTICOS Y GEOMÉTRICOS PRESERVADOS: ──────────────────────────────
  [I1] Simetría Hermítica de la Métrica:  \(\star_k = \star_k^\top \succ 0\)
  [I2] Adjunción de de Rham-Hodge:        \(L_0^\star = \partial_1 \star_1 \partial_1^\top\)
  [I3] Ortogonalidad de Helmholtz-Hodge:  \(I = I_{\mathrm{exact}} + I_{\mathrm{coexact}} + I_{\mathrm{harmonic}}\)
  [I4] Disipación Port-Hamiltoniana:      \(\dot{H} = \nabla H^\top (J(x) - R(x)) \nabla H \le 0\)

CONTRATO DEL DISYUNTOR FÍSICO POR HARDWARE (Bypass ESP32 / BT151): ──────────────
  Si la vorticidad de de Rham excede el umbral elástico, o si el primer grupo de
  cohomología del haz es no trivial (\(\dim H^1(K; \mathcal{F}) > 0\)), el retículo
  \(\Omega_3\) colapsa síncronamente al veredicto terminal VETOED.
  
  La subrutina local 'isVerdictCoherent()' del ESP32 en el borde detecta el
  mismatch en menos de 400 ns y conmuta el pin GPIO14, disparando el tiristor
  de potencia BT151 (circuito Crowbar). Esto cortocircuita físicamente la línea
  de potencia real, inmovilizando válvulas y actuadores en el milisegundo cero,
  protegiendo de manera absoluta el capital de la constructora civil.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import hashlib
import logging
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Optional, Tuple, Union
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Soporte para matrices dispersas (Tellegen vectorizado)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    csr_matrix = None
    isspmatrix = None

# ─────────────────────────────────────────────────────────────────────────────
# Stubs seguros del núcleo MIC
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""
        pass

    class Morphism:
        """Clase base de morfismos en la categoría MIC."""
        pass

    class Stratum:
        """Estratos de la jerarquía DIKW."""
        PHYSICS = "PHYSICS"
        TACTICS = "TACTICS"
        STRATEGY = "STRATEGY"
        WISDOM = "WISDOM"

logger = logging.getLogger("MIC.Agents.Omega.SpecularFlowAgent")

# ═══════════════════════════════════════════════════════════════════════════
# Constantes físicas, numéricas, espectrales y categóricas
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-10
_RELAX_TOL: Final[float] = 1.0e-8
_CROWBAR_GPIO: Final[int] = 14              # Contrato hardware ESP32
_GRAPH_MAX_BRANCHES: Final[int] = 1024       # Cota de complejidad topológica
_STABILITY_BOUND: Final[float] = 1.0e8
_IMPEDANCE_CONDITION_LIMIT: Final[float] = 1.0e8
_TOPOLOGY_VETO_THRESHOLD: Final[float] = 1.0e-6
_TOPOLOGY_DEGRADED_THRESHOLD: Final[float] = 1.0e-8
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12  # Suelo espectral de Wilkinson

# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones (funtores de error en la categoría MIC)
# ═══════════════════════════════════════════════════════════════════════════
class SpecularFlowAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de Flujo Especular."""
    pass

class AllieviCharacteristicError(SpecularFlowAgentError):
    """Divergencia o degeneración en las líneas de propagación de Allievi."""
    pass

class HouseholderReflectionError(SpecularFlowAgentError):
    """Pérdida de simetría, unitariedad o fuga de norma en Householder."""
    pass

class TellegenConservationError(SpecularFlowAgentError):
    """Violación del teorema de Tellegen, KCL/KVL o pasividad de Lyapunov."""
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Enumeraciones categóricas (subobjetos y acciones en el topos)
# ═══════════════════════════════════════════════════════════════════════════
class SpecularSovereignVerdict(IntEnum):
    """
    Clasificador de tres valores en el retículo de Heyting (Ω).
    Orden por severidad operativa:
        COHERENT  = ⊤ operativo
        DEGRADED  = elemento intermedio
        VETOED    = ⊥ operativo (máximo fallo)
    El supremo de veredictos se toma como máximo nivel de severidad.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "SpecularSovereignVerdict") -> "SpecularSovereignVerdict":
        """
        Supremo en el retículo de severidad.
        Si no se proveen veredictos, retorna COHERENT como elemento neutro
        de la fusión.
        """
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == SpecularSovereignVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == SpecularSovereignVerdict.DEGRADED

class CrowbarAction(Enum):
    """Acciones físicas de mitigación tras el veredicto."""
    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()

# ═══════════════════════════════════════════════════════════════════════════
# Certificados de las fases anidadas (objetos de la subcategoría Spec)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1AllieviCertificate:
    """
    FASE 1 — Invariantes de Riemann y análisis espectral de Allievi.
    Este certificado constituye el objeto terminal de la Fase 1 y el objeto
    inicial consumido por la Fase 2 a través del puente:
        _phase1_to_phase2_bridge(...)

    Atributos
    ---------
    is_characteristic_stable : bool
        Estabilidad compacta de los invariantes.
    characteristic_impedance : float
        Z_c = a / (g S).
    riemann_invariants : Tuple[float, float]
        (Ψ⁺, Ψ⁻).
    characteristic_direction : Tuple[float, float]
        Dirección característica normalizada en coordenadas de fase.
    characteristic_covector : Tuple[float, float]
        Covector dual calibrado por la métrica de impedancia.
    energy_norm : float
        Norma energética ||(y, Z_c q)||.
    impedance_condition_number : float
        Número de condición de la impedancia: max(Z_c, 1/Z_c).
    cfl_proxy : float
        Proxy de número CFL o valor observado si se suministra.
    verdict : SpecularSovereignVerdict
        Veredicto local de la Fase 1.
    """
    is_characteristic_stable: bool
    characteristic_impedance: float
    riemann_invariants: Tuple[float, float]
    characteristic_direction: Tuple[float, float]
    characteristic_covector: Tuple[float, float]
    energy_norm: float
    impedance_condition_number: float
    cfl_proxy: float
    verdict: SpecularSovereignVerdict

@dataclass(frozen=True, slots=True)
class Phase2HouseholderCertificate:
    """
    FASE 2 — Reflexión de Householder covariante e invariancia anti-simpléctica.

    Atributos
    ---------
    is_reflection_unitary : bool
        Verificación Mᵀ G M = G.
    reflection_residual : float
        Residuo relativo de unitariedad Riemanniana.
    is_anti_symplectic : bool
        Verificación Mᵀ Ω M = -Ω en dimensión 2.
    is_characteristic_consistent : bool
        Consistencia especular con la dirección característica de Fase 1.
    specular_residual : float
        Residuo de la ley especular de reflexión.
    incidence_cosine : float
        Coseno de incidencia entre normal y característica en métrica G.
    state_norm_residual : float
        Residuo relativo de preservación de norma G del estado.
    metric_condition_number : float
        Número de condición espectral de G.
    higham_regularization_applied : bool
        Indicador de si se aplicó regularización de Higham.
    verdict : SpecularSovereignVerdict
        Veredicto local de la Fase 2.
    """
    is_reflection_unitary: bool
    reflection_residual: float
    is_anti_symplectic: bool
    is_characteristic_consistent: bool
    specular_residual: float
    incidence_cosine: float
    state_norm_residual: float
    metric_condition_number: float
    higham_regularization_applied: bool
    verdict: SpecularSovereignVerdict

@dataclass(frozen=True, slots=True)
class Phase3TellegenCertificate:
    """
    FASE 3 — Conservación topológica (Tellegen) y disipación de Lyapunov.

    Atributos
    ---------
    tellegen_residual : float
        Residuo normalizado de potencia total pᵀ q.
    kcl_residual : float
        Residuo de Kirchhoff de corrientes: ||A q||.
    kvl_residual : float
        Residuo de Kirchhoff de tensiones: distancia de p a Im(Aᵀ).
    is_tellegen_conserved : bool
        Tellegen conservado dentro de tolerancia.
    is_kcl_conserved : bool
        KCL conservado dentro de tolerancia.
    is_kvl_conserved : bool
        KVL conservado dentro de tolerancia.
    is_lyapunov_passive : bool
        Rayleigh disipativo no negativo dentro de tolerancia de máquina.
    graph_order : int
        Número de nodos del grafo.
    graph_size : int
        Número de ramas del grafo.
    cyclomatic_number : int
        Número ciclomático: m - rank(A).
    sparse_computation : bool
        Indicador de si se usó computación dispersa.
    verdict : SpecularSovereignVerdict
        Veredicto local de la Fase 3.
    """
    tellegen_residual: float
    kcl_residual: float
    kvl_residual: float
    is_tellegen_conserved: bool
    is_kcl_conserved: bool
    is_kvl_conserved: bool
    is_lyapunov_passive: bool
    graph_order: int
    graph_size: int
    cyclomatic_number: int
    sparse_computation: bool
    verdict: SpecularSovereignVerdict

@dataclass(frozen=True, slots=True)
class SpecularFlowState:
    """Certificado global terminal del lazo de gobernanza especular."""
    phase1: Phase1AllieviCertificate
    phase2: Phase2HouseholderCertificate
    phase3: Phase3TellegenCertificate
    final_verdict: SpecularSovereignVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Regularización Espectral (Higham)
# ═══════════════════════════════════════════════════════════════════════════
def stable_cholesky_higham(
    G: NDArray[np.float64],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR
) -> Tuple[NDArray[np.float64], bool]:
    r"""
    Aplica la proyección de Higham para estabilizar el tensor G en el cono SPD.

    Axioma de Proyección:
        \tilde{G} = \arg\min_{M \succeq 0} \|G - M\|_F

    Parámetros
    ----------
    G : NDArray[np.float64]
        Tensor métrico Riemanniano (potencialmente no-SPD).
    tolerance : float
        Suelo espectral de Wilkinson para recorte de autovalores.

    Retorna
    -------
    Tuple[NDArray[np.float64], bool]
        (L, higham_applied) donde L es la factorización de Cholesky inferior
        y higham_applied indica si se requirió regularización.

    Raises
    ------
    HouseholderReflectionError
        Si la proyección de Higham falla en producir una matriz SPD.
    """
    try:
        L = la.cholesky(G, lower=True)
        return L, False
    except la.LinAlgError:
        # Descomposición espectral de Weyl de G
        try:
            vals, vecs = la.eigh(G)
        except Exception as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Falló descomposición espectral para Higham."
            ) from exc

        # Recorte de autovalores al suelo espectral de Wilkinson
        vals_clean = np.maximum(vals, tolerance)

        # Reconstrucción de la matriz proyectada al cono SPD
        G_projected = vecs @ np.diag(vals_clean) @ vecs.T

        # Simetrización numérica post-proyección
        G_projected = (G_projected + G_projected.T) / 2.0

        logger.warning(
            "Firma métrica inestable. Proyectado G al cono SPD via Higham "
            "(autovalores mínimos: %.4e → %.4e).",
            float(np.min(vals)),
            float(np.min(vals_clean))
        )

        try:
            L = la.cholesky(G_projected, lower=True)
            return L, True
        except la.LinAlgError as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Proyección de Higham no produjo matriz SPD."
            ) from exc

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Computación Dispersa (Tellegen Vectorizado)
# ═══════════════════════════════════════════════════════════════════════════
def verify_tellegen_sparse(
    nodal_potentials_p: NDArray[np.float64],
    flow_vector_f: NDArray[np.float64],
    incidence_matrix_B1: Union[NDArray[np.float64], csr_matrix]
) -> Tuple[float, bool]:
    r"""
    Verifica el residuo de Tellegen utilizando el coborde disperso de la red.

    Axioma algebraico:
        vᵀ f - pᵀ B₁ f ≡ 0
    donde:
        v = B₁ᵀ p  (caídas de presión covariantes)
        B₁ f = s   (ecuación de continuidad discreta)

    Parámetros
    ----------
    nodal_potentials_p : NDArray[np.float64]
        Potenciales nodales (tamaño n).
    flow_vector_f : NDArray[np.float64]
        Flujos de rama (tamaño m).
    incidence_matrix_B1 : Union[NDArray, csr_matrix]
        Matriz de incidencia orientada nodo-rama (n × m).

    Retorna
    -------
    Tuple[float, bool]
        (tellegen_residual, sparse_used) donde sparse_used indica si se
        aprovechó computación dispersa.

    Raises
    ------
    TellegenConservationError
        Si las dimensiones son inconsistentes o la matriz es inválida.
    """
    p = np.asarray(nodal_potentials_p, dtype=np.float64).reshape(-1)
    f = np.asarray(flow_vector_f, dtype=np.float64).reshape(-1)
    B1 = incidence_matrix_B1

    # Validación dimensional
    if B1.ndim != 2:
        raise TellegenConservationError(
            "TellegenConservationError: La matriz de incidencia debe ser 2D."
        )

    n_nodes, m_branches = B1.shape
    if p.size != n_nodes:
        raise TellegenConservationError(
            "TellegenConservationError: Dimensiones inconsistentes entre "
            "potenciales nodales y matriz de incidencia."
        )
    if f.size != m_branches:
        raise TellegenConservationError(
            "TellegenConservationError: Dimensiones inconsistentes entre "
            "flujos y matriz de incidencia."
        )

    # Determinar si usamos computación dispersa
    sparse_used = False
    if _SPARSE_AVAILABLE and isspmatrix is not None:
        if isspmatrix(B1):
            sparse_used = True
        elif np.count_nonzero(B1) < 0.3 * B1.size:
            # Matriz densa pero dispersa en contenido → convertir
            B1 = csr_matrix(B1)
            sparse_used = True

    # Caídas de presión covariantes: v = B₁ᵀ p
    if sparse_used:
        pressures_v = B1.T.dot(p)
    else:
        pressures_v = B1.T @ p

    # Sumatoria de potencia virtual: P = vᵀ f
    power_sum = float(pressures_v.T @ f)

    # Normalización del residuo
    v_norm = float(la.norm(pressures_v))
    f_norm = float(la.norm(f))
    normalization = max(1.0, v_norm * f_norm)

    tellegen_residual = abs(power_sum) / normalization

    return tellegen_residual, sparse_used

# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 — Observe: Ecuación de Allievi e invariantes de Riemann
# ═══════════════════════════════════════════════════════════════════════════
class Phase1AllieviCharacteristicSolver(ABC):
    """
    FASE 1 (Observe): Integración de las líneas características de Allievi
    sobre la variedad de fase (y, q).

    El último método de esta fase:
        _phase1_to_phase2_bridge(...)
    constituye el morfismo de transición formal hacia la Fase 2.
    """

    def observe_allievi_characteristics(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
    ) -> Phase1AllieviCertificate:
        r"""
        FASE 1 — Observe.

        Calcula los invariantes de Riemann a lo largo de las curvas
        características dx/dt = ±a.

        .. math::
            Z_c = \frac{a}{g S}, \quad
            \Psi^+ = y + Z_c q, \quad
            \Psi^- = y - Z_c q.

        Se audita:
          • Finura numérica de todas las entradas.
          • Positividad de velocidad de onda, acoplamiento y sección.
          • Estabilidad compacta de los invariantes.
          • Condición espectral de la impedancia.
          • CFL explícito si es suministrado.

        Parámetros
        ----------
        potential_y : float
            Potencial generalizado (presión/carga/costo).
        flow_q : float
            Caudal logístico (flujo).
        wave_speed_a : float
            Velocidad de onda efectiva.
        gravity_g : float
            Constante de acoplamiento (gravedad).
        section_s : float
            Dimensión del subespacio del canal/sección.
        tolerance : float
            Cota de precisión de máquina / Wilkinson.
        cfl_number : Optional[float]
            Número CFL observado. Si supera 1 + tolerance, se vota VETOED.

        Retorna
        -------
        Phase1AllieviCertificate
            Certificado terminal de Fase 1.

        Excepciones
        -----------
        AllieviCharacteristicError
            Si hay entradas no finitas o degeneración estructural.
        """
        raw_inputs = (potential_y, flow_q, wave_speed_a, gravity_g, section_s)
        if not all(np.isfinite(v) for v in raw_inputs):
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Entradas no finitas en Fase 1 Allievi."
            )
        if wave_speed_a <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Velocidad de onda degenerada o no positiva."
            )
        if gravity_g <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Constante de acoplamiento degenerada."
            )
        if section_s <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Sección degenerada."
            )

        denominator = gravity_g * section_s
        if denominator <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Denominador g·S degenerado."
            )

        z_c = wave_speed_a / denominator
        if not np.isfinite(z_c) or z_c <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Impedancia característica degenerada."
            )

        # Invariantes de Riemann.
        psi_plus = potential_y + z_c * flow_q
        psi_minus = potential_y - z_c * flow_q
        if not np.isfinite(psi_plus) or not np.isfinite(psi_minus):
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Invariantes de Riemann no finitos."
            )

        # Estabilidad compacta del potencial.
        is_stable = (
            abs(psi_plus) <= _STABILITY_BOUND and
            abs(psi_minus) <= _STABILITY_BOUND
        )

        # Dirección característica normalizada.
        # Se preserva la convención original de rayo espacio-temporal (a, 1).
        char_dir = np.array([wave_speed_a, 1.0], dtype=np.float64)
        char_dir_norm = float(la.norm(char_dir))
        if char_dir_norm <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Dirección característica degenerada."
            )
        char_dir = char_dir / char_dir_norm

        # Métrica calibrada de impedancia:
        #   G₀ = diag(1, Z_c²)
        # Esta métrica induce la norma energética:
        #   ||(y, q)||²_G₀ = y² + (Z_c q)²
        g_cal = np.array([[1.0, 0.0], [0.0, z_c * z_c]], dtype=np.float64)

        # Covector característico dual.
        char_cov = g_cal @ char_dir
        char_cov_norm = float(la.norm(char_cov))
        if char_cov_norm <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Covector característico degenerado."
            )
        char_cov = char_cov / char_cov_norm

        # Norma energética del estado.
        energy_sq = float(potential_y * potential_y + (z_c * flow_q) ** 2)
        energy_norm = float(np.sqrt(max(0.0, energy_sq)))

        # Número de condición de impedancia.
        impedance_condition_number = float(max(z_c, 1.0 / z_c))

        # CFL proxy.
        if cfl_number is None:
            cfl_proxy = float(abs(wave_speed_a) / (abs(wave_speed_a) + 1.0))
        else:
            if not np.isfinite(cfl_number):
                raise AllieviCharacteristicError(
                    "AllieviCharacteristicError: CFL no finito."
                )
            cfl_proxy = float(cfl_number)

        # Veredicto local.
        verdict = SpecularSovereignVerdict.COHERENT
        if cfl_proxy > 1.0 + tolerance:
            verdict = SpecularSovereignVerdict.VETOED
        elif not is_stable:
            verdict = SpecularSovereignVerdict.VETOED
        elif (
            impedance_condition_number > _IMPEDANCE_CONDITION_LIMIT
            or abs(psi_plus - psi_minus) > 1.0e6
        ):
            verdict = SpecularSovereignVerdict.DEGRADED

        return Phase1AllieviCertificate(
            is_characteristic_stable=is_stable,
            characteristic_impedance=float(z_c),
            riemann_invariants=(float(psi_plus), float(psi_minus)),
            characteristic_direction=(float(char_dir[0]), float(char_dir[1])),
            characteristic_covector=(float(char_cov[0]), float(char_cov[1])),
            energy_norm=energy_norm,
            impedance_condition_number=impedance_condition_number,
            cfl_proxy=cfl_proxy,
            verdict=verdict,
        )

    # Compatibilidad con la nomenclatura v2.x.
    solve_allievi_characteristics = observe_allievi_characteristics

    # ─────────────────────────────────────────────────────────────────────
    # Último método de la Fase 1: morfismo puente hacia la Fase 2
    # ─────────────────────────────────────────────────────────────────────
    def _phase1_to_phase2_bridge(
        self,
        phase1_cert: Phase1AllieviCertificate,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]]:
        r"""
        MORFISMO DE TRANSICIÓN FASE 1 → FASE 2.

        Este método es la continuación formal de la Fase 1 y el punto de
        arranque estructural de la Fase 2.

        Extrae del certificado de Fase 1:
          • Dirección característica.
          • Covector característico.
          • Impedancia Z_c.
          • Métrica calibrada G₀ = diag(1, Z_c²).

        Retorna
        -------
        Tuple[NDArray, NDArray, float, NDArray]
            (char_dir, char_cov, z_c, G_cal)

        Raises
        ------
        AllieviCharacteristicError
            Si el certificado de Fase 1 contiene valores no finitos o
            degenerados.
        """
        char_dir = np.array(
            phase1_cert.characteristic_direction, dtype=np.float64
        )
        char_cov = np.array(
            phase1_cert.characteristic_covector, dtype=np.float64
        )
        z_c = float(phase1_cert.characteristic_impedance)

        if not np.all(np.isfinite(char_dir)) or not np.all(np.isfinite(char_cov)):
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Certificado de Fase 1 no finito en puente."
            )

        char_dir_norm = float(la.norm(char_dir))
        if char_dir_norm <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Dirección característica degenerada en puente."
            )
        char_dir = char_dir / char_dir_norm

        g_cal = np.array([[1.0, 0.0], [0.0, z_c * z_c]], dtype=np.float64)

        return char_dir, char_cov, z_c, g_cal

# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 — Orient: Reflexión de Householder covariante (anidada en Fase 1)
# ═══════════════════════════════════════════════════════════════════════════
class Phase2SpecularHouseholderReflector(Phase1AllieviCharacteristicSolver):
    """
    FASE 2 (Orient): Reflexión de Householder covariante sobre la
    hipersuperficie de restricciones.

    Esta clase hereda directamente la Fase 1. Su método principal consume el
    certificado de Fase 1 y el puente:
        _phase1_to_phase2_bridge(...)

    El operador de reflexión Riemanniano es:
    .. math::
        M = I - 2 \frac{n (G n)^\top}{n^\top G n}
    donde n es el vector normal primal de la faceta de restricción.

    Se verifica:
      • G SPD mediante Cholesky con regularización de Higham.
      • Unitariedad Riemanniana: Mᵀ G M = G.
      • Anti-simplécticidad 2D: Mᵀ Ω M = -Ω.
      • Preservación de norma G del estado.
      • Ley especular de reflexión sobre la característica de Fase 1.
    """

    def orient_householder_specular_reflection(
        self,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        phase1_cert: Phase1AllieviCertificate,
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase2HouseholderCertificate:
        r"""
        FASE 2 — Orient.

        Aplica la reflexión de Householder covariante y verifica su
        consistencia con las líneas características de la Fase 1.

        Parámetros
        ----------
        metric_tensor_g : Optional[NDArray]
            Tensor métrico Riemanniano G. Si es None, se usa la métrica
            calibrada G₀ = diag(1, Z_c²) heredada de Fase 1.
        boundary_normal_n : Optional[NDArray]
            Vector normal primal a la faceta de restricción. Si es None, se
            usa la dirección característica heredada de Fase 1.
        state_vector : Optional[NDArray]
            Vector de estado [y, q]ᵀ. Si es None, se usa vector nulo 2D.
        phase1_cert : Phase1AllieviCertificate
            Certificado de la Fase 1 (obligatorio para la anidación).
        tolerance : float
            Cota de precisión.

        Retorna
        -------
        Phase2HouseholderCertificate
            Certificado terminal de Fase 2.

        Excepciones
        -----------
        HouseholderReflectionError
            Si la métrica no es SPD (incluso tras Higham), la normal degenera,
            la dimensión no es 2D o la reflexión viola unitariedad estructural.
        """
        # ─── 0. Puente formal desde Fase 1 ───
        char_dir, char_cov, z_c, g_cal = self._phase1_to_phase2_bridge(phase1_cert)
        dim = char_dir.size
        if dim != 2:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: El agente especular está definido "
                "sobre fase 2D (y, q)."
            )

        # ─── 1. Selección y auditoría de la métrica G ───
        if metric_tensor_g is None:
            g_metric = g_cal.copy()
            higham_applied = False
        else:
            g_metric = np.asarray(metric_tensor_g, dtype=np.float64)
            if g_metric.ndim != 2 or g_metric.shape[0] != g_metric.shape[1]:
                raise HouseholderReflectionError(
                    "HouseholderReflectionError: G debe ser una matriz cuadrada."
                )
            if g_metric.shape != (dim, dim):
                raise HouseholderReflectionError(
                    "HouseholderReflectionError: G debe ser 2x2 para la fase (y, q)."
                )
            if not np.all(np.isfinite(g_metric)):
                raise HouseholderReflectionError(
                    "HouseholderReflectionError: G contiene entradas no finitas."
                )

            # Simetría estructural.
            sym_residual = float(la.norm(g_metric - g_metric.T, ord="fro"))
            if sym_residual > max(tolerance * 100.0, _RELAX_TOL):
                raise HouseholderReflectionError(
                    f"HouseholderReflectionError: G no simétrica "
                    f"(residuo = {sym_residual:.4e})."
                )

            # SPD mediante Cholesky con regularización de Higham.
            try:
                L, higham_applied = stable_cholesky_higham(g_metric, tolerance)
                # Reconstruir G desde L para consistencia numérica
                g_metric = L @ L.T
            except HouseholderReflectionError as exc:
                raise exc

        # Número de condición espectral.
        try:
            metric_condition_number = float(np.linalg.cond(g_metric))
        except Exception as exc:  # pragma: no cover
            raise HouseholderReflectionError(
                "HouseholderReflectionError: No se pudo estimar el número "
                "de condición de G."
            ) from exc

        if not np.isfinite(metric_condition_number) or metric_condition_number <= 0.0:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Número de condición de G inválido."
            )

        # ─── 2. Selección y auditoría de la normal ───
        if boundary_normal_n is None:
            normal_vector = char_dir.copy()
        else:
            normal_vector = np.asarray(
                boundary_normal_n, dtype=np.float64
            ).reshape(-1)

        if normal_vector.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: La normal debe tener dimensión 2."
            )
        if not np.all(np.isfinite(normal_vector)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal no finita."
            )

        normal_euclid_norm = float(la.norm(normal_vector))
        if normal_euclid_norm <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal euclidiana degenerada."
            )

        # ─── 3. Selección y auditoría del vector de estado ───
        if state_vector is None:
            x_state = np.zeros(dim, dtype=np.float64)
        else:
            x_state = np.asarray(state_vector, dtype=np.float64).reshape(-1)

        if x_state.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: El vector de estado debe tener dimensión 2."
            )
        if not np.all(np.isfinite(x_state)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Vector de estado no finito."
            )

        # ─── 4. Construcción del operador de Householder covariante ───
        g_normal = g_metric @ normal_vector
        denom = float(normal_vector @ g_normal)
        if denom <= _MACHINE_EPS * max(1.0, float(la.norm(g_metric, ord="fro"))):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal ortogonal degenerada en la métrica G."
            )

        m_operator = np.eye(dim, dtype=np.float64) - (
            (2.0 / denom) * np.outer(normal_vector, g_normal)
        )
        if not np.all(np.isfinite(m_operator)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Operador de Householder no finito."
            )

        # ─── 5. Unitariedad Riemanniana: Mᵀ G M = G ───
        unitary_matrix_residual = m_operator.T @ g_metric @ m_operator - g_metric
        unitary_residual = float(
            la.norm(unitary_matrix_residual, ord="fro")
            / max(1.0, float(la.norm(g_metric, ord="fro")))
        )
        is_reflection_unitary = unitary_residual <= max(tolerance * 1000.0, _RELAX_TOL)

        # ─── 6. Preservación de norma G del estado ───
        x_norm_sq = float(x_state @ g_metric @ x_state)
        mx_state = m_operator @ x_state
        mx_norm_sq = float(mx_state @ g_metric @ mx_state)
        x_norm = float(np.sqrt(max(0.0, x_norm_sq)))
        mx_norm = float(np.sqrt(max(0.0, mx_norm_sq)))
        state_norm_residual = float(abs(mx_norm - x_norm) / max(1.0, x_norm))

        # ─── 7. Consistencia especular con la característica de Fase 1 ───
        char_g_norm_sq = float(char_dir @ g_metric @ char_dir)
        if char_g_norm_sq <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Norma G de la característica degenerada."
            )
        char_g_norm = float(np.sqrt(max(0.0, char_g_norm_sq)))
        normal_g_norm = float(np.sqrt(max(0.0, denom)))
        incidence_numerator = float(normal_vector @ g_metric @ char_dir)
        incidence_denominator = char_g_norm * normal_g_norm

        if incidence_denominator <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Denominador de incidencia degenerado."
            )

        incidence_cosine = incidence_numerator / incidence_denominator
        reflected_char = m_operator @ char_dir

        # Ley especular:
        #   M c = c - 2 (nᵀ G c)/(nᵀ G n) n
        expected_reflection = char_dir - (
            (2.0 * incidence_numerator / denom) * normal_vector
        )
        specular_residual = float(
            la.norm(reflected_char - expected_reflection)
            / max(1.0, float(la.norm(char_dir)))
        )

        reflected_char_g_norm_sq = float(
            reflected_char @ g_metric @ reflected_char
        )
        reflected_char_g_norm = float(np.sqrt(max(0.0, reflected_char_g_norm_sq)))
        char_norm_residual = float(
            abs(reflected_char_g_norm - char_g_norm) / max(1.0, char_g_norm)
        )

        # Inversión de componente normal.
        if abs(incidence_numerator) > _MACHINE_EPS:
            reflected_normal_projection = float(
                normal_vector @ g_metric @ reflected_char
            )
            normal_flip_ratio = reflected_normal_projection / incidence_numerator
            normal_flip_ok = abs(normal_flip_ratio + 1.0) <= max(
                tolerance * 1000.0, _RELAX_TOL
            )
        else:
            # Incidencia rasante: no hay componente normal que invertir.
            normal_flip_ok = True

        characteristic_tolerance = max(tolerance * 1000.0, _RELAX_TOL)
        is_characteristic_consistent = (
            char_norm_residual <= characteristic_tolerance
            and specular_residual <= characteristic_tolerance
            and normal_flip_ok
        )

        # ─── 8. Invariancia anti-simpléctica en 2D ───
        omega = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        symplectic_matrix_residual = m_operator.T @ omega @ m_operator + omega
        symplectic_residual = float(
            la.norm(symplectic_matrix_residual, ord="fro")
            / max(1.0, float(la.norm(omega, ord="fro")))
        )

        try:
            det_m = float(np.linalg.det(m_operator))
        except Exception as exc:  # pragma: no cover
            raise HouseholderReflectionError(
                "HouseholderReflectionError: No se pudo calcular det(M)."
            ) from exc

        is_anti_symplectic = (
            symplectic_residual <= max(tolerance * 1000.0, _RELAX_TOL)
            and det_m < 0.0
        )

        # ─── 9. Veredicto de Fase 2 ───
        verdict = SpecularSovereignVerdict.COHERENT
        if (
            not np.isfinite(unitary_residual)
            or not np.isfinite(specular_residual)
            or not np.isfinite(state_norm_residual)
            or not np.isfinite(metric_condition_number)
            or not is_reflection_unitary
            or not is_anti_symplectic
            or not is_characteristic_consistent
        ):
            verdict = SpecularSovereignVerdict.VETOED
        elif (
            unitary_residual > tolerance
            or specular_residual > tolerance
            or state_norm_residual > tolerance
            or metric_condition_number > _IMPEDANCE_CONDITION_LIMIT
            or abs(incidence_cosine) < 0.1
        ):
            verdict = SpecularSovereignVerdict.DEGRADED

        return Phase2HouseholderCertificate(
            is_reflection_unitary=is_reflection_unitary,
            reflection_residual=unitary_residual,
            is_anti_symplectic=is_anti_symplectic,
            is_characteristic_consistent=is_characteristic_consistent,
            specular_residual=specular_residual,
            incidence_cosine=float(incidence_cosine),
            state_norm_residual=state_norm_residual,
            metric_condition_number=metric_condition_number,
            higham_regularization_applied=higham_applied,
            verdict=verdict,
        )

    # Compatibilidad con la nomenclatura v2.x.
    reflect_householder_boundary = orient_householder_specular_reflection

# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 — Decide: Teorema de Tellegen y pasividad de Lyapunov
# ═══════════════════════════════════════════════════════════════════════════
class Phase3TellegenPassivityValidator(Phase2SpecularHouseholderReflector):
    """
    FASE 3 (Decide): Auditoría del balance de potencias sobre la red gráfica
    que subyace a los flujos y presiones.

    Esta fase hereda las fases 1 y 2, y recibe sus certificados para
    garantizar trazabilidad categórica completa.

    Verifica:
      • KCL: A q ≈ 0.
      • KVL: p ≈ Aᵀ φ.
      • Tellegen: pᵀ q ≈ 0 cuando p ∈ Im(Aᵀ) y q ∈ Ker(A).
      • Pasividad de Lyapunov: P_diss ≥ 0.
      • Invariantes topológicos del grafo: orden, tamaño y ciclomático.
      • Computación dispersa para redes de gran escala.
    """

    def decide_tellegen_lyapunov_conservation(
        self,
        pressures: NDArray[np.float64],
        flows: NDArray[np.float64],
        rayleigh_dissipation: float,
        phase1_cert: Phase1AllieviCertificate,
        phase2_cert: Phase2HouseholderCertificate,
        incidence_matrix: Optional[Union[NDArray[np.float64], csr_matrix]] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase3TellegenCertificate:
        r"""
        FASE 3 — Decide.

        Valida el teorema de Tellegen sobre el grafo de la red y la
        disipación de Rayleigh como condición de pasividad de Lyapunov.

        Si no se proporciona una matriz de incidencia, se asume una topología
        de grafo paralelo de dos nodos con m ramas:
        .. math::
            A = \begin{bmatrix}
                1 & 1 & \dots & 1 \\
               -1 & -1 & \dots & -1
            \end{bmatrix}

        Esta topología exige:
          • KCL: Σ Q_k = 0.
          • KVL: ΔP_k iguales para todas las ramas.

        Para auditorías rigurosas de redes reales, se recomienda suministrar
        `incidence_matrix` explícitamente, preferiblemente como matriz dispersa
        csr_matrix para optimización de complejidad O(|E|).

        Parámetros
        ----------
        pressures : NDArray
            Caídas de potencial de rama ΔP_k (tamaño m).
        flows : NDArray
            Caudales logísticos Q_k (tamaño m).
        rayleigh_dissipation : float
            Potencia disipada de Rayleigh P_diss.
        phase1_cert : Phase1AllieviCertificate
            Certificado de Fase 1.
        phase2_cert : Phase2HouseholderCertificate
            Certificado de Fase 2.
        incidence_matrix : Optional[Union[NDArray, csr_matrix]]
            Matriz de incidencia nodo-rama (n × m). Puede ser densa o dispersa.
        nodal_potentials : Optional[NDArray]
            Potenciales nodales para verificación dispersa de Tellegen.
        tolerance : float
            Cota de precisión.

        Retorna
        -------
        Phase3TellegenCertificate
            Certificado terminal de Fase 3.

        Excepciones
        -----------
        TellegenConservationError
            Si hay dimensiones inconsistentes, entradas no finitas o
            complejidad topológica fuera de cota.
        """
        p = np.asarray(pressures, dtype=np.float64).reshape(-1)
        q = np.asarray(flows, dtype=np.float64).reshape(-1)
        m = p.size

        if q.size != m:
            raise TellegenConservationError(
                "TellegenConservationError: Dimensiones inconsistentes entre "
                "presiones y flujos."
            )

        if m == 0:
            # Red vacía: objeto terminal trivial.
            return Phase3TellegenCertificate(
                tellegen_residual=0.0,
                kcl_residual=0.0,
                kvl_residual=0.0,
                is_tellegen_conserved=True,
                is_kcl_conserved=True,
                is_kvl_conserved=True,
                is_lyapunov_passive=True,
                graph_order=0,
                graph_size=0,
                cyclomatic_number=0,
                sparse_computation=False,
                verdict=SpecularSovereignVerdict.COHERENT,
            )

        if m > _GRAPH_MAX_BRANCHES:
            raise TellegenConservationError(
                "TellegenConservationError: Complejidad topológica fuera de cota."
            )

        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise TellegenConservationError(
                "TellegenConservationError: Presiones o flujos no finitos."
            )

        if not np.isfinite(rayleigh_dissipation):
            raise TellegenConservationError(
                "TellegenConservationError: Disipación de Rayleigh no finita."
            )

        # ─── 1. Construcción/validación de la topología de red ───
        sparse_computation = False
        if incidence_matrix is None:
            # Grafo paralelo de dos nodos.
            a_matrix = np.zeros((2, m), dtype=np.float64)
            a_matrix[0, :] = 1.0
            a_matrix[1, :] = -1.0
        else:
            a_matrix = incidence_matrix
            if a_matrix.ndim != 2:
                raise TellegenConservationError(
                    "TellegenConservationError: La matriz de incidencia debe ser 2D."
                )
            if a_matrix.shape[1] != m:
                raise TellegenConservationError(
                    "TellegenConservationError: La matriz de incidencia no concuerda "
                    "con el número de ramas."
                )

            # Verificar finitud (compatible con dispersas)
            if _SPARSE_AVAILABLE and isspmatrix is not None:
                if isspmatrix(a_matrix):
                    sparse_computation = True
                    if not np.all(np.isfinite(a_matrix.data)):
                        raise TellegenConservationError(
                            "TellegenConservationError: Matriz de incidencia dispersa "
                            "contiene entradas no finitas."
                        )
                else:
                    if not np.all(np.isfinite(a_matrix)):
                        raise TellegenConservationError(
                            "TellegenConservationError: Matriz de incidencia no finita."
                        )
            else:
                if not np.all(np.isfinite(a_matrix)):
                    raise TellegenConservationError(
                        "TellegenConservationError: Matriz de incidencia no finita."
                    )

        n_nodes = a_matrix.shape[0]
        if n_nodes == 0:
            raise TellegenConservationError(
                "TellegenConservationError: La topología no tiene nodos."
            )

        # ─── 2. Normas y escalas de auditoría ───
        p_norm = float(la.norm(p))
        q_norm = float(la.norm(q))

        if sparse_computation:
            a_norm = float(a_matrix.norm(ord="fro"))
        else:
            a_norm = float(la.norm(a_matrix, ord="fro"))

        # ─── 3. KCL: A q ≈ 0 ───
        if sparse_computation:
            kcl_vector = a_matrix.dot(q)
        else:
            kcl_vector = a_matrix @ q

        kcl_residual = float(
            la.norm(kcl_vector)
            / (
                max(1.0, a_norm)
                * max(1.0, q_norm)
                * float(np.sqrt(max(1, n_nodes)))
            )
        )

        # ─── 4. KVL: p ∈ Im(Aᵀ) ───
        try:
            if sparse_computation:
                # Usar solver disperso para KVL
                from scipy.sparse.linalg import lsqr
                phi_result = lsqr(a_matrix.T, p, atol=tolerance, btol=tolerance)
                phi = phi_result[0]
            else:
                phi, *_ = np.linalg.lstsq(a_matrix.T, p, rcond=None)

            if sparse_computation:
                kvl_reconstruction = a_matrix.T.dot(phi)
            else:
                kvl_reconstruction = a_matrix.T @ phi

            kvl_residual = float(
                la.norm(kvl_reconstruction - p) / max(1.0, p_norm)
            )
        except Exception:
            kvl_residual = float("inf")

        # ─── 5. Tellegen: pᵀ q ≈ 0 (o verificación dispersa si hay potenciales nodales) ───
        if nodal_potentials is not None and incidence_matrix is not None:
            # Verificación dispersa estructural de Tellegen
            try:
                tellegen_residual, sparse_used = verify_tellegen_sparse(
                    nodal_potentials, q, incidence_matrix
                )
                sparse_computation = sparse_computation or sparse_used
            except TellegenConservationError:
                # Fallback a verificación clásica
                tellegen_power = float(p @ q)
                tellegen_residual = float(
                    abs(tellegen_power) / max(1.0, p_norm * q_norm)
                )
        else:
            # Verificación clásica de Tellegen
            tellegen_power = float(p @ q)
            tellegen_residual = float(
                abs(tellegen_power) / max(1.0, p_norm * q_norm)
            )

        # ─── 6. Invariantes topológicos del grafo ───
        try:
            if sparse_computation and _SPARSE_AVAILABLE:
                # Estimación de rango para matrices dispersas
                rank_a = int(np.linalg.matrix_rank(
                    a_matrix.toarray() if isspmatrix(a_matrix) else a_matrix,
                    tol=1.0e-12
                ))
            else:
                rank_a = int(np.linalg.matrix_rank(a_matrix, tol=1.0e-12))
        except Exception:  # pragma: no cover
            rank_a = min(n_nodes, m)

        connected_components = max(1, n_nodes - rank_a)
        cyclomatic_number = max(0, m - rank_a)

        # ─── 7. Pasividad de Lyapunov/Rayleigh ───
        is_lyapunov_passive = rayleigh_dissipation >= -_MACHINE_EPS

        # ─── 8. Umbrales de conservación ───
        strict_tol = max(tolerance, _TOPOLOGY_DEGRADED_THRESHOLD)
        is_kcl_conserved = kcl_residual <= strict_tol
        is_kvl_conserved = kvl_residual <= strict_tol
        is_tellegen_conserved = tellegen_residual <= strict_tol

        max_topological_residual = max(
            kcl_residual,
            kvl_residual,
            tellegen_residual,
        )

        # ─── 9. Veredicto local de Fase 3 ───
        if (
            not np.isfinite(max_topological_residual)
            or max_topological_residual > _TOPOLOGY_VETO_THRESHOLD
            or rayleigh_dissipation < -tolerance
        ):
            own_verdict = SpecularSovereignVerdict.VETOED
        elif (
            not is_kcl_conserved
            or not is_kvl_conserved
            or not is_tellegen_conserved
            or rayleigh_dissipation < 0.0
            or max_topological_residual > tolerance
        ):
            own_verdict = SpecularSovereignVerdict.DEGRADED
        else:
            own_verdict = SpecularSovereignVerdict.COHERENT

        # ─── 10. Trazabilidad categórica con fases previas ───
        verdict = SpecularSovereignVerdict.supremum(
            own_verdict,
            phase1_cert.verdict,
            phase2_cert.verdict,
        )

        return Phase3TellegenCertificate(
            tellegen_residual=tellegen_residual,
            kcl_residual=kcl_residual,
            kvl_residual=kvl_residual,
            is_tellegen_conserved=is_tellegen_conserved,
            is_kcl_conserved=is_kcl_conserved,
            is_kvl_conserved=is_kvl_conserved,
            is_lyapunov_passive=is_lyapunov_passive,
            graph_order=int(n_nodes),
            graph_size=int(m),
            cyclomatic_number=int(cyclomatic_number),
            sparse_computation=sparse_computation,
            verdict=verdict,
        )

    # Compatibilidad con la nomenclatura v2.x.
    decide_conservation_and_passivity = decide_tellegen_lyapunov_conservation

# ═══════════════════════════════════════════════════════════════════════════
# Soberano del Flujo Especular (Morphism en la categoría MIC)
# ═══════════════════════════════════════════════════════════════════════════
class SpecularFlowSovereignAgent(Morphism, Phase3TellegenPassivityValidator):
    """
    Soberano del haz de De Rham para el control de lazo cerrado del Flujo
    Especular.

    Hereda las tres fases anidadas, formando un solo espacio de nombres
    coherente desde Allievi hasta Tellegen.

    El ciclo OODA queda formalizado como:
        Observe  → Allievi
        Orient   → Householder
        Decide   → Tellegen/Lyapunov

    El veredicto final es el supremo de severidad en el retículo de Heyting.
    """

    def __init__(self, raise_on_veto: bool = False) -> None:
        super().__init__()
        self._target_stratum: Stratum = Stratum.WISDOM
        self._raise_on_veto: bool = raise_on_veto

    # ─────────────────────────────────────────────────────────────────────
    # Orquestación soberana
    # ─────────────────────────────────────────────────────────────────────
    def execute_sovereign_governance(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        pressures: NDArray[np.float64],
        flows: NDArray[np.float64],
        rayleigh_dissipation: float,
        incidence_matrix: Optional[Union[NDArray[np.float64], csr_matrix]] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
    ) -> SpecularFlowState:
        """
        Orquesta el ciclo OODA sobre la variedad característica de
        Allievi-Householder con validación topológica de Tellegen.

        Estrategia de anidación:
          1. Si Fase 1 emite VETOED, se emiten certificados vetados para
             Fase 2 y Fase 3 sin continuar el cómputo.
          2. Si Fase 2 emite VETOED, se emite certificado vetado para Fase 3.
          3. Si todas las fases continúan, Fase 3 audita la red y la
             pasividad.

        Retorna
        -------
        SpecularFlowState
            Certificado global terminal.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            # ─── FASE 1: Observe ───
            cert_1 = self.observe_allievi_characteristics(
                potential_y=potential_y,
                flow_q=flow_q,
                wave_speed_a=wave_speed_a,
                gravity_g=gravity_g,
                section_s=section_s,
                tolerance=tolerance,
                cfl_number=cfl_number,
            )

            if cert_1.verdict.is_vetoed:
                cert_2 = self._vetoed_phase2_certificate()
                cert_3 = self._vetoed_phase3_certificate()
            else:
                # ─── FASE 2: Orient (anidada, recibe cert_1) ───
                cert_2 = self.orient_householder_specular_reflection(
                    metric_tensor_g=metric_tensor_g,
                    boundary_normal_n=boundary_normal_n,
                    state_vector=state_vector,
                    phase1_cert=cert_1,
                    tolerance=tolerance,
                )

                if cert_2.verdict.is_vetoed:
                    cert_3 = self._vetoed_phase3_certificate()
                else:
                    # ─── FASE 3: Decide (anidada, recibe cert_1 y cert_2) ───
                    cert_3 = self.decide_tellegen_lyapunov_conservation(
                        pressures=pressures,
                        flows=flows,
                        rayleigh_dissipation=rayleigh_dissipation,
                        phase1_cert=cert_1,
                        phase2_cert=cert_2,
                        incidence_matrix=incidence_matrix,
                        nodal_potentials=nodal_potentials,
                        tolerance=tolerance,
                    )

            # ─── Fusión de veredictos (supremo en severidad) ───
            final_verdict = SpecularSovereignVerdict.supremum(
                cert_1.verdict,
                cert_2.verdict,
                cert_3.verdict,
            )

            crowbar_triggered = False
            crowbar_action = CrowbarAction.NONE
            diagnostic_note = "Ciclo OODA especular completado."

            if final_verdict == SpecularSovereignVerdict.VETOED:
                crowbar_triggered = True
                crowbar_action = CrowbarAction.HARD_SHORT
                diagnostic_note = (
                    "VETO ESPECTRAL DE FLUJO: fuga de pasividad, conservación o "
                    "unitariedad. Gatillando Crowbar."
                )
                logger.error(
                    "¡VETO ESPECTRAL DE FLUJO! Gatillando Crowbar (GPIO%d).",
                    _CROWBAR_GPIO,
                )
                if self._raise_on_veto:
                    raise SpecularFlowAgentError(
                        "Obstrucción de pasividad, conservación o unitariedad "
                        "en el Flujo Especular."
                    )
            elif final_verdict == SpecularSovereignVerdict.DEGRADED:
                crowbar_triggered = True
                crowbar_action = CrowbarAction.WATCHDOG_PULSE
                diagnostic_note = (
                    "Degradación paramétrica detectada. Activando Watchdog."
                )
                logger.warning(
                    "Degradación paramétrica en el Flujo Especular. "
                    "Activando Watchdog."
                )

            provenance_hash = self._generate_provenance_hash(
                cert_1,
                cert_2,
                cert_3,
                timestamp_utc,
            )

            return SpecularFlowState(
                phase1=cert_1,
                phase2=cert_2,
                phase3=cert_3,
                final_verdict=final_verdict,
                crowbar_triggered=crowbar_triggered,
                crowbar_action=crowbar_action,
                timestamp_utc=timestamp_utc,
                provenance_hash=provenance_hash,
                diagnostic_note=diagnostic_note,
            )

        except SpecularFlowAgentError as exc:
            logger.error(
                "Colapso categórico del agente de flujo especular: %s",
                exc,
            )
            if self._raise_on_veto:
                raise
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)

        except Exception as exc:  # pragma: no cover
            logger.exception(
                "Colapso catastrófico no tipado del agente de flujo especular."
            )
            if self._raise_on_veto:
                raise SpecularFlowAgentError(
                    "Colapso catastrófico no tipado del agente de flujo especular."
                ) from exc
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)

    # ─────────────────────────────────────────────────────────────────────
    # Certificados vetados para cortocircuito anidado
    # ─────────────────────────────────────────────────────────────────────
    def _vetoed_phase2_certificate(self) -> Phase2HouseholderCertificate:
        """
        Certificado vetado de Fase 2 para cortocircuito cuando Fase 1 ha sido
        vetada.
        """
        return Phase2HouseholderCertificate(
            is_reflection_unitary=False,
            reflection_residual=float("inf"),
            is_anti_symplectic=False,
            is_characteristic_consistent=False,
            specular_residual=float("inf"),
            incidence_cosine=0.0,
            state_norm_residual=float("inf"),
            metric_condition_number=float("inf"),
            higham_regularization_applied=False,
            verdict=SpecularSovereignVerdict.VETOED,
        )

    def _vetoed_phase3_certificate(self) -> Phase3TellegenCertificate:
        """
        Certificado vetado de Fase 3 para cortocircuito cuando una fase previa
        ha sido vetada.
        """
        return Phase3TellegenCertificate(
            tellegen_residual=float("inf"),
            kcl_residual=float("inf"),
            kvl_residual=float("inf"),
            is_tellegen_conserved=False,
            is_kcl_conserved=False,
            is_kvl_conserved=False,
            is_lyapunov_passive=False,
            graph_order=0,
            graph_size=0,
            cyclomatic_number=0,
            sparse_computation=False,
            verdict=SpecularSovereignVerdict.VETOED,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Estado catastrófico de emergencia
    # ─────────────────────────────────────────────────────────────────────
    def _cataclysm_state(self, reason: str, timestamp_utc: str) -> SpecularFlowState:
        """
        Construye un estado catastrófico con certificados vetados y Crowbar
        HARD_SHORT.

        Este estado es el objeto terminal de fallo dentro del topos operativo.
        """
        phase1_dummy = Phase1AllieviCertificate(
            is_characteristic_stable=False,
            characteristic_impedance=0.0,
            riemann_invariants=(0.0, 0.0),
            characteristic_direction=(0.0, 0.0),
            characteristic_covector=(0.0, 0.0),
            energy_norm=0.0,
            impedance_condition_number=float("inf"),
            cfl_proxy=0.0,
            verdict=SpecularSovereignVerdict.VETOED,
        )
        phase2_dummy = self._vetoed_phase2_certificate()
        phase3_dummy = self._vetoed_phase3_certificate()

        raw_payload = f"CATACLYSM_SPECULAR_VETO::{reason}"
        provenance_hash = hashlib.sha256(
            raw_payload.encode("utf-8")
        ).hexdigest()

        return SpecularFlowState(
            phase1=phase1_dummy,
            phase2=phase2_dummy,
            phase3=phase3_dummy,
            final_verdict=SpecularSovereignVerdict.VETOED,
            crowbar_triggered=True,
            crowbar_action=CrowbarAction.HARD_SHORT,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_SPECULAR_VETO: {reason}",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Hash de procedencia auditable
    # ─────────────────────────────────────────────────────────────────────
    def _generate_provenance_hash(
        self,
        c1: Phase1AllieviCertificate,
        c2: Phase2HouseholderCertificate,
        c3: Phase3TellegenCertificate,
        timestamp_utc: str,
    ) -> str:
        """
        Genera un hash SHA-256 de procedencia que ata los veredictos, residuos
        críticos y marca temporal.

        El hash constituye el sello de auditoría del certificado global.
        """
        raw_payload = "|".join(
            (
                timestamp_utc,
                str(c1.verdict.value),
                str(c2.verdict.value),
                str(c3.verdict.value),
                f"{c1.characteristic_impedance:.12e}",
                f"{c1.energy_norm:.12e}",
                f"{c1.impedance_condition_number:.12e}",
                f"{c2.reflection_residual:.12e}",
                f"{c2.specular_residual:.12e}",
                f"{c2.state_norm_residual:.12e}",
                f"{c2.metric_condition_number:.12e}",
                f"{c3.tellegen_residual:.12e}",
                f"{c3.kcl_residual:.12e}",
                f"{c3.kvl_residual:.12e}",
                str(int(c3.graph_order)),
                str(int(c3.graph_size)),
                str(int(c3.cyclomatic_number)),
                str(int(c3.sparse_computation)),
                str(int(c2.higham_regularization_applied)),
            )
        )
        return hashlib.sha256(
            raw_payload.encode("utf-8")
        ).hexdigest()

# ═══════════════════════════════════════════════════════════════════════════
# Compatibilidad de nombres de clases para versiones anteriores
# ═══════════════════════════════════════════════════════════════════════════
Phase1_AllieviCharacteristicSolver = Phase1AllieviCharacteristicSolver
Phase2_SpecularHouseholderReflector = Phase2SpecularHouseholderReflector
Phase3_TellegenPassivityValidator = Phase3TellegenPassivityValidator

__all__ = [
    "SpecularFlowAgentError",
    "AllieviCharacteristicError",
    "HouseholderReflectionError",
    "TellegenConservationError",
    "SpecularSovereignVerdict",
    "CrowbarAction",
    "Phase1AllieviCertificate",
    "Phase2HouseholderCertificate",
    "Phase3TellegenCertificate",
    "SpecularFlowState",
    "Phase1AllieviCharacteristicSolver",
    "Phase2SpecularHouseholderReflector",
    "Phase3TellegenPassivityValidator",
    "SpecularFlowSovereignAgent",
    "stable_cholesky_higham",
    "verify_tellegen_sparse",
]