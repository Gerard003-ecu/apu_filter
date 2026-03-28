"""
=========================================================================================
Módulo: Deliberation Manifold (El Ágora Tensorial — Estrato Ω)
Ubicación: app/core/immune_system/deliberation_manifold.py
=========================================================================================

Naturaleza Ciber-Física:
    Actúa como el Funtor de Colapso de la Función de Estado del sistema. Intercepta 
    los tensores continuos provenientes de los subespacios de Topología (V_TACTICS) 
    y Finanzas (V_STRATEGY), sometiéndolos a la fricción geométrica del territorio 
    para colapsar el sistema en un vértice operativo determinista. Su función axiomática 
    es erradicar el libre albedrío estocástico del Modelo de Lenguaje (LLM), forzándolo 
    a acatar el límite superior del riesgo físico.

1. Ecuación de Estado (Estrés Ajustado Tensorial):
    El escalar continuo de estrés ajustado (σ*) se define mediante la contracción:
        σ* = T_int ⊗ F_ext ⊗ Λ ⊗ P_frag
    Donde:
    • T_int (Tensión Interna): Matriz de covarianza entre la desalineación estratégica 
      y el acoplamiento gravitacional del presupuesto.
    • F_ext (Fricción Externa): Evaluación sobre una métrica Riemanniana territorial,
      ponderando asimétricamente el clima, la logística y la entropía social.
    • Λ (Palanca de Improbabilidad): Operador de amplificación para anomalías en 
      la homología persistente.
    • P_frag (Penalización Estructural): Inverso del Índice de Estabilidad Piramidal 
      (Si Ψ < 1.0, P_frag amplifica exponencialmente el colapso).

2. Proyección sobre el Retículo de Severidad (Lattice Theory):
    El Manifold mapea el escalar continuo σ* ∈ ℝ⁺ hacia un espacio discreto utilizando 
    un Retículo Acotado Distributivo (VerdictLevel, ≤, ⊔, ⊓). 
    El veredicto final se extrae aplicando estrictamente la operación Supremo (⊔ / Join):
        Veredicto = ⊔ (σ* → L_i)
    Esto garantiza matemáticamente el "Worst-Case Scenario": si una sola dimensión 
    física o topológica cruza el umbral crítico, la operación Supremo satura la 
    ecuación a ⊤ (RECHAZAR), sin posibilidad de dilución estocástica.

3. Ley de Clausura Transitiva (Filtración DIKW):
    El morfismo respeta la filtración topológica estricta:
        V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_Ω ⊂ V_WISDOM
    Un fallo en un estrato subyacente F_i propaga invariablemente el colapso hacia 
    los estratos superiores F_j (para j < i). 

Invariantes Algebraicos y Numéricos Garantizados:
    • Finitud estricta: Todo float intermedio y resultante (σ*) es comprobablemente 
      finito (¬NaN, ¬±Inf), protegiendo la Unidad de Punto Flotante (FPU).
    • Idempotencia de consolidación: AnomalyData.consolidate() = (AnomalyData.consolidate())²
    • Pureza Funcional: La operación de colapso _collapse() es un homomorfismo puro, 
      carente de efectos secundarios en el espacio de fase.
=========================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum
from app.wisdom.semantic_translator import VerdictLevel
from app.core.immune_system.calibration.sheaf_cohomology_orchestrator import SheafDegeneracyError, HomologicalInconsistencyError, SheafCohomologyOrchestrator

logger = logging.getLogger("MIC.Omega.DeliberationManifold")


# =============================================================================
# CONSTANTES DEL MANIFOLD
# =============================================================================

# ---------------------------------------------------------------------------
# Umbrales del retículo de veredictos.
#
# Derivación empírica:
#   - σ* = 0 para sistema ideal (ψ=1, ROI=1, sin anomalías, sin territorio).
#   - σ* ≈ 0.5 para desalineación moderada sin anomalías.
#   - σ* ≈ 1.5 para fragilidad + anomalías moderadas.
#   - σ* ≈ 3.0 para fragilidad severa + anomalías + territorio hostil.
# ---------------------------------------------------------------------------
_VERDICT_THRESHOLD_VIABLE: float = 0.75
_VERDICT_THRESHOLD_CONDICIONAL: float = 1.75
_VERDICT_THRESHOLD_PRECAUCION: float = 3.00

# ---------------------------------------------------------------------------
# Pesos de la métrica territorial riemanniana simplificada.
# Suman 1.0 para que la fricción externa sea una media ponderada.
# ---------------------------------------------------------------------------
_FRICTION_WEIGHT_LOGISTICS: float = 0.4
_FRICTION_WEIGHT_SOCIAL: float = 0.4
_FRICTION_WEIGHT_CLIMATE: float = 0.2

# Verificación estática de que los pesos suman 1.0
assert math.isclose(
    _FRICTION_WEIGHT_LOGISTICS + _FRICTION_WEIGHT_SOCIAL + _FRICTION_WEIGHT_CLIMATE,
    1.0,
    rel_tol=1e-9,
), "Los pesos de fricción territorial deben sumar 1.0"

# ---------------------------------------------------------------------------
# Coeficientes de presión anómala.
# Representan el impacto relativo de cada tipo de anomalía topológica
# sobre la presión estructural.
# ---------------------------------------------------------------------------
_ANOMALY_COEFF_CYCLE: float = 0.08
_ANOMALY_COEFF_ISOLATED: float = 0.03
_ANOMALY_COEFF_STRESSED: float = 0.05

# ---------------------------------------------------------------------------
# Punto de inflexión de gravity_coupling.
# Calibrado para que fragility_norm ≈ 0.5 (neutral) produzca coupling ≈ 1.0
# y fragility_norm → 1 produzca coupling → 2.0.
# ---------------------------------------------------------------------------
_GRAVITY_INFLECTION: float = 0.5

# ---------------------------------------------------------------------------
# Factor de escala del improbability lever.
# Controla cuánto amplifican las anomalías combinatorias el estrés base.
# Valor 2.0 implica que el lever puede como máximo cuadruplicar el estrés
# (clamp a [1, 4]).
# ---------------------------------------------------------------------------
_IMPROBABILITY_SCALE_FACTOR: float = 2.0
_IMPROBABILITY_CLAMP_LOW: float = 1.0
_IMPROBABILITY_CLAMP_HIGH: float = 4.0

# ---------------------------------------------------------------------------
# Penalización máxima por fragilidad.
# penalty = 1 + min(MAX_PENALTY_DELTA, (1 - ψ) × MAX_PENALTY_DELTA)
# Con MAX_PENALTY_DELTA = 1.5, el penalty máximo es 2.5.
# ---------------------------------------------------------------------------
_FRAGILITY_PENALTY_MAX_DELTA: float = 1.5

# ---------------------------------------------------------------------------
# Rangos de clamp para inputs.
# ---------------------------------------------------------------------------
_PSI_CLAMP_LOW: float = 0.05
_PSI_CLAMP_HIGH: float = 5.0
_ROI_CLAMP_LOW: float = 0.0
_ROI_CLAMP_HIGH: float = 5.0
_FRICTION_CLAMP_LOW: float = 0.0
_FRICTION_CLAMP_HIGH: float = 5.0

# ---------------------------------------------------------------------------
# Epsilon numérico para evitar divisiones por cero.
# ---------------------------------------------------------------------------
_EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# Umbrales de interpretación semántica.
# ---------------------------------------------------------------------------
_PSI_FRAGILE_THRESHOLD: float = 0.75
_PSI_ROBUST_THRESHOLD: float = 1.25
_ROI_WEAK_THRESHOLD: float = 1.0
_ROI_MODERATE_THRESHOLD: float = 1.5
_FRICTION_FAVORABLE_THRESHOLD: float = 1.25
_FRICTION_MODERATE_THRESHOLD: float = 2.0
_STRESS_LOW_THRESHOLD: float = _VERDICT_THRESHOLD_VIABLE
_STRESS_MODERATE_THRESHOLD: float = _VERDICT_THRESHOLD_CONDICIONAL
_STRESS_HIGH_THRESHOLD: float = _VERDICT_THRESHOLD_PRECAUCION

# ---------------------------------------------------------------------------
# Límites por defecto del SynapticRegistry.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_CARTRIDGES: int = 16
_DEFAULT_MAX_CHARS: int = 12000


# =============================================================================
# 1. GESTIÓN DE CAPACIDADES EMERGENTES (TOON VITAMINS)
# =============================================================================


@dataclass(frozen=True)
class ToonCartridge:
    """Cartucho Sináptico (Vitamina Cognitiva).

    Contiene conocimiento de dominio comprimido en formato TOON para
    inyección contextual en prompts LLM.

    Invariantes:
    - ``name`` no puede estar vacío tras strip.
    - ``weight`` es un float finito ≥ 0.
    """

    name: str
    domain: str
    toon_payload: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("El cartucho debe tener un nombre no vacío")
        if not math.isfinite(self.weight) or self.weight < 0:
            object.__setattr__(self, "weight", max(0.0, _safe_float(self.weight, 1.0)))


class SynapticRegistry:
    """Registro en memoria de capacidades emergentes TOON.

    Thread-safety: NO es thread-safe. Uso previsto en contexto
    de request o pipeline secuencial.
    """

    _EMPTY_CONTEXT: str = "CONTEXTO_COGNITIVO|VACIO"

    def __init__(self) -> None:
        self._cartridges: Dict[str, ToonCartridge] = {}

    @property
    def cartridge_count(self) -> int:
        """Número de cartuchos registrados."""
        return len(self._cartridges)

    def load_cartridge(self, cartridge: ToonCartridge, telemetry_context: Optional[Any] = None) -> None:
        """Acopla una nueva vitamina cognitiva al sistema.

        Permite la Cirugía Topológica Exógena: Si se inyecta un Positrón (Antimateria Exógena),
        se busca un Electrón de masa equivalente. Si se encuentra, ambos se aniquilan del
        SynapticRegistry y se emite un GammaPhoton como prueba criptográfica a través
        de telemetry_context.

        Raises:
            TypeError: si ``cartridge`` no es instancia de ``ToonCartridge``.
            ValueError: si el nombre del cartucho está vacío.
        """
        import json

        if not isinstance(cartridge, ToonCartridge):
            raise TypeError(
                f"cartridge debe ser instancia de ToonCartridge, "
                f"se recibió {type(cartridge).__name__}"
            )

        # La validación de nombre vacío ya está en __post_init__,
        # pero la repetimos por defensa en profundidad.
        if not cartridge.name.strip():
            raise ValueError("El cartucho debe tener un nombre no vacío")

        # Evaluar Aniquilación de Electrón-Positrón
        if "PositronCartridge" in cartridge.toon_payload:
            try:
                positron_data = json.loads(cartridge.toon_payload.split("|")[-1])
                pos_mass = positron_data.get("inertial_mass")

                # Buscar electrón correspondiente para aniquilar
                for existing_name, existing_cartridge in list(self._cartridges.items()):
                    if "ElectronCartridge" in existing_cartridge.toon_payload:
                        electron_data = json.loads(existing_cartridge.toon_payload.split("|")[-1])
                        elec_mass = electron_data.get("inertial_mass")

                        if pos_mass == elec_mass:
                            del self._cartridges[existing_name]

                            # Emisión de radiación de auditoría (Fotón Gamma)
                            if telemetry_context is not None:
                                from app.core.telemetry_schemas import GammaPhoton
                                import time
                                import hashlib
                                data_hash = hashlib.sha256(str(pos_mass).encode()).hexdigest()
                                gamma = GammaPhoton(
                                    annihilation_energy=2 * pos_mass * (3e8)**2,
                                    data_hash=data_hash,
                                    timestamp_entry=time.time(),
                                    authorization_signature=positron_data.get("authorization_signature", "unknown")
                                )
                                telemetry_context.record_error(
                                    step_name="Exogenous_Topological_Surgery",
                                    error_message="Electron-Positron Annihilation in RAM.",
                                    error_type="GammaPhotonEmission",
                                    severity="INFO",
                                    stratum=Stratum.OMEGA,
                                    metadata={"gamma_photon": gamma.__dict__}
                                )
                            logger.info("💥 Aniquilación Electrón-Positrón completada. Radiación Gamma emitida.")
                            return
            except Exception as e:
                logger.warning(f"Error evaluando inyección de Positrón: {e}")

        self._cartridges[cartridge.name] = cartridge
        logger.info(
            "🧠 Cartucho Sináptico acoplado: %s [%s]",
            cartridge.name,
            cartridge.domain,
        )

    def get_active_context(
        self,
        max_items: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """Concatena los TOONs activos para inyección en prompt LLM.

        Política de selección:
        1. Ordena por peso descendente (desempate alfabético por nombre).
        2. Omite payloads vacíos tras strip.
        3. Respeta ``max_items`` (número máximo de cartuchos).
        4. Respeta ``max_chars`` (longitud máxima total incluyendo separadores).

        Retorna ``"CONTEXTO_COGNITIVO|VACIO"`` si no hay contenido disponible.
        """
        if not self._cartridges:
            return self._EMPTY_CONTEXT

        cartridges = sorted(
            self._cartridges.values(),
            key=lambda c: (-_safe_float(c.weight, 1.0), c.name),
        )

        payloads: List[str] = []
        total_chars = 0
        separator_len = 1  # len("\n")

        for cartridge in cartridges:
            payload = (cartridge.toon_payload or "").strip()
            if not payload:
                continue

            # Check max_items AFTER filtering empty payloads
            if max_items is not None and max_items >= 0 and len(payloads) >= max_items:
                break

            if max_chars is not None and max_chars >= 0:
                # Contabilizar separador entre payloads existentes
                separator_cost = separator_len if payloads else 0
                projected = total_chars + separator_cost + len(payload)
                if projected > max_chars:
                    break

            payloads.append(payload)
            # Acumular incluyendo separador previo
            if len(payloads) > 1:
                total_chars += separator_len
            total_chars += len(payload)

        return "\n".join(payloads) if payloads else self._EMPTY_CONTEXT


# =============================================================================
# 2. MODELOS FORMALES DEL MANIFOLD
# =============================================================================


@dataclass(frozen=True)
class OmegaInputs:
    """Coordenadas saneadas de entrada al manifold deliberativo.

    Todos los campos numéricos están clamped a rangos seguros en ``from_payload()``.

    Invariantes:
    - ``psi`` ∈ [_PSI_CLAMP_LOW, _PSI_CLAMP_HIGH]
    - ``roi`` ∈ [_ROI_CLAMP_LOW, _ROI_CLAMP_HIGH]
    - ``n_nodes``, ``n_edges`` ≥ 1
    - ``cycle_count``, ``isolated_count``, ``stressed_count`` ≥ 0
    - Fricciones ∈ [_FRICTION_CLAMP_LOW, _FRICTION_CLAMP_HIGH]
    """

    # Topología
    psi: float = 1.0
    n_nodes: int = 1
    n_edges: int = 1
    cycle_count: int = 0
    isolated_count: int = 0
    stressed_count: int = 0

    # Finanzas
    roi: float = 1.0

    # Territorio
    logistics_friction: float = 1.0
    social_friction: float = 1.0
    climate_entropy: float = 1.0
    territory_present: bool = False

    # Contexto Visual (Resolución de Homotopía)
    focus_node_id: Optional[str] = None
    zoom_level: Optional[int] = None

    # Raw states para trazabilidad (inmutabilidad de contenido no garantizada
    # por frozen=True en dicts; se documentan como read-only por contrato)
    topo_data: Dict[str, Any] = field(default_factory=dict)
    fin_data: Dict[str, Any] = field(default_factory=dict)
    territory_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> OmegaInputs:
        """Construye inputs saneados desde el payload del estado categórico.

        Política de saneamiento:
        - Valores ausentes → defaults neutrales.
        - Valores no-numéricos → defaults.
        - Valores fuera de rango → clamped.
        - territory_present = True solo si territory_data es dict NO vacío.
        """
        safe_payload = _safe_dict(payload)
        topo_data = _safe_dict(safe_payload.get("tactics_state"))
        fin_data = _safe_dict(safe_payload.get("strategy_state"))
        territory_data = _safe_dict(safe_payload.get("territory_state"))

        # Extraer parámetros de resolución visual
        focus_node_id = safe_payload.get("focus_node_id")
        zoom_level_raw = safe_payload.get("zoom_level")
        zoom_level = _safe_int(zoom_level_raw, 0) if zoom_level_raw is not None else None

        # Fibración Localizada: si existe un focus_node_id, el análisis se restringe
        # ESTRICTAMENTE a los datos del sub-grafo invocando conceptualmente el Mapa de Restricción.
        # El fallback a métricas globales se extirpa para preservar el axioma
        # de restricción de la Teoría de Haces Celulares.
        if focus_node_id:
            # Invocar el RestrictionMap desde el payload derivado del SheafCohomologyOrchestrator
            local_topo = topo_data.get("localized_metrics", {}).get(focus_node_id)
            local_fin = fin_data.get("localized_metrics", {}).get(focus_node_id)

            # Si el orquestador espectral falló durante la restricción (debido a falta de aristas,
            # β0>1 generalizado en el subgrafo, etc.), los datos locales no existirán o estarán marcados.
            SheafCohomologyOrchestrator.validate_local_restriction(
                focus_node_id, local_topo, local_fin
            )

            active_topo_data = local_topo
            active_fin_data = local_fin
        else:
            active_topo_data = topo_data
            active_fin_data = fin_data

        # Determinar presencia de territorio: dict no vacío
        territory_present = bool(territory_data)

        # Extraer fricciones solo si hay territorio presente
        if territory_present:
            logistics_friction = _clamp(
                _safe_float(territory_data.get("logistics_friction"), 1.0),
                _FRICTION_CLAMP_LOW,
                _FRICTION_CLAMP_HIGH,
            )
            social_friction = _clamp(
                _safe_float(territory_data.get("social_friction"), 1.0),
                _FRICTION_CLAMP_LOW,
                _FRICTION_CLAMP_HIGH,
            )
            climate_entropy = _clamp(
                _safe_float(territory_data.get("climate_entropy"), 1.0),
                _FRICTION_CLAMP_LOW,
                _FRICTION_CLAMP_HIGH,
            )
        else:
            logistics_friction = 1.0
            social_friction = 1.0
            climate_entropy = 1.0

        return cls(
            psi=_extract_topological_stability(active_topo_data),
            n_nodes=max(1, _safe_int(active_topo_data.get("n_nodes"), 1)),
            n_edges=max(1, _safe_int(active_topo_data.get("n_edges"), 1)),
            cycle_count=max(0, _safe_int(active_topo_data.get("cycle_count"), 0)),
            isolated_count=max(0, _safe_int(active_topo_data.get("isolated_count"), 0)),
            stressed_count=max(0, _safe_int(active_topo_data.get("stressed_count"), 0)),
            roi=_extract_profitability_index(active_fin_data),
            logistics_friction=logistics_friction,
            social_friction=social_friction,
            climate_entropy=climate_entropy,
            territory_present=territory_present,
            focus_node_id=focus_node_id,
            zoom_level=zoom_level,
            topo_data=active_topo_data,
            fin_data=active_fin_data,
            territory_data=territory_data,
        )


@dataclass(frozen=True)
class OmegaMetrics:
    """Magnitudes cuantitativas derivadas del manifold.

    Todas las magnitudes son floats finitos.
    Las que representan factores multiplicativos son ≥ 0.
    """

    fragility_norm: float        # Fragilidad normalizada a [0, 1]
    roi_norm: float              # ROI normalizado a [0, 1]
    misalignment: float          # |fragility_norm - roi_norm| ∈ [0, 1]
    gravity_coupling: float      # ∈ [1 - tanh(0.5), 1 + tanh(0.5)] ≈ [0.54, 1.46]

    internal_tension: float      # ≥ 0
    external_friction: float     # ≥ 1.0
    anomaly_pressure: float      # ≥ 1.0
    combinatorial_scale: float   # ≥ 1.0
    friction_scale: float        # ≥ 1.0
    improbability_lever: float   # ∈ [1, 4]

    base_stress: float           # ≥ 0
    fragility_penalty: float     # ∈ [1.0, 1 + _FRAGILITY_PENALTY_MAX_DELTA]
    total_stress: float          # ≥ 0
    adjusted_stress: float       # ≥ 0


@dataclass(frozen=True)
class OmegaDiagnostics:
    """Diagnósticos interpretables para auditoría humana y trazabilidad.

    Cada campo de status es una etiqueta semántica derivada de los umbrales
    definidos en las constantes del módulo.
    """

    topology_status: str
    financial_status: str
    territory_status: str
    stress_status: str

    dominant_risk_axis: str
    risk_contribution_breakdown: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    inputs_snapshot: Dict[str, Any] = field(default_factory=dict)
    derived_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa para almacenamiento/transmisión."""
        return asdict(self)


@dataclass(frozen=True)
class OmegaResult:
    """Resultado completo del colapso antes del empaquetado de estado.

    Invariantes:
    - ``verdict`` es un miembro válido de ``VerdictLevel``.
    - ``metrics.adjusted_stress`` es consistente con ``verdict``
      según los umbrales del retículo.
    """

    inputs: OmegaInputs
    metrics: OmegaMetrics
    diagnostics: OmegaDiagnostics
    verdict: VerdictLevel

    def to_payload(self, synaptic_context_toon: str) -> Dict[str, Any]:
        """Convierte el resultado a la estructura de payload esperada aguas abajo.

        Incluye todas las métricas intermedias para máxima auditabilidad.
        """
        payload = {
            "omega_metrics": {
                "topological_stability": round(self.inputs.psi, 6),
                "fragility_norm": round(self.metrics.fragility_norm, 6),
                "roi_norm": round(self.metrics.roi_norm, 6),
                "internal_tension": round(self.metrics.internal_tension, 6),
                "external_friction": round(self.metrics.external_friction, 6),
                "improbability_lever": round(self.metrics.improbability_lever, 6),
                "base_stress": round(self.metrics.base_stress, 6),
                "total_stress": round(self.metrics.total_stress, 6),
                "adjusted_stress": round(self.metrics.adjusted_stress, 6),
                "misalignment": round(self.metrics.misalignment, 6),
                "gravity_coupling": round(self.metrics.gravity_coupling, 6),
                "fragility_penalty": round(self.metrics.fragility_penalty, 6),
                "anomaly_pressure": round(self.metrics.anomaly_pressure, 6),
                "combinatorial_scale": round(self.metrics.combinatorial_scale, 6),
                "friction_scale": round(self.metrics.friction_scale, 6),
            },
            "verdict": self.verdict,
            "synaptic_context_toon": synaptic_context_toon,
            "omega_diagnostics": self.diagnostics.to_dict(),
        }

        # Si se aplicó un retracto de resolución, anexar contexto de fibración
        if self.inputs.focus_node_id:
            payload["resolution_retract"] = {
                "focus_node_id": self.inputs.focus_node_id,
                "zoom_level": self.inputs.zoom_level,
                "is_localized_deliberation": True
            }

        return payload


# =============================================================================
# 3. MORFISMO OMEGA
# =============================================================================


class OmegaDeliberationManifold(Morphism):
    """Morfismo de colapso tensorial-operacional.

    Toma un estado que contiene la superposición de TACTICS y STRATEGY
    y lo proyecta a un estado OMEGA mediante cálculo determinista.

    El cálculo interno (``_collapse``) es una **función pura**: no tiene
    efectos secundarios, no muta estado, y su resultado depende
    exclusivamente de los inputs.
    """

    def __init__(self, name: str = "omega_tensor_collapse") -> None:
        super().__init__(name)
        self.synaptic_registry = SynapticRegistry()
        self._domain: FrozenSet[Stratum] = frozenset([Stratum.TACTICS, Stratum.STRATEGY])
        self._codomain: Stratum = Stratum.OMEGA

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._codomain

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Ejecuta el colapso deliberativo sobre un estado categórico.

        Política de errores:
        - Si ``state.is_success`` es False, se retorna sin modificar.
        - Si el colapso falla, se retorna un estado de error con traza.
        - No captura KeyboardInterrupt ni SystemExit.
        """
        if not state.is_success:
            return state

        try:
            payload = state.payload if isinstance(state.payload, dict) else {}
            inputs = OmegaInputs.from_payload(payload)
            result = self._collapse(inputs)

            synaptic_context = self.synaptic_registry.get_active_context(
                max_items=_DEFAULT_MAX_CARTRIDGES,
                max_chars=_DEFAULT_MAX_CHARS,
            )

            collapsed_payload = result.to_payload(
                synaptic_context_toon=synaptic_context
            )

            verdict_name = (
                result.verdict.name
                if hasattr(result.verdict, "name")
                else str(result.verdict)
            )

            logger.info(
                "🌌 Colapso Tensorial completado. "
                "base=%.4f total=%.4f adjusted=%.4f → %s",
                result.metrics.base_stress,
                result.metrics.total_stress,
                result.metrics.adjusted_stress,
                verdict_name,
            )

            return state.with_update(
                new_payload={"omega_state": collapsed_payload},
                new_stratum=self.codomain,
            )

        except (KeyboardInterrupt, SystemExit):
            raise
        except (SheafDegeneracyError, HomologicalInconsistencyError) as e:
            # FASE 3: Saturación del Retículo y Colapso Determinista (Fast-Fail)
            logger.error("🛑 Saturación del Retículo por Singularidad Topológica Local: %s", e)

            # Recuperar payload de nuevo pero sin tratar de validarlo estrictamente,
            # solo para mantener contexto de la petición local
            payload = state.payload if isinstance(state.payload, dict) else {}
            safe_payload = _safe_dict(payload)
            focus_node_id = safe_payload.get("focus_node_id", "DESCONOCIDO")
            zoom_level = _safe_int(safe_payload.get("zoom_level"), 0)

            # Inyección Axiomática de Estrés Infinito
            collapsed_payload = {
                "omega_metrics": {
                    "topological_stability": 0.0,
                    "fragility_norm": 1.0,
                    "roi_norm": 0.0,
                    "internal_tension": math.inf,
                    "external_friction": math.inf,
                    "improbability_lever": math.inf,
                    "base_stress": math.inf,
                    "total_stress": math.inf,
                    "adjusted_stress": math.inf,
                    "misalignment": 1.0,
                    "gravity_coupling": 2.0,
                    "fragility_penalty": 2.5,
                    "anomaly_pressure": math.inf,
                    "combinatorial_scale": math.inf,
                    "friction_scale": math.inf,
                },
                "verdict": VerdictLevel.RECHAZAR,
                "synaptic_context_toon": self.synaptic_registry.get_active_context(),
                "omega_diagnostics": {
                    "topology_status": "fragil",
                    "financial_status": "retorno_debil",
                    "territory_status": "territorio_hostil",
                    "stress_status": "tension_critica",
                    "dominant_risk_axis": "Singularidad Topológica Local",
                    "risk_contribution_breakdown": {
                        "extremes": math.inf,
                        "fragility": math.inf,
                        "internal": math.inf,
                        "territory": math.inf,
                    },
                    "summary": (
                        f"Veredicto=RECHAZAR; "
                        f"Instrucción Axiomática: Singularidad Topológica Local en sub-espacio '{focus_node_id}'. "
                        f"El sub-grafo carece de soporte estructural. "
                        f"Causa: {e}"
                    ),
                    "inputs_snapshot": {},
                    "derived_snapshot": {}
                },
                "resolution_retract": {
                    "focus_node_id": focus_node_id,
                    "zoom_level": zoom_level,
                    "is_localized_deliberation": True
                }
            }

            # Se registra la anomalía en el estado y se avanza forzando el rechazo.
            # Esto aniquila el libre albedrío del LLM y fuerza una justificación forense de rechazo.
            if state.telemetry_context:
                state.telemetry_context.record_error(
                    step_name="deliberation_manifold_collapse",
                    error_message=f"Saturación del Retículo por Singularidad Topológica Local: {e}",
                    error_type="SheafDegeneracyError",
                    severity="CRITICAL",
                    stratum=Stratum.OMEGA,
                    propagate=True
                )

            return state.with_update(
                new_payload={"omega_state": collapsed_payload},
                new_stratum=self.codomain,
            ).with_error(f"SheafDegeneracyError/HomologicalInconsistencyError: {e}")
        except Exception as e:
            logger.exception("Fallo en el Manifold de Deliberación")
            return state.with_error(f"Colapso Tensorial fallido: {e}")

    # -------------------------------------------------------------------------
    # ORQUESTACIÓN FORMAL
    # -------------------------------------------------------------------------

    def _collapse(self, inputs: OmegaInputs) -> OmegaResult:
        """Orquesta el cálculo completo del manifold.

        Función pura: no tiene efectos secundarios.
        """
        metrics = self._compute_metrics(inputs)
        verdict = self._project_to_lattice(metrics.adjusted_stress)
        diagnostics = self._build_diagnostics(inputs, metrics, verdict)

        return OmegaResult(
            inputs=inputs,
            metrics=metrics,
            diagnostics=diagnostics,
            verdict=verdict,
        )

    def _compute_metrics(self, inputs: OmegaInputs) -> OmegaMetrics:
        """Calcula todas las magnitudes continuas del manifold.

        Pipeline de cálculo (DAG de dependencias):

            psi → fragility_norm ─┬─→ misalignment ─┬─→ internal_tension ─┐
            roi → roi_norm ───────┘                  │                     │
                                   fragility_norm ───→ gravity_coupling ───┘
                                                                           │
            territory → external_friction ──────────────┬──────────────────┤
                                                        │                  │
            anomalies → anomaly_pressure ──┐            │                  │
            n_nodes, n_edges → comb_scale ─┼─→ improbability_lever ────────┤
            external_friction → fric_scale ┘                               │
                                                                           ↓
            psi → fragility_penalty ──→ adjusted_stress = base × Λ × P
                                        base = T_int × F_ext
        """
        fragility_norm = self._compute_fragility_normalized(inputs.psi)
        roi_norm = self._normalize_roi(inputs.roi)
        misalignment = self._compute_misalignment(fragility_norm, roi_norm)
        gravity_coupling = self._compute_gravity_coupling(fragility_norm)

        internal_tension = self._compute_internal_tension(
            misalignment=misalignment,
            gravity_coupling=gravity_coupling,
        )

        external_friction = self._compute_external_friction(inputs)

        anomaly_pressure = self._compute_anomaly_pressure(inputs)
        combinatorial_scale = self._compute_combinatorial_scale(inputs)
        friction_scale = self._compute_friction_scale(external_friction)
        improbability_lever = self._compute_improbability_lever(
            anomaly_pressure=anomaly_pressure,
            combinatorial_scale=combinatorial_scale,
            friction_scale=friction_scale,
        )

        base_stress = internal_tension * external_friction
        total_stress = base_stress * improbability_lever
        fragility_penalty = self._compute_fragility_penalty(inputs.psi)

        # Magnetización TOON (Silo B): Acoplamiento Mínimo de Gauge
        # Si el estrato cuenta con Cartuchos TOON cargados, la atención p_mu sufre
        # deflexión determinista: p_mu -> p_mu - q_h A_mu, forzando un alineamiento
        # ortogonal con la física del proyecto (alineación de estrés) y restando
        # grados de libertad al libre albedrío del LLM.
        cartridges_loaded = self.synaptic_registry.cartridge_count
        gauge_deflection = 1.0 + 0.05 * cartridges_loaded  # Cada cartucho introduce fricción determinista

        # Mínima Acción Agéntica: Si la intención insiste en atravesar un socavón lógico
        # (alta anomalía topológica β_1 > 0 modelado en anomaly_pressure) bajo alta fricción,
        # la Energía de Dirichlet acoplada al tensor fuerza la saturación hacia el autoestado supremo.
        if anomaly_pressure > 1.25 and external_friction > 1.5:
            # Acoplamiento del tensor topológico
            adjusted_stress = math.inf
        else:
            adjusted_stress = total_stress * fragility_penalty * gauge_deflection

        return OmegaMetrics(
            fragility_norm=fragility_norm,
            roi_norm=roi_norm,
            misalignment=misalignment,
            gravity_coupling=gravity_coupling,
            internal_tension=internal_tension,
            external_friction=external_friction,
            anomaly_pressure=anomaly_pressure,
            combinatorial_scale=combinatorial_scale,
            friction_scale=friction_scale,
            improbability_lever=improbability_lever,
            base_stress=base_stress,
            fragility_penalty=fragility_penalty,
            total_stress=total_stress,
            adjusted_stress=adjusted_stress,
        )

    # -------------------------------------------------------------------------
    # MOTORES MATEMÁTICOS INTERNOS
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_fragility_normalized(psi: float) -> float:
        """Fragilidad estructural normalizada a [0, 1].

        Transformación:
            raw = log₂(1 + 1/ψ)    — monotónicamente decreciente en ψ
            norm = clamp(raw / log₂(1 + 1/ε), 0, 1)

        Propiedades:
        - ψ → 0⁺   ⟹ fragility → 1.0 (máxima fragilidad)
        - ψ = 1.0   ⟹ fragility = log₂(2) / factor ≈ 0.059
        - ψ → ∞     ⟹ fragility → 0.0

        La normalización usa log₂ en lugar de ln para mejor separación
        en el rango operativo [0.05, 5.0].
        """
        safe_psi = max(psi, _EPSILON)
        raw = math.log2(1.0 + 1.0 / safe_psi)
        # Factor de normalización: máximo teórico cuando psi = _PSI_CLAMP_LOW
        max_raw = math.log2(1.0 + 1.0 / _PSI_CLAMP_LOW)
        return _clamp(raw / max_raw, 0.0, 1.0)

    @staticmethod
    def _normalize_roi(roi: float) -> float:
        """Normaliza ROI al rango [0, 1] para comparabilidad con fragilidad.

        Transformación lineal:
            roi_norm = clamp(roi / _ROI_CLAMP_HIGH, 0, 1)

        Propiedad: ROI = 1.0 (neutral) → roi_norm = 0.2
        """
        return _clamp(roi / _ROI_CLAMP_HIGH, 0.0, 1.0)

    @staticmethod
    def _compute_misalignment(fragility_norm: float, roi_norm: float) -> float:
        """Desalineación entre fragilidad estructural y expectativa financiera.

        Ambos operandos están normalizados a [0, 1], por lo que
        misalignment ∈ [0, 1] sin necesidad de re-escalado.

        Interpretación:
        - misalignment ≈ 0: estructura y finanzas son coherentes.
        - misalignment ≈ 1: máxima disonancia (e.g., estructura frágil
          con expectativa de ROI alto, o viceversa).
        """
        return abs(fragility_norm - roi_norm)

    @staticmethod
    def _compute_gravity_coupling(fragility_norm: float) -> float:
        """Amplificación suave de la tensión por fragilidad.

        Usa tanh como sigmoide suave centrada en ``_GRAVITY_INFLECTION``:
            coupling = 1 + tanh(fragility_norm - inflection)

        Rango:
        - fragility_norm = 0 → coupling ≈ 1 - tanh(0.5) ≈ 0.54
        - fragility_norm = inflection → coupling = 1.0
        - fragility_norm = 1 → coupling ≈ 1 + tanh(0.5) ≈ 1.46

        La función es C∞ y monotónicamente creciente.
        """
        return 1.0 + math.tanh(fragility_norm - _GRAVITY_INFLECTION)

    @staticmethod
    def _compute_internal_tension(
        misalignment: float, gravity_coupling: float
    ) -> float:
        """Tensión interna efectiva.

        T_int = max(0, misalignment × gravity_coupling)

        El max(0, ...) es defensivo; con los rangos actuales siempre es ≥ 0.
        """
        return max(0.0, misalignment * gravity_coupling)

    @staticmethod
    def _compute_external_friction(inputs: OmegaInputs) -> float:
        """Métrica riemanniana simplificada del territorio.

        Si no hay datos territoriales, retorna 1.0 (neutral multiplicativo).

        La fricción es una media ponderada de las tres dimensiones
        territoriales, clamped a [1.0, ∞) para que nunca reduzca
        el estrés base.
        """
        if not inputs.territory_present:
            return 1.0

        metric = (
            inputs.logistics_friction * _FRICTION_WEIGHT_LOGISTICS
            + inputs.social_friction * _FRICTION_WEIGHT_SOCIAL
            + inputs.climate_entropy * _FRICTION_WEIGHT_CLIMATE
        )
        return max(1.0, metric)

    @staticmethod
    def _compute_anomaly_pressure(inputs: OmegaInputs) -> float:
        """Presión estructural inducida por anomalías topológicas.

        pressure = 1 + Σ(coeff_i × count_i)

        Siempre ≥ 1.0 (neutral multiplicativo sin anomalías).
        """
        return 1.0 + (
            _ANOMALY_COEFF_CYCLE * inputs.cycle_count
            + _ANOMALY_COEFF_ISOLATED * inputs.isolated_count
            + _ANOMALY_COEFF_STRESSED * inputs.stressed_count
        )

    @staticmethod
    def _compute_combinatorial_scale(inputs: OmegaInputs) -> float:
        """Escala logarítmica del tamaño del espacio combinatorio.

        scale = log₁₀(max(10, n_nodes × n_edges))

        Siempre ≥ 1.0.
        """
        opportunity_space = max(1, inputs.n_nodes) * max(1, inputs.n_edges)
        return math.log10(max(10.0, float(opportunity_space)))

    @staticmethod
    def _compute_friction_scale(external_friction: float) -> float:
        """Escala sublineal del efecto de fricción sobre las colas.

        scale = √(max(1, external_friction))

        La raíz cuadrada atenúa el impacto de fricciones extremas
        para evitar que el territorio domine el veredicto.
        """
        return math.sqrt(max(1.0, external_friction))

    @staticmethod
    def _compute_improbability_lever(
        anomaly_pressure: float,
        combinatorial_scale: float,
        friction_scale: float,
    ) -> float:
        """Palanca de eventos extremos (fat-tail risk amplifier).

        Λ = clamp((comb_scale × fric_scale × anomaly_pressure) / K, 1, 4)

        donde K = _IMPROBABILITY_SCALE_FACTOR.

        El clamp a [1, 4] garantiza que:
        - Sin anomalías ni complejidad, Λ ≈ 1 (no amplifica).
        - En el peor caso, Λ = 4 (cuadruplica el estrés base).
        """
        lever = (
            combinatorial_scale * friction_scale * anomaly_pressure
        ) / _IMPROBABILITY_SCALE_FACTOR
        return _clamp(lever, _IMPROBABILITY_CLAMP_LOW, _IMPROBABILITY_CLAMP_HIGH)

    @staticmethod
    def _compute_fragility_penalty(psi: float) -> float:
        """Penalización estructural por fragilidad.

        Si ψ ≥ 1.0: penalty = 1.0 (sin penalización).
        Si ψ < 1.0: penalty = 1 + min(δ_max, (1 - ψ) × δ_max)
            donde δ_max = _FRAGILITY_PENALTY_MAX_DELTA.

        Rango: [1.0, 1 + δ_max] = [1.0, 2.5].
        """
        if psi >= 1.0:
            return 1.0
        deficit = 1.0 - max(psi, 0.0)
        return 1.0 + min(_FRAGILITY_PENALTY_MAX_DELTA, deficit * _FRAGILITY_PENALTY_MAX_DELTA)

    @staticmethod
    def _project_to_lattice(adjusted_stress: float) -> VerdictLevel:
        """Proyecta el estrés ajustado al retículo discreto de veredictos.

        Mapeo:
            σ* < 0.75  →  VIABLE
            σ* < 1.75  →  CONDICIONAL
            σ* < 3.00  →  PRECAUCION
            σ* ≥ 3.00  →  RECHAZAR
        """
        if adjusted_stress < _VERDICT_THRESHOLD_VIABLE:
            return VerdictLevel.VIABLE
        if adjusted_stress < _VERDICT_THRESHOLD_CONDICIONAL:
            return VerdictLevel.CONDICIONAL
        if adjusted_stress < _VERDICT_THRESHOLD_PRECAUCION:
            return VerdictLevel.PRECAUCION
        return VerdictLevel.RECHAZAR

    def _build_diagnostics(
        self,
        inputs: OmegaInputs,
        metrics: OmegaMetrics,
        verdict: VerdictLevel,
    ) -> OmegaDiagnostics:
        """Construye un diagnóstico formal y auditable.

        Incluye:
        - Etiquetas semánticas para cada dimensión.
        - Snapshot numérico de inputs y métricas derivadas.
        - Identificación del eje de riesgo dominante con breakdown.
        """
        topology_status = _interpret_psi(inputs.psi)
        financial_status = _interpret_roi(inputs.roi)
        territory_status = _interpret_friction(metrics.external_friction)
        stress_status = _interpret_stress(metrics.adjusted_stress)

        dominant_axis, contributions = self._identify_dominant_risk_axis(metrics)

        verdict_name = (
            verdict.name if hasattr(verdict, "name") else str(verdict)
        )

        summary = (
            f"Veredicto={verdict_name}; "
            f"topologia={topology_status}; "
            f"finanzas={financial_status}; "
            f"territorio={territory_status}; "
            f"estres={stress_status}; "
            f"eje_dominante={dominant_axis}"
        )

        inputs_snapshot = {
            "psi": round(inputs.psi, 6),
            "roi": round(inputs.roi, 6),
            "n_nodes": inputs.n_nodes,
            "n_edges": inputs.n_edges,
            "cycle_count": inputs.cycle_count,
            "isolated_count": inputs.isolated_count,
            "stressed_count": inputs.stressed_count,
            "territory_present": inputs.territory_present,
            "logistics_friction": round(inputs.logistics_friction, 6),
            "social_friction": round(inputs.social_friction, 6),
            "climate_entropy": round(inputs.climate_entropy, 6),
        }

        derived_snapshot = {
            "fragility_norm": round(metrics.fragility_norm, 6),
            "roi_norm": round(metrics.roi_norm, 6),
            "misalignment": round(metrics.misalignment, 6),
            "gravity_coupling": round(metrics.gravity_coupling, 6),
            "internal_tension": round(metrics.internal_tension, 6),
            "external_friction": round(metrics.external_friction, 6),
            "anomaly_pressure": round(metrics.anomaly_pressure, 6),
            "combinatorial_scale": round(metrics.combinatorial_scale, 6),
            "friction_scale": round(metrics.friction_scale, 6),
            "improbability_lever": round(metrics.improbability_lever, 6),
            "base_stress": round(metrics.base_stress, 6),
            "fragility_penalty": round(metrics.fragility_penalty, 6),
            "total_stress": round(metrics.total_stress, 6),
            "adjusted_stress": round(metrics.adjusted_stress, 6),
        }

        return OmegaDiagnostics(
            topology_status=topology_status,
            financial_status=financial_status,
            territory_status=territory_status,
            stress_status=stress_status,
            dominant_risk_axis=dominant_axis,
            risk_contribution_breakdown=contributions,
            summary=summary,
            inputs_snapshot=inputs_snapshot,
            derived_snapshot=derived_snapshot,
        )

    @staticmethod
    def _identify_dominant_risk_axis(
        metrics: OmegaMetrics,
    ) -> Tuple[str, Dict[str, float]]:
        """Identifica el eje dominante del riesgo con breakdown numérico.

        Compara contribuciones de cada factor al estrés total:
        - internal:  tensión interna (misalignment × coupling)
        - territory: exceso de fricción territorial sobre el neutro
        - extremes:  exceso del lever de improbabilidad sobre el neutro
        - fragility: exceso de penalización por fragilidad sobre el neutro

        Retorna:
            Tupla (nombre_eje_dominante, {eje: contribución, ...})

        Si todas las contribuciones son ≤ 0, retorna "balanced".
        En caso de empate, se usa orden lexicográfico para determinismo.
        """
        contributions = {
            "extremes": round(metrics.improbability_lever - 1.0, 6),
            "fragility": round(metrics.fragility_penalty - 1.0, 6),
            "internal": round(metrics.internal_tension, 6),
            "territory": round(metrics.external_friction - 1.0, 6),
        }

        # Ordenar por valor descendente, luego por nombre ascendente (determinismo)
        sorted_items = sorted(
            contributions.items(),
            key=lambda item: (-item[1], item[0]),
        )

        dominant_name, dominant_value = sorted_items[0]
        if dominant_value <= 0:
            return "balanced", contributions
        return dominant_name, contributions


# =============================================================================
# 4. HELPERS NUMÉRICOS Y SEMÁNTICOS
# =============================================================================


def _safe_dict(value: Any) -> Dict[str, Any]:
    """Convierte valor a dict; retorna dict vacío si no es dict."""
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float) -> float:
    """Convierte valor a float finito; retorna ``default`` si imposible.

    Rechaza None, bool, NaN, ±Inf, y tipos no convertibles.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    """Convierte valor a int via float; retorna ``default`` si imposible.

    Rechaza None, bool, NaN, ±Inf, y tipos no convertibles.
    Trunca parte decimal.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return int(number)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    """Restringe ``value`` al intervalo cerrado [low, high].

    Precondiciones: low ≤ high, todos finitos.
    """
    return max(low, min(high, value))


def _extract_topological_stability(topo_data: Dict[str, Any]) -> float:
    """Extrae estabilidad topológica ψ con saneamiento robusto.

    Busca ``pyramid_stability`` en el dict topológico.

    Convención:
    - ψ ≈ 1.0: neutral/saludable.
    - ψ < 1.0: fragilidad estructural.
    - ψ > 1.0: estabilidad superior.

    Siempre retorna valor en [_PSI_CLAMP_LOW, _PSI_CLAMP_HIGH].
    """
    psi = _safe_float(topo_data.get("pyramid_stability"), 1.0)
    return _clamp(psi, _PSI_CLAMP_LOW, _PSI_CLAMP_HIGH)


def _extract_profitability_index(fin_data: Dict[str, Any]) -> float:
    """Extrae ``profitability_index`` con saneamiento robusto.

    Siempre retorna valor en [_ROI_CLAMP_LOW, _ROI_CLAMP_HIGH].
    """
    roi = _safe_float(fin_data.get("profitability_index"), 1.0)
    return _clamp(roi, _ROI_CLAMP_LOW, _ROI_CLAMP_HIGH)


def _interpret_psi(psi: float) -> str:
    """Clasifica ψ en etiqueta semántica."""
    if psi < _PSI_FRAGILE_THRESHOLD:
        return "fragil"
    if psi < _PSI_ROBUST_THRESHOLD:
        return "estable"
    return "robusto"


def _interpret_roi(roi: float) -> str:
    """Clasifica ROI en etiqueta semántica."""
    if roi < _ROI_WEAK_THRESHOLD:
        return "retorno_debil"
    if roi < _ROI_MODERATE_THRESHOLD:
        return "retorno_moderado"
    return "retorno_fuerte"


def _interpret_friction(friction: float) -> str:
    """Clasifica fricción territorial en etiqueta semántica."""
    if friction < _FRICTION_FAVORABLE_THRESHOLD:
        return "territorio_favorable"
    if friction < _FRICTION_MODERATE_THRESHOLD:
        return "territorio_moderado"
    return "territorio_hostil"


def _interpret_stress(stress: float) -> str:
    """Clasifica estrés ajustado en etiqueta semántica."""
    if stress < _STRESS_LOW_THRESHOLD:
        return "tension_baja"
    if stress < _STRESS_MODERATE_THRESHOLD:
        return "tension_moderada"
    if stress < _STRESS_HIGH_THRESHOLD:
        return "tension_alta"
    return "tension_critica"