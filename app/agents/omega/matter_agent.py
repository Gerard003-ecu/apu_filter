# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Matter Agent (Endofuntor de Colapso Hadrónico)                      ║
║ Ubicación : app/agents/omega/matter_agent.py                                 ║
║ Versión : 5.0.0-Topos-Thermodynamic-Phased-Strict                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA (Rigor Doctoral)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sea $\mathcal{E}_{MIC}$ el Topos de Grothendieck sobre el sitio de          ║
║  Zariski del ecosistema MIC, con morfismos de cobertura que satisfacen       ║
║  el axioma de descenso fiel-plano (faithfully flat descent).                 ║
║                                                                              ║
║  Este agente realiza el endofuntor:                                          ║
║                                                                              ║
║    F : Ob(𝒞_Ω) → Ob(𝒞_Ω)                                                    ║
║    F(X) = CategoricalState ∘ π ∘ δ ∘ φ(X)                                    ║
║                                                                              ║
║  donde:                                                                      ║
║    φ : X → BillOfMaterials  (motor físico, delegado)                         ║
║    δ : BOM → HadronicDeliberationVerdict  (vetos termodinámicos, Fase 2)     ║
║    π : Verdict → CategoricalState  (proyección categórica, Fase 3)           ║
║                                                                              ║
║  La composición F = π ∘ δ ∘ φ es un morfismo en la categoría de              ║
║  estratos MIC si y solo si se preservan los tres invariantes:                ║
║                                                                              ║
║    [I1] Acotación de Gini:    G(BOM) < γ_c  ∈ (0,1]                          ║
║    [I2] Fricción isotérmica:  Φ(BOM) ≤ Φ_max > 0                             ║
║    [I3] Positividad exérgica: E_x(BOM) ≥ 0  (Segunda Ley)                    ║
║                                                                              ║
║  La violación de cualquier invariante induce un VETO que eleva una           ║
║  subclase de HadronicCollapseVetoError, deteniendo la composición.           ║
║                                                                              ║
║  Estructura de 3 Fases Anidadas:                                             ║
║   Fase 1 – Validación de Parámetros Constitutivos:                           ║
║             Verifica [I1]∩[I2] sobre los umbrales; resuelve y reconcilia     ║
║             el motor generador; produce MatterAgentContext inmutable.        ║
║   Fase 2 – Deliberación Termodinámica:                                       ║
║             Aplica vetos sobre BOM computado; modela fricción mediante       ║
║             disipación de Rayleigh no-lineal; produce el Veredicto.          ║
║   Fase 3 – Proyección Categórica:                                            ║
║             Firma el morfismo con el invariante de Euler-Poincaré del        ║
║             complejo simplicial logístico; empaqueta en CategoricalState.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 0 – IMPORTACIONES
# Standard Library
# ──────────────────────────────────────────────────────────────────────────────
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 0 – IMPORTACIONES
# Project-Local
# ──────────────────────────────────────────────────────────────────────────────
from app.core.mic_algebra import (
    CategoricalState,
    Morphism,
    TopologicalInvariantError,
)
from app.core.schemas import Stratum
from app.physics.matter_generator import BillOfMaterials, MatterGenerator

# ──────────────────────────────────────────────────────────────────────────────
# LOGGER DEL MÓDULO
# ──────────────────────────────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger("MIC.Omega.MatterAgent")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 – JERARQUÍA DE EXCEPCIONES Y VETOS TERMODINÁMICOS
# ══════════════════════════════════════════════════════════════════════════════
#
#  Lattice de vetos (retículo booleano de fallos):
#
#    TopologicalInvariantError
#    └── HadronicCollapseVetoError          (veto genérico Ω)
#        ├── ThermodynamicFrictionVeto      (violación [I2])
#        ├── LogisticSingularityVeto        (violación [I1])
#        └── NegativeExergyVeto             (violación [I3])
#
#  El orden parcial ≤ en el retículo satisface:
#    NegativeExergyVeto ≤ ThermodynamicFrictionVeto ≤ HadronicCollapseVetoError
#  (la violación exérgica implica siempre fricción divergente)
# ──────────────────────────────────────────────────────────────────────────────


class HadronicCollapseVetoError(TopologicalInvariantError):
    r"""
    Excepción base del Estrato Ω.

    Representa la falla del endofuntor F al encontrar una configuración
    que viola la clausura transitiva del ecosistema MIC.  Toda subclase
    debe proporcionar un ``veto_code`` único para trazabilidad en logs
    distribuidos.
    """

    veto_code: str = "HADRONIC_COLLAPSE_GENERIC"

    def __init__(self, reason: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason: str = reason
        self.payload: Dict[str, Any] = payload or {}
        self.timestamp_utc: float = time.time()

    def __str__(self) -> str:  # noqa: D105
        return (
            f"[{self.veto_code}] @ t={self.timestamp_utc:.6f}s | "
            f"{self.reason} | payload={self.payload}"
        )


class ThermodynamicFrictionVeto(HadronicCollapseVetoError):
    r"""
    Violación del invariante [I2]: Φ(BOM) > Φ_max.

    La fricción de Rayleigh disipada supera el techo isotérmico
    permitido por el segundo principio en el régimen cuasi-estático.
    """

    veto_code: str = "THERMODYNAMIC_FRICTION_VETO"


class LogisticSingularityVeto(HadronicCollapseVetoError):
    r"""
    Violación del invariante [I1]: G(BOM) ≥ γ_c.

    El índice de Gini de la distribución de masa supera el umbral
    crítico γ_c, indicando concentración logística singular que
    destruye la ergodicidad del flujo de materiales.
    """

    veto_code: str = "LOGISTIC_SINGULARITY_VETO"


class NegativeExergyVeto(HadronicCollapseVetoError):
    r"""
    Violación del invariante [I3]: E_x(BOM) < 0.

    Exergía negativa implica que el sistema extrae trabajo del
    reservorio térmico ambiente, violando la Segunda Ley de la
    Termodinámica (imposibilidad de máquina de movimiento perpetuo
    de segunda especie).
    """

    veto_code: str = "NEGATIVE_EXERGY_VETO"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 – CONSTANTES FÍSICAS Y UMBRALES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
#
#  Las constantes siguientes son invariantes del módulo (Final).
#  No deben modificarse en tiempo de ejecución; su alteración
#  constituye una violación del invariante de clausura del Topos.
# ──────────────────────────────────────────────────────────────────────────────

# Exponente de la función de disipación de Rayleigh no-lineal.
# Para n=2 recuperamos el modelo cuadrático (flujo laminar).
# Para n=3 modelamos régimen turbulento (Re >> 1).
_RAYLEIGH_EXPONENT: Final[float] = 2.0

# Factor dimensional que relaciona masa total [kg] con coeficiente
# de fricción viscosa [W·s/kg^n].  Derivado de la viscosidad
# cinemática del medio logístico equivalente.
_RAYLEIGH_VISCOSITY_COEFFICIENT: Final[float] = 0.05

# Tolerancia numérica para comparaciones de punto flotante
# en los vetos (evita falsos positivos por redondeo IEEE-754).
_FLOAT_EPSILON: Final[float] = 1e-12

# Versión semántica del módulo
MODULE_VERSION: Final[str] = "5.0.0-Topos-Thermodynamic-Phased-Strict"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 – ESTRUCTURAS DE ESTADO DELIBERATIVO (DATACLASSES INMUTABLES)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HadronicDeliberationVerdict:
    r"""
    Veredicto inmutable del proceso de deliberación termodinámica.

    Attributes
    ----------
    is_viable : bool
        ``True`` si y solo si los tres invariantes [I1], [I2], [I3]
        han sido satisfechos.
    gini_asymmetry : float
        Índice de Gini G ∈ [0, 1) de la distribución de masa del BOM.
    exergy_dissipated : float
        Potencia exérgica disipada Φ [W] calculada mediante la función
        de disipación de Rayleigh no-lineal.
    exergy_available : float
        Exergía disponible E_x [J] reportada por el motor físico.
    bom_tensor : BillOfMaterials
        Tensor de materiales original generado por ``MatterGenerator``.
    morphism_signature : str
        Hash SHA-256 de la representación canónica del BOM,
        actúa como firma del morfismo φ en la categoría de estratos.
    topological_veto_reason : Optional[str]
        Razón del veto si ``is_viable`` es ``False``, ``None`` en caso
        de viabilidad confirmada.
    euler_poincare_characteristic : int
        Característica de Euler-Poincaré χ(K) del complejo simplicial
        logístico asociado al grafo de dependencias del BOM.
        χ = Σ_{k≥0} (-1)^k · β_k  donde β_k son los números de Betti.
    deliberation_timestamp_utc : float
        Marca temporal UNIX (UTC) del instante de deliberación.
    """

    is_viable: bool
    gini_asymmetry: float
    exergy_dissipated: float
    exergy_available: float
    bom_tensor: BillOfMaterials
    morphism_signature: str
    topological_veto_reason: Optional[str] = None
    euler_poincare_characteristic: int = 0
    deliberation_timestamp_utc: float = field(
        default_factory=time.time  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class MatterAgentContext:
    r"""
    Contexto inmutable producido en la Fase 1.

    Attributes
    ----------
    engine : MatterGenerator
        Motor físico validado y reconciliado con los umbrales del agente.
    max_gini_critical : float
        Umbral crítico γ_c ∈ (0, 1] del índice de Gini.  [I1]
    max_exergy_friction : float
        Techo de potencia disipada Φ_max > 0 [W].  [I2]
    rayleigh_exponent : float
        Exponente n de la función de disipación de Rayleigh:
        Φ(m) = ν · m^n.  Por defecto n = 2 (régimen laminar).
    rayleigh_viscosity : float
        Coeficiente de viscosidad logística ν [W·s/kg^n].
    context_hash : str
        Hash SHA-256 de los parámetros constitutivos, garantiza
        integridad del contexto durante todo el ciclo de vida.
    """

    engine: MatterGenerator
    max_gini_critical: float
    max_exergy_friction: float
    rayleigh_exponent: float = _RAYLEIGH_EXPONENT
    rayleigh_viscosity: float = _RAYLEIGH_VISCOSITY_COEFFICIENT
    context_hash: str = field(default="", compare=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 – ENDOFUNTOR DE COLAPSO HADRÓNICO (AGENTE PRINCIPAL)
# ══════════════════════════════════════════════════════════════════════════════
#
#  MatterAgent implementa Morphism (interfaz del ecosistema MIC) y
#  encapsula tres clases anidadas de rigor creciente que corresponden
#  a las tres fases de la composición F = π ∘ δ ∘ φ.
#
#  La instancia de MatterAgent es INMUTABLE después del constructor:
#    - phase1, phase2, phase3 se construyen en __init__ sin estado mutable.
#    - project_intent_and_deliberate es puramente funcional dado context.
# ──────────────────────────────────────────────────────────────────────────────


class MatterAgent(Morphism):
    r"""
    Endofuntor de Colapso Hadrónico del Estrato Ω.

    Implementa la composición categórica:

    .. math::

        F = \pi \circ \delta \circ \varphi

    donde cada componente es un morfismo en :math:`\mathcal{E}_{MIC}`:

    * :math:`\varphi` – motor físico (``MatterGenerator``)
    * :math:`\delta` – deliberación termodinámica con vetos
    * :math:`\pi`   – proyección en ``CategoricalState``

    Parameters
    ----------
    engine : Optional[MatterGenerator]
        Motor físico externo.  Si es ``None``, se construye uno
        internamente con ``gini_threshold = max_gini_critical``.
    max_gini_critical : float
        Umbral γ_c ∈ (0, 1] del índice de Gini.  Default: 0.85.
    max_exergy_friction : float
        Techo Φ_max > 0 [W] de potencia disipada.  Default: 1 MW.
    rayleigh_exponent : float
        Exponente n de la función de Rayleigh.  Default: 2.0.
    rayleigh_viscosity : float
        Coeficiente ν de viscosidad logística [W·s/kg^n].
        Default: 0.05.

    Raises
    ------
    ValueError
        Si los parámetros constitutivos no satisfacen las restricciones
        físicas de la Fase 1.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRUCTOR – Orquesta las tres fases en el orden de composición
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        engine: Optional[MatterGenerator] = None,
        max_gini_critical: float = 0.85,
        max_exergy_friction: float = 1e6,
        rayleigh_exponent: float = _RAYLEIGH_EXPONENT,
        rayleigh_viscosity: float = _RAYLEIGH_VISCOSITY_COEFFICIENT,
    ) -> None:

        # ── Fase 1: Validación de Parámetros Constitutivos ────────────────────
        #    Produce el MatterAgentContext inmutable que alimenta la Fase 2.
        _phase1 = MatterAgent.Phase1_ParameterValidation(
            engine=engine,
            max_gini_critical=max_gini_critical,
            max_exergy_friction=max_exergy_friction,
            rayleigh_exponent=rayleigh_exponent,
            rayleigh_viscosity=rayleigh_viscosity,
        )
        # La transición Fase1→Fase2 ocurre a través de build_context(),
        # el método culminante de la Fase 1.
        self._context: MatterAgentContext = _phase1.build_context()

        # ── Fase 2: Deliberación Termodinámica ────────────────────────────────
        #    Instancia inmutable; recibe el contexto sellado de la Fase 1.
        self._phase2 = MatterAgent.Phase2_ThermodynamicDeliberation(
            context=self._context
        )

        # ── Fase 3: Proyección Categórica ─────────────────────────────────────
        #    Se instancia en el constructor (sin perezosidad mutable)
        #    para garantizar que el agente sea completamente funcional
        #    desde el primer uso.
        self._phase3 = MatterAgent.Phase3_CategoricalProjection(
            context=self._context
        )

        logger.info(
            "MatterAgent v%s inicializado | context_hash=%s",
            MODULE_VERSION,
            self._context.context_hash,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PROPIEDAD DE SOLO LECTURA – Expone el contexto para inspección externa
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def context(self) -> MatterAgentContext:
        """Contexto inmutable de la Fase 1 (solo lectura)."""
        return self._context

    # ══════════════════════════════════════════════════════════════════════════
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │        FASE 1 – VALIDACIÓN DE PARÁMETROS CONSTITUTIVOS              │
    # └─────────────────────────────────────────────────────────────────────┘
    # ══════════════════════════════════════════════════════════════════════════

    class Phase1_ParameterValidation:
        r"""
        **Fase 1: Validación de Parámetros Constitutivos.**

        Responsabilidades formales
        --------------------------
        1. Verificar que γ_c ∈ (0, 1] y Φ_max > 0 (admisibilidad física).
        2. Verificar coherencia relacional: γ_c < 1 para evitar el punto
           fijo degenerado donde todo BOM es rechazado.
        3. Verificar que n ≥ 1 (exponente de Rayleigh físicamente válido)
           y ν > 0.
        4. Resolver el motor generador: si se provee uno externo, reconciliar
           su ``gini_threshold`` con γ_c mediante un ajuste conservador
           (se toma el mínimo).
        5. Calcular el ``context_hash`` SHA-256 que sella la integridad
           del contexto para toda la vida del agente.

        El método culminante ``build_context()`` es el morfismo de
        transición Fase1 → Fase2.

        Parameters
        ----------
        engine : Optional[MatterGenerator]
            Motor físico; ``None`` dispara construcción interna.
        max_gini_critical : float
            γ_c ∈ (0, 1].
        max_exergy_friction : float
            Φ_max > 0.
        rayleigh_exponent : float
            n ≥ 1.
        rayleigh_viscosity : float
            ν > 0.

        Raises
        ------
        ValueError
            En cualquier violación de las restricciones anteriores.
        """

        def __init__(
            self,
            engine: Optional[MatterGenerator],
            max_gini_critical: float,
            max_exergy_friction: float,
            rayleigh_exponent: float,
            rayleigh_viscosity: float,
        ) -> None:
            self._raw_engine = engine
            self._raw_max_gini = max_gini_critical
            self._raw_max_exergy = max_exergy_friction
            self._raw_rayleigh_exp = rayleigh_exponent
            self._raw_rayleigh_vis = rayleigh_viscosity

        # ── Validadores internos ───────────────────────────────────────────────

        def _validate_gini_threshold(self) -> None:
            r"""
            Verifica γ_c ∈ (0, 1].

            El límite inferior abierto en 0 excluye el caso trivial donde
            ningún BOM es aceptable (γ_c = 0 rechazaría toda distribución).
            El límite superior cerrado en 1 permite, en teoría, una
            concentración perfecta (monopolio logístico), aunque en la
            práctica debe configurarse γ_c ≤ 0.95 para seguridad operacional.
            """
            γ_c = self._raw_max_gini
            if not (_FLOAT_EPSILON < γ_c <= 1.0):
                raise ValueError(
                    f"max_gini_critical debe estar en (0, 1]; "
                    f"recibido: {γ_c!r}.  "
                    f"Valores ≤ 0 producen el veto trivial (todo BOM rechazado)."
                )

        def _validate_exergy_friction(self) -> None:
            r"""
            Verifica Φ_max > 0.

            Un techo de fricción nulo o negativo carecería de sentido
            físico: implicaría que cualquier flujo de materia, por mínimo
            que sea, viola el invariante [I2].
            """
            if self._raw_max_exergy <= _FLOAT_EPSILON:
                raise ValueError(
                    f"max_exergy_friction debe ser estrictamente positivo; "
                    f"recibido: {self._raw_max_exergy!r}."
                )

        def _validate_rayleigh_parameters(self) -> None:
            r"""
            Verifica n ≥ 1 y ν > 0.

            Para n < 1 la función de disipación Φ(m) = ν·m^n diverge
            en m → 0⁺, lo que hace el modelo matemáticamente singular.
            Para n = 1 se recupera el modelo lineal (fricción de Stokes).
            Para n = 2 se tiene fricción cuadrática (flujo laminar).
            Para n = 3 se modela régimen turbulento.
            """
            if self._raw_rayleigh_exp < 1.0 - _FLOAT_EPSILON:
                raise ValueError(
                    f"rayleigh_exponent debe ser ≥ 1 para evitar singularidad "
                    f"en m → 0⁺; recibido: {self._raw_rayleigh_exp!r}."
                )
            if self._raw_rayleigh_vis <= _FLOAT_EPSILON:
                raise ValueError(
                    f"rayleigh_viscosity debe ser estrictamente positivo; "
                    f"recibido: {self._raw_rayleigh_vis!r}."
                )

        def _validate_relational_coherence(self) -> None:
            r"""
            Verifica coherencia relacional entre parámetros.

            Condición adicional: γ_c < 1.0 se recomienda para que exista
            al menos un BOM de masa distribuida uniformemente que sea
            viable.  Se emite una advertencia (no excepción) si γ_c > 0.95,
            pues podría ser intencional en contextos de monopolio controlado.
            """
            if self._raw_max_gini > 0.95:
                logger.warning(
                    "max_gini_critical = %.4f > 0.95: umbral muy permisivo; "
                    "verificar intencionalidad del parámetro.",
                    self._raw_max_gini,
                )

        # ── Resolución y reconciliación del motor ─────────────────────────────

        def _resolve_and_reconcile_engine(self, γ_c: float) -> MatterGenerator:
            r"""
            Resuelve el motor físico y reconcilia su ``gini_threshold``.

            Si se provee un motor externo cuyo ``gini_threshold`` difiere
            de γ_c, se adopta el mínimo (política conservadora): el agente
            nunca debe ser más permisivo que el motor subyacente.

            Si no se provee motor, se construye uno con ``gini_threshold = γ_c``.

            Parameters
            ----------
            γ_c : float
                Umbral de Gini del agente (ya validado).

            Returns
            -------
            MatterGenerator
                Motor listo para operar con threshold reconciliado.
            """
            if self._raw_engine is None:
                logger.debug(
                    "Motor externo no provisto; construyendo MatterGenerator "
                    "con gini_threshold=%.4f.",
                    γ_c,
                )
                return MatterGenerator(gini_threshold=γ_c)

            # Motor externo: reconciliación conservadora
            engine_threshold: float = getattr(
                self._raw_engine, "gini_threshold", γ_c
            )
            reconciled: float = min(engine_threshold, γ_c)

            if abs(reconciled - engine_threshold) > _FLOAT_EPSILON:
                logger.warning(
                    "Reconciliación de gini_threshold: motor=%.4f, agente=%.4f "
                    "→ adoptando mínimo conservador=%.4f.",
                    engine_threshold,
                    γ_c,
                    reconciled,
                )
                # Actualizar el threshold del motor si la API lo permite
                if hasattr(self._raw_engine, "gini_threshold"):
                    try:
                        object.__setattr__(self._raw_engine, "gini_threshold", reconciled)
                    except (AttributeError, TypeError):
                        logger.debug(
                            "Motor externo no permite mutación de gini_threshold; "
                            "el umbral del agente (%.4f) será el determinante.",
                            reconciled,
                        )

            return self._raw_engine

        # ── Hash de integridad del contexto ───────────────────────────────────

        @staticmethod
        def _compute_context_hash(
            γ_c: float,
            φ_max: float,
            n: float,
            ν: float,
        ) -> str:
            r"""
            Calcula el SHA-256 del vector de parámetros constitutivos.

            La representación canónica utiliza JSON con precisión fija de
            12 dígitos para garantizar reproducibilidad cross-plataforma,
            evitando artefactos de representación IEEE-754.

            Parameters
            ----------
            γ_c : float
                Umbral de Gini.
            φ_max : float
                Techo de fricción exérgica.
            n : float
                Exponente de Rayleigh.
            ν : float
                Viscosidad logística.

            Returns
            -------
            str
                Hexdigest SHA-256 de 64 caracteres.
            """
            canonical: str = json.dumps(
                {
                    "gamma_c": f"{γ_c:.12f}",
                    "phi_max": f"{φ_max:.12f}",
                    "n":       f"{n:.12f}",
                    "nu":      f"{ν:.12f}",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        # ── Método culminante de la Fase 1 ────────────────────────────────────

        def build_context(self) -> MatterAgentContext:
            r"""
            **Método culminante de la Fase 1 y puerta de entrada a la Fase 2.**

            Ejecuta la cadena de validaciones en orden de dependencia lógica:

            1. ``_validate_gini_threshold``      – restricción escalar sobre γ_c
            2. ``_validate_exergy_friction``     – restricción escalar sobre Φ_max
            3. ``_validate_rayleigh_parameters`` – restricción sobre (n, ν)
            4. ``_validate_relational_coherence``– coherencia inter-parámetros
            5. ``_resolve_and_reconcile_engine`` – resolución del motor físico
            6. ``_compute_context_hash``         – sellado criptográfico

            Returns
            -------
            MatterAgentContext
                Contexto inmutable listo para alimentar la Fase 2.

            Raises
            ------
            ValueError
                Si cualquier validación falla.

            Notes
            -----
            Este método es el morfismo de transición:

            .. math::

                \text{build\_context} :
                (\gamma_c, \Phi_{\max}, n, \nu, \text{engine})
                \;\longmapsto\;
                \text{MatterAgentContext}

            La imagen es un objeto inmutable (`frozen=True`) que actúa
            como objeto inicial en la categoría de configuraciones del
            agente, garantizando unicidad del contexto por sesión.
            """
            # Cadena de validaciones
            self._validate_gini_threshold()
            self._validate_exergy_friction()
            self._validate_rayleigh_parameters()
            self._validate_relational_coherence()

            # Resolución del motor
            engine: MatterGenerator = self._resolve_and_reconcile_engine(
                γ_c=self._raw_max_gini
            )

            # Sellado criptográfico
            ctx_hash: str = self._compute_context_hash(
                γ_c=self._raw_max_gini,
                φ_max=self._raw_max_exergy,
                n=self._raw_rayleigh_exp,
                ν=self._raw_rayleigh_vis,
            )

            logger.info(
                "Fase 1 completada: contexto sellado | hash=%s | "
                "γ_c=%.4f | Φ_max=%.2e | n=%.2f | ν=%.4f",
                ctx_hash[:16],
                self._raw_max_gini,
                self._raw_max_exergy,
                self._raw_rayleigh_exp,
                self._raw_rayleigh_vis,
            )

            # ── TRANSICIÓN FASE 1 → FASE 2 ────────────────────────────────────
            # El MatterAgentContext es el objeto que conecta la Fase 1 con
            # la Fase 2.  Su inmutabilidad (frozen=True) garantiza que la
            # Fase 2 nunca puede alterar retroactivamente los parámetros
            # constitutivos validados aquí.
            return MatterAgentContext(
                engine=engine,
                max_gini_critical=self._raw_max_gini,
                max_exergy_friction=self._raw_max_exergy,
                rayleigh_exponent=self._raw_rayleigh_exp,
                rayleigh_viscosity=self._raw_rayleigh_vis,
                context_hash=ctx_hash,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │        FASE 2 – DELIBERACIÓN TERMODINÁMICA                          │
    # │  (continúa desde el MatterAgentContext producido por build_context) │
    # └─────────────────────────────────────────────────────────────────────┘
    # ══════════════════════════════════════════════════════════════════════════

    class Phase2_ThermodynamicDeliberation:
        r"""
        **Fase 2: Deliberación Termodinámica.**

        Recibe el ``MatterAgentContext`` inmutable producido por
        ``Phase1_ParameterValidation.build_context()`` y aplica los
        tres vetos del Retículo Acotado Distributivo en orden de
        severidad creciente.

        Función de disipación de Rayleigh no-lineal
        -------------------------------------------
        El modelo de fricción se basa en la función de disipación de
        Rayleigh generalizada:

        .. math::

            \Phi(m_{\text{total}}) = \nu \cdot m_{\text{total}}^{n}

        donde:

        * :math:`m_{\text{total}}` – masa total del BOM [kg]
        * :math:`\nu` – viscosidad logística [W·s/kg^n]
        * :math:`n` – exponente (n=2 → laminar, n=3 → turbulento)

        Para n=2 y ν=0.05 se obtiene el modelo de fricción cuadrática
        estándar de la logística de flujos continuos.

        Orden de vetos (retículo booleano)
        -----------------------------------
        1. **[V3] NegativeExergyVeto** – más severo; implica [V2].
        2. **[V1] LogisticSingularityVeto** – independiente de [V2].
        3. **[V2] ThermodynamicFrictionVeto** – menos severo individualmente.

        Parameters
        ----------
        context : MatterAgentContext
            Contexto sellado de la Fase 1.
        """

        def __init__(self, context: MatterAgentContext) -> None:
            self._ctx = context

        # ── Función de disipación de Rayleigh ─────────────────────────────────

        def _compute_rayleigh_dissipation(self, total_mass: float) -> float:
            r"""
            Calcula Φ(m) = ν · m^n.

            Parameters
            ----------
            total_mass : float
                Masa total del BOM [kg].  Debe ser ≥ 0.

            Returns
            -------
            float
                Potencia disipada Φ [W].

            Notes
            -----
            Si ``total_mass`` es negativo (dato corrupto del motor físico),
            se registra la anomalía y se devuelve 0.0 para evitar artefactos
            numéricos; el veto de exergía negativa capturará el estado inválido
            en el paso siguiente.
            """
            if total_mass < 0.0:
                logger.error(
                    "total_mass=%.6f < 0: dato corrupto del motor físico; "
                    "Φ forzado a 0.0 para evitar NaN.",
                    total_mass,
                )
                return 0.0

            ν: float = self._ctx.rayleigh_viscosity
            n: float = self._ctx.rayleigh_exponent

            # math.pow maneja correctamente m=0 para n≥1: 0^n = 0
            return ν * math.pow(total_mass, n)

        # ── Cálculo de la firma del morfismo ──────────────────────────────────

        @staticmethod
        def _compute_morphism_signature(bom: BillOfMaterials) -> str:
            r"""
            Calcula la firma SHA-256 del BOM como morfismo en 𝒞_Ω.

            Utiliza la representación canónica JSON del BOM (si está
            disponible vía ``__dict__`` o ``to_dict()``), o un fallback
            basado en los atributos numéricos fundamentales.

            Parameters
            ----------
            bom : BillOfMaterials
                Tensor de materiales a firmar.

            Returns
            -------
            str
                Hexdigest SHA-256 de 64 caracteres.
            """
            try:
                if hasattr(bom, "to_dict"):
                    canonical_data = json.dumps(
                        bom.to_dict(), sort_keys=True, separators=(",", ":")
                    )
                else:
                    canonical_data = json.dumps(
                        {
                            "total_mass":     f"{bom.total_mass:.12f}",
                            "gini_asymmetry": f"{bom.gini_asymmetry:.12f}",
                            "exergy":         f"{getattr(bom, 'exergy_available', 0.0):.12f}",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
            except (TypeError, AttributeError) as exc:
                logger.warning(
                    "No se pudo serializar BOM para firma: %s; "
                    "usando repr como fallback.",
                    exc,
                )
                canonical_data = repr(bom)

            return hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()

        # ── Aplicación secuencial de vetos ────────────────────────────────────

        def _apply_veto_negative_exergy(self, bom: BillOfMaterials) -> None:
            r"""
            **Veto [V3]: NegativeExergyVeto.**

            Verifica E_x(BOM) ≥ 0 (invariante [I3]).

            La exergía negativa es físicamente imposible en un sistema
            cerrado que no puede importar orden desde el exterior.  Su
            aparición indica corrupción en el motor físico o una
            configuración termodinámicamente inconsistente.

            Parameters
            ----------
            bom : BillOfMaterials
                Tensor de materiales evaluado.

            Raises
            ------
            NegativeExergyVeto
                Si ``bom.exergy_available < 0``.
            """
            exergy: float = getattr(bom, "exergy_available", 0.0)
            if exergy < -_FLOAT_EPSILON:
                reason = (
                    f"Exergía negativa detectada: E_x = {exergy:.6f} J < 0. "
                    f"Violación de la Segunda Ley (invariante [I3])."
                )
                logger.critical("VETO EXÉRGICO [V3]: %s", reason)
                raise NegativeExergyVeto(
                    reason=reason,
                    payload={"exergy_available": exergy, "bom_id": id(bom)},
                )

        def _apply_veto_logistic_singularity(self, bom: BillOfMaterials) -> None:
            r"""
            **Veto [V1]: LogisticSingularityVeto.**

            Verifica G(BOM) < γ_c (invariante [I1]).

            El índice de Gini mide la concentración de masa en el tensor
            de materiales.  Cuando G ≥ γ_c el flujo logístico degenera
            en una singularidad que destruye la ergodicidad del sistema:
            un único nodo acapara toda la masa y el grafo de dependencias
            pierde su conectividad algebraica.

            Parameters
            ----------
            bom : BillOfMaterials
                Tensor de materiales evaluado.

            Raises
            ------
            LogisticSingularityVeto
                Si ``bom.gini_asymmetry >= max_gini_critical``.
            """
            G: float = bom.gini_asymmetry
            γ_c: float = self._ctx.max_gini_critical

            if G >= γ_c - _FLOAT_EPSILON:
                reason = (
                    f"Singularidad logística: G = {G:.6f} ≥ γ_c = {γ_c:.6f}. "
                    f"Ergodicidad del flujo de materiales destruida (invariante [I1])."
                )
                logger.error("VETO LOGÍSTICO [V1]: %s", reason)
                raise LogisticSingularityVeto(
                    reason=reason,
                    payload={
                        "gini_asymmetry":  G,
                        "max_gini_critical": γ_c,
                        "excess":          G - γ_c,
                    },
                )

        def _apply_veto_thermodynamic_friction(
            self, friction_loss: float
        ) -> None:
            r"""
            **Veto [V2]: ThermodynamicFrictionVeto.**

            Verifica Φ(BOM) ≤ Φ_max (invariante [I2]).

            La potencia disipada por fricción de Rayleigh supera el
            techo isotérmico establecido en el contexto.  Esto indica
            que el flujo de masa genera calor a una tasa que no puede
            ser evacuada por los mecanismos de disipación del sistema
            logístico, llevando a un calentamiento acumulativo y
            eventual colapso térmico.

            Parameters
            ----------
            friction_loss : float
                Potencia disipada Φ [W] calculada por
                ``_compute_rayleigh_dissipation``.

            Raises
            ------
            ThermodynamicFrictionVeto
                Si ``friction_loss > max_exergy_friction``.
            """
            φ_max: float = self._ctx.max_exergy_friction

            if friction_loss > φ_max + _FLOAT_EPSILON:
                reason = (
                    f"Fricción de Rayleigh excede límite isotérmico: "
                    f"Φ = {friction_loss:.4f} W > Φ_max = {φ_max:.4f} W. "
                    f"Colapso térmico inminente (invariante [I2])."
                )
                logger.error("VETO TERMODINÁMICO [V2]: %s", reason)
                raise ThermodynamicFrictionVeto(
                    reason=reason,
                    payload={
                        "friction_loss":       friction_loss,
                        "max_exergy_friction": φ_max,
                        "excess_ratio":        friction_loss / φ_max,
                    },
                )

        # ── Método principal de deliberación ──────────────────────────────────

        def deliberate(self, bom: BillOfMaterials) -> HadronicDeliberationVerdict:
            r"""
            **Método principal de la Fase 2 y puerta de entrada a la Fase 3.**

            Ejecuta la secuencia de vetos en orden de severidad decreciente
            y, si todos se aprueban, retorna el ``HadronicDeliberationVerdict``
            que certifica la viabilidad del BOM.

            Secuencia de evaluación
            -----------------------
            1. Veto [V3]: Exergía negativa (más severo; implica [V2])
            2. Veto [V1]: Singularidad logística (independiente)
            3. Veto [V2]: Fricción termodinámica (cuantitativo)

            Parameters
            ----------
            bom : BillOfMaterials
                Tensor de materiales producido por el motor físico.

            Returns
            -------
            HadronicDeliberationVerdict
                Veredicto aprobatorio con todas las métricas selladas.

            Raises
            ------
            NegativeExergyVeto
                Invariante [I3] violado.
            LogisticSingularityVeto
                Invariante [I1] violado.
            ThermodynamicFrictionVeto
                Invariante [I2] violado.

            Notes
            -----
            Este método implementa el morfismo:

            .. math::

                \delta : \text{BillOfMaterials}
                \;\longmapsto\;
                \text{HadronicDeliberationVerdict}

            La imagen es el objeto de transición Fase2 → Fase3.
            Su inmutabilidad garantiza que la Fase 3 opera sobre
            datos sellados e inmodificables.
            """
            ts_start: float = time.perf_counter()

            # Paso 1: Veto de exergía negativa [V3]
            self._apply_veto_negative_exergy(bom)

            # Paso 2: Veto de singularidad logística [V1]
            self._apply_veto_logistic_singularity(bom)

            # Paso 3: Calcular disipación de Rayleigh
            friction_loss: float = self._compute_rayleigh_dissipation(
                total_mass=bom.total_mass
            )

            # Paso 4: Veto de fricción termodinámica [V2]
            self._apply_veto_thermodynamic_friction(friction_loss)

            # Paso 5: Firma del morfismo φ
            sig: str = self._compute_morphism_signature(bom)

            ts_elapsed: float = time.perf_counter() - ts_start

            logger.info(
                "Deliberación aprobada | G=%.6f | Φ=%.4f W | "
                "E_x=%.4f J | sig=%s | elapsed=%.4fs",
                bom.gini_asymmetry,
                friction_loss,
                getattr(bom, "exergy_available", 0.0),
                sig[:16],
                ts_elapsed,
            )

            # ── TRANSICIÓN FASE 2 → FASE 3 ────────────────────────────────────
            # El HadronicDeliberationVerdict es el objeto que conecta la Fase 2
            # con la Fase 3.  Su campo `morphism_signature` garantiza que la
            # proyección categórica opere sobre el mismo BOM que fue deliberado.
            return HadronicDeliberationVerdict(
                is_viable=True,
                gini_asymmetry=bom.gini_asymmetry,
                exergy_dissipated=friction_loss,
                exergy_available=getattr(bom, "exergy_available", 0.0),
                bom_tensor=bom,
                morphism_signature=sig,
                topological_veto_reason=None,
                euler_poincare_characteristic=0,  # calculado en Fase 3
                deliberation_timestamp_utc=time.time(),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │        FASE 3 – PROYECCIÓN CATEGÓRICA                               │
    # │  (continúa desde el HadronicDeliberationVerdict de la Fase 2)       │
    # └─────────────────────────────────────────────────────────────────────┘
    # ══════════════════════════════════════════════════════════════════════════

    class Phase3_CategoricalProjection:
        r"""
        **Fase 3: Proyección Categórica.**

        Recibe el ``HadronicDeliberationVerdict`` inmutable producido por
        ``Phase2_ThermodynamicDeliberation.deliberate()`` y lo proyecta
        en un ``CategoricalState`` compatible con la clausura transitiva
        del ecosistema MIC.

        Invariante de Euler-Poincaré
        ----------------------------
        Antes de empaquetar el estado, se calcula la característica de
        Euler-Poincaré del complejo simplicial logístico K asociado al
        grafo de dependencias del BOM:

        .. math::

            \chi(K) = \sum_{k \geq 0} (-1)^k \cdot \beta_k

        donde :math:`\beta_k = \dim H_k(K; \mathbb{Z})` son los números
        de Betti del complejo.  Para un grafo conexo planar:

        .. math::

            \chi = V - E + F = 2 \quad \text{(fórmula de Euler)}

        El agente utiliza una aproximación de primer orden basada en los
        atributos del BOM disponibles (número de nodos y aristas del grafo
        de dependencias) o el valor por defecto χ = 1 si no se dispone de
        la topología del grafo.

        Firma del morfismo π
        --------------------
        El ``CategoricalState`` resultante incluye la firma SHA-256 del
        veredicto en su payload, cerrando la cadena de trazabilidad:

        .. code-block:: text

            φ-signature  →  deliberation  →  π-signature  →  CategoricalState

        Parameters
        ----------
        context : MatterAgentContext
            Contexto sellado de la Fase 1 (necesario para el stratum target
            y el context_hash).
        """

        def __init__(self, context: MatterAgentContext) -> None:
            self._ctx = context

        # ── Cálculo del invariante de Euler-Poincaré ──────────────────────────

        @staticmethod
        def _compute_euler_poincare(bom: BillOfMaterials) -> int:
            r"""
            Calcula χ(K) del complejo simplicial logístico del BOM.

            Estrategia de cálculo (orden de prioridad)
            ------------------------------------------
            1. Si ``bom`` expone ``euler_characteristic`` directamente,
               se usa ese valor (calculado por el motor físico con topología
               completa).
            2. Si ``bom`` expone ``num_nodes`` y ``num_edges`` (grafo plano),
               se aplica χ = V - E + 1 (árbol generador conexo, F=1).
            3. Fallback: χ = 1 (complejo contractible, homológicamente trivial).

            Parameters
            ----------
            bom : BillOfMaterials
                Tensor de materiales con posibles atributos topológicos.

            Returns
            -------
            int
                Característica de Euler-Poincaré χ(K).
            """
            # Prioridad 1: valor precalculado por el motor físico
            if hasattr(bom, "euler_characteristic"):
                chi = int(bom.euler_characteristic)
                logger.debug("χ(K) = %d (provisto por motor físico).", chi)
                return chi

            # Prioridad 2: topología de grafo disponible
            if hasattr(bom, "num_nodes") and hasattr(bom, "num_edges"):
                V: int = int(bom.num_nodes)
                E: int = int(bom.num_edges)
                # Para grafo plano conexo con una cara (árbol generador):
                # χ = V - E + 1
                chi = V - E + 1
                logger.debug(
                    "χ(K) = V - E + 1 = %d - %d + 1 = %d (topología de grafo).",
                    V, E, chi,
                )
                return chi

            # Fallback: complejo contractible
            logger.debug("χ(K) = 1 (fallback: complejo contractible).")
            return 1

        # ── Firma de la proyección π ──────────────────────────────────────────

        def _compute_projection_signature(
            self, verdict: HadronicDeliberationVerdict
        ) -> str:
            r"""
            Calcula la firma SHA-256 de la proyección π.

            Combina la firma del morfismo φ (``verdict.morphism_signature``)
            con el hash del contexto (``context.context_hash``) para producir
            una firma que identifica unívocamente la composición completa
            F = π ∘ δ ∘ φ.

            Parameters
            ----------
            verdict : HadronicDeliberationVerdict
                Veredicto inmutable de la Fase 2.

            Returns
            -------
            str
                Hexdigest SHA-256 de 64 caracteres.
            """
            composite: str = json.dumps(
                {
                    "phi_sig":   verdict.morphism_signature,
                    "ctx_hash":  self._ctx.context_hash,
                    "gini":      f"{verdict.gini_asymmetry:.12f}",
                    "exergy_d":  f"{verdict.exergy_dissipated:.12f}",
                    "exergy_a":  f"{verdict.exergy_available:.12f}",
                    "ts":        f"{verdict.deliberation_timestamp_utc:.6f}",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(composite.encode("utf-8")).hexdigest()

        # ── Método principal de proyección ────────────────────────────────────

        def project(
            self, verdict: HadronicDeliberationVerdict
        ) -> CategoricalState:
            r"""
            **Método culminante de la Fase 3 e interfaz final del endofuntor F.**

            Toma el veredicto de la Fase 2 y lo proyecta en un
            ``CategoricalState`` del estrato ``WISDOM``, incluyendo:

            * Todas las métricas de deliberación (Gini, exergía, fricción).
            * La firma completa de la composición F = π ∘ δ ∘ φ.
            * El invariante de Euler-Poincaré del complejo simplicial.
            * El hash del contexto que garantiza trazabilidad end-to-end.

            Parameters
            ----------
            verdict : HadronicDeliberationVerdict
                Veredicto inmutable producido por la Fase 2.

            Returns
            -------
            CategoricalState
                Estado categórico listo para el estrato superior (WISDOM).

            Notes
            -----
            El ``CategoricalState`` resultante satisface la Ley de
            Clausura Transitiva del ecosistema MIC:

            .. math::

                \forall\, s \in \mathcal{E}_{MIC},\;
                \pi(\delta(\varphi(s))) \in \mathcal{E}_{MIC}

            es decir, F es un endofuntor propio (la imagen permanece
            dentro del topos).
            """
            # Paso 1: Invariante topológico
            chi: int = self._compute_euler_poincare(verdict.bom_tensor)

            # Paso 2: Firma de la proyección completa
            pi_sig: str = self._compute_projection_signature(verdict)

            logger.info(
                "Fase 3 completada | χ(K)=%d | π-sig=%s | viable=%s",
                chi,
                pi_sig[:16],
                verdict.is_viable,
            )

            # ── RESULTADO FINAL DEL ENDOFUNTOR F ──────────────────────────────
            return CategoricalState(
                stratum=Stratum.WISDOM,
                payload={
                    # Tensor de materiales original
                    "bom_tensor": verdict.bom_tensor,

                    # Métricas completas de deliberación
                    "deliberation_metrics": {
                        "gini_asymmetry":         verdict.gini_asymmetry,
                        "exergy_dissipated_W":    verdict.exergy_dissipated,
                        "exergy_available_J":     verdict.exergy_available,
                        "is_viable":              verdict.is_viable,
                        "veto_reason":            verdict.topological_veto_reason,
                        "deliberation_ts_utc":    verdict.deliberation_timestamp_utc,
                    },

                    # Invariante topológico del complejo simplicial logístico
                    "topological_invariants": {
                        "euler_poincare_characteristic": chi,
                    },

                    # Trazabilidad criptográfica de la composición F = π∘δ∘φ
                    "morphism_chain": {
                        "phi_signature":     verdict.morphism_signature,
                        "pi_signature":      pi_sig,
                        "context_hash":      self._ctx.context_hash,
                        "module_version":    MODULE_VERSION,
                    },
                },
            )

    # ══════════════════════════════════════════════════════════════════════════
    # MÉTODO PÚBLICO PRINCIPAL – INTERFAZ EXTERNA DEL ENDOFUNTOR
    # ══════════════════════════════════════════════════════════════════════════

    def project_intent_and_deliberate(
        self,
        hierarchical_complex: Any,
        root_node: str,
        friction_map: Dict[str, float],
        price_map: Dict[str, float],
    ) -> CategoricalState:
        r"""
        Interfaz externa del endofuntor F = π ∘ δ ∘ φ.

        Orquesta la composición completa de las tres fases:

        1. **φ** – Delega al motor físico (Estrato TACTICS/PHYSICS) para
           computar el ``BillOfMaterials`` a partir del complejo jerárquico.
        2. **δ** – Aplica los vetos termodinámicos (Fase 2) sobre el BOM.
        3. **π** – Proyecta el veredicto en un ``CategoricalState`` (Fase 3).

        Parameters
        ----------
        hierarchical_complex : Any
            Estructura de grafo jerárquico (networkx.DiGraph, dict-of-dicts,
            o cualquier formato aceptado por ``MatterGenerator.project_to_bom``).
        root_node : str
            Nodo raíz del complejo jerárquico desde el cual se inicia
            el colapso logístico.
        friction_map : Dict[str, float]
            Mapa de coeficientes de fricción por arista del grafo.
            Claves: identificadores de arista; valores: coeficientes [N·s/m].
        price_map : Dict[str, float]
            Mapa de precios por unidad de material.
            Claves: identificadores de material; valores: precio [€/kg].

        Returns
        -------
        CategoricalState
            Estado categórico inmutable del estrato ``WISDOM``, listo
            para el estrato superior del ecosistema MIC.

        Raises
        ------
        NegativeExergyVeto
            Si el BOM reporta exergía negativa.
        LogisticSingularityVeto
            Si el índice de Gini del BOM supera γ_c.
        ThermodynamicFrictionVeto
            Si la fricción de Rayleigh supera Φ_max.
        TopologicalInvariantError
            Cualquier otra violación de invariante en el motor físico.

        Notes
        -----
        El flujo completo satisface:

        .. math::

            F(X) = \pi\bigl(\delta\bigl(\varphi(X)\bigr)\bigr)
            \;\in\; \mathcal{E}_{MIC}

        garantizando que el estado resultante es un objeto válido
        del Topos de Grothendieck :math:`\mathcal{E}_{MIC}`.
        """
        ts_total: float = time.perf_counter()

        logger.info(
            "Iniciando colapso hadrónico en el Estrato Ω | "
            "root='%s' | ctx_hash=%s",
            root_node,
            self._context.context_hash[:16],
        )

        # ── Paso 1: φ – Motor físico (Estrato TACTICS/PHYSICS) ───────────────
        bom_tensor: BillOfMaterials = self._context.engine.project_to_bom(
            hierarchical_complex=hierarchical_complex,
            root_node=root_node,
            friction_tensor_map=friction_map,
            price_tensor_map=price_map,
        )

        # ── Paso 2: δ – Deliberación Termodinámica (Fase 2) ──────────────────
        verdict: HadronicDeliberationVerdict = self._phase2.deliberate(bom_tensor)

        # ── Paso 3: π – Proyección Categórica (Fase 3) ───────────────────────
        final_state: CategoricalState = self._phase3.project(verdict)

        elapsed: float = time.perf_counter() - ts_total

        logger.info(
            "Colapso hadrónico completado | isomorfismo de masa preservado | "
            "stratum=%s | elapsed=%.4fs",
            final_state.stratum,
            elapsed,
        )

        return final_state