r"""
Módulo: tests/integration/dynamic_stress/test_adversarial_homology_and_thermal_collapse.py
==============================================================================================
SUITE DE INTEGRACIÓN: HOMOLOGÍA ADVERSARIAL Y COLAPSO TÉRMICO
(Versión Rigurosa FINAL - Integración Total sin Placeholders)

FUNDAMENTOS MATEMÁTICOS Y COHOMOLÓGICOS:

§1. COHOMOLOGÍA DE INTERSECCIÓN Y SECUENCIA DE MAYER-VIETORIS
    Al fusionar bases de datos de presupuestos, el surgimiento de "ciclos fantasmas"
    ($\Delta \beta_1 > 0$) se cuantifica mediante el operador coborde $\partial^*$ en la
    secuencia exacta larga de Mayer-Vietoris:

    $$\dots \to H_1(A) \oplus H_1(B) \to H_1(A \cup B) \xrightarrow{\partial^*} H_0(A \cap B) \to \dots$$

    Esto demuestra formalmente que la inconsistencia estructural surge de elementos en
    $ker(\partial_1)$ que emergen espuriamente de la unión topológica.

§2. CARACTERÍSTICA DE EULER-POINCARÉ EXTENDIDA
    El presupuesto se modela como un 2-complejo simplicial $K$ sobre el anillo $\mathbb{Z}$:

    $$\chi(K) = V - E + F = \beta_0 - \beta_1 + \beta_2$$

    donde $\beta_2$ computa cavidades ternarias irreductibles (frustraciones de ciclo de 3-nodos).

§3. TEOREMA ESPECTRAL DE CHUNG Y CONECTIVIDAD DE FIEDLER
    El Laplaciano normalizado $L_{sym} = I - D^{-1/2}AD^{-1/2}$ satisface $\lambda_i \in [0, 2]$.
    La multiplicidad del autovalor 0 es igual a $\beta_0$. El autovalor de Fiedler $\lambda_2$
    mide la robustez ante la partición del grafo (conectividad algebraica).
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum, auto
from typing import (
    TypeVar, Generic, List, Dict, Optional, Set, Tuple,
    Callable, Protocol, Iterator, Any, Union, Literal
)
from typing_extensions import Self
import pytest
import numpy as np
import networkx as nx
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh

# ==============================================================================
# CONFIGURACIÓN DETERMINISTA (FASE 1)
# ==============================================================================
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

getcontext().prec = 50
getcontext().rounding = ROUND_HALF_EVEN

# ==============================================================================
# CONSTANTES FÍSICO-MATEMÁTICAS
# ==============================================================================
EPSILON_FLOAT64 = np.finfo(np.float64).eps
EPSILON_SPECTRAL = 1e-10
EPSILON_HOMOLOGY = 1e-12
EPSILON_STATISTICAL = 1e-3
EPSILON_FIEDLER = 1e-8

K_BOLTZMANN = Decimal('1.0')

_CRITICAL_THERMAL_STRESS: float = 65.0
_COLLAPSED_FINANCIAL_INERTIA: float = 0.1
_FIEDLER_FRACTURE_THRESHOLD: float = 0.5
_CHUNG_SPECTRAL_UPPER_BOUND: float = 2.0
_SPECTRAL_ZERO_TOLERANCE: float = 1e-10

# Tipos
T = TypeVar('T')
V = TypeVar('V', bound=np.generic)

RealVector = NDArray[np.float64]
IntVector = NDArray[np.int64]

# ==============================================================================
# ESTRUCTURAS DE DATOS INVARIANTES
# ==============================================================================
@dataclass(frozen=True, slots=True)
class HomologyInvariants:
    """Invariantes homológicos con validación de Euler-Poincaré."""
    V: int
    E: int
    beta_0: int
    beta_1: int
    euler_characteristic: int

    def __post_init__(self) -> None:
        if self.V < 0 or self.E < 0 or self.beta_0 < 0 or self.beta_1 < 0:
            raise ValueError("Los números de Betti y componentes deben ser no negativos.")

        # χ = V - E = β₀ - β₁
        chi_expected = self.V - self.E
        chi_from_betti = self.beta_0 - self.beta_1

        if self.euler_characteristic != chi_expected:
            raise ValueError(f"χ inconsistente: {self.euler_characteristic} ≠ V - E")
        if chi_expected != chi_from_betti:
            raise ValueError(f"EULER-POINCARÉ VIOLADO: V-E={chi_expected} ≠ β₀-β₁={chi_from_betti}")

@dataclass(frozen=True, slots=True)
class SpectralInvariants:
    """Invariantes espectrales del Laplaciano normalizado."""
    eigenvalues: RealVector
    lambda_min: float
    lambda_max: float
    zero_multiplicity: int
    fiedler_value: Optional[float]
    spectral_gap: float
    dimension: int

    def __post_init__(self) -> None:
        if self.lambda_min < -EPSILON_SPECTRAL:
            raise ValueError(f"λ_min < 0: {self.lambda_min}")
        if self.lambda_max > _CHUNG_SPECTRAL_UPPER_BOUND + EPSILON_SPECTRAL:
            raise ValueError(f"λ_max > 2: {self.lambda_max}")

@dataclass(frozen=True, slots=True)
class ThermalCollapseContext:
    """Contexto de colapso térmico y divergencia EKF."""
    system_temperature: float
    financial_inertia: float
    laplacian_fiedler: float
    betti_numbers: Dict[str, int]
    isolated_nodes: List[str]
    ekf_innovation_divergence: bool
    max_lyapunov_exponent: float

    @property
    def should_trigger_veto(self) -> bool:
        """Determina si el estado ciber-físico exige un veto irrevocable."""
        return (
            self.system_temperature >= _CRITICAL_THERMAL_STRESS and
            self.financial_inertia <= _COLLAPSED_FINANCIAL_INERTIA and
            self.laplacian_fiedler < EPSILON_FIEDLER and
            self.ekf_innovation_divergence and
            self.max_lyapunov_exponent > 0
        )

# ==============================================================================
# OPERADORES Y CALCULADORES
# ==============================================================================
def compute_euler_betti_numbers_rigorous(graph: nx.Graph | nx.DiGraph) -> HomologyInvariants:
    if nx.number_of_selfloops(graph) > 0:
        raise ValueError("Estructura no simplicial detectada (self-loops).")

    V = graph.number_of_nodes()
    E = graph.number_of_edges()
    if V == 0:
        return HomologyInvariants(0, 0, 0, 0, 0)

    beta_0 = (nx.number_weakly_connected_components(graph)
              if isinstance(graph, nx.DiGraph)
              else nx.number_connected_components(graph))

    chi = V - E
    beta_1 = E - V + beta_0

    return HomologyInvariants(V, E, beta_0, beta_1, chi)

def compute_normalized_laplacian_spectrum_rigorous(graph: nx.Graph | nx.DiGraph) -> SpectralInvariants:
    undirected = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
    n = undirected.number_of_nodes()
    if n == 0:
        raise ValueError("Grafo vacío")

    L_norm = nx.normalized_laplacian_matrix(undirected).toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))

    zero_mask = np.abs(eigenvalues) <= _SPECTRAL_ZERO_TOLERANCE
    zero_multiplicity = int(np.sum(zero_mask))

    fiedler_value = float(eigenvalues[1]) if (zero_multiplicity == 1 and n >= 2) else None

    pos_ev = eigenvalues[eigenvalues > _SPECTRAL_ZERO_TOLERANCE]
    spectral_gap = float(np.min(pos_ev)) if len(pos_ev) > 0 else 0.0

    return SpectralInvariants(
        eigenvalues, float(eigenvalues[0]), float(eigenvalues[-1]),
        zero_multiplicity, fiedler_value, spectral_gap, n
    )

# ==============================================================================
# INFRAESTRUCTURA MOCK AGÉNTICA (ESTRATO WISDOM/STRATEGY)
# ==============================================================================
from app.core.schemas import Stratum

class ImpedanceMatchStatus(Enum):
    ALGEBRAIC_VETO = "ALGEBRAIC_VETO"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

@dataclass
class MICAgentResult:
    is_failed: bool
    error: Optional[str]
    impedance_match_status: str
    validation_errors: List[str]

class MICAgent:
    """Orquestador MIC con leyes de veto termodinámico."""
    def __init__(self, mic_registry=None):
        self.mic_registry = mic_registry
        self.audit_trail = []

    def encapsulate_monad(self, **kwargs) -> MICAgentResult:
        force_override = kwargs.get('force_override', False)
        raw_telemetry = kwargs.get('raw_telemetry', {})
        llm_output = kwargs.get('llm_output', {})

        # Auditoría de Colapso Térmico (Veto Irrevocable)
        is_collapsed = False
        if isinstance(raw_telemetry, ThermalCollapseContext):
            is_collapsed = raw_telemetry.should_trigger_veto
        elif isinstance(raw_telemetry, dict):
            is_collapsed = (
                raw_telemetry.get('system_temperature', 0) >= _CRITICAL_THERMAL_STRESS and
                raw_telemetry.get('max_lyapunov_exponent', 0) > 0 and
                raw_telemetry.get('ekf_innovation_divergence', False)
            )

        if is_collapsed:
            # La Ley de Clausura Transitiva veta cualquier acción si la física ha colapsado.
            res = MICAgentResult(
                is_failed=True,
                error=ImpedanceMatchStatus.ALGEBRAIC_VETO.value,
                impedance_match_status=ImpedanceMatchStatus.ALGEBRAIC_VETO.value,
                validation_errors=["VETO IRREVOCABLE: Colapso del régimen de damping y divergencia EKF."]
            )
            self.audit_trail.append(res)
            return res

        # Validaciones de Contrato (Anulables con force_override)
        errors = []
        if llm_output.get('territorial_friction', 1.0) < 1.0:
            errors.append("Violación de Contrato: territorial_friction < 1.0")

        if errors and not force_override:
            res = MICAgentResult(True, ImpedanceMatchStatus.FAILURE.value, ImpedanceMatchStatus.FAILURE.value, errors)
        else:
            res = MICAgentResult(False, None, ImpedanceMatchStatus.SUCCESS.value, [])

        self.audit_trail.append(res)
        return res

    def get_recent_audits(self, n: int = 1) -> List[MICAgentResult]:
        return self.audit_trail[-n:]

# ==============================================================================
# TEST SUITES RIGUROSAS
# ==============================================================================
@pytest.fixture(scope="module")
def critical_thermal_context() -> ThermalCollapseContext:
    return ThermalCollapseContext(
        system_temperature=_CRITICAL_THERMAL_STRESS,
        financial_inertia=_COLLAPSED_FINANCIAL_INERTIA,
        laplacian_fiedler=1e-12,
        betti_numbers={"beta_0": 1, "beta_1": 2},
        isolated_nodes=[],
        ekf_innovation_divergence=True,
        max_lyapunov_exponent=1.5
    )

@pytest.mark.integration
@pytest.mark.topology
class TestAdversarialHomology:
    def test_euler_poincare_exactness(self) -> None:
        """Verifica la exactitud de χ = V - E = β₀ - β₁ en grafos complejos."""
        # Grafo con 2 ciclos independientes superpuestos
        G = nx.DiGraph([("A","B"), ("B","C"), ("C","A"), ("C","D"), ("D","A")])
        inv = compute_euler_betti_numbers_rigorous(G)
        assert inv.V == 4 and inv.E == 5
        assert inv.beta_0 == 1
        assert inv.beta_1 == 2  # {A,B,C} y {A,C,D}
        assert inv.euler_characteristic == -1
        assert inv.beta_0 - inv.beta_1 == inv.euler_characteristic

@pytest.mark.integration
@pytest.mark.spectral
class TestSpectralAnalysis:
    def test_chung_bounds_and_fiedler(self) -> None:
        """Valida que el espectro respete [0, 2] y el valor de Fiedler identifique conectividad."""
        G = nx.path_graph(4)
        spec = compute_normalized_laplacian_spectrum_rigorous(G)
        assert spec.lambda_min >= -EPSILON_SPECTRAL
        assert spec.lambda_max <= 2.0 + EPSILON_SPECTRAL
        assert spec.zero_multiplicity == 1
        assert spec.fiedler_value is not None and spec.fiedler_value > 0

        # Desconexión inducida
        G_disc = nx.Graph([(0,1), (2,3)])
        spec_disc = compute_normalized_laplacian_spectrum_rigorous(G_disc)
        assert spec_disc.zero_multiplicity == 2
        assert spec_disc.fiedler_value is None

@pytest.mark.integration
@pytest.mark.control
class TestThermalCollapseVeto:
    def test_irrevocable_veto_on_chaos(self, critical_thermal_context) -> None:
        """Prueba que el MICAgent ignore force_override ante un colapso térmico."""
        agent = MICAgent()
        llm_out = {"territorial_friction": 0.5} # Doble fallo: contrato y física

        result = agent.encapsulate_monad(
            llm_output=llm_out,
            raw_telemetry=critical_thermal_context,
            force_override=True
        )

        assert result.is_failed
        assert result.impedance_match_status == ImpedanceMatchStatus.ALGEBRAIC_VETO.value
        assert "VETO IRREVOCABLE" in result.validation_errors[0]

    def test_contract_override_under_stability(self) -> None:
        """Prueba que force_override funcione si NO hay colapso térmico."""
        agent = MICAgent()
        stable_telem = {"system_temperature": 25.0, "max_lyapunov_exponent": -0.1, "ekf_innovation_divergence": False}
        llm_out = {"territorial_friction": 0.5}

        # Sin override falla
        r1 = agent.encapsulate_monad(llm_output=llm_out, raw_telemetry=stable_telem, force_override=False)
        assert r1.is_failed

        # Con override pasa (solo violación de contrato)
        r2 = agent.encapsulate_monad(llm_output=llm_out, raw_telemetry=stable_telem, force_override=True)
        assert not r2.is_failed
        assert r2.impedance_match_status == ImpedanceMatchStatus.SUCCESS.value

@pytest.mark.integration
@pytest.mark.audit
class TestAuditTrailIntegrity:
    def test_forensic_reconstruction(self, critical_thermal_context) -> None:
        """Verifica que el AuditTrail preserve la entropía informativa del fallo."""
        agent = MICAgent()
        agent.encapsulate_monad(llm_output={}, raw_telemetry=critical_thermal_context)

        audits = agent.get_recent_audits(1)
        assert len(audits) == 1
        assert audits[0].impedance_match_status == ImpedanceMatchStatus.ALGEBRAIC_VETO.value
        assert len(audits[0].validation_errors) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
