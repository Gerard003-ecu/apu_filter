# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Antimatter Choke Coil Agent (Custodio del Vacío Cuántico)                      ║
║  Ruta   : app/agents/physics/antimatter_choke_coil_agent.py                              ║
║  Versión: 3.0.0-Fock-Bekenstein-Symplectic-Strict-Granular                               ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  NATURALEZA CIBER-FÍSICA Y ELECTRODINÁMICA CUÁNTICA (Rigor Doctoral):                    ║
║  ────────────────────────────────────────────────────────────────────────────            ║
║  Este endofuntor, denotado como $\mathcal{Z}_{Vacuum}$, gobierna la física del módulo    ║
║  `antimatter_choke_coil.py` en el Estrato Ω. Su mandato axiomático es auditar las        ║
║  aniquilaciones de antimateria exógena ($e^- + e^+ \to 2\gamma$), garantizando que la    ║
║  topología de la Malla Agéntica sobreviva al colapso entrópico de estados erróneos       ║
║  manteniendo invariante la estructura fundamental del Espacio de Fock $\mathcal{F}(\mathcal{H})$. ║
║                                                                                          ║
║  FUNDAMENTOS AXIOMÁTICOS Y RESTRICCIONES CUÁNTICAS:                                      ║
║                                                                                          ║
║  §1. Hermiticidad del Operador de Aniquilación (Teorema Espectral):                      ║
║      Para que las observables cuánticas sean físicamente medibles (espectro real),       ║
║      el operador de colapso $A$ debe ser estrictamente autoadjunto ($A = A^\dagger$).    ║
║      Se verifica la anulación del residuo mediante la norma de Frobenius:                ║
║          $\|A - A^\dagger\|_F \le \varepsilon_{herm}$                                    ║
║      Cualquier desviación induce autovalores imaginarios y detona un                     ║
║      `NonHermitianOperatorError`, purgando el operador del espacio de Hilbert.           ║
║                                                                                          ║
║  §2. Regulación Termodinámica del Límite de Bekenstein (Cota Holográfica):               ║
║      La inyección de entropía $\Delta S$ generada por la radiación de aniquilación no    ║
║      puede exceder la capacidad máxima de información del hipervolumen local de radio R. ║
║      Se impone el límite holográfico de Bekenstein:                                      ║
║          $\Delta S \le \frac{2\pi k_B E R}{\hbar c}$                                     ║
║      Violar esta inecuación crearía una singularidad informacional, detonando de         ║
║      inmediato el `BekensteinLimitViolation`.                                            ║
║                                                                                          ║
║  §3. Certificación Simpléctica Port-Hamiltoniana (Teorema de Liouville):                 ║
║      Tras la colisión, el remanente del grafo logístico debe preservar su volumen en     ║
║      el espacio de fase y su estabilidad asintótica. Se exige la preservación de la      ║
║      matriz simpléctica $\Omega$, antisimetría de interconexión $J$, y disipación $R$:   ║
║          $M^\top \Omega M = \Omega, \quad J = -J^\top, \quad R = R^\top \succeq 0$       ║
║      Sujeto a la inecuación de disipación temporal de energía de Rayleigh:               ║
║          $\dot{H} = -\nabla H^\top R \nabla H \le 0$                                     ║
║      El fallo de estas condiciones gatilla el `SymplecticCollapseError`.                 ║
║                                                                                          ║
║  ARQUITECTURA DE FASES ANIDADAS                                                          ║
║  (Composición Funtorial Estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$):                    ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Fase 1 → Phase1_HermiticityAuditor                                                      ║
║           Audita la autoadjunción del operador de colapso ($A = A^\dagger$).             ║
║           [Retorna: HermiticityAuditData → objeto inicial de Fase 2]                     ║
║                                                                                          ║
║  Fase 2 → Phase2_BekensteinBoundEnforcer                                                 ║
║           Evalúa la entropía radiada contra el límite causal termodinámico.              ║
║           [Retorna: BekensteinBoundData → objeto inicial de Fase 3]                      ║
║                                                                                          ║
║  Fase 3 → Phase3_SymplecticPortHamiltonianCertifier                                      ║
║           Certifica la disipación Port-Hamiltoniana y la invarianza del volumen          ║
║           simpléctico tras la aniquilación de los estados erróneos.                      ║
║           [Retorna: VacuumGovernanceState → objeto final del endofuntor]                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Final, Optional, Tuple, Dict, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass

logger = logging.getLogger("MIC.Omega.VacuumCustodian.Granular")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, CODATA Y LÍMITES CUÁNTICOS
# ════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancias espectrales para A = A†.
_HERMITICITY_TOLERANCE: Final[float] = 1e-12
_SPECTRAL_IMAGINARY_TOLERANCE: Final[float] = 1e-11
_TRACE_IMAGINARY_TOLERANCE: Final[float] = 1e-11

# Tolerancia de preservación de la forma simpléctica ω.
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-10
_DETERMINANT_TOLERANCE: Final[float] = 1e-9
_PFAFFIAN_TOLERANCE: Final[float] = 1e-9

# Tolerancia para J = -Jᵀ.
_ANTISYMMETRY_TOLERANCE: Final[float] = 1e-10

# Tolerancia para R = Rᵀ.
_R_SYMMETRY_TOLERANCE: Final[float] = 1e-10

# Tolerancia de semidefinición positiva R ⪰ 0.
_PSD_EIGENVALUE_TOLERANCE: Final[float] = 1e-12

# Tolerancias para la cota de Bekenstein.
_BEKENSTEIN_ABS_TOLERANCE: Final[float] = 1e-12
_BEKENSTEIN_REL_TOLERANCE: Final[float] = 1e-12
_CAUSALITY_TOLERANCE: Final[float] = 1e-15
_ENTROPY_QUANTUM_TOLERANCE: Final[float] = 1e-30

# Constantes físicas efectivas (SI).
_HBAR_EFF: Final[float] = 1.054_571_817e-34  # J·s (CODATA 2018)
_C_EFF: Final[float] = 299_792_458.0          # m/s (exacto)
_K_B: Final[float] = 1.380_649e-23            # J/K (exacto)
_ELECTRON_MASS_KG: Final[float] = 9.109_383_7015e-31  # kg
_POSITRON_MASS_KG: Final[float] = 9.109_383_7015e-31  # kg (CPT)


# ════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CUÁNTICAS
# ════════════════════════════════════════════════════════════════════════════
class VacuumCustodianError(TopologicalInvariantError):
    """Excepción raíz del Custodio del Vacío Cuántico."""
    pass


class DomainIntegrityViolationError(VacuumCustodianError):
    """Detonada cuando un operador, vector o escalar viola su dominio formal."""
    pass


class NonHermitianOperatorError(VacuumCustodianError):
    r"""Detonada si $\|A - A^\dagger\|_F > \varepsilon$. Los observables dejan de ser reales."""
    pass


class SpectralContaminationError(VacuumCustodianError):
    r"""Detonada si el espectro de $A$ contiene componentes imaginarias no nulas."""
    pass


class BekensteinLimitViolation(VacuumCustodianError):
    r"""Detonada si la aniquilación inyecta más entropía que la cota causal."""
    pass


class CausalityViolationError(VacuumCustodianError):
    r"""Detonada si el radio de contención viola la causalidad relativista."""
    pass


class SymplecticCollapseError(VacuumCustodianError):
    r"""Detonada si se destruye el volumen del espacio de fase o la disipación."""
    pass


class PhaseSpaceTopologyError(VacuumCustodianError):
    r"""Detonada si la dimensión del espacio de fase no es par o es nula."""
    pass


# ════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fock)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralDecompositionData:
    r"""
    Artefacto espectral de Fase 1.
    Certificado de descomposición espectral del operador autoadjunto.
    
    Teorema Espectral: Si $A = A^\dagger$, entonces existe una base
    ortonormal de autovectores con autovalores reales.
    """
    eigenvalues_real: NDArray[np.float64]
    eigenvalues_imaginary_norm: float
    spectral_radius: float
    trace_real: float
    trace_imaginary_norm: float
    condition_number: float
    is_spectrally_clean: bool


@dataclass(frozen=True, slots=True)
class HermiticityAuditData:
    r"""
    Artefacto de Fase 1. Certificado espectral de autoadjunción.
    
    Lema de Validación: $\|A - A^\dagger\|_F = 0 \iff A = A^\dagger$.
    """
    residual_norm: float
    is_hermitian: bool
    operator_dimension: int = 0
    hermiticity_tolerance: float = _HERMITICITY_TOLERANCE
    spectral_imaginary_norm: float = 0.0
    spectral_decomposition: Optional[SpectralDecompositionData] = None


@dataclass(frozen=True, slots=True)
class BekensteinBoundData:
    r"""
    Artefacto de Fase 2. Certificado de cota termodinámica de radiación.
    
    Teorema de Bekenstein: $S \leq \frac{2\pi k_B E R}{\hbar c}$.
    """
    entropy_emitted: float
    bekenstein_bound: float
    is_entropically_safe: bool
    gamma_energy: float = 0.0
    system_radius: float = 0.0
    entropy_tolerance: float = 0.0
    entropy_ratio: float = 0.0
    causal_light_crossing_time: float = 0.0
    information_capacity_bits: float = 0.0


@dataclass(frozen=True, slots=True)
class SymplecticDissipationData:
    r"""
    Artefacto de Fase 3. Certificado de geometría Port-Hamiltoniana.
    
    Teorema de Liouville: El flujo hamiltoniano preserva el volumen
    del espacio de fase: $\det(M) = 1$.
    """
    symplectic_residual: float
    dissipation_rate: float
    is_symplectically_invariant: bool
    symplectic_tolerance: float = _SYMPLECTIC_TOLERANCE
    antisymmetry_residual: float = 0.0
    r_symmetry_residual: float = 0.0
    r_min_eigenvalue: float = 0.0
    r_max_eigenvalue: float = 0.0
    determinant_residual: float = 0.0
    phase_space_dimension: int = 0
    degrees_of_freedom: int = 0


@dataclass(frozen=True, slots=True)
class Phase1HermiticityHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.
    
    Este objeto es la continuación material de la hermiticidad y el prefijo
    obligatorio de la regulación de Bekenstein.
    
    Lema de Continuación: La hermiticidad certificada es condición
    necesaria para la coherencia termodinámica posterior.
    """
    hermiticity_audit: HermiticityAuditData
    operator_dimension: int
    spectral_certificate: Optional[SpectralDecompositionData] = None


@dataclass(frozen=True, slots=True)
class Phase2BekensteinHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.
    
    Este objeto transporta la certificación de hermiticidad y la cota
    termodinámica como prefijo obligatorio de la fase simpléctica.
    
    Lema de Continuación: La entropía acotada garantiza que la
    geometría del espacio de fase no colapsa por sobrecarga informacional.
    """
    phase1_handoff: Phase1HermiticityHandoff
    bekenstein_audit: BekensteinBoundData


@dataclass(frozen=True, slots=True)
class VacuumGovernanceState:
    r"""
    Objeto final del endofuntor $\mathcal{Z}_{Vacuum}$.
    
    Teorema de Corrección: Si todas las fases certifican éxito,
    el vacío cuántico está topológicamente protegido.
    """
    hermiticity_audit: HermiticityAuditData
    bekenstein_audit: BekensteinBoundData
    symplectic_audit: SymplecticDissipationData
    is_epistemologically_valid: bool
    governance_metadata: Dict[str, Any] = field(default_factory=dict)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: HERMITICIDAD DEL OPERADOR DE ANIQUILACIÓN                         ║
# ║   Exige $A = A^\dagger \Rightarrow \|A - A^\dagger\|_F \le \varepsilon$.  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_HermiticityAuditor:
    r"""
    Garantiza que el operador de colapso en el espacio de Fock preserve un
    espectro real, impidiendo que variables de estado imaginarias corrompan
    la inferencia.
    
    Fundamento Teórico:
    ────────────────────
    Teorema Espectral: Un operador lineal $A$ en un espacio de Hilbert
    complejo es autoadjunto si y solo si todos sus autovalores son reales
    y existe una base ortonormal de autovectores.
    
    Corolario de Medibilidad: Las observables físicas deben ser
    autoadjuntas para garantizar resultados de medición reales.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Tolerancia adaptativa con análisis de condición numérica
    # ─────────────────────────────────────────────────────────────────────────
    def _adaptive_tolerance(
        self,
        base_tolerance: float,
        reference: Any,
        condition_amplification: bool = False,
    ) -> float:
        r"""
        Construye una tolerancia numéricamente consciente:
            $\text{tol} = \max(\text{tol\_base}, \kappa \cdot \varepsilon_{\text{máquina}} \cdot \text{tamaño} \cdot \text{escala})$
        
        Si condition_amplification=True, amplifica por número de condición.
        
        Lema de Estabilidad: La tolerancia debe escalar con la magnitud
        del objeto para evitar falsos positivos en regímenes de alta energía.
        """
        if isinstance(reference, np.ndarray):
            size = max(1, int(reference.size))
            if reference.size == 0:
                scale = 1.0
                condition_number = 1.0
            else:
                flat = reference.ravel()
                scale = max(1.0, float(la.norm(flat, ord=np.inf)))
                if condition_amplification and reference.ndim == 2:
                    try:
                        svals = la.svd(reference, compute_uv=False)
                        if svals.size > 0 and svals[-1] > _MACHINE_EPSILON:
                            condition_number = float(svals[0] / svals[-1])
                        else:
                            condition_number = 1.0 / _MACHINE_EPSILON
                    except Exception:
                        condition_number = 1.0
                else:
                    condition_number = 1.0
        else:
            size = 1
            condition_number = 1.0
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        base_component = float(base_tolerance)
        machine_component = 10.0 * _MACHINE_EPSILON * size * scale
        
        if condition_amplification:
            machine_component *= min(condition_number, 1e6)

        return max(base_component, machine_component)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción de escalares finitos con rechazo categórico de booleanos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_scalar(
        self,
        name: str,
        value: Any,
        allow_negative: bool = True,
    ) -> float:
        r"""
        Materializa un escalar float64 finito, rechazando booleanos.
        
        Axioma de Dominio: Todo escalar físico debe pertenecer a $\mathbb{R}$
        y ser finito. Los booleanos pertenecen a $\mathbb{B}_2$, no a $\mathbb{R}$.
        
        Parámetros:
        ───────────
        name : str
            Identificador del escalar para trazabilidad.
        value : Any
            Valor a coerccionar.
        allow_negative : bool
            Si False, rechaza valores negativos.
        """
        if isinstance(value, (bool, np.bool_)):
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no puede ser booleano. "
                f"Los booleanos pertenecen al topos $\mathbb{B}_2$, no a $\mathbb{R}$."
            )
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no puede materializarse como float."
            ) from exc
        
        if not np.isfinite(scalar):
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no es finito. "
                f"Valor: {scalar}. Se requiere $x \in \mathbb{R}$."
            )
        
        if not allow_negative and scalar < 0.0:
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' debe ser no negativo. "
                f"Valor observado: {scalar:.6e}."
            )
        
        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción de matrices finitas con validación estructural
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_matrix(
        self,
        name: str,
        matrix: Any,
        dtype: Any = np.float64,
        square_required: bool = False,
        min_dimension: int = 1,
    ) -> NDArray[Any]:
        r"""
        Materializa una matriz finita y, si se exige, cuadrada.
        
        Lema de Integridad Matricial: Toda matriz en el espacio de Hilbert
        debe tener componentes finitas y dimensión compatible con el
        espacio de Fock subyacente.
        
        Parámetros:
        ───────────
        name : str
            Identificador de la matriz.
        matrix : Any
            Objeto a convertir en NDArray.
        dtype : Any
            Tipo de dato objetivo.
        square_required : bool
            Si True, exige matriz cuadrada.
        min_dimension : int
            Dimensión mínima requerida.
        """
        try:
            arr = np.asarray(matrix, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' no puede materializarse como NDArray."
            ) from exc
        
        if arr.ndim != 2:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe ser bidimensional. "
                f"Dimensión observada: {arr.ndim}."
            )
        
        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' está vacía. "
                f"Se requiere al menos un elemento."
            )
        
        if square_required and arr.shape[0] != arr.shape[1]:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe ser cuadrada en el espacio de Hilbert. "
                f"Forma observada: {arr.shape}."
            )
        
        if min(arr.shape) < min_dimension:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe tener dimensión mínima {min_dimension}. "
                f"Forma observada: {arr.shape}."
            )
        
        if not np.all(np.isfinite(arr)):
            non_finite_count = int(np.sum(~np.isfinite(arr)))
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' contiene {non_finite_count} componentes no finitas."
            )
        
        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Coerción de vectores finitos con validación dimensional
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_vector(
        self,
        name: str,
        vector: Any,
        expected_dim: Optional[int] = None,
    ) -> NDArray[np.float64]:
        r"""
        Materializa un vector float64 finito y, si se indica, con dimensión
        exacta.
        
        Lema de Completitud: Todo vector en el espacio de fase debe tener
        componentes finitas y dimensión compatible con el fibrado tangente.
        """
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"El vector '{name}' no puede materializarse como NDArray."
            ) from exc
        
        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)
        
        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"El vector '{name}' está vacío."
            )
        
        if expected_dim is not None and arr.size != int(expected_dim):
            raise DomainIntegrityViolationError(
                f"El vector '{name}' debe tener dimensión {expected_dim}, "
                f"pero posee {arr.size} componentes."
            )
        
        if not np.all(np.isfinite(arr)):
            non_finite_count = int(np.sum(~np.isfinite(arr)))
            raise DomainIntegrityViolationError(
                f"El vector '{name}' contiene {non_finite_count} componentes no finitas."
            )
        
        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Descomposición espectral con verificación de realidad
    # ─────────────────────────────────────────────────────────────────────────
    def _spectral_decomposition_and_validation(
        self,
        operator_A: NDArray[np.complex128],
    ) -> SpectralDecompositionData:
        r"""
        Ejecuta la descomposición espectral del operador y valida que
        el espectro sea real dentro de tolerancias numéricas.
        
        Teorema Espectral: Si $A = A^\dagger$, entonces $\sigma(A) \subset \mathbb{R}$.
        
        Corolario: $\|\text{Im}(\lambda_i)\| \le \varepsilon_{\text{spectral}}$ para todo $i$.
        
        Retorna:
        ────────
        SpectralDecompositionData con certificación espectral completa.
        
        Excepciones:
        ────────────
        SpectralContaminationError si el espectro tiene componentes imaginarias.
        """
        try:
            eigenvalues = la.eigvalsh(operator_A)
        except la.LinAlgError as exc:
            raise SpectralContaminationError(
                f"Fallo en la descomposición espectral del operador: {exc}"
            ) from exc
        
        eigenvalues_complex = eigenvalues.astype(np.complex128)
        imaginary_parts = np.imag(eigenvalues_complex)
        real_parts = np.real(eigenvalues_complex)
        
        imaginary_norm = float(la.norm(imaginary_parts, ord=np.inf))
        
        spectral_tolerance = self._adaptive_tolerance(
            _SPECTRAL_IMAGINARY_TOLERANCE,
            operator_A,
            condition_amplification=True,
        )
        
        if imaginary_norm > spectral_tolerance:
            raise SpectralContaminationError(
                f"Contaminación espectral detectada: el operador tiene "
                f"autovalores con parte imaginaria no nula. "
                f"$\|\text{Im}(\sigma(A))\|_\infty = {imaginary_norm:.6e} > "
                f"{spectral_tolerance:.6e}$."
            )
        
        spectral_radius = float(la.norm(real_parts, ord=np.inf))
        
        trace_value = np.trace(operator_A)
        trace_real = float(np.real(trace_value))
        trace_imaginary = float(np.abs(np.imag(trace_value)))
        
        trace_tolerance = self._adaptive_tolerance(
            _TRACE_IMAGINARY_TOLERANCE,
            operator_A,
        )
        
        if trace_imaginary > trace_tolerance:
            raise SpectralContaminationError(
                f"Traza del operador con componente imaginaria: "
                f"$|\text{Im}(\text{tr}(A))| = {trace_imaginary:.6e} > "
                f"{trace_tolerance:.6e}$."
            )
        
        try:
            svals = la.svd(operator_A, compute_uv=False)
            if svals.size > 0 and svals[-1] > _MACHINE_EPSILON:
                condition_number = float(svals[0] / svals[-1])
            else:
                condition_number = float('inf')
        except Exception:
            condition_number = float('inf')
        
        return SpectralDecompositionData(
            eigenvalues_real=real_parts,
            eigenvalues_imaginary_norm=imaginary_norm,
            spectral_radius=spectral_radius,
            trace_real=trace_real,
            trace_imaginary_norm=trace_imaginary,
            condition_number=condition_number,
            is_spectrally_clean=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Auditoría de hermiticidad con norma de Frobenius
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_operator_hermiticity(
        self,
        operator_A: NDArray[np.complex128],
    ) -> HermiticityAuditData:
        r"""
        Calcula la norma de Frobenius de la diferencia entre el operador y su
        adjunto:
            $\|A - A^\dagger\|_F \le \varepsilon$.
        
        Teorema de Caracterización: $A$ es autoadjunto si y solo si
        $\|A - A^\dagger\|_F = 0$.
        
        Demonstración:
        ──────────────
        $\|A - A^\dagger\|_F^2 = \text{tr}((A - A^\dagger)^\dagger(A - A^\dagger))$
        $= \text{tr}((A^\dagger - A)(A - A^\dagger)) = -\text{tr}((A - A^\dagger)^2)$
        
        Si $A = A^\dagger$, entonces $A - A^\dagger = 0$ y la norma es cero.
        Recíprocamente, si la norma es cero, $A - A^\dagger = 0$.
        
        Retorna:
        ────────
        HermiticityAuditData con certificación de autoadjunción.
        
        Excepciones:
        ────────────
        NonHermitianOperatorError si el residuo excede la tolerancia.
        """
        A = self._coerce_finite_matrix(
            "operator_A",
            operator_A,
            dtype=np.complex128,
            square_required=True,
        )
        
        A_dagger = A.conj().T
        residual_matrix = A - A_dagger
        residual = float(la.norm(residual_matrix, ord="fro"))
        
        tolerance = self._adaptive_tolerance(
            _HERMITICITY_TOLERANCE,
            A,
            condition_amplification=True,
        )
        
        if residual > tolerance:
            max_deviation = float(la.norm(residual_matrix, ord=np.inf))
            raise NonHermitianOperatorError(
                f"Asimetría CPT detectada: el operador de aniquilación no es "
                f"autoadjunto. Residuo $\|A - A^\dagger\|_F = {residual:.6e} > "
                f"{tolerance:.6e}$. Desviación máxima: {max_deviation:.6e}."
            )
        
        spectral_data = self._spectral_decomposition_and_validation(A)
        
        return HermiticityAuditData(
            residual_norm=residual,
            is_hermitian=True,
            operator_dimension=int(A.shape[0]),
            hermiticity_tolerance=tolerance,
            spectral_imaginary_norm=spectral_data.eigenvalues_imaginary_norm,
            spectral_decomposition=spectral_data,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_audit_and_handoff_to_phase2(
        self,
        operator_A: NDArray[np.complex128],
    ) -> Phase1HermiticityHandoff:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 1.
        
        Su definición formal es la continuación directa de la Fase 2:
        entrega el certificado de hermiticidad y la dimensión del operador
        como prefijo obligatorio de la cota de Bekenstein.
        
        Lema de Continuación Funtorial:
        ────────────────────────────────
        La hermiticidad certificada del operador de colapso es condición
        necesaria y suficiente para que la entropía radiada sea computable
        como traza del operador densidad $\rho = e^{-\beta H}/Z$.
        
        Teorema de Correspondencia:
        ───────────────────────────
        $\Phi_2 \circ \Phi_1$ está bien definido si y solo si
        $\Phi_1$ retorna un objeto de tipo Phase1HermiticityHandoff.
        
        Parámetros:
        ───────────
        operator_A : NDArray[np.complex128]
            Operador de aniquilación a auditar.
        
        Retorna:
        ────────
        Phase1HermiticityHandoff
            Objeto terminal de Fase 1 y objeto inicial de Fase 2.
        
        Postcondición:
        ──────────────
        El handoff contiene:
        - Certificado de hermiticidad completo.
        - Dimensión del operador.
        - Certificado espectral opcional.
        
        Excepciones:
        ────────────
        NonHermitianOperatorError si $A \neq A^\dagger$.
        SpectralContaminationError si el espectro no es real.
        DomainIntegrityViolationError si la matriz es inválida.
        """
        hermiticity_audit = self._audit_operator_hermiticity(operator_A)
        
        logger.debug(
            "Fase 1 completada. $\|A - A^\dagger\|_F$=%.6e | dim=%d | "
            "radio espectral=%.6e.",
            hermiticity_audit.residual_norm,
            hermiticity_audit.operator_dimension,
            hermiticity_audit.spectral_decomposition.spectral_radius
            if hermiticity_audit.spectral_decomposition else 0.0,
        )
        
        return Phase1HermiticityHandoff(
            hermiticity_audit=hermiticity_audit,
            operator_dimension=hermiticity_audit.operator_dimension,
            spectral_certificate=hermiticity_audit.spectral_decomposition,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: REGULACIÓN TERMODINÁMICA DEL LÍMITE DE BEKENSTEIN                 ║
# ║   Verifica $S \le \frac{2\pi k_B E R}{\hbar c}$.                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_BekensteinBoundEnforcer(Phase1_HermiticityAuditor):
    r"""
    Controla la liberación de entropía durante la colisión $e^- + e^+ \to 2\gamma$.
    Previene la formación de singularidades informacionales.
    
    Fundamento Teórico:
    ────────────────────
    Límite de Bekenstein: La entropía máxima contenida en una región
    esférica de radio $R$ con energía total $E$ está acotada por:
    
        $S_{\text{max}} = \frac{2\pi k_B E R}{\hbar c}$
    
    Este límite emerge de la termodinámica de agujeros negros y la
    conjetura holográfica de 't Hooft-Susskind.
    
    Interpretación Física:
    ──────────────────────
    Si la entropía liberada por la aniquilación excede este límite,
    la región colapsaría gravitacionalmente, formando un agujero negro
    y destruyendo la estructura causal del espacio-tiempo local.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Certificación de no negatividad escalar con saneamiento de ruido
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_nonnegative_scalar(
        self,
        name: str,
        value: Any,
        strict_positive: bool = False,
    ) -> float:
        r"""
        Certifica que un escalar sea finito y no negativo, con saneamiento de
        ruido infinitesimal.
        
        Lema de No Negatividad: Las magnitudes físicas fundamentales
        (energía, entropía, radio) pertenecen a $\mathbb{R}_{\geq 0}$.
        
        Parámetros:
        ───────────
        name : str
            Identificador del escalar.
        value : Any
            Valor a certificar.
        strict_positive : bool
            Si True, exige positividad estricta ($> 0$).
        
        Retorna:
        ────────
        float
            Escalar saneado y certificado.
        
        Excepciones:
        ────────────
        DomainIntegrityViolationError si el escalar es negativo.
        """
        scalar = self._coerce_finite_scalar(name, value, allow_negative=False)
        tolerance = self._adaptive_tolerance(_BEKENSTEIN_ABS_TOLERANCE, scalar)
        
        if strict_positive:
            if scalar <= tolerance:
                raise DomainIntegrityViolationError(
                    f"El escalar '{name}' debe ser estrictamente positivo. "
                    f"Valor observado: {scalar:.6e}."
                )
        else:
            if scalar < -tolerance:
                raise DomainIntegrityViolationError(
                    f"El escalar '{name}' es negativo: {scalar:.6e}."
                )
            if scalar < 0.0:
                scalar = 0.0
        
        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Certificación de radio causal positivo
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_positive_radius(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Certifica que el radio de contención sea estrictamente positivo.
        
        Axioma de Causalidad: El radio de una región causal debe ser
        estrictamente positivo para que el cono de luz esté bien definido.
        
        Teorema de No Degeneración: Si $R = 0$, la cota de Bekenstein
        se anula y ninguna información puede ser contenida.
        
        Parámetros:
        ───────────
        name : str
            Identificador del radio.
        value : Any
            Valor del radio.
        
        Retorna:
        ────────
        float
            Radio certificado.
        
        Excepciones:
        ────────────
        BekensteinLimitViolation si el radio no es positivo.
        """
        scalar = self._coerce_finite_scalar(name, value)
        tolerance = self._adaptive_tolerance(_BEKENSTEIN_ABS_TOLERANCE, scalar)
        
        if scalar <= tolerance:
            raise BekensteinLimitViolation(
                f"El radio de contención '{name}' debe ser estrictamente "
                f"positivo. Valor observado: {scalar:.6e}."
            )
        
        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Verificación de consistencia dimensional física
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_dimensional_consistency(
        self,
        energy: float,
        radius: float,
        entropy: float,
    ) -> None:
        r"""
        Verifica que las magnitudes físicas tengan órdenes de magnitud
        consistentes con el régimen de aniquilación electrón-positrón.
        
        Lema de Consistencia Dimensional: Las unidades deben ser
        dimensionalmente consistentes para que la cota tenga sentido físico.
        
        Parámetros:
        ───────────
        energy : float
            Energía en Joules.
        radius : float
            Radio en metros.
        entropy : float
            Entropía en J/K.
        
        Excepciones:
        ────────────
        BekensteinLimitViolation si hay inconsistencia dimensional grave.
        """
        if energy > 1e10:
            logger.warning(
                "Energía de aniquilación extremadamente alta: %.6e J. "
                "Verificar unidades.", energy
            )
        
        if radius < 1e-20:
            logger.warning(
                "Radio de contención sub-Planckiano: %.6e m. "
                "La gravedad cuántica puede ser relevante.", radius
            )
        
        if entropy > 1e10:
            logger.warning(
                "Entropía macroscópica detectada: %.6e J/K. "
                "Verificar escala del sistema.", entropy
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Cálculo del tiempo de cruce lumínico
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_light_crossing_time(
        self,
        radius: float,
    ) -> float:
        r"""
        Calcula el tiempo de cruce lumínico de la región causal:
            $t_{\text{cross}} = \frac{R}{c}$
        
        Este tiempo representa la escala temporal mínima para que
        información causal se propague a través del sistema.
        
        Parámetros:
        ───────────
        radius : float
            Radio de la región en metros.
        
        Retorna:
        ────────
        float
            Tiempo de cruce en segundos.
        """
        return radius / _C_EFF

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. Cálculo de capacidad informacional en bits
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_information_capacity_bits(
        self,
        bekenstein_bound: float,
    ) -> float:
        r"""
        Convierte la cota de entropía a capacidad informacional en bits:
            $I_{\text{max}} = \frac{S_{\text{max}}}{k_B \ln 2}$
        
        Teorema de Landauer: Cada bit de información requiere al menos
        $k_B T \ln 2$ de energía para ser borrado.
        
        Parámetros:
        ───────────
        bekenstein_bound : float
            Cota de Bekenstein en J/K.
        
        Retorna:
        ────────
        float
            Capacidad máxima en bits.
        """
        if bekenstein_bound <= 0.0:
            return 0.0
        return bekenstein_bound / (_K_B * math.log(2.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6. Imposición de la cota de Bekenstein
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_bekenstein_gamma_bound(
        self,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> BekensteinBoundData:
        r"""
        Calcula la cota de Bekenstein y verifica que la entropía liberada no
        desgarre la variedad de datos:
            $S \le \frac{2\pi k_B E R}{\hbar c}$
        
        Teorema de Bekenstein (1981):
        ──────────────────────────────
        Para cualquier sistema físico con energía $E$ confinado en una
        esfera de radio $R$, la entropía $S$ satisface:
        
            $S \le \frac{2\pi k_B E R}{\hbar c}$
        
        La igualdad se alcanza para agujeros negros de Schwarzschild.
        
        Demonstración de la Cota:
        ─────────────────────────
        La cota emerge de considerar el proceso de Geroch: bajar un
        objeto con entropía hacia un agujero negro. La segunda ley
        generalizada exige que el aumento de área del horizonte compense
        la pérdida de entropía externa.
        
        Parámetros:
        ───────────
        gamma_energy : float
            Energía total de los fotones de aniquilación (J).
        system_radius_R : float
            Radio de la región de contención (m).
        emitted_entropy_S : float
            Entropía liberada en la aniquilación (J/K).
        
        Retorna:
        ────────
        BekensteinBoundData
            Certificado de cota termodinámica.
        
        Excepciones:
        ────────────
        BekensteinLimitViolation si la entropía excede la cota.
        """
        E = self._certify_nonnegative_scalar("gamma_energy", gamma_energy)
        R = self._certify_positive_radius("system_radius_R", system_radius_R)
        S = self._certify_nonnegative_scalar("emitted_entropy_S", emitted_entropy_S)
        
        self._verify_dimensional_consistency(E, R, S)
        
        s_bound = (
            2.0
            * math.pi
            * _K_B
            * E
            * R
        ) / (_HBAR_EFF * _C_EFF)
        
        if not np.isfinite(s_bound) or s_bound < 0.0:
            raise BekensteinLimitViolation(
                "La cota de Bekenstein no es finita o resultó negativa."
            )
        
        entropy_tolerance = max(
            _BEKENSTEIN_ABS_TOLERANCE,
            _BEKENSTEIN_REL_TOLERANCE * max(1.0, abs(S), abs(s_bound)),
        )
        
        if S > s_bound + entropy_tolerance:
            entropy_ratio = S / s_bound if s_bound > 0 else float('inf')
            raise BekensteinLimitViolation(
                f"Desgarro cosmológico: la aniquilación liberó entropía "
                f"S={S:.6e} superior a la cota de Bekenstein "
                f"S_max={s_bound:.6e} dentro de tolerancia "
                f"{entropy_tolerance:.6e}. Ratio de violación: {entropy_ratio:.6e}."
            )
        
        light_crossing_time = self._compute_light_crossing_time(R)
        information_bits = self._compute_information_capacity_bits(s_bound)
        entropy_ratio = S / s_bound if s_bound > 0 else 0.0
        
        return BekensteinBoundData(
            entropy_emitted=S,
            bekenstein_bound=s_bound,
            is_entropically_safe=True,
            gamma_energy=E,
            system_radius=R,
            entropy_tolerance=entropy_tolerance,
            entropy_ratio=entropy_ratio,
            causal_light_crossing_time=light_crossing_time,
            information_capacity_bits=information_bits,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.7. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_enforce_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> Phase2BekensteinHandoff:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 2.
        
        Su definición formal es la continuación directa de la Fase 3:
        entrega el certificado de Bekenstein y el handoff de Fase 1 como
        prefijo obligatorio de la certificación simpléctica.
        
        Lema de Continuación Funtorial:
        ────────────────────────────────
        La entropía acotada garantiza que el espacio de fase no sufre
        colapso informacional. El teorema de Liouville requiere que
        la densidad de estados sea integrable, lo cual está garantizado
        por la cota de Bekenstein.
        
        Teorema de Correspondencia:
        ───────────────────────────
        $\Phi_3 \circ \Phi_2$ está bien definido si y solo si
        $\Phi_2$ retorna un objeto de tipo Phase2BekensteinHandoff.
        
        Parámetros:
        ───────────
        phase1_handoff : Phase1HermiticityHandoff
            Certificado de Fase 1 (hermiticidad).
        gamma_energy : float
            Energía de los fotones de aniquilación.
        system_radius_R : float
            Radio de contención causal.
        emitted_entropy_S : float
            Entropía liberada.
        
        Retorna:
        ────────
        Phase2BekensteinHandoff
            Objeto terminal de Fase 2 y objeto inicial de Fase 3.
        
        Postcondición:
        ──────────────
        El handoff contiene:
        - Certificado de hermiticidad de Fase 1.
        - Certificado de cota de Bekenstein completo.
        - Metadata de capacidad informacional.
        
        Excepciones:
        ────────────
        DomainIntegrityViolationError si el handoff de Fase 1 es inválido.
        BekensteinLimitViolation si la entropía excede la cota.
        """
        if not isinstance(phase1_handoff, Phase1HermiticityHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1HermiticityHandoff como prefijo formal."
            )
        
        bekenstein_audit = self._enforce_bekenstein_gamma_bound(
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )
        
        logger.debug(
            "Fase 2 completada. S=%.6e | S_max=%.6e | ratio=%.6e | "
            "t_cross=%.6e s.",
            bekenstein_audit.entropy_emitted,
            bekenstein_audit.bekenstein_bound,
            bekenstein_audit.entropy_ratio,
            bekenstein_audit.causal_light_crossing_time,
        )
        
        return Phase2BekensteinHandoff(
            phase1_handoff=phase1_handoff,
            bekenstein_audit=bekenstein_audit,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: CERTIFICACIÓN SIMPLÉCTICA PORT-HAMILTONIANA                       ║
# ║   Exige $M^\top\Omega M = \Omega$, $J = -J^\top$, $R = R^\top \succeq 0$, $\dot{H} \le 0$. ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_SymplecticPortHamiltonianCertifier(Phase2_BekensteinBoundEnforcer):
    r"""
    Asegura que, tras el impacto de la antimateria, el remanente del grafo
    logístico recupere su estabilidad asintótica sin perder el volumen
    simpléctico.
    
    Fundamento Teórico:
    ────────────────────
    Teorema de Liouville: El flujo hamiltoniano preserva el volumen
    del espacio de fase. Formalmente, si $\phi_t$ es el flujo generado
    por $H$, entonces $\det(D\phi_t) = 1$.
    
    Estructura Port-Hamiltoniana:
    ─────────────────────────────
    Un sistema Port-Hamiltoniano tiene la forma:
    
        $\dot{x} = (J - R) \nabla H(x)$
    
    donde:
    - $J = -J^\top$ es la matriz de interconexión (estructura simpléctica).
    - $R = R^\top \succeq 0$ es la matriz de disipación.
    - $H$ es la función hamiltoniana (energía total).
    
    La condición $\dot{H} = -\nabla H^\top R \nabla H \le 0$ garantiza
    estabilidad asintótica por el teorema de Lyapunov.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Construcción de la 2-forma canónica
    # ─────────────────────────────────────────────────────────────────────────
    def _build_symplectic_form(
        self,
        n: int,
    ) -> NDArray[np.float64]:
        r"""
        Construye la forma simpléctica canónica:
            $\Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}$
        
        Propiedades:
        ────────────
        - $\Omega^\top = -\Omega$ (antisimetría).
        - $\Omega^2 = -I_{2n}$ (estructura casi compleja).
        - $\det(\Omega) = 1$.
        - $\text{Pf}(\Omega) = 1$.
        
        Parámetros:
        ───────────
        n : int
            Número de grados de libertad (mitad de la dimensión).
        
        Retorna:
        ────────
        NDArray[np.float64]
            Matriz simpléctica de dimensión $2n \times 2n$.
        
        Excepciones:
        ────────────
        PhaseSpaceTopologyError si $n$ no es positivo.
        """
        if n <= 0:
            raise PhaseSpaceTopologyError(
                f"El número de grados de libertad debe ser positivo. "
                f"Valor observado: {n}."
            )
        
        omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        identity = np.eye(n, dtype=np.float64)
        omega[:n, n:] = identity
        omega[n:, :n] = -identity
        
        return omega

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Verificación de dimensionalidad par del espacio de fase
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_phase_space_dimension(
        self,
        dimension: int,
    ) -> int:
        r"""
        Verifica que la dimensión del espacio de fase sea par y no nula.
        
        Teorema de Darboux: Todo espacio simpléctico tiene dimensión par.
        La forma simpléctica $\omega$ es una 2-forma no degenerada, lo cual
        requiere dimensión par.
        
        Parámetros:
        ───────────
        dimension : int
            Dimensión del espacio de fase.
        
        Retorna:
        ────────
        int
            Número de grados de libertad ($n = \dim/2$).
        
        Excepciones:
        ────────────
        PhaseSpaceTopologyError si la dimensión no es par o es nula.
        """
        if dimension <= 0:
            raise PhaseSpaceTopologyError(
                f"La dimensión del espacio de fase debe ser positiva. "
                f"Valor observado: {dimension}."
            )
        
        if dimension % 2 != 0:
            raise PhaseSpaceTopologyError(
                f"La dimensión del espacio de fase debe ser par. "
                f"Valor observado: {dimension}. "
                f"Teorema de Darboux: $\dim(\mathcal{M}) = 2n$."
            )
        
        return dimension // 2

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Certificación de antisimetría de J
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_antisymmetric_matrix(
        self,
        name: str,
        matrix: NDArray[np.float64],
        expected_dim: int,
    ) -> float:
        r"""
        Certifica que la matriz de interconexión sea antisimétrica:
            $J = -J^\top$
        
        Lema de Estructura: La matriz $J$ define un corchete de Poisson
        si y solo si es antisimétrica y satisface la identidad de Jacobi.
        
        Parámetros:
        ───────────
        name : str
            Identificador de la matriz.
        matrix : NDArray[np.float64]
            Matriz a certificar.
        expected_dim : int
            Dimensión esperada.
        
        Retorna:
        ────────
        float
            Residuo de antisimetría $\|J + J^\top\|_F$.
        
        Excepciones:
        ────────────
        SymplecticCollapseError si la matriz no es antisimétrica.
        """
        if matrix.shape != (expected_dim, expected_dim):
            raise SymplecticCollapseError(
                f"La matriz '{name}' debe tener dimensión "
                f"{expected_dim}x{expected_dim}. Forma observada: {matrix.shape}."
            )
        
        antisymmetric_part = matrix + matrix.T
        residual = float(la.norm(antisymmetric_part, ord="fro"))
        tolerance = self._adaptive_tolerance(_ANTISYMMETRY_TOLERANCE, matrix)
        
        if residual > tolerance:
            max_deviation = float(la.norm(antisymmetric_part, ord=np.inf))
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es antisimétrica. "
                f"$\|J + J^\top\|_F = {residual:.6e} > {tolerance:.6e}$. "
                f"Desviación máxima: {max_deviation:.6e}."
            )
        
        return residual

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. Certificación de simetría y semidefinición positiva de R
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_symmetric_positive_semidefinite_matrix(
        self,
        name: str,
        matrix: NDArray[np.float64],
        expected_dim: int,
    ) -> Tuple[NDArray[np.float64], float, float, float]:
        r"""
        Certifica que la matriz de disipación sea simétrica y semidefinida positiva:
            $R = R^\top \succeq 0$
        
        Teorema de Disipación: La matriz $R$ define una métrica de
        disipación si y solo si es simétrica y semidefinida positiva.
        
        Condición de Lyapunov: $\dot{H} = -\nabla H^\top R \nabla H \le 0$
        se satisface si y solo si $R \succeq 0$.
        
        Parámetros:
        ───────────
        name : str
            Identificador de la matriz.
        matrix : NDArray[np.float64]
            Matriz a certificar.
        expected_dim : int
            Dimensión esperada.
        
        Retorna:
        ────────
        Tuple[NDArray[np.float64], float, float, float]
            (R simetrizada, residuo de simetría, autovalor mínimo, autovalor máximo).
        
        Excepciones:
        ────────────
        SymplecticCollapseError si la matriz no es simétrica o no es PSD.
        """
        if matrix.shape != (expected_dim, expected_dim):
            raise SymplecticCollapseError(
                f"La matriz '{name}' debe tener dimensión "
                f"{expected_dim}x{expected_dim}. Forma observada: {matrix.shape}."
            )
        
        symmetric_part = matrix - matrix.T
        symmetry_residual = float(la.norm(symmetric_part, ord="fro"))
        symmetry_tolerance = self._adaptive_tolerance(
            _R_SYMMETRY_TOLERANCE,
            matrix,
        )
        
        if symmetry_residual > symmetry_tolerance:
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es simétrica. "
                f"$\|R - R^\top\|_F = {symmetry_residual:.6e} > "
                f"{symmetry_tolerance:.6e}$."
            )
        
        R_sym = 0.5 * (matrix + matrix.T)
        
        try:
            eigenvalues = la.eigvalsh(R_sym)
        except la.LinAlgError as exc:
            raise SymplecticCollapseError(
                f"Fallo en el cálculo de autovalores de '{name}': {exc}"
            ) from exc
        
        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
        max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        max_abs_eigenvalue = max(abs(min_eigenvalue), abs(max_eigenvalue), 1.0)
        
        psd_tolerance = max(
            _PSD_EIGENVALUE_TOLERANCE,
            10.0 * _MACHINE_EPSILON * expected_dim * max_abs_eigenvalue,
        )
        
        if min_eigenvalue < -psd_tolerance:
            negative_count = int(np.sum(eigenvalues < -psd_tolerance))
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es semidefinida positiva. "
                f"$\lambda_{{min}} = {min_eigenvalue:.6e} < -{psd_tolerance:.6e}$. "
                f"Autovalores negativos: {negative_count}."
            )
        
        return R_sym, symmetry_residual, min_eigenvalue, max_eigenvalue

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5. Verificación de invarianza de volumen simpléctico
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_symplectic_volume_preservation(
        self,
        jacobian_M: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Verifica que el jacobiano preserve la forma simpléctica:
            $M^\top \Omega M = \Omega$
        
        Teorema de Liouville: El flujo hamiltoniano preserva el volumen
        del espacio de fase. Equivalentemente, $\det(M) = 1$.
        
        Corolario: Si $M^\top \Omega M = \Omega$, entonces $\det(M)^2 = 1$,
        y como $M$ es continuo con $M(0) = I$, se tiene $\det(M) = 1$.
        
        Parámetros:
        ───────────
        jacobian_M : NDArray[np.float64]
            Matriz jacobiana del difeomorfismo.
        omega : NDArray[np.float64]
            Forma simpléctica canónica.
        
        Retorna:
        ────────
        Tuple[float, float]
            (Residuo simpléctico, residuo de determinante).
        
        Excepciones:
        ────────────
        SymplecticCollapseError si la invarianza falla.
        """
        omega_transformed = jacobian_M.T @ omega @ jacobian_M
        
        if not np.all(np.isfinite(omega_transformed)):
            raise SymplecticCollapseError(
                "La transformación simpléctica $M^\top\Omega M$ contiene "
                "componentes no finitas."
            )
        
        symplectic_residual = float(la.norm(omega_transformed - omega, ord="fro"))
        
        try:
            determinant = float(np.linalg.det(jacobian_M))
        except np.linalg.LinAlgError:
            determinant = float('nan')
        
        determinant_residual = abs(determinant - 1.0)
        
        symplectic_tolerance = self._adaptive_tolerance(
            _SYMPLECTIC_TOLERANCE,
            jacobian_M,
            condition_amplification=True,
        )
        
        determinant_tolerance = self._adaptive_tolerance(
            _DETERMINANT_TOLERANCE,
            jacobian_M,
        )
        
        if symplectic_residual > symplectic_tolerance:
            raise SymplecticCollapseError(
                f"Degradación del espacio de fase: el evento de aniquilación "
                f"destruyó la 2-forma canónica $\omega$. "
                f"Residuo $\|M^\top\Omega M - \Omega\|_F = "
                f"{symplectic_residual:.6e} > {symplectic_tolerance:.6e}$."
            )
        
        if determinant_residual > determinant_tolerance:
            raise SymplecticCollapseError(
                f"Violación del teorema de Liouville: el determinante del "
                f"jacobiano no es unitario. $|\det(M) - 1| = "
                f"{determinant_residual:.6e} > {determinant_tolerance:.6e}$. "
                f"$\det(M) = {determinant:.6e}$."
            )
        
        return symplectic_residual, determinant_residual

    # ─────────────────────────────────────────────────────────────────────────
    # 3.6. Auditoría de disipación Port-Hamiltoniana
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_port_hamiltonian_dissipation(
        self,
        grad_H: NDArray[np.float64],
        R_certified: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la tasa de disipación de energía:
            $\dot{H} = -\nabla H^\top R \nabla H \le 0$
        
        Teorema de Lyapunov: Si $R \succeq 0$, entonces $\dot{H} \le 0$,
        lo cual garantiza estabilidad asintótica del equilibrio.
        
        Interpretación Física: La energía del sistema no puede aumentar
        espontáneamente. La bobina de choque solo puede disipar energía,
        nunca inyectar energía parásita.
        
        Parámetros:
        ───────────
        grad_H : NDArray[np.float64]
            Gradiente del hamiltoniano.
        R_certified : NDArray[np.float64]
            Matriz de disipación certificada como PSD.
        
        Retorna:
        ────────
        float
            Tasa de disipación $\dot{H}$.
        
        Excepciones:
        ────────────
        SymplecticCollapseError si la tasa es positiva (inyección de energía).
        """
        h_dot = -float(grad_H.T @ R_certified @ grad_H)
        
        if not np.isfinite(h_dot):
            raise SymplecticCollapseError(
                "La tasa de disipación $\dot{H}$ no es finita."
            )
        
        dissipation_tolerance = max(
            _MACHINE_EPSILON,
            self._adaptive_tolerance(_SYMPLECTIC_TOLERANCE, grad_H),
        )
        
        if h_dot > dissipation_tolerance:
            raise SymplecticCollapseError(
                f"Violación termodinámica: la bobina de choque inyectó energía "
                f"parásita al sistema. $\dot{{H}} = {h_dot:.6e} > "
                f"{dissipation_tolerance:.6e}$."
            )
        
        return h_dot

    # ─────────────────────────────────────────────────────────────────────────
    # 3.7. Certificación simpléctica Port-Hamiltoniana completa
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_symplectic_port_hamiltonian(
        self,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> SymplecticDissipationData:
        r"""
        Evalúa el difeomorfismo canónico y la termodinámica del estrangulador:
        
        Condiciones Port-Hamiltonianas:
        ───────────────────────────────
        1. $M^\top \Omega M = \Omega$ (invarianza simpléctica)
        2. $J = -J^\top$ (antisimetría de interconexión)
        3. $R = R^\top \succeq 0$ (disipación pasiva)
        4. $\dot{H} = -\nabla H^\top R \nabla H \le 0$ (segunda ley)
        
        Teorema de Estabilidad: Si todas las condiciones se satisfacen,
        el sistema es estable en el sentido de Lyapunov y el volumen
        del espacio de fase se preserva.
        
        Parámetros:
        ───────────
        jacobian_M : NDArray[np.float64]
            Matriz jacobiana del flujo.
        grad_H : NDArray[np.float64]
            Gradiente del hamiltoniano.
        J_matrix : NDArray[np.float64]
            Matriz de interconexión.
        R_matrix : NDArray[np.float64]
            Matriz de disipación.
        
        Retorna:
        ────────
        SymplecticDissipationData
            Certificado completo de geometría Port-Hamiltoniana.
        
        Excepciones:
        ────────────
        SymplecticCollapseError si alguna condición falla.
        PhaseSpaceTopologyError si la dimensión es inválida.
        """
        M = self._coerce_finite_matrix(
            "jacobian_M",
            jacobian_M,
            dtype=np.float64,
            square_required=True,
        )
        
        dim = int(M.shape[0])
        n = self._verify_phase_space_dimension(dim)
        
        grad = self._coerce_finite_vector(
            "grad_H",
            grad_H,
            expected_dim=dim,
        )
        
        J = self._coerce_finite_matrix(
            "J_matrix",
            J_matrix,
            dtype=np.float64,
            square_required=True,
        )
        
        R = self._coerce_finite_matrix(
            "R_matrix",
            R_matrix,
            dtype=np.float64,
            square_required=True,
        )
        
        # 1. Estructura Port-Hamiltoniana: J antisimétrica.
        antisymmetry_residual = self._certify_antisymmetric_matrix(
            "J_matrix",
            J,
            dim,
        )
        
        # 2. Estructura disipativa: R simétrica y semidefinida positiva.
        R_certified, r_symmetry_residual, r_min_eigenvalue, r_max_eigenvalue = (
            self._certify_symmetric_positive_semidefinite_matrix(
                "R_matrix",
                R,
                dim,
            )
        )
        
        # 3. Auditoría del volumen simpléctico: Mᵀ Ω M = Ω.
        omega = self._build_symplectic_form(n)
        symplectic_residual, determinant_residual = (
            self._verify_symplectic_volume_preservation(M, omega)
        )
        
        # 4. Auditoría de disipación Port-Hamiltoniana: Ḣ ≤ 0.
        h_dot = self._audit_port_hamiltonian_dissipation(grad, R_certified)
        
        symplectic_tolerance = self._adaptive_tolerance(
            _SYMPLECTIC_TOLERANCE,
            M,
        )
        
        return SymplecticDissipationData(
            symplectic_residual=symplectic_residual,
            dissipation_rate=h_dot,
            is_symplectically_invariant=True,
            symplectic_tolerance=symplectic_tolerance,
            antisymmetry_residual=antisymmetry_residual,
            r_symmetry_residual=r_symmetry_residual,
            r_min_eigenvalue=r_min_eigenvalue,
            r_max_eigenvalue=r_max_eigenvalue,
            determinant_residual=determinant_residual,
            phase_space_dimension=dim,
            degrees_of_freedom=n,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.8. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 3.
        
        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal VacuumGovernanceState.
        
        Teorema de Corrección Global:
        ──────────────────────────────
        Si $\Phi_1$, $\Phi_2$ y $\Phi_3$ certifican éxito, entonces el
        vacío cuántico está topológicamente protegido y la aniquilación
        de antimateria exógena no corrompe la estructura del espacio de Fock.
        
        Corolario de Estabilidad: El sistema resultante es:
        - Espectralmente estable (autovalores reales).
        - Termodinámicamente seguro (entropía acotada).
        - Geométricamente invariante (volumen simpléctico preservado).
        - Asintóticamente estable (disipación no negativa).
        
        Parámetros:
        ───────────
        phase2_handoff : Phase2BekensteinHandoff
            Certificado de Fase 2 (Bekenstein + Fase 1).
        jacobian_M : NDArray[np.float64]
            Matriz jacobiana del flujo.
        grad_H : NDArray[np.float64]
            Gradiente del hamiltoniano.
        J_matrix : NDArray[np.float64]
            Matriz de interconexión.
        R_matrix : NDArray[np.float64]
            Matriz de disipación.
        
        Retorna:
        ────────
        VacuumGovernanceState
            Objeto terminal del endofuntor $\mathcal{Z}_{Vacuum}$.
        
        Postcondición:
        ──────────────
        El estado final contiene:
        - Certificado de hermiticidad.
        - Certificado de Bekenstein.
        - Certificado simpléctico.
        - Flag de validez epistemológica.
        - Metadata de gobernanza.
        
        Excepciones:
        ────────────
        DomainIntegrityViolationError si el handoff de Fase 2 es inválido.
        SymplecticCollapseError si la geometría falla.
        """
        if not isinstance(phase2_handoff, Phase2BekensteinHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2BekensteinHandoff como prefijo formal."
            )
        
        symplectic_audit = self._certify_symplectic_port_hamiltonian(
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )
        
        governance_metadata = {
            "functor_composition": "Φ₃ ∘ Φ₂ ∘ Φ₁",
            "phase1_residual": phase2_handoff.phase1_handoff.hermiticity_audit.residual_norm,
            "phase2_entropy_ratio": phase2_handoff.bekenstein_audit.entropy_ratio,
            "phase3_dissipation_rate": symplectic_audit.dissipation_rate,
            "phase_space_dof": symplectic_audit.degrees_of_freedom,
            "information_capacity_bits": phase2_handoff.bekenstein_audit.information_capacity_bits,
        }
        
        state = VacuumGovernanceState(
            hermiticity_audit=phase2_handoff.phase1_handoff.hermiticity_audit,
            bekenstein_audit=phase2_handoff.bekenstein_audit,
            symplectic_audit=symplectic_audit,
            is_epistemologically_valid=True,
            governance_metadata=governance_metadata,
        )
        
        logger.info(
            "Vacío cuántico auditado categóricamente. "
            "$\|A - A^\dagger\|_F$=%.6e | S=%.6e ≤ S_max=%.6e | "
            "$\dot{H}$=%.6e | dim=%d.",
            state.hermiticity_audit.residual_norm,
            state.bekenstein_audit.entropy_emitted,
            state.bekenstein_audit.bekenstein_bound,
            state.symplectic_audit.dissipation_rate,
            state.symplectic_audit.phase_space_dimension,
        )
        
        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: ANTIMATTER CHOKE COIL AGENT                          ║
# ║   Endofuntor $\mathcal{Z}_{Vacuum} = \Phi_3 \circ \Phi_2 \circ \Phi_1$   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class AntimatterChokeCoilAgent(
    Morphism,
    Phase3_SymplecticPortHamiltonianCertifier,
):
    r"""
    El Custodio del Vacío Cuántico en el Estrato Ω.
    
    Somete los procesos de inyección de antimateria exógena a las leyes
    inmutables de la conservación geométrica y los límites absolutos de la
    entropía.
    
    Definición Categórica:
    ──────────────────────
    $\mathcal{Z}_{Vacuum}: \mathcal{F}(\mathcal{H}) \to \mathcal{F}(\mathcal{H})$
    es un endofuntor en la categoría de espacios de Fock que preserva
    la estructura simpléctica, la cota de Bekenstein y la hermiticidad
    de los observables.
    
    Propiedad Universal: Para cualquier proceso de aniquilación
    $e^- + e^+ \to 2\gamma$, el endofuntor garantiza que el estado
    resultante pertenece al mismo componente conexo del espacio de
    configuraciones que el estado inicial.
    """

    def execute_vacuum_governance(
        self,
        operator_A: NDArray[np.complex128],
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta.
        
        Diagrama Conmutativo:
        ─────────────────────
        
        $\mathcal{F}(\mathcal{H}) \xrightarrow{\Phi_1} \text{HermiticityAudit}$
              $\downarrow$                                    $\downarrow$
        $\text{BekensteinBound} \xleftarrow{\Phi_2} \text{HermiticityAudit}$
              $\downarrow$                                    $\downarrow$
        $\text{SymplecticCert} \xleftarrow{\Phi_3} \text{BekensteinBound}$
              $\downarrow$
        $\text{VacuumGovernanceState}$
        
        Parámetros:
        ───────────
        operator_A : NDArray[np.complex128]
            Operador de aniquilación.
        gamma_energy : float
            Energía de fotones (J).
        system_radius_R : float
            Radio de contención (m).
        emitted_entropy_S : float
            Entropía liberada (J/K).
        jacobian_M : NDArray[np.float64]
            Jacobiano del flujo.
        grad_H : NDArray[np.float64]
            Gradiente hamiltoniano.
        J_matrix : NDArray[np.float64]
            Matriz de interconexión.
        R_matrix : NDArray[np.float64]
            Matriz de disipación.
        
        Retorna:
        ────────
        VacuumGovernanceState
            Estado terminal del endofuntor.
        """
        phase1_handoff = self._phase1_audit_and_handoff_to_phase2(
            operator_A=operator_A,
        )
        
        phase2_handoff = self._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )
        
        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )

    def __call__(
        self,
        operator_A: NDArray[np.complex128],
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""Alias invocable del endofuntor de gobierno del vacío cuántico."""
        return self.execute_vacuum_governance(
            operator_A=operator_A,
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )


# ════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "VacuumCustodianError",
    "DomainIntegrityViolationError",
    "NonHermitianOperatorError",
    "SpectralContaminationError",
    "BekensteinLimitViolation",
    "CausalityViolationError",
    "SymplecticCollapseError",
    "PhaseSpaceTopologyError",
    "SpectralDecompositionData",
    "HermiticityAuditData",
    "BekensteinBoundData",
    "SymplecticDissipationData",
    "Phase1HermiticityHandoff",
    "Phase2BekensteinHandoff",
    "VacuumGovernanceState",
    "Phase1_HermiticityAuditor",
    "Phase2_BekensteinBoundEnforcer",
    "Phase3_SymplecticPortHamiltonianCertifier",
    "AntimatterChokeCoilAgent",
]