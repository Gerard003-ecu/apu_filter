# -*- coding: utf-8 -*-
r"""
=========================================================================================
Módulo: Quantum Algebra (Núcleo de Mecánica Cuántica para la Malla Agéntica)
Ubicación: app/core/quantum_algebra.py
Versión: 1.0.0 (Fase 2: Consagración Cuántica)

NATURALEZA CIBER-FÍSICA:
Este módulo implementa el formalismo de la mecánica cuántica de sistemas cerrados
para modelar el espacio de decisión agéntica. Transmuta la deliberación booleana
en un operador de densidad $\rho$ actuando sobre un espacio de Hilbert $\mathcal{H}_N$.

FUNDAMENTOS MATEMÁTICOS:

§1. ESPACIO DE HILBERT $\mathcal{H}_N$ Y BASE ORTONORMAL:
Definimos el espacio vectorial complejo $\mathbb{C}^N$ equipado con el producto
interno hermítico $\langle \phi | \psi \rangle = \sum_i \phi_i^* \psi_i$.
La base canónica $\{ |e_i\rangle \}_{i=1}^N$ satisface:
$$ \langle e_i | e_j \rangle = \delta_{ij} $$
La matriz de Gram $G$ asociada a esta base es rigurosamente la identidad $I_N$.
La integridad del espacio se verifica mediante el rango del tensor métrico $T$:
$$ \text{rank}(T) = N $$
evaluado vía Descomposición en Valores Singulares (SVD) truncada bajo $\epsilon_{mach}$.

§2. MATRIZ DE DENSIDAD $\rho$ (OPERADOR DE ESTADO):
El estado del Consejo de Sabios no es un vector determinista, sino un ensamble
estadístico modelado por el operador de densidad $\rho: \mathcal{H}_N \to \mathcal{H}_N$.
Para un estado mixto con probabilidades $p_i$, el operador se ensambla como:
$$ \rho = \sum_{i=1}^{k} p_i |\psi_i\rangle \langle \psi_i| $$

AXIOMAS DE CONSISTENCIA (HARD VETOES):
Cualquier instancia de $\rho$ debe satisfacer incondicionalmente:
1. Hermiticidad: El tensor debe ser invariante bajo transposición conjugada.
   $$ \| \rho - \rho^\dagger \|_{\infty} \le \epsilon_{mach} \implies \text{Im}(\text{spec}(\rho)) \equiv 0 $$
2. Positividad Semidefinida: Las probabilidades negativas son aberraciones topológicas.
   $$ \lambda_{min}(\rho) \ge 0 $$
3. Traza Unitaria (Conservación de Decisión):
   $$ \text{Tr}(\rho) = \sum_{i=1}^{N} \lambda_i = 1 $$

=========================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Final
import logging
from app.adapters.tools_interface import MICRegistry, MICConfiguration, TopologicalInvariantError

logger = logging.getLogger("MIC.Physics.QuantumAlgebra")

@dataclass(frozen=True, slots=True)
class HilbertSpace:
    r"""
    Representación del espacio de Hilbert complejo $\mathcal{H}_N \cong \mathbb{C}^N$.

    Attributes:
        dimension: Dimensión $N$ del espacio.
        basis: Matriz $T$ cuyas columnas son los vectores de la base $\{ |e_i\rangle \}$.
        epsilon: Tolerancia para validaciones numéricas.
    """
    dimension: int
    basis: np.ndarray = field(repr=False)
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        r"""
        Valida la integridad del espacio de Hilbert mediante aserciones algebraicas.

        Aserción de Código: Valida que el tensor métrico del espacio de Hilbert no presente
        deficiencia de rango computando rank(T)=N mediante descomposición SVD truncada
        bajo la tolerancia de la máquina $\epsilon_{mach}$.

        1. Ortogonalidad Funcional: $\langle e_i | e_j \rangle = \delta_{ij}$.
        2. Independencia Lineal: $\text{rank}(T) = N$.
        """
        # Verificación de la Matriz de Gram G = B^H B coincidiendo con la Identidad
        # <e_i | e_j> = delta_ij
        gram = self.basis.conj().T @ self.basis
        identity = np.eye(self.dimension, dtype=np.complex128)

        gram_error = np.linalg.norm(gram - identity, ord=np.inf)
        if gram_error > self.epsilon:
            raise TopologicalInvariantError(
                f"Violación de Ortogonalidad Funcional: ||G - I||_inf = {gram_error:.2e} > {self.epsilon}"
            )

        # Verificación de rango vía SVD (Singular Value Decomposition)
        # rank(T) = N
        s = np.linalg.svd(self.basis, compute_uv=False)
        rank = np.sum(s > self.epsilon)
        if rank != self.dimension:
            raise TopologicalInvariantError(
                f"Deficiencia de Rango en el Tensor Métrico: rank(T) = {rank} < {self.dimension}"
            )

    @classmethod
    def create_canonical(cls, dimension: int, epsilon: float = 1e-12) -> HilbertSpace:
        r"""
        Implementa el constructor de la base ortonormal canónica $\{ |e_i\rangle \}_{i=1}^{N}$.
        """
        basis = np.eye(dimension, dtype=np.complex128)
        return cls(dimension=dimension, basis=basis, epsilon=epsilon)

@dataclass(frozen=True, slots=True)
class QuantumDensityOperator:
    r"""
    Operador de Densidad Cuántica $\rho$ (Estado de la Deliberación).

    Implementa la formalización de un estado cuántico que satisface los axiomas
    de Hermiticidad, Positividad y Traza Unitaria.
    """
    rho: np.ndarray
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        r"""
        Auditoría axiomática del operador de densidad (Hard Vetoes).

        El constructor somete la matriz $\rho$ a las tres pruebas fundamentales:

        1. Hermiticidad: $\| \rho - \rho^\dagger \|_{\infty} \le \epsilon_{mach}$
           Garantiza que el espectro sea real y el operador sea observable.

        2. Positividad Semidefinida: $\lambda_{min}(\rho) \ge 0$
           Asegura que no existan probabilidades negativas.

        3. Traza Unitaria: $\text{Tr}(\rho) = \sum \lambda_i = 1$
           Conserva la decisión total del ensamble.
        """
        if self.rho.shape[0] != self.rho.shape[1]:
            raise TopologicalInvariantError("La matriz de densidad debe ser cuadrada.")

        # 1. Verificación de Hermiticidad
        rho_dagger = self.rho.conj().T
        hermiticity_residue = np.linalg.norm(self.rho - rho_dagger, ord=np.inf)
        if hermiticity_residue > self.epsilon:
            raise TopologicalInvariantError(
                f"Violación de Hermiticidad: residue = {hermiticity_residue:.2e}. "
                "El operador de estado debe ser autoadjunto."
            )

        # 2. Verificación de Positividad Semidefinida
        # Se calcula la descomposición espectral para verificar los autovalores
        eigenvalues = np.linalg.eigvalsh(self.rho)
        lambda_min = np.min(eigenvalues)
        if lambda_min < -self.epsilon:
            raise TopologicalInvariantError(
                f"Violación de Positividad Semidefinida: lambda_min = {lambda_min:.2e}. "
                "Las probabilidades negativas son aberraciones topológicas."
            )

        # 3. Verificación de Traza Unitaria
        # Tr(rho) = 1
        trace = np.trace(self.rho)
        trace_residue = abs(trace - 1.0)
        if trace_residue > self.epsilon:
            raise TopologicalInvariantError(
                f"Violación de Traza Unitaria: Tr(rho) = {trace.real:.6f} + {trace.imag:.6f}j. "
                "Fallo en la conservación de la decisión total."
            )

    @classmethod
    def from_pure_state(cls, psi: np.ndarray, epsilon: float = 1e-12) -> QuantumDensityOperator:
        r"""
        Construye $\rho$ a partir de un estado puro $| \psi \rangle$.
        $$ \rho = | \psi \rangle \langle \psi | $$
        """
        # Normalización del vector de estado para asegurar traza unitaria
        norm = np.linalg.norm(psi)
        if norm < epsilon:
            raise TopologicalInvariantError("No se puede construir un estado a partir del vector nulo.")

        psi_norm = psi / norm
        rho = np.outer(psi_norm, psi_norm.conj())
        return cls(rho=rho, epsilon=epsilon)

    @classmethod
    def from_mixed_state(cls, weights: List[float], states: List[np.ndarray], epsilon: float = 1e-12) -> QuantumDensityOperator:
        r"""
        Construye $\rho$ a partir de un ensamble estadístico (estado mixto).
        $$ \rho = \sum_{i=1}^{k} p_i |\psi_i\rangle \langle \psi_i| $$
        """
        if len(weights) != len(states):
            raise ValueError("Pesos y estados deben tener la misma cardinalidad.")

        # Normalización de las probabilidades de selección p_i
        w_sum = sum(weights)
        if abs(w_sum) < epsilon:
             raise TopologicalInvariantError("La suma de pesos no puede ser cero.")

        p = [w / w_sum for w in weights]

        dim = states[0].shape[0]
        rho_sum = np.zeros((dim, dim), dtype=np.complex128)

        for pi, psi in zip(p, states):
            # Normalización de cada vector de base del ensamble
            n = np.linalg.norm(psi)
            if n > epsilon:
                u = psi / n
                rho_sum += pi * np.outer(u, u.conj())

        return cls(rho=rho_sum, epsilon=epsilon)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Retorna los autovalores del operador de densidad."""
        return np.linalg.eigvalsh(self.rho)

    def compute_von_neumann_entropy(self) -> float:
        r"""
        Fase 2.3: Termodinámica del Colapso (Entropía de Von Neumann).
        Computa la traza matricial sobre el logaritmo matricial:
        $$ S(\rho) = -\text{Tr}(\rho \ln \rho) = -\sum_{i} \lambda_i \ln \lambda_i $$

        Aserción Axiomática (Idempotencia vs Gibbs):
        - Estado Puro: \rho^2 = \rho \implies S(\rho) \le \epsilon_{mach}
        - Estado Mixto: S(\rho) > 0
        """
        ev = self.eigenvalues
        # Filtrar autovalores cero para evitar log(0)
        ev = ev[ev > self.epsilon]
        if len(ev) == 0:
            return 0.0
        return -float(np.sum(ev * np.log(ev)))

    def is_pure_state(self) -> bool:
        r"""
        Verifica empíricamente si \rho es un proyector idempotente (Estado Puro).
        """
        return self.compute_von_neumann_entropy() <= self.epsilon

class QuantumRegistry(MICRegistry):
    r"""
    Evolución del MICRegistry (Núcleo Cuántico).

    Gobierna la deliberación de la Malla Agéntica mediante el operador de densidad $\rho$
    en un espacio de Hilbert $\mathcal{H}_N$. Esta estructura trasciende la certidumbre
    booleana clásica para sumergirse en la mecánica cuántica de sistemas abiertos.

    Axioma de Estado Cuántico:
    El sistema se define por el operador $\rho \in \mathcal{L}(\mathcal{H}_N)$,
    que representa el ensamble estadístico de las intenciones agénticas.
    """

    __slots__ = ("_hilbert_space", "_state")

    def __init__(self, rho: np.ndarray, config: Optional[MICConfiguration] = None) -> None:
        r"""
        Constructor del Registro Cuántico (Consagración de la Fase 2).

        Somete la matriz de densidad $\rho$ a una auditoría axiomática implacable
        garantizando que el operador sea un estado físico legítimo.

        Hard Vetoes Verificados (Incondicionales):
        1. Hermiticidad: $\rho = \rho^\dagger$
           (Asegura que los observables de decisión sean reales: $\lambda_i \in \mathbb{R}$)
        2. Positividad Semidefinida: $\rho \ge 0$
           (Aniquila la posibilidad de probabilidades negativas, aberraciones topológicas)
        3. Traza Unitaria: $\text{Tr}(\rho) = 1$
           (Garantiza la conservación de la decisión total en el manifold de deliberación)

        Args:
            rho: Matriz de densidad inicial $\rho \in \mathbb{C}^{N \times N}$.
            config: Configuración MIC para tolerancias numéricas $\epsilon$.

        Raises:
            TopologicalInvariantError: Si la matriz $\rho$ viola algún axioma cuántico.
        """
        super().__init__(config=config)

        # Fase 2.1: Instanciación del Espacio de Hilbert HN y la Base Ortonormal
        # Se garantiza que <e_i | e_j> = delta_ij y rank(T) = N
        self._hilbert_space = HilbertSpace.create_canonical(
            dimension=rho.shape[0],
            epsilon=self._config.epsilon
        )

        # Fase 2.2: Construcción Invariante de la Matriz de Densidad rho
        # La instanciación de QuantumDensityOperator ejecuta los Hard Vetoes
        self._state = QuantumDensityOperator(
            rho=rho,
            epsilon=self._config.epsilon
        )

        logger.info(
            r"QuantumRegistry instanciado exitosamente. "
            r"Espacio de Hilbert $\mathcal{H}_{%d}$ certificado bajo $\epsilon_{mach}=%.2e$.",
            self._hilbert_space.dimension,
            self._config.epsilon
        )

    @property
    def quantum_state(self) -> QuantumDensityOperator:
        """Retorna el operador de densidad actual."""
        return self._state

    @property
    def hilbert_space(self) -> HilbertSpace:
        """Retorna el espacio de Hilbert asociado."""
        return self._hilbert_space

    def apply_observational_projectors(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Fase 2.4: Proyectores Observacionales (Quantum Admission Gate).
        Implementa los proyectores ortogonales P1 y P2 actuando sobre el estado de la función de onda.

        Axiomas de Resolución:
        1. Resolución de Identidad: P1 + P2 = I_obs
        2. Exclusión Mutua: P1 P2 = 0

        Args:
            psi: Vector de estado de la función de onda M.

        Returns:
            Tupla (P1|psi>, P2|psi>) representando la información filtrada y observada.
        """
        N = self._hilbert_space.dimension
        # Definimos una partición del espacio de Hilbert (ej. primera mitad vs resto)
        # En una arquitectura real, esto mapearía a QuantumAdmissionGate (P1) y HilbertObserverAgent (P2)
        mid = N // 2

        P1_diag = np.zeros(N)
        P1_diag[:mid] = 1.0
        P1 = np.diag(P1_diag)

        P2_diag = np.zeros(N)
        P2_diag[mid:] = 1.0
        P2 = np.diag(P2_diag)

        # Validación Axiomática: Resolución de Identidad
        identity_obs = np.eye(N)
        res_error = np.linalg.norm((P1 + P2) - identity_obs, ord=np.inf)
        if res_error > self._config.epsilon:
            raise TopologicalInvariantError(f"Violación de Resolución de Identidad: {res_error:.2e}")

        # Validación Axiomática: Exclusión Mutua
        mut_error = np.linalg.norm(P1 @ P2, ord=np.inf)
        if mut_error > self._config.epsilon:
            raise TopologicalInvariantError(f"Violación de Exclusión Mutua: {mut_error:.2e}")

        return P1 @ psi, P2 @ psi

    def calculate_wkb_transmission(self, energy: float, work_function: float) -> float:
        r"""
        Efecto Túnel Cuántico (WKB).
        Calcula la probabilidad de transmisión T si la energía incidente es menor que la función de trabajo.

        Axioma: T -> 0 ==> P1(psi) = 0 (Absorción de información en ausencia de amortiguamiento).
        """
        if energy >= work_function:
            return 1.0

        # Aproximación WKB simplificada para la barrera
        barrier_height = work_function - energy
        gamma = np.sqrt(barrier_height) # Simplificación del coeficiente de Gamow
        transmission = np.exp(-2 * gamma)

        return float(transmission)
