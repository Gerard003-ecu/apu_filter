# -*- coding: utf-8 -*-
r"""
=========================================================================================
Módulo: Quantum Algebra (Núcleo Axiomático de Mecánica Cuántica para Malla Agéntica)
Ubicación: app/core/quantum_algebra.py
Versión: 2.0.0 (Fase 2: Consagración Cuántica - Refactorización Rigurosa)

=========================================================================================
NATURALEZA CIBER-FÍSICA Y FUNDAMENTOS MATEMÁTICOS
=========================================================================================

Este módulo implementa el formalismo completo de la mecánica cuántica de sistemas
cerrados y abiertos para modelar el espacio de deliberación agéntica mediante
operadores actuando sobre espacios de Hilbert complejos.

§1. ESPACIO DE HILBERT $\mathcal{H}_N$ Y ESTRUCTURA MÉTRICA
────────────────────────────────────────────────────────────────────────────────────

Definición 1.1 (Espacio de Hilbert):
    $\mathcal{H}_N \cong \mathbb{C}^N$ equipado con el producto interno hermítico:
    $$ \langle \phi | \psi \rangle = \sum_{i=1}^{N} \overline{\phi_i} \psi_i $$
    
    donde $\overline{z}$ denota el conjugado complejo.

Definición 1.2 (Base Ortonormal Canónica):
    La base $\mathcal{B} = \{ |e_i\rangle \}_{i=1}^N$ satisface:
    $$ \langle e_i | e_j \rangle = \delta_{ij} $$
    
    donde $\delta_{ij}$ es el delta de Kronecker.

Teorema 1.3 (Matriz de Gram):
    Para una base ortonormal, la matriz de Gram $G \in \mathbb{C}^{N \times N}$ 
    definida como $G_{ij} = \langle e_i | e_j \rangle$ es la identidad:
    $$ G = I_N $$
    
    Prueba: Se construye explícitamente verificando que $G = B^{\dagger} B$ donde
    $B$ es la matriz cuyas columnas son los vectores de base.

Teorema 1.4 (Integridad del Espacio - Criterio de Rango):
    El espacio $\mathcal{H}_N$ es completo si y solo si:
    $$ \text{rank}(B) = N $$
    
    Verificación Numérica: Se utiliza SVD con tolerancia $\epsilon_{\text{mach}}$:
    $$ \text{rank}_{\epsilon}(B) = |\{ \sigma_i : \sigma_i > \epsilon \}| = N $$

§2. OPERADOR DE DENSIDAD $\rho$ (FORMALISMO DE VON NEUMANN)
────────────────────────────────────────────────────────────────────────────────────

Definición 2.1 (Operador de Densidad):
    El estado cuántico de un sistema se describe mediante un operador lineal
    $\rho: \mathcal{H}_N \to \mathcal{H}_N$ que satisface:
    
    (A1) Hermiticidad: $\rho = \rho^{\dagger}$
    (A2) Positividad Semidefinida: $\rho \geq 0$ (espectro no negativo)
    (A3) Traza Unitaria: $\text{Tr}(\rho) = 1$

Lema 2.2 (Consecuencias de Hermiticidad):
    Si $\rho = \rho^{\dagger}$, entonces:
    (i)  El espectro es real: $\lambda_i \in \mathbb{R}$
    (ii) Existe base ortonormal de autovectores: $\rho = \sum_i \lambda_i |i\rangle\langle i|$

Teorema 2.3 (Estados Puros vs Mixtos):
    Sea $\rho$ un operador de densidad. Entonces:
    
    (i)  $\rho$ es un estado puro $\iff$ $\rho^2 = \rho$ (idempotencia)
    (ii) $\rho$ es un estado mixto $\iff$ $\text{Tr}(\rho^2) < 1$
    
    Prueba: Para estado puro $|\psi\rangle$: $\rho = |\psi\rangle\langle\psi|$ 
            implica $\rho^2 = \rho$ por ortonormalidad.

§3. ENTROPÍA DE VON NEUMANN Y TERMODINÁMICA CUÁNTICA
────────────────────────────────────────────────────────────────────────────────────

Definición 3.1 (Entropía de Von Neumann):
    La entropía del estado cuántico $\rho$ se define como:
    $$ S(\rho) = -\text{Tr}(\rho \ln \rho) = -\sum_{i} \lambda_i \ln \lambda_i $$
    
    donde $\{\lambda_i\}$ son los autovalores de $\rho$ y por convención $0 \ln 0 = 0$.

Teorema 3.2 (Propiedades de la Entropía):
    (i)   $S(\rho) \geq 0$ con igualdad $\iff$ $\rho$ es un estado puro
    (ii)  $S(\rho) \leq \ln N$ con igualdad $\iff$ $\rho = I_N / N$ (estado maximal mixto)
    (iii) $S(\rho)$ es cóncava en el conjunto de estados cuánticos

§4. PROYECTORES ORTOGONALES Y MEDICIÓN CUÁNTICA
────────────────────────────────────────────────────────────────────────────────────

Definición 4.1 (Proyector Ortogonal):
    Un operador $P: \mathcal{H}_N \to \mathcal{H}_N$ es un proyector ortogonal si:
    (P1) $P^2 = P$ (idempotencia)
    (P2) $P^{\dagger} = P$ (hermiticidad)

Teorema 4.2 (Resolución de la Identidad - POVM):
    Una familia $\{P_k\}_{k=1}^m$ de proyectores forma una medición proyectiva si:
    $$ \sum_{k=1}^m P_k = I_N \quad \text{y} \quad P_i P_j = \delta_{ij} P_i $$

Corolario 4.3 (Conservación de Probabilidad):
    Para cualquier estado $|\psi\rangle$ y medición proyectiva $\{P_k\}$:
    $$ \sum_{k=1}^m \langle\psi|P_k|\psi\rangle = 1 $$

§5. APROXIMACIÓN WKB Y EFECTO TÚNEL
────────────────────────────────────────────────────────────────────────────────────

Teorema 5.1 (Coeficiente de Transmisión WKB):
    Para una barrera de potencial rectangular con altura $V_0$ y anchura $a$,
    la probabilidad de transmisión cuántica en el régimen $E < V_0$ es:
    
    $$ T \approx \exp\left(-2\int_{x_1}^{x_2} \kappa(x) \, dx\right) $$
    
    donde $\kappa(x) = \sqrt{2m(V(x) - E)/\hbar^2}$ es el número de onda imaginario.

Aproximación 5.2 (Barrera Rectangular):
    Para barrera rectangular de altura $\Phi = V_0 - E$ y anchura $a$:
    $$ T \approx \exp\left(-2a\sqrt{2m\Phi/\hbar^2}\right) = \exp(-2\gamma) $$
    
    donde $\gamma$ es el factor de Gamow.

=========================================================================================
INVARIANTES TOPOLÓGICOS Y ASERCIONES AXIOMÁTICAS
=========================================================================================

Los siguientes invariantes deben ser preservados bajo transformaciones del sistema:

(I1) Traza del Operador de Densidad: $\text{Tr}(\rho) = 1$ (conservación de probabilidad)
(I2) Positividad del Espectro: $\lambda_{\min}(\rho) \geq 0$ (coherencia física)
(I3) Hermiticidad: $\|\rho - \rho^{\dagger}\|_{\infty} \leq \epsilon_{\text{mach}}$
(I4) Rango del Espacio: $\text{rank}_{\epsilon}(B) = N$ (completitud)
(I5) Ortogonalidad: $\|B^{\dagger}B - I_N\|_{\infty} \leq \epsilon_{\text{mach}}$

=========================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from app.adapters.tools_interface import (
    MICConfiguration,
    MICRegistry,
    TopologicalInvariantError,
)

# =========================================================================================
# CONFIGURACIÓN DEL LOGGER
# =========================================================================================

logger: logging.Logger = logging.getLogger("MIC.Physics.QuantumAlgebra")

# =========================================================================================
# CONSTANTES FÍSICAS Y MATEMÁTICAS
# =========================================================================================

# Tolerancia numérica por defecto (épsilon de la máquina para float64)
EPSILON_MACHINE: Final[float] = 1e-12

# Constante de normalización para prevenir divisiones por cero
EPSILON_NORMALIZATION: Final[float] = 1e-15

# Umbral para clasificar autovalores como cero en cálculos entrópicos
EPSILON_ENTROPY: Final[float] = 1e-14


# =========================================================================================
# CLASE: HilbertSpace (Espacio de Hilbert con Validación Rigurosa)
# =========================================================================================


@dataclass(frozen=True, slots=True)
class HilbertSpace:
    r"""
    Representación inmutable del espacio de Hilbert complejo $\mathcal{H}_N \cong \mathbb{C}^N$.

    Este objeto encapsula la estructura geométrica del espacio vectorial complejo
    equipado con el producto interno hermítico, garantizando la integridad mediante
    validaciones algebraicas rigurosas.

    Attributes:
        dimension: Dimensión $N$ del espacio de Hilbert.
        basis: Matriz $B \in \mathbb{C}^{N \times N}$ cuyas columnas forman la base
               ortonormal $\{|e_i\rangle\}_{i=1}^N$.
        epsilon: Tolerancia numérica $\epsilon_{\text{mach}}$ para validaciones.

    Invariantes Verificados:
        (H1) Ortogonalidad: $\|B^{\dagger}B - I_N\|_{\infty} \leq \epsilon$
        (H2) Completitud: $\text{rank}_{\epsilon}(B) = N$

    Raises:
        TopologicalInvariantError: Si se viola algún invariante estructural.

    Examples:
        >>> hs = HilbertSpace.create_canonical(dimension=4)
        >>> hs.dimension
        4
        >>> hs.validate_orthonormality()  # No lanza excepción si es válido
    """

    dimension: int
    basis: NDArray[np.complex128] = field(repr=False)
    epsilon: float = EPSILON_MACHINE

    def __post_init__(self) -> None:
        """
        Validación post-construcción de los invariantes estructurales.

        Este método se invoca automáticamente tras la instanciación para
        garantizar la coherencia matemática del espacio de Hilbert.

        Validaciones Ejecutadas:
            1. Ortogonalidad Funcional (Matriz de Gram)
            2. Completitud del Espacio (Criterio de Rango)

        Raises:
            TopologicalInvariantError: Si alguna validación falla.
        """
        self._validate_orthonormality()
        self._validate_completeness()

    def _validate_orthonormality(self) -> None:
        r"""
        Verifica la ortogonalidad funcional mediante la matriz de Gram.

        Comprueba que:
        $$ G = B^{\dagger}B = I_N $$

        donde $G_{ij} = \langle e_i | e_j \rangle = \delta_{ij}$.

        Raises:
            TopologicalInvariantError: Si $\|G - I_N\|_{\infty} > \epsilon$.
        """
        gram_matrix: NDArray[np.complex128] = self.basis.conj().T @ self.basis
        identity_matrix: NDArray[np.complex128] = np.eye(
            self.dimension, dtype=np.complex128
        )

        deviation: float = float(
            np.linalg.norm(gram_matrix - identity_matrix, ord=np.inf)
        )

        if deviation > self.epsilon:
            logger.error(
                "Violación de Ortogonalidad: ||G - I||_∞ = %.4e > ε = %.4e",
                deviation,
                self.epsilon,
            )
            raise TopologicalInvariantError(
                f"Violación del Invariante (H1) - Ortogonalidad Funcional: "
                f"||B†B - I_N||_∞ = {deviation:.4e} > {self.epsilon:.4e}. "
                f"La matriz de Gram se desvía de la identidad, indicando que "
                f"la base no es ortonormal en el sentido del producto hermítico."
            )

        logger.debug(
            "Invariante (H1) verificado: Ortogonalidad funcional con ||G - I||_∞ = %.4e",
            deviation,
        )

    def _validate_completeness(self) -> None:
        r"""
        Verifica la completitud del espacio mediante el criterio de rango.

        Utiliza la Descomposición en Valores Singulares (SVD) para comprobar que:
        $$ \text{rank}_{\epsilon}(B) = |\{\sigma_i : \sigma_i > \epsilon\}| = N $$

        Raises:
            TopologicalInvariantError: Si el rango numérico es menor que $N$.
        """
        singular_values: NDArray[np.float64] = np.linalg.svd(
            self.basis, compute_uv=False
        )
        numerical_rank: int = int(np.sum(singular_values > self.epsilon))

        if numerical_rank != self.dimension:
            logger.error(
                "Deficiencia de Rango: rank_ε(B) = %d < N = %d",
                numerical_rank,
                self.dimension,
            )
            raise TopologicalInvariantError(
                f"Violación del Invariante (H2) - Completitud del Espacio: "
                f"rank_ε(B) = {numerical_rank} < N = {self.dimension}. "
                f"El tensor métrico presenta deficiencia de rango, lo que indica "
                f"dependencia lineal en los vectores de la base bajo tolerancia ε = {self.epsilon:.4e}. "
                f"Valores singulares: {singular_values}"
            )

        logger.debug(
            "Invariante (H2) verificado: Completitud con rank_ε(B) = %d",
            numerical_rank,
        )

    @classmethod
    def create_canonical(
        cls, dimension: int, epsilon: float = EPSILON_MACHINE
    ) -> HilbertSpace:
        r"""
        Constructor de la base ortonormal canónica (base computacional).

        Genera el espacio de Hilbert $\mathcal{H}_N$ con la base estándar:
        $$ |e_i\rangle = (0, \ldots, 0, 1, 0, \ldots, 0)^T $$
        donde el 1 está en la posición $i$-ésima.

        Args:
            dimension: Dimensión $N$ del espacio de Hilbert.
            epsilon: Tolerancia numérica para validaciones.

        Returns:
            Instancia validada de HilbertSpace.

        Raises:
            ValueError: Si dimension < 1.

        Examples:
            >>> hs = HilbertSpace.create_canonical(3)
            >>> hs.dimension
            3
            >>> np.allclose(hs.basis, np.eye(3))
            True
        """
        if dimension < 1:
            raise ValueError(
                f"La dimensión del espacio de Hilbert debe ser positiva: {dimension}"
            )

        canonical_basis: NDArray[np.complex128] = np.eye(
            dimension, dtype=np.complex128
        )

        logger.info(
            "Creando espacio de Hilbert canónico de dimensión N = %d con ε = %.2e",
            dimension,
            epsilon,
        )

        return cls(dimension=dimension, basis=canonical_basis, epsilon=epsilon)

    @property
    def metric_tensor(self) -> NDArray[np.complex128]:
        """
        Retorna la matriz de Gram (tensor métrico) $G = B^{\dagger}B$.

        Para una base ortonormal, debe coincidir con $I_N$.

        Returns:
            Matriz de Gram $G \in \mathbb{C}^{N \times N}$.
        """
        return self.basis.conj().T @ self.basis


# =========================================================================================
# CLASE: QuantumDensityOperator (Operador de Densidad con Axiomas de Von Neumann)
# =========================================================================================


@dataclass(frozen=True, slots=True)
class QuantumDensityOperator:
    r"""
    Operador de Densidad Cuántica $\rho \in \mathcal{L}(\mathcal{H}_N)$.

    Implementa el formalismo de von Neumann para estados cuánticos puros y mixtos,
    garantizando el cumplimiento de los tres axiomas fundamentales mediante
    validación rigurosa en tiempo de construcción.

    Attributes:
        rho: Matriz de densidad $\rho \in \mathbb{C}^{N \times N}$.
        epsilon: Tolerancia numérica para validaciones axiomáticas.

    Axiomas Verificados (Hard Vetoes):
        (A1) Hermiticidad: $\rho = \rho^{\dagger}$
        (A2) Positividad Semidefinida: $\lambda_{\min}(\rho) \geq 0$
        (A3) Traza Unitaria: $\text{Tr}(\rho) = 1$

    Raises:
        TopologicalInvariantError: Si se viola algún axioma cuántico.

    Examples:
        >>> psi = np.array([1, 0, 0], dtype=complex)
        >>> rho_op = QuantumDensityOperator.from_pure_state(psi)
        >>> rho_op.is_pure_state()
        True
    """

    rho: NDArray[np.complex128] = field(repr=False)
    epsilon: float = EPSILON_MACHINE

    def __post_init__(self) -> None:
        """
        Auditoría axiomática del operador de densidad.

        Ejecuta las tres validaciones fundamentales que definen un
        estado cuántico físicamente realizable.

        Raises:
            TopologicalInvariantError: Si algún axioma es violado.
        """
        self._validate_shape()
        self._validate_hermiticity()
        self._validate_positive_semidefiniteness()
        self._validate_unit_trace()

    def _validate_shape(self) -> None:
        """
        Verifica que $\rho$ sea una matriz cuadrada.

        Raises:
            TopologicalInvariantError: Si la matriz no es cuadrada.
        """
        if self.rho.ndim != 2 or self.rho.shape[0] != self.rho.shape[1]:
            raise TopologicalInvariantError(
                f"El operador de densidad debe ser una matriz cuadrada. "
                f"Forma recibida: {self.rho.shape}"
            )

    def _validate_hermiticity(self) -> None:
        r"""
        Verifica el Axioma (A1): Hermiticidad $\rho = \rho^{\dagger}$.

        Comprueba que:
        $$ \|\rho - \rho^{\dagger}\|_{\infty} \leq \epsilon $$

        Raises:
            TopologicalInvariantError: Si el residuo hermítico excede la tolerancia.
        """
        rho_dagger: NDArray[np.complex128] = self.rho.conj().T
        hermiticity_residue: float = float(
            np.linalg.norm(self.rho - rho_dagger, ord=np.inf)
        )

        if hermiticity_residue > self.epsilon:
            logger.error(
                "Violación de Hermiticidad: ||ρ - ρ†||_∞ = %.4e > ε = %.4e",
                hermiticity_residue,
                self.epsilon,
            )
            raise TopologicalInvariantError(
                f"Violación del Axioma (A1) - Hermiticidad: "
                f"||ρ - ρ†||_∞ = {hermiticity_residue:.4e} > {self.epsilon:.4e}. "
                f"El operador de densidad no es autoadjunto, lo que implica que "
                f"su espectro contiene componentes imaginarias no físicas."
            )

        logger.debug(
            "Axioma (A1) verificado: Hermiticidad con ||ρ - ρ†||_∞ = %.4e",
            hermiticity_residue,
        )

    def _validate_positive_semidefiniteness(self) -> None:
        r"""
        Verifica el Axioma (A2): Positividad Semidefinida $\rho \geq 0$.

        Comprueba que todos los autovalores sean no negativos:
        $$ \lambda_{\min}(\rho) \geq -\epsilon $$

        Raises:
            TopologicalInvariantError: Si existen autovalores negativos significativos.
        """
        eigenvalues: NDArray[np.float64] = np.linalg.eigvalsh(self.rho)
        lambda_min: float = float(np.min(eigenvalues))

        if lambda_min < -self.epsilon:
            logger.error(
                "Violación de Positividad: λ_min = %.4e < -ε = %.4e",
                lambda_min,
                -self.epsilon,
            )
            raise TopologicalInvariantError(
                f"Violación del Axioma (A2) - Positividad Semidefinida: "
                f"λ_min = {lambda_min:.4e} < -{self.epsilon:.4e}. "
                f"El operador presenta autovalores negativos, lo que corresponde "
                f"a probabilidades negativas (aberración topológica). "
                f"Espectro completo: {eigenvalues}"
            )

        logger.debug(
            "Axioma (A2) verificado: Positividad con λ_min = %.4e", lambda_min
        )

    def _validate_unit_trace(self) -> None:
        r"""
        Verifica el Axioma (A3): Traza Unitaria $\text{Tr}(\rho) = 1$.

        Comprueba que:
        $$ |\text{Tr}(\rho) - 1| \leq \epsilon $$

        Raises:
            TopologicalInvariantError: Si la traza se desvía de la unidad.
        """
        trace_value: complex = np.trace(self.rho)
        trace_residue: float = float(abs(trace_value - 1.0))

        if trace_residue > self.epsilon:
            logger.error(
                "Violación de Traza Unitaria: |Tr(ρ) - 1| = %.4e > ε = %.4e, Tr(ρ) = %.6f%+.6fj",
                trace_residue,
                self.epsilon,
                trace_value.real,
                trace_value.imag,
            )
            raise TopologicalInvariantError(
                f"Violación del Axioma (A3) - Traza Unitaria: "
                f"|Tr(ρ) - 1| = {trace_residue:.4e} > {self.epsilon:.4e}. "
                f"Tr(ρ) = {trace_value.real:.8f} + {trace_value.imag:.8f}j. "
                f"Fallo en la conservación de la probabilidad total (normalización)."
            )

        logger.debug(
            "Axioma (A3) verificado: Traza unitaria con |Tr(ρ) - 1| = %.4e",
            trace_residue,
        )

    @classmethod
    def from_pure_state(
        cls, psi: NDArray[np.complex128], epsilon: float = EPSILON_MACHINE
    ) -> QuantumDensityOperator:
        r"""
        Construye el operador de densidad a partir de un estado puro $|\psi\rangle$.

        Para un estado puro normalizado:
        $$ \rho = |\psi\rangle\langle\psi| $$

        Args:
            psi: Vector de estado $|\psi\rangle \in \mathbb{C}^N$.
            epsilon: Tolerancia numérica.

        Returns:
            Operador de densidad correspondiente al estado puro.

        Raises:
            TopologicalInvariantError: Si $\|\psi\| = 0$ (vector nulo).

        Examples:
            >>> psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
            >>> rho_op = QuantumDensityOperator.from_pure_state(psi)
            >>> np.isclose(np.trace(rho_op.rho), 1.0)
            True
        """
        norm: float = float(np.linalg.norm(psi))

        if norm < EPSILON_NORMALIZATION:
            raise TopologicalInvariantError(
                f"No se puede construir un operador de densidad a partir del vector nulo. "
                f"||ψ|| = {norm:.4e} < {EPSILON_NORMALIZATION:.4e}"
            )

        # Normalización del vector de estado
        psi_normalized: NDArray[np.complex128] = psi / norm

        # Construcción del producto externo |ψ⟩⟨ψ|
        rho_matrix: NDArray[np.complex128] = np.outer(
            psi_normalized, psi_normalized.conj()
        )

        logger.info(
            "Operador de densidad construido desde estado puro de dimensión N = %d",
            psi.shape[0],
        )

        return cls(rho=rho_matrix, epsilon=epsilon)

    @classmethod
    def from_mixed_state(
        cls,
        weights: List[float],
        states: List[NDArray[np.complex128]],
        epsilon: float = EPSILON_MACHINE,
    ) -> QuantumDensityOperator:
        r"""
        Construye el operador de densidad a partir de un ensamble estadístico (estado mixto).

        Para un ensamble con probabilidades $\{p_i\}$ y estados $\{|\psi_i\rangle\}$:
        $$ \rho = \sum_{i=1}^{k} p_i |\psi_i\rangle\langle\psi_i| $$

        donde $\sum_i p_i = 1$ y $p_i \geq 0$ para todo $i$.

        Args:
            weights: Lista de probabilidades $\{p_i\}$ (serán normalizadas).
            states: Lista de vectores de estado $\{|\psi_i\rangle\}$.
            epsilon: Tolerancia numérica.

        Returns:
            Operador de densidad correspondiente al estado mixto.

        Raises:
            ValueError: Si las listas tienen diferente longitud o están vacías.
            TopologicalInvariantError: Si la suma de pesos es cero o estados incompatibles.

        Examples:
            >>> psi1 = np.array([1, 0], dtype=complex)
            >>> psi2 = np.array([0, 1], dtype=complex)
            >>> rho_op = QuantumDensityOperator.from_mixed_state([0.5, 0.5], [psi1, psi2])
            >>> np.allclose(rho_op.rho, np.eye(2) / 2)
            True
        """
        if len(weights) != len(states):
            raise ValueError(
                f"Las listas de pesos y estados deben tener la misma longitud. "
                f"Recibido: {len(weights)} pesos y {len(states)} estados."
            )

        if len(weights) == 0:
            raise ValueError("El ensamble no puede estar vacío.")

        # Normalización de las probabilidades
        weight_sum: float = sum(weights)
        if abs(weight_sum) < EPSILON_NORMALIZATION:
            raise TopologicalInvariantError(
                f"La suma de pesos no puede ser cero: Σp_i = {weight_sum:.4e}"
            )

        normalized_probabilities: List[float] = [w / weight_sum for w in weights]

        # Verificación de consistencia dimensional
        dimension: int = states[0].shape[0]
        for idx, state in enumerate(states):
            if state.shape[0] != dimension:
                raise TopologicalInvariantError(
                    f"Inconsistencia dimensional en el estado {idx}: "
                    f"esperado {dimension}, recibido {state.shape[0]}"
                )

        # Construcción del operador de densidad mixto
        rho_accumulator: NDArray[np.complex128] = np.zeros(
            (dimension, dimension), dtype=np.complex128
        )

        for prob, state in zip(normalized_probabilities, states):
            norm: float = float(np.linalg.norm(state))

            if norm > EPSILON_NORMALIZATION:
                state_normalized: NDArray[np.complex128] = state / norm
                rho_accumulator += prob * np.outer(
                    state_normalized, state_normalized.conj()
                )
            else:
                logger.warning(
                    "Estado con norma casi nula ignorado en el ensamble: ||ψ|| = %.4e",
                    norm,
                )

        logger.info(
            "Operador de densidad construido desde ensamble mixto de %d estados (dim N = %d)",
            len(states),
            dimension,
        )

        return cls(rho=rho_accumulator, epsilon=epsilon)

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """
        Retorna los autovalores del operador de densidad ordenados de forma ascendente.

        Los autovalores representan las probabilidades de ocupación de los
        autoestados del sistema.

        Returns:
            Array de autovalores $\{\lambda_i\}$ ordenados.

        Note:
            Para operadores hermíticos, eigvalsh es más eficiente que eig.
        """
        return np.linalg.eigvalsh(self.rho)

    @property
    def eigensystem(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """
        Retorna el sistema completo de autovalores y autovectores.

        Returns:
            Tupla (autovalores, autovectores) donde autovectores[:, i] corresponde
            al autovalor autovalores[i].

        Note:
            La descomposición espectral permite escribir:
            ρ = Σ_i λ_i |i⟩⟨i|
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.rho)
        return eigenvalues, eigenvectors

    def compute_von_neumann_entropy(self) -> float:
        r"""
        Calcula la entropía de von Neumann del estado cuántico.

        La entropía cuantifica el grado de mezcla del estado:
        $$ S(\rho) = -\text{Tr}(\rho \ln \rho) = -\sum_{i} \lambda_i \ln \lambda_i $$

        Propiedades:
            - $S(\rho) = 0$ $\iff$ estado puro ($\rho^2 = \rho$)
            - $S(\rho) = \ln N$ $\iff$ estado maximal mixto ($\rho = I_N / N$)
            - $0 \leq S(\rho) \leq \ln N$

        Returns:
            Entropía de von Neumann en unidades de nats (logaritmo natural).

        Note:
            Se utiliza la convención $0 \ln 0 = 0$ (límite termodinámico).

        Examples:
            >>> psi = np.array([1, 0], dtype=complex)
            >>> rho_pure = QuantumDensityOperator.from_pure_state(psi)
            >>> rho_pure.compute_von_neumann_entropy()
            0.0
        """
        spectrum: NDArray[np.float64] = self.eigenvalues

        # Filtrar autovalores significativamente positivos (evitar log(0))
        positive_eigenvalues: NDArray[np.float64] = spectrum[
            spectrum > EPSILON_ENTROPY
        ]

        if len(positive_eigenvalues) == 0:
            logger.warning(
                "Todos los autovalores son despreciables (< %.2e). Entropía = 0.",
                EPSILON_ENTROPY,
            )
            return 0.0

        # Cálculo de la entropía: -Σ λ_i ln(λ_i)
        entropy: float = -float(np.sum(positive_eigenvalues * np.log(positive_eigenvalues)))

        logger.debug("Entropía de von Neumann calculada: S(ρ) = %.6f nats", entropy)

        return entropy

    def is_pure_state(self) -> bool:
        r"""
        Determina si el operador representa un estado puro.

        Criterio: $S(\rho) \leq \epsilon$ $\iff$ $\rho^2 \approx \rho$

        Returns:
            True si el estado es puro, False si es mixto.

        Note:
            Alternativamente, se podría verificar Tr(ρ²) ≈ 1, pero
            el criterio entrópico es más robusto numéricamente.

        Examples:
            >>> psi = np.array([1/np.sqrt(3)] * 3, dtype=complex)
            >>> rho_pure = QuantumDensityOperator.from_pure_state(psi)
            >>> rho_pure.is_pure_state()
            True
        """
        entropy: float = self.compute_von_neumann_entropy()
        is_pure: bool = entropy <= self.epsilon

        logger.debug(
            "Clasificación de estado: %s (S(ρ) = %.6e)",
            "PURO" if is_pure else "MIXTO",
            entropy,
        )

        return is_pure

    def compute_purity(self) -> float:
        r"""
        Calcula la pureza del estado cuántico.

        La pureza se define como:
        $$ \gamma = \text{Tr}(\rho^2) = \sum_{i} \lambda_i^2 $$

        Propiedades:
            - $\gamma = 1$ $\iff$ estado puro
            - $\gamma = 1/N$ $\iff$ estado maximal mixto
            - $1/N \leq \gamma \leq 1$

        Returns:
            Pureza del estado en el rango [1/N, 1].

        Note:
            La pureza está relacionada con la entropía mediante desigualdades
            de convexidad, pero proporciona información complementaria.

        Examples:
            >>> psi = np.array([1, 0, 0], dtype=complex)
            >>> rho = QuantumDensityOperator.from_pure_state(psi)
            >>> rho.compute_purity()
            1.0
        """
        spectrum: NDArray[np.float64] = self.eigenvalues
        purity: float = float(np.sum(spectrum**2))

        logger.debug("Pureza del estado calculada: γ = Tr(ρ²) = %.6f", purity)

        return purity


# =========================================================================================
# CLASE: QuantumRegistry (Núcleo Cuántico del MIC)
# =========================================================================================


class QuantumRegistry(MICRegistry):
    r"""
    Registro Cuántico del Marco de Interoperabilidad Categórica (MIC).

    Extiende el MICRegistry base para incorporar el formalismo de la mecánica
    cuántica en el núcleo de la deliberación agéntica. El sistema se modela
    mediante un operador de densidad $\rho$ actuando sobre un espacio de Hilbert
    $\mathcal{H}_N$, trascendiendo la lógica booleana clásica.

    Arquitectura Matemática:
        - Estado del Sistema: $\rho \in \mathcal{L}(\mathcal{H}_N)$
        - Espacio de Configuración: $\mathcal{H}_N \cong \mathbb{C}^N$
        - Dinámica: Evoluciones unitarias y proyecciones de medición

    Attributes:
        _hilbert_space: Espacio de Hilbert subyacente $\mathcal{H}_N$.
        _state: Operador de densidad actual $\rho$.

    Examples:
        >>> rho_matrix = np.eye(3, dtype=complex) / 3  # Estado maximal mixto
        >>> registry = QuantumRegistry(rho=rho_matrix)
        >>> registry.quantum_state.compute_von_neumann_entropy()
        1.0986122886681098  # ln(3)
    """

    __slots__ = ("_hilbert_space", "_state")

    def __init__(
        self,
        rho: NDArray[np.complex128],
        config: Optional[MICConfiguration] = None,
    ) -> None:
        r"""
        Constructor del Registro Cuántico con auditoría axiomática completa.

        El proceso de construcción ejecuta las siguientes fases:

        1. Inicialización del MICRegistry base
        2. Construcción y validación del espacio de Hilbert $\mathcal{H}_N$
        3. Construcción y validación del operador de densidad $\rho$
        4. Verificación de consistencia dimensional

        Args:
            rho: Matriz de densidad inicial $\rho \in \mathbb{C}^{N \times N}$.
            config: Configuración MIC (opcional, usa valores por defecto si es None).

        Raises:
            TopologicalInvariantError: Si $\rho$ viola algún axioma cuántico o
                                       hay inconsistencia dimensional.

        Note:
            La auditoría axiomática es HARD VETO: cualquier violación aborta
            la construcción del objeto mediante excepción.
        """
        super().__init__(config=config)

        dimension: int = rho.shape[0]

        # Fase 1: Construcción del Espacio de Hilbert con Base Canónica
        logger.info(
            "Inicializando espacio de Hilbert de dimensión N = %d con ε = %.2e",
            dimension,
            self._config.epsilon,
        )

        self._hilbert_space: HilbertSpace = HilbertSpace.create_canonical(
            dimension=dimension, epsilon=self._config.epsilon
        )

        # Fase 2: Construcción del Operador de Densidad con Auditoría Axiomática
        logger.info(
            "Construyendo operador de densidad y ejecutando auditoría axiomática..."
        )

        self._state: QuantumDensityOperator = QuantumDensityOperator(
            rho=rho, epsilon=self._config.epsilon
        )

        # Fase 3: Verificación de Consistencia Dimensional
        if self._state.rho.shape[0] != self._hilbert_space.dimension:
            raise TopologicalInvariantError(
                f"Inconsistencia dimensional: dim(ρ) = {self._state.rho.shape[0]} "
                f"≠ dim(H) = {self._hilbert_space.dimension}"
            )

        logger.info(
            "✓ QuantumRegistry inicializado exitosamente. "
            "Espacio H_%d certificado bajo ε = %.2e. "
            "Estado: %s (S = %.4f nats, γ = %.4f)",
            self._hilbert_space.dimension,
            self._config.epsilon,
            "PURO" if self._state.is_pure_state() else "MIXTO",
            self._state.compute_von_neumann_entropy(),
            self._state.compute_purity(),
        )

    @property
    def quantum_state(self) -> QuantumDensityOperator:
        """
        Retorna el operador de densidad actual del sistema.

        Returns:
            Operador de densidad $\rho$ que describe el estado cuántico.
        """
        return self._state

    @property
    def hilbert_space(self) -> HilbertSpace:
        """
        Retorna el espacio de Hilbert subyacente.

        Returns:
            Espacio de Hilbert $\mathcal{H}_N$ del sistema.
        """
        return self._hilbert_space

    def apply_observational_projectors(
        self, psi: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        Aplica proyectores ortogonales complementarios a un vector de estado.

        Implementa una medición proyectiva con dos resultados posibles,
        modelando el Quantum Admission Gate y el Hilbert Observer Agent.

        Construcción de los Proyectores:
            Sea $N$ la dimensión del espacio. Definimos una partición:
            - $P_1$: Proyector sobre el subespacio $\text{span}\{|e_i\rangle\}_{i=1}^{N/2}$
            - $P_2$: Proyector sobre el subespacio $\text{span}\{|e_i\rangle\}_{i=N/2+1}^{N}$

        Axiomas Verificados:
            (R1) Resolución de la Identidad: $P_1 + P_2 = I_N$
            (R2) Exclusión Mutua: $P_1 P_2 = 0$

        Args:
            psi: Vector de estado $|\psi\rangle \in \mathcal{H}_N$ a proyectar.

        Returns:
            Tupla $(P_1|\psi\rangle, P_2|\psi\rangle)$ de estados proyectados.

        Raises:
            TopologicalInvariantError: Si los proyectores violan los axiomas.
            ValueError: Si el vector tiene dimensión incompatible.

        Note:
            En una implementación física, $P_1$ y $P_2$ corresponderían a
            detectores complementarios o canales de medición ortogonales.

        Examples:
            >>> registry = QuantumRegistry(rho=np.eye(4, dtype=complex) / 4)
            >>> psi = np.array([1, 1, 1, 1], dtype=complex) / 2
            >>> proj1, proj2 = registry.apply_observational_projectors(psi)
            >>> np.allclose(np.linalg.norm(proj1)**2 + np.linalg.norm(proj2)**2, 1.0)
            True
        """
        N: int = self._hilbert_space.dimension

        if psi.shape[0] != N:
            raise ValueError(
                f"Vector de estado tiene dimensión incompatible: "
                f"esperado {N}, recibido {psi.shape[0]}"
            )

        # Construcción de los proyectores mediante partición del espacio
        mid_point: int = N // 2

        # P1: Proyector sobre el primer subespacio
        P1_diagonal: NDArray[np.float64] = np.zeros(N)
        P1_diagonal[:mid_point] = 1.0
        P1: NDArray[np.float64] = np.diag(P1_diagonal)

        # P2: Proyector sobre el segundo subespacio (complemento ortogonal)
        P2_diagonal: NDArray[np.float64] = np.zeros(N)
        P2_diagonal[mid_point:] = 1.0
        P2: NDArray[np.float64] = np.diag(P2_diagonal)

        # Validación Axiomática (R1): Resolución de la Identidad
        identity_obs: NDArray[np.float64] = np.eye(N)
        resolution_error: float = float(
            np.linalg.norm((P1 + P2) - identity_obs, ord=np.inf)
        )

        if resolution_error > self._config.epsilon:
            raise TopologicalInvariantError(
                f"Violación del Axioma (R1) - Resolución de la Identidad: "
                f"||P₁ + P₂ - I||_∞ = {resolution_error:.4e} > {self._config.epsilon:.4e}"
            )

        # Validación Axiomática (R2): Exclusión Mutua
        mutual_exclusion_error: float = float(np.linalg.norm(P1 @ P2, ord=np.inf))

        if mutual_exclusion_error > self._config.epsilon:
            raise TopologicalInvariantError(
                f"Violación del Axioma (R2) - Exclusión Mutua: "
                f"||P₁P₂||_∞ = {mutual_exclusion_error:.4e} > {self._config.epsilon:.4e}"
            )

        logger.debug(
            "Proyectores validados: ||P₁+P₂-I||_∞ = %.2e, ||P₁P₂||_∞ = %.2e",
            resolution_error,
            mutual_exclusion_error,
        )

        # Aplicación de los proyectores al vector de estado
        projected_state_1: NDArray[np.complex128] = P1 @ psi
        projected_state_2: NDArray[np.complex128] = P2 @ psi

        # Cálculo de las probabilidades de medición (Born rule)
        prob_1: float = float(np.linalg.norm(projected_state_1) ** 2)
        prob_2: float = float(np.linalg.norm(projected_state_2) ** 2)

        logger.info(
            "Proyección aplicada: P(subespacio 1) = %.4f, P(subespacio 2) = %.4f",
            prob_1,
            prob_2,
        )

        return projected_state_1, projected_state_2

    def calculate_wkb_transmission(
        self, energy: float, work_function: float, barrier_width: float = 1.0
    ) -> float:
        r"""
        Calcula el coeficiente de transmisión cuántica mediante aproximación WKB.

        Aproximación WKB (Wentzel-Kramers-Brillouin):
            Para una barrera de potencial rectangular con altura $\Phi = V_0 - E$
            y anchura $a$, el coeficiente de transmisión en el régimen túnel es:

            $$ T \approx \exp(-2\gamma) $$

            donde el factor de Gamow es:
            $$ \gamma = a \sqrt{2m\Phi/\hbar^2} $$

        Regímenes Físicos:
            - Si $E \geq V_0$: Transmisión clásica $T = 1$ (sobre la barrera)
            - Si $E < V_0$: Tunelamiento cuántico $T < 1$ (a través de la barrera)

        Args:
            energy: Energía incidente $E$ de la partícula.
            work_function: Función de trabajo $V_0$ (altura de la barrera).
            barrier_width: Anchura $a$ de la barrera (en unidades reducidas).

        Returns:
            Coeficiente de transmisión $T \in [0, 1]$.

        Note:
            En unidades naturales ($\hbar = 1$, $m = 1$), la expresión se simplifica.
            Para recuperar unidades SI, se debe reescalar apropiadamente.

        Examples:
            >>> registry = QuantumRegistry(rho=np.eye(2, dtype=complex) / 2)
            >>> T_classical = registry.calculate_wkb_transmission(energy=10.0, work_function=5.0)
            >>> T_classical
            1.0
            >>> T_tunnel = registry.calculate_wkb_transmission(energy=1.0, work_function=5.0)
            >>> 0.0 < T_tunnel < 1.0
            True
        """
        # Régimen Clásico: Energía suficiente para superar la barrera
        if energy >= work_function:
            logger.debug(
                "Régimen clásico: E = %.4f ≥ Φ = %.4f → T = 1.0",
                energy,
                work_function,
            )
            return 1.0

        # Régimen Cuántico: Tunelamiento a través de la barrera
        barrier_height: float = work_function - energy

        # Factor de Gamow (en unidades naturales: ℏ = m = 1)
        # γ = a√(2mΔΦ/ℏ²) = a√(ΔΦ) en unidades reducidas
        gamow_factor: float = barrier_width * np.sqrt(barrier_height)

        # Coeficiente de transmisión WKB: T ≈ exp(-2γ)
        transmission_coefficient: float = float(np.exp(-2.0 * gamow_factor))

        logger.debug(
            "Régimen túnel: E = %.4f < Φ = %.4f, γ = %.4f → T = %.6e",
            energy,
            work_function,
            gamow_factor,
            transmission_coefficient,
        )

        return transmission_coefficient

    def evolve_unitary(self, unitary: NDArray[np.complex128]) -> None:
        r"""
        Evoluciona el estado cuántico mediante un operador unitario.

        Aplica la transformación:
        $$ \rho' = U \rho U^{\dagger} $$

        donde $U$ es un operador unitario ($U^{\dagger}U = I$).

        Args:
            unitary: Operador unitario $U \in \mathbb{C}^{N \times N}$.

        Raises:
            ValueError: Si U no tiene la dimensión correcta.
            TopologicalInvariantError: Si U no es unitario o ρ' viola axiomas.

        Note:
            La evolución unitaria preserva la pureza del estado y conserva
            la entropía de von Neumann.

        Examples:
            >>> rho = np.eye(2, dtype=complex) / 2
            >>> registry = QuantumRegistry(rho=rho)
            >>> U = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X
            >>> registry.evolve_unitary(U)
            >>> # Estado permanece invariante (estado maximal mixto)
        """
        N: int = self._hilbert_space.dimension

        if unitary.shape != (N, N):
            raise ValueError(
                f"Operador unitario tiene dimensión incompatible: "
                f"esperado ({N}, {N}), recibido {unitary.shape}"
            )

        # Verificación de unitariedad: U†U = I
        unitarity_residue: float = float(
            np.linalg.norm(
                unitary.conj().T @ unitary - np.eye(N, dtype=np.complex128),
                ord=np.inf,
            )
        )

        if unitarity_residue > self._config.epsilon:
            raise TopologicalInvariantError(
                f"El operador no es unitario: ||U†U - I||_∞ = {unitarity_residue:.4e}"
            )

        # Evolución del estado: ρ' = UρU†
        rho_evolved: NDArray[np.complex128] = (
            unitary @ self._state.rho @ unitary.conj().T
        )

        # Actualización del estado con validación axiomática
        self._state = QuantumDensityOperator(
            rho=rho_evolved, epsilon=self._config.epsilon
        )

        logger.info(
            "Evolución unitaria aplicada. Nuevo estado: %s (S = %.4f nats)",
            "PURO" if self._state.is_pure_state() else "MIXTO",
            self._state.compute_von_neumann_entropy(),
        )


# =========================================================================================
# FIN DEL MÓDULO
# =========================================================================================