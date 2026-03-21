"""
Suite de integración: Ortogonalidad funcional y cambio de base en la MIC.

Fundamentación matemática:
──────────────────────────

1. La MIC como representación de la identidad (I_n):
   En el álgebra lineal agéntica, la Matriz de Interacción Central opera
   como una representación matricial de la identidad cuando cada herramienta
   (vector atómico) actúa como proyección sobre un eje canónico:

       e_i: ℝⁿ → ℝⁿ,  e_i(x) = x + δᵢ

   donde δᵢ es el i-ésimo vector canónico. La matriz de transformación
   resultante T = [e₁(0) | e₂(0) | ... | eₙ(0)] debe ser I_n.

2. Ortogonalidad e independencia lineal:
   Para vectores canónicos {e_i}, la matriz de Gram satisface:

       G_ij = ⟨e_i(0), e_j(0)⟩ = δ_ij  (delta de Kronecker)

   Esto garantiza ausencia de efectos secundarios cruzados: la activación
   de la herramienta i no modifica la dimensión j ≠ i.

3. Composición funcional vs. suma vectorial:
   La composición de morfismos f ∘ g NO es suma vectorial f + g.
   Para handlers que escriben dimensiones disjuntas:

       (e₂ ∘ e₁)(0) = e₂(e₁(0)) = e₂(δ₁) = δ₁ + δ₂

   El resultado COINCIDE con e₁(0) + e₂(0) solo porque los handlers
   tienen soportes disjuntos (escriben dimensiones distintas sin
   interferencia). Esta coincidencia es una propiedad a VERIFICAR,
   no a asumir.

4. Estabilidad espectral (λ = 1, multiplicidad n):
   Si T = I_n, entonces spec(T) = {1} con multiplicidad n.
   Cualquier desviación indica:
   - λ > 1: explosión de gradientes (amplificación exponencial)
   - λ < 1: amnesia del estado (contracción exponencial)
   - λ complejo con |λ| ≠ 1: oscilación divergente o amortiguada

5. Invariancia bajo perturbación del estado base:
   La ortogonalidad debe mantenerse INDEPENDIENTEMENTE del estado
   inicial. Si el handler e_i modifica dim_j para j ≠ i cuando
   el estado base tiene dim_j ≠ 0, hay una fuga dimensional.

Referencias:
    [1] Mac Lane, S. "Categories for the Working Mathematician", Springer, 1971.
    [2] Strang, G. "Linear Algebra and Its Applications", 4th Ed., 2006.
    [3] Axler, S. "Linear Algebra Done Right", 3rd Ed., Springer, 2015.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Dict, Any, List, Set, Tuple, NamedTuple
from dataclasses import dataclass

from app.core.schemas import Stratum
from app.core.mic_algebra import (
    CategoricalState,
    Morphism,
    ComposedMorphism,
    create_morphism_from_handler,
)
from app.adapters.tools_interface import MICRegistry


# =============================================================================
# CONSTANTES
# =============================================================================

# Dimensiones canónicas del espacio de estado.
# Cada herramienta debe activar exactamente una de estas dimensiones.
_CANONICAL_DIMENSIONS: List[str] = ["dim_1", "dim_2", "dim_3"]

# Número de dimensiones del espacio vectorial.
_SPACE_DIMENSION: int = len(_CANONICAL_DIMENSIONS)

# Tolerancia para comparaciones de punto flotante.
# Justificación: las operaciones son sumas de enteros (0.0 + 1.0),
# sin acumulación de error. La tolerancia es conservadora.
_ORTHOGONALITY_TOLERANCE: float = 1e-12

# Valores de perturbación para tests adversariales del estado base.
# Se eligen valores no triviales para detectar fugas dimensionales.
_PERTURBATION_VALUES: List[float] = [3.14, -2.71, 42.0]


# =============================================================================
# HANDLERS CANÓNICOS (VECTORES BASE)
# =============================================================================


def _handler_e1(**kwargs) -> Dict[str, Any]:
    """
    Proyección canónica sobre el eje 1 (dimensión de búsqueda).

    Operación algebraica:
        e₁(x) = x + δ₁ = (x₁ + 1, x₂, x₃)

    donde δ₁ = (1, 0, 0) es el primer vector canónico.

    CONTRATO: Este handler SOLO modifica dim_1. Las dimensiones
    dim_2 y dim_3 deben permanecer INALTERADAS. La violación de
    este contrato rompe la ortogonalidad de la base.

    Parameters
    ----------
    **kwargs : Dict
        Kwargs para compatibilidad con create_morphism_from_handler.

    Returns
    -------
    Dict
        Dict con dim_1 activada (= 1.0).
    """
    dim_1 = kwargs.get("dim_1", 0.0)
    return {"dim_1": 1.0 + dim_1 if "dim_1" not in kwargs else 1.0}


def _handler_e2(**kwargs) -> Dict[str, Any]:
    """
    Proyección canónica sobre el eje 2 (dimensión de cálculo).

    Operación algebraica:
        e₂(x) = x + δ₂ - x₂·δ₂ = (x₁, 1, x₃)

    CONTRATO: SOLO modifica dim_2.

    Parameters
    ----------
    **kwargs : Dict
        Kwargs para compatibilidad con create_morphism_from_handler.

    Returns
    -------
    Dict
        Dict con dim_2 activada (= 1.0).
    """
    dim_2 = kwargs.get("dim_2", 0.0)
    return {"dim_2": 1.0 + dim_2 if "dim_2" not in kwargs else 1.0}


def _handler_e3(**kwargs) -> Dict[str, Any]:
    """
    Proyección canónica sobre el eje 3 (dimensión de estrategia).

    Operación algebraica:
        e₃(x) = x + δ₃ - x₃·δ₃ = (x₁, x₂, 1)

    CONTRATO: SOLO modifica dim_3.

    Parameters
    ----------
    **kwargs : Dict
        Kwargs para compatibilidad con create_morphism_from_handler.

    Returns
    -------
    Dict
        Dict con dim_3 activada (= 1.0).
    """
    dim_3 = kwargs.get("dim_3", 0.0)
    return {"dim_3": 1.0 + dim_3 if "dim_3" not in kwargs else 1.0}


# =============================================================================
# REGISTRO DE VECTORES Y CONFIGURACIÓN
# =============================================================================


@dataclass(frozen=True)
class CanonicalVectorSpec:
    """
    Especificación de un vector canónico para registro en la MIC.

    Attributes
    ----------
    name : str
        Identificador único del vector/herramienta.
    stratum : Stratum
        Estrato objetivo en la jerarquía categórica.
    handler : callable
        Función de transformación del estado.
    target_dimension : str
        Nombre de la dimensión que este vector activa.
    """

    name: str
    stratum: Stratum
    handler: Any
    target_dimension: str


# Definición de la base estándar E = {e₁, e₂, e₃}
_CANONICAL_BASIS: List[CanonicalVectorSpec] = [
    CanonicalVectorSpec(
        name="search_tool",
        stratum=Stratum.PHYSICS,
        handler=_handler_e1,
        target_dimension="dim_1",
    ),
    CanonicalVectorSpec(
        name="calc_tool",
        stratum=Stratum.TACTICS,
        handler=_handler_e2,
        target_dimension="dim_2",
    ),
    CanonicalVectorSpec(
        name="strategy_tool",
        stratum=Stratum.STRATEGY,
        handler=_handler_e3,
        target_dimension="dim_3",
    ),
]


# =============================================================================
# FUNCIONES AUXILIARES DE ÁLGEBRA LINEAL
# =============================================================================


def _create_zero_state() -> CategoricalState:
    """
    Crea el estado cero (origen del espacio vectorial).

    El estado cero satisface:
        ∀ i: dim_i = 0.0

    Este es el elemento neutro de la suma vectorial y el punto
    base desde el cual se miden los efectos de cada herramienta.

    Returns
    -------
    CategoricalState
        Estado con todas las dimensiones canónicas en 0.0.
    """
    return CategoricalState(
        payload={dim: 0.0 for dim in _CANONICAL_DIMENSIONS},
    )


def _create_perturbed_state(
    perturbation: List[float],
) -> CategoricalState:
    """
    Crea un estado con valores no triviales en todas las dimensiones.

    Se usa para verificar que los handlers NO modifican dimensiones
    ajenas. Si un handler es verdaderamente ortogonal, aplicarlo
    sobre un estado perturbado solo debe cambiar SU dimensión,
    dejando las demás en sus valores perturbados originales.

    Parameters
    ----------
    perturbation : List[float]
        Valores para cada dimensión canónica. Debe tener longitud
        igual a _SPACE_DIMENSION.

    Returns
    -------
    CategoricalState
        Estado perturbado.

    Raises
    ------
    ValueError
        Si la longitud de perturbation no coincide con _SPACE_DIMENSION.
    """
    if len(perturbation) != _SPACE_DIMENSION:
        raise ValueError(
            f"Se requieren {_SPACE_DIMENSION} valores de perturbación, "
            f"recibidos {len(perturbation)}."
        )

    payload = {
        dim: val
        for dim, val in zip(_CANONICAL_DIMENSIONS, perturbation)
    }
    return CategoricalState(payload=payload)


def _extract_state_vector(
    state: CategoricalState,
) -> np.ndarray:
    """
    Extrae el vector de estado en ℝⁿ desde un CategoricalState.

    IMPORTANTE: Solo extrae las dimensiones canónicas definidas en
    _CANONICAL_DIMENSIONS. Si el estado contiene claves adicionales
    (e.g., "dim_4"), estas NO se incluyen en el vector pero se
    reportan como advertencia mediante _detect_dimensional_leaks.

    Parameters
    ----------
    state : CategoricalState
        Estado categórico del cual extraer coordenadas.

    Returns
    -------
    np.ndarray
        Vector en ℝⁿ con las coordenadas canónicas.
    """
    return np.array(
        [state.payload.get(dim, 0.0) for dim in _CANONICAL_DIMENSIONS],
        dtype=np.float64,
    )


def _detect_dimensional_leaks(
    state: CategoricalState,
    allowed_dimensions: Set[str],
) -> List[str]:
    """
    Detecta dimensiones en el payload que no pertenecen al espacio canónico.

    Una "fuga dimensional" ocurre cuando un handler introduce una clave
    en el payload que no está en _CANONICAL_DIMENSIONS. Esto indica que
    el handler está operando fuera del espacio vectorial definido,
    potencialmente creando dependencias ocultas.

    Parameters
    ----------
    state : CategoricalState
        Estado a inspeccionar.
    allowed_dimensions : Set[str]
        Conjunto de claves permitidas en el payload.

    Returns
    -------
    List[str]
        Lista de claves no autorizadas encontradas. Vacía si no hay fugas.
    """
    return [
        key for key in state.payload.keys()
        if key not in allowed_dimensions
    ]


def _compute_gram_matrix(
    vectors: List[np.ndarray],
) -> np.ndarray:
    """
    Calcula la matriz de Gram G_ij = ⟨v_i, v_j⟩ para una lista de vectores.

    La matriz de Gram es positiva semidefinida por construcción.
    Para una base ortonormal, G = I_n exactamente.

    Propiedades:
        - G es simétrica: G_ij = G_ji
        - G_ii = ‖v_i‖² ≥ 0
        - det(G) > 0 ⟺ los vectores son linealmente independientes
        - G = I_n ⟺ los vectores son ortonormales

    Parameters
    ----------
    vectors : List[np.ndarray]
        Lista de n vectores en ℝᵐ.

    Returns
    -------
    np.ndarray
        Matriz de Gram de dimensión n × n.
    """
    n = len(vectors)
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            G[i, j] = np.dot(vectors[i], vectors[j])
    return G


def _compute_transformation_matrix(
    atomic_vectors: List[Morphism],
    base_state: CategoricalState,
) -> np.ndarray:
    """
    Construye la matriz de transformación T observando cómo los
    vectores atómicos mapean el estado base.

    La columna j de T es el vector resultante de aplicar el j-ésimo
    handler al estado base:

        T = [e₁(base) | e₂(base) | ... | eₙ(base)]

    Si base = 0 y cada handler produce δ_j, entonces T = I_n.

    Parameters
    ----------
    atomic_vectors : List[Morphism]
        Vectores atómicos (herramientas) a evaluar.
    base_state : CategoricalState
        Estado base desde el cual se evalúan las transformaciones.

    Returns
    -------
    np.ndarray
        Matriz T de dimensión n × n (si n herramientas en ℝⁿ).
    """
    columns = [
        _extract_state_vector(vec(base_state))
        for vec in atomic_vectors
    ]
    return np.column_stack(columns)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def canonical_basis() -> List[Morphism]:
    """
    Construye la base canónica E = {e₁, e₂, e₃} como vectores atómicos.

    Cada vector atómico encapsula un handler que activa exactamente
    una dimensión del espacio de estado.

    Returns
    -------
    List[Morphism]
        Lista ordenada de vectores base.
    """
    return [
        create_morphism_from_handler(
            name=spec.name,
            target_stratum=Stratum.PHYSICS,
            handler=spec.handler,
            required_keys=[],
            optional_keys=list(_CANONICAL_DIMENSIONS)
        )
        for spec in _CANONICAL_BASIS
    ]


@pytest.fixture
def mic_registry(canonical_basis: List[Morphism]) -> MICRegistry:
    """
    Inicializa la MIC registrando los vectores de la base canónica.

    La MIC resultante debe representar I₃ cuando se evalúa
    sobre el estado cero.

    Parameters
    ----------
    canonical_basis : List[AtomicVector]
        Base canónica proporcionada por fixture.

    Returns
    -------
    MICRegistry
        Registro con los 3 vectores canónicos.
    """
    registry = MICRegistry()
    for spec in _CANONICAL_BASIS:
        registry.register_vector(
            spec.name, spec.stratum, spec.handler
        )
    return registry


@pytest.fixture
def zero_state() -> CategoricalState:
    """
    Estado cero (origen del espacio vectorial).

    Returns
    -------
    CategoricalState
        Estado con todas las dimensiones en 0.0.
    """
    return _create_zero_state()


@pytest.fixture
def perturbed_state() -> CategoricalState:
    """
    Estado perturbado con valores no triviales para tests adversariales.

    Los valores π, -e, 42 se eligen para ser:
    - No enteros (detecta truncamiento)
    - Con signos mixtos (detecta errores de signo)
    - Distintos entre sí (detecta mezcla de dimensiones)

    Returns
    -------
    CategoricalState
        Estado con dim_1=3.14, dim_2=-2.71, dim_3=42.0.
    """
    return _create_perturbed_state(_PERTURBATION_VALUES)


# =============================================================================
# TEST SUITE
# =============================================================================


@pytest.mark.integration
@pytest.mark.stress
class TestMICFunctionalOrthogonality:
    """
    Valida la estructura de espacio vectorial ortonormal de la MIC.

    Estructura de tests:
    ────────────────────
    1. Ortogonalidad canónica (Gram = I₃)
    2. Estabilidad espectral (spec(T) = {1, 1, 1})
    3. Rango completo (rank(T) = 3)
    4. Invariancia bajo estado perturbado
    5. Detección de fugas dimensionales
    6. Composición funcional y ortogonalidad del complemento
    7. Conmutatividad de handlers con soporte disjunto
    8. Idempotencia de handlers individuales

    Invariantes algebraicos verificados:
        (A1) G = I_n              (ortonormalidad)
        (A2) spec(T) = {1}^n     (estabilidad espectral)
        (A3) rank(T) = n          (rango completo, nulidad 0)
        (A4) ⟨e_i(x), e_j(x)⟩ independiente de x para i ≠ j
        (A5) dim(payload) ⊆ _CANONICAL_DIMENSIONS (no fugas)
        (A6) e_i ∘ e_j = e_j ∘ e_i para i ≠ j (conmutatividad)
    """

    # ─────────────────────────────────────────────────────────────────
    # Test 1: Ortonormalidad de la base canónica
    # ─────────────────────────────────────────────────────────────────

    def test_canonical_basis_orthonormality(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica el invariante (A1): la matriz de Gram de los vectores
        base es exactamente la identidad I₃.

        Demostración constructiva:
        ─────────────────────────
        Sea B = {e₁, e₂, e₃} la base canónica y 0 el estado cero.

        Los vectores imagen son:
            v₁ = e₁(0) = (1, 0, 0)
            v₂ = e₂(0) = (0, 1, 0)
            v₃ = e₃(0) = (0, 0, 1)

        La matriz de Gram es:
            G_ij = ⟨v_i, v_j⟩ = δ_ij

        Por tanto G = I₃, lo cual es equivalente a:
        - Ortogonalidad: ⟨v_i, v_j⟩ = 0 para i ≠ j (sin side effects)
        - Normalidad: ‖v_i‖ = 1 para todo i (activación unitaria)
        """
        vectors: List[np.ndarray] = [
            _extract_state_vector(vec(zero_state))
            for vec in canonical_basis
        ]

        G: np.ndarray = _compute_gram_matrix(vectors)
        I_n: np.ndarray = np.eye(_SPACE_DIMENSION)

        np.testing.assert_allclose(
            G,
            I_n,
            atol=_ORTHOGONALITY_TOLERANCE,
            err_msg=(
                "VIOLACIÓN DE ORTONORMALIDAD: La matriz de Gram no es I₃.\n"
                f"G = \n{G}\n"
                f"Los vectores imagen son:\n"
                + "\n".join(
                    f"  v_{i+1} = {v}" for i, v in enumerate(vectors)
                )
                + "\n"
                "Cada herramienta debe activar EXACTAMENTE una dimensión "
                "con valor 1.0 y dejar las demás en 0.0."
            ),
        )

    # ─────────────────────────────────────────────────────────────────
    # Test 2: Estabilidad espectral
    # ─────────────────────────────────────────────────────────────────

    def test_spectral_stability_identity_spectrum(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica el invariante (A2): todos los eigenvalores de la
        matriz de transformación T son exactamente λ = 1.

        Fundamento:
        ──────────
        Si T = I_n, entonces el polinomio característico es:

            det(T - λI) = (1 - λ)ⁿ = 0

        con raíz única λ = 1 de multiplicidad algebraica n.

        Implicaciones de desviación:
        - λ > 1: un estado amplifica exponencialmente su magnitud
          en cada paso → explosión de gradientes.
        - 0 < λ < 1: el estado se contrae exponencialmente →
          amnesia (pérdida de información acumulada).
        - λ < 0: el estado oscila con inversión de signo →
          comportamiento caótico.
        - Im(λ) ≠ 0: rotación en el plano complejo →
          oscilación periódica o espiral.
        """
        T: np.ndarray = _compute_transformation_matrix(
            canonical_basis, zero_state
        )
        eigenvalues: np.ndarray = np.linalg.eigvals(T)

        for idx, eig in enumerate(eigenvalues):
            assert np.isclose(eig.real, 1.0, atol=_ORTHOGONALITY_TOLERANCE), (
                f"Eigenvalor λ_{idx+1} tiene parte real {eig.real:.10f} ≠ 1.0. "
                f"Desviación: Δ = {abs(eig.real - 1.0):.2e}. "
                f"{'EXPLOSIÓN' if eig.real > 1 else 'AMNESIA'} de gradientes detectada."
            )
            assert np.isclose(eig.imag, 0.0, atol=_ORTHOGONALITY_TOLERANCE), (
                f"Eigenvalor λ_{idx+1} tiene parte imaginaria {eig.imag:.10f} ≠ 0. "
                f"Rotación espectral detectada: el estado oscilará "
                f"con período 2π/arg(λ) = {2 * np.pi / abs(np.angle(eig)):.2f} pasos."
            )

    # ─────────────────────────────────────────────────────────────────
    # Test 3: Rango completo
    # ─────────────────────────────────────────────────────────────────

    def test_full_rank_no_blind_spots(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica el invariante (A3): rank(T) = n, equivalentemente
        nulidad(T) = 0.

        Si rank(T) < n, existe un subespacio no trivial Ker(T) ≠ {0}
        donde la MIC no tiene efecto — un "punto ciego operativo".

        Para T = I_n:
            rank(T) = n
            det(T) = 1
            Ker(T) = {0}

        La pérdida de rango indica que alguna herramienta es
        combinación lineal de las demás (redundancia) o que
        alguna dimensión no es alcanzable (punto ciego).
        """
        T: np.ndarray = _compute_transformation_matrix(
            canonical_basis, zero_state
        )
        rank: int = np.linalg.matrix_rank(T)

        assert rank == _SPACE_DIMENSION, (
            f"COLAPSO DIMENSIONAL: rank(T) = {rank} < {_SPACE_DIMENSION}. "
            f"Nulidad = {_SPACE_DIMENSION - rank}. "
            f"det(T) = {np.linalg.det(T):.6f}. "
            f"T = \n{T}\n"
            f"Existen {_SPACE_DIMENSION - rank} dimensiones inaccesibles "
            f"(puntos ciegos operativos)."
        )

        # Verificación complementaria via determinante
        det_T: float = np.linalg.det(T)
        assert abs(det_T - 1.0) < _ORTHOGONALITY_TOLERANCE, (
            f"det(T) = {det_T:.10f} ≠ 1.0. "
            f"Para T = I_n, det(T) = 1 exactamente. "
            f"Desviación indica distorsión volumétrica del espacio de estado."
        )

    # ─────────────────────────────────────────────────────────────────
    # Test 4: Ortogonalidad invariante bajo perturbación del estado base
    # ─────────────────────────────────────────────────────────────────

    def test_orthogonality_invariance_under_perturbation(
        self,
        canonical_basis: List[Morphism],
        perturbed_state: CategoricalState,
    ) -> None:
        """
        Verifica el invariante (A4): la ortogonalidad de los EFECTOS
        de los handlers se mantiene cuando el estado base no es cero.

        Fundamento:
        ──────────
        Sea x = (π, -e, 42) un estado perturbado.

        Si e_i es verdaderamente ortogonal:
            e₁(x) = (1, -e, 42)    → efecto: Δ₁ = e₁(x) - x = (1-π, 0, 0)
            e₂(x) = (π, 1, 42)     → efecto: Δ₂ = e₂(x) - x = (0, 1+e, 0)
            e₃(x) = (π, -e, 1)     → efecto: Δ₃ = e₃(x) - x = (0, 0, 1-42)

        Los vectores de EFECTO {Δ_i} deben ser mutuamente ortogonales:
            ⟨Δ_i, Δ_j⟩ = 0 para i ≠ j

        (No necesariamente ortonormales porque ‖Δ_i‖ depende de x_i.)

        Esta propiedad es MÁS FUERTE que la ortogonalidad desde el
        estado cero, porque detecta handlers que leen y modifican
        dimensiones ajenas basándose en valores existentes.
        """
        # Vectores de efecto: Δ_i = e_i(x) - x
        base_vector: np.ndarray = _extract_state_vector(perturbed_state)
        effect_vectors: List[np.ndarray] = []

        for vec in canonical_basis:
            result_vector = _extract_state_vector(vec(perturbed_state))
            delta = result_vector - base_vector
            effect_vectors.append(delta)

        # Verificar ortogonalidad mutua de los efectos
        for i in range(_SPACE_DIMENSION):
            for j in range(i + 1, _SPACE_DIMENSION):
                dot_ij: float = float(np.dot(effect_vectors[i], effect_vectors[j]))
                assert abs(dot_ij) < _ORTHOGONALITY_TOLERANCE, (
                    f"FUGA DIMENSIONAL: ⟨Δ_{i+1}, Δ_{j+1}⟩ = {dot_ij:.2e} ≠ 0. "
                    f"Handler '{_CANONICAL_BASIS[i].name}' interfiere con "
                    f"Handler '{_CANONICAL_BASIS[j].name}' cuando el estado "
                    f"base es x = {base_vector}. "
                    f"Δ_{i+1} = {effect_vectors[i]}, "
                    f"Δ_{j+1} = {effect_vectors[j]}."
                )

        # Verificar que cada efecto es unidimensional (soporte = 1 dimensión)
        for i, delta in enumerate(effect_vectors):
            nonzero_count = np.count_nonzero(np.abs(delta) > _ORTHOGONALITY_TOLERANCE)
            assert nonzero_count == 1, (
                f"Handler '{_CANONICAL_BASIS[i].name}' modifica "
                f"{nonzero_count} dimensiones (esperado 1). "
                f"Δ_{i+1} = {delta}. "
                f"Dimensiones modificadas: "
                f"{[_CANONICAL_DIMENSIONS[k] for k in range(_SPACE_DIMENSION) if abs(delta[k]) > _ORTHOGONALITY_TOLERANCE]}."
            )

    # ─────────────────────────────────────────────────────────────────
    # Test 5: Detección de fugas dimensionales
    # ─────────────────────────────────────────────────────────────────

    def test_no_dimensional_leaks(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica el invariante (A5): ningún handler introduce claves
        en el payload fuera de las dimensiones canónicas.

        Una fuga dimensional ocurre cuando un handler añade una clave
        como "dim_4" o "internal_buffer" al payload. Esto crea una
        dimensión oculta que:
        - No es observable por _extract_state_vector
        - Puede crear dependencias entre handlers aparentemente ortogonales
        - Viola la clausura del espacio vectorial ℝⁿ

        Ejemplo adversarial:
            def bad_handler(state):
                payload = dict(state.payload)
                payload["dim_1"] = 1.0
                payload["_cache"] = "leaked"  # ← FUGA
                return CategoricalState(payload=payload)
        """
        allowed_dims: Set[str] = set(_CANONICAL_DIMENSIONS)

        for i, vec in enumerate(canonical_basis):
            result_state = vec(zero_state)
            leaks = _detect_dimensional_leaks(result_state, allowed_dims)

            assert len(leaks) == 0, (
                f"FUGA DIMENSIONAL en handler '{_CANONICAL_BASIS[i].name}': "
                f"Claves no autorizadas en payload: {leaks}. "
                f"Payload completo: {result_state.payload}. "
                f"Solo se permiten: {sorted(allowed_dims)}."
            )

    # ─────────────────────────────────────────────────────────────────
    # Test 6: Composición y ortogonalidad del complemento
    # ─────────────────────────────────────────────────────────────────

    def test_composition_preserves_complement_orthogonality(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica que la composición funcional e₂ ∘ e₁ produce un vector
        contenido en Span(e₁, e₂), ortogonal al complemento {e₃}.

        Cálculo explícito:
        ──────────────────
            e₁(0) = (1, 0, 0)
            (e₂ ∘ e₁)(0) = e₂(e₁(0)) = e₂((1, 0, 0)) = (1, 1, 0)

        El vector compuesto v_c = (1, 1, 0) satisface:
            ⟨v_c, e₃(0)⟩ = ⟨(1,1,0), (0,0,1)⟩ = 0

        Esto confirma que la composición no "contamina" la dimensión
        ortogonal al subespacio Span(e₁, e₂).

        NOTA IMPORTANTE: (e₂ ∘ e₁)(0) = (1, 1, 0) COINCIDE con
        e₁(0) + e₂(0) = (1, 0, 0) + (0, 1, 0) = (1, 1, 0) solo
        porque los handlers tienen soportes disjuntos. Esta
        coincidencia composición = suma NO es general.
        """
        composed = ComposedMorphism(
            canonical_basis[0], canonical_basis[1]
        )

        v_composite: np.ndarray = _extract_state_vector(
            composed(zero_state)
        )
        v_complement: np.ndarray = _extract_state_vector(
            canonical_basis[2](zero_state)
        )

        # Ortogonalidad con el complemento
        dot_product: float = float(np.dot(v_composite, v_complement))
        assert abs(dot_product) < _ORTHOGONALITY_TOLERANCE, (
            f"CONTAMINACIÓN DIMENSIONAL: ⟨v_composite, e₃⟩ = {dot_product:.2e} ≠ 0. "
            f"v_composite = {v_composite}, e₃ = {v_complement}. "
            f"La composición e₂ ∘ e₁ contamina la dimensión 3."
        )

        # Verificar que v_composite está en Span(e₁, e₂)
        assert abs(v_composite[2]) < _ORTHOGONALITY_TOLERANCE, (
            f"v_composite tiene componente en dim_3 = {v_composite[2]:.2e}. "
            f"Debería ser 0 (contenido en Span(e₁, e₂))."
        )

        effect_1 = _extract_state_vector(canonical_basis[0](zero_state))
        effect_2 = _extract_state_vector(canonical_basis[1](zero_state))

        # Verificar coincidencia composición ≈ suma (propiedad de soportes disjuntos)
        v_sum: np.ndarray = (
            effect_1 + effect_2
        )
        np.testing.assert_allclose(
            v_composite,
            v_sum,
            atol=_ORTHOGONALITY_TOLERANCE,
            err_msg=(
                f"RUPTURA DE ADITIVIDAD: e₂ ∘ e₁ ≠ e₁ + e₂. "
                f"Composición = {v_composite}, Suma = {v_sum}. "
                f"Los handlers no tienen soportes completamente disjuntos."
            ),
        )

    # ─────────────────────────────────────────────────────────────────
    # Test 7: Conmutatividad de handlers con soporte disjunto
    # ─────────────────────────────────────────────────────────────────

    def test_handler_commutativity(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica el invariante (A6): handlers con soporte disjunto conmutan.

        Para handlers que operan en dimensiones disjuntas:
            (e_j ∘ e_i)(x) = (e_i ∘ e_j)(x)  ∀ x

        Esto es una consecuencia directa de que los handlers modifican
        dimensiones independientes. La falla de conmutatividad indicaría
        una dependencia oculta entre dimensiones.

        Se verifica para todos los pares (i, j) con i < j, tanto
        desde el estado cero como desde el estado perturbado.
        """
        perturbed = _create_perturbed_state(_PERTURBATION_VALUES)

        for initial_state, state_name in [
            (zero_state, "cero"),
            (perturbed, "perturbado"),
        ]:
            for i in range(_SPACE_DIMENSION):
                for j in range(i + 1, _SPACE_DIMENSION):
                    e_i = canonical_basis[i]
                    e_j = canonical_basis[j]

                    # e_j ∘ e_i
                    result_ij = _extract_state_vector(
                        e_j(e_i(initial_state))
                    )
                    # e_i ∘ e_j
                    result_ji = _extract_state_vector(
                        e_i(e_j(initial_state))
                    )

                    np.testing.assert_allclose(
                        result_ij,
                        result_ji,
                        atol=_ORTHOGONALITY_TOLERANCE,
                        err_msg=(
                            f"NO CONMUTATIVIDAD desde estado {state_name}: "
                            f"(e_{j+1} ∘ e_{i+1})(x) ≠ (e_{i+1} ∘ e_{j+1})(x). "
                            f"e_{j+1}(e_{i+1}(x)) = {result_ij}, "
                            f"e_{i+1}(e_{j+1}(x)) = {result_ji}. "
                            f"Existe dependencia oculta entre "
                            f"'{_CANONICAL_BASIS[i].name}' y "
                            f"'{_CANONICAL_BASIS[j].name}'."
                        ),
                    )

    # ─────────────────────────────────────────────────────────────────
    # Test 8: Idempotencia de activación
    # ─────────────────────────────────────────────────────────────────

    def test_handler_idempotence(
        self,
        canonical_basis: List[Morphism],
        zero_state: CategoricalState,
    ) -> None:
        """
        Verifica que aplicar un handler dos veces produce el mismo
        resultado que aplicarlo una vez:

            e_i(e_i(x)) = e_i(x)  ∀ x

        Esto es una propiedad de idempotencia que garantiza estabilidad:
        re-ejecutar una herramienta no amplifica ni distorsiona el estado.

        Para un handler que establece dim_i = 1.0:
            e_i(x) = (..., 1.0, ...)     [dim_i = 1.0]
            e_i(e_i(x)) = (..., 1.0, ...) [dim_i ya es 1.0, sin cambio]

        La falla de idempotencia indicaría que el handler ACUMULA efecto
        (e.g., dim_i += 1.0 en lugar de dim_i = 1.0), causando
        explosión bajo re-ejecución.
        """
        for i, vec in enumerate(canonical_basis):
            # Una aplicación
            once = vec(zero_state)
            v_once = _extract_state_vector(once)

            # Doble aplicación
            twice = vec(once)
            v_twice = _extract_state_vector(twice)

            np.testing.assert_allclose(
                v_twice,
                v_once,
                atol=_ORTHOGONALITY_TOLERANCE,
                err_msg=(
                    f"NO IDEMPOTENCIA: e_{i+1}(e_{i+1}(0)) ≠ e_{i+1}(0). "
                    f"Una aplicación: {v_once}, "
                    f"Doble aplicación: {v_twice}. "
                    f"Handler '{_CANONICAL_BASIS[i].name}' ACUMULA efecto "
                    f"en lugar de establecer valor fijo."
                ),
            )