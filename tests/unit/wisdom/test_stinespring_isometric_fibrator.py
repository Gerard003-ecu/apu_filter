# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas: Stinespring Isometric Fibrator                             ║
║ Ubicación: tests/unit/wisdom/test_stinespring_isometric_fibrator.py          ║
║ Versión: 3.0.0-Rigorous-Doctoral-Test-Suite                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas con Fundamentos Matemáticos Formales:
────────────────────────────────────────────────────────────────────────────────

Esta suite implementa verificación exhaustiva de:

1. **Axiomas CPTP (Completely Positive Trace Preserving)**:
   - CP: (Φ ⊗ id_n)(ρ) ≥ 0 para todo n y todo ρ ≥ 0
   - TP: Tr[Φ(ρ)] = Tr[ρ] = 1 para todo estado ρ

2. **Teorema de Stinespring**:
   - ∀ canal CPTP Φ: ∃! isometría V minimal tal que Φ(ρ) = Tr_env[V ρ V^†]
   - Minimalidad: dim(env) = rank(Choi(Φ))

3. **Teorema de Choi-Jamiołkowski**:
   - Isomorfismo: Φ ↔ (I ⊗ Φ)(|Ψ⁺⟩⟨Ψ⁺|)
   - CP ⟺ Choi(Φ) ≥ 0

4. **Criterio PPT (Peres-Horodecki)**:
   - Para 2×2, 2×3: ρ separable ⟺ ρ^{T_B} ≥ 0

5. **Fidelidad de Uhlmann**:
   - F(ρ, σ) = [Tr√(√ρ σ √ρ)]²
   - Cotas de error para truncamiento espectral

6. **Propiedades de Canales Especiales**:
   - Unitales: Φ(I) = I
   - Entanglement Breaking: Choi separable
   - Depolarizantes, Pauli, fase-damping, amplitude-damping

╔═══════════════════════════════════════════════════════════════════════════════╗
║                        METODOLOGÍA DE PRUEBAS                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

- **Pruebas Parametrizadas**: pytest.mark.parametrize para cobertura exhaustiva
- **Generación Aleatoria**: Estados Haar-random para robustez estadística
- **Verificación Algebraica**: Comprobación de identidades matemáticas exactas
- **Análisis Numérico**: Monitoreo de estabilidad y condicionamiento
- **Casos Extremos**: Dimensiones límite, estados puros/mixtos, canales degenerados
- **Propiedades Categóricas**: Funtoridad, naturalidad, composición

"""

from __future__ import annotations

import logging
from typing import List, Tuple, Callable
from dataclasses import dataclass

import pytest
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Importaciones del módulo bajo prueba
from app.wisdom.stinespring_isometric_fibrator import (
    StinespringIsometricFibrator,
    SpectralAnalyzer,
    ChoiOperatorFactory,
    IsometryConstructor,
    QuantumChannelApplicator,
    EnvironmentTruncator,
    QuantumInformationMetrics,
    NumericalThresholds,
    SpectralDecomposition,
    ChoiOperator,
    IsometryTensor,
    ChannelFidelityMetrics,
)

from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix
from app.core.mic_algebra import NumericalInstabilityError
from app.wisdom.mac_algebra import TraceAnomalyError


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE GENERACIÓN CUÁNTICA
# ══════════════════════════════════════════════════════════════════════════════

class QuantumStateGenerator:
    """Generador de estados cuánticos con propiedades certificadas."""
    
    @staticmethod
    def random_density_matrix(dim: int, rank: Optional[int] = None) -> NDArray[np.complex128]:
        """
        Genera matriz de densidad aleatoria via método de Ginibre.
        
        Algoritmo:
            1. Generar matriz aleatoria compleja A ~ Ginibre(dim, rank)
            2. ρ = A A^† / Tr(A A^†)
        
        Garantiza: ρ ≥ 0, Tr(ρ) = 1, rank(ρ) = rank (si especificado)
        """
        if rank is None:
            rank = dim
        
        # Matriz de Ginibre: entradas i.i.d. ~ CN(0, 1)
        A = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
        A /= np.sqrt(2)  # Normalización estándar
        
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        
        return rho
    
    @staticmethod
    def pure_state(dim: int) -> NDArray[np.complex128]:
        """
        Genera estado puro aleatorio via distribución de Haar.
        
        Algoritmo QR: A ~ Ginibre, Q = QR(A)[0], |ψ⟩ = Q[:, 0]
        """
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        Q, _ = la.qr(A)
        psi = Q[:, 0]
        
        rho = np.outer(psi, psi.conj())
        return rho
    
    @staticmethod
    def maximally_mixed(dim: int) -> NDArray[np.complex128]:
        """Estado maximalmente mixto: I / dim."""
        return np.eye(dim, dtype=np.complex128) / dim
    
    @staticmethod
    def bell_state(bell_type: int = 0) -> NDArray[np.complex128]:
        """
        Estados de Bell (base EPR).
        
        |Φ⁺⟩ = (|00⟩ + |11⟩) / √2  [tipo 0]
        |Φ⁻⟩ = (|00⟩ - |11⟩) / √2  [tipo 1]
        |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2  [tipo 2]
        |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2  [tipo 3]
        """
        states = {
            0: np.array([1, 0, 0, 1], dtype=np.complex128),  # Φ⁺
            1: np.array([1, 0, 0, -1], dtype=np.complex128), # Φ⁻
            2: np.array([0, 1, 1, 0], dtype=np.complex128),  # Ψ⁺
            3: np.array([0, 1, -1, 0], dtype=np.complex128), # Ψ⁻
        }
        
        psi = states[bell_type] / np.sqrt(2)
        return np.outer(psi, psi.conj())
    
    @staticmethod
    def werner_state(dim: int, fidelity: float) -> NDArray[np.complex128]:
        """
        Estado de Werner con fidelidad F ∈ [0, 1].
        
        ρ_W(F) = F |Ψ⁺⟩⟨Ψ⁺| + (1-F) I/(d²)
        
        Separable ⟺ F ≤ 1/(d+1) para dimensión d×d
        """
        d = dim
        psi_plus = np.zeros((d*d, 1), dtype=np.complex128)
        for i in range(d):
            psi_plus[i * d + i] = 1.0
        psi_plus /= np.sqrt(d)
        
        rho_pure = psi_plus @ psi_plus.conj().T
        rho_mixed = np.eye(d*d, dtype=np.complex128) / (d*d)
        
        return fidelity * rho_pure + (1 - fidelity) * rho_mixed


class KrausOperatorGenerator:
    """Generador de operadores de Kraus para canales específicos."""
    
    @staticmethod
    def identity_channel(dim: int) -> List[NDArray[np.complex128]]:
        """Canal identidad: Φ(ρ) = ρ."""
        return [np.eye(dim, dtype=np.complex128)]
    
    @staticmethod
    def depolarizing_channel(dim: int, p: float) -> List[NDArray[np.complex128]]:
        """
        Canal depolarizante: Φ(ρ) = (1-p)ρ + p·I/d.
        
        Parámetro: p ∈ [0, 1]
        - p = 0: identidad
        - p = 1: completamente depolarizante
        
        Operadores de Kraus (para qubit, d=2):
            M_0 = √(1 - 3p/4) I
            M_1 = √(p/4) X
            M_2 = √(p/4) Y
            M_3 = √(p/4) Z
        """
        if dim != 2:
            # Generalización para d arbitrario (versión simplificada)
            M0 = np.sqrt(1 - p * (dim**2 - 1) / dim**2) * np.eye(dim, dtype=np.complex128)
            kraus_ops = [M0]
            
            # Generar bases ortogonales (bases de Gell-Mann generalizadas)
            # Para simplicidad, usamos proyectores ortonormales
            for i in range(dim**2 - 1):
                # Generar matriz base aleatoria ortogonal
                basis = np.zeros((dim, dim), dtype=np.complex128)
                idx1, idx2 = divmod(i + 1, dim)
                if idx1 == idx2:
                    basis[idx1, idx1] = 1.0
                else:
                    basis[idx1, idx2] = 1.0 / np.sqrt(2)
                    basis[idx2, idx1] = 1.0 / np.sqrt(2)
                
                kraus_ops.append(np.sqrt(p / (dim**2 - 1)) * basis)
            
            return kraus_ops
        
        # Caso específico qubit (d=2)
        pauli_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        pauli_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        
        return [
            np.sqrt(1 - 3*p/4) * identity,
            np.sqrt(p/4) * pauli_X,
            np.sqrt(p/4) * pauli_Y,
            np.sqrt(p/4) * pauli_Z,
        ]
    
    @staticmethod
    def amplitude_damping_channel(gamma: float) -> List[NDArray[np.complex128]]:
        """
        Canal amplitude-damping (decaimiento espontáneo).
        
        Modeliza emisión espontánea con tasa γ ∈ [0, 1].
        
        Kraus:
            M_0 = [[1, 0], [0, √(1-γ)]]
            M_1 = [[0, √γ], [0, 0]]
        """
        M0 = np.array([
            [1.0, 0.0],
            [0.0, np.sqrt(1 - gamma)]
        ], dtype=np.complex128)
        
        M1 = np.array([
            [0.0, np.sqrt(gamma)],
            [0.0, 0.0]
        ], dtype=np.complex128)
        
        return [M0, M1]
    
    @staticmethod
    def phase_damping_channel(gamma: float) -> List[NDArray[np.complex128]]:
        """
        Canal phase-damping (decoherencia de fase).
        
        Destruye coherencia sin afectar poblaciones.
        
        Kraus:
            M_0 = [[1, 0], [0, √(1-γ)]]
            M_1 = [[0, 0], [0, √γ]]
        """
        M0 = np.array([
            [1.0, 0.0],
            [0.0, np.sqrt(1 - gamma)]
        ], dtype=np.complex128)
        
        M1 = np.array([
            [0.0, 0.0],
            [0.0, np.sqrt(gamma)]
        ], dtype=np.complex128)
        
        return [M0, M1]
    
    @staticmethod
    def unitary_channel(U: NDArray[np.complex128]) -> List[NDArray[np.complex128]]:
        """Canal unitario: Φ(ρ) = U ρ U^†."""
        return [U]
    
    @staticmethod
    def random_unitary_channel(dim: int) -> List[NDArray[np.complex128]]:
        """Canal unitario aleatorio via distribución de Haar."""
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        U, _ = la.qr(A)
        return [U]
    
    @staticmethod
    def pauli_channel(px: float, py: float, pz: float) -> List[NDArray[np.complex128]]:
        """
        Canal de Pauli: Φ(ρ) = (1-px-py-pz)ρ + px X ρ X + py Y ρ Y + pz Z ρ Z.
        
        Parámetros: px, py, pz ≥ 0, px + py + pz ≤ 1
        """
        if px + py + pz > 1.0 + 1e-10:
            raise ValueError("Probabilidades de Pauli exceden 1")
        
        pauli_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        pauli_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        
        p0 = 1 - px - py - pz
        
        return [
            np.sqrt(p0) * identity,
            np.sqrt(px) * pauli_X,
            np.sqrt(py) * pauli_Y,
            np.sqrt(pz) * pauli_Z,
        ]
    
    @staticmethod
    def entanglement_breaking_channel(dim: int) -> List[NDArray[np.complex128]]:
        """
        Canal que rompe entrelazamiento (mide en base computacional).
        
        Φ(ρ) = ∑_i ⟨i|ρ|i⟩ |i⟩⟨i|
        
        Kraus: M_i = |i⟩⟨i|
        """
        kraus_ops = []
        for i in range(dim):
            M_i = np.zeros((dim, dim), dtype=np.complex128)
            M_i[i, i] = 1.0
            kraus_ops.append(M_i)
        
        return kraus_ops


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def logger_config():
    """Configura logging para pruebas."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )


@pytest.fixture
def small_dimensions():
    """Dimensiones pequeñas para pruebas rápidas."""
    return {"mic": 2, "mac": 2}


@pytest.fixture
def medium_dimensions():
    """Dimensiones medianas para pruebas de rendimiento."""
    return {"mic": 4, "mac": 4}


@pytest.fixture
def large_dimensions():
    """Dimensiones grandes para pruebas de escalabilidad."""
    return {"mic": 8, "mac": 8}


@pytest.fixture
def fibrator_small(small_dimensions):
    """Fibrador para dimensiones pequeñas."""
    return StinespringIsometricFibrator(
        mic_dim=small_dimensions["mic"],
        mac_dim=small_dimensions["mac"],
        max_env_dim=20
    )


@pytest.fixture
def fibrator_medium(medium_dimensions):
    """Fibrador para dimensiones medianas."""
    return StinespringIsometricFibrator(
        mic_dim=medium_dimensions["mic"],
        mac_dim=medium_dimensions["mac"],
        max_env_dim=50
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE GENERADORES CUÁNTICOS
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumStateGenerator:
    """Pruebas para generadores de estados cuánticos."""
    
    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 8])
    def test_random_density_matrix_properties(self, dim):
        """Verifica propiedades de matrices de densidad aleatorias."""
        rho = QuantumStateGenerator.random_density_matrix(dim)
        
        # Axioma D1: Hermiticidad
        assert np.allclose(rho, rho.conj().T), "Estado no hermítico"
        
        # Axioma D2: Positividad
        eigenvalues = la.eigvalsh(rho)
        assert np.all(eigenvalues >= -NumericalThresholds.POSITIVITY_TOL), \
            f"Estado no positivo: λ_min = {eigenvalues.min()}"
        
        # Axioma D3: Traza unitaria
        trace = np.trace(rho).real
        assert np.isclose(trace, 1.0, atol=1e-10), \
            f"Traza no unitaria: Tr(ρ) = {trace}"
        
        # Propiedad adicional: Pureza
        purity = np.trace(rho @ rho).real
        assert 1/dim <= purity <= 1.0 + 1e-10, \
            f"Pureza fuera de rango: γ = {purity}"
    
    @pytest.mark.parametrize("dim,rank", [
        (4, 1),  # Estado puro
        (4, 2),  # Rango parcial
        (4, 4),  # Rango completo
    ])
    def test_random_density_matrix_rank(self, dim, rank):
        """Verifica control de rango en estados aleatorios."""
        rho = QuantumStateGenerator.random_density_matrix(dim, rank=rank)
        
        numerical_rank = np.linalg.matrix_rank(
            rho, tol=NumericalThresholds.POSITIVITY_TOL
        )
        
        assert numerical_rank == rank, \
            f"Rango incorrecto: esperado {rank}, obtenido {numerical_rank}"
    
    @pytest.mark.parametrize("dim", [2, 3, 5, 7])
    def test_pure_state_properties(self, dim):
        """Verifica propiedades de estados puros."""
        rho = QuantumStateGenerator.pure_state(dim)
        
        # Estado puro ⟺ ρ² = ρ
        rho_squared = rho @ rho
        assert np.allclose(rho_squared, rho, atol=1e-10), \
            "Estado puro no satisface ρ² = ρ"
        
        # Pureza = 1
        purity = np.trace(rho @ rho).real
        assert np.isclose(purity, 1.0, atol=1e-10), \
            f"Pureza de estado puro ≠ 1: γ = {purity}"
        
        # Entropía = 0
        entropy = QuantumInformationMetrics.von_neumann_entropy(rho)
        assert entropy < 1e-10, \
            f"Entropía de estado puro ≠ 0: S = {entropy}"
    
    def test_maximally_mixed_properties(self):
        """Verifica propiedades del estado maximalmente mixto."""
        dim = 4
        rho = QuantumStateGenerator.maximally_mixed(dim)
        
        # Debe ser proporcional a identidad
        expected = np.eye(dim, dtype=np.complex128) / dim
        assert np.allclose(rho, expected), \
            "Estado maximalmente mixto ≠ I/d"
        
        # Pureza mínima
        purity = QuantumInformationMetrics.purity(rho)
        assert np.isclose(purity, 1/dim, atol=1e-10), \
            f"Pureza incorrecta: esperado {1/dim}, obtenido {purity}"
        
        # Entropía máxima
        entropy = QuantumInformationMetrics.von_neumann_entropy(rho)
        expected_entropy = np.log2(dim)
        assert np.isclose(entropy, expected_entropy, atol=1e-10), \
            f"Entropía incorrecta: esperado {expected_entropy}, obtenido {entropy}"
    
    @pytest.mark.parametrize("bell_type", [0, 1, 2, 3])
    def test_bell_states(self, bell_type):
        """Verifica propiedades de estados de Bell."""
        rho = QuantumStateGenerator.bell_state(bell_type)
        
        # Estados de Bell son puros
        purity = QuantumInformationMetrics.purity(rho)
        assert np.isclose(purity, 1.0, atol=1e-10), \
            f"Estado de Bell no es puro: γ = {purity}"
        
        # Traza parcial debe dar estado maximalmente mixto (entrelazamiento maximal)
        # Tr_B[|Ψ⟩⟨Ψ|] = I/2 para estados de Bell
        rho_reshaped = rho.reshape(2, 2, 2, 2)
        rho_A = np.trace(rho_reshaped, axis1=1, axis2=3)
        
        expected = np.eye(2, dtype=np.complex128) / 2
        assert np.allclose(rho_A, expected, atol=1e-10), \
            "Traza parcial de Bell no es maximalmente mixta"
    
    @pytest.mark.parametrize("fidelity", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_werner_state(self, fidelity):
        """Verifica construcción de estados de Werner."""
        dim = 2
        rho = QuantumStateGenerator.werner_state(dim, fidelity)
        
        # Verificar traza
        trace = np.trace(rho).real
        assert np.isclose(trace, 1.0, atol=1e-10), \
            f"Traza incorrecta: {trace}"
        
        # Verificar separabilidad via criterio PPT para dim=2
        # Werner separable ⟺ F ≤ 1/3 para qubits
        rho_reshaped = rho.reshape(dim, dim, dim, dim)
        rho_ppt = rho_reshaped.transpose(0, 3, 2, 1).reshape(dim*dim, dim*dim)
        
        eigenvalues_ppt = la.eigvalsh(rho_ppt)
        is_ppt = np.all(eigenvalues_ppt >= -NumericalThresholds.POSITIVITY_TOL)
        
        if fidelity <= 1/3 + 1e-6:
            assert is_ppt, f"Werner con F={fidelity} debe ser separable (PPT)"
        else:
            assert not is_ppt, f"Werner con F={fidelity} debe ser entrelazado (no PPT)"


class TestKrausOperatorGenerator:
    """Pruebas para generadores de operadores de Kraus."""
    
    def test_identity_channel(self):
        """Verifica canal identidad."""
        dim = 3
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        
        assert len(kraus_ops) == 1, "Canal identidad debe tener 1 Kraus"
        assert np.allclose(kraus_ops[0], np.eye(dim)), \
            "Kraus de identidad debe ser I"
        
        # Verificar completitud
        sum_MdM = sum(M.conj().T @ M for M in kraus_ops)
        assert np.allclose(sum_MdM, np.eye(dim)), \
            "Completitud de Kraus violada"
    
    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
    def test_depolarizing_channel_completeness(self, p):
        """Verifica completitud del canal depolarizante."""
        dim = 2
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, p)
        
        # ∑_k M_k^† M_k = I
        sum_MdM = sum(M.conj().T @ M for M in kraus_ops)
        assert np.allclose(sum_MdM, np.eye(dim), atol=1e-10), \
            f"Completitud violada para p={p}"
    
    @pytest.mark.parametrize("gamma", [0.0, 0.3, 0.7, 1.0])
    def test_amplitude_damping_channel(self, gamma):
        """Verifica canal amplitude-damping."""
        kraus_ops = KrausOperatorGenerator.amplitude_damping_channel(gamma)
        
        # Verificar completitud
        sum_MdM = sum(M.conj().T @ M for M in kraus_ops)
        assert np.allclose(sum_MdM, np.eye(2), atol=1e-12), \
            f"Completitud violada para γ={gamma}"
        
        # Verificar acción sobre estado excitado |1⟩
        # Debe decaer a |0⟩ con probabilidad γ
        rho_excited = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        rho_out = sum(M @ rho_excited @ M.conj().T for M in kraus_ops)
        
        # Población de |0⟩ debe ser γ
        assert np.isclose(rho_out[0, 0].real, gamma, atol=1e-10), \
            f"Población incorrecta después de amplitude damping"
    
    def test_pauli_channel_completeness(self):
        """Verifica canal de Pauli."""
        px, py, pz = 0.1, 0.2, 0.15
        kraus_ops = KrausOperatorGenerator.pauli_channel(px, py, pz)
        
        sum_MdM = sum(M.conj().T @ M for M in kraus_ops)
        assert np.allclose(sum_MdM, np.eye(2), atol=1e-12), \
            "Completitud de canal Pauli violada"
    
    def test_entanglement_breaking_channel(self):
        """Verifica canal que rompe entrelazamiento."""
        dim = 3
        kraus_ops = KrausOperatorGenerator.entanglement_breaking_channel(dim)
        
        # Debe tener dim operadores (proyectores)
        assert len(kraus_ops) == dim, \
            f"Número incorrecto de Kraus: esperado {dim}, obtenido {len(kraus_ops)}"
        
        # Cada operador debe ser proyector
        for i, M in enumerate(kraus_ops):
            assert np.allclose(M @ M, M, atol=1e-12), \
                f"M_{i} no es proyector"
        
        # Verificar que rompe entrelazamiento
        # Aplicar a estado de Bell → debe dar estado separable
        bell = QuantumStateGenerator.bell_state(0)  # 2x2
        
        # Extender canal a 2x2
        kraus_ops_qubit = KrausOperatorGenerator.entanglement_breaking_channel(2)
        rho_out = sum(M @ bell @ M.conj().T for M in kraus_ops_qubit)
        
        # Verificar PPT (debe ser positivo)
        rho_reshaped = rho_out.reshape(2, 2, 2, 2)
        rho_ppt = rho_reshaped.transpose(0, 3, 2, 1).reshape(4, 4)
        eigenvalues_ppt = la.eigvalsh(rho_ppt)
        
        assert np.all(eigenvalues_ppt >= -NumericalThresholds.POSITIVITY_TOL), \
            "Canal no rompió entrelazamiento (falla PPT)"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DEL ANALIZADOR ESPECTRAL
# ══════════════════════════════════════════════════════════════════════════════

class TestSpectralAnalyzer:
    """Pruebas rigurosas del analizador espectral."""
    
    @pytest.mark.parametrize("dim", [2, 3, 5, 8])
    def test_canonical_spectral_decomposition(self, dim):
        """Verifica descomposición espectral canónica."""
        # Generar operador hermítico aleatorio
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = 0.5 * (A + A.conj().T)
        
        spectral = SpectralAnalyzer.canonical_spectral_decomposition(H)
        
        # Verificar ordenamiento decreciente
        assert np.all(np.diff(spectral.eigenvalues) <= 0), \
            "Autovalores no ordenados decrecientemente"
        
        # Verificar reconstrucción: H = V Λ V^†
        reconstructed = (spectral.eigenvectors * spectral.eigenvalues) @ \
                       spectral.eigenvectors.conj().T
        
        assert np.allclose(reconstructed, H, atol=1e-10), \
            "Reconstrucción espectral incorrecta"
        
        # Verificar ortonormalidad: V^† V = I
        VdV = spectral.eigenvectors.conj().T @ spectral.eigenvectors
        assert np.allclose(VdV, np.eye(dim), atol=1e-12), \
            "Autovectores no ortonormales"
    
    def test_spectral_gauge_standardization(self):
        """Verifica gauge de fase estándar."""
        dim = 4
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = 0.5 * (H + H.conj().T)
        
        spectral = SpectralAnalyzer.canonical_spectral_decomposition(H)
        
        # Verificar que primer elemento no nulo de cada autovector es real positivo
        for i in range(dim):
            v = spectral.eigenvectors[:, i]
            non_zero_idx = np.argmax(np.abs(v) > NumericalThresholds.EPS_MACHINE)
            
            first_element = v[non_zero_idx]
            assert np.abs(np.imag(first_element)) < 1e-10, \
                f"Gauge de fase incorrecto en autovector {i}"
            assert np.real(first_element) >= -1e-10, \
                f"Gauge de fase negativo en autovector {i}"
    
    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_rank_determination(self, rank):
        """Verifica determinación de rango numérico."""
        dim = 5
        # Construir matriz con rango específico
        A = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
        H = A @ A.conj().T
        
        spectral = SpectralAnalyzer.canonical_spectral_decomposition(H)
        
        assert spectral.rank == rank, \
            f"Rango incorrecto: esperado {rank}, obtenido {spectral.rank}"
    
    def test_condition_number_computation(self):
        """Verifica cálculo de número de condición."""
        # Matriz bien condicionada
        H_good = np.diag([1.0, 0.9, 0.8, 0.7])
        spectral_good = SpectralAnalyzer.canonical_spectral_decomposition(H_good)
        assert spectral_good.condition_number < 2.0, \
            "Número de condición incorrecto para matriz bien condicionada"
        
        # Matriz mal condicionada
        H_bad = np.diag([1.0, 1e-10, 1e-11, 1e-12])
        spectral_bad = SpectralAnalyzer.canonical_spectral_decomposition(H_bad)
        assert spectral_bad.condition_number > 1e10, \
            "Número de condición incorrecto para matriz mal condicionada"
    
    def test_project_to_positive_cone(self):
        """Verifica proyección de Löwner al cono PSD."""
        dim = 4
        # Crear matriz con autovalores negativos
        eigenvalues = np.array([2.0, 1.0, -0.5, -1.0])
        V = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        V, _ = la.qr(V)  # Ortonormalizar
        
        H = (V * eigenvalues) @ V.conj().T
        
        # Proyectar
        H_projected, distance = SpectralAnalyzer.project_to_positive_cone(H)
        
        # Verificar positividad
        eigs_projected = la.eigvalsh(H_projected)
        assert np.all(eigs_projected >= -NumericalThresholds.POSITIVITY_TOL), \
            "Proyección no es positiva semidefinida"
        
        # Verificar distancia (debe corresponder a norma de autovalores negativos)
        expected_distance = np.sqrt(0.5**2 + 1.0**2)
        assert np.isclose(distance, expected_distance, atol=1e-10), \
            f"Distancia de proyección incorrecta: {distance} vs {expected_distance}"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE LA FÁBRICA DE OPERADORES DE CHOI
# ══════════════════════════════════════════════════════════════════════════════

class TestChoiOperatorFactory:
    """Pruebas de construcción y validación de operadores de Choi."""
    
    def test_choi_from_identity_channel(self):
        """Verifica Choi del canal identidad."""
        dim = 3
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # Choi de identidad es estado maximalmente entrelazado
        # Choi(id) = |Φ⁺⟩⟨Φ⁺| donde |Φ⁺⟩ = ∑_i |ii⟩ / √d
        psi_plus = np.zeros(dim * dim, dtype=np.complex128)
        for i in range(dim):
            psi_plus[i * dim + i] = 1.0
        psi_plus /= np.sqrt(dim)
        
        expected_choi = np.outer(psi_plus, psi_plus.conj())
        
        assert np.allclose(choi.matrix, expected_choi, atol=1e-10), \
            "Choi de identidad incorrecto"
        
        # Verificar propiedades
        assert choi.is_completely_positive, "Canal identidad debe ser CP"
        assert choi.is_trace_preserving, "Canal identidad debe ser TP"
        assert choi.rank == 1, "Rango de Choi(identidad) debe ser 1"
    
    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 1.0])
    def test_choi_from_depolarizing_channel(self, p):
        """Verifica Choi del canal depolarizante."""
        dim = 2
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, p)
        
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # Verificar CP
        assert choi.is_completely_positive, \
            f"Canal depolarizante (p={p}) debe ser CP"
        
        # Verificar TP
        assert choi.is_trace_preserving, \
            f"Canal depolarizante (p={p}) debe ser TP"
        
        # Verificar rango
        # Depolarizante tiene rango 4 (excepto p=0 que es identidad)
        if p == 0.0:
            assert choi.rank == 1, "Depolarizante(p=0) debe tener rango 1"
        else:
            assert choi.rank == 4, f"Depolarizante(p={p}) debe tener rango 4"
    
    def test_choi_trace_preservation_verification(self):
        """Verifica detección de violación de TP."""
        dim = 2
        # Crear operadores que NO preservan traza
        M1 = np.array([[1.0, 0], [0, 0.5]], dtype=np.complex128)
        M2 = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
        
        kraus_ops = [M1, M2]
        
        # Esto debe advertir pero NO fallar (solo advierte)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # Verificar que detecta que NO es TP
        assert not choi.is_trace_preserving, \
            "Debería detectar violación de TP"
    
    def test_choi_complete_positivity_enforcement(self):
        """Verifica proyección cuando canal no es CP."""
        dim = 2
        # Crear "canal" no CP artificialmente
        # (Esto es difícil, usamos matriz de Choi negativa directamente)
        
        # Crear Kraus que den Choi no positivo es complicado,
        # pero podemos probar la proyección directamente
        eigenvalues = np.array([1.0, 0.5, -0.2, -0.3])
        V = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        V, _ = la.qr(V)
        
        choi_matrix = (V * eigenvalues) @ V.conj().T
        
        # Proyectar
        choi_projected, _ = SpectralAnalyzer.project_to_positive_cone(choi_matrix)
        
        # Verificar que ahora es CP
        eigs = la.eigvalsh(choi_projected)
        assert np.all(eigs >= -NumericalThresholds.POSITIVITY_TOL), \
            "Proyección no restauró CP"
    
    @pytest.mark.parametrize("bell_type", [0, 1, 2, 3])
    def test_ppt_criterion_on_bell_states(self, bell_type):
        """Verifica criterio PPT en estados de Bell."""
        # Estados de Bell son entrelazados → deben fallar PPT
        bell = QuantumStateGenerator.bell_state(bell_type)
        
        # Aplicar transposición parcial
        bell_reshaped = bell.reshape(2, 2, 2, 2)
        bell_ppt = bell_reshaped.transpose(0, 3, 2, 1).reshape(4, 4)
        
        # Verificar autovalores
        eigs = la.eigvalsh(bell_ppt)
        has_negative = np.any(eigs < -NumericalThresholds.POSITIVITY_TOL)
        
        assert has_negative, \
            f"Estado de Bell {bell_type} debe fallar PPT (entrelazado)"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DEL CONSTRUCTOR DE ISOMETRÍAS
# ══════════════════════════════════════════════════════════════════════════════

class TestIsometryConstructor:
    """Pruebas de construcción de isometrías de Stinespring."""
    
    def test_isometry_from_identity_channel(self):
        """Verifica isometría del canal identidad."""
        dim = 3
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        isometry = IsometryConstructor.from_choi_operator(choi)
        
        # Isometría de identidad debe ser simplemente I
        expected_V = np.eye(dim, dtype=np.complex128)
        assert np.allclose(isometry.V_matrix, expected_V, atol=1e-10), \
            "Isometría de canal identidad incorrecta"
        
        # Verificar V^† V = I
        VdV = isometry.V_matrix.conj().T @ isometry.V_matrix
        assert np.allclose(VdV, np.eye(dim), atol=1e-12), \
            "Axioma V^† V = I violado"
    
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9])
    def test_isometry_from_amplitude_damping(self, gamma):
        """Verifica isometría de amplitude-damping."""
        kraus_ops = KrausOperatorGenerator.amplitude_damping_channel(gamma)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, 2, 2)
        
        isometry = IsometryConstructor.from_choi_operator(choi)
        
        # Verificar dimensión del entorno
        assert isometry.env_dimension == 2, \
            "Amplitude-damping debe tener 2 Kraus (env_dim=2)"
        
        # Verificar isometría
        IsometryConstructor.verify_isometry_axioms(isometry, 2)
    
    def test_isometry_axioms_verification(self):
        """Verifica verificación de axiomas isométricos."""
        dim = 2
        p = 0.3
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, p)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        isometry = IsometryConstructor.from_choi_operator(choi)
        
        # Esto no debe lanzar excepción
        IsometryConstructor.verify_isometry_axioms(isometry, dim)
        
        # Verificar explícitamente cada axioma
        identity = np.eye(dim, dtype=np.complex128)
        
        # A1: V^† V = I
        VdV = isometry.V_matrix.conj().T @ isometry.V_matrix
        assert np.allclose(VdV, identity, atol=1e-10), "Axioma A1 violado"
        
        # A2: ∑ M_k^† M_k = I
        sum_MdM = sum(M.conj().T @ M for M in isometry.kraus_operators)
        assert np.allclose(sum_MdM, identity, atol=1e-10), "Axioma A2 violado"
        
        # A3: rank(V) = dim
        rank_V = np.linalg.matrix_rank(isometry.V_matrix)
        assert rank_V == dim, f"Axioma A3 violado: rank={rank_V}"
    
    def test_isometry_minimal_dimension(self):
        """Verifica minimalidad de la dimensión del entorno."""
        # Canal con rango de Choi conocido
        dim = 2
        # Entanglement-breaking tiene rango = dim
        kraus_ops = KrausOperatorGenerator.entanglement_breaking_channel(dim)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        isometry = IsometryConstructor.from_choi_operator(choi)
        
        # env_dim debe ser igual a rank(Choi)
        assert isometry.env_dimension == choi.rank, \
            f"Dimensión del entorno no minimal: {isometry.env_dimension} vs {choi.rank}"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DEL APLICADOR DE CANALES CUÁNTICOS
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumChannelApplicator:
    """Pruebas de aplicación de canales cuánticos."""
    
    def test_identity_channel_preservation(self):
        """Verifica que canal identidad preserva estados."""
        dim = 3
        rho = QuantumStateGenerator.random_density_matrix(dim)
        
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        rho_out = QuantumChannelApplicator.apply_kraus_representation(kraus_ops, rho)
        
        assert np.allclose(rho_out, rho, atol=1e-12), \
            "Canal identidad no preserva estado"
    
    @pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 1.0])
    def test_depolarizing_channel_action(self, p):
        """Verifica acción del canal depolarizante."""
        dim = 2
        rho = QuantumStateGenerator.pure_state(dim)
        
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, p)
        rho_out = QuantumChannelApplicator.apply_kraus_representation(kraus_ops, rho)
        
        # Para p=0, debe ser identidad
        if p == 0.0:
            assert np.allclose(rho_out, rho, atol=1e-10), \
                "Depolarizante(p=0) no es identidad"
        
        # Para p=1, debe dar estado maximalmente mixto
        if np.isclose(p, 1.0):
            expected = np.eye(dim, dtype=np.complex128) / dim
            assert np.allclose(rho_out, expected, atol=1e-10), \
                "Depolarizante(p=1) no da estado maximalmente mixto"
        
        # Verificar que preserva traza
        trace_out = np.trace(rho_out).real
        assert np.isclose(trace_out, 1.0, atol=1e-12), \
            "Canal depolarizante no preserva traza"
    
    def test_partial_trace_equivalence(self):
        """Verifica equivalencia entre traza parcial y suma de Kraus."""
        dim = 2
        rho = QuantumStateGenerator.random_density_matrix(dim)
        
        gamma = 0.4
        kraus_ops = KrausOperatorGenerator.amplitude_damping_channel(gamma)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        isometry = IsometryConstructor.from_choi_operator(choi)
        
        # Método 1: Aplicar Kraus directamente
        rho_out_kraus = QuantumChannelApplicator.apply_kraus_representation(
            kraus_ops, rho
        )
        
        # Método 2: Traza parcial del entorno
        rho_out_trace = QuantumChannelApplicator.partial_trace_environment(
            isometry, rho
        )
        
        assert np.allclose(rho_out_kraus, rho_out_trace, atol=1e-10), \
            "Métodos de aplicación de canal no coinciden"
    
    def test_enforce_physicality_hermitization(self):
        """Verifica hermitización de estados."""
        dim = 3
        # Crear matriz no hermítica
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        
        rho_physical, metrics = QuantumChannelApplicator.enforce_physicality(
            A, preserve_trace=False
        )
        
        # Verificar hermiticidad
        assert np.allclose(rho_physical, rho_physical.conj().T, atol=1e-12), \
            "enforce_physicality no hermitizó"
        
        # Verificar positividad
        eigs = la.eigvalsh(rho_physical)
        assert np.all(eigs >= -NumericalThresholds.POSITIVITY_TOL), \
            "enforce_physicality no proyectó a PSD"
    
    def test_enforce_physicality_trace_normalization(self):
        """Verifica normalización de traza."""
        dim = 3
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_scaled = 2.5 * rho  # Traza = 2.5
        
        rho_physical, metrics = QuantumChannelApplicator.enforce_physicality(
            rho_scaled, preserve_trace=True
        )
        
        trace = np.trace(rho_physical).real
        assert np.isclose(trace, 1.0, atol=1e-12), \
            "enforce_physicality no normalizó traza"
        
        assert metrics['trace_normalization_applied'], \
            "Métrica de normalización incorrecta"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DEL TRUNCADOR DE ENTORNO
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentTruncator:
    """Pruebas de truncamiento espectral del entorno."""
    
    def test_no_truncation_when_below_threshold(self):
        """Verifica que no se trunca si rank ≤ max_env_dim."""
        dim = 2
        p = 0.3
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, p)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # max_rank > choi.rank
        truncated_kraus, metrics = EnvironmentTruncator.truncate_with_renormalization(
            choi, max_rank=10
        )
        
        # No debe truncar
        assert len(truncated_kraus) == len(kraus_ops), \
            "Truncó cuando no debería"
        
        # Fidelidad debe ser 1 (sin error)
        assert metrics.uhlmann_fidelity == 1.0, \
            "Fidelidad no es 1 sin truncamiento"
    
    def test_truncation_preserves_cptp(self):
        """Verifica que truncamiento preserva CPTP."""
        dim = 2
        # Crear canal con muchos Kraus (depolarizante tiene 4)
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.8)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # Truncar a 2 operadores
        truncated_kraus, metrics = EnvironmentTruncator.truncate_with_renormalization(
            choi, max_rank=2
        )
        
        assert len(truncated_kraus) <= 2, "No truncó correctamente"
        
        # Verificar TP
        sum_MdM = sum(M.conj().T @ M for M in truncated_kraus)
        identity = np.eye(dim, dtype=np.complex128)
        assert np.allclose(sum_MdM, identity, atol=1e-10), \
            "Truncamiento violó conservación de traza"
        
        # Verificar que fidelidad está acotada
        assert 0.0 <= metrics.uhlmann_fidelity <= 1.0, \
            "Fidelidad fuera de rango [0,1]"
    
    def test_truncation_error_bounds(self):
        """Verifica cotas de error de truncamiento."""
        dim = 2
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.5)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # Truncar drásticamente
        truncated_kraus, metrics = EnvironmentTruncator.truncate_with_renormalization(
            choi, max_rank=1
        )
        
        # Distancia de traza debe satisfacer D ≤ √(2ε)
        # donde ε es el spectral gap
        expected_upper_bound = np.sqrt(2 * metrics.spectral_gap)
        
        assert metrics.trace_distance <= expected_upper_bound + 1e-10, \
            f"Distancia de traza excede cota teórica: {metrics.trace_distance} > {expected_upper_bound}"
        
        # Fidelidad debe satisfacer F ≥ 1 - ε
        expected_lower_bound = 1.0 - metrics.spectral_gap
        
        assert metrics.uhlmann_fidelity >= expected_lower_bound - 1e-10, \
            f"Fidelidad bajo cota teórica: {metrics.uhlmann_fidelity} < {expected_lower_bound}"
    
    def test_truncation_convergence_to_identity(self):
        """Verifica que truncamiento extremo converge al canal trivial."""
        dim = 2
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.9)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus_ops, dim, dim)
        
        # Truncar a 1 solo operador
        truncated_kraus, _ = EnvironmentTruncator.truncate_with_renormalization(
            choi, max_rank=1
        )
        
        assert len(truncated_kraus) == 1, "Truncamiento a 1 falló"
        
        # Debe ser proporcional a identidad (canal unital)
        M = truncated_kraus[0]
        # Verificar ∑ M^† M = I (ya verificado arriba)
        # El canal con 1 Kraus es unitario
        assert np.allclose(M.conj().T @ M, np.eye(dim), atol=1e-10), \
            "Único Kraus no satisface unitariedad"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE MÉTRICAS DE INFORMACIÓN CUÁNTICA
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumInformationMetrics:
    """Pruebas de métricas de información cuántica."""
    
    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_von_neumann_entropy_pure_state(self, dim):
        """Verifica que entropía de estado puro es 0."""
        rho = QuantumStateGenerator.pure_state(dim)
        entropy = QuantumInformationMetrics.von_neumann_entropy(rho)
        
        assert entropy < 1e-10, \
            f"Entropía de estado puro ≠ 0: S = {entropy}"
    
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_von_neumann_entropy_maximally_mixed(self, dim):
        """Verifica entropía de estado maximalmente mixto."""
        rho = QuantumStateGenerator.maximally_mixed(dim)
        entropy = QuantumInformationMetrics.von_neumann_entropy(rho)
        
        expected = np.log2(dim)
        assert np.isclose(entropy, expected, atol=1e-10), \
            f"Entropía de estado maximalmente mixto incorrecta: {entropy} vs {expected}"
    
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_purity_bounds(self, dim):
        """Verifica cotas de pureza."""
        rho = QuantumStateGenerator.random_density_matrix(dim)
        purity = QuantumInformationMetrics.purity(rho)
        
        # 1/d ≤ γ ≤ 1
        assert 1/dim - 1e-10 <= purity <= 1.0 + 1e-10, \
            f"Pureza fuera de rango: γ = {purity}"
    
    def test_linear_entropy_relation(self):
        """Verifica relación S_L = 1 - γ."""
        dim = 4
        rho = QuantumStateGenerator.random_density_matrix(dim)
        
        purity = QuantumInformationMetrics.purity(rho)
        linear_entropy = QuantumInformationMetrics.linear_entropy(rho)
        
        assert np.isclose(linear_entropy, 1.0 - purity, atol=1e-12), \
            "Relación S_L = 1 - γ violada"
    
    def test_coherence_l1_norm(self):
        """Verifica coherencia cuántica."""
        # Estado diagonal (sin coherencia)
        rho_diagonal = np.diag([0.6, 0.3, 0.1]).astype(np.complex128)
        coherence_diag = QuantumInformationMetrics.coherence_l1_norm(rho_diagonal)
        
        assert np.isclose(coherence_diag, 0.0, atol=1e-12), \
            "Estado diagonal debe tener coherencia nula"
        
        # Estado con coherencia
        rho = QuantumStateGenerator.pure_state(3)
        coherence = QuantumInformationMetrics.coherence_l1_norm(rho)
        
        assert coherence > 0, \
            "Estado puro genérico debe tener coherencia positiva"
    
    def test_entropy_monotonicity_under_mixing(self):
        """Verifica que entropía aumenta al mezclar estados."""
        dim = 3
        rho_pure = QuantumStateGenerator.pure_state(dim)
        rho_mixed = QuantumStateGenerator.maximally_mixed(dim)
        
        # Mezcla: ρ_λ = λ ρ_pure + (1-λ) ρ_mixed
        lambda_val = 0.5
        rho_mixture = lambda_val * rho_pure + (1 - lambda_val) * rho_mixed
        
        S_pure = QuantumInformationMetrics.von_neumann_entropy(rho_pure)
        S_mixture = QuantumInformationMetrics.von_neumann_entropy(rho_mixture)
        S_mixed = QuantumInformationMetrics.von_neumann_entropy(rho_mixed)
        
        # S(ρ_pure) ≤ S(ρ_mixture) ≤ S(ρ_mixed)
        assert S_pure <= S_mixture + 1e-10, \
            "Entropía no monotónica al mezclar (1)"
        assert S_mixture <= S_mixed + 1e-10, \
            "Entropía no monotónica al mezclar (2)"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS INTEGRADAS DEL FIBRADOR DE STINESPRING
# ══════════════════════════════════════════════════════════════════════════════

class TestStinespringIsometricFibrator:
    """Pruebas integradas del fibrador completo."""
    
    def test_initialization(self, fibrator_small):
        """Verifica inicialización correcta."""
        assert fibrator_small.mic_dim == 2
        assert fibrator_small.mac_dim == 2
        assert fibrator_small.max_env_dim == 20
    
    def test_identity_channel_elevation(self, fibrator_small):
        """Verifica elevación con canal identidad."""
        dim = 2
        rho_mic = QuantumStateGenerator.random_density_matrix(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        
        rho_mac_atomic = fibrator_small.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        # Canal identidad debe preservar el estado
        assert np.allclose(rho_mac_atomic.matrix, rho_mic, atol=1e-10), \
            "Canal identidad no preservó el estado"
    
    @pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8])
    def test_depolarizing_channel_elevation(self, fibrator_small, p):
        """Verifica elevación con canal depolarizante."""
        dim = 2
        rho_mic = QuantumStateGenerator.pure_state(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, p)
        
        rho_mac_atomic = fibrator_small.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        rho_mac = rho_mac_atomic.matrix
        
        # Verificar propiedades físicas
        assert np.allclose(rho_mac, rho_mac.conj().T, atol=1e-12), \
            "Estado de salida no hermítico"
        
        eigs = la.eigvalsh(rho_mac)
        assert np.all(eigs >= -NumericalThresholds.POSITIVITY_TOL), \
            "Estado de salida no positivo"
        
        trace = np.trace(rho_mac).real
        assert np.isclose(trace, 1.0, atol=1e-12), \
            "Traza no preservada"
        
        # Para p=1, debe dar estado maximalmente mixto
        if np.isclose(p, 1.0, atol=1e-6):
            expected = np.eye(dim, dtype=np.complex128) / dim
            assert np.allclose(rho_mac, expected, atol=1e-8), \
                "Depolarizante completo no da estado maximalmente mixto"
    
    def test_amplitude_damping_elevation(self, fibrator_small):
        """Verifica elevación con amplitude-damping."""
        dim = 2
        gamma = 0.5
        
        # Estado excitado |1⟩
        rho_mic = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        kraus_ops = KrausOperatorGenerator.amplitude_damping_channel(gamma)
        
        rho_mac_atomic = fibrator_small.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        rho_mac = rho_mac_atomic.matrix
        
        # Verificar decaimiento: ⟨0|ρ_out|0⟩ ≈ γ
        population_ground = rho_mac[0, 0].real
        assert np.isclose(population_ground, gamma, atol=1e-10), \
            f"Decaimiento incorrecto: esperado {gamma}, obtenido {population_ground}"
    
    def test_truncation_integration(self):
        """Verifica integración de truncamiento."""
        # Crear fibrador con max_env_dim pequeño
        fibrator = StinespringIsometricFibrator(
            mic_dim=2, mac_dim=2, max_env_dim=2
        )
        
        dim = 2
        rho_mic = QuantumStateGenerator.random_density_matrix(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        # Canal con 4 Kraus → debe truncar a 2
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.7)
        
        rho_mac_atomic = fibrator.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        # Verificar que se aplicó truncamiento
        fidelity_metrics = fibrator.get_last_fidelity_metrics()
        assert fidelity_metrics is not None, \
            "Truncamiento no se aplicó cuando debería"
        
        # Fidelidad debe ser < 1 (hubo pérdida)
        assert fidelity_metrics.uhlmann_fidelity < 1.0, \
            "Fidelidad = 1 a pesar de truncamiento"
    
    def test_entropy_increase_under_decoherence(self, fibrator_small):
        """Verifica aumento de entropía bajo decoherencia."""
        dim = 2
        rho_mic = QuantumStateGenerator.pure_state(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        # Canal de fase-damping (aumenta entropía)
        gamma = 0.6
        kraus_ops = KrausOperatorGenerator.phase_damping_channel(gamma)
        
        S_initial = QuantumInformationMetrics.von_neumann_entropy(rho_mic)
        
        rho_mac_atomic = fibrator_small.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        S_final = QuantumInformationMetrics.von_neumann_entropy(rho_mac_atomic.matrix)
        
        # Entropía debe aumentar (segunda ley)
        assert S_final >= S_initial - 1e-10, \
            f"Entropía disminuyó: {S_initial} → {S_final}"
    
    def test_purity_decrease_under_mixing(self, fibrator_small):
        """Verifica disminución de pureza bajo canales mezcladores."""
        dim = 2
        rho_mic = QuantumStateGenerator.pure_state(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.5)
        
        purity_initial = QuantumInformationMetrics.purity(rho_mic)
        
        rho_mac_atomic = fibrator_small.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        purity_final = QuantumInformationMetrics.purity(rho_mac_atomic.matrix)
        
        # Pureza debe disminuir
        assert purity_final <= purity_initial + 1e-10, \
            f"Pureza aumentó: {purity_initial} → {purity_final}"
    
    def test_audit_report_generation(self, fibrator_small):
        """Verifica generación de reporte de auditoría."""
        dim = 2
        rho_mic = QuantumStateGenerator.random_density_matrix(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.4)
        
        fibrator_small.elevate_quantum_state(rho_mic_atomic, kraus_ops)
        
        report = fibrator_small.generate_audit_report()
        
        # Verificar estructura del reporte
        assert "channel_properties" in report
        assert "isometry_validation" in report
        assert "truncation_metrics" in report
        
        # Verificar propiedades del canal
        assert report["channel_properties"]["completely_positive"]
        assert report["channel_properties"]["trace_preserving"]
        
        # Verificar métricas de isometría
        assert report["isometry_validation"]["environment_dimension"] is not None
        assert report["isometry_validation"]["isometry_error"] < 1e-8
    
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_scalability(self, dim):
        """Verifica escalabilidad a diferentes dimensiones."""
        fibrator = StinespringIsometricFibrator(
            mic_dim=dim, mac_dim=dim, max_env_dim=50
        )
        
        rho_mic = QuantumStateGenerator.random_density_matrix(dim)
        rho_mic_atomic = AtomicDensityMatrix(matrix=rho_mic, dimension=dim)
        
        # Canal identidad (simple pero válido para todas las dimensiones)
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        
        rho_mac_atomic = fibrator.elevate_quantum_state(
            rho_mic_atomic, kraus_ops
        )
        
        # Verificar que el resultado es válido
        assert rho_mac_atomic.dimension == dim
        assert np.allclose(rho_mac_atomic.matrix, rho_mic, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE PROPIEDADES CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════════════════

class TestCategoricalProperties:
    """Pruebas de propiedades categóricas del funtor."""
    
    def test_functoriality_composition(self):
        """Verifica que F(Φ ∘ Ψ) = F(Φ) ∘ F(Ψ)."""
        dim = 2
        
        # Dos canales
        gamma1 = 0.3
        gamma2 = 0.4
        kraus1 = KrausOperatorGenerator.amplitude_damping_channel(gamma1)
        kraus2 = KrausOperatorGenerator.amplitude_damping_channel(gamma2)
        
        # Composición de canales: Φ(Ψ(ρ))
        # Kraus de composición: {M_i^Φ M_j^Ψ}
        kraus_composed = []
        for M1 in kraus1:
            for M2 in kraus2:
                kraus_composed.append(M1 @ M2)
        
        # Estado de prueba
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
        
        # Método 1: Aplicar composición directamente
        fibrator = StinespringIsometricFibrator(dim, dim, max_env_dim=20)
        rho_out1_atomic = fibrator.elevate_quantum_state(rho_atomic, kraus_composed)
        rho_out1 = rho_out1_atomic.matrix
        
        # Método 2: Aplicar secuencialmente
        rho_temp_atomic = fibrator.elevate_quantum_state(rho_atomic, kraus2)
        rho_out2_atomic = fibrator.elevate_quantum_state(rho_temp_atomic, kraus1)
        rho_out2 = rho_out2_atomic.matrix
        
        # Deben coincidir (funtoridad)
        assert np.allclose(rho_out1, rho_out2, atol=1e-9), \
            "Funtoridad violada: F(Φ∘Ψ) ≠ F(Φ)∘F(Ψ)"
    
    def test_functoriality_identity(self):
        """Verifica que F(id) = id."""
        dim = 3
        fibrator = StinespringIsometricFibrator(dim, dim, max_env_dim=20)
        
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
        
        kraus_id = KrausOperatorGenerator.identity_channel(dim)
        
        rho_out_atomic = fibrator.elevate_quantum_state(rho_atomic, kraus_id)
        
        # F(id)(ρ) = ρ
        assert np.allclose(rho_out_atomic.matrix, rho, atol=1e-12), \
            "F(id) ≠ id"
    
    def test_linearity_of_channel(self):
        """Verifica linealidad del canal: Φ(aρ + bσ) = aΦ(ρ) + bΦ(σ)."""
        dim = 2
        fibrator = StinespringIsometricFibrator(dim, dim, max_env_dim=20)
        
        rho1 = QuantumStateGenerator.random_density_matrix(dim)
        rho2 = QuantumStateGenerator.random_density_matrix(dim)
        
        a, b = 0.3, 0.7
        rho_combined = a * rho1 + b * rho2
        
        kraus_ops = KrausOperatorGenerator.depolarizing_channel(dim, 0.4)
        
        # Φ(aρ + bσ)
        rho_combined_atomic = AtomicDensityMatrix(matrix=rho_combined, dimension=dim)
        rho_out_combined_atomic = fibrator.elevate_quantum_state(
            rho_combined_atomic, kraus_ops
        )
        rho_out_combined = rho_out_combined_atomic.matrix
        
        # aΦ(ρ) + bΦ(σ)
        rho1_atomic = AtomicDensityMatrix(matrix=rho1, dimension=dim)
        rho2_atomic = AtomicDensityMatrix(matrix=rho2, dimension=dim)
        
        rho_out1_atomic = fibrator.elevate_quantum_state(rho1_atomic, kraus_ops)
        rho_out2_atomic = fibrator.elevate_quantum_state(rho2_atomic, kraus_ops)
        
        rho_out_linear = a * rho_out1_atomic.matrix + b * rho_out2_atomic.matrix
        
        assert np.allclose(rho_out_combined, rho_out_linear, atol=1e-10), \
            "Linealidad del canal violada"


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE CASOS EXTREMOS Y MANEJO DE ERRORES
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndErrorHandling:
    """Pruebas de casos extremos y manejo robusto de errores."""
    
    def test_non_positive_input_state_rejection(self, fibrator_small):
        """Verifica rechazo de estados no positivos."""
        dim = 2
        # Crear estado con autovalores negativos
        rho_invalid = np.array([
            [0.5, 0.6],
            [0.6, 0.5]
        ], dtype=np.complex128)
        # Autovalores: 1.1 y -0.1 (negativo)
        
        rho_atomic = AtomicDensityMatrix(matrix=rho_invalid, dimension=dim)
        kraus_ops = KrausOperatorGenerator.identity_channel(dim)
        
        with pytest.raises(TraceAnomalyError, match="no es positivo"):
            fibrator_small.elevate_quantum_state(rho_atomic, kraus_ops)
    
    def test_non_cptp_kraus_handling(self, fibrator_small):
        """Verifica manejo de operadores no-CPTP."""
        dim = 2
        # Crear Kraus que NO preservan traza
        M1 = np.array([[1.5, 0], [0, 1.5]], dtype=np.complex128)
        kraus_invalid = [M1]
        
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
        
        # Debe advertir pero intentar proceder con corrección
        # (dependiendo de implementación, puede lanzar excepción)
        with pytest.raises((TraceAnomalyError, NumericalInstabilityError)):
            fibrator_small.elevate_quantum_state(rho_atomic, kraus_invalid)
    
    def test_zero_rank_choi_handling(self):
        """Verifica manejo de canal con Choi de rango cero."""
        dim = 2
        # Crear Kraus nulos (caso degenerado)
        M_zero = np.zeros((dim, dim), dtype=np.complex128)
        kraus_zero = [M_zero]
        
        fibrator = StinespringIsometricFibrator(dim, dim)
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
        
        with pytest.raises((NumericalInstabilityError, TraceAnomalyError)):
            fibrator.elevate_quantum_state(rho_atomic, kraus_zero)
    
    def test_numerical_stability_with_ill_conditioned_kraus(self):
        """Verifica estabilidad con Kraus mal condicionados."""
        dim = 2
        # Crear Kraus muy mal condicionados
        M1 = np.array([[1.0, 1e-8], [1e-8, 1e-15]], dtype=np.complex128)
        M2 = np.array([[0, 1e-8], [1e-8, 0]], dtype=np.complex128)
        
        # Normalizar para que sean CPTP
        norm_factor = la.norm(M1.conj().T @ M1 + M2.conj().T @ M2)
        M1 /= np.sqrt(norm_factor)
        M2 /= np.sqrt(norm_factor)
        
        kraus_ill = [M1, M2]
        
        fibrator = StinespringIsometricFibrator(dim, dim)
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
        
        # Debe proceder con advertencias sobre condicionamiento
        try:
            rho_out_atomic = fibrator.elevate_quantum_state(rho_atomic, kraus_ill)
            # Verificar que salida sigue siendo física
            rho_out = rho_out_atomic.matrix
            assert np.allclose(rho_out, rho_out.conj().T, atol=1e-8)
            eigs = la.eigvalsh(rho_out)
            assert np.all(eigs >= -1e-6)
        except NumericalInstabilityError:
            # También aceptable si detecta inestabilidad crítica
            pass
    
    @pytest.mark.parametrize("dim", [1, 10, 20])
    def test_dimension_limits(self, dim):
        """Verifica manejo de dimensiones extremas."""
        if dim == 1:
            # Dimensión 1: todos los estados son idénticos
            fibrator = StinespringIsometricFibrator(dim, dim)
            rho = np.array([[1.0]], dtype=np.complex128)
            rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
            kraus = [np.array([[1.0]], dtype=np.complex128)]
            
            rho_out_atomic = fibrator.elevate_quantum_state(rho_atomic, kraus)
            assert np.isclose(rho_out_atomic.matrix[0, 0], 1.0)
        
        else:
            # Dimensiones grandes: verificar que no explota
            fibrator = StinespringIsometricFibrator(dim, dim, max_env_dim=min(dim*dim, 50))
            rho = QuantumStateGenerator.random_density_matrix(dim)
            rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
            kraus = KrausOperatorGenerator.identity_channel(dim)
            
            rho_out_atomic = fibrator.elevate_quantum_state(rho_atomic, kraus)
            assert rho_out_atomic.dimension == dim


# ══════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS DE RENDIMIENTO Y BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformanceAndBenchmarks:
    """Pruebas de rendimiento y benchmarking."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("dim", [2, 4, 8, 16])
    def test_scaling_with_dimension(self, dim, benchmark):
        """Benchmark: escalamiento con dimensión."""
        fibrator = StinespringIsometricFibrator(dim, dim, max_env_dim=100)
        rho = QuantumStateGenerator.random_density_matrix(dim)
        rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
        kraus = KrausOperatorGenerator.identity_channel(dim)
        
        def elevate():
            return fibrator.elevate_quantum_state(rho_atomic, kraus)
        
        result = benchmark(elevate)
        assert result.dimension == dim
    
    @pytest.mark.slow
    def test_truncation_performance(self, benchmark):
        """Benchmark: rendimiento de truncamiento."""
        dim = 2
        kraus = KrausOperatorGenerator.depolarizing_channel(dim, 0.8)
        choi = ChoiOperatorFactory.from_kraus_operators(kraus, dim, dim)
        
        def truncate():
            return EnvironmentTruncator.truncate_with_renormalization(choi, max_rank=2)
        
        truncated, metrics = benchmark(truncate)
        assert len(truncated) <= 2
    
    @pytest.mark.slow
    def test_memory_efficiency_large_env(self):
        """Verifica eficiencia de memoria con entornos grandes."""
        import tracemalloc
        
        dim = 4
        fibrator = StinespringIsometricFibrator(dim, dim, max_env_dim=20)
        
        tracemalloc.start()
        
        for _ in range(10):
            rho = QuantumStateGenerator.random_density_matrix(dim)
            rho_atomic = AtomicDensityMatrix(matrix=rho, dimension=dim)
            kraus = KrausOperatorGenerator.depolarizing_channel(dim, 0.5)
            
            fibrator.elevate_quantum_state(rho_atomic, kraus)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verificar que memoria no crece descontroladamente
        assert peak < 100 * 1024 * 1024, \
            f"Uso de memoria excesivo: {peak / 1024**2:.2f} MB"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configuración personalizada de pytest."""
    config.addinivalue_line(
        "markers", "slow: marca pruebas lentas (deshabilitadas por defecto)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-m", "not slow",  # Excluir pruebas lentas por defecto
    ])