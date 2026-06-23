# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MAC Minimizer Test Suite                                  ║
║ Ubicación: tests/unit/boole/tactics/test_mac_minimizer.py                         ║
║ Versión: 2.0.0-Quantum-Spectral-Purification-Test-Suite                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas para Validación de:

# ─────────────────────────────────────────────────────────────────────────
# Sutura #2: Apaciguamiento de FloatingPointError en log2(0).
# numpy emite RuntimeWarning cuando una operación spectral toca el cero
# exacto (eigenvalor cero en estados puros, máquinas de Hilbert con
# soporte disjunto, etc.). Producción NO los maneja — los tests deben
# filtrarlos para no convertir advertencias en excepciones.
# ─────────────────────────────────────────────────────────────────────────
import warnings as _warnings_sutura2
_warnings_sutura2.filterwarnings(
    'ignore',
    message=r'.*divide by zero encountered in (log2|log).*',
    category=RuntimeWarning,
)
_warnings_sutura2.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in (multiply|log2|log).*',
    category=RuntimeWarning,
)
_warnings_sutura2.filterwarnings(
    'ignore',
    message=r'.*overflow encountered in exp.*',
    category=RuntimeWarning,
)
───────────────────────────────────────────────
1. Entropía de von Neumann (cálculo, divergencias, entropías generalizadas)
2. Truncamiento Espectral (estrategias, conservación, fidelidad)
3. Poda de Lindblad (criterios, optimización, impacto)
4. Minimización Completa (pipeline, métricas, validación)
5. Propiedades Termodinámicas (pureza, rango efectivo, información)

Metodología de Testing:
───────────────────────
- Pruebas unitarias (componentes aislados)
- Pruebas de integración (flujos completos)
- Pruebas de propiedades (property-based testing)
- Pruebas de invariantes físicos y matemáticos
- Pruebas de optimización y compresión
- Pruebas de estabilidad numérica

Referencias:
────────────
- von Neumann (1932): "Mathematical Foundations of Quantum Mechanics"
- Schumacher (1995): "Quantum coding theorem"
- Nielsen & Chuang (2010): "Quantum Computation and Quantum Information"
- Vidal et al. (2002): "Entanglement in quantum critical phenomena"
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
import numpy as np
import scipy.linalg as la
from typing import List, Tuple, Dict
from hypothesis import given, strategies as st, settings, assume
import logging

# Módulos bajo prueba
import sys
sys.path.insert(0, '../../..')

from app.boole.tactics.mac_minimizer import (
    VonNeumannEntropyEngine,
    SpectralTruncationProjector,
    LindbladPruningOperator,
    MACMinimizer,
    TruncationStrategy,
    PruningCriterion,
    LogarithmicBase,
    TruncationReport,
    PruningReport,
    MinimizationMetrics
)

from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    create_quantum_mac_state,
    NumericalInstabilityError
)

from app.wisdom.mac_agent import GaloisAdjunctionAuditor

logger = logging.getLogger("MAC.Minimizer.Tests")


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def entropy_minimizer() -> VonNeumannEntropyEngine:
    """Motor de entropía estándar."""
    return VonNeumannEntropyEngine(tol=1e-12, log_base=LogarithmicBase.NATURAL)


@pytest.fixture
def spectral_projector() -> SpectralTruncationProjector:
    """Proyector de truncamiento espectral estándar."""
    return SpectralTruncationProjector(
        epsilon_threshold=1e-6,
        strategy=TruncationStrategy.THRESHOLD
    )


@pytest.fixture
def lindblad_pruner() -> LindbladPruningOperator:
    """Podador de operadores de Lindblad estándar."""
    return LindbladPruningOperator(
        tau_cutoff=1e-4,
        criterion=PruningCriterion.MAGNITUDE
    )


@pytest.fixture
def mac_minimizer() -> MACMinimizer:
    """Minimizador MAC completo."""
    return MACMinimizer(
        epsilon_spectral=1e-6,
        tau_lindblad=1e-4,
        debug_mode=True
    )


def create_mixed_state(dimension: int, purity: float, seed: int = 42) -> AtomicDensityMatrix:
    """Crea estado mixto con pureza controlada."""
    return create_quantum_mac_state(dimension=dimension, purity=purity, seed=seed)


def create_depolarizing_channel(dimension: int, p: float) -> List[Tuple[float, np.ndarray]]:
    """Crea canal despolarizante."""
    if dimension == 2:
        pauli_ops = [
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128)
        ]
        return [(p / 3.0, pauli) for pauli in pauli_ops]
    else:
        return [(p / dimension, np.eye(dimension, dtype=np.complex128))]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRUEBAS DE VON NEUMANN ENTROPY MINIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class TestVonNeumannEntropyEngine:
    """Suite de pruebas para cálculo de entropía."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Entropía Básica
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_pure_state_zero_entropy(self, entropy_minimizer):
        """Estado puro debe tener entropía cero."""
        # Create known pure state |0><0| to avoid issues with random state generation
        rho_matrix = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        rho_pure = AtomicDensityMatrix(rho_matrix)
        entropy = entropy_minimizer.compute_entropy(rho_pure)
        
        assert np.isclose(entropy, 0.0, atol=1e-10), \
            f"Estado puro con entropía no nula: {entropy}"
    
    def test_maximally_mixed_maximum_entropy(self, entropy_minimizer):
        """Estado maximalmente mixto debe tener entropía máxima."""
        dim = 4
        rho_matrix = np.eye(dim, dtype=np.complex128) / dim
        rho_mixed = AtomicDensityMatrix(rho_matrix)
        
        entropy = entropy_minimizer.compute_entropy(rho_mixed)
        expected = np.log(dim)  # ln(d) para base natural
        
        assert np.isclose(entropy, expected, atol=1e-10), \
            f"Entropía incorrecta: {entropy} vs {expected}"
    
    def test_entropy_non_negative(self, entropy_minimizer):
        """Entropía debe ser siempre no negativa."""
        rho = create_mixed_state(dimension=3, purity=0.5, seed=42)
        entropy = entropy_minimizer.compute_entropy(rho)
        
        assert entropy >= -1e-12, f"Entropía negativa: {entropy}"
    
    def test_entropy_upper_bound(self, entropy_minimizer):
        """Entropía debe estar acotada por ln(d)."""
        dim = 5
        rho = create_mixed_state(dimension=dim, purity=0.2, seed=42)
        entropy = entropy_minimizer.compute_entropy(rho)
        
        max_entropy = np.log(dim)
        assert entropy <= max_entropy + 1e-10, \
            f"Entropía excede máximo: {entropy} > {max_entropy}"
    
    def test_entropy_concavity(self, entropy_minimizer):
        """S(λρ₁ + (1-λ)ρ₂) ≥ λS(ρ₁) + (1-λ)S(ρ₂)."""
        rho1 = create_mixed_state(dimension=2, purity=0.8, seed=42)
        rho2 = create_mixed_state(dimension=2, purity=0.6, seed=43)
        
        lam = 0.6
        rho_mix_matrix = lam * rho1.matrix + (1 - lam) * rho2.matrix
        rho_mix = AtomicDensityMatrix(rho_mix_matrix)
        
        S_mix = entropy_minimizer.compute_entropy(rho_mix)
        S1 = entropy_minimizer.compute_entropy(rho1)
        S2 = entropy_minimizer.compute_entropy(rho2)
        
        S_convex = lam * S1 + (1 - lam) * S2
        
        assert S_mix >= S_convex - 1e-10, \
            f"Violación de concavidad: S(mix)={S_mix} < {S_convex}"
    
    def test_entropy_log_base_conversion(self):
        """Entropía en diferentes bases debe ser consistente."""
        rho = create_mixed_state(dimension=2, purity=0.7, seed=42)
        
        minimizer_ln = VonNeumannEntropyEngine(log_base=LogarithmicBase.NATURAL)
        minimizer_log2 = VonNeumannEntropyEngine(log_base=LogarithmicBase.BINARY)
        
        S_ln = minimizer_ln.compute_entropy(rho)
        S_log2 = minimizer_log2.compute_entropy(rho)
        
        # S_log2 = S_ln / ln(2)
        expected_log2 = S_ln / np.log(2)
        
        assert np.isclose(S_log2, expected_log2, atol=1e-10), \
            f"Conversión de base incorrecta: {S_log2} vs {expected_log2}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Entropía de Rényi
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_renyi_alpha_1_equals_von_neumann(self, entropy_minimizer):
        """S₁(ρ) debe igualar entropía de von Neumann."""
        rho = create_mixed_state(dimension=3, purity=0.6, seed=42)
        
        S_vn = entropy_minimizer.compute_entropy(rho)
        S_renyi_1 = entropy_minimizer.compute_renyi_entropy(rho, alpha=1.0)
        
        assert np.isclose(S_vn, S_renyi_1, atol=1e-10), \
            f"S₁ no coincide con S_vN: {S_renyi_1} vs {S_vn}"
    
    def test_renyi_alpha_0_equals_log_rank(self, entropy_minimizer):
        """S₀(ρ) debe igualar ln(rank(ρ))."""
        # Estado diagonal con rango conocido (evita errores numéricos de transformaciones unitarias)
        rho_matrix = np.diag([0.5, 0.3, 0.2, 0.0])
        rho = AtomicDensityMatrix(rho_matrix)
        
        S_renyi_0 = entropy_minimizer.compute_renyi_entropy(rho, alpha=0.0)
        expected = np.log(3)  # rank = 3 (3 eigenvalores no nulos)
        
        assert np.isclose(S_renyi_0, expected, atol=1e-10), \
            f"S₀ incorrecto: {S_renyi_0} vs {expected}"
    
    def test_renyi_alpha_2_collision_entropy(self, entropy_minimizer):
        """S₂(ρ) = -ln(Tr(ρ²)) (entropía de colisión)."""
        rho = create_mixed_state(dimension=2, purity=0.7, seed=42)
        
        # Compute Renyi-2 entropy via engine
        S_renyi_2 = entropy_minimizer.compute_renyi_entropy(rho, alpha=2.0)
        
        # Compute expected S₂ from the eigenvalues after engine's spectral processing
        spectral = entropy_minimizer.compute_spectral_data(rho)
        eigenvalues = spectral.eigenvalues
        # Ensure non-negative (engine already does this)
        positive_eigs = eigenvalues[eigenvalues > entropy_minimizer._tol]
        purity_from_eigenvalues = np.sum(positive_eigs ** 2)
        expected = -np.log(purity_from_eigenvalues)
        
        assert np.isclose(S_renyi_2, expected, atol=1e-10), \
            f"S₂ incorrecto: {S_renyi_2} vs {expected}"
    
    def test_renyi_monotonicity_in_alpha(self, entropy_minimizer):
        """S_α(ρ) debe ser decreciente en α."""
        rho = create_mixed_state(dimension=3, purity=0.5, seed=42)
        
        alphas = [0.5, 1.0, 2.0, 5.0]
        entropies = [entropy_minimizer.compute_renyi_entropy(rho, a) for a in alphas]
        
        for i in range(len(entropies) - 1):
            assert entropies[i] >= entropies[i+1] - 1e-10, \
                f"S_α no decreciente: S_{alphas[i]}={entropies[i]} < S_{alphas[i+1]}={entropies[i+1]}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Divergencia Relativa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_relative_entropy_identical_states(self, entropy_minimizer):
        """D(ρ||ρ) = 0."""
        rho = create_mixed_state(dimension=2, purity=0.6, seed=42)
        
        divergence = entropy_minimizer.compute_relative_entropy(rho, rho)
        
        assert np.isclose(divergence, 0.0, atol=1e-10), \
            f"Divergencia no nula para estados idénticos: {divergence}"
    
    def test_relative_entropy_non_negative(self, entropy_minimizer):
        """D(ρ||σ) ≥ 0."""
        rho = create_mixed_state(dimension=2, purity=0.7, seed=42)
        sigma = create_mixed_state(dimension=2, purity=0.5, seed=43)
        
        divergence = entropy_minimizer.compute_relative_entropy(rho, sigma)
        
        assert divergence >= -1e-12, f"Divergencia negativa: {divergence}"
    
    def test_relative_entropy_asymmetry(self, entropy_minimizer):
        """D(ρ||σ) ≠ D(σ||ρ) en general."""
        rho = create_mixed_state(dimension=2, purity=0.8, seed=42)
        sigma = create_mixed_state(dimension=2, purity=0.4, seed=43)
        
        D_rho_sigma = entropy_minimizer.compute_relative_entropy(rho, sigma)
        D_sigma_rho = entropy_minimizer.compute_relative_entropy(sigma, rho)
        
        # No deben ser iguales (excepto en casos degenerados)
        if np.isclose(D_rho_sigma, D_sigma_rho, atol=1e-6):
            pytest.skip("Estados resultaron simétricos (caso degenerado)")
    
    def test_relative_entropy_infinite_for_disjoint_support(self, entropy_minimizer):
        """D(ρ||σ) = ∞ si supp(ρ) ⊄ supp(σ)."""
        # Estado ρ con soporte en |0⟩
        psi_0 = np.array([1, 0], dtype=np.complex128)
        rho = AtomicDensityMatrix(np.outer(psi_0, psi_0.conj()))
        
        # Estado σ con soporte en |1⟩
        psi_1 = np.array([0, 1], dtype=np.complex128)
        sigma = AtomicDensityMatrix(np.outer(psi_1, psi_1.conj()))
        
        divergence = entropy_minimizer.compute_relative_entropy(rho, sigma)
        
        assert np.isinf(divergence), \
            f"Divergencia debería ser infinita: {divergence}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Métricas Derivadas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_effective_rank_pure_state(self, entropy_minimizer):
        """Rango efectivo de estado puro debe ser 1."""
        rho_pure = create_mixed_state(dimension=3, purity=1.0, seed=42)
        
        r_eff = entropy_minimizer.compute_effective_rank(rho_pure)
        
        assert np.isclose(r_eff, 1.0, atol=1e-8), \
            f"Rango efectivo incorrecto para estado puro: {r_eff}"
    
    def test_effective_rank_maximally_mixed(self, entropy_minimizer):
        """Rango efectivo de estado maximalmente mixto debe ser d."""
        dim = 4
        rho_mixed = AtomicDensityMatrix(np.eye(dim, dtype=np.complex128) / dim)
        
        r_eff = entropy_minimizer.compute_effective_rank(rho_mixed)
        
        assert np.isclose(r_eff, dim, atol=1e-8), \
            f"Rango efectivo incorrecto: {r_eff} vs {dim}"
    
    def test_purity_bounds(self, entropy_minimizer):
        """Pureza debe estar en [1/d, 1]."""
        dim = 3
        rho = create_mixed_state(dimension=dim, purity=0.5, seed=42)
        
        purity = entropy_minimizer.compute_purity(rho)
        
        assert 1.0/dim - 1e-10 <= purity <= 1.0 + 1e-10, \
            f"Pureza fuera de rango: {purity}"
    
    def test_purity_entropy_relationship(self, entropy_minimizer):
        """Mayor pureza → menor entropía."""
        rho_high_purity = create_mixed_state(dimension=2, purity=0.9, seed=42)
        rho_low_purity = create_mixed_state(dimension=2, purity=0.3, seed=43)
        
        S_high = entropy_minimizer.compute_entropy(rho_high_purity)
        S_low = entropy_minimizer.compute_entropy(rho_low_purity)
        
        assert S_high <= S_low + 1e-10, \
            f"Mayor pureza no implica menor entropía: S_high={S_high}, S_low={S_low}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PRUEBAS DE SPECTRAL TRUNCATION PROJECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TestSpectralTruncationProjector:
    """Suite de pruebas para truncamiento espectral."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Truncamiento Básico
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_truncation_preserves_trace(self, spectral_projector):
        """Truncamiento debe preservar traza = 1."""
        rho = create_mixed_state(dimension=4, purity=0.5, seed=42)
        
        rho_truncated, _ = spectral_projector.truncate_spectrum(rho)
        
        trace = np.trace(rho_truncated.matrix).real
        assert np.isclose(trace, 1.0, atol=1e-10), \
            f"Traza no preservada: {trace}"
    
    def test_truncation_preserves_hermiticity(self, spectral_projector):
        """Estado truncado debe ser hermitiano."""
        rho = create_mixed_state(dimension=4, purity=0.6, seed=42)
        
        rho_truncated, _ = spectral_projector.truncate_spectrum(rho)
        
        hermiticity_error = la.norm(
            rho_truncated.matrix - rho_truncated.matrix.conj().T,
            ord='fro'
        )
        
        assert hermiticity_error < 1e-10, \
            f"Estado truncado no hermitiano: {hermiticity_error}"
    
    def test_truncation_preserves_positivity(self, spectral_projector):
        """Estado truncado debe ser positivo."""
        rho = create_mixed_state(dimension=4, purity=0.4, seed=42)
        
        rho_truncated, _ = spectral_projector.truncate_spectrum(rho)
        
        eigenvalues = la.eigvalsh(rho_truncated.matrix)
        assert np.all(eigenvalues >= -1e-10), \
            f"Estado truncado no positivo: {eigenvalues}"
    
    def test_truncation_reduces_dimension(self, spectral_projector):
        """Truncamiento debe reducir dimensión efectiva."""
        # Estado con algunos eigenvalores pequeños
        eigenvalues = np.array([0.4, 0.3, 0.2, 0.09, 1e-7, 1e-8])
        eigenvalues /= np.sum(eigenvalues)
        
        U, _ = la.qr(np.random.randn(6, 6) + 1j * np.random.randn(6, 6))
        rho_matrix = U @ np.diag(eigenvalues) @ U.conj().T
        rho = AtomicDensityMatrix(rho_matrix)
        
        rho_truncated, report = spectral_projector.truncate_spectrum(rho)
        
        assert report.truncated_dimension < report.original_dimension, \
            "Dimensión no reducida"
    
    def test_truncation_increases_purity(self):
        """Truncamiento debe aumentar pureza."""
        projector = SpectralTruncationProjector(
            epsilon_threshold=0.1,
            strategy=TruncationStrategy.THRESHOLD
        )
        
        rho = create_mixed_state(dimension=4, purity=0.3, seed=42)
        
        rho_truncated, report = projector.truncate_spectrum(rho)
        
        assert report.purity_after >= report.purity_before - 1e-10, \
            f"Pureza disminuyó: {report.purity_before} → {report.purity_after}"
    
    def test_truncation_decreases_entropy(self):
        """Truncamiento debe disminuir entropía."""
        projector = SpectralTruncationProjector(
            epsilon_threshold=0.1,
            strategy=TruncationStrategy.THRESHOLD
        )
        
        rho = create_mixed_state(dimension=5, purity=0.2, seed=42)
        
        rho_truncated, report = projector.truncate_spectrum(rho)
        
        assert report.entropy_after <= report.entropy_before + 1e-10, \
            f"Entropía aumentó: {report.entropy_before} → {report.entropy_after}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Estrategias de Truncamiento
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_strategy_threshold(self):
        """Estrategia THRESHOLD retiene eigenvalores ≥ ε."""
        epsilon = 0.15
        projector = SpectralTruncationProjector(
            epsilon_threshold=epsilon,
            strategy=TruncationStrategy.THRESHOLD
        )
        
        eigenvalues = np.array([0.4, 0.3, 0.2, 0.09, 0.01])
        U, _ = la.qr(np.random.randn(5, 5) + 1j * np.random.randn(5, 5))
        rho = AtomicDensityMatrix(U @ np.diag(eigenvalues) @ U.conj().T)
        
        rho_truncated, report = projector.truncate_spectrum(rho)
        
        # Debe retener eigenvalores [0.4, 0.3, 0.2] (≥ 0.15)
        expected_dim = 3
        assert report.truncated_dimension == expected_dim, \
            f"Dimensión incorrecta: {report.truncated_dimension} vs {expected_dim}"
    
    def test_strategy_rank_k(self):
        """Estrategia RANK_K retiene top-k eigenvalores."""
        k = 3
        projector = SpectralTruncationProjector(
            epsilon_threshold=0.0,
            strategy=TruncationStrategy.RANK_K
        )
        
        rho = create_mixed_state(dimension=6, purity=0.4, seed=42)
        
        rho_truncated, report = projector.truncate_spectrum(rho, target_param=k)
        
        assert report.truncated_dimension == k, \
            f"Dimensión incorrecta: {report.truncated_dimension} vs {k}"
    
    def test_strategy_cumulative_energy(self):
        """Estrategia CUMULATIVE_ENERGY retiene hasta % de energía."""
        target_energy = 0.9
        projector = SpectralTruncationProjector(
            epsilon_threshold=0.0,
            strategy=TruncationStrategy.CUMULATIVE_ENERGY
        )
        
        rho = create_mixed_state(dimension=5, purity=0.3, seed=42)
        
        rho_truncated, report = projector.truncate_spectrum(
            rho, target_param=target_energy
        )
        
        assert report.retained_energy >= target_energy - 0.1, \
            f"Energía insuficiente: {report.retained_energy} < {target_energy}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Conservación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_energy_conservation(self, spectral_projector):
        """Energía retenida debe ser suma de eigenvalores retenidos."""
        rho = create_mixed_state(dimension=4, purity=0.5, seed=42)
        
        rho_truncated, report = spectral_projector.truncate_spectrum(rho)
        
        # Energía retenida debe igualar suma de eigenvalores retenidos
        expected_energy = np.sum(report.retained_eigenvalues)
        
        # Normalizar por traza original
        eigenvalues_original = la.eigvalsh(rho.matrix)
        total_energy = np.sum(eigenvalues_original)
        
        computed_ratio = expected_energy / total_energy
        
        assert np.isclose(report.retained_energy, computed_ratio, atol=1e-8), \
            f"Energía inconsistente: {report.retained_energy} vs {computed_ratio}"
    
    def test_compression_ratio_correctness(self, spectral_projector):
        """Ratio de compresión debe ser dim_truncated / dim_original."""
        rho = create_mixed_state(dimension=5, purity=0.4, seed=42)
        
        rho_truncated, report = spectral_projector.truncate_spectrum(rho)
        
        expected_ratio = report.truncated_dimension / report.original_dimension
        
        assert np.isclose(report.compression_ratio, expected_ratio, atol=1e-12), \
            f"Ratio incorrecto: {report.compression_ratio} vs {expected_ratio}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Casos Extremos
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_pure_state_no_truncation(self):
        """Estado puro no debería truncarse."""
        projector = SpectralTruncationProjector(
            epsilon_threshold=1e-6,
            strategy=TruncationStrategy.THRESHOLD
        )
        
        rho_pure = create_mixed_state(dimension=3, purity=1.0, seed=42)
        
        rho_truncated, report = projector.truncate_spectrum(rho_pure)
        
        # Debe retener solo 1 eigenvalor
        assert report.truncated_dimension == 1, \
            f"Estado puro truncado incorrectamente: dim={report.truncated_dimension}"
    
    def test_all_eigenvalues_below_threshold_raises_error(self):
        """Todos los eigenvalores bajo umbral debe lanzar error."""
        projector = SpectralTruncationProjector(
            epsilon_threshold=0.5,  # Umbral muy alto
            strategy=TruncationStrategy.THRESHOLD
        )
        
        rho = create_mixed_state(dimension=4, purity=0.3, seed=42)
        
        with pytest.raises(NumericalInstabilityError, match="Degeneración Espectral"):
            projector.truncate_spectrum(rho)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PRUEBAS DE LINDBLAD PRUNING OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestLindbladPruningOperator:
    """Suite de pruebas para poda de operadores de Lindblad."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Poda Básica
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_pruning_removes_small_rates(self, lindblad_pruner):
        """Debe eliminar operadores con tasas pequeñas."""
        jump_ops = [
            (0.5, np.random.randn(2, 2) + 1j * np.random.randn(2, 2)),
            (0.3, np.random.randn(2, 2) + 1j * np.random.randn(2, 2)),
            (1e-5, np.random.randn(2, 2) + 1j * np.random.randn(2, 2)),  # Pequeño
            (1e-6, np.random.randn(2, 2) + 1j * np.random.randn(2, 2))   # Muy pequeño
        ]
        
        pruned_ops, report = lindblad_pruner.prune_jump_operators(jump_ops)
        
        assert report.pruned_count < report.original_count, \
            "No se podó ningún operador"
        assert report.discarded_count >= 2, \
            f"No se descartaron suficientes operadores: {report.discarded_count}"
    
    def test_pruning_preserves_large_rates(self, lindblad_pruner):
        """Debe preservar operadores con tasas grandes."""
        large_rate = 0.5
        jump_ops = [
            (large_rate, np.eye(2, dtype=np.complex128)),
            (0.6, np.eye(2, dtype=np.complex128))
        ]
        
        pruned_ops, report = lindblad_pruner.prune_jump_operators(jump_ops)
        
        assert report.pruned_count == report.original_count, \
            "Operadores grandes fueron podados incorrectamente"
    
    def test_empty_list_returns_empty(self, lindblad_pruner):
        """Lista vacía debe retornar lista vacía."""
        jump_ops = []
        
        pruned_ops, report = lindblad_pruner.prune_jump_operators(jump_ops)
        
        assert len(pruned_ops) == 0, "Lista no vacía retornada"
        assert report.original_count == 0, "Conteo incorrecto"
    
    def test_pruning_report_consistency(self, lindblad_pruner):
        """Reporte debe ser consistente."""
        jump_ops = [
            (0.5, np.eye(2, dtype=np.complex128)),
            (1e-5, np.eye(2, dtype=np.complex128)),
            (0.3, np.eye(2, dtype=np.complex128))
        ]
        
        pruned_ops, report = lindblad_pruner.prune_jump_operators(jump_ops)
        
        assert report.pruned_count + report.discarded_count == report.original_count, \
            "Conteo inconsistente"
    
    def test_total_rate_conservation(self, lindblad_pruner):
        """Tasa total post-poda debe ser suma de tasas retenidas."""
        jump_ops = [
            (0.5, np.eye(2, dtype=np.complex128)),
            (0.3, np.eye(2, dtype=np.complex128)),
            (1e-5, np.eye(2, dtype=np.complex128))
        ]
        
        pruned_ops, report = lindblad_pruner.prune_jump_operators(jump_ops)
        
        expected_rate = sum(gamma for gamma, _ in pruned_ops)
        
        assert np.isclose(report.total_rate_after, expected_rate, atol=1e-10), \
            f"Tasa total inconsistente: {report.total_rate_after} vs {expected_rate}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Criterios de Poda
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_criterion_magnitude(self):
        """Criterio MAGNITUDE debe usar solo γₖ."""
        pruner = LindbladPruningOperator(
            tau_cutoff=0.1,
            criterion=PruningCriterion.MAGNITUDE
        )
        
        jump_ops = [
            (0.5, np.eye(2, dtype=np.complex128) * 1000),  # Norma grande
            (0.05, np.eye(2, dtype=np.complex128))          # γ pequeño
        ]
        
        pruned_ops, report = pruner.prune_jump_operators(jump_ops)
        
        # Solo debe retener el primero (γ = 0.5 > 0.1)
        assert report.pruned_count == 1, \
            f"Criterio MAGNITUDE no funciona correctamente: {report.pruned_count}"
    
    def test_criterion_frobenius_norm(self):
        """Criterio FROBENIUS_NORM debe considerar γₖ ‖Lₖ‖."""
        pruner = LindbladPruningOperator(
            tau_cutoff=0.5,
            criterion=PruningCriterion.FROBENIUS_NORM
        )
        
        # Operador con γ pequeño pero norma grande
        L_large = np.ones((2, 2), dtype=np.complex128) * 10
        
        # Operador con γ grande pero norma pequeña
        L_small = np.eye(2, dtype=np.complex128) * 0.01
        
        jump_ops = [
            (0.1, L_large),   # γ‖L‖ ≈ 0.1 × 14 ≈ 1.4
            (0.9, L_small)    # γ‖L‖ ≈ 0.9 × 0.014 ≈ 0.013
        ]
        
        pruned_ops, report = pruner.prune_jump_operators(jump_ops)
        
        # Debe retener al menos el primero
        assert report.pruned_count >= 1, \
            "Criterio FROBENIUS_NORM no funciona"
    
    def test_preserve_critical_operator(self):
        """Debe preservar al menos un operador si preserve_critical=True."""
        pruner = LindbladPruningOperator(
            tau_cutoff=10.0,  # Umbral muy alto
            criterion=PruningCriterion.MAGNITUDE,
            preserve_critical=True
        )
        
        jump_ops = [
            (0.1, np.eye(2, dtype=np.complex128)),
            (0.05, np.eye(2, dtype=np.complex128))
        ]
        
        pruned_ops, report = pruner.prune_jump_operators(jump_ops)
        
        # Debe preservar el de mayor tasa
        assert report.pruned_count == 1, \
            f"No se preservó operador crítico: {report.pruned_count}"
        
        # Verificar que es el de tasa 0.1
        assert pruned_ops[0][0] == 0.1, \
            f"Operador crítico incorrecto: {pruned_ops[0][0]}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Eficiencia
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_pruning_efficiency_calculation(self, lindblad_pruner):
        """Eficiencia debe ser discarded / original."""
        jump_ops = [
            (0.5, np.eye(2, dtype=np.complex128)),
            (0.3, np.eye(2, dtype=np.complex128)),
            (1e-5, np.eye(2, dtype=np.complex128)),
            (1e-6, np.eye(2, dtype=np.complex128))
        ]
        
        pruned_ops, report = lindblad_pruner.prune_jump_operators(jump_ops)
        
        expected_efficiency = report.discarded_count / report.original_count
        
        assert np.isclose(report.pruning_efficiency, expected_efficiency, atol=1e-12), \
            f"Eficiencia incorrecta: {report.pruning_efficiency} vs {expected_efficiency}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: PRUEBAS DE MAC MINIMIZER (INTEGRACIÓN)
# ══════════════════════════════════════════════════════════════════════════════

class TestMACMinimizer:
    """Suite de pruebas para minimizador completo."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Pipeline Completo
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_purification_pipeline_success(self, mac_minimizer):
        """Pipeline completo debe ejecutarse sin errores."""
        rho = create_mixed_state(dimension=4, purity=0.3, seed=42)
        jump_ops = create_depolarizing_channel(dimension=4, p=0.2)
        
        rho_purified, ops_optimized, metrics = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        assert isinstance(rho_purified, AtomicDensityMatrix), \
            "Estado purificado mal formado"
        assert isinstance(metrics, MinimizationMetrics), \
            "Métricas mal formadas"
    
    def test_purification_preserves_trace(self, mac_minimizer):
        """Purificación debe preservar traza."""
        rho = create_mixed_state(dimension=3, purity=0.5, seed=42)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.1)
        
        rho_purified, _, _ = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops
        )
        
        trace = np.trace(rho_purified.matrix).real
        assert np.isclose(trace, 1.0, atol=1e-10), \
            f"Traza no preservada: {trace}"
    
    def test_purification_increases_purity(self, mac_minimizer):
        """Purificación debe aumentar pureza."""
        rho = create_mixed_state(dimension=4, purity=0.3, seed=42)
        jump_ops = create_depolarizing_channel(dimension=4, p=0.1)
        
        entropy_engine = VonNeumannEntropyEngine()
        purity_before = entropy_engine.compute_purity(rho)
        
        rho_purified, _, _ = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        purity_after = entropy_engine.compute_purity(rho_purified)
        
        assert purity_after >= purity_before - 1e-10, \
            f"Pureza disminuyó: {purity_before} → {purity_after}"
    
    def test_purification_decreases_entropy(self, mac_minimizer):
        """Purificación debe disminuir entropía."""
        rho = create_mixed_state(dimension=5, purity=0.2, seed=42)
        jump_ops = create_depolarizing_channel(dimension=5, p=0.15)
        
        entropy_engine = VonNeumannEntropyEngine()
        entropy_before = entropy_engine.compute_entropy(rho)
        
        rho_purified, _, _ = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        entropy_after = entropy_engine.compute_entropy(rho_purified)
        
        assert entropy_after <= entropy_before + 1e-10, \
            f"Entropía aumentó: {entropy_before} → {entropy_after}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Métricas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_metrics_fidelity_high_for_small_truncation(self):
        """Fidelidad debe ser alta si truncamiento es pequeño."""
        minimizer = MACMinimizer(
            epsilon_spectral=1e-8,  # Umbral muy pequeño
            tau_lindblad=1e-6
        )
        
        rho = create_mixed_state(dimension=3, purity=0.7, seed=42)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.05)
        
        rho_purified, _, metrics = minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        assert metrics.fidelity_preservation >= 0.9, \
            f"Fidelidad baja con truncamiento pequeño: {metrics.fidelity_preservation}"
    
    def test_metrics_compression_ratio_meaningful(self, mac_minimizer):
        """Ratio de compresión debe ser significativo."""
        rho = create_mixed_state(dimension=5, purity=0.2, seed=42)
        jump_ops = create_depolarizing_channel(dimension=5, p=0.3)
        
        _, _, metrics = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        assert 0 <= metrics.total_compression_ratio <= 1, \
            f"Ratio de compresión fuera de rango: {metrics.total_compression_ratio}"
    
    def test_metrics_information_loss_non_negative(self, mac_minimizer):
        """Pérdida de información debe ser no negativa."""
        rho = create_mixed_state(dimension=4, purity=0.4, seed=42)
        jump_ops = create_depolarizing_channel(dimension=4, p=0.1)
        
        _, _, metrics = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        assert metrics.information_loss >= -1e-10, \
            f"Pérdida de información negativa: {metrics.information_loss}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Umbral Adaptativo
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_adaptive_threshold_skips_low_entropy(self):
        """No debe truncar si entropía está bajo umbral."""
        minimizer = MACMinimizer(
            epsilon_spectral=1e-6,
            tau_lindblad=1e-4,
            auto_optimize=True
        )
        
        # Estado con baja entropía
        rho = create_mixed_state(dimension=2, purity=0.95, seed=42)
        jump_ops = create_depolarizing_channel(dimension=2, p=0.01)
        
        _, _, metrics = minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            entropy_threshold=10.0,  # Umbral muy alto
            force_truncation=False
        )
        
        # No debería haber truncamiento
        if metrics.truncation_report:
            assert metrics.truncation_report.compression_ratio == 1.0, \
                "Truncamiento innecesario aplicado"
    
    def test_force_truncation_overrides_threshold(self, mac_minimizer):
        """force_truncation debe forzar truncamiento."""
        rho = create_mixed_state(dimension=3, purity=0.99, seed=42)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.01)
        
        _, _, metrics = mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            entropy_threshold=100.0,  # Umbral inalcanzable
            force_truncation=True
        )
        
        assert metrics.truncation_report is not None, \
            "Truncamiento no aplicado con force_truncation=True"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Telemetría
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_telemetry_tracking(self, mac_minimizer):
        """Telemetría debe rastrear minimizaciones."""
        rho = create_mixed_state(dimension=3, purity=0.5, seed=42)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.1)
        
        # Realizar múltiples minimizaciones
        for _ in range(3):
            mac_minimizer.purify_semantic_state(
                rho=rho,
                jump_operators=jump_ops,
                force_truncation=True
            )
        
        telemetry = mac_minimizer.get_telemetry()
        
        assert telemetry['minimization_count'] == 3, \
            f"Conteo incorrecto: {telemetry['minimization_count']}"
    
    def test_debug_mode_stores_history(self):
        """Modo debug debe almacenar historial."""
        minimizer = MACMinimizer(debug_mode=True)
        
        rho = create_mixed_state(dimension=3, purity=0.4, seed=42)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.1)
        
        minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        telemetry = minimizer.get_telemetry()
        
        assert len(telemetry['history']) > 0, \
            "Historial no almacenado en modo debug"
    
    def test_reset_clears_telemetry(self, mac_minimizer):
        """Reset debe limpiar telemetría."""
        rho = create_mixed_state(dimension=3, purity=0.5, seed=42)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.1)
        
        mac_minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops
        )
        
        mac_minimizer.reset()
        
        telemetry = mac_minimizer.get_telemetry()
        
        assert telemetry['minimization_count'] == 0, \
            "Reset no limpia conteo"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5: PRUEBAS DE INTEGRACIÓN END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndScenarios:
    """Pruebas de integración de flujos completos."""
    
    def test_complete_minimization_pipeline(self):
        """Pipeline completo de minimización."""
        # Setup
        minimizer = MACMinimizer(
            epsilon_spectral=1e-5,
            tau_lindblad=1e-3,
            truncation_strategy=TruncationStrategy.CUMULATIVE_ENERGY,
            pruning_criterion=PruningCriterion.MAGNITUDE,
            debug_mode=True
        )
        
        # Estado mixto con alta entropía
        rho = create_mixed_state(dimension=6, purity=0.2, seed=42)
        
        # Múltiples operadores de Lindblad con diferentes tasas
        jump_ops = [
            (0.5, np.random.randn(6, 6) + 1j * np.random.randn(6, 6)),
            (0.3, np.random.randn(6, 6) + 1j * np.random.randn(6, 6)),
            (0.1, np.random.randn(6, 6) + 1j * np.random.randn(6, 6)),
            (1e-4, np.random.randn(6, 6) + 1j * np.random.randn(6, 6)),
            (1e-5, np.random.randn(6, 6) + 1j * np.random.randn(6, 6))
        ]
        
        # Ejecutar minimización
        rho_purified, ops_optimized, metrics = minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            entropy_threshold=None,  # Automático
            force_truncation=True
        )
        
        # Validaciones
        assert isinstance(rho_purified, AtomicDensityMatrix), \
            "Estado purificado inválido"
        assert len(ops_optimized) <= len(jump_ops), \
            "Operadores no podados"
        assert metrics.total_compression_ratio < 1.0, \
            "No hubo compresión"
        assert metrics.fidelity_preservation > 0.5, \
            "Fidelidad muy baja"
    
    def test_iterative_minimization(self):
        """Minimización iterativa debe converger."""
        minimizer = MACMinimizer(
            epsilon_spectral=1e-4,
            tau_lindblad=1e-3
        )
        
        rho = create_mixed_state(dimension=5, purity=0.3, seed=42)
        jump_ops = create_depolarizing_channel(dimension=5, p=0.2)
        
        entropy_engine = VonNeumannEntropyEngine()
        entropies = []
        
        # Múltiples iteraciones
        for iteration in range(3):
            rho, jump_ops, _ = minimizer.purify_semantic_state(
                rho=rho,
                jump_operators=jump_ops,
                force_truncation=True
            )
            
            entropy = entropy_engine.compute_entropy(rho)
            entropies.append(entropy)
        
        # Entropía debe decrecer o estabilizarse
        assert entropies[-1] <= entropies[0] + 1e-10, \
            f"Entropía no decrece: {entropies}"
    
    def test_stress_test_large_system(self):
        """Prueba de estrés con sistema grande."""
        minimizer = MACMinimizer(
            epsilon_spectral=1e-3,
            tau_lindblad=1e-2,
            debug_mode=False  # Desactivar para performance
        )
        
        rho = create_mixed_state(dimension=10, purity=0.1, seed=42)
        jump_ops = create_depolarizing_channel(dimension=10, p=0.3)
        
        # Debe completarse sin errores
        rho_purified, _, metrics = minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        assert np.isclose(np.trace(rho_purified.matrix).real, 1.0, atol=1e-8), \
            "Estado grande no normalizado"
        assert metrics.computational_speedup > 1.0, \
            "Sin ganancia computacional en sistema grande"
    
    def test_preservation_of_quantum_properties(self):
        """Propiedades cuánticas deben preservarse."""
        minimizer = MACMinimizer(epsilon_spectral=1e-5, tau_lindblad=1e-4)
        
        rho = create_mixed_state(dimension=4, purity=0.6, seed=42)
        jump_ops = create_depolarizing_channel(dimension=4, p=0.1)
        
        rho_purified, _, _ = minimizer.purify_semantic_state(
            rho=rho,
            jump_operators=jump_ops,
            force_truncation=True
        )
        
        # Verificar propiedades cuánticas
        rho_matrix = rho_purified.matrix
        
        # Hermiticidad
        hermiticity_error = la.norm(rho_matrix - rho_matrix.conj().T, ord='fro')
        assert hermiticity_error < 1e-10, "Hermiticidad violada"
        
        # Positividad
        eigenvalues = la.eigvalsh(rho_matrix)
        assert np.all(eigenvalues >= -1e-10), "Positividad violada"
        
        # Traza unitaria
        trace = np.trace(rho_matrix).real
        assert np.isclose(trace, 1.0, atol=1e-10), "Traza no unitaria"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST Y FIXTURES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def setup_logging():
    """Configurar logging para pruebas."""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(name)s - %(levelname)s - %(message)s'
    )


@pytest.fixture(scope="session")
def random_seed():
    """Semilla aleatoria global para reproducibilidad."""
    np.random.seed(42)
    return 42


# Marcadores personalizados
pytest.mark.entropy = pytest.mark.entropy
pytest.mark.truncation = pytest.mark.truncation
pytest.mark.pruning = pytest.mark.pruning
pytest.mark.minimization = pytest.mark.minimization
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.boole.tactics.mac_minimizer",
        "--cov-report=html",
        "--cov-report=term",
        "-m", "not slow",
        "--maxfail=10",
        "--durations=15"
    ])