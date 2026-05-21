# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pytest

# Fase 1: Esterilización del Vacío Termodinámico
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from app.core.quantum_algebra import HilbertSpace, QuantumDensityOperator, QuantumRegistry
from app.adapters.tools_interface import TopologicalInvariantError

class TestQuantumCore:
    """
    Oráculo de Pruebas Cuánticas para la Fase 2.
    """

    def test_phase_2_1_hilbert_space_integrity(self):
        """
        Verifica la instanciación del Espacio de Hilbert HN y su Base Ortonormal.
        Axioma: rank(T) = N y G = Identity.
        """
        N = 5
        h_space = HilbertSpace.create_canonical(N)
        assert h_space.dimension == N
        # La integridad se verifica en el __post_init__ de HilbertSpace

        # Verificar manualmente la matriz de Gram
        gram = h_space.basis.conj().T @ h_space.basis
        assert np.allclose(gram, np.eye(N))

        # Verificar rango
        s = np.linalg.svd(h_space.basis, compute_uv=False)
        assert np.sum(s > 1e-12) == N

    def test_phase_2_2_density_matrix_axioms(self):
        """
        Verifica la construcción invariante de rho y sus Hard Vetoes.
        """
        # Estado Puro
        psi = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
        rho_pure = QuantumDensityOperator.from_pure_state(psi)
        assert np.allclose(np.trace(rho_pure.rho), 1.0)

        # Estado Mixto
        weights = [0.5, 0.5]
        states = [
            np.array([1, 0, 0, 0, 0], dtype=np.complex128),
            np.array([0, 1, 0, 0, 0], dtype=np.complex128)
        ]
        rho_mixed = QuantumDensityOperator.from_mixed_state(weights, states)
        assert np.allclose(np.trace(rho_mixed.rho), 1.0)
        assert np.allclose(rho_mixed.rho, rho_mixed.rho.conj().T)

    def test_phase_2_2_hard_vetoes(self):
        """
        Inyecta operadores degenerados y exige TopologicalInvariantError.
        """
        # 1. Violación de Traza Unitaria (Tr(rho) = 1.05)
        rho_bad_trace = np.eye(5, dtype=np.complex128) * (1.05 / 5.0)
        with pytest.raises(TopologicalInvariantError, match="Violación de Traza Unitaria"):
            QuantumDensityOperator(rho_bad_trace)

        # 2. Violación de Hermiticidad
        rho_non_hermitian = np.array([
            [0.5, 0.1],
            [0.2, 0.5]
        ], dtype=np.complex128)
        # Ajustar para dimension 5 o usar N=2
        rho_non_hermitian_full = np.eye(5, dtype=np.complex128) * 0.2
        rho_non_hermitian_full[0, 1] = 0.1
        rho_non_hermitian_full[1, 0] = 0.2
        # Pero Tr debe ser 1
        trace = np.trace(rho_non_hermitian_full)
        rho_non_hermitian_full = rho_non_hermitian_full / trace

        with pytest.raises(TopologicalInvariantError, match="Violación de Hermiticidad"):
            QuantumDensityOperator(rho_non_hermitian_full)

        # 3. Violación de Positividad Semidefinida (autovalor negativo)
        rho_negative = np.diag([1.1, -0.1, 0, 0, 0]).astype(np.complex128)
        with pytest.raises(TopologicalInvariantError, match="Violación de Positividad Semidefinida"):
            QuantumDensityOperator(rho_negative)

    def test_phase_2_3_von_neumann_entropy(self):
        """
        Verifica el cálculo de la Entropía de Von Neumann y la distinción de estados.
        """
        # 1. Estado Puro (Entropía Nula)
        psi = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
        rho_pure = QuantumDensityOperator.from_pure_state(psi)
        entropy_pure = rho_pure.compute_von_neumann_entropy()
        assert entropy_pure <= 1e-12
        assert rho_pure.is_pure_state()

        # 2. Estado Máximamente Mixto (Entropía Máxima)
        N = 5
        rho_mixed = np.eye(N, dtype=np.complex128) / N
        op_mixed = QuantumDensityOperator(rho_mixed)
        entropy_mixed = op_mixed.compute_von_neumann_entropy()

        expected_entropy = np.log(N)
        assert np.isclose(entropy_mixed, expected_entropy)
        assert not op_mixed.is_pure_state()
        assert entropy_mixed > 0

    def test_phase_2_4_observational_projectors(self):
        """
        Verifica la Resolución de Identidad y Exclusión Mutua de P1 y P2.
        """
        N = 4
        rho = np.eye(N, dtype=np.complex128) / N
        registry = QuantumRegistry(rho)

        psi = np.array([1, 1, 1, 1], dtype=np.complex128) / np.sqrt(4)
        p1_psi, p2_psi = registry.apply_observational_projectors(psi)

        # P1 + P2 = I => P1|psi> + P2|psi> = |psi>
        assert np.allclose(p1_psi + p2_psi, psi)

        # P1 P2 = 0 => <psi|P1 P2|psi> = 0
        assert np.allclose(np.vdot(p1_psi, p2_psi), 0.0)

    def test_phase_2_4_wkb_transmission(self):
        """
        Verifica el cálculo de transmisión WKB.
        """
        rho = np.eye(2, dtype=np.complex128) / 2.0
        registry = QuantumRegistry(rho)

        # E >= Phi => T = 1
        assert registry.calculate_wkb_transmission(10.0, 5.0) == 1.0

        # E < Phi => T < 1
        T = registry.calculate_wkb_transmission(5.0, 10.0)
        assert 0 < T < 1.0

if __name__ == "__main__":
    pytest.main([__file__])
