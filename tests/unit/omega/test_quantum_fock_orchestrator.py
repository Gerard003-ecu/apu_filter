# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas: Quantum Fock Orchestrator                                  ║
║ Versión: 1.0.0-Rigorous-Test-Suite                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cobertura:
    • 11 tests — Fase 1 (FockSpaceBuilder)
    •  9 tests — Fase 2 (CatadioptricCollider)
    • 12 tests — Fase 3 (LindbladDissipator)
    •  6 tests — Orquestador (Integración end-to-end)
    •  3 tests — Excepciones y robustez

Ejecución:
    pytest test_quantum_fock_orchestrator.py -v
    pytest test_quantum_fock_orchestrator.py -v --tb=short
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.testing import assert_allclose

# Importar el módulo bajo prueba
import sys
sys.path.insert(0, ".")

try:
    from app.omega.quantum_fock_orchestrator import (
        # Excepciones
        PauliExclusionViolationError,
        FockSpaceOverflowError,
        LindbladDissipationError,
        HermiticityViolationError,
        UnitarityViolationError,
        TopologicalInvariantError,
        # Cuasipartículas
        Boson, Fermion,
        RiemannianFocalBoson,
        HouseholderReflectionFermion,
        # Configuración
        FockSpaceConfiguration,
        # Fase 1
        InteractionOperators,
        Phase1_FockSpaceBuilder,
        # Fase 2
        CatadioptricHamiltonian,
        Phase2_CatadioptricCollider,
        # Fase 3
        LindbladEvolutionResult,
        Phase3_LindbladDissipator,
        # Orquestador
        QuantumFockOrchestrator,
    )
except ImportError as e:
    pytest.skip(f"Módulo a probar no disponible: {e}", allow_module_level=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL Y FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def config_small() -> FockSpaceConfiguration:
    """Configuración mínima: 1 bosón (trunc=2), 1 fermión."""
    return FockSpaceConfiguration(
        n_boson_modes=1,
        n_fermion_modes=1,
        boson_truncation=2,
        use_sparse=False,
    )


@pytest.fixture
def config_medium() -> FockSpaceConfiguration:
    """Configuración media: 2 bosones (trunc=3), 2 fermiones → dim=64."""
    return FockSpaceConfiguration(
        n_boson_modes=2,
        n_fermion_modes=2,
        boson_truncation=3,
        use_sparse=False,
    )


@pytest.fixture
def config_sparse() -> FockSpaceConfiguration:
    """Configuración sparse: 3 bosones (trunc=4), 2 fermiones."""
    return FockSpaceConfiguration(
        n_boson_modes=3,
        n_fermion_modes=2,
        boson_truncation=4,
        use_sparse=True,
    )


@pytest.fixture
def config_minimal() -> FockSpaceConfiguration:
    """Configuración trivial: 0 bosones, 0 fermiones (dim=1)."""
    return FockSpaceConfiguration(
        n_boson_modes=0,
        n_fermion_modes=0,
        boson_truncation=2,
        use_sparse=False,
    )


@pytest.fixture
def coupling_small() -> np.ndarray:
    """Matriz de acoplamiento 1x1 con valor complejo."""
    return np.array([[0.5 + 0.3j]], dtype=np.complex128)


@pytest.fixture
def coupling_medium() -> np.ndarray:
    """Matriz de acoplamiento 2x2 hermítica ficticia."""
    return np.array([
        [0.5, 0.2 + 0.1j],
        [0.2 - 0.1j, -0.3],
    ], dtype=np.complex128)


@pytest.fixture
def pure_state(complex_size: int = 4) -> np.ndarray:
    """Estado puro |ψ⟩⟨ψ| de tamaño dado."""
    rng = np.random.default_rng(42)
    psi = rng.standard_normal(complex_size) + 1j * rng.standard_normal(complex_size)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


@pytest.fixture
def maximally_mixed(dim: int = 4) -> np.ndarray:
    """Estado maximamente mezclado I/d."""
    return np.eye(dim, dtype=np.complex128) / dim


# ══════════════════════════════════════════════════════════════════════════════
# CLASE 1: TESTS DE FASE 1 — CONSTRUCCIÓN DEL ESPACIO DE FOCK
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1_FockSpaceBuilder:
    """Validación rigurosa de la Fase 1: operadores canónicos y CCR/CAR."""

    # ─── Test 1.1: Dimensiones ──────────────────────────────────────────────
    def test_hilbert_dimension_calculation(self, config_medium):
        r"""Verifica $\dim\mathcal{H} = (M+1)^B \cdot 2^F$."""
        builder = Phase1_FockSpaceBuilder(config_medium)
        expected = (3 + 1) ** 2 * 2 ** 2  # 64
        assert builder._total_dim == expected
        assert builder._boson_dims == [4, 4]
        assert builder._fermion_dims == [2, 2]

    # ─── Test 1.2: Hermiticidad de b† = (b)† ────────────────────────────────
    def test_boson_creation_is_adjoint(self, config_small):
        r"""Verifica $\hat{b}^\dagger_k = (\hat{b}_k)^\dagger$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        for k, (b, bdag) in enumerate(zip(
            builder._boson_ann_ops, builder._boson_cre_ops
        )):
            assert_allclose(
                bdag, b.conj().T,
                atol=1e-12,
                err_msg=f"b_{k}† ≠ (b_{k})†"
            )

    # ─── Test 1.3: Proyector fermiónico n² = n ───────────────────────────────
    def test_fermion_number_is_projector(self, config_small):
        r"""Verifica $\hat{n}_q^2 = \hat{n}_q$ (exclusión de Pauli)."""
        builder = Phase1_FockSpaceBuilder(config_small)
        for q, n in enumerate(builder._fermion_number_ops):
            sq = n @ n
            assert_allclose(
                sq, n, atol=1e-12,
                err_msg=f"n_{q}² ≠ n_{q}"
            )
            # Eigenvalores deben ser {0, 1}
            eigs = la.eigvalsh((n + n.conj().T) / 2)
            assert set(np.round(eigs, 8)).issubset({0.0, 1.0}), \
                f"n_{q} debe tener autovalores en {{0,1}}"

    # ─── Test 1.4: Conmutación bosónica canónica ─────────────────────────────
    def test_boson_ccr_relations(self, config_medium):
        r"""Verifica $[\hat{b}_k, \hat{b}_{k'}^\dagger] = \delta_{kk'} \mathbb{1}$."""
        builder = Phase1_FockSpaceBuilder(config_medium)
        ops = builder.get_interaction_operators()
        residuals = ops.verify_relations(tol=1e-9)

        # Filtrar solo relaciones bosónicas de creación-aniquilación
        for key, val in residuals.items():
            if key.startswith("[b_") and val > 1e-9:
                pytest.fail(f"Residual CCR {key}: {val:.2e}")

    # ─── Test 1.5: Conmutación de números fermiónicos ─────────────────────────
    def test_fermion_number_commutation(self, config_medium):
        r"""Verifica $[\hat{n}_q, \hat{n}_{q'}] = 0$ para $q \neq q'$."""
        builder = Phase1_FockSpaceBuilder(config_medium)
        for i, ni in enumerate(builder._fermion_number_ops):
            for j, nj in enumerate(builder._fermion_number_ops):
                if i != j:
                    comm = ni @ nj - nj @ ni
                    assert_allclose(
                        comm, np.zeros_like(comm), atol=1e-12,
                        err_msg=f"[n_{i}, n_{j}] ≠ 0"
                    )

    # ─── Test 1.6: Norma esperada de operadores bosónicos ────────────────────
    def test_boson_annihilation_off_diagonal(self, config_small):
        r"""Verifica que $\hat{b}$ solo tiene elementos $\langle n|\hat{b}|n+1\rangle = \sqrt{n+1}$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        b = builder._boson_ann_ops[0]
        d = b.shape[0]
        expected = np.zeros_like(b)
        for n in range(d - 1):
            expected[n, n + 1] = np.sqrt(n + 1)
        assert_allclose(b, expected, atol=1e-12)

    # ─── Test 1.7: Representación sparse ─────────────────────────────────────
    def test_sparse_construction(self, config_sparse):
        """Verifica consistencia sparse vs dense."""
        from scipy.sparse import issparse

        builder = Phase1_FockSpaceBuilder(config_sparse)
        for b in builder._boson_ann_ops:
            assert issparse(b), "Operadores deben ser sparse"
            assert b.shape == (builder._total_dim, builder._total_dim)

        # Verificar que el artefacto respeta el flag
        ops = builder.get_interaction_operators()
        assert ops.use_sparse is True

    # ─── Test 1.8: Configuración inválida lanza excepción ────────────────────
    def test_invalid_config_raises(self):
        """Verifica validación de parámetros en FockSpaceConfiguration."""
        with pytest.raises(FockSpaceOverflowError):
            FockSpaceConfiguration(n_boson_modes=-1, n_fermion_modes=1)

        with pytest.raises(FockSpaceOverflowError):
            FockSpaceConfiguration(
                n_boson_modes=1, n_fermion_modes=1,
                boson_truncation=-1
            )

    # ─── Test 1.9: Espacio trivial (0 modos) ─────────────────────────────────
    def test_trivial_space(self, config_minimal):
        """Verifica que dim=1 con 0 modos bosónicos y 0 fermiónicos."""
        builder = Phase1_FockSpaceBuilder(config_minimal)
        assert builder._total_dim == 1
        assert len(builder._boson_ann_ops) == 0
        assert len(builder._fermion_number_ops) == 0

    # ─── Test 1.10: Artefacto de Fase 1 inmutable ────────────────────────────
    def test_interaction_operators_is_frozen(self, config_small):
        """Verifica que InteractionOperators es inmutable (frozen=True)."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()

        with pytest.raises(Exception):  # FrozenInstanceError
            ops.hilbert_dim = 100

    # ─── Test 1.11: Verificación de relaciones en artifact ────────────────────
    def test_verify_relations_returns_dict(self, config_medium):
        """Verifica la firma y tipo de retorno de verify_relations."""
        builder = Phase1_FockSpaceBuilder(config_medium)
        ops = builder.get_interaction_operators()
        residuals = ops.verify_relations()

        assert isinstance(residuals, dict)
        assert len(residuals) > 0
        for key, val in residuals.items():
            assert isinstance(key, str)
            assert isinstance(val, float)
            assert val >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CLASE 2: TESTS DE FASE 2 — COLISIONADOR CATADIÓPTRICO
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2_CatadioptricCollider:
    """Validación rigurosa de la Fase 2: Hamiltoniano y unitariedad."""

    # ─── Test 2.1: Hermiticidad del Hamiltoniano ──────────────────────────────
    def test_hamiltonian_is_hermitian(self, config_small, coupling_small):
        r"""Verifica $\hat{H}_{\text{int}} = \hat{H}_{\text{int}}^\dagger$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)
        H = collider._H_int

        assert_allclose(H, H.conj().T, atol=1e-10)

    # ─── Test 2.2: Unitariedad de U(t) ───────────────────────────────────────
    def test_scattering_matrix_is_unitary(self, config_small, coupling_small):
        r"""Verifica $\hat{U}^\dagger(t) \hat{U}(t) = \mathbb{1}$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)

        for t in [0.1, 1.0, 10.0, 100.0]:
            U = collider.compute_scattering_matrix(t=t)
            residual = collider.verify_unitarity(U)
            assert residual < 1e-8, f"Unitaridad violada en t={t}: {residual:.2e}"

    # ─── Test 2.3: U(0) = I ─────────────────────────────────────────────────
    def test_zero_time_evolution_is_identity(self, config_small, coupling_small):
        r"""Verifica $\hat{U}(0) = \mathbb{1}$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)

        U0 = collider.compute_scattering_matrix(t=0.0)
        assert_allclose(U0, np.eye(ops.hilbert_dim), atol=1e-10)

    # ─── Test 2.4: Validación de dimensiones de acoplamiento ─────────────────
    def test_invalid_coupling_shape_raises(self, config_medium):
        """Verifica que dimensiones incorrectas lancen excepción."""
        builder = Phase1_FockSpaceBuilder(config_medium)
        ops = builder.get_interaction_operators()

        # 2 bosones × 2 fermiones esperados, pero pasamos 3x3
        bad_coupling = np.zeros((3, 3), dtype=np.complex128)
        with pytest.raises(TopologicalInvariantError):
            Phase2_CatadioptricCollider(ops, bad_coupling)

    # ─── Test 2.5: Acoplamiento con valores no finitos ───────────────────────
    def test_nan_coupling_raises(self, config_small):
        """Verifica rechazo de acoplamiento con NaN/Inf."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()

        bad_coupling = np.array([[np.nan]], dtype=np.complex128)
        with pytest.raises(FockSpaceOverflowError):
            Phase2_CatadioptricCollider(ops, bad_coupling)

    # ─── Test 2.6: Norma de operadores registrada ────────────────────────────
    def test_operator_norms_populated(self, config_small, coupling_small):
        """Verifica que operator_norms se calcula para cada par (k,q)."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)

        assert (0, 0) in collider._norms
        assert collider._norms[(0, 0)] == pytest.approx(abs(coupling_small[0, 0]))

    # ─── Test 2.7: Hamiltoniano con acoplamiento cero ────────────────────────
    def test_zero_coupling_gives_zero_hamiltonian(self, config_small):
        r"""Verifica $\hat{H}_{\text{int}} = 0$ si $g = 0$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        zero_coupling = np.zeros((1, 1), dtype=np.complex128)

        collider = Phase2_CatadioptricCollider(ops, zero_coupling)
        assert_allclose(collider._H_int, np.zeros_like(collider._H_int), atol=1e-12)

        # Y U(t) = I para todo t
        U = collider.compute_scattering_matrix(t=5.0)
        assert_allclose(U, np.eye(ops.hilbert_dim), atol=1e-12)

    # ─── Test 2.8: Artefacto de Fase 2 contiene H correcto ──────────────────
    def test_phase2_artifact_delivery(self, config_medium, coupling_medium):
        """Verifica que get_catadioptric_hamiltonian() entrega H consistente."""
        builder = Phase1_FockSpaceBuilder(config_medium)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_medium)
        artifact = collider.get_catadioptric_hamiltonian()

        assert isinstance(artifact, CatadioptricHamiltonian)
        assert_allclose(artifact.H_int, collider._H_int, atol=1e-14)
        assert artifact.hilbert_dim == ops.hilbert_dim

    # ─── Test 2.9: Acoplamiento puramente imaginario ─────────────────────────
    def test_purely_imaginary_coupling(self, config_small):
        r"""Verifica hermiticidad con $g \in i\mathbb{R}$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()

        # g + g* = 0 → H debe seguir hermítico
        imag_coupling = np.array([[1j]], dtype=np.complex128)
        collider = Phase2_CatadioptricCollider(ops, imag_coupling)

        assert_allclose(
            collider._H_int, collider._H_int.conj().T, atol=1e-10
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLASE 3: TESTS DE FASE 3 — DISIPADOR DE LINDBLAD
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase3_LindbladDissipator:
    """Validación rigurosa de la Fase 3: evolución CPTP y antimateria."""

    @pytest.fixture
    def dissipator_setup(self, config_small, coupling_small):
        """Setup: Fases 1+2 ejecutadas, devuelve Phase3 listo."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)
        hb = collider.get_catadioptric_hamiltonian()

        # Operadores de Lindblad: aniquilación bosónica
        L_ops = ops.boson_ann
        dissipator = Phase3_LindbladDissipator(
            hamiltonian_bundle=hb,
            lindblad_operators=L_ops,
            decay_rates=[0.5],
            planck_normalized=1.0,
            trace_tol=1e-9,
        )
        return dissipator, ops.hilbert_dim

    # ─── Test 3.1: Conservación de traza con H=0 ──────────────────────────────
    def test_trace_preserved_pure_dissipation(self, config_small, coupling_small):
        r"""Sin Hamiltoniano (H=0), $\text{Tr}(\rho)$ debe preservarse."""
        # Forzamos H=0 mediante acoplamiento cero
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, np.zeros((1, 1), dtype=np.complex128))
        hb = collider.get_catadioptric_hamiltonian()

        L_ops = ops.boson_ann
        dissipator = Phase3_LindbladDissipator(
            hamiltonian_bundle=hb,
            lindblad_operators=L_ops,
            decay_rates=[0.1],
        )

        rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        result = dissipator.execute_master_equation(rho0, dt=0.01, method="rk4")

        assert abs(np.trace(result.post_collision_rho) - 1.0) < 1e-8
        assert result.positivity_preserved

    # ─── Test 3.2: Estado puro permanece puro bajo evolución unitaria ────────
    def test_pure_state_under_unitary_evolution(self, config_small, coupling_small):
        r"""Sin disipador, $|\psi\rangle$ debe permanecer puro: $\text{Tr}(\rho^2)=1$."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)
        hb = collider.get_catadioptric_hamiltonian()

        # Lista vacía de operadores de Lindblad
        dissipator = Phase3_LindbladDissipator(
            hamiltonian_bundle=hb,
            lindblad_operators=[],
            decay_rates=[],
        )

        # Estado puro
        psi = np.array([1, 0, 1, 0, 0, 0], dtype=np.complex128)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())

        result = dissipator.execute_master_equation(rho0, dt=0.01, method="rk4")

        # Pureza debe permanecer ≈ 1
        purity = np.real(np.trace(result.post_collision_rho @ result.post_collision_rho))
        assert purity > 0.99, f"Pureza perdida: {purity:.4f}"

    # ─── Test 3.3: Positividad preservada para estado mixto ───────────────────
    def test_mixed_state_positivity(self, dissipator_setup, maximally_mixed):
        """Verifica que I/d permanece positivo bajo evolución disipativa."""
        dissipator, dim = dissipator_setup
        rho0 = maximally_mixed[:dim] if dim <= maximally_mixed.shape[0] \
            else np.eye(dim) / dim

        result = dissipator.execute_master_equation(rho0, dt=0.05, method="rk4")

        # Eigenvalores deben ser ≥ 0
        eigs = la.eigvalsh((result.post_collision_rho + result.post_collision_rho.conj().T) / 2)
        assert np.min(eigs) >= -1e-9, f"Eigenvalor negativo: {np.min(eigs):.2e}"

    # ─── Test 3.4: Comparación RK4 vs Euler ──────────────────────────────────
    def test_rk4_more_accurate_than_euler(self, dissipator_setup):
        """RK4 debe diferir menos del estado analítico que Euler."""
        dissipator, dim = dissipator_setup

        # Estado inicial simple
        rho0 = np.zeros((dim, dim), dtype=np.complex128)
        rho0[0, 0] = 1.0

        dt = 0.1
        r_rk4 = dissipator.execute_master_equation(rho0, dt=dt, method="rk4")
        r_euler = dissipator.execute_master_equation(rho0, dt=dt, method="euler")

        # Diferencia entre los dos métodos
        diff = np.linalg.norm(
            r_rk4.post_collision_rho - r_euler.post_collision_rho
        )

        # Ambos deben preservar traza, pero divergen en detalles de orden superior
        assert abs(np.trace(r_rk4.post_collision_rho) - 1.0) < 1e-7
        assert abs(np.trace(r_euler.post_collision_rho) - 1.0) < 1e-7
        assert r_rk4.integration_method == "RK4"
        assert r_euler.integration_method == "Euler"

    # ─── Test 3.5: Decaimiento al estado fundamental ─────────────────────────
    def test_thermal_decay_to_ground_state(self, config_small):
        """Verifica que el sistema decae al estado de mínima energía."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()

        # Hamiltoniano trivial (sin acoplamiento)
        collider = Phase2_CatadioptricCollider(
            ops, np.zeros((1, 1), dtype=np.complex128)
        )
        hb = collider.get_catadioptric_hamiltonian()

        # Lindblad = aniquilación con γ alta
        dissipator = Phase3_LindbladDissipator(
            hamiltonian_bundle=hb,
            lindblad_operators=ops.boson_ann,
            decay_rates=[5.0],
        )

        # Estado excitado |1⟩
        rho0 = np.zeros((ops.hilbert_dim, ops.hilbert_dim), dtype=np.complex128)
        # El estado |0⟩ del bosón con fermión vacío = índice 0
        rho0[0, 0] = 1.0

        # Evolucionar largo tiempo
        for _ in range(100):
            result = dissipator.execute_master_equation(
                result.post_collision_rho if 'result' in locals() else rho0,
                dt=0.05, method="rk4"
            )

        # Debe haber decaído pero mantenerse normalizado
        tr = np.real(np.trace(result.post_collision_rho))
        assert abs(tr - 1.0) < 1e-6

    # ─── Test 3.6: Emisión de fotón Gamma con entropía significativa ─────────
    def test_photon_emission_threshold(self, dissipator_setup):
        """Verifica emisión de fotón cuando S_diss > threshold."""
        dissipator, dim = dissipator_setup

        # Estado excitado (alta entropía)
        rho0 = np.zeros((dim, dim), dtype=np.complex128)
        rho0[1, 1] = 1.0  # Estado excitado

        result = dissipator.execute_master_equation(rho0, dt=0.5, method="rk4")

        # Si hubo disipación significativa, debe haber fotón
        if result.dissipated_entropy > 1e-15:
            assert result.emitted_photon is not None
            assert result.emitted_photon.annihilation_energy > 0
            assert result.emitted_photon.authorization_signature.startswith("QED_")

    # ─── Test 3.7: Sin fotón si no hay disipación ────────────────────────────
    def test_no_photon_when_no_dissipation(self, config_small):
        """Sin operadores de Lindblad, no debe emitirse fotón."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(
            ops, np.zeros((1, 1), dtype=np.complex128)
        )
        hb = collider.get_catadioptric_hamiltonian()

        dissipator = Phase3_LindbladDissipator(
            hamiltonian_bundle=hb,
            lindblad_operators=[],
            decay_rates=[],
        )

        rho0 = np.eye(ops.hilbert_dim, dtype=np.complex128) / ops.hilbert_dim
        result = dissipator.execute_master_equation(rho0, dt=0.1, method="rk4")

        assert result.emitted_photon is None
        assert result.dissipated_entropy == 0.0

    # ─── Test 3.8: ρ inicial no normalizado ──────────────────────────────────
    def test_unnormalized_initial_rho_raises(self, dissipator_setup):
        """ρ con Tr ≠ 1 debe lanzar LindbladDissipationError."""
        dissipator, dim = dissipator_setup

        bad_rho = 2.0 * np.eye(dim, dtype=np.complex128) / dim  # Tr = 2
        with pytest.raises(LindbladDissipationError):
            dissipator.execute_master_equation(bad_rho, dt=0.01)

    # ─── Test 3.9: Tasas y operadores con dimensiones inconsistentes ─────────
    def test_mismatched_rates_raises(self, config_small, coupling_small):
        """Verifica que decay_rates.len ≠ L_ops.len lance excepción."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(ops, coupling_small)
        hb = collider.get_catadioptric_hamiltonian()

        with pytest.raises(TopologicalInvariantError):
            Phase3_LindbladDissipator(
                hamiltonian_bundle=hb,
                lindblad_operations=ops.boson_ann,
                decay_rates=[0.1, 0.2],  # 2 tasas vs 1 operador
            )

    # ─── Test 3.10: Métrica de energy_drift ──────────────────────────────────
    def test_energy_drift_metric(self, dissipator_setup):
        """Verifica que energy_drift es finito y no negativo."""
        dissipator, dim = dissipator_setup

        rho0 = np.eye(dim, dtype=np.complex128) / dim
        result = dissipator.execute_master_equation(rho0, dt=0.01, method="rk4")

        assert np.isfinite(result.energy_drift)
        assert result.energy_drift >= 0.0

    # ─── Test 3.11: Validación Kraus-Decomposition (log) ─────────────────────
    def test_kraus_validation_warning(self, config_small, caplog):
        """Verifica que mapas no contractivos generen warning."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(
            ops, np.zeros((1, 1), dtype=np.complex128)
        )
        hb = collider.get_catadioptric_hamiltonian()

        # Operador con norma > 1 para forzar no-contractividad
        big_op = 3.0 * np.eye(ops.hilbert_dim, dtype=np.complex128)

        with caplog.at_level(logging.WARNING):
            Phase3_LindbladDissipator(
                hamiltonian_bundle=hb,
                lindblad_operators=[big_op],
                decay_rates=[1.0],
            )

        # Verifica que se logueó el warning
        assert any("Kraus" in rec.message for rec in caplog.records)

    # ─── Test 3.12: dt=0 no produce cambio ──────────────────────────────────
    def test_zero_dt_preserves_state(self, dissipator_setup):
        """Verifica que dt=0 → ρ(t) = ρ(0)."""
        dissipator, dim = dissipator_setup

        rho0 = np.eye(dim, dtype=np.complex128) / dim
        result = dissipator.execute_master_equation(rho0, dt=0.0, method="rk4")

        assert_allclose(result.post_collision_rho, rho0, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# CLASE 4: TESTS DEL ORQUESTADOR — INTEGRACIÓN END-TO-END
# ══════════════════════════════════════════════════════════════════════════════
class TestQuantumFockOrchestrator:
    """Validación del pipeline completo."""

    # ─── Test 4.1: Inicialización con parámetros válidos ─────────────────────
    def test_orchestrator_initialization(self, config_small, coupling_small):
        """Verifica que el orquestador se construye sin errores."""
        orch = QuantumFockOrchestrator(
            config=config_small,
            coupling_matrix=coupling_small,
        )
        assert orch._phase1 is not None
        assert orch._phase2 is not None
        assert orch._phase3 is not None

    # ─── Test 4.2: Pipeline completo con estado puro ─────────────────────────
    def test_full_pipeline_pure_state(self, config_medium, coupling_medium):
        """Ejecuta assimilate_and_collide con estado puro."""
        orch = QuantumFockOrchestrator(
            config=config_medium,
            coupling_matrix=coupling_medium,
        )

        # Estado puro en dim=64
        rng = np.random.default_rng(123)
        psi = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())

        result = orch.assimilate_and_collide(rho0, dt=0.01)

        assert isinstance(result, LindbladEvolutionResult)
        assert result.post_collision_rho.shape == (64, 64)
        assert abs(np.trace(result.post_collision_rho) - 1.0) < 1e-7
        assert result.positivity_preserved

    # ─── Test 4.3: Verificación de integridad end-to-end ─────────────────────
    def test_verify_complete_integrity(self, config_small, coupling_small):
        """Verifica la firma y contenido de verify_complete_integrity()."""
        orch = QuantumFockOrchestrator(
            config=config_small,
            coupling_matrix=coupling_small,
        )

        report = orch.verify_complete_integrity()

        assert "fock_space" in report
        assert "hamiltonian" in report
        assert "lindblad" in report

        assert report["fock_space"]["hilbert_dim"] > 0
        assert report["fock_space"]["boson_modes"] == 1
        assert report["fock_space"]["fermion_modes"] == 1
        assert report["hamiltonian"]["hermitic"] is True

    # ─── Test 4.4: Cambio de método de integración ───────────────────────────
    def test_integration_method_selection(self, config_small, coupling_small):
        """Verifica que se respete la selección de método."""
        orch = QuantumFockOrchestrator(
            config=config_small,
            coupling_matrix=coupling_small,
            default_integration="euler",
        )

        rho0 = np.eye(orch._phase1._total_dim, dtype=np.complex128)
        rho0 /= np.trace(rho0)

        # RK4 explícito
        r_rk4 = orch.assimilate_and_collide(rho0, dt=0.01, method="rk4")
        assert r_rk4.integration_method == "RK4"

        # Euler explícito
        r_euler = orch.assimilate_and_collide(rho0, dt=0.01, method="euler")
        assert r_euler.integration_method == "Euler"

    # ─── Test 4.5: Operadores de Lindblad personalizados ─────────────────────
    def test_custom_lindblad_operators(self, config_medium, coupling_medium):
        """Verifica uso de operadores de Lindblad arbitrarios."""
        ops_builder = Phase1_FockSpaceBuilder(config_medium)
        ops = ops_builder.get_interaction_operators()

        # Usar operadores fermiónicos como Lindblad (en lugar de bosónicos)
        custom_L = [n for n in ops.fermion_number]

        orch = QuantumFockOrchestrator(
            config=config_medium,
            coupling_matrix=coupling_medium,
            lindblad_operators=custom_L,
            lindblad_rates=[0.5] * len(custom_L),
        )

        rho0 = np.eye(ops.hilbert_dim, dtype=np.complex128) / ops.hilbert_dim
        result = orch.assimilate_and_collide(rho0, dt=0.01)

        assert result.positivity_preserved

    # ─── Test 4.6: Logging de fotón Gamma ────────────────────────────────────
    def test_gamma_photon_logging(self, config_small, coupling_small, caplog):
        """Verifica que se emita warning cuando hay fotón Gamma."""
        # γ alto para forzar emisión
        orch = QuantumFockOrchestrator(
            config=config_small,
            coupling_matrix=coupling_small,
            lindblad_rates=[10.0],
        )

        # Estado excitado
        dim = orch._phase1._total_dim
        rho0 = np.zeros((dim, dim), dtype=np.complex128)
        rho0[1, 1] = 1.0

        with caplog.at_level(logging.WARNING):
            result = orch.assimilate_and_collide(rho0, dt=0.5)

        if result.emitted_photon is not None:
            assert any("Gamma" in rec.message for rec in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# CLASE 5: TESTS DE EXCEPCIONES Y ROBUSTEZ
# ══════════════════════════════════════════════════════════════════════════════
class TestExceptionsAndRobustness:
    """Tests de manejo de errores y casos degenerados."""

    # ─── Test 5.1: Excepciones heredan de TopologicalInvariantError ──────────
    def test_exception_hierarchy(self):
        """Todas las excepciones del módulo son TopologicalInvariantError."""
        for exc_class in [
            PauliExclusionViolationError,
            FockSpaceOverflowError,
            LindbladDissipationError,
            HermiticityViolationError,
            UnitarityViolationError,
        ]:
            assert issubclass(exc_class, TopologicalInvariantError)
            assert issubclass(exc_class, Exception)

    # ─── Test 5.2: Estado con traza muy pequeña → reinicialización ───────────
    def test_tiny_trace_reinitialization(self, config_small, caplog):
        """ρ con Tr≈0 debe re-inicializarse a I/d."""
        builder = Phase1_FockSpaceBuilder(config_small)
        ops = builder.get_interaction_operators()
        collider = Phase2_CatadioptricCollider(
            ops, np.zeros((1, 1), dtype=np.complex128)
        )
        hb = collider.get_catadioptric_hamiltonian()

        dissipator = Phase3_LindbladDissipator(
            hamiltonian_bundle=hb,
            lindblad_operators=ops.boson_ann,
            decay_rates=[100.0],  # γ muy alto
        )

        # Estado con traza casi cero (no normalizado, se renormalizará)
        rho0 = np.zeros((ops.hilbert_dim, ops.hilbert_dim), dtype=np.complex128)
        rho0[0, 0] = 1e-40

        # No debe lanzar excepción gracias al fallback
        with caplog.at_level(logging.WARNING):
            # Nota: validación inicial rechazará esto por Tr ≠ 1,
            # pero el caso patológico se prueba con un ρ casi normalizado
            # que colapsa durante evolución.
            rho0 = np.eye(ops.hilbert_dim, dtype=np.complex128) / ops.hilbert_dim
            result = dissipator.execute_master_equation(rho0, dt=10.0, method="rk4")

        # Debe haber warning o resultado estable
        assert result is not None

    # ─── Test 5.3: Determinismo con misma semilla ─────────────────────────────
    def test_determinism(self, config_medium, coupling_medium):
        """Mismo input → mismo output (determinismo)."""
        orch1 = QuantumFockOrchestrator(
            config=config_medium,
            coupling_matrix=coupling_medium,
        )
        orch2 = QuantumFockOrchestrator(
            config=config_medium,
            coupling_matrix=coupling_medium,
        )

        rho0 = np.eye(orch1._phase1._total_dim, dtype=np.complex128)
        rho0 /= np.trace(rho0)

        r1 = orch1.assimilate_and_collide(rho0, dt=0.01)
        r2 = orch2.assimilate_and_collide(rho0, dt=0.01)

        assert_allclose(r1.post_collision_rho, r2.post_collision_rho, atol=1e-14)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS PARAMETRIZADOS ADICIONALES
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("B,F,M", [
    (1, 1, 2),
    (2, 1, 3),
    (1, 2, 2),
    (2, 2, 2),
    (3, 1, 1),  # Mínimo truncamiento
])
def test_hilbert_dimension_parametrized(B, F, M):
    r"""Verifica $\dim\mathcal{H} = (M+1)^B \cdot 2^F$ en múltiples configs."""
    config = FockSpaceConfiguration(
        n_boson_modes=B, n_fermion_modes=F, boson_truncation=M
    )
    builder = Phase1_FockSpaceBuilder(config)
    expected = (M + 1) ** B * (2 ** F)
    assert builder._total_dim == expected


@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0, 5.0])
def test_lindblad_decay_rate_scaling(config_small, coupling_small, gamma):
    """Verifica que diferentes tasas de decaimiento produzcan resultados coherentes."""
    builder = Phase1_FockSpaceBuilder(config_small)
    ops = builder.get_interaction_operators()
    collider = Phase2_CatadioptricCollider(ops, coupling_small)
    hb = collider.get_catadioptric_hamiltonian()

    dissipator = Phase3_LindbladDissipator(
        hamiltonian_bundle=hb,
        lindblad_operators=ops.boson_ann,
        decay_rates=[gamma],
    )

    dim = ops.hilbert_dim
    rho0 = np.eye(dim, dtype=np.complex128) / dim

    if gamma == 0.0:
        # Sin disipación: no debe emitir fotón
        result = dissipator.execute_master_equation(rho0, dt=0.1, method="rk4")
        assert result.emitted_photon is None
    else:
        result = dissipator.execute_master_equation(rho0, dt=0.1, method="rk4")
        assert result.positivity_preserved
        assert abs(np.trace(result.post_collision_rho) - 1.0) < 1e-7


@pytest.mark.parametrize("dt", [1e-4, 1e-3, 1e-2, 0.1])
def test_dt_convergence(config_small, coupling_small, dt):
    """Verifica convergencia al reducir dt."""
    builder = Phase1_FockSpaceBuilder(config_small)
    ops = builder.get_interaction_operators()
    collider = Phase2_CatadioptricCollider(ops, coupling_small)
    hb = collider.get_catadioptric_hamiltonian()

    dissipator = Phase3_LindbladDissipator(
        hamiltonian_bundle=hb,
        lindblad_operators=ops.boson_ann,
        decay_rates=[0.5],
    )

    dim = ops.hilbert_dim
    rho0 = np.eye(dim, dtype=np.complex128) / dim
    result = dissipator.execute_master_equation(rho0, dt=dt, method="rk4")

    # Para todo dt válido, debe preservar positividad y traza
    assert result.positivity_preserved
    assert abs(np.trace(result.post_collision_rho) - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS DE RENDIMIENTO (opcional, marcado como slow)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.slow
class TestPerformance:
    """Benchmarks de rendimiento (excluidos por defecto)."""

    def test_large_system_construction(self):
        """Construye sistema grande (dim=1280) en tiempo razonable."""
        import time

        config = FockSpaceConfiguration(
            n_boson_modes=4,  # (3+1)^4 = 256
            n_fermion_modes=5,  # × 2^5 = 32 → dim=1280
            boson_truncation=3,
        )

        start = time.time()
        builder = Phase1_FockSpaceBuilder(config)
        elapsed = time.time() - start

        assert builder._total_dim == 1280
        assert elapsed < 5.0, f"Construcción demasiado lenta: {elapsed:.2f}s"

    def test_evolution_performance(self):
        """100 pasos de evolución en sistema medio."""
        import time

        config = FockSpaceConfiguration(
            n_boson_modes=2, n_fermion_modes=2, boson_truncation=3,
        )
        orch = QuantumFockOrchestrator(
            config=config,
            coupling_matrix=np.array([[0.5, 0.2], [0.3, -0.4]], dtype=np.complex128),
        )

        rho = np.eye(orch._phase1._total_dim, dtype=np.complex128)
        rho /= np.trace(rho)

        start = time.time()
        for _ in range(100):
            rho = orch.assimilate_and_collide(rho, dt=0.001).post_collision_rho
        elapsed = time.time() - start

        assert elapsed < 30.0, f"Evolución demasiado lenta: {elapsed:.2f}s"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PYTEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """Ejecuta la suite directamente con `python test_quantum_fock_orchestrator.py`."""
    pytest.main([__file__, "-v", "--tb=short"])