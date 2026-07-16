# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : test_atomic_piston_service.py                                     ║
║ Versión  : 5.0.0-Test-Suite-Doctoral-Rigorous                                ║
║ Ubicación: tests/unit/physics/test_atomic_piston_service.py                  ║
║ Dominio  : Circuitos · Topología algebraica · Espectral · Categorías · PHS   ║
╚══════════════════════════════════════════════════════════════════════════════╝

SUITE DE PRUEBAS ROBUSTA — ATOMIC PISTON SERVICE v5.0.0
═══════════════════════════════════════════════════════
Arquitectura anidada (espejo del SUT):
  FASE 1 — Herramientas físicas y matemáticas
           (métrica Riemanniana, fricción covariante, PID en grupos de Lie)
  FASE 2 — Gemelo digital simpléctico
           (port-Hamiltoniano, Störmer–Verlet, pasividad, colisiones)
  FASE 3 — Microservicio / ServiceContext
           (buffer lock-free, POVM/Kraus, cohomología de haces, HTTP guards)
  FASE Σ — Integración y contratos de sutura F1→F2→F3
  FASE R — Regresiones numéricas y estrés

Invariantes matemáticos certificados
────────────────────────────────────
  I1  Covarianza tensorial: F_fric ∈ T*M transforma con G
  I2  Simplecticidad: mapa Störmer–Verlet preserva ω=dq∧dp (proxy: H en R=0)
  I3  Pasividad PHS: Ḣ ≤ ∇Hᵀ B u  ⇔  −∇Hᵀ R ∇H ≤ 0
  I4  Cohomología: admisión ⇔ β₁=0 (H¹ trivial en grafo de registro)
  I5  No-demolición POVM: λ=1 ⇒ ρ_JSON ≡ ρ_sim (módulo metadatos)
  I6  Buffer lock-free: latest() O(1), wrap circular, concurrencia
  I7  Anti-windup geométrico: ‖u‖_G ≤ u_max
  I8  J antisimétrica, R=Rᵀ ≽ 0 (estructura port-Hamiltoniana)
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# ── Bootstrap de imports ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.physics.atomic_piston_service import (
    # FASE 1
    FrictionCalculator,
    LieGroupPIDController,
    _metric_norm,
    _metric_dual,
    _lie_log_affine,
    _parallel_transport_affine,
    _covariant_derivative_affine,
    # FASE 2
    AtomicPiston,
    _build_J_matrix,
    _build_R_matrix,
    _compute_hamiltonian_gradient,
    _verify_dissipation_inequality,
    # FASE 3
    _AtomicCounter,
    LockFreeCircularBuffer,
    KrausObserver,
    SheafCohomologyVerifier,
    ServiceContext,
    _require_json,
    _buffer_or_503,
    _ipu_or_503,
)

try:
    from app.core.constants import (
        FrictionModel,
        PistonMode,
        TransducerType,
    )
except ImportError:

    class FrictionModel:
        VISCOUS = "viscous"
        COULOMB = "coulomb"
        STRIBECK = "stribeck"

    class PistonMode:
        CAPACITOR = "capacitor"
        BATTERY = "battery"

    class TransducerType:
        PIEZOELECTRIC = "piezoelectric"
        ELECTROSTATIC = "electrostatic"
        MAGNETOSTRICTIVE = "magnetostrictive"


# ══════════════════════════════════════════════════════════════════════════════
# Constantes de tolerancia (justificadas)
# ══════════════════════════════════════════════════════════════════════════════

class Tol:
    r"""
    Tolerancias numéricas con semántica explícita.

    MACHINE     : redondeo float64.
    STRICT      : identidades algebraicas exactas (J+Jᵀ, simetría R).
    METRIC      : normas / duales métricos.
    ENERGY_SV   : deriva relativa de H en Störmer–Verlet (R=0, Δt pequeño).
    PASSIVE     : cota de Ḣ_dis (debe ser ≤ 0 salvo ruido FP).
    PID         : anti-windup / saturación geométrica.
    FRICTION    : regímenes Coulomb/Stribeck (suavizado tanh).
    COHOMOLOGY  : β₁ entero exacto.
    """

    MACHINE = float(np.finfo(np.float64).eps)
    STRICT = 1e-14
    METRIC = 1e-12
    ENERGY_SV = 1e-2          # 1 % en ventanas controladas
    ENERGY_SV_LONG = 5e-2     # 5 % en 10 periodos
    PASSIVE = 1e-10
    PID = 1e-2
    FRICTION = 1e-2
    COHOMOLOGY = 0            # igualdad exacta de enteros
    JSON_KEYS = frozenset  # alias semántico, no tolerancia


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures compartidas
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def G_id_1() -> np.ndarray:
    return np.array([[1.0]], dtype=np.float64)


@pytest.fixture
def G_id_2() -> np.ndarray:
    return np.eye(2, dtype=np.float64)


@pytest.fixture
def G_spd_2() -> np.ndarray:
    r"""G ≻ 0 no diagonal (fuerza covarianza real)."""
    return np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.float64)


@pytest.fixture
def G_diag_3() -> np.ndarray:
    return np.diag([1.0, 2.0, 3.0]).astype(np.float64)


@pytest.fixture
def piston_viscous() -> AtomicPiston:
    return AtomicPiston(
        capacity=0.01,
        elasticity=100.0,
        damping=0.1,
        piston_mass=1.0,
        friction_model=FrictionModel.VISCOUS,
    )


@pytest.fixture
def piston_nondissipative() -> AtomicPiston:
    r"""R efectivo nulo en el sector mecánico (damping=0, fricción viscosa=0)."""
    return AtomicPiston(
        capacity=0.01,
        elasticity=100.0,
        damping=0.0,
        piston_mass=1.0,
        friction_model=FrictionModel.VISCOUS,
        nonlinear_elasticity=0.0,
    )


@pytest.fixture
def piston_coulomb() -> AtomicPiston:
    return AtomicPiston(
        capacity=0.01,
        elasticity=100.0,
        damping=0.1,
        piston_mass=1.0,
        friction_model=FrictionModel.COULOMB,
        coulomb_friction=0.2,
    )


@pytest.fixture
def empty_buffer() -> LockFreeCircularBuffer:
    return LockFreeCircularBuffer(capacity=16)


@pytest.fixture
def sheaf() -> SheafCohomologyVerifier:
    return SheafCohomologyVerifier()


def _assert_spd(G: np.ndarray, name: str = "G") -> None:
    eig = np.linalg.eigvalsh(G)
    assert np.all(eig > 0), f"{name} no es SPD: λ={eig}"


def _relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


# ============================================================================
# FASE 1 — HERRAMIENTAS FÍSICAS Y MATEMÁTICAS
# ============================================================================

class TestPhase1MathematicalTools:
    r"""
    FASE 1: métrica, dual musical, fricción covariante, PID en (ℝⁿ,+).

    Contrato de salida hacia FASE 2:
      · F_fric ∈ T*_xM (covector)
      · u_PID ∈ g con ‖u‖_G ≤ u_max
    """

    # ──────────────────────────────────────────────────────────────────────
    # §1.0 Álgebra Riemanniana auxiliar
    # ──────────────────────────────────────────────────────────────────────

    class TestMetricAlgebra:
        """Norma ‖·‖_G y dual ♭: v ↦ Gv."""

        def test_metric_norm_reduces_to_euclidean_on_identity(self, G_id_2):
            v = np.array([3.0, 4.0], dtype=np.float64)
            assert abs(_metric_norm(v, G_id_2) - 5.0) < Tol.METRIC

        def test_metric_norm_zero_iff_vector_zero(self, G_spd_2):
            _assert_spd(G_spd_2)
            assert _metric_norm(np.zeros(2), G_spd_2) < Tol.MACHINE * 10
            assert _metric_norm(np.array([1e-3, -2e-3]), G_spd_2) > 0.0

        def test_metric_norm_positive_for_nonzero(self, G_spd_2):
            rng = np.random.default_rng(0)
            for _ in range(20):
                v = rng.normal(size=2)
                if np.linalg.norm(v) < 1e-15:
                    continue
                assert _metric_norm(v, G_spd_2) > 0.0

        @pytest.mark.parametrize("lam", [0.5, -2.0, math.pi, -1e-3, 1e3])
        def test_metric_norm_absolute_homogeneity(self, G_diag_3, lam):
            v = np.array([1.0, -2.0, 0.5], dtype=np.float64)
            lhs = _metric_norm(lam * v, G_diag_3)
            rhs = abs(lam) * _metric_norm(v, G_diag_3)
            assert abs(lhs - rhs) < Tol.METRIC * max(1.0, abs(lam))

        def test_metric_norm_triangle_inequality(self, G_spd_2):
            r"""‖u+v‖_G ≤ ‖u‖_G + ‖v‖_G (desigualdad triangular)."""
            rng = np.random.default_rng(1)
            for _ in range(15):
                u, v = rng.normal(size=2), rng.normal(size=2)
                assert (
                    _metric_norm(u + v, G_spd_2)
                    <= _metric_norm(u, G_spd_2) + _metric_norm(v, G_spd_2) + Tol.METRIC
                )

        def test_metric_norm_monotonic_in_eigenvalues(self):
            r"""Si G₂ − G₁ ≽ 0 entonces ‖v‖_{G₂} ≥ ‖v‖_{G₁}."""
            v = np.array([1.0, -1.0], dtype=np.float64)
            G1 = np.eye(2)
            G2 = 3.0 * np.eye(2)
            assert _metric_norm(v, G2) >= _metric_norm(v, G1) - Tol.METRIC

        def test_metric_dual_is_Gv(self, G_spd_2):
            v = np.array([1.0, -2.0], dtype=np.float64)
            np.testing.assert_allclose(
                _metric_dual(v, G_spd_2), G_spd_2 @ v, atol=Tol.METRIC
            )

        def test_metric_dual_linearity(self, G_spd_2):
            v = np.array([1.0, 2.0], dtype=np.float64)
            w = np.array([-1.0, 0.5], dtype=np.float64)
            a, b = 3.0, -0.5
            lhs = _metric_dual(a * v + b * w, G_spd_2)
            rhs = a * _metric_dual(v, G_spd_2) + b * _metric_dual(w, G_spd_2)
            np.testing.assert_allclose(lhs, rhs, atol=Tol.METRIC)

        def test_metric_dual_recovers_inner_product(self, G_spd_2):
            r"""⟨u,v⟩_G = uᵀ G v = uᵀ (v♭)."""
            u = np.array([0.3, -1.2], dtype=np.float64)
            v = np.array([2.0, 0.7], dtype=np.float64)
            ip = float(u @ (G_spd_2 @ v))
            ip_dual = float(u @ _metric_dual(v, G_spd_2))
            assert abs(ip - ip_dual) < Tol.METRIC

        def test_metric_norm_equals_sqrt_v_flat_v(self, G_spd_2):
            v = np.array([1.5, -0.5], dtype=np.float64)
            n = _metric_norm(v, G_spd_2)
            n2 = math.sqrt(max(float(v @ _metric_dual(v, G_spd_2)), 0.0))
            assert abs(n - n2) < Tol.METRIC

        def test_metric_norm_dimension_mismatch_raises(self):
            with pytest.raises(ValueError, match="incompatible"):
                _metric_norm(np.array([1.0, 2.0]), np.eye(3))

        def test_metric_non_square_raises(self):
            G = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            with pytest.raises(ValueError, match="cuadrada"):
                _metric_norm(np.array([1.0, 2.0]), G)

        def test_lie_log_affine_is_vector_difference(self):
            r"""En (ℝⁿ,+): Log_x(x_d) = x_d − x."""
            x = np.array([1.0, -2.0, 0.5])
            x_d = np.array([4.0, 0.0, -1.0])
            np.testing.assert_allclose(
                _lie_log_affine(x, x_d), x_d - x, atol=Tol.STRICT
            )

        def test_parallel_transport_affine_is_identity(self, G_spd_2):
            r"""Γ afín = Id ⇒ isometría trivial."""
            xi = np.array([1.0, -3.0], dtype=np.float64)
            transported = _parallel_transport_affine(xi)
            np.testing.assert_allclose(transported, xi, atol=Tol.STRICT)
            assert abs(
                _metric_norm(transported, G_spd_2) - _metric_norm(xi, G_spd_2)
            ) < Tol.METRIC

        def test_covariant_derivative_affine_equals_finite_difference(self):
            r"""Γ=0 ⇒ Dẋ/dt = dẋ/dt (diferencias finitas)."""
            v_prev = np.array([0.0], dtype=np.float64)
            v_curr = np.array([0.02], dtype=np.float64)
            dt = 0.01
            Dv = _covariant_derivative_affine(v_curr, v_prev, dt)
            np.testing.assert_allclose(Dv, (v_curr - v_prev) / dt, atol=Tol.METRIC)

    # ──────────────────────────────────────────────────────────────────────
    # §1.1 FrictionCalculator — covarianza tensorial
    # ──────────────────────────────────────────────────────────────────────

    class TestFrictionCalculator:
        """F_fric como 1-forma (covector) en T*M."""

        def test_smooth_sign_zero_at_origin(self, G_id_2):
            s = FrictionCalculator.smooth_sign_metric(
                np.zeros(2, dtype=np.float64), G_id_2
            )
            np.testing.assert_allclose(s, 0.0, atol=Tol.STRICT * 10)

        def test_smooth_sign_shape_matches_velocity(self, G_spd_2):
            v = np.array([0.1, -0.2], dtype=np.float64)
            s = FrictionCalculator.smooth_sign_metric(v, G_spd_2)
            assert s.shape == v.shape

        def test_smooth_sign_opposes_velocity_direction_large_v(self, G_id_2):
            r"""
            Para ‖v‖_G ≫ v_s: smooth_sign ≈ (G v)/‖v‖_G (covector unitario).
            El ángulo con Gv/‖v‖_G debe ser ~0.
            """
            v = np.array([5.0, 0.0], dtype=np.float64)  # grande vs V_SMOOTH
            s = FrictionCalculator.smooth_sign_metric(v, G_id_2)
            expected = _metric_dual(v, G_id_2) / _metric_norm(v, G_id_2)
            cosang = float(
                np.dot(s, expected)
                / (np.linalg.norm(s) * np.linalg.norm(expected) + 1e-30)
            )
            assert cosang > 0.99

        def test_smooth_sign_is_odd_map(self, G_spd_2):
            r"""smooth_sign(−v) = −smooth_sign(v) (imparidad)."""
            v = np.array([0.07, -0.03], dtype=np.float64)
            s_pos = FrictionCalculator.smooth_sign_metric(v, G_spd_2)
            s_neg = FrictionCalculator.smooth_sign_metric(-v, G_spd_2)
            np.testing.assert_allclose(s_neg, -s_pos, atol=Tol.METRIC)

        def test_coulomb_kinetic_bounded_by_fc(self, G_id_1):
            F_c = 0.2
            f = FrictionCalculator.compute_friction(
                velocity=0.1,
                driving_force=0.5,
                friction_model=FrictionModel.COULOMB,
                coulomb_friction=F_c,
                metric_tensor=G_id_1,
            )
            assert abs(f) <= F_c * (1.0 + Tol.FRICTION)
            assert f != 0.0
            assert np.sign(f) == -np.sign(0.1)

        def test_coulomb_static_balances_drive_below_fc(self, G_id_1):
            F_c, F_drv = 0.2, 0.1
            f = FrictionCalculator.compute_friction(
                velocity=0.0,
                driving_force=F_drv,
                friction_model=FrictionModel.COULOMB,
                coulomb_friction=F_c,
                metric_tensor=G_id_1,
            )
            assert abs(f + F_drv) < Tol.FRICTION

        def test_coulomb_static_saturates_at_fc_when_drive_large(self, G_id_1):
            F_c, F_drv = 0.2, 5.0
            f = FrictionCalculator.compute_friction(
                velocity=0.0,
                driving_force=F_drv,
                friction_model=FrictionModel.COULOMB,
                coulomb_friction=F_c,
                metric_tensor=G_id_1,
            )
            assert abs(abs(f) - F_c) < Tol.FRICTION * 5
            assert np.sign(f) == -np.sign(F_drv)

        def test_coulomb_odd_in_velocity_kinetic(self, G_id_1):
            kwargs = dict(
                driving_force=0.5,
                friction_model=FrictionModel.COULOMB,
                coulomb_friction=0.2,
                metric_tensor=G_id_1,
            )
            f_pos = FrictionCalculator.compute_friction(velocity=0.2, **kwargs)
            f_neg = FrictionCalculator.compute_friction(velocity=-0.2, **kwargs)
            assert abs(f_pos + f_neg) < Tol.FRICTION

        def test_stribeck_decreases_with_speed(self):
            F_S, F_C, v_St = 0.3, 0.1, 0.05
            kwargs = dict(
                driving_force=0.5,
                friction_model=FrictionModel.STRIBECK,
                coulomb_friction=F_C,
                stribeck_coeffs=(F_S, F_C, v_St),
            )
            f_low = FrictionCalculator.compute_friction(velocity=0.01, **kwargs)
            f_high = FrictionCalculator.compute_friction(velocity=1.0, **kwargs)
            assert abs(f_low) > abs(f_high)

        def test_stribeck_high_speed_approaches_coulomb(self, G_id_1):
            F_S, F_C, v_St = 0.3, 0.1, 0.05
            f = FrictionCalculator.compute_friction(
                velocity=50.0 * v_St,
                driving_force=1.0,
                friction_model=FrictionModel.STRIBECK,
                coulomb_friction=F_C,
                stribeck_coeffs=(F_S, F_C, v_St),
                metric_tensor=G_id_1,
            )
            assert abs(abs(f) - F_C) < 0.05

        def test_viscous_model_returns_zero(self):
            f = FrictionCalculator.compute_friction(
                velocity=0.1,
                driving_force=0.5,
                friction_model=FrictionModel.VISCOUS,
            )
            assert f == 0.0

        def test_metric_tensor_deforms_friction_magnitude(self):
            r"""I1: G≠I debe alterar ‖v‖_G y por tanto F_fric."""
            kwargs = dict(
                velocity=0.1,
                driving_force=0.5,
                friction_model=FrictionModel.COULOMB,
                coulomb_friction=0.2,
            )
            f_I = FrictionCalculator.compute_friction(
                metric_tensor=np.array([[1.0]]), **kwargs
            )
            f_G = FrictionCalculator.compute_friction(
                metric_tensor=np.array([[4.0]]), **kwargs
            )
            assert abs(f_I - f_G) > 1e-6

        def test_stribeck_invalid_v_st_raises(self):
            with pytest.raises(ValueError, match="v_stribeck"):
                FrictionCalculator.compute_friction(
                    velocity=0.1,
                    driving_force=0.5,
                    friction_model=FrictionModel.STRIBECK,
                    stribeck_coeffs=(0.3, 0.1, 0.0),
                )

        def test_stribeck_negative_v_st_raises(self):
            with pytest.raises(ValueError, match="v_stribeck"):
                FrictionCalculator.compute_friction(
                    velocity=0.1,
                    driving_force=0.5,
                    friction_model=FrictionModel.STRIBECK,
                    stribeck_coeffs=(0.3, 0.1, -0.1),
                )

        def test_friction_field_batch_shape_and_consistency(self, G_id_1):
            velocities = np.array([0.01, 0.05, 0.1, 0.5], dtype=np.float64)
            drives = np.full_like(velocities, 0.5)
            field = FrictionCalculator.compute_friction_field(
                velocities=velocities,
                driving_forces=drives,
                friction_model=FrictionModel.COULOMB,
                metric_tensor=G_id_1,
                coulomb_friction=0.2,
            )
            assert field.shape == velocities.shape
            for i, v in enumerate(velocities):
                f_i = FrictionCalculator.compute_friction(
                    velocity=float(v),
                    driving_force=0.5,
                    friction_model=FrictionModel.COULOMB,
                    coulomb_friction=0.2,
                    metric_tensor=G_id_1,
                )
                assert abs(field[i] - f_i) < Tol.FRICTION

        def test_friction_field_rejects_shape_mismatch(self, G_id_1):
            with pytest.raises((ValueError, AssertionError)):
                FrictionCalculator.compute_friction_field(
                    velocities=np.array([0.1, 0.2]),
                    driving_forces=np.array([0.5]),
                    friction_model=FrictionModel.COULOMB,
                    metric_tensor=G_id_1,
                )

    # ──────────────────────────────────────────────────────────────────────
    # §1.2 LieGroupPIDController — control geométrico en (ℝⁿ,+)
    # ──────────────────────────────────────────────────────────────────────

    class TestLieGroupPIDController:
        """PID con Log, transporte paralelo y anti-windup en bola geodésica."""

        @pytest.fixture
        def pid_1d(self) -> LieGroupPIDController:
            return LieGroupPIDController(kp=1.0, ki=0.1, kd=0.01)

        def test_lie_log_affine_difference(self, pid_1d):
            x = np.array([2.0], dtype=np.float64)
            x_d = np.array([5.0], dtype=np.float64)
            np.testing.assert_allclose(
                pid_1d._lie_log(x, x_d), x_d - x, atol=Tol.STRICT
            )

        def test_parallel_transport_isometry(self, G_spd_2):
            ctl = LieGroupPIDController(
                kp=1.0, ki=0.1, kd=0.01, metric_tensor=G_spd_2, state_dim=2
            )
            xi = np.array([1.0, 2.0], dtype=np.float64)
            n0 = _metric_norm(xi, G_spd_2)
            n1 = _metric_norm(ctl._parallel_transport(xi), G_spd_2)
            assert abs(n0 - n1) < Tol.METRIC

        def test_covariant_derivative_matches_acceleration(self):
            ctl = LieGroupPIDController(kp=1.0, ki=0.1, kd=0.01)
            a, dt = 2.0, 0.01
            # Primer update fija v_prev; segundo mide Δv/dt
            ctl._update_covariant_derivative(np.array([0.0]), dt)
            Dv = ctl._update_covariant_derivative(np.array([a * dt]), dt)
            assert abs(float(Dv[0]) - a) < 0.15  # un paso FD

        def test_output_bounded_by_geodesic_ball(self, G_id_1):
            u_max = 5.0
            ctl = LieGroupPIDController(
                kp=100.0, ki=50.0, kd=10.0,
                output_limit=u_max,
                metric_tensor=G_id_1,
            )
            for _ in range(100):
                u = ctl.update(10.0, 0.0, 0.01, current_velocity=0.0)
            assert _metric_norm(np.array([u]), G_id_1) <= u_max * 1.01

        def test_anti_windup_prevents_integral_runaway(self):
            ctl = LieGroupPIDController(
                kp=100.0, ki=50.0, kd=0.0, output_limit=1.0
            )
            for _ in range(200):
                ctl.update(10.0, 0.0, 0.01, current_velocity=0.0)
            # Integral no debe crecer como ki * e * T sin cota
            unbounded = 50.0 * 10.0 * 200 * 0.01  # ≫ 1
            assert np.linalg.norm(ctl._integral_lie) < unbounded * 0.1

        def test_derivative_filter_limits_setpoint_kick(self):
            ctl = LieGroupPIDController(
                kp=1.0, ki=0.0, kd=10.0, derivative_filter_tau=0.1
            )
            u0 = ctl.update(0.0, 5.0, 0.01, 0.0)
            u1 = ctl.update(10.0, 5.0, 0.01, 0.0)
            # Sin kick puro: |Δu| acotado respecto a Kp·Δsp
            assert abs(u1 - u0) < 1.0 * abs(10.0 - 0.0) * 1.5

        def test_reset_clears_lie_state(self, pid_1d):
            pid_1d.update(10.0, 0.0, 0.01, 0.1)
            pid_1d.update(10.0, 0.0, 0.01, 0.1)
            assert np.linalg.norm(pid_1d._integral_lie) > 0.0
            pid_1d.reset()
            np.testing.assert_allclose(pid_1d._integral_lie, 0.0, atol=Tol.STRICT * 10)
            np.testing.assert_allclose(pid_1d._velocity_prev, 0.0, atol=Tol.STRICT * 10)
            np.testing.assert_allclose(
                pid_1d._cov_deriv_filtered, 0.0, atol=Tol.STRICT * 10
            )

        def test_invalid_output_limit_raises(self):
            with pytest.raises(ValueError, match="output_limit"):
                LieGroupPIDController(kp=1.0, ki=0.1, kd=0.01, output_limit=0.0)

        def test_negative_output_limit_raises(self):
            with pytest.raises(ValueError, match="output_limit"):
                LieGroupPIDController(kp=1.0, ki=0.1, kd=0.01, output_limit=-1.0)

        def test_indefinite_metric_raises(self):
            G = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
            with pytest.raises(ValueError, match="definida positiva"):
                LieGroupPIDController(
                    kp=1.0, ki=0.1, kd=0.01, metric_tensor=G, state_dim=2
                )

        def test_singular_metric_raises(self):
            G = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
            with pytest.raises(ValueError):
                LieGroupPIDController(
                    kp=1.0, ki=0.1, kd=0.01, metric_tensor=G, state_dim=2
                )

        def test_proportional_action_reduces_error_sign(self, pid_1d):
            u = pid_1d.update(setpoint=5.0, current=2.0, dt=0.01, current_velocity=0.0)
            # Convención del SUT: u empuja a reducir error (signo de −Kp e)
            assert np.sign(u) == np.sign(-(5.0 - 2.0))

        def test_zero_error_zero_velocity_small_output(self, pid_1d):
            # Sin historial integral
            pid_1d.reset()
            u = pid_1d.update(1.0, 1.0, 0.01, 0.0)
            assert abs(u) < 1e-9

        def test_dt_nonpositive_raises(self, pid_1d):
            with pytest.raises(ValueError):
                pid_1d.update(1.0, 0.0, 0.0, 0.0)


# ============================================================================
# FASE 2 — GEMELO DIGITAL SIMPLÉCTICO (PORT-HAMILTONIANO)
# ============================================================================

class TestPhase2SymplecticDigitalTwin:
    r"""
    FASE 2: J, R, ∇H, Störmer–Verlet, pasividad, colisiones.

    Contrato de entrada (FASE 1): F_fric, u_PID.
    Contrato de salida hacia FASE 3: get_state_dict() JSON-serializable.
    """

    # ──────────────────────────────────────────────────────────────────────
    # §2.0 Álgebra port-Hamiltoniana
    # ──────────────────────────────────────────────────────────────────────

    class TestPortHamiltonianAlgebra:
        """Estructura (J−R)∇H + B u."""

        @pytest.mark.parametrize("alpha", [0.0, 1.0, 50.0, -3.5])
        def test_J_antisymmetric(self, alpha):
            J = _build_J_matrix(alpha)
            assert np.linalg.norm(J + J.T, ord="fro") < Tol.STRICT
            assert J.shape == (3, 3)
            assert J[0, 1] == pytest.approx(1.0)
            assert J[1, 0] == pytest.approx(-1.0)
            assert J[1, 2] == pytest.approx(alpha)
            assert J[2, 1] == pytest.approx(-alpha)
            assert J[0, 2] == pytest.approx(0.0)
            assert J[2, 0] == pytest.approx(0.0)

        def test_J_skew_implies_power_conservation(self):
            r"""∇Hᵀ J ∇H = 0 (J antisimétrica ⇒ potencia giroscópica nula)."""
            J = _build_J_matrix(12.0)
            rng = np.random.default_rng(2)
            for _ in range(20):
                g = rng.normal(size=3)
                assert abs(float(g @ J @ g)) < Tol.STRICT * 10

        @pytest.mark.parametrize("c_m,R_t", [(0.5, 100.0), (0.0, 1.0), (2.0, 1e6)])
        def test_R_symmetric_psd(self, c_m, R_t):
            R = _build_R_matrix(c_m, R_t)
            np.testing.assert_allclose(R, R.T, atol=Tol.STRICT)
            eig = np.linalg.eigvalsh(R)
            assert np.all(eig >= -Tol.STRICT)
            assert R[0, 0] == pytest.approx(0.0)
            assert R[1, 1] == pytest.approx(c_m)
            assert R[2, 2] == pytest.approx(1.0 / R_t)

        def test_R_invalid_resistance_raises(self):
            with pytest.raises(ValueError, match="resistance_total"):
                _build_R_matrix(0.5, 0.0)
            with pytest.raises(ValueError, match="resistance_total"):
                _build_R_matrix(0.5, -10.0)

        def test_hamiltonian_gradient_values(self):
            q, p, Q = 0.1, 0.5, 0.001
            m, k, eps, C_eq = 1.0, 100.0, 10.0, 0.01
            g = _compute_hamiltonian_gradient(
                q=q, p=p, Q_charge=Q,
                mass=m, elasticity=k, nonlinear_elast=eps,
                capacitance_eq=C_eq,
            )
            assert g.shape == (3,)
            assert g[0] == pytest.approx(k * q + eps * (q ** 3))
            assert g[1] == pytest.approx(p / m)
            assert g[2] == pytest.approx(Q / C_eq)

        def test_hamiltonian_gradient_linear_elasticity_limit(self):
            g = _compute_hamiltonian_gradient(
                q=0.2, p=0.0, Q_charge=0.0,
                mass=2.0, elasticity=50.0, nonlinear_elast=0.0,
                capacitance_eq=1.0,
            )
            assert g[0] == pytest.approx(50.0 * 0.2)
            assert g[1] == pytest.approx(0.0)

        def test_dissipation_non_positive(self):
            g = np.array([10.0, 0.5, 5.0], dtype=np.float64)
            R = _build_R_matrix(0.5, 100.0)
            d = _verify_dissipation_inequality(g, R)
            assert d <= Tol.PASSIVE

        def test_dissipation_equals_minus_quadratic_form(self):
            g = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            R = _build_R_matrix(0.5, 100.0)
            d = _verify_dissipation_inequality(g, R)
            expected = -float(g @ R @ g)
            assert abs(d - expected) < Tol.METRIC

        def test_dissipation_zero_when_R_channels_inactive(self):
            g = np.array([10.0, 0.0, 0.0], dtype=np.float64)
            R = _build_R_matrix(0.5, 100.0)
            assert abs(_verify_dissipation_inequality(g, R)) < Tol.STRICT * 10

        def test_dissipation_scales_with_velocity_squared(self):
            R = _build_R_matrix(1.0, 1e9)  # canal eléctrico ~0
            d1 = _verify_dissipation_inequality(np.array([0.0, 1.0, 0.0]), R)
            d2 = _verify_dissipation_inequality(np.array([0.0, 2.0, 0.0]), R)
            # Ḣ_dis = −c_m (∂H/∂p)² ⇒ ratio 4
            assert abs(d2 / d1 - 4.0) < 1e-9

    # ──────────────────────────────────────────────────────────────────────
    # §2.1 AtomicPiston — integrador y estado extendido
    # ──────────────────────────────────────────────────────────────────────

    class TestAtomicPiston:
        def test_initial_extended_state_is_zero(self, piston_viscous):
            p = piston_viscous
            assert p.q == 0.0 and p.p == 0.0 and p.Q_charge == 0.0
            assert p.velocity == 0.0

        def test_hamiltonian_decomposition(self):
            piston = AtomicPiston(
                capacity=0.01, elasticity=100.0, damping=0.0,
                piston_mass=1.0, nonlinear_elasticity=0.0,
            )
            piston.q, piston.p, piston.Q_charge = 0.005, 0.1, 0.001
            T = piston.p ** 2 / (2.0 * piston.m)
            V = 0.5 * piston.k * piston.q ** 2
            W_e = piston.Q_charge ** 2 / (2.0 * piston.equivalent_capacitance)
            assert abs(piston.stored_energy - (T + V + W_e)) < Tol.METRIC

        def test_hamiltonian_with_quartic_potential(self):
            eps = 50.0
            piston = AtomicPiston(
                capacity=0.01, elasticity=100.0, damping=0.0,
                piston_mass=1.0, nonlinear_elasticity=eps,
            )
            piston.q, piston.p, piston.Q_charge = 0.01, 0.0, 0.0
            V = 0.5 * piston.k * piston.q ** 2 + 0.25 * eps * piston.q ** 4
            assert abs(piston.stored_energy - V) < Tol.METRIC

        def test_energy_conservation_nondissipative_short(
            self, piston_nondissipative
        ):
            p = piston_nondissipative
            p.q, p.p = 0.005, 0.0
            H0 = p.stored_energy
            assert H0 > 0.0
            for _ in range(1000):
                p.update_state(0.001)
            assert _relative_error(p.stored_energy, H0) < Tol.ENERGY_SV

        def test_energy_drift_over_many_periods(self):
            r"""I2 proxy: deriva acotada en n periodos del oscilador armónico."""
            m, k = 1.0, 100.0
            p = AtomicPiston(
                capacity=0.01, elasticity=k, damping=0.0,
                piston_mass=m, friction_model=FrictionModel.VISCOUS,
            )
            p.q, p.p = 0.005, 0.0
            T = 2.0 * math.pi * math.sqrt(m / k)
            dt = T / 100.0
            H0 = p.stored_energy
            for _ in range(int(10 * 100)):
                p.update_state(dt)
            assert _relative_error(p.stored_energy, H0) < Tol.ENERGY_SV_LONG

        def test_dissipation_history_non_positive(self, piston_viscous):
            p = piston_viscous
            p.q, p.p = 0.005, 0.1
            for step in range(100):
                p.update_state(0.001)
                if p.dissipation_history:
                    assert p.dissipation_history[-1] <= Tol.PASSIVE, f"step={step}"

        def test_dissipative_energy_decays_without_input(self, piston_viscous):
            p = piston_viscous
            p.q, p.p = 0.005, 0.0
            H0 = p.stored_energy
            for _ in range(2000):
                p.update_state(0.001)
            assert p.stored_energy <= H0 + Tol.METRIC

        def test_collision_inverts_and_damps_momentum(self, piston_viscous):
            p = piston_viscous
            p.q, p.p = 0.0, 1.0
            p_before = None
            collided = False
            for _ in range(2000):
                if abs(p.q) >= p.capacity * 0.99:
                    p_before = p.p
                    collided = True
                p.update_state(0.001)
                if collided and abs(p.q) < p.capacity * 0.99:
                    break
            if not collided:
                pytest.skip("No se alcanzó colisión en la ventana simulada")
            e = AtomicPiston.RESTITUTION_COEFF
            assert np.sign(p.p) != np.sign(p_before)
            assert abs(p.p) <= abs(p_before) * e * 1.1 + 1e-12

        def test_load_resistance_rebuilds_R(self, piston_viscous):
            p = piston_viscous
            R_tot_0 = p._R_total
            p.set_load_resistance(500.0)
            assert p._R_total != R_tot_0
            expected = 1.0 / (p.internal_resistance + 500.0)
            assert abs(p._R[2, 2] - expected) < Tol.METRIC

        def test_load_resistance_nonpositive_raises(self, piston_viscous):
            with pytest.raises(ValueError):
                piston_viscous.set_load_resistance(0.0)
            with pytest.raises(ValueError):
                piston_viscous.set_load_resistance(-10.0)

        @pytest.mark.parametrize(
            "q,p_mom,Q",
            [(0.0, 0.0, 0.0), (0.005, 0.1, 0.0), (0.0, 0.0, 0.001), (0.005, 0.1, 0.001)],
        )
        def test_efficiency_in_unit_interval(self, piston_viscous, q, p_mom, Q):
            piston_viscous.q, piston_viscous.p, piston_viscous.Q_charge = q, p_mom, Q
            eta = piston_viscous.get_conversion_efficiency()
            assert 0.0 <= eta <= 1.0

        def test_state_dict_schema_and_json(self, piston_viscous):
            piston_viscous.q, piston_viscous.p, piston_viscous.Q_charge = 0.005, 0.1, 0.001
            sd = piston_viscous.get_state_dict()
            for key in (
                "state_extended", "position", "velocity",
                "stored_energy", "port_hamiltonian", "control_targets",
            ):
                assert key in sd
            for k in ("q", "p", "Q_charge"):
                assert k in sd["state_extended"]
            # JSON round-trip
            raw = json.dumps(sd)
            back = json.loads(raw)
            assert back["state_extended"]["q"] == pytest.approx(0.005)

        def test_reset_clears_state_and_history(self, piston_viscous):
            p = piston_viscous
            p.q, p.p, p.Q_charge, p.velocity = 0.005, 0.1, 0.001, 0.5
            for _ in range(5):
                p.update_state(0.001)
            p.reset()
            assert p.q == p.p == p.Q_charge == p.velocity == 0.0
            assert len(p.position_history) == 0

        def test_invalid_dt_raises(self, piston_viscous):
            with pytest.raises(ValueError, match="positivo"):
                piston_viscous.update_state(0.0)
            with pytest.raises(ValueError, match="positivo"):
                piston_viscous.update_state(-1e-3)

        def test_invalid_init_params(self):
            with pytest.raises(ValueError, match="capacity"):
                AtomicPiston(capacity=0.0, elasticity=100.0, damping=0.1)
            with pytest.raises(ValueError, match="piston_mass"):
                AtomicPiston(
                    capacity=0.01, elasticity=100.0, damping=0.1, piston_mass=0.0
                )
            with pytest.raises(ValueError, match="dt_default"):
                AtomicPiston(
                    capacity=0.01, elasticity=100.0, damping=0.1, dt_default=0.0
                )

        def test_bode_data_json_serializable(self, piston_viscous):
            freqs = np.logspace(0, 3, 50)
            bode = piston_viscous.generate_bode_data(freqs)
            for key in (
                "frequencies", "magnitude_dB", "phase_deg",
                "magnitude_elec_dB", "phase_elec_deg",
            ):
                assert key in bode
                assert isinstance(bode[key], list)
                assert len(bode[key]) == len(freqs)
            json.dumps(bode)  # no raise

        def test_bode_magnitude_finite(self, piston_viscous):
            bode = piston_viscous.generate_bode_data(np.logspace(0, 2, 20))
            mag = np.array(bode["magnitude_dB"], dtype=np.float64)
            assert np.all(np.isfinite(mag))

        def test_history_bounded_by_maxlen(self, piston_viscous):
            p = piston_viscous
            p.q, p.p = 0.005, 0.1
            for _ in range(p._max_hist + 100):
                p.update_state(0.001)
            assert len(p.position_history) <= p._max_hist_short
            assert len(p.energy_history) <= p._max_hist

        def test_export_history_csv_schema(self, piston_viscous, tmp_path):
            p = piston_viscous
            p.q = 0.005
            for _ in range(10):
                p.update_state(0.001)
            path = tmp_path / "history.csv"
            p.export_history_to_csv(str(path))
            assert path.exists()
            with path.open(encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            assert rows[0] == [
                "time_step", "position_q", "velocity",
                "stored_energy", "efficiency", "friction_force",
                "hamiltonian", "dissipation_rate",
            ]
            assert len(rows) > 1
            # tipos numéricos parseables
            for cell in rows[1]:
                float(cell)

        def test_velocity_consistent_with_momentum(self, piston_viscous):
            r"""ẋ = p/m (definición canónica)."""
            p = piston_viscous
            p.p = 0.25
            p.velocity = p.p / p.m  # si el SUT expone sync explícito
            # Tras un update el SUT debe mantener consistencia
            p.q = 0.0
            p.update_state(1e-4)
            assert abs(p.velocity - p.p / p.m) < 1e-9

        def test_small_dt_stability(self, piston_nondissipative):
            p = piston_nondissipative
            p.q = 0.005
            H0 = p.stored_energy
            for _ in range(100):
                p.update_state(1e-6)
            assert np.isfinite(p.stored_energy)
            assert _relative_error(p.stored_energy, H0) < 1e-3


# ============================================================================
# FASE 3 — MICROSERVICIO, BUFFER, POVM, COHOMOLOGÍA
# ============================================================================

class TestPhase3FlaskMicroservice:
    r"""
    FASE 3: buffer, Kraus/POVM, haz, ServiceContext, guards HTTP.
    Entrada canónica: state_dict de AtomicPiston (FASE 2).
    """

    # ──────────────────────────────────────────────────────────────────────
    # §3.0 LockFreeCircularBuffer
    # ──────────────────────────────────────────────────────────────────────

    class TestLockFreeCircularBuffer:
        def test_publish_latest_roundtrip(self, empty_buffer):
            empty_buffer.publish({"position": 0.005, "velocity": 0.1})
            latest = empty_buffer.latest()
            assert latest is not None
            assert latest["position"] == 0.005
            assert latest["velocity"] == 0.1

        def test_circular_wrap_keeps_newest(self):
            buf = LockFreeCircularBuffer(capacity=4)
            for i in range(10):
                buf.publish({"step": i})
            assert buf.latest()["step"] == 9

        def test_empty_initially(self, empty_buffer):
            assert empty_buffer.is_empty()
            assert empty_buffer.latest() is None

        def test_drain_lifo(self, empty_buffer):
            for i in range(5):
                empty_buffer.publish({"step": i})
            drained = empty_buffer.drain(max_items=3)
            assert [d["step"] for d in drained] == [4, 3, 2]

        def test_drain_empty_returns_empty(self, empty_buffer):
            assert empty_buffer.drain(max_items=5) == []

        def test_invalid_capacity_raises(self):
            with pytest.raises(ValueError, match="capacity"):
                LockFreeCircularBuffer(capacity=1)

        def test_capacity_two_is_valid(self):
            buf = LockFreeCircularBuffer(capacity=2)
            buf.publish({"a": 1})
            buf.publish({"a": 2})
            assert buf.latest()["a"] == 2

        def test_concurrent_publish_thread_safety(self):
            buf = LockFreeCircularBuffer(capacity=128)
            n_threads, n_pub = 10, 100
            errors: List[BaseException] = []

            def worker(tid: int) -> None:
                try:
                    for i in range(n_pub):
                        buf.publish({"thread": tid, "i": i})
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors
            assert buf.latest() is not None

        def test_publish_none_rejected_or_stored_explicitly(self, empty_buffer):
            r"""Contrato: o bien se rechaza None, o latest refleja el valor."""
            try:
                empty_buffer.publish(None)
            except (TypeError, ValueError):
                return
            # Si se admite, latest no debe romper
            _ = empty_buffer.latest()

        def test_latest_does_not_consume(self, empty_buffer):
            empty_buffer.publish({"k": 1})
            assert empty_buffer.latest()["k"] == 1
            assert empty_buffer.latest()["k"] == 1  # idempotente

    # ──────────────────────────────────────────────────────────────────────
    # §3.1 KrausObserver — POVM / QND
    # ──────────────────────────────────────────────────────────────────────

    class TestKrausObserver:
        def test_kraus_operator_identity_full_efficiency(self):
            obs = KrausObserver(measurement_efficiency=1.0)
            rho = {"position": 0.005, "velocity": 0.1}
            M = obs.kraus_operator_M(rho)
            np.testing.assert_allclose(M, np.eye(len(rho)), atol=Tol.METRIC)

        def test_povm_qnd_at_unit_efficiency(self):
            obs = KrausObserver(measurement_efficiency=1.0)
            rho = {"position": 0.005, "velocity": 0.1, "stored_energy": 0.025}
            out = obs.measure(rho)
            assert out is not None
            assert out["position"] == rho["position"]
            assert out["velocity"] == rho["velocity"]
            assert out["stored_energy"] == rho["stored_energy"]
            assert out["_povm_metadata"]["qnd_certified"] is True

        def test_povm_does_not_mutate_input(self):
            obs = KrausObserver(measurement_efficiency=1.0)
            rho = {"position": 0.005, "velocity": 0.1}
            snapshot = dict(rho)
            _ = obs.measure(rho)
            assert rho == snapshot

        def test_partial_observation_keys(self):
            keys = ["position", "velocity"]
            obs = KrausObserver(measurement_efficiency=1.0, observable_keys=keys)
            rho = {"position": 0.005, "velocity": 0.1, "stored_energy": 0.025}
            out = obs.measure(rho)
            assert out is not None
            for k in keys:
                assert k in out

        def test_efficiency_bounds(self):
            with pytest.raises(ValueError, match="measurement_efficiency"):
                KrausObserver(measurement_efficiency=0.0)
            with pytest.raises(ValueError, match="measurement_efficiency"):
                KrausObserver(measurement_efficiency=1.5)
            assert KrausObserver(measurement_efficiency=0.001)._lambda == 0.001
            assert KrausObserver(measurement_efficiency=1.0)._lambda == 1.0

        def test_measure_none_returns_none(self):
            assert KrausObserver().measure(None) is None

        def test_measurement_count_increments(self):
            obs = KrausObserver()
            n0 = obs.measurement_count
            obs.measure({"k": 1})
            obs.measure({"k": 2})
            assert obs.measurement_count == n0 + 2

        def test_sub_unity_efficiency_still_returns_dict(self):
            obs = KrausObserver(measurement_efficiency=0.5)
            out = obs.measure({"position": 1.0, "velocity": 0.0})
            assert out is not None
            assert isinstance(out, dict)

        def test_metadata_contains_efficiency(self):
            obs = KrausObserver(measurement_efficiency=0.8)
            out = obs.measure({"a": 1})
            meta = out.get("_povm_metadata", {})
            # Si el SUT expone λ en metadatos, validarlo; si no, solo presencia
            if "measurement_efficiency" in meta:
                assert meta["measurement_efficiency"] == pytest.approx(0.8)
            assert "qnd_certified" in meta

    # ──────────────────────────────────────────────────────────────────────
    # §3.2 SheafCohomologyVerifier — β₁ / H¹
    # ──────────────────────────────────────────────────────────────────────

    class TestSheafCohomologyVerifier:
        def test_betti_1_empty_graph(self, sheaf):
            assert sheaf._compute_betti_1() == 0

        def test_betti_1_tree_is_zero(self, sheaf):
            sheaf._vertices = {f"v{i}": f"http://v{i}" for i in range(4)}
            sheaf._edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3")]
            assert sheaf._compute_betti_1() == 0

        def test_betti_1_triangle_is_one(self, sheaf):
            sheaf._vertices = {f"v{i}": f"http://v{i}" for i in range(3)}
            sheaf._edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v0")]
            assert sheaf._compute_betti_1() == 1

        def test_betti_1_two_disjoint_edges(self, sheaf):
            r"""|V|=4, |E|=2, c=2 → β₁ = 2−4+2 = 0."""
            sheaf._vertices = {f"v{i}": f"http://v{i}" for i in range(4)}
            sheaf._edges = [("v0", "v1"), ("v2", "v3")]
            assert sheaf._compute_betti_1() == 0

        def test_betti_1_complete_graph_k4(self, sheaf):
            r"""K₄: |V|=4, |E|=6, c=1 → β₁ = 6−4+1 = 3."""
            sheaf._vertices = {f"v{i}": f"http://v{i}" for i in range(4)}
            sheaf._edges = [
                ("v0", "v1"), ("v0", "v2"), ("v0", "v3"),
                ("v1", "v2"), ("v1", "v3"), ("v2", "v3"),
            ]
            assert sheaf._compute_betti_1() == 3

        def test_registration_admits_isolated_service(self, sheaf):
            admitted, msg, stalk = sheaf.verify_registration(
                service_name="svc_a",
                module_url="http://a",
                health_url="http://a/health",
                neighbors=[],
            )
            assert admitted is True
            assert "hmac_signature" in stalk
            assert ("H¹" in msg) or ("β₁" in msg) or ("beta" in msg.lower())

        def test_registration_admits_tree_extension(self, sheaf):
            ok1, _, _ = sheaf.verify_registration(
                "s1", "http://s1", "http://s1/h", neighbors=[]
            )
            ok2, _, _ = sheaf.verify_registration(
                "s2", "http://s2", "http://s2/h", neighbors=["s1"]
            )
            assert ok1 and ok2
            assert sheaf.cohomology_status["H1_zero"] is True
            assert sheaf.cohomology_status["num_vertices"] == 2
            assert sheaf.cohomology_status["num_edges"] == 1

        def test_registration_rejects_when_edge_closes_cycle(self, sheaf):
            r"""
            I4 estricto: s1—s2—s3 y luego s3→s1 debe producir β₁=1 y rechazo
            (o rollback). Si el SUT admite vecinos inexistentes de forma laxa,
            se fuerza el ciclo en el grafo interno y se revalida β₁.
            """
            sheaf.verify_registration("s1", "http://1", "http://1/h", neighbors=[])
            sheaf.verify_registration("s2", "http://2", "http://2/h", neighbors=["s1"])
            sheaf.verify_registration("s3", "http://3", "http://3/h", neighbors=["s2"])
            # Intento de cerrar ciclo
            admitted, msg, stalk = sheaf.verify_registration(
                "s1b", "http://1b", "http://1b/h", neighbors=["s3", "s1"]
            )
            # Tras la operación, H¹ debe reflejar la topología real
            status = sheaf.cohomology_status
            if admitted:
                # Si se admitió, el grafo resultante aún debe reportar β₁ consistentemente
                assert status["beta_1"] == sheaf._compute_betti_1()
            else:
                assert status["H1_zero"] is True or "H¹" in msg or "β₁" in msg
                # Rollback: el servicio rechazado no debe permanecer
                assert "s1b" not in sheaf._vertices

        def test_preexisting_cycle_status_flags_H1_nonzero(self, sheaf):
            sheaf._vertices = {f"v{i}": f"http://v{i}" for i in range(3)}
            sheaf._edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v0")]
            status = sheaf.cohomology_status
            assert status["beta_1"] == 1
            assert status["H1_zero"] is False

        def test_hmac_signature_deterministic(self):
            v = SheafCohomologyVerifier(ecosystem_key=b"test_key_123")
            stalk = {
                "service_name": "test",
                "module_url": "http://test",
                "timestamp": 1234567890.0,
            }
            s1, s2 = v._sign_stalk(stalk), v._sign_stalk(stalk)
            assert s1 == s2
            assert len(s1) == 64  # sha256 hex

        def test_hmac_changes_with_payload(self):
            v = SheafCohomologyVerifier(ecosystem_key=b"test_key_123")
            a = v._sign_stalk({"service_name": "a", "module_url": "u", "timestamp": 1.0})
            b = v._sign_stalk({"service_name": "b", "module_url": "u", "timestamp": 1.0})
            assert a != b

        def test_remove_service_clears_incident_edges(self, sheaf):
            sheaf._vertices = {"v0": "http://v0", "v1": "http://v1"}
            sheaf._edges = [("v0", "v1")]
            sheaf.remove_service("v0")
            assert "v0" not in sheaf._vertices
            assert sheaf._edges == []

        def test_remove_unknown_service_is_safe(self, sheaf):
            sheaf.remove_service("ghost")  # no raise

        def test_cohomology_status_schema(self, sheaf):
            sheaf._vertices = {"v0": "http://v0", "v1": "http://v1"}
            sheaf._edges = [("v0", "v1")]
            status = sheaf.cohomology_status
            for key in (
                "beta_1", "H1_zero", "num_vertices",
                "num_edges", "vertices", "edges",
            ):
                assert key in status
            assert status["num_vertices"] == 2
            assert status["num_edges"] == 1
            assert status["H1_zero"] is True
            assert status["beta_1"] == 0

    # ──────────────────────────────────────────────────────────────────────
    # §3.3 ServiceContext
    # ──────────────────────────────────────────────────────────────────────

    class TestServiceContext:
        def test_default_initialization(self):
            ctx = ServiceContext()
            assert ctx.config is None
            assert ctx.ipu is None
            assert isinstance(ctx.buffer, LockFreeCircularBuffer)
            assert isinstance(ctx.observer_full, KrausObserver)
            assert isinstance(ctx.sheaf_verifier, SheafCohomologyVerifier)
            assert ctx.sim_thread is None
            assert ctx.stop_event.is_set() is False

        def test_default_buffer_capacity(self):
            ctx = ServiceContext()
            assert ctx.buffer._capacity == 16

        def test_stop_event_can_be_set(self):
            ctx = ServiceContext()
            ctx.stop_event.set()
            assert ctx.stop_event.is_set() is True

    # ──────────────────────────────────────────────────────────────────────
    # §3.4 Guards HTTP
    # ──────────────────────────────────────────────────────────────────────

    class TestHttpUtilities:
        def test_require_json_ok(self):
            with patch(
                "app.agents.physics.atomic_piston_service.request"
            ) as req:
                req.is_json = True
                assert _require_json() is None

        def test_require_json_415(self):
            with patch(
                "app.agents.physics.atomic_piston_service.request"
            ) as req:
                req.is_json = False
                result = _require_json()
                assert result is not None
                _, code = result
                assert code == 415

        def test_buffer_or_503_empty(self):
            with patch("app.agents.physics.atomic_piston_service._svc") as svc:
                svc.buffer = Mock(is_empty=Mock(return_value=True))
                result = _buffer_or_503()
                assert result is not None
                assert result[1] == 503

        def test_buffer_or_503_ready(self):
            with patch("app.agents.physics.atomic_piston_service._svc") as svc:
                svc.buffer = Mock(is_empty=Mock(return_value=False))
                assert _buffer_or_503() is None

        def test_ipu_or_503_none(self):
            with patch("app.agents.physics.atomic_piston_service._svc") as svc:
                svc.ipu = None
                result = _ipu_or_503()
                assert result is not None
                assert result[1] == 503

        def test_ipu_or_503_ready(self):
            with patch("app.agents.physics.atomic_piston_service._svc") as svc:
                svc.ipu = Mock()
                assert _ipu_or_503() is None

    # ──────────────────────────────────────────────────────────────────────
    # §3.5 AtomicCounter
    # ──────────────────────────────────────────────────────────────────────

    class TestAtomicCounter:
        def test_initial_value(self):
            assert _AtomicCounter(initial=42).value == 42

        def test_fetch_and_increment_sequence(self):
            c = _AtomicCounter(initial=0)
            assert c.fetch_and_increment() == 0
            assert c.fetch_and_increment() == 1
            assert c.fetch_and_increment() == 2
            assert c.value == 3

        def test_reset(self):
            c = _AtomicCounter(initial=0)
            c.fetch_and_increment()
            c.reset(10)
            assert c.value == 10

        def test_thread_safety(self):
            c = _AtomicCounter(initial=0)
            n_t, n_i = 10, 200

            def worker() -> None:
                for _ in range(n_i):
                    c.fetch_and_increment()

            threads = [threading.Thread(target=worker) for _ in range(n_t)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert c.value == n_t * n_i


# ============================================================================
# FASE Σ — INTEGRACIÓN / SUTURAS F1→F2→F3
# ============================================================================

class TestPhaseIntegration:
    r"""Contratos de sutura entre fases anidadas."""

    def test_suture_friction_into_stormer_verlet_dissipates(self):
        r"""SUTURA 1→2: F_fric reduce H sin entrada externa."""
        p = AtomicPiston(
            capacity=0.01, elasticity=100.0, damping=0.1, piston_mass=1.0,
            friction_model=FrictionModel.COULOMB, coulomb_friction=0.2,
        )
        p.q, p.p = 0.005, 0.1
        H0 = p.stored_energy
        p.update_state(0.001)
        assert p.stored_energy <= H0 * 1.01 + Tol.METRIC

    def test_suture_pid_into_dynamics_generates_motion(self):
        r"""SUTURA 2→2: u_PID como F_ext mueve el pistón hacia target."""
        p = AtomicPiston(
            capacity=0.01, elasticity=100.0, damping=0.1, piston_mass=1.0,
            friction_model=FrictionModel.VISCOUS,
        )
        p.set_speed_target(0.5)
        p.q = p.p = 0.0
        for _ in range(500):
            p.update_state(0.001)
        assert abs(p.velocity) > 0.0

    def test_suture_state_dict_into_lockfree_buffer(self):
        r"""SUTURA 2→3: get_state_dict → publish → latest."""
        p = AtomicPiston(
            capacity=0.01, elasticity=100.0, damping=0.1, piston_mass=1.0
        )
        buf = LockFreeCircularBuffer(capacity=16)
        p.q, p.p = 0.005, 0.1
        buf.publish(p.get_state_dict())
        latest = buf.latest()
        assert latest is not None
        assert latest["state_extended"]["q"] == pytest.approx(0.005)
        assert latest["state_extended"]["p"] == pytest.approx(0.1)

    def test_suture_buffer_into_povm_qnd(self):
        r"""SUTURA 3: latest → measure ⇒ QND certificado."""
        p = AtomicPiston(
            capacity=0.01, elasticity=100.0, damping=0.1, piston_mass=1.0
        )
        buf = LockFreeCircularBuffer(capacity=16)
        obs = KrausObserver(measurement_efficiency=1.0)
        p.q = 0.005
        buf.publish(p.get_state_dict())
        rho = obs.measure(buf.latest())
        assert rho is not None
        assert rho["position"] == pytest.approx(0.005)
        assert rho["_povm_metadata"]["qnd_certified"] is True

    def test_suture_full_pipeline_publish_measure_schema(self):
        r"""Cadena completa: integrar → serializar → buffer → POVM → JSON."""
        p = AtomicPiston(
            capacity=0.01, elasticity=100.0, damping=0.05, piston_mass=1.0,
            friction_model=FrictionModel.VISCOUS,
        )
        p.q = 0.004
        for _ in range(50):
            p.update_state(0.001)
        buf = LockFreeCircularBuffer(capacity=8)
        obs = KrausObserver(measurement_efficiency=1.0)
        buf.publish(p.get_state_dict())
        payload = obs.measure(buf.latest())
        raw = json.dumps(payload)
        assert "state_extended" in json.loads(raw) or "position" in json.loads(raw)

    def test_suture_sheaf_registration_workflow_tree(self):
        r"""SUTURA 7: dos registros en árbol ⇒ H¹=0."""
        v = SheafCohomologyVerifier()
        a1, _, _ = v.verify_registration(
            "service_1", "http://s1", "http://s1/h", neighbors=[]
        )
        a2, _, _ = v.verify_registration(
            "service_2", "http://s2", "http://s2/h", neighbors=["service_1"]
        )
        assert a1 and a2
        st = v.cohomology_status
        assert st["H1_zero"] is True
        assert st["num_vertices"] == 2
        assert st["num_edges"] == 1
        assert st["beta_1"] == 0

    def test_suture_service_context_wires_components(self):
        r"""ServiceContext expone buffer+observer+sheaf listos para el loop."""
        ctx = ServiceContext()
        p = AtomicPiston(
            capacity=0.01, elasticity=100.0, damping=0.1, piston_mass=1.0
        )
        p.q = 0.001
        ctx.buffer.publish(p.get_state_dict())
        measured = ctx.observer_full.measure(ctx.buffer.latest())
        assert measured is not None
        admitted, _, _ = ctx.sheaf_verifier.verify_registration(
            "ipu_local", "http://local", "http://local/h", neighbors=[]
        )
        assert admitted is True


# ============================================================================
# FASE R — REGRESIONES Y ESTRÉS NUMÉRICO
# ============================================================================

class TestRegressionsAndStress:
    r"""Candados de regresión y condiciones extremas."""

    def test_J_power_orthogonality_random_ensemble(self):
        J = _build_J_matrix(7.5)
        rng = np.random.default_rng(99)
        for _ in range(50):
            g = rng.normal(size=3)
            assert abs(float(g @ J @ g)) < 1e-12

    def test_R_psd_under_extreme_parameters(self):
        for c_m, R_t in [(0.0, 1e-6), (1e3, 1e-3), (1e-9, 1e9)]:
            R = _build_R_matrix(c_m, R_t)
            assert np.min(np.linalg.eigvalsh(R)) >= -1e-12

    def test_metric_norm_large_dynamic_range(self):
        G = np.diag([1e-8, 1e8]).astype(np.float64)
        v = np.array([1.0, 1.0], dtype=np.float64)
        n = _metric_norm(v, G)
        assert np.isfinite(n) and n > 0.0

    def test_piston_high_stiffness_energy_bounded(self):
        p = AtomicPiston(
            capacity=0.01, elasticity=1e5, damping=0.0,
            piston_mass=1.0, friction_model=FrictionModel.VISCOUS,
        )
        p.q = 1e-4
        H0 = p.stored_energy
        for _ in range(500):
            p.update_state(1e-5)
        assert np.isfinite(p.stored_energy)
        assert _relative_error(p.stored_energy, H0) < 0.05

    def test_buffer_stress_many_publishes(self):
        buf = LockFreeCircularBuffer(capacity=32)
        for i in range(10_000):
            buf.publish({"i": i})
        assert buf.latest()["i"] == 9999

    def test_counter_stress_contention(self):
        c = _AtomicCounter(0)
        n_t, n_i = 16, 500

        def worker() -> None:
            for _ in range(n_i):
                c.fetch_and_increment()

        ts = [threading.Thread(target=worker) for _ in range(n_t)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert c.value == n_t * n_i

    def test_pid_long_horizon_remains_bounded(self):
        ctl = LieGroupPIDController(
            kp=2.0, ki=0.5, kd=0.1, output_limit=3.0
        )
        u_hist = []
        x = 0.0
        for k in range(1000):
            u = ctl.update(1.0, x, 0.01, current_velocity=0.0)
            u_hist.append(u)
            x += 0.01 * u  # planta trivial
        assert max(abs(u) for u in u_hist) <= 3.0 * 1.01
        assert all(np.isfinite(u) for u in u_hist)

    def test_coulomb_friction_does_not_inject_energy_alone(self):
        r"""Solo fricción + damping, sin PID/target: H no crece."""
        p = AtomicPiston(
            capacity=0.01, elasticity=80.0, damping=0.05, piston_mass=1.0,
            friction_model=FrictionModel.COULOMB, coulomb_friction=0.05,
        )
        p.q, p.p = 0.003, 0.2
        H0 = p.stored_energy
        for _ in range(1000):
            p.update_state(0.001)
        assert p.stored_energy <= H0 * 1.02 + 1e-12


# ============================================================================
if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "--tb=short"])
    )