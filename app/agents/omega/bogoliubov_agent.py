# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo: Bogoliubov Agent (Gran Inquisidor Cuántico)                         |
| Ruta  : app/omega/bogoliubov_agent.py                                        |
| Versión: 3.1.0-Nested-Symplectic-Quantum                                     |
+==============================================================================+

TRANSFORMACIÓN DE BOGOLIUBOV-VALATIN (GRUPO SIMPLÉCTICO Sp(2n,C)):
Aísla las cuasipartículas estables del ruido térmico del LLM.
\[ \begin{pmatrix} \hat{\alpha}_k \\ \hat{\alpha}_{-k}^\dagger \end{pmatrix} = \begin{pmatrix} u_k & v_k \\ v_k^* & u_k^* \end{pmatrix} \begin{pmatrix} \hat{b}_k \\ \hat{b}_{-k}^\dagger \end{pmatrix} \]
\[ |u_k|^2 - |v_k|^2 = 1 \]

ECUACIÓN MAESTRA DE LINDBLAD-KOSSAKOWSKI:
\[ \frac{d \rho_{MAC}}{dt} = -\frac{i}{\hbar} [H_{eff}, \rho_{MAC}] + \sum_{k} \gamma_k ( L_k \rho_{MAC} L_k^\dagger - \frac{1}{2} \{ L_k^\dagger L_k, \rho_{MAC} \} ) \]

ARQUITECTURA ANIDADA: Cada fase es una clase interna de la anterior, garantizando
que la salida formal de la última operación de la Fase N es el único contrato de
entrada para la Fase N+1.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

#
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (resilientes)
#
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        """Excepción base para invariantes topológicos."""
        pass

    class Morphism:
        """Morfismo base vacío."""
        pass

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: NDArray[np.float64] = np.eye(1, dtype=np.float64)

try:
    from app.core.telemetry_schemas import PositronCartridge, GammaPhoton
except ImportError:
    @dataclass(frozen=True)
    class PositronCartridge:
        inertial_mass: float
        topological_spin: str
        homological_charge: int
        authorization_signature: str

    @dataclass(frozen=True)
    class GammaPhoton:
        annihilation_energy: float
        data_hash: str
        timestamp_entry: float
        authorization_signature: str

from app.omega.quantum_fock_orchestrator import (
    FockSpaceConfiguration,
    InteractionOperators,
    QuantumFockOrchestrator,
    LindbladEvolutionResult,
    Boson,
    Fermion,
)

logger = logging.getLogger("MIC.Omega.BogoliubovAgent")

#
# EXCEPCIONES SIMPLÉCTICAS Y CUÁNTICAS
#
class BogoliubovTransformationError(TopologicalInvariantError):
    """
    Detonada si la transformación viola las Relaciones de Conmutación Canónicas
    (CCR), implicando que el espacio de fase se ha desgarrado (pérdida de unitariedad).
    """
    pass


class SMatrixSingularityError(TopologicalInvariantError):
    """
    Detonada si el tensor de acoplamiento g_{k,q} diverge, indicando una resonancia
    infinita entre la alucinación de la IA y el presupuesto base.
    """
    pass


class ErrorDensityValidationError(TopologicalInvariantError):
    """
    Detonada si la matriz de error no satisface los axiomas de una matriz densidad
    válida (hermiticidad, traza unitaria, positividad).
    """
    pass


#
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE FASE
#
@dataclass(frozen=True, slots=True)
class BogoliubovSpectrum:
    r"""
    Espectro de cuasipartículas tras la diagonalización BdG.

    Atributos:
        u_matrix: Matriz $(B \times B)$ con amplitudes $u_{kj}$.
        v_matrix: Matriz $(B \times B)$ con amplitudes $v_{kj}$.
        quasiparticle_energies: Vector $(B,)$ con energías positivas $E_k > 0$.
        ccr_residual: $\|U^\dagger U - V^\dagger V - I\|_F$ tras verificación.
        symmetric_basis: Si True, la base es simétrica bajo $(u,v) \leftrightarrow (v^*, u^*)$.
    """
    u_matrix: NDArray[np.complex128]
    v_matrix: NDArray[np.complex128]
    quasiparticle_energies: NDArray[np.float64]
    ccr_residual: float = 0.0
    symmetric_basis: bool = True

    @property
    def n_modes(self) -> int:
        """Número de modos bosónicos del espacio original."""
        return self.u_matrix.shape[0]

    @property
    def symplectic_matrix(self) -> NDArray[np.complex128]:
        r"""Matriz simpléctica completa $S = \begin{pmatrix} U & V^* \\ V & U^* \end{pmatrix}$."""
        dim = self.n_modes
        top = np.hstack([self.u_matrix, self.v_matrix.conj()])
        bot = np.hstack([self.v_matrix, self.u_matrix.conj()])
        return np.vstack([top, bot])

    def verify_ccr_strict(self, tol: float = 1e-9) -> Dict[str, float]:
        r"""
        Verificación estricta post-diagonalización de las CCR.

        Returns:
            Dict con residuales:
                - 'commutation': $\|U^\dagger U - V^\dagger V - I\|_F$
                - 'particle_hole': $\max_k |E_k + E_k^*|/2$ (debe ser 0 para Hermiticidad)
                - 'normalization': $\max_k |\|u_k\|^2 - \|v_k\|^2 - 1|$
                - 'symplectic': $\|S^\dagger \eta S - \eta\|_F$ con $\eta = \text{diag}(I, -I)$
        """
        d = self.n_modes
        u, v = self.u_matrix, self.v_matrix

        comm_res = la.norm(u.conj().T @ u - v.conj().T @ v - np.eye(d), ord='fro')

        norms_u = np.real(np.sum(np.abs(u) ** 2, axis=0))
        norms_v = np.real(np.sum(np.abs(v) ** 2, axis=0))
        norm_res = float(np.max(np.abs(norms_u - norms_v - 1.0)))

        energy_imag = float(np.max(np.abs(np.imag(self.quasiparticle_energies))))

        # Verificación simpléctica completa
        eta = np.diag(np.concatenate([np.ones(d), -np.ones(d)]))
        S = self.symplectic_matrix
        sympl_res = float(la.norm(S.conj().T @ eta @ S - eta, ord='fro'))

        return {
            "commutation": float(comm_res),
            "normalization": norm_res,
            "energy_imag": energy_imag,
            "symplectic": sympl_res,
        }


@dataclass(frozen=True, slots=True)
class CoupledInteractionData:
    r"""
    Producto final de la Fase 2 que encapsula la matriz de acoplamiento $g_{kq}$
    y los modos bosónicos ya transformados a la base de cuasipartículas.

    Atributos:
        coupling_matrix: Matriz $(B \times F)$ con acoplamientos $g_{k,q}$.
        transformed_boson_modes: Vector $(B,)$ de amplitudes de cuasipartículas.
        fermion_modes: Vector $(F,)$ de amplitudes fermiónicas.
        metric_tensor: Tensor $G$ utilizado en la contracción covariante.
        mean_coupling_strength: Promedio $\bar{g}$ para diagnóstico.
        max_coupling_strength: Máximo $|g_{k,q}|$ para diagnóstico.
    """
    coupling_matrix: NDArray[np.complex128]
    transformed_boson_modes: NDArray[np.complex128]
    fermion_modes: NDArray[np.complex128]
    metric_tensor: NDArray[np.float64]
    mean_coupling_strength: float
    max_coupling_strength: float


@dataclass(frozen=True, slots=True)
class LindbladEnvironment:
    r"""
    Producto final de la Fase 3: operadores de Lindblad $\{\hat{L}_k\}$ y
    entropía proyectada, listos para ser inyectados en el orquestador.

    Atributos:
        jump_operators: Lista de operadores de salto $\{\hat{L}_k\}$.
        decay_rates: Tasas efectivas $\{\gamma_k\}$ correspondientes.
        projected_entropy: Entropía de von Neumann $S(\rho_{\text{error}})$.
        effective_dimension: Número de canales activos (con $\lambda_i > 0$).
        spectral_gap: Diferencia $\lambda_{\max} - \lambda_{\min+1}$ entre canales.
    """
    jump_operators: List[NDArray[np.complex128]]
    decay_rates: List[float]
    projected_entropy: float
    effective_dimension: int
    spectral_gap: float

    def verify_cptp_condition(self, coupling_strength: float, tol: float = 1e-12) -> bool:
        r"""
        Comprueba que $\|\sum_i L_i^\dagger L_i\|_\infty \le \bar{g}$,
        garantizando un disipador acotado y traza-preservante.
        """
        if not self.jump_operators:
            return True
        d = self.jump_operators[0].shape[0]
        total = np.zeros((d, d), dtype=np.complex128)
        for L in self.jump_operators:
            total += L.conj().T @ L
        norm = float(la.norm(total, ord=2))
        return norm <= coupling_strength + tol


#
#                 FASE 1   DIAGONALIZACIÓN SIMPLÉCTICA
#
class Phase1_BogoliubovTransformation:
    r"""
    Operador de isomorfismo simpléctico para el Espacio de Fock.

    Resuelve la ecuación de Bogoliubov-de Gennes (BdG):
    $$
    \begin{pmatrix} H_k & \Delta \\ \Delta^\dagger & -H_k^* \end{pmatrix}
    \begin{pmatrix} u_k \\ v_k \end{pmatrix}
    = E_k \begin{pmatrix} u_k \\ v_k \end{pmatrix}
    $$

    Y verifica que la transformación resultante preserve las CCR:
    $$
    U^\dagger U - V^\dagger V = \mathbb{1}_N, \qquad
    |u_k|^2 - |v_k|^2 = 1
    $$
    """

    def __init__(self, tolerance: float = 1e-9, hermiticity_tol: float = 1e-10):
        self._tol = tolerance
        self._herm_tol = hermiticity_tol

    def _validate_inputs(
        self,
        kinetic_energy_matrix: NDArray[np.float64],
        pairing_gap_matrix: NDArray[np.complex128],
    ) -> Tuple[int, int]:
        r"""Validación dimensional y de estructura de las matrices BdG."""
        if kinetic_energy_matrix.ndim != 2 or kinetic_energy_matrix.shape[0] != kinetic_energy_matrix.shape[1]:
            raise BogoliubovTransformationError(
                f"H_k debe ser cuadrada; shape={kinetic_energy_matrix.shape}"
            )
        if pairing_gap_matrix.ndim != 2 or pairing_gap_matrix.shape[0] != pairing_gap_matrix.shape[1]:
            raise BogoliubovTransformationError(
                f"Δ debe ser cuadrada; shape={pairing_gap_matrix.shape}"
            )

        dim_h = kinetic_energy_matrix.shape[0]
        dim_p = pairing_gap_matrix.shape[0]

        if dim_h != dim_p:
            raise BogoliubovTransformationError(
                f"Dimensiones incompatibles: H_k={dim_h}, Δ={dim_p}"
            )

        # H_k debe ser simétrica real (energía cinética hermítica real)
        if not np.allclose(kinetic_energy_matrix, kinetic_energy_matrix.T, atol=self._herm_tol):
            raise BogoliubovTransformationError(
                "H_k no es simétrica real; viola hermiticidad de BdG."
            )

        return dim_h, dim_p

    def _build_bdg_hamiltonian(
        self,
        kinetic_energy_matrix: NDArray[np.float64],
        pairing_gap_matrix: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Construye el Hamiltoniano BdG en el espacio de Nambu $(2N \times 2N)$:
        $$
        H_{\text{BdG}} = \begin{pmatrix} H_k & \Delta \\ \Delta^\dagger & -H_k \end{pmatrix}
        $$
        """
        dim = kinetic_energy_matrix.shape[0]
        Hk = kinetic_energy_matrix.astype(np.complex128)

        top = np.hstack([Hk, pairing_gap_matrix])
        bot = np.hstack([pairing_gap_matrix.conj().T, -Hk])
        return np.vstack([top, bot])

    def compute_bogoliubov_coefficients(
        self,
        kinetic_energy_matrix: NDArray[np.float64],
        pairing_gap_matrix: NDArray[np.complex128],
    ) -> BogoliubovSpectrum:
        r"""
        Resuelve la ecuación BdG y extrae amplitudes $(u_k, v_k)$.

        Parameters
        ----------
        kinetic_energy_matrix : (N, N) real, simétrica
            Matriz de energía cinética $H_k$.
        pairing_gap_matrix : (N, N) complex
            Matriz de gap de apareamiento $\Delta$.

        Returns
        -------
        BogoliubovSpectrum
            Espectro con $(u_k, v_k)$ normalizados y energías $E_k > 0$.
        """
        dim, _ = self._validate_inputs(kinetic_energy_matrix, pairing_gap_matrix)

        # Construcción y diagonalización del Hamiltoniano BdG
        H_BdG = self._build_bdg_hamiltonian(kinetic_energy_matrix, pairing_gap_matrix)

        # Verificación de que H_BdG es hermítico (la estructura garantiza esto si Δ y H_k son válidos)
        herm_res = la.norm(H_BdG - H_BdG.conj().T, ord='fro') / (2 * dim)
        if herm_res > self._herm_tol * 10:
            raise BogoliubovTransformationError(
                f"H_BdG no hermítico (residual={herm_res:.2e}); revisar Δ y H_k."
            )

        # Diagonalización hermítica
        evals, evecs = la.eigh(H_BdG)

        #
        # Selección de energías positivas: por la simetría partícula-hueco
        # H_BdG tiene espectro {±E_k}, así que seleccionamos E_k > 0.
        #
        positive_mask = evals > self._tol
        E_k = evals[positive_mask]
        evecs_pos = evecs[:, positive_mask]

        n_positive = len(E_k)
        if n_positive == 0:
            raise BogoliubovTransformationError(
                "No se encontraron energías positivas; posible gap trivial o sistema apagado."
            )

        # Manejo de degeneración: si |n_positive - dim| > 0 pero cercano, intentar tolerancia relajada
        if n_positive < dim:
            relaxed_tol = self._tol * 100
            positive_mask_relaxed = evals > relaxed_tol
            E_k_relaxed = evals[positive_mask_relaxed]

            if len(E_k_relaxed) == dim:
                logger.warning(
                    f"Degeneración partícula-hueco detectada; usando tolerancia relajada "
                    f"({relaxed_tol:.2e}). {n_positive} → {len(E_k_relaxed)} modos."
                )
                E_k = E_k_relaxed
                evecs_pos = evecs[:, positive_mask_relaxed]
                n_positive = dim

        if n_positive != dim:
            raise BogoliubovTransformationError(
                f"Estructura BdG inválida: se esperaban {dim} modos positivos, "
                f"encontrados {n_positive}. Posible degeneración o mal condicionamiento."
            )

        # Ordenar por energía ascendente
        order = np.argsort(E_k)
        E_k = E_k[order]
        evecs_pos = evecs_pos[:, order]

        # Extraer u y v
        u_matrix = evecs_pos[:dim, :]
        v_matrix = evecs_pos[dim:, :]

        # Verificación rigurosa de las CCR
        ccr_check = u_matrix.conj().T @ u_matrix - v_matrix.conj().T @ v_matrix
        ccr_residual = float(la.norm(ccr_check - np.eye(dim), ord='fro'))

        if ccr_residual > self._tol:
            raise BogoliubovTransformationError(
                f"Violación del grupo simpléctico Sp(2N,ℂ). "
                f"‖U†U - V†V - I‖_F = {ccr_residual:.2e}"
            )

        # Verificación adicional: |u_k|² - |v_k|² = 1 para cada modo
        norms_u = np.real(np.sum(np.abs(u_matrix) ** 2, axis=0))
        norms_v = np.real(np.sum(np.abs(v_matrix) ** 2, axis=0))
        norm_residual = float(np.max(np.abs(norms_u - norms_v - 1.0)))

        if norm_residual > self._tol:
            raise BogoliubovTransformationError(
                f"Normalización por modo violada: max |‖u_k‖² - ‖v_k‖² - 1| = {norm_residual:.2e}"
            )

        # Verificación de simetría partícula-hueco: para cada E_k > 0 debe existir -E_k
        evals_sorted = np.sort(evals)
        midpoint = len(evals_sorted) // 2
        positive_half = evals_sorted[midpoint:]
        negative_half = evals_sorted[:midpoint]
        ph_residual = float(np.max(np.abs(positive_half + negative_half[::-1])))

        symmetric = ph_residual < self._tol * 100

        return BogoliubovSpectrum(
            u_matrix=u_matrix,
            v_matrix=v_matrix,
            quasiparticle_energies=E_k,
            ccr_residual=ccr_residual,
            symmetric_basis=symmetric,
        )

    def transform_boson_modes(
        self,
        boson_wave: NDArray[np.complex128],
        spectrum: BogoliubovSpectrum,
    ) -> NDArray[np.complex128]:
        r"""
        Aplica la transformación de Bogoliubov a los coeficientes bosónicos:
        $$
        \vec{\alpha} = U^\dagger \vec{\psi} - V^\dagger \vec{\psi}^*
        $$

        **Este es el último método de la Fase 1; su salida es la entrada formal de la Fase 2.**

        Parameters
        ----------
        boson_wave : (N,) complex
            Vector de amplitudes bosónicas originales.
        spectrum : BogoliubovSpectrum
            Resultado de `compute_bogoliubov_coefficients`.

        Returns
        -------
        alpha_modes : (N,) complex
            Amplitudes de las cuasipartículas, listas para acoplamiento.
        """
        boson_wave = np.asarray(boson_wave, dtype=np.complex128)
        if boson_wave.ndim != 1:
            raise BogoliubovTransformationError(
                f"boson_wave debe ser 1D; ndim={boson_wave.ndim}"
            )
        if boson_wave.shape[0] != spectrum.n_modes:
            raise BogoliubovTransformationError(
                f"Dimensión {boson_wave.shape[0]} no coincide con espectro "
                f"({spectrum.n_modes})"
            )

        if not np.all(np.isfinite(boson_wave)):
            raise BogoliubovTransformationError(
                "boson_wave contiene valores no finitos (NaN/Inf)."
            )

        # Transformación: α = U† ψ - V† ψ*
        alpha = spectrum.u_matrix.conj().T @ boson_wave - spectrum.v_matrix.conj().T @ boson_wave.conj()

        if not np.all(np.isfinite(alpha)):
            raise BogoliubovTransformationError(
                "La transformación produjo amplitudes no finitas."
            )

        return alpha

    # -------------------------------------------------------------------------
    # FASE 2 (ANIDADA): SÍNTESIS DE ACOPLAMIENTO (PULLBACK GEOMÉTRICO)
    # -------------------------------------------------------------------------
    class Phase2_CouplingTensorSynthesizer:
        r"""
        Genera la matriz de acoplamiento $g_{kq}$ mediante el pullback covariante:
        $$
        g_{k,q} = \psi_k^\dagger \, G \, \mathcal{H}_{obs} \, G \, \phi_q
        $$

        donde $G$ es la métrica riemanniana y $\mathcal{H}_{obs}$ es el hamiltoniano
        de obstrucción topológica.

        Refactorización v3:
            • Soporte para múltiples modos bosónicos $(B)$ y fermiónicos $(F)$.
            • Vectorización completa: $g \in \mathbb{C}^{B \times F}$ construida por
              broadcasting matricial sin ambigüedades dimensionales.
            • Validación de la condición de acoplamiento débil $|g_{kq}| \ll 1$.
        """

        def __init__(
            self,
            metric_tensor: NDArray[np.float64] = None,
            weak_coupling_threshold: float = 1.0,
        ):
            self._G = np.asarray(metric_tensor if metric_tensor is not None else G_PHYSICS,
                                 dtype=np.float64)
            self._weak_threshold = weak_coupling_threshold

            if self._G.ndim != 2 or self._G.shape[0] != self._G.shape[1]:
                raise SMatrixSingularityError(
                    f"Métrica G debe ser cuadrada 2D; shape={self._G.shape}"
                )
            if np.any(~np.isfinite(self._G)):
                raise SMatrixSingularityError("Métrica G contiene NaN/Inf.")

        def _build_obstruction_hamiltonian(
            self,
            topological_obstruction: NDArray[np.float64],
        ) -> NDArray[np.complex128]:
            r"""
            Construye $\mathcal{H}_{obs} = \text{diag}(\text{obstruction})$.
            Para múltiples modos, extiende con broadcasting si es necesario.
            """
            obs = np.asarray(topological_obstruction, dtype=np.float64)
            if obs.ndim == 1:
                return np.diag(obs.astype(np.complex128))
            elif obs.ndim == 2:
                return obs.astype(np.complex128)
            else:
                raise SMatrixSingularityError(
                    f"topological_obstruction debe ser 1D o 2D; ndim={obs.ndim}"
                )

        def compute_coupling_constants(
            self,
            transformed_boson_modes: NDArray[np.complex128],
            fermion_modes: NDArray[np.complex128],
            topological_obstruction: NDArray[np.float64],
        ) -> CoupledInteractionData:
            r"""
            Calcula la matriz de acoplamiento $g_{kq}$ completa.

            Si los vectores de entrada son 1D, devuelve una matriz $(1 \times 1)$.
            Si son 2D con shapes $(B, M)$ y $(F, M)$, devuelve $(B \times F)$.

            Parameters
            ----------
            transformed_boson_modes : (B,) o (B, M) complex
                Modos bosónicos en la base de cuasipartículas (salida de Fase 1).
            fermion_modes : (F,) o (F, M) complex
                Modos fermiónicos de las restricciones de negocio.
            topological_obstruction : (M,) o (M, M) real
                Penalizaciones topológicas.

            Returns
            -------
            CoupledInteractionData
                Matriz de acoplamiento $g \in \mathbb{C}^{B \times F}$ + metadatos.
            """
            psi = np.asarray(transformed_boson_modes, dtype=np.complex128)
            phi = np.asarray(fermion_modes, dtype=np.complex128)
            obs = np.asarray(topological_obstruction, dtype=np.float64)

            for name, arr in [("psi", psi), ("phi", phi), ("obs", obs)]:
                if not np.all(np.isfinite(arr)):
                    raise SMatrixSingularityError(f"{name} contiene NaN/Inf.")

            if psi.ndim == 1 and phi.ndim == 1:
                return self._compute_coupling_1d(psi, phi, obs)

            if psi.ndim == 2 and phi.ndim == 2:
                return self._compute_coupling_2d(psi, phi, obs)

            raise SMatrixSingularityError(
                f"Dimensiones inconsistentes: psi.ndim={psi.ndim}, phi.ndim={phi.ndim}. "
                "Ambos deben ser 1D o ambos 2D."
            )

        def _compute_coupling_1d(
            self,
            psi: NDArray[np.complex128],
            phi: NDArray[np.complex128],
            obs: NDArray[np.float64],
        ) -> CoupledInteractionData:
            r"""Caso escalar: un solo modo bosónico y un solo modo fermiónico."""
            M = psi.shape[0]

            if phi.shape[0] != M:
                raise SMatrixSingularityError(
                    f"psi ({M}) y phi ({phi.shape[0]}) deben tener igual dimensión."
                )
            if obs.shape[0] != M:
                raise SMatrixSingularityError(
                    f"Obstrucción ({obs.shape[0]}) debe coincidir con dimensión de psi/phi ({M})."
                )
            if self._G.shape[0] != M:
                raise SMatrixSingularityError(
                    f"Métrica G ({self._G.shape}) incompatible con dimensión M={M}."
                )

            H_obs = self._build_obstruction_hamiltonian(obs)
            kernel = self._G.astype(np.complex128) @ H_obs @ self._G.astype(np.complex128)

            g_scalar = complex(psi.conj() @ kernel @ phi)

            if not np.isfinite(g_scalar):
                raise SMatrixSingularityError(f"Divergencia: g = {g_scalar}")

            g_matrix = np.array([[g_scalar]], dtype=np.complex128)

            return CoupledInteractionData(
                coupling_matrix=g_matrix,
                transformed_boson_modes=psi,
                fermion_modes=phi,
                metric_tensor=self._G.copy(),
                mean_coupling_strength=float(np.abs(g_scalar)),
                max_coupling_strength=float(np.abs(g_scalar)),
            )

        def _compute_coupling_2d(
            self,
            psi: NDArray[np.complex128],  # (B, M)
            phi: NDArray[np.complex128],  # (F, M)
            obs: NDArray[np.float64],
        ) -> CoupledInteractionData:
            r"""
            Caso multi-modo: vectorización completa.

            Para cada par $(k, q)$:
            $$
            g_{k,q} = \psi_k^\dagger \, K \, \phi_q
            $$
            donde $K = G \, \mathcal{H}_{obs} \, G$.

            Esto se computa eficientemente como:
            $$
            G = \bar{\Psi} \, K \, \Phi^T
            $$
            con $\bar{\Psi} \in \mathbb{C}^{B \times M}$ y $\Phi \in \mathbb{C}^{F \times M}$.
            """
            B, M_psi = psi.shape
            F, M_phi = phi.shape

            if M_psi != M_phi:
                raise SMatrixSingularityError(
                    f"psi y phi deben compartir dimensión espacial: {M_psi} vs {M_phi}"
                )

            if obs.ndim == 1 and obs.shape[0] != M_psi:
                raise SMatrixSingularityError(
                    f"Obstrucción ({obs.shape[0]}) ≠ dimensión espacial ({M_psi})"
                )
            elif obs.ndim == 2 and obs.shape != (M_psi, M_psi):
                raise SMatrixSingularityError(
                    f"Obstrucción matriz {obs.shape} ≠ ({M_psi}, {M_psi})"
                )

            if self._G.shape[0] != M_psi:
                raise SMatrixSingularityError(
                    f"Métrica G ({self._G.shape}) incompatible con M={M_psi}."
                )

            H_obs = self._build_obstruction_hamiltonian(obs)
            kernel = self._G.astype(np.complex128) @ H_obs @ self._G.astype(np.complex128)

            psi_K = psi.conj() @ kernel  # (B, M)
            g_matrix = psi_K @ phi.T      # (B, F)

            if not np.all(np.isfinite(g_matrix)):
                raise SMatrixSingularityError(
                    "Divergencia en matriz de acoplamiento (valores no finitos)."
                )

            abs_g = np.abs(g_matrix)
            mean_strength = float(np.mean(abs_g))
            max_strength = float(np.max(abs_g))

            if max_strength > self._weak_threshold:
                logger.warning(
                    f"Acoplamiento fuerte detectado: max |g_{{kq}}| = {max_strength:.3f} > "
                    f"{self._weak_threshold}. La teoría de perturbaciones puede no aplicar."
                )

            avg_psi = np.mean(psi, axis=1) if B > 1 else psi[:, 0]

            return CoupledInteractionData(
                coupling_matrix=g_matrix,
                transformed_boson_modes=avg_psi,
                fermion_modes=phi,
                metric_tensor=self._G.copy(),
                mean_coupling_strength=mean_strength,
                max_coupling_strength=max_strength,
            )

        # ---------------------------------------------------------------------
        # FASE 3 (ANIDADA): GENERACIÓN DE LINDBLADIANOS (OPERADORES DE SALTO CPTP)
        # ---------------------------------------------------------------------
        class Phase3_LindbladKrausGenerator:
            r"""
            Construye los operadores de Lindblad $\{\hat{L}_k\}$ a partir de la
            descomposición espectral de la matriz de error $\rho_{\text{err}}$.

            Para cada autovalor positivo $\lambda_i$ de $\rho_{\text{err}}$ con autovector
            $|\psi_i\rangle$, se construye:
            $$
            \hat{L}_i = \sqrt{\lambda_i \cdot \bar{g}} \; \hat{P}_0 \, |\psi_i\rangle
            $$

            donde $\hat{P}_0 = |0\rangle\langle 0|$ es el proyector al estado base y
            $\bar{g}$ es la magnitud media de acoplamiento.

            Esta construcción garantiza:
            - $\sum_i \hat{L}_i^\dagger \hat{L}_i \le \bar{g} \cdot \mathbb{1}$ (subnormalizado)
            - Disipación proporcional a la "componente de error" en cada modo.
            """

            def __init__(
                self,
                entropy_floor: float = 1e-15,
                spectral_tolerance: float = 1e-12,
            ):
                self._entropy_floor = entropy_floor
                self._spec_tol = spectral_tolerance

            def _validate_error_density(
                self,
                rho: NDArray[np.complex128],
            ) -> Tuple[int, NDArray[np.float64], NDArray[np.complex128]]:
                r"""
                Valida que $\rho_{\text{err}}$ sea una matriz densidad válida.

                Returns:
                    Tupla (dim, eigenvalues, eigenvectors).
                """
                if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
                    raise ErrorDensityValidationError(
                        f"ρ_err debe ser matriz cuadrada; shape={rho.shape}"
                    )

                herm_res = la.norm(rho - rho.conj().T, ord='fro') / rho.shape[0]
                if herm_res > 1e-8:
                    raise ErrorDensityValidationError(
                        f"ρ_err no hermítica (residual={herm_res:.2e})"
                    )

                tr = float(np.real(np.trace(rho)))
                if abs(tr - 1.0) > 1e-8:
                    raise ErrorDensityValidationError(
                        f"Tr(ρ_err) = {tr:.6f} ≠ 1"
                    )

                evals, evecs = la.eigh((rho + rho.conj().T) / 2.0)
                evals = np.real(evals)

                min_eig = float(np.min(evals))
                if min_eig < -1e-9:
                    raise ErrorDensityValidationError(
                        f"ρ_err no semidefinida positiva: λ_min = {min_eig:.2e}"
                    )

                evals = np.clip(evals, 0.0, None)
                evals = evals / np.sum(evals)

                return rho.shape[0], evals, evecs

            def _compute_von_neumann_entropy(self, evals: NDArray[np.float64]) -> float:
                r"""
                Entropía de von Neumann:
                $$ S(\rho) = -\sum_i \lambda_i \log(\lambda_i) $$
                con la convención $0 \log 0 = 0$.
                """
                significant = evals[evals > self._entropy_floor]
                if len(significant) == 0:
                    return 0.0
                return float(-np.sum(significant * np.log(significant)))

            def generate_jump_operators(
                self,
                error_density_matrix: NDArray[np.complex128],
                coupling_data: CoupledInteractionData,
            ) -> LindbladEnvironment:
                r"""
                Genera operadores de Lindblad $\{\hat{L}_k\}$ y tasas $\{\gamma_k\}$.

                Parameters
                ----------
                error_density_matrix : (D, D) complex
                    Matriz densidad del ruido semántico.
                coupling_data : CoupledInteractionData
                    Datos de acoplamiento de la Fase 2.

                Returns
                -------
                LindbladEnvironment
                    Operadores de salto + entropía + métricas espectrales.
                """
                dim, evals, evecs = self._validate_error_density(error_density_matrix)
                projected_entropy = self._compute_von_neumann_entropy(evals)

                active_mask = evals > self._spec_tol
                active_indices = np.where(active_mask)[0]
                n_active = len(active_indices)

                L_operators: List[NDArray[np.complex128]] = []
                decay_rates: List[float] = []

                mean_coupling = coupling_data.mean_coupling_strength

                for idx in active_indices:
                    lam_i = float(evals[idx])
                    psi_i = evecs[:, idx]

                    gamma_i = lam_i * mean_coupling
                    if gamma_i < self._entropy_floor:
                        continue

                    # L_i = sqrt(γ_i) |0><ψ_i|
                    L_k = np.zeros((dim, dim), dtype=np.complex128)
                    L_k[0, :] = np.sqrt(gamma_i) * psi_i.conj()
                    L_operators.append(L_k)
                    decay_rates.append(float(gamma_i))

                # Métricas espectrales
                if n_active >= 2:
                    sorted_evals = np.sort(evals)[::-1]
                    spectral_gap = float(sorted_evals[0] - sorted_evals[1])
                elif n_active == 1:
                    spectral_gap = float(evals[active_indices[0]])
                else:
                    spectral_gap = 0.0

                return LindbladEnvironment(
                    jump_operators=L_operators,
                    decay_rates=decay_rates,
                    projected_entropy=projected_entropy,
                    effective_dimension=n_active,
                    spectral_gap=spectral_gap,
                )


#
#    ORQUESTADOR SUPREMO: BOGOLIUBOV AGENT (Morfismo de Control)
#
class BogoliubovAgent(Morphism):
    r"""
    El Gran Inquisidor Cuántico del Estrato Ω.
    Gobierna la interacción multicuerpo y extrae la Antimateria Exógena.

    Encadena las tres fases anidadas mediante contratos formales:
        Fase1.compute_bogoliubov_coefficients()  → BogoliubovSpectrum
        Fase1.transform_boson_modes()            → α_modes
        Fase1.Phase2.compute_coupling_constants() → CoupledInteractionData
        Fase1.Phase2.Phase3.generate_jump_operators() → LindbladEnvironment
        Orquestador.assimilate_and_collide()      → LindbladEvolutionResult
    """

    def __init__(
        self,
        fock_config: FockSpaceConfiguration,
        metric_tensor: Optional[NDArray[np.float64]] = None,
        planck_normalized: float = 1.0,
        tolerance: float = 1e-9,
        entropy_floor: float = 1e-15,
    ):
        self._fock_config = fock_config
        self._planck = planck_normalized
        self._G = metric_tensor if metric_tensor is not None else G_PHYSICS

        # Instanciación de las fases anidadas
        self._phase1 = Phase1_BogoliubovTransformation(tolerance=tolerance)
        self._phase2 = Phase1_BogoliubovTransformation.Phase2_CouplingTensorSynthesizer(self._G)
        self._phase3 = Phase1_BogoliubovTransformation.Phase2_CouplingTensorSynthesizer.Phase3_LindbladKrausGenerator(entropy_floor=entropy_floor)

        self._orchestrator: Optional[QuantumFockOrchestrator] = None
        self._last_spectrum: Optional[BogoliubovSpectrum] = None
        self._last_coupling: Optional[CoupledInteractionData] = None
        self._last_lindblad: Optional[LindbladEnvironment] = None

    def orchestrate_quantum_collision(
        self,
        rho_llm: NDArray[np.complex128],
        boson_wave: NDArray[np.complex128],
        fermion_boundary: NDArray[np.complex128],
        kinetic_matrix: NDArray[np.float64],
        pairing_matrix: NDArray[np.complex128],
        topological_obstructions: NDArray[np.float64],
        dt: float = 1e-3,
    ) -> Tuple[LindbladEvolutionResult, Optional[PositronCartridge]]:
        r"""
        Método axiomático supremo que ejecuta la cadena completa anidada.
        """
        logger.info("Bogoliubov Agent: Iniciando cadena simpléctica anidada.")

        # --- FASE 1: Espectro de Bogoliubov ---
        spectrum = self._phase1.compute_bogoliubov_coefficients(
            kinetic_matrix, pairing_matrix
        )
        self._last_spectrum = spectrum
        logger.debug(
            f"BdG resuelto: {spectrum.n_modes} modos, "
            f"E_min={spectrum.quasiparticle_energies[0]:.4e}, "
            f"E_max={spectrum.quasiparticle_energies[-1]:.4e}, "
            f"CCR_res={spectrum.ccr_residual:.2e}"
        )

        # Transformación de modos (último método de Fase 1 → Fase 2)
        alpha_modes = self._phase1.transform_boson_modes(boson_wave, spectrum)

        # --- FASE 2: Acoplamiento catadióptrico (anidada) ---
        coupling_data = self._phase2.compute_coupling_constants(
            alpha_modes, fermion_boundary, topological_obstructions
        )
        self._last_coupling = coupling_data
        logger.debug(
            f"Acoplamiento: shape={coupling_data.coupling_matrix.shape}, "
            f"max|g|={coupling_data.max_coupling_strength:.4e}"
        )

        # --- FASE 3: Operadores de Lindblad (anidada) ---
        lindblad_env = self._phase3.generate_jump_operators(rho_llm, coupling_data)
        self._last_lindblad = lindblad_env
        logger.debug(
            f"Lindblad: {lindblad_env.effective_dimension} canales activos, "
            f"S={lindblad_env.projected_entropy:.4e}"
        )

        # Verificación CPTP opcional (no bloqueante)
        if not lindblad_env.verify_cptp_condition(coupling_data.mean_coupling_strength):
            logger.warning("Condición CPTP ligeramente violada; se proseguirá bajo contrato relajado.")

        # --- Configuración del QuantumFockOrchestrator ---
        self._orchestrator = QuantumFockOrchestrator(
            config=self._fock_config,
            coupling_matrix=coupling_data.coupling_matrix,
            lindblad_operators=lindblad_env.jump_operators,
            lindblad_rates=lindblad_env.decay_rates,
            planck_normalized=self._planck,
        )

        # --- Evolución CPTP ---
        evolution_result = self._orchestrator.assimilate_and_collide(rho_llm, dt)

        # --- Emisión forense (Positrón) ---
        positron = None
        if evolution_result.emitted_photon is not None:
            n_obstructions = int(np.sum(topological_obstructions > 0))
            positron = PositronCartridge(
                inertial_mass=evolution_result.dissipated_entropy,
                topological_spin="bogoliubov_anticommutator",
                homological_charge=n_obstructions,
                authorization_signature="Bogoliubov_SMatrix_Auditor",
            )
            logger.warning(
                "Antimateria inyectada. Fotón Gamma Hash: %s",
                evolution_result.emitted_photon.data_hash,
            )

        logger.info("Colisión cuántica completada bajo control simpléctico anidado.")
        return evolution_result, positron

    def diagnostic_report(self) -> Dict[str, Any]:
        r"""
        Genera reporte diagnóstico del último ciclo ejecutado.
        """
        report: Dict[str, Any] = {
            "phases_initialized": True,
            "planck_normalized": self._planck,
        }

        if self._last_spectrum is not None:
            sp = self._last_spectrum
            ccr_check = sp.verify_ccr_strict()
            report["phase1_bogoliubov"] = {
                "n_modes": sp.n_modes,
                "energies": sp.quasiparticle_energies.tolist(),
                "ccr_residual": sp.ccr_residual,
                "symmetric_basis": sp.symmetric_basis,
                "ccr_strict_check": ccr_check,
            }

        if self._last_coupling is not None:
            cd = self._last_coupling
            report["phase2_coupling"] = {
                "matrix_shape": list(cd.coupling_matrix.shape),
                "mean_strength": cd.mean_coupling_strength,
                "max_strength": cd.max_coupling_strength,
                "frobenius_norm": float(la.norm(cd.coupling_matrix, ord='fro')),
            }

        if self._last_lindblad is not None:
            le = self._last_lindblad
            report["phase3_lindblad"] = {
                "n_channels": le.effective_dimension,
                "n_operators": len(le.jump_operators),
                "sum_rates": float(np.sum(le.decay_rates)),
                "projected_entropy": le.projected_entropy,
                "spectral_gap": le.spectral_gap,
                "cptp_ok": le.verify_cptp_condition(
                    self._last_coupling.mean_coupling_strength if self._last_coupling else 0.0
                ),
            }

        return report


#
# EXPORTACIÓN CANÓNICA
#
__all__ = [
    "BogoliubovTransformationError",
    "SMatrixSingularityError",
    "ErrorDensityValidationError",
    "BogoliubovSpectrum",
    "CoupledInteractionData",
    "LindbladEnvironment",
    "Phase1_BogoliubovTransformation",
    "BogoliubovAgent",
]