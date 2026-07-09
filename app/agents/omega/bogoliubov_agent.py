# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo: Bogoliubov Agent (Gran Inquisidor Cuántico)                         |
| Ruta  : app/omega/bogoliubov_agent.py                                        |
| Versión: 3.0.0-Rigorous-Bogoliubov-Valatin-Symplectic                        |
+==============================================================================+

TRANSFORMACIÓN DE BOGOLIUBOV-VALATIN (GRUPO SIMPLÉCTICO Sp(2n,C)):
Aísla las cuasipartículas estables del ruido térmico del LLM.
\[ \begin{pmatrix} \hat{\alpha}_k \\ \hat{\alpha}_{-k}^\dagger \end{pmatrix} = \begin{pmatrix} u_k & v_k \\ v_k^* & u_k^* \end{pmatrix} \begin{pmatrix} \hat{b}_k \\ \hat{b}_{-k}^\dagger \end{pmatrix} \]
\[ |u_k|^2 - |v_k|^2 = 1 \]

ECUACIÓN MAESTRA DE LINDBLAD-KOSSAKOWSKI:
\[ \frac{d \rho_{MAC}}{dt} = -\frac{i}{\hbar} [H_{eff}, \rho_{MAC}] + \sum_{k} \gamma_k ( L_k \rho_{MAC} L_k^\dagger - \frac{1}{2} \{ L_k^\dagger L_k, \rho_{MAC} \} ) \]
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
        pass

    class Morphism:
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
# EXCEPCIONES SIMPL CTICAS Y CU NTICAS
#
class BogoliubovTransformationError(TopologicalInvariantError):
    """
    Detonada si la transformaci n viola las Relaciones de Conmutaci n Can nicas
    (CCR), implicando que el espacio de fase se ha desgarrado (p rdida de unitariedad).
    """
    pass


class SMatrixSingularityError(TopologicalInvariantError):
    """
    Detonada si el tensor de acoplamiento g_{k,q} diverge, indicando una resonancia
    infinita entre la alucinaci n de la IA y el presupuesto base.
    """
    pass


class ErrorDensityValidationError(TopologicalInvariantError):
    """
    Detonada si la matriz de error no satisface los axiomas de una matriz densidad
    v lida (hermiticidad, traza unitaria, positividad).
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
        """N mero de modos bos nicos del espacio original."""
        return self.u_matrix.shape[0]

    def verify_ccr_strict(self, tol: float = 1e-9) -> Dict[str, float]:
        r"""
        Verificación estricta post-diagonalización de las CCR.

        Returns:
            Dict con residuales:
                - 'commutation': $\|U^\dagger U - V^\dagger V - I\|_F$
                - 'particle_hole': $\max_k |E_k + E_k^*|/2$ (debe ser 0 para Hermiticidad)
                - 'normalization': $\max_k |\|u_k\|^2 - \|v_k\|^2 - 1|$
        """
        d = self.n_modes
        u, v = self.u_matrix, self.v_matrix

        comm_res = la.norm(u.conj().T @ u - v.conj().T @ v - np.eye(d), ord='fro')

        # Para cada modo k: |u_k|^2 - |v_k|^2 = 1
        norms_u = np.real(np.sum(np.abs(u) ** 2, axis=0))
        norms_v = np.real(np.sum(np.abs(v) ** 2, axis=0))
        norm_res = float(np.max(np.abs(norms_u - norms_v - 1.0)))

        # Realitud de las energ as
        energy_imag = float(np.max(np.abs(np.imag(self.quasiparticle_energies))))

        return {
            "commutation": float(comm_res),
            "normalization": norm_res,
            "energy_imag": energy_imag,
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


#
#                 FASE 1   DIAGONALIZACI N SIMPL CTICA
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
                f"  debe ser cuadrada; shape={pairing_gap_matrix.shape}"
            )

        dim_h = kinetic_energy_matrix.shape[0]
        dim_p = pairing_gap_matrix.shape[0]

        if dim_h != dim_p:
            raise BogoliubovTransformationError(
                f"Dimensiones incompatibles: H_k={dim_h},  ={dim_p}"
            )

        # H_k debe ser sim trica real (energ a cin tica herm tica real)
        if not np.allclose(kinetic_energy_matrix, kinetic_energy_matrix.T, atol=self._herm_tol):
            raise BogoliubovTransformationError(
                "H_k no es sim trica real; viola hermiticidad de BdG."
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

        # Construcci n y diagonalizaci n del Hamiltoniano BdG
        H_BdG = self._build_bdg_hamiltonian(kinetic_energy_matrix, pairing_gap_matrix)

        # Verificaci n de que H_BdG es herm tico (la estructura garantiza esto si   y H_k son v lidos)
        herm_res = la.norm(H_BdG - H_BdG.conj().T, ord='fro') / (2 * dim)
        if herm_res > self._herm_tol * 10:
            raise BogoliubovTransformationError(
                f"H_BdG no herm tico (residual={herm_res:.2e}); revisar   y H_k."
            )

        # Diagonalizaci n herm tica
        evals, evecs = la.eigh(H_BdG)

        #
        # Selecci n de energ as positivas: por la simetr a part cula-hueco
        # H_BdG tiene espectro { E_k}, as  que seleccionamos E_k > 0.
        #
        positive_mask = evals > self._tol
        E_k = evals[positive_mask]
        evecs_pos = evecs[:, positive_mask]

        n_positive = len(E_k)
        if n_positive == 0:
            raise BogoliubovTransformationError(
                "No se encontraron energ as positivas; posible gap trivial o systema apagado."
            )

        # Manejo de degeneraci n: si |n_positive - dim| > 0 pero cercano, intentar tolerancia relajada
        if n_positive < dim:
            # Reintentar con tolerancia m s laxa
            relaxed_tol = self._tol * 100
            positive_mask_relaxed = evals > relaxed_tol
            E_k_relaxed = evals[positive_mask_relaxed]

            if len(E_k_relaxed) == dim:
                logger.warning(
                    f"Degeneraci n part cula-hueco detectada; usando tolerancia relajada "
                    f"({relaxed_tol:.2e}). {n_positive}   {len(E_k_relaxed)} modos."
                )
                E_k = E_k_relaxed
                evecs_pos = evecs[:, positive_mask_relaxed]
                n_positive = dim

        if n_positive != dim:
            raise BogoliubovTransformationError(
                f"Estructura BdG inv lida: se esperaban {dim} modos positivos, "
                f"encontrados {n_positive}. Posible degeneraci n o mal condicionamiento."
            )

        # Ordenar por energ a ascendente
        order = np.argsort(E_k)
        E_k = E_k[order]
        evecs_pos = evecs_pos[:, order]

        # Extraer u y v
        u_matrix = evecs_pos[:dim, :]
        v_matrix = evecs_pos[dim:, :]

        # Verificaci n rigurosa de las CCR
        ccr_check = u_matrix.conj().T @ u_matrix - v_matrix.conj().T @ v_matrix
        ccr_residual = float(la.norm(ccr_check - np.eye(dim), ord='fro'))

        if ccr_residual > self._tol:
            raise BogoliubovTransformationError(
                f"Violaci n del grupo simpl ctico Sp(2N, ). "
                f" U U - V V - I _F = {ccr_residual:.2e}"
            )

        # Verificaci n adicional: |u_k|  - |v_k|  = 1 para cada modo
        norms_u = np.real(np.sum(np.abs(u_matrix) ** 2, axis=0))
        norms_v = np.real(np.sum(np.abs(v_matrix) ** 2, axis=0))
        norm_residual = float(np.max(np.abs(norms_u - norms_v - 1.0)))

        if norm_residual > self._tol:
            raise BogoliubovTransformationError(
                f"Normalizaci n por modo violada: max | u_k   -  v_k   - 1| = {norm_residual:.2e}"
            )

        # Verificaci n de simetr a part cula-hueco: para cada E_k > 0 debe existir -E_k
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

        Este es el último método de la Fase 1 → Fase 2.

        Parameters
        ----------
        boson_wave : (N,) complex
            Vector de amplitudes bosónicas originales.
        spectrum : BogoliubovSpectrum
            Resultado de `compute_bogoliubov_coefficients`.

        Returns
        -------
        alpha_modes : (N,) complex
            Amplitudes de las cuasipartículas.
        """
        boson_wave = np.asarray(boson_wave, dtype=np.complex128)
        if boson_wave.ndim != 1:
            raise BogoliubovTransformationError(
                f"boson_wave debe ser 1D; ndim={boson_wave.ndim}"
            )
        if boson_wave.shape[0] != spectrum.n_modes:
            raise BogoliubovTransformationError(
                f"Dimensi n {boson_wave.shape[0]} no coincide con espectro "
                f"({spectrum.n_modes})"
            )

        # Validaci n: las amplitudes bos nicas no deben ser divergentes
        if not np.all(np.isfinite(boson_wave)):
            raise BogoliubovTransformationError(
                "boson_wave contiene valores no finitos (NaN/Inf)."
            )

        # Transformaci n:   = U    - V   *
        alpha = spectrum.u_matrix.conj().T @ boson_wave - spectrum.v_matrix.conj().T @ boson_wave.conj()

        # Verificaci n post-transformaci n
        if not np.all(np.isfinite(alpha)):
            raise BogoliubovTransformationError(
                "La transformaci n produjo amplitudes no finitas."
            )

        return alpha


#
#          FASE 2   S NTESIS DE ACOPLAMIENTO (PULLBACK GEOM TRICO)
#
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
          broadcasting matricial sin ambigü dimensionales.
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
                f"M trica G debe ser cuadrada 2D; shape={self._G.shape}"
            )
        if np.any(~np.isfinite(self._G)):
            raise SMatrixSingularityError("M trica G contiene NaN/Inf.")

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
            # Ya es matriz
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
            Modos bosónicos en la base de cuasipartículas.
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

        # Validaci n de finitud
        for name, arr in [("psi", psi), ("phi", phi), ("obs", obs)]:
            if not np.all(np.isfinite(arr)):
                raise SMatrixSingularityError(f"{name} contiene NaN/Inf.")

        # Caso 1D: vectors  nicos   matriz 1 1
        if psi.ndim == 1 and phi.ndim == 1:
            return self._compute_coupling_1d(psi, phi, obs)

        # Caso 2D: m ltiples modos
        if psi.ndim == 2 and phi.ndim == 2:
            return self._compute_coupling_2d(psi, phi, obs)

        # Caso mixto no soportado
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
                f"psi ({M}) y phi ({phi.shape[0]}) deben tener igual dimensi n."
            )
        if obs.shape[0] != M:
            raise SMatrixSingularityError(
                f"Obstrucci n ({obs.shape[0]}) debe coincidir con dimensi n de psi/phi ({M})."
            )
        if self._G.shape[0] != M:
            raise SMatrixSingularityError(
                f"M trica G ({self._G.shape}) incompatible con dimensi n M={M}."
            )

        H_obs = self._build_obstruction_hamiltonian(obs)
        kernel = self._G.astype(np.complex128) @ H_obs @ self._G.astype(np.complex128)

        # Contracci n covariante: g =      kernel
        g_scalar = complex(psi.conj() @ kernel @ phi)

        if not np.isfinite(g_scalar):
            raise SMatrixSingularityError(
                f"Divergencia: g = {g_scalar}"
            )

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
                f"psi y phi deben compartir dimensi n espacial: {M_psi} vs {M_phi}"
            )

        # Obstrucci n puede ser vector (M,) o matriz (M, M)
        if obs.ndim == 1 and obs.shape[0] != M_psi:
            raise SMatrixSingularityError(
                f"Obstrucci n ({obs.shape[0]})   dimensi n espacial ({M_psi})"
            )
        elif obs.ndim == 2 and obs.shape != (M_psi, M_psi):
            raise SMatrixSingularityError(
                f"Obstrucci n matriz {obs.shape}   ({M_psi}, {M_psi})"
            )

        if self._G.shape[0] != M_psi:
            raise SMatrixSingularityError(
                f"M trica G ({self._G.shape}) incompatible con M={M_psi}."
            )

        H_obs = self._build_obstruction_hamiltonian(obs)
        kernel = self._G.astype(np.complex128) @ H_obs @ self._G.astype(np.complex128)

        # Vectorizaci n: G[B,F] =  [B,M]^  @ K[M,M] @  [F,M]^T
        # Equivalente: G = (    K    .T) con broadcasting
        psi_K = psi.conj() @ kernel  # (B, M)
        g_matrix = psi_K @ phi.T  # (B, F)

        # Validaci n
        if not np.all(np.isfinite(g_matrix)):
            raise SMatrixSingularityError(
                "Divergencia en matriz de acoplamiento (valores no finitos)."
            )

        abs_g = np.abs(g_matrix)
        mean_strength = float(np.mean(abs_g))
        max_strength = float(np.max(abs_g))

        # Advertencia si se viola la condici n de acoplamiento d bil
        if max_strength > self._weak_threshold:
            logger.warning(
                f"Acoplamiento fuerte detectado: max |g_{{kq}}| = {max_strength:.3f} > "
                f"{self._weak_threshold}. La teor a de perturbaciones puede no aplicar."
            )

        # Promedio de   por modo bos nico para entrega a Fase 3
        avg_psi = np.mean(psi, axis=1) if B > 1 else psi[:, 0]

        return CoupledInteractionData(
            coupling_matrix=g_matrix,
            transformed_boson_modes=avg_psi,
            fermion_modes=phi,
            metric_tensor=self._G.copy(),
            mean_coupling_strength=mean_strength,
            max_coupling_strength=max_strength,
        )


#
#         FASE 3   GENERACI N DE LINDBLADIANOS (OPERADORES DE SALTO CPTP)
#
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
    - $\sum_i \hat{L}_i^\dagger \hat{L}_i = \bar{g} \cdot \hat{P}_0$ (subnormalizado, $\leq \bar{g} \mathbb{1}$)
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
                f" _err debe ser matriz cuadrada; shape={rho.shape}"
            )

        # Hermiticidad
        herm_res = la.norm(rho - rho.conj().T, ord='fro') / rho.shape[0]
        if herm_res > 1e-8:
            raise ErrorDensityValidationError(
                f" _err no herm tica (residual={herm_res:.2e})"
            )

        # Traza unitaria
        tr = float(np.real(np.trace(rho)))
        if abs(tr - 1.0) > 1e-8:
            raise ErrorDensityValidationError(
                f"Tr( _err) = {tr:.6f}   1"
            )

        # Diagonalizaci n
        evals, evecs = la.eigh((rho + rho.conj().T) / 2.0)
        evals = np.real(evals)

        # Positividad
        min_eig = float(np.min(evals))
        if min_eig < -1e-9:
            raise ErrorDensityValidationError(
                f" _err no semidefinida positiva:  _min = {min_eig:.2e}"
            )

        # Clip valores negativos peque os
        evals = np.clip(evals, 0.0, None)
        # Renormalizar tras clip
        evals = evals / np.sum(evals)

        return rho.shape[0], evals, evecs

    def _compute_von_neumann_entropy(self, evals: NDArray[np.float64]) -> float:
        r"""
        Entropía de von Neumann:
        $$ S(\rho) = -\sum_i \lambda_i \log(\lambda_i) $$
        con la convención $0 \log 0 = 0$.
        """
        # Filtrar autovalores significativos
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
        # 1. Validar matriz de error
        dim, evals, evecs = self._validate_error_density(error_density_matrix)

        # 2. Entrop a proyectada
        projected_entropy = self._compute_von_neumann_entropy(evals)

        # 3. Filtrar modos activos
        active_mask = evals > self._spec_tol
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)

        # 4. Construir operadores de Lindblad
        L_operators: List[NDArray[np.complex128]] = []
        decay_rates: List[float] = []

        mean_coupling = coupling_data.mean_coupling_strength

        for idx in active_indices:
            lam_i = float(evals[idx])
            psi_i = evecs[:, idx]  # autovector

            # Tasa efectiva:  _i =  _i
            gamma_i = lam_i * mean_coupling

            if gamma_i < self._entropy_floor:
                continue

            # Operador de salto: L_i =   _i   |0   _i|
            L_k = np.zeros((dim, dim), dtype=np.complex128)
            L_k[0, :] = np.sqrt(gamma_i) * psi_i.conj()
            L_operators.append(L_k)
            decay_rates.append(float(gamma_i))

        # 5. M tricas espectrales
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

    Encadena las tres fases mediante contratos formales:
        Fase1.compute_bogoliubov_coefficients() → BogoliubovSpectrum
        Fase1.transform_boson_modes()          → α_modes (entrada a Fase 2)
        Fase2.compute_coupling_constants()      → CoupledInteractionData
        Fase3.generate_jump_operators()         → LindbladEnvironment
        Orquestador.assimilate_and_collide()    → LindbladEvolutionResult
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

        self._phase1 = Phase1_BogoliubovTransformation(tolerance=tolerance)
        self._phase2 = Phase2_CouplingTensorSynthesizer(self._G)
        self._phase3 = Phase3_LindbladKrausGenerator(entropy_floor=entropy_floor)

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
        Método axiomático supremo que ejecuta la cadena completa.
        """
        logger.info("Bogoliubov Agent: Iniciando cadena simpl ctica.")

        #    FASE 1: Espectro de Bogoliubov
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

        # Transformaci n de modos ( ltimo m todo de Fase 1   Fase 2)
        alpha_modes = self._phase1.transform_boson_modes(boson_wave, spectrum)

        #    FASE 2: Acoplamiento catadi ptrico
        coupling_data = self._phase2.compute_coupling_constants(
            alpha_modes, fermion_boundary, topological_obstructions
        )
        self._last_coupling = coupling_data
        logger.debug(
            f"Acoplamiento: shape={coupling_data.coupling_matrix.shape}, "
            f"max|g|={coupling_data.max_coupling_strength:.4e}"
        )

        #    FASE 3: Operadores de Lindblad
        lindblad_env = self._phase3.generate_jump_operators(rho_llm, coupling_data)
        self._last_lindblad = lindblad_env
        logger.debug(
            f"Lindblad: {lindblad_env.effective_dimension} canales activos, "
            f"S={lindblad_env.projected_entropy:.4e}"
        )

        #    Configuraci n del QuantumFockOrchestrator
        # Para acoplamiento 1 1, expandir a las dimensiones del orquestador
        # Si el orquestador requiere B F pero tenemos 1 1, usamos la matriz tal cual
        # (la dimensi n del orquestador viene de fock_config).
        self._orchestrator = QuantumFockOrchestrator(
            config=self._fock_config,
            coupling_matrix=coupling_data.coupling_matrix,
            lindblad_operators=lindblad_env.jump_operators,
            lindblad_rates=lindblad_env.decay_rates,
            planck_normalized=self._planck,
        )

        #    Evoluci n CPTP
        evolution_result = self._orchestrator.assimilate_and_collide(rho_llm, dt)

        #    Emisi n forense (Positr n)
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
                "Antimateria inyectada. Fot n Gamma Hash: %s",
                evolution_result.emitted_photon.data_hash,
            )

        logger.info("Colisi n cu ntica completada bajo control simpl ctico.")
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
            }

        return report


#
# EXPORTACI N CAN NICA
#
__all__ = [
    "BogoliubovTransformationError",
    "SMatrixSingularityError",
    "ErrorDensityValidationError",
    "BogoliubovSpectrum",
    "CoupledInteractionData",
    "LindbladEnvironment",
    "Phase1_BogoliubovTransformation",
    "Phase2_CouplingTensorSynthesizer",
    "Phase3_LindbladKrausGenerator",
    "BogoliubovAgent",
]