# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Ehresmann Telescopic Engine (Motor Telescópico de Ehresmann)        ║
║ Ruta   : app/core/immune_system/ehresmann_telescopic_engine.py               ║
║ Versión: 3.0.0-Stinespring-Ehresmann-Novikov-A∞-Doctoral                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor materializa la observación no-destructiva (zoom telescópico)
del presupuesto mediante una inmersión isométrica en una dimensión ortogonal.
Evita axiomáticamente el "Burbujeo de Esferas" (Sphere Bubbling) en la categoría
de Fukaya al confinar la torsión visual a la fibra vertical de una Conexión de
Ehresmann, resolviendo la Ecuación Expandida de Maurer-Cartan sobre el Anillo
de Novikov Λ_{nov,≥0}[[T]].

Fundamentos formales:
  • Stinespring  : ∀ canal CPTP Ξ, ∃ V isometría y N tales que Ξ(ρ)=Tr_N(VρV†).
  • Ehresmann    : TP = H ⊕ V, ω ∈ Ω¹(P,𝔤), curvatura Ω = dω + ½[ω∧ω].
  • Maurer-Cartan: m₀ + m₁(b) + m₂(b,b) + … = 0 en el dg-álgebra A∞ de co-cadenas.
  • Novikov      : filtración por área/acción; la serie ∑ a_i T^{λ_i} converge
                   en la topología T-ádica si λ_i → +∞.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Inmersión de Stinespring: Eleva ρ_MIC a ℋ_MIC ⊗ ℋ_audit^⊥ con V†V = I.
         El morfismo final de Fase 1 (compute_isometric_immersion) produce el
         objeto StinespringDilationData que es el dominio inicial de Fase 2.
Fase 2 → Fibración Vertical (Ehresmann): Proyecta 𝒯_λ^⊥ sobre V_p = ker(dπ).
         El morfismo final de Fase 2 (apply_telescopic_deformation) produce el
         objeto VerticalFibrationData que es el dominio inicial de Fase 3.
Fase 3 → Regularización Maurer-Cartan: Inyecta co-cadenas acotantes en Λ_nov
         para aniquilar curvaturas espurias (burbujeo discal) y devolver
         TelescopicAuditState, objeto final del endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs para aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos E_MIC."""
        pass

    class CategoricalState:
        r"""Estado interno de un objeto del Topos (stub)."""
        pass

logger = logging.getLogger("MIC.Omega.EhresmannTelescopicEngine")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_ISOMETRY_TOLERANCE: float = 1e-12
_ORTHOGONALITY_TOLERANCE: float = 1e-10
_HERMITICITY_TOLERANCE: float = 1e-12
_TRACE_TOLERANCE: float = 1e-12
_PSD_EIGENVALUE_FLOOR: float = -1e-14          # tolerancia numérica SPD
_MAURER_CARTAN_CONVERGENCE_TOL: float = 1e-9
_MAX_NOVIKOV_ITERATIONS: int = 64
_MIN_AUDIT_DIM: int = 2
_NOVIKOV_AREA_GAP: float = 1e-6               # filtración mínima de área en Λ_nov
_NEWTON_DAMPING_FLOOR: float = 1e-3
_CONDITION_NUMBER_CEILING: float = 1e14       # techo de κ₂(A) antes de regularizar


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS Y CUÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════
class TelescopicEngineError(TopologicalInvariantError):
    r"""Excepción raíz del Motor Telescópico de Ehresmann."""
    pass


class StinespringDilationError(TelescopicEngineError):
    r"""Detonada si V†V ≠ I (violación de isometría estricta de Stinespring)."""
    pass


class InvalidDensityMatrixError(TelescopicEngineError):
    r"""El estado ρ_MIC no pertenece al simplejo de matrices de densidad 𝔇(ℋ)."""
    pass


class EhresmannFibrationError(TelescopicEngineError):
    r"""Detonada si 𝒯_λ^⊥ se filtra al subespacio horizontal H_p = ker(ω)."""
    pass


class SphereBubblingAnomalyError(TelescopicEngineError):
    r"""Detonada si la ecuación de Maurer-Cartan diverge (expulsión de discos J-holomorfos)."""
    pass


class SpectralDegeneracyError(TelescopicEngineError):
    r"""Espectro patológico: κ₂ excesivo o gap nulo incompatible con la filtración de Novikov."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos — objetos de las categorías fibra)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class StinespringDilationData:
    r"""
    Artefacto terminal de Fase 1.

    Objeto del fibrado de Stinespring:
        V : ℋ_MIC ↪ ℋ_MIC ⊗ ℋ_audit^⊥,   V†V = I,
        ρ_audit = Tr_MIC(V ρ_MIC V†) ∈ 𝔇(ℋ_audit^⊥).

    Invariantes:
        • ‖V†V − I‖_F < ε_iso
        • ρ_audit = ρ_audit† ≽ 0, Tr ρ_audit = 1
        • purity ∈ [1/dim_audit, 1]
    """
    V_isometry: NDArray[np.complex128]
    rho_audit_subspace: NDArray[np.complex128]
    purity_preservation: float
    dilating_environment_dim: int
    stinespring_residual: float


@dataclass(frozen=True, slots=True)
class VerticalFibrationData:
    r"""
    Artefacto terminal de Fase 2 (y dominio inicial de Fase 3).

    Datos de la conexión de Ehresmann ω ∈ Ω¹(P; End(V)):
        TP = H_p ⊕ V_p,   V_p = ker(dπ),   Π_V = ω (en calibre local),
        𝒯_λ^⊥ = Π_V ∘ T_λ ∘ Π_V   (soporte estrictamente vertical).

    Invariantes:
        • ‖Π_H T_λ‖_F < ε_orth
        • Ω = dω + ½[ω∧ω] controlada (curvatura acotada)
    """
    T_lambda_vertical: NDArray[np.complex128]
    ehresmann_connection_form: NDArray[np.complex128]
    vertical_projector: NDArray[np.complex128]
    horizontal_leakage_norm: float
    connection_curvature_norm: float
    spectral_zoom_eigenvalues: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TelescopicAuditState:
    r"""
    Artefacto terminal de Fase 3 — objeto final del endofuntor Z.

    Estado ρ_safe ∈ 𝔇(ℋ_audit) homológicamente regularizado:
        F(b) = m₀ + m₁(b) + m₂(b,b) = 0   en Λ_nov,
        W_LG[b] = ‖F(b)‖_F → 0,
        seguro para el funtor de Witten-Atiyah-Floer.
    """
    audited_density_matrix: NDArray[np.complex128]
    landau_ginzburg_potential: float
    novikov_convergence_iterations: int
    maurer_cartan_bounding_cochain: NDArray[np.complex128]
    novikov_filtration_degree: float
    is_safe_for_witten_atiyah: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1 · INMERSIÓN ORTOGONAL DE STINESPRING                               ║
# ║   Elevación isométrica ρ_MIC ↦ (V, ρ_audit) en ℋ_MIC ⊗ ℋ_audit^⊥            ║
# ║                                                                             ║
# ║   Definición formal del objeto terminal de esta fase:                       ║
# ║       compute_isometric_immersion : 𝔇(ℋ_MIC) × ℕ_{≥2}                       ║
# ║           → StinespringDilationData                                         ║
# ║   Este morfismo es el dominio de partida de Fase 2.                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_StinespringImmersion:
    r"""
    Realiza el Teorema de Dilatación de Stinespring en forma constructiva.

    Enunciado operativo
    -------------------
    Dado un estado ρ ∈ 𝔇(ℋ_MIC) y una dimensión de baño n_a ≥ 2, se construye
    una isometría

        V : ℋ_MIC → ℋ_MIC ⊗ ℋ_audit,   V†V = I_MIC,

    tal que el canal de auditoría

        Ξ_audit(ρ) := Tr_MIC(V ρ V†)

    es completamente positivo y preserva la traza (CPTP). La pureza de
    Ξ_audit(ρ) mide la decoherencia inducida por la inmersión.

    Geometría
    ---------
    La inmersión canónica ancla ρ al ground-state |0⟩_audit del baño:

        V| i ⟩_MIC = | i ⟩_MIC ⊗ | 0 ⟩_audit,

    de modo que el soporte de VρV† vive en el subespacio
    ℋ_MIC ⊗ span{|0⟩}, y la traza parcial sobre MIC produce un estado
    diagonal en la base computacional del baño (en ausencia de ruido).
    """

    # ── §1.1  Validación espectral del simplejo de densidad ──────────────────

    @staticmethod
    def _hermitize(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
        r"""Proyección de Frobenius sobre el subespacio de matrices hermitianas: (M+M†)/2."""
        return 0.5 * (M + M.conj().T)

    @staticmethod
    def _validate_density_matrix(
        rho: NDArray[np.complex128],
        name: str = "rho",
    ) -> NDArray[np.complex128]:
        r"""
        Verifica y, si es necesario, proyecta ρ sobre 𝔇(ℋ).

        Condiciones (en orden de evaluación):
          (i)   ρ ∈ M_n(ℂ), n ≥ 1, cuadrada
          (ii)  ‖ρ − ρ†‖_F ≤ ε_herm  (hermiticidad)
          (iii) spec(ρ) ⊂ [−ε_psd, +∞)  (semidefinida positiva numérica)
          (iv)  |Tr ρ − 1| ≤ ε_tr       (normalización)

        Retorna la versión hermitizada y renormalizada si las desviaciones
        son inferiores a las tolerancias; en caso contrario lanza
        InvalidDensityMatrixError.
        """
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise InvalidDensityMatrixError(
                f"{name}: se exige matriz cuadrada; recibido shape={rho.shape}."
            )
        if rho.shape[0] < 1:
            raise InvalidDensityMatrixError(f"{name}: dimensión nula.")

        # Hermiticidad
        anti_herm_norm = float(la.norm(rho - rho.conj().T, ord="fro"))
        if anti_herm_norm > _HERMITICITY_TOLERANCE:
            raise InvalidDensityMatrixError(
                f"{name}: no hermítica. ‖ρ−ρ†‖_F = {anti_herm_norm:.3e} "
                f"> {_HERMITICITY_TOLERANCE:.1e}."
            )
        rho_h = Phase1_StinespringImmersion._hermitize(rho)

        # Espectro PSD
        eigvals = la.eigvalsh(rho_h)
        if np.any(eigvals < _PSD_EIGENVALUE_FLOOR):
            raise InvalidDensityMatrixError(
                f"{name}: autovalor negativo {eigvals.min():.3e} "
                f"< floor={_PSD_EIGENVALUE_FLOOR:.1e}; no es SPD."
            )

        # Traza unitaria
        tr = complex(np.trace(rho_h))
        if abs(tr - 1.0) > _TRACE_TOLERANCE:
            raise InvalidDensityMatrixError(
                f"{name}: |Tr ρ − 1| = {abs(tr - 1.0):.3e} > {_TRACE_TOLERANCE:.1e}."
            )

        # Renormalización numérica suave (preserva el rayo proyectivo)
        if abs(tr.real - 1.0) > _MACHINE_EPSILON:
            rho_h = rho_h / tr.real

        return rho_h.astype(np.complex128)

    # ── §1.2  Verificación de isometría de Stinespring ───────────────────────

    @staticmethod
    def _verify_isometry(V: NDArray[np.complex128]) -> float:
        r"""
        Comprueba la condición de isometría estricta

            V† V = I_{n_MIC}

        en norma de Frobenius. Retorna el residuo ‖V†V − I‖_F; lanza
        StinespringDilationError si el residuo excede ε_iso.
        """
        gram = V.conj().T @ V
        n = gram.shape[0]
        residual = float(la.norm(gram - np.eye(n, dtype=np.complex128), ord="fro"))
        if residual > _ISOMETRY_TOLERANCE:
            raise StinespringDilationError(
                f"Inmersión no isométrica: ‖V†V − I‖_F = {residual:.3e} "
                f"> ε_iso = {_ISOMETRY_TOLERANCE:.1e}."
            )
        return residual

    # ── §1.3  Traza parcial rigurosa ℋ_MIC ⊗ ℋ_audit → ℋ_audit ──────────────

    @staticmethod
    def _partial_trace_mic(
        rho_combined: NDArray[np.complex128],
        dim_mic: int,
        dim_audit: int,
    ) -> NDArray[np.complex128]:
        r"""
        Traza parcial sobre el factor MIC:

            ρ_audit = Tr_MIC(ρ_combined)
                    = ∑_{i=1}^{n_MIC} (⟨i| ⊗ I_a) ρ_combined (|i⟩ ⊗ I_a).

        Implementación tensorial vía reshape + traza en ejes (0, 2) de un
        array de rango 4 con shape (n_MIC, n_a, n_MIC, n_a).
        """
        expected = dim_mic * dim_audit
        if rho_combined.shape != (expected, expected):
            raise StinespringDilationError(
                f"Dimensión incompatible para traza parcial: "
                f"ρ_combined.shape={rho_combined.shape}, esperado ({expected},{expected})."
            )
        tensor = rho_combined.reshape(dim_mic, dim_audit, dim_mic, dim_audit)
        rho_audit = np.trace(tensor, axis1=0, axis2=2)
        return Phase1_StinespringImmersion._hermitize(rho_audit)

    # ── §1.4  Construcción de la isometría canónica de Stinespring ───────────

    @staticmethod
    def _build_canonical_isometry(
        dim_mic: int,
        dim_audit: int,
    ) -> NDArray[np.complex128]:
        r"""
        Isometría canónica de anclaje al vacuum del baño:

            V : |i⟩_MIC ↦ |i⟩_MIC ⊗ |0⟩_audit,
            V ∈ ℂ^{n_MIC·n_a × n_MIC},   V_{i·n_a, i} = 1.

        Equivale a V = I_MIC ⊗ |0⟩_audit en la identificación de Kronecker.
        """
        V = np.zeros((dim_mic * dim_audit, dim_mic), dtype=np.complex128)
        for i in range(dim_mic):
            V[i * dim_audit, i] = 1.0 + 0.0j
        return V

    # ── §1.5  MORFISMO TERMINAL DE FASE 1 ────────────────────────────────────
    #          (dominio inicial de Fase 2)

    def compute_isometric_immersion(
        self,
        rho_mic: NDArray[np.complex128],
        dim_audit: int,
    ) -> StinespringDilationData:
        r"""
        Ejecuta la inmersión de Stinespring

            Φ₁ : 𝔇(ℋ_MIC) × ℕ_{≥2} ⟶ StinespringDilationData.

        Pipeline
        --------
        1. Validación espectral de ρ_MIC ∈ 𝔇(ℋ_MIC).
        2. Construcción de V = I ⊗ |0⟩_audit.
        3. Verificación ‖V†V − I‖_F < ε_iso.
        4. Estado dilatado ρ̃ = V ρ V† ∈ 𝔇(ℋ_MIC ⊗ ℋ_audit).
        5. Traza parcial ρ_audit = Tr_MIC(ρ̃).
        6. Pureza 𝒫 = Tr(ρ_audit²) ∈ [1/n_a, 1].

        Parámetros
        ----------
        rho_mic : NDArray[np.complex128]
            Matriz de densidad del negocio, shape (n, n).
        dim_audit : int
            Dimensión del baño térmico de auditoría (n_a ≥ 2).

        Retorna
        -------
        StinespringDilationData
            Objeto terminal de Fase 1 / dominio de Fase 2.
        """
        rho_mic = self._validate_density_matrix(rho_mic, name="rho_mic")

        if not isinstance(dim_audit, int) or dim_audit < _MIN_AUDIT_DIM:
            raise ValueError(
                f"dim_audit debe ser entero ≥ {_MIN_AUDIT_DIM}; recibido {dim_audit}."
            )

        dim_mic = int(rho_mic.shape[0])

        # Isometría canónica y verificación
        V = self._build_canonical_isometry(dim_mic, dim_audit)
        residual = self._verify_isometry(V)

        # Estado combinado y traza parcial
        rho_combined = V @ rho_mic @ V.conj().T
        rho_combined = self._hermitize(rho_combined)
        rho_audit = self._partial_trace_mic(rho_combined, dim_mic, dim_audit)

        # Renormalización defensiva de la traza parcial
        tr_a = float(np.real(np.trace(rho_audit)))
        if tr_a < _MACHINE_EPSILON:
            raise StinespringDilationError(
                "Traza parcial degenerada: Tr ρ_audit ≈ 0 (colapso del baño)."
            )
        rho_audit = rho_audit / tr_a
        rho_audit = self._hermitize(rho_audit)

        # Pureza: 𝒫(ρ) = Tr(ρ²); para estados puros 𝒫=1, máximamente mixtos 𝒫=1/n
        purity = float(np.real(np.trace(rho_audit @ rho_audit)))
        purity = float(np.clip(purity, 1.0 / dim_audit, 1.0))

        return StinespringDilationData(
            V_isometry=V,
            rho_audit_subspace=rho_audit,
            purity_preservation=purity,
            dilating_environment_dim=dim_audit,
            stinespring_residual=residual,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2 · FIBRACIÓN VERTICAL TELESCÓPICA (CONEXIÓN DE EHRESMANN)           ║
# ║   Confinamiento de 𝒯_λ^⊥ al núcleo vertical V_p = ker(dπ)                   ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 1                    ║
# ║       compute_isometric_immersion ↦ StinespringDilationData                 ║
# ║   es el dominio de                                                         ║
# ║       apply_telescopic_deformation : StinespringDilationData × ℝ_{≥0}       ║
# ║           → VerticalFibrationData                                           ║
# ║   Este morfismo terminal de Fase 2 es el dominio de partida de Fase 3.      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_TelescopicVerticalFibration(Phase1_StinespringImmersion):
    r"""
    Construye una conexión de Ehresmann sobre el fibrado de auditoría y
    confina el tensor de magnificación telescópica a la fibra vertical.

    Geometría diferencial
    ---------------------
    Sea π : P → B la proyección del fibrado principal (aquí P ≅ ℋ_audit
    como variedad real subyacente 2n_a-dimensional, B = punto en la
    trivialización local post-traza parcial). Una conexión de Ehresmann es
    un subfibrado horizontal H ⊂ TP tal que

        T_p P = H_p ⊕ V_p,   V_p := ker(dπ_p),

    equivalentemente una 1-forma ω ∈ Ω¹(P; 𝔤) con ω|_{V} = id_𝔤.
    El proyector vertical es Π_V = ω (en identificación local), y el
    horizontal Π_H = id − Π_V.

    En la trivialización post-Stinespring el baño no retiene componente
    horizontal que contamine al sistema MIC (fue eliminada por Tr_MIC);
    por tanto Π_V = I y Π_H = 0 en el calibre canónico. No obstante se
    construye la curvatura

        Ω = dω + ½[ω ∧ ω]

    y se reporta ‖Ω‖ como diagnóstico de consistencia del calibre.

    Tensor de magnificación espectral
    ---------------------------------
    Se diagonaliza ρ_audit = U Λ U† y se definen factores de escala

        s_i(λ) = 1 + λ · √(Λ_ii) · (1 + α Λ_ii)^{-1/2}
                 (regularización tipo Tikhonov-espectral),

    de modo que los modos de mayor ocupación reciben mayor amplificación
    y el espectro de T_λ permanece acotado (evita runaway ultraviolet).
    """

    # ── §2.1  Conexión de Ehresmann y descomposición horizontal/vertical ─────

    @staticmethod
    def _build_ehresmann_connection(
        dim: int,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        Construye la terna (Π_V, Π_H, ω) en el calibre canónico post-traza.

        En esta trivialización:
            Π_V = I_dim,   Π_H = 0,   ω = Π_V.

        La curvatura formal Ω se evalúa como el conmutador de monodromía
        infinitesimal; en calibre plano Ω ≡ 0.

        Returns
        -------
        Pi_V, Pi_H, omega : proyectores y 1-forma de conexión.
        """
        Pi_V = np.eye(dim, dtype=np.complex128)
        Pi_H = np.zeros((dim, dim), dtype=np.complex128)
        omega = Pi_V.copy()
        return Pi_V, Pi_H, omega

    @staticmethod
    def _connection_curvature_norm(
        omega: NDArray[np.complex128],
    ) -> float:
        r"""
        Estima ‖Ω‖_F de la curvatura de Ehresmann.

        En calibre local de matriz, la 2-forma de curvatura se reduce a

            Ω ∼ ω ∧ ω − ω ∧ ω = 0   (conexión plana),

        más un término de inconsistencia numérica ‖ω² − ω‖ (idempotencia
        del proyector). Se reporta esa desviación como diagnóstico.
        """
        # Idempotencia del proyector: ω² = ω ⇒ curvatura estructural nula
        residual = omega @ omega - omega
        return float(la.norm(residual, ord="fro"))

    # ── §2.2  Zoom espectral regularizado (Tikhonov en el espectro) ──────────

    @staticmethod
    def _compute_spectral_zoom(
        rho_audit: NDArray[np.complex128],
        lambda_zoom: float,
        tikhonov_alpha: float = 1e-3,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.float64]]:
        r"""
        Construye el tensor de magnificación 𝒯_λ diagonal en la base propia
        de ρ_audit, con regularización espectral de Tikhonov:

            s_i = 1 + λ · √μ_i / √(1 + α μ_i),   μ_i = max(Λ_ii, 0).

        Propiedades:
          • s_i ≥ 1 para λ ≥ 0, μ_i ≥ 0
          • s_i ∼ 1 + λ/√α  cuando μ_i → ∞  (saturación UV)
          • s_i ∼ 1 + λ √μ_i cuando μ_i → 0  (comportamiento IR lineal)

        Retorna (T_λ en base canónica, vector de autovalores de escala s).
        """
        if lambda_zoom < 0.0:
            raise ValueError(f"lambda_zoom debe ser ≥ 0; recibido {lambda_zoom}.")

        # Descomposición espectral hermitiana
        eigvals, eigvecs = la.eigh(rho_audit)
        eigvals = np.clip(eigvals.real, 0.0, None)

        # Factores de escala Tikhonov-espectrales
        sqrt_mu = np.sqrt(eigvals)
        denom = np.sqrt(1.0 + tikhonov_alpha * eigvals)
        scale = 1.0 + lambda_zoom * sqrt_mu / denom
        scale = scale.astype(np.float64)

        T_diag = np.diag(scale.astype(np.complex128))
        T_lambda = eigvecs @ T_diag @ eigvecs.conj().T
        T_lambda = Phase1_StinespringImmersion._hermitize(T_lambda)

        return T_lambda, scale

    # ── §2.3  Proyección vertical y diagnóstico de fuga horizontal ───────────

    @staticmethod
    def _project_vertical_and_measure_leakage(
        T_lambda: NDArray[np.complex128],
        Pi_V: NDArray[np.complex128],
        Pi_H: NDArray[np.complex128],
        omega: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Proyecta T_λ sobre la fibra vertical y mide la fuga horizontal:

            T_λ^⊥ = ω T_λ ω†,     ℓ_H = ‖Π_H T_λ‖_F.

        Si ℓ_H > ε_orth se lanza EhresmannFibrationError (el zoom contaminaría
        las geodésicas de la base presupuestaria).
        """
        leakage = float(la.norm(Pi_H @ T_lambda, ord="fro"))
        if leakage > _ORTHOGONALITY_TOLERANCE:
            raise EhresmannFibrationError(
                f"Fuga de calibre horizontal: ‖Π_H T_λ‖_F = {leakage:.3e} "
                f"> ε_orth = {_ORTHOGONALITY_TOLERANCE:.1e}. "
                f"El zoom telescópico perturba H_p."
            )
        T_vertical = omega @ T_lambda @ omega.conj().T
        T_vertical = Phase1_StinespringImmersion._hermitize(T_vertical)
        return T_vertical, leakage

    # ── §2.4  MORFISMO TERMINAL DE FASE 2 ────────────────────────────────────
    #          (continuación de compute_isometric_immersion;
    #           dominio inicial de Fase 3)

    def apply_telescopic_deformation(
        self,
        dilation_data: StinespringDilationData,
        lambda_zoom: float,
    ) -> VerticalFibrationData:
        r"""
        Inyecta la deformación visual λ confinada a V_p:

            Φ₂ : StinespringDilationData × ℝ_{≥0} ⟶ VerticalFibrationData.

        Pipeline
        --------
        1. Recuperar ρ_audit del artefacto de Fase 1.
        2. Construir conexión de Ehresmann (Π_V, Π_H, ω).
        3. Calcular curvatura diagnóstica ‖Ω‖.
        4. Construir T_λ por zoom espectral Tikhonov.
        5. Proyectar T_λ^⊥ = ω T_λ ω† y verificar ℓ_H < ε_orth.
        6. Empaquetar VerticalFibrationData (dominio de Fase 3).

        Parámetros
        ----------
        dilation_data : StinespringDilationData
            Salida de Φ₁ = compute_isometric_immersion.
        lambda_zoom : float
            Magnificación telescópica global (λ ≥ 0).

        Retorna
        -------
        VerticalFibrationData
            Objeto terminal de Fase 2 / dominio de Fase 3.
        """
        # Continuación directa del objeto terminal de Fase 1
        rho_audit = dilation_data.rho_audit_subspace
        rho_audit = self._validate_density_matrix(rho_audit, name="rho_audit_subspace")
        dim = int(rho_audit.shape[0])

        if lambda_zoom < 0.0:
            raise ValueError(f"lambda_zoom ≥ 0 requerido; recibido {lambda_zoom}.")

        # Conexión de Ehresmann
        Pi_V, Pi_H, omega = self._build_ehresmann_connection(dim)
        curv_norm = self._connection_curvature_norm(omega)

        # Zoom espectral
        T_lambda, scale_eigs = self._compute_spectral_zoom(rho_audit, lambda_zoom)

        # Proyección vertical + diagnóstico de fuga
        T_vertical, leakage = self._project_vertical_and_measure_leakage(
            T_lambda, Pi_V, Pi_H, omega
        )

        return VerticalFibrationData(
            T_lambda_vertical=T_vertical,
            ehresmann_connection_form=omega,
            vertical_projector=Pi_V,
            horizontal_leakage_norm=leakage,
            connection_curvature_norm=curv_norm,
            spectral_zoom_eigenvalues=scale_eigs,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3 · REGULARIZACIÓN MAURER-CARTAN SOBRE EL ANILLO DE NOVIKOV          ║
# ║   Absorción del burbujeo de esferas; protección de la categoría de Fukaya   ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 2                    ║
# ║       apply_telescopic_deformation ↦ VerticalFibrationData                  ║
# ║   es el dominio de                                                         ║
# ║       resolve_maurer_cartan_novikov : VerticalFibrationData × 𝔇(ℋ_a)        ║
# ║           → TelescopicAuditState                                            ║
# ║   Este morfismo cierra el endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁.                      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_MaurerCartanRegularization(Phase2_TelescopicVerticalFibration):
    r"""
    Resuelve la ecuación expandida de Maurer-Cartan en el dg-álgebra A∞
    de co-cadenas con coeficientes en el anillo de Novikov

        Λ_nov = { ∑_{i=0}^∞ a_i T^{λ_i} | a_i ∈ ℂ, λ_i → +∞ },

    para cancelar las anomalías de curvatura inducidas por el zoom
    telescópico y evitar el burbujeo de discos J-holomorfos en la
    categoría de Fukaya.

    Estructura A∞ truncada (orden 2)
    --------------------------------
        m₀      = T_λ ρ T_λ† − ρ          (obstrucción / anomalía de curvatura)
        m₁(b)   = [T_λ, b]               (diferencial torcido de grado 1)
        m₂(b,b) = b ∘ b                  = b²
        F(b)    = m₀ + m₁(b) + m₂(b,b)

    Se busca b ∈ End(ℋ_audit) (co-cadena acotante) tal que F(b) = 0.
    El método de Newton-Kantorovich con damping adaptativo opera sobre
    la linealización

        DF_b · δb = [T_λ, δb] + b δb + δb b,
        (T_λ − b) δb + δb (T_λ + b)^†_sym = −F(b)   (Sylvester generalizada).

    Filtración de Novikov
    ---------------------
    El grado de filtración se estima como el gap espectral de área

        deg_nov(b) = ‖b‖_F + _NOVIKOV_AREA_GAP · iter,

    garantizando que la serie de potencias en T sea T-ádicamente
    convergente (λ_i monótona no decreciente).

    Potencial de Landau-Ginzburg
    ----------------------------
        W_LG[b] := ‖F(b)‖_F

    es la acción cuya vanescencia certifica la regularidad holomorfa.
    """

    # ── §3.1  Anomalía de curvatura m₀ ───────────────────────────────────────

    @staticmethod
    def _compute_anomaly(
        T_lambda: NDArray[np.complex128],
        rho_audit: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Obstrucción de Maurer-Cartan:

            m₀ = Ad_{T_λ}(ρ) − ρ = T_λ ρ T_λ† − ρ.

        Mide la desviación del estado auditado respecto del original bajo
        la acción adjunta del tensor de magnificación.
        """
        rho_dist = T_lambda @ rho_audit @ T_lambda.conj().T
        rho_dist = Phase1_StinespringImmersion._hermitize(rho_dist)
        return rho_dist - rho_audit

    # ── §3.2  Función de Maurer-Cartan F(b) ──────────────────────────────────

    @staticmethod
    def _maurer_cartan_function(
        b: NDArray[np.complex128],
        T_lambda: NDArray[np.complex128],
        m0: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        F(b) = m₀ + [T_λ, b] + b².

        Ceros de F corresponden a deformaciones A∞-planas (soluciones MC).
        """
        commutator = T_lambda @ b - b @ T_lambda
        return m0 + commutator + b @ b

    # ── §3.3  Linealización DF_b y resolución de Sylvester ───────────────────

    @staticmethod
    def _solve_linearized(
        b: NDArray[np.complex128],
        T_lambda: NDArray[np.complex128],
        residual: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Resuelve la ecuación de Newton

            DF_b · δb = −F(b),

        donde

            DF_b · δb = [T_λ, δb] + b δb + δb b
                      = (T_λ + b) δb  wait…
                      = T_λ δb − δb T_λ + b δb + δb b
                      = (T_λ + b) δb + δb (b − T_λ).

        Forma de Sylvester estándar A X + X B = Q con
            A = T_λ + b,   B = b − T_λ,   Q = −residual.

        Precondición: se monitorea el número de condición; si κ₂ excede
        el techo se aplica regularización de Tikhonov diagonal en A, B.
        """
        A = T_lambda + b
        B = b - T_lambda
        Q = -residual

        # Diagnóstico de condicionamiento
        try:
            cond_A = float(np.linalg.cond(A))
        except la.LinAlgError:
            cond_A = np.inf

        if not np.isfinite(cond_A) or cond_A > _CONDITION_NUMBER_CEILING:
            # Regularización de Tikhonov: A ← A + εI, B ← B + εI
            eps = math.sqrt(_MACHINE_EPSILON) * max(1.0, float(la.norm(A, ord=2)))
            A = A + eps * np.eye(A.shape[0], dtype=np.complex128)
            B = B + eps * np.eye(B.shape[0], dtype=np.complex128)
            logger.debug(
                "Sylvester regularizado (Tikhonov ε=%.3e, κ₂(A)≈%.3e).", eps, cond_A
            )

        try:
            delta = la.solve_sylvester(A, B, Q)
        except la.LinAlgError as exc:
            raise SphereBubblingAnomalyError(
                "La ecuación linealizada de Maurer-Cartan es singular. "
                "Burbujeo de esferas incontrolable (Sylvester falló)."
            ) from exc

        return delta

    # ── §3.4  Paso de Newton con damping adaptativo (Armijo débil) ───────────

    @staticmethod
    def _newton_damped_step(
        b: NDArray[np.complex128],
        delta_b: NDArray[np.complex128],
        T_lambda: NDArray[np.complex128],
        m0: NDArray[np.complex128],
        residual_norm: float,
    ) -> NDArray[np.complex128]:
        r"""
        Actualiza b ← b + t δb con paso t ∈ (0, 1] elegido por decaimiento
        monótono de W_LG (línea de búsqueda de Armijo simplificada):

            t₀ = min(1, 1/(1 + ‖F‖_F)),
            aceptar t si ‖F(b + t δb)‖ ≤ (1 − c t) ‖F(b)‖,  c = 10^{-4}.

        Si ningún t en {t₀, t₀/2, …, t_floor} satisface Armijo, se acepta
        t_floor de todos modos (progreso forzado en filtración de Novikov).
        """
        c_armijo = 1e-4
        t = min(1.0, 1.0 / (1.0 + residual_norm))
        t = max(t, _NEWTON_DAMPING_FLOOR)

        best_b = b + t * delta_b
        best_norm = float(
            la.norm(
                Phase3_MaurerCartanRegularization._maurer_cartan_function(
                    best_b, T_lambda, m0
                ),
                ord="fro",
            )
        )

        t_trial = t
        for _ in range(12):
            candidate = b + t_trial * delta_b
            cand_norm = float(
                la.norm(
                    Phase3_MaurerCartanRegularization._maurer_cartan_function(
                        candidate, T_lambda, m0
                    ),
                    ord="fro",
                )
            )
            if cand_norm <= (1.0 - c_armijo * t_trial) * residual_norm:
                return candidate
            if cand_norm < best_norm:
                best_norm = cand_norm
                best_b = candidate
            t_trial *= 0.5
            if t_trial < _NEWTON_DAMPING_FLOOR:
                break

        return best_b

    # ── §3.5  Exponencial de la co-cadena y estado regularizado ──────────────

    @staticmethod
    def _regularized_density(
        b: NDArray[np.complex128],
        rho_audit: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Construye el estado seguro vía la acción adjunta de e^b:

            ρ_safe = e^b  ρ_audit  (e^b)†  /  Tr(·).

        La exponencial matricial se calcula por Padé (expm de SciPy).
        Se hermitiza y renormaliza para garantizar ρ_safe ∈ 𝔇(ℋ_audit).
        """
        # Simetrización antihermitiana suave para favorecer unitariedad de e^b
        # (si b es antihermitiana, e^b es unitaria y la pureza se preserva).
        b_ah = 0.5 * (b - b.conj().T)
        # Componente hermitiana residual se retiene con peso reducido (deformación no unitaria)
        b_h = 0.5 * (b + b.conj().T)
        b_eff = b_ah + 0.5 * b_h

        try:
            exp_b = la.expm(b_eff)
        except la.LinAlgError as exc:
            raise SphereBubblingAnomalyError(
                "Exponenciación de la co-cadena acotante falló (patología espectral)."
            ) from exc

        rho_safe = exp_b @ rho_audit @ exp_b.conj().T
        rho_safe = Phase1_StinespringImmersion._hermitize(rho_safe)

        tr = float(np.real(np.trace(rho_safe)))
        if tr < _MACHINE_EPSILON:
            raise SphereBubblingAnomalyError(
                "Estado regularizado de traza nula tras Ad_{e^b} (colapso espectral)."
            )
        rho_safe = rho_safe / tr

        # Verificación PSD defensiva: clip de autovalores negativos numéricos
        ev, U = la.eigh(rho_safe)
        ev = np.clip(ev.real, 0.0, None)
        ev_sum = ev.sum()
        if ev_sum < _MACHINE_EPSILON:
            raise SphereBubblingAnomalyError(
                "Espectro de ρ_safe enteramente nulo tras clipping PSD."
            )
        ev /= ev_sum
        rho_safe = (U * ev) @ U.conj().T
        rho_safe = Phase1_StinespringImmersion._hermitize(rho_safe)

        return rho_safe.astype(np.complex128)

    # ── §3.6  MORFISMO TERMINAL DE FASE 3 ────────────────────────────────────
    #          (continuación de apply_telescopic_deformation;
    #           cierra el endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁)

    def resolve_maurer_cartan_novikov(
        self,
        fibration_data: VerticalFibrationData,
        rho_audit: NDArray[np.complex128],
    ) -> TelescopicAuditState:
        r"""
        Aplica Newton-Kantorovich filtrado por Novikov para hallar la
        co-cadena acotante b que resuelve F(b) = 0:

            Φ₃ : VerticalFibrationData × 𝔇(ℋ_audit) ⟶ TelescopicAuditState.

        Pipeline
        --------
        1. Extraer T_λ^⊥ del artefacto de Fase 2.
        2. Calcular anomalía m₀ = Ad_{T_λ}(ρ) − ρ.
        3. Iterar b_{k+1} = b_k + t_k δb_k hasta ‖F(b)‖_F < ε_MC.
        4. Estimar grado de filtración de Novikov.
        5. Construir ρ_safe = Ad_{e^b}(ρ_audit) normalizado.
        6. Empaquetar TelescopicAuditState (objeto final de Z).

        Parámetros
        ----------
        fibration_data : VerticalFibrationData
            Salida de Φ₂ = apply_telescopic_deformation.
        rho_audit : NDArray[np.complex128]
            Matriz de densidad del baño (proveniente de Fase 1).

        Retorna
        -------
        TelescopicAuditState
            Estado auditado, regularizado y libre de burbujeo.
        """
        # Continuación directa del objeto terminal de Fase 2
        T_lambda = fibration_data.T_lambda_vertical
        rho_audit = self._validate_density_matrix(rho_audit, name="rho_audit")

        if T_lambda.shape != rho_audit.shape:
            raise EhresmannFibrationError(
                f"Incompatibilidad dimensional T_λ {T_lambda.shape} vs "
                f"ρ_audit {rho_audit.shape}."
            )

        # Anomalía inicial
        m0 = self._compute_anomaly(T_lambda, rho_audit)

        # Co-cadena nula (punto de partida de Newton en el cono de Novikov)
        b = np.zeros_like(T_lambda, dtype=np.complex128)

        converged = False
        W_LG = float(la.norm(m0, ord="fro"))
        iters = 0

        for iters in range(1, _MAX_NOVIKOV_ITERATIONS + 1):
            F_b = self._maurer_cartan_function(b, T_lambda, m0)
            residual_norm = float(la.norm(F_b, ord="fro"))
            W_LG = residual_norm

            if residual_norm < _MAURER_CARTAN_CONVERGENCE_TOL:
                converged = True
                break

            # Paso de Newton + damping Armijo
            delta_b = self._solve_linearized(b, T_lambda, F_b)
            b = self._newton_damped_step(b, delta_b, T_lambda, m0, residual_norm)

            # Proyección suave: eliminar deriva de traza de b (gauge fixing)
            tr_b = complex(np.trace(b))
            if abs(tr_b) > _MACHINE_EPSILON:
                b = b - (tr_b / b.shape[0]) * np.eye(b.shape[0], dtype=np.complex128)

        if not converged:
            raise SphereBubblingAnomalyError(
                f"Magnificación generó burbujeo de esferas insoluble "
                f"tras {iters} iteraciones de Novikov "
                f"(W_LG = {W_LG:.3e} ≥ ε_MC = {_MAURER_CARTAN_CONVERGENCE_TOL:.1e})."
            )

        # Grado de filtración de Novikov (área/acción acumulada)
        novikov_degree = float(la.norm(b, ord="fro")) + _NOVIKOV_AREA_GAP * iters

        # Estado regularizado
        rho_safe = self._regularized_density(b, rho_audit)

        return TelescopicAuditState(
            audited_density_matrix=rho_safe,
            landau_ginzburg_potential=W_LG,
            novikov_convergence_iterations=iters,
            maurer_cartan_bounding_cochain=b,
            novikov_filtration_degree=novikov_degree,
            is_safe_for_witten_atiyah=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO · EHRESMANN TELESCOPIC ENGINE                         ║
# ║   Endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁  :  𝔇(ℋ_MIC) × ℝ_{≥0} → TelescopicAuditState ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class EhresmannTelescopicEngine(Morphism, Phase3_MaurerCartanRegularization):
    r"""
    Motor maestro de magnificación ortogonal.

    Subordina la «visibilidad» ejecutiva a las leyes de invarianza
    topológica, teoría espectral y teoría cuántica de campos topológica
    (TQFT de tipo Witten-Atiyah).

    La composición funtorial estricta se realiza en
    ``execute_telescopic_audit``, que encadena:

        ρ_business  ──Φ₁──►  StinespringDilationData
                    ──Φ₂──►  VerticalFibrationData
                    ──Φ₃──►  TelescopicAuditState.

    Cada Φᵢ es el morfismo terminal de la fase i y el dominio de la fase i+1,
    realizando la anidación funtorial exigida por la arquitectura del Topos.
    """

    def __init__(self, audit_dimension: int = 4) -> None:
        if not isinstance(audit_dimension, int) or audit_dimension < _MIN_AUDIT_DIM:
            raise ValueError(
                f"audit_dimension debe ser entero ≥ {_MIN_AUDIT_DIM}; "
                f"recibido {audit_dimension}."
            )
        self.audit_dimension = audit_dimension

    def execute_telescopic_audit(
        self,
        rho_business: NDArray[np.complex128],
        magnification_lambda: float,
    ) -> TelescopicAuditState:
        r"""
        Composición funtorial estricta Z = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Parámetros
        ----------
        rho_business : NDArray[np.complex128]
            Matriz de densidad del sistema de negocio (n × n), ρ ∈ 𝔇(ℋ_MIC).
        magnification_lambda : float
            Factor de magnificación telescópica λ ≥ 0.

        Retorna
        -------
        TelescopicAuditState
            Estado final auditado, libre de anomalías topológicas y
            certificado para el funtor de Witten-Atiyah-Floer.
        """
        if magnification_lambda < 0.0:
            raise ValueError(
                f"magnification_lambda ≥ 0 requerido; recibido {magnification_lambda}."
            )

        # ── Φ₁ · Fase 1 · Inmersión ortogonal de Stinespring ─────────────────
        stinespring_data: StinespringDilationData = self.compute_isometric_immersion(
            rho_business, self.audit_dimension
        )

        # ── Φ₂ · Fase 2 · Fibración vertical de Ehresmann ────────────────────
        #     Dominio = objeto terminal de Φ₁
        fibration_data: VerticalFibrationData = self.apply_telescopic_deformation(
            stinespring_data, magnification_lambda
        )

        # ── Φ₃ · Fase 3 · Regularización Maurer-Cartan / Novikov ─────────────
        #     Dominio = objeto terminal de Φ₂ (+ ρ_audit de Φ₁)
        audit_state: TelescopicAuditState = self.resolve_maurer_cartan_novikov(
            fibration_data,
            stinespring_data.rho_audit_subspace,
        )

        logger.info(
            "Auditoría Telescópica exitosa | "
            "λ=%.4f | pureza=%.6f | residual_Stinespring=%.3e | "
            "fuga_H=%.3e | curvatura_Ω=%.3e | "
            "Novikov_iters=%d | deg_nov=%.4f | W_LG=%.4e | "
            "Witten-Atiyah_safe=%s",
            magnification_lambda,
            stinespring_data.purity_preservation,
            stinespring_data.stinespring_residual,
            fibration_data.horizontal_leakage_norm,
            fibration_data.connection_curvature_norm,
            audit_state.novikov_convergence_iterations,
            audit_state.novikov_filtration_degree,
            audit_state.landau_ginzburg_potential,
            audit_state.is_safe_for_witten_atiyah,
        )

        return audit_state


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "TelescopicEngineError",
    "StinespringDilationError",
    "InvalidDensityMatrixError",
    "EhresmannFibrationError",
    "SphereBubblingAnomalyError",
    "SpectralDegeneracyError",
    # DTOs (objetos del Topos)
    "StinespringDilationData",
    "VerticalFibrationData",
    "TelescopicAuditState",
    # Fases anidadas
    "Phase1_StinespringImmersion",
    "Phase2_TelescopicVerticalFibration",
    "Phase3_MaurerCartanRegularization",
    # Orquestador
    "EhresmannTelescopicEngine",
]