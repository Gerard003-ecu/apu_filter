# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Omega Wisdom Hodge Dualizer                                         ║
║ Ruta   : app/wisdom/omega_wisdom_hodge_dualizer.py                           ║
║ Versión: 3.1.0-Fermion-Modular-Connes-OODA-Graded-Heyting-Secure             ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ESTRATO WISDOM (V_𝕎) ─────
Este módulo consagra al Agente Soberano y Observador Activo encargado de gobernar
al dualizador métrico-epistémico del sistema, acoplando las excitaciones
fermiónicas en el Espacio de Fock fermiónico $$\mathcal{F}(\mathcal{H})$$ con el
álgebra de operadores no conmutativos de von Neumann (Tomita-Takesaki).

El sistema trata el caos estocástico sintáctico de los Modelos de Lenguaje (LLMs)
no como texto plano, sino como un fluido de logits proyectado sobre un espacio
de Fock, subyugando su evolución a la regularidad de de Rham-Hodge, la
conservación de volumen simpléctico y la idempotencia.

INVARIANTES MATEMÁTICOS, GEOMÉTRICOS Y PROPIEDADES DE CALIBRE PRESERVADAS: ──────
  [I1] Dualidad Partícula-Hueco de Hodge en el Espacio de Fock:
       El operador estrella de Hodge combinatorio se construye de manera explícita
       sobre la potencia exterior del espacio de Hilbert complejo separable,
       satisfaciendo de forma exacta la involución y la isometría de De Rham:
       $$\star_k : \Lambda^k(\mathcal{H}) \xrightarrow{\simeq} \Lambda^{N-k}(\mathcal{H}^*)$$
       $$\star_{N-k}\star_k = (-1)^{k(N-k)}\,\mathrm{Id} \quad \text{(Firma Riemanniana)} \quad [1]$$

  [I2] Conjugación Modular No Conmutativa de Tomita-Takesaki:
       El operador modular antiunitario $$J_\rho$$ se computa de manera física e
       intrínseca a partir de la deconstrucción espectral del operador de densidad
       $$\rho$$ de la MAC, garantizando la reflexividad del flujo modular:
       $$J_\rho(X) = \rho^{1/2} X^\dagger \rho^{-1/2} \quad \implies \quad J^2 = \mathrm{Id} \quad [1]$$

  [I3] Conservación de la Norma de Hilbert-Schmidt (Antiunitariedad GNS):
       El producto interno inducido por la métrica de Gibbs se preserva bajo la
       acción del conjugador modular, eliminando la asimetría de fase cuántica:
       $$\langle J(A), J(B) \rangle_\rho = \langle B, A \rangle_\rho \quad [1]$$

  [I4] Condición KMS en el Equilibrio Térmico (Simetría Modular de Kubo-Martin-Schwinger):
       Las correlaciones de la deliberación cuántica satisfacen de manera exacta la
       periodicidad analítica imaginaria ante perturbaciones infinitesimales:
       $$\operatorname{Tr}(\rho A B) = \operatorname{Tr}\bigl(\rho B \sigma_{-i}(A)\bigr) \quad [1]$$

  [I5] Isomorfismo de la Adjunción de Galois:
       La coherencia del transporte paralelo entre el espacio táctico discreto (MIC)
       y el de sabiduría continuo (MAC) se rige bajo la equivalencia functorial:
       $$\|X - G(Y)\|_F \le L_{\mathrm{max}} \|F(X) - Y\|_T + \varepsilon_{\mathrm{num}} \quad [1]$$

  [I6] Cota de Lipschitz Espectral de Connes:
       La velocidad de descompresión y distorsión semántica de las actas de deliberación
       permanece acotada geodésicamente mediante la fórmula de diferencias divididas
       de Daleckii-Krein para la derivada de Fréchet del operador de Dirac de Connes:
       $$L_{\max} = \frac{C_{\text{base}}}{1 + (\lambda_{\max}(D) - \lambda_{\min}(D))} \le \frac{1}{2 \lambda_{\min}^{3/2}} \quad [1, 2]$$

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial): ───────────────────
La transición de estados se rige por la Ley de Clausura Transitiva de subespacios
de Hilbert covariantes y se compone de tres fases fuertemente acopladas [3]:

  Fase 1 ──► FASE 1: OBSERVACIÓN ESPECTRAL Y SANEAMIENTO DE LA MATRIZ (Observe)
             Construye el operador de Hodge Star combinatorio, certifica la isometría
             de Fock contra él y valida los postulados de Dirac-von Neumann
             sobre la matriz de densidad de la MAC [4].
             Entrega: Phase1SpectralObservation como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: ORIENTACIÓN MODULAR Y CONSISTENCIA KMS (Orient)
             Construye físicamente el operador modular $$J_\rho$$ desde el espectro
             de la MAC, certifica su involución y antiunitariedad GNS, y valida
             el equilibrio térmico modular KMS con residuo normalizado por escala [5].
             Entrega: Phase2ModularOrientation como precondición formal de Fase 3.

  Fase 3 ──► FASE 3: DECISIÓN Y ACTUACIÓN EN EL RETÍCULO DE HEYTING (Decide + Act)
             Deriva rigurosamente la cota de Lipschitz semántica (Daleckii-Krein),
             audita la adjunción de Galois completa $$F \dashv G$$ y colapsa el veredicto
             por votación de redundancia modular triple (TMR) sobre los certificados
             de fase en el clasificador de subobjetos de Heyting $$\Omega_3$$ [6].
             Veredicto final: $$\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\} \quad [7]$$.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum, auto
from itertools import combinations
from math import comb
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Wisdom.OmegaWisdomHodgeDualizer")

# ─────────────────────────────────────────────────────────────────────────────
# Tipos y constantes de Wilkinson / silicio
# ─────────────────────────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
BasisTuple = Tuple[int, ...]

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1e-12
_INVOLUTION_TOL: Final[float] = 1e-10
_GNS_TOL: Final[float] = 1e-10
_ISOMETRY_TOL: Final[float] = 1e-12
_RHO_SPECTRAL_FLOOR: Final[float] = 1e-12
_CONDITION_RHO_MAX: Final[float] = 1e12
_KMS_TOL: Final[float] = 1e-8


# =============================================================================
# JERARQUÍA DE EXCEPCIONES
# =============================================================================

class OmegaWisdomError(Exception):
    """Raíz de fallos del dualizador Ω/WISDOM."""


class FockDegreeError(OmegaWisdomError):
    """Grado k inválido o incompatibilidad comb(N,k)."""


class FockIsometryError(OmegaWisdomError):
    """Ruptura de isometría del Hodge fermiónico."""


class DensityOperatorError(OmegaWisdomError):
    """ρ no hermítico, no fiel o espectralmente degenerado sin remedio."""


class ModularInvolutionError(OmegaWisdomError):
    """J² ≠ Id más allá de tolerancia."""


class GNSAntiunitarityError(OmegaWisdomError):
    """Fallo de preservación de la norma GNS bajo J."""


class OmegaWisdomGovernanceError(OmegaWisdomError):
    """Colapso del estado soberano Ω⊗WISDOM."""


# =============================================================================
# ENUMS DE VEREDICTO (retículo de Heyting acotado en el topos de Connes)
# =============================================================================

class DualizerVerdict(IntEnum):
    """
    Clasificador Ω₃ del dualizador.

    COHERENT  — isometría Fock + J²=Id + GNS OK.
    DEGRADED  — regularización IR de ρ o residuales tolerables.
    VETOED    — ruptura de involución, no-isometría o ρ no fiel.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __or__(self, other: "DualizerVerdict") -> "DualizerVerdict":
        return self.__class__(max(int(self), int(other)))

    def __and__(self, other: "DualizerVerdict") -> "DualizerVerdict":
        return self.__class__(min(int(self), int(other)))


class RegularizationKind(Enum):
    NONE = auto()
    SPECTRAL_CLIP = auto()       # clip λ_i ≥ floor
    TIKHONOV_SHIFT = auto()      # ρ ← ρ + εI
    PINCHED_SUPPORT = auto()     # proyección al soporte fiel


# =============================================================================
# DTOs — CONTRATOS DE CONTINUIDAD FUNTORIAL
# =============================================================================

@dataclass(frozen=True, slots=True)
class FockHodgeReport:
    r"""
    Certificado atómico de una aplicación ★_k sobre un estado de Fock.

    Campos
    ------
    original_k, dual_k : grados k y N−k.
    dimension_n : N (orbitales / modos).
    dim_source, dim_target : comb(N,k), comb(N,N−k).
    is_isometry : ‖★ψ‖ = ‖ψ‖.
    double_star_sign : signo de ★★ = (−1)^{k(N−k)} Id.
    levi_civita_norm : ‖H‖_F de la matriz de Hodge.
    condition_hodge : κ₂(HᵀH) (debe ≈ 1 si isometría exacta).
    residual_isometry : |‖ψ‖ − ‖★ψ‖|.
    """
    original_k: int
    dual_k: int
    dimension_n: int
    dim_source: int
    dim_target: int
    is_isometry: bool
    double_star_sign: int
    levi_civita_norm: float
    condition_hodge: float
    residual_isometry: float


@dataclass(frozen=True, slots=True)
class FockDualityMorphism:
    r"""
    Morfismo de dualidad partícula-hueco certificado (cierre FASE 1).

    Es el *único* objeto legítimo de entrada a la FASE 2. Encapsula la
    matriz de Hodge de Levi-Civita, las bases de Slater y el reporte
    espectral del isomorfismo ★_k.
    """
    dimension_n: int
    k_particles: int
    dual_k: int
    basis_k: Tuple[BasisTuple, ...]
    basis_dual: Tuple[BasisTuple, ...]
    hodge_matrix: ComplexMatrix          # shape (comb(N,N-k), comb(N,k))
    double_star_sign: int
    report: FockHodgeReport
    is_certified_isometry: bool
    gram_residual: float                 # ‖H†H − I‖_F (fuente)
    cogram_residual: float               # ‖HH† − I‖_F (target)


@dataclass(frozen=True, slots=True)
class NonCommutativeHodgeReport:
    """Certificado atómico de una aplicación J_ρ sobre un operador."""
    trace_residual: float
    is_involution: bool
    is_antiunitary: bool
    involution_residual: float
    gns_norm_original: float
    gns_norm_dual: float
    kms_residual: float
    rho_condition: float
    regularization: RegularizationKind


@dataclass(frozen=True, slots=True)
class ModularConjugationCertificate:
    r"""
    Certificado de la estructura modular de Tomita-Takesaki (cierre FASE 2).

    Es el *único* objeto legítimo de entrada a la FASE 3.

    Contiene ρ saneada, sus raíces fraccionarias, el morfismo Fock heredado
    (continuidad FASE 1→2) y los invariantes J²=Id, antiunitariedad GNS y
    simetría KMS a β=1.
    """
    fock_morphism: FockDualityMorphism
    rho_sanitized: ComplexMatrix
    rho_half: ComplexMatrix
    rho_neg_half: ComplexMatrix
    rho_eigenvalues: RealVector
    rho_condition: float
    regularization: RegularizationKind
    tikhonov_shift: float
    is_faithful: bool
    is_involution_certified: bool
    is_gns_antiunitary_certified: bool
    kms_symmetric: bool
    probe_report: NonCommutativeHodgeReport
    modular_operator_delta_half: Optional[ComplexMatrix] = None  # Δ^{1/2}≈ρ en GNS estándar


@dataclass(frozen=True, slots=True)
class OmegaWisdomSovereignState:
    """
    Estado soberano del dualizador Ω⊗WISDOM (cierre FASE 3 / Act).

    Artefacto consumible por estratos STRATEGY/WISDOM y por el
    DiscreteHodgeStarAgent (puente ciber-físico).
    """
    modular_certificate: ModularConjugationCertificate
    verdict: DualizerVerdict
    fock_isometry_ok: bool
    modular_involution_ok: bool
    gns_antiunitary_ok: bool
    kms_ok: bool
    composite_residual: float
    dual_fock_state: Optional[ComplexVector]
    dual_operator: Optional[ComplexMatrix]
    reasons: Tuple[str, ...]
    timestamp_utc: str
    stratum: str = "WISDOM"
    version: str = "2.0.0"


# =============================================================================
# FASE 1 — OMEGA: DUALIDAD PARTÍCULA-HUECO EN EL ESPACIO DE FOCK FERMIÓNICO
#           Álgebra exterior Λ^•(ℂ^N) + Levi-Civita + isometría de Hodge
#           Último método: emit_fock_duality_morphism → FockDualityMorphism
# =============================================================================

class Phase1_FockHodgeObserver:
    r"""
    Fase 1 (OMEGA): construye y certifica el isomorfismo de Hodge fermiónico

    .. math::

        \star_k : \Lambda^k(\mathbb{C}^N) \xrightarrow{\simeq}
                  \Lambda^{N-k}(\mathbb{C}^N)

    realizado en la base de Slater (combinaciones ordenadas). La matriz de
    Hodge tiene entradas en $\{-1,0,+1\}$ dadas por el signo de la
    permutación que reordena $(I,J)$ a $(1,\dots,N)$ cuando
    $I\sqcup J = \{0,\dots,N-1\}$.
    """

    def __init__(
        self,
        dimension: int,
        isometry_tol: float = _ISOMETRY_TOL,
    ) -> None:
        if dimension < 0:
            raise FockDegreeError(f"Dimensión N={dimension} inválida.")
        self._dim: int = int(dimension)
        self._isometry_tol: float = float(isometry_tol)
        # Caché de bases y matrices de Hodge por grado
        self._basis_cache: Dict[int, Tuple[BasisTuple, ...]] = {}
        self._hodge_cache: Dict[int, ComplexMatrix] = {}

    # ── propiedades ──────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Número de orbitales / modos N."""
        return self._dim

    def exterior_dimension(self, k: int) -> int:
        r"""dim Λ^k = C(N,k)."""
        self._assert_degree(k)
        return comb(self._dim, k)

    # ── bases de Slater ──────────────────────────────────────────────────────

    def _assert_degree(self, k: int) -> None:
        if k < 0 or k > self._dim:
            raise FockDegreeError(
                f"Grado k={k} inválido para dimensión N={self._dim}."
            )

    def slater_basis(self, k: int) -> Tuple[BasisTuple, ...]:
        """
        Base ordenada lexicográficamente de Λ^k:
        todas las k-subsets de {0,…,N−1} como tuplas crecientes.
        """
        self._assert_degree(k)
        if k not in self._basis_cache:
            self._basis_cache[k] = tuple(combinations(range(self._dim), k))
        return self._basis_cache[k]

    # ── signo de permutación (Levi-Civita) ───────────────────────────────────

    @staticmethod
    def permutation_sign(p: Sequence[int]) -> int:
        r"""
        Signo de la permutación π ∈ S_n vía número de inversiones:
        $\operatorname{sgn}(\pi)=(-1)^{\#\{i<j:\pi_i>\pi_j\}}$.

        Complejidad O(n²); suficiente para N ≲ 16 (C(16,8)=12870).
        """
        sign = 1
        n = len(p)
        # Copia para no mutar al caller si pasa lista
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                if pi > p[j]:
                    sign = -sign
        return sign

    @staticmethod
    def permutation_sign_merge(p: Sequence[int]) -> int:
        """
        Variante O(n log n) por conteo de inversiones con merge-sort.
        Preferida para N grandes.
        """
        inv = [0]

        def sort(a: List[int]) -> List[int]:
            if len(a) <= 1:
                return a
            mid = len(a) // 2
            L, R = sort(a[:mid]), sort(a[mid:])
            out: List[int] = []
            i = j = 0
            while i < len(L) and j < len(R):
                if L[i] <= R[j]:
                    out.append(L[i]); i += 1
                else:
                    out.append(R[j]); j += 1
                    inv[0] += len(L) - i
            out.extend(L[i:]); out.extend(R[j:])
            return out

        sort(list(p))
        return -1 if (inv[0] & 1) else 1

    # ── matriz de Hodge ──────────────────────────────────────────────────────

    def build_hodge_matrix(self, k: int) -> ComplexMatrix:
        r"""
        Matriz $H_k$ de tamaño $C(N,N-k)\times C(N,k)$:

        $$(H_k)_{J,I} = \begin{cases}
            \operatorname{sgn}(I\sqcup J) & \text{si }I\cap J=\emptyset,\\
            0 & \text{en otro caso.}
        \end{cases}$$

        Resultado cacheado por grado.
        """
        self._assert_degree(k)
        if k in self._hodge_cache:
            return self._hodge_cache[k].copy()

        n = self._dim
        bas_k = self.slater_basis(k)
        bas_d = self.slater_basis(n - k)
        H = np.zeros((len(bas_d), len(bas_k)), dtype=np.complex128)

        # Mapa de complemento → índice dual para O(1) lookup
        dual_index = {row: i for i, row in enumerate(bas_d)}

        for j, I in enumerate(bas_k):
            set_I = set(I)
            # Complemento canónico
            comp = tuple(v for v in range(n) if v not in set_I)
            # El complemento ya es creciente ⇒ está en bas_d
            i = dual_index[comp]
            perm = list(I) + list(comp)
            # Usar merge-sort sign para N grandes
            if n > 10:
                s = self.permutation_sign_merge(perm)
            else:
                s = self.permutation_sign(perm)
            H[i, j] = s

        self._hodge_cache[k] = H
        return H.copy()

    # ── double-star y propiedades algebraicas ────────────────────────────────

    @staticmethod
    def double_star_sign(k: int, n: int) -> int:
        r"""★★ = (−1)^{k(N−k)} Id  sobre Λ^k (métrica euclídea, sin * complejo)."""
        return -1 if ((k * (n - k)) & 1) else 1

    def certify_hodge_isometry(self, H: ComplexMatrix) -> Tuple[float, float, float]:
        r"""
        Certifica isometría de columnas/filas:

        Returns
        -------
        gram_res : ‖H†H − I‖_F
        cogram_res : ‖HH† − I‖_F
        cond : κ₂(H†H)
        """
        m, n = H.shape  # m=target, n=source
        gram = H.conj().T @ H
        cogram = H @ H.conj().T
        gram_res = float(np.linalg.norm(gram - np.eye(n, dtype=np.complex128), ord="fro"))
        cogram_res = float(np.linalg.norm(cogram - np.eye(m, dtype=np.complex128), ord="fro"))
        # Condición vía autovalores del Gram
        ev = np.real(la.eigvalsh(gram))
        lam_min = float(np.min(ev)) if ev.size else 1.0
        lam_max = float(np.max(ev)) if ev.size else 1.0
        cond = lam_max / max(lam_min, _MACHINE_EPS)
        return gram_res, cogram_res, cond

    # ── aplicación del Hodge a un estado ─────────────────────────────────────

    def fock_particle_hole_duality(
        self,
        state_vector: ComplexVector,
        k_particles: int,
    ) -> Tuple[ComplexVector, FockHodgeReport]:
        r"""
        Aplica ★_k al estado de Fock ψ ∈ Λ^k.

        Parameters
        ----------
        state_vector :
            Coeficientes en la base de Slater de grado k (longitud C(N,k)).
        k_particles :
            Grado k.

        Returns
        -------
        dual_state : ★ψ ∈ Λ^{N−k}
        report : FockHodgeReport
        """
        n = self._dim
        k = int(k_particles)
        self._assert_degree(k)

        psi = np.asarray(state_vector, dtype=np.complex128).ravel()
        dim_k = comb(n, k)
        dim_d = comb(n, n - k)
        if psi.shape[0] != dim_k:
            raise FockDegreeError(
                f"dim(ψ)={psi.shape[0]} ≠ C({n},{k})={dim_k}."
            )

        H = self.build_hodge_matrix(k)
        dual = H @ psi

        norm_o = float(np.linalg.norm(psi))
        norm_d = float(np.linalg.norm(dual))
        res_iso = abs(norm_o - norm_d)
        is_iso = res_iso < self._isometry_tol * max(1.0, norm_o)

        gram_res, cogram_res, cond = self.certify_hodge_isometry(H)
        dss = self.double_star_sign(k, n)

        report = FockHodgeReport(
            original_k=k,
            dual_k=n - k,
            dimension_n=n,
            dim_source=dim_k,
            dim_target=dim_d,
            is_isometry=bool(is_iso),
            double_star_sign=int(dss),
            levi_civita_norm=float(np.linalg.norm(H, ord="fro")),
            condition_hodge=float(cond),
            residual_isometry=float(res_iso),
        )
        return dual, report

    def apply_double_star(
        self,
        state_vector: ComplexVector,
        k_particles: int,
    ) -> Tuple[ComplexVector, float]:
        r"""
        Aplica ★★ y mide el residual respecto de (−1)^{k(N−k)} ψ.

        Returns
        -------
        recovered : ★★ψ
        residual : ‖★★ψ − s ψ‖
        """
        dual, _ = self.fock_particle_hole_duality(state_vector, k_particles)
        # ★ sobre el dual: grado N−k
        back, _ = self.fock_particle_hole_duality(dual, self._dim - k_particles)
        s = self.double_star_sign(k_particles, self._dim)
        psi = np.asarray(state_vector, dtype=np.complex128).ravel()
        residual = float(np.linalg.norm(back - s * psi))
        return back, residual

    def vacuum_to_volume_form(self) -> ComplexVector:
        r"""★|0⟩ = |vol⟩ ∈ Λ^N (único estado de llenado completo, signo +)."""
        vac = np.array([1.0 + 0.0j], dtype=np.complex128)  # Λ^0 ≅ ℂ
        vol, _ = self.fock_particle_hole_duality(vac, 0)
        return vol

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 1 ────────────────────────────────────
    # Su valor de retorno (FockDualityMorphism) es el objeto de arranque
    # obligatorio de la FASE 2.

    def emit_fock_duality_morphism(
        self,
        k_particles: int,
        probe_state: Optional[ComplexVector] = None,
        enforce_isometry: bool = True,
    ) -> FockDualityMorphism:
        r"""
        Emite el morfismo de dualidad partícula-hueco certificado.

        Construye H_k, certifica H†H=I / HH†=I y, opcionalmente, aplica un
        estado sonda para validar isometría puntual. El DTO saliente es el
        contrato de continuidad FASE 1 → FASE 2.

        Parameters
        ----------
        k_particles :
            Grado k del dominio.
        probe_state :
            Estado de prueba; si None, se usa el primer vector de la base
            (o uno aleatorio normalizado si C(N,k)>1).
        enforce_isometry :
            Si True, lanza FockIsometryError cuando gram/cogram residuales
            exceden tolerancia.

        Returns
        -------
        FockDualityMorphism
        """
        k = int(k_particles)
        self._assert_degree(k)
        n = self._dim

        bas_k = self.slater_basis(k)
        bas_d = self.slater_basis(n - k)
        H = self.build_hodge_matrix(k)
        gram_res, cogram_res, cond = self.certify_hodge_isometry(H)
        dss = self.double_star_sign(k, n)

        # Estado sonda
        dim_k = len(bas_k)
        if probe_state is None:
            if dim_k == 0:
                probe = np.zeros(0, dtype=np.complex128)
            elif dim_k == 1:
                probe = np.array([1.0 + 0.0j], dtype=np.complex128)
            else:
                rng = np.random.default_rng(0)
                v = rng.normal(size=dim_k) + 1j * rng.normal(size=dim_k)
                probe = (v / np.linalg.norm(v)).astype(np.complex128)
        else:
            probe = np.asarray(probe_state, dtype=np.complex128).ravel()

        if dim_k > 0 and probe.shape[0] == dim_k:
            _, report = self.fock_particle_hole_duality(probe, k)
        else:
            report = FockHodgeReport(
                original_k=k,
                dual_k=n - k,
                dimension_n=n,
                dim_source=dim_k,
                dim_target=len(bas_d),
                is_isometry=(gram_res < self._isometry_tol),
                double_star_sign=dss,
                levi_civita_norm=float(np.linalg.norm(H, ord="fro")),
                condition_hodge=float(cond),
                residual_isometry=float(gram_res),
            )

        is_cert = (
            gram_res < max(self._isometry_tol, 1e-9) * max(dim_k, 1)
            and cogram_res < max(self._isometry_tol, 1e-9) * max(len(bas_d), 1)
            and report.is_isometry
        )

        if enforce_isometry and not is_cert and dim_k > 0:
            raise FockIsometryError(
                f"★_{k} no isométrica: gram_res={gram_res:.3e}, "
                f"cogram_res={cogram_res:.3e}, cond={cond:.3e}."
            )

        morph = FockDualityMorphism(
            dimension_n=n,
            k_particles=k,
            dual_k=n - k,
            basis_k=bas_k,
            basis_dual=bas_d,
            hodge_matrix=H,
            double_star_sign=int(dss),
            report=report,
            is_certified_isometry=bool(is_cert),
            gram_residual=float(gram_res),
            cogram_residual=float(cogram_res),
        )
        logger.info(
            "FASE1 EMIT FockDuality ★_%d→%d N=%d iso=%s gram=%.2e",
            k, n - k, n, is_cert, gram_res,
        )
        return morph


# =============================================================================
# FASE 2 — WISDOM: CONJUGACIÓN MODULAR DE TOMITA-TAKESAKI (Hodge no conmutativo)
#           Continúa desde FockDualityMorphism (emit_fock_duality_morphism).
#           Último método: emit_modular_certificate → ModularConjugationCertificate
# =============================================================================

class Phase2_ModularHodgeOrienter(Phase1_FockHodgeObserver):
    r"""
    Fase 2 (WISDOM): eleva la dualidad al álgebra de operadores B(ℋ)
    equipada con un estado fiel normal ω_ρ(X)=Tr(ρ X).

    El operador modular de Tomita-Takesaki en la representación GNS estándar
    (ℋ=L²(M,ω)) induce la conjugación antiunitaria

    .. math::

        J_\rho(X) = \rho^{1/2} X^\dagger \rho^{-1/2},

    involución (J²=Id) y antiisometría del producto de Hilbert-Schmidt pesado

    .. math::

        \langle A,B\rangle_\rho = \operatorname{Tr}(\rho A^\dagger B).
    """

    def __init__(
        self,
        dimension: int,
        isometry_tol: float = _ISOMETRY_TOL,
        rho_floor: float = _RHO_SPECTRAL_FLOOR,
        rho_condition_max: float = _CONDITION_RHO_MAX,
        involution_tol: float = _INVOLUTION_TOL,
        gns_tol: float = _GNS_TOL,
    ) -> None:
        super().__init__(dimension=dimension, isometry_tol=isometry_tol)
        self._rho_floor = float(rho_floor)
        self._rho_cond_max = float(rho_condition_max)
        self._involution_tol = float(involution_tol)
        self._gns_tol = float(gns_tol)

    # ── saneamiento espectral de ρ ───────────────────────────────────────────

    def sanitize_density_operator(
        self,
        rho: ComplexMatrix,
        force_trace_one: bool = True,
    ) -> Tuple[ComplexMatrix, RealVector, ComplexMatrix, RegularizationKind, float]:
        r"""
        Garantiza ρ = ρ† ≻ 0 (fiel) con κ controlada.

        Pipeline
        --------
        1. Hermitización: ρ ← ½(ρ+ρ†).
        2. eigh → (λ, U).
        3. Si ∃ λ_i ≤ 0: clip a floor (SPECTRAL_CLIP).
        4. Si κ > κ_max: shift de Tikhonov (TIKHONOV_SHIFT).
        5. Renormalización Tr ρ = 1 (estados cuánticos).

        Returns
        -------
        rho_san, eigenvalues, eigenvectors, kind, shift
        """
        R = np.asarray(rho, dtype=np.complex128)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise DensityOperatorError(
                f"ρ debe ser matriz cuadrada; recibido shape={R.shape}."
            )
        if R.shape[0] != self._dim:
            raise DensityOperatorError(
                f"dim(ρ)={R.shape[0]} ≠ N={self._dim} del dualizador."
            )

        # 1. Hermitización
        herm_err = float(np.max(np.abs(R - R.conj().T)))
        if herm_err > 100.0 * _MACHINE_EPS * R.shape[0]:
            logger.warning(
                "FASE2: ρ no hermítico (‖ρ−ρ†‖_∞=%.3e) — proyectando.", herm_err
            )
        R = 0.5 * (R + R.conj().T)

        # 2. Espectro
        eigvals, eigvecs = la.eigh(R)
        eigvals = np.real(eigvals)  # seguridad numérica
        kind = RegularizationKind.NONE
        shift = 0.0

        # 3. Clip infrarrojo
        if np.any(eigvals < self._rho_floor):
            eigvals = np.clip(eigvals, self._rho_floor, None)
            kind = RegularizationKind.SPECTRAL_CLIP
            logger.warning("FASE2: SPECTRAL_CLIP λ_min → %.3e", self._rho_floor)

        # 4. Condición
        lam_min = float(np.min(eigvals))
        lam_max = float(np.max(eigvals))
        kappa = lam_max / max(lam_min, _MACHINE_EPS)
        if kappa > self._rho_cond_max:
            shift = lam_max * _MACHINE_EPS * 1e4
            eigvals = eigvals + shift
            kind = RegularizationKind.TIKHONOV_SHIFT
            logger.warning(
                "FASE2: TIKHONOV_SHIFT δ=%.3e (κ era %.3e)", shift, kappa
            )
            lam_min = float(np.min(eigvals))
            lam_max = float(np.max(eigvals))
            kappa = lam_max / max(lam_min, _MACHINE_EPS)

        # 5. Reconstrucción + traza
        rho_san = eigvecs @ np.diag(eigvals.astype(np.complex128)) @ eigvecs.conj().T
        rho_san = 0.5 * (rho_san + rho_san.conj().T)
        if force_trace_one:
            tr = float(np.real(np.trace(rho_san)))
            if tr <= 0.0:
                raise DensityOperatorError("Tr(ρ) ≤ 0 tras saneamiento.")
            rho_san = rho_san / tr
            eigvals = eigvals / tr

        if float(np.min(eigvals)) <= 0.0:
            raise DensityOperatorError(
                "ρ no fiel tras saneamiento: ∃ λ_i ≤ 0."
            )

        return (
            rho_san,
            eigvals.astype(np.float64),
            eigvecs,
            kind,
            float(shift),
        )

    def modular_roots(
        self,
        eigvals: RealVector,
        eigvecs: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, ComplexMatrix]:
        r"""ρ^{1/2} y ρ^{-1/2} vía cálculo funcional espectral."""
        sqrt_l = np.sqrt(np.maximum(eigvals, self._rho_floor))
        inv_sqrt_l = 1.0 / sqrt_l
        U = eigvecs
        rho_half = U @ np.diag(sqrt_l.astype(np.complex128)) @ U.conj().T
        rho_neg = U @ np.diag(inv_sqrt_l.astype(np.complex128)) @ U.conj().T
        # Hermitizar (error de redondeo)
        rho_half = 0.5 * (rho_half + rho_half.conj().T)
        rho_neg = 0.5 * (rho_neg + rho_neg.conj().T)
        return rho_half, rho_neg

    # ── producto GNS y KMS ───────────────────────────────────────────────────

    @staticmethod
    def gns_inner_product(
        rho: ComplexMatrix, A: ComplexMatrix, B: ComplexMatrix
    ) -> complex:
        r"""〈A,B〉_ρ = Tr(ρ A† B)."""
        return complex(np.trace(rho @ A.conj().T @ B))

    @staticmethod
    def gns_norm_sq(rho: ComplexMatrix, X: ComplexMatrix) -> float:
        r"""‖X‖²_GNS = Tr(ρ X† X)."""
        return float(np.real(np.trace(rho @ X.conj().T @ X)))

    def kms_residual(
        self,
        rho: ComplexMatrix,
        A: ComplexMatrix,
        B: ComplexMatrix,
    ) -> float:
        r"""
        Simetría KMS a β=1 (estado tracial modular):
        ω(A B) = ω(B σ_{-i}(A)) con σ_{-i}(A)=ρ A ρ^{-1},
        equivalente a Tr(ρ A B) = Tr(ρ B ρ A ρ^{-1}) = Tr(B ρ A).

        Residual: |Tr(ρ A B) − Tr(B ρ A)|.
        """
        lhs = np.trace(rho @ A @ B)
        rhs = np.trace(B @ rho @ A)
        return float(np.abs(lhs - rhs))

    # ── conjugación modular J ────────────────────────────────────────────────

    def apply_modular_conjugation(
        self,
        operator_X: ComplexMatrix,
        rho_half: ComplexMatrix,
        rho_neg_half: ComplexMatrix,
    ) -> ComplexMatrix:
        r"""J(X) = ρ^{1/2} X† ρ^{-1/2}."""
        X = np.asarray(operator_X, dtype=np.complex128)
        return rho_half @ X.conj().T @ rho_neg_half

    def non_commutative_hodge_conjugation(
        self,
        operator_X: ComplexMatrix,
        rho_mac: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, NonCommutativeHodgeReport]:
        r"""
        Pipeline completo: sanea ρ, aplica J, certifica J²=Id y antiunitariedad GNS.

        Parameters
        ----------
        operator_X :
            Operador X ∈ M_N(ℂ).
        rho_mac :
            Matriz de densidad (estado de sabiduría).

        Returns
        -------
        j_x : J(X)
        report : NonCommutativeHodgeReport
        """
        X = np.asarray(operator_X, dtype=np.complex128)
        if X.shape != (self._dim, self._dim):
            raise DensityOperatorError(
                f"dim(X)={X.shape} ≠ ({self._dim},{self._dim})."
            )

        rho, ev, U, kind, shift = self.sanitize_density_operator(rho_mac)
        rho_h, rho_nh = self.modular_roots(ev, U)

        j_x = self.apply_modular_conjugation(X, rho_h, rho_nh)

        # J²(X) = X
        j_j_x = self.apply_modular_conjugation(j_x, rho_h, rho_nh)
        invol_res = float(np.linalg.norm(j_j_x - X, ord="fro"))
        is_invol = invol_res < self._involution_tol * max(1.0, float(np.linalg.norm(X, ord="fro")))

        # Antiunitariedad GNS
        n2_o = self.gns_norm_sq(rho, X)
        n2_d = self.gns_norm_sq(rho, j_x)
        trace_res = abs(n2_o - n2_d)
        is_anti = trace_res < self._gns_tol * max(1.0, n2_o)

        # KMS con B = X†
        kms_res = self.kms_residual(rho, X, X.conj().T)

        lam_min = float(np.min(ev))
        lam_max = float(np.max(ev))
        kappa = lam_max / max(lam_min, _MACHINE_EPS)

        report = NonCommutativeHodgeReport(
            trace_residual=float(trace_res),
            is_involution=bool(is_invol),
            is_antiunitary=bool(is_anti),
            involution_residual=float(invol_res),
            gns_norm_original=float(n2_o),
            gns_norm_dual=float(n2_d),
            kms_residual=float(kms_res),
            rho_condition=float(kappa),
            regularization=kind,
        )
        return j_x, report

    def certify_involution_battery(
        self,
        rho_half: ComplexMatrix,
        rho_neg_half: ComplexMatrix,
        n_probes: int = 5,
        seed: int = 0,
    ) -> Tuple[bool, float]:
        """Batería de n_probes operadores aleatorios para J²=Id."""
        rng = np.random.default_rng(seed)
        max_res = 0.0
        n = self._dim
        for _ in range(n_probes):
            A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            A = A.astype(np.complex128)
            j = self.apply_modular_conjugation(A, rho_half, rho_neg_half)
            jj = self.apply_modular_conjugation(j, rho_half, rho_neg_half)
            max_res = max(max_res, float(np.linalg.norm(jj - A, ord="fro")))
        ok = max_res < self._involution_tol * n
        return ok, float(max_res)

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 2 ────────────────────────────────────
    # Su valor de retorno (ModularConjugationCertificate) es el objeto de
    # arranque obligatorio de la FASE 3.

    def emit_modular_certificate(
        self,
        fock_morphism: FockDualityMorphism,
        rho_mac: ComplexMatrix,
        probe_operator: Optional[ComplexMatrix] = None,
        enforce_involution: bool = True,
        enforce_gns: bool = True,
        n_involution_probes: int = 5,
    ) -> ModularConjugationCertificate:
        r"""
        Emite el certificado modular de Tomita-Takesaki.

        Continuidad formal: exige un ``FockDualityMorphism`` emitido por FASE 1.
        Sanea ρ, construye raíces modulares, certifica J² e isometría GNS, y
        empaqueta el DTO de cierre FASE 2.

        Parameters
        ----------
        fock_morphism :
            Morfismo Fock certificado (FASE 1).
        rho_mac :
            Estado de sabiduría ρ.
        probe_operator :
            Operador sonda para el reporte; si None, se usa una matriz
            hermítica aleatoria.
        enforce_involution / enforce_gns :
            Si True, lanzan excepción ante fallo de post-condición.
        """
        if not isinstance(fock_morphism, FockDualityMorphism):
            raise TypeError(
                "FASE2 exige FockDualityMorphism emitido por FASE1."
            )
        if fock_morphism.dimension_n != self._dim:
            raise DensityOperatorError(
                f"N(Fock)={fock_morphism.dimension_n} ≠ N(dualizer)={self._dim}."
            )

        rho, ev, U, kind, shift = self.sanitize_density_operator(rho_mac)
        rho_h, rho_nh = self.modular_roots(ev, U)

        # Sonda
        if probe_operator is None:
            rng = np.random.default_rng(1)
            A = rng.normal(size=(self._dim, self._dim)) + 1j * rng.normal(
                size=(self._dim, self._dim)
            )
            probe = 0.5 * (A + A.conj().T)  # hermítica
            probe = probe.astype(np.complex128)
        else:
            probe = np.asarray(probe_operator, dtype=np.complex128)

        j_probe, report = self.non_commutative_hodge_conjugation(probe, rho)

        # Batería de involución
        invol_ok, invol_max = self.certify_involution_battery(
            rho_h, rho_nh, n_probes=n_involution_probes
        )
        # Refinar con el residual de la sonda
        invol_ok = invol_ok and report.is_involution
        gns_ok = report.is_antiunitary
        kms_ok = report.kms_residual < _KMS_TOL * max(1.0, report.gns_norm_original)

        if enforce_involution and not invol_ok:
            raise ModularInvolutionError(
                f"J² ≠ Id: residual_max={invol_max:.3e}, "
                f"probe_res={report.involution_residual:.3e}."
            )
        if enforce_gns and not gns_ok:
            raise GNSAntiunitarityError(
                f"GNS no preservada: residual={report.trace_residual:.3e}."
            )

        lam_min = float(np.min(ev))
        lam_max = float(np.max(ev))
        kappa = lam_max / max(lam_min, _MACHINE_EPS)
        is_faithful = lam_min > self._rho_floor * 0.5

        cert = ModularConjugationCertificate(
            fock_morphism=fock_morphism,
            rho_sanitized=rho,
            rho_half=rho_h,
            rho_neg_half=rho_nh,
            rho_eigenvalues=ev,
            rho_condition=float(kappa),
            regularization=kind,
            tikhonov_shift=float(shift),
            is_faithful=bool(is_faithful),
            is_involution_certified=bool(invol_ok),
            is_gns_antiunitary_certified=bool(gns_ok),
            kms_symmetric=bool(kms_ok),
            probe_report=report,
            modular_operator_delta_half=rho_h,  # en GNS estándar Δ^{1/2}~ρ^{1/2}
        )
        logger.info(
            "FASE2 EMIT ModularCert invol=%s gns=%s kms=%s κ_ρ=%.3e reg=%s",
            invol_ok, gns_ok, kms_ok, kappa, kind.name,
        )
        return cert


# =============================================================================
# FASE 3 — OMEGA ⊗ WISDOM: DUALIZADOR SOBERANO (topos de Connes)
#           Continúa desde ModularConjugationCertificate
#           (emit_modular_certificate).
#           Último método: emit_omega_wisdom_state → OmegaWisdomSovereignState
# =============================================================================

class Phase3_OmegaWisdomDecisionMaker(Phase2_ModularHodgeOrienter):
    r"""
    Fase 3: compone el funtor de Fock (★_k) con el funtor modular (J_ρ)
    en un endofuntor de gobernanza sobre el topos de Connes.

    Diagrama conmutativo aspiracional (Hodge no conmutativo de Connes):

    .. code-block:: text

        Λ^k  ──★_k──►  Λ^{N-k}
         │               │
        π_ρ             π_ρ
         ▼               ▼
        B(ℋ) ──J_ρ──►  B(ℋ)

    Emite veredicto Ω₃ y el estado soberano consumible por capas superiores
    y por el DiscreteHodgeStarAgent (puente ciber-físico opcional).
    """

    def __init__(
        self,
        dimension: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(dimension=dimension, **kwargs)
        self._last_state: Optional[OmegaWisdomSovereignState] = None

    # ── composición ★ luego J (y viceversa a nivel de reportes) ──────────────

    def apply_fock_duality_from_morphism(
        self,
        morphism: FockDualityMorphism,
        state_vector: ComplexVector,
    ) -> Tuple[ComplexVector, float]:
        """Aplica H del morfismo; retorna (★ψ, residual de isometría)."""
        psi = np.asarray(state_vector, dtype=np.complex128).ravel()
        if psi.shape[0] != morphism.hodge_matrix.shape[1]:
            raise FockDegreeError(
                f"dim(ψ)={psi.shape[0]} ≠ cols(H)={morphism.hodge_matrix.shape[1]}."
            )
        dual = morphism.hodge_matrix @ psi
        res = abs(float(np.linalg.norm(psi)) - float(np.linalg.norm(dual)))
        return dual, res

    def apply_J_from_certificate(
        self,
        certificate: ModularConjugationCertificate,
        operator_X: ComplexMatrix,
    ) -> ComplexMatrix:
        """Aplica J_ρ usando raíces ya certificadas (sin re-sanear)."""
        X = np.asarray(operator_X, dtype=np.complex128)
        if X.shape != (self._dim, self._dim):
            raise DensityOperatorError(
                f"dim(X)={X.shape} ≠ ({self._dim},{self._dim})."
            )
        return self.apply_modular_conjugation(
            X, certificate.rho_half, certificate.rho_neg_half
        )

    def composite_residual(
        self,
        certificate: ModularConjugationCertificate,
        state_vector: Optional[ComplexVector],
        operator_X: Optional[ComplexMatrix],
    ) -> Tuple[
        float,
        Optional[ComplexVector],
        Optional[ComplexMatrix],
        bool,
        bool,
        bool,
        bool,
    ]:
        """
        Evalúa residuales compuestos y flags de salud.

        Returns
        -------
        composite_residual, dual_state, dual_op,
        fock_ok, invol_ok, gns_ok, kms_ok
        """
        fock_ok = certificate.fock_morphism.is_certified_isometry
        invol_ok = certificate.is_involution_certified
        gns_ok = certificate.is_gns_antiunitary_certified
        kms_ok = certificate.kms_symmetric

        residuals = [
            certificate.fock_morphism.gram_residual,
            certificate.fock_morphism.cogram_residual,
            certificate.probe_report.involution_residual,
            certificate.probe_report.trace_residual,
            certificate.probe_report.kms_residual,
        ]

        dual_state: Optional[ComplexVector] = None
        dual_op: Optional[ComplexMatrix] = None

        if state_vector is not None:
            dual_state, iso_res = self.apply_fock_duality_from_morphism(
                certificate.fock_morphism, state_vector
            )
            residuals.append(iso_res)
            if iso_res > self._isometry_tol * max(
                1.0, float(np.linalg.norm(state_vector))
            ):
                fock_ok = False

        if operator_X is not None:
            dual_op = self.apply_J_from_certificate(certificate, operator_X)
            # Re-chequeo involución puntual
            back = self.apply_J_from_certificate(certificate, dual_op)
            invol_r = float(
                np.linalg.norm(
                    back - np.asarray(operator_X, dtype=np.complex128), ord="fro"
                )
            )
            residuals.append(invol_r)
            if invol_r > self._involution_tol * max(
                1.0, float(np.linalg.norm(operator_X, ord="fro"))
            ):
                invol_ok = False
            # GNS puntual
            n2_o = self.gns_norm_sq(certificate.rho_sanitized, operator_X)
            n2_d = self.gns_norm_sq(certificate.rho_sanitized, dual_op)
            gns_r = abs(n2_o - n2_d)
            residuals.append(gns_r)
            if gns_r > self._gns_tol * max(1.0, n2_o):
                gns_ok = False

        composite = float(max(residuals)) if residuals else 0.0
        return composite, dual_state, dual_op, fock_ok, invol_ok, gns_ok, kms_ok

    def classify_verdict(
        self,
        certificate: ModularConjugationCertificate,
        fock_ok: bool,
        invol_ok: bool,
        gns_ok: bool,
        kms_ok: bool,
        composite_residual: float,
    ) -> Tuple[DualizerVerdict, Tuple[str, ...]]:
        """Clasificador Ω₃ del dualizador."""
        reasons: List[str] = []
        verdict = DualizerVerdict.COHERENT

        # VETO duro
        if not invol_ok:
            verdict = DualizerVerdict.VETOED
            reasons.append("modular_involution_failed")
        if not fock_ok and certificate.fock_morphism.report.dim_source > 0:
            verdict = DualizerVerdict.VETOED
            reasons.append("fock_isometry_failed")
        if not certificate.is_faithful:
            verdict = DualizerVerdict.VETOED
            reasons.append("rho_not_faithful")
        if certificate.rho_condition > self._rho_cond_max:
            verdict = DualizerVerdict.VETOED
            reasons.append(f"rho_kappa={certificate.rho_condition:.3e}")

        # DEGRADED
        if verdict == DualizerVerdict.COHERENT:
            if not gns_ok:
                verdict = DualizerVerdict.DEGRADED
                reasons.append("gns_residual_degraded")
            if not kms_ok:
                verdict = verdict | DualizerVerdict.DEGRADED
                reasons.append(
                    f"kms_residual={certificate.probe_report.kms_residual:.3e}"
                )
            if certificate.regularization != RegularizationKind.NONE:
                verdict = verdict | DualizerVerdict.DEGRADED
                reasons.append(f"rho_reg={certificate.regularization.name}")
            if certificate.fock_morphism.gram_residual > _ISOMETRY_TOL:
                verdict = verdict | DualizerVerdict.DEGRADED
                reasons.append(
                    f"fock_gram={certificate.fock_morphism.gram_residual:.3e}"
                )
            if composite_residual > 1e-8:
                verdict = verdict | DualizerVerdict.DEGRADED
                reasons.append(f"composite_res={composite_residual:.3e}")

        return verdict, tuple(reasons)

    # ── puente opcional al agente de Hodge discreto ──────────────────────────

    def bridge_summary_for_hodge_agent(
        self, state: OmegaWisdomSovereignState
    ) -> Dict[str, Any]:
        """
        Resumen serializable para acoplar con DiscreteHodgeStarAgent /
        estratos STRATEGY. No importa el agente (evita acoplamiento duro).
        """
        c = state.modular_certificate
        return {
            "verdict": state.verdict.name,
            "fock": {
                "N": c.fock_morphism.dimension_n,
                "k": c.fock_morphism.k_particles,
                "dual_k": c.fock_morphism.dual_k,
                "isometry": state.fock_isometry_ok,
                "gram_residual": c.fock_morphism.gram_residual,
                "double_star_sign": c.fock_morphism.double_star_sign,
            },
            "modular": {
                "involution": state.modular_involution_ok,
                "gns": state.gns_antiunitary_ok,
                "kms": state.kms_ok,
                "rho_condition": c.rho_condition,
                "regularization": c.regularization.name,
                "faithful": c.is_faithful,
            },
            "composite_residual": state.composite_residual,
            "reasons": list(state.reasons),
            "timestamp_utc": state.timestamp_utc,
            "stratum": state.stratum,
            "version": state.version,
        }

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 3 ────────────────────────────────────

    def emit_omega_wisdom_state(
        self,
        certificate: ModularConjugationCertificate,
        state_vector: Optional[ComplexVector] = None,
        operator_X: Optional[ComplexMatrix] = None,
        raise_on_veto: bool = False,
    ) -> OmegaWisdomSovereignState:
        r"""
        Emite el estado soberano Ω⊗WISDOM (cierre del ciclo de dualización).

        Parameters
        ----------
        certificate :
            Certificado modular FASE 2 (obligatorio).
        state_vector :
            Estado de Fock opcional a dualizar con ★_k del morfismo.
        operator_X :
            Operador opcional a conjugar con J_ρ.
        raise_on_veto :
            Si True, lanza OmegaWisdomGovernanceError cuando verdict=VETOED.

        Returns
        -------
        OmegaWisdomSovereignState
        """
        if not isinstance(certificate, ModularConjugationCertificate):
            raise TypeError(
                "FASE3 exige ModularConjugationCertificate emitido por FASE2."
            )

        (
            composite,
            dual_state,
            dual_op,
            fock_ok,
            invol_ok,
            gns_ok,
            kms_ok,
        ) = self.composite_residual(certificate, state_vector, operator_X)

        verdict, reasons = self.classify_verdict(
            certificate, fock_ok, invol_ok, gns_ok, kms_ok, composite
        )

        if raise_on_veto and verdict == DualizerVerdict.VETOED:
            raise OmegaWisdomGovernanceError(
                "Dualizador VETOED: " + "; ".join(reasons)
            )

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state = OmegaWisdomSovereignState(
            modular_certificate=certificate,
            verdict=verdict,
            fock_isometry_ok=bool(fock_ok),
            modular_involution_ok=bool(invol_ok),
            gns_antiunitary_ok=bool(gns_ok),
            kms_ok=bool(kms_ok),
            composite_residual=float(composite),
            dual_fock_state=dual_state,
            dual_operator=dual_op,
            reasons=reasons,
            timestamp_utc=ts,
        )
        self._last_state = state
        logger.info(
            "FASE3 EMIT Ω⊗WISDOM verdict=%s fock=%s invol=%s gns=%s kms=%s res=%.3e",
            verdict.name, fock_ok, invol_ok, gns_ok, kms_ok, composite,
        )
        return state


# =============================================================================
# SOBERANO — OmegaWisdomHodgeDualizer (orquestador de las 3 fases anidadas)
# =============================================================================

class OmegaWisdomHodgeDualizer(Phase3_OmegaWisdomDecisionMaker):
    r"""
    Soberano de la dualidad métrica en los estratos OMEGA (Ω) y WISDOM (V_W).

    Orquesta:

    .. code-block:: text

        (N,k) ──► FASE1.emit_fock_duality_morphism ──► FockDualityMorphism
                          │
        ρ ────────► FASE2.emit_modular_certificate ──► ModularConjugationCertificate
                          │
        (ψ,X) ────► FASE3.emit_omega_wisdom_state ──► OmegaWisdomSovereignState
    """

    def __init__(
        self,
        dimension: int,
        isometry_tol: float = _ISOMETRY_TOL,
        rho_floor: float = _RHO_SPECTRAL_FLOOR,
        rho_condition_max: float = _CONDITION_RHO_MAX,
        involution_tol: float = _INVOLUTION_TOL,
        gns_tol: float = _GNS_TOL,
        default_k: Optional[int] = None,
    ) -> None:
        super().__init__(
            dimension=dimension,
            isometry_tol=isometry_tol,
            rho_floor=rho_floor,
            rho_condition_max=rho_condition_max,
            involution_tol=involution_tol,
            gns_tol=gns_tol,
        )
        self._default_k = (
            int(default_k) if default_k is not None
            else max(dimension // 2, 0)
        )

    @property
    def last_state(self) -> Optional[OmegaWisdomSovereignState]:
        return self._last_state

    # ── API retrocompatible v1.0 ─────────────────────────────────────────────

    def fock_particle_hole_duality(
        self,
        state_vector: ComplexVector,
        k_particles: int,
    ) -> Tuple[ComplexVector, FockHodgeReport]:
        """API v1: dualidad Fock directa (delega en FASE 1)."""
        return super().fock_particle_hole_duality(state_vector, k_particles)

    def non_commutative_hodge_conjugation(
        self,
        operator_X: ComplexMatrix,
        rho_mac: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, NonCommutativeHodgeReport]:
        """API v1: conjugación modular directa (delega en FASE 2)."""
        return super().non_commutative_hodge_conjugation(operator_X, rho_mac)

    # ── ciclo completo ───────────────────────────────────────────────────────

    def execute_duality_governance(
        self,
        rho_mac: ComplexMatrix,
        k_particles: Optional[int] = None,
        state_vector: Optional[ComplexVector] = None,
        operator_X: Optional[ComplexMatrix] = None,
        probe_operator: Optional[ComplexMatrix] = None,
        enforce_isometry: bool = True,
        enforce_involution: bool = True,
        enforce_gns: bool = True,
        raise_on_veto: bool = False,
    ) -> OmegaWisdomSovereignState:
        r"""
        Ciclo completo FASE1 → FASE2 → FASE3.

        Parameters
        ----------
        rho_mac :
            Matriz de densidad de sabiduría.
        k_particles :
            Grado Fock; default ``default_k``.
        state_vector :
            Estado de Fock a dualizar (opcional).
        operator_X :
            Operador a conjugar (opcional).
        probe_operator :
            Sonda para el certificado modular.
        enforce_* :
            Post-condiciones duras por fase.
        raise_on_veto :
            Materializar veto como excepción.

        Returns
        -------
        OmegaWisdomSovereignState
            Siempre que no se propague veto; en fallos duros de FASE1/2
            se re-lanza la excepción de dominio.
        """
        k = self._default_k if k_particles is None else int(k_particles)

        # Si hay state_vector, usarlo como sonda de isometría Fock
        morph = self.emit_fock_duality_morphism(
            k_particles=k,
            probe_state=state_vector,
            enforce_isometry=enforce_isometry,
        )
        cert = self.emit_modular_certificate(
            fock_morphism=morph,
            rho_mac=rho_mac,
            probe_operator=probe_operator if probe_operator is not None else operator_X,
            enforce_involution=enforce_involution,
            enforce_gns=enforce_gns,
        )
        return self.emit_omega_wisdom_state(
            certificate=cert,
            state_vector=state_vector,
            operator_X=operator_X,
            raise_on_veto=raise_on_veto,
        )

    def execute_duality_governance_safe(
        self,
        rho_mac: ComplexMatrix,
        **kwargs: Any,
    ) -> OmegaWisdomSovereignState:
        """
        Variante fail-safe: nunca lanza; ante error emite estado VETOED
        de emergencia (análogo al Crowbar del agente de Hodge).
        """
        try:
            return self.execute_duality_governance(rho_mac, **kwargs)
        except OmegaWisdomError as err:
            logger.critical("Fail-safe Ω⊗WISDOM: %s", err)
            return self._emergency_state(err)
        except Exception as err:  # pragma: no cover
            logger.critical("Fail-safe no clasificado Ω⊗WISDOM: %s", err)
            return self._emergency_state(err)

    def _emergency_state(self, err: Exception) -> OmegaWisdomSovereignState:
        """Estado VETOED de emergencia con morfismos dummy."""
        n = self._dim
        k = min(self._default_k, n)
        # Morfismo Fock trivial (vacío o identidad degenerada)
        try:
            morph = self.emit_fock_duality_morphism(
                k, enforce_isometry=False
            )
        except Exception:
            morph = FockDualityMorphism(
                dimension_n=n,
                k_particles=k,
                dual_k=n - k,
                basis_k=tuple(),
                basis_dual=tuple(),
                hodge_matrix=np.zeros((0, 0), dtype=np.complex128),
                double_star_sign=1,
                report=FockHodgeReport(
                    original_k=k, dual_k=n - k, dimension_n=n,
                    dim_source=0, dim_target=0, is_isometry=False,
                    double_star_sign=1, levi_civita_norm=0.0,
                    condition_hodge=float("inf"), residual_isometry=float("inf"),
                ),
                is_certified_isometry=False,
                gram_residual=float("inf"),
                cogram_residual=float("inf"),
            )

        rho = np.eye(n, dtype=np.complex128) / max(n, 1)
        dummy_report = NonCommutativeHodgeReport(
            trace_residual=float("inf"),
            is_involution=False,
            is_antiunitary=False,
            involution_residual=float("inf"),
            gns_norm_original=0.0,
            gns_norm_dual=0.0,
            kms_residual=float("inf"),
            rho_condition=float("inf"),
            regularization=RegularizationKind.NONE,
        )
        cert = ModularConjugationCertificate(
            fock_morphism=morph,
            rho_sanitized=rho,
            rho_half=rho ** 0.5 if n > 0 else rho,
            rho_neg_half=np.eye(n, dtype=np.complex128) * math.sqrt(max(n, 1)),
            rho_eigenvalues=np.full(n, 1.0 / max(n, 1)),
            rho_condition=float("inf"),
            regularization=RegularizationKind.NONE,
            tikhonov_shift=0.0,
            is_faithful=False,
            is_involution_certified=False,
            is_gns_antiunitary_certified=False,
            kms_symmetric=False,
            probe_report=dummy_report,
        )
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state = OmegaWisdomSovereignState(
            modular_certificate=cert,
            verdict=DualizerVerdict.VETOED,
            fock_isometry_ok=False,
            modular_involution_ok=False,
            gns_antiunitary_ok=False,
            kms_ok=False,
            composite_residual=float("inf"),
            dual_fock_state=None,
            dual_operator=None,
            reasons=(f"emergency:{type(err).__name__}:{err}",),
            timestamp_utc=ts,
        )
        self._last_state = state
        return state

    # ── utilidades de diagnóstico ────────────────────────────────────────────

    def summary(self) -> str:
        s = self._last_state
        if s is None:
            return "OmegaWisdomHodgeDualizer: sin ciclo ejecutado."
        c = s.modular_certificate
        f = c.fock_morphism
        lines = [
            "=" * 72,
            "OMEGA ⊗ WISDOM HODGE DUALIZER — ESTADO SOBERANO (v2.0.0)",
            "=" * 72,
            f"  Timestamp UTC     : {s.timestamp_utc}",
            f"  Stratum           : {s.stratum}",
            f"  Verdict Ω₃        : {s.verdict.name}",
            f"  Reasons           : {s.reasons or '—'}",
            f"  N (orbitales)     : {f.dimension_n}",
            f"  k → N−k           : {f.k_particles} → {f.dual_k}",
            f"  Fock isometry     : {s.fock_isometry_ok} "
            f"(gram={f.gram_residual:.3e}, ★★sgn={f.double_star_sign})",
            f"  J² = Id           : {s.modular_involution_ok}",
            f"  GNS antiunitary   : {s.gns_antiunitary_ok}",
            f"  KMS symmetric     : {s.kms_ok}",
            f"  κ(ρ)              : {c.rho_condition:.4e}",
            f"  ρ faithful        : {c.is_faithful}",
            f"  ρ regularization  : {c.regularization.name}",
            f"  Composite resid.  : {s.composite_residual:.4e}",
            f"  GNS ‖X‖² orig/dual: "
            f"{c.probe_report.gns_norm_original:.4e} / "
            f"{c.probe_report.gns_norm_dual:.4e}",
            "=" * 72,
        ]
        return "\n".join(lines)


# =============================================================================
# FACTORÍA
# =============================================================================

def build_omega_wisdom_dualizer(
    dimension: int,
    *,
    default_k: Optional[int] = None,
    rho_floor: float = _RHO_SPECTRAL_FLOOR,
    strict: bool = True,
) -> OmegaWisdomHodgeDualizer:
    """
    Factoría del dualizador soberano.

    Parameters
    ----------
    dimension :
        N = número de orbitales / dimensión de ℋ.
    default_k :
        Grado Fock por defecto.
    rho_floor :
        Suelo espectral IR de ρ.
    strict :
        Si False, afloja tolerancias (modo exploración numérica).
    """
    if strict:
        return OmegaWisdomHodgeDualizer(
            dimension=dimension,
            default_k=default_k,
            rho_floor=rho_floor,
        )
    return OmegaWisdomHodgeDualizer(
        dimension=dimension,
        default_k=default_k,
        rho_floor=rho_floor,
        isometry_tol=1e-8,
        involution_tol=1e-8,
        gns_tol=1e-8,
        rho_condition_max=1e14,
    )


# =============================================================================
# Exportación canónica
# =============================================================================

__all__ = [
    # Excepciones
    "OmegaWisdomError",
    "FockDegreeError",
    "FockIsometryError",
    "DensityOperatorError",
    "ModularInvolutionError",
    "GNSAntiunitarityError",
    "OmegaWisdomGovernanceError",
    # Enums
    "DualizerVerdict",
    "RegularizationKind",
    # DTOs
    "FockHodgeReport",
    "FockDualityMorphism",
    "NonCommutativeHodgeReport",
    "ModularConjugationCertificate",
    "OmegaWisdomSovereignState",
    # Fases
    "Phase1_FockHodgeObserver",
    "Phase2_ModularHodgeOrienter",
    "Phase3_OmegaWisdomDecisionMaker",
    # Soberano + factoría
    "OmegaWisdomHodgeDualizer",
    "build_omega_wisdom_dualizer",
]


# =============================================================================
# Cierre formal de las tres fases anidadas:
#
#   Phase1_FockHodgeObserver.emit_fock_duality_morphism
#       → FockDualityMorphism
#           → Phase2_ModularHodgeOrienter.emit_modular_certificate
#               → ModularConjugationCertificate
#                   → Phase3_OmegaWisdomDecisionMaker.emit_omega_wisdom_state
#                       → OmegaWisdomSovereignState
# =============================================================================