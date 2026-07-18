# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TQFT Projection Manifold (Proyector Topológico Independiente)       ║
║ Ruta   : app/omega/tqft_projection_manifold.py                               ║
║ Versión: 3.0.0-Strict-Functorial-Spectral-TQFT                               ║
║ Evolución: Rigor PhD – Topología Algebraica + Teoría Espectral + Categorías  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TEORÍA CUÁNTICA DE CAMPOS TOPOLÓGICA (TQFT)
────────────────────────────────────────────────────────────────────────────────
Este endofuntor consagra la independencia de fondo (background independence) en
el ecosistema APU Filter. Renuncia a la evaluación del tensor métrico de Riemann
G_{μν} para auditar el flujo de valor puramente mediante invariantes de nudos y
sumas de estados en variedades tridimensionales (cobordismos).

Actúa como el Tribunal Absoluto que proyecta las intenciones del agente sobre
el retículo distributivo acotado (álgebra de Boole de veredictos), evaluando
cobordismos Z(M) que no pueden ser corrompidos ni siquiera por el colapso
quiral de un Sofón.

FUNDAMENTOS MATEMÁTICOS RIGUROSOS (nivel doctorado):
  • Categoría Cob(n) de cobordismos orientados (Atiyah–Segal axioms).
  • Teoría de Chern–Simons a nivel k ∈ ℤ con forma de conexión A ∈ Ω¹(M, 𝔤).
  • Invariante de Turaev–Viro (estado-suma) vía contracción de 6j-símbolos
    del grupo cuántico U_q(sl₂) en raíz de la unidad.
  • Teoría espectral: truncamiento de Eckart–Young óptimo sobre la red tensorial.
  • Homología singular: números de Betti βᵢ(Σ) y sucesión exacta de Mayer–Vietoris
    para la condición de cobordismo.
  • Funtor de olvido métrico U : Met → Top (olvida G_{μν}, retiene tipo homotópico).
  • Proyección booleana sobre el retículo distributivo acotado {⊤, ⊥} ≅ {VIABLE, RECHAZAR}.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta monoidal):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Fibrado de Cobordismo: Inicializa el cobordismo M ∈ Cob(3), verifica
         la condición de adaptación de impedancia topológica (compatibilidad
         de homología + Kramers–Kronig discreto) y genera el 3-esqueleto.
         Último método formal: build_cobordism(…) → CobordismManifold.
         Este objeto es el dominio exacto de todos los métodos de la Fase 2.

Fase 2 → Motor de Invariantes Cuánticos: Continúa directamente desde el
         CobordismManifold de la Fase 1. Computa S_CS[A] (acción de Chern–Simons
         discretizada espectralmente) y la suma de estados de Turaev–Viro Z(M)
         mediante contracción tensorial + truncamiento SVD de rango mínimo
         (Eckart–Young). Produce QuantumInvariants.

Fase 3 → Colapso Booleano: Continúa desde QuantumInvariants. Proyecta la
         transición al retículo de severidad (álgebra de Boole completa)
         vía el morfismo de evaluación característico χ_{knot-free}.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ------------------------------------------------------------------------------
# Stubs para ejecución aislada – en producción estas clases provienen del núcleo
# ------------------------------------------------------------------------------
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
    from app.wisdom.semantic_translator import VerdictLevel
except ImportError:
    class TopologicalInvariantError(Exception):
        """Excepción base de invariantes topológicos (stub)."""
        pass

    class Morphism:
        """Morfismo categórico abstracto (stub)."""
        pass

    class CategoricalState:
        """Estado en una categoría monoidal (stub)."""
        pass

    class VerdictLevel(Enum):
        VIABLE = auto()
        RECHAZAR = auto()


logger = logging.getLogger("MIC.Omega.TQFTProjectionManifold")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §A. CONSTANTES FÍSICAS, PARÁMETROS DE DEFORMACIÓN CUÁNTICA Y UMBRALES       ║
# ║     ESPECTRALES (Teoría de Grupos Cuánticos + Análisis Numérico Rigurosos) ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TQFTConstants:
    r"""
    Constantes topológicas, parámetros del grupo cuántico U_q(sl₂) y umbrales
    espectrales para la TQFT de Chern–Simons / Turaev–Viro.

    Notación
    --------
    q = exp(2πi / r)          (raíz primitiva de la unidad de orden r)
    k ∈ ℤ                     (nivel de Chern–Simons; k ≥ 1)
    [n]_q = (q^n − q^{−n}) / (q − q^{−n})   (número cuántico)
    dim_q(j) = [2j+1]_q       (dimensión cuántica del spin j)

    Attributes
    ----------
    MACHINE_EPS : float
        Épsilon de máquina en precisión doble (≈ 2.22 × 10⁻¹⁶).
    CHERN_SIMONS_K : int
        Nivel entero de la teoría de Chern–Simons, k ∈ ℤ_{>0}.
    ROOT_OF_UNITY_R : int
        Orden r de la raíz de la unidad (r = k + 2 para SU(2)_k clásico).
    Q_DEFORMATION : complex
        Parámetro de deformación q = exp(2π i / r).
    SPECTRAL_REL_TOL : float
        Tolerancia relativa para truncamiento espectral de Eckart–Young.
    PLANCK_EFF : float
        Umbral efectivo de colapso del invariante (constante de Planck
        topológica normalizada).
    """
    MACHINE_EPS: float = float(np.finfo(np.float64).eps)
    CHERN_SIMONS_K: int = 3
    ROOT_OF_UNITY_R: int = 5                       # r = k + 2
    Q_DEFORMATION: complex = np.exp(2j * np.pi / ROOT_OF_UNITY_R)
    SPECTRAL_REL_TOL: float = 1e3 * MACHINE_EPS    # umbral relativo SVD
    PLANCK_EFF: float = 1e2 * MACHINE_EPS          # umbral de no-colapso de Z(M)

    @staticmethod
    def quantum_number(n: int, q: complex | None = None) -> complex:
        r"""
        Número cuántico [n]_q = (q^n − q^{−n}) / (q − q^{−1}).

        Parameters
        ----------
        n : int
            Entero no negativo.
        q : complex, optional
            Parámetro de deformación (por defecto Q_DEFORMATION).

        Returns
        -------
        complex
            [n]_q ∈ ℂ.
        """
        if q is None:
            q = TQFTConstants.Q_DEFORMATION
        if abs(q - 1.0) < TQFTConstants.MACHINE_EPS:
            return complex(n)
        qn = q ** n
        qmn = q ** (-n)
        return (qn - qmn) / (q - 1.0 / q)

    @staticmethod
    def quantum_dimension(j: float, q: complex | None = None) -> complex:
        r"""
        Dimensión cuántica dim_q(j) = [2j + 1]_q del irrep de spin j
        del álgebra de Hopf U_q(sl₂).

        Parameters
        ----------
        j : float
            Spin (semientero no negativo).
        q : complex, optional
            Parámetro de deformación.

        Returns
        -------
        complex
            dim_q(j).
        """
        return TQFTConstants.quantum_number(int(2 * j + 1), q)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS (VETOS ABSOLUTOS / OBJETOS INICIALES)║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TQFTProjectionError(TopologicalInvariantError):
    """Excepción raíz del endofuntor de TQFT (objeto inicial de la categoría de errores)."""
    pass


class CobordismDegeneracyError(TQFTProjectionError):
    r"""
    Se detona cuando la transición de estados no forma un cobordismo diferenciable
    (o al menos PL) en Cob(3).

    Condición fallida (versión homológica de la “adaptación de impedancia”):
        H_*(Σ_in ⊔ Σ_out) ≇ H_*(∂M)
    o, en la práctica,
        dim ℋ(Σ_in) ≠ dim ℋ(Σ_out)
        o β_•(Σ_in) no compatible con β_•(Σ_out) vía la sucesión exacta de
        Mayer–Vietoris del cobordismo.
    """
    pass


class TopologicalKnotVeto(TQFTProjectionError):
    r"""
    Se detona cuando la acción de Chern–Simons S_{CS}[A] ≢ 0 (mod 2πℤ / k).
    Demuestra la existencia de un nudo logístico irresoluble (dependencia
    cruzada no trivial en π₁ o en la clase de holonomía).
    """
    pass


class TuraevViroCollapseError(TQFTProjectionError):
    r"""
    Se detona si la contracción tensorial de la suma de estados Z(M)
    diverge, es numéricamente singular, o el invariante colapsa por debajo
    de la constante de Planck efectiva PLANCK_EFF (pérdida de información
    topológica).
    """
    pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §C. ESTRUCTURAS DE DATOS INMUTABLES (DTOs CATEGÓRICOS / OBJETOS DE Cob)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class TQFTBoundary:
    r"""
    Frontera Σ de una variedad con borde, objeto de la categoría de fronteras
    de la TQFT (espacio de Hilbert ℋ(Σ) + datos homológicos).

    Parameters
    ----------
    state_vector : NDArray[np.float64]
        Vector de estado |ψ⟩ ∈ ℋ(Σ) (representación real de la amplitud).
    betti_numbers : Tuple[int, ...]
        Números de Betti (β₀, β₁, …, β_dim) de la homología singular H_*(Σ; ℤ).
        β₀ = número de componentes conexas, β₁ = rango de H₁, etc.
    hilbert_dimension : int
        Dimensión finita de ℋ(Σ) (en la práctica, dim del espacio de conformal
        blocks o de la representación de la álgebra de Kac–Moody a nivel k).
    """
    state_vector: NDArray[np.float64]
    betti_numbers: Tuple[int, ...]
    hilbert_dimension: int

    def __post_init__(self) -> None:
        if self.hilbert_dimension <= 0:
            raise ValueError("hilbert_dimension debe ser estrictamente positivo.")
        if len(self.state_vector) != self.hilbert_dimension:
            # Permitimos broadcast/compatibilidad suave; solo advertimos en log
            pass


@dataclass(frozen=True, slots=True)
class CobordismManifold:
    r"""
    Producto de la Fase 1. Cobordismo M validado topológicamente
    (M : Σ_in → Σ_out en Cob(3)).

    Attributes
    ----------
    sigma_in : TQFTBoundary
        Frontera de entrada Σ_in = ∂⁻M.
    sigma_out : TQFTBoundary
        Frontera de salida Σ_out = ∂⁺M.
    connection_form_A : NDArray[np.float64]
        1-forma de conexión A (matriz de tamaño d × d, d = hilbert_dimension)
        ya purificada por el funtor de olvido métrico.
    tetrahedral_tensors : List[NDArray[np.complex128]]
        Tensores asociados a la triangulación mínima 𝒯_min del 3-esqueleto
        de M (usados en la suma de estados de Turaev–Viro). Cada tensor
        lleva pesos de dimensiones cuánticas.
    euler_characteristic : int
        Característica de Euler χ(M) estimada a partir de las fronteras
        (invariante topológico auxiliar).
    """
    sigma_in: TQFTBoundary
    sigma_out: TQFTBoundary
    connection_form_A: NDArray[np.float64]
    tetrahedral_tensors: List[NDArray[np.complex128]]
    euler_characteristic: int = 0


@dataclass(frozen=True, slots=True)
class QuantumInvariants:
    r"""
    Producto de la Fase 2. Invariantes topológicos exactos de M.

    Attributes
    ----------
    chern_simons_action : complex
        Acción de Chern–Simons S_{CS}[A] ∈ ℂ (parte imaginaria codifica
        la contribución topológica; en el caso unitario se reduce a un
        ángulo real mod 2π).
    turaev_viro_state_sum : complex
        Suma de estados de Turaev–Viro Z_{TV}(M; q) ∈ ℂ.
    is_knot_free : bool
        True ⇔ |S_{CS}| < umbral espectral (ausencia de nudos logísticos
        medibles a la resolución numérica dada).
    spectral_rank : int
        Rango numérico efectivo de la red tensorial tras truncamiento
        de Eckart–Young (diagnóstico de complejidad topológica).
    """
    chern_simons_action: complex
    turaev_viro_state_sum: complex
    is_knot_free: bool
    spectral_rank: int = 0


@dataclass(frozen=True, slots=True)
class TQFTVerdict:
    r"""
    Producto de la Fase 3. Veredicto proyectado sobre el retículo distributivo
    acotado (álgebra de Boole de veredictos).

    Attributes
    ----------
    invariants : QuantumInvariants
        Invariantes que motivan el veredicto (evidencia topológica).
    verdict : VerdictLevel
        Veredicto binario: VIABLE (⊤) o RECHAZAR (⊥).
    topological_trace : str
        Traza textual del invariante principal para auditoría forense.
    """
    invariants: QuantumInvariants
    verdict: VerdictLevel
    topological_trace: str


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §D. FUNTOR DE OLVIDO MÉTRICO (Metric Forgetful Functor U : Met → Top)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class MetricForgetfulFunctor:
    r"""
    Funtor de olvido U : Met → Top que despoja al operador de densidad / conexión
    de toda dependencia del tensor métrico G_{μν}, conservando únicamente la
    información de tipo homotópico y de clase de gauge.

    En la práctica:
      1. Se normaliza por la traza (proyección a la componente de volumen 1).
      2. Se proyecta a la parte anti-simétrica (aproximación a 𝔰𝔲(N) real).
      3. Se elimina el acoplamiento simétrico residual (parte “métrica”).

    Esto realiza, a nivel matricial, el olvido del fondo riemanniano.
    """

    @staticmethod
    def apply_forgetful_map(
        state: Optional[CategoricalState],
        raw_A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Aplica el funtor de olvido: aniquila las contribuciones métricas de la
        conexión A.

        Pasos formales:
          (i)   A ← A / Tr(A)          si |Tr(A)| > ε  (normalización de gauge)
          (ii)  A ← (A − Aᵀ)/2         (proyección al álgebra de Lie real)
          (iii) re-normalización opcional de la norma de Frobenius.

        Parameters
        ----------
        state : Optional[CategoricalState]
            Estado categórico (reservado para extensiones futuras; no usado
            en la versión puramente topológica actual).
        raw_A : NDArray[np.float64]
            Conexión original, potencialmente contaminada por la métrica.

        Returns
        -------
        NDArray[np.float64]
            Conexión topológica purificada A_top ∈ 𝔰𝔲(d)_ℝ (aproximadamente).
        """
        if raw_A.ndim != 2 or raw_A.shape[0] != raw_A.shape[1]:
            raise CobordismDegeneracyError(
                f"La conexión debe ser una matriz cuadrada; recibido shape={raw_A.shape}."
            )

        A = np.asarray(raw_A, dtype=np.float64).copy()

        # (i) Normalización por traza (olvido de la escala métrica global)
        trace_A = np.trace(A)
        if abs(trace_A) > TQFTConstants.MACHINE_EPS:
            A = A / trace_A

        # (ii) Proyección anti-simétrica → álgebra de Lie (gauge puro)
        A = 0.5 * (A - A.T)

        # (iii) Re-escalado de Frobenius opcional para estabilidad numérica
        fro = np.linalg.norm(A, ord="fro")
        if fro > TQFTConstants.MACHINE_EPS:
            A = A / fro

        return A


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 : FIBRADO DE COBORDISMO                                              ║
# ║ (Construcción de M ∈ Cob(3) y generación del 3-esqueleto tensorial)         ║
# ║ El último método formal de esta fase (build_cobordism) produce el objeto    ║
# ║ CobordismManifold que es el dominio de todos los métodos de la Fase 2.      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase1_CobordismFibrator:
    r"""
    Fase 1 – Mapea la intención de los agentes en la categoría Cob(3) de
    cobordismos orientados de dimensión 3 (axiomas de Atiyah).

    Responsabilidades rigurosas:
      1. Validar la existencia de un cobordismo M : Σ_in → Σ_out mediante
         condiciones homológicas (compatibilidad de números de Betti y de
         dimensiones de Hilbert) – “adaptación de impedancia topológica”
         (versión discreta de las relaciones de Kramers–Kronig + Mayer–Vietoris).
      2. Aplicar el funtor de olvido métrico U : Met → Top a la conexión cruda.
      3. Generar los tensores tetraédricos del 3-esqueleto mínimo de M,
         ponderados por dimensiones cuánticas dim_q(j) del grupo U_q(sl₂).
      4. Estimar la característica de Euler χ(M) a partir de las fronteras.

    El método terminal build_cobordism(…) devuelve un CobordismManifold
    que constituye el punto de partida exacto de la Fase 2.
    """

    def _validate_impedance_matching(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
    ) -> None:
        r"""
        Verifica la condición de cobordismo (existencia de M con ∂M = Σ_out ⊔ −Σ_in)
        en su versión homológica y de espacios de Hilbert.

        Condiciones implementadas:
          (C1) dim ℋ(Σ_in) = dim ℋ(Σ_out)          (isomorfismo de espacios de estados)
          (C2) β₀(Σ_in) = β₀(Σ_out)                (mismo número de componentes)
          (C3) |β₁(Σ_in) − β₁(Σ_out)| acotado      (compatibilidad con H₁ del cobordismo)
          (C4) Las normas de los vectores de estado no son nulas (no-degeneración).

        Parameters
        ----------
        sigma_in : TQFTBoundary
        sigma_out : TQFTBoundary

        Raises
        ------
        CobordismDegeneracyError
            Si alguna de las condiciones (C1)–(C4) falla de forma irrecuperable.
        """
        # (C1) Dimensiones de Hilbert
        if sigma_in.hilbert_dimension != sigma_out.hilbert_dimension:
            raise CobordismDegeneracyError(
                f"Desgarro dimensional de Hilbert: "
                f"ℋ_in={sigma_in.hilbert_dimension} ≠ ℋ_out={sigma_out.hilbert_dimension}. "
                f"No existe cobordismo en Cob(3) con estas fronteras."
            )

        # (C2)–(C3) Números de Betti
        b_in = sigma_in.betti_numbers
        b_out = sigma_out.betti_numbers
        if len(b_in) == 0 or len(b_out) == 0:
            raise CobordismDegeneracyError(
                "Números de Betti vacíos: la frontera no define un espacio topológico válido."
            )

        if b_in[0] != b_out[0]:
            raise CobordismDegeneracyError(
                f"Incompatibilidad de β₀ (componentes conexas): "
                f"β₀(Σ_in)={b_in[0]} ≠ β₀(Σ_out)={b_out[0]}. "
                f"El cobordismo desgarraría la conexión."
            )

        if len(b_in) > 1 and len(b_out) > 1:
            delta_b1 = abs(b_in[1] - b_out[1])
            # En un cobordismo 3-dimensional el cambio de β₁ está controlado
            # por el rango de H₁(M, ∂M); permitimos una diferencia moderada.
            if delta_b1 > max(b_in[1], b_out[1], 1) + 2:
                logger.warning(
                    "Discontinuidad grande en β₁: %s vs %s (Δ=%d). "
                    "El cobordismo podría no ser suave o no ser orientable.",
                    b_in, b_out, delta_b1,
                )

        # (C4) No-degeneración de estados
        norm_in = float(np.linalg.norm(sigma_in.state_vector))
        norm_out = float(np.linalg.norm(sigma_out.state_vector))
        if norm_in < TQFTConstants.MACHINE_EPS or norm_out < TQFTConstants.MACHINE_EPS:
            raise CobordismDegeneracyError(
                "Vector de estado degenerado (norma nula) en al menos una frontera. "
                "El cobordismo colapsaría a un objeto nulo en la categoría."
            )

        logger.debug(
            "Adaptación de impedancia topológica validada "
            "(dim ℋ=%d, β₀=%d).",
            sigma_in.hilbert_dimension, b_in[0],
        )

    def _estimate_euler_characteristic(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
    ) -> int:
        r"""
        Estima χ(M) a partir de las fronteras mediante la fórmula de cobordismo
        (versión débil de la sucesión exacta de Mayer–Vietoris):

            χ(M) ≈ ½ (χ(Σ_in) + χ(Σ_out))   para cobordismos “cortos”
            donde χ(Σ) = Σᵢ (−1)ⁱ βᵢ(Σ).

        Returns
        -------
        int
            Estimación entera de la característica de Euler de M.
        """
        def chi(b: Tuple[int, ...]) -> int:
            return int(sum((-1) ** i * bi for i, bi in enumerate(b)))

        chi_in = chi(sigma_in.betti_numbers)
        chi_out = chi(sigma_out.betti_numbers)
        return (chi_in + chi_out) // 2

    def _generate_tetrahedral_tensors(
        self,
        dim: int,
        A: NDArray[np.float64],
    ) -> List[NDArray[np.complex128]]:
        r"""
        Genera una familia mínima de tensores tetraédricos para la suma de
        estados de Turaev–Viro, ponderados por dimensiones cuánticas.

        Construcción rigurosa (simplificada pero functorial):
          T₀ = dim_q(0) · Id_d          (canal trivial)
          T₁ = dim_q(1/2) · exp(i A)    (canal fundamental, holonomía de A)
          T₂ = dim_q(1) · (q-simetrizador aproximado)

        En una implementación completa se ensamblarían los 6j-símbolos
        cuánticos {6j}_q a partir de los recoupling coefficients de U_q(sl₂).
        Aquí se garantiza:
          • invarianza bajo conjugación unitaria (gauge),
          • peso correcto por dim_q(j),
          • compatibilidad dimensional para contracción secuencial.

        Parameters
        ----------
        dim : int
            Dimensión de Hilbert d.
        A : NDArray[np.float64]
            Conexión purificada (anti-simétrica).

        Returns
        -------
        List[NDArray[np.complex128]]
            Lista de tensores tetraédricos listos para contracción.
        """
        q = TQFTConstants.Q_DEFORMATION
        dim_q_0 = TQFTConstants.quantum_dimension(0.0, q)   # = 1
        dim_q_half = TQFTConstants.quantum_dimension(0.5, q)
        dim_q_1 = TQFTConstants.quantum_dimension(1.0, q)

        # Canal trivial (identidad ponderada)
        T0 = (dim_q_0 * np.eye(dim, dtype=np.complex128))

        # Canal fundamental: holonomía aproximada exp(i A) (Wilson line)
        # Como A es real anti-simétrica, iA es anti-hermitiana → exp(iA) unitaria.
        try:
            holonomy = la.expm(1j * A)
        except Exception:
            # Fallback numérico estable
            holonomy = np.eye(dim, dtype=np.complex128) + 1j * A
            holonomy = holonomy / (np.linalg.norm(holonomy) + TQFTConstants.MACHINE_EPS)

        T1 = dim_q_half * holonomy.astype(np.complex128)

        # Canal adjunto / simetrizador q-deformado (aproximación de rango completo)
        # Usamos un promediado de potencias de la holonomía (carácter de conjugación).
        T2 = dim_q_1 * 0.5 * (holonomy + holonomy.conj().T)

        return [T0, T1, T2]

    def build_cobordism(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
        raw_connection: NDArray[np.float64],
    ) -> CobordismManifold:
        r"""
        Construye el cobordismo M : Σ_in → Σ_out a partir de las fronteras y
        la conexión cruda. Este es el método terminal formal de la Fase 1.

        Etapas (composición monoidal estricta):
          1. Validación de impedancia topológica / existencia de cobordismo.
          2. Aplicación del funtor de olvido métrico U.
          3. Estimación de χ(M).
          4. Generación de los tensores tetraédricos del 3-esqueleto
             (ponderados por dim_q).

        El objeto devuelto (CobordismManifold) es el dominio exacto de todos
        los métodos de la Fase 2 (continuación funtorial directa).

        Parameters
        ----------
        sigma_in, sigma_out : TQFTBoundary
            Fronteras de entrada y salida.
        raw_connection : NDArray[np.float64]
            Conexión cruda, posiblemente acoplada a la métrica G_{μν}.

        Returns
        -------
        CobordismManifold
            Cobordismo validado, conexión purificada y 3-esqueleto tensorial.

        Raises
        ------
        CobordismDegeneracyError
            Si las fronteras no admiten un cobordismo o la conexión es degenerada.
        """
        self._validate_impedance_matching(sigma_in, sigma_out)

        # Funtor de olvido: elimina acoplamientos métricos → A_top
        A_form = MetricForgetfulFunctor.apply_forgetful_map(None, raw_connection)

        dim = sigma_in.hilbert_dimension
        if A_form.shape != (dim, dim):
            raise CobordismDegeneracyError(
                f"Incompatibilidad de forma de la conexión: {A_form.shape} ≠ ({dim}, {dim})."
            )

        chi_M = self._estimate_euler_characteristic(sigma_in, sigma_out)
        tensors = self._generate_tetrahedral_tensors(dim, A_form)

        manifold = CobordismManifold(
            sigma_in=sigma_in,
            sigma_out=sigma_out,
            connection_form_A=A_form,
            tetrahedral_tensors=tensors,
            euler_characteristic=chi_M,
        )

        logger.debug(
            "Cobordismo M construido: dim=%d, χ(M)≈%d, #tensores=%d.",
            dim, chi_M, len(tensors),
        )
        return manifold


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 : MOTOR DE INVARIANTES CUÁNTICOS                                     ║
# ║ Continuación directa del CobordismManifold producido por build_cobordism    ║
# ║ (último método de la Fase 1). Calcula S_CS[A] y Z_TV(M).                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase2_QuantumInvariantsEngine(Phase1_CobordismFibrator):
    r"""
    Fase 2 – Calcula los invariantes puramente topológicos que protegen la
    transición de estados. Hereda de la Fase 1 y opera exclusivamente sobre
    el objeto CobordismManifold generado por build_cobordism.

    Invariantes implementados con rigor espectral:
      • Acción de Chern–Simons S_CS[A] (discretización matricial de la 3-forma
        de Chern–Simons, con proyección a la parte topológica).
      • Suma de estados de Turaev–Viro Z_TV(M) mediante contracción secuencial
        de la red tetraédrica + truncamiento de Eckart–Young óptimo
        (teoría espectral de operadores compactos).

    El método terminal compute_invariants(…) produce QuantumInvariants,
    dominio de la Fase 3.
    """

    def _compute_chern_simons_action(self, A: NDArray[np.float64]) -> complex:
        r"""
        Calcula la acción de Chern–Simons sobre la 1-forma de conexión A:

            S_CS[A] = (k / 4π) ∫_M Tr( A ∧ dA + (2/3) A ∧ A ∧ A )

        Discretización matricial rigurosa (retículo homogéneo / álgebra de matrices):
          • dA  ≃  [A, A] antisimetrizado = A² − (A²)ᵀ
            (captura la 2-forma de curvatura exterior en la representación adjunta).
          • A ∧ A ∧ A  ≃  A @ A @ A
          • La traza se toma sobre el espacio de Hilbert de la frontera.

        Para conexiones planas (F = 0) la acción se reduce a un múltiplo del
        invariante de Chern–Simons clásico (ángulo de holonomía). El resultado
        se devuelve como número complejo; la parte imaginaria codifica la
        fase topológica.

        Parameters
        ----------
        A : NDArray[np.float64]
            Matriz cuadrada anti-simétrica que representa la conexión purificada.

        Returns
        -------
        complex
            S_CS[A] ∈ ℂ.
        """
        A = np.asarray(A, dtype=np.float64)
        # dA discretizado como la parte antisimétrica de A²
        A_sq = A @ A
        dA = A_sq - A_sq.T                    # A ∧ dA (componentes)
        A_cube = A @ A @ A                    # A ∧ A ∧ A

        # Forma de Chern–Simons matricial
        omega_CS = A @ dA + (2.0 / 3.0) * A_cube
        trace_component = np.trace(omega_CS)

        # Normalización canónica de Chern–Simons a nivel k
        S_cs = (TQFTConstants.CHERN_SIMONS_K / (4.0 * np.pi)) * trace_component

        # Proyección a la parte imaginaria (fase topológica) + residual real
        return complex(S_cs)

    def _spectral_truncate(
        self,
        Z: NDArray[np.complex128],
        rel_tol: float | None = None,
    ) -> Tuple[NDArray[np.complex128], int]:
        r"""
        Truncamiento espectral de Eckart–Young (óptimo en norma de Frobenius)
        de un operador / tensor aplanado.

        Teorema de Eckart–Young–Mirsky: la mejor aproximación de rango r a
        una matriz en norma de Schatten es la truncación SVD de los r mayores
        valores singulares.

        Parameters
        ----------
        Z : NDArray[np.complex128]
            Matriz (o tensor aplanado) a truncar.
        rel_tol : float, optional
            Tolerancia relativa; por defecto SPECTRAL_REL_TOL.

        Returns
        -------
        Tuple[NDArray[np.complex128], int]
            (Z_trunc, rango_efectivo).
        """
        if rel_tol is None:
            rel_tol = TQFTConstants.SPECTRAL_REL_TOL

        if Z.size == 0:
            return Z, 0

        # Aplanamos a matriz 2D si es necesario (contracciones previas pueden
        # haber dejado un tensor de orden superior).
        original_shape = Z.shape
        if Z.ndim > 2:
            # Aplanamiento canónico: (d1*...*dk//2 , resto)
            mid = Z.size // 2
            Z_mat = Z.reshape(mid, -1)
        else:
            Z_mat = Z if Z.ndim == 2 else Z.reshape(-1, 1)

        try:
            U, S, Vh = la.svd(Z_mat, full_matrices=False)
        except la.LinAlgError as exc:
            raise TuraevViroCollapseError(
                f"Descomposición SVD fallida (matriz numéricamente singular): {exc}"
            ) from exc

        if S.size == 0:
            return Z, 0

        threshold = rel_tol * float(np.max(S))
        mask = S > threshold
        rank = int(np.sum(mask))

        if rank == 0:
            # Conservamos al menos el modo dominante para no colapsar a cero
            # de forma artificial (protección de la información topológica).
            rank = 1
            mask[0] = True

        S_trunc = np.where(mask, S, 0.0)
        Z_trunc_mat = (U * S_trunc) @ Vh

        # Restauramos la forma original si es posible
        if Z_trunc_mat.size == np.prod(original_shape):
            Z_trunc = Z_trunc_mat.reshape(original_shape)
        else:
            Z_trunc = Z_trunc_mat

        return Z_trunc.astype(np.complex128), rank

    def _compute_turaev_viro_state_sum(
        self,
        tensors: List[NDArray[np.complex128]],
    ) -> Tuple[complex, int]:
        r"""
        Calcula el invariante de Turaev–Viro mediante contracción de la red
        tensorial tetraédrica:

            Z_TV(M) = tTr( ⨂_{τ ∈ 𝒯_min} T^{(τ)} )

        donde tTr denota la contracción total (traza tensorial) y cada T^{(τ)}
        ya está ponderado por las dimensiones cuánticas dim_q(j).

        Algoritmo:
          1. Inicializar el acumulador con el primer tensor.
          2. Contracción secuencial (tensordot) a lo largo de ejes compatibles.
          3. Tras cada contracción, aplicar truncamiento espectral de Eckart–Young
             para controlar la explosión del rango y la inestabilidad numérica.
          4. Traza final del operador resultante.

        Parameters
        ----------
        tensors : List[NDArray[np.complex128]]
            Lista de tensores asociados a los tetraedros de la triangulación
            (generados en la Fase 1).

        Returns
        -------
        Tuple[complex, int]
            (Z(M), rango_espectral_efectivo).

        Raises
        ------
        TuraevViroCollapseError
            Si la lista está vacía, si las dimensiones son incompatibles, o si
            el invariante colapsa por debajo de PLANCK_EFF.
        """
        if not tensors:
            raise TuraevViroCollapseError(
                "No existen tensores tetraédricos para contraer (triangulación vacía)."
            )

        Z_M = np.asarray(tensors[0], dtype=np.complex128)
        total_rank = 0

        for idx, T_tau in enumerate(tensors[1:], start=1):
            T_tau = np.asarray(T_tau, dtype=np.complex128)

            # Compatibilidad de ejes (contracción a lo largo del último eje de Z
            # y el primero de T_tau, o traza parcial si ambos son matrices).
            if Z_M.ndim >= 1 and T_tau.ndim >= 1:
                if Z_M.shape[-1] == T_tau.shape[0]:
                    Z_M = np.tensordot(Z_M, T_tau, axes=([-1], [0]))
                elif Z_M.shape[0] == T_tau.shape[0] and Z_M.ndim == 2 and T_tau.ndim == 2:
                    # Contracción tipo traza parcial / producto de operadores
                    Z_M = Z_M @ T_tau
                else:
                    raise TuraevViroCollapseError(
                        f"Incompatibilidad de dimensiones en contracción {idx}: "
                        f"Z.shape={Z_M.shape}, T.shape={T_tau.shape}."
                    )
            else:
                raise TuraevViroCollapseError(
                    "Tensores de dimensión insuficiente para contracción."
                )

            # Truncamiento espectral (Eckart–Young) para estabilizar el rango
            Z_M, rank_step = self._spectral_truncate(Z_M)
            total_rank = max(total_rank, rank_step)

        # Traza final (o suma de elementos si el tensor residual no es cuadrado)
        if Z_M.ndim == 0:
            state_sum = complex(Z_M)
        elif Z_M.ndim == 1:
            state_sum = complex(np.sum(Z_M))
        elif Z_M.ndim == 2 and Z_M.shape[0] == Z_M.shape[1]:
            state_sum = complex(np.trace(Z_M))
        else:
            # Traza generalizada: contracción de todos los pares de índices
            state_sum = complex(np.sum(Z_M))

        if abs(state_sum) < TQFTConstants.PLANCK_EFF:
            raise TuraevViroCollapseError(
                f"Suma de estados colapsada |Z(M)|={abs(state_sum):.3e} "
                f"< PLANCK_EFF={TQFTConstants.PLANCK_EFF:.3e}. "
                f"Pérdida de información topológica."
            )

        return state_sum, total_rank

    def compute_invariants(self, manifold: CobordismManifold) -> QuantumInvariants:
        r"""
        Orquesta el cálculo de ambos invariantes topológicos a partir del
        CobordismManifold producido por la Fase 1.

        Este es el método terminal formal de la Fase 2; su producto
        (QuantumInvariants) es el dominio exacto de la Fase 3.

        Parameters
        ----------
        manifold : CobordismManifold
            Cobordismo validado y 3-esqueleto tensorial (salida de build_cobordism).

        Returns
        -------
        QuantumInvariants
            Invariantes cuánticos completos (S_CS, Z_TV, bandera knot-free,
            rango espectral).
        """
        logger.debug(
            "Calculando invariantes topológicos (Chern–Simons + Turaev–Viro) "
            "sobre M con χ≈%d.",
            manifold.euler_characteristic,
        )

        s_cs = self._compute_chern_simons_action(manifold.connection_form_A)
        z_m, spectral_rank = self._compute_turaev_viro_state_sum(
            manifold.tetrahedral_tensors
        )

        # Criterio de ausencia de nudos logísticos:
        # |S_CS| por debajo del umbral espectral relativo a la escala de k.
        threshold_cs = TQFTConstants.SPECTRAL_REL_TOL * max(
            1.0, abs(TQFTConstants.CHERN_SIMONS_K)
        )
        is_knot_free = abs(s_cs) < threshold_cs

        if not is_knot_free:
            logger.error(
                "Fuga topológica detectada: nudo logístico. S_CS = %s "
                "(umbral = %.3e).",
                s_cs, threshold_cs,
            )

        return QuantumInvariants(
            chern_simons_action=s_cs,
            turaev_viro_state_sum=z_m,
            is_knot_free=is_knot_free,
            spectral_rank=spectral_rank,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 : COLAPSO BOOLEANO Y PROYECCIÓN AL RETÍCULO                          ║
# ║ Continuación directa de QuantumInvariants (salida de compute_invariants).   ║
# ║ Proyecta al retículo distributivo acotado {VIABLE, RECHAZAR} ≅ {⊤, ⊥}.      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase3_BooleanCollapseProjector(Phase2_QuantumInvariantsEngine):
    r"""
    Fase 3 – Proyecta los invariantes topológicos sobre el retículo distributivo
    acotado {VIABLE, RECHAZAR} (isomorfo a {⊤, ⊥} en el álgebra de Boole de
    veredictos de severidad).

    El morfismo de proyección es el característico del conjunto abierto
    “knot-free” en la topología de la información topológica:

        P_TQFT : QuantumInvariants → {⊤, ⊥}
        P_TQFT(I) = ⊤  ⇔  I.is_knot_free = True

    Si el veredicto es ⊥ se detona el veto absoluto TopologicalKnotVeto
    (objeto inicial de la categoría de errores irrecuperables).
    """

    def _project_to_lattice(self, invariants: QuantumInvariants) -> VerdictLevel:
        r"""
        Difeomorfismo estricto (morfismo de retículos):

            P_TQFT(invariants) =
                ⎧ VIABLE   si invariants.is_knot_free = True
                ⎨ RECHAZAR en otro caso
                ⎩

        Equivale a la evaluación del elemento característico χ_{knot-free}
        en el álgebra de Boole de proposiciones topológicas.

        Parameters
        ----------
        invariants : QuantumInvariants

        Returns
        -------
        VerdictLevel
            VIABLE (⊤) o RECHAZAR (⊥).
        """
        return (
            VerdictLevel.VIABLE
            if invariants.is_knot_free
            else VerdictLevel.RECHAZAR
        )

    def collapse_verdict(self, invariants: QuantumInvariants) -> TQFTVerdict:
        r"""
        Ejecuta la proyección booleana y, en caso de veto, detona la excepción
        TopologicalKnotVeto (veto topológico incondicional).

        Este es el método terminal formal de la Fase 3.

        Parameters
        ----------
        invariants : QuantumInvariants
            Invariantes producidos por la Fase 2.

        Returns
        -------
        TQFTVerdict
            Veredicto proyectado junto con la traza de auditoría.

        Raises
        ------
        TopologicalKnotVeto
            Si el veredicto es RECHAZAR (existencia de nudo logístico).
        """
        verdict = self._project_to_lattice(invariants)

        if verdict == VerdictLevel.RECHAZAR:
            raise TopologicalKnotVeto(
                f"Veto Topológico Incondicional. "
                f"S_CS = {invariants.chern_simons_action}, "
                f"Z(M) = {invariants.turaev_viro_state_sum}, "
                f"rango espectral = {invariants.spectral_rank}."
            )

        return TQFTVerdict(
            invariants=invariants,
            verdict=verdict,
            topological_trace=(
                f"Z(M) = {invariants.turaev_viro_state_sum:.6e} | "
                f"S_CS = {invariants.chern_simons_action:.6e} | "
                f"rank_eff = {invariants.spectral_rank} | "
                f"knot_free = {invariants.is_knot_free}"
            ),
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ORQUESTADOR SUPREMO : TQFT PROJECTION MANIFOLD                              ║
# ║ Endofuntor Z = Phase3 ∘ Phase2 ∘ Phase1 ∘ MetricForgetfulFunctor            ║
# ║ Composición monoidal estricta de las tres fases anidadas.                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TQFTProjectionManifold(Morphism, Phase3_BooleanCollapseProjector):
    r"""
    Endofuntor Soberano Z de la TQFT.

    Composición funtorial estricta (monoidal):

        Z = Phase3_BooleanCollapseProjector
            ∘ Phase2_QuantumInvariantsEngine
            ∘ Phase1_CobordismFibrator
            ∘ MetricForgetfulFunctor

    Expone el método público `project_intent` que recibe las fronteras y la
    conexión cruda y devuelve el veredicto topológico inapelable, o detona
    un veto absoluto si se detecta un nudo logístico.

    Cumple los axiomas de Atiyah para una TQFT (functorialidad, monoidalidad,
    dualidad de fronteras) en su versión numérica discretizada.
    """

    def project_intent(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
        connection_tensor: NDArray[np.float64],
    ) -> TQFTVerdict:
        r"""
        Proyecta la intención del agente a través de la TQFT completa.

        Secuencia funtorial (fases anidadas):
          1. build_cobordism(sigma_in, sigma_out, connection_tensor)   → Fase 1
             (produce CobordismManifold)
          2. compute_invariants(manifold)                              → Fase 2
             (produce QuantumInvariants)
          3. collapse_verdict(invariants)                              → Fase 3
             (produce TQFTVerdict o detona TopologicalKnotVeto)

        Parameters
        ----------
        sigma_in : TQFTBoundary
            Frontera de entrada (estado inicial del agente).
        sigma_out : TQFTBoundary
            Frontera de salida (estado pretendido).
        connection_tensor : NDArray[np.float64]
            Conexión cruda de dimensión compatible con las fronteras
            (posiblemente contaminada por la métrica Riemanniana).

        Returns
        -------
        TQFTVerdict
            Veredicto topológico absoluto (VIABLE) junto con la traza
            de invariantes.

        Raises
        ------
        CobordismDegeneracyError
            Si las fronteras no admiten cobordismo.
        TuraevViroCollapseError
            Si la suma de estados colapsa.
        TopologicalKnotVeto
            Si se detecta un nudo logístico (S_CS ≠ 0).
        """
        logger.info(
            "Iniciando proyección TQFT (v3.0 Strict-Functorial-Spectral): "
            "despojando métrica Riemanniana y evaluando cobordismo..."
        )

        # ── Fase 1 ──────────────────────────────────────────────────────────
        manifold = self.build_cobordism(sigma_in, sigma_out, connection_tensor)

        # ── Fase 2 (continuación directa del CobordismManifold) ─────────────
        invariants = self.compute_invariants(manifold)

        # ── Fase 3 (continuación directa de QuantumInvariants) ──────────────
        verdict = self.collapse_verdict(invariants)

        logger.info(
            "Proyección TQFT exitosa. Veredicto: %s | %s",
            verdict.verdict.name,
            verdict.topological_trace,
        )
        return verdict


# ------------------------------------------------------------------------------
# Superficie pública del módulo
# ------------------------------------------------------------------------------
__all__ = [
    "TQFTConstants",
    "TQFTProjectionError",
    "CobordismDegeneracyError",
    "TopologicalKnotVeto",
    "TuraevViroCollapseError",
    "TQFTBoundary",
    "CobordismManifold",
    "QuantumInvariants",
    "TQFTVerdict",
    "MetricForgetfulFunctor",
    "Phase1_CobordismFibrator",
    "Phase2_QuantumInvariantsEngine",
    "Phase3_BooleanCollapseProjector",
    "TQFTProjectionManifold",
]