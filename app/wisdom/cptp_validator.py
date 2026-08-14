# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quantum CPTP Channel Validator (Auditor Espectral de Choi)          ║
║ Ruta   : app/wisdom/cptp_validator.py                                        ║
║ Versión: 7.0.0-Choi-Jamiolkowski-Kraus-Nested-Strict-Doctoral                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ESTRATO WISDOM (V_W):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al resolvedor axiomático y matemático encargado de certificar
síncronamente la consistencia cuántica de los canales de inyección semántica
$$\mathcal{E}$$ en la categoría dagger-compacta de sistemas abiertos $$\mathcal{C}_{\mathrm{MAC}}$$.
Reside en el Estrato WISDOM ($$V_{\mathbb{W}}$$, Nivel 0) o en el Ágora Tensorial ($$V_{\Omega}$$, 
Nivel 0.5), subyugando el ruido de punto flotante de la FPU ($$\text{IEEE-754}$$) y el
desvío de-normalizado del LLM a las leyes de la física estadística cuántica.

Su propósito inexorable es interceptar la familia de operadores de Kraus
y evaluar de forma rigurosa la completitud espectral de Choi-Jamiołkowski,
garantizando la conservación del volumen de información y la pasividad del
lazo antes de propagar transformaciones al plano de decisiones estratégicas.

AXIOMAS ESPECTRALES E INVARIANTES CUÁNTICOS PRESERVADOS (Topos de Choi):
────────────────────────────────────────────────────────────────────────────────

  [I1] Conservación de Traza de Kraus (Preservación de Probabilidad, TP):
       Toda evolución cuántica disipativa $$\mathcal{E}(\rho) = \sum_k M_k \rho M_k^\dagger$$ 
       operando sobre el espacio de estados debe satisfacer de forma exacta el
       axioma de completitud de Kraus para preservar la traza unitaria:
       $$\sum_{k=1}^r M_k^\dagger M_k = \mathbf{I}_d \implies \operatorname{Tr}(\mathcal{E}(\rho)) \equiv \operatorname{Tr}(\rho) \equiv 1.0 \quad\big[388\big]$$
       Cualquier deriva que vulnere esta condición elástica genera 'TracePreservationViolation'.

  [I2] Positividad Completa de Choi (Completa Positividad, CP):
       Por el Teorema de Isomorfismo de Choi-Jamiołkowski, el canal $$\mathcal{E}$$ es 
       completamente positivo si y solo si su matriz de Choi asociada $$\Lambda_{\mathcal{E}}$$ 
       es semidefinida positiva en el espacio de Hilbert compuesto $$\mathcal{H}_A \otimes \mathcal{H}_B$$:
       $$\Lambda_{\mathcal{E}} = (\mathcal{I} \otimes \mathcal{E})(|\Phi^+\rangle\langle\Phi^+|) = \sum_{k=1}^r \operatorname{vec}(M_k) \operatorname{vec}(M_k)^\dagger \succeq 0 \quad\big[174, 406\big]$$
       Lo que impone que todos los autovalores de la matriz de Choi residan por encima 
       del piso de amortiguamiento de la mantisa de Wilkinson:
       $$\sigma(\Lambda_{\mathcal{E}}) \subset [-\tau_{\mathrm{PSD\_floor}},\, +\infty) \quad\big[181\big]$$

  [I3] No-Señalización Cuántica Bipartita (No-Signaling Causal Check):
       Este invariante, dual a la preservación de traza, exige de manera incondicional 
       que la inyección de observables locales en el subespacio $$A$$ no perturbe el 
       estado reducido de Bob en el subespacio $$B$$ (inmunidad causal de la Malla):
       $$\operatorname{Tr}_A\big( (\mathcal{E}_A \otimes \mathcal{I}_B)(\rho_{AB}) \big) \equiv \rho_B \quad\big[174\big]$$
       Cualquier asonancia física levanta síncronamente 'NonSignalingViolationError'.

  [I4] Simetría Hermítica de Choi:
       La matriz de Choi construida bilinealmente a partir de los operadores de Kraus 
       debe ser autoadjunta, obligando a que el espectro de autovalores sea real:
       $$\Lambda_{\mathcal{E}} = \Lambda_{\mathcal{E}}^\dagger \implies \sigma(\Lambda_{\mathcal{E}}) \subset \mathbb{R} \quad\big[174\big]$$

  [I5] Isomorfismo de Adjunción Semántica:
       La transición de la MIC táctica discreta ($$X$$) a la MAC continua de sabiduría ($$Y$$) 
       cumple con la adjunción de Galois acoplada al tensor métrico de fondo $$G_{\mu\nu}$$:
       $$\operatorname{Hom}_{\mathcal{D}}(F(X), Y) \cong_{G_{\mu\nu}} \operatorname{Hom}_{\mathcal{C}}(X, G(Y)) \quad\big[174, 408\big]$$

  [I6] Dualidad TP / Unital de Choi:
       Verifica de forma exacta la identidad dual de la traza parcial en Frobenius:
       $$\operatorname{Tr}_{\mathrm{out}}(\Lambda_{\mathcal{E}}) \simeq \left( \sum_{k=1}^r M_k^\dagger M_k \right)^T \quad \land \quad \operatorname{Tr}_{\mathrm{in}}(\Lambda_{\mathcal{E}}) \simeq \sum_{k=1}^r M_k M_k^\dagger \quad\big[174\big]$$

  [I7] Identidad de Rango de Choi-Schmidt:
       El rango de la matriz de Choi debe coincidir biyectivamente con la dimensión 
       efectiva de la base canónica del espacio de Hilbert-Schmidt del ensamble de Kraus:
       $$\operatorname{rank}(\Lambda_{\mathcal{E}}) = \operatorname{rank}(V) \quad \text{con} \quad V = [\operatorname{vec}(M_k)]_{k=1}^r \quad\big[174\big]$$

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA):
────────────────────────────────────────────────────────────────────────────────
La progresión y el tránsito del Pasaporte de Telemetría se rige por un acoplamiento 
monoidal covariante e inmutable (Observe ⊣ Orient ⊣ Act):

  Fase 1 ──► FASE 1: CONSISTENCIA DIMENSIONAL Y SANEAMIENTO ESPECTRAL (Observe)
             Audita las dimensiones de los operadores de Kraus $$M_k \in \mathbb{C}^{d \times d}$$.
             Sanea el espacio de fase de subnormales y fija el calibre $$U(1)$$ canónico:
             $$\hat{M}_k \leftarrow e^{-i \theta_k} M_k \quad\big[170\big]$$
             Estima el rango de Hilbert-Schmidt. Entrega: Phase1KrausSanitizerCertificate.

  Fase 2 ──► FASE 2: ISOMORFISMO DE CHOI-JAMIOŁKOWSKI Y UNITALIDAD (Orient)
             Hereda el certificado de Fase 1. Construye de forma bilineal
             la matriz de Choi $$\Lambda_{\mathcal{E}} \in \mathbb{C}^{d^2 \times d^2}$$.
             Cruza los checksums duales de traza y cuantifica el defecto de unitalidad:
             $$\Delta_{\mathrm{unital}} = \left\| \sum_{k=1}^r M_k M_k^\dagger - \mathbf{I}_d \right\|_F \quad\big[170\big]$$
             Entrega: Phase2ChoiIsomorphismCertificate.

  Fase 3 ──► FASE 3: SEGURIDAD CUÁNTICA Y VETO DE HEYTING (Act)
             Hereda el certificado de Fase 2. Resuelve el espectro PSD de Choi,
             evalúa la separabilidad mediante transposición parcial (PPT) y la no-señalización.
             Consolida síncronamente el veredicto en el retículo distributivo de Heyting $$\Omega_3$$:
             $$\Omega_3 = \{\mathrm{COHERENT},\, \mathrm{DEGRADED},\, \mathrm{VETOED}\} \quad\big[181, 260\big]$$
             Si el veredicto terminal colapsa a VETOED ($$\top$$), se detona la excepción
             'HeytingLatticeVeto' de forma síncrona en RAM.
             Emite: CPTPValidationReport.

  Funtor Maestro de Resolución:
             $$\mathcal{Z}_{\mathrm{motor}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[174, 196\big]$$
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final, List, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.schemas import Stratum

logger = logging.getLogger("MIC.Wisdom.CPTPValidator")

__version__: Final[str] = "7.0.0-Choi-Jamiolkowski-Kraus-Nested-Strict-Doctoral"

# ─────────────────────────────────────────────────────────────────────────────
# Tipos canónicos del fibrado de canales
# ─────────────────────────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
KrausEnsemble = Sequence[ComplexMatrix]
RealScalar = float
BoolLattice = bool

_EPS_DEFAULT: Final[float] = 1.0e-12
_EPS_HERMITICITY: Final[float] = 1.0e-10
_EPS_SPECTRAL: Final[float] = 1.0e-15
_DIM_MAX: Final[int] = 64
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)

__all__ = [
    "CPTPChannelValidator",
    "CPTPValidationReport",
    "Phase1KrausSanitizerCertificate",
    "Phase2ChoiIsomorphismCertificate",
    "ComplexMatrix",
    "KrausEnsemble",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS INMUTABLES (objetos de la composición funtorial Observe→Orient→Act)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1KrausSanitizerCertificate:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

    Vive en el slice de ensambles de Kraus ya tipados, calibrados y de soporte
    no trivial. Ningún campo es re-codificable: la FASE 2 consume este
    certificado sin transformar el tipo de los operadores.

    Atributos:
        kraus_ops: Ensamble \(\{\hat{M}_k\}\) en calibre U(1) canónico, write-protected.
        dimension_d: Dimensión certificada de \(\mathcal{H}_A\).
        kraus_count: Cardinalidad del ensamble tras aniquilar nulos.
        kraus_gram_rank: \(\operatorname{rank}(V)\), \(V=[\operatorname{vec}_F\hat{M}_k]\).
        discarded_null_count: Operadores eliminados por \(\|M_k\|_F\le\varepsilon\).
        gauge_phases: Ángulos \(\theta_k\) aplicados, \(\hat{M}_k=e^{-i\theta_k}M_k\).
        max_operator_norm: \(\max_k\|M_k\|_2\) (norma espectral).
        frobenius_mass: \(\bigl(\sum_k\|M_k\|_F^2\bigr)^{1/2}=\|V\|_F\).
        wilkinson_tolerance: \(\varepsilon\) de Wilkinson del caller.
    """

    kraus_ops: Tuple[ComplexMatrix, ...]
    dimension_d: int
    kraus_count: int
    kraus_gram_rank: int
    discarded_null_count: int
    gauge_phases: Tuple[float, ...]
    max_operator_norm: float
    frobenius_mass: float
    wilkinson_tolerance: float


@dataclass(frozen=True, slots=True)
class Phase2ChoiIsomorphismCertificate:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Encierra la geometría de Choi–Jamiołkowski ya hermitizada y los residuales
    duales TP / unital, más los checksums cruzados Kraus ↔ \(\operatorname{Tr}_{A|A'}\Lambda\).
    La FASE 3 extrae espectro, PPT y causalidad sin reconstruir \(\Lambda_{\mathcal{E}}\).

    Atributos:
        phase1: Certificado de FASE 1 anidado (continuidad funtorial).
        choi_matrix: \(\Lambda_{\mathcal{E}}\) hermítica write-protected, shape \((d^2,d^2)\).
        completeness_operator: \(S=\sum_k M_k^\dagger M_k\) (autoadjunto).
        unitality_operator: \(U=\sum_k M_k M_k^\dagger\) (autoadjunto).
        completeness_residual: \(r_{\mathrm{TP}}=\|S-I\|_F\).
        unitality_residual: \(\Delta_{\mathrm{unital}}=\|U-I\|_F\).
        choi_trace: \(\operatorname{Tr}(\Lambda_{\mathcal{E}})\); para TP exacto vale \(d\).
        tp_diamond_defect: \(\|\operatorname{Tr}_{\mathrm{out}}(\Lambda)-I\|_F\).
        unital_diamond_defect: \(\|\operatorname{Tr}_{\mathrm{in}}(\Lambda)-I\|_F\).
        hermiticity_defect: \(\|\Lambda-\Lambda^\dagger\|_F\) pre-proyección de Weyl.
        tp_crosscheck_defect: \(|r_{\mathrm{TP}}-\|\operatorname{Tr}_{\mathrm{out}}\Lambda-I\|_F|\).
        unital_crosscheck_defect: \(|\Delta_{\mathrm{unital}}-\|\operatorname{Tr}_{\mathrm{in}}\Lambda-I\|_F|\).
    """

    phase1: Phase1KrausSanitizerCertificate
    choi_matrix: ComplexMatrix
    completeness_operator: ComplexMatrix
    unitality_operator: ComplexMatrix
    completeness_residual: float
    unitality_residual: float
    choi_trace: float
    tp_diamond_defect: float
    unital_diamond_defect: float
    hermiticity_defect: float
    tp_crosscheck_defect: float
    unital_crosscheck_defect: float


@dataclass(frozen=True, slots=True)
class CPTPValidationReport:
    r"""
    Certificado inmutable de consistencia cuántica para el canal semántico.

    Vive en el topos de haces sobre el espectro de Choi–Jamiołkowski.
    Todos los campos son invariantes bajo isometrías de Stinespring
    (libertad unitaria de calibre en el espacio ancilla).

    Atributos:
        is_cptp (bool):
            \(\top\) sii el mapa satisface CP ∧ TP dentro de tolerancia.
        kraus_completeness_residual (float):
            Residuo de completitud
            \(\bigl\|\sum_k M_k^\dagger M_k - I\bigr\|_F\).
        choi_min_eigenvalue (float):
            \(\lambda_{\min}(\Lambda_{\mathcal{E}})\); debe ser ≥ −ε para CP.
        choi_rank (int):
            \(\operatorname{rank}(\Lambda_{\mathcal{E}})\) = nº efectivo de Kraus
            (teorema de Choi: rango = mínimo número de operadores de Kraus).
        is_separable (bool):
            Criterio PPT de Peres–Horodecki sobre \(\Lambda_{\mathcal{E}}^{T_A}\).
        choi_trace (float):
            \(\operatorname{Tr}(\Lambda_{\mathcal{E}})\); para TP normalizado = d.
        tp_diamond_defect (float):
            Defecto \(\|\operatorname{Tr}_{\mathrm{out}}(\Lambda)-I\|_F\)
            (consistencia Choi↔TP).
        kraus_count (int):
            Cardinalidad del ensamble de Kraus tras saneamiento.
        wilkinson_tolerance (float):
            ε de Wilkinson usado en las comparaciones de decisión CP/TP.
        unitality_residual (float):
            \(\bigl\|\sum_k M_k M_k^\dagger - I\bigr\|_F\).
        is_unital (bool):
            \(\top\) sii el residual de unitalidad ≤ ε (no exigido por CPTP).
        choi_max_eigenvalue (float):
            \(\lambda_{\max}(\Lambda_{\mathcal{E}})\).
        choi_spectral_gap (float):
            Menor autovalor estrictamente positivo (hueco al cono singular).
        choi_condition_number (float):
            \(\lambda_{\max}/\lambda_{\min}^+\) sobre el soporte espectral.
        hermiticity_defect (float):
            \(\|\Lambda-\Lambda^\dagger\|_F\) previo a la proyección de Weyl.
        kraus_gram_rank (int):
            Rango numérico de \(V=[\operatorname{vec}_F M_k]\) (pre-Choi).
        nosignal_defect (float):
            \(\bigl\|\operatorname{Tr}\circ\mathcal{E}(|i\rangle\langle j|)-\delta_{ij}\bigr\|_F\).
        tp_crosscheck_defect (float):
            Discrepancia absoluta entre residual Kraus-TP y traza parcial de Choi.
        discarded_null_kraus (int):
            Operadores anulados en el saneamiento de calibre numérico.
    """

    is_cptp: bool
    kraus_completeness_residual: float
    choi_min_eigenvalue: float
    choi_rank: int
    is_separable: bool
    choi_trace: float = 0.0
    tp_diamond_defect: float = 0.0
    kraus_count: int = 0
    wilkinson_tolerance: float = _EPS_DEFAULT
    unitality_residual: float = 0.0
    is_unital: bool = False
    choi_max_eigenvalue: float = 0.0
    choi_spectral_gap: float = 0.0
    choi_condition_number: float = float("inf")
    hermiticity_defect: float = 0.0
    kraus_gram_rank: int = 0
    nosignal_defect: float = 0.0
    tp_crosscheck_defect: float = 0.0
    discarded_null_kraus: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDADOR CPTP — MORFISMO EN LA CATEGORÍA MIC (dagger-compacta)
# ═══════════════════════════════════════════════════════════════════════════════
class CPTPChannelValidator(Morphism):
    r"""
    Operador de validación espectral para canales de inyección en la MAC.

    Realiza el funtor de auditoría

    \[
    \mathcal{V}_{\mathrm{CPTP}}
    :\mathbf{Kraus}(\mathcal{H}_d)
    \longrightarrow
    \mathbf{Report}(\Omega_{\mathrm{CPTP}})
    \]

    como composición de tres fases anidadas, cada una monádica sobre
    \({\tt TopologicalInvariantError}\):

    \[
    \mathrm{Seal}\circ\mathrm{CP\text{-}PPT\text{-}NS}\circ\mathrm{Choi}\circ\mathrm{TP/Unital}\circ\mathrm{Normalize}.
    \]

    Los objetos intermedios son certificados frozen:

    \[
    \mathrm{RawKraus}
    \xrightarrow{\mathrm{F1}}
    {\tt Phase1KrausSanitizerCertificate}
    \xrightarrow{\mathrm{F2}}
    {\tt Phase2ChoiIsomorphismCertificate}
    \xrightarrow{\mathrm{F3}}
    {\tt CPTPValidationReport}.
    \]
    """

    def __init__(self, target_stratum: Stratum = Stratum.WISDOM) -> None:
        """
        Inicializa el validador en el estrato de la Sabiduría cuántica.

        Args:
            target_stratum: Estrato MIC de anclaje (default WISDOM).
        """
        super().__init__()
        self._target_stratum: Stratum = target_stratum

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  UTILIDADES ESPECTRALES (núcleo C* compartido por las tres fases)     ║
    # ║  Weyl–Wilkinson, proyección de Cartan al subespacio autoadjunto,      ║
    # ║  congelación de tensores y calibre U(1) canónico.                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _wilkinson_spectral_floor(
        scale: float,
        tolerance: float,
        ambient_dim: int,
    ) -> float:
        r"""
        Suelo espectral de Weyl–Wilkinson.

        Para una perturbación hermítica \(\Delta\) con \(\|\Delta\|_2\le
        C\cdot\varepsilon_{\mathrm{mach}}\cdot\mathrm{scale}\cdot\mathrm{poly}(n)\),

        \[
        \bigl|\lambda_i(H)-\lambda_i(H+\Delta)\bigr|
        \le\|\Delta\|_2.
        \]

        El umbral de rango numérico (no el de decisión CP/TP, que permanece
        anclado al \(\varepsilon\) del caller) se toma como

        \[
        \varepsilon_W
        =\max\bigl(\varepsilon,\;
        n\cdot\varepsilon_{\mathrm{mach}}\cdot\max(\mathrm{scale},1),\;
        \varepsilon_{\mathrm{spectral}}\bigr).
        \]

        Args:
            scale: Escala natural (\(\|H\|_2\) o \(\sigma_{\max}\)).
            tolerance: ε del caller.
            ambient_dim: Dimensión del espacio ambiente (p.ej. \(d^2\)).

        Returns:
            Suelo \(\varepsilon_W>0\).
        """
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            _EPS_SPECTRAL,
        )

    @staticmethod
    def _hermitize_weyl(matrix: ComplexMatrix) -> ComplexMatrix:
        r"""
        Proyección de Weyl / Cartan sobre el subespacio autoadjunto:

        \[
        H\mapsto\tfrac12\bigl(H+H^\dagger\bigr).
        \]

        Es el projector ortogonal en norma de Frobenius sobre
        \(\mathfrak{u}(n)^\perp\simeq\mathrm{i}\,\mathfrak{u}(n)\).
        """
        return 0.5 * (matrix + matrix.conj().T)

    @staticmethod
    def _freeze_matrix(matrix: np.ndarray) -> ComplexMatrix:
        """Copia C-contigua ``complex128`` con write-protect (invariante de certificado)."""
        frozen = np.array(matrix, dtype=np.complex128, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _antihermitian_frobenius(matrix: ComplexMatrix) -> RealScalar:
        r"""Defecto antihermítico \(\|A-A^\dagger\|_F\)."""
        return float(la.norm(matrix - matrix.conj().T, ord="fro"))

    @staticmethod
    def _canonical_u1_phase(operator: ComplexMatrix, floor: float) -> float:
        r"""
        Fase canónica \(\theta\in(-\pi,\pi]\) del calibre U(1) de un Kraus.

        Convención (estable y determinista):

        * si \(|\operatorname{Tr} M|>\varepsilon\), \(\theta=\arg(\operatorname{Tr} M)\);
        * si no, \(\theta=\arg(M_{ij})\) de la entrada de mayor módulo;
        * si el operador es nulo, \(\theta=0\).

        Así \(\hat{M}=e^{-i\theta}M\) tiene el coeficiente de referencia real ≥ 0.
        """
        trace_m = np.trace(operator)
        if abs(trace_m) > floor:
            return float(np.angle(trace_m))
        flat = np.asarray(operator).reshape(-1)
        idx = int(np.argmax(np.abs(flat)))
        coeff = flat[idx]
        if abs(coeff) <= floor:
            return 0.0
        return float(np.angle(coeff))

    @staticmethod
    def _choi_tensor4(choi_matrix: ComplexMatrix, dimension_d: int) -> ComplexMatrix:
        r"""
        Reconstrucción índice-explícita del tensor de Choi.

        Convención \(\operatorname{vec}_F\): \(\alpha=i+d\cdot j\) indexa
        \(M_{ij}\) (fila de salida \(i\), columna de entrada \(j\)).
        Con ``reshape(..., order="F")`` se obtiene

        \[
        T[i,j,a,b]
        =\Lambda_{\mathcal{E}}\bigl[i+dj,\;a+db\bigr],
        \]

        de modo que \(\Lambda\in\mathcal{B}(\mathcal{H}_{\mathrm{out}}\otimes
        \mathcal{H}_{\mathrm{in}})\).
        """
        d = dimension_d
        return np.asarray(choi_matrix, dtype=np.complex128).reshape((d, d, d, d), order="F")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 — CONSISTENCIA DIMENSIONAL Y SANEAMIENTO ESPECTRAL            ║
    # ║  Objetos: {M_k} ⊂ ℬ(ℋ_d), tipado C*, soporte no vacío, calibre U(1)  ║
    # ║  Funtores: inclusión en Mat_d(ℂ), proyección de calibre numérico      ║
    # ║  Terminal: Phase1KrausSanitizerCertificate  →  objeto inicial FASE 2  ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase1_validate_hilbert_dimension(self, dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de la dimensión del espacio de Hilbert.

        Exige

        \[
        d\in\mathbb{Z}_{\ge 1}
        \qquad\text{y}\qquad
        d\le d_{\max}
        \]

        (\(d=1\) es el canal trivial sobre \(\mathbb{C}\); se admite pero se
        registra). En circuitos cuánticos \(d=2^n\) para \(n\) qubits; cualquier
        \(d\) no entero o no positivo es un cortocircuito ontológico del bus MAC.
        El techo \(d_{\max}\) acota la matriz de Choi (\(d^2\times d^2\)) a un
        régimen numéricamente auditables (Weyl: \(d\lesssim 32\), headroom 64).

        Args:
            dimension_d: Dimensión candidata de \(\mathcal{H}_A\).

        Returns:
            ``dimension_d`` validado (identidad tipada).

        Raises:
            TopologicalInvariantError: Si \(d<1\) o \(d>d_{\max}\).
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise TopologicalInvariantError(
                f"Dimensión de Hilbert inválida: d={dimension_d}. "
                "Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise TopologicalInvariantError(
                f"Dimensión de Hilbert d={d} excede d_max={_DIM_MAX}. "
                "La matriz de Choi (d²×d²) abandonaría el régimen de auditoría "
                "espectral estable del estrato WISDOM."
            )
        if d == 1:
            logger.debug(
                "FASE1.1: d=1 (canal escalar trivial); auditoría degenerada admisible."
            )
        return d

    def _phase1_validate_kraus_ensemble(
        self,
        kraus_operators: KrausEnsemble,
        dimension_d: int,
    ) -> List[ComplexMatrix]:
        r"""
        FASE 1.2 — Tipado, shape y coerción C* del ensamble de Kraus.

        Para cada \(M_k\) se verifica:

        \[
        M_k\in\mathrm{Mat}_d(\mathbb{C}),
        \qquad
        \|M_k\|_F < \infty,
        \qquad
        M_k\text{ libre de NaN/Inf}.
        \]

        El ensamble no puede ser vacío (el canal cero no es TP salvo \(d=0\),
        excluido en 1.1). Cada operador se coacciona a ``complex128`` y
        se materializa como lista propia (desacoplamiento referencial del
        caller).

        Análogo eléctrico: \(\{M_k\}\) son las matrices de scattering de un
        multipolo multipuerto; shape incorrecto ≡ puerto flotante.

        Args:
            kraus_operators: Secuencia bruta de operadores de Kraus.
            dimension_d: Dimensión ya certificada (salida de 1.1).

        Returns:
            Lista normalizada ``list[ComplexMatrix]`` con shape \((d, d)\).

        Raises:
            TopologicalInvariantError: Ensamble vacío, shape o no-finitud.
        """
        if kraus_operators is None or len(kraus_operators) == 0:
            raise TopologicalInvariantError(
                "Ensamble de Kraus vacío: no existe canal CPTP sin operadores. "
                "Inyecte al menos un M_k (p.ej. {I} para el canal identidad)."
            )

        normalized: List[ComplexMatrix] = []
        expected = (dimension_d, dimension_d)

        for idx, m_k in enumerate(kraus_operators):
            if not isinstance(m_k, np.ndarray):
                raise TopologicalInvariantError(
                    f"Kraus[{idx}] no es ndarray; tipo={type(m_k).__name__}."
                )
            if m_k.ndim != 2 or m_k.shape != expected:
                raise TopologicalInvariantError(
                    f"Inconsistencia dimensional en Kraus[{idx}]: "
                    f"esperado {expected}, obtenido {getattr(m_k, 'shape', None)}."
                )
            m_c = np.asarray(m_k, dtype=np.complex128)
            if not np.all(np.isfinite(m_c)):
                raise TopologicalInvariantError(
                    f"Kraus[{idx}] contiene NaN/Inf: canal no físico (norma no finita)."
                )
            normalized.append(m_c)

        logger.debug(
            "FASE1.2 Kraus: n_ops=%d, d=%d, ‖M_0‖_F=%.6e",
            len(normalized),
            dimension_d,
            float(la.norm(normalized[0], ord="fro")),
        )
        return normalized

    def _phase1_audit_kraus_operator_norms(
        self,
        kraus_ops: List[ComplexMatrix],
    ) -> Tuple[RealScalar, RealScalar]:
        r"""
        FASE 1.3 — Auditoría de normas \(C^*\) del ensamble.

        Calcula

        \[
        \|M_k\|_2=\sigma_{\max}(M_k),
        \qquad
        \mu_F=\Bigl(\sum_k\|M_k\|_F^2\Bigr)^{1/2}.
        \]

        Para un canal TP se tiene \(\mu_F=\sqrt{d}\) exactamente
        (\(\sum_k\operatorname{Tr}(M_k^\dagger M_k)=\operatorname{Tr} I=d\)).
        Esta masa de Frobenius es el *checksum precursor* que la FASE 2
        comparará con \(\operatorname{Tr}\Lambda_{\mathcal{E}}\).

        Args:
            kraus_ops: Ensamble tipado (salida de 1.2).

        Returns:
            ``(max_operator_norm, frobenius_mass)``.
        """
        max_op = 0.0
        mass_sq = 0.0
        for m_k in kraus_ops:
            fro = float(la.norm(m_k, ord="fro"))
            mass_sq += fro * fro
            spec = float(la.norm(m_k, ord=2))
            if spec > max_op:
                max_op = spec
        mass = math.sqrt(mass_sq)
        logger.debug(
            "FASE1.3 normas: max‖M‖₂=%.6e, μ_F=%.6e",
            max_op,
            mass,
        )
        return max_op, mass

    def _phase1_normalize_kraus_gauge(
        self,
        kraus_ops: List[ComplexMatrix],
        tolerance: float,
    ) -> Tuple[List[ComplexMatrix], int]:
        r"""
        FASE 1.4 — Limpieza de calibre numérico (operadores nulos / basura UV).

        Descarta operadores con

        \[
        \|M_k\|_F \le \varepsilon_{\mathrm{spectral}}
        \]

        (ruido de redondeo FPU que no contribuye al canal ni al rango de Choi).
        Si tras el filtrado el ensamble queda vacío se eleva error: el canal
        colapsó al morfismo nulo.

        Args:
            kraus_ops: Ensamble tipado (salida de 1.2).
            tolerance: Umbral de aniquilación de norma Frobenius.

        Returns:
            ``(sub_ensamble, discarded_null_count)`` con soporte no trivial.

        Raises:
            TopologicalInvariantError: Si todos los Kraus son nulos.
        """
        floor = max(tolerance * 1.0e-3, _EPS_SPECTRAL)
        cleaned: List[ComplexMatrix] = []
        discarded = 0
        for m_k in kraus_ops:
            if float(la.norm(m_k, ord="fro")) > floor:
                cleaned.append(m_k)
            else:
                discarded += 1
        if not cleaned:
            raise TopologicalInvariantError(
                "Todos los operadores de Kraus son numéricamente nulos "
                f"(‖M_k‖_F ≤ {floor:.3e}). Canal cero no es CPTP."
            )
        if discarded:
            logger.info(
                "FASE1.4 calibre: descartados %d operadores nulos de %d.",
                discarded,
                len(kraus_ops),
            )
        return cleaned, discarded

    def _phase1_fix_kraus_u1_gauge(
        self,
        kraus_ops: List[ComplexMatrix],
        tolerance: float,
    ) -> Tuple[List[ComplexMatrix], Tuple[float, ...]]:
        r"""
        FASE 1.5 — Fijación del calibre U(1) por operador
        \(\hat{M}_k\leftarrow e^{-i\theta_k}M_k\).

        La libertad \(M_k\mapsto e^{i\phi_k}M_k\) no altera
        \(\mathcal{E}(\rho)=\sum_k M_k\rho M_k^\dagger\) ni \(\Lambda_{\mathcal{E}}\).
        Fijarla hace determinista cualquier inspección coeficiente a coeficiente
        y reduce la fase espuria en \(\operatorname{Tr} M_k\).

        Args:
            kraus_ops: Ensamble de soporte no trivial (salida de 1.4).
            tolerance: Suelo para decidir el coeficiente de referencia.

        Returns:
            ``(kraus_calibrados, (θ_k)_k)``.
        """
        floor = max(tolerance * 1.0e-3, _EPS_SPECTRAL)
        phased: List[ComplexMatrix] = []
        thetas: List[float] = []
        for m_k in kraus_ops:
            theta = self._canonical_u1_phase(m_k, floor)
            hat = np.exp(-1.0j * theta) * m_k
            phased.append(np.asarray(hat, dtype=np.complex128))
            thetas.append(theta)
        logger.debug(
            "FASE1.5 gauge U(1): n=%d, max|θ|=%.6e",
            len(thetas),
            max(abs(t) for t in thetas) if thetas else 0.0,
        )
        return phased, tuple(thetas)

    def _phase1_kraus_hs_gram_rank(
        self,
        kraus_ops: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float,
    ) -> int:
        r"""
        FASE 1.6 — Rango de Hilbert–Schmidt del ensamble (precursor de Choi).

        Apila \(V=\bigl[\operatorname{vec}_F M_k\bigr]_k\in\mathbb{C}^{d^2\times K}\)
        y estima

        \[
        \operatorname{rank}_{\varepsilon}(V)
        =\#\{\sigma_i(V):\sigma_i>\varepsilon_W\}.
        \]

        En aritmética exacta este rango coincide con
        \(\operatorname{rank}(\Lambda_{\mathcal{E}})=r_{\mathrm{Kraus}}^{\min}\)
        (teorema de representación de Choi). La FASE 3 confrontará ambos.

        Args:
            kraus_ops: Ensamble calibrado (salida de 1.5).
            dimension_d: \(d\) certificado.
            tolerance: ε de Wilkinson del caller.

        Returns:
            Rango numérico de \(V\).
        """
        vecs = [
            np.asarray(m_k, dtype=np.complex128).reshape(dimension_d * dimension_d, order="F")
            for m_k in kraus_ops
        ]
        stacked = np.column_stack(vecs)
        singular = la.svdvals(stacked)
        scale = float(singular[0]) if singular.size else 0.0
        floor = self._wilkinson_spectral_floor(
            scale, tolerance, dimension_d * dimension_d
        )
        rank = int(np.sum(singular > floor))
        logger.debug(
            "FASE1.6 Gram/HS: rank=%d, σ_max=%.6e, floor=%.3e, K=%d",
            rank,
            scale,
            floor,
            len(kraus_ops),
        )
        return rank

    def _phase1_observe_kraus_certificate(
        self,
        kraus_operators: KrausEnsemble,
        dimension_d: int,
        tolerance: float,
    ) -> Phase1KrausSanitizerCertificate:
        r"""
        FASE 1.Ω — Composición terminal de Observación de Kraus (Observe).

        Morfismo compuesto de la FASE 1:

        \[
        \mathrm{ObserveKraus}
        =\mathrm{HSRank}\circ\mathrm{U(1)}\circ\mathrm{NullFilter}
        \circ\mathrm{Norms}\circ\mathrm{Type}\circ\mathrm{Dim}
        :\mathbf{RawKraus}
        \rightharpoonup
        {\tt Phase1KrausSanitizerCertificate}.
        \]

        **Este certificado es el objeto inicial de la FASE 2**: se inyecta
        sin re-tipado en ``_phase2_trace_preserving_residual`` y en
        ``_phase2_orient_choi_geometry``. Ningún método de FASE 2 acepta
        un ensamble crudo.

        Args:
            kraus_operators: Ensamble bruto.
            dimension_d: Dimensión candidata.
            tolerance: ε de Wilkinson del caller.

        Returns:
            ``Phase1KrausSanitizerCertificate`` sellado (precondición de FASE 2).
        """
        d = self._phase1_validate_hilbert_dimension(dimension_d)
        typed = self._phase1_validate_kraus_ensemble(kraus_operators, d)
        max_op, _mass_raw = self._phase1_audit_kraus_operator_norms(typed)
        cleaned, discarded = self._phase1_normalize_kraus_gauge(typed, tolerance)
        phased, thetas = self._phase1_fix_kraus_u1_gauge(cleaned, tolerance)
        # Recalcular masa y normas sobre el ensamble ya saneado (invariantes U(1)).
        max_op, mass = self._phase1_audit_kraus_operator_norms(phased)
        gram_rank = self._phase1_kraus_hs_gram_rank(phased, d, tolerance)
        frozen_ops = tuple(self._freeze_matrix(m_k) for m_k in phased)
        return Phase1KrausSanitizerCertificate(
            kraus_ops=frozen_ops,
            dimension_d=d,
            kraus_count=len(frozen_ops),
            kraus_gram_rank=gram_rank,
            discarded_null_count=discarded,
            gauge_phases=thetas,
            max_operator_norm=max_op,
            frobenius_mass=mass,
            wilkinson_tolerance=float(tolerance),
        )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 — TRACE-PRESERVING, UNITALIDAD Y CHOI–JAMIOŁKOWSKI            ║
    # ║  Continuación directa de Phase1KrausSanitizerCertificate (FASE 1.Ω)   ║
    # ║  Objetos: Σ M†M, Σ MM†, Λ_ℰ = V V†, Tr_out(Λ), Tr_in(Λ)              ║
    # ║  Teorías: Kraus TP, dual unital, isomorfismo de Choi, Hellwig–Kraus   ║
    # ║  Terminal: Phase2ChoiIsomorphismCertificate  →  objeto inicial FASE 3 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase2_trace_preserving_residual(
        self,
        certificate: Phase1KrausSanitizerCertificate,
    ) -> Tuple[RealScalar, ComplexMatrix]:
        r"""
        FASE 2.1 — Residuo de completitud de Kraus / Trace-Preserving (inicio Orient).

        **Continuación funtorial de FASE 1.Ω**: recibe exactamente el
        ``Phase1KrausSanitizerCertificate`` emitido por
        ``_phase1_observe_kraus_certificate``. No re-valida shapes ni dtypes.

        Calcula el operador de completitud

        \[
        S=\sum_{k}M_k^\dagger M_k
        \in\mathcal{B}(\mathcal{H}_d)_{\mathrm{sa}}
        \]

        y el residuo de TP en norma de Frobenius (topología C*):

        \[
        r_{\mathrm{TP}}
        =\bigl\|S-I_d\bigr\|_F
        =\Bigl(\sum_{ij}|S_{ij}-\delta_{ij}|^2\Bigr)^{1/2}.
        \]

        El canal es TP (no-destruir probabilidad) sii \(r_{\mathrm{TP}}=0\).
        Para TP-completo se exige \(S=I\) (no solo \(S\le I\), que
        correspondería a un instrumento / canal trace-non-increasing).

        En circuitos: \(S\) es la matriz de Gram de potencias disipadas; \(S=I\)
        ≡ pasividad unitaria del multipolo (conservación de energía de
        probabilidad).

        Args:
            certificate: Certificado terminal de FASE 1.

        Returns:
            ``(completeness_residual, completeness_operator S)``.
        """
        dimension_d = certificate.dimension_d
        s_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in certificate.kraus_ops:
            s_op += m_k.conj().T @ m_k

        s_op = self._hermitize_weyl(s_op)
        residual = float(
            la.norm(s_op - np.eye(dimension_d, dtype=np.complex128), ord="fro")
        )
        logger.debug(
            "FASE2.1 TP: r_TP=%.6e, ‖S‖₂=%.6e, λ_min(S)=%.6e",
            residual,
            float(la.norm(s_op, ord=2)),
            float(np.min(la.eigvalsh(s_op))),
        )
        return residual, s_op

    def _phase2_unitality_residual(
        self,
        certificate: Phase1KrausSanitizerCertificate,
    ) -> Tuple[RealScalar, ComplexMatrix]:
        r"""
        FASE 2.2 — Defecto de unitalidad (dual de Hellwig–Kraus).

        Un canal es unital sii \(\mathcal{E}(I)=I\), equivalentemente

        \[
        U=\sum_k M_k M_k^\dagger = I_d,
        \qquad
        \Delta_{\mathrm{unital}}=\|U-I\|_F.
        \]

        CPTP **no** exige unitalidad (el amortiguamiento de amplitud es el
        contraejemplo canónico). Se reporta como invariante geométrico:
        unital + TP ⇒ mapa doblemente estocástico cuántico (canales de
        Birkhoff–von Neumann / ruidos unitales).

        Args:
            certificate: Certificado terminal de FASE 1.

        Returns:
            ``(unitality_residual, unitality_operator U)``.
        """
        dimension_d = certificate.dimension_d
        u_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in certificate.kraus_ops:
            u_op += m_k @ m_k.conj().T

        u_op = self._hermitize_weyl(u_op)
        residual = float(
            la.norm(u_op - np.eye(dimension_d, dtype=np.complex128), ord="fro")
        )
        logger.debug(
            "FASE2.2 Unital: Δ_U=%.6e, ‖U‖₂=%.6e, λ_min(U)=%.6e",
            residual,
            float(la.norm(u_op, ord=2)),
            float(np.min(la.eigvalsh(u_op))),
        )
        return residual, u_op

    def _phase2_construct_choi_matrix(
        self,
        certificate: Phase1KrausSanitizerCertificate,
    ) -> ComplexMatrix:
        r"""
        FASE 2.3 — Isomorfismo de Choi–Jamiołkowski desde Kraus (BLAS-3).

        Reconstruye

        \[
        \Lambda_{\mathcal{E}}
        =V V^\dagger,
        \qquad
        V=\bigl[\operatorname{vec}_F(M_k)\bigr]_k
        \in\mathbb{C}^{d^2\times K},
        \]

        donde \(\operatorname{vec}_F\) es el apilado *column-major* (orden
        Fortran), convención estándar en información cuántica (Watrous,
        Nielsen–Chuang). Equivale a

        \[
        \Lambda_{\mathcal{E}}
        =(\mathcal{E}\otimes\mathrm{id})(|\widetilde{\Omega}\rangle\langle\widetilde{\Omega}|),
        \quad
        |\widetilde{\Omega}\rangle=\sum_i|i\rangle\otimes|i\rangle
        \]

        (versión no normalizada: \(\operatorname{Tr}\Lambda=d\) para canales TP).

        Complejidad: \(O(K d^4)\) en el producto \(VV^\dagger\); aceptable
        para \(d\lesssim 32\) en el estrato WISDOM.

        Args:
            certificate: Certificado terminal de FASE 1.

        Returns:
            ``choi_matrix`` (aún sin proyectar) de shape \((d^2,d^2)\).
        """
        d = certificate.dimension_d
        vecs = [
            np.asarray(m_k, dtype=np.complex128).reshape(d * d, order="F")
            for m_k in certificate.kraus_ops
        ]
        stacked = np.column_stack(vecs)
        choi = stacked @ stacked.conj().T
        return np.asarray(choi, dtype=np.complex128)

    def _phase2_certify_choi_hermiticity(
        self,
        choi_matrix: ComplexMatrix,
        tolerance: float,
    ) -> Tuple[ComplexMatrix, RealScalar]:
        r"""
        FASE 2.4 — Certificación C* de hermiticidad de \(\Lambda_{\mathcal{E}}\).

        Verifica

        \[
        \bigl\|\Lambda-\Lambda^\dagger\bigr\|_F
        \le
        \max\bigl(\varepsilon,\,\varepsilon_H,\,\varepsilon_H\|\Lambda\|_F\bigr).
        \]

        Si el defecto es tolerable se devuelve la proyección de Weyl.
        Si excede el umbral, el ensamble es internamente inconsistente
        (corrupción del isomorfismo \(VV^\dagger\)) y se eleva invariante.

        Args:
            choi_matrix: \(\Lambda_{\mathcal{E}}\) (salida de 2.3).
            tolerance: ε de Wilkinson.

        Returns:
            ``(choi_hermítico, hermiticity_defect)``.

        Raises:
            TopologicalInvariantError: Defecto antihermítico intolerable.
        """
        defect = self._antihermitian_frobenius(choi_matrix)
        scale = max(float(la.norm(choi_matrix, ord="fro")), 1.0)
        tol_h = max(tolerance, _EPS_HERMITICITY, _EPS_HERMITICITY * scale)
        if defect > tol_h:
            raise TopologicalInvariantError(
                f"La matriz de Choi viola hermiticidad: "
                f"‖Λ−Λ†‖_F={defect:.3e} > tol={tol_h:.3e}."
            )
        return self._hermitize_weyl(choi_matrix), defect

    def _phase2_choi_partial_traces(
        self,
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> Tuple[ComplexMatrix, ComplexMatrix]:
        r"""
        FASE 2.5 — Trazas parciales duales de Choi.

        Con el layout \(T[i,j,a,b]=\Lambda[i+dj,\,a+db]\):

        \[
        \operatorname{Tr}_{\mathrm{out}}(\Lambda)
        =\sum_i T[i,\cdot,i,\cdot]
        \;\simeq\;
        \Bigl(\sum_k M_k^\dagger M_k\Bigr)^{\!T},
        \]
        \[
        \operatorname{Tr}_{\mathrm{in}}(\Lambda)
        =\sum_j T[\cdot,j,\cdot,j]
        \;\simeq\;
        \sum_k M_k M_k^\dagger.
        \]

        Para un canal TP (resp. unital) se tiene
        \(\operatorname{Tr}_{\mathrm{out}}\Lambda=I\)
        (resp. \(\operatorname{Tr}_{\mathrm{in}}\Lambda=I\)).
        Como \(S\) es hermítica, \(\|S^T-I\|_F=\|S-I\|_F\).

        Args:
            choi_matrix: \(\Lambda_{\mathcal{E}}\) hermítica \((d^2\times d^2)\).
            dimension_d: \(d\).

        Returns:
            ``(tr_out, tr_in)`` ambas autoadjuntas \(d\times d\).
        """
        tensor = self._choi_tensor4(choi_matrix, dimension_d)
        tr_out = np.trace(tensor, axis1=0, axis2=2)
        tr_in = np.trace(tensor, axis1=1, axis2=3)
        return self._hermitize_weyl(tr_out), self._hermitize_weyl(tr_in)

    def _phase2_crosscheck_kraus_choi_duality(
        self,
        completeness_residual: RealScalar,
        unitality_residual: RealScalar,
        tp_diamond_defect: RealScalar,
        unital_diamond_defect: RealScalar,
        choi_trace: RealScalar,
        frobenius_mass: RealScalar,
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealScalar, RealScalar]:
        r"""
        FASE 2.6 — Checksums cruzados Kraus ↔ Choi (fail-secure).

        Confronta tres identidades de aritmética exacta:

        1. \(r_{\mathrm{TP}}=\|\operatorname{Tr}_{\mathrm{out}}\Lambda-I\|_F\),
        2. \(\Delta_{\mathrm{unital}}=\|\operatorname{Tr}_{\mathrm{in}}\Lambda-I\|_F\),
        3. \(\operatorname{Tr}\Lambda=\|V\|_F^2=\mu_F^2\).

        Una discrepancia por encima del suelo de Wilkinson delata corrupción
        del isomorfismo \(\operatorname{vec}_F\) / ``reshape`` y se eleva
        como invariante topológico (no como merma CPTP del canal).

        Args:
            completeness_residual: \(r_{\mathrm{TP}}\) vía \(\sum M^\dagger M\).
            unitality_residual: \(\Delta_U\) vía \(\sum MM^\dagger\).
            tp_diamond_defect: residual vía \(\operatorname{Tr}_{\mathrm{out}}\Lambda\).
            unital_diamond_defect: residual vía \(\operatorname{Tr}_{\mathrm{in}}\Lambda\).
            choi_trace: \(\operatorname{Tr}\Lambda\).
            frobenius_mass: \(\mu_F\) del certificado de FASE 1.
            dimension_d: \(d\).
            tolerance: ε de Wilkinson.

        Returns:
            ``(tp_crosscheck_defect, unital_crosscheck_defect)``.

        Raises:
            TopologicalInvariantError: Checksum de isomorfismo corrupto.
        """
        tp_x = abs(float(completeness_residual) - float(tp_diamond_defect))
        unital_x = abs(float(unitality_residual) - float(unital_diamond_defect))
        mass_x = abs(float(choi_trace) - float(frobenius_mass) ** 2)
        floor = self._wilkinson_spectral_floor(
            max(abs(choi_trace), 1.0),
            tolerance,
            dimension_d * dimension_d,
        )
        # Factor holgado: las tres rutas acumulan redondeo FMA independiente.
        checksum_tol = max(10.0 * floor, 1.0e-9)

        if tp_x > checksum_tol:
            raise TopologicalInvariantError(
                "Checksum Choi↔Kraus TP corrupto: "
                f"|r_TP − ‖Tr_out Λ − I‖_F|={tp_x:.3e} > {checksum_tol:.3e}. "
                "El isomorfismo vec_F/reshape no es fiel."
            )
        if unital_x > checksum_tol:
            raise TopologicalInvariantError(
                "Checksum Choi↔Kraus Unital corrupto: "
                f"|Δ_U − ‖Tr_in Λ − I‖_F|={unital_x:.3e} > {checksum_tol:.3e}."
            )
        if mass_x > checksum_tol:
            raise TopologicalInvariantError(
                "Checksum Tr(Λ)↔μ_F² corrupto: "
                f"|Tr(Λ) − μ_F²|={mass_x:.3e} > {checksum_tol:.3e}."
            )
        logger.debug(
            "FASE2.6 checksums: ΔTP=%.3e, ΔU=%.3e, ΔTr=%.3e (tol=%.3e)",
            tp_x,
            unital_x,
            mass_x,
            checksum_tol,
        )
        return tp_x, unital_x

    def _phase2_orient_choi_geometry(
        self,
        certificate: Phase1KrausSanitizerCertificate,
    ) -> Phase2ChoiIsomorphismCertificate:
        r"""
        FASE 2.Ω — Composición terminal de Orientación Choi (Orient).

        Morfismo compuesto de la FASE 2:

        \[
        \mathrm{OrientChoi}
        =(\mathrm{Cross},\,\mathrm{Tr}_{A|A'},\,\mathrm{Herm},\,\mathrm{Choi},
        \,\mathrm{Unital},\,\mathrm{TP})
        \circ{\tt Phase1KrausSanitizerCertificate}.
        \]

        **Este certificado es el objeto inicial de la FASE 3**: se inyecta
        en ``_phase3_spectral_complete_positivity`` y restantes morfismos
        de Decide/Act sin reconstruir \(\Lambda_{\mathcal{E}}\).

        Args:
            certificate: Objeto terminal de FASE 1.Ω.

        Returns:
            ``Phase2ChoiIsomorphismCertificate`` sellado (precondición de FASE 3).
        """
        tolerance = certificate.wilkinson_tolerance
        d = certificate.dimension_d

        r_tp, s_op = self._phase2_trace_preserving_residual(certificate)
        r_unital, u_op = self._phase2_unitality_residual(certificate)
        choi_raw = self._phase2_construct_choi_matrix(certificate)
        choi, h_def = self._phase2_certify_choi_hermiticity(choi_raw, tolerance)
        tr_out, tr_in = self._phase2_choi_partial_traces(choi, d)

        eye_d = np.eye(d, dtype=np.complex128)
        tp_diamond = float(la.norm(tr_out - eye_d, ord="fro"))
        unital_diamond = float(la.norm(tr_in - eye_d, ord="fro"))
        choi_trace = float(np.real(np.trace(choi)))

        tp_x, unital_x = self._phase2_crosscheck_kraus_choi_duality(
            completeness_residual=r_tp,
            unitality_residual=r_unital,
            tp_diamond_defect=tp_diamond,
            unital_diamond_defect=unital_diamond,
            choi_trace=choi_trace,
            frobenius_mass=certificate.frobenius_mass,
            dimension_d=d,
            tolerance=tolerance,
        )

        return Phase2ChoiIsomorphismCertificate(
            phase1=certificate,
            choi_matrix=self._freeze_matrix(choi),
            completeness_operator=self._freeze_matrix(s_op),
            unitality_operator=self._freeze_matrix(u_op),
            completeness_residual=r_tp,
            unitality_residual=r_unital,
            choi_trace=choi_trace,
            tp_diamond_defect=tp_diamond,
            unital_diamond_defect=unital_diamond,
            hermiticity_defect=h_def,
            tp_crosscheck_defect=tp_x,
            unital_crosscheck_defect=unital_x,
        )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 3 — CP ESPECTRAL, PPT, NO-SEÑALIZACIÓN Y CERTIFICADO            ║
    # ║  Continuación directa de Phase2ChoiIsomorphismCertificate (FASE 2.Ω)  ║
    # ║  Objetos: σ(Λ), rank(Λ), Λ^{T_A}, forma de señalización, Ω_CPTP       ║
    # ║  Teorías: Choi CP, Peres–Horodecki, Weyl–Wilkinson, no-signaling      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase3_spectral_complete_positivity(
        self,
        certificate: Phase2ChoiIsomorphismCertificate,
    ) -> Tuple[RealVector, RealScalar, RealScalar, int, RealScalar, RealScalar, BoolLattice]:
        r"""
        FASE 3.1 — Positividad completa vía espectro de Choi (inicio Decide).

        **Continuación funtorial de FASE 2.Ω**: recibe
        ``Phase2ChoiIsomorphismCertificate`` y lee \(\Lambda_{\mathcal{E}}\)
        ya hermitizada.

        Por el teorema de Choi (1975):

        \[
        \mathcal{E}\text{ es CP}
        \;\Longleftrightarrow\;
        \Lambda_{\mathcal{E}}\succeq 0
        \;\Longleftrightarrow\;
        \lambda_{\min}(\Lambda_{\mathcal{E}})\ge 0.
        \]

        Se admite un suelo de decisión \(-\varepsilon\) (ε del caller) por
        redondeo FPU. El rango de Choi usa el suelo de Wilkinson relativo

        \[
        r=\#\{\lambda\in\sigma(\Lambda):\lambda>\varepsilon_W\},
        \]

        igual al número mínimo de operadores de Kraus (rango de Stinespring).
        Se reportan además \(\lambda_{\max}\), el hueco al cono singular
        \(\lambda_{\min}^+\) y el número de condición del soporte.

        Args:
            certificate: Certificado terminal de FASE 2.

        Returns:
            ``(eigvals, λ_min, λ_max, choi_rank, gap, cond, is_cp)``.
        """
        tolerance = certificate.phase1.wilkinson_tolerance
        choi = certificate.choi_matrix
        eigvals = np.asarray(la.eigvalsh(choi), dtype=np.float64)
        min_eigen = float(eigvals[0])
        max_eigen = float(eigvals[-1])
        is_cp: BoolLattice = min_eigen >= -tolerance

        scale = max(abs(max_eigen), 1.0)
        floor = self._wilkinson_spectral_floor(
            scale, tolerance, certificate.phase1.dimension_d ** 2
        )
        positive = eigvals[eigvals > floor]
        choi_rank = int(positive.size)
        if choi_rank:
            gap = float(positive[0])
            cond = float(positive[-1] / positive[0]) if positive[0] > 0.0 else float("inf")
        else:
            gap = 0.0
            cond = float("inf")

        logger.debug(
            "FASE3.1 CP: λ_min=%.6e, λ_max=%.6e, rank=%d, gap=%.6e, κ=%.4e, is_cp=%s",
            min_eigen,
            max_eigen,
            choi_rank,
            gap,
            cond,
            is_cp,
        )
        return eigvals, min_eigen, max_eigen, choi_rank, gap, cond, is_cp

    def _phase3_ppt_separability(
        self,
        certificate: Phase2ChoiIsomorphismCertificate,
    ) -> BoolLattice:
        r"""
        FASE 3.2 — Criterio PPT de Peres–Horodecki sobre \(\Lambda_{\mathcal{E}}\).

        La transpuesta parcial respecto del factor de salida

        \[
        \bigl(\Lambda^{T_{\mathrm{out}}}\bigr)_{i j;\,a b}
        =\Lambda_{a j;\,i b}
        \]

        debe cumplir \(\Lambda^{T_{\mathrm{out}}}\succeq 0\) si el estado de Choi
        (normalizado) es separable. PPT es:

        * necesario y suficiente para \(d=2\times 2\) y \(2\times 3\);
        * necesario pero no suficiente en dimensión superior
          (estados bound-entangled de Horodecki).

        En el validador reportamos PPT como *proxy* de separabilidad
        del canal (canales entanglement-breaking ⇒ Choi separable ⇒ PPT).

        Layout: coherente con ``reshape(order="F")`` de la FASE 2.

        Args:
            certificate: Certificado terminal de FASE 2.

        Returns:
            ``is_separable`` ≡ \((\lambda_{\min}(\Lambda^{T_A})\ge-\varepsilon)\).
        """
        d = certificate.phase1.dimension_d
        tolerance = certificate.phase1.wilkinson_tolerance
        choi_dim = d * d
        block = self._choi_tensor4(certificate.choi_matrix, d)
        # PT sobre el factor de salida: intercambiar i ↔ a  (ejes 0 ↔ 2).
        ppt_block = block.transpose(2, 1, 0, 3)
        ppt_matrix = ppt_block.reshape((choi_dim, choi_dim), order="F")
        ppt_matrix = self._hermitize_weyl(ppt_matrix)

        ppt_eigs = la.eigvalsh(ppt_matrix)
        min_ppt = float(np.min(ppt_eigs))
        is_separable: BoolLattice = min_ppt >= -tolerance

        logger.debug(
            "FASE3.2 PPT: λ_min(Λ^{T_A})=%.6e, is_separable=%s, d=%d",
            min_ppt,
            is_separable,
            d,
        )
        return is_separable

    def _phase3_nosignal_causality_witness(
        self,
        certificate: Phase2ChoiIsomorphismCertificate,
    ) -> RealScalar:
        r"""
        FASE 3.3 — Testigo algebraico de No-Señalización / causalidad.

        Para todo canal de un solo factor, la condición de no-señalización
        al extender por \(\mathrm{id}_B\),

        \[
        \operatorname{Tr}_A\bigl((\mathcal{E}_A\otimes\mathrm{id}_B)(\rho_{AB})\bigr)
        =\operatorname{Tr}_A(\rho_{AB}),
        \]

        es **equivalente** a TP. El testigo independiente contrae la traza
        de salida sobre la base computacional:

        \[
        T_{ij}
        :=\operatorname{Tr}\bigl(\mathcal{E}(|i\rangle\langle j|)\bigr)
        =\sum_{k,a}(M_k)_{a i}\,\overline{(M_k)_{a j}},
        \qquad
        \delta_{\mathrm{NS}}=\|T-I\|_F.
        \]

        La contracción se realiza como suma sobre el índice de *salida*
        (\(T=\sum_k M_k^\top\overline{M_k}\)), ruta distinta de
        \(\sum_k M_k^\dagger M_k\) (FASE 2.1) y de \(\operatorname{Tr}_{\mathrm{out}}\Lambda\)
        (FASE 2.5).

        Args:
            certificate: Certificado terminal de FASE 2.

        Returns:
            ``nosignal_defect`` \(\delta_{\mathrm{NS}}\).
        """
        d = certificate.phase1.dimension_d
        signaling = np.zeros((d, d), dtype=np.complex128)
        for m_k in certificate.phase1.kraus_ops:
            # T[i,j] += Σ_a M[a,i] conj(M[a,j])  ≡ traza de salida de E(|i⟩⟨j|).
            signaling += m_k.T @ np.conjugate(m_k)
        signaling = self._hermitize_weyl(signaling)
        defect = float(
            la.norm(signaling - np.eye(d, dtype=np.complex128), ord="fro")
        )
        logger.debug("FASE3.3 No-señalización: δ_NS=%.6e", defect)
        return defect

    def _phase3_rank_consistency(
        self,
        certificate: Phase2ChoiIsomorphismCertificate,
        choi_rank: int,
    ) -> None:
        r"""
        FASE 3.4 — Consistencia de rango Choi ↔ Hilbert–Schmidt.

        En aritmética exacta
        \(\operatorname{rank}(\Lambda_{\mathcal{E}})=\operatorname{rank}(V)\).
        Un desacuerdo (típicamente off-by-one cerca del suelo espectral)
        no falsea CPTP pero se registra como telemetría de patología
        numérica.

        Args:
            certificate: Certificado terminal de FASE 2.
            choi_rank: Rango espectral de \(\Lambda\) (salida de 3.1).
        """
        gram_rank = certificate.phase1.kraus_gram_rank
        delta = abs(int(gram_rank) - int(choi_rank))
        if delta:
            logger.warning(
                "FASE3.4 rango: rank(Λ)=%d ≠ rank_HS(V)=%d (|Δ|=%d). "
                "Posible umbral de Wilkinson en el borde espectral.",
                choi_rank,
                gram_rank,
                delta,
            )
        else:
            logger.debug(
                "FASE3.4 rango: rank(Λ)=rank_HS(V)=%d (consistente).",
                choi_rank,
            )

    def _phase3_decide_cptp_lattice(
        self,
        is_tp: BoolLattice,
        is_cp: BoolLattice,
        completeness_residual: RealScalar,
        min_eigen: RealScalar,
    ) -> BoolLattice:
        r"""
        FASE 3.5 — Decisión en la retícula de Heyting \(\Omega_{\mathrm{CPTP}}\).

        \[
        \Omega_{\mathrm{CPTP}}
        =\{\bot\prec\mathrm{CPTP}\},
        \qquad
        \chi_{\mathrm{CPTP}}
        =\chi_{\mathrm{TP}}\wedge\chi_{\mathrm{CP}}.
        \]

        Unitalidad, PPT y el testigo de no-señalización **no** entran en
        \(\chi\) (unitalidad no es axioma CPTP; PPT es proxy de
        entanglement-breaking; no-señalización ≡ TP para un solo factor).
        Si \(\chi=\bot\) se emite telemetría forense; el raise lo decide
        el compositor público (FASE 3.Ω).

        Args:
            is_tp: \(r_{\mathrm{TP}}\le\varepsilon\).
            is_cp: \(\lambda_{\min}(\Lambda)\ge-\varepsilon\).
            completeness_residual, min_eigen: para logging.

        Returns:
            ``is_cptp`` \(\in\mathbb{B}\).
        """
        is_cptp: BoolLattice = bool(is_tp and is_cp)
        if not is_cptp:
            logger.error(
                "Morfismo cuántico inválido detectado. "
                "Residuo TP=%.4e | λ_min(Choi)=%.4e | TP=%s CP=%s",
                completeness_residual,
                min_eigen,
                is_tp,
                is_cp,
            )
        else:
            logger.info(
                "Canal CPTP certificado. r_TP=%.4e | λ_min=%.4e",
                completeness_residual,
                min_eigen,
            )
        return is_cptp

    def _phase3_seal_validation_report(
        self,
        certificate: Phase2ChoiIsomorphismCertificate,
        is_cptp: BoolLattice,
        min_eigen: RealScalar,
        max_eigen: RealScalar,
        choi_rank: int,
        spectral_gap: RealScalar,
        condition_number: RealScalar,
        is_separable: BoolLattice,
        nosignal_defect: RealScalar,
        is_unital: BoolLattice,
    ) -> CPTPValidationReport:
        r"""
        FASE 3.6 — Sellado del certificado inmutable (objeto terminal Act).

        Empaqueta todos los invariantes de las tres fases en
        ``CPTPValidationReport`` frozen. Ningún campo es mutable
        post-construcción (slots+frozen ⇒ hashable ⇒ clave de caché de
        topos / memoización de morfismos de canal).

        Args:
            certificate: Geometría de Choi (FASE 2) + Kraus (FASE 1 anidada).
            Resto: invariantes espectrales y de retícula de FASE 3.

        Returns:
            ``CPTPValidationReport`` sellado.
        """
        phase1 = certificate.phase1
        return CPTPValidationReport(
            is_cptp=is_cptp,
            kraus_completeness_residual=certificate.completeness_residual,
            choi_min_eigenvalue=min_eigen,
            choi_rank=choi_rank,
            is_separable=is_separable,
            choi_trace=certificate.choi_trace,
            tp_diamond_defect=certificate.tp_diamond_defect,
            kraus_count=phase1.kraus_count,
            wilkinson_tolerance=phase1.wilkinson_tolerance,
            unitality_residual=certificate.unitality_residual,
            is_unital=is_unital,
            choi_max_eigenvalue=max_eigen,
            choi_spectral_gap=spectral_gap,
            choi_condition_number=condition_number,
            hermiticity_defect=certificate.hermiticity_defect,
            kraus_gram_rank=phase1.kraus_gram_rank,
            nosignal_defect=nosignal_defect,
            tp_crosscheck_defect=certificate.tp_crosscheck_defect,
            discarded_null_kraus=phase1.discarded_null_count,
        )

    def _phase3_reject_non_cptp(
        self,
        report: CPTPValidationReport,
    ) -> None:
        r"""
        FASE 3.Ω — Rechazo monádico de morfismos no CPTP.

        Si el certificado sella ``is_cptp=⊥``, se eleva

        \[
        {\tt TopologicalInvariantError}
        \supset
        {\tt TraceAnomalyError}
        \]

        con payload forense. El caller del estrato WISDOM debe atrapar
        y activar Crowbar / cuarentena del canal semántico.

        Args:
            report: Certificado ya sellado.

        Raises:
            TopologicalInvariantError: Siempre que ``not report.is_cptp``.
        """
        if report.is_cptp:
            return
        raise TopologicalInvariantError(
            "TraceAnomalyError: los operadores inyectados violan los "
            "postulados CPTP de Dirac–von Neumann. "
            f"r_TP={report.kraus_completeness_residual:.4e}, "
            f"λ_min(Choi)={report.choi_min_eigenvalue:.4e}, "
            f"rank={report.choi_rank}, "
            f"δ_⋄={report.tp_diamond_defect:.4e}, "
            f"δ_NS={report.nosignal_defect:.4e}, "
            f"Δ_U={report.unitality_residual:.4e}."
        )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  COMPOSICIÓN PÚBLICA — FUNCTOR DE AUDITORÍA CPTP                      ║
    # ║  ObserveKraus (F1) ⟶ OrientChoi (F2) ⟶ CP/PPT/NS/Seal (F3)           ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def audit_quantum_channel(
        self,
        kraus_operators: KrausEnsemble,
        dimension_d: int,
        tolerance: float = 1e-12,
        *,
        strict: bool = True,
    ) -> CPTPValidationReport:
        r"""
        Audita el canal cuántico: completitud de Kraus + positividad de Choi.

        Composición estricta de las tres fases anidadas. El objeto terminal
        de cada fase es el objeto inicial de la siguiente:

        .. code-block:: text

            ┌──────────────────────────────────────────────────────────┐
            │ FASE 1  ObserveKraus                                     │
            │   1.1 validate_hilbert_dimension                         │
            │   1.2 validate_kraus_ensemble                            │
            │   1.3 audit_kraus_operator_norms                         │
            │   1.4 normalize_kraus_gauge                              │
            │   1.5 fix_kraus_u1_gauge                                 │
            │   1.6 kraus_hs_gram_rank                                 │
            │   1.Ω observe_kraus_certificate  ──► Phase1Cert ──┐      │
            ├───────────────────────────────────────────────────┼──────┤
            │ FASE 2  OrientChoi  ◄─────────────────────────────┘      │
            │   2.1 trace_preserving_residual                          │
            │   2.2 unitality_residual                                 │
            │   2.3 construct_choi_matrix                              │
            │   2.4 certify_choi_hermiticity                           │
            │   2.5 choi_partial_traces                                │
            │   2.6 crosscheck_kraus_choi_duality                      │
            │   2.Ω orient_choi_geometry  ──► Phase2Cert ──┐           │
            ├──────────────────────────────────────────────┼───────────┤
            │ FASE 3  CP / PPT / NS / Seal  ◄──────────────┘           │
            │   3.1 spectral_complete_positivity                       │
            │   3.2 ppt_separability                                   │
            │   3.3 nosignal_causality_witness                         │
            │   3.4 rank_consistency                                   │
            │   3.5 decide_cptp_lattice                                │
            │   3.6 seal_validation_report                             │
            │   3.Ω reject_non_cptp  (rama monádica de rechazo)        │
            └──────────────────────────────────────────────────────────┘

        Args:
            kraus_operators: Secuencia de operadores de Kraus \(\{M_k\}\).
            dimension_d: Dimensión de \(\mathcal{H}_A\).
            tolerance: ε de Wilkinson para comparaciones FPU de decisión
                CP/TP (default ``1e-12``). El rango numérico usa además un
                suelo relativo \(\varepsilon_W\).
            strict: Si es verdadero (default), un canal no CPTP eleva
                ``TopologicalInvariantError``. Si es falso, se devuelve el
                reporte sellado con ``is_cptp=False`` (soft-audit).

        Returns:
            CPTPValidationReport: certificado inmutable de consistencia.

        Raises:
            TopologicalInvariantError: Inconsistencia dimensional, Choi
                no hermítica, ensamble nulo, checksum de isomorfismo
                corrupto, o (si ``strict``) violación CPTP irreversible.

        Notas Matemáticas:
            \[
            \Lambda_{\mathcal{E}}
            =\sum_k\operatorname{vec}_F(M_k)\operatorname{vec}_F(M_k)^\dagger,
            \qquad
            \mathcal{E}\in\mathrm{CPTP}
            \iff
            \Lambda\succeq 0
            \;\land\;
            \sum_k M_k^\dagger M_k=I.
            \]
        """
        if not isinstance(tolerance, (int, float, np.floating)) or tolerance < 0.0 or not math.isfinite(float(tolerance)):
            raise TopologicalInvariantError(
                f"tolerance debe ser real no negativo y finito; recibido {tolerance}."
            )
        tolerance = float(tolerance)

        # ── FASE 1 · ObserveKraus ─────────────────────────────────────────
        phase1_cert = self._phase1_observe_kraus_certificate(
            kraus_operators, dimension_d, tolerance
        )

        # ── FASE 2 · OrientChoi  (continúa Phase1KrausSanitizerCertificate)
        phase2_cert = self._phase2_orient_choi_geometry(phase1_cert)
        is_tp: BoolLattice = phase2_cert.completeness_residual <= tolerance
        is_unital: BoolLattice = phase2_cert.unitality_residual <= tolerance

        # ── FASE 3 · CP / PPT / NS / Seal  (continúa Phase2Choi…) ────────
        (
            _eigs,
            min_eigen,
            max_eigen,
            choi_rank,
            gap,
            cond,
            is_cp,
        ) = self._phase3_spectral_complete_positivity(phase2_cert)
        is_separable = self._phase3_ppt_separability(phase2_cert)
        nosignal = self._phase3_nosignal_causality_witness(phase2_cert)
        self._phase3_rank_consistency(phase2_cert, choi_rank)
        is_cptp = self._phase3_decide_cptp_lattice(
            is_tp, is_cp, phase2_cert.completeness_residual, min_eigen
        )

        report = self._phase3_seal_validation_report(
            certificate=phase2_cert,
            is_cptp=is_cptp,
            min_eigen=min_eigen,
            max_eigen=max_eigen,
            choi_rank=choi_rank,
            spectral_gap=gap,
            condition_number=cond,
            is_separable=is_separable,
            nosignal_defect=nosignal,
            is_unital=is_unital,
        )

        if strict:
            self._phase3_reject_non_cptp(report)
        return report

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades de fábrica (canales de referencia para tests / calibración)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def identity_kraus(dimension_d: int) -> List[ComplexMatrix]:
        r"""Canal identidad: \(\mathcal{E}(\rho)=\rho\), Kraus \(\{I_d\}\)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        return [np.eye(dimension_d, dtype=np.complex128)]

    @staticmethod
    def depolarizing_kraus(dimension_d: int, p: float) -> List[ComplexMatrix]:
        r"""
        Canal despolarizante de Weyl–Heisenberg en dimensión \(d\):

        \[
        \mathcal{E}_p(\rho)
        =(1-p)\rho+p\,\frac{I}{d}\operatorname{Tr}(\rho),
        \qquad p\in[0,1].
        \]

        Kraus vía la base unitaria \(U_{jk}=X^j Z^k\) con pesos

        \[
        q_{00}=1-p+\frac{p}{d^2},
        \qquad
        q_{jk}=\frac{p}{d^2}\quad(j,k)\ne(0,0),
        \]

        de modo que \(\sum q_{jk}=1\) y \(\sum_k M_k^\dagger M_k=I\).
        Para \(d=2\) reduce al canal de Pauli estándar
        \(\sqrt{1-3p/4}\,I,\;\sqrt{p/4}\,X,Y,Z\).
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p debe estar en [0,1]; recibido {p}")
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        d = int(dimension_d)
        kraus: List[ComplexMatrix] = []
        q0 = 1.0 - p + p / float(d * d)
        qi = p / float(d * d)
        m0_scale = math.sqrt(max(q0, 0.0))
        kraus.append(m0_scale * np.eye(d, dtype=np.complex128))

        if qi <= 0.0:
            return kraus

        omega = np.exp(2.0j * np.pi / d)
        shift = np.zeros((d, d), dtype=np.complex128)
        clock = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            shift[i, (i - 1) % d] = 1.0 + 0.0j
            clock[i, i] = omega ** i

        noise_scale = math.sqrt(qi)
        for i in range(d):
            for j in range(d):
                if i == 0 and j == 0:
                    continue
                w_ij = np.linalg.matrix_power(shift, i) @ np.linalg.matrix_power(clock, j)
                kraus.append((noise_scale * w_ij).astype(np.complex128, copy=False))
        return kraus

    @staticmethod
    def amplitude_damping_kraus(gamma: float) -> List[ComplexMatrix]:
        r"""
        Canal de amortiguamiento de amplitud (qubit, \(d=2\)):

        \[
        M_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix},
        \quad
        M_1=\begin{pmatrix}0&\sqrt{\gamma}\\0&0\end{pmatrix},
        \qquad\gamma\in[0,1].
        \]

        Es CPTP y **no** unital para \(\gamma\in(0,1]\).
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma debe estar en [0,1]; recibido {gamma}")
        m0 = np.array(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, math.sqrt(1.0 - gamma) + 0.0j]],
            dtype=np.complex128,
        )
        m1 = np.array(
            [[0.0 + 0.0j, math.sqrt(gamma) + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
            dtype=np.complex128,
        )
        return [m0, m1]

    @staticmethod
    def phase_damping_kraus(lam: float) -> List[ComplexMatrix]:
        r"""
        Canal de amortiguamiento de fase (qubit, \(d=2\)):

        \[
        M_0=\begin{pmatrix}1&0\\0&\sqrt{1-\lambda}\end{pmatrix},
        \quad
        M_1=\begin{pmatrix}0&0\\0&\sqrt{\lambda}\end{pmatrix},
        \qquad\lambda\in[0,1].
        \]

        Es unital y CPTP (ruido puramente de-fasaje, entanglement-breaking
        en \(\lambda=1\)).
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam debe estar en [0,1]; recibido {lam}")
        m0 = np.array(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, math.sqrt(1.0 - lam) + 0.0j]],
            dtype=np.complex128,
        )
        m1 = np.array(
            [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, math.sqrt(lam) + 0.0j]],
            dtype=np.complex128,
        )
        return [m0, m1]

    @staticmethod
    def bit_flip_kraus(p: float) -> List[ComplexMatrix]:
        r"""
        Canal bit-flip (qubit): \(\mathcal{E}(\rho)=(1-p)\rho+p\,X\rho X\), \(p\in[0,1]\).
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p debe estar en [0,1]; recibido {p}")
        eye = np.eye(2, dtype=np.complex128)
        pauli_x = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
        return [math.sqrt(1.0 - p) * eye, math.sqrt(p) * pauli_x]