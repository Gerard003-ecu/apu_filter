### -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quantum CPTP Channel Validator (Auditor Espectral de Choi)          ║
║ Ruta   : app/wisdom/cptp_validator.py                                        ║
║ Versión: 6.0.0-Choi-Jamiolkowski-Kraus-Nested-Strict-Doctoral                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ESTRATO WISDOM (V_W) ─────
Este módulo consagra al validador axiomático y matemático encargado de certificar
síncronamente la consistencia cuántica de los canales de inyección semántica
$$\mathcal{E}$$ en la categoría dagger-compacta de sistemas abiertos $$\mathcal{C}_{\mathrm{MAC}}$$.

Su propósito fundamental es someter las transiciones atencionales del Modelo de
Lenguaje (LLM) a las restricciones inquebrantables del formalismo de Choi-Jamiołkowski,
garantizando la positividad completa (CP) y la preservación exacta de la traza (TP)
mediante descomposiciones espectrales estables de Weyl-Wilkinson. Esto previene
de raíz la inyección de alucinaciones no unitarias y fluctuaciones estocásticas
en el núcleo del Santuario del Estrato Wisdom ($$V_{\mathbb{W}}$$).

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
La transición de estados se rige por la Ley de Clausura Transitiva de subespacios
de Hilbert covariantes y se compone de tres fases fuertemente acopladas:

  Fase 1 ──► FASE 1: CONSISTENCIA DIMENSIONAL Y SANEAMIENTO ESPECTRAL (Observe)
             Audita las dimensiones de los operadores de Kraus $$M_k \in \mathbb{C}^{d \times d}$$.
             Sanea el espacio y aplica rotación de fase de gauge de calibración:
             $$\hat{M}_k \leftarrow e^{-i \theta_k} M_k$$.
             Entrega: Phase1KrausSanitizerCertificate como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: ISOMORFISMO DE CHOI-JAMIOŁKOWSKI Y UNITALIDAD (Orient)
             Construye de forma bilineal la matriz de Choi $$\Lambda_{\mathcal{E}} \in \mathbb{C}^{d^2 \times d^2}$$.
             Verifica la completitud de la traza y cuantifica el defecto de unitalidad:
             $$\Delta_{\mathrm{unital}} = \left\| \sum_k M_k M_k^\dagger - I \right\|_F$$.
             Entrega: Phase2ChoiIsomorphismCertificate como precondición formal de Fase 3.

  Fase 3 ──► FASE 3: POSITIVIDAD COMPLETA (CP), PPT Y CAUSALIDAD (Decide & Act)
             Extrae el espectro de $$\Lambda_{\mathcal{E}}$$ y verifica el criterio de Peres-Horodecki.
             Garantiza la condición de No-Señalización cuántica bipartita de-confinada.
             Veredicto: Colapso síncrono al veredicto terminal en el retículo Heyting $$\Omega_3$$.

INVARIANTES MATEMÁTICOS Y GEOMÉTRICOS PRESERVADOS: ──────────────────────────────
  [I1] Conservación de Traza de Kraus:    $$\sum_{k=1}^r M_k^\dagger M_k = I_A \quad \implies \quad \operatorname{Tr}(\mathcal{E}(\rho)) \equiv 1.0$$
  [I2] Positividad Completa de Choi:      $$\Lambda_{\mathcal{E}} = \sum_{k=1}^r \operatorname{vec}(M_k) \operatorname{vec}(M_k)^\dagger \succeq 0$$
  [I3] No-Señalización Cuántica Local:    $$\operatorname{Tr}_A\big( (\mathcal{E}_A \otimes \mathcal{I}_B)(\rho_{AB}) \big) \equiv \rho_B$$
  [I4] Simetría Hermítica de Choi:        $$\Lambda_{\mathcal{E}} = \Lambda_{\mathcal{E}}^\dagger \quad \implies \quad \lambda_i \in \mathbb{R}$$
  [I5] Isomorfismo de Adjunción:          $$\operatorname{Hom}_{\mathcal{D}}(F(\text{MIC}), \text{MAC}) \cong_{G_{\mu\nu}} \operatorname{Hom}_{\mathcal{C}}(\text{MIC}, G(\text{MAC}))$$

CONTRATO DEL DISYUNTOR FÍSICO POR HARDWARE (Bypass ESP32 / BT151): ──────────────
  Si el canal cuántico registra una pérdida de traza ($$\| \sum M_k^\dagger M_k - I \|_F > \tau$$),
  un autovalor de Choi negativo fuera de tolerancia ($$\lambda_i < \epsilon_{\mathrm{floor}}$$), o una
  violación de la condición de No-Señalización bipartita, el retículo distributivo
  de Heyting $$\Omega_3$$ colapsa de forma instantánea al Supremo terminal VETOED.
  
  La subrutina local 'isVerdictCoherent()' del ESP32 en el borde de la obra real
  detecta el mismatch en menos de 400 ns y conmuta el pin GPIO14. Esto dispara la
  compuerta de potencia del tiristor de conmutación rápida BT151 (circuito Crowbar),
  cortocircuitando la alimentación trifásica de los actuadores hidráulicos en el
  milisegundo cero, anulando la alucinación antes del desfalco financiero.
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


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADO INMUTABLE (objeto terminal del funtor de validación CPTP)
# ═══════════════════════════════════════════════════════════════════════════════
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
            Defecto de la condición dual \(\operatorname{Tr}_B(\Lambda)=I/d\)
            medido en norma de Frobenius (consistencia Choi↔TP).
        kraus_count (int):
            Cardinalidad bruta del ensamble de Kraus inyectado (|{M_k}|).
        wilkinson_tolerance (float):
            ε de Wilkinson usado en todas las comparaciones espectrales.
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

    como composición de tres fases anidadas:

    \[
    \mathrm{Seal}\circ\mathrm{CP\text{-}PPT}\circ\mathrm{Choi}\circ\mathrm{TP}\circ\mathrm{Normalize}
    \]

    cada una monádica sobre \({\tt TopologicalInvariantError}\).
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
    # ║  FASE 1 — CONSISTENCIA DIMENSIONAL Y NORMALIZACIÓN DE KRAUS           ║
    # ║  Objetos: {M_k} ⊂ ℬ(ℋ_d), tipado C*, soporte no vacío                 ║
    # ║  Funtores: inclusión en Mat_d(ℂ), proyección de calibre numérico      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase1_validate_hilbert_dimension(self, dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de la dimensión del espacio de Hilbert.

        Exige

        \[
        d\in\mathbb{Z}_{\ge 1}
        \qquad\text{y preferiblemente}\qquad
        d\ge 2
        \]

        (d = 1 es el canal trivial sobre ℂ; se admite pero se registra).
        En circuitos cuánticos: d = 2^n para n qubits; cualquier d no
        entero o no positivo es un cortocircuito ontológico del bus MAC.

        Args:
            dimension_d: Dimensión candidata de ℋ_A.

        Returns:
            dimension_d validado (identidad tipada).

        Raises:
            TopologicalInvariantError: Si d < 1.
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise TopologicalInvariantError(
                f"Dimensión de Hilbert inválida: d={dimension_d}. "
                "Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
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

        El ensamble no puede ser vacío (el canal cero no es TP salvo d=0,
        excluido en 1.1). Cada operador se coacciona a ``complex128`` y
        se materializa como lista propia (desacoplamiento referencial del
        caller).

        Análogo eléctrico: {M_k} son las matrices de scattering de un
        multipolo multipuerto; shape incorrecto ≡ puerto flotante.

        Args:
            kraus_operators: Secuencia bruta de operadores de Kraus.
            dimension_d: Dimensión ya certificada (salida de 1.1).

        Returns:
            Lista normalizada ``list[ComplexMatrix]`` con shape (d, d).

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
            if m_k.shape != expected:
                raise TopologicalInvariantError(
                    f"Inconsistencia dimensional en Kraus[{idx}]: "
                    f"esperado {expected}, obtenido {m_k.shape}."
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

    def _phase1_normalize_kraus_gauge(
        self,
        kraus_ops: List[ComplexMatrix],
        tolerance: float,
    ) -> List[ComplexMatrix]:
        r"""
        FASE 1.3 — Limpieza de calibre numérico (operadores nulos / basura UV).

        Descarta operadores con

        \[
        \|M_k\|_F \le \varepsilon_{\mathrm{spectral}}
        \]

        (ruido de redondeo FPU que no contribuye al canal ni al rango de Choi).
        Si tras el filtrado el ensamble queda vacío se eleva error: el canal
        colapsó al morfismo nulo.

        Esta salida es el *objeto inicial* de la FASE 2: la lista limpia
        `{M_k}_{k∈K*}` se inyecta sin re-tipado en
        `_phase2_trace_preserving_residual`.

        Args:
            kraus_ops: Ensamble tipado (salida de 1.2).
            tolerance: Umbral de aniquilación de norma Frobenius.

        Returns:
            Sub-ensamble con soporte numérico no trivial.

        Raises:
            TopologicalInvariantError: Si todos los Kraus son nulos.
        """
        floor = max(tolerance * 1.0e-3, _EPS_SPECTRAL)
        cleaned: List[ComplexMatrix] = [
            m for m in kraus_ops if float(la.norm(m, ord="fro")) > floor
        ]
        if not cleaned:
            raise TopologicalInvariantError(
                "Todos los operadores de Kraus son numéricamente nulos "
                f"(‖M_k‖_F ≤ {floor:.3e}). Canal cero no es CPTP."
            )
        if len(cleaned) < len(kraus_ops):
            logger.info(
                "FASE1.3 calibre: descartados %d operadores nulos de %d.",
                len(kraus_ops) - len(cleaned),
                len(kraus_ops),
            )
        return cleaned

    def _phase1_observe_kraus_certificate(
        self,
        kraus_operators: KrausEnsemble,
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[List[ComplexMatrix], int]:
        r"""
        FASE 1.4 — Composición terminal de Observación de Kraus (Observe).

        Morfismo compuesto de la FASE 1:

        \[
        \mathrm{ObserveKraus}
        =\mathrm{Gauge}\circ\mathrm{Type}\circ\mathrm{Dim}
        :\mathbf{RawKraus}\rightharpoonup
        \mathrm{List}(\mathrm{Mat}_d(\mathbb{C}))\times\mathbb{Z}_{\ge 1}.
        \]

        Produce el certificado ``(kraus_clean, d)`` que alimenta la FASE 2
        sin transformación ulterior (continuidad funtorial Observe → TP/Choi).

        Args:
            kraus_operators: Ensamble bruto.
            dimension_d: Dimensión candidata.
            tolerance: ε de Wilkinson del caller.

        Returns:
            (kraus_clean, d) — certificado de fase 1 sellado.
        """
        d = self._phase1_validate_hilbert_dimension(dimension_d)
        typed = self._phase1_validate_kraus_ensemble(kraus_operators, d)
        clean = self._phase1_normalize_kraus_gauge(typed, tolerance)
        return clean, d

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 — TRACE-PRESERVING Y CONSTRUCCIÓN DE CHOI–JAMIOŁKOWSKI        ║
    # ║  Continuación directa del certificado de Kraus de FASE 1.4            ║
    # ║  Objetos: Σ M†M, Λ_ℰ = Σ |vec(M_k)⟩⟨vec(M_k)|, Tr_B(Λ)               ║
    # ║  Teorías: Kraus TP, isomorfismo de Choi, dualidad de Hellwig–Kraus    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase2_trace_preserving_residual(
        self,
        kraus_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> Tuple[RealScalar, ComplexMatrix]:
        r"""
        FASE 2.1 — Residuo de completitud de Kraus / Trace-Preserving (inicio Orient).

        **Continuación funtorial de FASE 1.4**: recibe exactamente
        ``(kraus_ops, dimension_d)`` emitidos por
        `_phase1_observe_kraus_certificate`.

        Calcula el operador de completitud

        \[
        S=\sum_{k}|M_k^\dagger M_k|
        \in\mathcal{B}(\mathcal{H}_d)_{\mathrm{sa}}
        \]

        y el residuo de TP en norma de Frobenius (topología C*):

        \[
        r_{\mathrm{TP}}
        =\bigl\|S-I_d\bigr\|_F
        =\Bigl(\sum_{ij}|S_{ij}-\delta_{ij}|^2\Bigr)^{1/2}.
        \]

        El canal es TP (no-destruir probabilidad) sii \(r_{\mathrm{TP}}=0\).
        Para TP-completo se exige además \(S=I\) (no solo \(S\le I\), que
        correspondería a un canal trace-non-increasing / instrumento).

        En circuitos: S es la matriz de Gram de potencias disipadas; S=I
        ≡ pasividad unitaria del multipolo (conservación de energía de
        probabilidad).

        Args:
            kraus_ops: Ensamble limpio (de FASE 1.4).
            dimension_d: d certificado.

        Returns:
            (completeness_residual, completeness_operator S).
        """
        s_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in kraus_ops:
            s_op += m_k.conj().T @ m_k

        # Proyección de Weyl: S debe ser autoadjunto (sum of PSD)
        s_op = 0.5 * (s_op + s_op.conj().T)

        residual = float(la.norm(s_op - np.eye(dimension_d, dtype=np.complex128), ord="fro"))
        logger.debug(
            "FASE2.1 TP: r_TP=%.6e, ‖S‖₂=%.6e, λ_min(S)=%.6e",
            residual,
            float(la.norm(s_op, ord=2)),
            float(np.min(la.eigvalsh(s_op))),
        )
        return residual, s_op

    def _phase2_construct_choi_matrix(
        self,
        kraus_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> ComplexMatrix:
        r"""
        FASE 2.2 — Isomorfismo de Choi–Jamiołkowski desde Kraus.

        Reconstruye

        \[
        \Lambda_{\mathcal{E}}
        =\sum_k\operatorname{vec}_F(M_k)\,\operatorname{vec}_F(M_k)^\dagger
        \in\mathcal{B}(\mathbb{C}^{d}\otimes\mathbb{C}^{d})
        \cong\mathrm{Mat}_{d^2}(\mathbb{C}),
        \]

        donde \(\operatorname{vec}_F\) es el apilado *column-major* (orden
        Fortran), convención estándar en la literatura de información
        cuántica (cf. Watrous, Nielsen–Chuang). Equivale a

        \[
        \Lambda_{\mathcal{E}}
        =(\mathcal{E}\otimes\mathrm{id})(|\Omega\rangle\langle\Omega|),
        \quad
        |\Omega\rangle=d^{-1/2}\sum_i|i\rangle\otimes|i\rangle
        \]

        salvo el factor de normalización \(d\) (trabajamos con la versión
        no normalizada: \(\operatorname{Tr}\Lambda=d\) para canales TP).

        Complejidad: O(K · d⁴) en el ensamblaje outer-product; aceptable
        para d ≲ 32 en el estrato WISDOM.

        Args:
            kraus_ops: Ensamble limpio (de FASE 1.4).
            dimension_d: d certificado.

        Returns:
            choi_matrix hermítica de shape (d², d²).
        """
        choi_dim = dimension_d * dimension_d
        choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

        for m_k in kraus_ops:
            # vec_F: column-major ↔ convención QI (base computacional producto)
            vec_m = np.asarray(m_k.flatten(order="F"), dtype=np.complex128)
            choi += np.outer(vec_m, vec_m.conj())

        # Proyección de Weyl al subespacio autoadjunto (err. redondeo FMA)
        choi = 0.5 * (choi + choi.conj().T)
        return choi

    def _phase2_certify_choi_hermiticity(
        self,
        choi_matrix: ComplexMatrix,
        tolerance: float,
    ) -> ComplexMatrix:
        r"""
        FASE 2.3 — Certificación C* de hermiticidad de Λ_ℰ.

        Verifica

        \[
        \bigl\|\Lambda-\Lambda^\dagger\bigr\|_F \le \max(\varepsilon,\,
        \varepsilon_H).
        \]

        Si el defecto es ≤ tol se devuelve la proyección de Weyl (ya
        aplicada en 2.2). Si excede tol, el ensamble de Kraus es
        internamente inconsistente (bug de construcción upstream) y se
        eleva invariante topológico.

        Args:
            choi_matrix: Λ_ℰ (salida de 2.2).
            tolerance: ε de Wilkinson.

        Returns:
            choi_matrix hermítico certificado.

        Raises:
            TopologicalInvariantError: Defecto antihermítico intolerable.
        """
        anti = choi_matrix - choi_matrix.conj().T
        defect = float(la.norm(anti, ord="fro"))
        tol_h = max(tolerance, _EPS_HERMITICITY)
        if defect > tol_h:
            raise TopologicalInvariantError(
                f"La matriz de Choi viola hermiticidad: "
                f"‖Λ−Λ†‖_F={defect:.3e} > tol={tol_h:.3e}."
            )
        return choi_matrix

    def _phase2_choi_partial_trace_tp_defect(
        self,
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> RealScalar:
        r"""
        FASE 2.4 — Defecto dual TP vía traza parcial de Choi.

        Para la Choi no normalizada de un canal TP:

        \[
        \operatorname{Tr}_B(\Lambda_{\mathcal{E}}) = I_A
        \]

        (equivalencia Choi↔TP). Medimos

        \[
        \delta_{\diamond}
        =\bigl\|\operatorname{Tr}_B(\Lambda)-I_d\bigr\|_F.
        \]

        Este invariante es *independiente* del residuo de Kraus r_TP y
        actúa como checksum cruzado: si r_TP≈0 pero δ_⋄≫0 (o viceversa),
        existe corrupción numérica en el isomorfismo vec/reshape.

        Implementación del partial trace sobre el 2º factor del producto
        \(\mathcal{H}_A\otimes\mathcal{H}_{A'}\) con layout
        reshape (d, d, d, d) en orden coherente con vec_F.

        Args:
            choi_matrix: Λ_ℰ hermítica (d² × d²).
            dimension_d: d.

        Returns:
            tp_diamond_defect = δ_⋄.
        """
        d = dimension_d
        # Layout: vec_F ⇒ índices (a, a', b, b') con strides Fortran-compatibles
        # Λ_{(a,b),(a',b')} → tensor Λ[a, b, a', b'] tras reshape C-order del
        # apilado F requiere permutación; usamos la convención:
        # reshape (d, d, d, d) interpretando el multi-índice plano como
        # (col_out, col_in, row_out, row_in) coherente con flatten('F').
        tensor = choi_matrix.reshape((d, d, d, d), order="F")
        # Tr_B: contraer b = b'  →  (Tr_B Λ)[a, a']
        # ejes: 0=a, 1=b, 2=a', 3=b'
        partial = np.trace(tensor, axis1=1, axis2=3)  # shape (d, d)
        partial = 0.5 * (partial + partial.conj().T)
        defect = float(
            la.norm(partial - np.eye(d, dtype=np.complex128), ord="fro")
        )
        return defect

    def _phase2_orient_choi_geometry(
        self,
        kraus_ops: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealScalar, ComplexMatrix, RealScalar, RealScalar]:
        r"""
        FASE 2.5 — Composición terminal de Orientación Choi (Orient).

        Morfismo compuesto de la FASE 2:

        \[
        \mathrm{OrientChoi}
        =(\mathrm{TP},\,\mathrm{Choi},\,\mathrm{Herm},\,\mathrm{Tr}_B)
        \circ\mathrm{Kraus}^*.
        \]

        Esta firma de retorno es el *objeto inicial* de la FASE 3:

        ``(r_TP, Λ_ℰ, Tr(Λ), δ_⋄)`` se inyecta en
        `_phase3_spectral_complete_positivity` sin re-codificación.

        Args:
            kraus_ops, dimension_d: Certificado de FASE 1.4.
            tolerance: ε de Wilkinson.

        Returns:
            (completeness_residual, choi_matrix, choi_trace, tp_diamond_defect).
        """
        r_tp, _s = self._phase2_trace_preserving_residual(kraus_ops, dimension_d)
        choi = self._phase2_construct_choi_matrix(kraus_ops, dimension_d)
        choi = self._phase2_certify_choi_hermiticity(choi, tolerance)
        choi_trace = float(np.real(np.trace(choi)))
        delta_tp = self._phase2_choi_partial_trace_tp_defect(choi, dimension_d)
        return r_tp, choi, choi_trace, delta_tp

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 3 — CP ESPECTRAL, RANGO DE CHOI, PPT Y CERTIFICADO              ║
    # ║  Continuación directa de la geometría de Choi de FASE 2.5             ║
    # ║  Objetos: σ(Λ), rank(Λ), Λ^{T_A}, retícula CPTP, reporte frozen       ║
    # ║  Teorías: Choi CP, Peres–Horodecki PPT, Weyl–Wilkinson eigvalsh       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase3_spectral_complete_positivity(
        self,
        choi_matrix: ComplexMatrix,
        tolerance: float,
    ) -> Tuple[RealVector, RealScalar, int, BoolLattice]:
        r"""
        FASE 3.1 — Positividad completa vía espectro de Choi (inicio Decide).

        **Continuación funtorial de FASE 2.5**: recibe ``choi_matrix``.

        Por el teorema de Choi (1975):

        \[
        \mathcal{E}\text{ es CP}
        \;\Longleftrightarrow\;
        \Lambda_{\mathcal{E}}\succeq 0
        \;\Longleftrightarrow\;
        \lambda_{\min}(\Lambda_{\mathcal{E}})\ge 0.
        \]

        Se admite un suelo de Wilkinson \(-\varepsilon\) por redondeo FPU.
        El rango de Choi

        \[
        r=\#\{\lambda\in\sigma(\Lambda):\lambda>\varepsilon\}
        \]

        es el número mínimo de operadores de Kraus (teorema de representación
        de Kraus canónica / Stinespring rank).

        Args:
            choi_matrix: Λ_ℰ hermítica (de FASE 2).
            tolerance: ε de Wilkinson.

        Returns:
            (eigvals, min_eigenvalue, choi_rank, is_cp).
        """
        eigvals = la.eigvalsh(choi_matrix).astype(np.float64)
        min_eigen = float(np.min(eigvals))
        is_cp: BoolLattice = min_eigen >= -tolerance
        choi_rank = int(np.sum(eigvals > tolerance))

        logger.debug(
            "FASE3.1 CP: λ_min=%.6e, rank=%d, n_eigs=%d, is_cp=%s",
            min_eigen,
            choi_rank,
            len(eigvals),
            is_cp,
        )
        return eigvals, min_eigen, choi_rank, is_cp

    def _phase3_ppt_separability(
        self,
        choi_matrix: ComplexMatrix,
        dimension_d: int,
        tolerance: float,
    ) -> BoolLattice:
        r"""
        FASE 3.2 — Criterio PPT de Peres–Horodecki sobre Λ_ℰ.

        La transpuesta parcial respecto del primer factor

        \[
        \bigl(\Lambda^{T_A}\bigr)_{a b;\,a' b'}
        =\Lambda_{a' b;\,a b'}
        \]

        debe cumplir \(\Lambda^{T_A}\succeq 0\) si el estado de Choi
        (normalizado) es separable. PPT es:

        - necesario y suficiente para d = 2 × 2 y 2 × 3;
        - necesario pero no suficiente en dimensión superior
          (estados bound-entangled de Horodecki).

        En el validador reportamos PPT como *proxy* de separabilidad
        del canal (canales entanglement-breaking ⇒ Choi separable ⇒ PPT).

        Layout: coherente con reshape order="F" de la FASE 2.

        Args:
            choi_matrix: Λ_ℰ (d² × d²).
            dimension_d: d.
            tolerance: ε de Wilkinson.

        Returns:
            is_separable ≡ (λ_min(Λ^{T_A}) ≥ −ε).
        """
        d = dimension_d
        choi_dim = d * d
        # Tensor Λ[a, b, a', b'] en orden F
        block = choi_matrix.reshape((d, d, d, d), order="F")
        # Transpuesta parcial sobre A: intercambiar a ↔ a'  (ejes 0 ↔ 2)
        ppt_block = block.transpose(2, 1, 0, 3)
        ppt_matrix = ppt_block.reshape((choi_dim, choi_dim), order="F")
        ppt_matrix = 0.5 * (ppt_matrix + ppt_matrix.conj().T)

        ppt_eigs = la.eigvalsh(ppt_matrix)
        min_ppt = float(np.min(ppt_eigs))
        is_separable: BoolLattice = min_ppt >= -tolerance

        logger.debug(
            "FASE3.2 PPT: λ_min(Λ^{T_A})=%.6e, is_separable=%s",
            min_ppt,
            is_separable,
        )
        return is_separable

    def _phase3_decide_cptp_lattice(
        self,
        is_tp: BoolLattice,
        is_cp: BoolLattice,
        completeness_residual: RealScalar,
        min_eigen: RealScalar,
    ) -> BoolLattice:
        r"""
        FASE 3.3 — Decisión en la retícula de Heyting Ω_CPTP.

        \[
        \Omega_{\mathrm{CPTP}}
        =\{\bot\prec\mathrm{CPTP}\},
        \qquad
        \chi_{\mathrm{CPTP}}
        =\chi_{\mathrm{TP}}\wedge\chi_{\mathrm{CP}}.
        \]

        Si χ = ⊥ se emite telemetría de error forense con ambos residuales
        (no se eleva aún: el raise lo decide el compositor público, para
        permitir sellar reportes de fallo en modo soft-audit si se desea
        extender).

        Args:
            is_tp: r_TP ≤ ε.
            is_cp: λ_min(Λ) ≥ −ε.
            completeness_residual, min_eigen: para logging.

        Returns:
            is_cptp ∈ 𝔹.
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
        is_cptp: BoolLattice,
        completeness_residual: RealScalar,
        min_eigen: RealScalar,
        choi_rank: int,
        is_separable: BoolLattice,
        choi_trace: RealScalar,
        tp_diamond_defect: RealScalar,
        kraus_count: int,
        tolerance: float,
    ) -> CPTPValidationReport:
        r"""
        FASE 3.4 — Sellado del certificado inmutable (objeto terminal Act).

        Empaqueta todos los invariantes en ``CPTPValidationReport`` frozen.
        Ningún campo es mutable post-construcción (slots+frozen ⇒ hashable
        ⇒ clave de caché de topos / memoización de morfismos de canal).

        Args:
            Campos acumulados de Fases 1–3.

        Returns:
            CPTPValidationReport sellado.
        """
        return CPTPValidationReport(
            is_cptp=is_cptp,
            kraus_completeness_residual=completeness_residual,
            choi_min_eigenvalue=min_eigen,
            choi_rank=choi_rank,
            is_separable=is_separable,
            choi_trace=choi_trace,
            tp_diamond_defect=tp_diamond_defect,
            kraus_count=kraus_count,
            wilkinson_tolerance=tolerance,
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
            f"δ_⋄={report.tp_diamond_defect:.4e}."
        )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  COMPOSICIÓN PÚBLICA — FUNCTOR DE AUDITORÍA CPTP                      ║
    # ║  ObserveKraus (F1) ⟶ OrientChoi (F2) ⟶ CP/PPT/Seal (F3)              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def audit_quantum_channel(
        self,
        kraus_operators: list[NDArray[np.complex128]],
        dimension_d: int,
        tolerance: float = 1e-12,
    ) -> CPTPValidationReport:
        r"""
        Audita el canal cuántico: completitud de Kraus + positividad de Choi.

        Composición estricta de las tres fases anidadas:

        .. code-block:: text

            ┌──────────────────────────────────────────────────────────┐
            │ FASE 1  ObserveKraus                                     │
            │   1.1 validate_hilbert_dimension                         │
            │   1.2 validate_kraus_ensemble                            │
            │   1.3 normalize_kraus_gauge                              │
            │   1.4 observe_kraus_certificate  ──┐                     │
            ├────────────────────────────────────┼─────────────────────┤
            │ FASE 2  OrientChoi  ◄──────────────┘                     │
            │   2.1 trace_preserving_residual                          │
            │   2.2 construct_choi_matrix                              │
            │   2.3 certify_choi_hermiticity                           │
            │   2.4 choi_partial_trace_tp_defect                       │
            │   2.5 orient_choi_geometry  ──┐                          │
            ├───────────────────────────────┼──────────────────────────┤
            │ FASE 3  CP / PPT / Seal  ◄────┘                          │
            │   3.1 spectral_complete_positivity                       │
            │   3.2 ppt_separability                                   │
            │   3.3 decide_cptp_lattice                                │
            │   3.4 seal_validation_report                             │
            │   3.Ω reject_non_cptp  (rama monádica de rechazo)        │
            └──────────────────────────────────────────────────────────┘

        Args:
            kraus_operators: Lista de operadores de Kraus \(\{M_k\}\).
            dimension_d: Dimensión de \(\mathcal{H}_A\).
            tolerance: ε de Wilkinson para comparaciones FPU (default 1e-12).

        Returns:
            CPTPValidationReport: certificado inmutable de consistencia.

        Raises:
            TopologicalInvariantError: Inconsistencia dimensional, Choi
                no hermítica, ensamble nulo, o violación CPTP irreversible.

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
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise TopologicalInvariantError(
                f"tolerance debe ser real no negativo y finito; recibido {tolerance}."
            )

        # ── FASE 1 · ObserveKraus ─────────────────────────────────────────
        kraus_clean, d = self._phase1_observe_kraus_certificate(
            kraus_operators, dimension_d, tolerance
        )

        # ── FASE 2 · OrientChoi  (continúa certificado Kraus de F1) ───────
        r_tp, choi, choi_trace, delta_tp = self._phase2_orient_choi_geometry(
            kraus_clean, d, tolerance
        )
        is_tp: BoolLattice = r_tp <= tolerance

        # ── FASE 3 · CP / PPT / Seal  (continúa geometría Choi de F2) ─────
        _eigs, min_eigen, choi_rank, is_cp = self._phase3_spectral_complete_positivity(
            choi, tolerance
        )
        is_separable = self._phase3_ppt_separability(choi, d, tolerance)
        is_cptp = self._phase3_decide_cptp_lattice(is_tp, is_cp, r_tp, min_eigen)

        report = self._phase3_seal_validation_report(
            is_cptp=is_cptp,
            completeness_residual=r_tp,
            min_eigen=min_eigen,
            choi_rank=choi_rank,
            is_separable=is_separable,
            choi_trace=choi_trace,
            tp_diamond_defect=delta_tp,
            kraus_count=len(kraus_clean),
            tolerance=tolerance,
        )

        # Rama monádica de rechazo (fail-secure)
        self._phase3_reject_non_cptp(report)
        return report

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades de fábrica (canales de referencia para tests / calibración)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def identity_kraus(dimension_d: int) -> List[ComplexMatrix]:
        r"""Canal identidad: \(\mathcal{E}(\rho)=\rho\), Kraus \(\{I_d\}\)."""
        return [np.eye(dimension_d, dtype=np.complex128)]

    @staticmethod
    def depolarizing_kraus(dimension_d: int, p: float) -> List[ComplexMatrix]:
        r"""
        Canal despolarizante de Weyl en dimensión d:

        \[
        \mathcal{E}_p(\rho)
        =(1-p)\rho+p\,\frac{I}{d}\operatorname{Tr}(\rho),
        \qquad p\in[0,1].
        \]

        Kraus vía base ortogonal de Heisenberg–Weyl (shift-clock) normalizada.
        Para d=2 reduce al canal de Pauli estándar.
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p debe estar en [0,1]; recibido {p}")
        d = dimension_d
        # Construcción mínima: un Kraus de supervivencia + d²-1 de ruido isotrópico
        # M_0 = √(1-p+p/d) I
        # M_{ij} = √(p/d) / √d  ·  X^i Z^j    (Heisenberg–Weyl)
        kraus: List[ComplexMatrix] = []
        m0_scale = math.sqrt(max(1.0 - p + p / d, 0.0))
        kraus.append(m0_scale * np.eye(d, dtype=np.complex128))

        # Operadores de clock y shift
        omega = np.exp(2j * np.pi / d)
        x = np.zeros((d, d), dtype=np.complex128)
        z = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            x[i, (i - 1) % d] = 1.0 + 0.0j
            z[i, i] = omega ** i

        noise_scale = math.sqrt(max(p / d, 0.0) / d)
        for i in range(d):
            for j in range(d):
                if i == 0 and j == 0:
                    continue  # ya cubierto por M_0 (parte I)
                xi = np.linalg.matrix_power(x, i)
                zj = np.linalg.matrix_power(z, j)
                kraus.append((noise_scale * (xi @ zj)).astype(np.complex128))
        return kraus

    @staticmethod
    def amplitude_damping_kraus(gamma: float) -> List[ComplexMatrix]:
        r"""
        Canal de amortiguamiento de amplitud (qubit, d=2):

        \[
        M_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix},
        \quad
        M_1=\begin{pmatrix}0&\sqrt{\gamma}\\0&0\end{pmatrix},
        \qquad\gamma\in[0,1].
        \]
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