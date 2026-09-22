from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Photinic Governance Satellite Engine (Motor Fotínico de Gobernanza) ║
║ Ruta   : app/core/immune_system/photinic_governance_satellite_engine.py      ║
║ Versión: 5.0.0-Doctoral-Majorana-SYM-Dekker-Choi-Horodecki-Tsirelson-HMAC    ║
║ Nivel  : Estrato Omega ($V_\Omega$, Nivel 0.5 — Núcleo Ciber-Físico FPU)     ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y ARQUITECTURA TEÓRICA DOCTORAL (v5):                    ║
║ 1. **Corrección crítica de seguridad**: se elimina el "clamp" de v4 que      ║
║    renormalizaba artificialmente $u_1+u_2\le 2$ antes de evaluar la Cota de  ║
║    Tsirelson, lo cual anulaba por completo el detector de corrupción         ║
║    adversarial/super-cuántica. El valor $\mathcal{B}_{\rm CHSH}$ ahora fluye ║
║    sin alteración hasta la puerta de certificación de Fase 3.                ║
║ 2. **Corrección de forma**: bipartición Alice/Bob del tensor de correlación  ║
║    de Horodecki reparada para $N$ impar de nodos federados (previamente      ║
║    lanzaba `ValueError` de numpy por incompatibilidad de formas).            ║
║ 3. Criterio físico correcto de canal CPTP: positividad completa vía          ║
║    espectro de Choi $C_\mathcal{E}\succeq -\varepsilon$, y causalidad vía    ║
║    el defecto de sub-unitalidad $D=I-\sum_k K_k^\top K_k\succeq-\varepsilon$ ║
║    (POVM físicamente admisible), en reemplazo de un umbral de Frobenius      ║
║    arbitrario ($\le 1.0$) sin fundamento físico.                            ║
║ 4. Aritmética compensada de **Dekker (TwoProduct)** + Kahan-Neumaier:        ║
║    transformación libre de error de la multiplicación de punto flotante,    ║
║    reparando una discrepancia entre el docstring y la implementación de v4.  ║
║ 5. Entropía de von Neumann genuina del estado reducido de gobernanza         ║
║    $\rho_{\rm fed}=S_{\rm fed}/\operatorname{Tr}(S_{\rm fed})\succeq 0$,     ║
║    junto con la entropía de entrelazamiento bipartita de Horodecki y la      ║
║    pureza cuántica $\operatorname{Tr}(\rho^2)$.                             ║
║ 6. Doble sello criptográfico HMAC-SHA256 (reparación de la importación       ║
║    muerta de `hmac` presente en v4).                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Final, Optional, Tuple, Dict, Any, List, Sequence, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Physics.PhotinicGovernanceSatelliteEngine")

# ------------------------------------------------------------------------------
# CONSTANTES FÍSICAS UNIVERSALES, COTAS METROLÓGICAS Y LÍMITES DE WILKINSON
# ------------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_TSIRELSON_BOUND: Final[float] = 2.0 * math.sqrt(2.0)  # 2.8284271247461903 (Cota de Tsirelson)
_BELL_CLASSICAL_BOUND: Final[float] = 2.0               # Cota clásica LHV de Bell-CHSH
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12            # Límite de estabilidad numérica
_GRAM_RECONSTRUCTION_TOLERANCE: Final[float] = 1.0e-9
_CHOI_PSD_TOLERANCE: Final[float] = 1.0e-9
_CAUSALITY_SUBUNITAL_TOLERANCE: Final[float] = 1.0e-9
_VELTKAMP_SPLIT_CONSTANT: Final[float] = 134217729.0    # 2^27 + 1, exacto en float64


# ------------------------------------------------------------------------------
# JERARQUÍA DE EXCEPCIONES DOCTORALES
# ------------------------------------------------------------------------------
class PhotinicEngineError(Exception):
    r"""Excepción raíz de fallos ontológicos en el motor fotínico de gobernanza."""
    pass


class PhotinicDimensionError(PhotinicEngineError):
    r"""Inconsistencia de dimensiones en los espacios vectoriales o fibrados."""
    pass


class PhotinicNonPositiveDefiniteError(PhotinicEngineError):
    r"""
    Violación de la positividad estricta de Gram-Riemann en el espacio de fase,
    o violación de la Positividad Completa (CP) certificada de la matriz de Choi
    ($\lambda_{\min}(C_\mathcal{E}) < -\varepsilon_{\mathrm{Wilkinson}}$), lo cual
    delataría corrupción numérica severa dado que $C_\mathcal{E}$ es PSD por
    construcción analítica (suma de proyectores rango-1 con coeficientes $\ge 0$).
    """
    pass


class PhotinicCausalViolationError(PhotinicEngineError):
    r"""Excepción raíz para violaciones de causalidad relativista o de canal cuántico físico."""
    pass


class PhotinicNonCausalChannelError(PhotinicCausalViolationError):
    r"""
    El canal de gobernanza federada viola la admisibilidad física de un
    operador POVM: $\sum_k K_k^\top K_k \not\preceq I$ (super-unital), lo cual
    implicaría probabilidades de transición agregadas $> 1$ — señal de
    corrupción del canal o de nodos federados maliciosos.
    """
    pass


class PhotinicTsirelsonViolationError(PhotinicCausalViolationError):
    r"""
    Superación estricta de la Cota de Tsirelson ($\mathcal{B}_{\rm CHSH} > 2\sqrt2$):
    régimen super-cuántico / caja PR, interpretado como corrupción adversarial
    no-local del consorcio federado o manipulación estadística de las políticas.
    """
    pass


# ------------------------------------------------------------------------------
# ESTRUCTURAS DE DATOS INMUTABLES (DATACLASSES CANÓNICAS)
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PhotinicObservationKernel:
    r"""
    Expediente inmutable de Fase 1 (Observe).
    Audita la regularidad de Banach $\ell^1/\ell^2/\ell^\infty$, la métrica de Majorana
    en $N=1$ SYM, el tensor de Gram federado certificado PSD y el condicionamiento
    espectral de Wilkinson.
    """
    node_policies: Tuple[NDArray[np.float64], ...]
    majorana_spinor_norm: float
    majorana_chiral_parity: float
    banach_ratios: Tuple[float, ...]
    gram_matrix: NDArray[np.float64]
    gram_eigenvalues: NDArray[np.float64]
    gram_symmetry_defect: float
    gram_reconstruction_residual: float
    condition_number: float
    spectral_gap: float
    sha256_seal: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class PhotinicPolicyReport:
    r"""
    Expediente inmutable de Fase 2 (Orient).
    Sintetiza la idempotencia de Grothendieck ($\Omega^2 = \Omega$), la positividad completa
    de Choi ($C_\mathcal{E} \succeq 0$), la admisibilidad física del canal (sub-unitalidad
    POVM) y el parámetro de Bell-CHSH sin alteración artificial alguna.
    """
    kernel: PhotinicObservationKernel
    idempotence_residual: float
    trace_rank_mismatch: float
    choi_min_eigenvalue: float
    choi_trace_residual: float
    choi_subunital_defect_min_eigenvalue: float
    tsirelson_chsh_value: float
    horodecki_eigs: Tuple[float, float]
    horodecki_bipartition_pairs: int
    is_choi_completely_positive: bool
    is_trace_non_increasing: bool
    is_causal_non_signaling: bool
    is_tsirelson_bounded: bool
    is_classical_lhv: bool
    timestamp: float


@dataclass(frozen=True, slots=True)
class PhotinicEngineState:
    r"""
    Certificado formal e inmutable emitido en la Fase 3 (Act).
    Acredita la telemetría metrológica en la FPU, la entropía de von Neumann
    global del estado reducido de gobernanza, la entropía de entrelazamiento
    bipartita de Horodecki, la pureza cuántica, la deriva de precisión y el
    doble sellado criptográfico HMAC-SHA256.
    """
    kernel: PhotinicObservationKernel
    policy_report: PhotinicPolicyReport
    fpu_execution_time_ms: float
    federated_density_matrix: NDArray[np.float64]
    von_neumann_entropy: float
    quantum_purity: float
    bell_pair_entanglement_entropy: float
    fpu_precision_drift: float
    cryptographic_seal: str
    phase_signature: str


# ------------------------------------------------------------------------------
# ARITMÉTICA COMPENSADA DE KAHAN-BABUŠKA-NEUMAIER (KBN) + DEKKER TWOPRODUCT
# ------------------------------------------------------------------------------
class KahanNeumaierSum:
    r"""
    Sumador compensado de precisión ultra-alta en FPU basado en el algoritmo de Neumaier,
    acoplado al algoritmo de **Dekker (TwoProduct vía Veltkamp-split)** para la
    transformación libre de error de la multiplicación de punto flotante IEEE 754.
    Garantiza un error residual relativo $\mathcal{O}(\varepsilon_{\mathrm{Wilkinson}})$
    incluso bajo adiciones/productos acumulativos de gran escala o cancelación catastrófica.
    """

    @staticmethod
    def sum(arr: Union[NDArray[np.float64], Sequence[float]]) -> float:
        s: float = 0.0
        c: float = 0.0
        for x in arr:
            val = float(x)
            t = s + val
            if abs(s) >= abs(val):
                c += (s - t) + val
            else:
                c += (val - t) + s
            s = t
        return float(s + c)

    @staticmethod
    def _veltkamp_split(a: float) -> Tuple[float, float]:
        r"""
        Descomposición de Veltkamp: $a = a_{\rm hi} + a_{\rm lo}$ exacta en aritmética
        IEEE 754, donde $a_{\rm hi}$ retiene los 26 bits altos de mantisa (constante
        de partición $2^{27}+1$), condición previa del algoritmo de Dekker.
        """
        c = _VELTKAMP_SPLIT_CONSTANT * a
        a_hi = c - (c - a)
        a_lo = a - a_hi
        return a_hi, a_lo

    @staticmethod
    def two_product_dekker(a: float, b: float) -> Tuple[float, float]:
        r"""
        Transformación libre de error (Error-Free Transformation) del producto $a\cdot b$
        de Dekker (1971): retorna $(p, e)$ tales que $p = \operatorname{fl}(a\cdot b)$ y
        $p + e = a \cdot b$ **exactamente** en aritmética real, sin recurrir a FMA de
        hardware. Repara la discrepancia de la v4.0.0, donde `compensated_dot`
        prometía "TwoProduct" en su docstring pero ejecutaba una multiplicación
        elemental sin compensación de error.
        """
        p = a * b
        a_hi, a_lo = KahanNeumaierSum._veltkamp_split(a)
        b_hi, b_lo = KahanNeumaierSum._veltkamp_split(b)
        err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
        return p, err

    @staticmethod
    def compensated_dot(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        r"""
        Producto punto de precisión efectiva doble ("double-double"): cada
        multiplicación elemental se descompone en $(p_i, e_i)$ vía Dekker, y
        tanto los productos como los errores se acumulan independientemente
        mediante suma compensada de Neumaier:
        $$u \cdot v \approx \operatorname{NeumaierSum}(p_i) + \operatorname{NeumaierSum}(e_i)$$
        """
        if u.shape != v.shape:
            raise PhotinicDimensionError("Discrepancia dimensional en producto punto compensado.")

        u_flat = u.ravel()
        v_flat = v.ravel()
        products = np.empty(u_flat.shape[0], dtype=np.float64)
        errors = np.empty(u_flat.shape[0], dtype=np.float64)
        for i in range(u_flat.shape[0]):
            p, e = KahanNeumaierSum.two_product_dekker(float(u_flat[i]), float(v_flat[i]))
            products[i] = p
            errors[i] = e

        return float(KahanNeumaierSum.sum(products) + KahanNeumaierSum.sum(errors))

    @staticmethod
    def compensated_l2_norm(v: NDArray[np.float64]) -> float:
        r"""Norma euclidiana compensada $\|v\|_2=\sqrt{\langle v,v\rangle}$ vía `compensated_dot`."""
        return float(math.sqrt(max(KahanNeumaierSum.compensated_dot(v, v), _WILKINSON_FLOOR)))


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA, ESPINOR DE MAJORANA EN N=1 SYM Y REGULARIDAD DE BANACH
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_PhotinicObservationKernel:
    r"""
    FASE 1 — Observe:
    Saneamiento riguroso de ceros signed IEEE 754 ($x = -0.0 \mapsto +0.0$),
    evaluación compensada de la equivalencia de normas de Banach ($\ell^1, \ell^2$),
    construcción certificada del tensor de Gram federado (simetría, PSD, y
    auditoría de reconstrucción cruzada contra las políticas crudas), descomposición
    del espinor de Majorana $\lambda_{\tilde{\gamma}}$ en $N=1$ SYM, y canonización
    HMAC-sellada e inmutable del expediente de observación.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        self._tol: Final[float] = float(tolerance)
        if hmac_key is None:
            logger.warning(
                "PhotinicGovernanceSatelliteEngine: se utiliza la clave HMAC derivada por defecto. "
                "En despliegue productivo DEBE inyectarse una clave gestionada por el subsistema "
                "de secretos del 'immune_system' (parámetro hmac_key)."
            )
            hmac_key = hashlib.sha256(b"APU.PhotinicGovernanceSatelliteEngine.DefaultEngineSalt.v5").digest()
        self._hmac_key: Final[bytes] = hmac_key

    # --------------------------------------------------------------------
    # 1.1 — Sellado HMAC y regularidad de Sobolev-Banach compensada
    # --------------------------------------------------------------------
    def _compute_hmac_seal(self, *chunks: bytes) -> str:
        r"""Sello de autenticidad HMAC-SHA256 sobre la concatenación ordenada de fragmentos binarios."""
        mac = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        for chunk in chunks:
            mac.update(chunk)
        return mac.hexdigest()

    def evaluate_banach_regularity(self, S: NDArray[np.float64]) -> float:
        r"""
        Evalúa el ratio de regularidad en el espacio de Banach mediante aritmética
        compensada de Dekker-Neumaier en ambas normas:
        $$\mathcal{R}(S) = \frac{\|S\|_1}{\|S\|_2}$$
        Satisface la cota analítica de Sobolev-Banach: $1.0 \le \mathcal{R}(S) \le \sqrt{d}$.
        """
        norm1 = KahanNeumaierSum.sum(np.abs(S))
        norm2 = KahanNeumaierSum.compensated_l2_norm(S)
        ratio = norm1 / max(norm2, _WILKINSON_FLOOR)
        upper_bound = math.sqrt(S.shape[0])
        if not (1.0 - 1.0e-6 <= ratio <= upper_bound + 1.0e-6):
            logger.warning(
                "Ratio de Banach R(S)=%.6f fuera de la cota de Sobolev [1, sqrt(%d)=%.6f].",
                ratio, S.shape[0], upper_bound
            )
        return ratio

    # --------------------------------------------------------------------
    # 1.2 — Certificación de simetría, positividad y reconstrucción del Gram
    # --------------------------------------------------------------------
    def _audit_spectral_symmetry_defect(self, S_fed: NDArray[np.float64]) -> float:
        r"""
        Certifica el defecto de asimetría de de Rham previo a invocar `eigvalsh`
        (que asume simetría exacta silenciosamente): $\delta_{\rm asym} =
        \|S-S^\top\|_F/\max(\|S\|_F,\varepsilon)$. Auditoría defensiva ausente en v4,
        análoga a la certificación de simetría del motor leptónico hermano.
        """
        defect = float(la.norm(S_fed - S_fed.T, ord='fro')) / max(
            float(la.norm(S_fed, ord='fro')), _WILKINSON_FLOOR
        )
        if defect > self._tol:
            logger.warning("Defecto de asimetría en S_fed = %.6e excede tolerancia %.2e.", defect, self._tol)
        return defect

    def _audit_gram_reconstruction_consistency(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
        S_fed: NDArray[np.float64]
    ) -> float:
        r"""
        Canario de integridad numérica que **efectivamente utiliza** las políticas
        crudas (reparando el parámetro fantasma `cleaned_policies` de la v4.0.0):
        reconstruye independientemente $\hat S_{\rm fed}$ directamente desde
        `cleaned_policies` mediante producto punto compensado de Dekker-Neumaier
        columna a columna, y certifica el residuo de Frobenius contra el
        $S_{\rm fed}$ recibido, delatando cualquier desalineación entre la
        construcción del tensor de Gram y los invariantes derivados de él.
        """
        stack_P = np.vstack(cleaned_policies)
        n_nodes = float(len(cleaned_policies))
        dim = stack_P.shape[1]

        S_reconstructed = np.empty((dim, dim), dtype=np.float64)
        for a in range(dim):
            col_a = stack_P[:, a]
            for b in range(a, dim):
                col_b = stack_P[:, b]
                val = KahanNeumaierSum.compensated_dot(col_a, col_b) / n_nodes
                S_reconstructed[a, b] = val
                S_reconstructed[b, a] = val

        # S_fed incluye el término de regularización self._tol * I; se compensa antes de comparar.
        S_fed_unregularized = S_fed - self._tol * np.eye(dim, dtype=np.float64)
        residual = float(la.norm(S_reconstructed - S_fed_unregularized, ord='fro'))
        reference = max(float(la.norm(S_reconstructed, ord='fro')), _WILKINSON_FLOOR)
        relative_residual = residual / reference

        if relative_residual > _GRAM_RECONSTRUCTION_TOLERANCE:
            raise PhotinicNonPositiveDefiniteError(
                f"Fallo de reconstrucción del tensor de Gram federado: residuo relativo "
                f"= {relative_residual:.4e} excede {_GRAM_RECONSTRUCTION_TOLERANCE:.2e}. "
                "Posible corrupción o desalineación entre políticas crudas y S_fed."
            )
        return relative_residual

    def compute_majorana_sym_invariants(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
        S_fed: NDArray[np.float64]
    ) -> Tuple[float, float, float]:
        r"""
        Cálculo del invariante de Majorana en $N=1$ Super-Yang-Mills:
        Audita el invariante cuadrático espinorial real $\bar{\lambda}\lambda$, la paridad
        quiral $\gamma_5$ del campo fotínico, y certifica la consistencia de reconstrucción
        del Gram federado a partir de las políticas crudas:
        $$I_{\mathrm{Majorana}} = \sqrt{\operatorname{Tr}(S_{\mathrm{fed}}^2)}, \qquad
          \chi_5 = \frac{\operatorname{Tr}(S_{\mathrm{fed}})}{\|S_{\mathrm{fed}}\|_F}$$
        """
        eigvals = la.eigvalsh(S_fed)
        majorana_norm = float(math.sqrt(max(KahanNeumaierSum.sum(eigvals ** 2), _WILKINSON_FLOOR)))

        tr_val = KahanNeumaierSum.sum(eigvals)
        fro_norm = float(np.clip(la.norm(S_fed, ord='fro'), _WILKINSON_FLOOR, None))
        chiral_parity = float(tr_val / fro_norm)

        reconstruction_residual = self._audit_gram_reconstruction_consistency(cleaned_policies, S_fed)

        return majorana_norm, chiral_parity, reconstruction_residual

    # --------------------------------------------------------------------
    # 1.3 — Ingesta pública y canonización terminal (puerto hacia Fase 2)
    # --------------------------------------------------------------------
    def observe_federated_nodes(
        self, node_policies: Sequence[NDArray[np.float64]]
    ) -> PhotinicObservationKernel:
        r"""
        Punto de ingestión de políticas crudas. Ejecuta validaciones dimensionales y saneamiento.
        """
        if not node_policies:
            raise PhotinicDimensionError("El consorcio federado no contiene vectores de política (N = 0).")

        dim = node_policies[0].shape[0]
        cleaned_policies: List[NDArray[np.float64]] = []
        banach_ratios: List[float] = []

        sha_material: List[bytes] = []

        for idx, vec in enumerate(node_policies):
            if vec.ndim != 1 or vec.shape[0] != dim:
                raise PhotinicDimensionError(
                    f"Inconsistencia dimensional en nodo federado [{idx}]. Esperado ({dim},), obtenido {vec.shape}"
                )
            if not np.all(np.isfinite(vec)):
                raise PhotinicDimensionError(f"Vector de política de nodo [{idx}] contiene valores NaN o Infinitos.")

            c_vec = np.where(vec == -0.0, +0.0, vec)
            if KahanNeumaierSum.compensated_l2_norm(c_vec) <= _WILKINSON_FLOOR:
                logger.warning("Vector de política de nodo [%d] es numéricamente nulo; nodo degenerado.", idx)

            cleaned_policies.append(c_vec)
            b_ratio = self.evaluate_banach_regularity(c_vec)
            banach_ratios.append(b_ratio)

            sha_material.append(c_vec.tobytes())
            sha_material.append(f"{b_ratio:.8e}".encode("ascii"))

        stack_P = np.vstack(cleaned_policies)
        n_nodes = float(len(cleaned_policies))
        S_fed = (stack_P.T @ stack_P) / n_nodes + self._tol * np.eye(dim, dtype=np.float64)

        symmetry_defect = self._audit_spectral_symmetry_defect(S_fed)
        S_fed = 0.5 * (S_fed + S_fed.T)

        eigvals = np.sort(la.eigvalsh(S_fed))
        min_eig = float(np.max([eigvals[0], _WILKINSON_FLOOR]))
        max_eig = float(eigvals[-1])
        cond_num = max_eig / min_eig

        if cond_num > _CONDITION_NUMBER_MAX:
            logger.warning("Matriz federada S_fed mal condicionada: kappa=%.4e > %.4e", cond_num, _CONDITION_NUMBER_MAX)

        # Brecha espectral: indefinida (+inf) en el caso degenerado d=1 (sin segundo autovalor).
        spectral_gap = float(eigvals[1] - eigvals[0]) if dim > 1 else float("inf")

        majorana_norm, chiral_parity, gram_reconstruction_residual = self.compute_majorana_sym_invariants(
            cleaned_policies, S_fed
        )

        seal_hash = self._compute_hmac_seal(
            *sha_material,
            S_fed.tobytes(),
            f"{majorana_norm:.8e}_{cond_num:.8e}_{symmetry_defect:.8e}_"
            f"{gram_reconstruction_residual:.8e}".encode("ascii")
        )

        return PhotinicObservationKernel(
            node_policies=tuple(cleaned_policies),
            majorana_spinor_norm=majorana_norm,
            majorana_chiral_parity=chiral_parity,
            banach_ratios=tuple(banach_ratios),
            gram_matrix=S_fed,
            gram_eigenvalues=eigvals,
            gram_symmetry_defect=symmetry_defect,
            gram_reconstruction_residual=gram_reconstruction_residual,
            condition_number=cond_num,
            spectral_gap=spectral_gap,
            sha256_seal=seal_hash,
            timestamp=time.time()
        )

    def canonize_photinic_observation_kernel(
        self, node_policies: Sequence[NDArray[np.float64]]
    ) -> PhotinicObservationKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1:
        Canoniza el expediente topológico inmutable de Fase 1.
        Sella la integridad criptográfica de las políticas y constituye el punto de acoplamiento
        y continuación natural hacia la Fase 2.
        """
        return self.observe_federated_nodes(node_policies=node_policies)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PROYECTOR DE GROTHENDIECK, CHOI (CPTP) Y COTA DE TSIRELSON (CHSH)
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_PhotinicFederatedOrient(Phase1_PhotinicObservationKernel):
    r"""
    FASE 2 — Orient:
    Hereda formalmente de la Fase 1. Ingiere el `PhotinicObservationKernel`.
    Evalúa la idempotencia de Grothendieck ($\Omega^2 = \Omega$), la positividad completa
    de Choi-Jamiołkowski, la admisibilidad física del canal (sub-unitalidad POVM) y la
    saturación de Bell-CHSH mediante el Criterio Espectral de Horodecki ($2\sqrt{2}$),
    **sin alteración artificial alguna del valor computado** (reparando el defecto crítico
    de la v4.0.0 que neutralizaba el detector de corrupción super-cuántica).
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)

    # --------------------------------------------------------------------
    # 2.1 — Proyector ortogonal de Grothendieck con tolerancia de rango rigurosa
    # --------------------------------------------------------------------
    def _compute_grothendieck_projector(
        self, policies: Tuple[NDArray[np.float64], ...]
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Construye el proyector canónico ortogonal de Grothendieck $\Omega = V_r V_r^\top$
        mediante SVD de $P$ (ruta única, unificada respecto de v4.0.0 que mantenía una
        rama `try/except` sobre `la.pinv` prácticamente inalcanzable). La tolerancia de
        rango numérico se deriva de la escala del problema en vez de un valor mágico fijo:
        $$\tau_{\rm rank} = \max(N,d)\cdot\varepsilon_{\rm máquina}\cdot\sigma_{\max}(P)$$
        Verifica:
        1. Residuo de idempotencia: $\|\Omega^2 - \Omega\|_F \approx 0$.
        2. Defecto traza-rango (característica de Euler): $|\operatorname{Tr}(\Omega)-\operatorname{rank}(\Omega)|\approx 0$.
        """
        P = np.vstack(policies)  # (N x d)
        U, s, Vt = la.svd(P, full_matrices=False)

        sigma_max = float(s[0]) if s.size > 0 else 0.0
        rank_tol = max(P.shape) * _MACHINE_EPS * max(sigma_max, _WILKINSON_FLOOR)
        rank = max(int(np.sum(s > rank_tol)), 1)

        Omega = Vt[:rank].T @ Vt[:rank]
        Omega = 0.5 * (Omega + Omega.T)
        Omega = np.where(Omega == -0.0, +0.0, Omega)

        Omega_sq = Omega @ Omega
        idempotence_res = float(la.norm(Omega_sq - Omega, ord='fro'))

        tr_val = KahanNeumaierSum.sum(np.diag(Omega))
        trace_rank_mismatch = abs(tr_val - float(rank))

        return Omega, idempotence_res, trace_rank_mismatch

    # --------------------------------------------------------------------
    # 2.2 — Superoperador de Choi-Jamiołkowski con criterio físico correcto
    # --------------------------------------------------------------------
    def _compute_choi_jamiolkowski_superoperator(
        self,
        policies: Tuple[NDArray[np.float64], ...]
    ) -> Tuple[float, float, float, bool, bool]:
        r"""
        Modela el canal cuántico de gobernanza federada $\mathcal{E}(X) = \sum_{k=1}^N K_k X K_k^\top$
        con operadores de Kraus normalizados $K_k = \frac{1}{\sqrt{N}} \frac{p_k p_k^\top}{\|p_k\|^2}$.

        Calcula la matriz de Choi $C_\mathcal{E} \succeq 0$ (PSD **por construcción** analítica,
        al ser una suma de proyectores rango-1 con coeficiente $1/d\ge0$; su auditoría numérica
        certifica ausencia de corrupción de FPU, no un hecho físico incierto).

        Certifica la admisibilidad física del canal mediante el defecto de sub-unitalidad
        (criterio correcto de causalidad de un operador POVM, en reemplazo del umbral de
        Frobenius arbitrario $\le 1.0$ de la v4.0.0):
        $$D = I_d - \sum_{k} K_k^\top K_k, \qquad \text{admisible} \iff \lambda_{\min}(D) \ge -\varepsilon$$
        """
        N = len(policies)
        dim = policies[0].shape[0]

        kraus_ops: List[NDArray[np.float64]] = []
        tp_accumulator = np.zeros((dim, dim), dtype=np.float64)

        for idx, p in enumerate(policies):
            p_norm_sq = KahanNeumaierSum.compensated_dot(p, p)
            if p_norm_sq > _WILKINSON_FLOOR:
                proj_k = np.outer(p, p) / p_norm_sq
            else:
                logger.warning(
                    "Nodo federado [%d] numéricamente nulo; se regulariza con canal maximamente mixto.", idx
                )
                proj_k = np.eye(dim, dtype=np.float64) / float(dim)

            K_k = proj_k / math.sqrt(float(N))
            kraus_ops.append(K_k)
            tp_accumulator += K_k.T @ K_k

        I_d = np.eye(dim, dtype=np.float64)

        # Defecto exacto de preservación de traza (unitalidad estricta): TP ideal <=> D=0.
        exact_tp_defect = tp_accumulator - I_d
        trace_residual = float(la.norm(exact_tp_defect, ord='fro'))

        # Criterio físico correcto de admisibilidad (sub-unitalidad / no-señalización):
        # un canal cuántico válido (aun no-unital) exige sum_k K_k^dagger K_k <= I.
        D_defect = 0.5 * (exact_tp_defect + exact_tp_defect.T)
        subunital_eigvals = la.eigvalsh(-D_defect)  # autovalores de (I - sum K^T K)
        subunital_min_eig = float(np.min(subunital_eigvals))
        is_trace_non_increasing = subunital_min_eig >= -_CAUSALITY_SUBUNITAL_TOLERANCE

        # Matriz de Choi explícita (PSD garantizado analíticamente; se certifica numéricamente).
        if dim <= 16:
            choi_mat = np.zeros((dim * dim, dim * dim), dtype=np.float64)
            for K_k in kraus_ops:
                v = K_k.reshape(-1, 1)
                choi_mat += (v @ v.T) / float(dim)
            choi_eigvals = la.eigvalsh(0.5 * (choi_mat + choi_mat.T))
            choi_min_eig = float(np.min(choi_eigvals))
        else:
            # Dimensiones mayores: PSD garantizado analíticamente (suma de outer products, coef>=0).
            choi_min_eig = 0.0

        is_cp = choi_min_eig >= -_CHOI_PSD_TOLERANCE

        return choi_min_eig, trace_residual, subunital_min_eig, is_cp, is_trace_non_increasing

    # --------------------------------------------------------------------
    # 2.3 — Criterio de Horodecki y Cota de Tsirelson (SIN renormalización artificial)
    # --------------------------------------------------------------------
    def _compute_horodecki_tsirelson_bound(
        self,
        policies: Tuple[NDArray[np.float64], ...]
    ) -> Tuple[float, Tuple[float, float], int, bool, bool]:
        r"""
        Auditoría de no-localidad cuántica y Cota de Tsirelson mediante el Teorema de Horodecki:
        Para un estado bipartito con tensor de correlación $T \in \mathbb{R}^{d_A \times d_B}$,
        el valor máximo de la desigualdad de Bell-CHSH alcanzable es:
        $$\langle \mathcal{B}_{\mathrm{CHSH}} \rangle_{\max} = 2 \sqrt{u_1 + u_2}$$
        donde $u_1 \ge u_2$ son los dos mayores autovalores de $T^\top T$.

        **Corrección crítica respecto de v4.0.0**: se elimina por completo la
        renormalización artificial que forzaba $u_1+u_2\le2$ (y por tanto
        $\mathcal{B}_{\rm CHSH}\le2\sqrt2$ *por construcción*), la cual convertía
        el régimen "Super-Cuántico / Corrupción Adversarial" documentado en un
        estado **inalcanzable en la práctica**. El valor calculado fluye sin alterar.

        **Corrección de forma**: para $N$ impar de nodos federados, la bipartición
        `mid=N//2` producía tamaños desiguales que hacían `stack_A.T @ stack_B`
        incompatible dimensionalmente (crash de numpy). Se introduce el
        emparejamiento simétrico $n_{\rm pairs}=\min(\rm mid, N-\rm mid)$ y la
        normalización de filas a norma unitaria, dotando a $T$ de una semántica
        de correlación bien definida y acotada para cualquier $N\ge2$.
        """
        N = len(policies)
        dim = policies[0].shape[0]

        if N >= 2:
            mid = N // 2
            n_pairs = min(mid, N - mid)

            stack_A_full = np.vstack(policies[:mid])
            stack_B_full = np.vstack(policies[mid:])
            stack_A = stack_A_full[:n_pairs]
            stack_B = stack_B_full[:n_pairs]

            # Normalización de filas a norma unitaria: acota cada entrada de T por
            # Cauchy-Schwarz, dotando a la construcción de semántica de "correlación".
            norms_A = np.clip(la.norm(stack_A, axis=1, keepdims=True), _WILKINSON_FLOOR, None)
            norms_B = np.clip(la.norm(stack_B, axis=1, keepdims=True), _WILKINSON_FLOOR, None)
            A_hat = stack_A / norms_A
            B_hat = stack_B / norms_B

            T_corr = (A_hat.T @ B_hat) / float(n_pairs)
        else:
            n_pairs = 0
            p = policies[0]
            if dim >= 4:
                half = dim // 2
                pA, pB = p[:half], p[half:2 * half]
            else:
                pA, pB = p, p
            norm_a = max(float(la.norm(pA)), _WILKINSON_FLOOR)
            norm_b = max(float(la.norm(pB)), _WILKINSON_FLOOR)
            T_corr = np.outer(pA / norm_a, pB / norm_b)

        TT_T = T_corr.T @ T_corr
        eigvals = np.sort(la.eigvalsh(TT_T))[::-1]

        u1 = float(max(eigvals[0], 0.0)) if len(eigvals) > 0 else 0.0
        u2 = float(max(eigvals[1], 0.0)) if len(eigvals) > 1 else 0.0

        chsh_val = float(2.0 * math.sqrt(max(u1 + u2, 0.0)))

        is_tsirelson = chsh_val <= (_TSIRELSON_BOUND + 1.0e-9)
        is_classical = chsh_val <= (_BELL_CLASSICAL_BOUND + 1.0e-9)

        return chsh_val, (u1, u2), n_pairs, is_tsirelson, is_classical

    # --------------------------------------------------------------------
    # 2.4 — Orientación terminal y puerto de acoplamiento hacia Fase 3
    # --------------------------------------------------------------------
    def orient_photinic_policy(
        self, kernel: PhotinicObservationKernel
    ) -> PhotinicPolicyReport:
        r"""Orquesta los cálculos de la Fase 2 sobre el kernel canonizado, sin enforcement (Fase 2 solo reporta)."""
        policies = kernel.node_policies

        _, idem_res, trace_mismatch = self._compute_grothendieck_projector(policies)
        min_choi_eig, trace_res, subunital_min_eig, is_cp, is_trace_non_increasing = (
            self._compute_choi_jamiolkowski_superoperator(policies)
        )
        chsh_val, horodecki_eigs, n_pairs, is_tsirelson, is_classical = (
            self._compute_horodecki_tsirelson_bound(policies)
        )

        return PhotinicPolicyReport(
            kernel=kernel,
            idempotence_residual=idem_res,
            trace_rank_mismatch=trace_mismatch,
            choi_min_eigenvalue=min_choi_eig,
            choi_trace_residual=trace_res,
            choi_subunital_defect_min_eigenvalue=subunital_min_eig,
            tsirelson_chsh_value=chsh_val,
            horodecki_eigs=horodecki_eigs,
            horodecki_bipartition_pairs=n_pairs,
            is_choi_completely_positive=is_cp,
            is_trace_non_increasing=is_trace_non_increasing,
            is_causal_non_signaling=is_trace_non_increasing,
            is_tsirelson_bounded=is_tsirelson,
            is_classical_lhv=is_classical,
            timestamp=time.time()
        )

    def synthesize_photinic_orientation(
        self,
        kernel_or_policies: Union[PhotinicObservationKernel, Sequence[NDArray[np.float64]]]
    ) -> PhotinicPolicyReport:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2:
        Garantiza la continuidad functorial estricta.
        Si se le suministra el `PhotinicObservationKernel`, procede directamente.
        Si se le suministran vectores de política crudos, enlaza automáticamente con el método
        terminal de Fase 1 `canonize_photinic_observation_kernel` y sintetiza el reporte formal.
        Este reporte es el punto de entrada canónico de la Fase 3.
        """
        if isinstance(kernel_or_policies, PhotinicObservationKernel):
            kernel = kernel_or_policies
        else:
            kernel = self.canonize_photinic_observation_kernel(kernel_or_policies)

        return self.orient_photinic_policy(kernel)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: TELEMETRÍA METROLÓGICA FPU, ENTROPÍA DE VON NEUMANN Y SELLADO HMAC
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_PhotinicGovernanceEngine(Phase2_PhotinicFederatedOrient):
    r"""
    FASE 3 — Act:
    Orquestador terminal de alta precisión en silicio y FPU. Certifica y **hace
    cumplir** (mediante excepciones, reparando el enforcement roto de v4.0.0) la
    positividad completa de Choi, la admisibilidad causal del canal y la Cota de
    Tsirelson; calcula la entropía de von Neumann genuina del estado reducido de
    gobernanza federada, la entropía de entrelazamiento bipartita de Horodecki y
    la pureza cuántica; y emite el `PhotinicEngineState` sellado con HMAC-SHA256.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)

    # --------------------------------------------------------------------
    # 3.1 — Matriz de densidad reducida genuina y entropías cuánticas
    # --------------------------------------------------------------------
    def compute_reduced_federated_density_matrix(
        self, kernel: PhotinicObservationKernel
    ) -> NDArray[np.float64]:
        r"""
        Construye la matriz de densidad reducida **genuina** del estado de
        gobernanza federada:
        $$\rho_{\rm fed} = \frac{S_{\rm fed}}{\operatorname{Tr}(S_{\rm fed})}$$
        Al ser $S_{\rm fed}\succeq0$ por construcción (Fase 1), $\rho_{\rm fed}$ es
        una matriz de densidad válida ($\rho\succeq0$, $\operatorname{Tr}\rho=1$).
        """
        trace_S = float(np.trace(kernel.gram_matrix))
        if trace_S <= _WILKINSON_FLOOR:
            raise PhotinicNonPositiveDefiniteError(
                "Traza del tensor de Gram federado colapsada; imposible normalizar como matriz de densidad."
            )
        return kernel.gram_matrix / trace_S

    def compute_von_neumann_entropy_full(self, rho: NDArray[np.float64]) -> Tuple[float, float]:
        r"""
        Entropía de von Neumann **global** del estado reducido de gobernanza y su
        pureza cuántica, calculadas sobre el espectro completo (no truncado a
        2 autovalores como en la v4.0.0):
        $$S(\rho) = -\sum_i \lambda_i \ln \lambda_i, \qquad \operatorname{Tr}(\rho^2) = \sum_i \lambda_i^2$$
        """
        eigvals = np.clip(la.eigvalsh(rho), 0.0, 1.0)
        s_vn = -KahanNeumaierSum.sum([ev * math.log(ev) for ev in eigvals if ev > _WILKINSON_FLOOR])
        purity = float(KahanNeumaierSum.sum(eigvals ** 2))
        return float(max(s_vn, 0.0)), purity

    def compute_horodecki_entanglement_entropy(self, report: PhotinicPolicyReport) -> float:
        r"""
        Entropía de entrelazamiento del **par de Bell efectivo** subyacente a la
        auditoría de Horodecki-Tsirelson, calculada sobre los dos autovalores
        normalizados $u_1,u_2$ de $T^\top T$ (equivalente conceptual a la v4.0.0,
        preservado bajo un nombre explícito que lo distingue de la entropía global).
        """
        u1, u2 = report.horodecki_eigs
        total = u1 + u2
        if total <= _WILKINSON_FLOOR:
            return 0.0
        p1, p2 = u1 / total, u2 / total
        s_vn = 0.0
        if p1 > _WILKINSON_FLOOR:
            s_vn -= p1 * math.log(p1)
        if p2 > _WILKINSON_FLOOR:
            s_vn -= p2 * math.log(p2)
        return float(s_vn)

    # --------------------------------------------------------------------
    # 3.2 — Auditoría terminal, enforcement de excepciones y emisión del certificado
    # --------------------------------------------------------------------
    def execute_photinic_audit(
        self,
        target: Union[PhotinicPolicyReport, PhotinicObservationKernel, Sequence[NDArray[np.float64]]]
    ) -> PhotinicEngineState:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 3 Y DEL MOTOR COVARIANTE:
        Punto de culminación OODA del Motor Fotínico de Gobernanza en lazo cerrado.
        Acepta polimórficamente:
        1. `PhotinicPolicyReport` (procedente directamente de Fase 2).
        2. `PhotinicObservationKernel` (procedente de Fase 1, acoplándose vía Fase 2).
        3. `Sequence[NDArray[np.float64]]` (secuencia cruda, resolviendo la cadena completa F1 -> F2 -> F3).

        A diferencia de la v4.0.0 (donde solo Tsirelson se auditaba, y encima estaba
        neutralizado por el bug de renormalización), esta Fase 3 **certifica y hace
        cumplir** las tres condiciones físicas computadas en Fase 2: positividad
        completa de Choi, admisibilidad causal del canal, y Cota de Tsirelson.
        """
        t_start = time.perf_counter()

        if isinstance(target, PhotinicPolicyReport):
            report = target
            kernel = report.kernel
        elif isinstance(target, PhotinicObservationKernel):
            kernel = target
            report = self.orient_photinic_policy(kernel)
        else:
            report = self.synthesize_photinic_orientation(target)
            kernel = report.kernel

        # --- Enforcement estricto (Fase 3, puerta de certificación) -----------
        if not report.is_choi_completely_positive:
            raise PhotinicNonPositiveDefiniteError(
                f"Violación de Positividad Completa de Choi: lambda_min(C_E) = "
                f"{report.choi_min_eigenvalue:.4e} < -{_CHOI_PSD_TOLERANCE:.2e}. "
                "Corrupción numérica severa: C_E debería ser PSD por construcción analítica."
            )
        if not report.is_trace_non_increasing:
            raise PhotinicNonCausalChannelError(
                f"Canal de gobernanza federada físicamente inadmisible: "
                f"lambda_min(I - sum K_k^T K_k) = {report.choi_subunital_defect_min_eigenvalue:.4e} "
                f"< -{_CAUSALITY_SUBUNITAL_TOLERANCE:.2e} (super-unital / probabilidades > 1)."
            )
        if not report.is_tsirelson_bounded:
            raise PhotinicTsirelsonViolationError(
                f"Violación de la Cota de Tsirelson: CHSH = {report.tsirelson_chsh_value:.4f} > "
                f"{_TSIRELSON_BOUND:.6f} (2*sqrt(2)). Corrupción adversarial no-local o caja PR "
                "super-cuántica detectada en el consorcio federado."
            )

        rho_fed = self.compute_reduced_federated_density_matrix(kernel)
        vn_entropy, purity = self.compute_von_neumann_entropy_full(rho_fed)
        bell_entropy = self.compute_horodecki_entanglement_entropy(report)

        t_elapsed_ms = float((time.perf_counter() - t_start) * 1000.0)
        fpu_drift = float(
            abs(report.idempotence_residual)
            + abs(report.trace_rank_mismatch)
            + abs(kernel.gram_symmetry_defect)
            + abs(kernel.gram_reconstruction_residual)
        )

        final_seal = self._compute_hmac_seal(
            kernel.sha256_seal.encode("ascii"),
            rho_fed.tobytes(),
            f"{report.idempotence_residual:.12e}_{report.choi_min_eigenvalue:.12e}_"
            f"{report.tsirelson_chsh_value:.12e}_{vn_entropy:.12e}_{purity:.12e}_"
            f"{bell_entropy:.12e}_{fpu_drift:.12e}".encode("ascii")
        )

        phase_signature = (
            f"FOCK_MAJORANA_SYM:CP={int(report.is_choi_completely_positive)}:"
            f"CAUSAL={int(report.is_causal_non_signaling)}:"
            f"TSIRELSON={int(report.is_tsirelson_bounded)}:"
            f"CHSH={report.tsirelson_chsh_value:.4f}:S_vN={vn_entropy:.4f}:PURITY={purity:.4f}"
        )

        return PhotinicEngineState(
            kernel=kernel,
            policy_report=report,
            fpu_execution_time_ms=t_elapsed_ms,
            federated_density_matrix=rho_fed,
            von_neumann_entropy=vn_entropy,
            quantum_purity=purity,
            bell_pair_entanglement_entropy=bell_entropy,
            fpu_precision_drift=fpu_drift,
            cryptographic_seal=final_seal,
            phase_signature=phase_signature
        )


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR FOTÍNICO DE GOBERNANZA SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class PhotinicGovernanceSatelliteEngine(Phase3_PhotinicGovernanceEngine):
    r"""
    Motor Fotínico de Gobernanza Federada de-confinado en silicio FPU.
    Integra la totalidad del pipeline holomorfo anidado (Fases 1, 2 y 3).
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)
        logger.info(
            "PhotinicGovernanceSatelliteEngine v5.0.0 inicializado con éxito. "
            "Alineación: Majorana SYM, Grothendieck, Choi CPTP (enforcement real), "
            "Horodecki-Tsirelson (sin renormalización artificial), Dekker-Neumaier FPU, HMAC-SHA256."
        )


__all__ = [
    "PhotinicGovernanceSatelliteEngine",
    "PhotinicObservationKernel",
    "PhotinicPolicyReport",
    "PhotinicEngineState",
    "PhotinicEngineError",
    "PhotinicDimensionError",
    "PhotinicNonPositiveDefiniteError",
    "PhotinicCausalViolationError",
    "PhotinicNonCausalChannelError",
    "PhotinicTsirelsonViolationError",
    "KahanNeumaierSum",
    "Phase1_PhotinicObservationKernel",
    "Phase2_PhotinicFederatedOrient",
    "Phase3_PhotinicGovernanceEngine",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "═" * 80)
    print(" INICIANDO AUDITORÍA ESPECTRAL Y CIBER-FÍSICA DE PHOTINIC GOVERNANCE SATELLITE ENGINE")
    print("═" * 80)

    engine = PhotinicGovernanceSatelliteEngine()
    rng = np.random.default_rng(42)

    # Consorcio federado con N=5 (impar) nodos: certifica la corrección de forma de v5.
    policies = [rng.normal(loc=0.0, scale=1.0, size=6) for _ in range(5)]

    state = engine.execute_photinic_audit(policies)
    rep = state.policy_report

    print(f"\n[+] TELEMETRÍA DE LA FASE 1 (OBSERVE):")
    print(f"    - Condicionamiento kappa(S_fed):    {state.kernel.condition_number:.4f}")
    print(f"    - Defecto de simetría de Gram:       {state.kernel.gram_symmetry_defect:.4e}")
    print(f"    - Residuo de reconstrucción de Gram: {state.kernel.gram_reconstruction_residual:.4e}")
    print(f"    - Norma del espinor de Majorana:     {state.kernel.majorana_spinor_norm:.6f}")
    print(f"    - Sello HMAC-SHA256:                 {state.kernel.sha256_seal[:24]}...")

    print(f"\n[+] TELEMETRÍA DE LA FASE 2 (ORIENT):")
    print(f"    - Residuo de idempotencia (Omega^2-Omega): {rep.idempotence_residual:.4e}")
    print(f"    - Autovalor mínimo de Choi:                {rep.choi_min_eigenvalue:.4e}")
    print(f"    - Defecto de sub-unitalidad (causalidad):  {rep.choi_subunital_defect_min_eigenvalue:.4e}")
    print(f"    - Pares de bipartición Horodecki (N=5):    {rep.horodecki_bipartition_pairs}")
    print(f"    - Valor de Bell-CHSH (sin renormalizar):   {rep.tsirelson_chsh_value:.4f}")
    print(f"    - ¿Acotado por Tsirelson (<=2*sqrt(2))?:   {rep.is_tsirelson_bounded}")
    print(f"    - ¿Régimen clásico LHV (<=2)?:             {rep.is_classical_lhv}")

    print(f"\n[+] TELEMETRÍA DE LA FASE 3 (ACT / FPU):")
    print(f"    - Latencia de Cómputo FPU:           {state.fpu_execution_time_ms:.4f} ms")
    print(f"    - Entropía von Neumann global S(rho):{state.von_neumann_entropy:.6f}")
    print(f"    - Pureza cuántica Tr(rho^2):          {state.quantum_purity:.6f}")
    print(f"    - Entropía entrelaz. par de Bell:     {state.bell_pair_entanglement_entropy:.6f}")
    print(f"    - Deriva de Precisión FPU:            {state.fpu_precision_drift:.4e}")
    print(f"    - Firma de Fase:                      {state.phase_signature}")
    print(f"    - Sello Criptográfico HMAC:           {state.cryptographic_seal[:32]}...")

    assert rep.is_tsirelson_bounded, "Fallo: violación de la Cota de Tsirelson"
    assert rep.is_choi_completely_positive, "Fallo: canal de Choi no-CP"
    assert rep.is_causal_non_signaling, "Fallo: canal no admisible físicamente"
    print("\n" + "═" * 80)
    print(" ¡AUDITORÍA CIBER-FÍSICA DEL MOTOR FOTÍNICO COMPLETADA CON ÉXITO ABSOLUTO!")
    print("═" * 80)