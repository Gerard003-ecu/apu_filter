from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Leptonic Flavor Satellite Engine (Motor de Sabor y Oscilación)      ║
║ Ruta   : app/core/immune_system/leptonic_flavor_satellite_engine.py          ║
║ Versión: 5.0.0-Doctoral-PMNS-Majorana-MSW-Jarlskog-Banach-Graph-FPU-HMAC     ║
║ Nivel  : Estrato Omega ($V_\Omega$, Nivel 0.5 — Núcleo Ciber-Físico FPU)    ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y ARQUITECTURA TEÓRICA DOCTORAL (v5):                   ║
║ 1. Geometría de Sabor Leptónico y Mezcla PMNS con Fases de Majorana,         ║
║    con auditoría cruzada analítica/numérica del invariante de Jarlskog       ║
║    (invariancia de gauge frente a fases de Majorana demostrada por cómputo). ║
║ 2. Dinámica MSW exacta corregida: $H_{\rm eff}\cdot L = 2\,k\,(L/E)\,U M U^\dagger ║
║    + V_e L/(\hbar c)$, con $k=1.267\,\mathrm{eV^{-2}\,km^{-1}\,GeV}$,        ║
║    exponenciación de Lie simpléctica y auditoría de unitariedad de $S(L)$.   ║
║ 3. Teoría espectral de grafos de mezcla: Laplaciano de Hodge, conectividad   ║
║    de Fiedler, distribución estacionaria de Perron-Frobenius, entropía de    ║
║    Shannon de mezcla y constante de Kemeny (tiempo medio de recurrencia).    ║
║ 4. Mecánica cuántica rigurosa: la matriz de densidad reducida de sabor       ║
║    $\rho = S_{\rm Gram}/\mathrm{Tr}(S_{\rm Gram})$ es genuinamente PSD y de  ║
║    traza unitaria; se certifica pureza $\mathrm{Tr}(\rho^2)$ y entropía      ║
║    relativa cuántico-clásica (divergencia de Kullback-Leibler).             ║
║ 5. Certificación espectral de Bauer-Fike: para el Hamiltoniano hermítico     ║
║    $H_{\rm eff}$, el número de condición de sus autovalores es exactamente  ║
║    1 (matriz normal), lo cual se verifica computacionalmente.               ║
║ 6. Seguridad criptográfica reforzada: doble sello HMAC-SHA256 (se repara    ║
║    la importación muerta de `hmac` de la v4, coherente con el nombre del    ║
║    subsistema `immune_system`).                                            ║
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

logger = logging.getLogger("APU.Physics.LeptonicFlavorSatelliteEngine")

# ------------------------------------------------------------------------------
# CONSTANTES FÍSICAS UNIVERSALES, LÍMITES METROLÓGICOS DE WILKINSON Y FPU
# ------------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_PMNS_UNITARITY_TOLERANCE: Final[float] = 1.0e-10
_EVOLUTION_UNITARITY_TOLERANCE: Final[float] = 1.0e-9
_STOCHASTIC_DEFECT_TOLERANCE: Final[float] = 1.0e-6
_METRIC_INVERSION_TOLERANCE: Final[float] = 1.0e-8
_MARKOV_ERGODICITY_TOLERANCE: Final[float] = 1.0e-6
_BANACH_SOBOLEV_TOLERANCE: Final[float] = 1.0e-6
_HERMITICITY_CERTIFICATION_TOLERANCE: Final[float] = 1.0e-10

# Constante de conversión natural de oscilación: k = 1/(4 hbar c) en unidades
# tales que la fase de oscilación de dos sabores se escribe
#   phi = 1.267 * Delta m^2[eV^2] * L[km] / E[GeV]
_OSC_PHASE_CONVERSION_FACTOR: Final[float] = 1.266932808

# Inverso de hbar*c expresado en (eV * km)^{-1}, derivado de
#   hbar*c = 197.3269804 MeV*fm = 1.973269804e-10 eV*km
# Se emplea para convertir el potencial de materia V_e [eV] y la línea de base
# L [km] en una fase adimensional: phi_mat = V_e * L * _INV_HBAR_C_EV_KM.
_INV_HBAR_C_EV_KM: Final[float] = 5.06773e9


# ------------------------------------------------------------------------------
# JERARQUÍA DE EXCEPCIONES DOCTORALES
# ------------------------------------------------------------------------------
class LeptonicEngineError(Exception):
    r"""Excepción raíz para violaciones del motor de sabor leptónico."""
    pass


class DimensionMismatchError(LeptonicEngineError):
    r"""Inconsistencia dimensional en tensores de sabor o métricas riemannianas."""
    pass


class MetricIndefinitenessError(LeptonicEngineError):
    r"""Fallo en la positividad estricta de Gram-Riemann ($G \not\succ 0$) o en su inversión numérica."""
    pass


class PMNSUnitarityError(LeptonicEngineError):
    r"""Violación de la unitoridad estricta en el grupo de Lie $U(3)$ ($U U^\dagger \neq I$)."""
    pass


class LeptonicChargeViolationError(LeptonicEngineError):
    r"""Ruptura de la invariancia de corriente de Noether para la carga leptónica total."""
    pass


class EvolutionUnitarityError(LeptonicEngineError):
    r"""El operador de evolución $S(L)=e^{-iH_{\rm eff}L}$ no preserva la unitoridad dentro de tolerancia FPU."""
    pass


class NumericalInstabilityError(LeptonicEngineError):
    r"""Fallo de doble-estocasticidad en la matriz de transición unistocástica $P=|S|^2$."""
    pass


class MarkovErgodicityError(LeptonicEngineError):
    r"""La cadena de Markov de sabor no admite una distribución estacionaria de Perron-Frobenius válida."""
    pass


# ------------------------------------------------------------------------------
# ARITMÉTICA COMPENSADA DE KAHAN-BABUŠKA-NEUMAIER (KBN)
# ------------------------------------------------------------------------------
class KahanNeumaierSum:
    r"""
    Sumador compensado de precisión en FPU basado en el algoritmo de Neumaier.
    Elimina la acumulación de error en punto flotante IEEE 754 preservando
    la precisión hasta la cota inferior analítica de Wilkinson. Se emplea de
    forma transversal en las 3 fases para toda reducción numérica sensible
    (normas $\ell^1$, cargas de Noether, entropías de Shannon/von Neumann,
    constantes de Kemeny).
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
    def compensated_inner_product(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        r"""Producto interno compensado de dos vectores sobre $\mathbb{R}^d$."""
        if u.shape != v.shape:
            raise DimensionMismatchError("Discrepancia dimensional en producto interno.")
        return KahanNeumaierSum.sum(u * v)


# ------------------------------------------------------------------------------
# EXPEDIENTES Y ESTRUCTURAS DE DATOS INMUTABLES
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    r"""
    Caché espectral inmutable del tensor métrico riemanniano $G_{\mu\nu}$.
    Incluye certificados de simetría e inversión numérica ausentes en v4.
    """
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    condition_number: float
    hodge_volume_form: float
    dimension: int
    inversion_residual: float
    asymmetry_defect: float


@dataclass(frozen=True, slots=True)
class LeptonicObservationKernel:
    r"""
    Expediente inmutable de Fase 1 (Observe).
    Encapsula los estados de sabor ($e, \mu, \tau$), normas euclidianas y riemannianas,
    ratios de regularidad en el espacio de Banach $\ell^1 \hookrightarrow \ell^2$,
    el tensor de Gram de sabor con su espectro de positividad certificado, los ángulos
    de entrelazamiento geométrico entre familias leptónicas, y el sello HMAC-SHA256.
    """
    flavor_state_matrix: NDArray[np.float64]
    flavor_norms_euclidean: Tuple[float, float, float]
    flavor_norms_riemannian: Tuple[float, float, float]
    banach_ratios: Tuple[float, float, float]
    gram_matrix: NDArray[np.float64]
    gram_eigenvalues: Tuple[float, float, float]
    flavor_entanglement_angles: Tuple[float, float, float]
    metric_cache: BaseMetricCache
    sha256_seal: str
    observation_timestamp: float


@dataclass(frozen=True, slots=True)
class LeptonicFlavorReport:
    r"""
    Expediente inmutable de Fase 2 (Orient).
    Sintetiza la matriz PMNS con fases de Majorana, el invariante de Jarlskog $\mathcal{J}_{CP}$
    (numérico y analítico, con residuo cruzado de invariancia de gauge), el Hamiltoniano
    efectivo MSW, la matriz de oscilación $P(\nu_\alpha \to \nu_\beta)$, el Laplaciano del
    grafo de sabor, la distribución estacionaria de Perron-Frobenius, la entropía de Shannon
    de mezcla, la constante de Kemeny y la conservación de corriente de Noether.
    """
    kernel: LeptonicObservationKernel
    pmns_matrix: NDArray[np.complex128]
    pmns_unitarity_residual: float
    jarlskog_invariant: float
    jarlskog_invariant_analytic: float
    jarlskog_cross_residual: float
    is_pmns_unitary: bool
    effective_hamiltonian: NDArray[np.complex128]
    vacuum_hamiltonian_eigenvalues: NDArray[np.float64]
    evolution_unitarity_defect: float
    stochastic_normalization_defect: float
    oscillation_probabilities: NDArray[np.float64]
    msw_matter_potential: float
    flavor_graph_laplacian: NDArray[np.float64]
    flavor_graph_spectral_gap: float
    flavor_stationary_distribution: NDArray[np.float64]
    shannon_mixing_entropy: float
    kemeny_constant: float
    total_leptonic_charge: float
    charge_conservation_residual: float
    is_charge_conserved: bool
    orientation_timestamp: float


@dataclass(frozen=True, slots=True)
class LeptonicEngineState:
    r"""
    Certificado formal e inmutable emitido en la Fase 3 (Act).
    Acredita la telemetría cuántica en la FPU: matriz de densidad reducida genuina,
    entropía de von Neumann, pureza cuántica, entropía relativa cuántico-clásica,
    los factores de calidad topológico y resonante, la certificación espectral de
    Bauer-Fike y el doble sello criptográfico HMAC-SHA256.
    """
    kernel: LeptonicObservationKernel
    report: LeptonicFlavorReport
    fpu_execution_time_ms: float
    flavor_density_matrix: NDArray[np.complex128]
    von_neumann_flavor_entropy: float
    quantum_purity: float
    classical_quantum_relative_entropy: float
    equivalent_circuit_q_factor: float
    resonant_quality_factor: float
    spectral_stability_drift: float
    is_bauer_fike_certified: bool
    cryptographic_seal: str
    quantum_flavor_signature: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA, ESPACIO DE BANACH Y METROLOGÍA DE RIEMANN-CHOLESKY
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_LeptonicFlavorObserver:
    r"""
    FASE 1 — Observe:
    Saneamiento de ceros de signo IEEE 754 ($x = -0.0 \mapsto +0.0$),
    auditoría de simetría y factorización de Cholesky de la métrica
    $G_{\mu\nu} = L L^\top \succ 0$ con certificación de inversión numérica,
    evaluación de normas riemannianas $\|v\|_{\ell^2, G} = \sqrt{v^\top G v}$,
    geometría de Gram de sabor con ángulos de entrelazamiento, auditoría de
    inmersión en álgebras de Banach y canonización HMAC-sellada del Kernel.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        self._tol: Final[float] = float(tolerance)
        # Clave HMAC del motor: si no se suministra, se deriva de una sal fija
        # documentada. En producción DEBE inyectarse una clave gestionada por
        # el subsistema de secretos del `immune_system`.
        self._hmac_key: Final[bytes] = (
            hmac_key if hmac_key is not None
            else hashlib.sha256(b"APU.LeptonicFlavorSatelliteEngine.DefaultEngineSalt.v5").digest()
        )

    # --------------------------------------------------------------------
    # 1.1 — Saneamiento numérico IEEE 754 y sellado criptográfico HMAC
    # --------------------------------------------------------------------
    @staticmethod
    def _sanitize_signed_zero(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Normaliza ceros con signo negativo IEEE 754: $-0.0 \mapsto +0.0$."""
        return np.where(tensor == -0.0, 0.0, tensor)

    def _compute_hmac_seal(self, *chunks: bytes) -> str:
        r"""
        Calcula el sello de autenticidad HMAC-SHA256 sobre la concatenación
        ordenada de fragmentos binarios, usando la clave privada del motor.
        Repara la importación muerta de `hmac` presente en versiones previas,
        reforzando la autenticidad de origen exigida por un subsistema inmune.
        """
        mac = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        for chunk in chunks:
            mac.update(chunk)
        return mac.hexdigest()

    # --------------------------------------------------------------------
    # 1.2 — Auditoría de simetría, Cholesky y certificación de inversión de G
    # --------------------------------------------------------------------
    def _audit_metric_symmetry_defect(self, G_metric: NDArray[np.float64]) -> float:
        r"""
        Mide el defecto relativo de asimetría de de Rham antes de simetrizar:
        $$\delta_{\rm asym} = \frac{\|G - G^\top\|_F}{\max(\|G\|_F, \varepsilon)}$$
        Si $\delta_{\rm asym}$ excede la tolerancia del motor, se emite una
        advertencia explícita en lugar de simetrizar silenciosamente (como
        hacía la v4), preservando la trazabilidad forense del insumo.
        """
        asym_norm = float(la.norm(G_metric - G_metric.T, ord='fro'))
        total_norm = float(la.norm(G_metric, ord='fro'))
        defect = asym_norm / max(total_norm, _WILKINSON_FLOOR)
        if defect > self._tol:
            logger.warning(
                "Defecto de asimetría en G_metric = %.6e excede tolerancia %.2e. "
                "Se aplica simetrización de seguridad G <- (G + G^T)/2.",
                defect, self._tol
            )
        return defect

    def _audit_metric_regularity(self, G_metric: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Inspecciona el tensor métrico $G_{\mu\nu}$:
        1. Auditoría de asimetría de de Rham y simetrización de seguridad.
        2. Factorización de Cholesky $G = L L^\top$ para garantizar definitud positiva estricta.
        3. Forma de volumen de Hodge $\operatorname{vol}_G = \sqrt{\det G} = \prod L_{ii}$.
        4. Inversión simpléctica triangular con **certificación de residuo**
           $\|G G^{-1} - I\|_F$ (ausente en v4: una inversión mal condicionada
           podía pasar desapercibida).
        5. Número de condición $\kappa_2(G) = \lambda_{\max}/\lambda_{\min}$.
        """
        if G_metric.ndim != 2 or G_metric.shape[0] != G_metric.shape[1]:
            raise DimensionMismatchError(f"El tensor métrico G debe ser una matriz cuadrada. Forma: {G_metric.shape}")

        dim = G_metric.shape[0]
        asymmetry_defect = self._audit_metric_symmetry_defect(G_metric)
        G_sym = 0.5 * (G_metric + G_metric.T)
        G_sym = self._sanitize_signed_zero(G_sym)

        try:
            L_cholesky = la.cholesky(G_sym, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(
                f"El tensor métrico G no es estrictamente Definido Positivo (SPD): {exc}"
            ) from exc

        diag_L = np.diag(L_cholesky)
        hodge_vol = float(np.prod(diag_L))
        if hodge_vol <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(f"Forma de volumen métrica colapsada: det(G)^1/2 = {hodge_vol:.6e}")

        # Inversión estable triangular + certificación de residuo de inversión
        L_inv = la.solve_triangular(L_cholesky, np.eye(dim, dtype=np.float64), lower=True)
        G_inv = L_inv.T @ L_inv
        inversion_residual = float(la.norm(G_sym @ G_inv - np.eye(dim, dtype=np.float64), ord='fro'))
        if inversion_residual > _METRIC_INVERSION_TOLERANCE:
            raise MetricIndefinitenessError(
                f"Residuo de inversión ||G G^-1 - I||_F = {inversion_residual:.4e} "
                f"excede la tolerancia numérica {_METRIC_INVERSION_TOLERANCE:.2e}."
            )

        eigvals = la.eigvalsh(G_sym)
        min_eig = float(max(np.min(eigvals), _WILKINSON_FLOOR))
        max_eig = float(np.max(eigvals))
        cond_num = max_eig / min_eig

        if cond_num > _CONDITION_NUMBER_MAX:
            logger.warning(
                "¡Advertencia metrológica!: Número de condición kappa_2(G) = %.4e excede umbral de seguridad.",
                cond_num
            )

        return BaseMetricCache(
            g_base=G_sym,
            cholesky_factor=L_cholesky,
            g_inv=G_inv,
            condition_number=cond_num,
            hodge_volume_form=hodge_vol,
            dimension=dim,
            inversion_residual=inversion_residual,
            asymmetry_defect=asymmetry_defect
        )

    # --------------------------------------------------------------------
    # 1.3 — Norma riemanniana, geometría de Gram y ángulos de entrelazamiento
    # --------------------------------------------------------------------
    def _compute_riemannian_norm(self, v: NDArray[np.float64], G_base: NDArray[np.float64]) -> float:
        r"""Norma riemanniana canónica $\|v\|_{\ell^2, G} = \sqrt{v^\top G v}$."""
        quadratic_form = float(v @ G_base @ v)
        return float(math.sqrt(max(quadratic_form, _WILKINSON_FLOOR)))

    def _compute_flavor_gram_geometry(
        self,
        flavor_matrix: NDArray[np.float64],
        G_base: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Tuple[float, float, float], Tuple[float, float, float]]:
        r"""
        Construye el tensor de Gram de sabor $S_{\alpha\beta} = \langle \psi_\alpha, \psi_\beta \rangle_G$
        y certifica su semi-definitud positiva (Criterio de Sylvester, ausente en v4).
        Adicionalmente calcula los **ángulos de entrelazamiento geométrico** entre pares
        de familias leptónicas:
        $$\theta_{\alpha\beta} = \arccos\left(\frac{S_{\alpha\beta}}{\sqrt{S_{\alpha\alpha} S_{\beta\beta}}}\right)$$
        que cuantifican la no-ortogonalidad riemanniana entre los estados de sabor crudos.
        """
        gram = flavor_matrix @ G_base @ flavor_matrix.T
        gram = self._sanitize_signed_zero(gram)
        gram_sym = 0.5 * (gram + gram.T)

        eigvals_gram = np.sort(la.eigvalsh(gram_sym))
        if float(np.min(eigvals_gram)) < -1.0e-9:
            raise MetricIndefinitenessError(
                f"El tensor de Gram de sabor no es PSD: lambda_min = {float(np.min(eigvals_gram)):.4e}"
            )

        diag_gram = np.diag(gram_sym)
        pairs = ((0, 1), (0, 2), (1, 2))
        angles: List[float] = []
        for a, b in pairs:
            denom = math.sqrt(max(diag_gram[a] * diag_gram[b], _WILKINSON_FLOOR))
            cos_theta = float(np.clip(gram_sym[a, b] / denom, -1.0, 1.0))
            angles.append(math.acos(cos_theta))

        return gram_sym, (float(eigvals_gram[0]), float(eigvals_gram[1]), float(eigvals_gram[2])), tuple(angles)

    # --------------------------------------------------------------------
    # 1.4 — Regularidad de Sobolev-Banach
    # --------------------------------------------------------------------
    def evaluate_banach_regularity(
        self,
        v: NDArray[np.float64],
        norm_riemannian: float
    ) -> float:
        r"""
        Audita la regularidad de Sobolev-Banach sobre el fibrado:
        $$\mathcal{R}_B(v) = \frac{\|v\|_1}{\|v\|_{\ell^2, G}}$$
        Satisface la cota analítica de Sobolev: $1.0 \le \mathcal{R}_B(v) \le \sqrt{d}$.
        Se certifica explícitamente la cota (ausente en v4: la ratio se calculaba
        sin verificar su propia consistencia analítica).
        """
        norm_l1 = KahanNeumaierSum.sum(np.abs(v))
        ratio = float(norm_l1 / max(norm_riemannian, _WILKINSON_FLOOR))
        dim = v.shape[0]
        upper_bound = math.sqrt(dim)
        if not (1.0 - _BANACH_SOBOLEV_TOLERANCE <= ratio <= upper_bound + _BANACH_SOBOLEV_TOLERANCE):
            logger.warning(
                "Ratio de Banach R_B(v) = %.6f fuera de la cota de Sobolev [1, sqrt(%d)=%.6f].",
                ratio, dim, upper_bound
            )
        return ratio

    # --------------------------------------------------------------------
    # 1.5 — Ingesta pública y canonización terminal (puerto hacia Fase 2)
    # --------------------------------------------------------------------
    def observe_leptonic_flavors(
        self,
        e_state: NDArray[np.float64],
        mu_state: NDArray[np.float64],
        tau_state: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> LeptonicObservationKernel:
        r"""
        Ingesta los tensores de sabor crudos, ejecuta el saneamiento numérico de signo
        y transfiere el flujo a la canonización formal.
        """
        if e_state.ndim != 1 or mu_state.ndim != 1 or tau_state.ndim != 1:
            raise DimensionMismatchError("Los estados de sabor (e, mu, tau) deben ser tensores 1D.")

        dim = e_state.shape[0]
        if mu_state.shape[0] != dim or tau_state.shape[0] != dim:
            raise DimensionMismatchError(
                f"Discrepancia en dimensiones de sabores: e({dim}), mu({mu_state.shape[0]}), tau({tau_state.shape[0]})."
            )

        c_e = self._sanitize_signed_zero(e_state)
        c_mu = self._sanitize_signed_zero(mu_state)
        c_tau = self._sanitize_signed_zero(tau_state)

        return self.canonize_leptonic_observation_kernel(
            e_state=c_e,
            mu_state=c_mu,
            tau_state=c_tau,
            G_metric=G_metric
        )

    def canonize_leptonic_observation_kernel(
        self,
        e_state: NDArray[np.float64],
        mu_state: NDArray[np.float64],
        tau_state: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> LeptonicObservationKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1:
        Canoniza el expediente topológico inmutable de Fase 1. Construye el
        tensor de Gram de sabores certificado PSD, sus ángulos de entrelazamiento,
        evalúa las normas de Banach y genera el sello HMAC-SHA256.

        Este método constituye el **puerto canónico de acoplamiento directo**
        hacia la Fase 2: toda la clase `Phase2_PMNSOscillationOrient` hereda
        de esta clase precisamente para consumir su salida `LeptonicObservationKernel`
        como insumo primario de la orientación de sabor PMNS/MSW que sigue a continuación.
        """
        cache = self._audit_metric_regularity(G_metric)
        if cache.dimension != e_state.shape[0]:
            raise DimensionMismatchError(
                f"Dimensión de métrica G ({cache.dimension}) no coincide con los vectores de sabor ({e_state.shape[0]})."
            )

        flavor_matrix = np.vstack([e_state, mu_state, tau_state])  # 3 x d

        n_e_euc = float(la.norm(e_state, ord=2))
        n_mu_euc = float(la.norm(mu_state, ord=2))
        n_tau_euc = float(la.norm(tau_state, ord=2))

        n_e_rie = self._compute_riemannian_norm(e_state, cache.g_base)
        n_mu_rie = self._compute_riemannian_norm(mu_state, cache.g_base)
        n_tau_rie = self._compute_riemannian_norm(tau_state, cache.g_base)

        r_e = self.evaluate_banach_regularity(e_state, n_e_rie)
        r_mu = self.evaluate_banach_regularity(mu_state, n_mu_rie)
        r_tau = self.evaluate_banach_regularity(tau_state, n_tau_rie)

        gram_mat, gram_eigvals, entanglement_angles = self._compute_flavor_gram_geometry(
            flavor_matrix, cache.g_base
        )

        seal = self._compute_hmac_seal(
            flavor_matrix.tobytes(),
            cache.g_base.tobytes(),
            gram_mat.tobytes(),
            f"{cache.condition_number:.8e}_{cache.hodge_volume_form:.8e}_"
            f"{cache.inversion_residual:.8e}_{cache.asymmetry_defect:.8e}".encode("ascii")
        )

        return LeptonicObservationKernel(
            flavor_state_matrix=flavor_matrix,
            flavor_norms_euclidean=(n_e_euc, n_mu_euc, n_tau_euc),
            flavor_norms_riemannian=(n_e_rie, n_mu_rie, n_tau_rie),
            banach_ratios=(r_e, r_mu, r_tau),
            gram_matrix=gram_mat,
            gram_eigenvalues=gram_eigvals,
            flavor_entanglement_angles=entanglement_angles,
            metric_cache=cache,
            sha256_seal=seal,
            observation_timestamp=time.time()
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: MATRIZ PMNS MAJORANA, EFECTO MSW, JARLSKOG Y GRAFO DE SABOR
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_PMNSOscillationOrient(Phase1_LeptonicFlavorObserver):
    r"""
    FASE 2 — Orient:
    Hereda de la Fase 1 e ingiere directamente el `LeptonicObservationKernel`
    producido por `canonize_leptonic_observation_kernel`. Construye la matriz
    $U_{\mathrm{PMNS}}$ con fases de Dirac y Majorana en $U(3)$, certifica el
    invariante de Jarlskog $\mathcal{J}_{\mathrm{CP}}$ contra su fórmula cerrada,
    ensambla y propaga el Hamiltoniano efectivo MSW (con la corrección física de
    factor 2 y de unidades del potencial de materia), y audita la topología
    espectral del grafo de oscilación mediante teoría de Markov/Perron-Frobenius.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)

    # --------------------------------------------------------------------
    # 2.1 — Validación de dominio físico y síntesis de la matriz PMNS
    # --------------------------------------------------------------------
    @staticmethod
    def _validate_mixing_angles(theta12: float, theta23: float, theta13: float) -> None:
        r"""
        Verifica que los ángulos de mezcla habiten el dominio físico estándar
        $\theta_{ij} \in [0, \pi/2]$ (parametrización PDG). Emite advertencia
        no bloqueante para permitir exploración de física más allá del Modelo
        Estándar (BSM), preservando la trazabilidad del análisis.
        """
        for name, val in (("theta12", theta12), ("theta23", theta23), ("theta13", theta13)):
            if not (0.0 <= val <= (math.pi / 2.0) + 1.0e-9):
                logger.warning(
                    "Ángulo de mezcla %s = %.6f rad fuera del dominio físico PDG [0, pi/2].", name, val
                )

    def _build_full_pmns_matrix(
        self,
        theta12: float = 0.587,
        theta23: float = 0.855,
        theta13: float = 0.149,
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0
    ) -> Tuple[NDArray[np.complex128], float, float]:
        r"""
        Sintetiza la Matriz PMNS completa en $U(3)$ con fases de Majorana:
        $$U = R_{23} \Gamma_\delta R_{13} \Gamma_\delta^\dagger R_{12} \operatorname{diag}\left(1, e^{i \alpha_{21}/2}, e^{i \alpha_{31}/2}\right)$$
        Calcula el invariante de Jarlskog **numéricamente** desde $U$ completa
        (post-Majorana) y **analíticamente** desde la fórmula cerrada:
        $$\mathcal{J}_{\mathrm{CP}} = c_{12} s_{12} c_{23} s_{23} c_{13}^2 s_{13} \sin \delta_{\mathrm{CP}}$$
        El residuo entre ambos certifica computacionalmente que $\mathcal{J}_{CP}$
        es invariante bajo la reparametrización de fase de Majorana (teorema de
        invariancia de gauge, verificado empíricamente en cada llamada).
        """
        self._validate_mixing_angles(theta12, theta23, theta13)

        c12, s12 = math.cos(theta12), math.sin(theta12)
        c23, s23 = math.cos(theta23), math.sin(theta23)
        c13, s13 = math.cos(theta13), math.sin(theta13)

        cp_phase = np.exp(1j * delta_cp)
        cp_conj = np.conj(cp_phase)

        U_D = np.zeros((3, 3), dtype=np.complex128)
        U_D[0, 0] = c12 * c13
        U_D[0, 1] = s12 * c13
        U_D[0, 2] = s13 * cp_conj

        U_D[1, 0] = -s12 * c23 - c12 * s23 * s13 * cp_phase
        U_D[1, 1] = c12 * c23 - s12 * s23 * s13 * cp_phase
        U_D[1, 2] = s23 * c13

        U_D[2, 0] = s12 * s23 - c12 * c23 * s13 * cp_phase
        U_D[2, 1] = -c12 * s23 - s12 * c23 * s13 * cp_phase
        U_D[2, 2] = c23 * c13

        diag_majorana = np.diag([1.0 + 0.0j, np.exp(1j * alpha21 / 2.0), np.exp(1j * alpha31 / 2.0)])
        U_full = U_D @ diag_majorana

        jarlskog_numeric = float(
            np.imag(U_full[0, 0] * U_full[1, 1] * np.conj(U_full[0, 1]) * np.conj(U_full[1, 0]))
        )
        jarlskog_analytic = c12 * s12 * c23 * s23 * (c13 ** 2) * s13 * math.sin(delta_cp)

        return U_full, jarlskog_numeric, jarlskog_analytic

    # --------------------------------------------------------------------
    # 2.2 — Ensamblaje granular del Hamiltoniano efectivo MSW (vacío + materia)
    # --------------------------------------------------------------------
    def _build_effective_vacuum_hamiltonian(
        self,
        U_pmns: NDArray[np.complex128],
        baseline_L_km: float,
        energy_E_GeV: float,
        dm2_21: float,
        dm2_31: float
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Construye la contribución de vacío de la fase de evolución total
        $H_{\rm vac}\cdot L$ en la base de sabor. La fase acumulada por el
        autoestado de masa $i$ tras recorrer $L$ es $\phi_i = m_i^2 L/(2E)$
        (unidades naturales $\hbar=c=1$), que en las unidades convencionales
        (eV², km, GeV) se expresa como $\phi_i = 2\cdot 1.267\cdot \Delta m_i^2 L/E$.
        **Corrección respecto de v4**: el coeficiente de fase incluye ahora el
        factor 2 físicamente correcto (v4 lo omitía, subestimando la fase de
        oscilación por un factor exacto de 2).
        $$H_{\rm vac} L = \left(2\cdot 1.267\cdot \frac{L}{E}\right)\, U\,\operatorname{diag}(0,\Delta m_{21}^2,\Delta m_{31}^2)\,U^\dagger$$
        """
        E_eff = max(energy_E_GeV, _WILKINSON_FLOOR)
        phase_coefficient = 2.0 * _OSC_PHASE_CONVERSION_FACTOR * baseline_L_km / E_eff
        M_diag = np.diag([0.0, dm2_21, dm2_31]).astype(np.complex128)
        H_vac_L = phase_coefficient * (U_pmns @ M_diag @ U_pmns.conj().T)
        return H_vac_L, phase_coefficient

    def _build_matter_potential_operator(
        self,
        matter_potential_eV: float,
        baseline_L_km: float
    ) -> NDArray[np.complex128]:
        r"""
        Construye la contribución MSW del potencial efectivo de materia
        $V_e = \sqrt{2}G_F N_e$ como fase acumulada $V_e \cdot L$, convertida
        de (eV, km) a fase adimensional mediante $\hbar c$:
        $$\phi_{\rm mat} = V_e[\text{eV}]\cdot L[\text{km}]\cdot \frac{1}{\hbar c}, \qquad
        \frac{1}{\hbar c} \approx 5.06773\times10^{9}\ (\text{eV}\cdot\text{km})^{-1}$$
        **Corrección respecto de v4**: se elimina el factor espurio $10^{-9}$
        que anulaba en 9 órdenes de magnitud el efecto de materia.
        """
        if abs(matter_potential_eV) <= _WILKINSON_FLOOR:
            return np.zeros((3, 3), dtype=np.complex128)
        phase_matter = matter_potential_eV * baseline_L_km * _INV_HBAR_C_EV_KM
        return np.diag([phase_matter, 0.0, 0.0]).astype(np.complex128)

    def _propagate_symplectic_evolution(
        self,
        H_total: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Ejecuta la exponenciación matricial simpléctica de Lie
        $$S(L) = e^{-i H_{\rm total}}$$
        y **audita explícitamente** la unitoridad resultante
        $\|S S^\dagger - I\|_F$, condición necesaria para la conservación de
        probabilidad total; en v4 esta auditoría no existía (se asumía
        implícitamente sin verificación).
        """
        S_matrix = la.expm(-1j * H_total)
        unitarity_defect = float(
            la.norm(S_matrix @ S_matrix.conj().T - np.eye(3, dtype=np.complex128), ord='fro')
        )
        if unitarity_defect > _EVOLUTION_UNITARITY_TOLERANCE:
            raise EvolutionUnitarityError(
                f"El operador de evolución S(L) viola la unitoridad: "
                f"||S S^dagger - I||_F = {unitarity_defect:.4e} > {_EVOLUTION_UNITARITY_TOLERANCE:.2e}"
            )
        return S_matrix, unitarity_defect

    def _extract_transition_probabilities(
        self,
        S_matrix: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Extrae la matriz de probabilidades de transición con la convención
        física correcta $P(\nu_\alpha \to \nu_\beta) = |S_{\beta\alpha}|^2$,
        es decir, $P = (|S|^2)^\top$ (fila = sabor inicial, columna = sabor final).
        **Corrección respecto de v4**: la v4 omitía la transposición.

        Por el teorema de Birkhoff-von Neumann aplicado a matrices unistocásticas,
        $P$ es automáticamente **doblemente estocástica** cuando $S$ es unitaria:
        tanto las sumas por filas como por columnas de $|S|^2$ valen 1. Se audita
        simultáneamente ambas condiciones como un único `stochastic_defect`, y se
        eleva `NumericalInstabilityError` si el defecto excede el umbral FPU,
        lo cual indicaría una ruptura previa de unitoridad no capturada.
        """
        P_raw = (np.abs(S_matrix) ** 2).T
        row_sums = np.sum(P_raw, axis=1)
        col_sums = np.sum(P_raw, axis=0)
        stochastic_defect = float(
            max(np.max(np.abs(row_sums - 1.0)), np.max(np.abs(col_sums - 1.0)))
        )
        if stochastic_defect > _STOCHASTIC_DEFECT_TOLERANCE:
            raise NumericalInstabilityError(
                f"Defecto de doble-estocasticidad (Birkhoff-von Neumann) = {stochastic_defect:.4e} "
                f"excede la tolerancia {_STOCHASTIC_DEFECT_TOLERANCE:.2e}."
            )
        P = P_raw / np.clip(row_sums, _WILKINSON_FLOOR, None)[:, None]
        return np.clip(P, 0.0, 1.0), stochastic_defect

    def _compute_msw_oscillation_probabilities(
        self,
        U_pmns: NDArray[np.complex128],
        baseline_L_km: float = 100.0,
        energy_E_GeV: float = 1.0,
        dm2_21: float = 7.53e-5,
        dm2_31: float = 2.453e-3,
        matter_potential_eV: float = 0.0
    ) -> Tuple[NDArray[np.float64], float, float, NDArray[np.float64], NDArray[np.complex128]]:
        r"""
        Orquesta la resolución completa de la oscilación de 3 sabores en
        presencia de materia (MSW), ensamblando los 3 bloques atómicos
        precedentes (Hamiltoniano de vacío, operador de materia, propagación
        simpléctica) y extrayendo las probabilidades certificadas.
        Retorna adicionalmente el espectro real $\{\omega_i\}$ del Hamiltoniano
        efectivo (garantizado real por hermiticidad) y el propio Hamiltoniano,
        insumos necesarios para la certificación espectral de Fase 3.
        """
        H_vac_L, _phase_coefficient = self._build_effective_vacuum_hamiltonian(
            U_pmns, baseline_L_km, energy_E_GeV, dm2_21, dm2_31
        )
        H_mat_L = self._build_matter_potential_operator(matter_potential_eV, baseline_L_km)
        H_total = H_vac_L + H_mat_L

        S_matrix, evolution_defect = self._propagate_symplectic_evolution(H_total)
        P_matrix, stochastic_defect = self._extract_transition_probabilities(S_matrix)
        H_eigvals = np.sort(la.eigvalsh(H_total))

        return P_matrix, evolution_defect, stochastic_defect, H_eigvals, H_total

    # --------------------------------------------------------------------
    # 2.3 — Topología espectral del grafo de mezcla (Markov / Perron-Frobenius)
    # --------------------------------------------------------------------
    def _compute_flavor_graph_topology(
        self,
        P_matrix: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], float, NDArray[np.float64], float, float]:
        r"""
        Modela la oscilación de sabor como un grafo estocástico ponderado.
        1. Laplaciano simétrico de Hodge $L_{\rm flavor} = I_3 - \frac{1}{2}(P+P^\top)$
           y brecha de Fiedler $\lambda_2$.
        2. **Distribución estacionaria** $\pi$ vía el autovector izquierdo de
           Perron-Frobenius de $P$ (asociado al autovalor 1), certificando
           ergodicidad (masa negativa despreciable) antes de normalizar.
        3. **Entropía de Shannon de mezcla** $H(\pi) = -\sum_\alpha \pi_\alpha \ln \pi_\alpha$
           (prometida en la sinopsis original y ausente en la implementación v4).
        4. **Constante de Kemeny** $\mathcal{K} = \sum_{\lambda_i \neq 1} \frac{1}{1-\lambda_i}$,
           que mide el tiempo medio de recurrencia/mezcla de la cadena de Markov
           de sabor, invariante de la topología del grafo independiente del
           estado inicial (Grinstead & Snell).
        """
        P_sym = 0.5 * (P_matrix + P_matrix.T)
        L_flavor = np.eye(3, dtype=np.float64) - P_sym
        eigvals_L = np.sort(la.eigvalsh(L_flavor))
        fiedler_gap = float(max(eigvals_L[1], 0.0))

        eigvals_P, eigvecs_P_T = la.eig(P_matrix.T)
        idx_perron = int(np.argmin(np.abs(eigvals_P - 1.0)))
        pi_raw = np.real(eigvecs_P_T[:, idx_perron])

        negative_mass = float(np.sum(np.abs(pi_raw[pi_raw < 0.0])))
        if negative_mass > _MARKOV_ERGODICITY_TOLERANCE:
            raise MarkovErgodicityError(
                f"Cadena de Markov de sabor no ergódica: masa negativa en el autovector "
                f"de Perron-Frobenius = {negative_mass:.4e}"
            )

        pi_clipped = np.clip(pi_raw, 0.0, None)
        pi_sum = KahanNeumaierSum.sum(pi_clipped)
        stationary = pi_clipped / max(pi_sum, _WILKINSON_FLOOR)

        shannon_entropy = -KahanNeumaierSum.sum(
            [p * math.log(p) for p in stationary if p > _WILKINSON_FLOOR]
        )

        kemeny_terms: List[float] = []
        for lam in eigvals_P:
            if abs(lam - 1.0) < 1.0e-9:
                continue
            denom = 1.0 - lam
            if abs(denom) > _WILKINSON_FLOOR:
                kemeny_terms.append(float((1.0 / denom).real))
        kemeny_constant = float(KahanNeumaierSum.sum(kemeny_terms)) if kemeny_terms else 0.0

        return L_flavor, fiedler_gap, stationary, float(max(shannon_entropy, 0.0)), kemeny_constant

    # --------------------------------------------------------------------
    # 2.4 — Orientación terminal y puerto de acoplamiento hacia Fase 3
    # --------------------------------------------------------------------
    def orient_leptonic_flavor(
        self,
        kernel: LeptonicObservationKernel,
        theta12: float = 0.587,
        theta23: float = 0.855,
        theta13: float = 0.149,
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0,
        baseline_L_km: float = 100.0,
        energy_E_GeV: float = 1.0,
        matter_potential_eV: float = 0.0
    ) -> LeptonicFlavorReport:
        r"""
        Conduce la orientación completa de sabor leptónico sobre el Kernel canonizado.
        """
        U_pmns, jarlskog_numeric, jarlskog_analytic = self._build_full_pmns_matrix(
            theta12=theta12, theta23=theta23, theta13=theta13,
            delta_cp=delta_cp, alpha21=alpha21, alpha31=alpha31
        )
        jarlskog_cross_residual = abs(jarlskog_numeric - jarlskog_analytic)

        unitarity_res = float(
            la.norm(U_pmns @ U_pmns.conj().T - np.eye(3, dtype=np.complex128), ord='fro')
        )
        is_unitary = unitarity_res <= _PMNS_UNITARITY_TOLERANCE

        P_matrix, evolution_defect, stochastic_defect, H_eigvals, H_total = self._compute_msw_oscillation_probabilities(
            U_pmns=U_pmns, baseline_L_km=baseline_L_km, energy_E_GeV=energy_E_GeV,
            matter_potential_eV=matter_potential_eV
        )

        L_flavor, fiedler_gap, stationary, shannon_entropy, kemeny_constant = self._compute_flavor_graph_topology(
            P_matrix
        )

        n_e, n_mu, n_tau = kernel.flavor_norms_riemannian
        q_vec = np.array([n_e, n_mu, n_tau], dtype=np.float64)
        l_total = KahanNeumaierSum.sum(q_vec)

        q_projected = P_matrix @ q_vec
        l_projected = KahanNeumaierSum.sum(q_projected)

        charge_res = abs(l_projected - l_total)
        is_charge_conserved = charge_res <= 1.0e-8

        return LeptonicFlavorReport(
            kernel=kernel,
            pmns_matrix=U_pmns,
            pmns_unitarity_residual=unitarity_res,
            jarlskog_invariant=jarlskog_numeric,
            jarlskog_invariant_analytic=jarlskog_analytic,
            jarlskog_cross_residual=jarlskog_cross_residual,
            is_pmns_unitary=is_unitary,
            effective_hamiltonian=H_total,
            vacuum_hamiltonian_eigenvalues=H_eigvals,
            evolution_unitarity_defect=evolution_defect,
            stochastic_normalization_defect=stochastic_defect,
            oscillation_probabilities=P_matrix,
            msw_matter_potential=matter_potential_eV,
            flavor_graph_laplacian=L_flavor,
            flavor_graph_spectral_gap=fiedler_gap,
            flavor_stationary_distribution=stationary,
            shannon_mixing_entropy=shannon_entropy,
            kemeny_constant=kemeny_constant,
            total_leptonic_charge=l_total,
            charge_conservation_residual=charge_res,
            is_charge_conserved=is_charge_conserved,
            orientation_timestamp=time.time()
        )

    def synthesize_leptonic_orientation(
        self,
        kernel_or_e: Union[LeptonicObservationKernel, NDArray[np.float64]],
        mu_state: Optional[NDArray[np.float64]] = None,
        tau_state: Optional[NDArray[np.float64]] = None,
        G_metric: Optional[NDArray[np.float64]] = None,
        theta12: float = 0.587,
        theta23: float = 0.855,
        theta13: float = 0.149,
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0,
        baseline_L_km: float = 100.0,
        energy_E_GeV: float = 1.0,
        matter_potential_eV: float = 0.0
    ) -> LeptonicFlavorReport:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2:
        Garantiza la continuidad functorial estricta. Si recibe un
        `LeptonicObservationKernel`, procede directamente a la orientación;
        si recibe tensores crudos, invoca de forma transparente
        `canonize_leptonic_observation_kernel` de la Fase 1. El
        `LeptonicFlavorReport` devuelto constituye el **puerto de ingestión**
        de la Fase 3: la clase `Phase3_LeptonicGovernanceEngine` hereda de esta
        clase para consumir directamente este reporte en su telemetría final.
        """
        if isinstance(kernel_or_e, LeptonicObservationKernel):
            kernel = kernel_or_e
        else:
            if mu_state is None or tau_state is None or G_metric is None:
                raise DimensionMismatchError(
                    "Si no se suministra un LeptonicObservationKernel, deben suministrarse e, mu, tau y G_metric."
                )
            kernel = self.canonize_leptonic_observation_kernel(
                e_state=kernel_or_e,
                mu_state=mu_state,
                tau_state=tau_state,
                G_metric=G_metric
            )

        return self.orient_leptonic_flavor(
            kernel=kernel,
            theta12=theta12,
            theta23=theta23,
            theta13=theta13,
            delta_cp=delta_cp,
            alpha21=alpha21,
            alpha31=alpha31,
            baseline_L_km=baseline_L_km,
            energy_E_GeV=energy_E_GeV,
            matter_potential_eV=matter_potential_eV
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: TELEMETRÍA CUÁNTICA FPU, RESONANCIA RLC, BAUER-FIKE Y SELLO HMAC
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_LeptonicGovernanceEngine(Phase2_PMNSOscillationOrient):
    r"""
    FASE 3 — Act:
    Orquestador terminal de alta precisión en silicio y FPU. Construye la
    matriz de densidad reducida **genuina** de sabor a partir del tensor de
    Gram (Fase 1), calcula su entropía de von Neumann y pureza, contrasta la
    distribución cuántica diagonal con la distribución estacionaria clásica
    (Fase 2) vía divergencia de Kullback-Leibler, modela el factor de calidad
    $Q$ topológico y el $Q$ resonante fundado en las frecuencias físicas del
    Hamiltoniano, certifica la estabilidad espectral mediante el teorema de
    Bauer-Fike y emite el `LeptonicEngineState` sellado con HMAC-SHA256.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)

    # --------------------------------------------------------------------
    # 3.1 — Matriz de densidad reducida genuina y entropías cuánticas
    # --------------------------------------------------------------------
    def compute_reduced_flavor_density_matrix(
        self,
        kernel: LeptonicObservationKernel
    ) -> NDArray[np.complex128]:
        r"""
        Construye la matriz de densidad reducida **genuina** de sabor:
        $$\rho_{\rm flavor} = \frac{S_{\rm Gram}}{\operatorname{Tr}(S_{\rm Gram})}$$
        donde $S_{\rm Gram}$ es el tensor de Gram certificado PSD de la Fase 1.
        Al ser $S_{\rm Gram} \succeq 0$ por construcción, $\rho$ es una matriz
        de densidad válida ($\rho \succeq 0$, $\operatorname{Tr}\rho = 1$).
        **Corrección respecto de v4**: la v4 usaba $\rho=(P+P^\top)/6$, una
        combinación ad hoc de la matriz estocástica clásica sin traza unitaria
        garantizada ni positividad semidefinida certificada.
        """
        gram = kernel.gram_matrix
        trace_gram = float(np.trace(gram))
        if trace_gram <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                "Traza del tensor de Gram de sabor colapsada; imposible normalizar como matriz de densidad."
            )
        rho = (gram / trace_gram).astype(np.complex128)
        return rho

    def compute_von_neumann_flavor_entropy(
        self,
        rho: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        r"""
        Calcula la entropía cuántica de von Neumann de la matriz de densidad genuina:
        $$S(\rho) = -\operatorname{Tr}(\rho \ln \rho) = -\sum_i \lambda_i \ln \lambda_i$$
        y la pureza $\operatorname{Tr}(\rho^2) \in [1/3, 1]$ (cota inferior de
        máxima mixtura para $d=3$, cota superior de estado puro), certificando
        implícitamente la consistencia física del estado reducido.
        """
        eigvals = np.clip(la.eigvalsh(rho).real, 0.0, 1.0)
        s_vn = -KahanNeumaierSum.sum(
            [ev * math.log(ev) for ev in eigvals if ev > _WILKINSON_FLOOR]
        )
        purity = float(KahanNeumaierSum.sum(eigvals ** 2))
        return float(max(s_vn, 0.0)), purity

    def compute_classical_quantum_relative_entropy(
        self,
        rho: NDArray[np.complex128],
        stationary_distribution: NDArray[np.float64]
    ) -> float:
        r"""
        Calcula la divergencia de Kullback-Leibler entre la distribución
        estacionaria clásica $\pi$ (Fase 2, teoría de Markov) y la distribución
        diagonal cuántica $q_\alpha = \rho_{\alpha\alpha}$ (Fase 3, teoría cuántica):
        $$D_{\rm KL}(\pi \| q) = \sum_\alpha \pi_\alpha \ln\frac{\pi_\alpha}{q_\alpha} \ge 0$$
        Cuantifica la coherencia entre la descripción probabilística clásica del
        grafo de mezcla y la descripción cuántica de la matriz de densidad de
        sabor; un valor cercano a cero certifica consistencia clásico-cuántica.
        """
        q_diag = np.clip(np.real(np.diag(rho)), _WILKINSON_FLOOR, 1.0)
        p_dist = np.clip(stationary_distribution, _WILKINSON_FLOOR, 1.0)
        kl_terms = p_dist * np.log(p_dist / q_diag)
        return float(KahanNeumaierSum.sum(kl_terms))

    # --------------------------------------------------------------------
    # 3.2 — Factores de calidad RLC: topológico (espectral) y resonante (físico)
    # --------------------------------------------------------------------
    def compute_equivalent_rlc_q_factor(
        self,
        fiedler_gap: float,
        pmns_unitarity_residual: float
    ) -> float:
        r"""
        Factor de Calidad topológico en la analogía de circuito resonante RLC:
        $$Q_{\rm top} \cong \frac{\lambda_2(L_{\mathrm{flavor}})}{\max(\mathcal{R}_{\mathrm{unitarity}}, \varepsilon_{\mathrm{Wilkinson}})}$$
        Mide la coherencia de fase del satélite frente a la amortiguación
        ohmica inducida por defectos de unitoridad numérica.
        """
        damping = max(pmns_unitarity_residual, 1.0e-11)
        q_val = float(fiedler_gap / damping)
        return float(np.clip(q_val, 0.0, 1.0e8))

    def compute_resonant_quality_factor(
        self,
        hamiltonian_eigenvalues: NDArray[np.float64],
        evolution_unitarity_defect: float,
        charge_conservation_residual: float
    ) -> float:
        r"""
        Factor de Calidad **resonante físico**, fundado directamente en las
        frecuencias propias reales del Hamiltoniano efectivo MSW
        $\omega_i \in \operatorname{spec}(H_{\rm eff})$ (garantizadas reales
        por hermiticidad):
        $$Q_{\rm res} = \frac{\omega_{\max}}{\gamma}, \qquad
        \gamma = \max\left(\delta_{\rm evol} + \delta_{\rm carga},\ \varepsilon\right)$$
        donde $\gamma$ actúa como tasa de amortiguación/decoherencia efectiva
        derivada de los residuos de unitoridad y conservación de Noether.
        Complementa a `compute_equivalent_rlc_q_factor` (de origen topológico)
        con una magnitud de origen estrictamente espectral-físico.
        """
        freqs = np.sort(np.abs(hamiltonian_eigenvalues))
        omega_max = float(freqs[-1]) if freqs.size > 0 else 0.0
        damping = max(evolution_unitarity_defect + charge_conservation_residual, 1.0e-11)
        q_res = float(omega_max / damping)
        return float(np.clip(q_res, 0.0, 1.0e10))

    # --------------------------------------------------------------------
    # 3.3 — Certificación espectral de Bauer-Fike
    # --------------------------------------------------------------------
    def _audit_bauer_fike_stability(
        self,
        H_effective: NDArray[np.complex128]
    ) -> Tuple[bool, float]:
        r"""
        Certifica la estabilidad numérica del problema de autovalores del
        Hamiltoniano efectivo mediante el **Teorema de Bauer-Fike**: para toda
        matriz normal (en particular, hermítica) $H = V\Lambda V^\dagger$ con
        $V$ unitaria, el número de condición de la base de autovectores es
        exactamente $\kappa_2(V) = 1$, por lo que
        $$|\lambda_{\rm computado} - \lambda_{\rm exacto}| \le \|\delta H\|_2$$
        es decir, los autovalores están **óptimamente condicionados** —
        cualquier perturbación de la FPU se propaga sin amplificación. La
        auditoría verifica computacionalmente la hermiticidad de $H$
        (condición suficiente para invocar el teorema).
        """
        hermitian_defect = float(la.norm(H_effective - H_effective.conj().T, ord='fro'))
        is_certified = hermitian_defect <= _HERMITICITY_CERTIFICATION_TOLERANCE
        if not is_certified:
            logger.warning(
                "Certificación Bauer-Fike fallida: defecto de hermiticidad = %.4e excede %.2e",
                hermitian_defect, _HERMITICITY_CERTIFICATION_TOLERANCE
            )
        return is_certified, hermitian_defect

    # --------------------------------------------------------------------
    # 3.4 — Auditoría terminal, sellado HMAC y emisión del certificado formal
    # --------------------------------------------------------------------
    def execute_leptonic_audit(
        self,
        target_or_e: Union[LeptonicFlavorReport, LeptonicObservationKernel, NDArray[np.float64]],
        mu_state: Optional[NDArray[np.float64]] = None,
        tau_state: Optional[NDArray[np.float64]] = None,
        G_metric: Optional[NDArray[np.float64]] = None,
        theta12: float = 0.587,
        theta23: float = 0.855,
        theta13: float = 0.149,
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0,
        baseline_L_km: float = 100.0,
        energy_E_GeV: float = 1.0,
        matter_potential_eV: float = 0.0
    ) -> LeptonicEngineState:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 3 Y DEL MOTOR COVARIANTE:
        Punto culminante OODA del Motor de Sabor Leptónico.
        Acepta polimórficamente:
        1. `LeptonicFlavorReport` (procedente directamente de Fase 2).
        2. `LeptonicObservationKernel` (procedente de Fase 1, acoplándose vía Fase 2).
        3. Tensores crudos ($e, \mu, \tau, G$), resolviendo la cadena completa F1 -> F2 -> F3.
        """
        t_start = time.perf_counter()

        if isinstance(target_or_e, LeptonicFlavorReport):
            report = target_or_e
            kernel = report.kernel
        elif isinstance(target_or_e, LeptonicObservationKernel):
            kernel = target_or_e
            report = self.orient_leptonic_flavor(
                kernel=kernel, theta12=theta12, theta23=theta23, theta13=theta13,
                delta_cp=delta_cp, alpha21=alpha21, alpha31=alpha31,
                baseline_L_km=baseline_L_km, energy_E_GeV=energy_E_GeV,
                matter_potential_eV=matter_potential_eV
            )
        else:
            report = self.synthesize_leptonic_orientation(
                kernel_or_e=target_or_e, mu_state=mu_state, tau_state=tau_state, G_metric=G_metric,
                theta12=theta12, theta23=theta23, theta13=theta13,
                delta_cp=delta_cp, alpha21=alpha21, alpha31=alpha31,
                baseline_L_km=baseline_L_km, energy_E_GeV=energy_E_GeV,
                matter_potential_eV=matter_potential_eV
            )
            kernel = report.kernel

        if not report.is_pmns_unitary:
            raise PMNSUnitarityError(
                f"Violación de unitoridad PMNS: residual = {report.pmns_unitarity_residual:.4e} > {_PMNS_UNITARITY_TOLERANCE:.2e}"
            )
        if not report.is_charge_conserved:
            raise LeptonicChargeViolationError(
                f"Violación de conservación de carga leptónica: residual = {report.charge_conservation_residual:.4e}"
            )

        rho = self.compute_reduced_flavor_density_matrix(kernel)
        vn_entropy, purity = self.compute_von_neumann_flavor_entropy(rho)
        relative_entropy = self.compute_classical_quantum_relative_entropy(
            rho, report.flavor_stationary_distribution
        )

        q_topological = self.compute_equivalent_rlc_q_factor(
            report.flavor_graph_spectral_gap, report.pmns_unitarity_residual
        )
        q_resonant = self.compute_resonant_quality_factor(
            report.vacuum_hamiltonian_eigenvalues,
            report.evolution_unitarity_defect,
            report.charge_conservation_residual
        )

        bauer_fike_certified, hermitian_defect = self._audit_bauer_fike_stability(report.effective_hamiltonian)

        spectral_drift = float(
            abs(report.pmns_unitarity_residual)
            + abs(report.charge_conservation_residual)
            + abs(hermitian_defect)
        )

        cryptographic_seal = self._compute_hmac_seal(
            kernel.sha256_seal.encode("ascii"),
            report.pmns_matrix.tobytes(),
            report.oscillation_probabilities.tobytes(),
            rho.tobytes(),
            f"{report.jarlskog_invariant:.12e}_{vn_entropy:.12e}_{q_topological:.12e}_"
            f"{q_resonant:.12e}_{purity:.12e}".encode("ascii")
        )

        quantum_signature = (
            f"LEPTONIC_PMNS_U3:J_CP={report.jarlskog_invariant:.6e}:"
            f"GAP={report.flavor_graph_spectral_gap:.6f}:"
            f"Qtop={q_topological:.2e}:Qres={q_resonant:.2e}:"
            f"S_vN={vn_entropy:.6f}:PURITY={purity:.6f}:"
            f"BAUER_FIKE={int(bauer_fike_certified)}:UNITARY={int(report.is_pmns_unitary)}"
        )

        t_elapsed_ms = float((time.perf_counter() - t_start) * 1000.0)

        return LeptonicEngineState(
            kernel=kernel,
            report=report,
            fpu_execution_time_ms=t_elapsed_ms,
            flavor_density_matrix=rho,
            von_neumann_flavor_entropy=vn_entropy,
            quantum_purity=purity,
            classical_quantum_relative_entropy=relative_entropy,
            equivalent_circuit_q_factor=q_topological,
            resonant_quality_factor=q_resonant,
            spectral_stability_drift=spectral_drift,
            is_bauer_fike_certified=bauer_fike_certified,
            cryptographic_seal=cryptographic_seal,
            quantum_flavor_signature=quantum_signature
        )


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR SATELITAL DE SABOR LEPTÓNICO SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class LeptonicFlavorSatelliteEngine(Phase3_LeptonicGovernanceEngine):
    r"""
    Motor Satelital de Sabor y Oscilación Leptónica de-confinado en la FPU.
    Integra la totalidad del pipeline holomorfo anidado (Fases 1, 2 y 3).
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)
        logger.info(
            "LeptonicFlavorSatelliteEngine v5.0.0 inicializado con éxito. "
            "Alineación: PMNS U(3), Fases Majorana, MSW Matter (unidades corregidas), "
            "Jarlskog (auditado analítico/numérico), Banach, Markov/Kemeny, "
            "densidad cuántica genuina, Bauer-Fike, FPU KBN, sellado HMAC-SHA256."
        )


__all__ = [
    "LeptonicFlavorSatelliteEngine",
    "BaseMetricCache",
    "LeptonicObservationKernel",
    "LeptonicFlavorReport",
    "LeptonicEngineState",
    "LeptonicEngineError",
    "DimensionMismatchError",
    "MetricIndefinitenessError",
    "PMNSUnitarityError",
    "LeptonicChargeViolationError",
    "EvolutionUnitarityError",
    "NumericalInstabilityError",
    "MarkovErgodicityError",
    "KahanNeumaierSum",
    "Phase1_LeptonicFlavorObserver",
    "Phase2_PMNSOscillationOrient",
    "Phase3_LeptonicGovernanceEngine",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "═" * 80)
    print(" INICIANDO AUDITORÍA ESPECTRAL Y METROLÓGICA DE LEPTONIC FLAVOR SATELLITE ENGINE")
    print("═" * 80)

    engine = LeptonicFlavorSatelliteEngine()

    dim = 4
    G_test = np.array([
        [2.0, 0.2, 0.0, 0.1],
        [0.2, 1.8, 0.1, 0.0],
        [0.0, 0.1, 2.2, 0.3],
        [0.1, 0.0, 0.3, 1.5]
    ], dtype=np.float64)

    e_vec = np.array([1.0, 0.2, -0.1, 0.05], dtype=np.float64)
    mu_vec = np.array([0.1, 2.0, 0.3, -0.1], dtype=np.float64)
    tau_vec = np.array([0.0, 0.1, 3.0, 0.2], dtype=np.float64)

    state = engine.execute_leptonic_audit(
        target_or_e=e_vec,
        mu_state=mu_vec,
        tau_state=tau_vec,
        G_metric=G_test,
        matter_potential_eV=1.5e-13
    )

    rep = state.report
    print(f"\n[+] TELEMETRÍA DE LA FASE 1 (OBSERVE):")
    print(f"    - Condicionamiento kappa_2(G):     {state.kernel.metric_cache.condition_number:.4f}")
    print(f"    - Residual de inversión ||GG^-1-I||: {state.kernel.metric_cache.inversion_residual:.4e}")
    print(f"    - Forma de volumen de Hodge:       {state.kernel.metric_cache.hodge_volume_form:.6f}")
    print(f"    - Ratios de regularidad Banach:    {tuple(round(x, 4) for x in state.kernel.banach_ratios)}")
    print(f"    - Ángulos de entrelazamiento (rad):{tuple(round(x, 4) for x in state.kernel.flavor_entanglement_angles)}")
    print(f"    - Sello HMAC-SHA256 de Fase 1:     {state.kernel.sha256_seal[:24]}...")

    print(f"\n[+] TELEMETRÍA DE LA FASE 2 (ORIENT):")
    print(f"    - Residual Unitoridad PMNS:        {rep.pmns_unitarity_residual:.4e}")
    print(f"    - Jarlskog num./analítico/residuo: {rep.jarlskog_invariant:.6e} / "
          f"{rep.jarlskog_invariant_analytic:.6e} / {rep.jarlskog_cross_residual:.2e}")
    print(f"    - Defecto de evolución/estoc.:     {rep.evolution_unitarity_defect:.2e} / "
          f"{rep.stochastic_normalization_defect:.2e}")
    print(f"    - Brecha de Fiedler (Grafo):       {rep.flavor_graph_spectral_gap:.6f}")
    print(f"    - Distribución estacionaria pi:    {tuple(round(x, 4) for x in rep.flavor_stationary_distribution)}")
    print(f"    - Entropía de Shannon de mezcla:   {rep.shannon_mixing_entropy:.6f}")
    print(f"    - Constante de Kemeny:             {rep.kemeny_constant:.6f}")
    print(f"    - Carga Leptónica Total L:         {rep.total_leptonic_charge:.6f}")
    print(f"    - Residual Noether Carga:          {rep.charge_conservation_residual:.4e}")
    print(f"    - Matriz de Transición P(nu_a -> nu_b):")
    for r in range(3):
        print(f"      [{rep.oscillation_probabilities[r, 0]:.4f}, "
              f"{rep.oscillation_probabilities[r, 1]:.4f}, "
              f"{rep.oscillation_probabilities[r, 2]:.4f}]")

    print(f"\n[+] TELEMETRÍA DE LA FASE 3 (ACT / FPU):")
    print(f"    - Latencia de Cómputo FPU:         {state.fpu_execution_time_ms:.4f} ms")
    print(f"    - Entropía von Neumann S(rho):     {state.von_neumann_flavor_entropy:.6f}")
    print(f"    - Pureza cuántica Tr(rho^2):       {state.quantum_purity:.6f}")
    print(f"    - Entropía relativa clás.-cuánt.:  {state.classical_quantum_relative_entropy:.6e}")
    print(f"    - Factor de Calidad Q topológico:  {state.equivalent_circuit_q_factor:.2e}")
    print(f"    - Factor de Calidad Q resonante:   {state.resonant_quality_factor:.2e}")
    print(f"    - Certificación Bauer-Fike:        {state.is_bauer_fike_certified}")
    print(f"    - Deriva Espectral Wilkinson:      {state.spectral_stability_drift:.4e}")
    print(f"    - Firma Cuántica de Sabor:         {state.quantum_flavor_signature}")
    print(f"    - Doble Sello Criptográfico HMAC:  {state.cryptographic_seal[:32]}...")

    assert rep.is_pmns_unitary, "Fallo: PMNS no es unitaria dentro de tolerancia"
    assert rep.is_charge_conserved, "Fallo: Violación de la conservación de carga leptónica"
    assert state.is_bauer_fike_certified, "Fallo: Certificación espectral Bauer-Fike no superada"
    print("\n" + "═" * 80)
    print(" ¡AUDITORÍA METROLÓGICA Y DOCTORAL COMPLETADA CON ÉXITO ABSOLUTO!")
    print("═" * 80)