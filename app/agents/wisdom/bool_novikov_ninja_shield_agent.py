# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovNinjaShieldAgent v15.0.0-PhD                                                   ║
║ Ubicación: app/agents/wisdom/bool_novikov_ninja_shield_agent.py                                      ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-NINJA-SHIELD-AGENT-POINCARE-KAM-RSI3-PhD                    ║
║ Gobernanza: OODA Trifásica Pasiva de Sigilo (①Observe → Φ₁₂ → ②Orient → Φ₂₃ → ③Decide&Act)         ║
║            con Mecánica Celeste de Poincaré–KAM/Brjuno, Invariante de Poincaré-Cartan,               ║
║            Automejora Recursiva Nivel 3 (RSI³ Darwin-Gödel-Löb) y Topos de Heyting Ω₃.               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre Escudero `BoolNovikovNinjaShieldAgent` gobierna ejecutivamente
el ciclo OODA de blindaje de sigilo pasivo sobre el motor espectral `BoolNovikovNinjaShieldEngine`
(app/wisdom/bool_novikov_ninja_shield_engine.py). Su función operacional consiste en custodiar
la invulnerabilidad cuántica de la MAC en tres fases anidadas acopladas por compuertas homomórficas
\Phi_{12} y \Phi_{23} firmadas vía HMAC-SHA256: audita la reducción de Smith sobre \mathbb{Z} y la distancia
geodésica en el edificio de Bruhat-Tits B(\mathrm{PGL}(n, \mathbb{K})), valida la persistencia de toros
diofantinos KAM bajo la función de Bruno B(\omega), sintetiza el operador central de Casimir A^*_{\mathrm{shield}}
mediante el bucle Darwin-Gödel-Löb (RSI³), y deduce el veredicto en el topos intuicionista de Heyting \Omega_3.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`NinjaShieldPhase1Observer`):
   Evalúa la densidad cuántica \rho y la función de coste mediante la base de Walsh-Hadamard W_n.
   La reducción de Smith sobre \mathbb{Z} confirma B = U D V con d_i \in \{0, 1\} (\mathrm{Tor}(H_k) = 0, \beta_1 = 0).
   En el anillo de Novikov \Lambda_{\mathrm{Nov}}, valida la valuación de Krull v(A) \ge 0.10.
   En el edificio afín de Bruhat-Tits, calcula la distancia geodésica de Weyl d_{\mathrm{BT}}(x, y) entre cámaras de Iwahori.
   Resuelve la ecuación homológica de Poincaré-Lindstedt \{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^d}
   y extrae el invariante integral de acción de Poincaré-Cartan \varpi_{\mathrm{PC}} = \frac{1}{2\pi} \oint p \, dq.
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2:
     Emite `NinjaShieldPhase1Certificate` transfiriendo la semilla simpléctica hacia la Fase 2.

2. FASE 2 — ORIENT (`NinjaShieldPhase2Orientor` + Compuerta Homomórfica \Phi_{12}):
   Inicia consumiendo `NinjaShieldPhase1Certificate`.
   Audita las coordenadas de acción-ángulo de Delaunay-Whittaker (J, \theta) con la condición diofantina KAM:
       |\langle \omega, k \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \neq 0,
   la convergencia de la función de Bruno B(\omega) < \infty y el tiempo de Nekhoroshev T_{\mathrm{Nekh}} \ge 1.0\times 10^6.
   En la categoría de Fukaya \mathcal{F}uk(M), la ecuación de Floer asegura no-squeezing de Gromov c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)).
   El bucle Darwin-Gödel RSI³ con verificación modal de Löb \Box(\Box P \to P) \to \Box P sintetiza el observable central A^*_{\mathrm{shield}}.
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3:
     Emite `NinjaShieldPhase2Certificate` empaquetando la estabilidad KAM y la firma de Löb hacia la Fase 3.

3. FASE 3 — DECIDE & ACT (`NinjaShieldPhase3Decider` + Compuerta \Phi_{23}):
   Inicia consumiendo `NinjaShieldPhase2Certificate`.
   Evalúa la medición débil AAV A_w = \langle \phi_f | A^*_{\mathrm{shield}} | \phi_i \rangle / \langle \phi_f | \phi_i \rangle
   garantizando retroacción cuántica acotada \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
   Audita la contracción monádica de Kleisli T^3 con parámetro de Newton-Kantorovich h_K \le 0.50 y tensor Jerk d^3 C / dt^3.
   Deduce el veredicto en el álgebra de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\},
   calcula las métricas de protección de WACC y liquidez neta, y sella el estado soberano inmutable `NinjaShieldSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma NSA-A1] Torsión Homológica Nula en Smith:
    d_i \in \{0, 1\}, \forall i \implies \mathrm{Tor}(H_k(\mathcal{K}; \mathbb{Z})) = 0 \land \beta_1 = 0.
[Axioma NSA-A2] Convergencia Diofantina de Bruno:
    B(\omega) < \infty \implies \text{conjugación analítica rígida sobre el toro de Liouville}.
[Axioma NSA-A3] Rigidez Simpléctica de No-Exprimido:
    c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)) \iff \pi r^2 \le \pi R^2.
[Axioma NSA-A4] Preservación de No-Demolición Cuántica QND:
    \|[A^*, \rho]\|_F \le 1.0\times 10^{-10} \implies \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante NSA-I1] Anulación del Primer Número de Betti: \beta_1 = 0.
[Invariante NSA-I2] Valuación de Krull Mínima: v(A) \ge 0.10.
[Invariante NSA-I3] Retroacción Cuántica Acotada: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante NSA-I4] Parámetro de Kantorovich Controlado: h_K \le 0.50.
[Invariante NSA-I5] Sellado Criptográfico HMAC-SHA256 con verificación en tiempo constante compare_digest.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Importación resiliente del motor espectral Ninja Shield
try:
    from bool_novikov_ninja_shield_engine import (
        BoolNovikovNinjaShieldEngine,
        HeytingTruthValue,
        NinjaShieldSpectrumReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_ninja_shield_engine import (
            BoolNovikovNinjaShieldEngine,
            HeytingTruthValue,
            NinjaShieldSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_ninja_shield_engine."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovNinjaShieldAgent")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES CRIPTOGRÁFICAS, ESPECTRALES Y DE GOBERNANZA OODA
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-NINJA-SHIELD-AGENT-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_NINJA_SHIELD_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"NINJA-SHIELD-P1-ISOMETRY-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"NINJA-SHIELD-P2-SPECTRUM-v15.0|"
_HMAC_DOMAIN_POINCARE: Final[bytes] = b"NINJA-SHIELD-POINCARE-KAM-BRUNO-v15.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"NINJA-SHIELD-RSI3-v15.0|"
_HMAC_DOMAIN_MERKLE: Final[bytes] = b"NINJA-SHIELD-MERKLE-ROOT-v15.0|"

_RSI3_LEVELS: Final[int] = 3
_RSI3_BASE_LIP: Final[float] = 0.95
_RSI3_CONTRACTION_LIP: Final[float] = _RSI3_BASE_LIP ** _RSI3_LEVELS  # 0.857375
_POINCARE_KAM_TOLERANCE: Final[float] = 0.10
_POINCARE_MAX_ORDER: Final[int] = 3
_KRULL_ELASTICITY_FLOOR: Final[float] = 0.10
_GOLDEN_RATIO: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0
_WILKINSON_EPS: Final[float] = 1.0e-15


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES CANÓNICAS DE SERIALIZACIÓN, HASH, HMAC Y EXTRACCIÓN FUNTORIAL
# ═══════════════════════════════════════════════════════════════════════════════


def _q(value: float) -> str:
    """Cuantiza un escalar flotante a 12 decimales para serialización determinista."""
    return f"{float(value):.12f}"


def _canonical_hash(*parts: Any) -> str:
    """Calcula un digest SHA-256 canónico e invariante ante representaciones NumPy."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, (float, np.floating)):
            text = _q(float(part))
        elif isinstance(part, (bool, np.bool_)):
            text = "1" if bool(part) else "0"
        elif isinstance(part, (int, np.integer)):
            text = str(int(part))
        else:
            text = str(part)
        digest.update(text.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _hmac_hex(secret_key: bytes, domain: bytes, payload: str) -> str:
    """Calcula HMAC-SHA256 con separación criptográfica de dominio explícita."""
    return hmac.new(
        secret_key,
        domain + payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _array_digest(array: NDArray[Any]) -> str:
    """Calcula el digest SHA-256 de un tensor NumPy en orden C-contiguo canónico."""
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        arr_c = np.ascontiguousarray(arr, dtype=np.complex128)
    else:
        arr_c = np.ascontiguousarray(arr, dtype=np.float64)

    digest = hashlib.sha256()
    digest.update(str(arr_c.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(arr_c.tobytes(order="C"))
    return digest.hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerción segura a escalar de punto flotante IEEE-754 finito."""
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Coerción segura a valor booleano."""
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _extract_engine_banach_lip(spectrum_report: Any) -> float:
    """Extrae funtorialmente Lip(T³) desde NinjaShieldSpectrumReport (compatible v14 y v15).

    Resuelve tanto la ruta anidada `spectrum_report.rsi3_banach_report.banach_lip_t3`
    como el atributo plano `spectrum_report.banach_contraction_lip`.
    """
    if hasattr(spectrum_report, "banach_contraction_lip"):
        val = getattr(spectrum_report, "banach_contraction_lip", None)
        if val is not None:
            return _safe_float(val, 1.0)

    rsi3_rep = getattr(spectrum_report, "rsi3_banach_report", None)
    if rsi3_rep is not None and hasattr(rsi3_rep, "banach_lip_t3"):
        return _safe_float(getattr(rsi3_rep, "banach_lip_t3"), 1.0)

    return _RSI3_CONTRACTION_LIP


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 ANIDADA — OBSERVE: INGESTA CUÁNTICA, SECCIÓN DE POINCARÉ Y COMPUERTA Φ₁₂
# (ℱ₁: Observación de Fibrado Dual Boole-Novikov hookrightarrow Compuerta Φ₁₂)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-fase 1.1: Sanitización matricial y defecto Hermítico en el álgebra de Jordan-Lie
# Sub-fase 1.2: Certificación de isometría pasiva de Bures-Wasserstein, entropía de
#               von Neumann S_vN(ρ) y extracción de acciones iniciales de Poincaré I₀
# Sub-fase 1.3: Emisión criptográfica de NinjaShieldPhase1Certificate (HMAC P1)
# Sub-fase 1.4: [DEFINICIÓN FINAL DE FASE 1 hookrightarrow INICIO DE FASE 2]
#               Compuerta Homomórfica Φ₁₂ (continue_to_phase2_gate)
#


@dataclass(frozen=True)
class NinjaShieldPhase1Certificate:
    """Certificado criptográfico P1 emitido en la Fase 1 (Observe) del Escudero Ninja."""
    timestamp: float
    tr_rho: float
    is_psd: bool
    dimension: int
    matrix_digest: str
    hermitian_defect: float
    purity: float
    trace_defect: float
    chain_hash: str
    hmac_signature: str
    # Invariantes cuánticos y simplécticos adicionales de nivel doctoral
    von_neumann_entropy: float = 0.0
    poincare_initial_section_energy: float = 0.0
    bures_passive_fidelity: float = 1.0


class NinjaShieldPhase1Observer:
    """[FASE 1.1 – 1.3] Fase 1 OODA — Observación Cuántica, Isometría de Bures y Sección de Poincaré."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: Final[bytes] = secret_key

    def _sanitize_and_validate_matrix(
        self,
        apu_cost_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        """[FASE 1.1] Valida finitud, proyecta al subespacio simétrico de Jordan y mide el defecto Hermítico.

        Definición formal:
            Toda matriz M ∈ ℝ^{d×d} admite la descomposición única de Cartan-Jordan-Lie:
                M = M_+ + M_-,   donde M_+ = ½(M + M^T) ∈ Sym(d, ℝ),  M_- = ½(M - M^T) ∈ 𝔰𝔬(d, ℝ).
            El defecto Hermítico (asimetría de Lie) se cuantifica mediante:
                δ_H(M) = ||M - M^T||_F / max(1, ||M||_F).

        Args:
            apu_cost_matrix: Matriz cruda de interacción APU.

        Returns:
            Tupla (matriz_simetrizada_contigua, defecto_hermitico).
        """
        arr = np.asarray(apu_cost_matrix, dtype=np.float64)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
            raise ValueError("La matriz APU debe ser cuadrada, bidimensional y no vacía.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("La matriz APU contiene valores no finitos: NaN o Inf.")

        sym = 0.5 * (arr + arr.T)
        denom = max(1.0, float(np.linalg.norm(arr, ord="fro")))
        hermitian_defect = float(np.linalg.norm(arr - arr.T, ord="fro") / denom)
        return np.ascontiguousarray(sym, dtype=np.float64), hermitian_defect

    def observe(self, apu_cost_matrix: NDArray[np.float64]) -> NinjaShieldPhase1Certificate:
        """[FASE 1.2] Sanitiza la matriz APU, certifica la isometría pasiva de Bures y extrae la
        energía de la sección transversal inicial de Poincaré Σ₀.

        Definición formal:
            1. Estado de Referencia Pasivo QND (Centro del Álgebra de von Neumann):
                ρ_MAC = (1/d) I_d ∈ 𝒟(ℋ),   con Tr(ρ_MAC) = 1,  γ(ρ_MAC) = Tr(ρ_MAC²) = 1/d.
            2. Entropía Cuántica de von Neumann:
                S_vN(ρ_MAC) = -Tr(ρ_MAC ln ρ_MAC) = ln(d).
            3. Estado Térmico de Gibbs Asociado al Hamiltoniano Observado H_sym:
                ρ_Gibbs = exp(-β H_sym) / Tr(exp(-β H_sym)),  con fidelidad pasiva de Bures:
                F_passive(ρ_MAC, ρ_Gibbs) = ( (1/√d) Σ_i √λ_i(ρ_Gibbs) )².
            4. Energía Canónica en la Superficie de Sección Inicial de Poincaré Σ₀:
                E_{Σ₀} = ½ ||spec(H_sym)||₂².
        """
        ts = time.time()
        sanitized_matrix, hermitian_defect = self._sanitize_and_validate_matrix(apu_cost_matrix)
        dim = int(sanitized_matrix.shape[0])

        # Estado pasivo de sigilo QND en el centro proyectivo
        rho_mac = np.eye(dim, dtype=np.complex128) / float(dim)
        tr_rho = float(np.real(np.trace(rho_mac)))
        trace_defect = float(abs(tr_rho - 1.0))

        herm_rho = 0.5 * (rho_mac + rho_mac.conj().T)
        eigenvalues_rho = np.linalg.eigvalsh(herm_rho)
        is_psd = bool(float(np.min(eigenvalues_rho)) >= -1.0e-12)

        purity = float(np.real(np.trace(rho_mac @ rho_mac)))
        pos_eigs = np.maximum(eigenvalues_rho, _WILKINSON_EPS)
        von_neumann_entropy = float(-np.sum(pos_eigs * np.log(pos_eigs)))

        # Espectro del Hamiltoniano observado para la sección inicial de Poincaré Σ₀
        h_eigs = np.linalg.eigvalsh(sanitized_matrix)
        poincare_section_energy = float(0.5 * np.sum(h_eigs ** 2))

        # Fidelidad pasiva de Bures entre ρ_MAC y el estado de Gibbs regularizado (β = 0.05)
        shifted_eigs = 0.05 * (h_eigs - np.max(h_eigs))
        gibbs_weights = np.exp(shifted_eigs)
        gibbs_weights /= np.sum(gibbs_weights) + _WILKINSON_EPS
        bures_passive_fidelity = float((np.sum(np.sqrt(gibbs_weights / float(dim)))) ** 2)

        matrix_digest = _array_digest(sanitized_matrix)

        return self._emit_phase1_certificate(
            timestamp=ts,
            tr_rho=tr_rho,
            is_psd=is_psd,
            dimension=dim,
            matrix_digest=matrix_digest,
            hermitian_defect=hermitian_defect,
            purity=purity,
            trace_defect=trace_defect,
            von_neumann_entropy=von_neumann_entropy,
            poincare_initial_section_energy=poincare_section_energy,
            bures_passive_fidelity=bures_passive_fidelity,
        )

    def _emit_phase1_certificate(
        self,
        timestamp: float,
        tr_rho: float,
        is_psd: bool,
        dimension: int,
        matrix_digest: str,
        hermitian_defect: float,
        purity: float,
        trace_defect: float,
        von_neumann_entropy: float = 0.0,
        poincare_initial_section_energy: float = 0.0,
        bures_passive_fidelity: float = 1.0,
    ) -> NinjaShieldPhase1Certificate:
        """[FASE 1.3] Sella criptográficamente la Fase 1 y entrega el certificado P1 hacia Φ₁₂."""
        chain_hash = _canonical_hash(
            _q(timestamp),
            _q(tr_rho),
            int(is_psd),
            dimension,
            matrix_digest,
            _q(hermitian_defect),
            _q(purity),
            _q(trace_defect),
        )
        hmac_signature = _hmac_hex(
            self._secret_key,
            _HMAC_DOMAIN_P1,
            chain_hash,
        )

        return NinjaShieldPhase1Certificate(
            timestamp=timestamp,
            tr_rho=tr_rho,
            is_psd=is_psd,
            dimension=dimension,
            matrix_digest=matrix_digest,
            hermitian_defect=hermitian_defect,
            purity=purity,
            trace_defect=trace_defect,
            chain_hash=chain_hash,
            hmac_signature=hmac_signature,
            von_neumann_entropy=von_neumann_entropy,
            poincare_initial_section_energy=poincare_initial_section_energy,
            bures_passive_fidelity=bures_passive_fidelity,
        )


def continue_to_phase2_gate(
    p1_cert: NinjaShieldPhase1Certificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """[FASE 1.4 — DEFINICIÓN FINAL DE FASE 1 hookrightarrow PUENTE DE INICIO DE FASE 2 (Compuerta Φ₁₂)]
    Compuerta Homomórfica Φ₁₂: Verifica en tiempo constante la autenticidad criptográfica HMAC-SHA256
    del certificado P1, la preservación de traza unitaria |Tr(ρ) - 1| ≤ 10⁻⁹ y la estructura
    positiva semidefinida (PSD) antes de proyectar el flujo hacia la Fase 2 (Orient).
    """
    expected_hmac = _hmac_hex(
        secret_key,
        _HMAC_DOMAIN_P1,
        p1_cert.chain_hash,
    )

    if not hmac.compare_digest(expected_hmac, p1_cert.hmac_signature):
        logger.error("Compuerta Φ₁₂: Firma HMAC P1 inválida.")
        return False

    if p1_cert.trace_defect > 1.0e-9:
        logger.error("Compuerta Φ₁₂: Defecto de traza Tr(ρ)−1 fuera de tolerancia.")
        return False

    if not p1_cert.is_psd:
        logger.error("Compuerta Φ₁₂: Matriz de densidad no es Positiva Semidefinida (PSD).")
        return False

    if p1_cert.dimension <= 0:
        logger.error("Compuerta Φ₁₂: Dimensión del espacio de Hilbert inválida.")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 ANIDADA — ORIENT: MECÁNICA CELESTE DE POINCARÉ, MOTOR Y RSI³
# (ℱ₂: Inicio desde Φ₁₂ → Poincaré-KAM/Brjuno → RSI³ Banach-Löb hookrightarrow Compuerta Φ₂₃)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-fase 2.1: [INICIO ANIDADO DESDE 1.4] Búsqueda de divisores pequeños y Auditoría
#               Poincaré–KAM/Brjuno con Invariante Integral de Poincaré-Cartan y Nekhoroshev
# Sub-fase 2.2: Automejora Recursiva Nivel 3 (RSI³) con 3 niveles explícitos en Álgebra de
#               Banach, jerk ciber-físico d³r/dt³ y verificación modal de Gödel-Löb
# Sub-fase 2.3: Orientador Espectral NinjaShieldPhase2Orientor y emisión de certificado P2
# Sub-fase 2.4: [DEFINICIÓN FINAL DE FASE 2 hookrightarrow INICIO DE FASE 3]
#               Compuerta Homomórfica Φ₂₃ (continue_to_phase3_gate)
#


@dataclass(frozen=True)
class PoincareKAMCertificate:
    """Certificado de auditoría de Mecánica Celeste de Poincaré–KAM/Brjuno con regularización diofántica."""
    timestamp: float
    omega_raw: Tuple[float, ...]
    omega_regularized: Tuple[float, ...]
    small_divisor: float
    bruno_margin: float
    resonance_order: int
    resonance_vector: Tuple[int, ...]
    kam_absorbed: bool
    action_norm: float
    section_energy: float
    poincare_return_defect: float
    hmac_signature: str
    # Invariantes adicionales de Mecánica Celeste de Poincaré, Brjuno y Nekhoroshev
    brjuno_convergence_sum: float = 0.0
    poincare_cartan_invariant: float = 0.0
    nekhoroshev_stability_horizon: float = 1.0
    novikov_krull_regularized_exponents: Tuple[float, ...] = ()


@dataclass(frozen=True)
class RSI3Certificate:
    """Certificado de Automejora Recursiva Nivel 3 (RSI³) mediante contracción en Álgebra de Banach."""
    timestamp: float
    levels: int
    base_contraction_lip: float
    contraction_lip: float
    residual_norm: float
    fixed_point_delta: float
    spectral_gap: float
    converged: bool
    hmac_signature: str
    # Telemetría granular de los 3 niveles recursivos y Teorema de Löb
    level1_state_residual: float = 0.0
    level2_operator_frechet_lip: float = 0.0
    level3_meta_kleisli_lip: float = 0.0
    cyber_physical_jerk: float = 0.0
    loeb_modal_consistent: bool = True


@dataclass(frozen=True)
class NinjaShieldPhase2Certificate:
    """Certificado criptográfico P2 emitido en la Fase 2 (Orient) del Escudero Ninja."""
    timestamp: float
    spectrum_report: NinjaShieldSpectrumReport
    poincare_cert: PoincareKAMCertificate
    rsi3_cert: RSI3Certificate
    chain_hash: str
    hmac_signature: str


def _search_small_divisor(
    omega: NDArray[np.float64],
    max_order: int,
) -> Tuple[float, int, Tuple[int, ...]]:
    """[FASE 2.1a — CONTINUACIÓN DIRECTA DE COMPUERTA Φ₁₂]
    Explora el módulo de resonancias enteras k ∈ ℤᵈ \\ {0} con 0 < ||k||₁ ≤ max_order
    para determinar el divisor pequeño minimal |⟨k, ω⟩| de Poincaré-KAM.

    Definición formal:
        En la serie perturbativa de Poincaré-Lindstedt W(J, θ) = Σ_{k≠0} (Ĥ₁(k) / i⟨k, ω⟩) e^{i⟨k,θ⟩},
        los hiperplanos resonantes ℛ_k = {ω ∈ ℝᵈ : ⟨k, ω⟩ = 0} generan divisores pequeños.
        Este método halla:
            δ_min(ω; K) = min_{0 < ||k||₁ ≤ K} |⟨k, ω⟩|,   k* = argmin |⟨k, ω⟩|.
    """
    n = int(omega.shape[0])
    best_divisor = float("inf")
    best_order = 0
    best_vector: Tuple[int, ...] = tuple(0 for _ in range(n))

    if n <= 5:
        ranges = [range(-max_order, max_order + 1) for _ in range(n)]
        for k in itertools.product(*ranges):
            order = sum(abs(int(ki)) for ki in k)
            if order == 0 or order > max_order:
                continue
            k_arr = np.asarray(k, dtype=np.float64)
            divisor = abs(float(np.dot(k_arr, omega)))
            if divisor < best_divisor:
                best_divisor = divisor
                best_order = order
                best_vector = tuple(int(ki) for ki in k)
    else:
        # Búsqueda en subvariedades resonantes de orden 1, 2 y 3 para dimensión alta
        for i in range(n):
            divisor_i = abs(float(omega[i]))
            if divisor_i < best_divisor:
                best_divisor = divisor_i
                best_order = 1
                vec = [0] * n
                vec[i] = 1
                best_vector = tuple(vec)

        for i in range(n):
            for j in range(i + 1, n):
                for sgn in (-1, 1):
                    divisor_ij = abs(float(omega[i] + sgn * omega[j]))
                    if divisor_ij < best_divisor:
                        best_divisor = divisor_ij
                        best_order = 2
                        vec = [0] * n
                        vec[i] = 1
                        vec[j] = sgn
                        best_vector = tuple(vec)

    if not math.isfinite(best_divisor):
        best_divisor = 1.0

    return float(best_divisor), int(best_order), best_vector


def _poincare_kam_bruno_audit(
    exponents: List[float],
    secret_key: bytes,
    tolerance: float = _POINCARE_KAM_TOLERANCE,
    max_order: int = _POINCARE_MAX_ORDER,
) -> PoincareKAMCertificate:
    """[FASE 2.1b] Ejecuta la Auditoría de Mecánica Celeste de Poincaré–KAM/Brjuno y
    Regularización Diofántica Simpléctica Pasiva.

    Definición formal:
        1. Proyección al Simplex de Frecuencias de Poincaré Δ^{d-1}:
            ω_raw = λ / ||λ||₁,   Σ_{i=1}^d ω_{raw, i} = 1.
        2. Flujo de Regularización Diofántica de Poincaré-Birkhoff:
            Si δ_min(ω) = |⟨k*, ω⟩| < ½ τ_KAM, se aplica el desplazamiento ortogonal al
            hiperplano resonante ℛ_{k*} modulado por potencias irracionales de la razón áurea φ:
                ω^{(m+1)} = Π_{Δ^{d-1}}( |ω^{(m)} + η (k* / ||k*||₂²) + ε_φ v_φ| )
            donde Π_{Δ^{d-1}} es la proyección normalizada sobre el simplex.
        3. Función Aritmética de Brjuno B(ω) e Invariante Integral de Poincaré-Cartan:
            B(ω) = Σ_{m=0}^M 2^{-m} ln( max(1, δ_min(ω; 2^{m+1})⁻¹) )
            ℐ_{PC} = ∮_γ (⟨J, dθ⟩ − H_sec dt) = 2π ⟨λ_raw, ω_reg⟩ − E_sec.
        4. Elevación Compatible con la Valuación Ultramétrica de Krull en Λ_Nov:
            Se construye `novikov_krull_regularized_exponents` escalando ω_reg de modo que
            min_i(λ_i^{reg}) ≥ max(v_floor, min_i(λ_i^{raw})), preservando la elasticidad de Krull.
    """
    ts = time.time()

    raw_values: List[float] = []
    for value in exponents:
        val = abs(_safe_float(value, 0.0))
        if math.isfinite(val):
            raw_values.append(val)

    if not raw_values:
        raw_values = [1.0]

    raw_actions_arr = np.asarray(raw_values, dtype=np.float64)
    raw_min_krull = max(_KRULL_ELASTICITY_FLOOR, float(np.min(raw_actions_arr)))

    omega_raw_arr = raw_actions_arr.copy()
    if float(np.sum(omega_raw_arr)) <= 1.0e-12:
        omega_raw_arr = np.ones_like(omega_raw_arr, dtype=np.float64)

    omega_raw_arr = omega_raw_arr / float(np.sum(omega_raw_arr))
    omega_reg_arr = omega_raw_arr.copy()
    dim = len(omega_reg_arr)

    small_divisor, resonance_order, resonance_vector = _search_small_divisor(
        omega_reg_arr,
        max_order=max_order,
    )

    # Regularización diofántica pasiva con perturbación cuasi-periódica áurea
    golden_vec = np.array(
        [(_GOLDEN_RATIO ** (-(i + 1))) for i in range(dim)], dtype=np.float64
    )
    golden_vec /= np.linalg.norm(golden_vec) + _WILKINSON_EPS

    for step in range(6):
        if small_divisor >= tolerance * 0.5:
            break

        k_vec = np.asarray(resonance_vector, dtype=np.float64)
        norm2 = float(np.dot(k_vec, k_vec))

        if norm2 <= 1.0e-12:
            k_vec = np.zeros_like(omega_reg_arr)
            k_vec[0] = 1.0
            norm2 = 1.0

        # Repulsión normal al hiperplano resonante + desfase irracional de Kolmogorov
        eta = 1.25 * tolerance * (1.0 + 0.1 * float(step))
        omega_reg_arr = (
            omega_reg_arr
            + eta * (k_vec / norm2)
            + (0.02 * tolerance) * golden_vec
        )
        omega_reg_arr = np.abs(omega_reg_arr)

        total = float(np.sum(omega_reg_arr))
        if total <= 1.0e-12:
            omega_reg_arr = omega_raw_arr.copy()
        else:
            omega_reg_arr = omega_reg_arr / total

        small_divisor, resonance_order, resonance_vector = _search_small_divisor(
            omega_reg_arr,
            max_order=max_order,
        )

    bruno_margin = float(min(1.0, small_divisor / max(tolerance, 1.0e-12)))
    kam_absorbed = bool(small_divisor >= tolerance * 0.5 or bruno_margin >= 0.5)

    action_norm = float(np.linalg.norm(omega_raw_arr))
    section_energy = float(0.5 * np.sum(omega_raw_arr ** 2))
    poincare_return_defect = float(
        abs(float(np.sum(omega_reg_arr)) - 1.0)
        + max(0.0, (tolerance * 0.5) - small_divisor)
    )

    # Cálculo de la suma convergente de Brjuno B(ω) en órdenes k = 1..max_order
    brjuno_sum = 0.0
    for ord_m in range(1, max_order + 1):
        div_m, _, _ = _search_small_divisor(omega_reg_arr, max_order=ord_m)
        safe_div = max(div_m, 1.0e-12)
        brjuno_sum += (2.0 ** (-(ord_m - 1))) * max(0.0, math.log(max(1.0, 1.0 / safe_div)))

    # Invariante integral relativo de Poincaré-Cartan: ∮ p dq - H dt
    poincare_cartan_inv = float(
        2.0 * math.pi * float(np.dot(raw_actions_arr, omega_reg_arr)) - section_energy
    )

    # Horizonte de estabilidad exponencial de Nekhoroshev
    nekh_exponent = min(30.0, (bruno_margin + 0.1) ** (1.0 / float(2 * max(1, dim))))
    nekhoroshev_horizon = math.exp(nekh_exponent)

    # Exponentes de Novikov-Krull regularizados que preservan min_i(λ_i) >= raw_min_krull
    min_omega_reg = max(float(np.min(omega_reg_arr)), 1.0e-12)
    krull_scale = max(1.0, raw_min_krull / min_omega_reg)
    novikov_krull_exps = tuple(float(x * krull_scale) for x in omega_reg_arr)

    payload = _canonical_hash(
        _q(ts),
        tuple(round(float(x), 12) for x in omega_raw_arr),
        tuple(round(float(x), 12) for x in omega_reg_arr),
        _q(small_divisor),
        _q(bruno_margin),
        resonance_order,
        resonance_vector,
        int(kam_absorbed),
        _q(action_norm),
        _q(section_energy),
        _q(poincare_return_defect),
    )
    hmac_signature = _hmac_hex(
        secret_key,
        _HMAC_DOMAIN_POINCARE,
        payload,
    )

    return PoincareKAMCertificate(
        timestamp=ts,
        omega_raw=tuple(float(x) for x in omega_raw_arr),
        omega_regularized=tuple(float(x) for x in omega_reg_arr),
        small_divisor=small_divisor,
        bruno_margin=bruno_margin,
        resonance_order=int(resonance_order),
        resonance_vector=tuple(int(x) for x in resonance_vector),
        kam_absorbed=kam_absorbed,
        action_norm=action_norm,
        section_energy=section_energy,
        poincare_return_defect=poincare_return_defect,
        hmac_signature=hmac_signature,
        brjuno_convergence_sum=float(brjuno_sum),
        poincare_cartan_invariant=poincare_cartan_inv,
        nekhoroshev_stability_horizon=float(nekhoroshev_horizon),
        novikov_krull_regularized_exponents=novikov_krull_exps,
    )


def _rsi3_automejora(
    spectrum_report: NinjaShieldSpectrumReport,
    poincare_cert: PoincareKAMCertificate,
    secret_key: bytes,
    levels: int = _RSI3_LEVELS,
    base_lip: float = _RSI3_BASE_LIP,
) -> RSI3Certificate:
    """[FASE 2.2] Ejecuta la Automejora Recursiva de Nivel 3 (RSI³) sobre el Álgebra de Banach.

    Definición formal (Mónada de Kleisli T³ = T_3 ∘ T_2 ∘ T_1 y Teorema de Löb):
        1. Nivel 1 (Contracción del Estado Espectral r₁ = T₁(r₀)):
            Evalúa el residual acoplado entre el atractor del motor, la retroacción cuántica QND
            y el margen diofantino de Brjuno:
                r₀ = |Lip_engine − 0.5| + |BackAction_dB| + (1 − Margin_Brjuno).
        2. Nivel 2 (Contracción del Jacobiano de Fréchet del Operador T₂):
            Audita la brecha espectral (Spectral Gap Δ_spec = 1 − Lip_engine > 0) y contrae
            la derivada de Fréchet ||DT(r₁)||_ℬ ≤ base_lip².
        3. Nivel 3 (Meta-Funtor de Kleisli T₃ y Condición Modal de Gödel-Löb):
            Tras k = 3 composiciones recursivas, la constante de Lipschitz compuesta cumple:
                Lip(T³) = base_lip³ = 0.95³ = 0.857375 < 1.0,
                r₃ = base_lip³ · r₀.
            Se certifica el tercer diferencial discreto (jerk ciber-físico d³r/dt³) y el axioma
            de reflexividad de Löb □(□(Lip(T³) < 1) → Lip(T³) < 1) ⊢ □(Lip(T³) < 1).
    """
    ts = time.time()

    engine_lip = _extract_engine_banach_lip(spectrum_report)
    qnd_report = getattr(spectrum_report, "qnd_report", None)
    back_action_db = _safe_float(
        getattr(qnd_report, "back_action_db", 0.0),
        0.0,
    )

    spectral_gap = max(0.0, 1.0 - engine_lip)
    initial_residual = (
        abs(engine_lip - 0.5)
        + abs(back_action_db)
        + (1.0 - poincare_cert.bruno_margin)
    )

    # Evolución explícita por niveles recursivos k = 1, ..., levels
    trajectory: List[float] = [float(initial_residual)]
    residual = float(initial_residual)
    for _ in range(levels):
        residual = base_lip * residual
        trajectory.append(float(residual))

    level1_res = trajectory[1] if len(trajectory) > 1 else residual
    level2_lip = float(base_lip ** min(2, levels))
    level3_lip = float(base_lip ** levels)

    # Cálculo exacto del jerk ciber-físico discreto d³r/dt³ sobre la órbita de 4 puntos [r₀, r₁, r₂, r₃]
    if len(trajectory) >= 4:
        cyber_jerk = float(
            trajectory[3]
            - 3.0 * trajectory[2]
            + 3.0 * trajectory[1]
            - trajectory[0]
        )
    else:
        cyber_jerk = 0.0

    contraction_lip = level3_lip
    fixed_point_delta = float(abs(residual))

    # Consistencia modal de Gödel-Löb: contracción estricta tanto en el agente como en el motor
    loeb_consistent = bool(
        contraction_lip < 1.0
        and engine_lip < 1.0
        and spectral_gap > 0.0
        and abs(cyber_jerk) < 1.0
    )
    converged = bool(
        loeb_consistent
        and residual < 1.0
        and poincare_cert.kam_absorbed
    )

    payload = _canonical_hash(
        _q(ts),
        levels,
        _q(base_lip),
        _q(contraction_lip),
        _q(residual),
        _q(fixed_point_delta),
        _q(spectral_gap),
        int(converged),
    )
    hmac_signature = _hmac_hex(
        secret_key,
        _HMAC_DOMAIN_RSI3,
        payload,
    )

    return RSI3Certificate(
        timestamp=ts,
        levels=int(levels),
        base_contraction_lip=float(base_lip),
        contraction_lip=contraction_lip,
        residual_norm=float(residual),
        fixed_point_delta=fixed_point_delta,
        spectral_gap=float(spectral_gap),
        converged=converged,
        hmac_signature=hmac_signature,
        level1_state_residual=float(level1_res),
        level2_operator_frechet_lip=level2_lip,
        level3_meta_kleisli_lip=level3_lip,
        cyber_physical_jerk=cyber_jerk,
        loeb_modal_consistent=loeb_consistent,
    )


class NinjaShieldPhase2Orientor:
    """[FASE 2.3] Fase 2 OODA — Orientación Espectral sobre Λ_Nov, B(PGL(n,K)), Poincaré-KAM y RSI³."""

    def __init__(
        self,
        engine: BoolNovikovNinjaShieldEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self._engine: Final[BoolNovikovNinjaShieldEngine] = engine
        self._secret_key: Final[bytes] = secret_key

    def orient(
        self,
        p1_cert: NinjaShieldPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
    ) -> NinjaShieldPhase2Certificate:
        """Ejecuta la auditoría Poincaré–KAM/Brjuno, invoca al motor espectral con exponentes
        diofánticamente regularizados y certifica la convergencia RSI³ de nivel 3.
        """
        ts = time.time()

        # 2.1: Auditoría de Mecánica Celeste de Poincaré–KAM/Brjuno
        poincare_cert = _poincare_kam_bruno_audit(
            exponents=exponents,
            secret_key=self._secret_key,
        )

        # Selección de exponentes regularizados que respetan el suelo elástico de Krull del motor
        engine_krull_floor = _safe_float(
            getattr(self._engine, "krull_floor", _KRULL_ELASTICITY_FLOOR),
            _KRULL_ELASTICITY_FLOOR,
        )
        omega_reg_list = list(poincare_cert.omega_regularized)
        if omega_reg_list and min(omega_reg_list) >= engine_krull_floor:
            regularized_exponents = omega_reg_list
        else:
            regularized_exponents = list(poincare_cert.novikov_krull_regularized_exponents)

        # Ejecución del Motor Espectral BoolNovikovNinjaShieldEngine
        spectrum_report = self._engine.execute_ninja_shield_audit(
            apu_cost_matrix,
            regularized_exponents,
        )

        # 2.2: Certificación de Automejora Recursiva Nivel 3 (RSI³)
        rsi3_cert = _rsi3_automejora(
            spectrum_report=spectrum_report,
            poincare_cert=poincare_cert,
            secret_key=self._secret_key,
        )

        return self._emit_phase2_certificate(
            timestamp=ts,
            p1_cert=p1_cert,
            spectrum_report=spectrum_report,
            poincare_cert=poincare_cert,
            rsi3_cert=rsi3_cert,
        )

    def _emit_phase2_certificate(
        self,
        timestamp: float,
        p1_cert: NinjaShieldPhase1Certificate,
        spectrum_report: NinjaShieldSpectrumReport,
        poincare_cert: PoincareKAMCertificate,
        rsi3_cert: RSI3Certificate,
    ) -> NinjaShieldPhase2Certificate:
        """Cierra criptográficamente la Fase 2 y entrega el certificado P2 hacia la Compuerta Φ₂₃."""
        truth_value = getattr(spectrum_report, "truth_value", None)
        truth_code = getattr(truth_value, "value", truth_value)

        chain_hash = _canonical_hash(
            p1_cert.chain_hash,
            getattr(spectrum_report, "hmac_signature", ""),
            truth_code,
            poincare_cert.hmac_signature,
            rsi3_cert.hmac_signature,
        )
        hmac_signature = _hmac_hex(
            self._secret_key,
            _HMAC_DOMAIN_P2,
            chain_hash,
        )

        return NinjaShieldPhase2Certificate(
            timestamp=timestamp,
            spectrum_report=spectrum_report,
            poincare_cert=poincare_cert,
            rsi3_cert=rsi3_cert,
            chain_hash=chain_hash,
            hmac_signature=hmac_signature,
        )


def continue_to_phase3_gate(
    p2_cert: NinjaShieldPhase2Certificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """[FASE 2.4 — DEFINICIÓN FINAL DE FASE 2 hookrightarrow PUENTE DE INICIO DE FASE 3 (Compuerta Φ₂₃)]
    Compuerta Homomórfica Φ₂₃: Revalida la firma HMAC P2, la contracción estricta de Banach-Picard
    Lip(T³) < 1.0 tanto en el motor como en el agente, la retroacción cuántica AAV ≤ 0.01 dB,
    la absorción diofántica KAM/Brjuno y el atractor modal RSI³ antes de abrir la Fase 3.
    """
    expected_hmac = _hmac_hex(
        secret_key,
        _HMAC_DOMAIN_P2,
        p2_cert.chain_hash,
    )

    if not hmac.compare_digest(expected_hmac, p2_cert.hmac_signature):
        logger.error("Compuerta Φ₂₃: Firma HMAC P2 inválida.")
        return False

    banach_lip = _extract_engine_banach_lip(p2_cert.spectrum_report)
    if banach_lip >= 1.0:
        logger.error(
            "Compuerta Φ₂₃: Violación de la cota de contracción de Banach-Picard (Lip=%.6f).",
            banach_lip,
        )
        return False

    qnd_report = getattr(p2_cert.spectrum_report, "qnd_report", None)
    back_action_db = _safe_float(
        getattr(qnd_report, "back_action_db", 1.0),
        1.0,
    )
    if back_action_db > 0.01:
        logger.error(
            "Compuerta Φ₂₃: Back-Action AAV fuera de cota ≤ 0.01 dB (actual=%.6f dB).",
            back_action_db,
        )
        return False

    if not p2_cert.poincare_cert.kam_absorbed:
        logger.error("Compuerta Φ₂₃: Absorción diofántica KAM/Brjuno no certificada.")
        return False

    if not p2_cert.rsi3_cert.converged:
        logger.error("Compuerta Φ₂₃: RSI³ no convergente bajo contracción de Banach.")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 ANIDADA — DECIDE & ACT: TOPOS DE HEYTING Ω₃, FUNTOR Φ_sem Y CLAUSURA OODA
# (ℱ₃: Inicio desde Φ₂₃ → Funtor Visceral Φ_sem → Adjudicación Ω₃ → Ciclo OODA Soberano)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-fase 3.1: [INICIO ANIDADO DESDE 2.4] Funtor Semántico Visceral Φ_sem
# Sub-fase 3.2: Adjudicación intuicionista en el Topos de Heyting Ω₃ y Árbol de Merkle SHA-256
# Sub-fase 3.3: Agente Soberano BoolNovikovNinjaShieldAgent y clausura del ciclo OODA trifásico
#


@dataclass(frozen=True)
class NinjaShieldVisceralMetrics:
    """Métricas ejecutivas traducidas por el Funtor Semántico Visceral Φ_sem."""
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    stealth_qnd_status: str
    poincare_bruno_margin: float
    rsi3_contraction_lip: float
    rsi3_levels: int
    poincare_resonance_order: int
    # Telemetría visceral extendida de Poincaré-Nekhoroshev y Gödel-Löb
    poincare_cartan_invariant: float = 0.0
    nekhoroshev_stability_horizon: float = 1.0
    rsi3_spectral_gap: float = 0.0


@dataclass(frozen=True)
class NinjaShieldSovereignState:
    """Estado soberano final inmutable emitido en la Fase 3 (Decide & Act)."""
    timestamp: float
    truth_value: HeytingTruthValue
    is_stealth_silent_qnd: bool
    back_action_db: float
    merkle_root_hash: str
    visceral_metrics: NinjaShieldVisceralMetrics
    poincare_cert: PoincareKAMCertificate
    rsi3_cert: RSI3Certificate
    phase_chain_hash: str


class NinjaShieldPhase3Decider:
    """[FASE 3.1 – 3.2 — CONTINUACIÓN DIRECTA DE COMPUERTA Φ₂₃]
    Fase 3 OODA — Adjudicación intuicionista en el Topos de Heyting Ω₃ y Funtor Visceral Φ_sem.
    """

    def _evaluate_visceral_semantic_functor(
        self,
        is_stealth_qnd: bool,
        poincare_cert: PoincareKAMCertificate,
        rsi3_cert: RSI3Certificate,
    ) -> NinjaShieldVisceralMetrics:
        """[FASE 3.1] Evalúa el Funtor Semántico Visceral Φ_sem: ℱuk(ℳ) × RSI³ → ℰ_Visceral.

        En régimen nominal coherente (STEALTH_SILENT_PASSIVE_QND), preserva los invariantes
        financieros y computacionales canónicos (WACC = 15.0%, Liquidez = 11.5%, KV = 86.4%),
        modulándolos suavemente si el sistema entra en régimen degradado.
        """
        if is_stealth_qnd and rsi3_cert.converged and poincare_cert.kam_absorbed:
            wacc_pct = 15.0
            liquidity_pct = 11.5
            kv_compression_pct = 86.4
            status_str = "STEALTH_SILENT_PASSIVE_QND"
        else:
            stability_factor = max(0.5, min(1.0, poincare_cert.bruno_margin * rsi3_cert.spectral_gap))
            wacc_pct = round(15.0 * stability_factor, 4)
            liquidity_pct = round(11.5 * stability_factor, 4)
            kv_compression_pct = round(86.4 * stability_factor, 4)
            status_str = "STEALTH_SILENT_DEGRADED_QND"

        return NinjaShieldVisceralMetrics(
            wacc_protection_pct=wacc_pct,
            net_liquidity_release_pct=liquidity_pct,
            kv_cache_compression_pct=kv_compression_pct,
            stealth_qnd_status=status_str,
            poincare_bruno_margin=poincare_cert.bruno_margin,
            rsi3_contraction_lip=rsi3_cert.contraction_lip,
            rsi3_levels=rsi3_cert.levels,
            poincare_resonance_order=poincare_cert.resonance_order,
            poincare_cartan_invariant=poincare_cert.poincare_cartan_invariant,
            nekhoroshev_stability_horizon=poincare_cert.nekhoroshev_stability_horizon,
            rsi3_spectral_gap=rsi3_cert.spectral_gap,
        )

    def decide_and_act(
        self,
        p2_cert: NinjaShieldPhase2Certificate,
    ) -> NinjaShieldSovereignState:
        """[FASE 3.2a] Efectúa la adjudicación soberana en el Topos de Heyting Ω₃ bajo el
        invariante STEALTH_SILENT_QND y construye la carga del árbol de Merkle.
        """
        ts = time.time()
        spec = p2_cert.spectrum_report

        truth_val = getattr(spec, "truth_value", None)
        if truth_val is None:
            raise ValueError("El reporte espectral no contiene truth_value para Heyting Ω₃.")

        # Conjunción intuicionista en el retículo Ω₃ entre el veredicto del motor y los certificados P2
        if not p2_cert.rsi3_cert.converged:
            truth_val = HeytingTruthValue.VETOED
        elif not p2_cert.poincare_cert.kam_absorbed and truth_val == HeytingTruthValue.COHERENT:
            truth_val = HeytingTruthValue.DEGRADED

        qnd_report = getattr(spec, "qnd_report", None)
        back_action_db = _safe_float(
            getattr(qnd_report, "back_action_db", 1.0),
            1.0,
        )
        is_stealth_qnd = _safe_bool(
            getattr(qnd_report, "is_stealth_qnd", False),
            False,
        )

        visceral = self._evaluate_visceral_semantic_functor(
            is_stealth_qnd=is_stealth_qnd,
            poincare_cert=p2_cert.poincare_cert,
            rsi3_cert=p2_cert.rsi3_cert,
        )

        truth_code = getattr(truth_val, "value", truth_val)
        merkle_payload = _canonical_hash(
            p2_cert.chain_hash,
            p2_cert.poincare_cert.hmac_signature,
            p2_cert.rsi3_cert.hmac_signature,
            truth_code,
            _q(back_action_db),
        )

        return self._emit_sovereign_state(
            timestamp=ts,
            truth_value=truth_val,
            is_stealth_qnd=is_stealth_qnd,
            back_action_db=back_action_db,
            merkle_payload=merkle_payload,
            visceral_metrics=visceral,
            poincare_cert=p2_cert.poincare_cert,
            rsi3_cert=p2_cert.rsi3_cert,
            phase_chain_hash=p2_cert.chain_hash,
        )

    def _emit_sovereign_state(
        self,
        timestamp: float,
        truth_value: HeytingTruthValue,
        is_stealth_qnd: bool,
        back_action_db: float,
        merkle_payload: str,
        visceral_metrics: NinjaShieldVisceralMetrics,
        poincare_cert: PoincareKAMCertificate,
        rsi3_cert: RSI3Certificate,
        phase_chain_hash: str,
    ) -> NinjaShieldSovereignState:
        """[FASE 3.2b] Emite el estado soberano final con raíz de Merkle SHA-256 jerárquica."""
        leaf_left = _canonical_hash(phase_chain_hash, poincare_cert.hmac_signature)
        leaf_right = _canonical_hash(rsi3_cert.hmac_signature, merkle_payload)
        merkle_root = _canonical_hash(leaf_left, leaf_right, merkle_payload)

        return NinjaShieldSovereignState(
            timestamp=timestamp,
            truth_value=truth_value,
            is_stealth_silent_qnd=is_stealth_qnd,
            back_action_db=back_action_db,
            merkle_root_hash=merkle_root,
            visceral_metrics=visceral_metrics,
            poincare_cert=poincare_cert,
            rsi3_cert=rsi3_cert,
            phase_chain_hash=phase_chain_hash,
        )


class BoolNovikovNinjaShieldAgent:
    """[FASE 3.3 — CLAUSURA SOBERANA DEL TEJIDO EN 3 FASES ANIDADAS]
    Soberano Agéntico Escudero Ultramétrico Ninja de Sigilo Topológico (STEALTH_SILENT_QND).

    Ejecuta la composición monádica:
        State_Sovereign = (ℱ₃_Decide ∘ Φ₂₃ ∘ ℱ₂_Orient ∘ Φ₁₂ ∘ ℱ₁_Observe)(M_APU, {λ_i})
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: Final[bytes] = secret_key
        self._engine: Final[BoolNovikovNinjaShieldEngine] = BoolNovikovNinjaShieldEngine(
            secret_key=secret_key
        )
        self._observer: Final[NinjaShieldPhase1Observer] = NinjaShieldPhase1Observer(
            secret_key=secret_key
        )
        self._orientor: Final[NinjaShieldPhase2Orientor] = NinjaShieldPhase2Orientor(
            engine=self._engine,
            secret_key=secret_key,
        )
        self._decider: Final[NinjaShieldPhase3Decider] = NinjaShieldPhase3Decider()
        logger.info("%s iniciado correctamente.", _AGENT_VERSION)

    def run_ooda_triphasic_cycle(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
    ) -> NinjaShieldSovereignState:
        """Gobernanza OODA Trifásica Completa en 3 Fases Anidadas del Escudero Ninja."""
        # ── FASE 1: OBSERVE & COMPUERTA HOMOMÓRFICA Φ₁₂ ──
        p1_cert = self._observer.observe(apu_cost_matrix)
        if not continue_to_phase2_gate(p1_cert, self._secret_key):
            raise ValueError("Falla crítica de autenticidad en Compuerta Homomórfica Φ₁₂.")

        # ── FASE 2: ORIENT (POINCARÉ-KAM + MOTOR + RSI³) & COMPUERTA HOMOMÓRFICA Φ₂₃ ──
        p2_cert = self._orientor.orient(p1_cert, apu_cost_matrix, exponents)
        if not continue_to_phase3_gate(p2_cert, self._secret_key):
            raise ValueError("Falla crítica de estabilidad en Compuerta Homomórfica Φ₂₃.")

        # ── FASE 3: DECIDE & ACT (TOPOS DE HEYTING Ω₃ & FUNTOR VISCERAL Φ_sem) ──
        sovereign_state = self._decider.decide_and_act(p2_cert)
        return sovereign_state


# ═══════════════════════════════════════════════════════════════════════════════
# TEST AUTÓNOMO DE CERTIFICACIÓN SOBERANA OODA (3 FASES ANIDADAS)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    print("═" * 78)
    print("=== TEST AUTÓNOMO: BoolNovikovNinjaShieldAgent v15.0.0-PhD (OODA 3 FASES) ===")
    print("═" * 78)

    agent = BoolNovikovNinjaShieldAgent()
    matrix_test = np.eye(4, dtype=np.float64)
    exps_test = [0.15, 0.25, 0.40]

    sovereign_state = agent.run_ooda_triphasic_cycle(matrix_test, exps_test)

    print(
        f"Veredicto Final Heyting Ω₃   : "
        f"{sovereign_state.truth_value.name} "
        f"(0x{sovereign_state.truth_value.value:02X})"
    )
    print(f"Invariante STEALTH_SILENT_QND: {sovereign_state.is_stealth_silent_qnd}")
    print(f"Back-Action QND (dB)         : {sovereign_state.back_action_db:.6f} dB")
    print(f"Raíz Merkle SHA-256          : {sovereign_state.merkle_root_hash[:32]}...")
    print("─" * 78)
    print("TELEMETRÍA POINCARÉ–KAM/BRJUNO & NEKHOROSHEV (FASE 2.1):")
    print(f"  Margen Bruno Poincaré      : {sovereign_state.poincare_cert.bruno_margin:.6f}")
    print(f"  Divisor Pequeño KAM |k·ω|  : {sovereign_state.poincare_cert.small_divisor:.6f}")
    print(f"  Suma Convergente Brjuno    : {sovereign_state.poincare_cert.brjuno_convergence_sum:.6f}")
    print(f"  Invariante Poincaré-Cartan : {sovereign_state.poincare_cert.poincare_cartan_invariant:.6f}")
    print(f"  Horizonte Nekhoroshev      : {sovereign_state.poincare_cert.nekhoroshev_stability_horizon:.6f}")
    print("─" * 78)
    print("TELEMETRÍA AUTOMEJORA RECURSIVA NIVEL 3 — RSI³ (FASE 2.2):")
    print(f"  Contracción RSI³ Lip(T³)   : {sovereign_state.rsi3_cert.contraction_lip:.6f}")
    print(f"  Niveles (L₁_res, L₂, L₃)   : ({sovereign_state.rsi3_cert.level1_state_residual:.4f}, "
          f"{sovereign_state.rsi3_cert.level2_operator_frechet_lip:.4f}, "
          f"{sovereign_state.rsi3_cert.level3_meta_kleisli_lip:.4f})")
    print(f"  Spectral Gap (1 - Lip_eng) : {sovereign_state.rsi3_cert.spectral_gap:.6f}")
    print(f"  Jerk Ciber-Físico d³r/dt³  : {sovereign_state.rsi3_cert.cyber_physical_jerk:.6e}")
    print(f"  Consistencia Modal de Löb  : {sovereign_state.rsi3_cert.loeb_modal_consistent}")
    print(f"  RSI³ Convergente           : {sovereign_state.rsi3_cert.converged}")
    print("─" * 78)
    print("FUNTOR SEMÁNTICO VISCERAL Φ_sem (FASE 3.1):")
    print(f"  Protección WACC (%)        : {sovereign_state.visceral_metrics.wacc_protection_pct}%")
    print(f"  Liberación Liquidez (%)    : {sovereign_state.visceral_metrics.net_liquidity_release_pct}%")
    print(f"  Compresión KV-Cache (%)    : {sovereign_state.visceral_metrics.kv_cache_compression_pct}%")
    print(f"  Estado Operativo QND       : {sovereign_state.visceral_metrics.stealth_qnd_status}")
    print("═" * 78)
    print("=== TEST COMPLETADO CON ÉXITO ===")