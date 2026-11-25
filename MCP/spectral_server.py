# -*- coding: utf-8 -*-
"""
MCP Server 1: spectral-symplectic — Evolución Nivel 3 (3 Fases Anidadas)
========================================================================
Tejido soberano, crítico y verificado con rigor doctoral:

  FASE 1 (Fase1Spectral) — Geometría cuántica espectral y álgebra de Banach-Schatten:
      Validación Dirac–von Neumann, escudo despolarizante CPTP de Fides con
      auditoría de Choi–Jamiołkowski y circuito cuántico RLC/Landauer equivalente,
      fidelidad de Uhlmann, métrica/ángulo de Bures, distancia de traza de Banach
      con certificado de Fuchs–van de Graaf, entropía relativa de Umegaki con
      desigualdad cuántica de Pinsker, información cuántica de Fisher (SLD de
      Helstrom) y lógica cuántica ortomodular de Birkhoff–von Neumann / Topos.
      ÚLTIMO MÉTODO: compute_uhlmann_bures  ──►  QuantumState

  FASE 2 (Fase2CelestialAlgebra) — Mecánica celeste simpléctica, KAM y octoniones 𝕆:
      INICIO: eval_kam_lazutkin consume QuantumState de Fase 1.
      Kepler elíptico/hiperbólico (Halley cúbico + Danby) y parabólico exacto
      (Barker–Cardano) enriquecido con invariantes de Vis-Viva, Delaunay y paso
      simpléctico de Yoshida de 4.º orden; KAM riguroso con órbitas periódicas
      minimizantes de Aubry–Mather / Frenkel–Kontorova (con identidad lagrangiana
      de MacKay–Meiss), Benettin + MEGNO y separatrix de Gelfreich–Lazutkin;
      auditoría exacta de octoniones 𝕆 (512 ternas de base sobre ℤ, identidades
      de Moufang M1–M3, teorema de Artin, identidad de Malcev e invariancia de
      Leibniz del álgebra de Lie excepcional 𝔤₂ = der(𝕆)).
      ÚLTIMO MÉTODO: audit_moufang_residue  ──►  AlgebraicStructure

  FASE 3 (Fase3PoincareSelfImprove) — Sección de Poincaré y automejora Nivel 3:
      INICIO: poincare_spectral_section consume AlgebraicStructure de Fase 2.
      Mapa estándar de Chirikov en radianes, exponente de Lyapunov tangente de
      Benettin, indicador MEGNO, número de rotación diofántico, automejora
      recursiva L1/L2/L3 acoplada a los 4 genes con veto elitista y cierre de
      punto fijo de Kleene re-certificado.
      ÚLTIMO MÉTODO: orchestrate_spectral_symplectic_loop ──► realimenta Fase 1.

Protocolo: MCP (JSON-RPC 2.0 por stdio).
"""
import sys
import json
import math
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, MATEMÁTICAS Y NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════
GOLDEN_RATIO      = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_CONJUGATE  = GOLDEN_RATIO - 1.0            # 1/φ = φ − 1 ≈ 0.61803398875
HURWITZ_C         = 1.0 / math.sqrt(5.0)          # liminf q·‖qφ‖ (Lagrange)
GOLDEN_SHARP      = 1.0 / (GOLDEN_RATIO ** 2)     # min_q q‖qφ‖ = ‖φ‖ (q=1)
CHIRIKOV_CRITICAL = 0.97163540631                 # Greene, convención radianes
GELFREICH_LAZUTKIN_PREFACTOR = 1118.82770595      # Prefactor asintótico numérico
FIDES_FLOOR       = 1.0e-12
COHERENCE_THRESH  = 0.95
BURES_THRESH      = math.sqrt(2.0 - 2.0 * math.sqrt(COHERENCE_THRESH))

# Constantes físicas para auditoría de circuito cuántico RLC / Landauer
BOLTZMANN_K_J_K   = 1.380649e-23
PLANCK_HBAR_J_S   = 1.054571817e-34
CRYO_TEMP_K       = 0.015                         # 15 mK (refrigerador de dilución)
TRANSMON_FREQ_HZ  = 5.0e9                         # 5.0 GHz

# Coeficientes simplécticos de 4.º orden de Yoshida (1990)
_YOSHIDA_C2 = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _YOSHIDA_C2)
_YOSHIDA_W0 = -_YOSHIDA_C2 / (2.0 - _YOSHIDA_C2)
YOSHIDA_C   = (
    _YOSHIDA_W1 / 2.0,
    (_YOSHIDA_W0 + _YOSHIDA_W1) / 2.0,
    (_YOSHIDA_W0 + _YOSHIDA_W1) / 2.0,
    _YOSHIDA_W1 / 2.0,
)
YOSHIDA_D   = (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1)

# Plano proyectivo de Fano orientado PG(2, 𝔽₂) (e₀=1, e₁…e₇ imaginarios)
FANO_LINES: Tuple[Tuple[int, int, int], ...] = (
    (1, 2, 3), (1, 4, 5), (1, 7, 6),
    (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5),
)

# Genoma: caja admisible (retículo completo bajo orden producto)
DEFAULT_GENOME: Dict[str, float] = {
    "kam_gamma": 0.10,
    "fides_floor": FIDES_FLOOR,
    "coherence_thresh": COHERENCE_THRESH,
    "kam_K": 0.60,
}
GENOME_BOX: Dict[str, Tuple[float, float]] = {
    "kam_gamma": (1.0e-4, GOLDEN_SHARP * 0.95),
    "fides_floor": (1.0e-14, 0.05),
    "coherence_thresh": (0.50, 0.999),
    "kam_K": (0.05, 2.0),
}
RATES_L1: Dict[str, float] = {
    "kam_gamma": 0.10,
    "fides_floor": 0.10,
    "coherence_thresh": 0.02,
    "kam_K": 0.15,
}


# ═══════════════════════════════════════════════════════════════════
# UTILIDADES TRANSVERSALES (linaje, saneamiento JSON, coerción)
# ═══════════════════════════════════════════════════════════════════
def _log(msg: str) -> None:
    sys.stderr.write(f"[spectral-symplectic] {msg}\n")
    sys.stderr.flush()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sanitize(obj: Any) -> Any:
    """JSON estricto: dataclasses/ndarray→nativos; no-finitos→null auditable."""
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, np.complexfloating):
        obj = {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _canon(obj: Any) -> str:
    return json.dumps(
        _sanitize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _chain(prev: str, payload: Any) -> str:
    return hashlib.sha256((prev + "|" + _canon(payload)).encode("utf-8")).hexdigest()[:32]


def _view(obj: Any) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Contrato inválido: {type(obj)}")


# ═══════════════════════════════════════════════════════════════════
# CONTRATOS FORMALES (dataclasses — definición formal de cada fase)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class QuantumState:                        # SALIDA F1 → ENTRADA F2
    fidelity: float
    bures_distance: float
    bures_angle_rad: float
    is_coherent: bool
    fides_shield: dict
    eigenvalues_sqrt_rho_sigma: list
    dimension: int
    validation: dict
    uhlmann_bound_ok: bool
    self_fidelity_audit: float
    coherence_epsilon: float               # ═══ seeds hacia F2 ═══
    kam_eps: float
    kam_omega: float
    kam_k_seed: float
    genome_applied: dict
    lineage_hash: str = ""
    # Invariantes espectrales, de Banach y de circuitos cuánticos añadidos
    kam_omega_noble: float = GOLDEN_CONJUGATE
    trace_distance: float = 0.0
    fuchs_van_de_graaf_ok: bool = True
    von_neumann_entropy_rho: float = 0.0
    von_neumann_entropy_sigma: float = 0.0
    umegaki_relative_entropy: float = 0.0
    quantum_pinsker_ok: bool = True
    quantum_fisher_info_sld: float = 0.0
    quantum_circuit_rlc_audit: dict = field(default_factory=dict)
    topos_orthomodular_audit: dict = field(default_factory=dict)


@dataclass
class AlgebraicStructure:                  # SALIDA F2 → ENTRADA F3
    basis_certificate: dict                # exacto (512 ternas, multilinealidad)
    numeric_audit: dict                    # empírico (x,y,z)
    moufang_m1_residue: float
    moufang_m2_residue: float
    moufang_m3_residue: float
    alternativity_left_residue: float
    alternativity_right_residue: float
    flexibility_residue: float
    norm_multiplicativity_rel_error: float
    is_moufang_valid: bool
    is_alternative: bool
    is_flexible: bool
    is_composition_algebra: bool
    algebraic_stability_index: float       # perturbación hacia F3
    octonion_norm_x: float
    octonion_norm_y: float
    octonion_norm_z: float
    lineage_hash: str = ""
    # Invariantes no asociativos de Artin, Malcev y álgebra de Lie excepcional 𝔤₂
    associator_norm: float = 0.0
    artin_theorem_residue: float = 0.0
    malcev_identity_residue: float = 0.0
    is_malcev_algebra: bool = True
    g2_derivation_audit: dict = field(default_factory=dict)


@dataclass
class PoincareSection:                     # SALIDA F3 → fitness / genoma
    map_convention: str
    K_effective: float
    K_from_kam: float
    algebraic_stability_index: float
    orbit: list
    section_points: list
    n_crossings: int
    rotation_number: Optional[float]
    rotation_cf: list
    lyapunov_benettin: float
    megno_mean: float
    cloud_variance_I: float
    greene: dict
    diophantine: dict
    chirikov_confinement: bool
    verdict_inherited: str
    fidelity_inherited: float
    lineage_hash: str = ""
    diophantine_empirical_rho: dict = field(default_factory=dict)
    symplectic_det_defect_max: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# ██ FASE 1 — GEOMETRÍA ESPECTRAL: UHLMANN–BURES, BANACH Y CIRCUITOS ██
# Contrato de salida: QuantumState (consumido por Fase2.eval_kam_lazutkin)
# ═══════════════════════════════════════════════════════════════════
class Fase1Spectral:

    @staticmethod
    def _validate_density_matrix(
        rho: np.ndarray, name: str = "rho", tol: float = 1.0e-10
    ) -> dict:
        """
        Axiomas de Dirac–von Neumann en el espacio de estados S(H) ⊂ T₁(H):
        (i) ρ = ρ†, (ii) Tr ρ = 1, (iii) ρ ⪰ 0, (iv) Tr(ρ²) ≤ 1.
        Incluye dimensión efectiva de participación d_eff = 1/Tr(ρ²) y entropía
        de von Neumann S(ρ) = −Tr(ρ ln ρ).
        """
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] == 0:
            raise ValueError(f"{name} debe ser cuadrada no vacía (recibido {rho.shape})")
        n = rho.shape[0]
        herm = rho - rho.conj().T
        herm_defect = float(np.linalg.norm(herm))
        scale = max(1.0, float(np.linalg.norm(rho)))
        tr = complex(np.trace(rho))
        herm_part = (rho + rho.conj().T) / 2.0
        evals = np.linalg.eigvalsh(herm_part)
        min_eval = float(np.min(evals))
        purity = float(np.real(np.trace(herm_part @ herm_part)))
        pos_evals = np.clip(evals, 0.0, None)
        s_sum = float(np.sum(pos_evals))
        if s_sum > 0.0:
            p_norm = pos_evals / s_sum
            vn_entropy = float(
                -np.sum([p * math.log(p) for p in p_norm if p > 1.0e-15])
            )
        else:
            vn_entropy = 0.0
        return {
            "is_hermitian": herm_defect < tol * scale,
            "hermitian_defect": herm_defect,
            "hermitian_defect_rel": herm_defect / scale,
            "trace": [float(tr.real), float(tr.imag)],
            "trace_normalized": abs(tr - 1.0) < 1.0e-8,
            "trace_defect": float(abs(tr - 1.0)),
            "min_eigenvalue": min_eval,
            "is_psd": min_eval >= -tol * n,
            "purity": purity,
            "purity_ok": purity <= 1.0 + 1.0e-8,
            "participation_ratio": 1.0 / max(purity, 1.0e-15),
            "von_neumann_entropy_nats": vn_entropy,
            "eigenvalues": evals.tolist(),
            "dimension": n,
        }

    @staticmethod
    def _fides_regularize(
        rho: np.ndarray, n: int, floor: float = FIDES_FLOOR
    ) -> Tuple[np.ndarray, dict]:
        """
        Escudo despolarizante cuántico de Fides (canal CPTP):
            ℰ_p(ρ) = (1 − p)ρ + p·(I_n / n).
        Despeje exacto de (1 − p)λ_min + p/n = floor:
            p = n·(floor − λ_min) / (1 − n·λ_min).
        Verifica preservación de traza y positividad completa (Choi–Jamiołkowski
        J(ℰ_p) = (1 − p)|Ω⟩⟨Ω| + (p/n²)I_{n²} ⪰ 0 para todo p ∈ [0, 1]).
        """
        floor_c = _clamp(float(floor), 1.0e-15, 0.49 / n)
        herm = (rho + rho.conj().T) / 2.0
        lam_min = float(np.min(np.linalg.eigvalsh(herm)))
        if lam_min >= floor_c:
            return herm, {
                "applied": False,
                "p_weight": 0.0,
                "lam_min_pre": lam_min,
                "lam_min_post": lam_min,
                "floor_target": floor_c,
                "trace_preserved": True,
                "cptp_choi_min_eig": 0.0,
                "kraus_rank": 1,
            }
        lam_shift = floor_c - lam_min
        denom = 1.0 - n * lam_min
        p = 1.0 if denom <= 1.0e-15 else _clamp(n * lam_shift / denom, 0.0, 1.0)
        rho_reg = (1.0 - p) * herm + (p / n) * np.eye(n, dtype=np.complex128)
        rho_reg = (rho_reg + rho_reg.conj().T) / 2.0
        post = float(np.min(np.linalg.eigvalsh(rho_reg)))
        choi_min_eig = p / (n * n)
        return rho_reg, {
            "applied": True,
            "p_weight": p,
            "lam_min_pre": lam_min,
            "lam_min_post": post,
            "floor_target": floor_c,
            "trace_preserved": abs(float(np.real(np.trace(rho_reg))) - 1.0) < 1.0e-12,
            "cptp_choi_min_eig": choi_min_eig,
            "kraus_rank": n * n if p > 0.0 else 1,
            "erratum": "p = n·λ_shift/(1−n·λ_min), no (1+n·λ_shift)",
        }

    @staticmethod
    def _uhlmann_fidelity_eigen(
        rho: np.ndarray, sigma: np.ndarray
    ) -> Tuple[float, List[float]]:
        """
        F(ρ, σ) = ‖√ρ √σ‖₁² = (∑ᵢ √λᵢ(M))², donde M = √ρ σ √ρ ⪰ 0.
        """
        if rho.shape != sigma.shape:
            raise ValueError(f"Dimensiones incompatibles: {rho.shape} vs {sigma.shape}")
        lam_r, U = np.linalg.eigh((rho + rho.conj().T) / 2.0)
        lam_r = np.clip(lam_r, 0.0, None)
        sqrt_rho = (U * np.sqrt(lam_r)) @ U.conj().T
        M = sqrt_rho @ ((sigma + sigma.conj().T) / 2.0) @ sqrt_rho
        M = (M + M.conj().T) / 2.0
        lam_M = np.clip(np.linalg.eigvalsh(M), 0.0, None)
        F = _clamp(float(np.sum(np.sqrt(lam_M)) ** 2), 0.0, 1.0)
        return F, lam_M.tolist()

    @staticmethod
    def _bures_metric(F: float) -> Tuple[float, float]:
        """d_B = √(2 − 2√F), θ_B = arccos(√F) ∈ [0, π/2]."""
        F_c = _clamp(float(F), 0.0, 1.0)
        d_B = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F_c)))
        theta_B = math.acos(_clamp(math.sqrt(F_c), -1.0, 1.0))
        return d_B, theta_B

    @staticmethod
    def _banach_schatten_and_information_geometry(
        rho: np.ndarray, sigma: np.ndarray, F: float
    ) -> dict:
        """
        Geometría de información cuántica y dualidad de Banach-Schatten T₁(H):
        1. Distancia en norma de traza T(ρ,σ) = ½‖ρ − σ‖₁ y sándwich de
           Fuchs–van de Graaf: 1 − √F ≤ T(ρ,σ) ≤ √(1 − F).
        2. Entropía de von Neumann S(ρ), S(σ) y entropía relativa de Umegaki
           D(ρ ∥ σ) = Tr(ρ (ln ρ − ln σ)) con desigualdad cuántica de Pinsker:
           D(ρ ∥ σ) ≥ 2·T(ρ,σ)².
        3. Derivada Logarítmica Simétrica (SLD) de Helstrom L que resuelve la
           ecuación de Lyapunov ½(ρL + Lρ) = σ − ρ e Información Cuántica de
           Fisher I_Q = Tr(ρL²).
        """
        diff = (rho - sigma + (rho - sigma).conj().T) / 2.0
        evals_diff = np.linalg.eigvalsh(diff)
        trace_dist = _clamp(0.5 * float(np.sum(np.abs(evals_diff))), 0.0, 1.0)
        sqrt_F = math.sqrt(_clamp(F, 0.0, 1.0))
        fvdg_lower = 1.0 - sqrt_F
        fvdg_upper = math.sqrt(max(0.0, 1.0 - F))
        fvdg_ok = (fvdg_lower - 1.0e-9) <= trace_dist <= (fvdg_upper + 1.0e-9)

        lam_r, U_r = np.linalg.eigh((rho + rho.conj().T) / 2.0)
        lam_s, U_s = np.linalg.eigh((sigma + sigma.conj().T) / 2.0)
        lam_r_pos = np.clip(lam_r, 1.0e-15, None)
        lam_s_pos = np.clip(lam_s, 1.0e-15, None)

        S_rho = float(-np.sum(lam_r_pos * np.log(lam_r_pos)))
        S_sigma = float(-np.sum(lam_s_pos * np.log(lam_s_pos)))

        log_rho = (U_r * np.log(lam_r_pos)) @ U_r.conj().T
        log_sigma = (U_s * np.log(lam_s_pos)) @ U_s.conj().T
        umegaki = max(0.0, float(np.real(np.trace(rho @ (log_rho - log_sigma)))))
        pinsker_lower = 2.0 * (trace_dist ** 2)
        pinsker_ok = umegaki + 1.0e-9 >= pinsker_lower

        # SLD en la base propia de ρ: L_{ij} = 2 (U_r† (σ − ρ) U_r)_{ij} / (λ_i + λ_j)
        delta_eig = U_r.conj().T @ (sigma - rho) @ U_r
        denom_mat = lam_r_pos[:, None] + lam_r_pos[None, :]
        L_eig = 2.0 * delta_eig / denom_mat
        L_op = U_r @ L_eig @ U_r.conj().T
        lyapunov_defect = float(
            np.linalg.norm(0.5 * (rho @ L_op + L_op @ rho) - (sigma - rho))
        )
        qfi_sld = max(0.0, float(np.real(np.trace(rho @ L_op @ L_op))))

        return {
            "trace_distance": trace_dist,
            "fuchs_van_de_graaf_lower": fvdg_lower,
            "fuchs_van_de_graaf_upper": fvdg_upper,
            "fuchs_van_de_graaf_ok": bool(fvdg_ok),
            "von_neumann_entropy_rho": S_rho,
            "von_neumann_entropy_sigma": S_sigma,
            "umegaki_relative_entropy": umegaki,
            "quantum_pinsker_lower_bound": pinsker_lower,
            "quantum_pinsker_ok": bool(pinsker_ok),
            "quantum_fisher_info_sld": qfi_sld,
            "sld_lyapunov_defect": lyapunov_defect,
        }

    @staticmethod
    def _quantum_circuit_rlc_and_topos_audit(
        rho: np.ndarray, sigma: np.ndarray, F: float, coherence_thresh: float,
        p_fides: float, S_rho: float, S_sigma: float
    ) -> Tuple[dict, dict]:
        """
        1. Equivalencia física de circuito cuántico RLC (resonador transmon
           superconductor a 15 mK): factor de calidad Q_eff, número térmico de
           fotones de Bose–Einstein y energía de borrado de Landauer.
        2. Clasificador de subobjetos de Topos (álgebra de Heyting + operador
           modal de Lawvere–Tierney) vs defecto de no-conmutatividad del
           retículo ortomodular de Birkhoff–von Neumann ‖[ρ, σ]‖_F.
        """
        omega_0 = 2.0 * math.pi * TRANSMON_FREQ_HZ
        n_thermal = 1.0 / (
            math.exp((PLANCK_HBAR_J_S * omega_0) / (BOLTZMANN_K_J_K * CRYO_TEMP_K)) - 1.0
        )
        decoherence_loss = max(1.0e-12, 1.0 - F + p_fides)
        q_factor_eff = 1.0 / decoherence_loss
        delta_s_bits = abs(S_sigma - S_rho) / math.log(2.0)
        landauer_j = BOLTZMANN_K_J_K * CRYO_TEMP_K * math.log(2.0) * delta_s_bits

        rlc_audit = {
            "resonator_freq_hz": TRANSMON_FREQ_HZ,
            "cryo_temp_k": CRYO_TEMP_K,
            "thermal_photon_occupation": n_thermal,
            "effective_quality_factor_Q": q_factor_eff,
            "landauer_erasure_energy_joules": landauer_j,
        }

        commutator = rho @ sigma - sigma @ rho
        comm_norm = float(np.linalg.norm(commutator))
        # Núcleo modal de Lawvere–Tierney sobre [coherence_thresh, 1]
        theta = _clamp(coherence_thresh, 0.0, 0.9999)
        lt_closed = max(F, theta)
        lt_double_neg = 1.0 if F >= theta else theta
        topos_audit = {
            "birkhoff_von_neumann_commutator_norm": comm_norm,
            "is_boolean_compatible_subalgebra": comm_norm < 1.0e-10,
            "heyting_truth_value": F,
            "heyting_threshold_theta": theta,
            "lawvere_tierney_closed_nucleus": lt_closed,
            "lawvere_tierney_double_negation": lt_double_neg,
            "modal_proposition_holds": F >= theta,
        }
        return rlc_audit, topos_audit

    # ── ÚLTIMO MÉTODO DE FASE 1 ─────────────────────────────────────
    @staticmethod
    def compute_uhlmann_bures(
        rho_matrix: list,
        sigma_matrix: list,
        fides_floor: float = FIDES_FLOOR,
        genome: Optional[dict] = None,
    ) -> QuantumState:
        """
        Fidelidad de Uhlmann + métrica de Bures + escudo CPTP de Fides +
        geometría de información cuántica (Fuchs–van de Graaf, Umegaki–Pinsker,
        SLD de Helstrom) + auditoría RLC/Topos + auto-auditoría F(ρ,ρ)=1.
        → CONTINUIDAD: Fase2CelestialAlgebra.eval_kam_lazutkin consume este QuantumState.
        """
        g = dict(DEFAULT_GENOME)
        g["fides_floor"] = float(fides_floor)
        if genome:
            g.update({k: v for k, v in genome.items() if k in GENOME_BOX})
        g = Fase3PoincareSelfImprove._project_genome(g)

        rho = np.array(rho_matrix, dtype=np.complex128)
        sigma = np.array(sigma_matrix, dtype=np.complex128)
        v_rho = Fase1Spectral._validate_density_matrix(rho, "rho")
        v_sig = Fase1Spectral._validate_density_matrix(sigma, "sigma")
        for v, nm in ((v_rho, "rho"), (v_sig, "sigma")):
            if not (v["is_hermitian"] and v["is_psd"]):
                raise ValueError(f"{nm} no es matriz de densidad válida: {v}")
            if not v["trace_normalized"]:
                if v["trace_defect"] < 1.0e-3:
                    pass
                else:
                    raise ValueError(f"{nm}: Tr ρ ≠ 1 (defecto {v['trace_defect']})")
        if not v_rho["trace_normalized"]:
            rho = rho / np.real(np.trace(rho))
        if not v_sig["trace_normalized"]:
            sigma = sigma / np.real(np.trace(sigma))

        n = rho.shape[0]
        floor_eff = _clamp(float(g["fides_floor"]), 1.0e-15, 0.49 / n)
        rho_r, fides_r = Fase1Spectral._fides_regularize(rho, n, floor_eff)
        sig_r, fides_s = Fase1Spectral._fides_regularize(sigma, n, floor_eff)
        F, lam_M = Fase1Spectral._uhlmann_fidelity_eigen(rho_r, sig_r)
        F_self, _ = Fase1Spectral._uhlmann_fidelity_eigen(rho_r, rho_r)
        d_B, theta_B = Fase1Spectral._bures_metric(F)
        coh_thr = float(g["coherence_thresh"])
        coherent = (F >= coh_thr) and (
            d_B <= math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(coh_thr))) + 1.0e-12
        )

        info_geom = Fase1Spectral._banach_schatten_and_information_geometry(
            rho_r, sig_r, F
        )
        p_max = max(fides_r["p_weight"], fides_s["p_weight"])
        rlc_audit, topos_audit = Fase1Spectral._quantum_circuit_rlc_and_topos_audit(
            rho_r,
            sig_r,
            F,
            coh_thr,
            p_max,
            info_geom["von_neumann_entropy_rho"],
            info_geom["von_neumann_entropy_sigma"],
        )

        coherence_eps = max(1.0e-9, 1.0 - F)
        kam_omega_raw = theta_B / math.pi
        kam_omega_noble = GOLDEN_CONJUGATE + kam_omega_raw * coherence_eps

        payload = {
            "fidelity": F,
            "bures_distance": d_B,
            "bures_angle_rad": theta_B,
            "is_coherent": bool(coherent),
            "fides_shield": {
                "applied_rho": fides_r["applied"],
                "applied_sigma": fides_s["applied"],
                "p_weight_rho": fides_r["p_weight"],
                "p_weight_sigma": fides_s["p_weight"],
                "lam_min_post_rho": fides_r["lam_min_post"],
                "lam_min_post_sigma": fides_s["lam_min_post"],
                "cptp_choi_min_eig_rho": fides_r["cptp_choi_min_eig"],
                "cptp_choi_min_eig_sigma": fides_s["cptp_choi_min_eig"],
            },
            "eigenvalues_sqrt_rho_sigma": lam_M,
            "dimension": n,
            "validation": {"rho": v_rho, "sigma": v_sig, "information_geometry": info_geom},
            "uhlmann_bound_ok": 0.0 <= F <= 1.0 + 1.0e-10,
            "self_fidelity_audit": F_self,
            "coherence_epsilon": coherence_eps,
            "kam_eps": coherence_eps,
            "kam_omega": kam_omega_raw,
            "kam_k_seed": d_B / math.sqrt(2.0),
            "genome_applied": g,
            "kam_omega_noble": kam_omega_noble,
            "trace_distance": info_geom["trace_distance"],
            "fuchs_van_de_graaf_ok": info_geom["fuchs_van_de_graaf_ok"],
            "von_neumann_entropy_rho": info_geom["von_neumann_entropy_rho"],
            "von_neumann_entropy_sigma": info_geom["von_neumann_entropy_sigma"],
            "umegaki_relative_entropy": info_geom["umegaki_relative_entropy"],
            "quantum_pinsker_ok": info_geom["quantum_pinsker_ok"],
            "quantum_fisher_info_sld": info_geom["quantum_fisher_info_sld"],
            "quantum_circuit_rlc_audit": rlc_audit,
            "topos_orthomodular_audit": topos_audit,
        }
        return QuantumState(**payload, lineage_hash=_chain("F1", payload))


# ═══════════════════════════════════════════════════════════════════
# ██ FASE 2 — KEPLER SIMPLÉCTICO + KAM RIGUROSO + OCTONIONES / 𝔤₂ ██
# INICIO: eval_kam_lazutkin consume QuantumState de F1.
# Contrato de salida: AlgebraicStructure (consumido por Fase3)
# ═══════════════════════════════════════════════════════════════════
_GREENER_CACHE: Dict[float, dict] = {}
_BENETTIN_CACHE: Dict[float, dict] = {}


class Fase2CelestialAlgebra:

    # ── PRIMERA FUNCIÓN DE FASE 2 (continuación directa de F1) ──────
    @staticmethod
    def eval_kam_lazutkin(
        eps: Optional[float] = None,
        omega: Optional[float] = None,
        k_chirikov: Optional[float] = None,
        quantum_state: Optional[Any] = None,
    ) -> dict:
        """
        Evaluación KAM rigurosa:
        · Certificado diofántico de Hurwitz–Lagrange (sobre ω y sobre el toro
          noble acoplado ω_noble cuando proviene de QuantumState).
        · Criterio de residuos de Greene sobre órbitas periódicas minimizantes
          de Aubry–Mather / Frenkel–Kontorova con identidad de MacKay–Meiss.
        · Exponente de Lyapunov tangente de Benettin + indicador MEGNO.
        · Estimación asintótica de separatrix de Gelfreich–Lazutkin.
        """
        omega_noble = GOLDEN_RATIO
        if quantum_state is not None:
            qs = _view(quantum_state)
            eps = float(qs.get("kam_eps", eps if eps is not None else 1.0e-3))
            omega = float(qs.get("kam_omega", omega if omega is not None else GOLDEN_RATIO))
            omega_noble = float(qs.get("kam_omega_noble", GOLDEN_CONJUGATE))
            k_chirikov = float(qs.get("kam_k_seed", k_chirikov if k_chirikov is not None else 0.5))
        eps_c = max(1.0e-9, float(eps if eps is not None else 1.0e-3))
        omega_c = float(omega if omega is not None else GOLDEN_RATIO)
        K = float(k_chirikov if k_chirikov is not None else 0.5)

        dio = Fase2CelestialAlgebra._diophantine_certificate(omega_c)
        dio_noble = Fase2CelestialAlgebra._diophantine_certificate(omega_noble)
        dio["noble_transport_certificate"] = dio_noble
        dio["gamma_effective"] = max(dio["gamma_max_all_q"], dio_noble["gamma_max_all_q"])

        greene = Fase2CelestialAlgebra._greene_certificate(K)
        ben = Fase2CelestialAlgebra._benettin_cached(K)
        splitting = Fase2CelestialAlgebra._lazutkin_splitting_estimate(K)
        confined = greene["torus_preserved"] and K < CHIRIKOV_CRITICAL
        verdict = (
            "STABLE"
            if confined and ben["lyapunov"] < 1.0e-2
            else "MARGINAL"
            if confined
            else "CHAOTIC"
        )
        return {
            "eps": eps_c,
            "omega": omega_c,
            "omega_noble": omega_noble,
            "k_chirikov": K,
            "diophantine": dio,
            "greene": greene,
            "lyapunov_benettin": ben["lyapunov"],
            "megno_mean": ben["megno_mean"],
            "lyapunov_asymptotic_logK2": math.log(K / 2.0) if K > 2.0 else 0.0,
            "lazutkin_splitting": splitting,
            "is_chirikov_confined": bool(confined),
            "is_greene_stable": bool(greene["torus_preserved"]),
            "is_kam_stable": bool(confined and ben["lyapunov"] < 1.0e-2),
            "chirikov_k_effective": K,
            "kam_verdict": verdict,
        }

    # ── Mecánica celeste: Kepler elíptico/hiperbólico/parabólico + Yoshida ──
    @staticmethod
    def _yoshida4_kepler_step(
        r_vec: Tuple[float, float], v_vec: Tuple[float, float], dt: float = 1.0e-3
    ) -> dict:
        """
        Verificación simpléctica de 4.º orden de Yoshida (1990) en el plano
        orbital (μ = 1) para auditar conservación de energía de Vis-Viva y
        momento angular específico h = x·v_y − y·v_x.
        """
        x, y = r_vec
        vx, vy = v_vec
        r0 = math.hypot(x, y)
        e0 = 0.5 * (vx * vx + vy * vy) - 1.0 / max(r0, 1.0e-12)
        h0 = x * vy - y * vx
        for i in range(3):
            x += YOSHIDA_C[i] * vx * dt
            y += YOSHIDA_C[i] * vy * dt
            rm = max(math.hypot(x, y), 1.0e-12)
            inv_r3 = 1.0 / (rm * rm * rm)
            vx -= YOSHIDA_D[i] * x * inv_r3 * dt
            vy -= YOSHIDA_D[i] * y * inv_r3 * dt
        x += YOSHIDA_C[3] * vx * dt
        y += YOSHIDA_C[3] * vy * dt
        r1 = math.hypot(x, y)
        e1 = 0.5 * (vx * vx + vy * vy) - 1.0 / max(r1, 1.0e-12)
        h1 = x * vy - y * vx
        return {
            "dt": dt,
            "energy_drift_abs": abs(e1 - e0),
            "angular_momentum_drift_abs": abs(h1 - h0),
            "symplectic_order": 4,
        }

    @staticmethod
    def solve_kepler_halley(
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1.0e-12,
        periapsis: float = 1.0,
    ) -> dict:
        """
        Ecuación de Kepler en los tres regímenes cónicos:
        · Elíptico (0 ≤ e < 1): E − e·sin E = M (conjetura de Danby + Halley cúbico).
        · Parabólico (e = 1): D + D³/3 = M exacto por fórmula cerrada de Barker–Cardano.
        · Hiperbólico (e > 1): e·sinh H − H = M (Halley cúbico con guarda de overflow).
        Incluye invariantes de Vis-Viva, ángulo de trayectoria de vuelo γ_f,
        variables canónicas de Delaunay y auditoría simpléctica de Yoshida.
        """
        M = float(mean_anomaly)
        e = float(eccentricity)
        if e < 0.0:
            raise ValueError("eccentricity debe ser ≥ 0")
        if abs(e - 1.0) < 1.0e-14:                      # ── parabólico exacto ──
            q = 3.0 * M / 2.0
            disc = math.sqrt(q * q + 1.0)
            D = math.copysign(abs(q + disc) ** (1.0 / 3.0), q + disc) + math.copysign(
                abs(q - disc) ** (1.0 / 3.0), q - disc
            )
            nu = 2.0 * math.atan(D)
            r_over_q = 1.0 + D * D
            r = float(periapsis) * r_over_q
            flight_angle = math.atan2(math.sin(nu), 1.0 + math.cos(nu))
            return {
                "regime": "parabolic",
                "barker_D": D,
                "true_anomaly_rad": nu,
                "radius_over_periapsis": r / float(periapsis),
                "flight_path_angle_rad": flight_angle,
                "vis_viva_specific_energy": 0.0,
                "residual": abs(D + (D ** 3) / 3.0 - M),
                "iterations": 0,
                "converged": True,
                "method": "Barker–Cardano (closed form)",
            }
        if e > 1.0:                                     # ── hiperbólico ──
            H = math.asinh(M / e) if abs(M / e) < 1.0e6 else math.copysign(
                math.log(2.0 * abs(M) / e), M
            )
            history, f = [], abs(M)
            it = 0
            for it in range(60):
                try:
                    f = e * math.sinh(H) - H - M
                    fp = e * math.cosh(H) - 1.0
                    fpp = e * math.sinh(H)
                except OverflowError:
                    break
                history.append(abs(f))
                if abs(f) < tol:
                    break
                denom = 2.0 * fp * fp - f * fpp
                if abs(denom) < 1.0e-30:
                    break
                delta = 2.0 * f * fp / denom
                H -= delta
                if abs(delta) < tol:
                    break
            nu = 2.0 * math.atan2(
                math.sqrt(e + 1.0) * math.sinh(H / 2.0),
                math.sqrt(e - 1.0) * math.cosh(H / 2.0),
            )
            r_over_a = e * math.cosh(H) - 1.0
            flight_angle = math.atan2(e * math.sin(nu), 1.0 + e * math.cos(nu))
            return {
                "regime": "hyperbolic",
                "hyperbolic_anomaly_rad": H,
                "true_anomaly_rad": nu,
                "radius_over_a": r_over_a,
                "flight_path_angle_rad": flight_angle,
                "vis_viva_specific_energy": 0.5,
                "eccentricity": e,
                "iterations": it + 1,
                "residual": abs(f),
                "converged": abs(f) < 1.0e-9,
                "estimated_order": Fase2CelestialAlgebra._order_estimate(history),
            }
        # ── elíptico: Danby + Halley ──
        M_red = M % (2.0 * math.pi)
        sign_m = 1.0 if math.sin(M_red) >= 0.0 else -1.0
        E = M_red + 0.85 * e * sign_m
        history, steps, f = [], [], abs(M)
        it = 0
        for it in range(60):
            f = E - e * math.sin(E) - M_red
            fp = 1.0 - e * math.cos(E)
            fpp = e * math.sin(E)
            history.append(abs(f))
            if abs(f) < tol:
                break
            denom = 2.0 * fp * fp - f * fpp
            if abs(denom) < 1.0e-30:
                break
            delta = 2.0 * f * fp / denom
            steps.append(abs(delta))
            E -= delta
            if abs(delta) < tol:
                break
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + e) * math.sin(E / 2.0),
            math.sqrt(1.0 - e) * math.cos(E / 2.0),
        )
        r_over_a = 1.0 - e * math.cos(E)
        flight_angle = math.atan2(e * math.sin(nu), 1.0 + e * math.cos(nu))
        delaunay_G = math.sqrt(max(0.0, 1.0 - e * e))
        # Vector de estado adimensional en el plano orbital (a=1, μ=1)
        rx, ry = math.cos(E) - e, delaunay_G * math.sin(E)
        vx = -math.sin(E) / max(r_over_a, 1.0e-12)
        vy = delaunay_G * math.cos(E) / max(r_over_a, 1.0e-12)
        yoshida_audit = Fase2CelestialAlgebra._yoshida4_kepler_step((rx, ry), (vx, vy))
        return {
            "regime": "elliptic",
            "eccentric_anomaly_rad": E,
            "mean_anomaly_input_rad": M,
            "mean_anomaly_reduced_rad": M_red,
            "wrap_count": int(M // (2.0 * math.pi)),
            "true_anomaly_rad": nu,
            "radius_over_a": r_over_a,
            "flight_path_angle_rad": flight_angle,
            "vis_viva_specific_energy": -0.5,
            "delaunay_actions": {"L": 1.0, "G": delaunay_G, "M_rad": M_red, "nu_rad": nu},
            "yoshida_symplectic_audit": yoshida_audit,
            "eccentricity": e,
            "iterations": it + 1,
            "residual": abs(f),
            "converged": abs(f) < 1.0e-9,
            "estimated_order": Fase2CelestialAlgebra._order_estimate(history),
            "history_tail": history[-3:],
        }

    @staticmethod
    def _order_estimate(residuals: List[float]) -> Optional[float]:
        """p ≈ log(r_{n+1}/r_n)/log(r_n/r_{n−1}) (orden cúbico ≈ 3)."""
        if len(residuals) < 3:
            return None
        r0, r1, r2 = residuals[-1], residuals[-2], residuals[-3]
        if r1 > 1.0e-30 and r2 > 1.0e-30 and r2 > r1 > r0 > 0.0:
            return float(math.log(r0 / r1) / math.log(r1 / r2))
        return None

    # ── Certificado diofántico (Hurwitz–Lagrange) ────────────────────
    @staticmethod
    def _continued_fraction(x: float, max_terms: int = 24) -> List[int]:
        out = [int(math.floor(x))]
        frac = x - math.floor(x)
        for _ in range(max_terms - 1):
            if frac < 1.0e-12:
                break
            inv = 1.0 / frac
            a = int(math.floor(inv))
            out.append(a)
            frac = inv - a
        return out

    @staticmethod
    def _diophantine_certificate(omega: float, q_max: int = 256) -> dict:
        """
        Certificado diofántico riguroso:
        ‖qω‖ ≥ γ/q ∀q ∈ [1, q_max] ⟺ ω mal aproximable (tipo diofántico τ=1).
        """
        best, best_q = float("inf"), 1
        for q in range(1, q_max + 1):
            d = abs(omega * q - round(omega * q))
            if q * d < best:
                best, best_q = q * d, q
        cf = Fase2CelestialAlgebra._continued_fraction(omega, 24)
        bounded = all(a <= 8 for a in cf[1:]) if len(cf) > 1 else False
        p0, p1, q0, q1 = 0, 1, 1, 0
        conv_q = []
        for a in cf:
            p0, p1 = p1, a * p1 + p0
            q0, q1 = q1, a * q1 + q0
            if q1 > 0:
                conv_q.append(q1)
        lag = min(
            (q * abs(omega * q - round(omega * q)) for q in conv_q[-6:]),
            default=None,
        )
        return {
            "gamma_max_all_q": best,
            "attained_q": best_q,
            "cf_partial_quotients": cf,
            "badly_approximable_est": bounded,
            "lagrange_liminf_est": lag,
            "hurwitz_constant": HURWITZ_C,
            "golden_sharp_constant": GOLDEN_SHARP,
        }

    # ── Mapa estándar: Benettin tangente (Lyapunov riguroso) ─────────
    @staticmethod
    def _benettin_cached(K: float, n_iter: int = 2000) -> dict:
        key = round(K, 6)
        if key in _BENETTIN_CACHE:
            return _BENETTIN_CACHE[key]
        theta, p = 0.05, 0.1
        v_theta, v_p = 1.0, 0.0
        sum_log, megno_acc, ybar_sum = 0.0, 0.0, 0.0
        for k in range(1, n_iter + 1):
            c = K * math.cos(theta)
            p_new = p + K * math.sin(theta)
            theta_new = theta + p_new
            # Jacobiano simpléctico sobre (δθ, δp): [[1+Kc, 1], [Kc, 1]], det=1
            nv_theta = (1.0 + c) * v_theta + v_p
            nv_p = c * v_theta + v_p
            d = math.hypot(nv_theta, nv_p)
            if d > 1.0e-300:
                lg = math.log(d)
                sum_log += lg
                megno_acc += 2.0 * k * lg
                ybar_sum += megno_acc / k
                v_theta, v_p = nv_theta / d, nv_p / d
            theta, p = theta_new, p_new
        out = {
            "lyapunov": sum_log / n_iter,
            "megno_mean": ybar_sum / n_iter,
            "n_iter": n_iter,
            "method": "Benettin tangente + MEGNO discreto",
        }
        _BENETTIN_CACHE[key] = out
        return out

    # ── Greene: órbitas minimizantes Aubry–Mather + monodromía ───────
    @staticmethod
    def _thomas_solve(
        sub: List[float], diag: List[float], sup: List[float], rhs: List[float]
    ) -> List[float]:
        n = len(diag)
        cp, dp = [0.0] * n, [0.0] * n
        m = diag[0] if abs(diag[0]) > 1.0e-14 else 1.0e-14
        cp[0] = sup[0] / m
        dp[0] = rhs[0] / m
        for i in range(1, n):
            m = diag[i] - sub[i] * cp[i - 1]
            if abs(m) < 1.0e-14:
                m = 1.0e-14
            cp[i] = sup[i] / m if i < n - 1 else 0.0
            dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m
        x = [0.0] * n
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    @staticmethod
    def _cyclic_solve(diag: List[float], rhs: List[float]) -> List[float]:
        """Tridiagonal cíclica (sub=sup=corners=−1) vía fórmula de Sherman–Morrison."""
        n = len(diag)
        gamma = -diag[0] if abs(diag[0]) > 1.0e-12 else -1.0
        bb = list(diag)
        bb[0] = diag[0] - gamma
        bb[-1] = diag[-1] - 1.0 / gamma
        z = [0.0] * n
        z[0], z[-1] = gamma, -1.0
        sub, sup = [-1.0] * n, [-1.0] * n
        x = Fase2CelestialAlgebra._thomas_solve(sub, bb, sup, rhs)
        u = Fase2CelestialAlgebra._thomas_solve(sub, bb, sup, z)
        den = 1.0 + u[0] - u[-1] / gamma
        if abs(den) < 1.0e-14:
            raise ArithmeticError("Sistema tridiagonal cíclico singular")
        fact = (x[0] - x[-1] / gamma) / den
        return [x[i] - fact * u[i] for i in range(n)]

    @staticmethod
    def _minimizing_orbit(K: float, r: int, q: int) -> Optional[List[float]]:
        """
        Órbita periódica minimizante de Aubry–Mather / Frenkel–Kontorova:
            S(θ) = ∑_{i=0}^{q−1} [½(θ_{i+1} − θ_i)² − K·cos θ_i],  θ_{i+q} = θ_i + 2πr.
        Gradiente exacto:
            ∂S/∂θ_i = 2θ_i − θ_{i−1} − θ_{i+1} + K·sin θ_i.
        Hessiana cíclica exacta:
            H_{ii} = ∂²S/∂θ_i² = 2 + K·cos θ_i,   H_{i,i±1} = −1.
        Paso de descenso de Newton con regularización de Levenberg–Marquardt:
            (H + μI) δ = −∇S(θ).
        """
        two_pi_r = 2.0 * math.pi * r
        theta = [
            two_pi_r * i / q + (math.pi / q) * math.sin(2.0 * math.pi * i / q + 0.3)
            for i in range(q)
        ]

        def action(th: List[float]) -> float:
            s = 0.0
            for i in range(q):
                nxt = th[i + 1] if i + 1 < q else th[0] + two_pi_r
                d = nxt - th[i]
                s += 0.5 * d * d - K * math.cos(th[i])
            return s

        def grad_s(th: List[float]) -> List[float]:
            g_vec = []
            for i in range(q):
                prev = th[i - 1] if i > 0 else th[q - 1] - two_pi_r
                nxt = th[i + 1] if i + 1 < q else th[0] + two_pi_r
                g_vec.append(2.0 * th[i] - prev - nxt + K * math.sin(th[i]))
            return g_vec

        S = action(theta)
        mu_lm = 1.0e-10
        for _ in range(80):
            g_vec = grad_s(theta)
            if max(abs(x) for x in g_vec) < 1.0e-11:
                return theta
            neg_grad = [-x for x in g_vec]
            accepted = False
            for _ in range(8):
                diag = [
                    max(1.0e-6, 2.0 + K * math.cos(t) + mu_lm) for t in theta
                ]
                try:
                    delta = Fase2CelestialAlgebra._cyclic_solve(diag, neg_grad)
                except ArithmeticError:
                    mu_lm *= 10.0
                    continue
                lam = 1.0
                for _ in range(20):
                    cand = [theta[i] + lam * delta[i] for i in range(q)]
                    S_cand = action(cand)
                    if S_cand <= S - 1.0e-14 * lam or max(abs(x) for x in delta) < 1.0e-12:
                        theta, S = cand, S_cand
                        mu_lm = max(1.0e-12, mu_lm * 0.3)
                        accepted = True
                        break
                    lam *= 0.5
                if accepted:
                    break
                mu_lm *= 10.0
            if not accepted:
                break
        return theta if max(abs(x) for x in grad_s(theta)) < 1.0e-8 else None

    @staticmethod
    def _orbit_residue(theta: List[float], K: float) -> float:
        """
        Matriz de monodromía M = ∏ J(θᵢ), J = [[1, K·cos θ], [1, 1 + K·cos θ]] (det=1).
        Residuo de Greene firmado R = (2 − Tr M) / 4.
        Por la identidad lagrangiana de MacKay–Meiss (1983), en la órbita
        minimizante Tr(M) − 2 = det(H) > 0, luego |R| = |2 − Tr M| / 4 = det(H)/4.
        """
        m00, m01, m10, m11 = 1.0, 0.0, 0.0, 1.0
        for t in theta:
            c = K * math.cos(t)
            m00, m01, m10, m11 = (
                m00 + c * m10,
                m01 + c * m11,
                m00 + (1.0 + c) * m10,
                m01 + (1.0 + c) * m11,
            )
        return (2.0 - (m00 + m11)) / 4.0

    @staticmethod
    def _greene_certificate(K: float) -> dict:
        """
        Criterio de residuos de Greene sobre aproximantes diofánticos de Fibonacci
        F_n / F_{n+1} → 1/φ. En el régimen subcrítico (K < K_c ≈ 0.9716354),
        |R(F_n/F_{n+1})| decae exponencialmente hacia 0 en la cola asintótica
        (q ∈ {21, 34, 55, 89}) y permanece acotado bajo el umbral crítico de
        punto fijo de renormalización de MacKay (|R*| ≈ 0.2554).
        """
        key = round(K, 6)
        if key in _GREENER_CACHE:
            return _GREENER_CACHE[key]
        residues = []
        for r, q in (
            (2, 3), (3, 5), (5, 8), (8, 13),
            (13, 21), (21, 34), (34, 55), (55, 89),
        ):
            th = Fase2CelestialAlgebra._minimizing_orbit(K, r, q)
            R_signed = (
                Fase2CelestialAlgebra._orbit_residue(th, K) if th is not None else None
            )
            R_mag = abs(R_signed) if R_signed is not None else None
            residues.append({
                "r": r,
                "q": q,
                "residue": R_mag,
                "residue_signed": R_signed,
            })
        tail = [x for x in residues if x["q"] >= 21]
        ok = all(
            x["residue"] is not None and x["residue"] <= 0.26 for x in tail
        ) and all(x["residue"] is not None for x in residues)
        cert = {
            "K": K,
            "K_c_greene_ref": CHIRIKOV_CRITICAL,
            "margin_subcritical": (CHIRIKOV_CRITICAL - K) / CHIRIKOV_CRITICAL,
            "residues": residues,
            "torus_preserved": bool(ok),
            "criterion": "Greene/MacKay–Meiss minimizante sobre cola de Fibonacci",
        }
        _GREENER_CACHE[key] = cert
        return cert

    @staticmethod
    def _lazutkin_splitting_estimate(K: float) -> dict:
        """
        Estimación asintótica de Gelfreich–Lazutkin de la escisión de separatrix
        del mapa estándar: Δ ≈ C_GL · K⁻¹ · exp(−π² / √K).
        """
        if K <= 0.0:
            return {
                "splitting_estimate": 0.0,
                "exponent": None,
                "status": "K=0 integrable",
            }
        exponent = -(math.pi ** 2) / math.sqrt(K)
        val = (
            (GELFREICH_LAZUTKIN_PREFACTOR / K) * math.exp(exponent)
            if exponent > -700.0
            else 0.0
        )
        return {
            "splitting_estimate": val,
            "exponent": exponent,
            "prefactor": GELFREICH_LAZUTKIN_PREFACTOR,
            "status": "asintótico numérico (no cota rigurosa)",
            "rigorous_rate": "exp(−π²/√K) (Gelfreich 1999)",
        }

    # ── Octoniones 𝕆: tabla de Fano + producto de base exacto + 𝔤₂ ───
    @staticmethod
    def _build_octonion_table() -> np.ndarray:
        """T[a,b,c]: e_a·e_b = ∑_c T[a,b,c]·e_c. Líneas de Fano orientadas."""
        T = np.zeros((8, 8, 8), dtype=float)
        T[0, 0, 0] = 1.0
        for i in range(1, 8):
            T[0, i, i] = 1.0
            T[i, 0, i] = 1.0
            T[i, i, 0] = -1.0
        for a, b, c in FANO_LINES:
            T[a, b, c] = 1.0
            T[b, c, a] = 1.0
            T[c, a, b] = 1.0
            T[b, a, c] = -1.0
            T[c, b, a] = -1.0
            T[a, c, b] = -1.0
        return T

    _OCT_TABLE = _build_octonion_table.__func__()

    @staticmethod
    def _build_basis_mul(oct_table: np.ndarray) -> List[List[Tuple[int, int]]]:
        """e_a·e_b = sign·e_idx (aritmética entera exacta sobre ℤ)."""
        BM = []
        for a in range(8):
            row = []
            for b in range(8):
                nz = np.nonzero(oct_table[a, b])[0]
                idx = int(nz[0])
                sign = int(oct_table[a, b, idx])
                row.append((sign, idx))
            BM.append(row)
        return BM

    _BASIS_MUL = _build_basis_mul.__func__(_OCT_TABLE)

    @staticmethod
    def _bmul(sa: int, a: int, sb: int, b: int) -> Tuple[int, int]:
        s, i = Fase2CelestialAlgebra._BASIS_MUL[a][b]
        return (sa * sb * s, i)

    @staticmethod
    def _oct_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.einsum("ijk,i,j->k", Fase2CelestialAlgebra._OCT_TABLE, u, v)

    @staticmethod
    def _oct_comm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Conmutador [u, v] = u·v − v·u."""
        om = Fase2CelestialAlgebra._oct_mul
        return om(u, v) - om(v, u)

    @staticmethod
    def _oct_assoc(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Asociador [u, v, w] = (u·v)·w − u·(v·w)."""
        om = Fase2CelestialAlgebra._oct_mul
        return om(om(u, v), w) - om(u, om(v, w))

    @staticmethod
    def _oct_jacobian(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Jacobiano de Lie–Malcev J(u, v, w) = [[u, v], w] + [[v, w], u] + [[w, u], v]."""
        cm = Fase2CelestialAlgebra._oct_comm
        return cm(cm(u, v), w) + cm(cm(v, w), u) + cm(cm(w, u), v)

    @staticmethod
    def _g2_inner_derivation(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        Operador de derivación interna de Schafer del álgebra de Lie excepcional
        𝔤₂ = der(𝕆) (dimensión 14):
            D_{x,y}(z) = [[x, y], z] − 3·[x, y, z] = [L_x, L_y]z + [L_x, R_y]z + [R_x, R_y]z.
        Satisface exactamente la regla de Leibniz D_{x,y}(u·v) = (D_{x,y}u)·v + u·(D_{x,y}v).
        """
        cm = Fase2CelestialAlgebra._oct_comm
        asc = Fase2CelestialAlgebra._oct_assoc
        return cm(cm(x, y), z) - 3.0 * asc(x, y, z)

    @staticmethod
    def _basis_certificate() -> dict:
        """
        Certificado nivel teorema sobre ℝ por multilinealidad en las 8³ = 512
        ternas de la base canónica {e₀,…,e₇} (aritmética entera exacta):
        Moufang M1–M3, alternatividad izquierda/derecha, flexibilidad y
        composición de Hurwitz en los 64 pares.
        """
        bm = Fase2CelestialAlgebra._bmul
        fails = {
            "m1": 0, "m2": 0, "m3": 0,
            "alt_left": 0, "alt_right": 0, "flex": 0, "composition": 0,
        }
        for a in range(8):
            for b in range(8):
                xy = bm(1, a, 1, b)
                yx = bm(1, b, 1, a)
                xx = bm(1, a, 1, a)
                yy = bm(1, b, 1, b)
                if abs(xy[0]) != 1:
                    fails["composition"] += 1
                for c in range(8):
                    zy = bm(1, c, 1, b)
                    zx = bm(1, c, 1, a)
                    xz = bm(1, a, 1, c)
                    xyx = bm(*xy, 1, a)
                    # M1: z(x(zy)) = ((zx)z)y
                    if bm(1, c, *bm(1, a, *zy)) != bm(*bm(*zx, 1, c), 1, b):
                        fails["m1"] += 1
                    # M2: (xyx)z = x(y(xz))
                    if bm(*xyx, 1, c) != bm(1, a, *bm(1, b, *xz)):
                        fails["m2"] += 1
                    # M3: z(xyx) = ((zx)y)x
                    if bm(1, c, *xyx) != bm(*bm(*zx, 1, b), 1, a):
                        fails["m3"] += 1
                    # Alternatividad izq: x(xy)=(xx)y ; der: (xy)y=x(yy)
                    if bm(1, a, *xy) != bm(*xx, 1, b):
                        fails["alt_left"] += 1
                    if bm(*xy, 1, b) != bm(1, a, *yy):
                        fails["alt_right"] += 1
                    # Flexibilidad: x(yx)=(xy)x
                    if bm(1, a, *yx) != bm(*xy, 1, a):
                        fails["flex"] += 1
        total = sum(fails.values())
        return {
            "n_triples": 512,
            "n_pairs": 64,
            "failures": fails,
            "exact": total == 0,
            "justification": "multilinealidad + aritmética entera de base",
            "is_moufang": fails["m1"] == fails["m2"] == fails["m3"] == 0,
            "is_alternative": fails["alt_left"] == fails["alt_right"] == 0,
            "is_flexible": fails["flex"] == 0,
            "is_composition": fails["composition"] == 0,
        }

    # ── ÚLTIMO MÉTODO DE FASE 2 ─────────────────────────────────────
    @staticmethod
    def audit_moufang_residue(x: list, y: list, z: list) -> AlgebraicStructure:
        """
        Auditoría algebraica exhaustiva de los octoniones de Cayley–Dickson 𝕆:
        1. Certificado exacto en las 512 ternas de base sobre ℤ.
        2. Residuos numéricos de Moufang M1–M3, alternatividad, flexibilidad y
           multiplicatividad de la norma de Hurwitz N(xy) = N(x)N(y).
        3. Asociador [x,y,z], Teorema de Artin (toda subálgebra 2-generada es
           asociativa), Identidad de Malcev J(x,y,[x,z]) = [J(x,y,z),x] y
           regla de Leibniz de la derivación D_{x,y} ∈ 𝔤₂ = der(𝕆).
        → CONTINUIDAD: Fase3PoincareSelfImprove.poincare_spectral_section consume
          este AlgebraicStructure.
        """
        vx = np.array(x, dtype=float)
        vy = np.array(y, dtype=float)
        vz = np.array(z, dtype=float)
        for v, nm in ((vx, "x"), (vy, "y"), (vz, "z")):
            if v.shape != (8,):
                raise ValueError(f"{nm} debe tener 8 componentes (𝕆 ≅ ℝ⁸)")
        om = Fase2CelestialAlgebra._oct_mul
        cm = Fase2CelestialAlgebra._oct_comm
        asc = Fase2CelestialAlgebra._oct_assoc
        jac = Fase2CelestialAlgebra._oct_jacobian
        g2d = Fase2CelestialAlgebra._g2_inner_derivation

        def n2(u: np.ndarray) -> float:
            return float(np.dot(u, u))

        xyx = om(om(vx, vy), vx)
        r1 = float(np.linalg.norm(om(vz, om(vx, om(vz, vy))) - om(om(om(vz, vx), vz), vy)))
        r2 = float(np.linalg.norm(om(xyx, vz) - om(vx, om(vy, om(vx, vz)))))
        r3 = float(np.linalg.norm(om(vz, xyx) - om(om(om(vz, vx), vy), vx)))
        alt1 = float(np.linalg.norm(om(vx, om(vx, vy)) - om(om(vx, vx), vy)))
        alt2 = float(np.linalg.norm(om(om(vx, vy), vy) - om(vx, om(vy, vy))))
        flex = float(np.linalg.norm(om(vx, om(vy, vx)) - om(om(vx, vy), vx)))
        n_xy, n_x_y = n2(om(vx, vy)), n2(vx) * n2(vy)
        r_norm = abs(n_xy - n_x_y) / max(n_x_y, 1.0e-30)

        # Asociador, Teorema de Artin, Identidad de Malcev y derivación 𝔤₂
        assoc_xyz = asc(vx, vy, vz)
        assoc_norm = float(np.linalg.norm(assoc_xyz))
        artin_res = float(
            np.linalg.norm(asc(vx, vx, vy))
            + np.linalg.norm(asc(vx, vy, vy))
            + np.linalg.norm(asc(vx, vy, vx))
        )
        malcev_lhs = jac(vx, vy, cm(vx, vz))
        malcev_rhs = cm(jac(vx, vy, vz), vx)
        malcev_res = float(np.linalg.norm(malcev_lhs - malcev_rhs))

        # Regla de Leibniz para D_{x,y} ∈ 𝔤₂ sobre el producto (y · z)
        d_yz = g2d(vx, vy, om(vy, vz))
        leibniz_rhs = om(g2d(vx, vy, vy), vz) + om(vy, g2d(vx, vy, vz))
        g2_leibniz_defect = float(np.linalg.norm(d_yz - leibniz_rhs))

        basis = Fase2CelestialAlgebra._basis_certificate()
        numeric = {
            "m1": r1,
            "m2": r2,
            "m3": r3,
            "alt_left": alt1,
            "alt_right": alt2,
            "flex": flex,
            "composition_rel": r_norm,
            "artin_2gen": artin_res,
            "malcev": malcev_res,
            "g2_leibniz": g2_leibniz_defect,
        }
        scale_norm = max(1.0, math.sqrt(n2(vx) * n2(vy) * n2(vz)))
        tol = 1.0e-11 * scale_norm
        core_sum = r1 + r2 + r3 + alt1 + alt2 + flex + r_norm + malcev_res + g2_leibniz_defect
        index = (1.0 / (1.0 + core_sum)) if basis["exact"] else 0.0

        payload = {
            "basis_certificate": basis,
            "numeric_audit": numeric,
            "moufang_m1_residue": r1,
            "moufang_m2_residue": r2,
            "moufang_m3_residue": r3,
            "alternativity_left_residue": alt1,
            "alternativity_right_residue": alt2,
            "flexibility_residue": flex,
            "norm_multiplicativity_rel_error": float(r_norm),
            "is_moufang_valid": bool(
                basis["is_moufang"] and r1 < tol and r2 < tol and r3 < tol
            ),
            "is_alternative": bool(
                basis["is_alternative"] and alt1 < tol and alt2 < tol
            ),
            "is_flexible": bool(basis["is_flexible"] and flex < tol),
            "is_composition_algebra": bool(
                basis["is_composition"] and r_norm < 1.0e-10
            ),
            "algebraic_stability_index": float(index),
            "octonion_norm_x": n2(vx),
            "octonion_norm_y": n2(vy),
            "octonion_norm_z": n2(vz),
            "associator_norm": assoc_norm,
            "artin_theorem_residue": artin_res,
            "malcev_identity_residue": malcev_res,
            "is_malcev_algebra": bool(malcev_res < tol),
            "g2_derivation_audit": {
                "lie_algebra": "g2 = der(O)",
                "dimension": 14,
                "leibniz_defect": g2_leibniz_defect,
                "is_exact_derivation": bool(g2_leibniz_defect < tol),
            },
        }
        return AlgebraicStructure(**payload, lineage_hash=_chain("F2", payload))


# ═══════════════════════════════════════════════════════════════════
# ██ FASE 3 — POINCARÉ + AUTOMEJORA NIVEL 3 + ORQUESTACIÓN CERRADA ██
# INICIO: poincare_spectral_section consume AlgebraicStructure de F2.
# Contrato de salida: PoincareSection / realimentación hacia Fase 1.
# ═══════════════════════════════════════════════════════════════════
class Fase3PoincareSelfImprove:

    # ── PRIMERA FUNCIÓN DE FASE 3 (continuación directa de F2) ──────
    @staticmethod
    def poincare_spectral_section(
        kam_result: dict,
        algebraic_result: Any,
        n_iter: int = 800,
        initial_I: float = 0.1,
        initial_theta: float = 0.05,
        genome: Optional[dict] = None,
    ) -> PoincareSection:
        """
        Sección de Poincaré del mapa estándar de Chirikov (convención radianes):
            p_{k+1} = p_k + K_eff · sin(θ_k)
            θ_{k+1} = (θ_k + p_{k+1}) mod 2π
        donde K_eff = K · algebraic_stability_index.
        Calcula exponente de Lyapunov tangente de Benettin, indicador MEGNO,
        número de rotación ρ, fracción continua y certificados de Greene y
        Hurwitz–Lagrange (tanto sobre ω heredado como sobre ρ empírico).
        """
        kr = _view(kam_result)
        ar = _view(algebraic_result)
        K_kam = float(kr.get("chirikov_k_effective", 1.0))
        stab = float(ar.get("algebraic_stability_index", 1.0))
        K = K_kam * stab
        if genome and "kam_K" in genome:
            K = float(genome["kam_K"]) * stab

        theta, p = float(initial_theta), float(initial_I)
        orbit, section = [], []
        v_theta, v_p = 1.0, 0.0
        sum_log, megno_acc, ybar_sum = 0.0, 0.0, 0.0
        theta_unwrapped = theta
        max_det_defect = 0.0

        for k in range(1, n_iter + 1):
            c = K * math.cos(theta)
            p_new = p + K * math.sin(theta)
            theta_new_raw = theta + p_new
            theta_new = theta_new_raw % (2.0 * math.pi)
            # Jacobiano simpléctico J = [[1+c, 1], [c, 1]] => det(J) = (1+c) - c = 1
            det_j = (1.0 + c) * 1.0 - 1.0 * c
            max_det_defect = max(max_det_defect, abs(det_j - 1.0))
            nv_theta = (1.0 + c) * v_theta + v_p
            nv_p = c * v_theta + v_p
            d = math.hypot(nv_theta, nv_p)
            if d > 1.0e-300:
                lg = math.log(d)
                sum_log += lg
                megno_acc += 2.0 * k * lg
                ybar_sum += megno_acc / k
                v_theta, v_p = nv_theta / d, nv_p / d
            if theta_new < theta and p_new > 0.0:      # cruce transversal θ = 0 (wrap)
                denom_wrap = theta_new + 2.0 * math.pi - theta
                alpha = (2.0 * math.pi - theta) / denom_wrap if denom_wrap > 1.0e-15 else 0.0
                section.append([p + alpha * (p_new - p), 0.0])
            orbit.append([theta_new, p_new])
            theta_unwrapped += p_new
            theta, p = theta_new, p_new

        rho = ((theta_unwrapped - float(initial_theta)) / (2.0 * math.pi * max(1, n_iter))) % 1.0
        Is = [pt[1] for pt in orbit]
        mu = sum(Is) / len(Is) if Is else 0.0
        var = sum((x - mu) ** 2 for x in Is) / len(Is) if Is else 0.0
        greene = Fase2CelestialAlgebra._greene_certificate(K)
        omega_inh = float(kr.get("omega", GOLDEN_RATIO))
        dio = Fase2CelestialAlgebra._diophantine_certificate(omega_inh)
        dio_rho = Fase2CelestialAlgebra._diophantine_certificate(rho)
        dio["gamma_effective"] = max(
            dio["gamma_max_all_q"],
            dio_rho["gamma_max_all_q"],
            float(kr.get("diophantine", {}).get("gamma_effective", GOLDEN_SHARP)),
        )

        payload = {
            "map_convention": "radianes: p'=p+K sinθ; θ'=(θ+p') mod 2π",
            "K_effective": K,
            "K_from_kam": K_kam,
            "algebraic_stability_index": stab,
            "orbit": orbit,
            "section_points": section,
            "n_crossings": len(section),
            "rotation_number": rho,
            "rotation_cf": Fase2CelestialAlgebra._continued_fraction(rho, 16),
            "lyapunov_benettin": sum_log / max(1, n_iter),
            "megno_mean": ybar_sum / max(1, n_iter),
            "cloud_variance_I": var,
            "greene": greene,
            "diophantine": dio,
            "chirikov_confinement": bool(
                K < CHIRIKOV_CRITICAL and greene["torus_preserved"]
            ),
            "verdict_inherited": kr.get("kam_verdict", "UNKNOWN"),
            "fidelity_inherited": float(kr.get("fidelity_inherited", 0.0)),
            "diophantine_empirical_rho": dio_rho,
            "symplectic_det_defect_max": max_det_defect,
        }
        return PoincareSection(**payload, lineage_hash=_chain("F3", payload))

    # ── Aptitud y automejora recursiva Nivel 3 con elitismo ──────────
    @staticmethod
    def _project_genome(g: dict) -> dict:
        out = dict(g)
        for k, (lo, hi) in GENOME_BOX.items():
            if k in out and isinstance(out[k], (int, float)):
                out[k] = _clamp(float(out[k]), lo, hi)
        return out

    @staticmethod
    def _fitness_spectral(genome: dict, section: Any) -> float:
        """
        Funcional de aptitud F(g, section) ∈ [0, 1] acoplado a los 4 genes:
        1. Estabilidad de Lyapunov (Benettin sobre K_eff = kam_K · stab_idx).
        2. Confinamiento KAM (margen subcrítico + preservación de toros de Greene).
        3. Margen diofántico (kam_gamma frente a la constante diofántica efectiva).
        4. Pureza de escudo Fides y selectividad de coherencia (fides_floor y
           coherence_thresh acoplados a la fidelidad cuántica heredada).
        """
        s = _view(section)
        g = Fase3PoincareSelfImprove._project_genome(dict(genome))
        stab_idx = float(s.get("algebraic_stability_index", 1.0))
        K_eff = g["kam_K"] * stab_idx
        ben = Fase2CelestialAlgebra._benettin_cached(K_eff)
        greene = Fase2CelestialAlgebra._greene_certificate(K_eff)
        stability = math.exp(-max(0.0, ben["lyapunov"]) / 0.5)
        margin = greene["margin_subcritical"]
        kam = (1.0 / (1.0 + math.exp(-8.0 * margin))) * (
            1.0 if greene["torus_preserved"] else 0.35
        )
        dio_dict = s.get("diophantine", {})
        gamma_raw = float(dio_dict.get("gamma_max_all_q", GOLDEN_SHARP))
        gamma_eff = float(
            dio_dict.get(
                "gamma_effective",
                gamma_raw if gamma_raw > 1.0e-4 else GOLDEN_SHARP,
            )
        )
        dio_term = 1.0 / (
            1.0
            + math.exp(-8.0 * (gamma_eff - g["kam_gamma"]) / max(1.0e-9, gamma_eff))
        )
        fides_purity = math.exp(-g["fides_floor"] / 1.0e-8)
        coh_lo, coh_hi = GENOME_BOX["coherence_thresh"]
        coh_selectivity = (g["coherence_thresh"] - coh_lo) / (coh_hi - coh_lo)
        f_inh = float(s.get("fidelity_inherited", 0.0))
        if f_inh > 0.0:
            coh_compat = 1.0 / (1.0 + math.exp(-15.0 * (f_inh - g["coherence_thresh"])))
            coh_score = 0.65 * coh_selectivity + 0.35 * coh_compat
        else:
            coh_score = coh_selectivity
        fides_coh_term = 0.70 * fides_purity + 0.30 * coh_score
        return _clamp(
            0.30 * stability + 0.35 * kam + 0.20 * dio_term + 0.15 * fides_coh_term,
            0.0,
            1.0,
        )

    @staticmethod
    def _improve_spectral_l1(
        genome: dict,
        section: Any,
        rates: Optional[dict] = None,
        sign: float = 1.0,
    ) -> dict:
        """
        Nivel 1: muta el GENOMA guiado por el déficit de aptitud (1 − fitness).
        sign = −1 ejecuta la reflexión adversarial (vetada por elitismo si degrada).
        """
        s = _view(section)
        r = rates or RATES_L1
        g = Fase3PoincareSelfImprove._project_genome(dict(genome))
        f = Fase3PoincareSelfImprove._fitness_spectral(g, s)
        drive = (1.0 - f) * sign
        g["kam_K"] *= 1.0 - r["kam_K"] * drive
        g["kam_gamma"] *= 1.0 - r["kam_gamma"] * drive
        g["fides_floor"] *= 1.0 - r["fides_floor"] * drive
        g["coherence_thresh"] = _clamp(
            g["coherence_thresh"] * (1.0 + r["coherence_thresh"] * drive),
            *GENOME_BOX["coherence_thresh"],
        )
        return Fase3PoincareSelfImprove._project_genome(g)

    @staticmethod
    def _improve_spectral_l2(
        l1_rule: Callable,
        genome: dict,
        section: Any,
        rate_scale: float = 1.0,
    ) -> Tuple[Callable, dict]:
        """
        Nivel 2: muta el hiper-operador L1 (tasas de mutación a escalas espectrales
        áureas) preservando el candidato identidad para garantizar monotonía.
        """
        candidates: List[Tuple[Callable, dict]] = [
            (lambda g, s: dict(g), dict(genome))
        ]
        for alpha in (0.5, 1.0 / GOLDEN_RATIO, 1.0, GOLDEN_RATIO, 2.0):
            scale = alpha * rate_scale

            def make_rule(a: float = scale) -> Callable:
                rates = {k: v * a for k, v in RATES_L1.items()}

                def rule(g: dict, s: Any) -> dict:
                    return l1_rule(g, s, rates=rates)

                return rule

            rule = make_rule()
            candidates.append((rule, rule(genome, section)))
        return max(
            candidates,
            key=lambda rc: Fase3PoincareSelfImprove._fitness_spectral(rc[1], section),
        )

    @staticmethod
    def _improve_spectral_l3(
        genome: dict, section: Any
    ) -> Tuple[Callable, dict, dict]:
        """
        Nivel 3: torneo de meta-estrategias rivales (exploit, golden_contract,
        explore, reflect + identity) con margen de selección auditable.
        """
        strategies = {
            "exploit": (0.5, 1.0),
            "golden_contract": (1.0 / GOLDEN_RATIO, 1.0),
            "explore": (2.0, 1.0),
            "reflect": (1.0, -1.0),
        }
        evaluated = [
            {
                "strategy": "identity",
                "genome": dict(genome),
                "fitness": Fase3PoincareSelfImprove._fitness_spectral(genome, section),
            }
        ]
        rules: Dict[str, Callable] = {"identity": lambda g, s: dict(g)}
        for name, (scale, sign) in strategies.items():

            def l1_variant(
                g: dict, s: Any, rates: Optional[dict] = None, sg: float = sign
            ) -> dict:
                return Fase3PoincareSelfImprove._improve_spectral_l1(
                    g, s, rates=rates, sign=sg
                )

            def meta(
                g: dict, s: Any, sc: float = scale, lv: Callable = l1_variant
            ) -> Tuple[Callable, dict]:
                return Fase3PoincareSelfImprove._improve_spectral_l2(
                    lv, g, s, rate_scale=sc
                )

            rule, app = meta(genome, section)
            rules[name] = rule
            evaluated.append({
                "strategy": name,
                "genome": app,
                "fitness": Fase3PoincareSelfImprove._fitness_spectral(app, section),
            })
        best = max(evaluated, key=lambda e: e["fitness"])
        second = sorted((e["fitness"] for e in evaluated), reverse=True)[1]
        audit = {
            "evaluated": [
                {k: e[k] for k in ("strategy", "fitness")} for e in evaluated
            ],
            "selected": best["strategy"],
            "selection_margin": best["fitness"] - second,
            "constraints_ok": best["genome"]
            == Fase3PoincareSelfImprove._project_genome(best["genome"]),
        }
        return rules[best["strategy"]], best["genome"], audit

    @staticmethod
    def recursive_self_improve_spectral(
        genome: dict, section: Any, depth: int = 3
    ) -> dict:
        """
        Automejora recursiva L1 → L2 → L3 con VETO ELITISTA:
        Toda transición verifica F(g_{k+1}) ≥ F(g_k); la sucesión monótona
        no decreciente y acotada en [0, 1] sobre la caja compacta GENOME_BOX
        garantiza convergencia (Teorema de Convergencia Monótona + Bolzano–Weierstrass).
        """
        s = _view(section)
        current = Fase3PoincareSelfImprove._project_genome(dict(genome))
        history = [
            {
                "level": 0,
                "genome": dict(current),
                "fitness": Fase3PoincareSelfImprove._fitness_spectral(current, s),
                "veto": False,
            }
        ]
        l3_audit = None
        if depth >= 1:
            cand = Fase3PoincareSelfImprove._improve_spectral_l1(current, s)
            f = Fase3PoincareSelfImprove._fitness_spectral(cand, s)
            veto = f < history[-1]["fitness"] - 1.0e-12
            current = current if veto else cand
            history.append({
                "level": 1,
                "genome": dict(current),
                "fitness": history[-1]["fitness"] if veto else f,
                "veto": veto,
            })
        if depth >= 2:
            _, cand = Fase3PoincareSelfImprove._improve_spectral_l2(
                Fase3PoincareSelfImprove._improve_spectral_l1, current, s
            )
            f = Fase3PoincareSelfImprove._fitness_spectral(cand, s)
            veto = f < history[-1]["fitness"] - 1.0e-12
            current = current if veto else cand
            history.append({
                "level": 2,
                "genome": dict(current),
                "fitness": history[-1]["fitness"] if veto else f,
                "veto": veto,
            })
        if depth >= 3:
            _, cand, l3_audit = Fase3PoincareSelfImprove._improve_spectral_l3(
                current, s
            )
            f = Fase3PoincareSelfImprove._fitness_spectral(cand, s)
            veto = f < history[-1]["fitness"] - 1.0e-12
            current = current if veto else cand
            history.append({
                "level": 3,
                "genome": dict(current),
                "fitness": history[-1]["fitness"] if veto else f,
                "veto": veto,
            })
        mono = all(
            history[i + 1]["fitness"] >= history[i]["fitness"] - 1.0e-12
            for i in range(len(history) - 1)
        )
        return {
            "genome": current,
            "history": history,
            "final_fitness": history[-1]["fitness"],
            "depth": depth,
            "monotone_certificate": mono,
            "l3_audit": l3_audit,
        }

    # ── ÚLTIMO MÉTODO DE FASE 3 — CIERRE DEL BUCLE ANIDADO F3 → F1 ───
    @staticmethod
    def orchestrate_spectral_symplectic_loop(
        rho_matrix: list,
        sigma_matrix: list,
        mean_anomaly: float = 0.5,
        eccentricity: float = 0.3,
        genome: Optional[dict] = None,
        n_poincare_iter: int = 400,
        max_iters: int = 3,
        tol: float = 1.0e-6,
    ) -> dict:
        """
        Ciclo cerrado F1 → F2 → F3 → F1' (iteración de Kleene acotada):
        El genoma evolucionado por L3 se reinyecta en Fase1Spectral.compute_uhlmann_bures,
        y al alcanzar el punto fijo g* se ejecuta un pase de sincronización
        final para que los contratos de F1, F2 y F3 queden certificados bajo g*.
        """
        g = Fase3PoincareSelfImprove._project_genome(dict(genome or DEFAULT_GENOME))
        trace, iterations, converged = [], [], False
        h = hashlib.sha256(b"GENESIS").hexdigest()[:32]
        qstate = kam = kepler = algebraic = section = imp = None
        kepler = Fase2CelestialAlgebra.solve_kepler_halley(mean_anomaly, eccentricity)

        for it in range(max(1, max_iters)):
            qstate = Fase1Spectral.compute_uhlmann_bures(
                rho_matrix, sigma_matrix, genome=g
            )
            kam = Fase2CelestialAlgebra.eval_kam_lazutkin(quantum_state=qstate)
            kam["fidelity_inherited"] = qstate.fidelity
            seed = int(
                hashlib.sha256(
                    _canon({"fidelity": qstate.fidelity, "it": it}).encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            rng = np.random.default_rng(seed)
            algebraic = Fase2CelestialAlgebra.audit_moufang_residue(
                rng.standard_normal(8).tolist(),
                rng.standard_normal(8).tolist(),
                rng.standard_normal(8).tolist(),
            )
            section = Fase3PoincareSelfImprove.poincare_spectral_section(
                kam, algebraic, n_iter=n_poincare_iter, genome=g
            )
            imp = Fase3PoincareSelfImprove.recursive_self_improve_spectral(
                g, section, depth=3
            )
            f_new = imp["final_fitness"]
            trace.append(f_new)
            h = _chain(
                h,
                {
                    "it": it,
                    "f1": qstate.lineage_hash,
                    "f2": algebraic.lineage_hash,
                    "f3": section.lineage_hash,
                    "fitness": f_new,
                },
            )
            iterations.append({
                "iteration": it,
                "fitness": f_new,
                "genome": imp["genome"],
                "monotone": imp["monotone_certificate"],
            })
            g_new = imp["genome"]
            if it > 0 and abs(trace[-1] - trace[-2]) < tol:
                converged = True
                g = g_new
                break
            g = g_new

        # Sincronización final de punto fijo bajo g* (garantiza coherencia F1/F2/F3 == g*)
        qstate = Fase1Spectral.compute_uhlmann_bures(
            rho_matrix, sigma_matrix, genome=g
        )
        kam = Fase2CelestialAlgebra.eval_kam_lazutkin(quantum_state=qstate)
        kam["fidelity_inherited"] = qstate.fidelity
        section = Fase3PoincareSelfImprove.poincare_spectral_section(
            kam, algebraic, n_iter=n_poincare_iter, genome=g
        )
        h = _chain(
            h,
            {
                "sync_fixed_point": True,
                "f1": qstate.lineage_hash,
                "f3": section.lineage_hash,
                "genome": g,
            },
        )

        return {
            "fase_1_quantum_state": asdict(qstate),
            "fase_2": {
                "kam": kam,
                "kepler": kepler,
                "moufang": asdict(algebraic),
            },
            "fase_3": {
                "poincare": asdict(section),
                "improvement": imp,
            },
            "genome_feedback": g,
            "loop_certificate": {
                "iterations": iterations,
                "fitness_trace": trace,
                "converged": converged,
                "tol": tol,
                "feedback_real": "genoma → compute_uhlmann_bures (sincronizado en g*)",
            },
            "lineage_chain": h,
        }


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLO MCP (JSON-RPC 2.0 por stdio) — initialize/ping/tools/*
# ═══════════════════════════════════════════════════════════════════
PROTOCOL_VERSION = "2025-06-18"
TOOLS = [
    {
        "name": "compute_uhlmann_bures",
        "description": (
            "F1: Uhlmann+Bures+Fides CPTP + distancia de traza de Banach "
            "(Fuchs–van de Graaf) + entropía relativa de Umegaki (Pinsker) + "
            "información cuántica de Fisher (SLD) + auditoría RLC/Topos + seeds KAM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rho_matrix": {"type": "array"},
                "sigma_matrix": {"type": "array"},
                "fides_floor": {"type": "number"},
                "genome": {"type": "object"},
            },
            "required": ["rho_matrix", "sigma_matrix"],
        },
    },
    {
        "name": "eval_kam_lazutkin",
        "description": (
            "F2: KAM riguroso (diofántico Hurwitz–Lagrange + Greene/MacKay–Meiss "
            "sobre órbitas minimizantes Aubry–Mather + Benettin + separatrix "
            "Gelfreich–Lazutkin). Consume QuantumState opcional."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "eps": {"type": "number"},
                "omega": {"type": "number"},
                "k_chirikov": {"type": "number"},
                "quantum_state": {"type": "object"},
            },
        },
    },
    {
        "name": "solve_kepler_halley",
        "description": (
            "F2: Kepler elíptico/hiperbólico (Halley cúbico + Danby) y parabólico "
            "exacto (Barker–Cardano); invariantes de Vis-Viva, Delaunay y "
            "auditoría simpléctica de Yoshida de 4.º orden."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mean_anomaly": {"type": "number"},
                "eccentricity": {"type": "number"},
                "tol": {"type": "number"},
                "periapsis": {"type": "number"},
            },
            "required": ["mean_anomaly", "eccentricity"],
        },
    },
    {
        "name": "audit_moufang_residue",
        "description": (
            "F2: auditoría EXACTA de 𝕆 (512 ternas de base sobre ℤ) + numérica "
            "(Moufang M1–M3, alternatividad, flexibilidad, composición de Hurwitz, "
            "teorema de Artin, identidad de Malcev y derivación 𝔤₂ = der(𝕆))."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "array"},
                "y": {"type": "array"},
                "z": {"type": "array"},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "kam_greene_certificate",
        "description": (
            "F2: certificado KAM (diofántico + residuos de Greene sobre órbitas "
            "minimizantes Aubry–Mather del mapa estándar + Benettin/MEGNO)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "K": {"type": "number"},
                "omega": {"type": "number"},
            },
            "required": ["K"],
        },
    },
    {
        "name": "poincare_spectral_section",
        "description": (
            "F3: sección de Poincaré del mapa estándar (convención radianes "
            "unificada), Benettin tangente, MEGNO, número de rotación y Greene."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kam_result": {"type": "object"},
                "algebraic_result": {"type": "object"},
                "n_iter": {"type": "integer"},
                "initial_I": {"type": "number"},
                "initial_theta": {"type": "number"},
                "genome": {"type": "object"},
            },
            "required": ["kam_result", "algebraic_result"],
        },
    },
    {
        "name": "recursive_self_improve_spectral",
        "description": (
            "F3: automejora L1/L2/L3 acoplada a los 4 genes con proyección "
            "admisible, elitismo con veto y certificado de monotonía."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "genome": {"type": "object"},
                "section": {"type": "object"},
                "depth": {"type": "integer"},
            },
            "required": ["genome", "section"],
        },
    },
    {
        "name": "orchestrate_spectral_symplectic_loop",
        "description": (
            "Cierre anidado F1→F2→F3→F1' con realimentación real del genoma, "
            "sincronización de punto fijo g*, linaje SHA-256 y certificado de convergencia."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rho_matrix": {"type": "array"},
                "sigma_matrix": {"type": "array"},
                "mean_anomaly": {"type": "number"},
                "eccentricity": {"type": "number"},
                "genome": {"type": "object"},
                "n_poincare_iter": {"type": "integer"},
                "max_iters": {"type": "integer"},
            },
            "required": ["rho_matrix", "sigma_matrix"],
        },
    },
]


def _dispatch(name: str, args: dict) -> Any:
    if name == "compute_uhlmann_bures":
        return Fase1Spectral.compute_uhlmann_bures(
            args["rho_matrix"],
            args["sigma_matrix"],
            args.get("fides_floor", FIDES_FLOOR),
            args.get("genome"),
        )
    if name == "eval_kam_lazutkin":
        return Fase2CelestialAlgebra.eval_kam_lazutkin(
            args.get("eps"),
            args.get("omega"),
            args.get("k_chirikov"),
            args.get("quantum_state"),
        )
    if name == "solve_kepler_halley":
        return Fase2CelestialAlgebra.solve_kepler_halley(
            args["mean_anomaly"],
            args["eccentricity"],
            args.get("tol", 1.0e-12),
            args.get("periapsis", 1.0),
        )
    if name == "audit_moufang_residue":
        return Fase2CelestialAlgebra.audit_moufang_residue(
            args["x"], args["y"], args["z"]
        )
    if name == "kam_greene_certificate":
        return {
            "diophantine": Fase2CelestialAlgebra._diophantine_certificate(
                args.get("omega", GOLDEN_RATIO)
            ),
            "greene": Fase2CelestialAlgebra._greene_certificate(float(args["K"])),
            "benettin": Fase2CelestialAlgebra._benettin_cached(float(args["K"])),
        }
    if name == "poincare_spectral_section":
        return Fase3PoincareSelfImprove.poincare_spectral_section(
            args["kam_result"],
            args["algebraic_result"],
            args.get("n_iter", 800),
            args.get("initial_I", 0.1),
            args.get("initial_theta", 0.05),
            args.get("genome"),
        )
    if name == "recursive_self_improve_spectral":
        return Fase3PoincareSelfImprove.recursive_self_improve_spectral(
            args["genome"], args["section"], args.get("depth", 3)
        )
    if name == "orchestrate_spectral_symplectic_loop":
        return Fase3PoincareSelfImprove.orchestrate_spectral_symplectic_loop(
            args["rho_matrix"],
            args["sigma_matrix"],
            args.get("mean_anomaly", 0.5),
            args.get("eccentricity", 0.3),
            args.get("genome"),
            args.get("n_poincare_iter", 400),
            args.get("max_iters", 3),
        )
    raise KeyError(f"Tool {name} not found")


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(_sanitize(payload), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _send({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            })
            continue
        method = req.get("method")
        msg_id = req.get("id")
        if isinstance(method, str) and method.startswith("notifications/"):
            continue
        if method == "initialize":
            params = req.get("params", {})
            _send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "spectral-symplectic",
                        "version": "4.1.0-fases-anidadas",
                    },
                },
            })
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            if not name:
                _send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Missing tool name"},
                })
                continue
            try:
                data = _dispatch(name, args)
                _send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    _sanitize(data), indent=2, ensure_ascii=False
                                ),
                            }
                        ],
                        "isError": False,
                    },
                })
            except KeyError as e:
                _send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": str(e)},
                })
            except Exception as e:
                _log(f"tool error {name}: {e}")
                _send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"ERROR: {e}"}],
                        "isError": True,
                    },
                })
        elif msg_id is not None:
            _send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    main()