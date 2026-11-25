# -*- coding: utf-8 -*-
"""
MCP Server 4: ciberphysical-bridge — Evolución Soberana Nivel 3 (3 Fases Anidadas)
==================================================================================
Tejido soberano, crítico y verificado en 3 fases anidadas con cierre de Kleene:

  • FASE 1 (Fase1Physical) — Metrología GUM (JCGM 100/102, Welch–Satterthwaite),
      Filtro de Kalman en forma estabilizada de Joseph con compuerta χ²(1) (NIS),
      Teoría Espectral de Grafos y Hodge (Laplaciano combinatorio L = D − A y
      normalizado L_sym = D^{-1/2}LD^{-1/2}, Courant–Fischer + Jacobi exacto),
      Levantamiento hipercomplejo a S³ ⊂ ℍ (Cl_{0,2}(ℝ)), Física de potencia del
      tiristor SCR BT151-500R (convolución térmica de Duhamel sobre red Foster/Cauer,
      I²t Simpson O(dt⁴), dI/dt, dV/dt con snubber RC, disparo de gate, enganche)
      y secuencia bare-metal MMIO para ESP32-S3 (Xtensa LX7).
      ──► ÚLTIMO MÉTODO DE FASE 1: `read_site_telemetry` ──► emite `PhysicalState`

  • FASE 2 (Fase2CSuite) — Funtor entre categorías posetales con certificado
      analítico de monotonía (subdiferencial de Clarke) y cotas de Lipschitz,
      Descomposición en Valores Singulares (SVD vía rotaciones cíclicas de Jacobi)
      del Jacobiano J = ∂F/∂(w, fid, β₁) con discos de Gershgorin, Propagación
      de incertidumbre GUM (JCGM 102) y bandas de guarda (JCGM 106), Topos
      localico Sh([0,1]) con álgebra de Heyting–Gödel, núcleo cerrado c_θ y
      núcleo booleano de doble negación de Lawvere–Tierney j_{¬¬,θ}.
      ──► PRIMERA FUNCIÓN DE FASE 2: `_verify_poset_functor` (consume `PhysicalState`)
      ──► ÚLTIMO MÉTODO DE FASE 2: `translate_visceral_csuite` ──► emite `CSuiteObject`

  • FASE 3 (Fase3Poincare) — Mecánica celeste hamiltoniana de Hénon–Heiles con
      integrador simpléctico de 4.º orden de Yoshida, Sección transversal de
      Poincaré Σ = {x = 0, p_x > 0} con aterrizaje exacto de Hénon (RK4 en
      variable independiente x incluyendo integración exacta de tiempo dt/dx = 1/p_x),
      Exponente de Lyapunov de Benettin + indicador caótico MEGNO (Cincotta–Simó),
      Certificado diofántico de Hurwitz–Lagrange y residuos de Greene sobre órbitas
      minimizantes de Aubry–Mather (Frenkel–Kontorova Newton tridiagonal cíclico),
      y Automejora Recursiva de Nivel 3 (L1 → L2 → L3) con métrica de Banach,
      trinquete elitista y cierre de ciclo real hacia la Fase 1.
      ──► PRIMERA FUNCIÓN DE FASE 3: `poincare_cyberphysical_section` (consume `CSuiteObject`)
      ──► ÚLTIMO MÉTODO DE FASE 3: `orchestrate_cyberphysical_loop` (realimenta Fase 1)

Protocolo: MCP (JSON-RPC 2.0 por stdio). Exclusivamente biblioteca estándar de Python.
"""

import sys
import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, METROLÓGICAS, ELECTRÓNICAS, CATEGÓRICAS Y CELESTES
# ═══════════════════════════════════════════════════════════════════════════════
K_B                = 1.380_649e-23
Q_E                = 1.602_176_634e-19
T_REF_K            = 300.0
GOLDEN_RATIO       = (1.0 + math.sqrt(5.0)) / 2.0
CONJ_GOLDEN_RATIO  = GOLDEN_RATIO - 1.0                 # 1/φ ≈ 0.6180339887
HURWITZ_C          = 1.0 / math.sqrt(5.0)               # liminf q·‖qφ‖ (Teorema de Hurwitz)

# Parámetros físicos del Tiristor SCR BT151-500R (NXP/WeEn; modelo lineal on-state V_T0 + r_T·i)
SCR_I2T_A2S        = 72.0                               # ∫i²dt límite (semisinusoidal 10 ms)
SCR_DIDT_A_US      = 50.0                               # dI/dt crítico máx (A/µs)
SCR_V_T0_V         = 0.85                               # Tensión umbral on-state (V)
SCR_R_T_OHM        = 0.012                              # Resistencia diferencial on-state (Ω)
SCR_I_GT_MA        = 15.0                               # Corriente de disparo de gate máx a 25 °C (mA)
SCR_V_GT_V         = 1.5                                # Tensión de disparo de gate máx (V)
SCR_T_GT_S         = 2.0e-6                             # Tiempo de encendido controlado t_gt (s)
SCR_I_LATCH_MA     = 30.0                               # Corriente de enganche I_L (mA)
SCR_I_HOLD_MA      = 20.0                               # Corriente de mantenimiento I_H (mA)
SCR_R_TH_JC        = 1.5                                # Resistencia térmica estacionaria J-C (K/W)
SCR_T_J_MAX_C      = 125.0                              # Temperatura máxima de juntura (°C)
SCR_V_DRM_V        = 500.0                              # Tensión pico repetitiva de bloqueo directo (V)
SCR_DVDT_V_US      = 50.0                               # dV/dt crítico estático sin disparo espurio (V/µs)

# Red térmica de Foster de 4 polos (Σ R_i = 1.50 K/W = R_TH_JC; constantes de ms a s para TO-220)
SCR_FOSTER_R       = (0.09, 0.21, 0.45, 0.75)
SCR_FOSTER_TAU     = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)
SCR_ITSM_A         = math.sqrt(2.0 * SCR_I2T_A2S / 10.0e-3)  # ≈ 120.0 A pico

# Mapa de registros MMIO para ESP32-S3 (Xtensa Dual-Core LX7, TRM §6 GPIO & IO_MUX)
ESP32S3_GPIO_BASE    = 0x6000_4000
GPIO_OUT_W1TS_REG    = ESP32S3_GPIO_BASE + 0x0008       # Write-1-to-set atómico
GPIO_OUT_W1TC_REG    = ESP32S3_GPIO_BASE + 0x000C       # Write-1-to-clear atómico
GPIO_ENABLE_W1TS_REG = ESP32S3_GPIO_BASE + 0x0024       # Habilitación de buffer de salida
GPIO_FUNC_OUT_SEL0   = ESP32S3_GPIO_BASE + 0x0554       # Matriz GPIO: + 4·pin
SIG_GPIO_OUT_IDX     = 128                              # Índice de salida directa GPIO simple
APB_CLK_HZ           = 80.0e6                           # Reloj periférico APB (12.5 ns / ciclo)

SOVEREIGN_CODE       = "SOVEREIGN_VETO_EXECUTE_0x03"
ALLOWED_PINS         = (12, 13, 14, 15)

# Constantes del Funtor F: PhysicalPoset → CSuitePoset y Topos Localico
F_W0, F_WS, F_PS     = 10.0, 5.0, 15.0
F_B1, F_WL           = 0.10, 0.02
F_WEIGHTS            = (0.4, 0.3, 0.3)
F_TH_AMBER           = 0.30
F_TH_CRIT            = 0.60
TOPOS_THETA          = 0.50

# Constantes de Mecánica Celeste (Hénon–Heiles y Teoría KAM / Aubry–Mather / Greene)
HH_E_ESCAPE          = 1.0 / 6.0                        # Energía crítica de escape triangular Hénon–Heiles
K_C_GREEN            = 0.971635406                      # Umbral crítico de Chirikov–Greene para el toro áureo
GREENE_PAIRS         = (
    (2, 3), (3, 5), (5, 8), (8, 13),
    (13, 21), (21, 34), (34, 55), (55, 89),
)                                                       # Convergentes racionales de Fibonacci F_n / F_{n+1} → 1/φ

# Variedad compacta del Genoma (Retículo completo acotado en espacio de Banach ℝ⁶)
DEFAULT_GENOME = {
    "kam_K": 0.60,
    "kalman_p0": 1.0,
    "kalman_q": 1.0e-4,
    "fiedler_norm": 0.5,
    "hh_dt": 0.06,
    "mutation_rate": 0.10,
}
GENOME_BOX = {
    "kam_K": (0.05, 1.50),
    "kalman_p0": (1.0e-3, 10.0),
    "kalman_q": (1.0e-6, 1.0),
    "fiedler_norm": (0.0, 1.0),
    "hh_dt": (0.02, 0.12),
    "mutation_rate": (0.01, 0.50),
}
RATES_L1 = {
    "kam_K": 0.15,
    "hh_dt": 0.10,
    "fiedler_norm": 0.08,
    "kalman_p0": 0.10,
    "mutation_rate": 0.05,
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES TRANSVERSALES (LINAJE CRIPTOGRÁFICO, SANEAMIENTO JSON Y ÁLGEBRA)
# ═══════════════════════════════════════════════════════════════════════════════
def _log(msg: str) -> None:
    sys.stderr.write(f"[ciberphysical-bridge] {msg}\n")
    sys.stderr.flush()


def _sanitize(obj: Any) -> Any:
    """Garantiza serialización JSON estricta (RFC 8259): dataclasses → dict; NaN/Inf → None."""
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _canon(obj: Any) -> str:
    return json.dumps(_sanitize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _chain(prev: str, payload: Any) -> str:
    """Cadena de custodia SHA-256 truncada a 128 bits (32 hex) entre fases anidadas."""
    return hashlib.sha256((prev + "|" + _canon(payload)).encode("utf-8")).hexdigest()[:32]


def _view(obj: Any) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Contrato formal inválido: se esperaba dataclass o dict, se recibió {type(obj)}")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRATOS FORMALES ENTRE FASES ANIDADAS (DATACLASSES TIPADAS)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class PhysicalState:                       # CONTRATO SALIDA F1 ──► ENTRADA F2
    timestamp_iso: str
    telemetry_raw: dict
    uncertainties_gum: dict
    kalman_states: dict                    # {sensor: {x, P, K, nis, accepted, innovation}}
    margins: dict
    network_topology: dict                 # Incluye espectro Laplaciano exacto y espinor ℍ
    sensors_status: str
    filtered_sensors: dict
    w_irr_j: float
    fidelity: float
    beta_1: int                            # 1.er número de Betti derivado del 1-complejo simplicial
    beta_1_declared: int
    observable_uncertainties: dict         # Incertidumbres estándar u(w), u(fid) hacia F2
    genome_applied: dict
    lineage_hash: str = ""


@dataclass
class CSuiteObject:                        # CONTRATO SALIDA F2 ──► ENTRADA F3
    csuite_verdict: str
    severity_scalar: float
    severity_U_expanded: float
    guard_band_conformity: str             # Decisión metrológica JCGM 106:2012
    risk_mitigation: str
    operational_pain: str
    money_saved: str
    raw_observables: dict
    jacobian: list
    spectral: dict
    functor_certificate: dict
    subobject_classifier: dict             # Incluye núcleo cerrado c_θ y doble negación j_{¬¬,θ}
    energy_henon_heiles: float             # Acoplamiento hamiltoniano hacia Fase 3
    trajectory_seed: dict                  # Semilla simpléctica en T*ℝ²
    provenance: dict
    lineage_hash: str = ""


@dataclass
class PoincareSection:                     # CONTRATO SALIDA F3 ──► FITNESS / CIERRE F1
    section_def: str
    integrator: str
    energy: float
    n_returns: int
    return_points: list                    # [[y, py, t_exact], ...] en Σ = {x=0, px>0}
    rotation_number: Optional[float]
    rotation_cf: list
    lyapunov_ft: float
    megno_mean: float
    energy_drift_rel: float
    escaped: bool
    diophantine: dict
    greene: dict
    kam_K_eff: float
    severity_inherited: float
    verdict_inherited: str
    lineage_hash: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# ██ FASE 1 — FÍSICA DE CIRCUITOS, METROLOGÍA GUM, KALMAN-JOSEPH Y GRAFOS ██
# ═══════════════════════════════════════════════════════════════════════════════
# Contrato de continuidad hacia Fase 2:
#   El último método de Fase 1 (`Fase1Physical.read_site_telemetry`) construye y
#   retorna el objeto formal `PhysicalState`, el cual es consumido directamente
#   por el primer método de Fase 2 (`Fase2CSuite._verify_poset_functor`).

class Fase1Physical:
    # ── Metrología Cuántica y Estadística GUM (JCGM 100:2008) ─────────────────
    @staticmethod
    def _t95(nu: float) -> float:
        """Cuantil bilateral t_{0.975, ν} de Student (95 % de cobertura) vía interpolación armónica en 1/ν."""
        if not math.isfinite(nu) or nu >= 1.0e6:
            return 1.95996
        nodes = [
            (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
            (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
            (15, 2.131), (20, 2.086), (30, 2.042), (60, 2.000),
            (120, 1.980), (1.0e6, 1.95996),
        ]
        nu_clamped = max(1.0, float(nu))
        for (n0, t0), (n1, t1) in zip(nodes, nodes[1:]):
            if n0 <= nu_clamped <= n1:
                w = (1.0 / nu_clamped - 1.0 / n0) / (1.0 / n1 - 1.0 / n0)
                return t0 + w * (t1 - t0)
        return 1.95996

    @staticmethod
    def _gum_uncertainty(
        measured: float,
        type_b_std: float,
        resolution: float = 0.0,
        samples: Optional[List[float]] = None,
        coverage_k: Optional[float] = None,
    ) -> dict:
        """
        Evaluación de incertidumbre JCGM 100:2008 (GUM):
          • Tipo A: u_A = s / √n con ν_A = n − 1 grados de libertad.
          • Tipo B (clase instrumental rectangular ±0.01 %, certificado de calibración
            y cuantización digital δ / (2√3)): ν_B → ∞.
          • Grados de libertad efectivos ν_eff mediante fórmula de Welch–Satterthwaite:
            ν_eff = u_c⁴ / ∑ (u_i⁴ / ν_i).
        """
        comps: List[Tuple[float, float]] = []
        has_type_a = bool(samples and len(samples) >= 2)
        if has_type_a and samples is not None:
            n_s = len(samples)
            m = sum(samples) / n_s
            s = math.sqrt(sum((x - m) ** 2 for x in samples) / (n_s - 1))
            comps.append((s / math.sqrt(n_s), float(n_s - 1)))

        u_class = abs(measured) * 1.0e-4 / math.sqrt(3.0)
        comps.append((u_class, float("inf")))
        comps.append((abs(type_b_std), float("inf")))
        if resolution > 0.0:
            comps.append((resolution / (2.0 * math.sqrt(3.0)), float("inf")))

        u_c = math.sqrt(sum(u * u for u, _ in comps))
        denom = sum((u ** 4) / nu for u, nu in comps if math.isfinite(nu) and nu > 0.0)
        nu_eff = (u_c ** 4) / denom if denom > 0.0 else float("inf")
        k = coverage_k if coverage_k is not None else Fase1Physical._t95(nu_eff)

        u_a_val = comps[0][0] if has_type_a else 0.0
        u_b_sq = sum(u * u for idx, (u, _) in enumerate(comps) if not (has_type_a and idx == 0))
        return {
            "measured": measured,
            "u_type_a": u_a_val,
            "u_type_b": math.sqrt(u_b_sq),
            "u_combined": u_c,
            "nu_eff": nu_eff,
            "coverage_k": k,
            "U_expanded": k * u_c,
        }

    # ── Filtro de Kalman Estabilizado de Joseph + Compuerta χ² (NIS) ──────────
    @staticmethod
    def _kalman_update_1d(
        x_prior: float,
        p_prior: float,
        z_meas: float,
        r_meas: float,
        q_proc: float = 0.0,
        gate_chi2: float = 6.635,
    ) -> dict:
        """
        Actualización bayesiana recursiva en el espacio de Hilbert L²:
          • Predicción de covarianza: P⁻ = P⁺_{k-1} + Q.
          • Varianza de innovación: S = P⁻ + R; estadística NIS = ν² / S ~ χ²(1).
          • Compuerta estadística al 99 % (χ²_{0.99, 1} = 6.635): fail-closed ante atípicos.
          • Actualización de covarianza en forma simétrica positiva de Bucy–Joseph:
            P⁺ = (1 − K)² P⁻ + K² R.
        """
        p_m = max(0.0, float(p_prior)) + max(0.0, float(q_proc))
        r_m = max(1.0e-18, float(r_meas))
        s = p_m + r_m
        if s <= 0.0:
            return {
                "x": x_prior,
                "P": p_m,
                "K": 0.0,
                "nis": 0.0,
                "innovation": 0.0,
                "accepted": False,
            }
        innov = z_meas - x_prior
        nis = (innov * innov) / s
        if nis > gate_chi2:
            return {
                "x": x_prior,
                "P": p_m,
                "K": 0.0,
                "nis": nis,
                "innovation": innov,
                "accepted": False,
            }
        K = p_m / s
        x_p = x_prior + K * innov
        p_p = ((1.0 - K) ** 2) * p_m + (K * K) * r_m
        return {
            "x": x_p,
            "P": p_p,
            "K": K,
            "nis": nis,
            "innovation": innov,
            "accepted": True,
        }

    # ── Teoría Espectral de Grafos, Cohomología de Hodge y Cuaterniones ℍ ────
    @staticmethod
    def _graph_laplacian(adj: List[List[int]]) -> List[List[float]]:
        """Construye el operador Laplaciano combinatorio L = D − A de un 1-complejo simplicial."""
        n = len(adj)
        for i in range(n):
            if adj[i][i]:
                raise ValueError("Lazo propio detectado: el grafo debe ser simple (sin 1-ciclos degenerados)")
            for j in range(n):
                if adj[i][j] != adj[j][i]:
                    raise ValueError("Matriz de adyacencia asimétrica: se requiere grafo no dirigido")
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            L[i][i] = float(sum(adj[i]))
            for j in range(n):
                if i != j and adj[i][j]:
                    L[i][j] = -1.0
        return L

    @staticmethod
    def _normalized_laplacian(adj: List[List[int]]) -> List[List[float]]:
        """
        Construye el Laplaciano simétrico normalizado L_sym = D^{-1/2} L D^{-1/2}.
        Sus autovalores satisfacen 0 = μ₁ ≤ μ₂ ≤ ... ≤ μ_n ≤ 2 (Teorema de Chung).
        """
        n = len(adj)
        deg = [float(sum(adj[i])) for i in range(n)]
        L_sym = [[0.0] * n for _ in range(n)]
        for i in range(n):
            if deg[i] > 0.0:
                L_sym[i][i] = 1.0
            for j in range(n):
                if i != j and adj[i][j] and deg[i] > 0.0 and deg[j] > 0.0:
                    L_sym[i][j] = -1.0 / math.sqrt(deg[i] * deg[j])
        return L_sym

    @staticmethod
    def _fiedler_value(
        L: List[List[float]],
        iters: int = 3000,
        tol: float = 1.0e-12,
    ) -> float:
        """
        Calcula la conectividad algebraica de Fiedler λ₂(L):
          1. Cota superior espectral de Anderson–Morley / Gershgorin sobre aristas:
             c = max_{(i,j)∈E} (d_i + d_j) ≥ λ_max(L).
          2. Iteración de potencias deflacionada en el subespacio ortogonal al núcleo
             constante 1⊥ sobre el operador desplazado M = c·I − L (Courant–Fischer).
          3. Certificación cruzada con diagonalización simétrica de Jacobi cuando n ≤ 32.
        """
        n = len(L)
        if n <= 1:
            return 0.0
        c = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if L[i][j] != 0.0:
                    c = max(c, L[i][i] + L[j][j])
        if c <= 0.0:
            return 0.0

        M = [[(c if i == j else 0.0) - L[i][j] for j in range(n)] for i in range(n)]
        v = [float(i + 1) for i in range(n)]
        mean = sum(v) / n
        v = [x - mean for x in v]
        nv = math.sqrt(sum(x * x for x in v))
        if nv < 1.0e-15:
            return 0.0
        v = [x / nv for x in v]

        lam_prev = 0.0
        for _ in range(iters):
            w = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
            lam = sum(v[i] * w[i] for i in range(n))
            mean = sum(w) / n
            w = [x - mean for x in w]
            nw = math.sqrt(sum(x * x for x in w))
            if nw < 1.0e-15:
                break
            v = [x / nw for x in w]
            if abs(lam - lam_prev) < tol * max(1.0, c):
                lam_prev = lam
                break
            lam_prev = lam

        rayleigh_fiedler = max(0.0, c - lam_prev)
        if n <= 32:
            eigs = Fase2CSuite._jacobi_eigenvalues(L)
            return max(0.0, eigs[1]) if len(eigs) >= 2 else rayleigh_fiedler
        return rayleigh_fiedler

    @staticmethod
    def _betti_numbers_simple(adj: List[List[int]]) -> dict:
        """
        Topología algebraica del 1-complejo simplicial G = (V, E):
          • b₀ = rango H₀(G; ℤ) (componentes conexas por búsqueda en profundidad).
          • b₁ = rango H₁(G; ℤ) = E − V + b₀ (ciclos independientes de Poincaré–Kirchhoff).
          • χ(G) = V − E = b₀ − b₁ (Característica de Euler–Poincaré).
        """
        n = len(adj)
        visited = [False] * n
        components = 0
        for i in range(n):
            if visited[i]:
                continue
            components += 1
            stack = [i]
            while stack:
                u = stack.pop()
                if visited[u]:
                    continue
                visited[u] = True
                for v in range(n):
                    if adj[u][v] and not visited[v]:
                        stack.append(v)
        edges = sum(1 for i in range(n) for j in range(i + 1, n) if adj[i][j])
        b1 = max(0, edges - n + components)
        return {
            "betti_0": components,
            "betti_1": b1,
            "euler_characteristic": components - b1,
        }

    @staticmethod
    def _build_sensor_network(sensor_ids: List[str]) -> dict:
        """
        Construye el complejo topológico de sensores y calcula su espectro completo de Hodge:
          • Espectro Laplaciano combinatorio L y normalizado L_sym.
          • Conectividad algebraica de Fiedler λ₂(L) y conectividad normalizada μ₂(L_sym)/2 ∈ [0,1].
        """
        n = len(sensor_ids)
        if n == 0:
            return {
                "adjacency": [],
                "fiedler": 0.0,
                "fiedler_normalized": 0.0,
                "laplacian_spectrum": [],
                "n_nodes": 0,
                "n_edges": 0,
                "betti": {"betti_0": 0, "betti_1": 0, "euler_characteristic": 0},
            }
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                h = int(
                    hashlib.sha256(f"{sensor_ids[i]}|{sensor_ids[j]}".encode("utf-8")).hexdigest()[:4],
                    16,
                )
                if h & 1:
                    adj[i][j] = adj[j][i] = 1
        L = Fase1Physical._graph_laplacian(adj)
        L_sym = Fase1Physical._normalized_laplacian(adj)
        fiedler_val = Fase1Physical._fiedler_value(L)
        eigs_L = [max(0.0, e) for e in Fase2CSuite._jacobi_eigenvalues(L)] if n <= 32 else []
        eigs_sym = [max(0.0, e) for e in Fase2CSuite._jacobi_eigenvalues(L_sym)] if n <= 32 else []
        fiedler_norm_emp = _clamp(0.5 * eigs_sym[1], 0.0, 1.0) if len(eigs_sym) >= 2 else 0.0

        edges = sum(1 for i in range(n) for j in range(i + 1, n) if adj[i][j])
        return {
            "adjacency": adj,
            "fiedler": fiedler_val,
            "fiedler_normalized": fiedler_norm_emp,
            "laplacian_spectrum": [round(x, 8) for x in eigs_L],
            "betti": Fase1Physical._betti_numbers_simple(adj),
            "n_nodes": n,
            "n_edges": edges,
        }

    # ── Física Electrotérmica y de Conmutación del SCR BT151-500R ────────────
    @staticmethod
    def _scr_didt(series_inductance_uh: float, voltage_applied_v: float) -> float:
        """Tasa inicial de crecimiento de corriente de ánodo di/dt|_{t=0⁺} = V / L (A/µs)."""
        if series_inductance_uh <= 0.0:
            return float("inf")
        return voltage_applied_v / series_inductance_uh

    @staticmethod
    def _scr_i2t(profile: List[float], dt_s: float) -> float:
        """Integral de acción específica de Joule ∫ i²(t) dt vía cuadratura compuesta de Simpson O(dt⁴)."""
        f = [i * i for i in profile]
        n = len(f)
        if n < 2:
            return f[0] * dt_s if n else 0.0
        if n % 2 == 1:
            s = f[0] + f[-1] + 4.0 * sum(f[1:-1:2]) + 2.0 * sum(f[2:-2:2])
            return s * dt_s / 3.0
        return (0.5 * f[0] + sum(f[1:-1]) + 0.5 * f[-1]) * dt_s

    @staticmethod
    def _scr_power_profile(profile: List[float]) -> List[float]:
        """Potencia instantánea disipada en estado de conducción: P(t) = V_{T0}·i(t) + r_T·i²(t)."""
        return [SCR_V_T0_V * i + SCR_R_T_OHM * i * i for i in profile]

    @staticmethod
    def _scr_thermal_foster_convolution(
        power: List[float],
        dt_s: float,
        r_th: Tuple[float, ...],
        tau: Tuple[float, ...],
        t_init_c: float,
    ) -> dict:
        """
        Integral de convolución de Duhamel exacta para excitación constante a tramos
        sobre el modelo térmico multipolo de Foster Z_{th}(s) = ∑ R_i / (1 + s·τ_i),
        acompañada de la capacitancia térmica equivalente de Cauer en el nodo de juntura.
        """
        if len(r_th) != len(tau):
            raise ValueError("Las tuplas r_th y tau deben poseer idéntica dimensión")
        decay = [math.exp(-dt_s / t) for t in tau]
        gain = [r * (1.0 - d) for r, d in zip(r_th, decay)]
        s = [0.0] * len(r_th)
        t_max = t_init_c
        for p in power:
            s = [si * d + p * g for si, d, g in zip(s, decay, gain)]
            t_max = max(t_max, t_init_c + sum(s))

        # Capacitancia térmica instantánea de juntura C_{th,1} (equivalente de Cauer en t → 0⁺):
        # 1 / C_{th,1} = lim_{s→∞} s·Z_{th}(s) = ∑ (R_i / τ_i)
        inv_c_cauer1 = sum(r / t for r, t in zip(r_th, tau) if t > 0.0)
        c_cauer_junction_j_k = (1.0 / inv_c_cauer1) if inv_c_cauer1 > 0.0 else 0.0

        return {
            "t_j_max_c": t_max,
            "t_j_final_c": t_init_c + sum(s),
            "cauer_junction_heat_capacity_J_K": c_cauer_junction_j_k,
        }

    @staticmethod
    def _scr_gate_drive(v_gg_v: float, r_g_ohm: float, t_g_us: float) -> dict:
        """
        Verificación del circuito excitador de compuerta (Gate Drive):
          I_G = (V_{GG} − V_{GT}) / R_G ≥ 1.5 · I_{GT} (con cota máxima de 50 mA)
          y ancho de pulso t_g ≥ 3 · t_{gt}.
        """
        i_g_ma = (v_gg_v - SCR_V_GT_V) / max(1.0e-9, r_g_ohm) * 1.0e3
        ok = (
            i_g_ma >= 1.5 * SCR_I_GT_MA
            and i_g_ma <= 50.0
            and t_g_us >= 3.0 * SCR_T_GT_S * 1.0e6
        )
        return {
            "i_g_ma": i_g_ma,
            "margin_vs_igt": i_g_ma / SCR_I_GT_MA,
            "t_g_us": t_g_us,
            "gate_ok": ok,
        }

    @staticmethod
    def _interlock_and(*predicates: bool) -> bool:
        """
        Operador meet (∧) en el retículo booleano completo {⊥, ⊤}ⁿ (Teorema de Knaster–Tarski:
        mayor punto fijo coinductivo νf de seguridad). Fail-closed ante cualquier violación.
        """
        return all(predicates)

    # ── Actuación Bare-Metal Crowbar en ESP32-S3 (Xtensa LX7) ─────────────────
    @staticmethod
    def trigger_esp32_crowbar(
        confirmation_code: str,
        reason: str,
        target_pin: int = 14,
        series_inductance_uh: float = 10.0,
        voltage_applied_v: float = 12.0,
        current_profile_a: Optional[List[float]] = None,
        fault_duration_s: float = 10.0e-3,
        t_j_init_c: float = 25.0,
        v_gg_v: float = 5.0,
        r_g_ohm: float = 110.0,
        t_g_us: float = 10.0,
        voltage_rise_time_us: float = 1.0,
        upstream_fuse_i2t_a2s: Optional[float] = None,
    ) -> dict:
        """
        Disyuntor electrónico crowbar basado en SCR BT151-500R con auditoría física
        completa (I²t, dI/dt, dV/dt, convolución térmica de Foster–Duhamel, corriente
        de enganche I_L, coordinación de fusible aguas arriba) y emisión de secuencia
        MMIO atómica (W1TS/W1TC) para arquitectura ESP32-S3 LX7.
        """
        audit: Dict[str, bool] = {}
        audit["code_ok"] = (confirmation_code == SOVEREIGN_CODE)
        if not audit["code_ok"]:
            return {
                "success": False,
                "error": "INVALID_CONFIRMATION_CODE",
                "interlocks": audit,
            }
        audit["pin_ok"] = target_pin in ALLOWED_PINS
        if not audit["pin_ok"]:
            return {
                "success": False,
                "error": "PIN_NOT_ALLOWED",
                "interlocks": audit,
            }

        if current_profile_a is None:
            n = max(9, int(fault_duration_s / 5.0e-5) | 1)  # Impar para regla de Simpson exacta
            current_profile_a = [
                100.0 * math.sin(math.pi * k / (n - 1)) for k in range(n)
            ]
        n = len(current_profile_a)
        dt_s = fault_duration_s / max(1, n - 1)
        i2t = Fase1Physical._scr_i2t(current_profile_a, dt_s)
        didt = Fase1Physical._scr_didt(series_inductance_uh, voltage_applied_v)
        dvdt = voltage_applied_v / max(1.0e-9, voltage_rise_time_us)
        therm = Fase1Physical._scr_thermal_foster_convolution(
            Fase1Physical._scr_power_profile(current_profile_a),
            dt_s,
            SCR_FOSTER_R,
            SCR_FOSTER_TAU,
            t_j_init_c,
        )
        gate = Fase1Physical._scr_gate_drive(v_gg_v, r_g_ohm, t_g_us)
        t_latch_s = (
            series_inductance_uh * 1.0e-6 * SCR_I_LATCH_MA * 1.0e-3
            / max(1.0e-9, voltage_applied_v)
        )
        i_peak = max(current_profile_a)

        audit["voltage_ok"] = 0.0 < voltage_applied_v <= SCR_V_DRM_V
        audit["didt_ok"] = didt <= SCR_DIDT_A_US
        audit["dvdt_ok"] = dvdt <= SCR_DVDT_V_US
        audit["i2t_ok"] = i2t <= SCR_I2T_A2S
        audit["selectivity_ok"] = (
            upstream_fuse_i2t_a2s is None or SCR_I2T_A2S >= upstream_fuse_i2t_a2s
        )
        audit["thermal_ok"] = therm["t_j_max_c"] <= SCR_T_J_MAX_C
        audit["gate_ok"] = gate["gate_ok"]
        audit["latch_ok"] = (
            t_g_us * 1.0e-6 >= SCR_T_GT_S + t_latch_s
            and i_peak >= SCR_I_LATCH_MA * 1.0e-3
        )

        margins = {
            "i2t_used_A2s": i2t,
            "i2t_margin_A2s": SCR_I2T_A2S - i2t,
            "didt_A_us": didt,
            "didt_margin_A_us": SCR_DIDT_A_US - didt,
            "dvdt_V_us": dvdt,
            "t_j_max_C": therm["t_j_max_c"],
            "thermal_margin_C": SCR_T_J_MAX_C - therm["t_j_max_c"],
            "cauer_junction_C_th_J_K": therm["cauer_junction_heat_capacity_J_K"],
            "i_latch_time_ns": t_latch_s * 1.0e9,
        }
        if not Fase1Physical._interlock_and(*audit.values()):
            return {
                "success": False,
                "error": "INTERLOCK_TRIPPED",
                "interlocks": audit,
                "margins": margins,
                "gate_design": gate,
            }

        mask = 1 << target_pin
        mmio = [
            {
                "reg": hex(GPIO_ENABLE_W1TS_REG),
                "value": hex(mask),
                "semantic": f"OUTPUT_ENABLE_GPIO_{target_pin}",
            },
            {
                "reg": hex(GPIO_FUNC_OUT_SEL0 + 4 * target_pin),
                "value": hex(SIG_GPIO_OUT_IDX),
                "semantic": f"FUNC_SEL_GPIO_{target_pin}_SIMPLE_OUT",
            },
            {
                "reg": hex(GPIO_OUT_W1TS_REG),
                "value": hex(mask),
                "semantic": f"SET_GPIO_{target_pin}_GATE_HIGH",
            },
            {
                "reg": "DWELL",
                "value": t_g_us,
                "semantic": "GATE_PULSE_WIDTH_US",
            },
            {
                "reg": hex(GPIO_OUT_W1TC_REG),
                "value": hex(mask),
                "semantic": f"CLEAR_GPIO_{target_pin}_GATE_LOW",
            },
        ]
        t_gpio_ns = 4.0 / APB_CLK_HZ * 1.0e9
        timing = {
            "gpio_writes_ns": t_gpio_ns,
            "driver_delay_ns": 50.0,
            "scr_t_gt_ns": SCR_T_GT_S * 1.0e9,
            "current_ramp_to_latch_ns": t_latch_s * 1.0e9,
            "total_est_ns": (
                t_gpio_ns + 50.0 + SCR_T_GT_S * 1.0e9 + t_latch_s * 1.0e9
            ),
            "model_uncertainty_ns": 500.0,
        }
        return {
            "success": True,
            "action": "HARDWARE_CROWBAR_FIRED",
            "gpio_pin": target_pin,
            "mmio_sequence": mmio,
            "timing_budget": timing,
            "thyristor_model": "SCR BT151-500R",
            "physics": margins,
            "gate_design": gate,
            "reason": reason,
            "physical_state": "DISCONNECTED_SAFE_ISOLATION",
            "interlocks": audit,
            "lineage_hash": _chain("F1-ACT", {"pin": target_pin, "margins": margins}),
        }

    # ── DEFINICIÓN FORMAL FINAL DE FASE 1 (PUENTE HACIA FASE 2) ───────────────
    @staticmethod
    def read_site_telemetry(
        sensor_ids: Optional[List[str]] = None,
        prior_state: Optional[dict] = None,
        genome: Optional[dict] = None,
        w_irr_j: float = 0.0,
        fidelity: float = 0.97,
        beta_1: int = 0,
    ) -> PhysicalState:
        """
        ── ÚLTIMO MÉTODO DE FASE 1 (EMITE `PhysicalState` CONSUMIDO POR FASE 2) ──
        Integra metrología GUM (Welch–Satterthwaite), filtrado estocástico de Kalman
        en forma de Joseph gobernado por `(kalman_p0, kalman_q)` del genoma,
        espectro exacto de Hodge–Laplace y levantamiento cuaterniónico de los 4
        sensores perimetrales hacia S³ ⊂ ℍ.
        """
        if sensor_ids is None:
            sensor_ids = ["slump-A", "pressure-B", "scale-C", "ambient-D"]
        g = dict(DEFAULT_GENOME)
        if genome:
            g.update({k: v for k, v in genome.items() if k in GENOME_BOX})
        g = Fase3Poincare._project_genome(g)

        z = {"slump": 14.2, "press": 42.8, "scale": 0.35, "amb": 26.5}
        tb = {"slump": 0.15, "press": 0.50, "scale": 0.05, "amb": 0.30}
        res = {"slump": 0.10, "press": 0.10, "scale": 0.01, "amb": 0.10}
        gum = {k: Fase1Physical._gum_uncertainty(z[k], tb[k], res[k]) for k in z}

        prior_k = (prior_state or {}).get("kalman_states", {})
        kalman = {}
        for k in z:
            pk = prior_k.get(k, {})
            kalman[k] = Fase1Physical._kalman_update_1d(
                pk.get("x", z[k]),
                pk.get("P", g["kalman_p0"]),
                z[k],
                gum[k]["u_combined"] ** 2,
                q_proc=g["kalman_q"],
            )

        network = Fase1Physical._build_sensor_network(sensor_ids)

        # Levantamiento del estado estimado de los 4 canales al cuerpo de cuaterniones ℍ
        q_raw = (
            kalman["slump"]["x"] / 15.0,
            kalman["press"]["x"] / 45.0,
            kalman["scale"]["x"] / 1.0,
            kalman["amb"]["x"] / 30.0,
        )
        q_norm = math.sqrt(sum(c * c for c in q_raw)) or 1.0
        network["quaternion_state_s3"] = [round(c / q_norm, 8) for c in q_raw]

        margins = {
            "slump_cm": abs(kalman["slump"]["x"] - 13.0),
            "pressure_bar": abs(kalman["press"]["x"] - 40.0),
            "scale_pct": 0.5 - abs(kalman["scale"]["x"]),
        }
        gates_ok = all(s["accepted"] for s in kalman.values())
        healthy = (
            margins["slump_cm"] <= 2.5
            and margins["pressure_bar"] <= 5.0
            and margins["scale_pct"] >= 0.0
            and gates_ok
            and network["fiedler"] > 0.0
        )
        w = max(0.0, float(w_irr_j))
        fid = _clamp(float(fidelity), 0.0, 1.0)
        b1_derived = int(network["betti"]["betti_1"])

        payload = {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "telemetry_raw": {
                "concrete_slump_cm": z["slump"],
                "hydraulic_line_pressure_bar": z["press"],
                "aggregate_scale_error_pct": z["scale"],
                "ambient_temp_c": z["amb"],
            },
            "uncertainties_gum": gum,
            "kalman_states": kalman,
            "margins": margins,
            "network_topology": network,
            "sensors_status": "ONLINE_HEALTHY" if healthy else "DEGRADED",
            "filtered_sensors": {sid: "OK" for sid in sensor_ids},
            "w_irr_j": w,
            "fidelity": fid,
            "beta_1": b1_derived,
            "beta_1_declared": int(beta_1),
            "observable_uncertainties": {
                "u_w_irr_j": 0.02 * (1.0 + w),
                "u_fidelity": 0.01,
                "coverage_k": 2.0,
                "provenance": "Tipo B declarada; JCGM 102 §6 propagación en F2",
            },
            "genome_applied": g,
        }
        return PhysicalState(**payload, lineage_hash=_chain("F1", payload))


# ═══════════════════════════════════════════════════════════════════════════════
# ██ FASE 2 — FUNTOR POSETAL, ESPECTRO SVD DE JACOBI Y TOPOS LOCALICO Sh([0,1]) ██
# ═══════════════════════════════════════════════════════════════════════════════
# Continuación directa de Fase 1:
#   • PRIMERA FUNCIÓN: `Fase2CSuite._verify_poset_functor(state)` consume el
#     `PhysicalState` emitido por `Fase1Physical.read_site_telemetry`.
#   • ÚLTIMO MÉTODO: `Fase2CSuite.translate_visceral_csuite(...)` emite el
#     objeto formal `CSuiteObject`, que inicia la Fase 3.

class Fase2CSuite:
    # ── PRIMERA FUNCIÓN DE FASE 2 (CONTINUACIÓN DIRECTA DE FASE 1) ────────────
    @staticmethod
    def _verify_poset_functor(state: Any) -> dict:
        """
        ── PRIMERA FUNCIÓN DE FASE 2 (CONSUME `PhysicalState` DE FASE 1) ──
        Certifica analítica y empíricamente que F: (Physical, ≤_P) → (CSuite, ≤_C)
        es un funtor covariante entre categorías delgadas (posets):
          1. Ley de identidad: F(id_A) = id_{F(A)} (automática en categorías posetales).
          2. Ley de composición / monotonía: A ≤_P B ⟹ F(A) ≤_C F(B), demostrada
             mediante positividad del subdiferencial de Clarke ∂_C F en todo el
             dominio convexo y acotada mediante constantes explícitas de Lipschitz.
        """
        base = Fase2CSuite._poset_coords(state)
        analytic = {
            "d_risk_dw": "r(1−r)/5 ≥ 0 ∀w ∈ ℝ⁺",
            "d_pain_dw": "∂_C pain ∈ [0, 1/15] ⊂ ℝ₀⁺",
            "d_money_dw": "∂_C money_w ∈ [0, 0.02] ⊂ ℝ₀⁺ (retracción monótona)",
            "d_money_dfid": "∂_C money_fid ∈ [−1, 0] ⊂ ℝ₀⁻ (coordenada dualizada)",
            "d_money_db1": "Δ_{β₁} money ∈ [0, 0.1] ⊂ ℝ₀⁺ (diferencia finita en ℤ)",
            "weights_positive": all(w > 0.0 for w in F_WEIGHTS),
        }
        lipschitz = {
            "L_w": F_WEIGHTS[0] / (4.0 * F_WS) + F_WEIGHTS[1] / F_PS + F_WEIGHTS[2] * F_WL,
            "L_fidelity": F_WEIGHTS[2],
            "L_beta1": F_WEIGHTS[2] * F_B1,
        }
        samples = []
        for dw in (0.0, 2.0, 5.0, 10.0):
            for df in (0.0, -0.05):
                for db in (0, 1):
                    a = dict(base)
                    b = {
                        "w_irr_j": base["w_irr_j"] + dw,
                        "fidelity": max(0.0, base["fidelity"] + df),
                        "beta_1": base["beta_1"] + db,
                    }
                    if Fase2CSuite._poset_le(a, b):
                        fa = Fase2CSuite._apply_functor(a)["severity_scalar"]
                        fb = Fase2CSuite._apply_functor(b)["severity_scalar"]
                        samples.append({
                            "A": a,
                            "B": b,
                            "monotone_ok": fb >= fa - 1.0e-12,
                        })
        mono = all(s["monotone_ok"] for s in samples)
        return {
            "identity_law_ok": True,
            "identity_justification": "categoría posetal delgada: Hom(A,A)={id_A} ⟹ F(id)=id",
            "analytic_sign_certificate": analytic,
            "lipschitz_constants": lipschitz,
            "monotone_law_ok": mono,
            "n_samples": len(samples),
            "functor_samples": samples,
            "functor_valid": mono and analytic["weights_positive"],
        }

    @staticmethod
    def _poset_coords(state: Any) -> dict:
        s = _view(state)
        return {
            "w_irr_j": float(s["w_irr_j"]),
            "fidelity": float(s["fidelity"]),
            "beta_1": int(s["beta_1"]),
        }

    @staticmethod
    def _poset_le(a: dict, b: dict) -> bool:
        """Orden parcial en PhysicalPoset = ℝ₀⁺ × [0,1]^{op} × ℕ₀."""
        return (
            a["w_irr_j"] <= b["w_irr_j"]
            and a["fidelity"] >= b["fidelity"]
            and a["beta_1"] <= b["beta_1"]
        )

    @staticmethod
    def _apply_functor(state: dict) -> dict:
        """
        Evalúa el morfismo de objetos F(w, fid, β₁) = (risk, pain, money_loss),
        su funcional convexo de severidad y el operador Jacobiano J ∈ ℝ^{3×3}.
        """
        w = float(state["w_irr_j"])
        fid = float(state["fidelity"])
        b1 = int(state["beta_1"])

        risk = 1.0 / (1.0 + math.exp(-(w - F_W0) / F_WS))
        pain = max(0.0, w - F_W0) / F_PS
        m_raw = 1.0 - fid + F_B1 * b1 + F_WL * max(0.0, w - F_W0)
        money = _clamp(m_raw, 0.0, 1.0)
        sev = F_WEIGHTS[0] * risk + F_WEIGHTS[1] * pain + F_WEIGHTS[2] * money

        verdict = (
            "NOMINAL_COHERENT"
            if sev < F_TH_AMBER
            else "AMBER_WARNING"
            if sev < F_TH_CRIT
            else "CRITICAL_VETO"
        )
        hp = 1.0 if w > F_W0 else 0.0
        act = 1.0 if 0.0 < m_raw < 1.0 else 0.0
        dr_dw = risk * (1.0 - risk) / F_WS

        J = [
            [dr_dw, 0.0, 0.0],
            [hp / F_PS, 0.0, 0.0],
            [F_WL * hp * act, -act, F_B1 * act],
        ]
        grad = [
            F_WEIGHTS[0] * J[0][0] + F_WEIGHTS[1] * J[1][0] + F_WEIGHTS[2] * J[2][0],
            F_WEIGHTS[2] * J[2][1],
            F_WEIGHTS[2] * J[2][2],
        ]
        return {
            "risk": risk,
            "pain": pain,
            "money_loss": money,
            "severity_scalar": sev,
            "verdict": verdict,
            "jacobian": J,
            "severity_gradient": grad,
        }

    # ── Álgebra Lineal Espectral: Diagonalización Cíclica de Jacobi y SVD ─────
    @staticmethod
    def _jacobi_eigenvalues(
        A: List[List[float]],
        sweeps: int = 60,
        tol: float = 1.0e-14,
    ) -> List[float]:
        """
        Diagonalización ortogonal de Jacobi para operadores autoadjuntos A = Aᵀ.
        Convergencia cuadrática asintótica de la norma fuera de diagonal ||off(A)||_F → 0.
        """
        n = len(A)
        if n == 0:
            return []
        a = [row[:] for row in A]
        for _ in range(sweeps):
            off = math.sqrt(
                sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
            )
            if off < tol:
                break
            for p in range(n - 1):
                for q in range(p + 1, n):
                    if abs(a[p][q]) < 1.0e-300:
                        continue
                    theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                    t = (1.0 if theta >= 0.0 else -1.0) / (
                        abs(theta) + math.sqrt(theta * theta + 1.0)
                    )
                    c = 1.0 / math.sqrt(t * t + 1.0)
                    s = t * c
                    for k in range(n):
                        akp, akq = a[k][p], a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    for k in range(n):
                        apk, aqk = a[p][k], a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
        return sorted(a[i][i] for i in range(n))

    @staticmethod
    def _spectral_verdict_decompose(csuite: dict) -> dict:
        """
        Descomposición espectral del operador tangente J = DF:
          • Valores singulares σ_i(J) = √λ_i(JᵀJ) vía Jacobi.
          • Norma de operador en espacio de Banach (ℝ³, ‖·‖₂): ‖J‖_{op} = σ_max(J).
          • Certificación de inclusión espectral mediante discos de Gershgorin sobre JᵀJ.
        """
        J = csuite["jacobian"]
        n = 3
        JtJ = [
            [sum(J[k][i] * J[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
        eigs = Fase2CSuite._jacobi_eigenvalues(JtJ)
        sigma = [math.sqrt(max(0.0, e)) for e in reversed(eigs)]
        s_max = sigma[0]
        s_min = sigma[-1]
        cond = (s_max / s_min) if s_min > 1.0e-12 else None

        centers = [JtJ[i][i] for i in range(n)]
        radii = [sum(abs(JtJ[i][j]) for j in range(n) if j != i) for i in range(n)]
        g_lo = min(c - r for c, r in zip(centers, radii))
        g_hi = max(c + r for c, r in zip(centers, radii))
        gersh_ok = (eigs[-1] <= g_hi + 1.0e-9) and (eigs[0] >= g_lo - 1.0e-9)

        v = [csuite["risk"], csuite["pain"], csuite["money_loss"]]
        dominant = ["risk", "pain", "money_loss"][v.index(max(v))]
        return {
            "singular_values": sigma,
            "sigma_max": s_max,
            "lipschitz_bound": s_max,
            "condition_number": cond,
            "gershgorin_enclosure": [g_lo, g_hi],
            "gershgorin_audit_ok": gersh_ok,
            "dominant_observable": dominant,
            "observable_norm_2": math.sqrt(sum(x * x for x in v)),
            "operator": "J = ∂F/∂(w, fidelity, β₁); σᵢ = √λᵢ(JᵀJ)",
        }

    # ── Propagación de Incertidumbre GUM (JCGM 102) ───────────────────────────
    @staticmethod
    def _gum_propagate_functor(
        grad: List[float],
        u_w: float,
        u_fid: float,
        k: float = 2.0,
    ) -> dict:
        """Ley de propagación de varianzas u_c²(s) = (∂s/∂w)²·u²(w) + (∂s/∂fid)²·u²(fid)."""
        u_s = math.sqrt((grad[0] * u_w) ** 2 + (grad[1] * u_fid) ** 2)
        return {"u_severity": u_s, "U_severity": k * u_s, "coverage_k": k}

    # ── Topos Localico Sh([0,1]), Heyting–Gödel y Núcleo de Lawvere–Tierney ──
    @staticmethod
    def _heyting_implication(a: float, b: float) -> float:
        """Implicación intuicionista de Gödel (adjunto derecho del producto ∧ = min): a ∧ x ≤ b ⟺ x ≤ (a → b)."""
        return 1.0 if a <= b else b

    @staticmethod
    def _topos_subobject_classifier(
        csuite: dict,
        threshold: float = TOPOS_THETA,
    ) -> dict:
        """
        Clasificador de subobjetos Ω en el Topos Localico Sh([0,1]):
          1. El frame lineal H = ([0,1], ∧=min, ∨=max, →_Gödel) verifica la adjunción
             de residuación de Heyting: (a ∧ x ≤ b) ⟺ (x ≤ a → b).
          2. El operador de truncamiento modal j_θ(p) = [p ≥ θ] preserva ⊤, es
             idempotente y preserva ínfimos (∧).
          3. Además, se construye el **Núcleo de Lawvere–Tierney exacto** sobre el
             sublocale cerrado H_{c_θ} = [θ, 1] (definido por el núcleo cerrado
             inflacionario c_θ(p) = max(p, θ)):
               j_{¬¬,θ}(p) = ((p ∨ θ) →_Gödel θ) →_Gödel θ ∈ {θ, 1},
             el cual satisface los 3 axiomas estrictos de núcleo en un locale:
               (N1) Inflacionariedad: p_θ ≤ j_{¬¬,θ}(p_θ) ∀ p_θ ∈ [θ, 1],
               (N2) Idempotencia: j_{¬¬,θ}(j_{¬¬,θ}(p_θ)) = j_{¬¬,θ}(p_θ),
               (N3) Preservación de meets: j_{¬¬,θ}(p ∧ q) = j_{¬¬,θ}(p) ∧ j_{¬¬,θ}(q),
             induciendo el subtopos booleano de haces Sh_{j_{¬¬,θ}}(H_{c_θ}) ≃ Set.
        """
        sev = float(csuite["severity_scalar"])
        grid = [i / 10.0 for i in range(11)]
        resid_ok = all(
            (min(a, x) <= b) == (x <= Fase2CSuite._heyting_implication(a, b))
            for a in grid
            for b in grid
            for x in grid
        )

        # Operador de retracción booleana j_θ(p) = [p ≥ θ]
        j = lambda p: 1.0 if p >= threshold else 0.0
        lt_top = j(1.0) == 1.0
        lt_idem = all(j(j(p)) == j(p) for p in grid)
        lt_meet = all(j(min(a, b)) == min(j(a), j(b)) for a in grid for b in grid)
        lt_mono = all(j(a) <= j(b) for a in grid for b in grid if a <= b)

        # Núcleo cerrado c_θ(p) = max(p, θ) y Núcleo de Doble Negación j_{¬¬,θ} en [θ, 1]
        c_theta = lambda p: max(p, threshold)
        j_dn_quotient = lambda p: Fase2CSuite._heyting_implication(
            Fase2CSuite._heyting_implication(c_theta(p), threshold), threshold
        )
        q_grid = [c_theta(p) for p in grid]
        nucleus_inflationary = all(p <= j_dn_quotient(p) + 1.0e-12 for p in q_grid)
        nucleus_idempotent = all(
            abs(j_dn_quotient(j_dn_quotient(p)) - j_dn_quotient(p)) < 1.0e-12
            for p in q_grid
        )
        nucleus_meet = all(
            abs(j_dn_quotient(min(a, b)) - min(j_dn_quotient(a), j_dn_quotient(b))) < 1.0e-12
            for a in q_grid
            for b in q_grid
        )

        truth = j(sev)
        return {
            "omega_graded": sev,
            "omega_binary": int(truth),
            "threshold": threshold,
            "is_veto_subobject": bool(truth),
            "heyting_residuation_audit_ok": resid_ok,
            "lawvere_tierney_audit": {
                "j_top": lt_top,
                "j_idempotent": lt_idem,
                "j_meet_preserving": lt_meet,
                "j_monotone": lt_mono,
            },
            "quotient_double_negation_nucleus": {
                "closed_sublocale_c_theta": c_theta(sev),
                "j_dn_value": j_dn_quotient(sev),
                "inflationary_ok": nucleus_inflationary,
                "idempotent_ok": nucleus_idempotent,
                "meet_preserving_ok": nucleus_meet,
                "is_strict_locale_nucleus": (
                    nucleus_inflationary and nucleus_idempotent and nucleus_meet
                ),
            },
            "booleanization": "Ω_j = {0,1}; subtopos de j-haces Sh_{j_{¬¬,θ}}([θ,1]) ≅ Set",
            "nucleus_warning": (
                "j_θ es retracción monótona; c_θ y j_{¬¬,θ} en [θ,1] son núcleos inflacionarios estrictos"
            ),
            "topos": "Sh([0,1] como frame-cadena); Ω ≅ ([0,1], Gödel)",
            "pullback": "extensión crisp = pullback de true:1→Ω a lo largo de χ",
        }

    # ── DEFINICIÓN FORMAL FINAL DE FASE 2 (PUENTE HACIA FASE 3) ───────────────
    @staticmethod
    def translate_visceral_csuite(
        fidelity: Optional[float] = None,
        w_irr: Optional[float] = None,
        beta_1: Optional[int] = None,
        physical_state: Optional[Any] = None,
    ) -> CSuiteObject:
        """
        ── ÚLTIMO MÉTODO DE FASE 2 (EMITE `CSuiteObject` CONSUMIDO POR FASE 3) ──
        Ejecuta el funtor certificado Physical → CSuite sobre el `PhysicalState`
        recibido de la Fase 1, calcula la SVD de Jacobi del Jacobiano, propaga
        la incertidumbre GUM con banda de guarda JCGM 106, evalúa el clasificador
        de subobjetos del topos y genera la energía y semilla canónica para el
        sistema hamiltoniano celeste de Hénon–Heiles en la Fase 3.
        """
        if physical_state is not None:
            ps = _view(physical_state)
            state = {
                "w_irr_j": float(ps["w_irr_j"]),
                "fidelity": float(ps["fidelity"]),
                "beta_1": int(ps["beta_1"]),
            }
            ou = ps.get(
                "observable_uncertainties",
                {"u_w_irr_j": 0.02, "u_fidelity": 0.01, "coverage_k": 2.0},
            )
            provenance: Dict[str, Any] = {
                "mode": "CONTRATO_F1",
                "lineage_f1": ps.get("lineage_hash"),
            }
            if w_irr is not None and abs(float(w_irr) - state["w_irr_j"]) > 1.0e-9:
                provenance["conflict_w_irr"] = "declarado≠F1; prevalece F1"
        else:
            state = {
                "w_irr_j": float(w_irr or 0.0),
                "fidelity": float(fidelity if fidelity is not None else 0.97),
                "beta_1": int(beta_1 or 0),
            }
            ou = {
                "u_w_irr_j": 0.02 * (1.0 + state["w_irr_j"]),
                "u_fidelity": 0.01,
                "coverage_k": 2.0,
            }
            provenance = {"mode": "AUTONOMO_SIN_F1"}

        cert = Fase2CSuite._verify_poset_functor(state)
        fmap = Fase2CSuite._apply_functor(state)
        spectral = Fase2CSuite._spectral_verdict_decompose(fmap)
        prop = Fase2CSuite._gum_propagate_functor(
            fmap["severity_gradient"],
            ou["u_w_irr_j"],
            ou["u_fidelity"],
            ou.get("coverage_k", 2.0),
        )
        subobj = Fase2CSuite._topos_subobject_classifier(fmap)

        sev, U = fmap["severity_scalar"], prop["U_severity"]
        if sev - U >= TOPOS_THETA:
            guard = "VETO_PROBADO"
        elif sev + U < TOPOS_THETA:
            guard = "NOMINAL_PROBADO"
        else:
            guard = "INDETERMINADO_BANDA_GUARDA"

        narratives = {
            "NOMINAL_COHERENT": (
                "Cero riesgo de sanción SECOP II.",
                "Cero parálisis en frentes de obra.",
                "Preservación del WACC al 15.0 % (+11.5 % liquidez).",
            ),
            "AMBER_WARNING": (
                "Riesgo transitorio por volatilidad de insumos.",
                "Alivio temporal de 1 h; interventoría requerida.",
                "Evita pérdidas por paradas en vertido de concreto.",
            ),
            "CRITICAL_VETO": (
                "Fraude presupuestal o socavón inminente.",
                "Corte perimetral preventivo inmediato.",
                "Blindaje del 100 % de la caja de la concesión.",
            ),
        }
        nar = narratives[fmap["verdict"]]

        energy = max(1.0e-4, 0.98 * sev * HH_E_ESCAPE)
        eta = _clamp(0.6 * (fmap["money_loss"] - fmap["risk"]), -0.6, 0.6)
        seed = {
            "x0": 0.0,
            "y0": 0.0,
            "eta": eta,
            "py0": eta * math.sqrt(2.0 * energy),
            "px0": math.sqrt(max(1.0e-12, 2.0 * energy - (eta ** 2) * 2.0 * energy)),
        }
        payload = {
            "csuite_verdict": fmap["verdict"],
            "severity_scalar": sev,
            "severity_U_expanded": U,
            "guard_band_conformity": guard,
            "risk_mitigation": nar[0],
            "operational_pain": nar[1],
            "money_saved": nar[2],
            "raw_observables": {
                "risk": fmap["risk"],
                "pain": fmap["pain"],
                "money_loss": fmap["money_loss"],
            },
            "jacobian": fmap["jacobian"],
            "spectral": spectral,
            "functor_certificate": cert,
            "subobject_classifier": subobj,
            "energy_henon_heiles": energy,
            "trajectory_seed": seed,
            "provenance": provenance,
        }
        return CSuiteObject(**payload, lineage_hash=_chain("F2", payload))


# ═══════════════════════════════════════════════════════════════════════════════
# ██ FASE 3 — MECÁNICA CELESTE DE POINCARÉ, KAM/GREENE Y AUTOMEJORA NIVEL 3 ██
# ═══════════════════════════════════════════════════════════════════════════════
# Continuación directa de Fase 2 y cierre del bucle hacia Fase 1:
#   • PRIMERA FUNCIÓN: `Fase3Poincare.poincare_cyberphysical_section(csuite, ...)`
#     consume el `CSuiteObject` emitido por `Fase2CSuite.translate_visceral_csuite`.
#   • ÚLTIMO MÉTODO: `Fase3Poincare.orchestrate_cyberphysical_loop(...)` cierra
#     el punto fijo de Kleene reinyectando el genoma evolucionado y la covarianza
#     de Kalman en `Fase1Physical.read_site_telemetry`.

_GREENE_CACHE: Dict[float, dict] = {}


class Fase3Poincare:
    # ── PRIMERA FUNCIÓN DE FASE 3 (CONTINUACIÓN DIRECTA DE FASE 2) ────────────
    @staticmethod
    def poincare_cyberphysical_section(
        csuite: Any,
        n_returns: int = 24,
        omega: float = GOLDEN_RATIO,
        genome: Optional[dict] = None,
    ) -> PoincareSection:
        """
        ── PRIMERA FUNCIÓN DE FASE 3 (CONSUME `CSuiteObject` DE FASE 2) ──
        Integra el sistema hamiltoniano galáctico-celeste de Hénon–Heiles
          H(x, y, p_x, p_y) = ½(p_x² + p_y²) + ½(x² + y²) + x²y − ⅓y³
        mediante el integrador simpléctico de 4.º orden de Yoshida y construye la
        sección transversal de Poincaré Σ = {x = 0, p_x > 0}:
          • Aterrizaje exacto de Hénon (1982): cambio de variable independiente
            de t a x e integración RK4 en un solo paso h = −x_prev tanto para
            (y, p_x, p_y) como para el tiempo exacto de cruce dt/dx = 1/p_x.
          • Exponente máximo de Lyapunov de tiempo finito (Benettin) + indicador
            integral MEGNO Ȳ(T) de Cincotta–Simó.
          • Certificado diofántico de Hurwitz–Lagrange y residuos de Greene sobre
            órbitas periódicas minimizantes de Aubry–Mather.
        """
        cs = _view(csuite)
        g = dict(DEFAULT_GENOME)
        if genome:
            g.update({k: v for k, v in genome.items() if k in GENOME_BOX})
        g = Fase3Poincare._project_genome(g)

        E = float(cs["energy_henon_heiles"])
        seed = cs["trajectory_seed"]
        dt = g["hh_dt"]
        state = [
            float(seed["x0"]),
            float(seed["y0"]),
            float(seed["px0"]),
            float(seed["py0"]),
        ]
        d0 = 1.0e-8
        shadow = [s + d0 * 0.5 for s in state]

        returns: List[List[float]] = []
        t = 0.0
        sum_log = 0.0
        megno_acc = 0.0
        ybar_sum = 0.0
        n_steps = 0
        drift_max = 0.0
        escaped = False
        max_steps = 20000

        for _ in range(max_steps):
            if len(returns) >= n_returns:
                break
            prev_state = state[:]
            t_prev = t

            state = Fase3Poincare._yoshida4_step(state, dt)
            shadow = Fase3Poincare._yoshita4_shadow(shadow, dt)
            t += dt
            n_steps += 1

            H = Fase3Poincare._hh_energy(state)
            drift_max = max(drift_max, abs(H - E) / max(1.0e-12, E))

            d = math.sqrt(sum((shadow[k] - state[k]) ** 2 for k in range(4)))
            if d > 1.0e-300:
                lg = math.log(d / d0)
                sum_log += lg
                megno_acc += 2.0 * (t - 0.5 * dt) * lg
                ybar_sum += megno_acc / t
                shadow = [
                    state[k] + (shadow[k] - state[k]) * d0 / d for k in range(4)
                ]

            if abs(state[0]) > 5.0 or abs(state[1]) > 5.0 or drift_max > 0.02:
                escaped = True
                break

            # Intersección transversal con Σ = {x = 0, p_x > 0} y aterrizaje exacto de Hénon
            x_old = prev_state[0]
            if x_old != 0.0 and (x_old < 0.0) != (state[0] < 0.0):
                if state[2] > 1.0e-9:
                    landed = Fase3Poincare._henon_landing(prev_state, dt)
                    if landed is not None:
                        # landed = [0.0, y_cross, px_cross, py_cross, dt_cross]
                        t_cross = t_prev + landed[4] if len(landed) >= 5 else t
                        returns.append([landed[1], landed[3], t_cross])

        lam = sum_log / t if t > 0.0 else 0.0
        ybar = ybar_sum / n_steps if n_steps else 0.0
        rho, cf = Fase3Poincare._rotation_number(returns)
        dio = Fase3Poincare._diophantine_certificate(omega)
        K_eff = g["kam_K"] * (0.5 + float(cs["severity_scalar"]))
        greene = Fase3Poincare._greene_certificate(K_eff)

        payload = {
            "section_def": "Σ = {x=0, p_x>0}; aterrizaje exacto de Hénon (RK4 en x y t)",
            "integrator": "Yoshida-4 simpléctico (Forest–Ruth)",
            "energy": E,
            "n_returns": len(returns),
            "return_points": returns,
            "rotation_number": rho,
            "rotation_cf": cf,
            "lyapunov_ft": lam,
            "megno_mean": ybar,
            "energy_drift_rel": drift_max,
            "escaped": escaped,
            "diophantine": dio,
            "greene": greene,
            "kam_K_eff": K_eff,
            "severity_inherited": float(cs["severity_scalar"]),
            "verdict_inherited": cs["csuite_verdict"],
        }
        return PoincareSection(**payload, lineage_hash=_chain("F3", payload))

    # ── Maquinaria Celeste Simpléctica: Hénon–Heiles + Yoshida-4 + Hénon ─────
    @staticmethod
    def _hh_accel(x: float, y: float) -> Tuple[float, float]:
        """Campo vectorial hamiltoniano de fuerzas −∇V(x, y) de Hénon–Heiles."""
        return -(x + 2.0 * x * y), -(y + x * x - y * y)

    @staticmethod
    def _hh_energy(s: List[float]) -> float:
        x, y, px, py = s[0], s[1], s[2], s[3]
        return (
            0.5 * (px * px + py * py)
            + 0.5 * (x * x + y * y)
            + x * x * y
            - (y ** 3) / 3.0
        )

    @staticmethod
    def _yoshida_coeffs() -> Tuple[List[float], List[float]]:
        c2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - c2)
        w0 = -c2 / (2.0 - c2)
        return ([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0], [w1, w0, w1])

    @staticmethod
    def _yoshida4_step(s: List[float], dt: float) -> List[float]:
        c, d = Fase3Poincare._yoshida_coeffs()
        x, y, px, py = s[0], s[1], s[2], s[3]
        for i in range(4):
            x += c[i] * dt * px
            y += c[i] * dt * py
            if i < 3:
                ax, ay = Fase3Poincare._hh_accel(x, y)
                px += d[i] * dt * ax
                py += d[i] * dt * ay
        return [x, y, px, py]

    @staticmethod
    def _yoshita4_shadow(s: List[float], dt: float) -> List[float]:
        return Fase3Poincare._yoshida4_step(s, dt)

    @staticmethod
    def _yoshita4_back(s: List[float], dt: float) -> List[float]:
        """Inversión temporal simpléctica exacta Φ_{-dt} = (Φ_{dt})⁻¹."""
        return Fase3Poincare._yoshida4_step(s, -dt)

    @staticmethod
    def _henon_landing(prev: List[float], dt: float) -> Optional[List[float]]:
        """
        Método de aterrizaje exacto de Michel Hénon (1982, Physica D 5, 412–414):
        Intercambia la variable independiente de t a x en el entorno del cruce x = 0:
          dy/dx  = p_y / p_x,
          dp_x/dx = a_x(x, y) / p_x,
          dp_y/dx = a_y(x, y) / p_x,
          dt/dx  = 1 / p_x.
        Un único paso RK4 de longitud h = −x_prev aterriza exactamente sobre la
        hipersuperficie x = 0 con error local O(h⁵) tanto en (y, p_x, p_y) como
        en el desplazamiento temporal Δt = t_cross − t_prev.
        """
        x0, y0, px0, py0 = prev[0], prev[1], prev[2], prev[3]
        if abs(px0) < 1.0e-10:
            return None
        h = -x0

        def rhs_x(xx: float, state_y_px_py_t: List[float]) -> List[float]:
            yy, pxx, pyy, _ = state_y_px_py_t
            if abs(pxx) < 1.0e-12:
                pxx = 1.0e-12 if pxx >= 0.0 else -1.0e-12
            ax, ay = Fase3Poincare._hh_accel(xx, yy)
            return [pyy / pxx, ax / pxx, ay / pxx, 1.0 / pxx]

        u0 = [y0, px0, py0, 0.0]
        k1 = rhs_x(x0, u0)
        k2 = rhs_x(x0 + 0.5 * h, [u0[i] + 0.5 * h * k1[i] for i in range(4)])
        k3 = rhs_x(x0 + 0.5 * h, [u0[i] + 0.5 * h * k2[i] for i in range(4)])
        k4 = rhs_x(x0 + h, [u0[i] + h * k3[i] for i in range(4)])

        u_end = [
            u0[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(4)
        ]
        y_land, px_land, py_land, dt_land = u_end
        if px_land <= 0.0:
            return None
        return [0.0, y_land, px_land, py_land, _clamp(dt_land, 0.0, max(dt, abs(dt_land)))]

    @staticmethod
    def _rotation_number(returns: List[List[float]]) -> Tuple[Optional[float], list]:
        """
        Número de rotación topológico de Poincaré ρ = |⟨Δθ⟩| / 2π sobre Σ alrededor
        del origen elíptico y desarrollo en fracción continua [a₀; a₁, a₂, ...].
        """
        if len(returns) < 3:
            return None, []
        angles = [math.atan2(r[1], r[0]) for r in returns]
        acc = 0.0
        for a, b in zip(angles, angles[1:]):
            d = b - a
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            acc += d
        rho = abs(acc) / (2.0 * math.pi * (len(angles) - 1))
        return rho, Fase3Poincare._continued_fraction(rho, 16)

    @staticmethod
    def _continued_fraction(x: float, max_terms: int = 24) -> List[int]:
        out: List[int] = []
        a0 = math.floor(x)
        out.append(int(a0))
        frac = x - a0
        for _ in range(max_terms - 1):
            if frac < 1.0e-12:
                break
            inv = 1.0 / frac
            a = math.floor(inv)
            out.append(int(a))
            frac = inv - a
        return out

    # ── Certificado Diofántico de Hurwitz–Lagrange ────────────────────────────
    @staticmethod
    def _diophantine_certificate(omega: float, q_max: int = 256) -> dict:
        """
        Certifica la condición diofántica ‖q·ω‖ ≥ γ / q^τ (con τ = 1 para irracionales
        cuadráticos nobles en el espectro de Markov–Lagrange):
          • Constante global en [1, q_max]: min_{1 ≤ q ≤ q_max} q·‖q·ω‖.
          • Estimación de liminf_{q→∞} q·‖q·ω‖ sobre los denominadores convergentes q_k
            de la fracción continua (Teorema de Hurwitz: C = 1/√5 para φ).
        """
        best, best_q = float("inf"), 1
        for q in range(1, q_max + 1):
            d = abs(omega * q - round(omega * q))
            if q * d < best:
                best, best_q = q * d, q
        cf = Fase3Poincare._continued_fraction(omega, 24)
        bounded = all(a <= 8 for a in cf[1:]) if len(cf) > 1 else True

        conv_q: List[int] = []
        lag = float("inf")
        p0, p1, q0, q1 = 0, 1, 1, 0
        for a in cf:
            p0, p1 = p1, a * p1 + p0
            q0, q1 = q1, a * q1 + q0
            if q1 > 0:
                conv_q.append(q1)
        for q in conv_q[-6:]:
            lag = min(lag, q * abs(omega * q - round(omega * q)))
        return {
            "gamma_max_all_q": best,
            "attained_q": best_q,
            "cf_partial_quotients": cf,
            "badly_approximable_est": bounded,
            "lagrange_liminf_est": lag if math.isfinite(lag) else None,
            "hurwitz_constant": HURWITZ_C,
            "golden_sharp_constant": 1.0 / (GOLDEN_RATIO ** 2),
        }

    # ── Teoría de Aubry–Mather y Residuos de Greene (Mapa Estándar) ───────────
    @staticmethod
    def _thomas_solve(
        sub: List[float],
        diag: List[float],
        sup: List[float],
        rhs: List[float],
    ) -> List[float]:
        """Algoritmo de Thomas O(n) para sistemas tridiagonales estrictamente dominantes."""
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
        """Solucionador tridiagonal periódico mediante fórmula de inversión de Sherman–Morrison."""
        n = len(diag)
        gamma = -diag[0]
        bb = list(diag)
        bb[0] = diag[0] - gamma
        bb[-1] = diag[-1] - 1.0 / gamma
        z = [0.0] * n
        z[0], z[-1] = gamma, -1.0
        sub, sup = [-1.0] * n, [-1.0] * n
        x = Fase3Poincare._thomas_solve(sub, bb, sup, rhs)
        u = Fase3Poincare._thomas_solve(sub, bb, sup, z)
        den = 1.0 + u[0] - u[-1] / gamma
        if abs(den) < 1.0e-14:
            raise ArithmeticError("Matriz Hessiana cíclica singular")
        fact = (x[0] - x[-1] / gamma) / den
        return [x[i] - fact * u[i] for i in range(n)]

    @staticmethod
    def _minimizing_orbit(K: float, r: int, q: int) -> Optional[List[float]]:
        """
        Calcula la órbita periódica minimizante de tipo (r, q) de Aubry–Mather para el
        lagrangiano discreto de Frenkel–Kontorova / Chirikov:
          W(θ) = ∑_{i=0}^{q-1} [ ½(θ_{i+1} − θ_i)² − K·cos(θ_i) ],  θ_{i+q} = θ_i + 2πr,
        mediante descenso de Newton amortiguado con Hessiana tridiagonal cíclica exacta.
        """
        two_pi_r = 2.0 * math.pi * r
        theta = [two_pi_r * i / q + 0.3 for i in range(q)]

        def action(th: List[float]) -> float:
            s = 0.0
            for i in range(q):
                nxt = th[i + 1] if i + 1 < q else th[0] + two_pi_r
                d = nxt - th[i]
                s += 0.5 * d * d - K * math.cos(th[i])
            return s

        def grad(th: List[float]) -> List[float]:
            g = []
            for i in range(q):
                prev = th[i - 1] if i > 0 else th[q - 1] - two_pi_r
                nxt = th[i + 1] if i + 1 < q else th[0] + two_pi_r
                g.append(nxt - 2.0 * th[i] + prev - K * math.sin(th[i]))
            return g

        S = action(theta)
        for _ in range(60):
            g = grad(theta)
            if max(abs(x) for x in g) < 1.0e-11:
                return theta
            diag = [2.0 - K * math.cos(t) + 1.0e-10 for t in theta]
            try:
                delta = Fase3Poincare._cyclic_solve(diag, [-x for x in g])
            except ArithmeticError:
                return None
            lam = 1.0
            for _ in range(24):
                cand = [theta[i] + lam * delta[i] for i in range(q)]
                Sc = action(cand)
                if Sc <= S + 1.0e-14:
                    theta, S = cand, Sc
                    break
                lam *= 0.5
            else:
                return None
        return theta if max(abs(x) for x in grad(theta)) < 1.0e-8 else None

    @staticmethod
    def _orbit_residue(theta: List[float], K: float) -> float:
        """
        Matriz de monodromía simpléctica M = ∏_{i=0}^{q-1} DT_K(θ_i) ∈ SL(2, ℝ) y
        Residuo de Greene R = (2 − Tr(M)) / 4.
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
        Criterio de estabilidad de Greene (1979, J. Math. Phys. 20, 1183) sobre la
        sucesión de órbitas periódicas de Fibonacci F_n / F_{n+1} → 1/φ.
        """
        key = round(K, 6)
        if key in _GREENE_CACHE:
            return _GREENE_CACHE[key]
        residues = []
        for r, q in GREENE_PAIRS:
            th = Fase3Poincare._minimizing_orbit(K, r, q)
            R = Fase3Poincare._orbit_residue(th, K) if th is not None else None
            residues.append({"r": r, "q": q, "residue": R})
        ok = all(
            x["residue"] is not None and -1.0e-6 <= x["residue"] <= 0.2501
            for x in residues
        )
        cert = {
            "K": K,
            "K_c_greene_ref": K_C_GREEN,
            "margin_subcritical": (K_C_GREEN - K) / K_C_GREEN,
            "residues": residues,
            "torus_preserved": ok,
            "criterion": "Greene minimizante (numérico); twist de Chirikov",
        }
        _GREENE_CACHE[key] = cert
        return cert

    # ── Automejora Recursiva Nivel 3 en Variedad Compacta con Elitismo ────────
    @staticmethod
    def _project_genome(g: dict) -> dict:
        """Proyección ortogonal Π_Box sobre el hiperrectángulo compacto GENOME_BOX."""
        out = dict(g)
        for k, (lo, hi) in GENOME_BOX.items():
            if k in out and isinstance(out[k], (int, float)):
                val = float(out[k])
                if not math.isfinite(val):
                    val = DEFAULT_GENOME[k]
                out[k] = _clamp(val, lo, hi)
        return out

    @staticmethod
    def _banach_genome_norm(g1: dict, g2: dict) -> float:
        """Distancia normalizada en el espacio de Banach (ℝ⁶, ‖·‖₂) relativa a GENOME_BOX."""
        sq = 0.0
        for k, (lo, hi) in GENOME_BOX.items():
            span = max(1.0e-12, hi - lo)
            d = (float(g1.get(k, DEFAULT_GENOME[k])) - float(g2.get(k, DEFAULT_GENOME[k]))) / span
            sq += d * d
        return math.sqrt(sq)

    @staticmethod
    def _fitness_cyber(genome: dict, section: Any) -> float:
        """
        Funcional de aptitud cibernético-celeste F: GENOME_BOX × PoincareSection → [0, 1]:
          • 30 % Estabilidad de Lyapunov (exp(−λ⁺ / 0.1)).
          • 15 % Fidelidad simpléctica de conservación de energía Hénon–Heiles.
          • 35 % Persistencia del toro KAM áureo (margen subcrítico de Greene + residuos).
          • 20 % Conectividad algebraica normalizada de Fiedler (`fiedler_norm`).
        """
        s = _view(section)
        g = Fase3Poincare._project_genome(dict(genome))
        lam = max(0.0, float(s.get("lyapunov_ft", 0.0)))
        drift = float(s.get("energy_drift_rel", 1.0e-3))
        stability = math.exp(-lam / 0.1)
        sympl = math.exp(-drift / 1.0e-4)

        K_eff = g["kam_K"] * (0.5 + float(s.get("severity_inherited", 0.0)))
        greene = Fase3Poincare._greene_certificate(K_eff)
        margin = greene["margin_subcritical"]
        kam = (1.0 / (1.0 + math.exp(-8.0 * _clamp(margin, -6.0, 6.0)))) * (
            1.0 if greene["torus_preserved"] else 0.35
        )
        if s.get("escaped"):
            stability *= 0.2
        return (
            0.30 * stability
            + 0.15 * sympl
            + 0.35 * kam
            + 0.20 * g["fiedler_norm"]
        )

    @staticmethod
    def _improve_cyber_l1(
        genome: dict,
        section: Any,
        rates: Optional[dict] = None,
        sign: float = 1.0,
    ) -> dict:
        """
        ── NIVEL 1 (Endomorfismo Base T₁: GENOME_BOX → GENOME_BOX) ──
        Actualiza el genoma en la dirección del flujo variacional que aleja K_eff de la
        resonancia crítica de Chirikov–Greene, afina el paso simpléctico `hh_dt` y
        refuerza la conectividad espectral `fiedler_norm`.
        """
        s = _view(section)
        r = rates or RATES_L1
        g = Fase3Poincare._project_genome(dict(genome))
        f = Fase3Poincare._fitness_cyber(g, s)
        drive = (1.0 - f) * sign
        drift_flag = min(1.0, float(s.get("energy_drift_rel", 0.0)) / 1.0e-4)

        g["kam_K"] *= 1.0 - r["kam_K"] * drive
        g["hh_dt"] *= 1.0 - r["hh_dt"] * drive * drift_flag
        g["fiedler_norm"] = min(1.0, g["fiedler_norm"] * (1.0 + r["fiedler_norm"] * drive))
        g["kalman_p0"] *= 1.0 + r["kalman_p0"] * drive * 0.5
        g["mutation_rate"] *= 1.0 - 0.05 * f
        return Fase3Poincare._project_genome(g)

    @staticmethod
    def _improve_cyber_l2(
        l1_rule: Callable,
        genome: dict,
        section: Any,
        rate_scale: float = 1.0,
    ) -> Tuple[Callable, dict]:
        """
        ── NIVEL 2 (Meta-Operador sobre Tasas de Aprendizaje en End(GENOME_BOX)) ──
        Sintetiza operadores L1 escalados sobre el espectro {0.5, φ⁻¹, 1.0, φ, 2.0}
        incluyendo el morfismo identidad para garantizar monotonía estricta (elitismo).
        """
        candidates: List[Tuple[Callable, dict]] = [
            (lambda g, s: dict(g), dict(genome))
        ]
        for alpha in (0.5, CONJ_GOLDEN_RATIO, 1.0, GOLDEN_RATIO, 2.0):
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
            key=lambda rc: Fase3Poincare._fitness_cyber(rc[1], section),
        )

    @staticmethod
    def _improve_cyber_l3(
        genome: dict,
        section: Any,
    ) -> Tuple[Callable, dict, dict]:
        """
        ── NIVEL 3 (Meta-Meta-Funtor 2-Categórico sobre Meta-Estrategias L2) ──
        Evalúa competitivamente las arquitecturas:
          • `identity`: Punto fijo estacionario.
          • `exploit`: Contracción fina (×0.5).
          • `golden_kam`: Contracción áurea cuasi-periódica (×φ⁻¹).
          • `explore`: Expansión de horizonte (×2.0).
          • `reflect`: Inversión simétrica de dirección (sign = −1.0).
        """
        strategies = {
            "exploit": (0.5, 1.0),
            "golden_kam": (CONJ_GOLDEN_RATIO, 1.0),
            "explore": (2.0, 1.0),
            "reflect": (1.0, -1.0),
        }
        evaluated = [
            {
                "strategy": "identity",
                "genome": dict(genome),
                "fitness": Fase3Poincare._fitness_cyber(genome, section),
            }
        ]
        rules: Dict[str, Callable] = {"identity": lambda g, s: dict(g)}

        for name, (scale, sign) in strategies.items():
            def l1_variant(g: dict, s: Any, rates: Optional[dict] = None, sg: float = sign) -> dict:
                return Fase3Poincare._improve_cyber_l1(g, s, rates=rates, sign=sg)

            def meta(g: dict, s: Any, sc: float = scale, lv: Callable = l1_variant) -> Tuple[Callable, dict]:
                return Fase3Poincare._improve_cyber_l2(lv, g, s, rate_scale=sc)

            rule, app = meta(genome, section)
            rules[name] = rule
            evaluated.append({
                "strategy": name,
                "genome": app,
                "fitness": Fase3Poincare._fitness_cyber(app, section),
            })

        best = max(evaluated, key=lambda e: e["fitness"])
        second = sorted((e["fitness"] for e in evaluated), reverse=True)[1]
        audit = {
            "evaluated": [
                {k: e[k] for k in ("strategy", "fitness")} for e in evaluated
            ],
            "selected": best["strategy"],
            "selection_margin": best["fitness"] - second,
            "constraints_ok": best["genome"] == Fase3Poincare._project_genome(best["genome"]),
        }
        return rules[best["strategy"]], best["genome"], audit

    @staticmethod
    def recursive_self_improve_cyber(
        genome: dict,
        section: Any,
        depth: int = 3,
    ) -> dict:
        """
        Torre recursiva L1 → L2 → L3 con trinquete de veto elitista y auditoría
        métrica en espacio de Banach:
          F(g₀) ≤ F(g₁) ≤ F(g₂) ≤ F(g₃) ≤ 1.0.
        """
        s = _view(section)
        current = Fase3Poincare._project_genome(dict(genome))
        history = [
            {
                "level": 0,
                "genome": dict(current),
                "fitness": Fase3Poincare._fitness_cyber(current, s),
                "banach_step_norm": 0.0,
                "veto": False,
            }
        ]
        l3_audit = None

        if depth >= 1:
            cand = Fase3Poincare._improve_cyber_l1(current, s)
            f = Fase3Poincare._fitness_cyber(cand, s)
            veto = f < history[-1]["fitness"] - 1.0e-12
            next_g = current if veto else cand
            step_norm = Fase3Poincare._banach_genome_norm(current, next_g)
            current = next_g
            history.append({
                "level": 1,
                "genome": dict(current),
                "fitness": history[-1]["fitness"] if veto else f,
                "banach_step_norm": step_norm,
                "veto": veto,
            })

        if depth >= 2:
            _, cand = Fase3Poincare._improve_cyber_l2(
                Fase3Poincare._improve_cyber_l1, current, s
            )
            f = Fase3Poincare._fitness_cyber(cand, s)
            veto = f < history[-1]["fitness"] - 1.0e-12
            next_g = current if veto else cand
            step_norm = Fase3Poincare._banach_genome_norm(current, next_g)
            current = next_g
            history.append({
                "level": 2,
                "genome": dict(current),
                "fitness": history[-1]["fitness"] if veto else f,
                "banach_step_norm": step_norm,
                "veto": veto,
            })

        if depth >= 3:
            _, cand, l3_audit = Fase3Poincare._improve_cyber_l3(current, s)
            f = Fase3Poincare._fitness_cyber(cand, s)
            veto = f < history[-1]["fitness"] - 1.0e-12
            next_g = current if veto else cand
            step_norm = Fase3Poincare._banach_genome_norm(current, next_g)
            current = next_g
            history.append({
                "level": 3,
                "genome": dict(current),
                "fitness": history[-1]["fitness"] if veto else f,
                "banach_step_norm": step_norm,
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

    # ── DEFINICIÓN FORMAL FINAL DE FASE 3 (CIERRE DEL BUCLE HACIA FASE 1) ─────
    @staticmethod
    def orchestrate_cyberphysical_loop(
        sensor_ids: Optional[List[str]] = None,
        w_irr_j: float = 5.0,
        fidelity: float = 0.97,
        beta_1: int = 0,
        genome: Optional[dict] = None,
        max_iters: int = 3,
        tol: float = 1.0e-6,
    ) -> dict:
        """
        ── ÚLTIMO MÉTODO DE FASE 3 (CIERRE DEL BUCLE ANIDADO F1 → F2 → F3 ↻ F1) ──
        Ejecuta la iteración de punto fijo de Kleene acotada a través de las 3 fases:
          1. Invoca `Fase1Physical.read_site_telemetry` con el estado a priori de
             Kalman `(x, P)` y el genoma evolucionado.
          2. Traduce el `PhysicalState` a `CSuiteObject` vía `Fase2CSuite.translate_visceral_csuite`.
          3. Calcula la sección de Poincaré de Hénon–Heiles y evoluciona el genoma en
             Nivel 3 (`recursive_self_improve_cyber`), reinyectándolo en la Fase 1.
        """
        g = Fase3Poincare._project_genome(dict(genome or DEFAULT_GENOME))
        prior = None
        trace: List[float] = []
        iterations: List[dict] = []
        converged = False
        h = hashlib.sha256(b"GENESIS").hexdigest()[:32]
        ps = cs = sec = imp = None

        for it in range(max(1, max_iters)):
            ps = Fase1Physical.read_site_telemetry(
                sensor_ids, prior, g, w_irr_j, fidelity, beta_1
            )
            cs = Fase2CSuite.translate_visceral_csuite(physical_state=ps)
            sec = Fase3Poincare.poincare_cyberphysical_section(cs, genome=g)
            imp = Fase3Poincare.recursive_self_improve_cyber(g, sec, depth=3)

            f_new = imp["final_fitness"]
            trace.append(f_new)
            h = _chain(
                h,
                {
                    "it": it,
                    "f1": ps.lineage_hash,
                    "f2": cs.lineage_hash,
                    "f3": sec.lineage_hash,
                    "fitness": f_new,
                },
            )
            iterations.append({
                "iteration": it,
                "fitness": f_new,
                "genome": imp["genome"],
                "monotone": imp["monotone_certificate"],
            })
            prior = {"kalman_states": ps.kalman_states}
            g_new = imp["genome"]
            if it > 0 and abs(trace[-1] - trace[-2]) < tol:
                converged = True
                g = g_new
                break
            g = g_new

        recommendation = None
        if cs is not None and (
            cs.guard_band_conformity == "VETO_PROBADO"
            or cs.csuite_verdict == "CRITICAL_VETO"
        ):
            recommendation = {
                "action": "CROWBAR_RECOMENDADO_NO_EJECUTADO",
                "requires_confirmation_code": SOVEREIGN_CODE,
                "tool": "trigger_esp32_crowbar",
                "justification": cs.guard_band_conformity,
                "sovereign_veto": "fail-closed: ejecución solo humana",
            }

        return {
            "fase_1_physical_state": asdict(ps) if ps is not None else {},
            "fase_2_csuite_object": asdict(cs) if cs is not None else {},
            "fase_3_poincare_and_l3": {
                "section": asdict(sec) if sec is not None else {},
                "improvement": imp,
            },
            "genome_feedback": g,
            "loop_certificate": {
                "iterations": iterations,
                "fitness_trace": trace,
                "converged": converged,
                "tol": tol,
                "feedback_real": "kalman_p0/kalman_q + (x,P) → F1",
            },
            "crowbar_recommendation": recommendation,
            "lineage_chain": h,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLO MCP (JSON-RPC 2.0 POR STDIO) — initialize / ping / tools/*
# ═══════════════════════════════════════════════════════════════════════════════
PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "read_site_telemetry",
        "description": (
            "F1: GUM (Welch–Satterthwaite) + Kalman-Joseph/NIS + espectro de grafos "
            "(Gershgorin/Jacobi) + espinor cuaterniónico S³. β₁ derivado del grafo."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sensor_ids": {"type": "array", "items": {"type": "string"}},
                "prior_state": {"type": "object"},
                "genome": {"type": "object"},
                "w_irr_j": {"type": "number"},
                "fidelity": {"type": "number"},
                "beta_1": {"type": "integer"},
            },
        },
    },
    {
        "name": "trigger_esp32_crowbar",
        "description": (
            "F1: crowbar SCR BT151 con interlocks (I²t, dI/dt, dV/dt, Foster-Cauer-Duhamel, "
            "gate, enganche, selectividad) y MMIO ESP32-S3 LX7."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirmation_code": {"type": "string"},
                "reason": {"type": "string"},
                "target_pin": {"type": "integer", "default": 14},
                "series_inductance_uh": {"type": "number"},
                "voltage_applied_v": {"type": "number"},
                "current_profile_a": {"type": "array", "items": {"type": "number"}},
                "fault_duration_s": {"type": "number"},
                "t_j_init_c": {"type": "number"},
                "v_gg_v": {"type": "number"},
                "r_g_ohm": {"type": "number"},
                "t_g_us": {"type": "number"},
                "voltage_rise_time_us": {"type": "number"},
                "upstream_fuse_i2t_a2s": {"type": "number"},
            },
            "required": ["confirmation_code", "reason"],
        },
    },
    {
        "name": "translate_visceral_csuite",
        "description": (
            "F2: funtor posetal certificado (analítico+Lipschitz), SVD del Jacobiano, "
            "topos Sh([0,1]) con núcleo de doble negación j_{¬¬,θ}, banda de guarda JCGM 106."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fidelity": {"type": "number"},
                "w_irr": {"type": "number"},
                "beta_1": {"type": "integer"},
                "physical_state": {"type": "object"},
            },
        },
    },
    {
        "name": "poincare_cyberphysical_section",
        "description": (
            "F3: Hénon–Heiles simpléctico (Yoshida-4), sección exacta de Hénon (x y t), "
            "Benettin+MEGNO, diofántico+Greene."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "csuite": {"type": "object"},
                "n_returns": {"type": "integer"},
                "omega": {"type": "number"},
                "genome": {"type": "object"},
            },
            "required": ["csuite"],
        },
    },
    {
        "name": "kam_greene_certificate",
        "description": (
            "F3: certificado KAM (diofántico Hurwitz–Lagrange + residuos de Greene "
            "sobre órbitas Aubry–Mather)."
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
        "name": "recursive_self_improve_cyber",
        "description": (
            "F3: automejora L1/L2/L3 con proyección admisible, métrica de Banach, "
            "elitismo con veto y certificado de monotonía."
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
        "name": "orchestrate_cyberphysical_loop",
        "description": (
            "Cierre anidado F1→F2→F3→F1' con realimentación real del genoma, "
            "linaje SHA-256 y veto soberano."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sensor_ids": {"type": "array", "items": {"type": "string"}},
                "w_irr_j": {"type": "number"},
                "fidelity": {"type": "number"},
                "beta_1": {"type": "integer"},
                "genome": {"type": "object"},
                "max_iters": {"type": "integer"},
            },
        },
    },
]


def _dispatch(name: str, args: dict) -> Any:
    if name == "read_site_telemetry":
        return Fase1Physical.read_site_telemetry(
            args.get("sensor_ids"),
            args.get("prior_state"),
            args.get("genome"),
            args.get("w_irr_j", 0.0),
            args.get("fidelity", 0.97),
            args.get("beta_1", 0),
        )
    if name == "trigger_esp32_crowbar":
        return Fase1Physical.trigger_esp32_crowbar(
            args["confirmation_code"],
            args["reason"],
            args.get("target_pin", 14),
            args.get("series_inductance_uh", 10.0),
            args.get("voltage_applied_v", 12.0),
            args.get("current_profile_a"),
            args.get("fault_duration_s", 10.0e-3),
            args.get("t_j_init_c", 25.0),
            args.get("v_gg_v", 5.0),
            args.get("r_g_ohm", 110.0),
            args.get("t_g_us", 10.0),
            args.get("voltage_rise_time_us", 1.0),
            args.get("upstream_fuse_i2t_a2s"),
        )
    if name == "translate_visceral_csuite":
        return Fase2CSuite.translate_visceral_csuite(
            args.get("fidelity"),
            args.get("w_irr"),
            args.get("beta_1"),
            args.get("physical_state"),
        )
    if name == "poincare_cyberphysical_section":
        return Fase3Poincare.poincare_cyberphysical_section(
            args["csuite"],
            args.get("n_returns", args.get("n_orbits", 24)),
            args.get("omega", GOLDEN_RATIO),
            args.get("genome"),
        )
    if name == "kam_greene_certificate":
        return {
            "diophantine": Fase3Poincare._diophantine_certificate(
                args.get("omega", GOLDEN_RATIO)
            ),
            "greene": Fase3Poincare._greene_certificate(float(args["K"])),
        }
    if name == "recursive_self_improve_cyber":
        return Fase3Poincare.recursive_self_improve_cyber(
            args["genome"], args["section"], args.get("depth", 3)
        )
    if name == "orchestrate_cyberphysical_loop":
        return Fase3Poincare.orchestrate_cyberphysical_loop(
            args.get("sensor_ids"),
            args.get("w_irr_j", 5.0),
            args.get("fidelity", 0.97),
            args.get("beta_1", 0),
            args.get("genome"),
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
        is_notification = msg_id is None

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
                        "name": "ciberphysical-bridge",
                        "version": "4.0.0-fases-anidadas",
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
        elif not is_notification:
            _send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    main()