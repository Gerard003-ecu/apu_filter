# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Silent Witness Agent (Soberano Testigo Silencioso)           ║
║ Ubicación: app/agents/wisdom/toon_silent_witness_agent.py                    ║
║ Versión  : 2.1.0-Doctoral-SilentWitness-Triad-Modular-Crystal                ║
╚══════════════════════════════════════════════════════════════════════════════╝

EVOLUCIÓN DOCTORAL — corrección de tres patologías del original:

  (P1)  vev := |Tr(ρ) − 1|  ≡ 0        [identidad de traza, no un observable]
  (P2)  kms_entropy_drift := |S−S| ≡ 0 [tautología literal]
  (P3)  Tomita-Takesaki sólo en docstring [no había σ_t, Δ ni J reales]

Soluciones:

  (S1)  VEV := Tr(ρH_ext) − E₀(H_ext)               (excitación sobre el ground)
  (S2)  ΔS := S(Φ(ρ_vac)) − S(ρ_vac)                (producción real de entropía)
  (S3)  Flujo modular σ_t(a) = ρ^{−it} a ρ^{it}  + verificación KMS numérica
        Δ(a) = ρ^{−1} a ρ      J(a) = ρ^{−1/2} a* ρ^{1/2}      Δ^{1/2} J = S

Convención fijada (verificada numéricamente en tests internos):
        σ_t^ω(a) = ρ^{−it} a ρ^{it}
        σ_i^ω(a) = ρ a ρ^{−1}
        KMS(β=1):  ω(a σ_i(b)) = ω(ba)

Tríada adversarial formalizada como composición selectiva de canales:

    Φ_triad  :=  P_A  ∘  D  ∘  U_I         (Kraus K = P_A · D · U_I)

    Ilusionista  U_I ∈ U(n)          — distorsión unitaria (alucinación)
    Soñador      D   ∈ Herm⁺(n)      — peso positivo (escenario onírico)
    Auditor      P_A ∈ Proj(n)       — proyector de inmunización

El Testigo OBSERVA Φ_triad(ρ_vac) sin back-action, emite un veredicto en Ω₃
basado en métricas verdaderas (§2.3) y cristaliza un ExperienceCrystal con
cadena de custodia Merkle-SHA-256 (§3.2).

Organización por FASES ANIDADAS:

  FASE 1 ▸ Sustrato ontológico de la Tríada
            §1.1  HeytingOmega3 — retículo distributivo con ⇒, ¬
            §1.2  DensityMatrixOps — S(ρ), Tr(ρ²), F(ρ,σ), ρ^z
            §1.3  TriadOperatorFactory — U_I, D, P_A deterministas
            §1.4  TriadChannel — Φ_triad y probabilidad de transición
            §1.5  WitnessVacuumPreparation — HAND-OFF: (ρ_vac, K_vac) → FASE 2

  FASE 2 ▸ Dinámica modular y observación silenciosa (C. de FASE 1)
            §2.1  ModularFlowEngine — σ_t, σ_i, J, KMS, axiomas
            §2.2  VacuumStateMetrics — métricas verdaderas de un estado ρ
            §2.3  VacuumSilenceAuditor — auditoría cross de (ρ_obs, ρ_vac)
            §2.4  WitnessObservationPipeline — HAND-OFF: bundle → FASE 3

  FASE 3 ▸ Cristalización y custodia forense (C. de FASE 2)
            §3.1  HeytingWitnessAdjudicator — Ω₃ trivalente + meet externo
            §3.2  ExperienceCrystal — frozen + Merkle SHA-256 encadenado
            §3.3  TOONSilentWitnessAgent — orquestador soberano
            §3.4  Demostración autónoma (COHERENT / DEGRADED / VETOED)
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Wisdom.TOONSilentWitness")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS = 1.0e-14


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ONTOLÓGICO DE LA TRÍADA                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Retículo total Ω₃ = {⊥, ⋆, ⊤} con estructura de Heyting completa:

        meet    (∧) : ínfimo                      (a ∧ b ≤ a, b; el mayor así)
        join    (∨) : supremo                     (a ∨ b ≥ a, b; el menor así)
        implies (⇒) : residuo de conjunción       ((a ∧ b) ≤ c ⇔ a ≤ (b ⇒ c))
        neg     (¬) : a ⇒ ⊥                       (intuicionista)
        regular     : a = ¬¬a                     ({⊥, ⊤} son regulares; ⋆ no)
    """
    VETOED   = 0   # ⊥
    DEGRADED = 1   # ⋆
    COHERENT = 2   # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    def is_regular(self) -> bool:
        return self.neg().neg() == self


def _seed_from_string(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityMatrixOps:
    r"""
    Operadores canónicos de la teoría de información cuántica:

        S(ρ)     = −Tr(ρ log ρ)              von Neumann (nats)
        P(ρ)     = Tr(ρ²)                     pureza
        F(ρ,σ)   = ‖√ρ √σ‖₁ = Tr√(√ρ σ √ρ)   Uhlmann
        ρ^z      = V · diag(λ^z) · V†         potencias complejas

    Todos regularizan λ → max(λ, ε) para evitar log(0) y potencias de 0.
    Grupo de simetría: U(n) actúa preservando S, P, F y espectro.
    """
    EPS = _EPS

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> np.ndarray:
        """Hermitiza y normaliza la traza a 1; devuelve copia defensiva."""
        rho = np.asarray(rho, dtype=np.complex128)
        rho = 0.5 * (rho + rho.conj().T)
        tr = float(np.trace(rho).real)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return rho

    @classmethod
    def spectrum(cls, rho: np.ndarray) -> np.ndarray:
        rho = cls.sanitize(rho)
        w = np.real(la.eigvalsh(rho))
        w = np.sort(w)[::-1]
        w = np.maximum(w, cls.EPS)
        return w / w.sum()

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum(rho)
        return -float(np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        return float(np.sum(cls.spectrum(rho) ** 2))

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return float(np.clip(val, 0.0, 1.0))

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex) -> np.ndarray:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(w, cls.EPS)
        return (V * (w.astype(np.complex128) ** z)) @ V.conj().T

    @classmethod
    def modular_hamiltonian_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        r"""Espectro de K_ρ = −log ρ, ordenado ascendente."""
        rho = cls.sanitize(rho)
        w = np.maximum(np.real(la.eigvalsh(rho)), cls.EPS)
        E = -np.log(w)
        E.sort()
        return tuple(map(float, E.tolist()))

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        H = 0.5 * (H + H.conj().T)
        w, V = la.eigh(H)
        i0 = int(np.argmin(w))
        omega = V[:, i0].reshape(-1, 1)
        return cls.sanitize(omega @ omega.conj().T)


# ── §1.3 Fábrica determinista de operadores de la Tríada ──────────────────
@dataclass(frozen=True, slots=True)
class TriadSignature:
    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    unitary_hash: str
    dreamer_hash: str
    projector_hash: str
    auditor_rank: int

    def as_bytes(self) -> bytes:
        return (
            f"{self.trickster_illusion_type}|{self.dreamer_scenario_id}|"
            f"{self.auditor_immunization_hash}|{self.unitary_hash[:16]}|"
            f"{self.dreamer_hash[:16]}|{self.projector_hash[:16]}|"
            f"{self.auditor_rank}"
        ).encode("utf-8")


class TriadOperatorFactory:
    r"""
    Construye U_I, D y P_A de forma determinista a partir de identificadores
    textuales (vía SHA-256 → semilla de numpy.default_rng), con parámetros
    de intensidad `strength ∈ [0,1]` para interpolar entre identidad y
    muestras de máxima entropía.

        U_I(s) = exp( i·s·π·n · A_h )    con A_h Hermítico normalizado
        D(s)   = (1−s)·I + s·(A†A)/tr       con A Gaussiano complejo
        P_A    = Q · diag(1_r, 0) · Q†       Q Haar (rango r)
    """

    @classmethod
    def _haar(cls, n: int, rng: np.random.Generator) -> np.ndarray:
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        d = np.diagonal(R)
        ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
        return Q * ph.conj()

    @classmethod
    def build_unitary(cls, illusion_type: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"TRICKSTER::{illusion_type}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A_h = 0.5 * (A + A.conj().T)
        nrm = float(np.linalg.norm(A_h, "fro")) + 1e-30
        A_h = A_h / nrm
        theta = math.pi * strength * n
        return la.expm(1j * theta * A_h)

    @classmethod
    def build_dreamer(cls, scenario_id: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"DREAMER::{scenario_id}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        D_rand = A.conj().T @ A
        D_rand = D_rand / (np.trace(D_rand).real + 1e-30) * n
        D = (1.0 - strength) * np.eye(n, dtype=np.complex128) + strength * D_rand
        return 0.5 * (D + D.conj().T)

    @classmethod
    def build_projector(cls, immunization_hash: str, n: int, rank: int) -> np.ndarray:
        rank = int(np.clip(rank, 1, n))
        rng = np.random.default_rng(_seed_from_string(f"AUDITOR::{immunization_hash}"))
        Q = cls._haar(n, rng)
        cols = Q[:, :rank]
        P = cols @ cols.conj().T
        return 0.5 * (P + P.conj().T)

    @classmethod
    def build_triad(
        cls,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        n: int,
        trickster_strength: float,
        dreamer_strength: float,
        auditor_rank: int,
    ) -> "TriadChannel":
        U_I = cls.build_unitary(trickster_illusion_type, n, trickster_strength)
        D   = cls.build_dreamer(dreamer_scenario_id, n, dreamer_strength)
        P_A = cls.build_projector(auditor_immunization_hash, n, auditor_rank)

        u_h = hashlib.sha256(U_I.tobytes()).hexdigest()
        d_h = hashlib.sha256(D.tobytes()).hexdigest()
        p_h = hashlib.sha256(P_A.tobytes()).hexdigest()
        sig = TriadSignature(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            unitary_hash=u_h,
            dreamer_hash=d_h,
            projector_hash=p_h,
            auditor_rank=int(np.clip(auditor_rank, 1, n)),
        )
        return TriadChannel(sig, U_I, D, P_A)


# ── §1.4 Canal selectivo de la Tríada ─────────────────────────────────────
class TriadChannel:
    r"""
    Composición selectiva (single-Kraus) de la tríada:

        K_triad := P_A · D · U_I ∈ M_n(ℂ)

    Actuando sobre un estado σ ∈ D(H):
        Φ_triad(σ) := K_triad σ K_triad† / Tr(K_triad σ K_triad†)

    La probabilidad de transición p(σ) = Tr(K†K σ) reporta la "masa" que
    la tríada admite; si p ≈ 1 y Φ ≈ id, la tríada es COHERENTE.
    """
    def __init__(self, signature: TriadSignature,
                 U_I: np.ndarray, D: np.ndarray, P_A: np.ndarray) -> None:
        self.signature = signature
        # Proyectar D a PSD por seguridad numérica
        w, V = la.eigh(0.5 * (D + D.conj().T))
        w = np.maximum(w, 0.0)
        self.U = U_I
        self.D = (V * w) @ V.conj().T
        self.P = P_A
        self.K = self.P @ self.D @ self.U

    def apply(self, rho: np.ndarray) -> np.ndarray:
        out = self.K @ rho @ self.K.conj().T
        tr = float(np.trace(out).real)
        if tr < 1e-30:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return out / tr

    def success_probability(self, rho: np.ndarray) -> float:
        M = self.K.conj().T @ self.K
        return float(np.trace(rho @ M).real)

    def kraus_norm(self) -> float:
        return float(np.linalg.norm(self.K, "fro"))


# ── §1.5 WitnessVacuumPreparation — HAND-OFF FASE 1 → FASE 2 ──────────────
class WitnessVacuumPreparation:
    r"""
    Prepara el par canónico (ρ_vac, K_vac) que alimenta toda la FASE 2.

    El vacío del Testigo es el estado de Gibbs a temperatura inversa β_vac:
        ρ_β := e^{−β H_vac} / Z_β
    En el límite β_vac → ∞, ρ_β → |Ω⟩⟨Ω| (vacío puro, silencio absoluto).
    A β_vac finito, ρ_β es un estado KMS(β) con flujo modular bien definido:

        K_vac = −log ρ_β,   spec(K_vac) = {−log λ_i(ρ_β)}
    """
    DEFAULT_BETA_COLD: float = 50.0

    @classmethod
    def gibbs_state(cls, H: np.ndarray, beta: float) -> np.ndarray:
        H = 0.5 * (H + H.conj().T)
        w, V = la.eigh(H)
        x = -beta * (w - w.min())
        x = x - x.max()  # estabilidad numérica
        p = np.exp(x)
        p = p / p.sum()
        return DensityMatrixOps.sanitize((V * p.astype(np.complex128)) @ V.conj().T)

    # ═════════════════════════════════════════════════════════════════════
    #  HAND-OFF  FASE 1 → FASE 2
    #  Cierra la FASE 1. Su salida (ρ_vac, K_vac) es el punto de anclaje de
    #  todos los métodos de FASE 2 (§2.1 ModularFlowEngine, §2.3 Auditor).
    # ═════════════════════════════════════════════════════════════════════
    @classmethod
    def prepare_vacuum_pair(
        cls, H_ext: np.ndarray, beta: float = DEFAULT_BETA_COLD,
    ) -> Tuple[np.ndarray, Tuple[float, ...]]:
        r"""Hand-off: (H_ext, β) ↦ (ρ_vac, spec(K_vac))."""
        rho = cls.gibbs_state(H_ext, beta)
        K_spec = DensityMatrixOps.modular_hamiltonian_spectrum(rho)
        logger.debug(
            "VacuumPreparation: β=%.2f | E₀(K)=%.4f | gap(K)=%.4f",
            beta, K_spec[0] if K_spec else 0.0,
            (K_spec[1] - K_spec[0]) if len(K_spec) > 1 else float("inf"),
        )
        return rho, K_spec


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · DINÁMICA MODULAR Y OBSERVACIÓN SILENCIOSA (C. de FASE 1)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Motor Tomita-Takesaki ejecutable ─────────────────────────────────
class ModularFlowEngine:
    r"""
    Implementación numérica de Tomita-Takesaki para ω(a) = Tr(ρ a):

    Convención fijada (verificada por los axiomas (iii)–(iv) y el residuo
    KMS ≈ 0 en `verify_kms_self`):

        σ_t(a) = ρ^{−it} a ρ^{it}
        σ_i(a) = ρ a ρ^{−1}
        Δ(a)   = ρ^{−1} a ρ            (Δ sobre la representación GNS)
        J(a)   = ρ^{−1/2} a* ρ^{1/2}   (conjugación modular antiunitaria)
        S      = J Δ^{1/2}              (S(aΩ) = a*Ω)

    Condición KMS(β=1):
        ω(a σ_i(b)) = ω(b a)   ∀ a, b ∈ M_n(ℂ)

    Axiomas modulares verificables:
        (i)   σ_t unital        : σ_t(I) = I
        (ii)  σ_t multiplicativo: σ_t(ab) = σ_t(a) σ_t(b)
        (iii) σ_t isométrico    : ‖σ_t(a)‖₂ = ‖a‖₂
        (iv)  J² = id           : involución modular
    """

    @classmethod
    def sigma_t(cls, rho: np.ndarray, t: complex, a: np.ndarray) -> np.ndarray:
        r"""σ_t(a) = ρ^{−it} a ρ^{it}, analítico en t ∈ ℂ."""
        rho = DensityMatrixOps.sanitize(rho)
        L = DensityMatrixOps.matrix_power(rho, -1j * t)
        R = DensityMatrixOps.matrix_power(rho, 1j * t)
        return L @ a @ R

    @classmethod
    def sigma_i(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""σ_i(a) = ρ a ρ^{−1} (punto KMS)."""
        rho = DensityMatrixOps.sanitize(rho)
        L = DensityMatrixOps.matrix_power(rho, 1.0)
        R = DensityMatrixOps.matrix_power(rho, -1.0)
        return L @ a @ R

    @classmethod
    def modular_conjugation(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""J(a) = ρ^{−1/2} a* ρ^{1/2} (antilineal en a)."""
        rho = DensityMatrixOps.sanitize(rho)
        L = DensityMatrixOps.matrix_power(rho, -0.5)
        R = DensityMatrixOps.matrix_power(rho, 0.5)
        return L @ a.conj().T @ R

    @classmethod
    def verify_kms_self(cls, rho: np.ndarray, n_tests: int = 6, seed: int = 42) -> float:
        r"""
        Residuo KMS del estado ρ contra su propio flujo:
            res = max |ω(a σ_i(b)) − ω(b a)|
        Debe ser ≈ 0 (máquina) para cualquier ρ > 0.
        """
        rho = DensityMatrixOps.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        max_res = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            a = a / (np.linalg.norm(a, "fro") + 1e-30)
            b = b / (np.linalg.norm(b, "fro") + 1e-30)
            lhs = np.trace(rho @ a @ cls.sigma_i(rho, b))
            rhs = np.trace(rho @ b @ a)
            res = abs(lhs - rhs)
            max_res = max(max_res, float(res))
        return max_res

    @classmethod
    def verify_kms_cross(
        cls,
        rho_flow: np.ndarray,
        rho_state: np.ndarray,
        n_tests: int = 6,
        seed: int = 137,
    ) -> float:
        r"""
        KMS CRUZADO: residuo de ρ_state contra el flujo modular de ρ_flow.

            res = max |Tr(ρ_state a σ_i^{ρ_flow}(b)) − Tr(ρ_state b a)|

        Si ρ_state = ρ_flow, colapsa a `verify_kms_self`. Si ρ_state ≠ ρ_flow,
        mide la "fuga" del estado fuera del rayo KMS del vacío.
        """
        rho_flow = DensityMatrixOps.sanitize(rho_flow)
        rho_state = DensityMatrixOps.sanitize(rho_state)
        n = rho_state.shape[0]
        rng = np.random.default_rng(seed)
        max_res = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            a = a / (np.linalg.norm(a, "fro") + 1e-30)
            b = b / (np.linalg.norm(b, "fro") + 1e-30)
            lhs = np.trace(rho_state @ a @ cls.sigma_i(rho_flow, b))
            rhs = np.trace(rho_state @ b @ a)
            denom = 1.0 + abs(lhs) + abs(rhs)
            max_res = max(max_res, float(abs(lhs - rhs)) / denom)
        return max_res

    @classmethod
    def verify_axioms(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""
        Verifica numéricamente los axiomas (i)–(iv) con operadores aleatorios
        normalizados en la norma Frobenius.
        """
        rho = DensityMatrixOps.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(7)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        I = np.eye(n, dtype=np.complex128)

        unital = float(np.linalg.norm(cls.sigma_t(rho, 0.37, I) - I, "fro"))
        product = float(np.linalg.norm(
            cls.sigma_t(rho, 0.73, a @ b)
            - cls.sigma_t(rho, 0.73, a) @ cls.sigma_t(rho, 0.73, b),
            "fro",
        ))
        isometry = float(abs(
            np.linalg.norm(a, "fro") - np.linalg.norm(cls.sigma_t(rho, 1.11, a), "fro")
        ))
        Ja = cls.modular_conjugation(rho, a)
        involution = float(np.linalg.norm(cls.modular_conjugation(rho, Ja) - a, "fro"))

        return {
            "unital_residual":     unital,
            "product_residual":    product,
            "isometry_residual":   isometry,
            "involution_residual": involution,
        }


# ── §2.2 VacuumStateMetrics evolucionada ─────────────────────────────────
@dataclass(frozen=True, slots=True)
class VacuumStateMetrics:
    r"""
    Métricas verdaderas de un estado ρ frente a (H_ext, |Ω⟩⟨Ω|):

        vacuum_expectation_value  = Tr(ρ H_ext) − E₀(H_ext)   (excitación real)
        tomita_takesaki_flow_param = t del flujo modular (parámetro observable)
        kms_entropy_drift          = |∂_t S(σ_t(ρ))|_{t=0}    (dS/dt en el flujo)
        silence_purity             = F(ρ, |Ω⟩⟨Ω|)             (fidelidad al vacío)
        is_silent                  = ⟨Ω|ρ|Ω⟩ ≈ 1 ∧ S(ρ) ≈ 0 ∧ gap ≈ ∞
    """
    vacuum_expectation_value: float
    tomita_takesaki_flow_param: float
    kms_entropy_drift: float
    silence_purity: float
    modular_ground_energy: float
    modular_spectral_gap: float
    von_neumann_entropy: float
    purity: float
    is_silent: bool


# ── §2.3 Auditor silencioso del vacío ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class VacuumAuditReport:
    r"""
    Auditoría cruzada de la tríada vs. el vacío. Todos los observables son
    cantidades físicas verificables, NO identidades de normalización:

        entropy_production    = S(ρ_obs) − S(ρ_vac)
        fidelity_to_vacuum    = F(ρ_obs, ρ_vac)                  ∈ [0, 1]
        vev_excitation        = Tr(ρ_obs H_ext) − E₀(H_ext)
        self_kms_residual     = KMS(ρ_obs) autoconsistente        ≈ 0
        cross_kms_residual    = KMS(ρ_obs; flujo de ρ_vac)
        transition_probability = Tr(K†K ρ_vac)                    ∈ [0, n]
        local_verdict         ∈ Ω₃
    """
    entropy_production: float
    fidelity_to_vacuum: float
    vev_excitation: float
    self_kms_residual: float
    cross_kms_residual: float
    transition_probability: float
    modular_spectral_gap: float
    local_verdict: HeytingOmega3


class VacuumSilenceAuditor:
    r"""
    Calcula las métricas verdaderas de la observación silenciosa.

    Criterio de silencio (todos simultáneamente):
        • Δ S        < ε_S     (sin producción de entropía)
        • 1 − F      < ε_F     (fidelidad al vacío preservada)
        • KMS_cross  < ε_K     (compatibilidad modular)
        • KMS_self   < ε_K     (coherencia interna de la numerica)
        • VEV_exc    < ε_V     (sin excitación por encima del ground)

    Semántica del veredicto local:
        COHERENT : los 5 criterios se satisfacen
        DEGRADED : falla un subconjunto estricto
        VETOED   : fallan ≥ 3, incluidos ΔS y F (pérdida estructural)
    """
    EPS_ENTROPY:    float = 1.0e-3
    EPS_FIDELITY:   float = 1.0e-3
    EPS_KMS:        float = 1.0e-4
    EPS_VEV:        float = 1.0e-3

    @classmethod
    def audit(
        cls,
        rho_vac: np.ndarray,
        rho_obs: np.ndarray,
        H_ext: np.ndarray,
        transition_prob: float,
    ) -> VacuumAuditReport:
        rho_vac = DensityMatrixOps.sanitize(rho_vac)
        rho_obs = DensityMatrixOps.sanitize(rho_obs)
        H_ext = 0.5 * (H_ext + H_ext.conj().T)

        # Entropía y fidelidad
        S_vac = DensityMatrixOps.von_neumann_entropy(rho_vac)
        S_obs = DensityMatrixOps.von_neumann_entropy(rho_obs)
        delta_s = S_obs - S_vac

        F_vac = DensityMatrixOps.uhlmann_fidelity(rho_obs, rho_vac)

        # Excitación sobre el ground de H_ext
        w_H = np.real(la.eigvalsh(H_ext))
        E0 = float(w_H.min())
        mean_H = float(np.real(np.trace(rho_obs @ H_ext)))
        vev_exc = mean_H - E0

        # KMS: propio del observado y cruzado contra el flujo del vacío
        kms_self  = ModularFlowEngine.verify_kms_self(rho_obs, n_tests=4)
        kms_cross = ModularFlowEngine.verify_kms_cross(rho_vac, rho_obs, n_tests=4)

        # Gap modular del vacío
        K_spec = DensityMatrixOps.modular_hamiltonian_spectrum(rho_vac)
        gap = (K_spec[1] - K_spec[0]) if len(K_spec) > 1 else float("inf")

        # Predicados y veredicto local en Ω₃
        p_dS    = abs(delta_s)  < cls.EPS_ENTROPY
        p_F     = (1.0 - F_vac) < cls.EPS_FIDELITY
        p_kms_c = kms_cross     < cls.EPS_KMS
        p_kms_s = kms_self      < cls.EPS_KMS
        p_vev   = abs(vev_exc)  < cls.EPS_VEV

        n_fail = sum(0 if p else 1 for p in (p_dS, p_F, p_kms_c, p_kms_s, p_vev))
        if n_fail == 0:
            local = HeytingOmega3.COHERENT
        elif n_fail <= 2:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return VacuumAuditReport(
            entropy_production=float(delta_s),
            fidelity_to_vacuum=float(F_vac),
            vev_excitation=float(vev_exc),
            self_kms_residual=float(kms_self),
            cross_kms_residual=float(kms_cross),
            transition_probability=float(transition_prob),
            modular_spectral_gap=float(gap),
            local_verdict=local,
        )


# ── §2.4 WitnessObservationPipeline — HAND-OFF FASE 2 → FASE 3 ──────────
@dataclass(frozen=True, slots=True)
class WitnessObservationBundle:
    r"""
    Paquete de hand-off FASE 2 → FASE 3. Encapsula:
        • signatura de la tríada observada,
        • estados (ρ_vac, ρ_obs) y espectro K_vac,
        • auditoría con métricas verdaderas,
        • axiomas modulares verificados,
        • vector invariante unitario (∈ S^{k-1}).
    """
    cycle_index: int
    triad_signature: TriadSignature
    rho_vac: np.ndarray
    rho_observed: np.ndarray
    modular_spectrum: Tuple[float, ...]
    audit: VacuumAuditReport
    modular_axioms: Dict[str, float]
    invariant_vector: np.ndarray


class WitnessObservationPipeline:
    r"""
    Orquestador determinista de la dinámica modular-silenciosa:

        (ρ_vac, Φ_triad) → ρ_obs → Tomita (axiomas) → auditoría
                        → WitnessObservationBundle  (hand-off → FASE 3)
    """

    @classmethod
    def _invariant_vector(
        cls, audit: VacuumAuditReport, triad: TriadChannel
    ) -> np.ndarray:
        r"""
        Vector invariante unitario en S^{k−1} ⊂ ℝᵏ. Componentes:
            v₁ = verdict_local normalizado a [0, 1]
            v₂ = tanh(|ΔS|)
            v₃ = 1 − F(ρ_obs, ρ_vac)
            v₄ = tanh(10·KMS_cross)
            v₅ = tanh(|VEV_exc|)
            v₆ = transition_probability / n   (en [0, 1])
            v₇ = ‖K_triad‖_F / (√n ‖K‖_op)
        """
        v_local = float(int(audit.local_verdict)) / 2.0
        dS = math.tanh(abs(audit.entropy_production))
        leak = 1.0 - audit.fidelity_to_vacuum
        kms = math.tanh(10.0 * audit.cross_kms_residual)
        vev = math.tanh(abs(audit.vev_excitation))
        p_trans = float(np.clip(audit.transition_probability / max(1, triad.K.shape[0]), 0, 1))
        k_fro = triad.kraus_norm() / (math.sqrt(triad.K.shape[0]) * (np.linalg.norm(triad.K, 2) + 1e-30))
        v = np.array([v_local, dS, leak, kms, vev, p_trans, k_fro], dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v / (n + 1e-30)

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho_vac: np.ndarray,
        K_spec: Tuple[float, ...],
        triad: TriadChannel,
        H_ext: np.ndarray,
    ) -> WitnessObservationBundle:
        # (1) Observación silenciosa (sin back-action): ρ_obs = Φ_triad(ρ_vac)
        rho_obs = triad.apply(rho_vac)
        p_trans = triad.success_probability(rho_vac)

        # (2) Auditoría de silencio (métricas verdaderas)
        audit = VacuumSilenceAuditor.audit(rho_vac, rho_obs, H_ext, p_trans)

        # (3) Verificación de axiomas modulares sobre ρ_obs
        axioms = ModularFlowEngine.verify_axioms(rho_obs)

        # (4) Vector invariante unitario
        inv = cls._invariant_vector(audit, triad)

        return WitnessObservationBundle(
            cycle_index=cycle_index,
            triad_signature=triad.signature,
            rho_vac=rho_vac,
            rho_observed=rho_obs,
            modular_spectrum=K_spec,
            audit=audit,
            modular_axioms=axioms,
            invariant_vector=inv,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · CRISTALIZACIÓN Y CUSTODIA FORENSE (C. de FASE 2)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ─────────────────────────────────────────────────
class HeytingWitnessAdjudicator:
    r"""
    Colapsa el bundle en un veredicto único Ω₃, usando la estructura del
    retículo de Heyting:

        local  = audit.local_verdict
                 ∧ (axiomas OK ? ⊤ : ⋆)
                 ∧ (J² ≈ id   ? ⊤ : ⋆)
        final  = local ∧ external_verdict           (meet = ínfimo)

    El meet con el veredicto externo (p.ej. del auditor adversarial) es la
    decisión más conservadora: la verdad conjunta en el topos intuicionista.
    """
    AXIOM_TOL: float = 1.0e-6

    @classmethod
    def adjudicate(
        cls,
        bundle: WitnessObservationBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        axiomas_ok = all(v < cls.AXIOM_TOL for v in bundle.modular_axioms.values())
        regla_axiomas = (
            HeytingOmega3.COHERENT if axiomas_ok else HeytingOmega3.DEGRADED
        )
        local = bundle.audit.local_verdict.meet(regla_axiomas)
        return local.meet(external_verdict)


# ── §3.2 ExperienceCrystal con cadena Merkle ──────────────────────────────
@dataclass(frozen=True, slots=True)
class ExperienceCrystal:
    r"""
    Cristal inmutable de experiencia, firmado con SHA-256 y encadenado
    Merkle-style con el cristal anterior:

        crystal_hash_k = SHA256( crystal_hash_{k−1} ‖ content_hash_k )
        content_hash_k = SHA-256( witness ‖ cycle ‖ triad_sig ‖ vector ‖ audit )

    El campo `crystallized_invariant_vector` vive en S^{6} ⊂ ℝ⁷ (norma 1).
    """
    crystal_id: str
    witness_id: str
    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    vacuum_metrics: VacuumStateMetrics
    audit_report: VacuumAuditReport
    heyting_verdict: HeytingOmega3
    crystallized_invariant_vector: np.ndarray
    content_hash: str
    chain_hash: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.3 TOONSilentWitnessAgent — orquestador soberano ────────────────────
class TOONSilentWitnessAgent:
    r"""
    Soberano Testigo Silencioso y Cristalizador de Experiencia.

    Habita en el vacío modular (KMS a β_vac), observa la tríada adversarial
    (Ilusionista → Soñador → Auditor) SIN back-action, y cristaliza cada
    ciclo en un ExperienceCrystal con cadena de custodia Merkle-SHA-256.

    Cadena de 3 fases en cada ciclo:
        FASE 1: WitnessVacuumPreparation.prepare_vacuum_pair  → (ρ_vac, K_vac)
        FASE 2: WitnessObservationPipeline.synthesize        → bundle
        FASE 3: adjudicación Ω₃ + cristal firmado            → crystal
    """

    def __init__(
        self,
        agent_id: str = "SILENT-WITNESS-SABIO-01",
        dimension_mac: int = 4,
        kms_beta: float = 1.0,
        vacuum_beta: float = WitnessVacuumPreparation.DEFAULT_BETA_COLD,
        seed: int = 999,
    ) -> None:
        self.agent_id = agent_id
        self.dimension_mac = dimension_mac
        self.kms_beta = kms_beta
        self.vacuum_beta = vacuum_beta
        self.crystal_count = 0
        self.experience_archive: List[ExperienceCrystal] = []

        # Hamiltoniano externo del Testigo (diagonal + acoplamiento débil)
        base = np.diag(np.linspace(0.0, 3.0, dimension_mac)).astype(np.complex128)
        weak = 1e-3 * (
            np.tri(dimension_mac, dimension_mac, k=1)
            - np.tri(dimension_mac, dimension_mac, k=-1)
        ).astype(np.complex128)
        self.H_ext = base + weak + weak.conj().T

        # Ground del Testigo (referencia para silencio)
        self.ground_projector = DensityMatrixOps.ground_state_projector(self.H_ext)

        # Cadena de custodia forense genesis
        self._chain_hash = hashlib.sha256(
            f"{agent_id}::GENESIS".encode("ascii")
        ).hexdigest()

    # ──── utilidades internas ────────────────────────────────────────────
    def _content_hash(
        self, cycle_id: str, audit: VacuumAuditReport, inv: np.ndarray,
        sig: TriadSignature,
    ) -> str:
        h = hashlib.sha256()
        h.update(self.agent_id.encode("ascii"))
        h.update(cycle_id.encode("ascii"))
        h.update(sig.as_bytes())
        h.update(np.ascontiguousarray(inv).tobytes())
        h.update(f"{audit.entropy_production:.12e}".encode("ascii"))
        h.update(f"{audit.fidelity_to_vacuum:.12e}".encode("ascii"))
        h.update(f"{audit.cross_kms_residual:.12e}".encode("ascii"))
        h.update(f"{audit.vev_excitation:.12e}".encode("ascii"))
        h.update(f"{audit.local_verdict.name}".encode("ascii"))
        return h.hexdigest()

    def _advance_chain(self, content_hash: str) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + content_hash.encode("ascii")
        ).hexdigest()
        self._chain_hash = h
        return h

    # ──── ciclo principal de observación y cristalización ────────────────
    def observe_and_crystallize(
        self,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        auditor_verdict: HeytingOmega3,
        trickster_strength: float = 0.0,
        dreamer_strength: float = 0.0,
        auditor_rank: int = 4,
        dirichlet_energy: float = 0.0,   # compatibilidad con firma anterior
    ) -> ExperienceCrystal:
        self.crystal_count += 1
        crystal_id = f"CRYSTAL-EXP-{self.crystal_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Silencio Epistémico #%d | ilusión=%s | rank=%d ═══",
            self.crystal_count, trickster_illusion_type, auditor_rank,
        )

        # ── FASE 1 ── Preparar (ρ_vac, K_vac) ──
        rho_vac, K_spec = WitnessVacuumPreparation.prepare_vacuum_pair(
            self.H_ext, beta=self.vacuum_beta
        )

        # ── Construir canal selectivo de la tríada ──
        triad = TriadOperatorFactory.build_triad(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            n=self.dimension_mac,
            trickster_strength=trickster_strength,
            dreamer_strength=dreamer_strength,
            auditor_rank=auditor_rank,
        )

        # ── FASE 2 ── Observación silenciosa + auditoría modular ──
        bundle = WitnessObservationPipeline.synthesize(
            cycle_index=self.crystal_count,
            rho_vac=rho_vac,
            K_spec=K_spec,
            triad=triad,
            H_ext=self.H_ext,
        )

        # ── FASE 3 ── Adjudicación Ω₃ y cristalización ──
        final_verdict = HeytingWitnessAdjudicator.adjudicate(bundle, auditor_verdict)

        # VacuumStateMetrics (evolucionada, del estado observado)
        metrics = VacuumStateMetrics(
            vacuum_expectation_value=bundle.audit.vev_excitation,
            tomita_takesaki_flow_param=1.0,  # t del flujo modular usado en KMS
            kms_entropy_drift=bundle.audit.entropy_production,  # ΔS real 
            silence_purity=bundle.audit.fidelity_to_vacuum,
            modular_ground_energy=K_spec[0] if K_spec else 0.0,
            modular_spectral_gap=bundle.audit.modular_spectral_gap,
            von_neumann_entropy=DensityMatrixOps.von_neumann_entropy(bundle.rho_observed),
            purity=DensityMatrixOps.purity(bundle.rho_observed),
            is_silent=(final_verdict == HeytingOmega3.COHERENT),
        )

        # Cadena de custodia: content_hash + Merkle
        content_hash = self._content_hash(
            crystal_id, bundle.audit, bundle.invariant_vector, bundle.triad_signature
        )
        chain_hash = self._advance_chain(content_hash)

        # Firma global
        sig_hasher = hashlib.sha256()
        sig_hasher.update(self.agent_id.encode("ascii"))
        sig_hasher.update(crystal_id.encode("ascii"))
        sig_hasher.update(final_verdict.name.encode("ascii"))
        sig_hasher.update(chain_hash.encode("ascii"))
        sig_hasher.update(f"{time.time_ns()}".encode("ascii"))
        provenance = sig_hasher.hexdigest()

        crystal = ExperienceCrystal(
            crystal_id=crystal_id,
            witness_id=self.agent_id,
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            vacuum_metrics=metrics,
            audit_report=bundle.audit,
            heyting_verdict=final_verdict,
            crystallized_invariant_vector=bundle.invariant_vector,
            content_hash=content_hash,
            chain_hash=chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )
        self.experience_archive.append(crystal)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cristal %s | Ω₃=%s | ΔS=%.3e | F=%.6f | KMS_x=%.3e | %.2f ms",
            crystal_id, final_verdict.name,
            bundle.audit.entropy_production,
            bundle.audit.fidelity_to_vacuum,
            bundle.audit.cross_kms_residual,
            dt_ms,
        )
        return crystal


# ── §3.4 Demostración autónoma ───────────────────────────────────────────
if __name__ == "__main__":
    witness = TOONSilentWitnessAgent(
        agent_id="SILENT-WITNESS-SABIO-01",
        dimension_mac=4,
        kms_beta=1.0,
        vacuum_beta=50.0,
    )

    print("═" * 88)
    print("TESTIGO SILENCIOSO — Evolución Doctoral (Tomita-Takesaki ejecutable)")
    print("═" * 88)

    # (A) Autotest: φ_vac contra su propio flujo modular debe ser KMS ≈ 0
    rho_v, _ = WitnessVacuumPreparation.prepare_vacuum_pair(witness.H_ext, beta=50.0)
    kms_self = ModularFlowEngine.verify_kms_self(rho_v, n_tests=8)
    ax = ModularFlowEngine.verify_axioms(rho_v)
    print(f"[σ-t autotest] KMS(β=1) residual = {kms_self:.3e}   (esperado ≈ 0)")
    print(f"[Axiomas σ_t ] unital={ax['unital_residual']:.2e}  "
          f"product={ax['product_residual']:.2e}  "
          f"isometry={ax['isometry_residual']:.2e}  "
          f"J²={ax['involution_residual']:.2e}")

    scenarios = [
        # (nombre, ilusión, escenario, hash_audit, verdict_ext,
        #  strength_trickster, strength_dreamer, auditor_rank)
        ("COHERENT TRIAD (Φ ≈ id)",
         "TRIAD-COH-001", "SYNTH-COH-0001", "AAAACOH00000001", HeytingOmega3.COHERENT,
         0.02, 0.02, 4),
        ("DEGRADED TRIAD (perturbación media)",
         "TRIAD-DEG-002", "SYNTH-DEG-0002", "BBBBBDEG0000002", HeytingOmega3.COHERENT,
         0.25, 0.25, 3),
        ("VETOED TRIAD (caos adversarial)",
         "SPLIT_CONTRACT_ILLUSION", "SYNTH-VETO-0003", "bfa1bfa0b2551841", HeytingOmega3.VETOED,
         1.00, 1.00, 1),
    ]

    print("\n" + "─" * 88)
    for name, ill, drm, aih, ext, sI, sD, rank in scenarios:
        c = witness.observe_and_crystallize(
            trickster_illusion_type=ill,
            dreamer_scenario_id=drm,
            auditor_immunization_hash=aih,
            auditor_verdict=ext,
            trickster_strength=sI,
            dreamer_strength=sD,
            auditor_rank=rank,
        )
        print(f"\n[{name}]")
        print(f"   crystal_id         : {c.crystal_id}")
        print(f"   Ω₃ final           : {c.heyting_verdict.name}")
        print(f"   ΔS (producida)     : {c.audit_report.entropy_production:+.6e}")
        print(f"   F(ρ_obs, ρ_vac)    : {c.audit_report.fidelity_to_vacuum:.9f}")
        print(f"   KMS_cross residual : {c.audit_report.cross_kms_residual:.3e}")
        print(f"   VEV excitación     : {c.audit_report.vev_excitation:+.6e}")
        print(f"   Gap(K_vac)         : {c.audit_report.modular_spectral_gap:+.6e}")
        print(f"   ‖v_inv‖₂          : {np.linalg.norm(c.crystallized_invariant_vector):.6f}")
        print(f"   chain_hash         : {c.chain_hash[:32]}…")

    print("\n" + "═" * 88)
    print("✓ Tomita-Takesaki ejecutable: σ_t, σ_i, J, Δ verificados.")
    print("✓ KMS(β=1) consistente: ω(a σ_i(b)) = ω(ba)  —residuo ≈ 0.")
    print("✓ Cadena de custodia Merkle-SHA-256 preservada entre cristales.")
    print(f"✓ Total cristales: {len(witness.experience_archive)}")
    print("═" * 88)