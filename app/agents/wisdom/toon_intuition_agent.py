# -*- coding: utf-8 -*-
r"""Soberano de la Intuición y Reflejo Flash Geodésico.

Ubicación: app/agents/wisdom/toon_intuition_agent.py
Versión  : 2.2.0-Doctoral-Nested-Bures-Jacobian-Kelly-Latency-Merkle

Este módulo define la entidad ejecutiva e inalienable "Soberano de la Intuición"
en el dominio WISDOM de la arquitectura COGNITIVE TOON / APU Filter. Su misión es
procesar reflejos flash sub-milisegundo ante emergencias ciber-físicas u oportunidades
de decisión en obra, proyectando estados semilla germinados sobre variedades de decisión
en el espacio de Grassmann y evaluando la métrica de Bures, el criterio de Kelly y la
intervención de hardware (crowbar interlock).

================================================================================
I. FORMALIZACIÓN MATEMÁTICA Y GEOMETRÍA DE DECISIÓN
================================================================================

1. Variedad de Decisión en el Espacio de Grassmann Gr(r, n):
   El espacio de decisiones válidas se representa como un subespacio de dimensión $r$ en $\mathbb{C}^n$,
   parametrizado por una base ortonormal $B \in \mathrm{St}(r, n)$ ($\mathbb{C}^{n \times r}$, $B^\dagger B = I_r$)
   obtenida por muestreo Haar determinista en el grupo de Stiefel mediante el digest SHA-256 del ID del agente.
   El proyector ortogonal asociado es $P = B B^\dagger \in M_n(\mathbb{C})$, $P^2 = P = P^\dagger$.

2. Funcional de Energía de Dirichlet y Paso Cauchy Flash:
   Dado el estado densidad germinado $\rho \in \mathfrak{D}_n$, la energía de Dirichlet mide el apartamiento de $\mathrm{ran}(P)$:
       $$\mathcal{E}(\rho) = \frac{1}{2} \|\rho - P \rho P\|_F^2 \ge 0$$
   Su gradiente tangente en $T_\rho \mathfrak{D}_n$ es $\nabla \mathcal{E}|_T = (\rho - P \rho P) - \frac{\mathrm{Tr}(\rho - P \rho P)}{n} I$.
   El reflejo flash unipaso realiza una actualización de Newton-Cauchy exacta ($\eta^* = 1$):
       $$\rho_{\mathrm{flash}} = \mathrm{proj}_{\mathfrak{D}_n}(\rho - \eta^* \nabla \mathcal{E}|_T)$$

3. Geometría de Bures-Wasserstein $W_2$ y Fidelidad de Uhlmann:
   Entre el estado flash $\rho_{\mathrm{flash}}$ y el atractor objetivo $\rho_{\mathrm{target}} = \frac{P \rho P}{\mathrm{Tr}(P \rho P)}$,
   la fidelidad de Uhlmann y la distancia geodésica de Bures son:
       $$F(\rho, \sigma) = \left( \mathrm{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2 \in [0, 1]$$
       $$d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}} \in [0, \sqrt{2}], \quad \theta_B = \arccos(\sqrt{F})$$

4. Criterio de Apuesta $\kappa$-Kelly y Crecimiento Logarítmico:
   Con probabilidad efectiva $p_{\mathrm{eff}} = F(\rho_{\mathrm{flash}}, \rho_{\mathrm{target}})$, la fracción óptima de Kelly es:
       $$f^* = \max(0, 2 p_{\mathrm{eff}} - 1), \quad s = \kappa \cdot f^* \quad (\kappa \in (0, 1])$$
   El crecimiento logarítmico esperado de la inversión es:
       $$G(s) = p_{\mathrm{eff}} \log(1 + s) + (1 - p_{\mathrm{eff}}) \log(1 - s) = \log 2 - h_2(p_{\mathrm{eff}})$$

5. Adjudicación de Heyting $\Omega_3$ e Interlock Ciber-Físico ESP32:
   El veredicto final en $\Omega_3 = \{\bot (\mathrm{VETOED}) < \star (\mathrm{DEGRADED}) < \top (\mathrm{COHERENT})\}$
   aplica meet conservador con detección de fraude. Si $v_{\mathrm{final}} = \bot$, se dispara de forma
   inmediata el interlock de hardware (GPIO14 $\to$ HIGH, MOSFET BT151) con latencia nominal $< 400\text{ ns}$.

================================================================================
II. ESTRUCTURA FUNTORIAL Y ARQUITECTURA
================================================================================

El Soberano opera como el funtor estricto $F = F_3 \circ F_2 \circ F_1$:
    $$F : \mathrm{IntuitiveFlashRequest} \times \mathrm{Gr}(r, n) \longrightarrow \mathrm{IntuitionFlashCertificate}$$

  • $F_1$ (`FlashHandoff.build`): $\mathrm{IntuitiveFlashRequest} \times \mathrm{Gr}(r, n) \to \mathrm{FlashHandoff}$.
    Sanitización $C^*$, proyector $P = B B^\dagger$, energía inicial $\mathcal{E}(\rho_0)$ y auditoría de fraude.
  • $F_2$ (`FlashPipeline.synthesize`): $\mathrm{FlashHandoff} \to \mathrm{FlashTrajectoryBundle}$.
    Paso Cauchy $\eta^*$, distancia geodésica $d_B$, espectro del Jacobiano $DT_\eta$, Kelly $\kappa$ y contrato de latencia $p99$.
  • $F_3$ (`TOONIntuitionAgent._phase3_certify`): $\mathrm{FlashTrajectoryBundle} \to \mathrm{IntuitionFlashCertificate}$.
    Adjudicación por meets, disparo ciber-físico ESP32 Crowbar si $\bot$, traducción visceral y firma Merkle.

================================================================================
III. INVARIANTES FORMALES Y AXIOMAS DEL SISTEMA
================================================================================

- Axioma 1 (Autonomía de Bures): $d_B(\rho, \sigma)$ es una distancia riemanniana intrínseca sobre $\mathfrak{D}_n$.
- Axioma 2 (Invariante Determinista de Stiefel): La base $B \in \mathrm{St}(r, n)$ se genera exclusivamente mediante la semilla SHA-256 del ID.
- Axioma 3 (Prioridad Absoluta del Crowbar): Si $v_{\mathrm{final}} = \bot$, la respuesta es una interrupción física inmediata ($\mathrm{interlock\_fired} = \mathrm{True}$).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


logger = logging.getLogger("APU.Wisdom.TOONIntuition")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


def _seed_from_string(s: str) -> int:
    """Proyección SHA-256 → ℕ/2³², determinista, libre de plataforma."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 · SUSTRATO GEOMÉTRICO                                             ║
# ║                                                                           ║
# ║  Objetos: Ω₃, 𝔇_n, Gr(r,n), Request.                                      ║
# ║  Morfismo terminal: FlashHandoff.continue_into_phase2.                    ║
# ║  Ese morfismo ES el dominio de todos los funtores de la FASE 2.           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ──────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel de 3 valores)

        Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}

    Estructura:
        meet    (∧) : ínfimo = min
        join    (∨) : supremo = max
        implies (⇒) : residuo  (a ∧ b ≤ c ⇔ a ≤ (b ⇒ c))
                      a ⇒ b = ⊤  si a ≤ b,  else b
        neg     (¬) : a ⇒ ⊥     (seudocomplemento intuicionista)
        iff     (⇔) : (a ⇒ b) ∧ (b ⇒ a)

    Propiedades que fallan respecto de un álgebra de Boole:
        ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤          (tercio excluso)
        ¬¬⋆ = ⊤ ≠ ⋆             (⋆ no es regular)
        {⊥, ⊤}  ↪  Ω₃           (subálgebra Booleana de regulares)

    Interpretación en el topos de prefaisceaux sobre el poset Ω₃:
        ⊤ clasifica subobjetos totales (flash coherente / luz verde),
        ⋆ clasifica subobjetos densos no cerrados (alerta / pies de plomo),
        ⊥ clasifica el subobjeto vacío (veto / crowbar).
    """
    VETOED: int = 0      # ⊥
    DEGRADED: int = 1    # ⋆
    COHERENT: int = 2    # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    def leq(self, other: "HeytingOmega3") -> bool:
        """Orden total del poset: ⊥ ≤ ⋆ ≤ ⊤."""
        return int(self) <= int(other)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if self.leq(other) else other

    def neg(self) -> "HeytingOmega3":
        """Seudocomplemento ¬a := a ⇒ ⊥.  ¬⋆ = ⊥,  ¬⊥ = ⊤,  ¬⊤ = ⊥."""
        return self.implies(HeytingOmega3.VETOED)

    def iff(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.implies(other).meet(other.implies(self))

    def is_regular(self) -> bool:
        """a es regular ⟺ a = ¬¬a.  Sólo ⊥ y ⊤ lo son."""
        return self.neg().neg() == self

    def as_weight(self) -> float:
        """Inmersión afín Ω₃ ↪ [0, 1] : ⊥↦0, ⋆↦½, ⊤↦1."""
        return float(int(self)) / 2.0


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre el compacto convexo de estados

        𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†,  ρ ≥ 0,  Tr ρ = 1 }.

    Espacio tangente afín (métrica plana de Frobenius):

        T_ρ 𝔇_n ≅ { A = A† : Tr A = 0 }.

    Funcionales (unitariamente invariantes):

        S(ρ)     = −Tr(ρ log ρ)                 von Neumann (nats)
        P(ρ)     = Tr(ρ²)                       pureza ∈ [1/n, 1]
        F(ρ,σ)   = ‖√ρ √σ‖₁                     Uhlmann–Jozsa ∈ [0, 1]
        d_B(ρ,σ) = √(2 − 2√F)                   Bures ∈ [0, √2]
        θ_B(ρ,σ) = arccos(√F)                   ángulo de Bures ∈ [0, π/2]
        ρ^z      = exp(z log ρ)                 cálculo funcional

    sanitize = proyección euclídea sobre 𝔇_n (Hermitiza + PSD-clip Higham
    + renormalización de traza).  Es no expansiva en ‖·‖_F.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""Proyección afín sobre 𝔇_n: Hermitiza, PSD-clip, Tr = 1."""
        if not cls.is_square(rho):
            raise ValueError(
                f"DensityOperatorAlgebra.sanitize: no cuadrada {np.shape(rho)}"
            )
        rho_h = np.asarray(rho, dtype=np.complex128)
        rho_h = 0.5 * (rho_h + rho_h.conj().T)
        w, V = la.eigh(rho_h)
        w = np.maximum(np.real(w), cls.EPS)
        rho_h = (V * w) @ V.conj().T
        tr = float(np.trace(rho_h).real)
        if abs(tr) > _EPS_TRACE:
            rho_h = rho_h / tr
        return rho_h

    @classmethod
    def spectrum_descending(cls, rho: np.ndarray) -> RealVector:
        rho = cls.sanitize(rho)
        w = np.real(la.eigvalsh(rho))
        w = np.sort(w)[::-1]
        w = np.maximum(w, cls.EPS)
        s = float(w.sum())
        if s <= 0.0:
            n = w.size
            return np.full(n, 1.0 / n, dtype=np.float64)
        return (w / s).astype(np.float64)

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def frobenius(cls, A: np.ndarray) -> float:
        return float(np.linalg.norm(A, "fro"))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD
    ) -> ComplexMatrix:
        r"""ρ^z = V diag(λ_i^z) V†  (Holstein–Rellich, corte principal)."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return float(np.clip(val, 0.0, 1.0))

    @classmethod
    def bures_angle(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""θ_B(ρ,σ) = arccos(√F(ρ,σ)) ∈ [0, π/2]."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.acos(float(np.clip(math.sqrt(F), 0.0, 1.0))))

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""d_B(ρ,σ) = √(2 − 2√F(ρ,σ)) ∈ [0, √2].  Geodésica, no Frobenius."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def tangent_project(cls, A: np.ndarray, n: Optional[int] = None) -> ComplexMatrix:
        r"""Proyección euclídea sobre T 𝔇_n: Hermitiza y resta (Tr A / n) I."""
        H = np.asarray(A, dtype=np.complex128)
        H = 0.5 * (H + H.conj().T)
        dim = int(n if n is not None else H.shape[0])
        beta = float(np.trace(H).real) / max(dim, 1)
        return H - beta * np.eye(dim, dtype=np.complex128)


# ── §1.3 Geometría del subespacio de decisiones ───────────────────────────
@dataclass(frozen=True, slots=True)
class DecisionManifoldGeometry:
    r"""
    Punto de Grassmann Gr(r, n): variedad de decisión M_decision ⊂ ℂⁿ.

        basis       : B ∈ St(r, n) ⊂ ℂ^{n×r},  B†B = I_r
        projector   : P = B B†,  P² = P = P†
        rank, codim : r, n−r
        hash        : SHA-256(B ‖ P)   trazabilidad forense (P3)

    El complemento P_⊥ = I − P es el único ortocomplemento.  E[ρ] mide
    la masa de ρ fuera de ran(P) más las coherencias P–P_⊥.
    """
    basis: np.ndarray
    projector: np.ndarray
    rank: int
    codim: int
    is_isometry: bool
    is_projector: bool
    isometry_residual: float
    projector_residual: float
    hash: str

    @property
    def complement(self) -> ComplexMatrix:
        n = self.projector.shape[0]
        return np.eye(n, dtype=np.complex128) - self.projector

    def mass(self, rho: np.ndarray) -> float:
        """Tr(P ρ) ∈ [0, 1] — masa de ρ sobre ran(P)."""
        rho_s = DensityOperatorAlgebra.sanitize(rho)
        return float(np.real(np.trace(self.projector @ rho_s)))


class DecisionManifoldFactory:
    r"""
    Construye B ∈ St(r, n) por QR-Haar (Stewart 1980: Ginibre → Haar en U(n);
    las primeras r columnas son Haar en Stiefel).  Semilla = SHA-256(key)
    (P3: determinista por agent_id, no rng(42)).

    Se fuerza r ∈ [1, n−1] porque r = n ⇒ P = I ⇒ E ≡ 0 (decisión trivial).
    """

    @classmethod
    def haar_unitary(cls, n: int, rng: np.random.Generator) -> ComplexMatrix:
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        d = np.diagonal(R)
        ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
        return (Q * ph.conj()).astype(np.complex128)

    @classmethod
    def principal_angles(cls, B1: np.ndarray, B2: np.ndarray) -> RealVector:
        r"""θ_i = arccos(σ_i(B1† B2)) ∈ [0, π/2]."""
        M = B1.conj().T @ B2
        sig = np.clip(np.real(la.svdvals(M)), 0.0, 1.0)
        return np.arccos(sig).astype(np.float64)

    @classmethod
    def grassmann_distance(cls, B1: np.ndarray, B2: np.ndarray) -> float:
        """Distancia de Grassmann ‖θ‖₂ entre ran(B1) y ran(B2)."""
        return float(np.linalg.norm(cls.principal_angles(B1, B2), ord=2))

    @classmethod
    def build(cls, n: int, rank: int, key: str) -> DecisionManifoldGeometry:
        if n < 2:
            raise ValueError("DecisionManifoldFactory.build: n ≥ 2")
        rank = int(np.clip(rank, 1, n - 1))
        rng = np.random.default_rng(_seed_from_string(f"M_DECISION::{key}"))
        Q = cls.haar_unitary(n, rng)
        B = np.ascontiguousarray(Q[:, :rank])
        P = B @ B.conj().T
        P = 0.5 * (P + P.conj().T)

        iso_err = float(np.linalg.norm(B.conj().T @ B - np.eye(rank), "fro"))
        proj_err = float(np.linalg.norm(P @ P - P, "fro"))
        sub_hash = _sha256_bytes(
            np.ascontiguousarray(B).tobytes(),
            np.ascontiguousarray(P).tobytes(),
        )
        return DecisionManifoldGeometry(
            basis=B,
            projector=P,
            rank=rank,
            codim=n - rank,
            is_isometry=bool(iso_err < 1e-10),
            is_projector=bool(proj_err < 1e-10),
            isometry_residual=iso_err,
            projector_residual=proj_err,
            hash=sub_hash,
        )


# ── §1.4 Solicitud validada ───────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntuitiveFlashRequest:
    """Petición cruda del cultivo germinado + contexto de obra."""
    crop_origin_id: str
    seed_crystal_id: str
    germinated_density_matrix: np.ndarray
    site_context_payload: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class RequestAuditReport:
    r"""
    Verificación dimensional del request (antes de proyectar a 𝔇_n):

        is_valid := |Tr ρ − 1| < τ  ∧  ‖ρ−ρ†‖_F < τ_H  ∧  λ_min ≥ −ε
        has_critical_fraud : flag contextual (P4: veto duro aguas abajo)
        payload_hash       : SHA-256 canónico del payload ‖ ρ_saneada
    """
    purity: float
    entropy: float
    trace_residual: float
    hermiticity_residual: float
    lambda_min: float
    has_critical_fraud: bool
    cost_risk_amount: float
    payload_hash: str
    is_valid: bool


class FlashRequestSanitizer:
    r"""
    Valida y sanea la solicitud.  El flag `has_critical_fraud` NO se infiere
    del espectro: es un predicado contextual del payload (P4).  El motor
    lo transporta intacto hasta el adjudicador, que lo interpreta como ⊥.
    """
    TRACE_TOL: Final[float] = 1.0e-6
    HERM_TOL: Final[float] = 1.0e-8

    @classmethod
    def _payload_bytes(cls, payload: Dict[str, Any]) -> bytes:
        try:
            return json.dumps(
                payload, sort_keys=True, default=str, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            return repr(sorted(payload.items())).encode("utf-8")

    @classmethod
    def sanitize(
        cls, request: IntuitiveFlashRequest
    ) -> Tuple[ComplexMatrix, RequestAuditReport]:
        rho_raw = np.asarray(request.germinated_density_matrix, dtype=np.complex128)
        if not DensityOperatorAlgebra.is_square(rho_raw):
            raise ValueError(
                f"FlashRequestSanitizer: matriz no cuadrada {np.shape(rho_raw)}"
            )

        herm_res = float(np.linalg.norm(rho_raw - rho_raw.conj().T, "fro"))
        trace_res = abs(float(np.trace(rho_raw).real) - 1.0)

        rho = DensityOperatorAlgebra.sanitize(rho_raw)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        purity = float(np.sum(w ** 2))
        entropy = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        lam_min = float(np.real(la.eigvalsh(rho)).min())

        payload = dict(request.site_context_payload or {})
        fraud = bool(payload.get("has_critical_fraud", False))
        cost = float(payload.get("cost_risk_amount", 0.0) or 0.0)

        payload_hash = _sha256_bytes(
            cls._payload_bytes(payload),
            np.ascontiguousarray(rho).tobytes(),
        )
        is_valid = (
            trace_res < cls.TRACE_TOL
            and herm_res < cls.HERM_TOL
            and lam_min > -1e-6
        )
        return rho, RequestAuditReport(
            purity=purity,
            entropy=entropy,
            trace_residual=trace_res,
            hermiticity_residual=herm_res,
            lambda_min=lam_min,
            has_critical_fraud=fraud,
            cost_risk_amount=cost,
            payload_hash=payload_hash,
            is_valid=bool(is_valid),
        )


# ── §1.5 FlashHandoff — HAND-OFF FASE 1 → FASE 2 ─────────────────────────
@dataclass(frozen=True, slots=True)
class FlashHandoff:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

        rho_seed      : ρ ∈ 𝔇_n  (saneada)
        rho_target    : PρP / Tr(PρP)   atractor estático, E≈0
        geometry      : (B, P) ∈ Gr(r, n)
        seed_energy   : E[ρ_seed] = ½‖ρ − PρP‖_F²
        seed_fidelity : F(ρ_seed, ρ_target)
        mass_on_P     : Tr(P ρ)
        spectral_hash : SHA-256(ρ ‖ spec)
    """
    cycle_index: int
    request_id: str
    crop_origin_id: str
    seed_crystal_id: str
    rho_seed: np.ndarray
    rho_target: np.ndarray
    geometry: DecisionManifoldGeometry
    seed_energy: float
    seed_fidelity: float
    mass_on_P: float
    request_audit: RequestAuditReport
    spectral_hash: str

    @classmethod
    def static_target(cls, rho: np.ndarray, P: np.ndarray) -> ComplexMatrix:
        PrP = P @ rho @ P
        tr = float(np.trace(PrP).real)
        if tr < 1e-15:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return DensityOperatorAlgebra.sanitize(PrP / tr)

    @classmethod
    def dirichlet_energy(cls, rho: np.ndarray, P: np.ndarray) -> float:
        residual = rho - P @ rho @ P
        return 0.5 * float(np.linalg.norm(residual, "fro") ** 2)

    @classmethod
    def build(
        cls,
        cycle_index: int,
        request: IntuitiveFlashRequest,
        geometry: DecisionManifoldGeometry,
    ) -> "FlashHandoff":
        r"""
        Cierra la FASE 1 como objeto.  El morfismo de continuación
        hacia FASE 2 es `continue_into_phase2`.
        """
        rho, audit = FlashRequestSanitizer.sanitize(request)
        P = geometry.projector
        rho_target = cls.static_target(rho, P)
        seed_energy = cls.dirichlet_energy(rho, P)
        seed_fid = DensityOperatorAlgebra.uhlmann_fidelity(rho, rho_target)
        mass = float(np.real(np.trace(P @ rho)))

        w = DensityOperatorAlgebra.spectrum_descending(rho)
        spec_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(w).tobytes(),
        )
        return cls(
            cycle_index=cycle_index,
            request_id=f"{request.crop_origin_id}::{request.seed_crystal_id}",
            crop_origin_id=request.crop_origin_id,
            seed_crystal_id=request.seed_crystal_id,
            rho_seed=rho,
            rho_target=rho_target,
            geometry=geometry,
            seed_energy=float(seed_energy),
            seed_fidelity=float(seed_fid),
            mass_on_P=mass,
            request_audit=audit,
            spectral_hash=spec_hash,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "seed_energy": self.seed_energy,
            "seed_fidelity": self.seed_fidelity,
            "mass_on_P": self.mass_on_P,
            "geometry_hash": self.geometry.hash,
            "has_critical_fraud": self.request_audit.has_critical_fraud,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        eta_star: float,
        latency_contract_ns: float,
        kelly_kappa: float,
    ) -> "FlashTrajectoryBundle":
        r"""
        Último morfismo de la FASE 1  ∧  primer morfismo de la FASE 2.

        Identidad de composición:

            continue_into_phase2 ∘ build
                = FlashPipeline.synthesize ∘ build
                : Request × Gr(r,n) → FlashTrajectoryBundle.

        En el sentido de categorías, la FASE 2 es el comma-category
        (FlashHandoff ↓ Flash₂).  Invocar Dirichlet/Bures/Kelly sin un
        FlashHandoff es un error de tipo.
        """
        return FlashPipeline.synthesize(
            handoff=self,
            eta_star=eta_star,
            latency_contract_ns=latency_contract_ns,
            kelly_kappa=kelly_kappa,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 · DINÁMICA FLASH + MÉTRICA BURES + KELLY                          ║
# ║                                                                           ║
# ║  Dominio = FlashHandoff (codominio de §1.5).                              ║
# ║  Codominio = FlashTrajectoryBundle, dominio de toda la FASE 3.            ║
# ║                                                                           ║
# ║  §2.1 se lee como la continuación literal de continue_into_phase2.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Funcional de Dirichlet del proyector ─────────────────────────────
@dataclass(frozen=True, slots=True)
class DirichletLandscape:
    r"""
    Paisaje de E en un punto ρ ∈ 𝔇_n.

        E(ρ)          = ½‖ρ − PρP‖_F²
        residual      = ρ − PρP = P_⊥ρP + PρP_⊥ + P_⊥ρP_⊥
        coherent_mass = ‖P_⊥ ρ P‖_F²
        leak_mass     = ‖P_⊥ ρ P_⊥‖_F²
        Teorema: ‖ρ−PρP‖_F² = 2·coherent_mass + leak_mass
        Lip_F(∇E) ≤ 1
    """
    rho: np.ndarray
    energy: float
    grad_full: np.ndarray
    grad_tan: np.ndarray
    grad_norm: float
    residual_norm: float
    coherent_mass: float
    leak_mass: float


class FlashDirichletFunctional:
    r"""
    Energía de Dirichlet (continuación de FlashHandoff.continue_into_phase2):

        E[ρ] = ½ ‖ρ − P ρ P‖_F² ≥ 0,
        E[ρ] = 0  ⟺  ρ = PρP  ⟺  supp(ρ) ⊆ ran(P) y [ρ, P] = 0.

    Gradiente euclídeo ⟨A,B⟩_F = Tr(A† B):

        ∇E(ρ) = ρ − PρP,
        ∇E|_T = ∇E − (Tr ∇E / n) I  ∈ T_ρ 𝔇_n.

    Hessiano: ∇²E = Id − Ad_P, spec ⊆ {0,1}, luego Lip_F(∇E) ≤ 1 y el
    paso Cauchy η = 1 es Newton exacto en el bloque transversal (P2).
    """
    LIPSCHITZ: Final[float] = 1.0

    @classmethod
    def decompose_residual(
        cls, rho: np.ndarray, P: np.ndarray
    ) -> Tuple[ComplexMatrix, float, float]:
        I = np.eye(P.shape[0], dtype=np.complex128)
        Pc = I - P
        cross = Pc @ rho @ P
        leak = Pc @ rho @ Pc
        residual = rho - P @ rho @ P
        coherent_mass = float(np.linalg.norm(cross, "fro") ** 2)
        leak_mass = float(np.linalg.norm(leak, "fro") ** 2)
        return residual, coherent_mass, leak_mass

    @classmethod
    def evaluate(cls, rho: np.ndarray, P: np.ndarray) -> DirichletLandscape:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        residual, coherent_mass, leak_mass = cls.decompose_residual(rho, P)
        energy = 0.5 * float(np.linalg.norm(residual, "fro") ** 2)
        grad_tan = DensityOperatorAlgebra.tangent_project(residual, n)
        return DirichletLandscape(
            rho=rho,
            energy=energy,
            grad_full=residual,
            grad_tan=grad_tan,
            grad_norm=float(np.linalg.norm(grad_tan, "fro")),
            residual_norm=float(np.linalg.norm(residual, "fro")),
            coherent_mass=coherent_mass,
            leak_mass=leak_mass,
        )

    @classmethod
    def cauchy_step(cls, rho: np.ndarray, P: np.ndarray, eta: float) -> ComplexMatrix:
        r"""
        T_η(ρ) = sanitize(ρ − η ∇E|_T).

        Para η = 1, antes de Higham:

            ρ − ∇E|_T = PρP + (Tr(ρ−PρP)/n) I.
        """
        land = cls.evaluate(rho, P)
        return DensityOperatorAlgebra.sanitize(rho - float(eta) * land.grad_tan)


# ── §2.2 Métrica geodésica de Bures ───────────────────────────────────────
class BuresGeodesicMetric:
    r"""
    Métrica de Bures sobre 𝔇_n (única Riemanniana CPTP-contractiva, Petz 1996):

        F(ρ, σ)   = ‖√ρ √σ‖₁ ∈ [0, 1]
        θ_B(ρ, σ) = arccos(√F) ∈ [0, π/2]
        d_B(ρ, σ) = √(2 − 2√F) ∈ [0, √2]

    Geodésica de Bures–Wasserstein (Takatsu / Bhatia–Jain–Lim):

        C_ρσ = ρ^{-1/2} (ρ^{1/2} σ ρ^{1/2})^{1/2} ρ^{-1/2}
        γ(t) = [(1−t)I + t C] ρ [(1−t)I + t C]
    """

    @classmethod
    def fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return DensityOperatorAlgebra.uhlmann_fidelity(rho, sigma)

    @classmethod
    def angle(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return DensityOperatorAlgebra.bures_angle(rho, sigma)

    @classmethod
    def distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return DensityOperatorAlgebra.bures_distance(rho, sigma)

    @classmethod
    def triangle_residual(
        cls, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> float:
        """res = max(0, d(a,c) − d(a,b) − d(b,c)).  Cero ⇔ desigualdad OK."""
        dab = cls.distance(a, b)
        dbc = cls.distance(b, c)
        dac = cls.distance(a, c)
        return float(max(0.0, dac - dab - dbc))

    @classmethod
    def uhlmann_map(cls, rho: np.ndarray, sigma: np.ndarray) -> ComplexMatrix:
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(sigma)
        rho_h = DensityOperatorAlgebra.matrix_power(rho, 0.5)
        rho_invh = DensityOperatorAlgebra.matrix_power(rho, -0.5)
        inner = rho_h @ sigma @ rho_h
        inner_h = DensityOperatorAlgebra.matrix_power(inner, 0.5)
        C = rho_invh @ inner_h @ rho_invh
        return 0.5 * (C + C.conj().T)

    @classmethod
    def geodesic(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float
    ) -> ComplexMatrix:
        t = float(np.clip(t, 0.0, 1.0))
        rho = DensityOperatorAlgebra.sanitize(rho)
        if t <= 0.0:
            return rho
        if t >= 1.0:
            return DensityOperatorAlgebra.sanitize(sigma)
        C = cls.uhlmann_map(rho, sigma)
        n = rho.shape[0]
        S = (1.0 - t) * np.eye(n, dtype=np.complex128) + t * C
        return DensityOperatorAlgebra.sanitize(S @ rho @ S.conj().T)


# ── §2.3 Radio espectral del Jacobiano flash ──────────────────────────────
@dataclass(frozen=True, slots=True)
class FlashJacobianSpectrum:
    r"""
    Linealización de T_η(ρ) = ρ − η ∇E|_T  (antes de Higham) en el afín
    Hermítico de traza 1.

    Descomposición en bloques de Ad_P (P2):

        ran(Ad_P)   (dim r²,  ρ = PρP)     :  spec(DT) = {1}      marginal
        ran(Id−Ad_P) (dim n²−r², transversal): spec(DT) = {1 − η}

    Luego:
        transverse_rate = |1 − η*|
        ρ_spec(DT)      = max(1, |1−η*|)     (siempre ≥ 1 por el bloque P)
        contractivo transversal ⟺ η* ∈ (0, 2)
    """
    eta_star: float
    transverse_rate: float
    spectral_radius: float
    numerical_transverse_gain: float
    is_contractive_transverse: bool
    local_verdict: HeytingOmega3


class FlashSpectralJacobian:
    r"""
    Auditoría analítica + sonda numérica del Jacobiano flash.

    La sonda aplica DT_η a perturbaciones aleatorias del bloque P_⊥ y
    mide ‖DT(v)‖_F / ‖v‖_F ≈ |1−η| (testigo de la fórmula de bloques).
    """
    N_PROBES: Final[int] = 8

    @classmethod
    def _dt_apply(cls, A: np.ndarray, P: np.ndarray, eta: float) -> ComplexMatrix:
        RA = A - P @ A @ P
        n = A.shape[0]
        DT = A - eta * RA + (eta * float(np.trace(RA).real) / n) * np.eye(
            n, dtype=np.complex128
        )
        return 0.5 * (DT + DT.conj().T)

    @classmethod
    def numerical_transverse_gain(
        cls, P: np.ndarray, eta: float, key: str = "J-PROBE"
    ) -> float:
        n = int(P.shape[0])
        I = np.eye(n, dtype=np.complex128)
        Pc = I - P
        rng = np.random.default_rng(_seed_from_string(f"JACOBIAN::{key}"))
        gains: List[float] = []
        for _ in range(cls.N_PROBES):
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            A = 0.5 * (A + A.conj().T)
            A = Pc @ A @ Pc
            A = DensityOperatorAlgebra.tangent_project(A, n)
            nrm = DensityOperatorAlgebra.frobenius(A)
            if nrm < 1e-14:
                continue
            DT = cls._dt_apply(A, P, eta)
            gains.append(DensityOperatorAlgebra.frobenius(DT) / nrm)
        if not gains:
            return abs(1.0 - eta)
        return float(np.mean(gains))

    @classmethod
    def audit(
        cls,
        eta_star: float,
        P: Optional[np.ndarray] = None,
        probe_key: str = "J-PROBE",
    ) -> FlashJacobianSpectrum:
        eta = float(np.clip(eta_star, 1e-9, 2.0 - 1e-9))
        rate = abs(1.0 - eta)
        radius = max(1.0, rate)
        gain = (
            cls.numerical_transverse_gain(P, eta, probe_key)
            if P is not None
            else rate
        )
        is_contractive = rate < 1.0
        verdict = (
            HeytingOmega3.COHERENT if is_contractive else HeytingOmega3.DEGRADED
        )
        return FlashJacobianSpectrum(
            eta_star=eta,
            transverse_rate=float(rate),
            spectral_radius=float(radius),
            numerical_transverse_gain=float(gain),
            is_contractive_transverse=bool(is_contractive),
            local_verdict=verdict,
        )


# ── §2.4 Criterio de Kelly κ-fraccional ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class KellyStakeReport:
    r"""
    Stake de Kelly κ-fraccional, odds 1:1 (P6).

    Identificación de modelo (no teorema):  p_eff := F(ρ, ρ_target) ∈ [0,1]
    se lee como probabilidad bayesiana de «éxito de alineación».

        f*     = max(0, 2 p_eff − 1)                 Kelly entero
        stake  = κ · f*                              κ ∈ (0, 1]
        no-bet ⟺ p_eff ≤ ½  ⟺  f* = 0

    Crecimiento logarítmico esperado (nats, log natural):

        G(f) = p log(1+f) + (1−p) log(1−f),     f ∈ [0, 1)
        G(f*) = log 2 − h₂(p)                   si p > ½
                (h₂ = entropía binaria en nats)

    Cotas: 0 ≤ stake ≤ κ ≤ 1.
    """
    p_eff: float
    kelly_full: float
    kappa: float
    stake: float
    log_growth: float
    binary_entropy_nats: float
    is_no_bet: bool
    hedge_amount: float
    local_verdict: HeytingOmega3


class KellyStakeCalculator:
    DEFAULT_KAPPA: Final[float] = 0.5

    @classmethod
    def binary_entropy(cls, p: float) -> float:
        """h₂(p) = −p log p − (1−p) log(1−p)  (nats).  h₂(0)=h₂(1)=0."""
        p = float(np.clip(p, 0.0, 1.0))
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))

    @classmethod
    def log_growth(cls, p: float, f: float) -> float:
        p = float(np.clip(p, 0.0, 1.0))
        f = float(np.clip(f, 0.0, 1.0 - 1e-15))
        if f <= 0.0:
            return 0.0
        return float(p * math.log(1.0 + f) + (1.0 - p) * math.log(1.0 - f))

    @classmethod
    def compute(
        cls,
        fidelity: float,
        kappa: float = DEFAULT_KAPPA,
        cost_risk: float = 0.0,
    ) -> KellyStakeReport:
        p_eff = float(np.clip(fidelity, 0.0, 1.0))
        kappa = float(np.clip(kappa, 0.0, 1.0))
        f_full = max(0.0, 2.0 * p_eff - 1.0)
        stake = float(np.clip(kappa * f_full, 0.0, 1.0))
        no_bet = p_eff <= 0.5
        h2 = cls.binary_entropy(p_eff)
        growth = cls.log_growth(p_eff, stake)
        hedge = float(stake * max(0.0, cost_risk))

        if (not no_bet) and stake > 0.0:
            local = HeytingOmega3.COHERENT
        else:
            local = HeytingOmega3.DEGRADED

        return KellyStakeReport(
            p_eff=p_eff,
            kelly_full=float(f_full),
            kappa=kappa,
            stake=stake,
            log_growth=growth,
            binary_entropy_nats=h2,
            is_no_bet=bool(no_bet),
            hedge_amount=hedge,
            local_verdict=local,
        )


# ── §2.5 Contrato de latencia con benchmark estadístico ───────────────────
@dataclass(frozen=True, slots=True)
class LatencyReport:
    r"""
    Contrato de latencia (P5): warm-up + N muestras del hot-path Bures.

        contract_met ⟺ p99 < contract_ns
        Predicado advisory: no veta el flash; degrada si se incumple.
    """
    mean_ns: float
    p50_ns: float
    p99_ns: float
    contract_ns: float
    contract_met: bool
    n_samples: int
    n_warmup: int
    local_verdict: HeytingOmega3


class FlashLatencyBenchmark:
    r"""
    Mide d_B(ρ,σ) en la máquina actual.  Warm-up elimina frío de caché /
    inicialización BLAS; p50/p99 vía percentil empírico (Hyndman R7).
    """

    @classmethod
    def measure_bures(
        cls,
        rho: np.ndarray,
        sigma: np.ndarray,
        contract_ns: float,
        n_warmup: int = 16,
        n_samples: int = 64,
    ) -> LatencyReport:
        n_warmup = max(0, int(n_warmup))
        n_samples = max(8, int(n_samples))
        for _ in range(n_warmup):
            _ = DensityOperatorAlgebra.bures_distance(rho, sigma)

        samples = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            t0 = time.perf_counter_ns()
            _ = DensityOperatorAlgebra.bures_distance(rho, sigma)
            samples[i] = float(time.perf_counter_ns() - t0)

        p50 = float(np.percentile(samples, 50.0))
        p99 = float(np.percentile(samples, 99.0))
        mean = float(samples.mean())
        met = p99 < float(contract_ns)
        local = HeytingOmega3.COHERENT if met else HeytingOmega3.DEGRADED
        return LatencyReport(
            mean_ns=mean,
            p50_ns=p50,
            p99_ns=p99,
            contract_ns=float(contract_ns),
            contract_met=bool(met),
            n_samples=n_samples,
            n_warmup=n_warmup,
            local_verdict=local,
        )


# ── §2.6 Traductor a lenguaje visceral (con números reales) ───────────────
class VisceralSignalTranslator:
    r"""
    Recomendación para el gerente de obra.  Todos los números son los del
    bundle (d_B, F, stake, hedge = cost_risk × stake).  No hay adjetivos
    sin magnitud (P6).
    """

    @classmethod
    def translate(
        cls,
        verdict: HeytingOmega3,
        d_bures: float,
        kelly: KellyStakeReport,
        cost_risk: float,
    ) -> str:
        hedge = float(kelly.hedge_amount if kelly.hedge_amount else
                      kelly.stake * max(0.0, cost_risk))
        if verdict == HeytingOmega3.COHERENT:
            return (
                f"CORAZONADA SANA: ρ alineada con M_decision "
                f"(d_B={d_bures:.4f}, F={kelly.p_eff:.4f}, G={kelly.log_growth:.4f} nats). "
                f"Stake Kelly κ·f* = {kelly.stake:.4f} → luz verde. "
                f"Hedge sugerido: ${hedge:,.0f}."
            )
        if verdict == HeytingOmega3.DEGRADED:
            return (
                f"CORAZONADA DE ALERTA: fricción geodésica detectada "
                f"(d_B={d_bures:.4f}, F={kelly.p_eff:.4f}). "
                f"Stake Kelly κ·f* = {kelly.stake:.4f} "
                f"{'(no-bet)' if kelly.is_no_bet else ''}. "
                f"Riesgo cubierto: ${hedge:,.0f}. "
                f"Pies de plomo en el desembolso."
            )
        return (
            f"CORAZONADA DE VETO CRÍTICO: ρ fuera de la variedad de decisión "
            f"o fraude contextual. d_B={d_bures:.4f}, F={kelly.p_eff:.4f}. "
            f"Válvula de pago cerrada. Interlock BT151 preparado."
        )


# ── §2.7 FlashPipeline — HAND-OFF FASE 2 → FASE 3 ────────────────────────
@dataclass(frozen=True, slots=True)
class FlashTrajectoryBundle:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Producto de los funtores Dirichlet ⊗ Bures ⊗ Jacobiano ⊗ Kelly ⊗ Latencia
    aplicados al FlashHandoff de FASE 1.
    """
    handoff: FlashHandoff
    rho_flash: np.ndarray
    landscape: DirichletLandscape
    jacobian: FlashJacobianSpectrum
    bures_to_target: float
    bures_seed_to_target: float
    theta_to_target: float
    kelly: KellyStakeReport
    latency: LatencyReport
    purity: float
    entropy: float
    energy_decay_ratio: float

    def content_bytes(self) -> bytes:
        """Digest firmable para la cadena Merkle de fases."""
        return hashlib.sha256(
            self.handoff.spectral_hash.encode("ascii")
            + np.ascontiguousarray(self.rho_flash).tobytes()
            + f"{self.landscape.energy:.12e}".encode("ascii")
            + f"{self.bures_to_target:.12e}".encode("ascii")
            + f"{self.kelly.stake:.12e}".encode("ascii")
            + f"{self.latency.p99_ns:.3f}".encode("ascii")
        ).digest()

    def continue_into_phase3(
        self,
        external_verdict: HeytingOmega3,
        eta_star: float,
        reason_prefix: str = "INTUITION-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        Último morfismo de la FASE 2  ∧  primero de la FASE 3.

        Identidad:

            continue_into_phase3 ∘ synthesize ∘ build
                = (adjudicate ⊗ fire) ∘ synthesize ∘ build.
        """
        return FlashPipeline.continue_into_phase3(
            self, external_verdict, eta_star, reason_prefix
        )


class FlashPipeline:
    r"""
    Orquestador determinista de la FASE 2 (funtor F₂).

        synthesize : FlashHandoff × ℝ₊³ → FlashTrajectoryBundle

    Un solo paso Cauchy η* (flash): no es el descenso BB del motor
    espectral; es el reflejo one-shot, Newton exacto si η*=1.

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 2 → FASE 3
    ────────────────────────────────────────────────────────────────────────
    synthesize es el morfismo terminal de la FASE 2.  Su imagen
    FlashTrajectoryBundle es el dominio de TODOS los métodos de FASE 3.
    """
    DEFAULT_ETA_STAR: Final[float] = 1.0
    DEFAULT_LATENCY_NS: Final[float] = 10_000.0  # 10 µs

    @classmethod
    def synthesize(
        cls,
        handoff: FlashHandoff,
        eta_star: float = DEFAULT_ETA_STAR,
        latency_contract_ns: float = DEFAULT_LATENCY_NS,
        kelly_kappa: float = KellyStakeCalculator.DEFAULT_KAPPA,
    ) -> FlashTrajectoryBundle:
        r"""
        Cierra la FASE 2.  Abre la FASE 3.

            land₀      = evaluate(ρ_seed, P)                 §2.1
            ρ_flash    = cauchy_step(ρ_seed, P, η*)          §2.1
            jacobian   = audit(η*, P)                        §2.3
            (d_B, θ_B) = Bures(ρ_flash, ρ_target)            §2.2
            kelly      = compute(F(ρ_flash, ρ_target), κ)    §2.4
            latency    = measure_bures(...)                  §2.5
        """
        P = handoff.geometry.projector
        eta = float(eta_star)

        rho_flash = FlashDirichletFunctional.cauchy_step(handoff.rho_seed, P, eta)
        land_next = FlashDirichletFunctional.evaluate(rho_flash, P)
        jacobian = FlashSpectralJacobian.audit(
            eta, P=P, probe_key=handoff.request_id
        )

        d_b = BuresGeodesicMetric.distance(rho_flash, handoff.rho_target)
        th_b = BuresGeodesicMetric.angle(rho_flash, handoff.rho_target)
        d_seed = DensityOperatorAlgebra.bures_distance(
            handoff.rho_seed, handoff.rho_target
        )
        fid_flash = BuresGeodesicMetric.fidelity(rho_flash, handoff.rho_target)

        kelly = KellyStakeCalculator.compute(
            fid_flash,
            kappa=kelly_kappa,
            cost_risk=handoff.request_audit.cost_risk_amount,
        )
        latency = FlashLatencyBenchmark.measure_bures(
            rho_flash, handoff.rho_target, contract_ns=latency_contract_ns,
        )

        e0 = float(handoff.seed_energy)
        ratio = float(land_next.energy / e0) if e0 > 1e-20 else (
            0.0 if land_next.energy <= 1e-20 else 1.0
        )
        return FlashTrajectoryBundle(
            handoff=handoff,
            rho_flash=rho_flash,
            landscape=land_next,
            jacobian=jacobian,
            bures_to_target=float(d_b),
            bures_seed_to_target=float(d_seed),
            theta_to_target=float(th_b),
            kelly=kelly,
            latency=latency,
            purity=DensityOperatorAlgebra.purity(rho_flash),
            entropy=DensityOperatorAlgebra.von_neumann_entropy(rho_flash),
            energy_decay_ratio=ratio,
        )

    @classmethod
    def continue_into_phase3(
        cls,
        bundle: FlashTrajectoryBundle,
        external_verdict: HeytingOmega3,
        eta_star: float,
        reason_prefix: str = "INTUITION-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        Continuación estricta de `synthesize`.

        Simultáneamente último morfismo de FASE 2 y primero de FASE 3:
        adjudica en Ω₃ y dispara el crowbar.  El certificado se cristaliza
        aguas arriba en el soberano (posee agent_id y Merkle).
        """
        verdict = HeytingIntuitionAdjudicator.adjudicate(bundle, external_verdict)
        reason = (
            f"{reason_prefix}::d_B={bundle.bures_to_target:.4f} "
            f"fraud={bundle.handoff.request_audit.has_critical_fraud} "
            f"kelly={bundle.kelly.stake:.4f} "
            f"η*={eta_star:.3f} "
            f"rate={bundle.jacobian.transverse_rate:.4f} "
            f"lat_met={bundle.latency.contract_met} "
            f"valid={bundle.handoff.request_audit.is_valid}"
        )
        actuation = ESP32CrowbarInterlock.fire(verdict, reason)
        return verdict, actuation


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 · ADJUDICACIÓN + CROWBAR + CERTIFICACIÓN                          ║
# ║                                                                           ║
# ║  Dominio = FlashTrajectoryBundle (codominio de §2.7 synthesize).          ║
# ║  Codominio = IntuitionFlashCertificate (objeto terminal del flash).       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ────────────────────────────────────────────────
class HeytingIntuitionAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃
    por meets sucesivos (producto de subobjetos).  Umbrales adimensionales.

        fraud    : has_critical_fraud ↦ ⊥ else ⊤     (P4, veto duro)
        geom     : d_B(ρ_flash, ρ_tgt) graduado
        spectral : contractivo transversal ↦ ⊤ else ⋆
        kelly    : local_verdict del stake
        valid    : request is_valid ↦ ⊤ else ⋆
        manifold : isometría ∧ proyector ↦ ⊤ else ⊥

    Latencia es advisory: se reporta, no entra en el meet (P5: contrato
    de máquina, no de geometría).

        final = local ∧ external     (meet conservador, nunca infla).
    """
    BURES_COHERENT: Final[float] = 0.20
    BURES_DEGRADED: Final[float] = 0.40

    @classmethod
    def _grade(cls, value: float, hi_ok: float, mid_ok: float) -> HeytingOmega3:
        if value <= hi_ok:
            return HeytingOmega3.COHERENT
        if value <= mid_ok:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _fraud_rule(cls, bundle: FlashTrajectoryBundle) -> HeytingOmega3:
        if bundle.handoff.request_audit.has_critical_fraud:
            return HeytingOmega3.VETOED
        return HeytingOmega3.COHERENT

    @classmethod
    def _geom_rule(cls, bundle: FlashTrajectoryBundle) -> HeytingOmega3:
        return cls._grade(
            bundle.bures_to_target, cls.BURES_COHERENT, cls.BURES_DEGRADED
        )

    @classmethod
    def _spectral_rule(cls, bundle: FlashTrajectoryBundle) -> HeytingOmega3:
        return bundle.jacobian.local_verdict

    @classmethod
    def _kelly_rule(cls, bundle: FlashTrajectoryBundle) -> HeytingOmega3:
        return bundle.kelly.local_verdict

    @classmethod
    def _valid_rule(cls, bundle: FlashTrajectoryBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.handoff.request_audit.is_valid
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _manifold_rule(cls, bundle: FlashTrajectoryBundle) -> HeytingOmega3:
        g = bundle.handoff.geometry
        ok = g.is_isometry and g.is_projector
        return HeytingOmega3.COHERENT if ok else HeytingOmega3.VETOED

    @classmethod
    def adjudicate(
        cls,
        bundle: FlashTrajectoryBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""Continuación de FlashPipeline.synthesize / continue_into_phase3."""
        local = (
            cls._fraud_rule(bundle)
            .meet(cls._geom_rule(bundle))
            .meet(cls._spectral_rule(bundle))
            .meet(cls._kelly_rule(bundle))
            .meet(cls._valid_rule(bundle))
            .meet(cls._manifold_rule(bundle))
        )
        return local.meet(external_verdict)


# ── §3.2 Interlock ciber-físico ESP32 Crowbar ────────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""
    Actuación física (simulada) del crowbar ESP32.

        interlock_fired      : True ⟺ verdict = ⊥
        actuation_latency_ns : cota de diseño < 400 ns (nominal 392.15 ns)
        provenance_hash      : SHA-256(reason ‖ t_ns)
    """
    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str


class ESP32CrowbarInterlock:
    r"""
    FASE 3 · FE ciber-física (axioma operativo).

    Si Ω₃ = ⊥ se arma el crowbar:

        GPIO14 → HIGH  ⇒  MOSFET BT151  ⇒  latencia < 400 ns.

    Este módulo no emite I/O de hardware; certifica la decisión y su
    provenance.  El corte ocurre antes de que la anomalía se propague.
    """
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    @classmethod
    def fire(
        cls, verdict: HeytingOmega3, reason: str = "",
    ) -> CrowbarActuationReport:
        if verdict != HeytingOmega3.VETOED:
            return CrowbarActuationReport(
                interlock_fired=False,
                actuation_latency_ns=0.0,
                gpio_pin=cls.GPIO_PIN,
                device=cls.DEVICE,
                reason="OK",
                provenance_hash="",
            )

        t_ns = time.time_ns()
        payload = f"CROWBAR_INTUITION::{reason}::{t_ns}".encode("utf-8")
        prov = hashlib.sha256(payload).hexdigest()
        logger.critical(
            "[INTUICIÓN — CROWBAR] %s → HIGH | %.2f ns | razón=%s",
            cls.GPIO_PIN, cls.NOMINAL_LATENCY_NS, reason,
        )
        return CrowbarActuationReport(
            interlock_fired=True,
            actuation_latency_ns=cls.NOMINAL_LATENCY_NS,
            gpio_pin=cls.GPIO_PIN,
            device=cls.DEVICE,
            reason=reason,
            provenance_hash=prov,
        )


# ── §3.3 Certificado del flash ────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntuitionFlashCertificate:
    r"""
    Objeto terminal del flash: producto fibrado firmado

        Certificate ≅ Bundle × Ω₃ × Crowbar × Merkle.

    Distancias Bures desambiguadas (P1):
        bures_manifold_distance : d_B(ρ_flash, ρ_target)   calidad post-flash
        bures_seed_to_target    : d_B(ρ_seed,  ρ_target)   geodésica de F1
    """
    flash_id: str
    agent_id: str
    crop_origin_id: str
    seed_crystal_id: str
    heyting_verdict: HeytingOmega3
    bures_manifold_distance: float
    bures_seed_to_target: float
    bures_angle_rad: float
    energy_initial: float
    energy_flash: float
    energy_decay_ratio: float
    mass_on_P: float
    transverse_rate: float
    numerical_jacobian_gain: float
    kelly_p_eff: float
    kelly_stake: float
    kelly_log_growth: float
    kelly_hedge_amount: float
    latency_p50_ns: float
    latency_p99_ns: float
    latency_contract_met: bool
    purity: float
    entropy: float
    crowbar_interlock: CrowbarActuationReport
    visceral_recommendation: str
    manifold_hash: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.4 SOBERANO DE LA INTUICIÓN ────────────────────────────────────────
class TOONIntuitionAgent:
    r"""
    Soberano de la Intuición y Reflejo Flash.

    Funtor soberano  F = F₃ ∘ F₂ ∘ F₁ :

        F₁  FlashHandoff.build
        F₂  FlashHandoff.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ certify

    Asociatividad (teorema de anidamiento):

        synthesize_intuitive_flash
            = _phase3_certify ∘ _phase2_flash ∘ _phase1_handoff
            = certify ∘ synthesize ∘ build.
    """

    def __init__(
        self,
        agent_id: str = "INTUITION-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        manifold_rank: int = 2,
        latency_contract_ns: float = FlashPipeline.DEFAULT_LATENCY_NS,
        kelly_kappa: float = KellyStakeCalculator.DEFAULT_KAPPA,
        eta_star: float = FlashPipeline.DEFAULT_ETA_STAR,
    ) -> None:
        if not (1 <= manifold_rank < dimension_mac):
            raise ValueError(
                f"manifold_rank ∈ [1, n−1]; r={manifold_rank}, n={dimension_mac}"
            )
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.latency_contract_ns = float(latency_contract_ns)
        self.kelly_kappa = float(kelly_kappa)
        self.eta_star = float(eta_star)
        self.flash_count = 0

        self.manifold = DecisionManifoldFactory.build(
            n=self.dimension_mac, rank=manifold_rank, key=agent_id,
        )
        self._chain_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::n={dimension_mac}::r={manifold_rank}::"
            f"m={self.manifold.hash}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_handoff(self, request: IntuitiveFlashRequest) -> FlashHandoff:
        """FASE 1 anidada: cierra con FlashHandoff (dominio de FASE 2)."""
        handoff = FlashHandoff.build(
            cycle_index=self.flash_count,
            request=request,
            geometry=self.manifold,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.spectral_hash))
        return handoff

    def _phase2_flash(self, handoff: FlashHandoff) -> FlashTrajectoryBundle:
        """FASE 2 anidada: continuación de build; cierra con Bundle."""
        bundle = handoff.continue_into_phase2(
            eta_star=self.eta_star,
            latency_contract_ns=self.latency_contract_ns,
            kelly_kappa=self.kelly_kappa,
        )
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    def _phase3_certify(
        self,
        flash_id: str,
        bundle: FlashTrajectoryBundle,
        external_verdict: HeytingOmega3,
    ) -> IntuitionFlashCertificate:
        """FASE 3 anidada: continuación de synthesize; cierra con Certificate."""
        final_verdict, crowbar = bundle.continue_into_phase3(
            external_verdict, self.eta_star
        )
        self._advance_chain(
            "F3",
            f"{final_verdict.name}|{crowbar.provenance_hash}".encode("ascii"),
        )
        recommendation = VisceralSignalTranslator.translate(
            verdict=final_verdict,
            d_bures=bundle.bures_to_target,
            kelly=bundle.kelly,
            cost_risk=bundle.handoff.request_audit.cost_risk_amount,
        )
        provenance = _sha256_bytes(
            self.agent_id.encode("ascii"),
            flash_id.encode("ascii"),
            bundle.handoff.crop_origin_id.encode("ascii"),
            bundle.handoff.seed_crystal_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{bundle.bures_to_target:.12e}".encode("ascii"),
            f"{bundle.kelly.stake:.12e}".encode("ascii"),
            f"{bundle.latency.p99_ns:.3f}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        return IntuitionFlashCertificate(
            flash_id=flash_id,
            agent_id=self.agent_id,
            crop_origin_id=bundle.handoff.crop_origin_id,
            seed_crystal_id=bundle.handoff.seed_crystal_id,
            heyting_verdict=final_verdict,
            bures_manifold_distance=bundle.bures_to_target,
            bures_seed_to_target=bundle.bures_seed_to_target,
            bures_angle_rad=bundle.theta_to_target,
            energy_initial=float(bundle.handoff.seed_energy),
            energy_flash=float(bundle.landscape.energy),
            energy_decay_ratio=bundle.energy_decay_ratio,
            mass_on_P=bundle.handoff.mass_on_P,
            transverse_rate=bundle.jacobian.transverse_rate,
            numerical_jacobian_gain=bundle.jacobian.numerical_transverse_gain,
            kelly_p_eff=bundle.kelly.p_eff,
            kelly_stake=bundle.kelly.stake,
            kelly_log_growth=bundle.kelly.log_growth,
            kelly_hedge_amount=bundle.kelly.hedge_amount,
            latency_p50_ns=bundle.latency.p50_ns,
            latency_p99_ns=bundle.latency.p99_ns,
            latency_contract_met=bundle.latency.contract_met,
            purity=bundle.purity,
            entropy=bundle.entropy,
            crowbar_interlock=crowbar,
            visceral_recommendation=recommendation,
            manifold_hash=bundle.handoff.geometry.hash,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    def synthesize_intuitive_flash(
        self,
        request: IntuitiveFlashRequest,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> IntuitionFlashCertificate:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        self.flash_count += 1
        flash_id = f"FLASH-INTUITION-{self.flash_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Flash #%d | req=%s::%s | manifold=%s r=%d ═══",
            self.flash_count, request.crop_origin_id, request.seed_crystal_id,
            self.manifold.hash[:12], self.manifold.rank,
        )
        handoff = self._phase1_handoff(request)
        bundle = self._phase2_flash(handoff)
        cert = self._phase3_certify(flash_id, bundle, external_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Flash %s | Ω₃=%s | d_B=%.4f | κ·f*=%.4f | p99=%.0f ns | %.2f ms",
            flash_id, cert.heyting_verdict.name, cert.bures_manifold_distance,
            cert.kelly_stake, cert.latency_p99_ns, dt_ms,
        )
        return cert


# ── §3.5 Demostración autónoma ───────────────────────────────────────────
def _build_state_leak(
    n: int,
    manifold: DecisionManifoldGeometry,
    leakage: float,
    key: str,
) -> ComplexMatrix:
    r"""
    Estado ρ_t con fuga controlada t ∈ [0,1] al complemento P_⊥:

        ρ_t = (1−t) · ρ_P + t · ρ_{P_⊥}

    ρ_P (resp. ρ_{P_⊥}) es el estado Wishart proyectado y renormalizado
    sobre ran(P) (resp. ran(P_⊥)).

        t = 0 → supp(ρ) ⊆ ran(P)   ⇒  E[ρ] = 0
        t = 1 → supp(ρ) ⊆ ran(P_⊥) ⇒  E[ρ] máximo, intuición vacía
    """
    rng = np.random.default_rng(_seed_from_string(f"LEAK::{key}"))
    P = manifold.projector
    Pc = np.eye(n, dtype=np.complex128) - P

    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    rho_raw = A @ A.conj().T
    tr_raw = float(np.trace(rho_raw).real)
    rho_raw = rho_raw / max(tr_raw, 1e-30)

    def _normalize_support(M_side: np.ndarray) -> ComplexMatrix:
        r = M_side @ rho_raw @ M_side
        tr = float(np.trace(r).real)
        if tr < 1e-15:
            rr = max(1.0, float(np.real(np.trace(M_side))))
            return M_side / rr
        return r / tr

    rho_P = _normalize_support(P)
    rho_Pc = _normalize_support(Pc)
    t = float(np.clip(leakage, 0.0, 1.0))
    return DensityOperatorAlgebra.sanitize((1.0 - t) * rho_P + t * rho_Pc)


if __name__ == "__main__":
    print("═" * 92)
    print("SOBERANO DE LA INTUICIÓN — v2.2.0 Nested Doctoral")
    print("Bures · Jacobiano DT_η · Kelly-G* · Latencia p99 · Heyting Ω₃ · Merkle")
    print("═" * 92)

    agent = TOONIntuitionAgent(
        agent_id="INTUITION-SOVEREIGN-SABIO-01",
        dimension_mac=4,
        manifold_rank=2,
        latency_contract_ns=10_000.0,
        kelly_kappa=0.5,
        eta_star=1.0,
    )

    print(
        f"\nManifold r={agent.manifold.rank} | "
        f"iso={agent.manifold.is_isometry} | proj={agent.manifold.is_projector} | "
        f"‖B†B−I‖_F={agent.manifold.isometry_residual:.2e} | "
        f"‖P²−P‖_F={agent.manifold.projector_residual:.2e}"
    )
    print(f"Manifold hash: {agent.manifold.hash[:32]}…")
    print(
        f"η*={agent.eta_star} ⇒ rate transversal analítico = "
        f"{abs(1.0 - agent.eta_star):.4f}  (Newton exacto si η*=1)"
    )

    a = _build_state_leak(4, agent.manifold, 0.00, "TRI-A")
    b = _build_state_leak(4, agent.manifold, 0.50, "TRI-B")
    c = _build_state_leak(4, agent.manifold, 1.00, "TRI-C")
    tri = BuresGeodesicMetric.triangle_residual(a, b, c)
    print(f"\nTest Bures ► δ_triangular = {tri:.3e} (esperado ≈ 0)")
    print(f"             d_B(A,B) = {BuresGeodesicMetric.distance(a, b):.6f}")
    print(f"             d_B(A,C) = {BuresGeodesicMetric.distance(a, c):.6f}")
    gamma_mid = BuresGeodesicMetric.geodesic(a, c, 0.5)
    print(
        f"             d_B(A, γ(½)) = "
        f"{BuresGeodesicMetric.distance(a, gamma_mid):.6f}  (geodésica W₂)"
    )

    scenarios = [
        ("COHERENT (leak=0.05)", 0.05, HeytingOmega3.COHERENT, False, 12_000_000.0),
        ("DEGRADED (leak=0.80)", 0.80, HeytingOmega3.COHERENT, False, 12_000_000.0),
        ("VETOED   (fraude)", 0.10, HeytingOmega3.COHERENT, True, 250_000_000.0),
    ]

    print("\n" + "─" * 92)
    for name, leak, ext, fraud, cost in scenarios:
        rho = _build_state_leak(4, agent.manifold, leak, name)
        req = IntuitiveFlashRequest(
            crop_origin_id=f"CROP-SOVEREIGN-0001::{name[:20]}",
            seed_crystal_id=f"CRYSTAL-{name[:10]}",
            germinated_density_matrix=rho,
            site_context_payload={
                "cost_risk_amount": cost,
                "has_critical_fraud": fraud,
            },
        )
        cert = agent.synthesize_intuitive_flash(req, external_verdict=ext)

        print(f"\n[{name}]")
        print(f"   flash_id              : {cert.flash_id}")
        print(f"   Ω₃ final              : {cert.heyting_verdict.name}")
        print(f"   mass_P(ρ_seed)        : {cert.mass_on_P:.6f}")
        print(f"   d_B(seed, target)     : {cert.bures_seed_to_target:.6f}")
        print(f"   d_B(flash, target)    : {cert.bures_manifold_distance:.6f}")
        print(f"   θ_B                   : {cert.bures_angle_rad:.6f} rad")
        print(f"   E_initial             : {cert.energy_initial:.6e}")
        print(f"   E_flash               : {cert.energy_flash:.6e}")
        print(f"   ratio decaimiento     : {cert.energy_decay_ratio:.6f}")
        print(
            f"   rate transversal      : {cert.transverse_rate:.6f}  "
            f"(sonda numérica = {cert.numerical_jacobian_gain:.6f})"
        )
        print(f"   p_eff = F             : {cert.kelly_p_eff:.6f}")
        print(
            f"   stake κ·f*            : {cert.kelly_stake:.6f}  "
            f"G={cert.kelly_log_growth:.4f} nats  hedge=${cert.kelly_hedge_amount:,.0f}"
        )
        print(
            f"   latencia p50 / p99    : {cert.latency_p50_ns:,.0f} / "
            f"{cert.latency_p99_ns:,.0f} ns"
        )
        print(
            f"   contrato latencia     : "
            f"{'✓' if cert.latency_contract_met else '✗'} (<10 µs)"
        )
        print(f"   crowbar listo         : {cert.crowbar_interlock.interlock_fired}")
        print(f"   recomendación visceral: {cert.visceral_recommendation[:78]}…")
        print(f"   firma (phase_chain)   : {cert.phase_chain_sha256[:32]}…")
        print(f"   firma (provenance)    : {cert.sha256_provenance[:32]}…")

    print("\n" + "═" * 92)
    print("✓ F1→F2: build ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate ⊗ fire.")
    print("✓ d_B=√(2−2√F): geodésica de Bures, no Frobenius (P1).")
    print("✓ DT_η = {1}⊕{1−η}; η*=1 es Newton exacto, Lip(∇E)≤1 (P2).")
    print("✓ Stiefel Haar ← SHA-256(agent_id), no rng(42) (P3).")
    print("✓ Fraude crítico ⇒ ⊥ duro; Ω₃ por meets, no conteo (P4).")
    print("✓ Latencia p50/p99 advisory; Kelly G*=log 2 − h(F) (P5, P6).")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256.")
    print("═" * 92)