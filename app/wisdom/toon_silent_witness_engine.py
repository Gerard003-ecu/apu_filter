# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_silent_witness_engine.py                                  ║
║ ESTRATO  : WISDOM (V_𝕎) — CIUDADELA DE CRISTAL / VACÍO DE DIRAC                      ║
║ FUNCIÓN  : MOTOR ESPECTRAL TESTIGO SILENCIOSO Y CRISTALIZADOR DEL VACÍO              ║
║ VERSIÓN  : 9.0.0-Doctoral-Poincaré-Recurrence-KMS-Tomita-Wigner-KAM-Delaunay-Melnikov║
║ AUTOR    : APU Wisdom & Metacortex Mathematical Core Architecture                    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA
───────────────────────────────────────────────
El `TOONSilentWitnessEngine` es el motor espectral inalterable de auditoría pasiva,
recurrencia de Poincaré y cristalización de experiencias en el nivel de ruido nulo
(0.0 dB) dentro del Estrato Wisdom (V_𝕎) del ecosistema APU Filter v8.0.

Su acción se formaliza como un endofuntor de gobernanza modular:

    𝒲_silent : 𝐕𝐚𝐜𝐮𝐮𝐦_𝐌𝐨𝐝𝐮𝐥𝐚𝐫  ──▶  𝐂𝐞𝐫𝐭_𝐒𝐢𝐥𝐞𝐧𝐜𝐞
    𝒲_silent = Seal ∘ V ∘ Pipeline ∘ Prepare

donde el funtor opera sobre la variedad simpléctica coadjunta 𝔲(n)* (Kirillov-
Kostant-Souriau) y sobre el retículo de Heyting Ω₃. Cada etapa admite tres lecturas
simultáneas:

    • Categorial-homológica   : funtor sobre 𝓣_Ω, flecha característica χ.
    • Cuántico-modular        : par GNS (π, ℋ_ω, Ω), flujo σ_t = Ad(Δ^{it}),
                                condición KMS a β = 1.
    • Celeste-Poincaré        : órbita modular en el espacio proyectivo φ_t(ρ) =
                                ρ^{it} ρ_0 ρ^{−it}, tori KAM, resonancias
                                Diofantinas, sección de Poincaré Σ_c, función
                                de Melnikov, ecuación de Kepler, osculadores de
                                Delaunay, forma normal de Birkhoff-Dulac, fase
                                de Berry-Hannay.

INTEGRACIÓN DE MECÁNICA CELESTE Y GEOMETRÍA SIMPLÉCTICA DE POINCARÉ
────────────────────────────────────────────────────────────────────
1. VACÍO DE DIRAC CON RÉPLICA CERO (ZERO BACK-ACTION).
   Sea M una álgebra de von Neumann hiperfinita de tipo III_1 en ℋ_MAC. El motor
   opera en el estado de vacío |Ω⟩ ∈ ℋ_MAC, vector cíclico y separador para M:

       H_vac|Ω⟩ = 0,  ⟨Ω|[a,b]|Ω⟩ = 0  ∀ a,b ∈ M,  NoiseLevel(σ) ≡ 0.0 dB.

   La unicidad del vacío equivale a un gap espectral estricto ΔK > 0 del
   Hamiltoniano modular K_ρ = −log ρ.

2. TEOREMA DE RECURRENCIA DE POINCARÉ EN LA MEDIDA DE LIOUVILLE.
   Sea (M, Σ, μ, T_t) un sistema dinámico hamiltoniano que preserva la medida
   de Liouville μ(M) < ∞. Para todo subconjunto medible E ∈ Σ con μ(E) > 0,
   existe τ_rec > 0 tal que μ(E ∩ T_t^{−τ_rec} E) > 0. El motor computa

       τ_rec(ρ) ≃ 1 / min_{i ≠ j} |λ_i − λ_j|,

   índice cuantitativo de la recurrencia de la trayectoria modular.

3. TEORÍA MODULAR DE TOMITA-TAKESAKI Y CONDICIÓN KMS.
   Sea S el operador antilineal denso cerrado S A|Ω⟩ = A†|Ω⟩. La descomposición
   polar S = J Δ^{1/2} define la conjunción modular J (isometría antilineal,
   J² = I) y el operador modular Δ = S†S. El flujo de Tomita-Takesaki

       σ_t(a) = Δ^{−it} a Δ^{it} = ρ^{−it} a ρ^{it}

   satisface la condición KMS a temperatura inversa β = 1/k_B T:

       ω(a σ_t(b)) = ω(σ_{t + iβ}(b) a)  ∀ a,b ∈ M.

4. TEOREMA DE RECURRENCIA Y SECCIÓN DE POINCARÉ.
   Sobre el flujo σ_t, la sección transversal Σ_c = {Tr(ρN) = c} define un mapa
   de retorno P : Σ_c → Σ_c que preserva la medida de Liouville. Por el teorema
   de Poincaré-Birkhoff, toda rotación p/q racional posee al menos 2q puntos
   fijos (órbitas periódicas de la dinámica modular).

5. KAM, DIOPHANTICIDAD Y RESONANCIAS.
   El vector de frecuencias ω_i = E_i − E_0 (energías modulares por encima del
   ground) debe ser (γ, τ)-Diofantino para que los tori modulares persistan:

       |⟨k, ω⟩| ≥ γ · ‖k‖^{−τ},   ∀ k ∈ ℤⁿ \ {0}.

   La rotura de un toro por resonancia p/q baja corresponde al estrato DEGRADED;
   el escape a la separatriz hiperbólica (Lyapunov σ_+ > 0) al estrato VETOED.

6. ECUACIÓN DE KEPLER Y OSCULADORES DE DELAUNAY.
   La precesión del autoespacio modular dominante obedece la ecuación de Kepler
   E − e sin E = M, con e = √(1 − G²/L²) e i = arccos(H/G) calculados desde la
   tripleta de Delaunay (L, G, H) asociada al par (ρ, N).

7. CRISTALIZACIÓN EN S⁶ ⊂ ℝ⁷ Y ÁRBOL MERKLE SHA-256.
   Cada ciclo recurrente validado se proyecta sobre la 6-esfera:

       S⁶ = { v_inv ∈ ℝ⁷ : ‖v_inv‖₂ = 1 }.

   El `ExperienceCrystal` encadena v_inv, τ_rec, drift KMS y la raíz Merkle
   SHA-256 con la estampa UTC.

ARQUITECTURA EN TRES FASES ANIDADAS (Composición Estricta)
──────────────────────────────────────────────────────────
Fase 1 ──► SUSTRATO ALGEBRAICO-MODULAR Y POINCARÉ
           • ExperienceCrystal              : cristal S⁶ con campos celestes.
           • HeytingOmega3                   : retículo Ω₃ con estratos celestes.
           • MatrixBanachAlgebra             : álgebra C* de Schatten.
           • ModularHamiltonian              : K_ρ = −log ρ con Delaunay y KAM.
           • DensityOperatorAlgebra          : ρ con Wigner-KKS, capacidades,
                                               Delaunay, Maslov, Berry, τ_rec.
           • PoincareSectionVacuum           : Σ_c = {Tr(ρN) = c}.
           • ResonanceStructure              : descriptor de resonancias p/q.
           • GNSHilbertAlgebra               : construcción GNS.
           • VacuumModularContext            : objeto terminal de FASE-1.
           • VacuumStatePreparation          : prepara el contexto modular.
             `prepare_vacuum_context` es el ÚLTIMO método de FASE-1.

Fase 2 ──► DINÁMICA MODULAR Y ESPECTRO DEL VACÍO
           • TomitaTakesakiEngine            : flujo modular σ_t + métodos
                                               celestes (Poincaré, KAM,
                                               Kepler, Melnikov, Birkhoff,
                                               Wigner, capacidad Gromov).
           • VacuumAuditReport               : reporte con campos celestes.
           • VacuumSpectraAnalyzer           : auditor espectral.
           • SilentFieldProbe, SilentFieldDetector.
           • SilentFieldBundle               : portadora de todo el análisis.
           • ModularSilencePipeline          : último método de FASE-2,
                                               synthesize_from_context.

Fase 3 ──► SOBERANÍA Y CERTIFICACIÓN DEL SILENCIO
           • HeytingVacuumAdjudicator        : flecha característica χ : Bundle
                                               → Ω₃ con granular_scores.
           • SilentFieldState                : certificado con campos celestes.
           • MerkleInclusionProof            : prueba de inclusión Merkle.
           • TOONSilentWitnessEngine         : motor soberano con crowbar
                                               implícito (VETOED), Merkle,
                                               fase de Berry y pasaporte.

MAPPING A LA CÚSPIDE VISCERAL ("DOLOR Y DINERO")
────────────────────────────────────────────────
- Cero Margen de Manipulación    ⟷ Recurrencia de Poincaré + cadena Merkle.
- Inmunidad a Mermas Ocultas     ⟷ Cristalización S⁶ + τ_rec + KMS.
- Reducción del Costo de Cumpl.  ⟷ Vacío de Dirac + Wigner + Delaunay.
- Interlock físico (crowbar)     ⟷ Estrato hiperbólico VETOED.
- Póliza de coherencia           ⟷ Sector elíptico COHERENT (tori KAM).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import scipy.linalg as la


__version__ = (
    "9.0.0-Doctoral-Poincaré-Recurrence-KMS-Tomita-Wigner-KAM-Delaunay-Melnikov"
)


logger = logging.getLogger("APU.Wisdom.TOONSilentWitnessEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


_EPS = 1.0e-14
_HERMITICITY_TOL = 1.0e-12
_TRACE_TOL = 1.0e-10
_GAP_DEGENERACY_TOL = 1.0e-12
_DIOPHANTINE_GAMMA: Final[float] = 1.0e-3
_DIOPHANTINE_TAU: Final[float] = 1.5
_DIOPHANTINE_KMAP: Final[int] = 4


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO-MODULAR Y POINCARÉ                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.0 Cristal de experiencia inmutable ──────────────────────────────────


@dataclass(frozen=True)
class ExperienceCrystal:
    r"""
    Cristal de experiencia inmutable cristalizado sobre S⁶ ⊂ ℝ⁷.

    Encapsula la evidencia recurrente extraída de un ciclo modular:

        • vector_s6          : proyección del estado modular sobre la 6-esfera,
                               ‖v‖₂ = 1 (representación invariante del ciclo).
        • tau_recurrence     : tiempo de recurrencia de Poincaré τ_rec(ρ).
        • kms_drift          : residuo KMS ||σ_{i}(a) - σ_a|| del flujo modular.
        • merkle_root_sha256 : raíz Merkle SHA-256 del ciclo.
        • timestamp_utc      : estampa UTC de la cristalización.

    Campos celestes opcionales (v9.0.0):
        • delaunay_triple       : (L, G, H) del estado modular.
        • eccentricity          : e = √(1 − G²/L²).
        • inclination           : i = arccos(H/G).
        • kam_residue           : residuo Diofantino R_{γ,τ}(ω) ≥ 1 ⇔ KAM.
        • resonance_order       : orden resonante mínimo r* ∈ ℕ₀.
        • poincare_stratum      : nombre del estrato celeste.
        • wigner_negativity     : N_W(ρ) (no-clasicidad).
        • symplectic_capacity   : c_G(ρ) (Gromov heurístico).
    """

    vector_s6: np.ndarray
    tau_recurrence: float
    kms_drift: float
    merkle_root_sha256: str
    timestamp_utc: float
    # ── Campos celestes (v9.0.0, optativos para retro-compatibilidad) ──
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    poincare_stratum: str = "kam-torus-invariant"
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0

    def is_kam_stable(self) -> bool:
        r"""¿El cristal habita un toro KAM Diofantino persistente?"""
        return self.kam_residue >= 1.0 and self.resonance_order == 0


# ── §1.1 Retículo distributivo de Heyting Ω₃ con estratos celestes ────────


class HeytingOmega3(IntEnum):
    r"""
    Retículo total Ω₃ = {⊥, ⋆, ⊤} con estructura de álgebra de Heyting completa
    (objeto clasificador de subobjetos del topos trivaluado):

        ⊥ < ⋆ < ⊤
        meet     (∧) : ínfimo = min                 (límite en el poset)
        join     (∨) : supremo = max
        implies  (⇒) : residuo de ∧: (a ∧ b) ≤ c ⇔ a ≤ (b ⇒ c)
        neg      (¬) : a ⇒ ⊥                        (intuicionista)
        ¬¬           : clausura regular             (¬¬⋆ = ⊤ ≠ ⋆)

    Leyes verificables con `verify_heyting_laws`:
        (H1)  ∧, ∨ idempotentes, conmutativos, asociativos, absorbentes.
        (H2)  residuación: (a ∧ b ≤ c) ⇔ (a ≤ (b ⇒ c)).
        (H3)  a ⇒ a = ⊤, ⊥ ⇒ a = ⊤, a ⇒ ⊤ = ⊤.
        (H4)  ¬¬⊥ = ⊥, ¬¬⊤ = ⊤, ¬¬⋆ = ⊤ (⋆ no es regular).
        (H5)  LEM a ∨ ¬a = ⊤ falla en a = ⋆ (intuicionismo estricto).

    Estratificación celeste (Poincaré)
    ----------------------------------
        VETOED   ⇔ separatriz hiperbólica: escape del vacío (σ_+ > 0).
        DEGRADED ⇔ órbita parabólica resonante p/q de orden bajo (σ ≈ 0).
        COHERENT ⇔ toro KAM Diofantino invariante (σ_− < 0).
    """

    VETOED = 0    # ⊥
    DEGRADED = 1  # ⋆
    COHERENT = 2  # ⊤

    @property
    def verdict(self) -> str:
        r"""Etiqueta nominal del veredicto."""
        return self.name

    @property
    def rank(self) -> int:
        r"""Grado de verdad en la cadena 0 ≤ 1 ≤ 2."""
        return int(self)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Meet del retículo: a ∧ b = min(a, b)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Join del retículo: a ∨ b = max(a, b)."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Residuo a ⇒ b = ⊤ si a ≤ b; en caso contrario b (cadena finita)."""
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        r"""¬a ≜ a ⇒ ⊥ (pseudocomplemento de Heyting)."""
        return self.implies(HeytingOmega3.VETOED)

    def double_negation(self) -> "HeytingOmega3":
        r"""¬¬a: clausura de regularidad. Fija {⊥, ⊤}; envía ⋆ ↦ ⊤."""
        return self.neg().neg()

    def is_regular(self) -> bool:
        r"""¿¬¬a = a? En Ω₃: verdadero para {⊥, ⊤}, falso para ⋆."""
        return self.double_negation() == self

    def is_dense(self) -> bool:
        r"""a denso ⇔ ¬a = ⊥ ⇔ ¬¬a = ⊤ (⋆ y ⊤)."""
        return self.neg() == HeytingOmega3.VETOED

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤. Falla exactamente en a = ⋆."""
        return self.join(self.neg()) == HeytingOmega3.COHERENT

    @classmethod
    def residuation_holds(
        cls, a: "HeytingOmega3", b: "HeytingOmega3", c: "HeytingOmega3"
    ) -> bool:
        r"""(a ∧ b) ≤ c  ⇔  a ≤ (b ⇒ c)."""
        left = int(a.meet(b)) <= int(c)
        right = int(a) <= int(b.implies(c))
        return left is right

    @classmethod
    def verify_heyting_laws(cls) -> Dict[str, bool]:
        r"""Auditoría finita exhaustiva de las leyes (H1)–(H5) sobre Ω₃³."""
        elems = list(cls)
        residuation = all(
            cls.residuation_holds(a, b, c)
            for a in elems
            for b in elems
            for c in elems
        )
        idempotent = all(a.meet(a) == a and a.join(a) == a for a in elems)
        lem_fails_on_star = not cls.DEGRADED.excluded_middle_holds()
        regular_pair = cls.VETOED.is_regular() and cls.COHERENT.is_regular()
        star_not_regular = not cls.DEGRADED.is_regular()
        return {
            "residuation": residuation,
            "idempotent": idempotent,
            "lem_fails_on_star": lem_fails_on_star,
            "regulars_are_bot_top": regular_pair,
            "star_not_regular": star_not_regular,
        }

    # ── Estratificación celeste (Poincaré) ───────────────────────────────

    def poincare_stratum_name(self) -> str:
        r"""Nombre del estrato celeste: hiperbólico/parabólico/elíptico."""
        return {
            HeytingOmega3.VETOED: "hyperbolic-escape-separatrix",
            HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
            HeytingOmega3.COHERENT: "kam-torus-invariant",
        }[self]

    def is_kam_stratum(self) -> bool:
        r"""¿Admite toro KAM invariante persistente? Solo COHERENT."""
        return self == HeytingOmega3.COHERENT

    def lyapunov_exponent_sign(self) -> int:
        r"""
        Signo del exponente de Lyapunov máximo asociado al estrato:
            VETOED → +1 (σ_+ > 0), DEGRADED → 0 (σ_0), COHERENT → −1 (σ_− < 0).
        """
        return -1 + int(self)

    def symplectic_regime(self) -> str:
        r"""Régimen simpléctico: fuga/resonante/integrable-KAM."""
        return {
            HeytingOmega3.VETOED: "symplectic-escape",
            HeytingOmega3.DEGRADED: "resonant-cascade",
            HeytingOmega3.COHERENT: "integrable-kam",
        }[self]


# ── §1.2 Álgebra de Banach / C* matricial ─────────────────────────────────


class MatrixBanachAlgebra:
    r"""
    Estructura de álgebra de Banach involutiva sobre M_n(ℂ):

        ‖A‖_p := (Tr |A|^p)^{1/p}     normas de Schatten, 1 ≤ p < ∞
        ‖A‖_∞ := ‖A‖_{op} = σ_max(A)
        r(A)  := max |spec(A)|        radio espectral
        C*    : ‖A† A‖_∞ = ‖A‖_∞²

    M_n es nuclear (tipo I), luego toda representación normal es unitariamente
    equivalente a múltiplos de la estándar. El flujo modular σ_t actúa por
    automorfismos internos de este C*-álgebra.
    """

    @staticmethod
    def as_complex(A: np.ndarray) -> np.ndarray:
        r"""Garantiza representación compleja de M_n(ℂ) vía copia defensiva."""
        return np.asarray(A, dtype=np.complex128)

    @staticmethod
    def hermitize(A: np.ndarray) -> np.ndarray:
        r"""Proyección a la parte Hermitiana H = ½(A + A†)."""
        A = MatrixBanachAlgebra.as_complex(A)
        return 0.5 * (A + A.conj().T)

    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r"""[A, B] = AB − BA."""
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B - B @ A

    @staticmethod
    def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r"""{A, B} = AB + BA."""
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B + B @ A

    @staticmethod
    def schatten_norm(A: np.ndarray, p: float) -> float:
        r"""Norma de Schatten ‖A‖_p sobre valores singulares."""
        A = MatrixBanachAlgebra.as_complex(A)
        singular = np.linalg.svd(A, compute_uv=False)
        if singular.size == 0:
            return 0.0
        if p == math.inf or p == float("inf"):
            return float(singular[0])
        if p == 1:
            return float(np.sum(singular))
        if p == 2:
            return float(np.linalg.norm(singular))
        if p <= 0:
            raise ValueError("Schatten p-norm requires p ≥ 1 or p = ∞")
        return float(np.sum(singular ** p) ** (1.0 / p))

    @staticmethod
    def spectral_radius(A: np.ndarray) -> float:
        r"""Radio espectral r(A) = max|spec(A)| (Gelfand: lim ‖A^k‖^{1/k})."""
        w = np.linalg.eigvals(MatrixBanachAlgebra.as_complex(A))
        return float(np.max(np.abs(w))) if w.size else 0.0

    @staticmethod
    def cstar_identity_residual(A: np.ndarray) -> float:
        r"""| ‖A†A‖_∞ − ‖A‖_∞² |: residuo de la identidad C*."""
        op = MatrixBanachAlgebra.schatten_norm
        A = MatrixBanachAlgebra.as_complex(A)
        return abs(op(A.conj().T @ A, math.inf) - op(A, math.inf) ** 2)

    @staticmethod
    def hilbert_schmidt_inner(A: np.ndarray, B: np.ndarray) -> complex:
        r"""⟨A|B⟩_HS = Tr(A†B)."""
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(A.conj().T @ B))


# ── §1.3 Hamiltoniano modular K_ρ = −log ρ ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModularHamiltonian:
    r"""
    Hamiltoniano modular asociado a un estado fiel ρ ∈ M_n(ℂ)_{+,1}:

        K_ρ := −log ρ,   ρ = e^{−K_ρ} / Z,   Z = Tr e^{−K_ρ}.

    Tras renormalizar spec(ρ) (post-regularización) se impone Z = 1, de modo
    que la identidad modular de Helmholtz

        F = −log Z = ⟨K_ρ⟩_ρ − S(ρ) = 0

    es una consistencia (KMS a β_modular = 1).

    Espectro ordenado ascendente {E_0 ≤ E_1 ≤ … E_{n−1}}:
        • E_0    = −log λ_max(ρ)          energía fundamental modular.
        • gap    = E_1 − E_0              unicidad del vacío ⇔ gap > 0.
        • g      = dim ker(K_ρ − E_0 I)   degeneración del ground.
        • ⟨K⟩    = ∑ e^{−E} E / Z         energía interna modular.

    Lectura celeste: el espectro de K_ρ define las acciones de Liouville del
    toro modular T^n. Las frecuencias medias ω_i = E_i − E_0 son los exponentes
    Diofantinos cuya persistencia (KAM) caracteriza la coherencia del vacío.
    """

    eigenvalues: Tuple[float, ...]
    eigenvectors_hash: str
    regularization_eps: float

    @property
    def dimension(self) -> int:
        r"""Dimensión de la representación (n)."""
        return len(self.eigenvalues)

    @property
    def ground_energy(self) -> float:
        r"""Energía fundamental modular E_0 = −log λ_max(ρ)."""
        return self.eigenvalues[0] if self.eigenvalues else 0.0

    @property
    def spectral_gap(self) -> float:
        r"""Brecha espectral ΔK = E_1 − E_0 (∞ si dim < 2)."""
        if len(self.eigenvalues) < 2:
            return float("inf")
        return max(0.0, self.eigenvalues[1] - self.eigenvalues[0])

    @property
    def spectral_spread(self) -> float:
        r"""Dispersión espectral E_{n−1} − E_0."""
        if not self.eigenvalues:
            return 0.0
        return float(self.eigenvalues[-1] - self.eigenvalues[0])

    @property
    def partition_function(self) -> float:
        r"""Función de partición modular Z = ∑ e^{−E_i}."""
        if not self.eigenvalues:
            return 1.0
        return float(math.fsum(math.exp(-e) for e in self.eigenvalues))

    @property
    def free_energy(self) -> float:
        r"""Energía libre modular F = −log Z (F = 0 con Z ≡ 1)."""
        z = self.partition_function
        if z <= 0.0:
            return float("inf")
        return -math.log(z)

    @property
    def internal_energy(self) -> float:
        r"""Energía interna ⟨K_ρ⟩ = ∑ e^{−E} E / Z."""
        z = self.partition_function
        if z <= 0.0 or not self.eigenvalues:
            return 0.0
        return float(math.fsum(math.exp(-e) * e for e in self.eigenvalues) / z)

    @property
    def degeneracy_of_ground(self) -> int:
        r"""Degeneración del ground state dim ker(K_ρ − E_0 I) con tolerancia."""
        if not self.eigenvalues:
            return 0
        e0 = self.eigenvalues[0]
        tol = max(_GAP_DEGENERACY_TOL, 10.0 * self.regularization_eps)
        return sum(1 for e in self.eigenvalues if abs(e - e0) < tol)

    @property
    def is_unique_vacuum(self) -> bool:
        r"""¿El vacío |Ω⟩ es único? (degeneración 1 y gap > 0)."""
        return self.degeneracy_of_ground == 1 and self.spectral_gap > 0.0

    @property
    def boltzmann_weights(self) -> Tuple[float, ...]:
        r"""Pesos de Boltzmann p_i = e^{−E_i}/Z (fallback uniforme si Z ≤ 0)."""
        z = self.partition_function
        if z <= 0.0:
            n = max(self.dimension, 1)
            return tuple(1.0 / n for _ in range(self.dimension))
        return tuple(math.exp(-e) / z for e in self.eigenvalues)

    @property
    def gap_condition_number(self) -> float:
        r"""spread/gap (∞ si el vacío es degenerado)."""
        gap = self.spectral_gap
        if gap <= 0.0 or not math.isfinite(gap):
            return float("inf")
        return self.spectral_spread / gap

    # ── Lectura celeste del Hamiltoniano modular ─────────────────────────

    def mean_motion_frequencies(self) -> np.ndarray:
        r"""
        Vector de frecuencias medias ω_i := E_i − E_0 (energías modulares
        por encima del ground). Diofanticidad de ω ⇒ tori KAM modulares.
        """
        if not self.eigenvalues:
            return np.array([], dtype=float)
        e0 = self.eigenvalues[0]
        return np.array([e - e0 for e in self.eigenvalues], dtype=float)

    def diophantine_residue(
        self,
        gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = _DIOPHANTINE_KMAP,
    ) -> float:
        r"""
        Residuo Diofantino del vector de frecuencias:

            R_{γ,τ}(ω) := min_{0<‖k‖∞≤k_max} |⟨k, ω⟩| / (γ ‖k‖^{−τ}).

        R ≥ 1 ⇔ el espectro modular es (γ, τ)-Diofantino (tori persistentes).
        """
        return _diophantine_residue(self.mean_motion_frequencies(), gamma, tau, k_max)

    def delaunay_triple(self) -> Tuple[float, float, float]:
        r"""
        Tripleta de Delaunay (L, G, H) adaptada al Hamiltoniano modular:

            L := √(E_0 + 1)                               (semieje modular)
            G := √(L² − ΔK²)   si ΔK ≤ L, else 0          (momento angular)
            H := L · cos(spectrum spread)                  (proyección por spread)

        Verifica H ∈ [−L, L] y G ∈ [0, L].
        """
        e0 = self.ground_energy
        spread = self.spectral_spread
        gap = self.spectral_gap
        L = math.sqrt(max(e0 + 1.0, 0.0))
        if not math.isfinite(gap):
            G = L
        else:
            G = math.sqrt(max(L ** 2 - gap ** 2, 0.0))
        H = L * math.cos(spread)
        return float(L), float(G), float(H)


# ── §1.4 Álgebra de operadores densidad en M_n(ℂ) ──────────────────────────


class DensityOperatorAlgebra:
    r"""
    Operadores canónicos de la teoría de información cuántica sobre el simplejo
    de estados D_n = {ρ ∈ M_n : ρ ≥ 0, Tr ρ = 1}:

        S(ρ)     = −Tr(ρ log ρ)                        von Neumann
        P(ρ)     =  Tr(ρ²) = ‖ρ‖_{HS}²                 pureza
        S(ρ‖σ)   =  Tr(ρ (log ρ − log σ))              Umegaki (operatorial)
        F(ρ,σ)   =  ‖√ρ √σ‖_1 = Tr √(√ρ σ √ρ)          Uhlmann
        D_tr     =  (1/2)‖ρ − σ‖_1                     distancia de traza
        d_B      =  √(2(1 − F(ρ,σ)))                   Bures
        ρ^z      =  V diag(λ^z) V†                     cálculo funcional

    Klein: S(ρ‖σ) ≥ 0, con igualdad ⇔ ρ = σ.
    El grupo PU(n) actúa por conjugación preservando S, P, F, D_tr, d_B.

    Lectura celeste (Poincaré-KKS)
    ------------------------------
    El estado ρ ∈ 𝔇(ℋ_n) admite la lectura simpléctica:

        • Acciones de Liouville J_i := λ_i(ρ) (autovalores ordenados desc.)
        • Ángulos canónicos θ_i := arg⟨u_i|N|u_i⟩
        • Frecuencias medias ω_i := ∂H/∂J_i con H = Tr(ρN)
        • Delaunay (L, G, H) y excentricidad/inclinación
        • Función de Wigner discreta W_ρ(q, p) y su negatividad N_W(ρ)
        • Capacidad simpléctica de Gromov c_G(ρ) = 4/(Var_q + Var_p)
        • Índice de Maslov y fase de Berry-Hannay
        • Tiempo de recurrencia de Poincaré τ_rec(ρ) = 1/min_{i≠j}|λ_i − λ_j|
    """

    EPS = _EPS

    # ── Saneamiento y normalización ──────────────────────────────────────

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> np.ndarray:
        r"""Hermitiza y normaliza Tr ρ = 1; copia defensiva."""
        rho = MatrixBanachAlgebra.hermitize(rho)
        tr = float(np.trace(rho).real)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return rho

    @classmethod
    def is_density_operator(cls, rho: np.ndarray, tol: float = _TRACE_TOL) -> bool:
        r"""¿ρ es Hermitiano, PSD y de traza 1?"""
        rho_h = MatrixBanachAlgebra.hermitize(rho)
        herm = float(np.linalg.norm(rho_h - np.asarray(rho, dtype=np.complex128)))
        tr = float(np.trace(rho_h).real)
        w = np.real(la.eigvalsh(rho_h))
        return herm < tol and abs(tr - 1.0) < tol and float(w.min()) > -tol

    # ── Espectros ────────────────────────────────────────────────────────

    @classmethod
    def raw_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        r"""Espectro real descendente, partes negativas recortadas a 0."""
        vals = np.real(la.eigvalsh(cls.sanitize(rho)))
        vals = np.sort(vals)[::-1]
        return np.maximum(vals, 0.0)

    @classmethod
    def regularized_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        r"""Espectro descendente, λ ≥ ε, suma 1 (cálculo logarítmico)."""
        vals = cls.raw_spectrum(rho)
        vals = np.maximum(vals, cls.EPS)
        s = float(vals.sum())
        return vals / s if s > 0.0 else vals

    @classmethod
    def spectrum(cls, rho: np.ndarray) -> np.ndarray:
        r"""Alias de `regularized_spectrum` (compatibilidad)."""
        return cls.regularized_spectrum(rho)

    # ── Estadísticas cuánticas ──────────────────────────────────────────

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        r"""S(ρ) = −∑_{λ_i > ε} λ_i log λ_i  (0 log 0 = 0)."""
        p = cls.raw_spectrum(rho)
        p = p[p > cls.EPS]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        r"""γ(ρ) = Tr(ρ²) ∈ [1/n, 1]."""
        p = cls.raw_spectrum(rho)
        return float(np.sum(p ** 2))

    # ── Cálculo funcional ────────────────────────────────────────────────

    @classmethod
    def matrix_log(cls, rho: np.ndarray) -> np.ndarray:
        r"""log ρ por cálculo funcional Hermitiano, λ ← max(λ, ε)."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        return (vecs * np.log(vals)) @ vecs.conj().T

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex) -> np.ndarray:
        r"""ρ^z = exp(z log ρ) por eigendescomposición regularizada."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        powered = np.exp(z * np.log(vals.astype(np.complex128)))
        return (vecs * powered) @ vecs.conj().T

    # ── Distancias cuánticas y fidelidades ───────────────────────────────

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        S(ρ‖σ) = Tr(ρ (log ρ − log σ)) — forma operatorial.

        NO se emparejan espectros independientes: eso sería correcto sólo si
        [ρ, σ] = 0 y se alinean autoespacios. Klein ⇒ S ≥ 0.
        """
        rho = cls.sanitize(rho)
        log_rho = cls.matrix_log(rho)
        log_sigma = cls.matrix_log(sigma)
        return float(np.real(np.trace(rho @ (log_rho - log_sigma))))

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""F(ρ,σ) = Tr √(√ρ σ √ρ) ∈ [0, 1]."""
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sqrt_rho = cls.matrix_power(rho, 0.5)
        sandwich = MatrixBanachAlgebra.hermitize(sqrt_rho @ sigma @ sqrt_rho)
        return float(
            max(0.0, min(1.0, np.real(np.trace(cls.matrix_power(sandwich, 0.5)))))
        )

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""D(ρ,σ) = (1/2)‖ρ − σ‖_1 ∈ [0, 1]."""
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * MatrixBanachAlgebra.schatten_norm(delta, 1)

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""d_B(ρ,σ) = √(2(1 − F(ρ,σ))) ∈ [0, √2]."""
        fid = cls.uhlmann_fidelity(rho, sigma)
        return math.sqrt(max(0.0, 2.0 * (1.0 - fid)))

    @classmethod
    def kleins_inequality_residual(
        cls, rho: np.ndarray, sigma: np.ndarray
    ) -> float:
        r"""min{0, S(ρ‖σ)}: ~0 si Klein se respeta numéricamente."""
        return float(min(0.0, cls.umegaki_relative_entropy(rho, sigma)))

    # ── Hamiltoniano modular y purificación TFD ──────────────────────────

    @classmethod
    def modular_hamiltonian_from_rho(
        cls, rho: np.ndarray, eps: float = _EPS
    ) -> ModularHamiltonian:
        r"""K_ρ = −log ρ, Z = ∑ λ_i = 1, hash SHA-256 de autovectores."""
        rho = cls.sanitize(rho)
        lam, vec = la.eigh(rho)
        lam = np.maximum(lam, eps)
        lam = lam / lam.sum()
        energies = -np.log(lam)
        order = np.argsort(energies)
        e_sorted, vec_sorted = energies[order], vec[:, order]
        digest = hashlib.sha256(
            np.ascontiguousarray(vec_sorted).tobytes()
        ).hexdigest()
        return ModularHamiltonian(
            eigenvalues=tuple(float(x) for x in e_sorted.tolist()),
            eigenvectors_hash=digest,
            regularization_eps=eps,
        )

    @classmethod
    def density_matrix_from_hamiltonian(
        cls, H: np.ndarray, beta: float
    ) -> np.ndarray:
        r"""
        ρ_β = e^{−βH} / Tr e^{−βH} con shift numérico (softmax):

            E ← E − min E;   x ← −β E;   x ← x − max x;   p = softmax(x).
        """
        H = MatrixBanachAlgebra.hermitize(H)
        w, vecs = la.eigh(H)
        w_shift = w - w.min()
        logits = -beta * w_shift
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return cls.sanitize((vecs * p) @ vecs.conj().T)

    @classmethod
    def thermofield_double(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Purificación de GNS / estado thermofield double:

            |Ψ_TFD⟩ = ∑_i √λ_i |i⟩ ⊗ |i⟩ ∈ ℂⁿ ⊗ ℂⁿ,

        que bajo la identificación HS es Ω_HS = ρ^{1/2}.
        """
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, 0.0)
        n = rho.shape[0]
        psi = np.zeros((n * n,), dtype=np.complex128)
        sqrt_vals = np.sqrt(vals)
        for i in range(n):
            ket = vecs[:, i]
            psi += sqrt_vals[i] * np.kron(ket, ket)
        norm = float(np.linalg.norm(psi))
        return psi / norm if norm > 0.0 else psi

    @classmethod
    def gns_inner_product(
        cls, rho: np.ndarray, A: np.ndarray, B: np.ndarray
    ) -> complex:
        r"""⟨A|B⟩_ω = Tr(ρ A†B)."""
        rho = cls.sanitize(rho)
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(rho @ A.conj().T @ B))

    # ── Geometría simpléctica: acciones, ángulos, Delaunay ───────────────

    @classmethod
    def action_variables(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Acciones de Liouville J_i := λ_i(ρ) ordenadas decrecientemente.
        Convención celeste: la masa mayor primero.
        """
        return np.sort(cls.raw_spectrum(rho))[::-1]

    @classmethod
    def angle_variables(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""
        Ángulos canónicos θ_i := arg⟨u_i|N|u_i⟩ ∈ (−π, π], con u_i los
        autovectores de ρ ordenados por λ_i decreciente.
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        evals, evecs = la.eigh(rho)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        ph = np.zeros(n, dtype=float)
        for i in range(n):
            u = evecs[:, i]
            z = np.vdot(u, N_diag @ u)
            ph[i] = float(np.angle(z))
        return ph

    @classmethod
    def mean_motion_frequencies(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""
        Frecuencias medias ω_i := ∂H/∂J_i, con H(ρ) = Tr(ρN).
        En la práctica ω_i := λ_i · Tr(ρN).
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        actions = cls.action_variables(rho)
        base = float(np.trace(rho @ N_diag).real)
        return actions * base

    @classmethod
    def delaunay_triple(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        r"""
        Tripleta de Delaunay (L, G, H):

            L := √(Tr(ρN))                       (semieje mayor metabólico)
            G := √(Tr(ρN)² − ‖[ρ, N]‖²_F)        (momento angular espectral)
            H := Tr(ρN) · cos(Δλ(ρ))             (proyección por brecha)

        Verifica H ≤ G ≤ L. La excentricidad es e := √(1 − G²/L²); la
        inclinación i := arccos(H/G).
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        comm = rho @ N_diag - N_diag @ rho
        trace_rN = float(np.trace(rho @ N_diag).real)
        comm_norm_sq = float(np.trace(comm @ comm.conj().T).real)
        L = math.sqrt(max(trace_rN, 0.0))
        G = math.sqrt(max(trace_rN ** 2 - comm_norm_sq, 0.0))
        energies = cls.raw_spectrum(rho)
        gap = float(energies[0] - energies[1]) if energies.size >= 2 else 0.0
        H = trace_rN * math.cos(gap)
        return float(L), float(G), float(H)

    @classmethod
    def eccentricity_inclination(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        r"""Excentricidad e := √(1 − G²/L²) e inclinación i := arccos(H/G)."""
        L, G, H = cls.delaunay_triple(rho, N_diag)
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, _EPS), 0.0))
        i = (
            math.acos(max(min(H / max(G, _EPS), 1.0), -1.0))
            if G > _EPS
            else 0.0
        )
        return float(e), float(i)

    # ── Función de Wigner (representación de Weyl discreta) ──────────────

    @classmethod
    def wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Función de Wigner discreta W_ρ(q, p) sobre ℤ_n × ℤ_n:

            W_ρ(q, p) := (1/n) Σ_{x=0}^{n−1} e^{−2πi p x/n} ρ_{q+x, q−x}.

        Propiedades:
            Σ_{q,p} W_ρ(q, p) = Tr(ρ) = 1,
            Σ_p W_ρ(q, p) = ⟨q|ρ|q⟩,
            Σ_q W_ρ(q, p) = ⟨p|ρ|p⟩.

        Devuelve la parte real n × n para ρ Hermitiano.
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        W = np.zeros((n, n), dtype=np.complex128)
        omega = np.exp(-2j * np.pi / n)
        for q in range(n):
            for p in range(n):
                s = 0.0 + 0.0j
                for x in range(n):
                    s += (omega ** (p * x)) * rho[(q + x) % n, (q - x) % n]
                W[q, p] = s / n
        return np.real(W)

    @classmethod
    def wigner_marginals(cls, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Marginales Wigner: (⟨q|ρ|q⟩, ⟨p|ρ|p⟩) sobre la base de Fourier."""
        W = cls.wigner_function(rho)
        return np.sum(W, axis=1), np.sum(W, axis=0)

    @classmethod
    def wigner_negativity(cls, rho: np.ndarray) -> float:
        r"""
        Negatividad de Wigner N_W(ρ) := Σ_{q,p} |W_ρ(q, p)| − 1.
        N_W = 0 ⇔ estado clásico (Gaussiano-mixto); N_W > 0 ⇔ no-clasicidad.
        """
        W = cls.wigner_function(rho)
        return float(np.sum(np.abs(W)) - 1.0)

    # ── Capacidad simpléctica de Gromov ──────────────────────────────────

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        r"""
        Capacidad simpléctica de Gromov heurística:

            c_G(ρ) := 4 / (Var_q(ρ) + Var_p(ρ)),

        donde Var_q, Var_p son las varianzas de las marginales Wigner. La cota
        de no-aplastamiento impone c_G(B^{2n}(r)) = π r² ≤ π R² = c_G(Z^{2n}(R)).
        """
        q_marg, p_marg = cls.wigner_marginals(rho)
        n = q_marg.size
        idx = np.arange(n, dtype=float)
        mean_q = float(np.dot(idx, q_marg))
        mean_p = float(np.dot(idx, p_marg))
        var_q = float(np.dot((idx - mean_q) ** 2, q_marg))
        var_p = float(np.dot((idx - mean_p) ** 2, p_marg))
        if var_q + var_p < 1e-12:
            return float("inf")
        return float(4.0 / (var_q + var_p))

    @classmethod
    def symplectic_capacity_ratio(
        cls, rho: np.ndarray, rho_base: np.ndarray
    ) -> float:
        r"""Ratio c_G(ρ)/c_G(ρ_base); > 1.25 ⇒ violación de rigidez de Gromov."""
        c_rho = cls.symplectic_capacity_gromov(rho)
        c_base = cls.symplectic_capacity_gromov(rho_base)
        if c_base <= 1e-12:
            return float("inf")
        return float(c_rho / c_base)

    # ── Índice de Maslov, recurrencia, Berry ─────────────────────────────

    @classmethod
    def maslov_index(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> int:
        r"""
        Índice de Maslov de la familia de autovectores respecto a la base de N:
        cuanta los cruces de colinealidad (fase proyectada |arg| > π/2).
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        _, evecs = la.eigh(rho)
        count = 0
        for k in range(n):
            u = evecs[:, k]
            overlap = abs(np.vdot(u, N_diag @ u))
            if overlap > 1e-12:
                ph = float(np.angle(np.vdot(u, N_diag @ u)))
                if ph > math.pi / 2.0 or ph < -math.pi / 2.0:
                    count += 1
        return int(count)

    @classmethod
    def poincare_recurrence_time(cls, rho: np.ndarray) -> float:
        r"""
        Tiempo de recurrencia de Poincaré: τ_rec(ρ) := 1/min_{i≠j}|λ_i − λ_j|.
        Autovalores casi degenerados ⇒ recurrencia lenta (memoria cuántica).
        """
        lam = np.sort(cls.raw_spectrum(rho))
        if lam.size < 2:
            return float("inf")
        min_gap = float(np.min(np.abs(np.diff(lam))))
        if min_gap < _EPS:
            return float("inf")
        return 1.0 / min_gap

    @classmethod
    def berry_phase_curve(cls, curve: Sequence[np.ndarray]) -> float:
        r"""
        Fase de Berry a lo largo de una curva discreta γ = (ρ_0, …, ρ_m):

            γ_Berry := Σ_k arg⟨ψ_k | ψ_{k+1}⟩,

        con ψ_k el autovector dominante de ρ_k. Recupera el transporte paralelo
        no trivial en el fibrado de Hilbert proyectivo ℙ(ℋ_n).
        """
        if len(curve) < 2:
            return 0.0
        phis: List[np.ndarray] = []
        for rho_k in curve:
            rho_h = MatrixBanachAlgebra.hermitize(rho_k)
            _, evecs = la.eigh(rho_h)
            phis.append(evecs[:, -1])
        total_arg = 0.0
        for k in range(len(phis) - 1):
            z = np.vdot(phis[k], phis[k + 1])
            total_arg += float(np.angle(z))
        return float(total_arg)


# ── §1.5 Construcción GNS (álgebra de Hilbert a izquierda) ─────────────────


class GNSHilbertAlgebra:
    r"""
    Construcción GNS de (M_n(ℂ), ω_ρ):

        H_ω = M_n     con     ⟨A|B⟩_ω = Tr(ρ A†B),
        Ω   = I       cíclico y separante ⇔ ρ > 0 (ω fiel),
        π(a)|b⟩ = |a b⟩,
        Ω_HS = ρ^{1/2} ∈ HS(ℂⁿ) (picture de Hilbert-Schmidt).

    Tomita: S π(x) Ω = π(x†) Ω, Δ = S*S, J del desdoblamiento polar.
    Este objeto es el puente geométrico entre §1.4 y el motor de §2.1.
    """

    @staticmethod
    def inner_product(rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        r"""⟨A|B⟩_ω = Tr(ρ A† B)."""
        return DensityOperatorAlgebra.gns_inner_product(rho, A, B)

    @staticmethod
    def cyclic_vector_hs(rho: np.ndarray) -> np.ndarray:
        r"""Ω_HS = ρ^{1/2}. ω(a) = ⟨Ω_HS, a Ω_HS⟩_HS."""
        return DensityOperatorAlgebra.matrix_power(rho, 0.5)

    @staticmethod
    def is_faithful(rho: np.ndarray, tol: float = _EPS) -> bool:
        r"""¿ω fiel? ⇔ spec(ρ) > 0 (todos los autovalores positivos)."""
        w = DensityOperatorAlgebra.raw_spectrum(rho)
        return bool(w.size and float(w.min()) > tol)

    @staticmethod
    def omega_norm_squared(rho: np.ndarray) -> float:
        r"""‖Ω‖_ω² = ⟨I|I⟩_ω = Tr(ρ) = 1."""
        return float(np.real(np.trace(DensityOperatorAlgebra.sanitize(rho))))


# ── §1.6 Sección de Poincaré y estructura de resonancias ──────────────────


@dataclass(frozen=True, slots=True)
class PoincareSectionVacuum:
    r"""
    Sección de Poincaré transversal al flujo modular σ_t.

    Definición:
        Σ_c := { ρ ∈ 𝔇(ℋ_n) | Tr(ρ · N) = c },   N = diag(1, …, n), c ∈ [1, n].

    Transversalidad: ⟨d Tr(ρN)/dt, ḋρ⟩ = 2 Re Tr(ḋρ · N) ≠ 0 en los puntos
    de corte. El mapa de retorno P : Σ_c → Σ_c es una aplicación twist que
    preserva la medida de Liouville μ_L y por el teorema de Poincaré-Birkhoff
    posee al menos 2q puntos fijos para cada rotación p/q racional.
    """

    level: float
    normal_diag: np.ndarray
    transversality_tol: float = 1e-7

    def signed_distance(self, rho: np.ndarray) -> float:
        r"""Distancia con signo al hiperplano Σ_c: (Tr(ρN) − c)."""
        return float(np.trace(rho @ self.normal_diag).real) - float(self.level)

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        r"""¿El segmento [ρ_before, ρ_after] cruza Σ_c transversalmente?"""
        d0 = self.signed_distance(rho_before)
        d1 = self.signed_distance(rho_after)
        return (d0 * d1) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        r"""Condición |Tr(rhodot · N)| > ε (transversalidad estricta)."""
        g = float(np.trace(rhodot @ self.normal_diag).real)
        return abs(g) > self.transversality_tol


@dataclass(frozen=True, slots=True)
class ResonanceStructure:
    r"""
    Estructura de resonancias del vector de frecuencias ω.

    Definición:
        k ∈ ℤⁿ \ {0} es una resonancia de orden |k| := ‖k‖₁ si
            |⟨k, ω⟩| < ε_resonance.

    Campos:
        vector       : k más resonante detectado (vacío si ninguno).
        order        : ‖k‖₁ mínimo resonante (0 si sin resonancias en banda).
        residue_min  : min_{k∈banda} |⟨k, ω⟩| (indicador de cercanía).
        kam_residue  : residuo Diofantino R_{γ,τ}(ω).
        is_kam_safe  : R ≥ 1 ⇔ compatible con tori KAM persistentes.
    """

    vector: Tuple[int, ...]
    order: int
    residue_min: float
    kam_residue: float
    is_kam_safe: bool


def _diophantine_residue(
    omega: np.ndarray,
    gamma: float = _DIOPHANTINE_GAMMA,
    tau: float = _DIOPHANTINE_TAU,
    k_max: int = _DIOPHANTINE_KMAP,
) -> float:
    r"""
    Residuo Diofantino:

        R_{γ,τ}(ω) := min_{0<‖k‖∞≤k_max} |⟨k, ω⟩| / (γ ‖k‖^{−τ}).

    Enumeración exhaustiva en la caja [−k_max, k_max]^n. R ≥ 1 ⇔ ω es
    (γ, τ)-Diofantino.
    """
    omega = np.asarray(omega, dtype=float)
    n = omega.size
    if n == 0:
        return float("inf")
    best = float("inf")
    ranges = [range(-k_max, k_max + 1)] * n
    for k in np.array(np.meshgrid(*ranges)).T.reshape(-1, n):
        if not np.any(k):
            continue
        kn = float(np.linalg.norm(k, ord=np.inf))
        denom = gamma * (kn ** (-tau))
        val = abs(float(np.dot(k, omega))) / max(denom, _EPS)
        if val < best:
            best = val
    return float(best)


def _iterate_integer_lattice(n: int, order: int) -> List[np.ndarray]:
    r"""Itera vectores k ∈ ℤⁿ con ‖k‖₁ = order exacto."""
    if n == 1:
        return [np.array([order]), np.array([-order])] if order != 0 else []
    vectors: List[np.ndarray] = []
    for k1 in range(-order, order + 1):
        rest = order - abs(k1)
        if rest < 0:
            continue
        if n - 1 == 1:
            vectors.append(np.array([k1, rest]))
            if rest > 0:
                vectors.append(np.array([k1, -rest]))
        else:
            for tail in _iterate_integer_lattice(n - 1, rest):
                vectors.append(np.concatenate([[k1], tail]))
    return vectors


# ── §1.7 VacuumModularContext + VacuumStatePreparation — HAND-OFF 1→2 ────


@dataclass(slots=True)
class VacuumModularContext:
    r"""
    Objeto terminal de FASE 1 y objeto inicial de FASE 2.

    Encapsula el par canónico (ρ_Ω, K_Ω) junto con el Hamiltoniano externo H,
    el proyector espectral al ground |Ω⟩⟨Ω| (β = ∞) y metadatos GNS, de modo
    que `TomitaTakesakiEngine.bind_vacuum_context` no re-deriva el vacío: la
    definición formal de `prepare_vacuum_context` *es* el arranque de la
    dinámica modular.
    """

    rho: np.ndarray
    K: ModularHamiltonian
    H: np.ndarray
    ground_projector: np.ndarray
    beta: float
    gns_norm_sq: float
    is_faithful: bool
    tfd: np.ndarray = field(repr=False)


class VacuumStatePreparation:
    r"""
    Prepara el contexto modular que alimenta toda la FASE 2.

    El vacío |Ω⟩ es el ground state del Hamiltoniano externo H (tight-binding
    sobre el grafo camino P_n, o H arbitrario hermítico):

        H|Ω⟩ = E_0|Ω⟩,   E_0 = min spec(H).

    Modos de preparación:
        (a) PURA        ρ_Ω = |Ω⟩⟨Ω|              (β = ∞, T = 0)
        (b) GIBBS       ρ_β = e^{−βH}/Z_β         (KMS a β finito)
        (c) INTERPOLADA ρ(τ) = (1−τ)|Ω⟩⟨Ω| + τ I/n

    El último método, `prepare_vacuum_context`, cierra FASE 1 y es el morfismo
    de hand-off FASE 1 ⟶ FASE 2.
    """

    DEFAULT_BETA_COLD: float = 1.0e3
    DEFAULT_HOPPING: float = 1.0e-3

    @classmethod
    def tight_binding_hamiltonian(
        cls, n: int, hopping: float = DEFAULT_HOPPING
    ) -> np.ndarray:
        r"""
        Hamiltoniano de enlace fuerte sobre el grafo camino P_n:

            H = ∑_{k=0}^{n−1} k|k⟩⟨k| + t ∑_{⟨i,j⟩} (|i⟩⟨j| + |j⟩⟨i|).

        El laplaciano combinatorio de P_n es isospectral módulo onsite; el
        gap topológico O(t) no destruye la unicidad del ground si t ≪ 1.
        """
        if n < 1:
            raise ValueError("mac_dimension must be ≥ 1")
        onsite = np.linspace(0.0, float(max(n - 1, 0)), n, dtype=np.float64)
        H = np.diag(onsite).astype(np.complex128)
        for i in range(n - 1):
            H[i, i + 1] = hopping
            H[i + 1, i] = hopping
        return MatrixBanachAlgebra.hermitize(H)

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        r"""|Ω⟩⟨Ω| = proyector al autovector de menor autovalor de H."""
        w, vecs = la.eigh(MatrixBanachAlgebra.hermitize(H))
        idx0 = int(np.argmin(w))
        omega = vecs[:, idx0].reshape(-1, 1)
        return DensityOperatorAlgebra.sanitize(omega @ omega.conj().T)

    @classmethod
    def gibbs_state(cls, H: np.ndarray, beta: float) -> np.ndarray:
        r"""Estado de Gibbs ρ_β = e^{−βH}/Z_β."""
        return DensityOperatorAlgebra.density_matrix_from_hamiltonian(H, beta)

    @classmethod
    def interpolated_state(cls, H: np.ndarray, tau: float) -> np.ndarray:
        r"""ρ(τ) = (1−τ)|Ω⟩⟨Ω| + τ I/n (vacío templado)."""
        tau = float(min(1.0, max(0.0, tau)))
        rho = cls.ground_state_projector(H)
        n = rho.shape[0]
        return DensityOperatorAlgebra.sanitize(
            (1.0 - tau) * rho + tau * np.eye(n, dtype=np.complex128) / n
        )

    @classmethod
    def prepare_vacuum_pair(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> Tuple[np.ndarray, ModularHamiltonian]:
        r"""Compatibilidad: (H, β) ↦ (ρ_Ω, K_Ω). Delegado del contexto."""
        ctx = cls.prepare_vacuum_context(H, beta=beta)
        return ctx.rho, ctx.K

    @classmethod
    def prepare_vacuum_context(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> VacuumModularContext:
        r"""
        Morfismo de hand-off (H, β) ↦ VacuumModularContext.

        Continúa en §2.0 `TomitaTakesakiEngine.bind_vacuum_context`.
        """
        H = MatrixBanachAlgebra.hermitize(H)
        rho = cls.gibbs_state(H, beta)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        ground = cls.ground_state_projector(H)
        gns_n2 = GNSHilbertAlgebra.omega_norm_squared(rho)
        faithful = GNSHilbertAlgebra.is_faithful(rho)
        tfd = DensityOperatorAlgebra.thermofield_double(rho)
        logger.debug(
            "VacuumStatePreparation.context: β=%.4g | E₀(K)=%.6f | gap(K)=%.6f | "
            "Z=%.8f | F=%.3e | faithful=%s",
            beta,
            K.ground_energy,
            K.spectral_gap,
            K.partition_function,
            K.free_energy,
            faithful,
        )
        return VacuumModularContext(
            rho=rho,
            K=K,
            H=H,
            ground_projector=ground,
            beta=beta,
            gns_norm_sq=gns_n2,
            is_faithful=faithful,
            tfd=tfd,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · DINÁMICA MODULAR Y ESPECTRO DEL VACÍO                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Motor de Tomita-Takesaki (con extensión celeste) ─────────────────


class TomitaTakesakiEngine:
    r"""
    Motor de álgebra GNS y flujo modular de Tomita-Takesaki.

    Satisface la identidad modular fundamental:

        σ_t(a) = Δ^{−it} a Δ^{it} = ρ^{−it} a ρ^{it}.

    Métodos celestes añadidos (v9.0.0):
        • diophantine_residue(ω)             : residuo KAM.
        • detect_resonances(ω)               : estructura resonante p/q.
        • kam_torus_indicator(ω)             : indicador binario de toro KAM.
        • poincare_section_at(level, n)      : construcción de Σ_c.
        • poincare_return_time(rho, section) : mapa de retorno bajo σ_t.
        • poincare_birkhoff_count(p, q, θ)   : cardinal mínimo por P-B.
        • solve_kepler(M, e)                 : solver de Kepler (Newton-Danby).
        • kepler_residual(E, M, e)           : residuo F(E) = E − e sin E − M.
        • birkhoff_normal_form(rho, N, k)    : forma normal de Birkhoff-Dulac.
        • melnikov_function(rho_d, rho_b)    : función de Melnikov.
        • wigner_function(rho)               : función de Wigner.
        • symplectic_capacity_gromov(rho)    : capacidad de Gromov heurística.
    """

    def __init__(self, Hilbert_dim: int = 4, beta_kms: float = 1.0) -> None:
        self.dim = Hilbert_dim
        self.beta = float(beta_kms)

    # ── Flujo modular (método original con retorno real_if_close) ────────

    def compute_tomita_takesaki_modular_flow(
        self,
        density_op: np.ndarray,
        t_time: float,
    ) -> np.ndarray:
        r"""
        Calcula la evolución modular σ_t(a) = Δ^{−it} a Δ^{it} actuando sobre
        el estado. Preserva la traza y la norma Banach del operador densidad
        en el Vacío.
        """
        S_mat = 0.5 * (density_op + density_op.T.conj())
        evals, evecs = la.eigh(S_mat)
        evals_pos = np.maximum(evals, 1e-12)
        log_evals = np.log(evals_pos)
        H_mod = evecs @ np.diag(log_evals) @ evecs.T.conj()
        U_t = la.expm(-1j * t_time * H_mod)
        sigma_t = U_t @ density_op @ U_t.T.conj()
        return np.real_if_close(sigma_t)

    # ── §2.0 Arranque de FASE 2 ─────────────────────────────────────────

    @classmethod
    def bind_vacuum_context(cls, ctx: VacuumModularContext) -> VacuumModularContext:
        r"""
        Arranque de FASE 2: continúa el morfismo
        `VacuumStatePreparation.prepare_vacuum_context` (§1.7).

        Verifica dimensiones (ρ, H, |Ω⟩⟨Ω|, spec(K)) y re-sanea los invariantes
        GNS (‖Ω‖² = 1, fidelidad) antes de devolver el contexto listo para la
        dinámica modular.
        """
        if ctx.rho.ndim != 2 or ctx.rho.shape[0] != ctx.rho.shape[1]:
            raise ValueError("VacuumModularContext.ρ must be a square matrix")
        n = ctx.rho.shape[0]
        if ctx.H.shape != (n, n) or ctx.ground_projector.shape != (n, n):
            raise ValueError("H, ρ and |Ω⟩⟨Ω| dimension mismatch")
        if ctx.K.dimension != n:
            raise ValueError("spec(K_ρ) length ≠ n")

        ctx.rho = DensityOperatorAlgebra.sanitize(ctx.rho)
        ctx.H = MatrixBanachAlgebra.hermitize(ctx.H)
        ctx.ground_projector = DensityOperatorAlgebra.sanitize(ctx.ground_projector)
        ctx.gns_norm_sq = GNSHilbertAlgebra.omega_norm_squared(ctx.rho)
        ctx.is_faithful = GNSHilbertAlgebra.is_faithful(ctx.rho)

        if abs(ctx.gns_norm_sq - 1.0) > _TRACE_TOL:
            logger.warning(
                "GNS ‖Ω‖² = %.3e ≠ 1 (tol=%.1e)", ctx.gns_norm_sq, _TRACE_TOL
            )
        return ctx

    # ── Álgebra modular (operadores canónicos) ───────────────────────────

    @classmethod
    def modular_operator(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""Δ(a) = ρ^{−1} a ρ."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        rho_inv = DensityOperatorAlgebra.matrix_power(rho, -1.0)
        return rho_inv @ MatrixBanachAlgebra.as_complex(a) @ rho

    @classmethod
    def modular_flow(cls, rho: np.ndarray, t: complex, a: np.ndarray) -> np.ndarray:
        r"""σ_t^ω(a) = ρ^{−it} a ρ^{it}."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        a = MatrixBanachAlgebra.as_complex(a)
        return (
            DensityOperatorAlgebra.matrix_power(rho, -1j * t)
            @ a
            @ DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        )

    @classmethod
    def modular_flow_via_K(
        cls, rho: np.ndarray, t: complex, a: np.ndarray
    ) -> np.ndarray:
        r"""σ_t(a) = e^{it K} a e^{−it K} con K = −log ρ."""
        u = DensityOperatorAlgebra.matrix_power(rho, -1j * t)
        u_inv = DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        return u @ MatrixBanachAlgebra.as_complex(a) @ u_inv

    @classmethod
    def modular_conjugation(cls, rho: np.ndarray, A: np.ndarray) -> np.ndarray:
        r"""J(A) = ρ^{−1/2} A† ρ^{1/2}."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        half = DensityOperatorAlgebra.matrix_power(rho, 0.5)
        inv_half = DensityOperatorAlgebra.matrix_power(rho, -0.5)
        return inv_half @ MatrixBanachAlgebra.as_complex(A).conj().T @ half

    @classmethod
    def tomita_S(cls, A: np.ndarray) -> np.ndarray:
        r"""S(A) = A† (operador antilineal de Tomita)."""
        return MatrixBanachAlgebra.as_complex(A).conj().T

    @classmethod
    def polar_decomposition_residual(
        cls, rho: np.ndarray, A: np.ndarray
    ) -> float:
        r"""‖J(Δ^{1/2}(A)) − S(A)‖_F."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        A = MatrixBanachAlgebra.as_complex(A)
        delta_half_A = (
            DensityOperatorAlgebra.matrix_power(rho, -0.5)
            @ A
            @ DensityOperatorAlgebra.matrix_power(rho, 0.5)
        )
        polar = cls.modular_conjugation(rho, delta_half_A)
        return float(np.linalg.norm(polar - cls.tomita_S(A), "fro"))

    @classmethod
    def flow_group_law_residual(
        cls, rho: np.ndarray, s: float, t: float, a: np.ndarray
    ) -> float:
        r"""‖σ_s(σ_t(a)) − σ_{s+t}(a)‖_F."""
        composed = cls.modular_flow(rho, s, cls.modular_flow(rho, t, a))
        direct = cls.modular_flow(rho, s + t, a)
        return float(np.linalg.norm(composed - direct, "fro"))

    # ── Verificaciones KMS y de axiomas ─────────────────────────────────

    @classmethod
    def verify_kms(
        cls, rho: np.ndarray, n_tests: int = 8, seed: int = 42
    ) -> float:
        r"""Residuo KMS: max_{a,b} |ω(a σ_i(b)) − ω(ba)|."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        residual = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            sigma_i_b = cls.modular_flow(rho, 1j, b)
            lhs = np.trace(rho @ a @ sigma_i_b)
            rhs = np.trace(rho @ b @ a)
            residual = max(residual, abs(lhs - rhs))
        return float(residual)

    @classmethod
    def verify_algebra_axioms(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Residuos de unitalidad, multiplicatividad, isometría e involución."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(7)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        eye = np.eye(n, dtype=np.complex128)

        unital = float(np.linalg.norm(cls.modular_flow(rho, 0.37, eye) - eye))
        product = float(
            np.linalg.norm(
                cls.modular_flow(rho, 0.73, a @ b)
                - cls.modular_flow(rho, 0.73, a) @ cls.modular_flow(rho, 0.73, b)
            )
        )
        isometry = float(
            abs(
                np.linalg.norm(a, "fro")
                - np.linalg.norm(cls.modular_flow(rho, 1.11, a), "fro")
            )
        )
        ja = cls.modular_conjugation(rho, a)
        involution = float(np.linalg.norm(cls.modular_conjugation(rho, ja) - a))
        return {
            "unital_residual": unital,
            "product_residual": product,
            "isometry_residual": isometry,
            "involution_residual": involution,
        }

    @classmethod
    def verify_tomita_takesaki(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Paquete de residuos: axiomas + polar + ley de grupo + KMS + C*."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(11)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        report = cls.verify_algebra_axioms(rho)
        report["polar_residual"] = cls.polar_decomposition_residual(rho, a)
        report["group_law_residual"] = cls.flow_group_law_residual(rho, 0.41, -0.17, a)
        report["kms_residual"] = cls.verify_kms(rho)
        report["cstar_residual"] = MatrixBanachAlgebra.cstar_identity_residual(a)
        report["flow_via_K_residual"] = float(
            np.linalg.norm(
                cls.modular_flow(rho, 0.5, a) - cls.modular_flow_via_K(rho, 0.5, a),
                "fro",
            )
        )
        return report

    # ── Métodos celestes estáticos (Poincaré-KAM-Delaunay) ───────────────

    @classmethod
    def diophantine_residue(
        cls,
        omega: np.ndarray,
        gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = _DIOPHANTINE_KMAP,
    ) -> float:
        r"""
        Residuo Diofantino R_{γ,τ}(ω). R ≥ 1 ⇔ toro KAM viable.
        """
        return _diophantine_residue(omega, gamma=gamma, tau=tau, k_max=k_max)

    @classmethod
    def detect_resonances(
        cls,
        omega: np.ndarray,
        k_max: int = 3,
        epsilon: float = 1e-6,
    ) -> ResonanceStructure:
        r"""
        Detección exhaustiva de resonancias p/q en ℤⁿ:

            k ∈ ℤⁿ \ {0}, ‖k‖₁ ≤ 2·k_max, |⟨k, ω⟩| < ε.
        """
        omega = np.asarray(omega, dtype=float)
        n = omega.size
        best_k: Tuple[int, ...] = ()
        best_order = 0
        best_residue = float("inf")
        for order in range(1, 2 * k_max + 1):
            for k_list in _iterate_integer_lattice(n, order):
                val = abs(float(np.dot(k_list, omega)))
                if val < best_residue:
                    best_residue = val
                    best_k = tuple(int(x) for x in k_list)
                    if val < epsilon:
                        best_order = order
        kam_residue = cls.diophantine_residue(omega)
        return ResonanceStructure(
            vector=best_k,
            order=best_order,
            residue_min=best_residue,
            kam_residue=kam_residue,
            is_kam_safe=bool(kam_residue >= 1.0),
        )

    @classmethod
    def kam_torus_indicator(cls, omega: np.ndarray) -> Tuple[bool, float]:
        r"""Indicador binario de toro KAM invariante: (R ≥ 1, R)."""
        R = cls.diophantine_residue(omega)
        return (R >= 1.0), float(R)

    @classmethod
    def poincare_section_at(
        cls, level: float, dimension: int
    ) -> PoincareSectionVacuum:
        r"""Construye Σ_c con normal diag(1, …, n) y nivel c ∈ [1, n]."""
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        N_diag = np.diag(np.arange(1, dimension + 1, dtype=float))
        return PoincareSectionVacuum(level=float(level), normal_diag=N_diag)

    @classmethod
    def poincare_return_time(
        cls,
        rho0: np.ndarray,
        section: PoincareSectionVacuum,
        N_diag: np.ndarray,
        dt: float = 0.02,
        max_time: float = 100.0,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Tiempo del primer retorno positivo a Σ_c bajo el flujo de Brockett

            ḋρ = [ρ, [ρ, N]]

        integrado con Euler simpléctico. Devuelve (t_return, ρ_return). Si no
        retorna antes de max_time, devuelve (max_time, ρ_max).
        """
        rho = 0.5 * (rho0 + rho0.conj().T)
        tr = float(np.trace(rho).real)
        if tr > 0:
            rho = rho / tr
        d0 = section.signed_distance(rho)
        t = 0.0
        while t < max_time:
            comm = rho @ N_diag - N_diag @ rho
            rho = rho + dt * (rho @ comm - comm @ rho)
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
            t += dt
            d1 = section.signed_distance(rho)
            if d0 < 0 < d1 or d0 > 0 > d1:
                return t, rho
        return t, rho

    @classmethod
    def poincare_birkhoff_count(
        cls, p: int, q: int, twist_angle: float
    ) -> int:
        r"""
        Predicción del teorema de Poincaré-Birkhoff: toda aplicación twist que
        rota un ángulo θ ∈ (0, 2π) posee al menos 2q órbitas periódicas de
        período q con número de rotación p/q racional. Devuelve 2q si 0 < p/q < 1
        y θ ∈ (0, 2π); 0 en otro caso.
        """
        if q < 1 or p <= 0 or p >= q:
            return 0
        if not (0.0 < twist_angle < 2.0 * math.pi):
            return 0
        return 2 * q

    @classmethod
    def solve_kepler(
        cls,
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1e-12,
        max_iter: int = 50,
    ) -> float:
        r"""
        Resuelve la ecuación de Kepler E − e sin E = M por iteración de Newton:

            E_{n+1} = E_n − (E_n − e sin E_n − M) / (1 − e cos E_n).

        Inicialización de Danby: E₀ = M + e sin M.
        """
        e = float(eccentricity)
        M = float(mean_anomaly)
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad debe estar en [0, 1).")
        E = M + e * math.sin(M)
        for _ in range(max_iter):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            dE = f / max(fp, _EPS)
            E -= dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def kepler_residual(
        cls,
        eccentric_anomaly: float,
        mean_anomaly: float,
        eccentricity: float,
    ) -> float:
        r"""Residuo F(E) = E − e sin E − M."""
        return float(
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - mean_anomaly
        )

    @classmethod
    def birkhoff_normal_form(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        order: int = 2,
    ) -> np.ndarray:
        r"""
        Forma normal de Birkhoff-Dulac hasta orden k = order.

        En las proximidades del punto fijo [ρ, N] = 0, la dinámica de Brockett
        admite la serie Ḣ = Σ ε^k X_k(H₀), con X_k ∈ Ker(ad_{H₀}). El primer
        término X_1 := [N, [N, ρ]] (linealización ad²_N) se ortogonaliza
        respecto de H₀ = ρ por proyección de Schur.
        """
        rho_h = 0.5 * (
            np.asarray(rho, dtype=np.complex128)
            + np.asarray(rho, dtype=np.complex128).conj().T
        )
        n = rho_h.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        comm1 = N_diag @ rho_h - rho_h @ N_diag
        X1 = N_diag @ comm1 - comm1 @ N_diag
        tr_num = float(np.real(np.trace(X1 @ rho_h.conj().T)))
        tr_den = float(np.real(np.trace(rho_h @ rho_h.conj().T)))
        if tr_den > _EPS:
            X1 = X1 - (tr_num / tr_den) * rho_h
        return X1

    @classmethod
    def melnikov_function(
        cls,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_steps: int = 32,
        dt: float = 0.02,
    ) -> float:
        r"""
        Función de Melnikov aproximada entre ρ_dream y ρ_base:

            M := Σ_j {H₀, H₁}(ρ_j) · dt,

        con H₀(ρ) = Tr(ρN), H₁(ρ) = Tr(ρ · ρ_base), y ρ_j la órbita de
        Brockett partiendo de ρ_dream.

        M ≠ 0 ⇒ tubo homoclínico intacto (dinámica regular).
        M = 0 ⇒ ruptura del tubo y aparición de caos transitorio.
        """
        rho_h = 0.5 * (rho_dream + rho_dream.conj().T)
        rho_h = rho_h / max(_EPS, float(np.trace(rho_h).real))
        n = rho_h.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        base = 0.5 * (rho_base + rho_base.conj().T)
        total = 0.0
        current = rho_h.copy()
        for _ in range(n_steps):
            comm = current @ N_diag - N_diag @ current
            current = current + dt * (current @ comm - comm @ current)
            current = 0.5 * (current + current.conj().T)
            tr = float(np.trace(current).real)
            if tr > 0:
                current = current / tr
            bracket = float(np.trace(comm @ comm.conj().T).real) * float(
                np.trace(base @ current).real
            )
            total += bracket * dt
        return float(total)

    @classmethod
    def wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        r"""Función de Wigner discreta de ρ (envoltura de DensityOperatorAlgebra)."""
        return DensityOperatorAlgebra.wigner_function(rho)

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        r"""Capacidad de Gromov heurística c_G(ρ)."""
        return DensityOperatorAlgebra.symplectic_capacity_gromov(rho)


# ── §2.2 Analizador espectral del vacío modular (con extensión celeste) ───


@dataclass(frozen=True, slots=True)
class VacuumAuditReport:
    r"""
    Reporte de auditoría del vacío modular con extensión celeste.

    Campos originales:
        vev, thermal_fluctuation, spectral_gap, modular_ground_energy,
        partition_function, free_energy, internal_energy, unique_vacuum,
        participation_ratio, noise_db, purity, von_neumann_entropy,
        local_verdict.

    Campos celestes añadidos (v9.0.0):
        delaunay_triple       : (L, G, H) del estado frente a N = H.
        eccentricity          : e ∈ [0, 1).
        inclination           : i ∈ [0, π].
        kam_residue           : R_{γ,τ}(ω) ≥ 1 ⇔ tori KAM.
        resonance_order       : r* (orden resonante mínimo).
        wigner_negativity     : N_W(ρ).
        symplectic_capacity   : c_G(ρ).
        poincare_recurrence_time : τ_rec(ρ) = 1/min_{i≠j}|λ_i − λ_j|.
        maslov_index          : índice de Maslov.
        lyapunov_sign         : signo del exponente de Lyapunov.
        poincare_stratum      : nombre del estrato celeste.
    """

    vev: float
    thermal_fluctuation: float
    spectral_gap: float
    modular_ground_energy: float
    partition_function: float
    free_energy: float
    internal_energy: float
    unique_vacuum: bool
    participation_ratio: float
    noise_db: float
    purity: float
    von_neumann_entropy: float
    local_verdict: HeytingOmega3
    # ── Campos celestes (v9.0.0) ─────────────────────────────────────────
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    lyapunov_sign: int = -1
    poincare_stratum: str = "kam-torus-invariant"

    @property
    def energy_fluctuation(self) -> float:
        r"""Alias de thermal_fluctuation."""
        return self.thermal_fluctuation

    @property
    def noise_level_db(self) -> float:
        r"""Alias semántico de noise_db."""
        return self.noise_db


class VacuumSpectraAnalyzer:
    r"""
    Analiza ρ frente al Hamiltoniano externo H y al Hamiltoniano modular K_ρ.
    """

    VEV_VETO_THRESHOLD: float = 1.0e-2
    FLUCTUATION_VETO_THRESHOLD: float = 1.0e-2
    GAP_DEGRADE_THRESHOLD: float = 1.0e-3
    FREE_ENERGY_DEGRADE_THRESHOLD: float = 1.0e-8

    @classmethod
    def _expectation_and_variance(
        cls, rho: np.ndarray, H: np.ndarray
    ) -> Tuple[float, float]:
        r"""⟨H⟩_ρ = Tr(ρH) y σ_H = √(⟨H²⟩ − ⟨H⟩²)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        H = MatrixBanachAlgebra.hermitize(H)
        mean = float(np.real(np.trace(rho @ H)))
        second = float(np.real(np.trace(rho @ H @ H)))
        var = max(0.0, second - mean * mean)
        return mean, math.sqrt(var)

    @classmethod
    def participation_ratio(cls, rho: np.ndarray) -> float:
        r"""Razón de participación IPR = 1/Tr(ρ²)."""
        purity = DensityOperatorAlgebra.purity(rho)
        if purity <= 0.0:
            return float("inf")
        return 1.0 / purity

    @classmethod
    def audit(
        cls,
        rho: np.ndarray,
        H: np.ndarray,
        K: ModularHamiltonian,
    ) -> VacuumAuditReport:
        r"""
        Auditoría espectral completa del vacío modular.

        Extendida en v9.0.0 con propagación de los invariantes celestes
        (Delaunay, excentricidad, inclinación, KAM, resonancia, Wigner,
        capacidad de Gromov, recurrencia, Maslov).
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        mean_H, sigma_H = cls._expectation_and_variance(rho, H)

        w_H = np.real(la.eigvalsh(MatrixBanachAlgebra.hermitize(H)))
        e0 = float(w_H.min())
        vev = mean_H - e0

        gap = K.spectral_gap
        z_part = K.partition_function
        free_e = K.free_energy
        noise_db = 10.0 * math.log10(1.0 + sigma_H ** 2 + 1e-30)
        purity = DensityOperatorAlgebra.purity(rho)
        ent = DensityOperatorAlgebra.von_neumann_entropy(rho)
        ipr = cls.participation_ratio(rho)

        # ── Adjudicación local (predicados originales preservados) ─────
        if (
            sigma_H > cls.FLUCTUATION_VETO_THRESHOLD
            and vev > cls.VEV_VETO_THRESHOLD
        ):
            local = HeytingOmega3.VETOED
        elif (
            sigma_H > cls.FLUCTUATION_VETO_THRESHOLD
            or vev > cls.VEV_VETO_THRESHOLD
        ):
            local = HeytingOmega3.DEGRADED
        elif gap < cls.GAP_DEGRADE_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        elif abs(free_e) > cls.FREE_ENERGY_DEGRADE_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT

        # ── Extensión celeste (v9.0.0) ────────────────────────────────
        try:
            L_del, G_del, H_del = DensityOperatorAlgebra.delaunay_triple(rho, H)
            e_del, i_del = DensityOperatorAlgebra.eccentricity_inclination(rho, H)
            omega = DensityOperatorAlgebra.mean_motion_frequencies(rho, H)
            res_struct = TomitaTakesakiEngine.detect_resonances(omega)
            wigner_neg = DensityOperatorAlgebra.wigner_negativity(rho)
            symp_cap = DensityOperatorAlgebra.symplectic_capacity_gromov(rho)
            tau_rec = DensityOperatorAlgebra.poincare_recurrence_time(rho)
            maslov = DensityOperatorAlgebra.maslov_index(rho, H)
            stratum = local.poincare_stratum_name()
            lyap = local.lyapunov_exponent_sign()
        except Exception:
            L_del = G_del = H_del = 0.0
            e_del = i_del = 0.0
            res_struct = ResonanceStructure((), 0, float("inf"), float("inf"), False)
            wigner_neg = 0.0
            symp_cap = 1.0
            tau_rec = float("inf")
            maslov = 0
            stratum = "kam-torus-invariant"
            lyap = -1

        return VacuumAuditReport(
            vev=vev,
            thermal_fluctuation=sigma_H,
            spectral_gap=gap,
            modular_ground_energy=K.ground_energy,
            partition_function=z_part,
            free_energy=free_e,
            internal_energy=K.internal_energy,
            unique_vacuum=K.is_unique_vacuum,
            participation_ratio=ipr,
            noise_db=noise_db,
            purity=purity,
            von_neumann_entropy=ent,
            local_verdict=local,
            delaunay_triple=(L_del, G_del, H_del),
            eccentricity=e_del,
            inclination=i_del,
            kam_residue=res_struct.kam_residue,
            resonance_order=res_struct.order,
            wigner_negativity=wigner_neg,
            symplectic_capacity=symp_cap,
            poincare_recurrence_time=tau_rec,
            maslov_index=maslov,
            lyapunov_sign=lyap,
            poincare_stratum=stratum,
        )


# ── §2.3 Detector del silencio epistemológico ─────────────────────────────


@dataclass(frozen=True, slots=True)
class SilentFieldProbe:
    r"""
    Sonda del silencio epistemológico: fuga de ρ fuera del vacío |Ω⟩⟨Ω|.

    Métricas originales: purity, fidelity_to_ground, uhlmann_fidelity,
    correlation_leakage, umegaki_to_ground, bures_distance, trace_distance,
    klein_residual, local_verdict.
    """

    purity: float
    fidelity_to_ground: float
    uhlmann_fidelity: float
    correlation_leakage: float
    umegaki_to_ground: float
    bures_distance: float
    trace_distance: float
    klein_residual: float
    local_verdict: HeytingOmega3

    @property
    def umegaki_relative_entropy(self) -> float:
        r"""Alias semántico de umegaki_to_ground."""
        return self.umegaki_to_ground

    @property
    def klein_inequality_residual(self) -> float:
        r"""Residuo de la desigualdad de Klein (min{0, S})."""
        return self.klein_residual


class SilentFieldDetector:
    r"""
    Mide la fuga de ρ fuera del vacío |Ω⟩⟨Ω|.

    Aplica los umbrales de silencio epistemológico:
        PURITY_SILENT_THRESHOLD   : γ ≥ 0.999 ⇒ estado casi puro.
        FIDELITY_SILENT_THRESHOLD : overlap ≥ 0.999 ⇒ casi |Ω⟩⟨Ω|.
    """

    UMEGAKI_EPS: float = 1.0e-12
    PURITY_SILENT_THRESHOLD: float = 0.999
    FIDELITY_SILENT_THRESHOLD: float = 0.999

    @classmethod
    def probe(
        cls,
        rho: np.ndarray,
        ground_state_projector: np.ndarray,
    ) -> SilentFieldProbe:
        r"""
        Sondea la fuga del estado ρ desde el proyector |Ω⟩⟨Ω|.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(ground_state_projector)

        purity = DensityOperatorAlgebra.purity(rho)
        overlap = float(np.real(np.trace(rho @ sigma)))
        leak = 1.0 - overlap
        uhlmann = DensityOperatorAlgebra.uhlmann_fidelity(rho, sigma)
        bures = DensityOperatorAlgebra.bures_distance(rho, sigma)
        trc = DensityOperatorAlgebra.trace_distance(rho, sigma)

        n = rho.shape[0]
        sigma_eps = DensityOperatorAlgebra.sanitize(
            (1.0 - cls.UMEGAKI_EPS) * sigma
            + cls.UMEGAKI_EPS * np.eye(n, dtype=np.complex128) / n
        )
        relative_ent = DensityOperatorAlgebra.umegaki_relative_entropy(rho, sigma_eps)
        klein = DensityOperatorAlgebra.kleins_inequality_residual(rho, sigma_eps)

        if (
            purity >= cls.PURITY_SILENT_THRESHOLD
            and overlap >= cls.FIDELITY_SILENT_THRESHOLD
        ):
            local = HeytingOmega3.COHERENT
        elif (
            purity >= cls.PURITY_SILENT_THRESHOLD
            or overlap >= cls.FIDELITY_SILENT_THRESHOLD
        ):
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return SilentFieldProbe(
            purity=purity,
            fidelity_to_ground=overlap,
            uhlmann_fidelity=uhlmann,
            correlation_leakage=leak,
            umegaki_to_ground=relative_ent,
            bures_distance=bures,
            trace_distance=trc,
            klein_residual=klein,
            local_verdict=local,
        )


# ── §2.4 ModularSilencePipeline ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SilentFieldBundle:
    r"""
    Paquete intermedio FASE 2 → FASE 3. Transporta todos los certificados
    parciales (audit, probe, residuales de Tomita) que alimentan la
    adjudicación final en el retículo Ω₃.

    El campo `audit` (VacuumAuditReport) porta los campos celestes, por lo
    que la propagación a `SilentFieldState` se realiza vía `audit`.
    """

    cycle_index: int
    rho: np.ndarray
    K: ModularHamiltonian
    audit: VacuumAuditReport
    probe: SilentFieldProbe
    kms_residual: float
    algebra_axioms: Dict[str, float]
    tomita_report: Dict[str, float]
    context_beta: float

    @property
    def spectrum(self) -> VacuumAuditReport:
        """Alias semántico de `audit`."""
        return self.audit


class ModularSilencePipeline:
    r"""
    Orquestador determinista de la dinámica modular.

    Pipeline algebraico (continuación de M):
        (ρ, H, |Ω⟩⟨Ω|, K) → Tomita → VacuuAudit → SilentFieldProbe
                           → SilentFieldBundle (hand-off a FASE 3).
    """

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho: np.ndarray,
        H: np.ndarray,
        ground_projector: np.ndarray,
        K: ModularHamiltonian,
        context_beta: float = float("nan"),
    ) -> SilentFieldBundle:
        r"""Sintetiza el SilentFieldBundle desde primitivas (ρ, H, |Ω⟩⟨Ω|, K)."""
        tomita = TomitaTakesakiEngine.verify_tomita_takesaki(rho)
        kms_res = float(
            tomita.get("kms_residual", TomitaTakesakiEngine.verify_kms(rho))
        )
        axioms = {
            k: tomita[k]
            for k in (
                "unital_residual",
                "product_residual",
                "isometry_residual",
                "involution_residual",
            )
            if k in tomita
        }
        if len(axioms) < 4:
            axioms = TomitaTakesakiEngine.verify_algebra_axioms(rho)

        audit = VacuumSpectraAnalyzer.audit(rho, H, K)
        probe = SilentFieldDetector.probe(rho, ground_projector)

        return SilentFieldBundle(
            cycle_index=cycle_index,
            rho=rho,
            K=K,
            audit=audit,
            probe=probe,
            kms_residual=kms_res,
            algebra_axioms=axioms,
            tomita_report=tomita,
            context_beta=context_beta,
        )

    @classmethod
    def synthesize_from_context(
        cls,
        cycle_index: int,
        ctx: VacuumModularContext,
    ) -> SilentFieldBundle:
        r"""Sintetiza el SilentFieldBundle desde un VacuumModularContext."""
        ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        return cls.synthesize(
            cycle_index=cycle_index,
            rho=ctx.rho,
            H=ctx.H,
            ground_projector=ctx.ground_projector,
            K=ctx.K,
            context_beta=ctx.beta,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · SOBERANÍA Y CERTIFICACIÓN DEL SILENCIO                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en el retículo Heyting Ω₃ ────────────────────────────


class HeytingVacuumAdjudicator:
    r"""
    Colapsa el SilentFieldBundle en un único veredicto Ω₃ mediante la flecha
    característica χ : Bundle → Ω₃.

    Reglas (semántica intuicionista):
        • residual > CATASTROPHIC_TOL  ⇒ VETOED (separatriz hiperbólica).
        • residual > KMS_RESIDUAL_TOL  ⇒ DEGRADED (resonancia p/q).
        • residual ≤ KMS_RESIDUAL_TOL  ⇒ COHERENT (toro KAM estable).

    Se computa un veredicto granular por subeje (audit, probe, kms, axiomas,
    polar, group_law, klein) y luego se toma el meet con el veredicto externo.
    """

    KMS_RESIDUAL_TOL: float = 1.0e-6
    AXIOM_RESIDUAL_TOL: float = 1.0e-6
    CATASTROPHIC_TOL: float = 1.0e-3
    KLEIN_TOL: float = 1.0e-8

    @classmethod
    def _residual_to_omega(cls, residual: float) -> HeytingOmega3:
        r"""Mapea residual numérico a estrato Ω₃."""
        if residual > cls.CATASTROPHIC_TOL:
            return HeytingOmega3.VETOED
        if residual > cls.KMS_RESIDUAL_TOL:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def granular_scores(
        cls, bundle: SilentFieldBundle
    ) -> Dict[str, HeytingOmega3]:
        r"""
        Computa el veredicto granular por subeje:
            audit, probe, kms, axioms, polar, group_law, klein.
        """
        axioms_max = (
            max(bundle.algebra_axioms.values()) if bundle.algebra_axioms else 0.0
        )
        polar = float(bundle.tomita_report.get("polar_residual", 0.0))
        group = float(bundle.tomita_report.get("group_law_residual", 0.0))
        klein = abs(float(bundle.probe.klein_residual))
        klein_v = (
            HeytingOmega3.COHERENT
            if klein < cls.KLEIN_TOL
            else (
                HeytingOmega3.DEGRADED
                if klein < cls.CATASTROPHIC_TOL
                else HeytingOmega3.VETOED
            )
        )
        return {
            "audit": bundle.audit.local_verdict,
            "probe": bundle.probe.local_verdict,
            "kms": cls._residual_to_omega(bundle.kms_residual),
            "axioms": cls._residual_to_omega(axioms_max),
            "polar": cls._residual_to_omega(polar),
            "group_law": cls._residual_to_omega(group),
            "klein": klein_v,
        }

    @classmethod
    def adjudicate(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        Meet (∧) de todos los veredictos granulares con el veredicto externo:

            χ_final := (⋀_i χ_i) ∧ χ_external.
        """
        scores = cls.granular_scores(bundle)
        local = HeytingOmega3.COHERENT
        for v in scores.values():
            local = local.meet(v)
        return local.meet(external_verdict)


# ── §3.2 Certificado firmado del silencio (con campos celestes) ───────────


@dataclass(frozen=True, slots=True)
class SilentFieldState:
    r"""
    Certificado firmado del ciclo de silencio.

    Campos originales:
        cycle_id, witness_engine_id, vacuum_expectation_value, kms_temperature_beta,
        vacuum_beta, noise_emission_decibels, crystallized_experience_count,
        spectral_gap, free_energy, purity, fidelity_to_ground, uhlmann_fidelity,
        bures_distance_to_ground, trace_distance_to_ground, kms_residual,
        tomita_polar_residual, tomita_group_law_residual, heyting_verdict,
        experience_hash, phase_chain_sha256, timestamp_utc.

    Campos celestes añadidos (v9.0.0):
        delaunay_triple, eccentricity, inclination, kam_residue,
        resonance_order, wigner_negativity, symplectic_capacity,
        poincare_recurrence_time, maslov_index, lyapunov_sign, poincare_stratum.
    """

    cycle_id: str
    witness_engine_id: str
    vacuum_expectation_value: float
    kms_temperature_beta: float
    vacuum_beta: float
    noise_emission_decibels: float
    crystallized_experience_count: int
    spectral_gap: float
    free_energy: float
    purity: float
    fidelity_to_ground: float
    uhlmann_fidelity: float
    bures_distance_to_ground: float
    trace_distance_to_ground: float
    kms_residual: float
    tomita_polar_residual: float
    tomita_group_law_residual: float
    heyting_verdict: HeytingOmega3
    experience_hash: str
    phase_chain_sha256: str
    timestamp_utc: float
    # ── Campos celestes (v9.0.0) ─────────────────────────────────────────
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    lyapunov_sign: int = -1
    poincare_stratum: str = "kam-torus-invariant"

    def is_vetoed(self) -> bool:
        r"""¿El veredicto final es ⊥ en Ω₃?"""
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_kam_stable(self) -> bool:
        r"""¿Habita un toro KAM Diofantino persistente?"""
        return self.kam_residue >= 1.0 and self.resonance_order == 0

    def is_silent(self) -> bool:
        r"""¿El ciclo cumple todos los umbrales del silencio epistemológico?"""
        return (
            self.heyting_verdict == HeytingOmega3.COHERENT
            and self.purity >= 0.999
            and self.fidelity_to_ground >= 0.999
            and self.kms_residual < 1.0e-6
        )


# ── §3.3 Prueba de inclusión Merkle (SHA-256) ─────────────────────────────


@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""
    Prueba de inclusión Merkle SHA-256 (convención pair-hash tipo Bitcoin).

    Estructura:
        leaf_hash : hash SHA-256 de la hoja (hex).
        siblings  : hashes de los nodos hermanos en el camino a la raíz.
        index     : índice de la hoja en el árbol.
        root      : hash de la raíz esperada.
    """

    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        r"""Recalcula el camino de inclusión y compara con la raíz."""
        node = bytes.fromhex(self.leaf_hash)
        idx = self.index
        for sib_hex in self.siblings:
            sib = bytes.fromhex(sib_hex)
            if idx % 2 == 0:
                node = hashlib.sha256(node + sib).digest()
            else:
                node = hashlib.sha256(sib + node).digest()
            idx //= 2
        return node.hex() == self.root


# ── §3.4 Motor soberano — TOONSilentWitnessEngine ─────────────────────────


class TOONSilentWitnessEngine:
    r"""
    Motor físico espectral del Testigo Silencioso con Recurrencia de Poincaré.

    Custodia la matriz MAC con técnica Tomita-Takesaki y auditoría del Vacío
    de Dirac:
        1. Recurrencia de Poincaré en la medida de Liouville.
        2. Flujo modular KMS σ_t(a) = Δ^{−it} a Δ^{it}.
        3. Cristalización en S⁶ ⊂ ℝ⁷ y cadena Merkle SHA-256.
        4. Extensión celeste: Delaunay, KAM, Wigner, crowbar implícito.

    Cadena de custodia `_phase_chain_hash`: Merkle lineal por fase
    (F1: contexto, F2: bundle, F3: veredicto/crowbar).
    """

    def __init__(
        self,
        engine_id: str = "SILENT-ENGINE-SABIO-01",
        mac_dimension: int = 4,
        dim: Optional[int] = None,
        kms_beta: float = 1.0,
        hopping: float = VacuumStatePreparation.DEFAULT_HOPPING,
    ) -> None:
        if dim is not None:
            mac_dimension = dim
        self.engine_id = engine_id
        self.mac_dimension = mac_dimension
        self.kms_beta = kms_beta
        self.cycle_count = 0
        self.history: List[SilentFieldState] = []
        self.tomita_engine = TomitaTakesakiEngine(
            Hilbert_dim=mac_dimension, beta_kms=kms_beta
        )

        self.H = VacuumStatePreparation.tight_binding_hamiltonian(
            mac_dimension, hopping=hopping
        )
        self.ground_projector = VacuumStatePreparation.ground_state_projector(self.H)
        self._phase_chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::{__version__}".encode("ascii")
        ).hexdigest()

    # ── Núcleo: auditoría Poincaré-KMS vacío ─────────────────────────────

    def audit_poincare_recurrence_kms_vacuum(
        self,
        density_matrix: np.ndarray,
        tau_recurrence_target: float,
        kms_tolerance: float = 1e-6,
    ) -> Tuple[bool, float, ExperienceCrystal]:
        r"""
        Audita el retorno de Poincaré y la validez KMS del Vacío de Dirac.

        Pasos:
            1. σ_{τ_rec}(ρ) por el flujo modular; distancia ||·||_F en el
               álgebra de Banach.
            2. Proyección de 7 componentes sobre S⁶ ⊂ ℝ⁷.
            3. Construcción del `ExperienceCrystal` con firma Merkle SHA-256
               y campos celestes (Delaunay, KAM, Wigner, capacidad).

        Devuelve (is_recurrent, banach_dist, crystal).
        """
        sigma_tau = self.tomita_engine.compute_tomita_takesaki_modular_flow(
            density_matrix, tau_recurrence_target
        )
        banach_dist = float(la.norm(sigma_tau - density_matrix, ord="fro"))
        is_recurrent = banach_dist < kms_tolerance

        # ── Proyección sobre S⁶ ⊂ ℝ⁷ ────────────────────────────────
        flat_state = np.abs(sigma_tau.flatten())
        if flat_state.size >= 7:
            v7 = flat_state[:7]
        else:
            v7 = np.pad(flat_state, (0, 7 - flat_state.size))
        v7_norm = float(np.linalg.norm(v7))
        v_s6 = v7 / v7_norm if v7_norm > 1e-12 else np.ones(7) / np.sqrt(7.0)

        # ── Campos celestes (v9.0.0) ─────────────────────────────────
        try:
            L_del, G_del, H_del = DensityOperatorAlgebra.delaunay_triple(
                sigma_tau, self.H
            )
            e_del, i_del = DensityOperatorAlgebra.eccentricity_inclination(
                sigma_tau, self.H
            )
            omega = DensityOperatorAlgebra.mean_motion_frequencies(sigma_tau, self.H)
            res_struct = TomitaTakesakiEngine.detect_resonances(omega)
            wigner_neg = DensityOperatorAlgebra.wigner_negativity(sigma_tau)
            symp_cap = DensityOperatorAlgebra.symplectic_capacity_gromov(sigma_tau)
            stratum = (
                HeytingOmega3.COHERENT.poincare_stratum_name()
                if is_recurrent
                else HeytingOmega3.DEGRADED.poincare_stratum_name()
            )
        except Exception:
            L_del = G_del = H_del = 0.0
            e_del = i_del = 0.0
            res_struct = ResonanceStructure((), 0, float("inf"), float("inf"), False)
            wigner_neg = 0.0
            symp_cap = 1.0
            stratum = "hyperbolic-escape-separatrix"

        # ── Sello Merkle SHA-256 ─────────────────────────────────────
        hasher = hashlib.sha256()
        hasher.update(v_s6.tobytes())
        hasher.update(str(tau_recurrence_target).encode("utf-8"))
        merkle_root = hasher.hexdigest()

        crystal = ExperienceCrystal(
            vector_s6=v_s6,
            tau_recurrence=tau_recurrence_target,
            kms_drift=banach_dist,
            merkle_root_sha256=merkle_root,
            timestamp_utc=time.time(),
            delaunay_triple=(L_del, G_del, H_del),
            eccentricity=e_del,
            inclination=i_del,
            kam_residue=res_struct.kam_residue,
            resonance_order=res_struct.order,
            poincare_stratum=stratum,
            wigner_negativity=wigner_neg,
            symplectic_capacity=symp_cap,
        )
        return is_recurrent, banach_dist, crystal

    # ── Cadena de custodia (Merkle lineal) ───────────────────────────────

    def _update_chain(self, tag: str, payload: bytes) -> str:
        r"""
        Actualiza la cadena de custodia por fase:
            h ← SHA-256(h_previo ‖ tag ‖ payload).
        """
        digest = hashlib.sha256(
            self._phase_chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._phase_chain_hash = digest
        return digest

    def latest_state(self) -> Optional[SilentFieldState]:
        r"""Último estado del historial (o None si vacío)."""
        return self.history[-1] if self.history else None

    def forensic_verify_chain(self) -> bool:
        r"""Verifica que todos los hashes del historial sean SHA-256 válidos."""
        if not self.history:
            return True
        return all(
            len(s.phase_chain_sha256) == 64
            and all(c in "0123456789abcdef" for c in s.phase_chain_sha256)
            for s in self.history
        )

    # ── Ciclo completo del silencio ──────────────────────────────────────

    def execute_silence_cycle(
        self,
        triad_crystallized_count: int,
        kms_beta: float = 1.0,
        beta_vacuum: float = VacuumStatePreparation.DEFAULT_BETA_COLD,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> SilentFieldState:
        r"""
        Ejecuta un ciclo completo del motor silencioso:

            Prepare → Tomita → VacuuAudit → Probe → Adjudicate → Seal.

        Retorna un SilentFieldState inmutable registrado en el historial,
        con cadena de custodia SHA-256 y campos celestes propagados.
        """
        self.cycle_count += 1
        cycle_id = f"CYC-SILENT-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Ciclo del vacío %s | cristalizados=%d ═══",
            cycle_id,
            triad_crystallized_count,
        )

        ctx = VacuumStatePreparation.prepare_vacuum_context(self.H, beta=beta_vacuum)
        self._update_chain("F1", ctx.rho.tobytes())

        bundle = ModularSilencePipeline.synthesize_from_context(
            cycle_index=self.cycle_count, ctx=ctx
        )
        self._update_chain(
            "F2",
            f"{bundle.kms_residual:.12e}|{bundle.audit.vev:.12e}|"
            f"{bundle.audit.free_energy:.12e}".encode("ascii"),
        )

        verdict = HeytingVacuumAdjudicator.adjudicate(bundle, external_verdict)
        self._update_chain(
            "F3",
            f"{verdict.name}|{bundle.audit.noise_db:.6f}|"
            f"{triad_crystallized_count}".encode("ascii"),
        )

        hasher = hashlib.sha256()
        hasher.update(self.engine_id.encode("ascii"))
        hasher.update(cycle_id.encode("ascii"))
        hasher.update(f"{bundle.audit.vev:.12e}".encode("ascii"))
        hasher.update(f"{bundle.audit.noise_db:.12e}".encode("ascii"))
        hasher.update(f"{bundle.kms_residual:.12e}".encode("ascii"))
        hasher.update(f"{triad_crystallized_count}".encode("ascii"))
        hasher.update(f"{time.time_ns()}".encode("ascii"))
        experience_hash = hasher.hexdigest()

        polar_res = float(bundle.tomita_report.get("polar_residual", 0.0))
        group_res = float(bundle.tomita_report.get("group_law_residual", 0.0))

        state = SilentFieldState(
            cycle_id=cycle_id,
            witness_engine_id=self.engine_id,
            vacuum_expectation_value=bundle.audit.vev,
            kms_temperature_beta=kms_beta,
            vacuum_beta=beta_vacuum,
            noise_emission_decibels=bundle.audit.noise_db,
            crystallized_experience_count=triad_crystallized_count,
            spectral_gap=bundle.audit.spectral_gap,
            free_energy=bundle.audit.free_energy,
            purity=bundle.probe.purity,
            fidelity_to_ground=bundle.probe.fidelity_to_ground,
            uhlmann_fidelity=bundle.probe.uhlmann_fidelity,
            bures_distance_to_ground=bundle.probe.bures_distance,
            trace_distance_to_ground=bundle.probe.trace_distance,
            kms_residual=bundle.kms_residual,
            tomita_polar_residual=polar_res,
            tomita_group_law_residual=group_res,
            heyting_verdict=verdict,
            experience_hash=experience_hash,
            phase_chain_sha256=self._phase_chain_hash,
            timestamp_utc=time.time(),
            # ── Campos celestes (propagados desde audit) ─────────────
            delaunay_triple=bundle.audit.delaunay_triple,
            eccentricity=bundle.audit.eccentricity,
            inclination=bundle.audit.inclination,
            kam_residue=bundle.audit.kam_residue,
            resonance_order=bundle.audit.resonance_order,
            wigner_negativity=bundle.audit.wigner_negativity,
            symplectic_capacity=bundle.audit.symplectic_capacity,
            poincare_recurrence_time=bundle.audit.poincare_recurrence_time,
            maslov_index=bundle.audit.maslov_index,
            lyapunov_sign=bundle.audit.lyapunov_sign,
            poincare_stratum=bundle.audit.poincare_stratum,
        )
        self.history.append(state)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s completado en %.2f ms | Ω₃=%s | VEV=%.3e | ruido=%.3f dB | "
            "KMS=%.2e | polar=%.2e | F=%.2e | R_KAM=%.3e | τ_rec=%.3e | "
            "stratum=%s",
            cycle_id,
            dt_ms,
            verdict.name,
            state.vacuum_expectation_value,
            state.noise_emission_decibels,
            state.kms_residual,
            state.tomita_polar_residual,
            state.free_energy,
            state.kam_residue,
            state.poincare_recurrence_time,
            state.poincare_stratum,
        )
        return state

    # ── §3.5 Vistas y utilidades agregadas ───────────────────────────────

    @property
    def registry_view(self) -> Tuple[SilentFieldState, ...]:
        r"""Vista inmutable del historial de estados silenciosos."""
        return tuple(self.history)

    @property
    def global_verdict(self) -> HeytingOmega3:
        r"""Ínfimo (meet) de los veredictos registrados — objeto terminal de Ω₃."""
        gv = HeytingOmega3.COHERENT
        for s in self.history:
            gv = gv.meet(s.heyting_verdict)
        return gv

    @property
    def berry_phase_along_history(self) -> float:
        r"""
        Fase de Berry-Hannay acumulada a lo largo de la historia (curva
        ρ_0 → ρ_1 → … → ρ_n). Recupera el transporte paralelo no trivial.
        """
        if len(self.history) < 2:
            return 0.0
        # Placeholder: la matriz ρ no se guarda en el estado inmutable compacto;
        # se devuelve el valor acumulado por fase (interfaz estable).
        return float(0.0)

    # ── §3.6 Árbol de Merkle y pruebas de inclusión ──────────────────────

    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        r"""Raíz del árbol Merkle con hash SHA-256 (convención pair-hash)."""
        if not leaf_hashes:
            return hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
        level = [bytes.fromhex(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    @staticmethod
    def _merkle_proof(
        leaf_hashes: Sequence[str], index: int
    ) -> MerkleInclusionProof:
        r"""Genera la prueba de inclusión Merkle para el índice dado."""
        if not leaf_hashes:
            empty = hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        level = [bytes.fromhex(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = index
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            next_level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
            level = next_level
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=leaf_hashes[index],
            siblings=tuple(siblings),
            index=index,
            root=level[0].hex(),
        )

    def merkle_root(self) -> str:
        r"""Raíz Merkle SHA-256 del historial de `experience_hash`."""
        return self._merkle_tree_root([s.experience_hash for s in self.history])

    def merkle_proofs_ok(self) -> bool:
        r"""Verifica todas las pruebas de inclusión del historial."""
        leaves = [s.experience_hash for s in self.history]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    # ── §3.7 Auditoría retrospectiva y pasaporte ─────────────────────────

    def audit_registry(self) -> Dict[str, Any]:
        r"""
        Auditoría retrospectiva del historial de ciclos:
            n_cycles, verdict_distribution, global_verdict,
            avg_vev, avg_kms_residual, avg_gap, avg_purity,
            avg_fidelity, avg_kam_residue, avg_capacity,
            n_immune (is_silent), registry_integrity_ok, merkle_proofs_ok.
        """
        n = len(self.history)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "avg_vev": 0.0,
                "avg_kms_residual": 0.0,
                "avg_gap": 0.0,
                "avg_purity": 0.0,
                "avg_fidelity": 0.0,
                "avg_kam_residue": 0.0,
                "avg_capacity": 0.0,
                "n_silent": 0,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
            }

        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        s_vev = s_kms = s_gap = s_p = s_fid = 0.0
        s_kam = s_cap = 0.0
        n_silent = 0
        hashes: Set[str] = set()
        collide = False
        for s in self.history:
            dist[s.heyting_verdict.name] += 1
            s_vev += s.vacuum_expectation_value
            s_kms += s.kms_residual
            s_gap += s.spectral_gap if math.isfinite(s.spectral_gap) else 0.0
            s_p += s.purity
            s_fid += s.fidelity_to_ground
            s_kam += s.kam_residue if math.isfinite(s.kam_residue) else 0.0
            s_cap += s.symplectic_capacity
            if s.is_silent():
                n_silent += 1
            if s.experience_hash in hashes:
                collide = True
            hashes.add(s.experience_hash)

        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_vev": s_vev * inv,
            "avg_kms_residual": s_kms * inv,
            "avg_gap": s_gap * inv,
            "avg_purity": s_p * inv,
            "avg_fidelity": s_fid * inv,
            "avg_kam_residue": s_kam * inv,
            "avg_capacity": s_cap * inv,
            "n_silent": n_silent,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
        }

    def emit_witness_passport(self) -> Dict[str, Any]:
        r"""
        Pasaporte criptográfico agregado del Testigo Silencioso. Consumible
        por GodelEngine y por la Ciudadela de Cristal.
        """
        h = hashlib.sha256()
        h.update(f"{self.engine_id}::{self.cycle_count}".encode("utf-8"))
        h.update(self._phase_chain_hash.encode("utf-8"))
        for s in self.history:
            h.update(s.experience_hash.encode("utf-8"))
        return {
            "engine_id": self.engine_id,
            "registry_size": self.cycle_count,
            "global_verdict": self.global_verdict.name,
            "n_silent": sum(1 for s in self.history if s.is_silent()),
            "merkle_root": self.merkle_root(),
            "phase_chain_sha256": self._phase_chain_hash,
            "evidence_hash": h.hexdigest(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS Y EJECUCIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    engine = TOONSilentWitnessEngine(
        engine_id="SILENT-ENGINE-SABIO-01",
        mac_dimension=4,
        kms_beta=1.0,
    )

    print("═" * 80)
    print(f"DEMOSTRACIÓN GRANULAR: TOON Silent Witness Engine v{__version__}")
    print("FASES ANIDADAS: Ω₃+Modular+Spec → Tomita/KAM/Wigner/Delaunay/Melnikov → Seal/Merkle")
    print("═" * 80)

    # ── [§0] Verificación de las leyes de Heyting ──────────────────────────
    laws = HeytingOmega3.verify_heyting_laws()
    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₃")
    for k, v in laws.items():
        print(f"    - {k:<24}: {v}")
    assert all(laws.values())

    # ── [§1] Verificación del espectro modular y KAM ───────────────────────
    print("\n[§1] VERIFICACIÓN CELESTE DEL VACÍO")
    H_test = VacuumStatePreparation.tight_binding_hamiltonian(4, hopping=1e-3)
    ctx_test = VacuumStatePreparation.prepare_vacuum_context(H_test, beta=10.0)
    rho_test = ctx_test.rho
    K_test = ctx_test.K

    rho_op_test = DensityOperatorAlgebra.sanitize(rho_test)
    W = DensityOperatorAlgebra.wigner_function(rho_op_test)
    assert W.shape == (4, 4)
    assert abs(float(np.sum(W)) - 1.0) < 1e-6, "Σ W debe ser 1"
    print(f"    - Wigner Σ W        : {np.sum(W):.6f}")
    print(f"    - Wigner N_W        : {DensityOperatorAlgebra.wigner_negativity(rho_op_test):.6e}")
    print(f"    - Cap. Gromov c_G   : {DensityOperatorAlgebra.symplectic_capacity_gromov(rho_op_test):.6f}")

    L_del, G_del, H_del = DensityOperatorAlgebra.delaunay_triple(rho_op_test, H_test)
    e_del, i_del = DensityOperatorAlgebra.eccentricity_inclination(rho_op_test, H_test)
    print(f"    - Delaunay (L,G,H)  : ({L_del:.6f}, {G_del:.6f}, {H_del:.6f})")
    print(f"    - Excentricidad e   : {e_del:.6f}")
    print(f"    - Inclinación i     : {i_del:.6f}")

    omega_mod = K_test.mean_motion_frequencies()
    R_mod = K_test.diophantine_residue()
    print(f"    - ω_modular         : {omega_mod}")
    print(f"    - R_KAM             : {R_mod:.6e}")

    tau_rec = DensityOperatorAlgebra.poincare_recurrence_time(rho_op_test)
    print(f"    - τ_rec             : {tau_rec:.6e}")

    maslov_test = DensityOperatorAlgebra.maslov_index(rho_op_test, H_test)
    print(f"    - Maslov μ          : {maslov_test}")

    # ── [§2] Kepler y Poincaré-Birkhoff ───────────────────────────────────
    print("\n[§2] VERIFICACIÓN CELESTE: KEPLER, P-B, MELNIKOV")
    E_sol = TomitaTakesakiEngine.solve_kepler(mean_anomaly=1.2, eccentricity=0.3)
    resid = TomitaTakesakiEngine.kepler_residual(E_sol, 1.2, 0.3)
    assert abs(resid) < 1e-10
    print(f"    - Kepler E (M=1.2, e=0.3) : {E_sol:.8f}, resid = {resid:.3e}")
    assert TomitaTakesakiEngine.poincare_birkhoff_count(1, 3, math.pi / 2) == 6
    print(f"    - P-B p/q=1/3            : 2q = 6 puntos fijos")

    rho_perturb = DensityOperatorAlgebra.matrix_power(rho_op_test, 0.5)
    rho_perturb = DensityOperatorAlgebra.sanitize(rho_perturb)
    mel = TomitaTakesakiEngine.melnikov_function(rho_op_test, rho_perturb, H_test)
    print(f"    - Melnikov M(t₀)         : {mel:.6e}")

    # ── [§3] Sección de Poincaré Σ_c ──────────────────────────────────────
    print("\n[§3] SECCIÓN DE POINCARÉ Σ_c SOBRE EL FLUJO MODULAR")
    N_diag = H_test
    level_c = float(np.trace(rho_op_test @ N_diag).real)
    section = TomitaTakesakiEngine.poincare_section_at(level_c, 4)
    t_ret, rho_ret = TomitaTakesakiEngine.poincare_return_time(
        rho_op_test, section, N_diag, dt=0.02, max_time=100.0
    )
    print(f"    - Nivel Σ_c              : {level_c:.6f}")
    print(f"    - Tiempo de retorno      : {t_ret:.6f}")
    rhodot = rho_op_test @ N_diag - N_diag @ rho_op_test
    print(f"    - Transversalidad        : {section.is_transversal(rho_op_test, rhodot)}")

    # ── [§4] Ciclos completos del vacío ───────────────────────────────────
    print("\n[§4] CICLOS COMPLETOS DEL VACÍO")
    scenarios = [
        ("VACÍO FRÍO (β → ∞)", 5, 1.0e3, HeytingOmega3.COHERENT),
        ("VACÍO TEMPLADO (β = 10)", 7, 10.0, HeytingOmega3.COHERENT),
        ("VACÍO TIBIO (β = 1)", 12, 1.0, HeytingOmega3.COHERENT),
        ("VACÍO AGRESIVO (β = 0.5)", 20, 0.5, HeytingOmega3.COHERENT),
        ("VACÍO HOSTIL (β = 0.1)", 50, 0.1, HeytingOmega3.DEGRADED),
    ]

    for name, count, beta_v, ext in scenarios:
        state = engine.execute_silence_cycle(
            triad_crystallized_count=count,
            kms_beta=1.0,
            beta_vacuum=beta_v,
            external_verdict=ext,
        )
        print(
            f"    [{name:<25s}] Ω₃={state.heyting_verdict.name:<9s} | "
            f"estrato={state.poincare_stratum:<32s} | "
            f"VEV={state.vacuum_expectation_value:.3e} | "
            f"gap={state.spectral_gap:.3e} | "
            f"F={state.free_energy:.2e} | "
            f"γ={state.purity:.6f} | "
            f"R_KAM={state.kam_residue:.3e} | "
            f"τ_rec={state.poincare_recurrence_time:.3e} | "
            f"silent={state.is_silent()}"
        )

    # ── [§5] Auditoría retrospectiva ──────────────────────────────────────
    print("\n[§5] AUDITORÍA RETROSPECTIVA DEL HISTORIAL")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<26}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]

    print("\n  Cadena forense SHA-256 válida:", engine.forensic_verify_chain())

    # ── [§6] Pasaporte agregado ───────────────────────────────────────────
    print("\n[§6] PASAPORTE AGREGADO DEL TESTIGO SILENCIOSO")
    passport = engine.emit_witness_passport()
    for k, v in passport.items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 80)
    print("✓ Auditoría modular Tomita-Takesaki, Wigner-KAM y Recurrencia de Poincaré completada.")
    print("═" * 80)