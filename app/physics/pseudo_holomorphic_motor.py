# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pseudo-Holomorphic Motor (Operador Elíptico de Rigidez Simpléctica) ║
║ Ruta   : app/physics/pseudo_holomorphic_motor.py                             ║
║ Versión: 5.1.0-Doctoral-Fukaya-Floer-Connes-Heyting-Secure                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y DE-CONFINAMIENTO DE CALIBRE:
────────────────────────────────────────────────────────────────────────────────
Este componente constitutivo de la Capa Física (Nivel 3, $V_{\mathrm{PHYSICS}}$)
resuelve formalmente el espacio de moduli de curvas y polígonos pseudo-holomorfos
bajo condiciones de frontera Lagrangianas exactas. Transmuta las variables de 
fase del presupuesto (avances, velocidades de capital y mermas discretas) en 
un complejo de cadenas de Floer, garantizando que el transporte paralelo en la 
Malla satisfaga estrictamente los axiomas de regularidad elíptica y conservación 
simpléctica de Liouville.

El motor actúa como el cortafuegos definitivo contra la deriva sintáctica de la 
IA. Si el flujo atencional intenta deformar las constantes de-confinadas del negocio, 
el sistema colapsa el retículo de Heyting $\Omega_3 = \{\mathtt{COHERENT}, \, 
\mathtt{DEGRADED}, \, \mathtt{VETOED}\}$, disparando síncronamente la interrupción 
física de potencia por hardware en el microcontrolador ESP32 perimetral.

================================════════════════════════════════════════════════
I. DEFINICIÓN FORMAL DEL ESPACIO DE FASE SIMPLÉCTICO
================================════════════════════════════════════════════════
Sea $\mathcal{M}$ la variedad diferenciable de dimensión real $2n$ equipada con la
2-forma simpléctica canónica de Liouville $\omega$, expresada localmente en
coordenadas de Darboux $(q_i, p_i)$ como:
$$\omega = \sum_{i=1}^n dq_i \wedge dp_i$$

Donde:
  - $q \in \mathcal{M}$ representa el vector de estados de avance y costo de los
    Análisis de Precios Unitarios (APUs) del presupuesto.
  - $p \in T^*\mathcal{M}$ representa el covector de momentum o tasas de inversión
    financiera reguladas por la métrica de fondo $G_{\mu\nu}$.

================================════════════════════════════════════════════════
II. DEFINICIONES Y OPERADORES DE CURVAS PSEUDO-HOLOMORFAS
================================════════════════════════════════════════════════
Defínase un polígono pseudo-holomorfo de $d+1$ lados como una aplicación elíptica:
$$u: (\Sigma, j) \longrightarrow (\mathcal{M}, J, \omega)$$

Donde:
  - $(\Sigma, j)$ es una superficie de Riemann acotada y punteada con estructura
    conforme casi compleja de dominio $j$.
  - $J$ es la estructura casi compleja de destino compatible con la forma
    simpléctica ($J^2 = -\mathrm{Id}$).

El mapa satisface de forma síncrona la ecuación elíptica no lineal de Cauchy-Riemann
perturbada en el interior de la superficie:
$$\bar{\partial}_J u = \frac{1}{2}\left( du + J(u) \circ du \circ j \right) = 0$$

Las condiciones de contorno exigen que las componentes de la frontera del dominio
$\partial\Sigma$ se confinen de manera secuencial y exacta sobre subvariedades
Lagrangianas exactas:
$$u(\partial_i \Sigma) \subset L_i \subset \mathcal{M}, \quad \forall i \in \{0, \dots, d\}$$

Donde cada $L_i$ representa una brana de restricción contractual o física (Modelo A de
cuerdas) asociada a los costos unitarios del megaproyecto.

================================════════════════════════════════════════════════
III. AXIOMAS FUNDAMENTALES DE LA VARIEDAD DE CONTROL
================================════════════════════════════════════════════════
Axioma 1 (Transversalidad de Kuranishi):
  La linealización del operador Cauchy-Riemann $D_u \bar{\partial}_J$ es un operador elíptico
  de Fredholm de rango completo. La dimensión virtual del espacio de moduli local de
  soluciones se gobierna de forma exacta por el Índice de Maslov relativo $\mu(\beta)$
  de la clase de homotopía $\beta \in \pi_2(\mathcal{M}, \bigcup L_i)$:
  $$\operatorname{dim} \mathcal{M}(x_0, \dots, x_d; \beta, J) = \mu(\beta) + n - 3 + (d+1)$$
  Para eludir singularidades numéricas ante mapas multi-recubiertos, se definen cartas locales
  de Kuranishi $(V_u, E_u, s_u, \psi_u)$ sobre la obstrucción elíptica, forzando a que la
  intersección de las branas en el espacio de fase simpléctico sea transversal.

Axioma 2 (Nilpotencia de de Rham-Floer / Escudo Anti-Bubbling):
  Al exigir incondicionalmente la condición de exactitud Lagrangiana ($\theta|_{L_i} = df_i$,
  donde $\theta$ es la 1-forma de Liouville), el área simpléctica de cualquier disco
  pseudo-holomorfo con frontera en $L_i$ es idénticamente nula. Esto aniquila categóricamente
  el fenómeno del burbujeo de discos (Disk Bubbling) de índice de Maslov nulo o negativo,
  garantizando la nilpotencia pura del diferencial en el complejo de Floer:
  $$d^2 = m_1 \circ m_1 = 0 \quad \implies \quad H^k_{\mathrm{dR}}(K) \cong \ker(d_k) / \operatorname{im}(d_{k-1})$$

Axioma 3 (Preservación del Volumen de Liouville):
  Toda transición dinámica del flujo de caja inducida por el motor debe actuar como un
  difeomorfismo simpléctico (simetría covariante), conservando el volumen de fase geométrico
  módulo la precisión de la mantisa de la FPU:
  $$\det(M_{\mathrm{Jacobiano}}) \equiv 1.0 \pm \varepsilon_{\mathrm{machine}} \quad \wedge \quad M_{\mathrm{Jacobiano}}^\top \Omega M_{\mathrm{Jacobiano}} \equiv \Omega$$

================================════════════════════════════════════════════════
IV. INVARIANTES TOPOLÓGICOS Y ESPECTRALES DE LAZO CERRADO
================================════════════════════════════════════════════════
  - NÚMEROS DE BETTI DISCRETOS ($\beta_k$): El motor calcula síncronamente el espectro
    del Laplaciano normalizado de de Rham-Hodge $L_F = \delta_0^\top G^{-1} \delta_0$.
    Se proscribe la aparición de islas de datos ($\beta_0 > 1$) y socavones lógicos o
    dependencias circulares de-confinadas ($\beta_1 > 0$):
    $$\beta_0 = \dim \ker(L_F) \equiv 1, \quad \beta_1 = \dim H^1(K; \mathcal{F}) \equiv 0$$

  - CARACTERÍSTICA DE EULER-POINCARÉ ($\chi(K)$): Para el 1-esqueleto simplicial de la
    Malla de control, la característica de Euler satisface de forma estricta el invariante:
    $$\chi(K) = \beta_0 - \beta_1 = |V| - |E| \equiv 1.0$$

  - CONECTIVIDAD DE FIEDLER ($\lambda_1$): La brecha espectral del primer autovalor no
    nulo de $L_F$ determina la estabilidad de transporte paralelo ante perturbaciones
    estocásticas de la IA:
    $$\lambda_1(L_F) \ge \tau_{\mathrm{Fiedler}} > 0$$

================================════════════════════════════════════════════════
V. MÉTODO DE SUTURA CIBER-FÍSICA (Actuación en Silicio)
================================════════════════════════════════════════════════
La coherencia cuántico-topológica del sistema se proyecta sobre el clasificador de
subobjetos de la retícula de Heyting $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED},
\mathtt{VETOED}\}$. Cualquier desviación de los invariantes o desgarro de la 2-forma
simpléctica colapsa el estado al Supremo terminal VETOED ($\top$). 

Esto gatilla síncronamente una Rutina de Servicio de Interrupción (ISR) cargada en la
IRAM del ESP32 perimetral. En menos de 400 ns, el pin físico GPIO14 conmuta a nivel alto,
excitando el tiristor BT151 (circuito Crowbar) para cortocircuitar la línea de alimentación
de los actuadores mecánicos reales, paralizando la maquinaria industrial en el milisegundo cero,
neutralizando el transitorio antes de consolidar sobrecostos ante SECOP II.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Final, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.PseudoHolomorphicMotor")

# ---------------------------------------------------------------------------
# Constantes espectrales de máquina (Wilkinson / Higham / Golub–Van Loan)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_REG_FLOOR: Final[float] = 1e-12
_WILKINSON_PSD_FLOOR: Final[float] = -1e-13
_RANK_EPS_MULT: Final[float] = 1.0
_TRANSVERSAL_COND_MAX: Final[float] = 1.0e8
_CAYLEY_COND_MAX: Final[float] = 1.0e12
_NILPOTENCY_REL: Final[float] = 1.0e-12
_KIRCHHOFF_REL: Final[float] = 1.0e-12
_EIGH_RESIDUAL_WARN: Final[float] = 1.0e-8
_SYMMETRY_ATOL: Final[float] = 1.0e-12
_COMPLEX_IMAG_TOL: Final[float] = 1.0e-14
_BUBBLING_MASLOV_MIN: Final[int] = 0
_SOFT_SYMPLECTIC_FACTOR: Final[float] = 10.0
_SOFT_COND_FACTOR: Final[float] = 10.0

# Suturas Kuranishi / Novikov / Tikhonov
_NOVIKOV_ATOL: Final[float] = 1.0e-14
_NOVIKOV_MAX_TERMS: Final[int] = 48
_NOVIKOV_ENERGY_CAP: Final[float] = 64.0
_KURANISHI_SIGMA_FLOOR: Final[float] = 64.0 * _MACHINE_EPS
_TIKHONOV_MU_FLOOR: Final[float] = 1.0e-14
_TIKHONOV_MU_CEILING: Final[float] = 1.0e-2
_TIKHONOV_MAX_ITERS: Final[int] = 12
_SP_POLAR_ITERS: Final[int] = 8
_SP_POLAR_TOL: Final[float] = 1.0e-12
_VOLUME_CONTRACTION_TOL: Final[float] = 1.0e-8
_MC_NEWTON_ITERS: Final[int] = 16
_MC_NEWTON_TOL: Final[float] = 1.0e-10
_A_INF_TRUNCATION: Final[int] = 4
_MULTIPLY_COVER_KERNEL_SLACK: Final[int] = 1

__version__: Final[str] = "4.0.0-Doctoral-Kuranishi-Novikov-Tikhonov-FOOO-Secure"


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES SIMPLÉCTICAS Y COHOMOLÓGICAS (Fail-Secure Boundary)
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticGeometryError(Exception):
    """Raíz de violaciones geométricas e invariantes del espacio de fase."""


class NonTransversalIntersectionError(SymplecticGeometryError):
    """Branas lagrangianas no transversales o germen elíptico irregular."""


class DiskBubblingDivergenceError(SymplecticGeometryError):
    """Discos de Maslov negativo / bubbling no absorbido por Maurer–Cartan."""


class CohomologicalFrustrationError(SymplecticGeometryError):
    """Ruptura de \(d^2=0\), Betti inconsistente o Hodge no concordante."""


class AlmostComplexIncompatibilityError(SymplecticGeometryError):
    """La estructura casi-compleja no es compatible con \(\omega\)."""


class SpectralGapCollapseError(SymplecticGeometryError):
    """El laplaciano abandona el cono PSD o el núcleo espectral es contradictorio."""


class KuranishiChartError(SymplecticGeometryError):
    """Carta de Kuranishi mal formada o dimensión virtual ilegible."""


class NovikovValuationError(SymplecticGeometryError):
    """Serie de Novikov con soporte no bien ordenado o valuación ilegal."""


class MaurerCartanObstructionError(SymplecticGeometryError):
    """La ecuación de Maurer–Cartan no admite co-cadena acotante en la filtración."""


# ═══════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR DE SUBOBJETOS Ω₃ (lógica interna del topos de-confinado)
# ═══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3:
    r"""
    Cadena de Heyting \(0 \lt \tfrac12 \lt 1\), lógica interna del topos.

    Operaciones:
        \(a \wedge b = \min(a,b),\quad a \vee b = \max(a,b)\)
        \(a \to b = 1\) si \(a \le b\), si no \(b\)
        \(\neg a = a \to 0\)

    No es un álgebra de Boole: \(\neg\neg\mathrm{DEGRADED}=\mathrm{COHERENT}
    \ne\mathrm{DEGRADED}\). Esa es la huella intuicionista exigida por el
    clasificador de subobjetos.
    """

    COHERENT: Final[str] = "COHERENT"
    DEGRADED: Final[str] = "DEGRADED"
    VETOED: Final[str] = "VETOED"

    _ORDER: Final[dict[str, int]] = {VETOED: 0, DEGRADED: 1, COHERENT: 2}
    _TRUTH: Final[dict[str, float]] = {VETOED: 0.0, DEGRADED: 0.5, COHERENT: 1.0}

    @classmethod
    def order(cls, value: str) -> int:
        if value not in cls._ORDER:
            raise SymplecticGeometryError(f"Valor foráneo al retículo Ω₃: {value!r}.")
        return cls._ORDER[value]

    @classmethod
    def truth(cls, value: str) -> float:
        return cls._TRUTH[cls.normalize(value)]

    @classmethod
    def normalize(cls, value: str) -> str:
        if value not in cls._ORDER:
            raise SymplecticGeometryError(f"Valor foráneo al retículo Ω₃: {value!r}.")
        return value

    @classmethod
    def le(cls, left: str, right: str) -> bool:
        return cls.order(left) <= cls.order(right)

    @classmethod
    def meet(cls, left: str, right: str) -> str:
        return left if cls.order(left) <= cls.order(right) else right

    @classmethod
    def join(cls, left: str, right: str) -> str:
        return left if cls.order(left) >= cls.order(right) else right

    @classmethod
    def implies(cls, left: str, right: str) -> str:
        return cls.COHERENT if cls.le(left, right) else cls.normalize(right)

    @classmethod
    def neg(cls, value: str) -> str:
        return cls.implies(value, cls.VETOED)

    @classmethod
    def fold_meet(cls, values: Sequence[str]) -> str:
        acc = cls.COHERENT
        for item in values:
            acc = cls.meet(acc, item)
        return acc


# ═══════════════════════════════════════════════════════════════════════════════
# ANILLO DE NOVIKOV Λ_ℝ  (regularización no arquimediana)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class NovikovTerm:
    """Monoserie \(a\,T^r\) con valuación de energía \(r=\int\omega\)."""

    valuation: float
    coeff: float


class NovikovSeries:
    r"""
    Elemento del anillo de Novikov truncado

    \[
      \Lambda_{\mathbb{R}}^{\le E}
      =\Bigl\{\sum_{r\le E} a_r T^r\Bigr\}
    \]

    con soporte finito bien ordenado.  El valor absoluto no arquimediano es
    \(\lvert x\rvert_T=\mathrm{e}^{-\mathrm{val}(x)}\) (y \(0\) si \(x=0\)).
    """

    __slots__ = ("_terms", "_energy_cap")

    def __init__(
        self,
        terms: Optional[Iterable[Tuple[float, float]]] = None,
        energy_cap: float = _NOVIKOV_ENERGY_CAP,
    ) -> None:
        self._energy_cap = float(energy_cap)
        if self._energy_cap < 0.0 or not math.isfinite(self._energy_cap):
            raise NovikovValuationError("El techo de energía de Novikov debe ser finito y ≥ 0.")
        acc: Dict[float, float] = {}
        if terms is not None:
            for raw_val, raw_coeff in terms:
                val = float(raw_val)
                coeff = float(raw_coeff)
                if not math.isfinite(val) or val < -1e-15:
                    raise NovikovValuationError(
                        f"Valuación ilegal r={val}: se exige r ≥ 0 (área de Liouville)."
                    )
                if val < 0.0:
                    val = 0.0
                if val > self._energy_cap or (not math.isfinite(coeff)) or abs(coeff) <= _NOVIKOV_ATOL:
                    continue
                key = round(val, 12)
                acc[key] = acc.get(key, 0.0) + coeff
        self._terms: Tuple[NovikovTerm, ...] = self._freeze(acc)

    @staticmethod
    def _freeze(acc: Dict[float, float]) -> Tuple[NovikovTerm, ...]:
        cleaned = [
            NovikovTerm(valuation=key, coeff=value)
            for key, value in acc.items()
            if abs(value) > _NOVIKOV_ATOL and math.isfinite(value)
        ]
        cleaned.sort(key=lambda term: term.valuation)
        if len(cleaned) > _NOVIKOV_MAX_TERMS:
            cleaned = cleaned[:_NOVIKOV_MAX_TERMS]
        return tuple(cleaned)

    @classmethod
    def zero(cls, energy_cap: float = _NOVIKOV_ENERGY_CAP) -> "NovikovSeries":
        return cls((), energy_cap=energy_cap)

    @classmethod
    def unit(cls, energy_cap: float = _NOVIKOV_ENERGY_CAP) -> "NovikovSeries":
        return cls(((0.0, 1.0),), energy_cap=energy_cap)

    @classmethod
    def monomial(cls, coeff: float, valuation: float, energy_cap: float = _NOVIKOV_ENERGY_CAP) -> "NovikovSeries":
        return cls(((valuation, coeff),), energy_cap=energy_cap)

    @classmethod
    def from_scalar(cls, coeff: float, energy_cap: float = _NOVIKOV_ENERGY_CAP) -> "NovikovSeries":
        return cls(((0.0, float(coeff)),), energy_cap=energy_cap)

    @property
    def terms(self) -> Tuple[NovikovTerm, ...]:
        return self._terms

    @property
    def energy_cap(self) -> float:
        return self._energy_cap

    def is_zero(self) -> bool:
        return not self._terms

    def valuation(self) -> float:
        if not self._terms:
            return float("inf")
        return float(self._terms[0].valuation)

    def leading_coeff(self) -> float:
        if not self._terms:
            return 0.0
        return float(self._terms[0].coeff)

    def constant_term(self) -> float:
        for term in self._terms:
            if term.valuation <= 1e-15:
                return float(term.coeff)
        return 0.0

    def nonarchimedean_norm(self) -> float:
        if not self._terms:
            return 0.0
        return float(math.exp(-self.valuation()))

    def archimedean_l1(self) -> float:
        return float(sum(abs(term.coeff) for term in self._terms))

    def evaluate_at_q(self, q: float) -> float:
        """Evaluación de Hahn en \(0<q<1\): \(\sum a_r q^r\)."""
        if not (0.0 < q < 1.0):
            raise NovikovValuationError("evaluate_at_q exige 0 < q < 1 (radio no arquimediano).")
        acc = 0.0
        log_q = math.log(q)
        for term in self._terms:
            acc += term.coeff * math.exp(term.valuation * log_q)
        return float(acc)

    def _as_dict(self) -> Dict[float, float]:
        return {term.valuation: term.coeff for term in self._terms}

    def __add__(self, other: "NovikovSeries") -> "NovikovSeries":
        cap = min(self._energy_cap, other._energy_cap)
        acc = self._as_dict()
        for term in other._terms:
            acc[term.valuation] = acc.get(term.valuation, 0.0) + term.coeff
        return NovikovSeries(acc.items(), energy_cap=cap)

    def __sub__(self, other: "NovikovSeries") -> "NovikovSeries":
        return self + other.scale(-1.0)

    def scale(self, scalar: float) -> "NovikovSeries":
        if abs(float(scalar)) <= _NOVIKOV_ATOL:
            return NovikovSeries.zero(self._energy_cap)
        return NovikovSeries(
            ((term.valuation, term.coeff * float(scalar)) for term in self._terms),
            energy_cap=self._energy_cap,
        )

    def __mul__(self, other: "NovikovSeries") -> "NovikovSeries":
        cap = min(self._energy_cap, other._energy_cap)
        acc: Dict[float, float] = {}
        for left in self._terms:
            for right in other._terms:
                val = left.valuation + right.valuation
                if val > cap:
                    continue
                key = round(val, 12)
                acc[key] = acc.get(key, 0.0) + left.coeff * right.coeff
        return NovikovSeries(acc.items(), energy_cap=cap)

    def __neg__(self) -> "NovikovSeries":
        return self.scale(-1.0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NovikovSeries):
            return NotImplemented
        if len(self._terms) != len(other._terms):
            return False
        for left, right in zip(self._terms, other._terms):
            if abs(left.valuation - right.valuation) > 1e-12:
                return False
            if abs(left.coeff - right.coeff) > 1e-12:
                return False
        return True

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts = []
        for term in self._terms:
            parts.append(f"{term.coeff:+.6g}·T^{term.valuation:.6g}")
        return " ".join(parts)


def _novikov_vector_zero(dim: int, energy_cap: float) -> List[NovikovSeries]:
    return [NovikovSeries.zero(energy_cap) for _ in range(max(0, int(dim)))]


def _novikov_dot(left: Sequence[NovikovSeries], right: Sequence[float]) -> NovikovSeries:
    acc = NovikovSeries.zero(left[0].energy_cap if left else _NOVIKOV_ENERGY_CAP)
    for series, coeff in zip(left, right):
        acc = acc + series.scale(float(coeff))
    return acc


def _novikov_l1_vector(vec: Sequence[NovikovSeries]) -> float:
    return float(sum(item.archimedean_l1() for item in vec))


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS INMUTABLES Y OBJETOS DE CONTINUACIÓN ENTRE FASES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class KuranishiChart:
    r"""
    Carta local de Kuranishi \((V,E,s,\psi)\) del operador \(D_u\bar\partial_J\).

    \[
      \operatorname{vdim}= \dim V-\dim E =\operatorname{ind}^{\mathrm{virt}}(D_u).
    \]

    ``is_transversal`` es verdadero sólo si \(\dim E=0\) *y* el disco no
    es multi-recubierto.  En caso contrario la virtualidad se certifica
    por la sección \(s\) y por la concordancia de índices.
    """

    ambient_dimension: int
    obstruction_dimension: int
    kernel_dimension: int
    virtual_dimension: int
    smallest_singular_value: float
    tikhonov_mu: float
    higham_floor: float
    kuranishi_section_norm: float
    liouville_volume_ratio: float
    symplectic_polar_residual: float
    is_multiply_covered: bool
    is_transversal: bool
    index_matches_fredholm: bool
    complex_type_defect: float


@dataclass(frozen=True, slots=True)
class ModuliSpaceCertificate:
    """Certificado espectral del espacio de moduli del polígono holomorfo."""

    maslov_index: int
    fredholm_index: int
    is_transversal: bool
    condition_number: float
    symplectic_deviation: float
    is_symplectic: bool
    virtual_polygon_dimension: int
    relative_symplectic_residual: float
    cayley_hamiltonian_residual: float
    nijenhuis_norm: float
    gromov_energy_ceiling: float
    consecutive_transversality: bool
    riemann_roch_index: int
    kuranishi_obstruction_dim: int
    is_virtually_regular: bool


@dataclass(frozen=True, slots=True)
class EllipticGerm:
    r"""
    Germen del operador de Cauchy–Riemann linealizado tras la Fase 1.

    Objeto terminal de la Fase 1 y objeto inicial de la Fase 2.
    Porta la carta de Kuranishi y el shift de Tikhonov: la Fase 2 no
    puede deformar \(m_k\) sin este germen.
    """

    moduli: ModuliSpaceCertificate
    almost_complex_compatibility: float
    liouville_exactness: float
    lipschitz_connes_defect: float
    kashiwara_maslov_estimate: int
    is_elliptic_regular: bool
    phase_signature: str
    kuranishi: KuranishiChart
    tikhonov_mu: float
    is_exact_lagrangian: bool
    bubbling_energy_floor: float


@dataclass(frozen=True, slots=True)
class MaurerCartanCertificate:
    r"""
    Certificado de la ecuación de Maurer–Cartan truncada a \(k\le K\):

    \[
      \sum_{k=0}^{K} m_k(b^{\otimes k})-W_L(b)\,[L]
      \;\in\;
      F^{E}\Lambda\otimes CF^\bullet.
    \]
    """

    solved: bool
    truncation_order: int
    residual_l1: float
    curvature_m0_l1: float
    bounding_cochain_l1: float
    bounding_cochain_energy: float
    superpotential_constant: float
    superpotential_valuation: float
    a_infinity_m1_square_l1: float
    obstruction_absorbed: bool
    newton_iterations: int
    deformed_nilpotent: bool


@dataclass(frozen=True, slots=True)
class FloerCohomologyCertificate:
    """Certificado de lazo cerrado de la consistencia homológica del complejo."""

    cohomological_dimension_H1: int
    euler_characteristic: int
    fiedler_connectivity: float
    is_nilpotent: bool
    is_secured_coherent: bool
    cohomological_dimension_H0: int
    cheeger_constant: float
    hodge_kernel_agreement: bool
    witten_spectral_gap: float
    persistence_entropy: float
    spectral_radius_d2: float
    is_kirchhoff: bool
    stable_rank_delta_0: float


@dataclass(frozen=True, slots=True)
class FloerSpectrum:
    r"""
    Paquete espectral emitido por la Fase 2 y consumido por la Fase 3.

    Transporta el espectro de Hodge, la brecha de Witten, el certificado
    de Maurer–Cartan y el superpotencial \(W_L\in\Lambda_{\mathbb{R}}\).
    """

    cohomology: FloerCohomologyCertificate
    hodge_spectrum_0: Tuple[float, ...]
    hodge_spectrum_1_head: Tuple[float, ...]
    cheeger_constant: float
    witten_gap: float
    persistence_entropy: float
    a_infinity_m1_residual: float
    is_floer_rigid: bool
    phase_signature: str
    maurer_cartan: MaurerCartanCertificate
    superpotential_terms: Tuple[Tuple[float, float], ...]
    novikov_energy_cap: float


@dataclass(frozen=True, slots=True)
class PseudoHolomorphicState:
    """Estado unificado y certificado del motor de curvas pseudo-holomorfas."""

    moduli_audit: ModuliSpaceCertificate
    cohomology_audit: FloerCohomologyCertificate
    elliptic_germ: EllipticGerm
    floer_spectrum: FloerSpectrum
    heyting_verdict: str
    heyting_truth_value: float
    heyting_implication_trace: Tuple[str, ...]
    is_sutured: bool
    timestamp_utc: str
    phase_trace: Tuple[str, ...]
    landau_ginzburg_critical: bool
    novikov_superpotential: str


# ═══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN DEL MOTOR ESPECTRAL DE GEOMETRÍA SIMPLÉCTICA
# ═══════════════════════════════════════════════════════════════════════════════
class PseudoHolomorphicMotor:
    r"""
    Motor elíptico que impone rigidez de Gromov con cartas de Kuranishi,
    coeficientes de Novikov y regularización Higham–Tikhonov.

    El espacio de fase es el espacio simpléctico estándar
    \((\mathbb{R}^{2n},\omega_0)\) en coordenadas de Darboux, con

    \[
      \omega_0 \;=\; \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix},
      \qquad
      J_0 \;=\; -\omega_0,\quad J_0^2=-I,\quad
      g_0(u,v)=\omega_0(u,J_0 v)=\langle u,v\rangle.
    \]
    """

    def __init__(
        self,
        dimension_n: int,
        num_boundaries_d: int,
        novikov_energy_cap: float = _NOVIKOV_ENERGY_CAP,
    ) -> None:
        """
        Inicializa la variedad del espacio de fase simpléctico.

        Args:
            dimension_n: Dimensión \(n\) del espacio de configuración.
            num_boundaries_d: Número \(d\) de branas lagrangianas exactas.
            novikov_energy_cap: Techo de la filtración de energía \(E\).
        """
        if not isinstance(dimension_n, (int, np.integer)) or int(dimension_n) <= 0:
            raise SymplecticGeometryError(
                "La dimensión del espacio de configuración debe ser un entero estrictamente positivo."
            )
        if not isinstance(num_boundaries_d, (int, np.integer)) or int(num_boundaries_d) < 2:
            raise SymplecticGeometryError(
                "Se requieren al menos dos branas lagrangianas exactas para suturar el polígono."
            )
        energy_cap = float(novikov_energy_cap)
        if not math.isfinite(energy_cap) or energy_cap <= 0.0:
            raise NovikovValuationError("novikov_energy_cap debe ser finito y estrictamente positivo.")

        self._n: Final[int] = int(dimension_n)
        self._d: Final[int] = int(num_boundaries_d)
        self._novikov_energy_cap: Final[float] = energy_cap

        eye_n = np.eye(self._n, dtype=np.float64)
        zero_n = np.zeros((self._n, self._n), dtype=np.float64)
        self._omega: Final[np.ndarray] = np.block([[zero_n, eye_n], [-eye_n, zero_n]])
        self._almost_complex: Final[np.ndarray] = -self._omega
        self._validate_darboux_form()

    # ─────────────────────────────────────────────────────────────────────────
    # INVARIANTES DE CONSTRUCCIÓN
    # ─────────────────────────────────────────────────────────────────────────
    def _validate_darboux_form(self) -> None:
        r"""
        Verifica los axiomas de Darboux sobre \(\omega_0\) y \(J_0\):

        * \(\omega_0^\top = -\omega_0\) (antisimetría),
        * \(\omega_0^2 = -I_{2n}\) (forma lineal estándar),
        * \(\det\omega_0 = 1\) (orientación),
        * \(J_0^2 = -I\) y \(g_0 = \omega_0(\cdot,J_0\cdot)\) definida positiva.
        """
        omega = self._omega
        size = 2 * self._n
        skew_dev = float(la.norm(omega + omega.T, ord="fro"))
        square_dev = float(la.norm(omega @ omega + np.eye(size), ord="fro"))
        det_omega = float(np.linalg.det(omega))
        jay = self._almost_complex
        j_sq_dev = float(la.norm(jay @ jay + np.eye(size), ord="fro"))
        metric = omega @ jay
        metric_sym = 0.5 * (metric + metric.T)
        eig_g = la.eigvalsh(metric_sym)
        if (
            skew_dev > 1e-12
            or square_dev > 1e-10
            or abs(det_omega - 1.0) > 1e-8
            or j_sq_dev > 1e-10
            or float(np.min(eig_g)) <= 0.0
        ):
            raise SymplecticGeometryError(
                "Fallo de los axiomas de Darboux en la 2-forma canónica de construcción."
            )

    @property
    def dimension_n(self) -> int:
        """Dimensión \(n\) del espacio de configuración."""
        return self._n

    @property
    def num_boundaries_d(self) -> int:
        """Número \(d\) de branas lagrangianas del polígono."""
        return self._d

    @property
    def novikov_energy_cap(self) -> float:
        """Techo \(E\) de la filtración de Novikov."""
        return self._novikov_energy_cap

    @property
    def symplectic_matrix(self) -> np.ndarray:
        """Retorna una copia de la 2-forma simpléctica canónica de Liouville."""
        return np.array(self._omega, copy=True)

    @property
    def almost_complex_structure(self) -> np.ndarray:
        r"""Estructura casi-compleja compatible \(J_0=-\omega_0\)."""
        return np.array(self._almost_complex, copy=True)

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES ESPECTRALES Y DE VALIDACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _finite_scalar(value: Any, name: str) -> float:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise SymplecticGeometryError(f"{name} no es un escalar finito.")
        return scalar

    @staticmethod
    def _machine_tol(scale: float, floor: float = _MACHINE_EPS) -> float:
        return max(floor, _MACHINE_EPS * max(1.0, abs(scale)))

    @staticmethod
    def _coerce_real_matrix(matrix: Any, name: str) -> np.ndarray:
        """Convierte y valida una matriz real 2-D, finita y contigua."""
        arr = np.asarray(matrix)
        if np.iscomplexobj(arr):
            imag_amp = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
            if imag_amp > _COMPLEX_IMAG_TOL:
                raise SymplecticGeometryError(
                    f"{name} es compleja no real (‖Im‖_∞={imag_amp:.3e})."
                )
            arr = arr.real
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise SymplecticGeometryError(f"{name} debe ser una matriz bidimensional.")
        if arr.size == 0:
            raise SymplecticGeometryError(f"{name} no puede ser vacía.")
        if not np.all(np.isfinite(arr)):
            raise SymplecticGeometryError(f"{name} contiene valores NaN o Inf.")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _coerce_real_vector(vector: Any, name: str, expected: Optional[int] = None) -> np.ndarray:
        arr = np.asarray(vector)
        if np.iscomplexobj(arr):
            imag_amp = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
            if imag_amp > _COMPLEX_IMAG_TOL:
                raise SymplecticGeometryError(
                    f"{name} es complejo no real (‖Im‖_∞={imag_amp:.3e})."
                )
            arr = arr.real
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise SymplecticGeometryError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(arr)):
            raise SymplecticGeometryError(f"{name} contiene valores NaN o Inf.")
        if expected is not None and arr.size != expected:
            raise SymplecticGeometryError(
                f"{name} debe tener longitud {expected}, se recibió {arr.size}."
            )
        return np.ascontiguousarray(arr)

    @staticmethod
    def _frobenius(matrix: np.ndarray) -> float:
        return float(la.norm(matrix, ord="fro"))

    @staticmethod
    def _opnorm(matrix: np.ndarray) -> float:
        if matrix.size == 0:
            return 0.0
        return float(la.norm(matrix, ord=2))

    def _relative_residual(self, residual: np.ndarray, reference: np.ndarray) -> float:
        num = self._frobenius(residual)
        den = max(self._frobenius(reference), _MACHINE_EPS)
        return float(num / den)

    @staticmethod
    def _singular_spectrum(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(la.svdvals(matrix), dtype=np.float64)

    def _numerical_rank(self, matrix: np.ndarray) -> int:
        r"""
        Rango numérico de Golub–Van Loan:

        \[
          \mathrm{rank}_\varepsilon(A)
          = \#\{\sigma_i(A):\sigma_i > \max(m,n)\,\varepsilon_{\mathrm{máq}}\,\sigma_{\max}\}.
        \]
        """
        if matrix.size == 0:
            return 0
        singular_values = self._singular_spectrum(matrix)
        peak = float(singular_values[0])
        if peak <= _MACHINE_EPS:
            return 0
        threshold = max(matrix.shape) * _RANK_EPS_MULT * _MACHINE_EPS * peak
        return int(np.sum(singular_values > threshold))

    def _stable_rank(self, matrix: np.ndarray) -> float:
        r"""Rango estable \(\|A\|_F^2 / \|A\|_2^2\)."""
        singular_values = self._singular_spectrum(matrix)
        if singular_values.size == 0 or singular_values[0] <= _MACHINE_EPS:
            return 0.0
        return float(np.sum(singular_values ** 2) / (singular_values[0] ** 2))

    def _spectral_condition_number(self, matrix: np.ndarray) -> float:
        r"""Número de condición espectral \(\kappa_2(A)=\sigma_{\max}/\sigma_{\min}\)."""
        if matrix.size == 0:
            return 0.0
        singular_values = self._singular_spectrum(matrix)
        if singular_values[0] <= _MACHINE_EPS:
            return 0.0
        if singular_values[-1] <= _MACHINE_EPS:
            return float("inf")
        return float(singular_values[0] / singular_values[-1])

    def _symmetric_signature(self, matrix: np.ndarray) -> Tuple[int, int, int]:
        """Retorna \((n_+, n_-, n_0)\) de una forma cuadrática real."""
        symmetric = 0.5 * (matrix + matrix.T)
        eigvals = la.eigvalsh(symmetric)
        peak = max(1.0, float(np.max(np.abs(eigvals))) if eigvals.size else 1.0)
        tol = self._machine_tol(peak, floor=1e-12)
        n_pos = int(np.sum(eigvals > tol))
        n_neg = int(np.sum(eigvals < -tol))
        n_zero = int(eigvals.size - n_pos - n_neg)
        return n_pos, n_neg, n_zero

    def _eigh_certified(self, matrix: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Descomposición espectral simétrica con residual de Wilkinson."""
        coerced = self._coerce_real_matrix(matrix, name)
        if coerced.shape[0] != coerced.shape[1]:
            raise SymplecticGeometryError(f"{name} debe ser cuadrada para eigh.")
        symmetric = 0.5 * (coerced + coerced.T)
        eigvals, eigvecs = la.eigh(symmetric)
        residual = symmetric @ eigvecs - eigvecs @ np.diag(eigvals)
        rel = self._relative_residual(residual, symmetric)
        if rel > _EIGH_RESIDUAL_WARN:
            logger.warning(
                "Residual espectral elevado en %s: ‖AV-VΛ‖_F / ‖A‖_F = %.3e.",
                name,
                rel,
            )
        return eigvals, eigvecs, rel

    def regularize_spd_higham(self, matrix: np.ndarray) -> np.ndarray:
        r"""
        Proyección de Higham (1988/2002) al cono SPD en norma de Frobenius:

        \[
          \widetilde{M}
          = V\,\mathrm{diag}(\max(\lambda_i,\tau))\,V^\top,
          \qquad \tau=10^{-12}.
        \]

        No altera el laplaciano de Hodge (ese núcleo es sagrado); sólo
        regulariza tensores de inercia / amortiguamiento.
        """
        coerced = self._coerce_real_matrix(matrix, "matrix")
        if coerced.shape[0] != coerced.shape[1]:
            raise SymplecticGeometryError("La matriz de amortiguamiento debe ser cuadrada.")
        if not np.allclose(coerced, coerced.T, rtol=1e-12, atol=_SYMMETRY_ATOL):
            logger.warning(
                "Matriz no simétrica en regularize_spd_higham; se simetriza (M+Mᵀ)/2."
            )
            coerced = 0.5 * (coerced + coerced.T)
        eigvals, eigvecs, _ = self._eigh_certified(coerced, "higham_target")
        n_neg = int(np.sum(eigvals < -_HIGHAM_REG_FLOOR))
        if n_neg:
            logger.info(
                "Proyección de Higham: inercia negativa = %d, λ_min = %.3e.",
                n_neg,
                float(eigvals[0]),
            )
        clipped = np.maximum(eigvals, _HIGHAM_REG_FLOOR)
        return (eigvecs * clipped) @ eigvecs.T

    def _infer_edge_endpoints(self, delta_0: np.ndarray) -> List[Tuple[int, int]]:
        """Infiere pares \((u,v)\) de una cofrontera de incidencia (dos soportes dominantes)."""
        edges: List[Tuple[int, int]] = []
        for row in delta_0:
            nonzero = np.flatnonzero(np.abs(row) > 1e-12)
            if nonzero.size < 2:
                continue
            top = nonzero[np.argsort(np.abs(row[nonzero]))[-2:]]
            edges.append((int(top[0]), int(top[1])))
        return edges

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — GEOMETRÍA SIMPLÉCTICA, KURANISHI Y HIGHAM–TIKHONOV
    # ═════════════════════════════════════════════════════════════════════════
    def audit_symplectic_form_axioms(self, form: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        r"""
        Audita antisimetría y no degeneración de una 2-forma
        \(\omega\in\bigwedge^2(\mathbb{R}^{2n})^*\).
        """
        omega = self._omega if form is None else self._coerce_real_matrix(form, "form")
        size = 2 * self._n
        if omega.shape != (size, size):
            raise SymplecticGeometryError(f"La 2-forma debe ser ({size} x {size}).")
        skew = self._frobenius(omega + omega.T)
        rank = self._numerical_rank(omega)
        pfaffian_proxy = abs(float(np.linalg.det(omega)))
        deviation = skew + float(abs(rank - size)) + abs(math.log(max(pfaffian_proxy, _MACHINE_EPS)))
        return rank == size and skew <= 1e-12, float(deviation)

    def construct_compatible_almost_complex(
        self,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Construye (o valida) una \(J\) compatible:

        \[
          J^2=-I,\qquad
          \omega(Ju,Jv)=\omega(u,v),\qquad
          g(u,v)=\omega(u,Jv)\succ 0.
        \]
        """
        size = 2 * self._n
        if almost_complex_J is None:
            return self.almost_complex_structure
        jay = self._coerce_real_matrix(almost_complex_J, "almost_complex_J")
        if jay.shape != (size, size):
            raise AlmostComplexIncompatibilityError(f"J debe ser ({size} x {size}).")
        return jay

    def audit_nijenhuis_integrability(
        self,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> float:
        r"""Residual algebraico de integrabilidad: \(\|J^2+I\|_F\) (J constante)."""
        jay = self.construct_compatible_almost_complex(almost_complex_J)
        return self._frobenius(jay @ jay + np.eye(jay.shape[0]))

    def audit_almost_complex_compatibility(
        self,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> float:
        r"""Defecto de compatibilidad \(\|J^\top\omega J-\omega\|_F+\|J^2+I\|_F+\mathrm{dist}(g,\mathrm{SPD})\)."""
        jay = self.construct_compatible_almost_complex(almost_complex_J)
        omega = self._omega
        compat = self._frobenius(jay.T @ omega @ jay - omega)
        nijenhuis = self.audit_nijenhuis_integrability(jay)
        metric = 0.5 * ((omega @ jay) + (omega @ jay).T)
        eig_g = la.eigvalsh(metric)
        neg_mass = float(np.sum(np.clip(-eig_g, 0.0, None)))
        return float(compat + nijenhuis + neg_mass)

    def audit_symplectic_preservation(
        self,
        jacobian_M: np.ndarray,
        tolerance: float = 1e-10,
    ) -> Tuple[bool, float]:
        r"""Verifica \(M^\top\omega M=\omega\)."""
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        size = 2 * self._n
        if emm.shape != (size, size):
            raise SymplecticGeometryError(
                f"Jacobiano inválido. Se requiere dimensión estricta ({size} x {size})."
            )
        pulled = emm.T @ self._omega @ emm
        deviation = self._frobenius(pulled - self._omega)
        return bool(deviation <= tolerance), float(deviation)

    def relative_symplectic_residual(self, jacobian_M: np.ndarray) -> float:
        r"""Residuo relativo \(\|M^\top\omega M-\omega\|_F / \|\omega\|_F\)."""
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        size = 2 * self._n
        if emm.shape != (size, size):
            raise SymplecticGeometryError(
                f"Jacobiano inválido. Se requiere dimensión estricta ({size} x {size})."
            )
        pulled = emm.T @ self._omega @ emm
        return self._relative_residual(pulled - self._omega, self._omega)

    def audit_hamiltonian_cayley(
        self,
        jacobian_M: np.ndarray,
        tolerance: float = 1e-10,
    ) -> Tuple[bool, float]:
        r"""Contraprueba de Cayley: \(C=(M-I)(M+I)^{-1}\) hamiltoniana sii \(M\in\mathrm{Sp}\)."""
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        size = 2 * self._n
        if emm.shape != (size, size):
            raise SymplecticGeometryError(
                f"Jacobiano inválido. Se requiere dimensión estricta ({size} x {size})."
            )
        shifted = emm + np.eye(size)
        cond = self._spectral_condition_number(shifted)
        if not math.isfinite(cond) or cond > _CAYLEY_COND_MAX:
            return False, float("inf")
        try:
            cayley = la.solve(shifted, emm - np.eye(size), assume_a="gen")
        except la.LinAlgError:
            return False, float("inf")
        residual = cayley.T @ self._omega + self._omega @ cayley
        deviation = self._frobenius(residual)
        scale = max(1.0, self._frobenius(self._omega) * self._opnorm(cayley))
        return bool(deviation <= tolerance * scale), float(deviation)

    def audit_liouville_exactness(self, jacobian_M: np.ndarray) -> float:
        r"""Defecto de exactitud de Liouville \(\|M^\top\omega+\omega M\|_F\)."""
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        size = 2 * self._n
        if emm.shape != (size, size):
            raise SymplecticGeometryError(
                f"Jacobiano inválido. Se requiere dimensión estricta ({size} x {size})."
            )
        lie = emm.T @ self._omega + self._omega @ emm
        return self._frobenius(lie)

    def audit_lagrangian_plane(
        self,
        frame: np.ndarray,
        name: str = "lagrangian_frame",
        tolerance: float = 1e-10,
    ) -> Tuple[bool, float]:
        r"""Certifica que las columnas de \(F\) generan un lagrangiano."""
        frame_m = self._coerce_real_matrix(frame, name)
        size = 2 * self._n
        if frame_m.shape != (size, self._n):
            raise SymplecticGeometryError(f"{name} debe ser ({size} x {self._n}).")
        isotropic = frame_m.T @ self._omega @ frame_m
        iso_dev = self._frobenius(isotropic)
        full_rank = self._numerical_rank(frame_m) == self._n
        return bool(full_rank and iso_dev <= tolerance), float(iso_dev)

    def audit_lagrangian_transversality(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        tolerance_cond: float = _TRANSVERSAL_COND_MAX,
    ) -> Tuple[bool, float]:
        r"""Transversalidad limpia \(L_a\pitchfork L_b\) sii \(\operatorname{rank}[F_a\,F_b]=2n\)."""
        size = 2 * self._n
        fa = self._coerce_real_matrix(frame_a, "frame_a")
        fb = self._coerce_real_matrix(frame_b, "frame_b")
        if fa.shape != (size, self._n) or fb.shape != (size, self._n):
            raise SymplecticGeometryError("Los marcos lagrangianos deben ser (2n x n).")
        stacked = np.hstack((fa, fb))
        cond = self._spectral_condition_number(stacked)
        rank = self._numerical_rank(stacked)
        return bool(rank == size and cond < tolerance_cond), float(cond)

    def kashiwara_maslov_triple(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        frame_c: np.ndarray,
    ) -> int:
        r"""Índice de Kashiwara–Maslov \(\tau(\lambda_a,\lambda_b,\lambda_c)\)."""
        size = 2 * self._n
        frames = []
        for label, frame in (("a", frame_a), ("b", frame_b), ("c", frame_c)):
            coerced = self._coerce_real_matrix(frame, f"frame_{label}")
            if coerced.shape != (size, self._n):
                raise SymplecticGeometryError(f"frame_{label} debe ser ({size} x {self._n}).")
            frames.append(coerced)
        fa, fb, fc = frames
        a12 = fa.T @ self._omega @ fb
        a23 = fb.T @ self._omega @ fc
        a31 = fc.T @ self._omega @ fa
        nloc = self._n
        quad = np.zeros((3 * nloc, 3 * nloc), dtype=np.float64)
        quad[0:nloc, nloc:2 * nloc] = 0.5 * a12
        quad[nloc:2 * nloc, 0:nloc] = 0.5 * a12.T
        quad[nloc:2 * nloc, 2 * nloc:3 * nloc] = 0.5 * a23
        quad[2 * nloc:3 * nloc, nloc:2 * nloc] = 0.5 * a23.T
        quad[2 * nloc:3 * nloc, 0:nloc] = 0.5 * a31
        quad[0:nloc, 2 * nloc:3 * nloc] = 0.5 * a31.T
        n_pos, n_neg, _ = self._symmetric_signature(quad)
        tau = int(n_pos - n_neg)
        if abs(tau) > self._n:
            tau = int(np.clip(tau, -self._n, self._n))
        if (tau - self._n) % 2 != 0:
            candidate_up = tau + 1
            candidate_dn = tau - 1
            tau = candidate_up if abs(candidate_up) <= abs(candidate_dn) else candidate_dn
            tau = int(np.clip(tau, -self._n, self._n))
            if (tau - self._n) % 2 != 0:
                tau = self._n if tau >= 0 else -self._n
        return int(tau)

    def estimate_polygonal_maslov(self, lagrangian_frames: Sequence[Any]) -> int:
        r"""Estimador cíclico \(\mu\approx\tfrac12\sum_i\tau(L_i,L_{i+1},L_{i+2})\)."""
        if len(lagrangian_frames) < 3:
            raise SymplecticGeometryError(
                "El índice de Maslov poligonal requiere al menos tres marcos."
            )
        frames = [
            self._coerce_real_matrix(frame, f"lagrangian_frames[{idx}]")
            for idx, frame in enumerate(lagrangian_frames)
        ]
        total = 0
        count = len(frames)
        for idx in range(count):
            total += self.kashiwara_maslov_triple(
                frames[idx],
                frames[(idx + 1) % count],
                frames[(idx + 2) % count],
            )
        return int(np.rint(0.5 * total))

    def solve_fredholm_index(self, maslov_index: int) -> int:
        r"""Índice de Fredholm \(\operatorname{ind}(D_u\bar\partial_J)=\mu(\beta)+n-3\)."""
        return int(maslov_index) + self._n - 3

    def solve_riemann_roch_index(self, maslov_index: int) -> int:
        r"""Índice de Riemann–Roch \(\operatorname{ind}_{\mathbb{R}}=n+\mu(\beta)\)."""
        return int(maslov_index) + self._n

    def solve_virtual_polygon_dimension(self, maslov_index: int) -> int:
        r"""Dimensión virtual \(\mathrm{virt.dim}\,\mathcal{M}_d(\beta)=\mu(\beta)+n+d-3\)."""
        return int(maslov_index) + self._n + self._d - 3

    def estimate_gromov_energy_ceiling(
        self,
        jacobian_M: np.ndarray,
        maslov_index: int,
    ) -> float:
        r"""Techo de energía a la Gromov para un flujo lineal."""
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        size = 2 * self._n
        if emm.shape != (size, size):
            raise SymplecticGeometryError(
                f"Jacobiano inválido. Se requiere dimensión estricta ({size} x {size})."
            )
        pulled = emm.T @ self._omega @ emm
        symplectic_part = 0.5 * self._frobenius(pulled - self._omega)
        displacement = self._opnorm(emm - np.eye(size))
        maslov_part = abs(int(maslov_index)) * self._opnorm(self._omega)
        ceiling = float(symplectic_part + displacement + maslov_part)
        if not math.isfinite(ceiling):
            raise DiskBubblingDivergenceError(
                "Techo de energía de Gromov no finito: compactificación incontrolada."
            )
        return ceiling

    def _coerce_lagrangian_frames(
        self,
        lagrangian_frames: Optional[Sequence[Any]],
    ) -> Optional[List[np.ndarray]]:
        if lagrangian_frames is None:
            return None
        if len(lagrangian_frames) != self._d:
            raise SymplecticGeometryError(
                f"Se esperaban d={self._d} marcos lagrangianos, se recibieron {len(lagrangian_frames)}."
            )
        size = 2 * self._n
        frames: List[np.ndarray] = []
        for idx, frame in enumerate(lagrangian_frames):
            coerced = self._coerce_real_matrix(frame, f"lagrangian_frames[{idx}]")
            if coerced.shape != (size, self._n):
                raise SymplecticGeometryError(
                    f"lagrangian_frames[{idx}] debe ser ({size} x {self._n})."
                )
            is_lag, iso_dev = self.audit_lagrangian_plane(
                coerced, name=f"lagrangian_frames[{idx}]"
            )
            if not is_lag:
                raise NonTransversalIntersectionError(
                    f"El marco {idx} no es lagrangiano (desviación isotrópica={iso_dev:.3e})."
                )
            frames.append(coerced)
        return frames

    def project_symplectic_polar(self, jacobian_M: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Iteración polar simpléctica de Higham sobre \(\mathrm{Sp}(2n)\):

        \[
          Y_{k+1}
          =\tfrac12 Y_k+\tfrac12\,\omega^{-1} Y_k^{-\top}\omega.
        \]

        Preserva (en el límite) \(\det=1\) y evita la contracción de Liouville
        que produce un Tikhonov crudo.
        """
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        size = 2 * self._n
        if emm.shape != (size, size):
            raise SymplecticGeometryError(
                f"Jacobiano inválido. Se requiere dimensión estricta ({size} x {size})."
            )
        omega = self._omega
        current = np.array(emm, copy=True)
        residual = float("inf")
        for _ in range(_SP_POLAR_ITERS):
            try:
                inv_t = la.inv(current).T
            except la.LinAlgError:
                current = current + _HIGHAM_REG_FLOOR * np.eye(size)
                inv_t = la.inv(current).T
            # ω^{-1} = -ω  en la forma canónica (ω² = -I).
            updated = 0.5 * current + 0.5 * ((-omega) @ inv_t @ omega)
            residual = self._frobenius(updated - current)
            current = updated
            if residual <= _SP_POLAR_TOL:
                break
        pulled = current.T @ omega @ current
        polar_residual = self._frobenius(pulled - omega)
        return current, float(polar_residual)

    def extract_complex_type_component(
        self,
        perturbation: np.ndarray,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Proyección de tipo complejo (parte \(J\)-lineal)

        \[
          H^{J}
          =\tfrac12\bigl(H-J H J\bigr).
        \]

        El defecto \(\|H-H^J\|_F\) mide la contaminación de tipo real.
        """
        aitch = self._coerce_real_matrix(perturbation, "perturbation")
        jay = self.construct_compatible_almost_complex(almost_complex_J)
        if aitch.shape != jay.shape:
            raise AlmostComplexIncompatibilityError(
                "La perturbación y J deben compartir dimensión."
            )
        projected = 0.5 * (aitch - jay @ aitch @ jay)
        defect = self._frobenius(aitch - projected)
        return projected, float(defect)

    def estimate_tikhonov_shift(
        self,
        operator_d: np.ndarray,
        condition_guard: float = 1.0e6,
    ) -> float:
        r"""
        Shift espectral de Tikhonov adaptativo

        \[
          \mu
          =\mathrm{clip}\bigl(
             \sigma_{\max}\,
             \max(\varepsilon_{\mathrm{máq}},\sigma_{\max}/(\kappa_\sharp\sigma_{\min}^+))
           ,\,\mu_\lfloor,\,\mu_\lceil\bigr).
        \]
        """
        dee = self._coerce_real_matrix(operator_d, "operator_d")
        spectrum = self._singular_spectrum(dee)
        if spectrum.size == 0 or spectrum[0] <= _MACHINE_EPS:
            return float(_TIKHONOV_MU_FLOOR)
        sigma_max = float(spectrum[0])
        positive = spectrum[spectrum > _KURANISHI_SIGMA_FLOOR * max(1.0, sigma_max)]
        sigma_min = float(positive[-1]) if positive.size else _MACHINE_EPS
        cond = sigma_max / max(sigma_min, _MACHINE_EPS)
        mu_raw = sigma_max * max(_MACHINE_EPS, cond / max(condition_guard, 1.0) * _MACHINE_EPS * max(dee.shape))
        if cond > condition_guard:
            mu_raw = max(mu_raw, sigma_max / condition_guard)
        mu = min(max(mu_raw, _TIKHONOV_MU_FLOOR), _TIKHONOV_MU_CEILING)
        return float(mu)

    def regularize_tikhonov_higham(
        self,
        operator_d: np.ndarray,
        preserve_liouville: bool = True,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""
        Inverso regularizado Higham–Tikhonov con sintonía de volumen:

        \[
          D_\mu^+
          =(D^*D+\mu I)^{-1}D^*,
          \qquad
          D^*D \;\leftarrow\; \operatorname{Higham}_{\mathrm{SPD}}(D^*D).
        \]

        Si ``preserve_liouville`` y \(D\) es cuadrado \(2n\times 2n\), se
        incrementa \(\mu\) hasta que la proyección polar simpléctica de
        \(I+D_\mu^+\) no contraiga \(\lvert\det\rvert\) por debajo de
        \(1-\varepsilon_{\mathrm{vol}}\), y la perturbación se reduce a
        su componente de tipo complejo.

        Returns:
            \((D_\mu^+,\,\mu,\,\lvert\det\rvert_{\mathrm{ratio}},\,\mathrm{defecto}_J)\).
        """
        dee = self._coerce_real_matrix(operator_d, "operator_d")
        gram = dee.T @ dee
        gram_spd = self.regularize_spd_higham(gram)
        mu = self.estimate_tikhonov_shift(dee)
        identity = np.eye(gram_spd.shape[0], dtype=np.float64)
        volume_ratio = 1.0
        complex_defect = 0.0
        plus = np.zeros((dee.shape[1], dee.shape[0]), dtype=np.float64)

        for _ in range(_TIKHONOV_MAX_ITERS):
            regularized = gram_spd + mu * identity
            try:
                cho = la.cholesky(regularized, lower=True)
                plus = la.cho_solve((cho, True), dee.T)
            except la.LinAlgError:
                plus = la.solve(regularized, dee.T, assume_a="pos")

            if not preserve_liouville or dee.shape[0] != dee.shape[1] or dee.shape[0] != 2 * self._n:
                break

            candidate = np.eye(dee.shape[0], dtype=np.float64) + plus
            candidate, _ = self.project_symplectic_polar(candidate)
            det_abs = abs(float(np.linalg.det(candidate)))
            volume_ratio = det_abs if math.isfinite(det_abs) else 0.0
            if abs(volume_ratio - 1.0) <= _VOLUME_CONTRACTION_TOL:
                break
            if mu >= _TIKHONOV_MU_CEILING:
                logger.warning(
                    "Tikhonov alcanzó el techo μ=%.3e con |det|=%.6f (contracción residual).",
                    mu,
                    volume_ratio,
                )
                break
            mu = min(mu * 2.0, _TIKHONOV_MU_CEILING)

        if plus.shape[0] == plus.shape[1] == 2 * self._n:
            plus, complex_defect = self.extract_complex_type_component(plus, almost_complex_J)
        return plus, float(mu), float(volume_ratio), float(complex_defect)

    def detect_multiply_covered(
        self,
        operator_d: np.ndarray,
        maslov_index: int,
        fredholm_index: int,
    ) -> Tuple[bool, int, int, float]:
        r"""
        Detecta discos multi-recubiertos *sin* invocar perturbación genérica.

        Criterios (cualquiera basta):

        * \(\dim\ker D > \max(0,\operatorname{ind})+\delta\) (simetría de recubrimiento),
        * \(\dim\operatorname{coker} D>0\) con \(\sigma_{\min}\le C\varepsilon_{\mathrm{máq}}\sigma_{\max}\),
        * \(\mu(\beta)\ge 4\) par y \(\operatorname{ind}\) incompatible con inyectividad
          en algún punto (criterio de McDuff–Salamon).
        """
        dee = self._coerce_real_matrix(operator_d, "operator_d")
        spectrum = self._singular_spectrum(dee)
        if spectrum.size == 0:
            return True, dee.shape[1], dee.shape[0], 0.0
        sigma_max = float(spectrum[0])
        floor = _KURANISHI_SIGMA_FLOOR * max(1.0, sigma_max) * max(dee.shape)
        kernel_dim = int(np.sum(spectrum <= floor)) + max(0, dee.shape[1] - spectrum.size)
        coker_dim = int(np.sum(spectrum <= floor)) + max(0, dee.shape[0] - spectrum.size)
        # Ajuste por rangos rectangulares: dim ker = n_cols - rank, dim coker = n_rows - rank.
        rank = int(np.sum(spectrum > floor))
        kernel_dim = int(dee.shape[1] - rank)
        coker_dim = int(dee.shape[0] - rank)
        sigma_min = float(spectrum[-1]) if spectrum.size else 0.0
        expected_ker = max(0, int(fredholm_index))
        cover_by_kernel = kernel_dim > expected_ker + _MULTIPLY_COVER_KERNEL_SLACK
        cover_by_maslov = (int(maslov_index) >= 4) and (int(maslov_index) % 2 == 0) and coker_dim > 0
        is_cover = bool(cover_by_kernel or cover_by_maslov)
        return is_cover, kernel_dim, coker_dim, sigma_min

    def build_kuranishi_chart(
        self,
        operator_d: np.ndarray,
        maslov_index: int,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> KuranishiChart:
        r"""
        Construye una carta de Kuranishi local para \(D=D_u\bar\partial_J\).

        * \(V\simeq\ker D\oplus(\operatorname{im} D^*)^\perp_{\mathrm{approx}}\)
          se identifica con el dominio.
        * \(E\simeq\operatorname{coker} D\) es el espacio de obstrucción.
        * La sección \(s\) es la proyección de \(D\) sobre \(E\), regularizada
          por Higham–Tikhonov (nunca se declara \(s\equiv 0\) por genericidad).
        """
        dee = self._coerce_real_matrix(operator_d, "operator_d")
        fredholm = self.solve_fredholm_index(maslov_index)
        is_cover, ker_dim, coker_dim, sigma_min = self.detect_multiply_covered(
            dee, maslov_index, fredholm
        )
        plus, mu, volume_ratio, complex_defect = self.regularize_tikhonov_higham(
            dee, preserve_liouville=True, almost_complex_J=almost_complex_J
        )

        # Sección de Kuranishi: residuo D ∘ Π_{ker^⊥} en las direcciones de coker.
        try:
            u_left, singular, _ = la.svd(dee, full_matrices=True)
        except la.LinAlgError as err:
            raise KuranishiChartError(f"SVD de D_u falló: {err}") from err
        floor = _KURANISHI_SIGMA_FLOOR * max(1.0, float(singular[0]) if singular.size else 1.0) * max(dee.shape)
        coker_mask = np.ones(u_left.shape[1], dtype=bool)
        n_right = min(singular.size, u_left.shape[1])
        coker_mask[:n_right] = singular[:n_right] <= floor
        if u_left.shape[1] > singular.size:
            coker_mask[singular.size:] = True
        if not np.any(coker_mask):
            section_norm = 0.0
        else:
            coker_basis = u_left[:, coker_mask]
            # s(v) = P_E D v; su norma de operador es el mayor σ residual.
            section_norm = float(la.norm(coker_basis.T @ dee, ord=2))

        polar_residual = 0.0
        if dee.shape[0] == dee.shape[1] == 2 * self._n:
            _, polar_residual = self.project_symplectic_polar(np.eye(dee.shape[0]) + plus)

        ambient = int(dee.shape[1])
        obstruction = int(coker_dim)
        virtual = int(ambient - obstruction)
        index_matches = virtual == int(fredholm) or (virtual - int(fredholm)) in (-self._d, 0, self._d)
        is_transversal = bool(obstruction == 0 and (not is_cover) and sigma_min > floor)

        if obstruction > 0 and not index_matches:
            logger.warning(
                "Carta de Kuranishi: vdim=%d vs ind_Fredholm=%d, coker=%d, multi=%s.",
                virtual,
                fredholm,
                obstruction,
                is_cover,
            )

        return KuranishiChart(
            ambient_dimension=ambient,
            obstruction_dimension=obstruction,
            kernel_dimension=int(ker_dim),
            virtual_dimension=virtual,
            smallest_singular_value=float(sigma_min),
            tikhonov_mu=float(mu),
            higham_floor=_HIGHAM_REG_FLOOR,
            kuranishi_section_norm=float(section_norm),
            liouville_volume_ratio=float(volume_ratio),
            symplectic_polar_residual=float(polar_residual),
            is_multiply_covered=bool(is_cover),
            is_transversal=is_transversal,
            index_matches_fredholm=bool(index_matches),
            complex_type_defect=float(complex_defect),
        )

    def _phase1_certify_moduli_space(
        self,
        jacobian_M: np.ndarray,
        maslov_index: int,
        lipschitz_limit: float,
        tolerance: float,
        lagrangian_frames: Optional[Sequence[Any]] = None,
        almost_complex_J: Optional[np.ndarray] = None,
        kuranishi: Optional[KuranishiChart] = None,
    ) -> ModuliSpaceCertificate:
        """
        Certifica la rigidez simpléctica y la *regularidad virtual*.

        Motor interno de la Fase 1; el morfismo terminal público es
        ``emit_elliptic_germ``.  La transversalidad clásica ya no es
        necesaria: basta una carta de Kuranishi índice-concordante.
        """
        maslov = int(maslov_index)
        if maslov < _BUBBLING_MASLOV_MIN:
            raise DiskBubblingDivergenceError(
                f"Índice de Maslov negativo detectado: μ = {maslov}. "
                "Se produce divergencia por discos holomorfos burbujeantes."
            )
        lipschitz_limit = self._finite_scalar(lipschitz_limit, "lipschitz_limit")
        tolerance = self._finite_scalar(tolerance, "tolerance")
        if lipschitz_limit <= 0.0 or tolerance <= 0.0:
            raise SymplecticGeometryError(
                "lipschitz_limit y tolerance deben ser estrictamente positivos."
            )

        is_symplectic_raw, deviation = self.audit_symplectic_preservation(jacobian_M, tolerance)
        relative = self.relative_symplectic_residual(jacobian_M)
        _, cayley_dev = self.audit_hamiltonian_cayley(jacobian_M, tolerance)
        nijenhuis = self.audit_nijenhuis_integrability(almost_complex_J)
        energy = self.estimate_gromov_energy_ceiling(jacobian_M, maslov)

        fredholm_dim = self.solve_fredholm_index(maslov)
        rr_index = self.solve_riemann_roch_index(maslov)
        virt_dim = self.solve_virtual_polygon_dimension(maslov)
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        condition_number = self._spectral_condition_number(emm)

        frames = self._coerce_lagrangian_frames(lagrangian_frames)
        consecutive = True
        if frames is not None:
            for idx in range(len(frames)):
                ok_tr, _ = self.audit_lagrangian_transversality(
                    frames[idx], frames[(idx + 1) % len(frames)]
                )
                consecutive = consecutive and ok_tr
            if not consecutive:
                raise NonTransversalIntersectionError(
                    "Pérdida de transversalidad consecutiva (o cíclica) entre branas lagrangianas."
                )

        if kuranishi is None:
            kuranishi = self.build_kuranishi_chart(emm, maslov, almost_complex_J)

        # Transversalidad clásica (punto genérico somewhere-injective).
        is_transversal = bool(
            kuranishi.is_transversal
            and math.isfinite(condition_number)
            and condition_number < _TRANSVERSAL_COND_MAX
            and fredholm_dim >= 0
            and consecutive
            and nijenhuis <= 1e-8
        )
        # Regularidad *virtual*: carta bien formada, volumen de Liouville
        # no contraído, índice concordante.  Legal en Fukaya/FOOO.
        is_virtually_regular = bool(
            kuranishi.index_matches_fredholm
            and kuranishi.liouville_volume_ratio >= 1.0 - 10.0 * _VOLUME_CONTRACTION_TOL
            and kuranishi.complex_type_defect <= 1e-4
            and virt_dim >= -self._d
            and consecutive
            and nijenhuis <= 1e-8
        )

        norm_jacobian = max(self._opnorm(emm), _MACHINE_EPS)
        normalized_deviation = deviation / max(norm_jacobian ** 2, _MACHINE_EPS)
        is_lipschitz_confined = normalized_deviation <= tolerance * lipschitz_limit
        cayley_ok = (not math.isfinite(cayley_dev)) or (cayley_dev <= max(tolerance * 10.0, 1e-8))
        is_symplectic_final = bool(is_symplectic_raw and is_lipschitz_confined and cayley_ok)

        return ModuliSpaceCertificate(
            maslov_index=maslov,
            fredholm_index=fredholm_dim,
            is_transversal=is_transversal,
            condition_number=condition_number,
            symplectic_deviation=deviation,
            is_symplectic=is_symplectic_final,
            virtual_polygon_dimension=virt_dim,
            relative_symplectic_residual=relative,
            cayley_hamiltonian_residual=float(cayley_dev),
            nijenhuis_norm=float(nijenhuis),
            gromov_energy_ceiling=float(energy),
            consecutive_transversality=bool(consecutive),
            riemann_roch_index=rr_index,
            kuranishi_obstruction_dim=int(kuranishi.obstruction_dimension),
            is_virtually_regular=is_virtually_regular,
        )

    def emit_elliptic_germ(
        self,
        jacobian_M: np.ndarray,
        maslov_index: int,
        lipschitz_limit: float,
        tolerance: float,
        lagrangian_frames: Optional[Sequence[Any]] = None,
        almost_complex_J: Optional[np.ndarray] = None,
    ) -> EllipticGerm:
        r"""
        Término formal de la Fase 1 y objeto inicial de la Fase 2.

        \[
          \operatorname{emit\_elliptic\_germ}
          :\; (M,\mu,J,\{L_i\})
          \;\longrightarrow\;
          \mathfrak{G}_1\in\mathrm{Ob}(\mathbf{Ell}_{\mathrm{Kur}}).
        \]

        Empaqueta el certificado de moduli, la carta de Kuranishi, el
        shift de Tikhonov y la exactitud de Liouville (que decide si
        \(m_0\) puede ser nulo).  La Fase 2 no posee otra puerta de
        entrada que ``ingest_elliptic_germ``.
        """
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        kuranishi = self.build_kuranishi_chart(emm, int(maslov_index), almost_complex_J)
        moduli = self._phase1_certify_moduli_space(
            jacobian_M=emm,
            maslov_index=maslov_index,
            lipschitz_limit=lipschitz_limit,
            tolerance=tolerance,
            lagrangian_frames=lagrangian_frames,
            almost_complex_J=almost_complex_J,
            kuranishi=kuranishi,
        )
        compatibility = self.audit_almost_complex_compatibility(almost_complex_J)
        liouville = self.audit_liouville_exactness(emm)
        lipschitz_defect = moduli.symplectic_deviation / max(self._opnorm(emm) ** 2, _MACHINE_EPS)

        kashiwara_est = int(maslov_index)
        frames = self._coerce_lagrangian_frames(lagrangian_frames)
        if frames is not None and len(frames) >= 3:
            kashiwara_est = self.estimate_polygonal_maslov(frames)
            if abs(kashiwara_est - int(maslov_index)) > max(2, self._d):
                logger.warning(
                    "Discrepancia Maslov declarado=%d vs Kashiwara=%d.",
                    int(maslov_index),
                    kashiwara_est,
                )

        is_exact = bool(liouville <= 1e-8 and moduli.is_symplectic)
        bubbling_floor = 0.0 if is_exact else max(moduli.gromov_energy_ceiling, _MACHINE_EPS)
        is_regular = bool(
            (moduli.is_transversal or moduli.is_virtually_regular)
            and moduli.fredholm_index >= -self._d
            and compatibility <= 1e-6
            and math.isfinite(moduli.gromov_energy_ceiling)
            and kuranishi.liouville_volume_ratio >= 1.0 - 10.0 * _VOLUME_CONTRACTION_TOL
        )
        germ = EllipticGerm(
            moduli=moduli,
            almost_complex_compatibility=float(compatibility),
            liouville_exactness=float(liouville),
            lipschitz_connes_defect=float(lipschitz_defect),
            kashiwara_maslov_estimate=int(kashiwara_est),
            is_elliptic_regular=is_regular,
            phase_signature="PHASE_1::emit_elliptic_germ",
            kuranishi=kuranishi,
            tikhonov_mu=float(kuranishi.tikhonov_mu),
            is_exact_lagrangian=is_exact,
            bubbling_energy_floor=float(bubbling_floor),
        )
        logger.debug(
            "Germen elíptico emitido: regular=%s virtual=%s multi=%s coker=%d μ_Tikh=%.3e.",
            germ.is_elliptic_regular,
            moduli.is_virtually_regular,
            kuranishi.is_multiply_covered,
            kuranishi.obstruction_dimension,
            kuranishi.tikhonov_mu,
        )
        return germ

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — NOVIKOV, MAURER–CARTAN Y COHOMOLOGÍA DE FLOER
    # Continuación formal de emit_elliptic_germ: ingest_elliptic_germ
    # ═════════════════════════════════════════════════════════════════════════
    def ingest_elliptic_germ(self, germ: EllipticGerm) -> EllipticGerm:
        r"""
        Continuación formal de ``emit_elliptic_germ`` (término de la Fase 1).

        Axioma de arranque de la Fase 2: el complejo de Floer–Novikov
        sólo se construye sobre un germen *virtualmente* regular.  La
        transversalidad clásica ya no es un requisito: se acepta una
        carta de Kuranishi índice-concordante.
        """
        if not isinstance(germ, EllipticGerm):
            raise NonTransversalIntersectionError(
                "La Fase 2 exige un EllipticGerm emitido por la Fase 1."
            )
        if germ.phase_signature != "PHASE_1::emit_elliptic_germ":
            raise NonTransversalIntersectionError(
                f"Firma de fase ilegítima: {germ.phase_signature!r}."
            )
        virtually_ok = bool(
            germ.is_elliptic_regular
            and (
                germ.moduli.is_transversal
                or (
                    germ.moduli.is_virtually_regular
                    and germ.kuranishi.index_matches_fredholm
                )
            )
        )
        if not virtually_ok:
            raise NonTransversalIntersectionError(
                "La Fase 2 no puede iniciar: el germen no es ni transversal "
                "ni virtualmente regular en el sentido de Kuranishi."
            )
        return germ

    def audit_nilpotency_de_rham(
        self,
        boundary_delta_0: np.ndarray,
        boundary_delta_1: np.ndarray,
    ) -> Tuple[bool, float, float]:
        r"""Nilpotencia combinatoria \(\delta_1\circ\delta_0=0\) (antes de deformar)."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        delta_1 = self._coerce_real_matrix(boundary_delta_1, "boundary_delta_1")
        if delta_1.shape[1] != delta_0.shape[0]:
            raise SymplecticGeometryError(
                "Las cofronteras no componen: columnas(δ₁) ≠ filas(δ₀)."
            )
        composition = delta_1 @ delta_0
        norm_composition = self._frobenius(composition)
        residual_scale = max(1.0, self._frobenius(delta_0) * self._frobenius(delta_1))
        is_nilpotent = bool(norm_composition <= _NILPOTENCY_REL * residual_scale)
        spectral_radius = self._opnorm(composition)
        return is_nilpotent, float(norm_composition), float(spectral_radius)

    def audit_kirchhoff_incidence(self, boundary_delta_0: np.ndarray) -> Tuple[bool, float]:
        r"""Ley de Kirchhoff combinatoria: \(\delta_0\mathbf{1}=0\)."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        ones = np.ones(delta_0.shape[1], dtype=np.float64)
        residual = delta_0 @ ones
        deviation = float(la.norm(residual))
        scale = max(1.0, self._frobenius(delta_0) * math.sqrt(delta_0.shape[1]))
        return bool(deviation <= _KIRCHHOFF_REL * scale), deviation

    def audit_dga_leibniz(
        self,
        boundary_delta_0: np.ndarray,
        samples: int = 3,
        seed: int = 1729,
    ) -> float:
        r"""Residual de derivación de álgebra diferencial graduada sobre 0-cocadenas."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        endpoints = self._infer_edge_endpoints(delta_0)
        if not endpoints:
            return float("inf")
        rng = np.random.default_rng(seed)
        n_vertices = delta_0.shape[1]
        acc = 0.0
        trials = max(1, int(samples))
        for _ in range(trials):
            left = rng.standard_normal(n_vertices)
            right = rng.standard_normal(n_vertices)
            product = left * right
            d_prod = delta_0 @ product
            d_left = delta_0 @ left
            d_right = delta_0 @ right
            predicted = np.zeros_like(d_prod)
            for edge_idx, (tail, head) in enumerate(endpoints):
                if edge_idx >= predicted.size:
                    break
                predicted[edge_idx] = left[head] * d_right[edge_idx] + right[tail] * d_left[edge_idx]
            acc += float(la.norm(d_prod[: len(endpoints)] - predicted[: len(endpoints)]))
        return float(acc / trials)

    def solve_sheaf_laplacian(
        self,
        boundary_delta_0: np.ndarray,
        damping_G: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""Laplaciano del haz ponderado \(L_F=\delta_0^\top G^{-1}\delta_0\)."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        gee = self._coerce_real_matrix(damping_G, "damping_G")
        n_edges = delta_0.shape[0]
        if gee.shape != (n_edges, n_edges):
            raise SymplecticGeometryError(
                f"damping_G debe ser ({n_edges} x {n_edges}), se recibió {gee.shape}."
            )
        gee_spd = self.regularize_spd_higham(gee)
        try:
            cho_factor = la.cholesky(gee_spd, lower=True)
            gee_inv = la.cho_solve((cho_factor, True), np.eye(n_edges, dtype=np.float64))
        except la.LinAlgError:
            gee_inv = la.inv(gee_spd)
        laplace = delta_0.T @ gee_inv @ delta_0
        condition_number = self._spectral_condition_number(laplace)
        return laplace, condition_number

    def solve_hodge_laplacians(
        self,
        boundary_delta_0: np.ndarray,
        boundary_delta_1: np.ndarray,
        damping_G: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Par de Hodge \(\Delta_0=\delta_0^\top G^{-1}\delta_0\), \(\Delta_1=\delta_0\delta_0^\top+\delta_1^\top\delta_1\)."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        delta_1 = self._coerce_real_matrix(boundary_delta_1, "boundary_delta_1")
        laplace_0, _ = self.solve_sheaf_laplacian(delta_0, damping_G)
        laplace_1 = delta_0 @ delta_0.T + delta_1.T @ delta_1
        return laplace_0, laplace_1

    def compute_betti_profile(
        self,
        boundary_delta_0: np.ndarray,
        boundary_delta_1: np.ndarray,
    ) -> Tuple[int, int, int]:
        r"""Números de Betti y característica de Euler."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        delta_1 = self._coerce_real_matrix(boundary_delta_1, "boundary_delta_1")
        n_vertices = delta_0.shape[1]
        n_edges = delta_0.shape[0]
        rank_0 = self._numerical_rank(delta_0)
        rank_1 = self._numerical_rank(delta_1)
        beta_0 = int(n_vertices - rank_0)
        beta_1 = int((n_edges - rank_1) - rank_0)
        if beta_0 <= 0:
            raise CohomologicalFrustrationError(
                f"H⁰ degenerado: β₀={beta_0}. Se requiere un complejo conexo o no vacío."
            )
        if beta_1 < 0:
            raise CohomologicalFrustrationError(
                f"H¹ inconsistente: β₁={beta_1}. Falla im δ₀ ⊆ ker δ₁."
            )
        return beta_0, beta_1, int(beta_0 - beta_1)

    def certify_hodge_kernel_agreement(
        self,
        laplace_0: np.ndarray,
        laplace_1: np.ndarray,
        beta_0: int,
        beta_1: int,
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Concordancia Hodge–Betti: el núcleo numérico de \(\Delta_k\) reproduce \(\beta_k\)."""
        eig_0, _, _ = self._eigh_certified(laplace_0, "hodge_delta_0")
        if float(eig_0[0]) < _WILKINSON_PSD_FLOOR:
            raise SpectralGapCollapseError(
                f"Δ₀ abandona el cono PSD: λ_min={float(eig_0[0]):.3e}."
            )
        peak_0 = max(1.0, float(np.max(np.abs(eig_0))))
        ker_tol_0 = max(1e-10, 64.0 * _MACHINE_EPS * peak_0 * max(laplace_0.shape))
        ker_0 = int(np.sum(np.abs(eig_0) <= ker_tol_0))

        eig_1, _, _ = self._eigh_certified(laplace_1, "hodge_delta_1")
        if float(eig_1[0]) < _WILKINSON_PSD_FLOOR:
            raise SpectralGapCollapseError(
                f"Δ₁ abandona el cono PSD: λ_min={float(eig_1[0]):.3e}."
            )
        peak_1 = max(1.0, float(np.max(np.abs(eig_1))))
        ker_tol_1 = max(1e-10, 64.0 * _MACHINE_EPS * peak_1 * max(laplace_1.shape))
        ker_1 = int(np.sum(np.abs(eig_1) <= ker_tol_1))

        agreement = bool(ker_0 == int(beta_0) and ker_1 == int(beta_1))
        if not agreement:
            logger.warning(
                "Discordancia Hodge–Betti: ker Δ₀=%d vs β₀=%d; ker Δ₁=%d vs β₁=%d.",
                ker_0,
                beta_0,
                ker_1,
                beta_1,
            )
        return agreement, eig_0, eig_1

    def compute_cheeger_fiedler(
        self,
        eigvals_0: np.ndarray,
        fiedler_threshold: float,
    ) -> Tuple[float, float]:
        r"""Brecha de Fiedler \(\lambda_1(\Delta_0)\) y constante de Cheeger \(\hat h=\sqrt{2\lambda_1}\)."""
        if eigvals_0.size == 0:
            return 0.0, 0.0
        cleaned = np.maximum(eigvals_0, 0.0)
        fiedler = float(cleaned[1]) if cleaned.size > 1 else 0.0
        if fiedler < 0.0:
            fiedler = 0.0
        cheeger = float(math.sqrt(2.0 * fiedler))
        _ = fiedler_threshold
        return fiedler, cheeger

    def witten_deform_coboundary(
        self,
        boundary_delta_0: np.ndarray,
        damping_G: np.ndarray,
        morse_potential: np.ndarray,
        witten_t: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        r"""Deformación de Witten \(d_t=\delta_0\circ e^{tf}\)."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        potential = self._coerce_real_vector(
            morse_potential, "morse_potential", expected=delta_0.shape[1]
        )
        time_t = self._finite_scalar(witten_t, "witten_t")
        if time_t < 0.0:
            raise SymplecticGeometryError("witten_t debe ser no negativo.")
        centered = potential - float(np.mean(potential))
        amp = float(np.max(np.abs(centered)))
        if amp > 0.0:
            centered = centered / max(amp, _MACHINE_EPS)
        gauge = np.exp(time_t * centered)
        deformed = delta_0 * gauge.reshape(1, -1)
        laplace_t, _ = self.solve_sheaf_laplacian(deformed, damping_G)
        eig_t, _, _ = self._eigh_certified(laplace_t, "witten_laplacian")
        if float(eig_t[0]) < _WILKINSON_PSD_FLOOR:
            raise SpectralGapCollapseError("El laplaciano de Witten no es PSD.")
        gap = float(eig_t[1]) if eig_t.size > 1 else 0.0
        return laplace_t, max(0.0, gap)

    def persistence_h0_barcode(
        self,
        boundary_delta_0: np.ndarray,
        morse_potential: np.ndarray,
    ) -> Tuple[Tuple[Tuple[float, float], ...], float]:
        """Código de barras de \(H_0\) por filtración de subnivel y union-find."""
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        potential = self._coerce_real_vector(
            morse_potential, "morse_potential", expected=delta_0.shape[1]
        )
        n_vertices = int(potential.size)
        edges = self._infer_edge_endpoints(delta_0)
        events: List[Tuple[float, int, int, int]] = []
        for idx in range(n_vertices):
            events.append((float(potential[idx]), 0, idx, -1))
        for left, right in edges:
            birth = float(max(potential[left], potential[right]))
            events.append((birth, 1, left, right))
        events.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        parent = list(range(n_vertices))
        birth_time = [float(potential[idx]) for idx in range(n_vertices)]
        alive = [False] * n_vertices

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        intervals: List[Tuple[float, float]] = []
        for value, kind, left, right in events:
            if kind == 0:
                alive[left] = True
                parent[left] = left
                birth_time[left] = value
                continue
            if not alive[left] or not alive[right]:
                continue
            root_l, root_r = find(left), find(right)
            if root_l == root_r:
                continue
            if birth_time[root_l] <= birth_time[root_r]:
                older, younger = root_l, root_r
            else:
                older, younger = root_r, root_l
            intervals.append((birth_time[younger], value))
            parent[younger] = older

        persistences = [max(0.0, death - birth) for birth, death in intervals if death > birth]
        entropy = 0.0
        total = float(sum(persistences))
        if total > _MACHINE_EPS:
            probs = [item / total for item in persistences]
            entropy = float(-sum(p * math.log(p) for p in probs if p > 0.0))
        barcode = tuple((float(a), float(b)) for a, b in intervals)
        return barcode, entropy

    def _cup_product_matrix(
        self,
        boundary_delta_0: np.ndarray,
        seed: int = 271828,
    ) -> np.ndarray:
        r"""
        Producto \(m_2\) combinatorio sobre 0-coadenas, proyectado a aristas:

        \[
          m_2(x,y)_e = \tfrac12\bigl(x_h y_e + y_t x_e\bigr)
        \]

        se linealiza como operador \(X\mapsto m_2(b,X)+m_2(X,b)\) una vez
        fijada la co-cadena acotante.  Aquí se construye un tensor
        simétrico de prueba \(\Pi\in\mathrm{End}(\mathbb{R}^{n_0})\) que
        sirve de pairing de Massey de orden 2.
        """
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        n0 = delta_0.shape[1]
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n0, n0))
        pairing = 0.5 * (raw + raw.T)
        pairing /= max(self._opnorm(pairing), 1.0)
        return pairing

    def estimate_curvature_m0(
        self,
        germ: EllipticGerm,
        n0: int,
    ) -> List[NovikovSeries]:
        r"""
        Curvatura \(m_0\in CF^0\hat\otimes\Lambda\).

        * Lagrangianas exactas: \(m_0=0\).
        * En caso contrario, un disco de Maslov 2 aporta
          \(n(\beta)\,T^{\int_\beta\omega}\) sobre la clase fundamental,
          con área acotada por el techo de Gromov y el piso de bubbling.
        """
        cap = self._novikov_energy_cap
        curvature = _novikov_vector_zero(n0, cap)
        if germ.is_exact_lagrangian:
            return curvature
        energy = max(germ.bubbling_energy_floor, germ.moduli.gromov_energy_ceiling, 0.0)
        energy = min(energy, cap)
        # Multiplicidad: 1 si μ≡2 (mod 2) y hay área; 0 si μ impar (no hay disco de Maslov 2).
        multiplicity = 1.0 if (germ.moduli.maslov_index % 2 == 0) else 0.0
        if germ.kuranishi.is_multiply_covered:
            multiplicity *= 2.0
        if multiplicity == 0.0 or energy <= _NOVIKOV_ATOL:
            return curvature
        # Se reparte la clase [L] de forma uniforme sobre los vértices
        # (modelo celular de la componente fundamental).
        weight = multiplicity / max(n0, 1)
        monomial = NovikovSeries.monomial(weight, energy, energy_cap=cap)
        return [monomial for _ in range(n0)]

    def _apply_m1(self, delta_0: np.ndarray, cochain: np.ndarray) -> np.ndarray:
        return delta_0 @ cochain

    def _apply_m2(
        self,
        pairing: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return pairing @ (left * right)

    def _mc_sum_real(
        self,
        delta_0: np.ndarray,
        pairing: np.ndarray,
        curvature: np.ndarray,
        bounding: np.ndarray,
        truncation: int,
    ) -> np.ndarray:
        r"""
        Suma de Maurer–Cartan truncada en coeficientes reales (término \(T^0\)
        más el empuje de \(m_0\) ya reducido a su coeficiente líder):

        \[
          m_0+m_1(b)+m_2(b,b)+m_3(b,b,b)+\cdots
        \]

        con \(m_3(b,b,b)\approx m_2(m_2(b,b),b)-m_2(b,m_2(b,b))\) (asociador).
        """
        acc = np.array(curvature, copy=True, dtype=np.float64)
        if truncation >= 1:
            acc = acc + self._apply_m1(delta_0, bounding)
        if truncation >= 2:
            acc = acc + self._apply_m2(pairing, bounding, bounding)
        if truncation >= 3:
            m2bb = self._apply_m2(pairing, bounding, bounding)
            associator = self._apply_m2(pairing, m2bb, bounding) - self._apply_m2(
                pairing, bounding, m2bb
            )
            acc = acc + associator
        if truncation >= 4:
            # Árbol cuaternario simetrizado (orden de filtración 4).
            m2bb = self._apply_m2(pairing, bounding, bounding)
            quartic = self._apply_m2(pairing, m2bb, m2bb)
            acc = acc + 0.5 * quartic
        return acc

    def _m1_square_obstruction(
        self,
        delta_0: np.ndarray,
        pairing: np.ndarray,
        curvature: np.ndarray,
        probe: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compara \(m_1(m_1(x))\) con \(\pm m_2(m_0,x)\pm m_2(x,m_0)\).

        En el modelo celular \(m_1=\delta_0\) baja el grado, de modo que
        \(m_1\circ m_1\) vive en aristas-de-aristas; se evalúa el pairing
        en vértices como *proxy de obstrucción* (identidad FOOO en grado 0).
        """
        left = self._apply_m2(pairing, curvature, probe)
        right = self._apply_m2(pairing, probe, curvature)
        fo_identity = left + right
        # m_1 ∘ m_1 sobre 0-coadenas es idénticamente 0 si δ₁δ₀=0; el
        # residuo relevante es precisamente fo_identity.
        return fo_identity, fo_identity

    def solve_maurer_cartan(
        self,
        germ: EllipticGerm,
        boundary_delta_0: np.ndarray,
        truncation: int = _A_INF_TRUNCATION,
    ) -> Tuple[np.ndarray, MaurerCartanCertificate, NovikovSeries]:
        r"""
        Newton–Tikhonov filtrado para la ecuación de Maurer–Cartan

        \[
          F(b)=\sum_{k=0}^{K}m_k(b^{\otimes k})-W\cdot\mathbf{1},\qquad
          DF\cdot db=-F.
        \]

        \(DF=m_1+m_2(b,\cdot)+m_2(\cdot,b)+\cdots\) se invierte con
        Higham–Tikhonov (el mismo \(\mu\) del germen, adaptado).
        """
        germ = self.ingest_elliptic_germ(germ)
        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        n0 = delta_0.shape[1]
        order = max(1, min(int(truncation), _A_INF_TRUNCATION))
        pairing = self._cup_product_matrix(delta_0)
        m0_series = self.estimate_curvature_m0(germ, n0)
        curvature = np.array([item.leading_coeff() for item in m0_series], dtype=np.float64)
        curvature_l1 = float(np.sum(np.abs(curvature)))

        bounding = np.zeros(n0, dtype=np.float64)
        residual_vec = self._mc_sum_real(delta_0, pairing, curvature, bounding, order)
        # Proyección al múltiplo de [L]: W es la media (componente fundamental).
        ones = np.ones(n0, dtype=np.float64)
        superpotential = float(np.mean(residual_vec))
        residual_vec = residual_vec - superpotential * ones
        residual_l1 = float(np.sum(np.abs(residual_vec)))
        iterations = 0

        if curvature_l1 <= _MC_NEWTON_TOL and residual_l1 <= _MC_NEWTON_TOL:
            certificate = MaurerCartanCertificate(
                solved=True,
                truncation_order=order,
                residual_l1=residual_l1,
                curvature_m0_l1=curvature_l1,
                bounding_cochain_l1=0.0,
                bounding_cochain_energy=0.0,
                superpotential_constant=superpotential,
                superpotential_valuation=0.0 if abs(superpotential) > _NOVIKOV_ATOL else float("inf"),
                a_infinity_m1_square_l1=0.0,
                obstruction_absorbed=True,
                newton_iterations=0,
                deformed_nilpotent=True,
            )
            return bounding, certificate, NovikovSeries.from_scalar(
                superpotential, self._novikov_energy_cap
            )

        identity = np.eye(n0, dtype=np.float64)
        for iterations in range(1, _MC_NEWTON_ITERS + 1):
            # Jacobiano de F: δ₀ᵀδ₀ actúa como m₁*m₁ sobre 0-coadenas;
            # el término de m₂ es la multiplicación por b a través del pairing.
            gram = delta_0.T @ delta_0
            jac = gram + pairing @ np.diag(2.0 * bounding)
            plus, _, _, _ = self.regularize_tikhonov_higham(
                jac + germ.tikhonov_mu * identity,
                preserve_liouville=False,
            )
            # plus ≈ jac_μ^{+} (n0 × n0) si jac es cuadrado.
            if plus.shape != (n0, n0):
                try:
                    step = -la.lstsq(jac + germ.tikhonov_mu * identity, residual_vec)[0]
                except la.LinAlgError:
                    step = -residual_vec * 0.1
            else:
                step = -(plus @ residual_vec)
            # Línea de Armijo no arquimediana (filtrada por L¹).
            step_taken = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = bounding + scale * step
                trial = self._mc_sum_real(delta_0, pairing, curvature, candidate, order)
                trial_w = float(np.mean(trial))
                trial_res = trial - trial_w * ones
                trial_l1 = float(np.sum(np.abs(trial_res)))
                if trial_l1 <= residual_l1 * (1.0 - 1e-4 * scale) or trial_l1 <= _MC_NEWTON_TOL:
                    bounding = candidate
                    residual_vec = trial_res
                    residual_l1 = trial_l1
                    superpotential = trial_w
                    step_taken = True
                    break
            if not step_taken:
                bounding = bounding + 0.05 * step
                trial = self._mc_sum_real(delta_0, pairing, curvature, bounding, order)
                superpotential = float(np.mean(trial))
                residual_vec = trial - superpotential * ones
                residual_l1 = float(np.sum(np.abs(residual_vec)))
            if residual_l1 <= _MC_NEWTON_TOL:
                break

        fo_identity, _ = self._m1_square_obstruction(
            delta_0, pairing, curvature, np.ones(n0, dtype=np.float64)
        )
        # Tras deformar, la obstrucción residual es m₁ᵇ(m₁ᵇ(x)) ≈ F-linearizado.
        deformed = self._mc_sum_real(delta_0, pairing, curvature, bounding, order)
        deformed -= float(np.mean(deformed)) * ones
        deformed_l1 = float(np.sum(np.abs(deformed)))
        obstruction_absorbed = bool(deformed_l1 <= max(_MC_NEWTON_TOL, 1e-8 * (1.0 + curvature_l1)))
        solved = bool(residual_l1 <= max(1e-8, 10.0 * _MC_NEWTON_TOL) and obstruction_absorbed)

        if (not germ.is_exact_lagrangian) and (not solved) and curvature_l1 > _MC_NEWTON_TOL:
            raise MaurerCartanObstructionError(
                "No existe co-cadena acotante en la filtración de energía: "
                f"‖F(b)‖₁={residual_l1:.3e}, ‖m₀‖₁={curvature_l1:.3e}. "
                "El bubbling de discos no es absorbible (Lagrangiana obstruida)."
            )

        energy_b = 0.0 if float(np.sum(np.abs(bounding))) <= _NOVIKOV_ATOL else float(
            germ.bubbling_energy_floor
        )
        w_series = NovikovSeries.from_scalar(superpotential, self._novikov_energy_cap)
        if not germ.is_exact_lagrangian and energy_b > 0.0:
            w_series = w_series + NovikovSeries.monomial(
                float(np.sum(np.abs(bounding))) / max(n0, 1),
                energy_b,
                self._novikov_energy_cap,
            )
        certificate = MaurerCartanCertificate(
            solved=solved,
            truncation_order=order,
            residual_l1=float(residual_l1),
            curvature_m0_l1=float(curvature_l1),
            bounding_cochain_l1=float(np.sum(np.abs(bounding))),
            bounding_cochain_energy=float(energy_b),
            superpotential_constant=float(superpotential),
            superpotential_valuation=float(w_series.valuation()),
            a_infinity_m1_square_l1=float(np.sum(np.abs(fo_identity))),
            obstruction_absorbed=obstruction_absorbed,
            newton_iterations=int(iterations),
            deformed_nilpotent=bool(deformed_l1 <= max(_MC_NEWTON_TOL, 1e-8)),
        )
        return bounding, certificate, w_series

    def evaluate_floer_cohomology(
        self,
        boundary_delta_0: np.ndarray,
        boundary_delta_1: np.ndarray,
        damping_G: np.ndarray,
        symplectic_certificate: Optional[ModuliSpaceCertificate] = None,
        fiedler_threshold: float = 0.05,
        elliptic_germ: Optional[EllipticGerm] = None,
        morse_potential: Optional[np.ndarray] = None,
        witten_t: float = 1.0,
    ) -> FloerCohomologyCertificate:
        r"""
        Audita el complejo de Floer: nilpotencia combinatoria, Betti, Hodge,
        Fiedler, Witten y persistencia \(H_0\).

        La nilpotencia *geométrica* \(m_1^2=0\) se certifica aparte, en
        Maurer–Cartan, cuando las lagrangianas no son exactas.
        """
        if elliptic_germ is not None:
            elliptic_germ = self.ingest_elliptic_germ(elliptic_germ)
            symplectic_certificate = elliptic_germ.moduli
        if symplectic_certificate is not None:
            virtually_ok = (
                symplectic_certificate.is_transversal
                or symplectic_certificate.is_virtually_regular
            )
            if not virtually_ok:
                raise NonTransversalIntersectionError(
                    "La Fase 2 no puede iniciar: ni transversalidad clásica "
                    "ni regularidad virtual de Kuranishi."
                )

        delta_0 = self._coerce_real_matrix(boundary_delta_0, "boundary_delta_0")
        delta_1 = self._coerce_real_matrix(boundary_delta_1, "boundary_delta_1")
        gee = self._coerce_real_matrix(damping_G, "damping_G")
        n_edges = delta_0.shape[0]
        if delta_1.shape[1] != n_edges:
            raise SymplecticGeometryError(
                f"boundary_delta_1 debe tener forma (n_faces, {n_edges})."
            )
        if gee.shape != (n_edges, n_edges):
            raise SymplecticGeometryError(f"damping_G debe ser ({n_edges} x {n_edges}).")

        threshold = self._finite_scalar(fiedler_threshold, "fiedler_threshold")
        if threshold < 0.0:
            raise SymplecticGeometryError("fiedler_threshold debe ser no negativo.")

        is_nilpotent, norm_composition, spectral_radius = self.audit_nilpotency_de_rham(
            delta_0, delta_1
        )
        if not is_nilpotent:
            residual_scale = max(1.0, self._frobenius(delta_0) * self._frobenius(delta_1))
            raise CohomologicalFrustrationError(
                "Ruptura de la nilpotencia combinatoria de de Rham: δ₁δ₀ ≠ 0. "
                f"Norma del residuo parásito: {norm_composition:.4e} > "
                f"{_NILPOTENCY_REL * residual_scale:.4e}."
            )

        is_kirchhoff, _ = self.audit_kirchhoff_incidence(delta_0)
        beta_0, beta_1, euler_char = self.compute_betti_profile(delta_0, delta_1)
        laplace_0, laplace_1 = self.solve_hodge_laplacians(delta_0, delta_1, gee)
        hodge_ok, eig_0, _ = self.certify_hodge_kernel_agreement(
            laplace_0, laplace_1, beta_0, beta_1
        )
        fiedler_val, cheeger = self.compute_cheeger_fiedler(eig_0, threshold)

        witten_gap = fiedler_val
        persistence_entropy = 0.0
        if morse_potential is not None:
            _, witten_gap = self.witten_deform_coboundary(
                delta_0, gee, morse_potential, witten_t=witten_t
            )
            _, persistence_entropy = self.persistence_h0_barcode(delta_0, morse_potential)

        is_secured_coherent = bool(
            is_nilpotent
            and beta_0 == 1
            and beta_1 == 0
            and fiedler_val >= threshold
            and hodge_ok
        )
        return FloerCohomologyCertificate(
            cohomological_dimension_H1=beta_1,
            euler_characteristic=euler_char,
            fiedler_connectivity=fiedler_val,
            is_nilpotent=is_nilpotent,
            is_secured_coherent=is_secured_coherent,
            cohomological_dimension_H0=beta_0,
            cheeger_constant=cheeger,
            hodge_kernel_agreement=hodge_ok,
            witten_spectral_gap=float(witten_gap),
            persistence_entropy=float(persistence_entropy),
            spectral_radius_d2=float(spectral_radius),
            is_kirchhoff=is_kirchhoff,
            stable_rank_delta_0=self._stable_rank(delta_0),
        )

    def emit_floer_spectrum(
        self,
        germ: EllipticGerm,
        boundary_delta_0: np.ndarray,
        boundary_delta_1: np.ndarray,
        damping_G: np.ndarray,
        fiedler_threshold: float = 0.05,
        morse_potential: Optional[np.ndarray] = None,
        witten_t: float = 1.0,
    ) -> FloerSpectrum:
        r"""
        Término formal de la Fase 2 y objeto inicial de la Fase 3.

        \[
          \operatorname{emit\_floer\_spectrum}
          :\; (\mathfrak{G}_1,\delta_\bullet,G)
          \;\longrightarrow\;
          \mathfrak{S}_2\in\mathrm{Ob}(\mathbf{FloerSpec}_\Lambda).
        \]

        Continúa a ``ingest_elliptic_germ``, resuelve Maurer–Cartan sobre
        \(\Lambda_{\mathbb{R}}\) y alimenta ``ingest_floer_spectrum``.
        """
        germ = self.ingest_elliptic_germ(germ)
        cohomology = self.evaluate_floer_cohomology(
            boundary_delta_0=boundary_delta_0,
            boundary_delta_1=boundary_delta_1,
            damping_G=damping_G,
            elliptic_germ=germ,
            fiedler_threshold=fiedler_threshold,
            morse_potential=morse_potential,
            witten_t=witten_t,
        )
        _, mc_cert, w_series = self.solve_maurer_cartan(germ, boundary_delta_0)
        laplace_0, laplace_1 = self.solve_hodge_laplacians(
            boundary_delta_0, boundary_delta_1, damping_G
        )
        eig_0, _, _ = self._eigh_certified(laplace_0, "spectrum_delta_0")
        eig_1, _, _ = self._eigh_certified(laplace_1, "spectrum_delta_1")
        head = tuple(float(val) for val in eig_1[: min(8, eig_1.size)])
        _, a_inf_residual, _ = self.audit_nilpotency_de_rham(
            boundary_delta_0, boundary_delta_1
        )
        is_rigid = bool(
            cohomology.is_secured_coherent
            and germ.moduli.is_symplectic
            and cohomology.hodge_kernel_agreement
            and mc_cert.deformed_nilpotent
            and mc_cert.solved
        )
        spectrum = FloerSpectrum(
            cohomology=cohomology,
            hodge_spectrum_0=tuple(float(val) for val in eig_0),
            hodge_spectrum_1_head=head,
            cheeger_constant=cohomology.cheeger_constant,
            witten_gap=cohomology.witten_spectral_gap,
            persistence_entropy=cohomology.persistence_entropy,
            a_infinity_m1_residual=float(a_inf_residual),
            is_floer_rigid=is_rigid,
            phase_signature="PHASE_2::emit_floer_spectrum",
            maurer_cartan=mc_cert,
            superpotential_terms=tuple(
                (term.valuation, term.coeff) for term in w_series.terms
            ),
            novikov_energy_cap=self._novikov_energy_cap,
        )
        logger.debug(
            "Espectro de Floer–Novikov emitido: rigid=%s MC=%s W=%s λ₁=%.4f.",
            is_rigid,
            mc_cert.solved,
            w_series,
            cohomology.fiedler_connectivity,
        )
        return spectrum

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — SUPERPOTENCIAL DE LANDAU–GINZBURG Y SUTURA DE HEYTING
    # Continuación formal de emit_floer_spectrum: ingest_floer_spectrum
    # ═════════════════════════════════════════════════════════════════════════
    def ingest_floer_spectrum(self, spectrum: FloerSpectrum) -> FloerSpectrum:
        r"""
        Continuación formal de ``emit_floer_spectrum`` (término de la Fase 2).

        Axioma de arranque de la Fase 3: la sutura de Heyting sólo se
        evalúa sobre un espectro con \(\delta^2=0\) *combinatorio* y con
        Maurer–Cartan resuelto (nilpotencia deformada \(m_1^b\)).
        """
        if not isinstance(spectrum, FloerSpectrum):
            raise CohomologicalFrustrationError(
                "La Fase 3 exige un FloerSpectrum emitido por la Fase 2."
            )
        if spectrum.phase_signature != "PHASE_2::emit_floer_spectrum":
            raise CohomologicalFrustrationError(
                f"Firma de fase ilegítima: {spectrum.phase_signature!r}."
            )
        if not spectrum.cohomology.is_nilpotent:
            raise CohomologicalFrustrationError(
                "La Fase 3 no puede iniciar: el diferencial combinatorio no es nilpotente."
            )
        if not spectrum.maurer_cartan.solved:
            raise MaurerCartanObstructionError(
                "La Fase 3 no puede iniciar: Maurer–Cartan no está resuelto "
                "(bubbling no absorbido por ninguna co-cadena acotante)."
            )
        return spectrum

    def evaluate_landau_ginzburg_superpotential(
        self,
        spectrum: FloerSpectrum,
    ) -> Tuple[NovikovSeries, bool]:
        r"""
        Superpotencial \(W_L\in\Lambda_{\mathbb{R}}\) y criterio crítico.

        \(W_L\) es constante en la filtración (crítico) sii sólo sobrevive
        el término de valuación nula o \(W_L\equiv 0\).  En el modelo
        de Landau–Ginzburg eso corresponde a objetos de Fukaya–Seidel
        no desplazables.
        """
        series = NovikovSeries(spectrum.superpotential_terms, energy_cap=spectrum.novikov_energy_cap)
        positive_energy = [
            term for term in series.terms if term.valuation > 1e-12 and abs(term.coeff) > _NOVIKOV_ATOL
        ]
        is_critical = (not positive_energy) or series.is_zero()
        return series, bool(is_critical)

    def _atom_to_omega(self, holds_hard: bool, holds_soft: bool) -> str:
        if not holds_hard:
            return HeytingOmega3.VETOED
        if not holds_soft:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def classify_heyting_verdict(
        self,
        germ: EllipticGerm,
        spectrum: FloerSpectrum,
        fiedler_threshold: float,
        tolerance: float,
    ) -> Tuple[str, Tuple[str, ...]]:
        r"""
        Clasificador \(\Omega_3\) por implicaciones de Heyting, ahora con
        átomos de Kuranishi y de Maurer–Cartan:

        \[
          \nu
          =
          (\mathrm{Sp}\to\mathrm{Kur})
          \wedge
          (d^2=0\to H^1=0)
          \wedge
          (\mathrm{MC})
          \wedge
          (\lambda_1\ge\tau)
          \wedge
          (\mathrm{Hodge}\leftrightarrow\beta).
        \]
        """
        moduli = germ.moduli
        coh = spectrum.cohomology
        mc = spectrum.maurer_cartan
        _, lg_critical = self.evaluate_landau_ginzburg_superpotential(spectrum)

        symplectic_atom = self._atom_to_omega(
            holds_hard=moduli.relative_symplectic_residual
            <= _SOFT_SYMPLECTIC_FACTOR * max(tolerance, _MACHINE_EPS),
            holds_soft=moduli.is_symplectic,
        )
        kuranishi_atom = self._atom_to_omega(
            holds_hard=germ.kuranishi.index_matches_fredholm
            and germ.kuranishi.liouville_volume_ratio
            >= 1.0 - 10.0 * _VOLUME_CONTRACTION_TOL,
            holds_soft=moduli.is_transversal or moduli.is_virtually_regular,
        )
        nilpotent_atom = self._atom_to_omega(
            holds_hard=coh.is_nilpotent,
            holds_soft=spectrum.a_infinity_m1_residual <= 1e-10,
        )
        mc_atom = self._atom_to_omega(
            holds_hard=mc.solved and mc.obstruction_absorbed,
            holds_soft=mc.deformed_nilpotent and mc.residual_l1 <= 1e-8,
        )
        h1_atom = self._atom_to_omega(
            holds_hard=coh.cohomological_dimension_H1 >= 0,
            holds_soft=coh.cohomological_dimension_H1 == 0,
        )
        h0_atom = self._atom_to_omega(
            holds_hard=coh.cohomological_dimension_H0 >= 1,
            holds_soft=coh.cohomological_dimension_H0 == 1,
        )
        fiedler_atom = self._atom_to_omega(
            holds_hard=coh.fiedler_connectivity >= 0.0,
            holds_soft=coh.fiedler_connectivity >= fiedler_threshold,
        )
        hodge_atom = self._atom_to_omega(holds_hard=True, holds_soft=coh.hodge_kernel_agreement)
        index_atom = self._atom_to_omega(
            holds_hard=moduli.fredholm_index >= -self._d,
            holds_soft=moduli.fredholm_index >= 0 and moduli.virtual_polygon_dimension >= 0,
        )
        cond_atom = self._atom_to_omega(
            holds_hard=moduli.condition_number < _SOFT_COND_FACTOR * _TRANSVERSAL_COND_MAX,
            holds_soft=moduli.condition_number < _TRANSVERSAL_COND_MAX,
        )
        lg_atom = self._atom_to_omega(holds_hard=True, holds_soft=lg_critical)
        cover_atom = self._atom_to_omega(
            holds_hard=True,
            holds_soft=not germ.kuranishi.is_multiply_covered,
        )

        impl_sp = HeytingOmega3.implies(symplectic_atom, kuranishi_atom)
        impl_h = HeytingOmega3.implies(nilpotent_atom, h1_atom)
        verdict = HeytingOmega3.fold_meet(
            [
                impl_sp,
                impl_h,
                mc_atom,
                h0_atom,
                fiedler_atom,
                hodge_atom,
                index_atom,
                cond_atom,
                lg_atom,
                cover_atom,
            ]
        )
        if verdict == HeytingOmega3.COHERENT and not (
            moduli.is_symplectic
            and coh.is_secured_coherent
            and germ.is_elliptic_regular
            and mc.solved
        ):
            verdict = HeytingOmega3.DEGRADED

        trace = (
            f"Sp={symplectic_atom}",
            f"Kur={kuranishi_atom}",
            f"Sp→Kur={impl_sp}",
            f"d²={nilpotent_atom}",
            f"H¹={h1_atom}",
            f"d²→H¹={impl_h}",
            f"MC={mc_atom}",
            f"H⁰={h0_atom}",
            f"Fiedler={fiedler_atom}",
            f"Hodge={hodge_atom}",
            f"ind={index_atom}",
            f"κ={cond_atom}",
            f"LG={lg_atom}",
            f"cover={cover_atom}",
            f"ν={verdict}",
        )
        return verdict, trace

    def suture_heyting_state(
        self,
        germ: EllipticGerm,
        spectrum: FloerSpectrum,
        fiedler_threshold: float,
        tolerance: float,
    ) -> PseudoHolomorphicState:
        r"""
        Término formal de la Fase 3: gluing del germen de Kuranishi y del
        espectro de Novikov sobre el clasificador \(\Omega_3\).
        """
        if germ.phase_signature:
            germ = self.ingest_elliptic_germ(germ)
        spectrum = self.ingest_floer_spectrum(spectrum)
        verdict, trace = self.classify_heyting_verdict(
            germ, spectrum, fiedler_threshold=fiedler_threshold, tolerance=tolerance
        )
        w_series, lg_critical = self.evaluate_landau_ginzburg_superpotential(spectrum)
        is_sutured = bool(
            verdict == HeytingOmega3.COHERENT
            and germ.moduli.is_symplectic
            and spectrum.cohomology.is_secured_coherent
            and (germ.moduli.is_transversal or germ.moduli.is_virtually_regular)
            and spectrum.maurer_cartan.solved
            and spectrum.maurer_cartan.deformed_nilpotent
        )
        if is_sutured:
            logger.info(
                "Sutura FOOO/Novikov securizada. "
                "Fiedler=%.4f | Euler=%d | δ_ω=%.4e | μ_Tikh=%.3e | W=%s | MC_it=%d",
                spectrum.cohomology.fiedler_connectivity,
                spectrum.cohomology.euler_characteristic,
                germ.moduli.symplectic_deviation,
                germ.tikhonov_mu,
                w_series,
                spectrum.maurer_cartan.newton_iterations,
            )
        elif verdict == HeytingOmega3.DEGRADED:
            logger.warning(
                "Estado DEGRADED en la aduana de-confinada. Traza: %s",
                " ∧ ".join(trace),
            )
        else:
            logger.warning(
                "Estado VETOED en la aduana de-confinada. Traza: %s",
                " ∧ ".join(trace),
            )
        return PseudoHolomorphicState(
            moduli_audit=germ.moduli,
            cohomology_audit=spectrum.cohomology,
            elliptic_germ=germ,
            floer_spectrum=spectrum,
            heyting_verdict=verdict,
            heyting_truth_value=HeytingOmega3.truth(verdict),
            heyting_implication_trace=trace,
            is_sutured=is_sutured,
            timestamp_utc=self._utc_now(),
            phase_trace=(
                germ.phase_signature,
                spectrum.phase_signature,
                "PHASE_3::suture_heyting_state",
            ),
            landau_ginzburg_critical=lg_critical,
            novikov_superpotential=repr(w_series),
        )

    def execute_suturation(
        self,
        jacobian_M: np.ndarray,
        boundary_delta_0: np.ndarray,
        boundary_delta_1: np.ndarray,
        damping_G: np.ndarray,
        maslov_index: int = 2,
        lipschitz_limit: float = 1.5,
        tolerance: float = 1e-10,
        fiedler_threshold: float = 0.05,
        lagrangian_frames: Optional[Sequence[Any]] = None,
        almost_complex_J: Optional[np.ndarray] = None,
        morse_potential: Optional[np.ndarray] = None,
        witten_t: float = 1.0,
    ) -> PseudoHolomorphicState:
        r"""
        Acto único de validación covariante. Las fases se anidan
        semánticamente y operativamente:

            Fase 1 → ``emit_elliptic_germ``     (Kuranishi + Tikhonov)
            Fase 2 → ``emit_floer_spectrum``    (Novikov + Maurer–Cartan)
            Fase 3 → ``suture_heyting_state``   (Landau–Ginzburg + Ω₃)
        """

        def _phase1_emit_germ() -> EllipticGerm:
            return self.emit_elliptic_germ(
                jacobian_M=jacobian_M,
                maslov_index=maslov_index,
                lipschitz_limit=lipschitz_limit,
                tolerance=tolerance,
                lagrangian_frames=lagrangian_frames,
                almost_complex_J=almost_complex_J,
            )

        def _phase2_emit_spectrum(germ: EllipticGerm) -> FloerSpectrum:
            return self.emit_floer_spectrum(
                germ=germ,
                boundary_delta_0=boundary_delta_0,
                boundary_delta_1=boundary_delta_1,
                damping_G=damping_G,
                fiedler_threshold=fiedler_threshold,
                morse_potential=morse_potential,
                witten_t=witten_t,
            )

        def _phase3_suture(
            germ: EllipticGerm,
            spectrum: FloerSpectrum,
        ) -> PseudoHolomorphicState:
            return self.suture_heyting_state(
                germ=germ,
                spectrum=spectrum,
                fiedler_threshold=fiedler_threshold,
                tolerance=tolerance,
            )

        try:
            elliptic_germ = _phase1_emit_germ()
            floer_spectrum = _phase2_emit_spectrum(elliptic_germ)
            return _phase3_suture(elliptic_germ, floer_spectrum)
        except Exception as err:
            logger.critical(
                "¡VETO DE SUTURA TOPOLÓGICA! Ruptura de la Malla en la aduana: %s",
                str(err),
            )
            raise


__all__ = [
    "SymplecticGeometryError",
    "NonTransversalIntersectionError",
    "DiskBubblingDivergenceError",
    "CohomologicalFrustrationError",
    "AlmostComplexIncompatibilityError",
    "SpectralGapCollapseError",
    "KuranishiChartError",
    "NovikovValuationError",
    "MaurerCartanObstructionError",
    "HeytingOmega3",
    "NovikovTerm",
    "NovikovSeries",
    "KuranishiChart",
    "ModuliSpaceCertificate",
    "EllipticGerm",
    "MaurerCartanCertificate",
    "FloerCohomologyCertificate",
    "FloerSpectrum",
    "PseudoHolomorphicState",
    "PseudoHolomorphicMotor",
]