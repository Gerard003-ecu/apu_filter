# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Homotopic Tesserarios Agent (Capa 3 de Calibre Homotópico de Rham)  ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_tesserarios.py        ║
║ Versión: 3.0.0-Doctoral-Nested-Quillen-Stasheff-Cech-Gerbes-Heyting          ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y HOMOTÓPICA DE DE RHAM:
────────────────────────────────────────────────────────────────────────────────
Supervisa la consistencia homotópica no abeliana en la Capa 3 ($V_{\mathrm{STRATEGY}} \subset V_{\mathrm{TESSERARIOS}}$)
de las deliberaciones agénticas para proscribir bifurcaciones lógicas o inyecciones de nudos no triviales mediante
tres aduanas de control:

1. ADUANA DE CATEGORÍAS DE MODELOS DE QUILLEN:
   Somete la transición Jacobiana $f: X \to Y$ a la factorización functorial $f = p \circ i$,
   donde $i$ es una cofibración acíclica y $p$ es una fibración estricta disipativa.
   Evalúa síncronamente la conservación de la 2-forma simpléctica de Liouville $\Omega$:
   $$\epsilon_{\mathrm{Quillen}} = \| M^\top \Omega M - \Omega \|_F \le \tau_{\mathrm{Quillen}}$$

2. ADUANA DE ASOCIAEDROS DE STASHEFF ($A_\infty$-ÁLGEBRAS):
   Audita el tensor asociador de tercer orden $m_3$ que mide la no-asociatividad 
   de la multiplicación de APUs en el Ágora. Exige el cumplimiento de las relaciones 
   de coherencia del pentágono $K_4$ de Stasheff:
   $$\|m_3(a,b,c)\|_F \le \tau_{\mathrm{Stasheff}}$$
   Donde desviaciones por encima del umbral rompen la nilpotencia de de Rham-Floer ($d^2 \neq \mathbf{0}$).

3. ADUANA DE OBSTRUCCIÓN DE ČECH PARA GERBES NO ABELIANOS:
   Calcula la clase de obstrucción no abeliana Čech $[\alpha] \in \check{H}^2(\mathcal{U}, \mathcal{G})$
   aplicando SVD sobre la matriz de co-cadenas Čech de-confinadas para aniquilar 
   triangulaciones circulares de blanqueo o colusión en el presupuesto.

INVARIANTES CATEGÓRICOS Y DE HARDWARE PERIMETRAL:
────────────────────────────────────────────────────────────────────────────────
- Nilpotencia estricta del operador de coborde discreto simplicial: $\delta_{k+1} \circ \delta_k \equiv \mathbf{0}$.
- Preservación de la regularidad simpléctica bajo homotopías contractibles.
- Invarianza homotópica bajo deformaciones de equivalencia débil.
- Veto en el retículo de Heyting $\Omega_3 = \{\text{COHERENT}, \text{DEGRADED}, \text{VETOED}\}$ ($\top = \text{VETOED}$).
- Interrupción perimetral ESP32 en IRAM ($t_{\text{actuation}} \le 400\,\text{ns}$) activando el tiristor BT151 (Crowbar) vía GPIO14.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Agents.HomotopicTesserarios")

# =============================================================================
# CONSTANTES UNIVERSALES DE PRECISIÓN METROLÓGICA (WILKINSON / HIGHAM)
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_HARD_STASHEFF_CEILING: Final[float] = 1.0e-8
_HARD_QUILLEN_TOLERANCE: Final[float] = 1.0e-8
_HARD_GERBE_TOLERANCE: Final[float] = 1.0e-4
_HARD_SULLIVAN_TOLERANCE: Final[float] = 1.0e-8
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_NS: Final[float] = 5.0
_CROWBAR_T_MIN_NS: Final[float] = 380.0
_CROWBAR_T_MAX_NS: Final[float] = 420.0
_MIN_SINGULAR_VALUE_FLOOR: Final[float] = 1.0e-12
_PENTAGON_EINSUM_DIM_CAP: Final[int] = 24
_DEGRADATION_FACTOR: Final[float] = 0.01


# #############################################################################
#                                                                             #
#  FASE I                                                                     #
#  NÚCLEO ESPECTRAL, LIOUVILLE, QUILLEN-POLAR, HOCHSCHILD, ČECH               #
#                                                                             #
#  Objetos: jacobiano de transición, producto m₂, homotopía m₃,               #
#           1-/2-cocadena de Čech.                                            #
#  Morfismos: certificación de Ω, residual simpléctico, factorización         #
#             polar (cofibración acíclica ∘ fibración), asociador A∞,         #
#             coborde de Čech, masa de Dixmier–Douady.                        #
#  Cierre formal: assemble_homotopy_jet  →  _HomotopyJet                      #
#                 (dominio de todos los métodos de la Fase II).               #
#                                                                             #
# #############################################################################


class HeytingVerdict(Enum):
    """
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

    Orden de verdad (permiso / coherencia):
        VETOED ≤ DEGRADED ≤ COHERENT

    Operaciones:
        a ∧ b = min(a, b) ,   a ∨ b = max(a, b)
        a → b = ⊤ si a ≤ b, else b
        ¬a    = a → ⊥
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(self.value, other.value))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[token]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc


@dataclass(frozen=True, slots=True)
class _SymplecticForm:
    """
    2-forma canónica de Liouville \(\Omega\in\bigwedge^2(T^*Q)^*\).

    Axiomas certificados por ``from_dimension``:
      • n = 2q par (carta de Darboux);
      • \(\Omega^\top = -\Omega\), \(\Omega^2 = -I\) (casi-compleja);
      • \(\det\Omega = 1\) (volumen de Liouville).
    """

    matrix: np.ndarray
    half_dim: int
    frobenius_norm: float

    @classmethod
    def from_dimension(cls, n: int) -> "_SymplecticForm":
        r"""
        Forma simpléctica estándar sobre \(T^*Q\cong\mathbb{R}^{n}\):

        .. math::

            \Omega = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix},
            \qquad
            \Omega^\top=-\Omega,\quad \Omega^2=-I,\quad
            \|\Omega\|_F=\sqrt{n}.
        """
        if n <= 0 or n % 2 != 0:
            raise ValueError(
                f"Darboux requiere dimensión par positiva; recibido n={n}."
            )
        half = n // 2
        i_half = np.eye(half, dtype=np.float64)
        z_half = np.zeros((half, half), dtype=np.float64)
        omega = np.block([[z_half, i_half], [-i_half, z_half]])
        return cls(
            matrix=omega,
            half_dim=half,
            frobenius_norm=float(np.sqrt(n)),
        )

    def verify(self, atol: float = _STRUCTURE_ATOL) -> None:
        """Certifica anti-simetría, estructura casi-compleja y \(\det\Omega=1\)."""
        omega = self.matrix
        n = omega.shape[0]
        scale = max(self.frobenius_norm, 1.0)
        skew = float(la.norm(omega + omega.T, "fro"))
        ac = float(la.norm(omega @ omega + np.eye(n), "fro"))
        det_res = abs(float(np.linalg.det(omega)) - 1.0)
        if skew > atol * scale:
            raise ValueError(f"Ω no es anti-simétrica: ‖Ω+Ωᵀ‖_F={skew:.3e}")
        if ac > atol * scale:
            raise ValueError(f"Ω no es casi-compleja: ‖Ω²+I‖_F={ac:.3e}")
        if det_res > 1.0e-6 * n:
            raise ValueError(f"det(Ω)≠1: |det-1|={det_res:.3e}")


@dataclass(frozen=True, slots=True)
class _QuillenWitness:
    """Testigos de la factorización polar \(M=UP\) en la categoría de modelos."""

    symplectic_residual: float
    symplectic_residual_rel: float
    det_residual: float
    cofibration_residual: float
    fibration_we_residual: float
    polar_condition: float


@dataclass(frozen=True, slots=True)
class _HomotopyJet:
    """
    1-jet homotópico inmutable. Cierre formal de la Fase I y objeto inicial
    de la Fase II (todo Tesserario factoriza por este tipo).

    Contiene residuales *brutos* (sin veredicto). La decisión metrológica
    es un morfismo de la Fase II.
    """

    n: int
    quillen: _QuillenWitness
    stasheff_norm: float
    stasheff_associator: float
    stasheff_pentagon: float
    sullivan_commutator: float
    gerbe_cech_residual: float
    gerbe_cech_residual_rel: float
    gerbe_sv_mass: float
    input_fault: Optional[str]


class _HomotopySpectralCore:
    r"""
    Núcleo de cómputo espectral y homológico.

    Opera en \(\mathrm{M}_n(\mathbb{R})\) con norma de Hilbert–Schmidt y en
    el complejo de Hochschild / Čech asociado a las operaciones de
    deliberación. Las deflaciones siguen a Wilkinson; la polar, a Higham.
    """

    # ------------------------------------------------------------------
    # I.1  Formas, pisos de Wilkinson y residuales relativos
    # ------------------------------------------------------------------

    @staticmethod
    def frobenius(array: np.ndarray) -> float:
        return float(la.norm(np.asarray(array, dtype=np.float64), "fro"))

    @classmethod
    def relative_frobenius(cls, residual: np.ndarray, scale_of: np.ndarray) -> float:
        denom = max(cls.frobenius(scale_of), _MIN_SINGULAR_VALUE_FLOOR)
        return cls.frobenius(residual) / denom

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        """
        Piso de deflación adaptativo:
        \(\varepsilon = \max(\|A\|_F\,\epsilon_{\mathrm{mach}}\,\kappa,\;\varepsilon_{\min})\).
        """
        if matrix.size == 0:
            return _MIN_SINGULAR_VALUE_FLOOR
        fro_norm = float(la.norm(matrix, "fro"))
        return max(
            fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
            _MIN_SINGULAR_VALUE_FLOOR,
        )

    @staticmethod
    def _as_real(name: str, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/Inf.")
        if np.iscomplexobj(arr) and np.max(np.abs(np.imag(arr))) > 1.0e-14:
            raise ValueError(f"{name} no es real (parte imaginaria no nula).")
        return np.real(arr).astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # I.2  Factorización de Quillen en el modelo simpléctico
    # ------------------------------------------------------------------

    @classmethod
    def symplectic_pullback_residual(
        cls,
        jacobian: np.ndarray,
        omega: np.ndarray,
    ) -> Tuple[float, float]:
        r"""
        Obstrucción a \(M\in\mathrm{Sp}(n,\mathbb{R})\):

        .. math::

            \varepsilon = \|M^\top\Omega M-\Omega\|_F,
            \qquad
            \varepsilon_{\mathrm{rel}} = \varepsilon/\|\Omega\|_F.
        """
        pulled = jacobian.T @ omega @ jacobian
        residual = pulled - omega
        abs_res = cls.frobenius(residual)
        rel_res = abs_res / max(cls.frobenius(omega), _MIN_SINGULAR_VALUE_FLOOR)
        return abs_res, rel_res

    @classmethod
    def polar_quillen_factor(
        cls,
        jacobian: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Factorización polar (Higham) como testigo de Quillen
        \(f = p\circ i\):

        .. math::

            M = U P,\qquad
            P=\sqrt{M^\top M}\succeq 0,\quad
            U=M P^{-1}\in\mathrm{O}(n).

        Interpretación en la categoría de modelos de \(\mathrm{GL}^+(n)\):
          • \(i:\; *\to P\) es cofibración acíclica (el cono SPD es
            contráctil; \(P\simeq I\) iff \(M\) ya es isometría);
          • \(p:\; U\) es fibración; es equivalencia débil simpléctica
            iff \(U^*\Omega=\Omega\).
        """
        u_svd, svals, vt = la.svd(jacobian, full_matrices=False)
        cond = (
            float(svals[0] / svals[-1])
            if svals.size and svals[-1] > _MIN_SINGULAR_VALUE_FLOOR
            else float("inf")
        )
        p_spd = (vt.T * svals) @ vt
        u_orth = u_svd @ vt
        return u_orth, p_spd, cond

    @classmethod
    def compute_quillen_witness(
        cls,
        jacobian: np.ndarray,
        omega: np.ndarray,
    ) -> _QuillenWitness:
        """Empaqueta residual simpléctico, \(\lvert\det M-1\rvert\) y testigos polar."""
        abs_res, rel_res = cls.symplectic_pullback_residual(jacobian, omega)
        det_res = abs(float(np.linalg.det(jacobian)) - 1.0)
        u_orth, p_spd, cond = cls.polar_quillen_factor(jacobian)
        cofib = cls.frobenius(p_spd - np.eye(p_spd.shape[0]))
        fib_we, _ = cls.symplectic_pullback_residual(u_orth, omega)
        return _QuillenWitness(
            symplectic_residual=abs_res,
            symplectic_residual_rel=rel_res,
            det_residual=float(det_res),
            cofibration_residual=float(cofib),
            fibration_we_residual=float(fib_we),
            polar_condition=float(cond),
        )

    # ------------------------------------------------------------------
    # I.3  Stasheff A∞ / Hochschild / Sullivan
    # ------------------------------------------------------------------

    @classmethod
    def compute_stasheff_norm(cls, m3: np.ndarray) -> float:
        """Masa de Hilbert–Schmidt de la operación \(m_3\) (homotopía asociativa)."""
        return cls.frobenius(m3)

    @classmethod
    def hochschild_associator(cls, m2: np.ndarray) -> np.ndarray:
        r"""
        Asociador de Hochschild de un producto bilinear \(m_2:V\otimes V\to V\),
        con constantes de estructura \(m_2[a,b,c]=(e_b\cdot e_c)_a\):

        .. math::

            \alpha(x,y,z)=(xy)z-x(yz).
        """
        left = np.einsum("akd,kbc->abcd", m2, m2, optimize=True)
        right = np.einsum("abk,kcd->abcd", m2, m2, optimize=True)
        return left - right

    @classmethod
    def sullivan_commutator_residual(cls, m2: np.ndarray) -> float:
        r"""
        Fallo de conmutatividad graduada (cdga de Sullivan, grado par):

        .. math::

            \|m_2(x,y)-m_2(y,x)\|_F.
        """
        return cls.frobenius(m2 - np.swapaxes(m2, 1, 2))

    @classmethod
    def stasheff_pentagon_residual(
        cls,
        m2: np.ndarray,
        m3: np.ndarray,
    ) -> float:
        r"""
        Identidad A∞ de orden 4 (pentágono de Stasheff \(K_4\)), con \(m_1=m_4=0\):

        .. math::

            -m_2(m_3\otimes\mathrm{id})-m_2(\mathrm{id}\otimes m_3)
            +m_3(m_2\otimes\mathrm{id}^{\otimes 2})
            -m_3(\mathrm{id}\otimes m_2\otimes\mathrm{id})
            +m_3(\mathrm{id}^{\otimes 2}\otimes m_2)=0.

        Signos: convención \(\sum_{r+s+t=n}(-1)^{r+st}
        m_{r+1+t}\circ(\mathrm{id}^{\otimes r}\otimes m_s\otimes\mathrm{id}^{\otimes t})\).
        """
        n = m2.shape[0]
        if m3.ndim != 4 or m3.shape != (n, n, n, n):
            raise ValueError("El pentágono exige m₃ de forma (n,n,n,n).")

        if n > _PENTAGON_EINSUM_DIM_CAP:
            acc = 0.0
            for last in range(n):
                m3_e = m3[..., last]
                term = np.zeros((n, n, n, n), dtype=np.float64)
                term -= np.einsum("ake,kbcd->abcd", m2, m3[..., last, :], optimize=True)
                term -= np.einsum("abk,kcde->abcd", m2, m3[:, :, :, last], optimize=True)
                term += np.einsum("akde,kbc->abcd", m3[..., last], m2, optimize=True)
                term -= np.einsum("abke,kcd->abcd", m3[..., last], m2, optimize=True)
                term += np.einsum("abck,kde->abcd", m3, m2[:, :, last], optimize=True)
                acc += float(np.square(term).sum())
                del term, m3_e
            return float(np.sqrt(acc))

        term = (
            -np.einsum("ake,kbcde->abcde", m2, m3, optimize=True)
            - np.einsum("abk,kcde->abcde", m2, m3, optimize=True)
            + np.einsum("akde,kbc->abcde", m3, m2, optimize=True)
            - np.einsum("abke,kcd->abcde", m3, m2, optimize=True)
            + np.einsum("abck,kde->abcde", m3, m2, optimize=True)
        )
        return cls.frobenius(term)

    # ------------------------------------------------------------------
    # I.4  Čech / gerbes / Dixmier–Douady
    # ------------------------------------------------------------------

    @classmethod
    def cech_coboundary_1_norm2(cls, cochain: np.ndarray) -> float:
        r"""
        \(\|\delta C\|_F^2\) del coborde aditivo de una 1-cocadena cuadrada
        (vértices del cubrimiento = filas/columnas):

        .. math::

            (\delta C)_{ijk}=C_{jk}-C_{ik}+C_{ij}.

        Identidad contraída (evita el 3-tensor):

        .. math::

            \|\delta C\|_F^2
              = 3n\|C\|_F^2 - 2\|C\mathbf{1}\|_2^2
                - 2\|C^\top\mathbf{1}\|_2^2 + 2\,\mathbf{1}^\top C^2\mathbf{1}.
        """
        c = np.asarray(cochain, dtype=np.float64)
        n = c.shape[0]
        fro2 = float(np.square(c).sum())
        row = c @ np.ones(n)
        col = c.T @ np.ones(n)
        ones = np.ones(n)
        mixed = float(ones @ (c @ (c @ ones)))
        return (
            3.0 * n * fro2
            - 2.0 * float(row @ row)
            - 2.0 * float(col @ col)
            + 2.0 * mixed
        )

    @classmethod
    def cech_coboundary_2_norm(cls, cochain: np.ndarray) -> float:
        r"""
        \(\|\delta G\|_F\) de una 2-cocadena aditiva \(G_{ijk}\):

        .. math::

            (\delta G)_{ijkl}
              = G_{jkl}-G_{ikl}+G_{ijl}-G_{ijk}.
        """
        g = np.asarray(cochain, dtype=np.float64)
        n = g.shape[0]
        if n > _PENTAGON_EINSUM_DIM_CAP:
            acc = 0.0
            for i in range(n):
                slice_i = (
                    g[:, :, :]
                    - g[i, :, :][None, :, :]
                    + g[i, :, :][:, None, :]
                    - g[i, :, :][:, :, None]
                )
                # (δG)_{i j k l} = G_{jkl} - G_{ikl} + G_{ijl} - G_{ijk}
                delta = (
                    g
                    - g[i][None, :, :]
                    + np.expand_dims(g[i], axis=1)
                    - np.expand_dims(g[i], axis=2)
                )
                acc += float(np.square(delta).sum())
                del slice_i, delta
            return float(np.sqrt(acc))
        delta = (
            g[np.newaxis, :, :, :]
            - g[:, np.newaxis, :, :]
            + g[:, :, np.newaxis, :]
            - g[:, :, :, np.newaxis]
        )
        return cls.frobenius(delta)

    @classmethod
    def compute_gerbe_obstruction(
        cls,
        cech_cochain: np.ndarray,
    ) -> Tuple[float, float, float]:
        r"""
        Obstrucción de gerbe.

        • Matriz cuadrada: 1-cocadena → \(\|\delta C\|_F\) (clase en
          \(\check H^2(\mathfrak{U},\mathfrak{g})\) trivial iff coborde nulo
          cuando \(C\) ya es 1-coborde; aquí se audita el fallo de cociclo
          si se interpreta \(C\) como 1-cochain que *debería* ser un
          1-cociclo de un gerbe plano, i.e. \(\delta C=0\)).
        • 3-tensor: 2-cocadena → \(\|\delta G\|_F\), clase de Dixmier–Douady.
        • Masa singular (secundaria): \(\sum_{\sigma_i>\varepsilon}\sigma_i\).

        Retorna ``(cech_residual, cech_residual_rel, sv_mass)``.
        """
        arr = np.asarray(cech_cochain, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0, 0.0

        floor = cls.wilkinson_deflation_floor(arr.reshape(arr.shape[0], -1))
        flat = arr.reshape(arr.shape[0], -1)
        svals = la.svd(flat, compute_uv=False)
        valid = svals[svals > floor]
        sv_mass = float(np.sum(valid)) if valid.size else 0.0
        scale = max(cls.frobenius(arr), _MIN_SINGULAR_VALUE_FLOOR)

        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            residual = float(np.sqrt(max(cls.cech_coboundary_1_norm2(arr), 0.0)))
        elif arr.ndim == 3 and arr.shape[0] == arr.shape[1] == arr.shape[2]:
            residual = cls.cech_coboundary_2_norm(arr)
        else:
            residual = sv_mass

        return residual, residual / scale, sv_mass

    # ------------------------------------------------------------------
    # I.ω  ÚLTIMO MORFISMO DE LA FASE I
    #      Codominio ≡ dominio de HomotopicTesserariosAgent (Fase II)
    # ------------------------------------------------------------------

    @classmethod
    def assemble_homotopy_jet(
        cls,
        dimension_n: int,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        omega: _SymplecticForm,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> _HomotopyJet:
        r"""
        Cierre formal de la Fase I / unidad de la adjunción con la Fase II.

        Evalúa *todas* las métricas homotópicas sobre los tensores de un
        ciclo OODA y las congela en un 1-jet inmutable. No decide veredictos:
        la decisión es un morfismo de la Fase II sobre este objeto.

        El jacobiano se audita como candidato a equivalencia débil en la
        estructura de modelos simpléctica (pullback de \(\Omega\) + polar
        de Quillen). \(m_2,m_3\) alimentan el complejo A∞. La cocadena de
        Čech alimenta la clase de gerbe.
        """
        fault: Optional[str] = None
        zero_q = _QuillenWitness(
            symplectic_residual=float("inf"),
            symplectic_residual_rel=float("inf"),
            det_residual=float("inf"),
            cofibration_residual=float("inf"),
            fibration_we_residual=float("inf"),
            polar_condition=float("inf"),
        )

        try:
            if omega.matrix.shape != (dimension_n, dimension_n):
                raise ValueError("Ω incompatible con dimension_n.")
            jac = cls._as_real("jacobian_matrix", jacobian_matrix)
            if jac.shape != (dimension_n, dimension_n):
                raise ValueError(
                    f"Jacobiana de forma {jac.shape}, esperada "
                    f"({dimension_n}, {dimension_n})."
                )
            quillen = cls.compute_quillen_witness(jac, omega.matrix)
        except ValueError as exc:
            fault = f"quillen:{exc}"
            quillen = zero_q
            logger.error("Fallo al ensamblar el testigo de Quillen: %s", exc)

        try:
            m3 = cls._as_real("m3_homotopy_tensor", m3_homotopy_tensor)
            if m3.ndim not in (3, 4):
                raise ValueError(f"m₃ debe ser de orden 3 o 4; ndim={m3.ndim}.")
            expected3 = (dimension_n,) * 3
            expected4 = (dimension_n,) * 4
            if m3.shape not in (expected3, expected4):
                raise ValueError(
                    f"m₃ de forma {m3.shape}, esperada {expected3} o {expected4}."
                )
            stasheff_norm = cls.compute_stasheff_norm(m3)
        except ValueError as exc:
            fault = (fault + "|" if fault else "") + f"stasheff:{exc}"
            m3 = np.zeros((dimension_n,) * 3, dtype=np.float64)
            stasheff_norm = float("inf")
            logger.error("Fallo al ensamblar m₃: %s", exc)

        associator_res = 0.0
        pentagon_res = 0.0
        sullivan_res = 0.0
        if m2_product_tensor is not None:
            try:
                m2 = cls._as_real("m2_product_tensor", m2_product_tensor)
                if m2.shape != (dimension_n,) * 3:
                    raise ValueError(
                        f"m₂ de forma {m2.shape}, esperada {(dimension_n,) * 3}."
                    )
                associator_res = cls.frobenius(cls.hochschild_associator(m2))
                sullivan_res = cls.sullivan_commutator_residual(m2)
                if m3.ndim == 4 and np.isfinite(stasheff_norm):
                    pentagon_res = cls.stasheff_pentagon_residual(m2, m3)
            except ValueError as exc:
                fault = (fault + "|" if fault else "") + f"ainfty:{exc}"
                associator_res = float("inf")
                pentagon_res = float("inf")
                sullivan_res = float("inf")
                logger.error("Fallo A∞/Sullivan: %s", exc)

        try:
            cech = (
                np.zeros((0, 0), dtype=np.float64)
                if np.asarray(cech_cochain_matrix).size == 0
                else cls._as_real("cech_cochain_matrix", cech_cochain_matrix)
            )
            g_res, g_rel, g_mass = cls.compute_gerbe_obstruction(cech)
        except ValueError as exc:
            fault = (fault + "|" if fault else "") + f"gerbe:{exc}"
            g_res, g_rel, g_mass = float("inf"), float("inf"), float("inf")
            logger.error("Fallo al ensamblar la cocadena de Čech: %s", exc)

        return _HomotopyJet(
            n=dimension_n,
            quillen=quillen,
            stasheff_norm=float(stasheff_norm),
            stasheff_associator=float(associator_res),
            stasheff_pentagon=float(pentagon_res),
            sullivan_commutator=float(sullivan_res),
            gerbe_cech_residual=float(g_res),
            gerbe_cech_residual_rel=float(g_rel),
            gerbe_sv_mass=float(g_mass),
            input_fault=fault,
        )


# #############################################################################
#                                                                             #
#  FASE II                                                                    #
#  TESSERARIOS · ADUANAS DE QUILLEN / STASHEFF / GERBE                        #
#                                                                             #
#  Continuación directa del último morfismo de la Fase I:                     #
#      _HomotopyJet  ↦  HomotopicTesserariosAgent                             #
#                                                                             #
#  Cierre formal: compile_tesserarios_sheaf  →  _TesserariosSheaf             #
#                 (dominio de la Cámara de Coherencia, Fase III).             #
#                                                                             #
# #############################################################################


@dataclass(frozen=True, slots=True)
class _TesserariosAuditResult:
    """Resultado de una aduana individual: métrica, tolerancia y veredicto."""

    metric_value: float
    tolerance: float
    verdict: str
    ancilla: Dict[str, float]


@dataclass(frozen=True, slots=True)
class _TesserariosSheaf:
    """
    Gavilla de auditorías. Cierre de la Fase II y objeto inicial de la
    Fase III (la Cámara no recompute residuales: consume este morfismo
    ya certificado).
    """

    jet: _HomotopyJet
    quillen: _TesserariosAuditResult
    stasheff: _TesserariosAuditResult
    gerbe: _TesserariosAuditResult


class HomotopicTesserariosAgent:
    r"""
    Tesserarios de Integridad Homotópica (Capa 3).

    Consume el 1-jet de la Fase I y evalúa contractibilidad (Quillen),
    coherencia de asociaedros (Stasheff) y trivialidad de gerbes (Čech)
    sobre los flujos de deliberación.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        omega: Optional[_SymplecticForm] = None,
    ) -> None:
        """
        Inicio formal de la Fase II.

        La 2-forma se obtiene del constructor de la Fase I
        (``_SymplecticForm.from_dimension``) salvo que se inyecte un
        \(\Omega\) ya certificado, para no reconstruir Darboux en cada agente.
        """
        if dimension_n <= 0 or dimension_n % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio simpléctico n={dimension_n} debe ser par."
            )
        if safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser estrictamente positivo.")

        if omega is None:
            omega = _SymplecticForm.from_dimension(dimension_n)
        elif omega.matrix.shape != (dimension_n, dimension_n):
            raise ValueError("Ω inyectada incompatible con dimension_n.")
        omega.verify()

        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._omega: Final[_SymplecticForm] = omega

    def ingest_tensors(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> _HomotopyJet:
        """Reenvía los tensores del ciclo al último morfismo de la Fase I."""
        return _HomotopySpectralCore.assemble_homotopy_jet(
            dimension_n=self._n,
            jacobian_matrix=jacobian_matrix,
            m3_homotopy_tensor=m3_homotopy_tensor,
            cech_cochain_matrix=cech_cochain_matrix,
            omega=self._omega,
            m2_product_tensor=m2_product_tensor,
        )

    def _verdict_from_metric(
        self,
        metric: float,
        base_tolerance: float,
        degradation_factor: float = _DEGRADATION_FACTOR,
    ) -> Tuple[str, float]:
        """
        Clasificador metrológico de tres niveles:

          • COHERENT — \(\mathrm{métrica}\le \tau\cdot\delta\)
          • DEGRADED — \(\tau\cdot\delta < \mathrm{métrica}\le\tau\)
          • VETOED   — \(\mathrm{métrica}>\tau\)  o no finita
        """
        tol = float(base_tolerance) * self._safety_margin
        if (not np.isfinite(metric)) or metric > tol:
            return "VETOED", tol
        if metric > tol * degradation_factor:
            return "DEGRADED", tol
        return "COHERENT", tol

    def audit_quillen_factorization(
        self,
        jet: _HomotopyJet,
    ) -> _TesserariosAuditResult:
        r"""
        [TESSERARIO 1 — ADUANA DE MODELOS DE QUILLEN]

        Un jacobiano es equivalencia débil en la estructura de modelos
        simpléctica iff \(M^*\Omega=\Omega\) (y entonces \(\det M=1\)).
        Se audita el residual absoluto de pullback; los testigos polar
        (cofibración \(\|P-I\|_F\), fibración \(\|U^*\Omega-\Omega\|_F\))
        viajan como ancilla.
        """
        if jet.n != self._n:
            logger.error("Jet de dimensión %d, agente n=%d.", jet.n, self._n)
            return _TesserariosAuditResult(
                metric_value=float("inf"),
                tolerance=_HARD_QUILLEN_TOLERANCE * self._safety_margin,
                verdict="VETOED",
                ancilla={},
            )
        metric = jet.quillen.symplectic_residual
        verdict, tol = self._verdict_from_metric(metric, _HARD_QUILLEN_TOLERANCE)
        if jet.quillen.det_residual > max(tol * 10.0, 1.0e-6):
            verdict = "VETOED"
        return _TesserariosAuditResult(
            metric_value=metric,
            tolerance=tol,
            verdict=verdict,
            ancilla={
                "symplectic_residual_rel": jet.quillen.symplectic_residual_rel,
                "det_residual": jet.quillen.det_residual,
                "cofibration_residual": jet.quillen.cofibration_residual,
                "fibration_we_residual": jet.quillen.fibration_we_residual,
                "polar_condition": jet.quillen.polar_condition,
            },
        )

    def audit_stasheff_coherence_relation(
        self,
        jet: _HomotopyJet,
    ) -> _TesserariosAuditResult:
        r"""
        [TESSERARIO 2 — COHERENCIA DE STASHEFF \(A_\infty\)]

        Métrica dominante: \(\|m_3\|_F\). Si el jet porta \(m_2\), se
        elevan además el asociador de Hochschild (identidad \(K_3\)) y
        el pentágono \(K_4\); cualquiera de ellos por encima del techo
        endurece el veredicto.
        """
        metric = jet.stasheff_norm
        verdict, tol = self._verdict_from_metric(metric, _HARD_STASHEFF_CEILING)
        for extra, name in (
            (jet.stasheff_associator, "associator"),
            (jet.stasheff_pentagon, "pentagon"),
            (jet.sullivan_commutator, "sullivan"),
        ):
            extra_verdict, _ = self._verdict_from_metric(
                extra,
                _HARD_STASHEFF_CEILING
                if name != "sullivan"
                else _HARD_SULLIVAN_TOLERANCE,
            )
            if HeytingVerdict.from_token(extra_verdict).value < HeytingVerdict.from_token(verdict).value:
                verdict = extra_verdict
                logger.debug("Stasheff endurecido por %s → %s", name, verdict)
        return _TesserariosAuditResult(
            metric_value=metric,
            tolerance=tol,
            verdict=verdict,
            ancilla={
                "associator": jet.stasheff_associator,
                "pentagon": jet.stasheff_pentagon,
                "sullivan_commutator": jet.sullivan_commutator,
            },
        )

    def audit_non_abelian_gerbe_obstruction(
        self,
        jet: _HomotopyJet,
    ) -> _TesserariosAuditResult:
        r"""
        [TESSERARIO 3 — OBSTRUCCIÓN DE GERBES]

        Se exige \([\alpha]=0\) en el \(\check H^2\) (o \(\check H^3\) si
        la entrada es 2-cocadena): \(\|\delta\alpha\|_F\le\tau_{\mathrm{Gerbe}}\).
        La masa singular es un invariante secundario de Dixmier–Douady.
        """
        metric = jet.gerbe_cech_residual
        verdict, tol = self._verdict_from_metric(metric, _HARD_GERBE_TOLERANCE)
        return _TesserariosAuditResult(
            metric_value=metric,
            tolerance=tol,
            verdict=verdict,
            ancilla={
                "cech_residual_rel": jet.gerbe_cech_residual_rel,
                "sv_mass": jet.gerbe_sv_mass,
            },
        )

    # ------------------------------------------------------------------
    # II.ω  ÚLTIMO MORFISMO DE LA FASE II
    #       Codominio ≡ dominio de TesserariosCoherenceChamber (Fase III)
    # ------------------------------------------------------------------

    def compile_tesserarios_sheaf(
        self,
        jet: _HomotopyJet,
    ) -> _TesserariosSheaf:
        """
        Cierre formal de la Fase II / unidad de la adjunción con la Fase III.

        Pega las tres aduanas sobre el 1-jet. Si el jet declara
        ``input_fault``, la gavilla se marca VETOED en el eje fallido
        (ya reflejado por residuales infinitos).
        """
        if jet.input_fault:
            logger.error("Jet ensamblado con fallos de entrada: %s", jet.input_fault)
        return _TesserariosSheaf(
            jet=jet,
            quillen=self.audit_quillen_factorization(jet),
            stasheff=self.audit_stasheff_coherence_relation(jet),
            gerbe=self.audit_non_abelian_gerbe_obstruction(jet),
        )


# #############################################################################
#                                                                             #
#  FASE III                                                                   #
#  CÁMARA DE COHERENCIA · HEYTING G₃ · OODA · CROWBAR BT151                   #
#                                                                             #
#  Continuación directa del último morfismo de la Fase II:                    #
#      _TesserariosSheaf  ↦  TesserariosCoherenceChamber                      #
#                                                                             #
# #############################################################################


@dataclass
class _ThyristorCrowbar:
    r"""
    Modelo lumped del tiristor BT151 como bypass de silicio.

    Física de circuito (actuador de seguridad de Capa 3, no un exploit):
      • disparo de puerta → enganche mientras \(I_A>I_H\);
      • latencia de puerta \(\sim 400\,\mathrm{ns}\) con jitter térmico
        \(\delta t\sim\mathcal{N}(0,\sigma^2)\).
    Una vez enganchado permanece conductor durante el ciclo OODA.
    """

    latched: bool = False
    last_latency_ns: float = 0.0

    def fire(self, rng: np.random.Generator) -> float:
        jitter = float(rng.normal(0.0, _CROWBAR_JITTER_NS))
        latency = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_T_MIN_NS,
                _CROWBAR_T_MAX_NS,
            )
        )
        self.latched = True
        self.last_latency_ns = latency
        return latency

    def reset(self) -> None:
        self.latched = False
        self.last_latency_ns = 0.0


class TesserariosCoherenceChamber:
    """
    Cámara de Coherencia de la Capa 3.

    Consume la `_TesserariosSheaf` (cierre de la Fase II), calcula el
    ínfimo de Heyting de las tres aduanas y dispara el crowbar si el
    ínfimo es ⊥.
    """

    def __init__(
        self,
        agent: HomotopicTesserariosAgent,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.agent = agent
        self._crowbar = _ThyristorCrowbar()
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    @classmethod
    def assemble_from_spectral_seed(
        cls,
        dimension_n: int,
        safety_margin: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "TesserariosCoherenceChamber":
        """
        Composición functorial Fase I → Fase II → Fase III.

        `from_dimension` (I.0) produce Ω; el agente (II.0) lo consume;
        la cámara es el objeto terminal del topos de seguridad homotópico.
        """
        omega = _SymplecticForm.from_dimension(dimension_n)
        omega.verify()
        agent = HomotopicTesserariosAgent(
            dimension_n=dimension_n,
            safety_margin=safety_margin,
            omega=omega,
        )
        return cls(agent, rng=rng)

    @staticmethod
    def _heyting_meet3(a: str, b: str, c: str) -> HeytingVerdict:
        """Ínfimo de Heyting de las tres aduanas (= más restrictivo)."""
        return (
            HeytingVerdict.from_token(a)
            .meet(HeytingVerdict.from_token(b))
            .meet(HeytingVerdict.from_token(c))
        )

    def fuse_and_actuate(self, sheaf: _TesserariosSheaf) -> Dict[str, Any]:
        """
        Ciclo OODA unificado de Capa 3 sobre una gavilla ya compilada.

        1. Observar  — residuales del 1-jet (Fase I.ω).
        2. Orientar  — veredictos de las tres aduanas (Fase II.ω).
        3. Decidir   — ínfimo de Heyting \(\nu_Q\wedge\nu_S\wedge\nu_G\).
        4. Actuar    — latch del BT151 si el ínfimo es VETOED.
        """
        final_heyting = self._heyting_meet3(
            sheaf.quillen.verdict,
            sheaf.stasheff.verdict,
            sheaf.gerbe.verdict,
        )
        final_verdict = final_heyting.name

        interlock_fired = False
        latency_ns = 0.0
        if final_heyting is HeytingVerdict.VETOED:
            latency_ns = self._crowbar.fire(self._rng)
            interlock_fired = True
            logger.critical(
                "¡VETO ATÓMICO DE TESSERARIOS HOMOTÓPICOS! "
                "Anomalía no abeliana de-confinada. "
                "Disyuntor Crowbar BT151 [GPIO14] gatillado síncronamente en IRAM. "
                "Latencia de actuación física: %.2f ns. Obra civil paralizada. "
                "ε_Q=%.3e  ‖m₃‖=%.3e  ‖δα‖=%.3e",
                latency_ns,
                sheaf.quillen.metric_value,
                sheaf.stasheff.metric_value,
                sheaf.gerbe.metric_value,
            )

        return {
            "heyting_verdict": final_verdict,
            "heyting_value": final_heyting.value,
            "quillen_residual": sheaf.quillen.metric_value,
            "quillen_tolerance": sheaf.quillen.tolerance,
            "quillen_verdict": sheaf.quillen.verdict,
            "quillen_ancilla": dict(sheaf.quillen.ancilla),
            "stasheff_norm": sheaf.stasheff.metric_value,
            "stasheff_tolerance": sheaf.stasheff.tolerance,
            "stasheff_verdict": sheaf.stasheff.verdict,
            "stasheff_ancilla": dict(sheaf.stasheff.ancilla),
            "gerbe_obstruction": sheaf.gerbe.metric_value,
            "gerbe_tolerance": sheaf.gerbe.tolerance,
            "gerbe_verdict": sheaf.gerbe.verdict,
            "gerbe_ancilla": dict(sheaf.gerbe.ancilla),
            "input_fault": sheaf.jet.input_fault,
            "hardware_interlock_fired": interlock_fired,
            "hardware_crowbar_latched": self._crowbar.latched,
            "actuation_latency_ns": latency_ns,
        }

    def process_coherence_cycle(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Atajo I.ω → II.ω → III sobre tensores crudos de un ciclo."""
        jet = self.agent.ingest_tensors(
            jacobian_matrix,
            m3_homotopy_tensor,
            cech_cochain_matrix,
            m2_product_tensor=m2_product_tensor,
        )
        sheaf = self.agent.compile_tesserarios_sheaf(jet)
        return self.fuse_and_actuate(sheaf)


def execute_tesserarios_cycle(
    agent: HomotopicTesserariosAgent,
    jacobian_matrix: np.ndarray,
    m3_homotopy_tensor: np.ndarray,
    cech_cochain_matrix: np.ndarray,
    m2_product_tensor: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Fachada de compatibilidad: delega en la Cámara de la Fase III."""
    chamber = TesserariosCoherenceChamber(agent, rng=rng)
    return chamber.process_coherence_cycle(
        jacobian_matrix,
        m3_homotopy_tensor,
        cech_cochain_matrix,
        m2_product_tensor=m2_product_tensor,
    )


# Conserva el método en el agente para no romper llamadores existentes.
def _bind_agent_cycle() -> None:
    def _cycle(
        self: HomotopicTesserariosAgent,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        r"""
        Orquesta el ciclo OODA de lazo cerrado para los Tesserarios.

        Composición \(I.\omega\to II.\omega\to III\): ensambla el 1-jet,
        compila la gavilla y fusiona en Heyting. Detona el crowbar ante VETOED.
        """
        return execute_tesserarios_cycle(
            self,
            jacobian_matrix,
            m3_homotopy_tensor,
            cech_cochain_matrix,
            m2_product_tensor=m2_product_tensor,
        )

    HomotopicTesserariosAgent.execute_tesserarios_cycle = _cycle  # type: ignore[attr-defined]


_bind_agent_cycle()


__all__ = [
    "HeytingVerdict",
    "HomotopicTesserariosAgent",
    "TesserariosCoherenceChamber",
    "execute_tesserarios_cycle",
]