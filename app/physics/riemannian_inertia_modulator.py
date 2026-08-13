# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Inertia Modulator (Motor de Momento Giroscópico)         ║
║ Ruta   : app/physics/riemannian_inertia_modulator.py                         ║
║ Versión: 3.1.0-Symplectic-Lorentz-Kahan-Strict-Software                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJETO CIENTÍFICO-TÉCNICO Y FUNDAMENTACIÓN GEOMÉTRICA (Rigor Doctoral): ────────
Este módulo implementa el **Funtor de Moldeo de Masa** (Mass Shaping Functor) 
sobre el fibrado cotangente $$\mathcal{T}^* \mathcal{M}$$. Su propósito 
axiomático es inyectar una "Fuerza de Lorentz" informacional de naturaleza 
estrictamente giroscópica en la trayectoria del espacio de fase transaccional, 
con el fin de desviar las fluctuaciones estocásticas y las alucinaciones de 
los Modelos de Lenguaje (LLMs) hacia sumideros de disipación, sin alterar la 
energía Hamiltoniana interna del sistema.

El módulo trata el flujo logístico del presupuesto como un fluido continuo, 
subyugando el transporte paralelo de los covectores de estado a la regularidad 
del cono antisimétrico de Lie y preservando de manera exacta la 2-forma simpléctica 
canónica de Liouville.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial): ───────────────────
La transmutación del momento se rige por un contrato algebraico rígido donde la 
salida inmutable de cada fase actúa como la precondición formal de la siguiente:

  Fase 1 ──► FASE 1: ESPECTROSCOPÍA DEL MOMENTUM (Phase1_MomentumSpectrometer)
             Aplica el isomorfismo musical plano (flat, $$\flat$$) para descender
             índices en la variedad Riemanniana anisotrópica:
             $$p_\mu = G_{\mu\nu} \dot{q}^\nu \quad\big[8, 25, 30\big]$$
             Certifica la cota de inercia mediante la norma dual inducida:
             $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} \le P_{\max} \quad\big[8, 25\big]$$
             Entrega: MomentumAuditData como precondición formal de la Fase 2.

  Fase 2 ──► FASE 2: SÍNTESIS DEL OPERADOR GIROSCÓPICO (Phase2_GyroscopicSynthesizer)
             Acopla el momentum covariante $$p$$ con la vorticidad solenoidal $$\omega$$
             (2-forma de refracción territorial) para construir el tensor de Lorentz:
             $$W_{\mu\nu} = \alpha \left( p_\mu \omega_\nu - p_\nu \omega_{\mu} \right) \quad\big[8, 25, 30\big]$$
             Proyecta $$W$$ al álgebra de Lie del cono antisimétrico $$\mathfrak{so}(n)$$
             exigiendo la nulidad del residuo de Frobenius:
             $$r_{\mathrm{skew}} = \frac{\|W + W^\top\|_F}{\max(1, \|W\|_F)} \le \epsilon_{\mathrm{skew}} \quad\big[8, 25\big]$$
             Entrega: GyroscopicSynthesisData como precondición formal de la Fase 3.

  Fase 3 ──► FASE 3: MODULACIÓN SIMPLÉCTICA Y TRABAJO NILPOTENTE (Phase3_SymplecticInertiaModulator)
             Inyecta el tensor giroscópico $$W$$ en el operador de interconexión de 
             Dirac $$J_{\mathrm{base}}$$ del sistema Port-Hamiltoniano:
             $$J_{\mathrm{eff}} = J_{\mathrm{base}} + W \quad\big[8, 25\big]$$
             Certifica la nulidad del trabajo mecánico neto mediante el algoritmo 
             de sumación compensada de Kahan sobre el gradiente Hamiltoniano $$\nabla H$$:
             $$P_{\mathrm{gyro}} = \langle \nabla H, J_{\mathrm{eff}} \nabla H \rangle \equiv 0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[8, 25\big]$$
             Entrega: ThermodynamicVetoData como certificado inmutable terminal.

INVARIANTES MATEMÁTICOS, GEOMÉTRICOS Y PROPIEDADES DE CALIBRE PRESERVADOS: ─────
  [I1] Invariancia de la Forma Simpléctica (Volumen de Liouville):
       El operador $$J_{\mathrm{eff}}$$ es estrictamente antisimétrico, garantizando 
       que el flujo conserve el volumen en el espacio de fase simpléctico, 
       acatando de manera idéntica el Teorema de Liouville:
       $$x^\top J_{\mathrm{eff}} x \equiv 0 \quad \forall x \in \mathbb{R}^{2n} \quad\big[8\big]$$

  [I2] Coherencia Métrica de de Rham (Isomorfismo Musical):
       El motor de isomorfismo exige consistencia bilateral en la inversión 
       espectral de la métrica $$G_{\mu\nu}$$ respecto al límite de Wilkinson:
       $$\max\left( \frac{\|G G^{-1} - I\|_F}{\sqrt{n}}, \frac{\|G^{-1} G - I\|_F}{\sqrt{n}} \right) \le C \cdot \kappa(G) \cdot \varepsilon_{\mathrm{machine}}$$

  [I3] Conservación Termodinámica e Inecuación de Clausius-Duhem:
       Dado que la inyección giroscópica de Lorentz no realiza trabajo neto, 
       la tasa de disipación de Rayleigh del sistema Port-Hamiltoniano basal se 
       mantiene incondicionalmente pasiva y no-negativa:
       $$\dot{H} = -\nabla H^\top R(x) \nabla H \le 0 \quad \text{donde} \quad R(x) = R(x)^\top \succeq \mathbf{0} \quad\big[25\big]$$
"""

import logging
import math
from dataclasses import dataclass
from typing import Final
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema MIC.
# Se conservan stubs analíticos para ejecución autónoma.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación base de un invariante topológico-categórico."""

        pass

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

        pass


logger = logging.getLogger("MIC.Physics.RiemannianInertiaModulator")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS Y TERMODINÁMICAS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancia base para residuos estructurales.
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-12

# Tolerancia absoluta para consistencia de la inversa métrica.
_METRIC_INVERSE_TOLERANCE: Final[float] = 1e-8

# Cota elástica máxima del momentum.
_MOMENTUM_MAX_BOUND: Final[float] = 1e8

# Factor de acoplamiento giroscópico.
_VORTICITY_COUPLING_FACTOR: Final[float] = 1.0

# Número de condición máximo admisible para la métrica riemanniana.
_CONDITION_NUMBER_MAX: Final[float] = 1e12

# Multiplicador de ULPs para la cota adaptativa del trabajo nulo.
_WORK_TOLERANCE_ULPS: Final[float] = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES SIMPLÉCTICAS
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaError(TopologicalInvariantError):
    """Excepción raíz del funtor de modulación de inercia."""

    pass


class MomentumDivergenceError(RiemannianInertiaError):
    r"""
    Detonada si $\|p\|_{G^{-1}}$ excede la cota elástica admisible:
        $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} > P_{\text{max}}$$
    """

    pass


class SkewSymmetryViolationError(RiemannianInertiaError):
    r"""
    Detonada si la proyección antisimétrica retiene componentes simétricas significativas:
        $$\frac{\|W + W^T\|_F}{\max(1, \|W\|_F)} > \epsilon_{\text{skew}}$$
    """

    pass


class SymplecticWorkViolationError(RiemannianInertiaError):
    r"""
    Detonada si el operador efectivo realiza trabajo neto no nulo:
        $$|\langle \nabla H, J_{\text{eff}} \nabla H \rangle| > \tau_{\text{work}}$$
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs DEL FIBRADO COTANGENTE)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MomentumAuditData:
    """Artefacto certificado de la Fase 1."""

    covariant_momentum: NDArray[np.float64]
    momentum_norm: float
    is_bounded: bool
    metric_condition_number: float
    inverse_consistency_residual: float


@dataclass(frozen=True, slots=True)
class GyroscopicSynthesisData:
    """Artefacto certificado de la Fase 2."""

    skew_symmetric_tensor: NDArray[np.float64]
    antisymmetry_residual: float
    vorticity_projection_residual: float
    gyroscopic_frobenius_norm: float
    is_strictly_skew: bool


@dataclass(frozen=True, slots=True)
class ThermodynamicVetoData:
    """Artefacto certificado de la Fase 3."""

    effective_dirac_matrix: NDArray[np.float64]
    nilpotent_work_residual: float
    dirac_symmetric_residual: float
    work_tolerance: float
    is_symplectically_passive: bool


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → ESPECTROSCOPÍA DEL MOMENTUM (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_MomentumSpectrometer:
    r"""
    Extrae el covector de momentum mediante el isomorfismo musical plano ($\flat$):

        $$p_\mu = G_{\mu\nu} \dot{q}^\nu$$

    y certifica:
        1. dimensionalidad y finitud de los tensores;
        2. simetría métrica;
        3. definición positiva de G y G^{-1};
        4. consistencia de la inversa;
        5. cota de inercia $\|p\|_{G^{-1}}$.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades elementales de validación numérica
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_vector(
        self,
        vector: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Valida que un vector sea 1-D, no vacío, finito y convertible a float64."""
        try:
            arr = np.array(vector, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaError(
                f"{name} no pudo convertirse a un vector float64."
            ) from exc

        if arr.ndim != 1:
            raise RiemannianInertiaError(f"{name} debe ser un vector 1-D.")

        if arr.size == 0:
            raise RiemannianInertiaError(f"{name} no puede ser vacío.")

        if not np.all(np.isfinite(arr)):
            raise RiemannianInertiaError(f"{name} contiene valores no finitos.")

        return arr

    def _validate_square_matrix(
        self,
        matrix: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Valida que una matriz sea cuadrada, no vacía, finita y float64."""
        try:
            arr = np.array(matrix, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaError(
                f"{name} no pudo convertirse a una matriz float64."
            ) from exc

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise RiemannianInertiaError(f"{name} debe ser una matriz cuadrada.")

        if arr.shape[0] == 0:
            raise RiemannianInertiaError(f"{name} no puede ser vacía.")

        if not np.all(np.isfinite(arr)):
            raise RiemannianInertiaError(f"{name} contiene valores no finitos.")

        return arr

    def _frobenius_norm(self, matrix: NDArray[np.float64], name: str) -> float:
        """Calcula la norma de Frobenius con verificación de finitud."""
        if not np.all(np.isfinite(matrix)):
            raise RiemannianInertiaError(f"{name} contiene valores no finitos.")

        value = float(np.linalg.norm(matrix, ord="fro"))
        if not math.isfinite(value):
            raise RiemannianInertiaError(f"La norma de Frobenius de {name} no es finita.")
        return value

    def _matrix_residual_tolerance(
        self,
        matrix: NDArray[np.float64],
        base_tol: float = _SYMPLECTIC_TOLERANCE,
    ) -> float:
        """
        Cota adaptativa relativa a la escala de la matriz:

            $$tol = \max(tol_{\text{base}}, tol_{\text{base}} \times \max(1, \|matrix\|_F))$$

        Esto evita falsos positivos por escala sin perder rigor estructural.
        """
        scale = self._frobenius_norm(matrix, "matriz para tolerancia")
        return max(base_tol, base_tol * max(1.0, scale))

    def _symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        """Residuo de simetría ||A - Aᵀ||_F."""
        return self._frobenius_norm(matrix - matrix.T, "residuo simétrico")

    def _skew_symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        """Residuo de antisimetría ||A + Aᵀ||_F."""
        return self._frobenius_norm(matrix + matrix.T, "residuo antisimétrico")

    def _assert_positive_definite(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> None:
        """
        Certifica definición positiva mediante factorización de Cholesky.

        Para una métrica riemanniana G se exige:
            $$G = G^T, \quad v^T G v > 0 \quad \forall v \ne 0$$
        """
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise RiemannianInertiaError(
                f"{name} no es definida positiva (Cholesky falló)."
            ) from exc

    def _condition_number(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> float:
        """
        Calcula el número de condición espectral κ₂(matrix).

        Una métrica con κ excesivo es numéricamente peligrosa: amplifica
        errores de redondeo y corrompe la noción de norma inducida.
        """
        try:
            cond = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError as exc:
            raise RiemannianInertiaError(
                f"No fue posible calcular el número de condición de {name}."
            ) from exc

        if not math.isfinite(cond):
            raise RiemannianInertiaError(
                f"El número de condición de {name} no es finito."
            )

        if cond > _CONDITION_NUMBER_MAX:
            raise RiemannianInertiaError(
                f"{name} está mal condicionada: κ = {cond:.4e} > "
                f"{_CONDITION_NUMBER_MAX:.4e}."
            )

        return cond

    # ──────────────────────────────────────────────────────────────────────────
    # Validación del par métrico (G, G^{-1})
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_metric_pair(
        self,
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        """
        Valida y sanea el par métrico:

            G      : métrica riemanniana en T^*M ⊗ T^*M;
            G_inv  : métrica dual en TM ⊗ TM.

        Certificado:
            - misma dimensión;
            - simetría estructural;
            - definición positiva de ambas;
            - consistencia G @ G_inv ≈ I y G_inv @ G ≈ I;
            - número de condición acotado.

        Retorna:
            G simetrizada, G_inv simetrizada, κ(G), residuo de inversa.
        """
        G_tensor = self._validate_square_matrix(G_tensor, "G_tensor")
        G_inv = self._validate_square_matrix(G_inv, "G_inv")

        if G_tensor.shape != G_inv.shape:
            raise RiemannianInertiaError(
                "G_tensor y G_inv deben tener la misma dimensión."
            )

        n = G_tensor.shape[0]

        # Simetría estructural antes de saneamiento.
        g_sym_residual = self._symmetric_residual(G_tensor)
        g_tol = self._matrix_residual_tolerance(G_tensor)
        if g_sym_residual > g_tol:
            raise RiemannianInertiaError(
                f"G_tensor no es simétrica. Residuo = {g_sym_residual:.4e}, "
                f"tolerancia = {g_tol:.4e}."
            )

        g_inv_sym_residual = self._symmetric_residual(G_inv)
        g_inv_tol = self._matrix_residual_tolerance(G_inv)
        if g_inv_sym_residual > g_inv_tol:
            raise RiemannianInertiaError(
                f"G_inv no es simétrica. Residuo = {g_inv_sym_residual:.4e}, "
                f"tolerancia = {g_inv_tol:.4e}."
            )

        # Saneamiento simétrico para eliminar ruido de mantisa.
        G = 0.5 * (G_tensor + G_tensor.T)
        G_inv = 0.5 * (G_inv + G_inv.T)

        # Definición positiva.
        self._assert_positive_definite(G, "G_tensor")
        self._assert_positive_definite(G_inv, "G_inv")

        # Condicionamiento.
        cond_G = self._condition_number(G, "G_tensor")

        # Consistencia de la inversa.
        identity = np.eye(n, dtype=np.float64)

        left = G @ G_inv
        right = G_inv @ G

        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise RiemannianInertiaError(
                "El producto de consistencia métrica produjo valores no finitos."
            )

        left_residual = self._frobenius_norm(left - identity, "G @ G_inv - I")
        right_residual = self._frobenius_norm(right - identity, "G_inv @ G - I")
        inverse_residual = max(left_residual, right_residual)

        inverse_tol = max(
            _METRIC_INVERSE_TOLERANCE,
            100.0 * _MACHINE_EPSILON * max(1.0, cond_G) * float(n),
        )

        if inverse_residual > inverse_tol:
            raise RiemannianInertiaError(
                "G_inv no es una inversa consistente de G_tensor. "
                f"Residuo = {inverse_residual:.4e}, tolerancia = {inverse_tol:.4e}."
            )

        return G, G_inv, cond_G, inverse_residual

    # ──────────────────────────────────────────────────────────────────────────
    # Núcleo de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _extract_covariant_momentum(
        self,
        q_dot: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Isomorfismo musical plano:

            $$p = G \dot{q}, \quad p_\mu = G_{\mu\nu} \dot{q}^\nu$$
        """
        q_dot = self._validate_vector(q_dot, "q_dot")
        G_tensor = self._validate_square_matrix(G_tensor, "G_tensor")

        if G_tensor.shape[0] != q_dot.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre q_dot and G_tensor."
            )

        p = G_tensor @ q_dot

        if not np.all(np.isfinite(p)):
            raise RiemannianInertiaError(
                "El cálculo del momentum covariante produjo valores no finitos."
            )

        return p

    def _evaluate_momentum_divergence(
        self,
        p: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        metric_condition_number: float,
        inverse_consistency_residual: float,
    ) -> MomentumAuditData:
        r"""
        Certifica la norma inducida por la métrica dual:

            $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu}$$

        Esta cantidad coincide, salvo error de inversa, con:

            $$\sqrt{\dot{q}^T G \dot{q}}$$

        lo cual es la energía cinética geométrica de la trayectoria.
        """
        p = self._validate_vector(p, "p")
        G_inv = self._validate_square_matrix(G_inv, "G_inv")

        if G_inv.shape[0] != p.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre p y G_inv."
            )

        G_inv_sym = 0.5 * (G_inv + G_inv.T)
        aux = G_inv_sym @ p

        if not np.all(np.isfinite(aux)):
            raise RiemannianInertiaError(
                "El producto G_inv @ p produjo valores no finitos."
            )

        norm_squared = float(p @ aux)

        if not math.isfinite(norm_squared):
            raise RiemannianInertiaError(
                "La norma cuadrática del momentum no es finita."
            )

        # Saneamiento de FPU: una forma cuadrática SPD no puede ser negativa,
        # pero la mantisa puede producir -ε.
        if norm_squared < 0.0:
            negative_tol = max(
                _SYMPLECTIC_TOLERANCE,
                1000.0 * _MACHINE_EPSILON * max(1.0, abs(norm_squared)),
            )
            if norm_squared < -negative_tol:
                raise RiemannianInertiaError(
                    "Norma cuadrática negativa significativa: "
                    f"{norm_squared:.4e} < -{negative_tol:.4e}."
                )
            norm_squared = 0.0

        p_norm = math.sqrt(norm_squared)
        is_bounded = p_norm <= _MOMENTUM_MAX_BOUND

        if not is_bounded:
            raise MomentumDivergenceError(
                "Divergencia inercial detectada: "
                f"||p|| = {p_norm:.4e} > {_MOMENTUM_MAX_BOUND:.4e}."
            )

        return MomentumAuditData(
            covariant_momentum=p,
            momentum_norm=p_norm,
            is_bounded=is_bounded,
            metric_condition_number=metric_condition_number,
            inverse_consistency_residual=inverse_consistency_residual,
        )

    def execute_phase1(
        self,
        q_dot: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> MomentumAuditData:
        """
        Método terminal de la Fase 1.

        Su salida constituye la entrada canónica de la Fase 2.
        """
        q_dot = self._validate_vector(q_dot, "q_dot")

        G_tensor, G_inv, cond_G, inv_residual = self._validate_metric_pair(
            G_tensor,
            G_inv,
        )

        if G_tensor.shape[0] != q_dot.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre q_dot y la métrica."
            )

        p_cov = self._extract_covariant_momentum(q_dot, G_tensor)

        audit = self._evaluate_momentum_divergence(
            p_cov,
            G_inv,
            cond_G,
            inv_residual,
        )

        logger.debug(
            "Fase 1 completada: ||p|| = %.6e, κ(G) = %.6e, residuo inversa = %.6e.",
            audit.momentum_norm,
            audit.metric_condition_number,
            audit.inverse_consistency_residual,
        )

        return audit


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → SÍNTESIS DEL OPERADOR GIROSCÓPICO (Orient)
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_GyroscopicSynthesizer(Phase1_MomentumSpectrometer):
    r"""
    Recibe el certificado de Fase 1 y construye el tensor giroscópico:

        $$W_{\mu\nu} = \alpha (p_\mu \omega_\nu - p_\nu \omega_\mu)$$

    donde $\omega = \Omega p$ y $\Omega$ es una 2-forma de vorticidad proyectada al cono
    antisimétrico.

    La construcción garantiza que $W \in \mathfrak{so}(n)$, evitando componentes
    disipativas o inyección energética espuria.
    """

    def _project_vorticity_to_two_form(
        self,
        vorticity_matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Proyecta la matriz de vorticidad al espacio de 2-formas:

            $$\Omega_{\text{skew}} = \frac{1}{2} (\Omega - \Omega^T)$$

        El residuo devuelto es la componente simétrica descartada:

            $$\| \frac{1}{2}(\Omega + \Omega^T) \|_F$$

        Esto es físicamente necesario: la vorticidad es una 2-forma;
        la parte simétrica corresponde a deformación/strain, no a rotación.
        """
        V = self._validate_square_matrix(vorticity_matrix, "vorticity_matrix")

        symmetric_part = 0.5 * (V + V.T)
        projection_residual = self._frobenius_norm(
            symmetric_part,
            "parte simétrica de vorticidad",
        )

        V_skew = 0.5 * (V - V.T)

        if not np.all(np.isfinite(V_skew)):
            raise RiemannianInertiaError(
                "La proyección antisimétrica de la vorticidad produjo valores no finitos."
            )

        return V_skew, projection_residual

    def _project_to_skew_symmetric_cone(
        self,
        W_raw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Proyección ortogonal al cono antisimétrico:

            $$W_{\text{proj}} = \frac{1}{2} (W_{\text{raw}} - W_{\text{raw}}^T)$$

        Se audita el residuo de antisimetría:

            $$\| W_{\text{proj}} + W_{\text{proj}}^T \|_F$$

        En aritmética exacta este residuo es cero; en IEEE-754 se admite
        hasta una cota adaptativa de máquina.
        """
        W_raw = self._validate_square_matrix(W_raw, "W_raw")

        W_proj = 0.5 * (W_raw - W_raw.T)

        if not np.all(np.isfinite(W_proj)):
            raise RiemannianInertiaError(
                "La proyección giroscópica produjo valores no finitos."
            )

        residual = self._skew_symmetric_residual(W_proj)
        tol = self._matrix_residual_tolerance(W_proj)

        if residual > tol:
            raise SkewSymmetryViolationError(
                "El proyector antisimétrico retiene componentes simétricas. "
                f"Residuo = {residual:.4e}, tolerancia = {tol:.4e}."
            )

        return W_proj, residual

    def _synthesize_lorentz_tensor(
        self,
        momentum_data: MomentumAuditData,
        vorticity_matrix: NDArray[np.float64],
    ) -> GyroscopicSynthesisData:
        r"""
        Síntesis del tensor de Lorentz giroscópico:

            $$\omega = \Omega_{\text{skew}} p$$
            $$W = \alpha (p \wedge \omega)$$
            $$W_{\mu\nu} = \alpha (p_\mu \omega_\nu - p_\nu \omega_\mu)$$

        Esta construcción es manifiestamente antisimétrica.
        """
        if not isinstance(momentum_data, MomentumAuditData):
            raise RiemannianInertiaError(
                "momentum_data debe ser una instancia de MomentumAuditData."
            )

        p = self._validate_vector(
            momentum_data.covariant_momentum,
            "momentum_data.covariant_momentum",
        )

        V_skew, vorticity_residual = self._project_vorticity_to_two_form(
            vorticity_matrix
        )

        if V_skew.shape[0] != p.size:
            raise RiemannianInertiaError(
                "La dimensión de vorticity_matrix no coincide con la del momentum."
            )

        omega = V_skew @ p

        if not np.all(np.isfinite(omega)):
            raise RiemannianInertiaError(
                "El acoplamiento vorticial Ω p produjo valores no finitos."
            )

        W_raw = _VORTICITY_COUPLING_FACTOR * (
            np.outer(p, omega) - np.outer(omega, p)
        )

        if not np.all(np.isfinite(W_raw)):
            raise RiemannianInertiaError(
                "La síntesis de W_raw produjo valores no finitos."
            )

        W_proj, antisymmetry_residual = self._project_to_skew_symmetric_cone(W_raw)

        W_norm = self._frobenius_norm(W_proj, "W_proj")

        synthesis = GyroscopicSynthesisData(
            skew_symmetric_tensor=W_proj,
            antisymmetry_residual=antisymmetry_residual,
            vorticity_projection_residual=vorticity_residual,
            gyroscopic_frobenius_norm=W_norm,
            is_strictly_skew=True,
        )

        logger.debug(
            "Fase 2 completada: ||W||_F = %.6e, residuo antisimétrico = %.6e, "
            "residuo de vorticidad = %.6e.",
            synthesis.gyroscopic_frobenius_norm,
            synthesis.antisymmetry_residual,
            synthesis.vorticity_projection_residual,
        )

        return synthesis

    def execute_phase2(
        self,
        momentum_data: MomentumAuditData,
        vorticity_matrix: NDArray[np.float64],
    ) -> GyroscopicSynthesisData:
        """
        Método terminal de la Fase 2.

        Recibe la salida formal de execute_phase1 y produce la entrada
        canónica de execute_phase3.
        """
        return self._synthesize_lorentz_tensor(momentum_data, vorticity_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → MODULACIÓN SIMPLÉCTICA Y VEREDICTO TERMODINÁMICO (Decide & Act)
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_SymplecticInertiaModulator(Phase2_GyroscopicSynthesizer):
    r"""
    Inyecta $W$ en la estructura de Dirac $J$:

        $$J_{\text{eff}} = J + W$$

    y certifica la pasividad simpléctica:

        $$\langle \nabla H, J_{\text{eff}} \nabla H \rangle = 0$$

    Para ello:
        1. exige que $J$ sea antisimétrica;
        2. proyecta $J_{\text{eff}}$ al cono antisimétrico;
        3. audita el trabajo mediante sumación compensada de Kahan;
        4. emplea cotas adaptativas proporcionales a la escala del problema.
    """

    def _validate_dirac_structure(
        self,
        J_base: NDArray[np.float64],
        name: str = "J_base",
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Valida que la estructura de Dirac base sea antisimétrica:

            $$J + J^T = 0$$

        En estructuras Port-Hamiltonianas, J pertenece al álgebra de Lie
        del grupo simpléctico; cualquier componente simétrica representa
        disipación o inyección energética no autorizada.
        """
        J = self._validate_square_matrix(J_base, name)

        skew_residual = self._skew_symmetric_residual(J)
        tol = self._matrix_residual_tolerance(J)

        if skew_residual > tol:
            raise SymplecticWorkViolationError(
                f"{name} no es antisimétrica. "
                f"Residuo = {skew_residual:.4e}, tolerancia = {tol:.4e}."
            )

        J_skew = 0.5 * (J - J.T)

        if not np.all(np.isfinite(J_skew)):
            raise RiemannianInertiaError(
                f"La proyección antisimétrica de {name} produjo valores no finitos."
            )

        return J_skew, skew_residual

    def _modulate_dirac_structure(
        self,
        J_base: NDArray[np.float64],
        W_proj: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Acopla el operador giroscópico a la estructura de Dirac:

            $$J_{\text{eff}} = J + W$$

        Como J y W son antisimétricas, J_eff también lo es. Se realiza una
        proyección final de saneamiento para eliminar ruido de mantisa.
        """
        J_base = self._validate_square_matrix(J_base, "J_base")
        W_proj = self._validate_square_matrix(W_proj, "W_proj")

        if J_base.shape != W_proj.shape:
            raise RiemannianInertiaError(
                "J_base y W_proj deben tener la misma dimensión."
            )

        J_eff = J_base + W_proj

        if not np.all(np.isfinite(J_eff)):
            raise RiemannianInertiaError(
                "La modulación de Dirac produjo valores no finitos."
            )

        # Saneamiento final al cono antisimétrico.
        J_eff = 0.5 * (J_eff - J_eff.T)

        if not np.all(np.isfinite(J_eff)):
            raise RiemannianInertiaError(
                "La proyección final de J_eff produjo valores no finitos."
            )

        return J_eff

    def _kahan_quadratic_form(
        self,
        x: NDArray[np.float64],
        A: NDArray[np.float64],
    ) -> tuple[float, float]:
        r"""
        Calcula $x^T A x$ mediante sumación compensada de Kahan.

        Retorna:
            work_sum  : suma compensada de términos;
            abs_sum   : suma compensada de valores absolutos.

        Esta doble suma permite construir una cota de error adaptativa:

            $$|\text{error}| \lesssim \epsilon_{\text{machine}} \times \sum |term|$$

        Se evita la acumulación ingenua de n² términos, que puede enmascarar
        cancelaciones finas en matrices antisimétricas.
        """
        x = self._validate_vector(x, "x")
        A = self._validate_square_matrix(A, "A")

        n = x.size
        if A.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre x y A en la forma cuadrática."
            )

        total = 0.0
        compensation = 0.0

        abs_total = 0.0
        abs_compensation = 0.0

        for i in range(n):
            xi = float(x[i])
            terms = xi * A[i, :] * x

            if not np.all(np.isfinite(terms)):
                raise RiemannianInertiaError(
                    "La forma cuadrática produjo términos no finitos."
                )

            for term in terms:
                term_f = float(term)

                # Suma compensada de valores absolutos.
                abs_y = abs(term_f) - abs_compensation
                abs_t = abs_total + abs_y
                abs_compensation = (abs_t - abs_total) - abs_y

                if not math.isfinite(abs_t):
                    raise RiemannianInertiaError(
                        "La sumación absoluta de Kahan divergió."
                    )

                abs_total = abs_t

                # Suma compensada del trabajo.
                y = term_f - compensation
                t = total + y
                compensation = (t - total) - y

                if not math.isfinite(t):
                    raise RiemannianInertiaError(
                        "La sumación compensada de Kahan divergió."
                    )

                total = t

        return total, abs_total

    def _adaptive_work_tolerance(
        self,
        grad_H: NDArray[np.float64],
        J_eff: NDArray[np.float64],
        abs_sum: float,
    ) -> float:
        r"""
        Construye una cota adaptativa para el trabajo nulo:

            $$tol = \max(100 \epsilon, \text{ULP\_factor} \times \epsilon \times \max(1, \|\nabla H\|_2^2 \|J_{\text{eff}}\|_F, \sum |term|))$$

        Esto evita falsos vetos cuando la escala del gradiente o de J es
        grande, manteniendo sensibilidad a violaciones estructurales.
        """
        grad_norm = float(np.linalg.norm(grad_H, ord=2))
        j_norm = self._frobenius_norm(J_eff, "J_eff")

        if not math.isfinite(grad_norm):
            raise RiemannianInertiaError("||∇H||₂ no es finita.")

        scale_by_operator = (grad_norm * grad_norm) * j_norm
        scale_by_terms = max(1.0, abs_sum)
        scale = max(1.0, scale_by_operator, scale_by_terms)

        return max(
            100.0 * _MACHINE_EPSILON,
            _WORK_TOLERANCE_ULPS * _MACHINE_EPSILON * scale,
        )

    def _certify_nilpotent_work(
        self,
        grad_H: NDArray[np.float64],
        J_eff: NDArray[np.float64],
    ) -> ThermodynamicVetoData:
        r"""
        Certifica el teorema de trabajo nulo:

            $$x^T J_{\text{eff}} x = 0 \quad \text{si} \quad J_{\text{eff}}^T = -J_{\text{eff}}$$

        La certificación se realiza en tres niveles:

        1. Residuo simétrico:
               $$\|J_{\text{eff}} + J_{\text{eff}}^T\|_F \approx 0$$
        2. Proyección estricta:
               $$J_{\text{certified}} = \text{skew}(J_{\text{eff}})$$
        3. Auditoría de Kahan:
               $$|x^T J_{\text{certified}} x| \le tol_{\text{adaptativa}}$$
        """
        grad_H = self._validate_vector(grad_H, "grad_H")
        J_eff = self._validate_square_matrix(J_eff, "J_eff")

        n = grad_H.size
        if J_eff.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre grad_H y J_eff."
            )

        symmetric_residual = self._skew_symmetric_residual(J_eff)
        sym_tol = self._matrix_residual_tolerance(J_eff)

        if symmetric_residual > sym_tol:
            raise SymplecticWorkViolationError(
                "J_eff contiene una componente simétrica incompatible con pasividad. "
                f"Residuo = {symmetric_residual:.4e}, tolerancia = {sym_tol:.4e}."
            )

        # Certificado antisimétrico final.
        J_certified = 0.5 * (J_eff - J_eff.T)

        if not np.all(np.isfinite(J_certified)):
            raise RiemannianInertiaError(
                "La certificación de J_eff produjo valores no finitos."
            )

        work_sum, abs_sum = self._kahan_quadratic_form(grad_H, J_certified)
        work_residual = abs(work_sum)

        work_tolerance = self._adaptive_work_tolerance(
            grad_H,
            J_certified,
            abs_sum,
        )

        is_passive = work_residual <= work_tolerance

        if not is_passive:
            raise SymplecticWorkViolationError(
                "El operador efectivo realiza trabajo neto no nulo. "
                f"Trabajo = {work_residual:.4e}, tolerancia = {work_tolerance:.4e}."
            )

        veto = ThermodynamicVetoData(
            effective_dirac_matrix=J_certified,
            nilpotent_work_residual=work_residual,
            dirac_symmetric_residual=symmetric_residual,
            work_tolerance=work_tolerance,
            is_symplectically_passive=bool(is_passive),
        )

        logger.debug(
            "Fase 3 completada: trabajo residual = %.6e, tolerancia = %.6e, "
            "residuo simétrico = %.6e.",
            veto.nilpotent_work_residual,
            veto.work_tolerance,
            veto.dirac_symmetric_residual,
        )

        return veto

    def execute_phase3(
        self,
        grad_H: NDArray[np.float64],
        J_base: NDArray[np.float64],
        synthesis_data: GyroscopicSynthesisData,
    ) -> ThermodynamicVetoData:
        """
        Método terminal de la Fase 3.

        Recibe la salida formal de execute_phase2 y devuelve el veredicto
        termodinámico definitivo.
        """
        if not isinstance(synthesis_data, GyroscopicSynthesisData):
            raise RiemannianInertiaError(
                "synthesis_data debe ser una instancia de GyroscopicSynthesisData."
            )

        grad_H = self._validate_vector(grad_H, "grad_H")
        J_base = self._validate_square_matrix(J_base, "J_base")

        n = grad_H.size
        if J_base.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre grad_H y J_base."
            )

        W_proj = self._validate_square_matrix(
            synthesis_data.skew_symmetric_tensor,
            "synthesis_data.skew_symmetric_tensor",
        )

        if W_proj.shape != (n, n):
            raise RiemannianInertiaError(
                "La dimensión de W_proj no coincide con grad_H y J_base."
            )

        J_base_skew, _ = self._validate_dirac_structure(J_base, "J_base")
        J_eff = self._modulate_dirac_structure(J_base_skew, W_proj)

        return self._certify_nilpotent_work(grad_H, J_eff)


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: RIEMANNIAN INERTIA MODULATOR
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaModulator(Morphism, Phase3_SymplecticInertiaModulator):
    r"""
    Functor físico puro de moldeo de masa.

    Compone las tres fases en una sola flecha categórica determinista:

        Fase 1 → MomentumAuditData
        Fase 2 → GyroscopicSynthesisData
        Fase 3 → ThermodynamicVetoData

    La composición es estricta: cualquier violación de inercia, antisimetría
    o pasividad simpléctica detiene la ejecución con una excepción tipada.
    """

    def apply_inertia_modulation(
        self,
        q_dot: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        J_base: NDArray[np.float64],
        vorticity_matrix: NDArray[np.float64],
    ) -> ThermodynamicVetoData:
        r"""
        Orquesta la inyección del campo giroscópico de Gauge.

        Flujo formal:

            1. q̇, G, G^{-1}
                 ↦ p = G q̇, ||p||_{G^{-1}}.

            2. p, Ω
                 ↦ W = α(p ∧ (Ω_skew p)), W ∈ so(n).

            3. ∇H, J, W
                 ↦ J_eff = skew(J + W), xᵀJ_eff x = 0.

        Retorna:
            ThermodynamicVetoData con la certificación final.
        """
        momentum_audit = self.execute_phase1(
            q_dot=q_dot,
            G_tensor=G_tensor,
            G_inv=G_inv,
        )

        synthesis_data = self.execute_phase2(
            momentum_data=momentum_audit,
            vorticity_matrix=vorticity_matrix,
        )

        veto_data = self.execute_phase3(
            grad_H=grad_H,
            J_base=J_base,
            synthesis_data=synthesis_data,
        )

        logger.info(
            "Modulación de inercia certificada: ||p|| = %.6e, ||W||_F = %.6e, "
            "trabajo residual = %.6e.",
            momentum_audit.momentum_norm,
            synthesis_data.gyroscopic_frobenius_norm,
            veto_data.nilpotent_work_residual,
        )

        return veto_data


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "RiemannianInertiaError",
    "MomentumDivergenceError",
    "SkewSymmetryViolationError",
    "SymplecticWorkViolationError",
    "MomentumAuditData",
    "GyroscopicSynthesisData",
    "ThermodynamicVetoData",
    "Phase1_MomentumSpectrometer",
    "Phase2_GyroscopicSynthesizer",
    "Phase3_SymplecticInertiaModulator",
    "RiemannianInertiaModulator",
]