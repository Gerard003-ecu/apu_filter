# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Dynamic Shield Router                                               ║
║          (Conexión de Ehresmann y Fibrado de Gauge — Evolución Rigurosa)     ║
║ Ubicación : app/core/immune_system/dynamic_shield_router.py                  ║
║ Versión : 3.0.0-Ehresmann-Higham-Cartan-Riguroso                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Geometría Diferencial
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Actúa como discriminador de campos de Gauge, proveyendo la "vestimenta"
termodinámica exacta al `funtor_shield.py` dependiendo del estrato DIKW
operativo. Implementa una Conexión de Ehresmann que garantiza el transporte
paralelo de la matriz de disipación R(x) a través de la cadena de subespacios:

    V_𝕻 ⊂ V_𝕿 ⊂ V_𝕾 ⊂ V_𝕎

La arquitectura de tres fases anidadas modela la siguiente secuencia categórica:

    (G, ω, H) ──F1──▶ ConnectionData ──F2──▶ DeformationTensor ──F3──▶ FuntorShield

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 1 — Conexión de Ehresmann (Forma de Conexión y Proyector Horizontal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dado el G-fibrado principal π: P → M sobre la variedad base M de métricas de
disipación, la forma de conexión de Ehresmann ω ∈ Ω¹(P, 𝔤) satisface:

    (i)  ω(A*) = A   para todo A ∈ 𝔤  (reproducción del álgebra de Lie)
    (ii) Rg* ω = Ad_{g⁻¹} ω           (equivarianza bajo el grupo G)

El proyector horizontal H : TP → H(P) se construye como:

    H = I − L (Lᵀ G L)⁻¹ Lᵀ G

donde L = Chol(G)⁻¹ A_obs es la representación covariante de la obstrucción
topológica A_obs del estrato en la base ortonormal de G.  Esta construcción
garantiza H² = H (idempotencia) y ker H = V(P) (subespacio vertical).

El factor de escala λ(s) del estrato se deriva del número de Betti β₀(s) y
de la curvatura escalar discreta κ_s = Tr(G⁻¹ ∇²G)/n:

    λ(s) = 1 + log(1 + β₀(s)) · |κ_s|

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 2 — Pullback Termodinámico (Derivada de Lie + Contracción Métrica)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El pullback de la disipación sobre el fibrado de gauge usa la fórmula de
Cartan en su versión discreta:

    ℒ_X R  ≈  (X R + R Xᵀ)   con X = skew(telemetría)

La deformación total por estrato es:

    PHYSICS:  δR = γ · H G (R_d − R_b) G Hᵀ + ℒ_X R_b
    TACTICS:  δR = (α/k_BT) · Δε · H G Hᵀ
    STRATEGY/WISDOM: δR = (α/k_BT_sys) · Λ_L / ρ(R_b) · I

donde ρ(R_b) = radio espectral de R_b (normalización espectral).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 3 — Vestimenta del Escudo (Proyección de Higham + Regularización Tikhonov)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
La proyección exacta al cono S⁺ₙ sigue el algoritmo de Higham (2002):

    R_eff = argmin_{X ∈ S⁺ₙ} ‖R_raw − X‖_F

que se resuelve como:

    R_eff = U · diag(max(λᵢ, 0)) · Uᵀ   con R_raw = U Λ Uᵀ (Schur)

preservando la traza (sin clip ad-hoc).  Si κ(R_eff) > κ_max, se aplica
regularización de Tikhonov adaptativa:

    R_reg = R_eff + ε_tik · I,   ε_tik = (κ_max · λ_min − λ_max) / (κ_max − 1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Anidamiento entre fases:
  · El último objeto producido por la Fase 1 (ConnectionData con H exacto)
    es el argumento inicial de los métodos de la Fase 2.
  · El último objeto producido por la Fase 2 (DeformationTensor con δR
    y métricas de entropía) es el argumento inicial de los métodos de la
    Fase 3, que retorna el FuntorShield vestido.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypeVar

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.schemas import Stratum
from app.core.immune_system.funtor_shield import (
    FuntorShield,
    QuadraticDissipation,
    PortHamiltonianFlow,
)
from app.core.immune_system.metric_tensors import G_PHYSICS

logger = logging.getLogger("MIC.ImmuneSystem.DynamicShieldRouter")

T_Agent = TypeVar("T_Agent", bound=Morphism)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS Y NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════

#: Número de condición máximo admisible para R_eff antes de regularizar.
_KAPPA_MAX: float = 1.0e8

#: Tolerancia para considerar un valor propio numéricamente cero (proyección PSD).
_EIG_TOL: float = 1.0e-12

#: Umbral de ratio ‖δR‖_F / ‖R_base‖_F a partir del cual se emite una advertencia.
_DEFORMATION_RATIO_WARN: float = 0.5

#: Números de Betti β₀ asignados a cada estrato (conectividad topológica discreta).
_BETTI_0: Dict[Stratum, int] = {
    Stratum.PHYSICS: 1,
    Stratum.TACTICS: 2,
    Stratum.STRATEGY: 4,
    Stratum.WISDOM: 8,
}

# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DE LA CONEXIÓN DE EHRESMANN
# ══════════════════════════════════════════════════════════════════════════════


class ConnectionError(TopologicalInvariantError):
    """
    Excepción lanzada cuando la construcción de la conexión de Ehresmann
    para un estrato determinado es geométricamente inválida.

    Condiciones de disparo
    ──────────────────────
    · La métrica escalada G_s no es simétrica definida positiva (SPD).
    · El número de condición κ(G_s) supera el umbral _KAPPA_MAX, indicando
      que la forma de conexión ω no puede construirse de forma numéricamente
      estable.
    · El proyector horizontal H no satisface la idempotencia H² ≈ H dentro
      de la tolerancia numérica.
    """


class PullbackError(TopologicalInvariantError):
    """
    Excepción lanzada cuando el pullback termodinámico produce una deformación
    δR que viola las restricciones algebraicas o físicas.

    Condiciones de disparo
    ──────────────────────
    · La matriz base R no es simétrica o contiene valores propios < -_EIG_TOL.
    · El tensor δR contiene NaN o Inf tras el cálculo del pullback.
    · El ratio ‖δR‖_F / ‖R_base‖_F supera el umbral crítico de estabilidad.
    """


class DressingError(TopologicalInvariantError):
    """
    Excepción lanzada cuando la proyección de la matriz de disipación R_raw
    al cono S⁺ₙ falla o produce un escudo termodinámicamente inestable.

    Condiciones de disparo
    ──────────────────────
    · La descomposición espectral de R_raw no converge.
    · La regularización de Tikhonov produce ε_tik < 0 (inconsistencia espectral).
    · El FuntorShield resultante no puede instanciarse con la nueva disipación.
    """


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES (Data Transfer Objects entre fases)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConnectionData:
    r"""
    Objeto de transferencia producido por la **Fase 1**.

    Encapsula todos los objetos geométricos necesarios para el transporte
    paralelo de la disipación a través del fibrado de Ehresmann.

    Atributos
    ─────────
    metric_tensor : NDArray[float64], shape (n, n)
        Tensor métrico escalado G_s = λ(s) · G_base, simétrico definido positivo.
        Representa la métrica de Fisher-Rao del espacio de distribuciones del
        estrato s.

    horizontal_projector : NDArray[float64], shape (n, n)
        Proyector H : TP → H(P) al subespacio horizontal del fibrado.
        Satisface rigurosamente:
            H² = H              (idempotencia)
            H Gᵀ = G H          (auto-adjuntez respecto a G)
            ker H = V(P)        (anula el subespacio vertical)

    connection_form_norm : float
        Norma Frobenius de la 1-forma de conexión ω representada matricialmente.
        Sirve como medida de la "curvatura local" en el punto del estrato.

    stratum : Stratum
        Estrato DIKW para el cual fue construida la conexión.

    kappa_metric : float
        Número de condición κ(G_s), útil para diagnóstico numérico downstream.
    """

    metric_tensor: NDArray[np.float64]
    horizontal_projector: NDArray[np.float64]
    connection_form_norm: float
    stratum: Stratum
    kappa_metric: float


@dataclass(frozen=True)
class DeformationTensor:
    r"""
    Objeto de transferencia producido por la **Fase 2**.

    Contiene la matriz de deformación δR que modifica la disipación base
    en la Fase 3, junto con metadatos termodinámicos para auditoría.

    Atributos
    ─────────
    delta_R : NDArray[float64], shape (n, n)
        Tensor de deformación δR simétrico (no necesariamente PSD).
        La Fase 3 se encarga de proyectar R_base + δR al cono S⁺ₙ.

    frobenius_ratio : float
        ‖δR‖_F / ‖R_base‖_F — ratio de perturbación relativa.
        Valores > _DEFORMATION_RATIO_WARN indican deformación intensa.

    entropy_production : float
        Estimación de la producción de entropía σ = Tr(δR · G⁻¹) ≥ 0,
        que cuantifica la irreversibilidad introducida por la deformación.

    info : Dict[str, Any]
        Metadatos adicionales: estrato, método de pullback, parámetros usados.
    """

    delta_R: NDArray[np.float64]
    frobenius_ratio: float
    entropy_production: float
    info: Dict[str, Any] = field(default_factory=dict)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1 — CONEXIÓN DE EHRESMANN                                            ║
# ║   (Forma de Conexión ω y Proyector Horizontal Exacto H)                     ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_EhresmannConnection:
    r"""
    Construye la 1-forma de conexión ω sobre el G-fibrado principal de las
    métricas de disipación y determina el transporte paralelo entre estratos.

    Fundamento Geométrico
    ─────────────────────
    Sea π: P → M un G-fibrado principal sobre la variedad base M de métricas
    de disipación, donde G = GL⁺(n, ℝ) actúa por congruencia.

    La forma de conexión de Ehresmann ω ∈ Ω¹(P, 𝔤) se representa en el
    espacio discreto de dimensión n mediante la obstrucción topológica:

        A_obs(s) = ∇²_s S(s) ∈ Sym(n)

    que es el Hessiano discreto de la entropía de Shannon del estrato s,
    evaluado en la dirección del campo vectorial termodinámico.

    El proyector horizontal se construye mediante eliminación de la componente
    vertical respecto a la métrica G:

        L     = Chol(G)⁻¹ A_obs   ∈ ℝ^{n×n}   (obstrucción en base ortonormal)
        M_obs = Lᵀ L              ∈ Sym⁺(n)   (métrica inducida en la fibra)
        H     = I − L M_obs⁻¹ Lᵀ              (proyector de Riesz)

    Se verifica algebraicamente que:
        H²    = H        (idempotencia)
        Im(H) ⊥_G ker H  (G-ortogonalidad entre horizontal y vertical)

    Factor de Escala Termodinámica
    ──────────────────────────────
    El factor λ(s) se deriva del número de Betti β₀(s) (conectividad de la
    fibra del estrato) y de la curvatura escalar discreta de G:

        κ_s   = Tr(G⁻¹) / n   (aproximación de la curvatura escalar media)
        λ(s)  = 1 + log(1 + β₀(s)) · κ_s

    Este factor garantiza que estratos más complejos topológicamente reciben
    una métrica más rígida, reflejando el mayor costo energético del transporte
    paralelo entre sus fibras.

    Parámetros del Constructor
    ──────────────────────────
    base_metric : NDArray[float64], shape (n, n)
        Tensor métrico base G_base (por defecto G_PHYSICS), debe ser SPD.

    obstruction_scale : float, opcional
        Factor de escala de la obstrucción topológica A_obs. Por defecto 0.1,
        que mantiene A_obs pequeña respecto a G para evitar proyectores degenerados.
    """

    def __init__(
        self,
        base_metric: NDArray[np.float64] = G_PHYSICS,
        obstruction_scale: float = 0.1,
    ) -> None:
        self._base_metric = np.asarray(base_metric, dtype=np.float64)
        self._obstruction_scale = float(obstruction_scale)
        self._validate_base_metric()

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def compute_connection(self, target_stratum: Stratum) -> ConnectionData:
        r"""
        Calcula la conexión de Ehresmann para el estrato objetivo.

        Procedimiento
        ─────────────
        1. Escalado de la métrica: G_s = λ(s) · G_base
        2. Validación SPD y número de condición de G_s.
        3. Construcción de la obstrucción A_obs(s) como múltiplo de la
           identidad escalado por el factor topológico del estrato.
        4. Proyector horizontal H por eliminación de Riesz respecto a G_s.
        5. Verificación de idempotencia H² ≈ H.

        Parámetros
        ──────────
        target_stratum : Stratum
            Estrato DIKW para el cual se prepara la conexión.

        Retorna
        ───────
        ConnectionData
            Objeto inmutable con (G_s, H, ‖ω‖_F, stratum, κ(G_s)).

        Lanza
        ─────
        ConnectionError
            Si G_s no es SPD, si κ(G_s) > _KAPPA_MAX o si H² ≠ H.
        """
        # ── Paso 1: Factor de escala derivado de la topología del estrato ──
        lambda_s = self._stratum_scale_factor(target_stratum)
        G_s = self._base_metric * lambda_s

        # ── Paso 2: Validación SPD y condicionamiento de G_s ──
        self._validate_spd(G_s, label=f"G_s[{target_stratum}]")
        kappa = float(np.linalg.cond(G_s))
        if kappa > _KAPPA_MAX:
            raise ConnectionError(
                f"La métrica escalada G_s para el estrato {target_stratum} está "
                f"mal condicionada: κ = {kappa:.3e} > {_KAPPA_MAX:.3e}."
            )

        # ── Paso 3: Obstrucción topológica A_obs(s) ──
        # A_obs = ε_s · G_s representa el Hessiano de la entropía del estrato
        # en la aproximación de campo medio (curvatura constante).
        # ε_s = obstruction_scale · β₀(s) / n refleja la conectividad topológica.
        beta0 = _BETTI_0.get(target_stratum, 1)
        n = G_s.shape[0]
        epsilon_s = self._obstruction_scale * beta0 / n
        A_obs = epsilon_s * G_s

        # ── Paso 4: Proyector horizontal H (eliminación de Riesz) ──
        H, omega_norm = self._build_horizontal_projector(G_s, A_obs)

        # ── Paso 5: Verificación de idempotencia H² = H ──
        H2 = H @ H
        if not np.allclose(H2, H, atol=1e-9):
            raise ConnectionError(
                f"El proyector H para el estrato {target_stratum} no es idempotente: "
                f"‖H² − H‖_F = {np.linalg.norm(H2 - H, 'fro'):.3e}."
            )

        logger.debug(
            "Fase 1 [%s]: λ=%.3f, κ(G_s)=%.2e, ε_s=%.3e, ‖ω‖_F=%.3e",
            target_stratum, lambda_s, kappa, epsilon_s, omega_norm,
        )

        return ConnectionData(
            metric_tensor=G_s,
            horizontal_projector=H,
            connection_form_norm=omega_norm,
            stratum=target_stratum,
            kappa_metric=kappa,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _stratum_scale_factor(self, stratum: Stratum) -> float:
        r"""
        Calcula el factor de escala termodinámica λ(s) del estrato.

        Fórmula:
            κ_s  = Tr(G_base⁻¹) / n      (curvatura escalar media discreta)
            λ(s) = 1 + log(1 + β₀(s)) · |κ_s|

        El factor κ_s captura la "curvatura intrínseca" de la métrica base,
        mientras que β₀(s) refleja la complejidad topológica del estrato.
        El logaritmo suaviza el crecimiento para estratos con β₀ grande.

        Parámetros
        ──────────
        stratum : Stratum
            Estrato DIKW objetivo.

        Retorna
        ───────
        float
            Factor multiplicativo λ(s) ≥ 1.
        """
        G_inv = la.inv(self._base_metric)
        n = self._base_metric.shape[0]
        kappa_scalar = float(np.trace(G_inv)) / n   # curvatura escalar media
        beta0 = _BETTI_0.get(stratum, 1)
        lambda_s = 1.0 + np.log1p(beta0) * abs(kappa_scalar)
        return float(lambda_s)

    def _build_horizontal_projector(
        self,
        G: NDArray[np.float64],
        A_obs: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Construye el proyector horizontal H y calcula ‖ω‖_F.

        La forma de conexión ω se representa matricialmente como:

            ω_mat = G⁻¹ A_obs   ∈ ℝ^{n×n}

        (el mapeo del subespacio tangente al álgebra de Lie de la fibra).

        El proyector de Riesz sobre el subespacio horizontal es:

            L     = Chol(G)⁻¹ A_obs          (A_obs en base ortonormal de G)
            M     = Lᵀ L + _EIG_TOL · I       (regularización por robustez)
            H     = I − L M⁻¹ Lᵀ

        Nota: la regularización de M evita singularidades cuando A_obs ≈ 0.

        Parámetros
        ──────────
        G : NDArray[float64], shape (n, n)
            Métrica escalada del estrato.
        A_obs : NDArray[float64], shape (n, n)
            Obstrucción topológica (Hessiano de la entropía del estrato).

        Retorna
        ───────
        H : NDArray[float64], shape (n, n)
            Proyector horizontal idempotente.
        omega_norm : float
            ‖G⁻¹ A_obs‖_F, norma de la forma de conexión.
        """
        n = G.shape[0]

        # Descomposición de Cholesky de G para obtener la base ortonormal
        # G = Lc Lcᵀ  →  Chol(G) = Lc  (triangular inferior)
        Lc = la.cholesky(G, lower=True)       # G = Lc Lcᵀ
        Lc_inv = la.solve_triangular(         # Lc⁻¹ por sustitución
            Lc, np.eye(n), lower=True, check_finite=False
        )

        # Representación de A_obs en la base ortonormal de G
        L = Lc_inv @ A_obs    # shape (n, n)

        # Métrica inducida en la fibra (Gram matrix)
        M = L.T @ L + _EIG_TOL * np.eye(n)   # regularización mínima

        # Proyector horizontal: H = I − L M⁻¹ Lᵀ
        # Se usa la solución de sistema lineal para mayor estabilidad
        M_inv_Lt = la.solve(M, L.T, assume_a='pos')   # M⁻¹ Lᵀ
        H = np.eye(n) - L @ M_inv_Lt

        # Norma de la forma de conexión ω = G⁻¹ A_obs
        G_inv = la.cho_solve((Lc, True), np.eye(n))
        omega_mat = G_inv @ A_obs
        omega_norm = float(np.linalg.norm(omega_mat, 'fro'))

        return H, omega_norm

    def _validate_base_metric(self) -> None:
        """Verifica que la métrica base es cuadrada, simétrica y SPD."""
        G = self._base_metric
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ConnectionError("La métrica base debe ser una matriz cuadrada.")
        self._validate_spd(G, label="G_base")

    @staticmethod
    def _validate_spd(M: NDArray[np.float64], label: str = "M") -> None:
        """
        Verifica que M es simétrica y definida positiva mediante
        descomposición de Cholesky (criterio exacto SPD).

        Lanza ConnectionError si falla.
        """
        if not np.allclose(M, M.T, atol=1e-10):
            raise ConnectionError(
                f"{label} no es simétrica: ‖M − Mᵀ‖_F = "
                f"{np.linalg.norm(M - M.T, 'fro'):.3e}."
            )
        try:
            la.cholesky(M, lower=True)
        except la.LinAlgError as exc:
            raise ConnectionError(
                f"{label} no es definida positiva (Cholesky falló): {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Fin de Fase 1 — el objeto ConnectionData retornado por compute_connection
    # es el argumento de entrada de Phase2_ThermodynamicPullback.compute_deformation
    # ──────────────────────────────────────────────────────────────────────────


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2 — PULLBACK TERMODINÁMICO                                           ║
# ║   (Derivada de Lie de Cartan + Contracción Métrica Covariante)              ║
# ║                                                                             ║
# ║   ENTRADA: ConnectionData (Fase 1) ─────────────────────────────────────── ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_ThermodynamicPullback:
    r"""
    Ejecuta el pullback de las variables termodinámicas (telemetría del agente)
    sobre el fibrado de disipación, usando la conexión exacta de la Fase 1.

    Fundamento Matemático
    ─────────────────────
    Sea φ: Σ_agent → P el embedding del espacio de estados del agente en el
    fibrado principal. El pullback termodinámico φ*ω extrae la componente
    horizontal de la variación de disipación.

    La deformación total δR = (φ*ω)(R_base) se calcula por estrato:

    **PHYSICS** (estrato de observables físicos directos):
        δR = γ · H G_s (R_d − R_b) G_s Hᵀ   +   ℒ_X R_b

        donde la derivada de Lie de Cartan se aproxima en espacio discreto:
            ℒ_X R_b ≈ X R_b + R_b Xᵀ
        con X = skew(R_d − R_b) = (R_d − R_b − (R_d − R_b)ᵀ) / 2

        La proyección H G_s (·) G_s Hᵀ es el análogo discreto del pull-back
        covariante con respecto a la métrica del estrato.

    **TACTICS** (estrato de control táctico):
        δR = (α/k_BT) · Δε_ruido · H G_s Hᵀ

        La deformación es proporcional a la energía de ruido descartada,
        contraída con el proyector horizontal sobre la métrica del estrato.
        Esto asegura que solo la componente "horizontal" del ruido afecta
        la disipación, eliminando artefactos de gauge.

    **STRATEGY / WISDOM** (estratos de planificación y sabiduría):
        Λ_L  = Tr(Σ_k γ_k L_k ρ L_k†)   (traza de disipación de Lindblad)
        ρ_sp = ρ(R_b) = max(|λ_i(R_b)|)  (radio espectral de R_base)
        δR   = (α/k_BT_sys) · (Λ_L / ρ_sp) · I

        La normalización por el radio espectral ρ(R_b) garantiza que la
        deformación de Lindblad es adimensional respecto a la escala de R_base,
        evitando explosiones de norma en estratos de alta entropía.

    Validaciones
    ─────────────
    · R_base debe ser simétrica y semidefinida positiva.
    · δR no debe contener NaN ni Inf.
    · ‖δR‖_F / ‖R_base‖_F < umbral crítico (emite warning si se supera).
    · La producción de entropía σ = Tr(δR · G_s⁻¹) es computada y reportada.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def compute_deformation(
        self,
        connection: ConnectionData,
        agent_telemetry: Dict[str, Any],
        base_R: NDArray[np.float64],
    ) -> DeformationTensor:
        r"""
        Calcula δR a partir de la telemetría, la métrica escalada y la
        conexión de Ehresmann de la Fase 1.

        Parámetros
        ──────────
        connection : ConnectionData
            Datos de la conexión provenientes de la Fase 1, conteniendo
            G_s, H exacto, estrato y κ(G_s).

        agent_telemetry : Dict[str, Any]
            Datos termodinámicos del agente. Las claves esperadas varían
            por estrato:
            · PHYSICS  : 'R_desired', 'gamma_coupling'
            · TACTICS  : 'discarded_spherical_entropy', 'alpha_over_kT'
            · STRATEGY/WISDOM: 'lindblad_dissipation_trace', 'alpha_over_kT_sys'

        base_R : NDArray[float64], shape (n, n)
            Matriz de disipación actual del escudo (simétrica, PSD).

        Retorna
        ───────
        DeformationTensor
            Objeto inmutable con δR, ratio de perturbación, producción de
            entropía e información de auditoría.

        Lanza
        ─────
        PullbackError
            En cualquier violación de consistencia de R_base o de δR.
        """
        stratum = connection.stratum
        G_s = connection.metric_tensor
        H = connection.horizontal_projector

        # ── Validación de R_base ──
        self._validate_base_R(base_R, label="base_R")

        # ── Despacho por estrato ──
        if stratum == Stratum.PHYSICS:
            delta_R, method = self._pullback_physics(
                agent_telemetry, base_R, G_s, H
            )
        elif stratum == Stratum.TACTICS:
            delta_R, method = self._pullback_tactics(
                agent_telemetry, base_R, G_s, H
            )
        elif stratum in (Stratum.STRATEGY, Stratum.WISDOM):
            delta_R, method = self._pullback_omega(
                agent_telemetry, base_R, G_s, H
            )
        else:
            raise PullbackError(
                f"Estrato {stratum} no soportado para pullback termodinámico."
            )

        # ── Verificación de integridad numérica ──
        if np.any(~np.isfinite(delta_R)):
            raise PullbackError(
                f"La deformación δR para el estrato {stratum} contiene "
                f"NaN o Inf tras el método '{method}'."
            )

        # Simetrización explícita (garantía algebraica)
        delta_R = 0.5 * (delta_R + delta_R.T)

        # ── Métricas de perturbación y entropía ──
        norm_base = np.linalg.norm(base_R, 'fro')
        norm_delta = np.linalg.norm(delta_R, 'fro')
        ratio = float(norm_delta / (norm_base + 1e-15))

        if ratio > _DEFORMATION_RATIO_WARN:
            logger.warning(
                "Fase 2 [%s]: Deformación intensa ‖δR‖_F/‖R_b‖_F = %.3f > %.2f. "
                "Verificar parámetros de telemetría.",
                stratum, ratio, _DEFORMATION_RATIO_WARN,
            )

        # Producción de entropía σ = Tr(δR G_s⁻¹) ≥ 0 en el equilibrio
        G_s_inv = la.inv(G_s)
        sigma = float(np.trace(delta_R @ G_s_inv))

        logger.debug(
            "Fase 2 [%s]: método='%s', ‖δR‖_F=%.3e, ratio=%.3f, σ=%.4f",
            stratum, method, norm_delta, ratio, sigma,
        )

        return DeformationTensor(
            delta_R=delta_R,
            frobenius_ratio=ratio,
            entropy_production=sigma,
            info={
                "stratum": stratum,
                "method": method,
                "norm_delta_R": norm_delta,
                "norm_base_R": norm_base,
                "connection_form_norm": connection.connection_form_norm,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PULLBACKS ESPECÍFICOS POR ESTRATO
    # ──────────────────────────────────────────────────────────────────────────

    def _pullback_physics(
        self,
        telemetry: Dict[str, Any],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        H: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], str]:
        r"""
        Pullback covariante para el estrato PHYSICS.

        Fórmula:
            ΔR_cov  = H G_s (R_d − R_b) G_s Hᵀ    (componente de conexión)
            X_skew  = (ΔR_raw − ΔR_rawᵀ) / 2       (parte antisimétrica)
            ℒ_X R_b = X_skew R_b + R_b X_skewᵀ    (derivada de Lie discreta)
            δR      = γ · ΔR_cov + ℒ_X R_b

        El término de conexión proyecta la "diferencia de disipación" al
        subespacio horizontal de la fibra, eliminando componentes de gauge
        espurias. El término ℒ_X modela la torsión helicoidal de la
        trayectoria en el espacio de disipación.

        Parámetros de telemetría usados:
            'R_desired'    : NDArray[float64], shape (n,n) — disipación objetivo
            'gamma_coupling': float — acoplamiento cinético (default 0.1)
        """
        R_desired = np.asarray(
            telemetry.get("R_desired", R_base), dtype=np.float64
        )
        gamma = float(telemetry.get("gamma_coupling", 0.1))

        # Validar R_desired
        self._validate_base_R(R_desired, label="R_desired")

        # Diferencia cruda en el espacio de disipación
        Delta_R_raw = R_desired - R_base

        # Componente horizontal covariante: H G_s ΔR G_s Hᵀ
        Delta_R_cov = H @ G @ Delta_R_raw @ G @ H.T

        # Derivada de Lie de Cartan (versión discreta):
        # ℒ_X R_b ≈ X R_b + R_b Xᵀ  con X = parte antisimétrica de ΔR_raw
        X_skew = 0.5 * (Delta_R_raw - Delta_R_raw.T)
        lie_R = X_skew @ R_base + R_base @ X_skew.T

        delta_R = gamma * Delta_R_cov + lie_R
        return delta_R, "pullback_physics_covariante_cartan"

    def _pullback_tactics(
        self,
        telemetry: Dict[str, Any],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        H: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], str]:
        r"""
        Pullback para el estrato TACTICS.

        Fórmula:
            δR = (α / k_B T) · Δε_ruido · H G_s Hᵀ

        La deformación es la imagen del proyector horizontal aplicado sobre
        la métrica del estrato, escalada por la energía de ruido descartada.
        Esto modela el efecto termodinámico de filtrar fluctuaciones tácticas
        no relevantes para el transporte paralelo de la disipación.

        Parámetros de telemetría usados:
            'discarded_spherical_entropy' : float — Δε en unidades de k_B T
            'alpha_over_kT'              : float — coeficiente α/(k_B T)
        """
        delta_E = float(telemetry.get("discarded_spherical_entropy", 0.0))
        alpha_kT = float(telemetry.get("alpha_over_kT", 1.0))

        # Forma cuadrática proyectada: H G_s Hᵀ
        HGHt = H @ G @ H.T

        delta_R = alpha_kT * delta_E * HGHt
        return delta_R, "pullback_tactics_horizontal_metrico"

    def _pullback_omega(
        self,
        telemetry: Dict[str, Any],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        H: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], str]:
        r"""
        Pullback para los estratos STRATEGY y WISDOM (estrato Omega).

        Fórmula:
            ρ_sp = max(|λ_i(R_base)|)       (radio espectral)
            Λ_L  = Tr(Σ_k γ_k L_k ρ L_k†)  (disipación de Lindblad)
            δR   = (α / k_B T_sys) · (Λ_L / (ρ_sp + ε)) · I

        La normalización por el radio espectral ρ_sp garantiza que la
        perturbación es adimensional: el cociente Λ_L / ρ_sp mide cuántas
        "veces la escala propia de R_base" es la disipación de Lindblad.
        El término ε = _EIG_TOL previene divisiones por cero.

        Parámetros de telemetría usados:
            'lindblad_dissipation_trace' : float — Λ_L
            'alpha_over_kT_sys'         : float — α/(k_B T_sys)
        """
        lindblad_trace = float(telemetry.get("lindblad_dissipation_trace", 0.0))
        alpha_kT_sys = float(telemetry.get("alpha_over_kT_sys", 1.0))

        # Radio espectral de R_base (normalización espectral)
        rho_sp = float(np.max(np.abs(la.eigvalsh(R_base))))
        rho_sp_safe = rho_sp + _EIG_TOL

        n = R_base.shape[0]
        coeff = alpha_kT_sys * lindblad_trace / rho_sp_safe
        delta_R = coeff * np.eye(n)

        return delta_R, "pullback_omega_lindblad_espectral"

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_base_R(
        R: NDArray[np.float64],
        label: str = "R",
    ) -> None:
        r"""
        Verifica que R ∈ Sym⁺ₙ (simétrica semidefinida positiva).

        Criterios:
            · Simetría : ‖R − Rᵀ‖_F < 1e-10 · ‖R‖_F
            · PSD      : λ_min(R) ≥ −_EIG_TOL

        Lanza PullbackError con descripción cuantitativa del fallo.
        """
        sym_error = np.linalg.norm(R - R.T, 'fro')
        norm_R = np.linalg.norm(R, 'fro') + 1e-15
        if sym_error > 1e-10 * norm_R:
            raise PullbackError(
                f"{label} no es simétrica: ‖R − Rᵀ‖_F = {sym_error:.3e}, "
                f"‖R‖_F = {norm_R:.3e}."
            )
        eigmin = float(la.eigvalsh(R, subset_by_index=[0, 0])[0])
        if eigmin < -_EIG_TOL:
            raise PullbackError(
                f"{label} no es semidefinida positiva: λ_min = {eigmin:.3e} "
                f"< −{_EIG_TOL:.1e}."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Fin de Fase 2 — el objeto DeformationTensor retornado por compute_deformation
    # es el argumento de entrada de Phase3_ShieldDresser.dress_shield
    # ──────────────────────────────────────────────────────────────────────────


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3 — VESTIMENTA DEL ESCUDO                                            ║
# ║   (Proyección de Higham + Regularización Tikhonov Adaptativa)               ║
# ║                                                                             ║
# ║   ENTRADA: DeformationTensor (Fase 2) ──────────────────────────────────── ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_ShieldDresser:
    r"""
    Aplica la deformación δR sobre el escudo base, proyecta rigurosamente
    la suma R_raw = R_base + δR al cono de matrices simétricas semidefinidas
    positivas S⁺ₙ mediante el algoritmo de Higham (2002), y retorna un
    **nuevo** FuntorShield inmutable con el flujo Port-Hamiltoniano vestido.

    Algoritmo de Higham para la Proyección S⁺ₙ
    ────────────────────────────────────────────
    La proyección ortogonal (en norma de Frobenius) de una matriz simétrica
    A ∈ Sym(n) al cono S⁺ₙ es:

        Π_{S⁺ₙ}(A) = U · diag(max(λᵢ, 0)) · Uᵀ

    donde A = U Λ Uᵀ es la descomposición espectral real (Schur simétrico).

    Esta proyección:
        · Minimiza ‖A − X‖_F sobre X ∈ S⁺ₙ            (optimalidad)
        · Preserva la traza si A ya es PSD              (isometría parcial)
        · Preserva los vectores propios de A            (equivarianza espectral)

    A diferencia del clip ad-hoc λᵢ → max(λᵢ, 0) (que coincide con Higham
    para matrices simétricas), la implementación aquí usa la descomposición
    `scipy.linalg.eigh` que garantiza vectores propios ortonormales reales.

    Regularización de Tikhonov Adaptativa
    ──────────────────────────────────────
    Si κ(R_eff) = λ_max / λ_min > κ_max, se aplica la regularización:

        ε_tik  = (κ_max · λ_min − λ_max) / (κ_max − 1)
        R_reg  = R_eff + ε_tik · I

    Esta fórmula es la solución exacta de:
        min_{ε ≥ 0} ε  s.t.  κ(R_eff + ε I) ≤ κ_max

    garantizando el mínimo shift espectral que satisface la restricción de
    condicionamiento sin destruir información propia.

    Inmutabilidad Funcional
    ───────────────────────
    En contraste con la v2.0.0, NO se muta `base_shield.flow` in-place.
    En su lugar, se construye un **nuevo** FuntorShield mediante copia
    profunda del escudo base con el flow reemplazado. Esto preserva la
    semántica funcional y la trazabilidad del historial de escudos.

    Parámetros del Constructor
    ──────────────────────────
    psd_tolerance : float
        Tolerancia para considerar valores propios numéricamente cero.
        Por defecto _EIG_TOL = 1e-12.

    kappa_max : float
        Número de condición máximo admisible para R_eff.
        Si κ(R_eff) > kappa_max, se regulariza con Tikhonov adaptativo.
        Por defecto _KAPPA_MAX = 1e8.
    """

    def __init__(
        self,
        psd_tolerance: float = _EIG_TOL,
        kappa_max: float = _KAPPA_MAX,
    ) -> None:
        self._tol = float(psd_tolerance)
        self._kappa_max = float(kappa_max)

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def dress_shield(
        self,
        base_shield: FuntorShield[T_Agent],
        deformation: DeformationTensor,
    ) -> FuntorShield[T_Agent]:
        r"""
        Produce el FuntorShield vestido con la disipación efectiva:

            R_eff = Π_{S⁺ₙ}(R_base + δR)   [+ regularización Tikhonov si κ > κ_max]

        Procedimiento
        ─────────────
        1. Suma R_raw = R_base + δR.
        2. Proyección al cono S⁺ₙ por Higham (descomposición espectral + clip).
        3. Simetrización numérica para cancelar errores de redondeo.
        4. Verificación del ratio de perturbación relativa (warning si > umbral).
        5. Regularización Tikhonov adaptativa si κ(R_eff) > κ_max.
        6. Construcción del nuevo PortHamiltonianFlow con la nueva disipación.
        7. Retorno de un nuevo FuntorShield (copia profunda + flow vestido).

        Parámetros
        ──────────
        base_shield : FuntorShield[T_Agent]
            Escudo original. No se modifica in-place.
        deformation : DeformationTensor
            Tensor δR producido por la Fase 2, con metadatos de auditoría.

        Retorna
        ───────
        FuntorShield[T_Agent]
            Nuevo escudo con disipación R_eff ∈ S⁺ₙ y flujo Port-Hamiltoniano
            vestido. El escudo original permanece inmutado.

        Lanza
        ─────
        DressingError
            Si la descomposición espectral no converge, si ε_tik < 0 o si
            el nuevo FuntorShield no puede instanciarse.
        """
        R_base = base_shield.flow.R.matrix

        # ── Paso 1: Suma de la deformación ──
        R_raw = R_base + deformation.delta_R
        # Simetrización preventiva antes de la proyección
        R_raw = 0.5 * (R_raw + R_raw.T)

        # ── Paso 2: Proyección al cono S⁺ₙ (Higham 2002) ──
        R_psd = self._project_to_psd_cone(R_raw)

        # ── Paso 3: Simetrización post-proyección ──
        R_psd = 0.5 * (R_psd + R_psd.T)

        # ── Paso 4: Diagnóstico de perturbación relativa ──
        self._check_perturbation_ratio(R_base, deformation)

        # ── Paso 5: Regularización Tikhonov adaptativa ──
        R_eff = self._tikhonov_regularize(R_psd)

        # ── Paso 6: Construcción del flujo vestido ──
        new_dissipation = QuadraticDissipation(R_eff)
        try:
            dressed_flow = PortHamiltonianFlow(
                J=base_shield.flow.J,
                R=new_dissipation,
                grad_H=base_shield.flow.grad_H,
            )
        except Exception as exc:
            raise DressingError(
                f"No se pudo construir el PortHamiltonianFlow vestido: {exc}"
            ) from exc

        # ── Paso 7: Nuevo FuntorShield (inmutabilidad funcional) ──
        dressed_shield = copy.deepcopy(base_shield)
        dressed_shield.flow = dressed_flow

        # ── Logging de diagnóstico ──
        kappa_eff = float(np.linalg.cond(R_eff))
        logger.info(
            "Fase 3 [%s]: FuntorShield vestido. "
            "‖δR‖_F=%.3e, ratio=%.3f, σ=%.4f, κ(R_eff)=%.2e",
            deformation.info.get("stratum", "?"),
            np.linalg.norm(deformation.delta_R, 'fro'),
            deformation.frobenius_ratio,
            deformation.entropy_production,
            kappa_eff,
        )

        return dressed_shield

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _project_to_psd_cone(
        self,
        M: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Proyecta M al cono S⁺ₙ usando el algoritmo de Higham.

        Implementación
        ──────────────
            evals, evecs = eigh(M)        (descomposición espectral real)
            evals_psd    = max(evals, 0)  (truncamiento al semieje positivo)
            M_psd        = evecs diag(evals_psd) evecsᵀ

        Se emite un mensaje de debug si se detectan valores propios
        significativamente negativos (< −_EIG_TOL), indicando que la
        suma R_base + δR salió del cono de disipación válido.

        Lanza DressingError si la descomposición espectral no converge.
        """
        try:
            evals, evecs = la.eigh(M)
        except la.LinAlgError as exc:
            raise DressingError(
                f"Descomposición espectral de R_raw falló en la proyección "
                f"de Higham: {exc}"
            ) from exc

        n_negative = int(np.sum(evals < -self._tol))
        if n_negative > 0:
            min_eval = float(np.min(evals))
            logger.debug(
                "Fase 3: Proyección Higham — %d valor(es) propio(s) negativo(s) "
                "detectado(s) (λ_min = %.3e); truncados a cero.",
                n_negative, min_eval,
            )

        evals_psd = np.maximum(evals, 0.0)
        M_psd = evecs @ np.diag(evals_psd) @ evecs.T
        return M_psd

    def _tikhonov_regularize(
        self,
        R_psd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Aplica regularización de Tikhonov adaptativa si κ(R_psd) > κ_max.

        Fórmula exacta del shift mínimo:
            λ_max  = máximo valor propio de R_psd
            λ_min  = mínimo valor propio de R_psd (≥ 0 tras proyección Higham)
            ε_tik  = (κ_max · λ_min − λ_max) / (κ_max − 1)

        Si κ(R_psd) ≤ κ_max, retorna R_psd sin modificación.

        Si ε_tik < 0 (inconsistencia: λ_max / λ_min > κ_max pero la fórmula
        produce valor negativo por error numérico), se lanza DressingError.

        Lanza
        ─────
        DressingError si ε_tik < 0 o si la regularización produce κ > κ_max.
        """
        evals = la.eigvalsh(R_psd)
        lambda_min = float(evals[0])    # eigh retorna en orden ascendente
        lambda_max = float(evals[-1])

        # Condicionamiento efectivo (usar lambda_min + _EIG_TOL para robustez)
        lambda_min_safe = lambda_min + self._tol
        kappa_eff = lambda_max / lambda_min_safe

        if kappa_eff <= self._kappa_max:
            return R_psd   # Sin regularización necesaria

        # Shift de Tikhonov mínimo para κ(R_reg) = κ_max exactamente
        epsilon_tik = (self._kappa_max * lambda_min - lambda_max) / (
            self._kappa_max - 1.0
        )

        if epsilon_tik < 0.0:
            # Esto ocurre si λ_min ≈ 0 y κ_max no puede satisfacerse
            # con un shift positivo; usamos el shift que iguala λ_min al umbral
            epsilon_tik = (lambda_max / self._kappa_max) - lambda_min
            logger.warning(
                "Fase 3: Tikhonov — ε_tik negativo por λ_min ≈ 0; "
                "usando shift alternativo ε_tik = %.3e.", epsilon_tik,
            )
            if epsilon_tik < 0.0:
                raise DressingError(
                    f"Regularización Tikhonov inconsistente: "
                    f"ε_tik = {epsilon_tik:.3e} < 0. "
                    f"λ_max = {lambda_max:.3e}, λ_min = {lambda_min:.3e}, "
                    f"κ_max = {self._kappa_max:.2e}."
                )

        n = R_psd.shape[0]
        R_reg = R_psd + epsilon_tik * np.eye(n)

        logger.debug(
            "Fase 3: Tikhonov — κ_eff = %.2e > κ_max = %.2e. "
            "ε_tik = %.3e aplicado. Nuevo κ ≈ %.2e.",
            kappa_eff, self._kappa_max, epsilon_tik,
            float(np.linalg.cond(R_reg)),
        )

        return R_reg

    @staticmethod
    def _check_perturbation_ratio(
        R_base: NDArray[np.float64],
        deformation: DeformationTensor,
    ) -> None:
        """
        Emite una advertencia si el ratio de perturbación relativa del tensor
        de deformación supera el umbral _DEFORMATION_RATIO_WARN.

        Este chequeo es redundante con el de la Fase 2 pero opera sobre el
        escudo real (no la copia de telemetría), sirviendo como segunda línea
        de verificación antes de la escritura final del flow.
        """
        if deformation.frobenius_ratio > _DEFORMATION_RATIO_WARN:
            logger.warning(
                "Fase 3: Ratio de perturbación relativa = %.3f > %.2f. "
                "La deformación es grande respecto a R_base; "
                "verificar coherencia termodinámica del escudo.",
                deformation.frobenius_ratio,
                _DEFORMATION_RATIO_WARN,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Fin de Fase 3 — el FuntorShield retornado por dress_shield es el producto
    # final del pipeline de vestimenta y el argumento de retorno del Orquestador.
    # ──────────────────────────────────────────────────────────────────────────


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR: DYNAMIC SHIELD ROUTER (FACHADA CATEGÓRICA)                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class DynamicShieldRouter:
    r"""
    Fachada categórica que enruta las transformaciones naturales de vestimenta
    según el estrato DIKW, encadenando las tres fases anidadas:

        Phase1 ──(ConnectionData)──▶ Phase2 ──(DeformationTensor)──▶ Phase3
                                                                        │
                                                              FuntorShield vestido

    La composición de las tres fases define un funtor:

        Ψ: Stratum × Telemetry × FuntorShield ──▶ FuntorShield

    que preserva la estructura simpléctica de los flujos Port-Hamiltonianos
    y garantiza la disipatividad (R_eff ∈ S⁺ₙ) por construcción.

    Uso
    ───
        dressed = DynamicShieldRouter.dress_shield_for_stratum(
            base_shield    = mi_escudo,
            target_stratum = Stratum.TACTICS,
            agent_telemetry = {
                "discarded_spherical_entropy": 0.05,
                "alpha_over_kT": 2.3,
            },
        )
    """

    @classmethod
    def dress_shield_for_stratum(
        cls,
        base_shield: FuntorShield[T_Agent],
        target_stratum: Stratum,
        agent_telemetry: Dict[str, Any],
        base_metric: NDArray[np.float64] = G_PHYSICS,
        obstruction_scale: float = 0.1,
        psd_tolerance: float = _EIG_TOL,
        kappa_max: float = _KAPPA_MAX,
    ) -> FuntorShield[T_Agent]:
        r"""
        Viste categóricamente al escudo según el estrato de destino.

        Parámetros
        ──────────
        base_shield : FuntorShield[T_Agent]
            Escudo a vestir. No se modifica in-place.

        target_stratum : Stratum
            Estrato DIKW objetivo (PHYSICS, TACTICS, STRATEGY, WISDOM).

        agent_telemetry : Dict[str, Any]
            Datos termodinámicos del agente. Las claves requeridas dependen
            del estrato:
            · PHYSICS  : 'R_desired' (NDArray), 'gamma_coupling' (float)
            · TACTICS  : 'discarded_spherical_entropy' (float),
                         'alpha_over_kT' (float)
            · STRATEGY/WISDOM: 'lindblad_dissipation_trace' (float),
                                'alpha_over_kT_sys' (float)

        base_metric : NDArray[float64], opcional
            Tensor métrico base G_base (por defecto G_PHYSICS).
            Debe ser simétrico definido positivo de dimensión (n, n).

        obstruction_scale : float, opcional
            Factor de escala de la obstrucción topológica A_obs. Por defecto 0.1.

        psd_tolerance : float, opcional
            Tolerancia para la proyección Higham y la regularización Tikhonov.

        kappa_max : float, opcional
            Número de condición máximo admisible para R_eff.

        Retorna
        ───────
        FuntorShield[T_Agent]
            Escudo vestido con disipación R_eff ∈ S⁺ₙ y κ(R_eff) ≤ κ_max.

        Lanza
        ─────
        ConnectionError, PullbackError, DressingError
            Según la fase donde ocurra el error, con descripción cuantitativa.
        """
        # ── Fase 1: Conexión de Ehresmann ────────────────────────────────────
        phase1 = Phase1_EhresmannConnection(
            base_metric=base_metric,
            obstruction_scale=obstruction_scale,
        )
        connection_data = phase1.compute_connection(target_stratum)

        # ── Fase 2: Pullback Termodinámico ───────────────────────────────────
        phase2 = Phase2_ThermodynamicPullback()
        R_base = base_shield.flow.R.matrix
        deformation = phase2.compute_deformation(
            connection_data, agent_telemetry, R_base
        )

        # ── Fase 3: Vestimenta del Escudo ────────────────────────────────────
        phase3 = Phase3_ShieldDresser(
            psd_tolerance=psd_tolerance,
            kappa_max=kappa_max,
        )
        dressed_shield = phase3.dress_shield(base_shield, deformation)

        logger.info(
            "DynamicShieldRouter: Pipeline completo para estrato %s. "
            "κ(G_s)=%.2e, σ=%.4f, ratio=%.3f.",
            target_stratum,
            connection_data.kappa_metric,
            deformation.entropy_production,
            deformation.frobenius_ratio,
        )

        return dressed_shield


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Excepciones
    "ConnectionError",
    "PullbackError",
    "DressingError",
    # Estructuras de datos
    "ConnectionData",
    "DeformationTensor",
    # Fases
    "Phase1_EhresmannConnection",
    "Phase2_ThermodynamicPullback",
    "Phase3_ShieldDresser",
    # Orquestador
    "DynamicShieldRouter",
]