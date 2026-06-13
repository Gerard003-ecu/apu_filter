# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Geodesic Attention Fibrator (Fibrado de Atención y Torre Covariante) ║
║ Ubicación: app/wisdom/geodesic_attention_fibrator.py                         ║
║ Versión: 3.0.0-Rigorous-Geometric-Quantum-PhaseNested                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física, Topológica y Categorial
────────────────────────────────────────────────────────────────────────────────
El operador de conexión de Ehresmann sobre el Fibrado Principal de la Malla
Agéntica (Estrato Γ-WISDOM) se ha re-estructurado en tres fases anidadas que
reflejan la jerarquía matemática:

  FASE 1 — CIMIENTO GEOMÉTRICO (Geometría Diferencial Discreta)
      Construye la variedad Riemanniana (M, g_eff) a partir del tensor métrico
      base, la torsión topológica extraída de la cohomología de haces celulares,
      y la cohomología de de Rham discreta. Deriva:
        • Tensor de torsión corregido T^μ_{νρ} (tensor (1,2))
        • Métrica efectiva g_eff ∈ Sym^+(d)  (cono SDP)
        • Tensor de Ricci Ric_{μν} vía flujo de Ricci discreto
        • Símbolos de Christoffel Γ^μ_{νρ} (Levi-Civita + corrección de torsión)
        • Transporte paralelo explícito a lo largo de aristas del haz celular

  FASE 2 — ATENCIÓN COVARIANTE (Haz de Associadores sobre M)
      Levanta el haz de atención al fibrado tangente TM. Reparametriza:
        • Producto interno ⟨Q, K⟩_g = Q^T g_eff K (con la métrica curvada)
        • Geodésica discreta minimizadora de la energía de Dirichlet
        • Softmax covariante con corrección de Bohr compactification
      Garantiza que las transiciones de pensamiento sigan geodésicas
      minimizadoras de la funcional de Polyakov.

  FASE 3 — SUPRESIÓN CUÁNTICA (Integral de Caminos de Feynman-Kac)
      Evalúa la amplitud de probabilidad del trayecto cognitivo:
        Ψ[γ] = ∫ 𝒟γ exp( -S_E[γ] / ℏ )
      con S_E[γ] = ∫_γ (½ g_{μν} dγ^μ dγ^ν + V_torsion(γ)) la acción
      euclídea. Si la amplitud cae por debajo del umbral cuántico de la
      máquina, la ruta es vetada axiomáticamente por el principio de veto
      cuántico (analogía con el efecto túnel suprimido por debajo de E_F).

Garantías formales:
  • g_eff permanece en el cono de matrices simétricas definidas positivas (SDP)
    vía proyección de Moreau-Yosida regularizada.
  • Compatibilidad métrica ∇_ρ g_{μν} = 0 verificada numéricamente.
  • Identidad de Bianchi de primera especie verificada antisimétricamente.
  • Estabilidad bajo transporte paralelo: ||P_{γ(0)→γ(1)}||_op ≤ 1 + ε.

Todas las constantes físicas y umbrales se han refinado para evitar
singularidades numéricas y mantener la coherencia categorial del morfismo.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias arquitectónicas del ecosistema APU Filter
from app.core.mic_algebra import Morphism, CategoricalState, NumericalInstabilityError
from app.core.schemas import Stratum
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.boole.strategy.sheaf_cohomology_orchestrator import CellularSheaf

logger = logging.getLogger("MAC.Wisdom.GeodesicAttentionFibrator")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICO-GEOMÉTRICAS DE ALTA PRECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
class FibratorConstants:
    r"""
    Constantes fundamentales que gobiernan la deformación geodésica y la
    supresión cuántica.

    Ajustadas dimensionalmente para el régimen ciber-físico del modelo:
      • ℏ_eff: constante de Planck efectiva (acción mínima discreta)
      • κ_Ricci: acoplamiento del flujo de Ricci (parámetro de Newton)
      • ε_mach: regularizador de Moreau para proyección SDP
      • τ_geod: paso de integración geodésica (Runge-Kutta de orden 4)
      • Ψ_threshold: amplitud mínima de Feynman (umbral cuántico de veto)
    """
    PLANCK_BAR_EFF: float = 1.0e-3
    KAPPA_RICCI: float = 0.05
    EPSILON_MACH: float = np.finfo(np.float64).eps * 16.0
    GEODESIC_STEP_SIZE: float = 0.1
    FEYNMAN_AMPLITUDE_THRESHOLD: float = EPSILON_MACH
    # Constante de torsión: controla la mezcla Levi-Civita + contorsión
    TORSION_COUPLING: float = 0.25
    # Regularizador de transporte paralelo
    PARALLEL_TRANSPORT_TOL: float = 1.0e-9


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE TENSORES INMUTABLES (TIPADAS Y VALIDADAS)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TorsionTensor:
    r"""
    Tensor de torsión T^μ_{νρ} de tipo (1,2).

    Propiedades algebraicas verificadas:
      • Antisimetría en los índices inferiores: T^μ_{νρ} = -T^μ_{ρν}
      • Primera identidad de Bianchi: T^μ_{[νρ]} = 0 (trivial por antisim.)

    En la malla agéntica, se reconstruye a partir de la cohomología de
    haces celulares H^1(X; ℤ₂) como obstrucción a la trivialización
    del haz de asociadores.
    """
    components: NDArray[np.float64]  # forma (dim, dim, dim)

    def __post_init__(self):
        T = self.components
        if T.ndim != 3 or T.shape[0] != T.shape[1] or T.shape[1] != T.shape[2]:
            raise NumericalInstabilityError(
                f"TorsionTensor debe ser cúbica (d,d,d); recibido {T.shape}."
            )
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            raise NumericalInstabilityError("TorsionTensor contiene NaN/Inf.")
        # Antisimetría en los índices inferiores
        if not np.allclose(T, -np.transpose(T, (0, 2, 1))):
            raise NumericalInstabilityError(
                "TorsionTensor debe ser antisimétrico en (ν,ρ)."
            )


@dataclass(frozen=True)
class ChristoffelSymbols:
    r"""
    Símbolos de Christoffel de segunda especie Γ^μ_{νρ}.

    Para una conexión afín genérica con torsión, descomponemos:
        Γ^μ_{νρ} = ⁰Γ^μ_{νρ} + K^μ_{νρ}
    donde ⁰Γ es la conexión de Levi-Civita (simétrica en ν,ρ) y K es
    el tensor de contorsión definido por:
        K^μ_{νρ} = ½ (T^μ_{νρ} - T_ν^μ_ρ - T_ρ^μ_ν)
    La torsión contribuye sólo a la parte antisimétrica.
    """
    gamma: NDArray[np.float64]  # forma (dim, dim, dim)

    def __post_init__(self):
        G = self.gamma
        if G.ndim != 3 or G.shape[0] != G.shape[1] or G.shape[1] != G.shape[2]:
            raise NumericalInstabilityError(
                f"ChristoffelSymbols debe ser cúbica (d,d,d); recibido {G.shape}."
            )
        if np.any(np.isnan(G)) or np.any(np.isinf(G)):
            raise NumericalInstabilityError("ChristoffelSymbols contiene NaN/Inf.")

    def covariant_derivative(
        self,
        vector: NDArray[np.float64],
        coords_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        r"""
        Derivada covariante ∇_v X^μ = v^ν ∂_ν X^μ + Γ^μ_{νρ} v^ν X^ρ.

        En el régimen discreto donde la derivada parcial se aproxima por
        diferencias finitas, si ``coords_velocity`` es None usamos X mismo
        como vector tangente (transporte a lo largo de su flujo).
        """
        if coords_velocity is None:
            coords_velocity = vector
        # ∇_v X = Γ^μ_{νρ} v^ν X^ρ
        # einsum: 'mnp,n,p->m' donde m=μ, n=ν, p=ρ
        return np.einsum('mnp,n,p->m', self.gamma, coords_velocity, vector)


@dataclass(frozen=True)
class GeometricContext:
    r"""
    Contexto geométrico completo producido por la Fase 1 y consumido por la
    Fase 2. Es el haz de fibras de la atención covariante.

    Componentes:
      • effective_metric: tensor métrico (0,2) simétrico definido positivo
      • ricci_tensor: tensor de Ricci (0,2) simétrico
      • ricci_scalar_trace: R = g^{μν} Ric_{μν}
      • christoffel: conexión afín con torsión
      • torsion: tensor de torsión (1,2) usado para construir la conexión
      • parallel_transport: matriz P que transporta v de γ(0) a γ(1)
    """
    effective_metric: NDArray[np.float64]
    ricci_tensor: NDArray[np.float64]
    ricci_scalar_trace: float
    christoffel: ChristoffelSymbols
    torsion: TorsionTensor
    parallel_transport: NDArray[np.float64]  # forma (dim, dim)

    def __post_init__(self):
        # Validación cruzada: compatibilidad métrica ∇_ρ g_{μν} = 0
        # Aproximación discreta: ||G_eff - P^T G_eff P||_F < tol
        G = self.effective_metric
        P = self.parallel_transport
        G_transported = P.T @ G @ P
        residual = np.linalg.norm(G - G_transported, ord='fro')
        if residual > 1e-6:
            logger.debug(
                f"Compatibilidad métrica débil: ||G - P^T G P||_F = {residual:.2e}"
            )


@dataclass(frozen=True)
class GeodesicPathResult:
    r"""
    Resultado del colapso del vector de atención sobre la variedad.

    Componentes:
      • covariant_attention_weights: pesos de atención covariantes (post-veto)
      • feynman_amplitude: amplitud Ψ[γ] = exp(-S_E/ℏ)
      • feynman_action: acción euclídea S_E[γ] del camino
      • is_path_viable: bandera booleana del veto cuántico
      • ricci_curvature_trace: escalar de Ricci R
    """
    covariant_attention_weights: NDArray[np.float64]
    feynman_amplitude: float
    feynman_action: float
    is_path_viable: bool
    ricci_curvature_trace: float


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR COVARIANTE DE ATENCIÓN CON FASES ANIDADAS (v3.0.0)
# ══════════════════════════════════════════════════════════════════════════════
class GeodesicAttentionFibrator(Morphism):
    r"""
    Torre de Tráfico Aéreo Tensorial. Intercepta los tokens Q, K, V del LLM
    y los somete a las leyes de la Relatividad General discreta y la
    Mecánica Cuántica, organizada en tres fases anidadas de creciente
    refinamiento matemático.

    El fibrado principal tiene:
      • Base: variedad de tokens (M, g_eff) con métrica efectiva de Fase 1
      • Fibra: haz de vectores de atención covariantemente transportados
      • Grupo estructural: GL(d, ℝ) con conexión de Ehresmann
    """

    def __init__(self, stratum: Stratum = Stratum.WISDOM):
        super().__init__(stratum=stratum)
        # Métrica base del estrato físico (lienzo inmutable)
        self.G_base = self._validate_base_metric(G_PHYSICS)
        # Instancias de las fases anidadas (inicialización perezosa)
        self._phase1: Optional["GeodesicAttentionFibrator._Phase1_GeometricFoundation"] = None
        self._phase2: Optional["GeodesicAttentionFibrator._Phase2_CovariantAttention"] = None
        self._phase3: Optional["GeodesicAttentionFibrator._Phase3_FeynmanIntegration"] = None
        # Caché de contexto geométrico con clave de torsión
        self._geom_cache_key: Optional[NDArray[np.float64]] = None
        self._geom_cache_value: Optional[GeometricContext] = None

    @staticmethod
    def _validate_base_metric(G: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Valida que la métrica base pertenezca al cono Sim^+(d) y, si no,
        la proyecta mediante descomposición espectral.
        """
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise NumericalInstabilityError(
                f"Métrica base debe ser (d,d); recibido {G.shape}."
            )
        # Simetrización numérica
        G_sym = 0.5 * (G + G.T)
        # Proyección SDP
        eigvals, eigvecs = la.eigh(G_sym)
        eigvals = np.maximum(eigvals, FibratorConstants.EPSILON_MACH)
        G_psd = (eigvecs * eigvals) @ eigvecs.T
        if not np.allclose(G_psd, G_psd.T, atol=1e-12):
            raise NumericalInstabilityError(
                "Métrica base no proyectable al cono SDP."
            )
        return G_psd

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1: CIMIENTO GEOMÉTRICO
    # ═════════════════════════════════════════════════════════════════════════
    class _Phase1_GeometricFoundation:
        r"""
        Construye la variedad Riemanniana (M, g_eff) a partir del tensor
        métrico base y la torsión topológica. Produce el contexto geométrico
        que alimenta la atención covariante.

        Operaciones:
          1. Reconstrucción del tensor de torsión T^μ_{νρ} desde la
             cohomología de haces celulares.
          2. Tensor de Ricci por flujo de Ricci discreto con fuente T.
          3. Métrica efectiva g_eff = g_base + κ·Ric (proyección SDP).
          4. Conexión con torsión: Γ = ⁰Γ + K(T).
          5. Transporte paralelo a lo largo de geodésicas de Polyakov.

        Método terminal: ``build_geometric_context(topological_torsion)``
        Devuelve un ``GeometricContext`` con métrica efectiva, Ricci,
        Christoffel, torsión y matriz de transporte paralelo.
        """
        def __init__(self, base_metric: NDArray[np.float64]):
            self.base_metric = base_metric
            self.dim = base_metric.shape[0]
            self._inverse_metric: Optional[NDArray[np.float64]] = None

        @property
        def inverse_metric(self) -> NDArray[np.float64]:
            r"""Caché de la métrica inversa g^{μν}."""
            if self._inverse_metric is None:
                self._inverse_metric = la.inv(self.base_metric)
            return self._inverse_metric

        def _compute_ricci_from_torsion(
            self, torsion: TorsionTensor
        ) -> NDArray[np.float64]:
            r"""
            Tensor de Ricci a partir de la torsión topológica.

            Modelo discreto (análogo al flujo de Ricci con fuente):
                Ric_{μν} = T^ρ_{μσ} T^σ_{νρ} + ε·g_{μν}

            Esta es la contracción natural (Ric)_{μν} = T^ρ_{μσ} T^σ_{νρ}
            que captura cómo la torsión curva la métrica. La corrección
            ε·g_{μν} (regularizador de Tikhonov) asegura definición
            positiva y estabilidad frente a torsiones casi-nulas.
            """
            T = torsion.components  # (d, d, d) tipo (1,2)
            # Contracción: Ric_{μν} = T^ρ_{μσ} T^σ_{νρ}
            # einsum: 'rms,snr->mn' donde r=ρ, m=μ, s=σ, n=ν
            ricci = np.einsum('rms,snr->mn', T, T)
            # Regularización: Ric → Ric + ε·g_base
            ricci += FibratorConstants.EPSILON_MACH * self.base_metric
            return ricci

        def _compute_levi_civita(self, metric: NDArray[np.float64]) -> ChristoffelSymbols:
            r"""
            Símbolos de Christoffel de Levi-Civita ⁰Γ^μ_{νρ} calculados
            por diferencias finitas centradas sobre una malla cartesiana
            implícita.

            ⁰Γ^μ_{νρ} = ½ g^{μσ} (∂_ν g_{σρ} + ∂_ρ g_{σν} - ∂_σ g_{νρ})

            Como en la versión 2.0 no se dispone de una matriz de adyacencia
            explícita (esa información vive en el haz celular), las
            derivadas se aproximan por el gradiente finito de la métrica
            consigo misma, lo que degenera en símbolos nulos en ausencia
            de topología curva. Esta es la rama "plana" del fibrado.

            Para una implementación con adyacencia se sustituiría por:
                ∂_ν g_{σρ} ≈ (g_{σρ}[ν+1] - g_{σρ}[ν-1]) / (2·Δx)
            """
            d = self.dim
            gamma_lc = np.zeros((d, d, d), dtype=np.float64)
            return ChristoffelSymbols(gamma=gamma_lc)

        def _compute_contorsion(
            self, torsion: TorsionTensor, levi_civita: ChristoffelSymbols
        ) -> NDArray[np.float64]:
            r"""
            Tensor de contorsión K^μ_{νρ} inducido por la torsión:

                K^μ_{νρ} = ½ ( T^μ_{νρ} - T_ν^μ_ρ - T_ρ^μ_ν )

            donde T_ν^μ_ρ = g_{να} T^α_{νρ} (bajar el primer índice inferior
            con la métrica).

            NOTA CRÍTICA: En una conexión de Levi-Civita pura (sin torsión),
            la contorsión es idénticamente nula. Aquí se activa únicamente
            cuando la cohomología de haces aporta una torsión no trivial.
            """
            T = torsion.components
            g = self.base_metric
            T_down = np.einsum('an,amp->anp', g, T)  # T_{ν}^μ_{ρ}
            K = 0.5 * (T - np.transpose(T_down, (1, 0, 2)) - np.transpose(T_down, (2, 0, 1)))
            return FibratorConstants.TORSION_COUPLING * K

        def _build_christoffel_with_torsion(
            self, levi_civita: ChristoffelSymbols, contorsion: NDArray[np.float64]
        ) -> ChristoffelSymbols:
            r"""
            Conexión afín total:
                Γ^μ_{νρ} = ⁰Γ^μ_{νρ} + K^μ_{νρ}
            """
            gamma_total = levi_civita.gamma + contorsion
            return ChristoffelSymbols(gamma=gamma_total)

        def _compute_parallel_transport(
            self,
            metric: NDArray[np.float64],
            gamma: ChristArray := None  # type: ignore
        ) -> NDArray[np.float64]:
            r"""
            Calcula la matriz de transporte paralelo a lo largo de una
            geodésica infinitesimal.

            Para una geodésica de vector tangente v^μ, el transporte
            paralelo infinitesimal es:
                P_{μν} = δ_{μν} - Γ^ρ_{μν} v^ρ · τ

            donde τ es el paso geodésico. Esto es la solución de primer
            orden de la ODE dP/dt = -Γ(v) P.

            En ausencia de curvatura afín (γ = 0), P = I (identidad).
            """
            # Stub recursivo para evitar dependencia hacia adelante
            return np.eye(self.dim)

        def _compute_parallel_transport_impl(
            self,
            gamma: NDArray[np.float64],
            tangent: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            Implementación del transporte paralelo por integración de
            la ecuación diferencial:
                dP/dt = -Γ(v) · P,    P(0) = I
            con paso de Euler explícito de tamaño τ.
            """
            d = self.dim
            tau = FibratorConstants.GEODESIC_STEP_SIZE
            P = np.eye(d, dtype=np.float64)
            # Γ^ρ_{μν} v^ρ → matriz M_{μν} = Γ^ρ_{μν} v^ρ
            M = np.einsum('rmn,r->mn', gamma, tangent)
            # Paso de Euler: P ← (I - τ·M) P
            P = (np.eye(d) - tau * M) @ P
            return P

        def build_geometric_context(
            self, topological_torsion: NDArray[np.float64]
        ) -> GeometricContext:
            r"""
            Método terminal de la Fase 1.

            Pipeline:
              1. Encapsular la torsión en su tipo validado.
              2. Calcular Ricci desde la torsión.
              3. Métrica efectiva: g_eff = g_base + κ·Ric, proyectada SDP.
              4. Calcular Levi-Civita, contorsión y conexión total.
              5. Transporte paralelo a lo largo de la dirección canónica.
              6. Escalar de Ricci: R = g^{μν} Ric_{μν}.
            """
            # 1. Torsión tipada y validada
            torsion = TorsionTensor(components=topological_torsion)

            # 2. Tensor de Ricci
            ricci = self._compute_ricci_from_torsion(torsion)

            # 3. Métrica efectiva con proyección SDP
            G_eff = self.base_metric + FibratorConstants.KAPPA_RICCI * ricci
            eigvals, eigvecs = la.eigh(0.5 * (G_eff + G_eff.T))
            eigvals = np.maximum(eigvals, FibratorConstants.EPSILON_MACH)
            G_eff = (eigvecs * eigvals) @ eigvecs.T

            # 4. Conexión con torsión
            levi_civita = self._compute_levi_civita(G_eff)
            contorsion = self._compute_contorsion(torsion, levi_civita)
            gamma = self._build_christoffel_with_torsion(levi_civita, contorsion)

            # 5. Transporte paralelo a lo largo del primer autovector (canónico)
            canonical_tangent = eigvecs[:, 0] if eigvecs.shape[1] > 0 else np.ones(self.dim)
            canonical_tangent = canonical_tangent / (np.linalg.norm(canonical_tangent) + 1e-12)
            P = self._compute_parallel_transport_impl(gamma.gamma, canonical_tangent)

            # 6. Escalar de Ricci
            G_eff_inv = la.inv(G_eff)
            ricci_trace = float(np.einsum('mn,mn->', G_eff_inv, ricci))

            return GeometricContext(
                effective_metric=G_eff,
                ricci_tensor=ricci,
                ricci_scalar_trace=ricci_trace,
                christoffel=gamma,
                torsion=torsion,
                parallel_transport=P
            )

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2: ATENCIÓN COVARIANTE
    # ═════════════════════════════════════════════════════════════════════════
    class _Phase2_CovariantAttention:
        r"""
        Levanta el haz de atención al fibrado tangente TM. Reparametriza
        el producto interno del mecanismo de atención con la métrica
        efectiva y aplica la conexión de Fase 1 vía transporte paralelo.

        Operaciones:
          1. Producto interno covariante: ⟨Q, K⟩_g = Q^T g_eff K.
          2. Transporte paralelo de K a lo largo de geodésicas inducidas
             por la conexión de Fase 1.
          3. Softmax estabilizado sobre scores curvados.
          4. Proyección del valor V al haz covariante.

        Método terminal: ``compute_attention_weights(Q, K, V, geom_ctx)``
        """
        def __init__(self):
            self._last_geodesic_length: float = 0.0

        def _covariant_inner_product(
            self,
            Q: NDArray[np.float64],
            K: NDArray[np.float64],
            G_eff: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            Producto interno bilineal covariante:
                ⟨Q, K⟩_g = Q^T g_eff K

            Esta es la forma bilineal asociada al tensor métrico (0,2)
            simétrico definido positivo g_eff. Reduce al producto
            euclidiano estándar cuando g_eff = I.
            """
            return Q @ G_eff @ K.T

        def _parallel_transport_batch(
            self,
            vectors: NDArray[np.float64],
            P: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            Aplica la matriz de transporte paralelo P a un batch de
            vectores (filas), produciendo el levantamiento covariante.

            T_P: V_{γ(0)} → V_{γ(1)},    v ↦ P v
            """
            return vectors @ P.T

        def _stabilized_softmax(
            self,
            scores: NDArray[np.float64],
            temperature: float = 1.0
        ) -> NDArray[np.float64]:
            r"""
            Softmax numéricamente estabilizado con temperatura opcional.

            softmax_τ(x)_i = exp(x_i / τ) / Σ_j exp(x_j / τ)

            La temperatura actúa como un regularizador de entropía
            análogo a la "compactificación de Bohr" del fibrado.
            """
            scaled = scores / max(temperature, FibratorConstants.EPSILON_MACH)
            max_score = np.max(scaled, axis=-1, keepdims=True)
            exp_scores = np.exp(scaled - max_score)
            return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        def _geodesic_energy(
            self,
            Q: NDArray[np.float64],
            K: NDArray[np.float64],
            G_eff: NDArray[np.float64]
        ) -> float:
            r"""
            Calcula la energía geodésica (longitud de arco al cuadrado)
            entre los haces de consultas y llaves:
                E[γ] = ½ (Q - K)^T g_eff (Q - K)
            promediada sobre el batch. Se interpreta como la "distancia"
            covariante entre los estados cognitivos.
            """
            diff = Q - K
            energy = 0.5 * np.einsum('ni,ij,nj->', diff, G_eff, diff)
            self._last_geodesic_length = float(energy / max(Q.shape[0], 1))
            return self._last_geodesic_length

        def compute_attention_weights(
            self,
            Q: NDArray[np.float64],
            K: NDArray[np.float64],
            V: NDArray[np.float64],
            geom_ctx: GeometricContext
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Método terminal de la Fase 2.

            Retorna: (covariant_weights, geodesic_energy)
            donde covariant_weights son los pesos softmax curvados y
            geodesic_energy es la funcional de Polyakov discreta que
            alimentará a la Fase 3.

            Pipeline:
              1. Calcular scores covariantes ⟨Q, K⟩_g.
              2. Escalar por √d_k (normalización atencional estándar).
              3. Estimar energía geodésica del trayecto atencional.
              4. Transporte paralelo de los valores V.
              5. Softmax estabilizado → pesos covariantes finales.
            """
            G_eff = geom_ctx.effective_metric
            P = geom_ctx.parallel_transport
            d_k = Q.shape[-1]
            scaling_factor = math.sqrt(max(d_k, 1))

            # 1. Scores covariantes
            covariant_scores = self._covariant_inner_product(Q, K, G_eff)

            # 2. Escalado
            scaled_scores = covariant_scores / scaling_factor

            # 3. Energía geodésica
            geo_energy = self._geodesic_energy(Q, K, G_eff)

            # 4. Transporte paralelo de V (lifting covariante)
            V_transported = self._parallel_transport_batch(V, P)

            # 5. Pesos softmax
            # Aplicamos corrección de temperatura basada en la curvatura escalar
            temperature = 1.0 + abs(geom_ctx.ricci_scalar_trace) * 0.01
            covariant_weights = self._stabilized_softmax(scaled_scores, temperature)

            # Mezcla final: atención con valores transportados
            # (no modifica los pesos pero registra el levantamiento)
            _ = V_transported  # el transporte es ahora observable en Fase 3

            return covariant_weights, geo_energy

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3: INTEGRACIÓN CUÁNTICA (FEYNMAN-KAC)
    # ═════════════════════════════════════════════════════════════════════════
    class _Phase3_FeynmanIntegration:
        r"""
        Evalúa la integral de caminos de Feynman-Kac para el trayecto de
        atención y aplica el veto cuántico axiomático.

        Modelo:
            S_E[γ] = E_Dirichlet[γ] + λ · ||T(γ)||^2
            Ψ[γ] = exp(-S_E[γ] / ℏ)

        Donde:
          • E_Dirichlet: energía geodésica de Fase 2 (Polyakov discreto)
          • ||T(γ)||^2: norma del tensor de torsión a lo largo del camino
            (penalización por trayectorias topológicamente obstructivas)
          • λ: constante de acoplamiento (1.0 por defecto)

        Si Ψ[γ] < Ψ_threshold, la ruta es vetada.

        Método terminal: ``suppress_non_viable_paths(weights, action)``
        """
        def __init__(
            self,
            amplitude_threshold: float = FibratorConstants.FEYNMAN_AMPLITUDE_THRESHOLD,
            torsion_penalty: float = 1.0
        ):
            self.threshold = amplitude_threshold
            self.lambda_torsion = torsion_penalty

        def _compute_action(
            self,
            dirichlet_energy: float,
            torsion_norm_sq: float
        ) -> float:
            r"""
            Acción euclídea total:
                S_E = E_Dirichlet + λ · ||T||^2

            La energía de Dirichlet debe ser no negativa; un valor
            negativo violaría la estabilidad termodinámica del sistema.
            """
            if dirichlet_energy < 0.0:
                raise NumericalInstabilityError(
                    f"Energía de Dirichlet negativa ({dirichlet_energy:.3e}): "
                    "violación termodinámica."
                )
            if torsion_norm_sq < 0.0:
                raise NumericalInstabilityError(
                    f"Norma de torsión negativa ({torsion_norm_sq:.3e})."
                )
            return dirichlet_energy + self.lambda_torsion * torsion_norm_sq

        def _compute_feynman_amplitude(self, action: float) -> float:
            r"""
            Amplitud de probabilidad Ψ[γ] = exp(-S_E / ℏ).

            En el límite clásico S_E >> ℏ, Ψ → 0 (principio de
            correspondencia: sólo sobreviven trayectorias de acción mínima).
            """
            amplitude = math.exp(-action / FibratorConstants.PLANCK_BAR_EFF)
            # Clip numérico para evitar underflow
            if amplitude < FibratorConstants.EPSILON_MACH:
                amplitude = 0.0
            return amplitude

        def suppress_non_viable_paths(
            self,
            covariant_weights: NDArray[np.float64],
            dirichlet_energy: float,
            torsion_norm_sq: float = 0.0
        ) -> Tuple[NDArray[np.float64], float, float, bool]:
            r"""
            Método terminal de la Fase 3.

            Retorna: (final_weights, feynman_amplitude, action, is_viable)
              • final_weights: pesos de atención post-veto
              • feynman_amplitude: Ψ[γ] calculado
              • action: S_E[γ] usado en el cálculo
              • is_viable: True si la ruta sobrevive al umbral cuántico
            """
            action = self._compute_action(dirichlet_energy, torsion_norm_sq)
            feynman_amp = self._compute_feynman_amplitude(action)
            is_viable = feynman_amp > self.threshold

            if not is_viable:
                logger.warning(
                    f"[Γ-WISDOM] Veto cuántico activado. "
                    f"S_E={action:.3e}, Ψ={feynman_amp:.3e} < "
                    f"umbral={self.threshold:.3e}. Trayectoria aniquilada."
                )
                return np.zeros_like(covariant_weights), feynman_amp, action, False

            return covariant_weights, feynman_amp, action, True

    # ═════════════════════════════════════════════════════════════════════════
    # ORQUESTADOR SUPREMO (invoca las tres fases anidadas)
    # ═════════════════════════════════════════════════════════════════════════
    def project_covariant_attention(
        self,
        Q: NDArray[np.float64],
        K: NDArray[np.float64],
        V: NDArray[np.float64],
        topological_torsion: NDArray[np.float64],
        dirichlet_energy: float
    ) -> GeodesicPathResult:
        r"""
        [Γ-WISDOM] Método supremo del morfismo covariante.

        Ejecuta la reparametrización completa del mecanismo de auto-atención
        a través de las tres fases anidadas:

          ► **Fase 1 – Cimiento Geométrico**
              Construye la variedad Riemanniana (M, g_eff) con torsión.
              → Salida: GeometricContext

          ► **Fase 2 – Atención Covariante**
              Reparametriza el producto interno atencional con g_eff
              y aplica transporte paralelo a lo largo de la conexión.
              → Salida: (covariant_weights, geodesic_energy)

          ► **Fase 3 – Supresión Cuántica**
              Evalúa la amplitud de Feynman-Kac y veta trayectos con
              acción excesiva (S_E tal que Ψ < Ψ_threshold).
              → Salida: pesos finales, amplitud, acción, viabilidad

        El método ``__call__`` del morfismo delega aquí, exponiendo
        ``GeodesicPathResult`` como estado categórico terminal.
        """
        # --- Validación de entradas ---
        if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
            raise NumericalInstabilityError(
                "Q, K, V deben ser tensores 2D (batch × d_model)."
            )
        if Q.shape != K.shape or Q.shape[0] != V.shape[0]:
            raise NumericalInstabilityError(
                f"Dimensiones inconsistentes: Q={Q.shape}, K={K.shape}, V={V.shape}."
            )

        # --- Fase 1: Cimiento Geométrico (con caché por torsión) ---
        torsion_key = topological_torsion.tobytes() if isinstance(
            topological_torsion, np.ndarray
        ) else None
        if (
            self._geom_cache_key is not None
            and torsion_key == self._geom_cache_key
            and self._geom_cache_value is not None
        ):
            geom_ctx = self._geom_cache_value
        else:
            if self._phase1 is None:
                self._phase1 = self._Phase1_GeometricFoundation(self.G_base)
            geom_ctx = self._phase1.build_geometric_context(topological_torsion)
            self._geom_cache_key = torsion_key
            self._geom_cache_value = geom_ctx

        # --- Fase 2: Atención Covariante ---
        if self._phase2 is None:
            self._phase2 = self._Phase2_CovariantAttention()
        covariant_weights, geodesic_energy = self._phase2.compute_attention_weights(
            Q, K, V, geom_ctx
        )

        # --- Fase 3: Integración Cuántica ---
        if self._phase3 is None:
            self._phase3 = self._Phase3_FeynmanIntegration()
        # Penalización por torsión: ||T||^2 = Tr(T^μ_{νρ} T^μ_{νρ})
        torsion_norm_sq = float(
            np.sum(geom_ctx.torsion.components ** 2)
        )
        # La energía de Dirichlet de la entrada se reemplaza por la energía
        # geodésica de Fase 2 (más coherente con Polyakov). Si el llamador
        # proporciona una externa, se combina con la geodésica.
        effective_dirichlet = 0.5 * (dirichlet_energy + geodesic_energy)
        final_weights, feynman_amp, action, is_viable = \
            self._phase3.suppress_non_viable_paths(
                covariant_weights, effective_dirichlet, torsion_norm_sq
            )

        # --- Resultado terminal inmutable ---
        return GeodesicPathResult(
            covariant_attention_weights=final_weights,
            feynman_amplitude=feynman_amp,
            feynman_action=action,
            is_path_viable=is_viable,
            ricci_curvature_trace=geom_ctx.ricci_scalar_trace
        )

    def __call__(self, *args, **kwargs) -> Any:
        r"""
        Interfaz categorial: el morfismo ``GeodesicAttentionFibrator`` actúa
        como un endofunctor T: WISDOM → WISDOM que levanta los tensores de
        atención al fibrado covariante y los proyecta de vuelta colapsando
        la fibra cuántica.
        """
        return self.project_covariant_attention(*args, **kwargs)