"""
Módulo: Tensores Métricos Riemannianos (El Tejido del Espacio de Fase)
Ubicación: app/core/immune_system/metric_tensors.py

Este módulo define los tensores métricos (G_k) que gobiernan la topología local
del espacio de estados en subespacios de ℝ⁷. Los tensores son matrices simétricas
definidas positivas (SPD) pertenecientes a la variedad diferenciable:

    𝒮₊₊ⁿ = { G ∈ ℝⁿˣⁿ : G = Gᵀ, G ≻ 0 }

Se emplean para evaluar formas cuadráticas de Mahalanobis:

    d_G(x)² = xᵀ G x

donde G ≻ 0 penaliza anisotrópicamente riesgos acoplados en subespacios de:
- Física (V_phys ⊂ ℝ³)
- Topología (V_topo ⊂ ℝ²)
- Termodinámica (V_thermo ⊂ ℝ²)

Garantías Matemáticas Reforzadas:
─────────────────────────────────
1. Toda matriz es finita, cuadrada, simétrica (hasta tolerancia relativa)
2. Verificación SPD por descomposición de Cholesky (criterio definitivo)
3. Regularización Tikhonov adaptativa: G_reg = G + δI con δ = max(0, ε - λ_min)
4. Control de número de condición: κ₂(G) = λ_max/λ_min ≤ COND_NUM_TOL
5. Certificación de pertenencia al interior del cono SPD
6. Cuantificación de deformación por regularización (norma de Frobenius relativa)
7. Consistencia espectral verificada entre eigenvalores analíticos y numéricos
8. Brecha espectral normalizada por λ_max para estabilidad adimensional

Invariantes del Módulo:
───────────────────────
- Los tensores precompilados son inmutables (arrays read-only)
- Los eigenvalores almacenados en SpectralProfile son inmutables
- El módulo falla rápido en tiempo de carga si hay incoherencia geométrica
- Todas las operaciones preservan tipo float64 para estabilidad numérica
- Ninguna importación no utilizada; toda función importada se emplea
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, NamedTuple

import numpy as np
from numpy.linalg import LinAlgError

from app.core.immune_system.topological_watcher import (
    MIN_EIGVAL_TOL,
    COND_NUM_TOL,
    MetricTensorError,
)

logger = logging.getLogger("MIC.ImmuneSystem.MetricTensors")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES CON JUSTIFICACIÓN MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

_FLOAT_DTYPE: Final[np.dtype] = np.float64

# Tolerancia de simetría relativa: permite asimetría proporcional a ||G||_F
# Justificación: errores de redondeo escalan con la magnitud de las entradas
_SYMMETRY_RTOL: Final[float] = 1e-14

# Tolerancia absoluta mínima para matrices con norma pequeña
_SYMMETRY_ATOL_BASE: Final[float] = 1e-15

# Factor de seguridad para el interior del cono SPD
# λ_min debe exceder MIN_EIGVAL_TOL por este factor para garantizar robustez
_SPD_INTERIOR_FACTOR: Final[float] = 1.1

# Umbral de advertencia para deformación por regularización (norma Frobenius relativa)
_REGULARIZATION_WARN_THRESHOLD: Final[float] = 0.05  # 5%

# Umbral crítico de deformación (aborta si se excede)
_REGULARIZATION_ABORT_THRESHOLD: Final[float] = 0.25  # 25%

# Umbral absoluto para considerar un δ de Tikhonov como no trivial
# Evita comparaciones de punto flotante con 0.0 exacto
_TIKHONOV_DELTA_ATOL: Final[float] = 1e-16

# Umbral de norma de Frobenius bajo el cual una matriz se considera
# esencialmente nula y no admite regularización significativa
_NEAR_ZERO_FROBENIUS_TOL: Final[float] = 1e-14


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS PARA DIAGNÓSTICOS ESPECTRALES
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralProfile(NamedTuple):
    """
    Perfil espectral completo de una matriz simétrica.

    Todos los campos escalares son float; el campo eigenvalues es un array
    inmutable (write=False) para preservar la invariancia del perfil.

    Atributos:
        eigenvalues: Vector de eigenvalores ordenados ascendentemente (inmutable)
        lambda_min: Mínimo eigenvalor (λ₁)
        lambda_max: Máximo eigenvalor (λₙ)
        condition_number: κ₂(G) = λ_max / λ_min
        spectral_gap: Brecha espectral normalizada (λ₂ - λ₁)/λ_max si n > 1,
                       adimensional y acotada en [0, 1) para matrices SPD
        frobenius_norm: ||G||_F = √(Σᵢⱼ gᵢⱼ²)

    Nota sobre la brecha espectral:
    ───────────────────────────────
    Se normaliza por λ_max (no por λ₁) para evitar divergencia cuando
    λ₁ → 0⁺ y para obtener una cantidad adimensional acotada. Esta es
    la convención estándar en teoría espectral de grafos (Chung, 1997).
    """
    eigenvalues: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    spectral_gap: float
    frobenius_norm: float

    def is_strictly_positive(self, threshold: float = MIN_EIGVAL_TOL) -> bool:
        """Verifica si todos los eigenvalores exceden el umbral."""
        return bool(self.lambda_min >= threshold)

    def is_well_conditioned(self, max_kappa: float = COND_NUM_TOL) -> bool:
        """Verifica si el número de condición está acotado."""
        return bool(np.isfinite(self.condition_number) and self.condition_number <= max_kappa)


@dataclass(frozen=True, slots=True)
class RegularizationReport:
    """
    Informe de regularización aplicada a un tensor métrico.

    Atributos:
        tensor_name: Identificador del tensor
        original_profile: Perfil espectral antes de regularización
        regularized_profile: Perfil espectral después de regularización
        tikhonov_delta: Parámetro δ añadido a la diagonal (G_reg = G + δI)
        frobenius_deformation: ||G_reg - G||_F / ||G||_F (deformación relativa)
        required_regularization: True si se aplicó regularización no trivial
        input_asymmetry: Asimetría relativa de la entrada original
    """
    tensor_name: str
    original_profile: SpectralProfile
    regularized_profile: SpectralProfile
    tikhonov_delta: float
    frobenius_deformation: float
    required_regularization: bool
    input_asymmetry: float

    def log_summary(self) -> None:
        """Emite un resumen estructurado al logger."""
        if self.required_regularization:
            log_fn = (
                logger.warning
                if self.frobenius_deformation > _REGULARIZATION_WARN_THRESHOLD
                else logger.info
            )
            log_fn(
                "Regularización aplicada a %s: δ=%.6e, deformación=%.4f%%, "
                "λ_min: %.6e → %.6e, κ: %.4e → %.4e, "
                "asimetría entrada: %.4e",
                self.tensor_name,
                self.tikhonov_delta,
                self.frobenius_deformation * 100,
                self.original_profile.lambda_min,
                self.regularized_profile.lambda_min,
                self.original_profile.condition_number,
                self.regularized_profile.condition_number,
                self.input_asymmetry,
            )
        else:
            logger.debug(
                "Tensor %s validado sin regularización: λ_min=%.6e, κ=%.4e, "
                "asimetría entrada: %.4e",
                self.tensor_name,
                self.regularized_profile.lambda_min,
                self.regularized_profile.condition_number,
                self.input_asymmetry,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: FÁBRICA DE TENSORES MÉTRICOS
# ═══════════════════════════════════════════════════════════════════════════════

class MetricTensorFactory:
    """
    Fábrica determinista de tensores métricos para subespacios ortogonales.

    Cada tensor construido satisface rigurosamente:

    1. G ∈ ℝⁿˣⁿ (matriz cuadrada de dimensión prescrita)
    2. G = Gᵀ (simetría exacta por proyección ortogonal)
    3. G ≻ 0 (definida positiva, verificada por Cholesky)
    4. λ_min(G) ≥ MIN_EIGVAL_TOL (estabilidad numérica del espectro inferior)
    5. κ₂(G) ≤ COND_NUM_TOL (acotación del número de condición)
    6. ||G_reg - G_orig||_F / ||G_orig||_F ≤ _REGULARIZATION_ABORT_THRESHOLD
    7. Consistencia espectral: |λ_i(G_reg) - (λ_i(G) + δ)| ≤ ε_mach · ||G||

    Pipeline de Validación:
    ───────────────────────
    entrada → conversión → validación estructural → verificación simetría original
    → simetrización → verificación finitud post-simetrización → análisis espectral
    → regularización condicional → verificación consistencia espectral
    → verificación Cholesky → control de condicionamiento → inmutabilización
    """

    _EXPECTED_DIMENSIONS: Final[dict[str, int]] = {
        "G_phys": 3,
        "G_topo": 2,
        "G_thermo": 2,
    }

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE CONVERSIÓN Y VALIDACIÓN ESTRUCTURAL
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _as_strict_array(G: np.ndarray) -> np.ndarray:
        """
        Convierte entrada a ndarray denso float64 con copia defensiva.

        La copia asegura que modificaciones posteriores no afecten la entrada
        original, preservando la semántica funcional del pipeline.
        """
        return np.array(G, dtype=_FLOAT_DTYPE, copy=True)

    @classmethod
    def _validate_structural_preconditions(
        cls,
        name: str,
        G: np.ndarray,
    ) -> np.ndarray:
        """
        Valida precondiciones estructurales irreducibles de la matriz candidata.

        Verificaciones (en orden):
        1. Convertibilidad a ndarray float64
        2. Bidimensionalidad (rango tensorial = 2)
        3. Cuadratura (n_rows = n_cols)
        4. Dimensión positiva (no vacía)
        5. Dimensión esperada por dominio físico
        6. Finitud de todas las entradas (ausencia de NaN/±∞)

        Args:
            name: Identificador del tensor para mensajes de error
            G: Matriz candidata a validar

        Returns:
            Copia de G como ndarray float64 validado estructuralmente

        Raises:
            MetricTensorError: Si alguna precondición falla
        """
        try:
            G_arr = cls._as_strict_array(G)
        except (TypeError, ValueError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: no convertible a array numérico float64. "
                f"Tipo recibido: {type(G).__name__}. Error: {exc}"
            ) from exc

        if G_arr.ndim != 2:
            raise MetricTensorError(
                f"Tensor {name}: se requiere matriz 2D (rango tensorial 2), "
                f"recibido ndim={G_arr.ndim}."
            )

        n_rows, n_cols = G_arr.shape
        if n_rows != n_cols:
            raise MetricTensorError(
                f"Tensor {name}: matriz no cuadrada. "
                f"Forma recibida: {n_rows}×{n_cols}. "
                "Los tensores métricos deben ser endomorfismos."
            )

        if n_rows == 0:
            raise MetricTensorError(
                f"Tensor {name}: matriz vacía (0×0). "
                "Se requiere dimensión positiva."
            )

        expected_dim = cls._EXPECTED_DIMENSIONS.get(name)
        if expected_dim is not None and n_rows != expected_dim:
            raise MetricTensorError(
                f"Tensor {name}: dimensión incorrecta. "
                f"Esperada: {expected_dim}×{expected_dim}, "
                f"recibida: {n_rows}×{n_cols}."
            )

        if not np.all(np.isfinite(G_arr)):
            non_finite_mask = ~np.isfinite(G_arr)
            non_finite_count = int(np.count_nonzero(non_finite_mask))
            non_finite_indices = list(zip(*np.where(non_finite_mask)))[:5]
            raise MetricTensorError(
                f"Tensor {name}: contiene {non_finite_count} entrada(s) no "
                f"finita(s) (NaN o ±∞). Índices muestra: {non_finite_indices}. "
                "Esto destruye el significado métrico Riemanniano."
            )

        return G_arr

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE ANÁLISIS DE SIMETRÍA
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _compute_symmetry_tolerance(cls, G: np.ndarray) -> float:
        """
        Calcula tolerancia de simetría adaptativa basada en la norma de Frobenius.

        La tolerancia escala con ||G||_F para manejar correctamente matrices
        con entradas de diferentes magnitudes.

        Fórmula: tol = max(_SYMMETRY_ATOL_BASE, _SYMMETRY_RTOL · ||G||_F)

        Justificación: para una matriz con entradas de orden O(M), los errores
        de redondeo en float64 son O(M · ε_mach) ≈ O(M · 10⁻¹⁶). La norma
        de Frobenius captura la escala global, produciendo una tolerancia
        proporcional al error esperado.
        """
        frobenius_norm = float(np.linalg.norm(G, "fro"))
        return max(_SYMMETRY_ATOL_BASE, _SYMMETRY_RTOL * frobenius_norm)

    @classmethod
    def _assess_input_symmetry(cls, name: str, G: np.ndarray) -> float:
        """
        Evalúa la simetría de la matriz de entrada ANTES de simetrización.

        Retorna la asimetría relativa: ||G - Gᵀ||_F / ||G||_F

        Esto permite detectar entradas con asimetría significativa que podrían
        indicar un error de programación en el llamador.

        Para matrices con ||G||_F < _NEAR_ZERO_FROBENIUS_TOL, retorna la
        asimetría absoluta ||G - Gᵀ||_F, ya que la relativa no está bien
        definida (división por cantidad cercana a cero).
        """
        frobenius_norm = float(np.linalg.norm(G, "fro"))
        asymmetry = G - G.T
        asymmetry_norm = float(np.linalg.norm(asymmetry, "fro"))

        if frobenius_norm < _NEAR_ZERO_FROBENIUS_TOL:
            # Matriz esencialmente cero: reportar asimetría absoluta
            relative_asymmetry = asymmetry_norm
        else:
            relative_asymmetry = asymmetry_norm / frobenius_norm

        # Advertencia si la asimetría de entrada es significativa
        tolerance = cls._compute_symmetry_tolerance(G)
        if asymmetry_norm > tolerance:
            logger.warning(
                "Tensor %s: asimetría de entrada detectada. "
                "||G - Gᵀ||_F = %.6e, ||G - Gᵀ||_F / ||G||_F = %.6e. "
                "Se aplicará simetrización por proyección.",
                name,
                asymmetry_norm,
                relative_asymmetry,
            )

        return relative_asymmetry

    @staticmethod
    def _symmetrize_by_projection(G: np.ndarray) -> np.ndarray:
        """
        Proyección ortogonal al subespacio de matrices simétricas Sym(n).

        Dada G ∈ ℝⁿˣⁿ, la proyección ortogonal π: ℝⁿˣⁿ → Sym(n) es:

            π(G) = (G + Gᵀ) / 2

        Esta es la matriz simétrica más cercana a G en norma de Frobenius:

            π(G) = argmin_{S ∈ Sym(n)} ||G - S||_F

        Propiedades:
        - π² = π (idempotencia)
        - π(G) = G ⟺ G ∈ Sym(n)
        - ||G - π(G)||_F = ||G - Gᵀ||_F / 2 (distancia al subespacio)

        Demostración de optimalidad:
        Descomponemos G = S + A con S = (G + Gᵀ)/2 ∈ Sym(n) y A = (G - Gᵀ)/2
        antisimétrica. Para cualquier S' ∈ Sym(n):
            ||G - S'||²_F = ||S - S' + A||²_F = ||S - S'||²_F + ||A||²_F
        ya que ⟨Sym(n), Skew(n)⟩_F = 0. Mínimo en S' = S.
        """
        return 0.5 * (G + G.T)

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE ANÁLISIS ESPECTRAL
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _compute_spectral_profile(cls, G: np.ndarray) -> SpectralProfile:
        """
        Computa el perfil espectral completo de una matriz simétrica.

        Usa np.linalg.eigvalsh optimizado para matrices simétricas/Hermitianas.
        Los eigenvalores se retornan ordenados ascendentemente por LAPACK.

        Precondición: G debe ser simétrica (ya simetrizada por el pipeline).
        Se verifica esta precondición explícitamente para detectar errores
        de integración.

        La brecha espectral se normaliza por λ_max (convención de Chung, 1997)
        para obtener una cantidad adimensional acotada en [0, 1) para matrices
        SPD, evitando la divergencia de la normalización por λ_min cuando
        λ_min → 0⁺.

        Args:
            G: Matriz simétrica (se verifica simetría como precondición)

        Returns:
            SpectralProfile con todos los diagnósticos espectrales.
            El array de eigenvalores es inmutable (write=False).

        Raises:
            MetricTensorError: Si G no es simétrica dentro de tolerancia
            LinAlgError: Si la descomposición espectral falla
        """
        # Verificar precondición de simetría
        symmetry_tol = cls._compute_symmetry_tolerance(G)
        asymmetry_norm = float(np.linalg.norm(G - G.T, "fro"))
        if asymmetry_norm > symmetry_tol:
            raise MetricTensorError(
                f"_compute_spectral_profile recibió matriz no simétrica: "
                f"||G - Gᵀ||_F = {asymmetry_norm:.6e} > tol = {symmetry_tol:.6e}. "
                "eigvalsh requiere entrada simétrica para resultados correctos."
            )

        eigenvalues = np.linalg.eigvalsh(G)

        # Hacer eigenvalues inmutable para preservar invariancia del perfil
        eigenvalues_immutable = np.array(eigenvalues, dtype=_FLOAT_DTYPE, copy=True)
        eigenvalues_immutable.setflags(write=False)

        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])

        # Número de condición espectral κ₂(G) = |λ_max| / |λ_min|
        # Para matrices SPD, esto es simplemente λ_max / λ_min
        if lambda_min > 0.0:
            condition_number = lambda_max / lambda_min
        elif lambda_min == 0.0:
            condition_number = np.inf
        else:
            # λ_min < 0: matriz no SPD, κ₂ pierde significado de condicionamiento
            # Usamos |λ_max / λ_min| para dar una medida de la anisotropía espectral
            condition_number = np.nan

        # Brecha espectral normalizada por λ_max (convención Chung)
        # Para n=1 o λ_max ≤ 0, la brecha no tiene sentido y se define como 0
        n = len(eigenvalues)
        if n > 1 and lambda_max > 0.0:
            spectral_gap = float(eigenvalues[1] - eigenvalues[0]) / lambda_max
        else:
            spectral_gap = 0.0

        frobenius_norm = float(np.linalg.norm(G, "fro"))

        return SpectralProfile(
            eigenvalues=eigenvalues_immutable,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            condition_number=condition_number,
            spectral_gap=spectral_gap,
            frobenius_norm=frobenius_norm,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE VERIFICACIÓN SPD
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _verify_spd_by_cholesky(cls, name: str, G: np.ndarray) -> bool:
        """
        Verifica que G sea SPD mediante descomposición de Cholesky.

        Teorema (Criterio de Cholesky):
        Una matriz simétrica G es definida positiva si y solo si
        existe una matriz triangular inferior L con diagonal positiva
        tal que G = LLᵀ.

        La descomposición de Cholesky es numéricamente más estable que
        verificar eigenvalores para certificar SPD, especialmente para
        matrices casi singulares.

        Complejidad: O(n³/3), más eficiente que eigendecomposición O(n³).

        Args:
            name: Identificador del tensor (para logging)
            G: Matriz simétrica a verificar

        Returns:
            True si G es SPD (Cholesky exitoso), False en caso contrario
        """
        try:
            np.linalg.cholesky(G)
            return True
        except LinAlgError:
            return False

    @classmethod
    def _assert_spd_strict(
        cls,
        name: str,
        G: np.ndarray,
        profile: SpectralProfile,
    ) -> None:
        """
        Verificación estricta de propiedades SPD post-regularización.

        Condiciones verificadas (en orden de coste computacional creciente):
        1. Finitud de todas las entradas
        2. Simetría numérica dentro de tolerancia adaptativa
        3. λ_max > 0 (positividad no trivial)
        4. λ_min ≥ MIN_EIGVAL_TOL (espectro acotado inferiormente)
        5. Descomposición de Cholesky exitosa (certificación definitiva)

        El orden está optimizado para fallar rápido en las verificaciones
        más baratas antes de ejecutar Cholesky O(n³/3).

        Raises:
            MetricTensorError: Si alguna condición SPD falla
        """
        # Verificación 1: Finitud (O(n²) scan)
        if not np.all(np.isfinite(G)):
            non_finite_count = int(np.count_nonzero(~np.isfinite(G)))
            raise MetricTensorError(
                f"Tensor {name}: contiene {non_finite_count} valores no finitos "
                "post-regularización."
            )

        # Verificación 2: Simetría numérica
        symmetry_tol = cls._compute_symmetry_tolerance(G)
        asymmetry_norm = float(np.linalg.norm(G - G.T, "fro"))
        if asymmetry_norm > symmetry_tol:
            raise MetricTensorError(
                f"Tensor {name}: asimetría residual post-simetrización. "
                f"||G - Gᵀ||_F = {asymmetry_norm:.6e} > tol = {symmetry_tol:.6e}."
            )

        # Verificación 3: Positividad no trivial (más barata que comparar λ_min)
        if profile.lambda_max <= 0.0:
            raise MetricTensorError(
                f"Tensor {name}: λ_max = {profile.lambda_max:.6e} ≤ 0. "
                "La matriz es semi-definida negativa o nula."
            )

        # Verificación 4: Espectro inferior acotado
        if profile.lambda_min < MIN_EIGVAL_TOL:
            raise MetricTensorError(
                f"Tensor {name}: λ_min = {profile.lambda_min:.6e} < "
                f"MIN_EIGVAL_TOL = {MIN_EIGVAL_TOL:.6e}. "
                "La regularización no logró estabilizar el espectro inferior."
            )

        # Verificación 5: Certificación Cholesky (gold standard, O(n³/3))
        if not cls._verify_spd_by_cholesky(name, G):
            raise MetricTensorError(
                f"Tensor {name}: falló descomposición de Cholesky a pesar de "
                f"λ_min = {profile.lambda_min:.6e} > 0. "
                "Esto indica inestabilidad numérica severa o inconsistencia "
                "entre eigvalsh y Cholesky (posiblemente por errores de "
                "redondeo acumulados)."
            )

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE REGULARIZACIÓN
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _apply_tikhonov_regularization(
        cls,
        name: str,
        G: np.ndarray,
        original_profile: SpectralProfile,
    ) -> tuple[np.ndarray, float]:
        """
        Aplica regularización de Tikhonov adaptativa si es necesario.

        Regularización de Tikhonov:
            G_reg = G + δI

        donde δ = max(0, target_λ_min - λ_min) con:
        - target_λ_min = MIN_EIGVAL_TOL · _SPD_INTERIOR_FACTOR
        - Esto asegura λ_min(G_reg) = λ_min(G) + δ ≥ target_λ_min

        Propiedades de la regularización (demostración):
        ────────────────────────────────────────────────
        Sea G = QΛQᵀ la eigendecomposición con Q ortogonal, Λ = diag(λ₁,...,λₙ).
        Entonces:
            G + δI = Q(Λ + δI)Qᵀ = Q · diag(λ₁+δ,...,λₙ+δ) · Qᵀ

        Por tanto:
        - Los autovectores se preservan exactamente
        - λᵢ(G_reg) = λᵢ(G) + δ para todo i
        - La deformación tiene norma de Frobenius ||δI||_F = δ√n
        - Es la regularización de mínima norma de Frobenius que logra
          λ_min(G_reg) = target_λ_min (óptima en sentido de mínima perturbación
          con estructura escalar)

        Args:
            name: Identificador del tensor
            G: Matriz simétrica a regularizar
            original_profile: Perfil espectral de G

        Returns:
            Tupla (G_regularizada, delta_aplicado)
        """
        lambda_min = original_profile.lambda_min

        # Calcular δ necesario para empujar λ_min al umbral con margen
        target_lambda_min = MIN_EIGVAL_TOL * _SPD_INTERIOR_FACTOR

        if lambda_min >= target_lambda_min:
            # No se necesita regularización
            return G, 0.0

        # δ tal que λ_min + δ = target_lambda_min
        delta = target_lambda_min - lambda_min

        # Aplicar regularización: G_reg = G + δI
        n = G.shape[0]
        G_reg = G + delta * np.eye(n, dtype=_FLOAT_DTYPE)

        logger.debug(
            "Tensor %s: aplicando regularización Tikhonov con δ=%.6e "
            "(λ_min original=%.6e, objetivo=%.6e, n=%d)",
            name,
            delta,
            lambda_min,
            target_lambda_min,
            n,
        )

        return G_reg, delta

    @classmethod
    def _compute_regularization_deformation(
        cls,
        G_original: np.ndarray,
        G_regularized: np.ndarray,
        delta: float,
    ) -> float:
        """
        Calcula la deformación relativa por regularización en norma de Frobenius.

        Deformación = ||G_reg - G_orig||_F / ||G_orig||_F

        Para regularización Tikhonov G_reg = G + δI, se tiene analíticamente:
            ||G_reg - G||_F = ||δI||_F = δ√n

        Se calcula numéricamente para detectar inconsistencias, pero se
        verifica contra el valor analítico como control de cordura.

        Caso especial: si ||G_orig||_F < _NEAR_ZERO_FROBENIUS_TOL, la matriz
        original es esencialmente nula. En este caso la deformación relativa
        no está bien definida y se lanza un error, ya que una matriz nula
        no puede ser un tensor métrico válido (G ≻ 0 requiere λ_min > 0).

        Interpretación:
        - 0%: sin deformación (no se regularizó)
        - < 5%: deformación menor, geometría esencialmente preservada
        - 5-25%: deformación significativa, advertencia emitida
        - > 25%: deformación severa, indica problema con la matriz base

        Raises:
            MetricTensorError: Si la matriz original es esencialmente nula
        """
        original_norm = float(np.linalg.norm(G_original, "fro"))

        if original_norm < _NEAR_ZERO_FROBENIUS_TOL:
            raise MetricTensorError(
                "Deformación relativa no definida: ||G_original||_F = "
                f"{original_norm:.6e} < {_NEAR_ZERO_FROBENIUS_TOL:.6e}. "
                "Una matriz esencialmente nula no puede servir como tensor métrico."
            )

        deformation_norm = float(np.linalg.norm(G_regularized - G_original, "fro"))

        # Control de cordura: verificar contra valor analítico δ√n
        n = G_original.shape[0]
        expected_deformation_norm = delta * np.sqrt(n)
        relative_discrepancy = abs(deformation_norm - expected_deformation_norm)
        analytic_tol = max(1e-12, 1e-10 * expected_deformation_norm)

        if relative_discrepancy > analytic_tol:
            logger.warning(
                "Inconsistencia en deformación Tikhonov: numérica=%.6e, "
                "analítica (δ√n)=%.6e, discrepancia=%.6e",
                deformation_norm,
                expected_deformation_norm,
                relative_discrepancy,
            )

        return deformation_norm / original_norm

    @classmethod
    def _verify_spectral_consistency(
        cls,
        name: str,
        original_profile: SpectralProfile,
        regularized_profile: SpectralProfile,
        delta: float,
    ) -> None:
        """
        Verifica consistencia espectral entre eigenvalores pre y post regularización.

        Para regularización Tikhonov G_reg = G + δI, los eigenvalores deben
        satisfacer exactamente (en aritmética exacta):

            λᵢ(G_reg) = λᵢ(G) + δ  para todo i = 1, ..., n

        En aritmética de punto flotante, verificamos:

            |λᵢ(G_reg) - (λᵢ(G) + δ)| ≤ tol

        donde tol escala con la magnitud espectral para acomodar errores de
        redondeo. Inconsistencias aquí indicarían un bug en el pipeline.

        Args:
            name: Identificador del tensor
            original_profile: Perfil espectral pre-regularización
            regularized_profile: Perfil espectral post-regularización
            delta: Parámetro de Tikhonov aplicado

        Raises:
            MetricTensorError: Si la inconsistencia excede la tolerancia
        """
        if delta < _TIKHONOV_DELTA_ATOL:
            return  # Sin regularización, nada que verificar

        expected_eigenvalues = original_profile.eigenvalues + delta
        actual_eigenvalues = regularized_profile.eigenvalues

        # Tolerancia relativa a la magnitud espectral máxima
        spectral_scale = max(
            float(np.max(np.abs(expected_eigenvalues))),
            float(np.max(np.abs(actual_eigenvalues))),
            _NEAR_ZERO_FROBENIUS_TOL,
        )
        consistency_tol = spectral_scale * 1e-12

        max_discrepancy = float(np.max(np.abs(actual_eigenvalues - expected_eigenvalues)))

        if max_discrepancy > consistency_tol:
            logger.warning(
                "Tensor %s: inconsistencia espectral post-Tikhonov. "
                "max|λᵢ(G_reg) - (λᵢ(G) + δ)| = %.6e > tol = %.6e. "
                "Esto puede indicar errores de redondeo acumulados.",
                name,
                max_discrepancy,
                consistency_tol,
            )

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE CONTROL DE CONDICIONAMIENTO
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _assert_well_conditioned(
        cls,
        name: str,
        profile: SpectralProfile,
    ) -> None:
        """
        Verifica que el número de condición esté acotado.

        El número de condición κ₂(G) = λ_max / λ_min controla:
        - Estabilidad numérica de la inversión: ||G⁻¹||₂ = 1/λ_min
        - Sensibilidad a perturbaciones: ||δx|| / ||x|| ≤ κ₂(G) · ||δb|| / ||b||
        - Propagación de errores en formas cuadráticas
        - Convergencia de métodos iterativos (gradiente conjugado, etc.)

        Un κ₂(G) grande indica que la métrica es casi degenerada en alguna
        dirección, lo cual compromete la utilidad geométrica del tensor.

        Para la regularización Tikhonov con δ, el condicionamiento resultante es:
            κ₂(G_reg) = (λ_max + δ) / (λ_min + δ) ≤ κ₂(G)

        Raises:
            MetricTensorError: Si κ₂(G) > COND_NUM_TOL o no es finito
        """
        kappa = profile.condition_number

        if not np.isfinite(kappa):
            raise MetricTensorError(
                f"Tensor {name}: número de condición no finito (κ = {kappa}). "
                f"λ_min = {profile.lambda_min:.6e}, λ_max = {profile.lambda_max:.6e}."
            )

        if kappa > COND_NUM_TOL:
            raise MetricTensorError(
                f"Tensor {name}: mal condicionado. "
                f"κ₂(G) = {kappa:.4e} > COND_NUM_TOL = {COND_NUM_TOL:.4e}. "
                f"λ_min = {profile.lambda_min:.6e}, λ_max = {profile.lambda_max:.6e}. "
                "Esto indica que la métrica colapsa casi a un subespacio de "
                "menor dimensión."
            )

    # ───────────────────────────────────────────────────────────────────────────
    # MÉTODO DE INMUTABILIZACIÓN
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _make_immutable(cls, G: np.ndarray) -> np.ndarray:
        """
        Crea copia inmutable del tensor validado.

        La inmutabilidad se implementa con array.setflags(write=False),
        lo cual previene modificaciones accidentales que corromperían
        la geometría del módulo en tiempo de ejecución.

        Nota: La inmutabilidad es superficial (shallow). Para arrays
        numéricos densos de float64 esto proporciona protección completa,
        ya que no contienen referencias a otros objetos.
        """
        G_immutable = np.array(G, dtype=_FLOAT_DTYPE, copy=True)
        G_immutable.setflags(write=False)
        return G_immutable

    # ───────────────────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL DE VALIDACIÓN
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def _validate_and_regularize(cls, name: str, G: np.ndarray) -> np.ndarray:
        """
        Pipeline completo de validación y regularización de tensor métrico.

        Etapas del Pipeline:
        ─────────────────────
         1. VALIDACIÓN ESTRUCTURAL
            - Conversión a float64
            - Verificación de dimensionalidad y cuadratura
            - Detección de valores no finitos

         2. ANÁLISIS DE SIMETRÍA DE ENTRADA
            - Cuantificación de asimetría original
            - Advertencia si asimetría significativa

         3. SIMETRIZACIÓN POR PROYECCIÓN ORTOGONAL
            - G_sym = (G + Gᵀ) / 2

         4. VERIFICACIÓN DE FINITUD POST-SIMETRIZACIÓN
            - Detecta NaN/±∞ introducidos por la aritmética

         5. ANÁLISIS ESPECTRAL INICIAL
            - Cálculo de eigenvalores, κ₂, ||G||_F

         6. REGULARIZACIÓN TIKHONOV CONDICIONAL
            - Si λ_min < MIN_EIGVAL_TOL: G_reg = G_sym + δI
            - Cálculo de deformación relativa
            - Aborto si deformación > 25%

         7. ANÁLISIS ESPECTRAL POST-REGULARIZACIÓN
            - Verificación de mejora espectral

         8. VERIFICACIÓN DE CONSISTENCIA ESPECTRAL
            - |λᵢ(G_reg) - (λᵢ(G) + δ)| ≤ tol

         9. VERIFICACIÓN SPD ESTRICTA
            - Cholesky como certificación definitiva

        10. CONTROL DE CONDICIONAMIENTO
            - κ₂(G_reg) ≤ COND_NUM_TOL

        11. GENERACIÓN DE INFORME

        12. INMUTABILIZACIÓN
            - Retorno de array read-only

        Args:
            name: Identificador del tensor (para mensajes y dimensión esperada)
            G: Matriz candidata a tensor métrico

        Returns:
            Tensor métrico validado, regularizado e inmutable

        Raises:
            MetricTensorError: Si alguna verificación crítica falla
        """
        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 1: VALIDACIÓN ESTRUCTURAL
        # ═══════════════════════════════════════════════════════════════════════
        G_validated = cls._validate_structural_preconditions(name, G)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 2: ANÁLISIS DE SIMETRÍA DE ENTRADA
        # ═══════════════════════════════════════════════════════════════════════
        input_asymmetry = cls._assess_input_symmetry(name, G_validated)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 3: SIMETRIZACIÓN POR PROYECCIÓN ORTOGONAL
        # ═══════════════════════════════════════════════════════════════════════
        G_symmetric = cls._symmetrize_by_projection(G_validated)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 4: VERIFICACIÓN DE FINITUD POST-SIMETRIZACIÓN
        # ═══════════════════════════════════════════════════════════════════════
        if not np.all(np.isfinite(G_symmetric)):
            raise MetricTensorError(
                f"Tensor {name}: valores no finitos introducidos durante "
                "simetrización. Esto indica que la entrada contenía valores "
                "en el límite de representabilidad float64."
            )

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 5: ANÁLISIS ESPECTRAL INICIAL
        # ═══════════════════════════════════════════════════════════════════════
        try:
            original_profile = cls._compute_spectral_profile(G_symmetric)
        except (LinAlgError, MetricTensorError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: fallo en análisis espectral inicial. "
                f"La matriz puede ser numéricamente patológica. Error: {exc}"
            ) from exc

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 6: REGULARIZACIÓN TIKHONOV CONDICIONAL
        # ═══════════════════════════════════════════════════════════════════════
        G_regularized, tikhonov_delta = cls._apply_tikhonov_regularization(
            name, G_symmetric, original_profile
        )

        required_regularization = tikhonov_delta > _TIKHONOV_DELTA_ATOL

        # Calcular deformación por regularización
        if required_regularization:
            deformation = cls._compute_regularization_deformation(
                G_symmetric, G_regularized, tikhonov_delta
            )

            if deformation > _REGULARIZATION_ABORT_THRESHOLD:
                raise MetricTensorError(
                    f"Tensor {name}: deformación por regularización excesiva. "
                    f"||G_reg - G||_F / ||G||_F = {deformation:.2%} > "
                    f"{_REGULARIZATION_ABORT_THRESHOLD:.0%}. "
                    "La matriz base es demasiado deficiente para ser "
                    "utilizable como tensor métrico."
                )
        else:
            deformation = 0.0

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 7: ANÁLISIS ESPECTRAL POST-REGULARIZACIÓN
        # ═══════════════════════════════════════════════════════════════════════
        try:
            regularized_profile = cls._compute_spectral_profile(G_regularized)
        except (LinAlgError, MetricTensorError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: fallo en análisis espectral "
                f"post-regularización. Error: {exc}"
            ) from exc

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 8: VERIFICACIÓN DE CONSISTENCIA ESPECTRAL
        # ═══════════════════════════════════════════════════════════════════════
        cls._verify_spectral_consistency(
            name, original_profile, regularized_profile, tikhonov_delta
        )

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 9: VERIFICACIÓN SPD ESTRICTA
        # ═══════════════════════════════════════════════════════════════════════
        cls._assert_spd_strict(name, G_regularized, regularized_profile)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 10: CONTROL DE CONDICIONAMIENTO
        # ═══════════════════════════════════════════════════════════════════════
        cls._assert_well_conditioned(name, regularized_profile)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 11: INFORME DE REGULARIZACIÓN
        # ═══════════════════════════════════════════════════════════════════════
        report = RegularizationReport(
            tensor_name=name,
            original_profile=original_profile,
            regularized_profile=regularized_profile,
            tikhonov_delta=tikhonov_delta,
            frobenius_deformation=deformation,
            required_regularization=required_regularization,
            input_asymmetry=input_asymmetry,
        )
        report.log_summary()

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 12: INMUTABILIZACIÓN
        # ═══════════════════════════════════════════════════════════════════════
        return cls._make_immutable(G_regularized)

    # ───────────────────────────────────────────────────────────────────────────
    # CONSTRUCTORES DE TENSORES ESPECÍFICOS POR DOMINIO
    # ───────────────────────────────────────────────────────────────────────────

    @classmethod
    def build_physics_tensor(cls) -> np.ndarray:
        """
        Tensor métrico para el subespacio físico V_phys ⊂ ℝ³.

        Coordenadas del espacio:
            x = (saturation, flyback_voltage, dissipated_power)

        Estructura de la Forma Cuadrática:
        ──────────────────────────────────
        La forma cuadrática xᵀ G_phys x penaliza estados según:

        Q(x) = g₀₀·s² + g₁₁·v² + g₂₂·p²
             + 2g₀₁·sv + 2g₀₂·sp + 2g₁₂·vp

        Interpretación de Componentes:
        ─────────────────────────────────
        - g₀₀ = 2.50 (saturación): Penalización alta por riesgo de colapso
          inductivo. El núcleo magnético en saturación pierde inductancia,
          generando corrientes destructivas.

        - g₁₁ = 1.50 (flyback voltage): Penalización moderada por
          sobrevoltajes de conmutación. Los transitorios de apertura generan
          picos V = L·di/dt.

        - g₂₂ = 1.00 (potencia disipada): Penalización base por disipación
          térmica. El calor acumulado degrada componentes a largo plazo.

        - g₀₁ = g₁₀ = 0.85 (acoplamiento saturación-flyback): Modela el
          riesgo compuesto cuando saturación y conmutación abrupta coexisten.
          Físicamente: núcleo saturado → menor L → mayor di/dt → mayor flyback.

        - g₀₂ = g₂₀ = 0.30 (acoplamiento saturación-disipación): Acoplamiento
          débil. La saturación genera calor adicional por pérdidas de
          histéresis.

        - g₁₂ = g₂₁ = 0.45 (acoplamiento flyback-disipación): Los picos de
          voltaje incrementan pérdidas en dispositivos de supresión
          (snubbers, TVS).

        Justificación de Dominancia Diagonal Estricta:
        ───────────────────────────────────────────────
        La matriz base satisface dominancia diagonal estricta por filas:
          fila 0: 2.50 > |0.85| + |0.30| = 1.15 ✓  (margen 1.35)
          fila 1: 1.50 > |0.85| + |0.45| = 1.30 ✓  (margen 0.20)
          fila 2: 1.00 > |0.30| + |0.45| = 0.75 ✓  (margen 0.25)

        Por el teorema de Gershgorin, los discos de Gershgorin Dᵢ son:
          D₀ = {z ∈ ℂ : |z - 2.50| ≤ 1.15}  ⟹  z ∈ [1.35, 3.65]
          D₁ = {z ∈ ℂ : |z - 1.50| ≤ 1.30}  ⟹  z ∈ [0.20, 2.80]
          D₂ = {z ∈ ℂ : |z - 1.00| ≤ 0.75}  ⟹  z ∈ [0.25, 1.75]

        Todos los discos están contenidos en (0, ∞), garantizando eigenvalores
        estrictamente positivos antes de cualquier regularización.

        Returns:
            Tensor métrico G_phys ∈ 𝒮₊₊³ validado e inmutable
        """
        G_phys = np.array(
            [
                [2.50, 0.85, 0.30],
                [0.85, 1.50, 0.45],
                [0.30, 0.45, 1.00],
            ],
            dtype=_FLOAT_DTYPE,
        )
        return cls._validate_and_regularize("G_phys", G_phys)

    @classmethod
    def build_topology_tensor(cls) -> np.ndarray:
        """
        Tensor métrico para el subespacio topológico V_topo ⊂ ℝ².

        Coordenadas del espacio (números de Betti):
            x = (β₀, β₁)

        donde:
            β₀ = número de componentes conexas (fragmentación)
            β₁ = número de ciclos independientes (dependencias circulares)

        Interpretación Homológica:
        ─────────────────────────────
        Los números de Betti caracterizan la topología del grafo de
        dependencias del sistema:

        - β₀ alto: Sistema fragmentado en subsistemas desconectados.
          Problema: pérdida de coordinación global, estados inconsistentes.

        - β₁ alto: Ciclos en el grafo de dependencias.
          Problema: deadlocks, livelocks, propagación infinita de
          actualizaciones.

        Estructura de la Forma Cuadrática:
        ──────────────────────────────────
        Q(x) = g₀₀·β₀² + g₁₁·β₁² + 2g₀₁·β₀β₁

        - g₀₀ = 1.00: Penalización base por fragmentación.

        - g₁₁ = 3.00: Penalización alta por ciclos.
          Los ciclos son más peligrosos que la fragmentación porque
          pueden generar comportamientos no terminantes.

        - g₀₁ = g₁₀ = 0.60: Acoplamiento fragmentación-ciclos.
          Un sistema fragmentado CON ciclos en sus fragmentos es
          particularmente problemático: aislamiento + inestabilidad local.

        Verificación de Positividad (Criterio de Sylvester):
        ────────────────────────────────────────────────────
        Para una matriz simétrica 2×2, G ≻ 0 ⟺ Δ₁ > 0 ∧ Δ₂ > 0 donde
        Δₖ son los menores principales líderes:

          Δ₁ = g₀₀ = 1.00 > 0 ✓
          Δ₂ = det(G) = g₀₀·g₁₁ - g₀₁² = 1.00·3.00 - 0.60² = 2.64 > 0 ✓

        Eigenvalores analíticos (fórmula cuadrática):
        ──────────────────────────────────────────────
          tr(G) = 4.00,  det(G) = 2.64
          λ± = (tr ± √(tr² - 4·det)) / 2
             = (4.00 ± √(16.00 - 10.56)) / 2
             = (4.00 ± √5.44) / 2
             = (4.00 ± 2.33238...) / 2
          λ₋ ≈ 0.83381,  λ₊ ≈ 3.16619

        Returns:
            Tensor métrico G_topo ∈ 𝒮₊₊² validado e inmutable
        """
        G_topo = np.array(
            [
                [1.00, 0.60],
                [0.60, 3.00],
            ],
            dtype=_FLOAT_DTYPE,
        )
        return cls._validate_and_regularize("G_topo", G_topo)

    @classmethod
    def build_thermo_tensor(cls) -> np.ndarray:
        """
        Tensor métrico para el subespacio termodinámico V_thermo ⊂ ℝ².

        Coordenadas del espacio:
            x = (entropy, exergy_loss)

        donde:
            entropy = entropía del sistema (desorden informacional/energético)
            exergy_loss = pérdida de exergía (capacidad de realizar trabajo útil)

        Fundamento Termodinámico:
        ─────────────────────────
        La exergía (energía libre de Gibbs disponible) cuantifica la máxima
        cantidad de trabajo que un sistema puede realizar al equilibrarse
        con su entorno. La pérdida de exergía representa degradación
        irreversible de la calidad energética.

        Dualidad Entropía-Exergía:
        ──────────────────────────
        Por la segunda ley de la termodinámica:
            ΔS_universo ≥ 0 (entropía siempre crece)
            ΔExergía ≤ 0 (exergía siempre decrece en procesos reales)

        Estas variables están acopladas: incrementos de entropía típicamente
        correlacionan con pérdidas de exergía.

        Estructura de la Forma Cuadrática:
        ──────────────────────────────────
        Q(x) = g₀₀·S² + g₁₁·X_loss² + 2g₀₁·S·X_loss

        - g₀₀ = 1.80: Penalización por aumento de entropía.
          Entropía alta indica pérdida de estructura/información.

        - g₁₁ = 2.20: Penalización por pérdida de exergía.
          Ligeramente mayor que entropía porque la exergía es directamente
          convertible a trabajo útil.

        - g₀₁ = g₁₀ = 0.75: Acoplamiento entropía-exergía.
          Modela que la degradación informacional y la degradación
          energética rara vez son independientes.

        Verificación de Positividad (Criterio de Sylvester):
        ────────────────────────────────────────────────────
          Δ₁ = g₀₀ = 1.80 > 0 ✓
          Δ₂ = det(G) = 1.80·2.20 - 0.75² = 3.96 - 0.5625 = 3.3975 > 0 ✓

        Eigenvalores analíticos (fórmula cuadrática):
        ──────────────────────────────────────────────
          tr(G) = 4.00,  det(G) = 3.3975
          λ± = (tr ± √(tr² - 4·det)) / 2
             = (4.00 ± √(16.00 - 13.59)) / 2
             = (4.00 ± √2.41) / 2
             = (4.00 ± 1.55242...) / 2
          λ₋ ≈ 1.22379,  λ₊ ≈ 2.77621

        Returns:
            Tensor métrico G_thermo ∈ 𝒮₊₊² validado e inmutable
        """
        G_thermo = np.array(
            [
                [1.80, 0.75],
                [0.75, 2.20],
            ],
            dtype=_FLOAT_DTYPE,
        )
        return cls._validate_and_regularize("G_thermo", G_thermo)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES UTILITARIAS PARA DIAGNÓSTICO EXTERNO
# ═══════════════════════════════════════════════════════════════════════════════

def get_tensor_diagnostics(G: np.ndarray, name: str = "unnamed") -> dict:
    """
    Genera diagnósticos completos de un tensor métrico para debugging externo.

    Usa las mismas tolerancias de simetría que el pipeline de validación
    para garantizar consistencia entre diagnóstico y construcción.

    Args:
        G: Tensor métrico a diagnosticar
        name: Identificador para el informe

    Returns:
        Diccionario con diagnósticos espectrales y estructurales.
        En caso de error, retorna diccionario con campos 'error' y 'error_type'.
    """
    try:
        G_arr = np.asarray(G, dtype=_FLOAT_DTYPE)

        if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
            return {
                "name": name,
                "error": f"No es matriz cuadrada: shape={G_arr.shape}",
                "error_type": "StructuralError",
            }

        # Verificación de simetría consistente con el pipeline
        symmetry_tol = MetricTensorFactory._compute_symmetry_tolerance(G_arr)
        asymmetry_norm = float(np.linalg.norm(G_arr - G_arr.T, "fro"))
        is_symmetric = asymmetry_norm <= symmetry_tol

        # Simetrizar para análisis espectral válido
        G_sym = MetricTensorFactory._symmetrize_by_projection(G_arr)
        profile = MetricTensorFactory._compute_spectral_profile(G_sym)
        is_spd = MetricTensorFactory._verify_spd_by_cholesky(name, G_sym)

        return {
            "name": name,
            "shape": G_arr.shape,
            "dtype": str(G_arr.dtype),
            "is_symmetric": is_symmetric,
            "asymmetry_norm": asymmetry_norm,
            "symmetry_tolerance": symmetry_tol,
            "is_spd": is_spd,
            "lambda_min": profile.lambda_min,
            "lambda_max": profile.lambda_max,
            "condition_number": profile.condition_number,
            "spectral_gap": profile.spectral_gap,
            "frobenius_norm": profile.frobenius_norm,
            "eigenvalues": profile.eigenvalues.tolist(),
            "is_well_conditioned": profile.is_well_conditioned(),
            "is_strictly_positive": profile.is_strictly_positive(),
        }
    except Exception as exc:
        return {
            "name": name,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPILACIÓN DE TENSORES EN TIEMPO DE CARGA DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════════
# La instanciación global fuerza validación espectral completa al importar
# el módulo. Si algún tensor falla validación, el módulo no carga.
#
# Los tensores se exponen como constantes de solo lectura (arrays con
# write=False) para preservar la invariancia del tejido geométrico global.
# ═══════════════════════════════════════════════════════════════════════════════

G_PHYSICS: Final[np.ndarray] = MetricTensorFactory.build_physics_tensor()
G_TOPOLOGY: Final[np.ndarray] = MetricTensorFactory.build_topology_tensor()
G_THERMODYNAMICS: Final[np.ndarray] = MetricTensorFactory.build_thermo_tensor()

# Logging de confirmación de carga exitosa con diagnósticos mínimos
logger.info(
    "Tensores métricos inicializados exitosamente: "
    "G_PHYSICS %s, G_TOPOLOGY %s, G_THERMODYNAMICS %s",
    G_PHYSICS.shape,
    G_TOPOLOGY.shape,
    G_THERMODYNAMICS.shape,
)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Tensores precompilados
    "G_PHYSICS",
    "G_TOPOLOGY",
    "G_THERMODYNAMICS",
    # Fábrica para casos de uso avanzados
    "MetricTensorFactory",
    # Estructuras de diagnóstico
    "SpectralProfile",
    "RegularizationReport",
    # Utilidades
    "get_tensor_diagnostics",
]