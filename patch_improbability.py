import re

file_path = "app/core/immune_system/improbability_drive.py"
with open(file_path, "r") as f:
    content = f.read()

handler_replacement = """    def _morphism_handler(self, **kwargs: Any) -> Dict[str, Any]:
        \"\"\"
        Funtor natural para la MIC con Clausura Transitiva DIKW.

        Aplica Fast-Fail si β₁ > 0 en TACTICS. Extrae Ψ y ROI rigurosamente.
        \"\"\"
        try:
            telemetry = kwargs.get("telemetry_context")

            # Verificación del fibrado: Ley de Clausura Transitiva
            if telemetry:
                # Comprobar β1 (ciclos lógicos) en la capa Táctica
                tactics_betti_1 = telemetry.get_metric("tactics_betti_1", default=0)
                if tactics_betti_1 > 0:
                    return ImprobabilityResult.failure(
                        error_type="HomologicalInconsistencyError",
                        error_message=f"Fast-Fail: β₁ = {tactics_betti_1} > 0. Veto topológico, clausura transitiva violada."
                    ).to_dict()

            # Extracción Invariante del Pasaporte (o kwargs directos como fallback)
            # Ψ proviene de BusinessTopologicalAnalyzer (TACTICS)
            psi = kwargs.get("psi")
            if psi is None and telemetry:
                psi = telemetry.get_metric("business_pyramidal_stability", default=None)

            # ROI proviene de LaplaceOracle/FinancialEngine (STRATEGY)
            roi = kwargs.get("roi")
            if roi is None and telemetry:
                roi = telemetry.get_metric("strategy_roi", default=None)

            if psi is None or roi is None:
                return ImprobabilityResult.failure(
                    error_type="DimensionalMismatchError",
                    error_message="Parámetros 'psi' y 'roi' no se encontraron en kwargs ni en TelemetryContext"
                ).to_dict()

            psi_val = float(psi)
            roi_val = float(roi)

            # Computación
            penalty = self._tensor.compute_penalty(psi_val, roi_val)
            gradient = self._tensor.compute_gradient(psi_val, roi_val)

            result = ImprobabilityResult.success(
                penalty=penalty,
                kappa=self._tensor.kappa,
                gamma=self._tensor.gamma,
                psi=psi_val,
                roi=roi_val,
                gradient=gradient
            )

            return result.to_dict()

        except (ValueError, TypeError) as e:
            logger.warning("Error de validación: %s", str(e))
            return ImprobabilityResult.failure(
                error_type=type(e).__name__,
                error_message=str(e)
            ).to_dict()

        except (NumericalInstabilityError, AxiomViolationError) as e:
            logger.error("Error matemático: %s", str(e), exc_info=True)
            return ImprobabilityResult.failure(
                error_type=type(e).__name__,
                error_message=str(e)
            ).to_dict()

        except Exception as e:
            logger.error("Error inesperado: %s", str(e), exc_info=True)
            return ImprobabilityResult.failure(
                error_type=type(e).__name__,
                error_message=str(e)
            ).to_dict()"""

content = re.sub(
    r"    def _morphism_handler\(self, \*\*kwargs: Any\) -> Dict\[str, Any\]:.*?(?=    def batch_compute|    # ════════════════════════════════════════════════════════════════════════════\n    # MÓDULO 11: FÁBRICA DE TENSORES)",
    handler_replacement + "\n\n",
    content,
    flags=re.DOTALL
)

with open(file_path, "w") as f:
    f.write(content)
