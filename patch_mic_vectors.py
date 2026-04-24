import re

file_path = "app/adapters/mic_vectors.py"
with open(file_path, "r") as f:
    content = f.read()

new_vector = """def vector_calculate_improbability_tensor(
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    \"\"\"
    Vector de nivel STRATEGY.

    Aplica el Tensor de Improbabilidad para la deformación del espacio de
    probabilidades y evaluar el riesgo de colas pesadas (Fat-Tail Risk).
    \"\"\"
    with MetricsCollector() as collector:
        if payload is None:
            return _build_error(
                stratum=Stratum.STRATEGY,
                error_type="MissingPayloadError",
                message="El tensor de improbabilidad requiere parámetros de payload explícitos."
            )

        try:
            from app.core.immune_system.improbability_drive import ImprobabilityDriveService
            # Se necesita la instancia de MIC, pero en vectores puros no la tenemos inyectada directamente,
            # pero el servicio puede instanciarse sin MICRegistry usando kwargs de context/payload
            # o simplemente utilizando el Tensor matemáticamente.
            # actually _morphism_handler expects **kwargs

            # Since ImprobabilityDriveService needs MICRegistry for full init, let's instantiate the Tensor manually
            # Or use ImprobabilityDriveService with a dummy/None MIC just to call _morphism_handler since _morphism_handler doesn't use self.mic
            service = ImprobabilityDriveService(None)

            # Combine context and payload
            kwargs = {}
            if context:
                kwargs.update(context)
            kwargs.update(payload)

            result_dict = service._morphism_handler(**kwargs)

            if not result_dict.get("success", False):
                return _build_error(
                    stratum=Stratum.STRATEGY,
                    error_type=result_dict.get("error_type", "CalculationError"),
                    message=result_dict.get("error_message", "Fallo al calcular el tensor de improbabilidad"),
                    details=result_dict
                )

            return VectorResult(
                status=VectorResultStatus.SUCCESS,
                stratum=Stratum.STRATEGY,
                output=result_dict,
                metrics=collector.metrics,
                metadata={"tensor_applied": True}
            )
        except Exception as e:
            logger.error("Error en vector_calculate_improbability_tensor: %s", str(e))
            return _build_error(
                stratum=Stratum.STRATEGY,
                error_type=type(e).__name__,
                message=str(e),
            )

"""

content += "\n" + new_vector

with open(file_path, "w") as f:
    f.write(content)
