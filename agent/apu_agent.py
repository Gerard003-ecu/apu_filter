import logging
import os
import time

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AutonomousAgent:
    """
    Agente autónomo que opera bajo un ciclo OODA (Observe, Orient, Decide, Act).
    Monitorea la salud del Core y toma decisiones basadas en métricas de telemetría.
    """

    def __init__(self):
        # Default to localhost for dev, apu_core for docker
        self.core_api_url = os.getenv("CORE_API_URL", "http://localhost:5002")
        self.telemetry_endpoint = f"{self.core_api_url}/api/telemetry/status"
        self.check_interval = 10  # Seconds
        logging.info(f"AutonomousAgent initialized. Monitoring {self.core_api_url}")

    def observe(self):
        """
        Pivote 1: Observe.
        Realiza un GET /api/telemetry/status.
        Retorna las métricas crudas o None si hay error.
        """
        try:
            response = requests.get(self.telemetry_endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()
            logging.debug(f"[OBSERVE] Telemetry data received: {data}")
            return data
        except requests.exceptions.RequestException as e:
            logging.warning(f"[OBSERVE] Failed to connect to Core: {e}")
            return None

    def orient(self, telemetry_data):
        """
        Pivote 2: Orient (El Análisis).
        Analiza los datos para determinar el estado del sistema.
        """
        if not telemetry_data:
            return "UNKNOWN"

        # Extract metrics safely
        flyback_voltage = telemetry_data.get("flyback_voltage", 0)
        saturation = telemetry_data.get("saturation", 0)

        # Analysis Logic
        if flyback_voltage > 0.5:
            return "INESTABLE"
        elif saturation > 0.9:
            return "SATURADO"
        else:
            return "NOMINAL"

    def decide(self, status):
        """
        Pivote 3: Decide (La Toma de Decisiones).
        Determina la acción a tomar basada en el estado orientado.
        """
        if status == "INESTABLE":
            return "RECOMENDAR_LIMPIEZA"
        elif status == "SATURADO":
            return "RECOMENDAR_REDUCIR_VELOCIDAD"
        elif status == "NOMINAL":
            return "HEARTBEAT"
        else:
            return "WAIT"

    def act(self, decision):
        """
        Pivote 4: Act (La Ejecución).
        Ejecuta la acción decidida.
        """
        if decision == "RECOMENDAR_LIMPIEZA":
            logging.warning("[BRAIN] ⚠️ DETECTADA INESTABILIDAD - Se recomienda revisión de CSV")
        elif decision == "RECOMENDAR_REDUCIR_VELOCIDAD":
            logging.warning("[BRAIN] ⚠️ SATURACION DETECTADA - Se recomienda reducir la velocidad de carga")
        elif decision == "HEARTBEAT":
            logging.info("[BRAIN] ✅ Sistema NOMINAL - Operación normal")
        elif decision == "WAIT":
            logging.info("[BRAIN] ⏳ Esperando conexión con Core...")

        # Future: Call /api/tools/clean here

    def run(self):
        """
        Bucle Principal.
        Ejecuta el ciclo OODA continuamente.
        """
        logging.info("Starting OODA Loop...")
        while True:
            try:
                # 1. Observe
                data = self.observe()

                # 2. Orient
                status = self.orient(data)

                # 3. Decide
                decision = self.decide(status)

                # 4. Act
                self.act(decision)

            except Exception as e:
                logging.error(f"Unexpected error in OODA loop: {e}")

            time.sleep(self.check_interval)


if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.run()
