import numpy as np


def run_monte_carlo_simulation(apu_details, num_simulations=1000):
    """
    Ejecuta una simulación de Monte Carlo sobre el desglose de un APU.

    Args:
        apu_details (list): Lista de diccionarios, donde cada diccionario
        representa un insumo del APU.
        num_simulations (int): Número de simulaciones a ejecutar.

    Returns:
        dict: Un diccionario con los resultados estadísticos de la simulación.
    """
    total_costs = []

    for _ in range(num_simulations):
        simulation_cost = 0
        for item in apu_details:
            base_price = item.get("VALOR_UNITARIO", 0)
            quantity = item.get("CANTIDAD", 0)
            category = item.get("CATEGORIA", "")

            if category == "MATERIALES":
                # Simular volatilidad del 5% en el precio de los materiales
                simulated_price = np.random.normal(loc=base_price, scale=base_price * 0.05)
                simulation_cost += simulated_price * quantity
            elif category == "MANO DE OBRA":
                # Simular variabilidad del 10% en el rendimiento (cantidad de tiempo)
                simulated_quantity = np.random.normal(loc=quantity, scale=quantity * 0.10)
                simulation_cost += base_price * simulated_quantity
            else:
                # Otros costos se mantienen fijos
                simulation_cost += base_price * quantity

        total_costs.append(simulation_cost)

    # Asegurarse de que los costos no sean negativos
    total_costs = [max(0, cost) for cost in total_costs]

    # Calcular resultados estadísticos
    mean_cost = np.mean(total_costs)
    std_dev = np.std(total_costs)
    percentile_5 = np.percentile(total_costs, 5)
    percentile_95 = np.percentile(total_costs, 95)

    return {
        "mean": mean_cost,
        "std_dev": std_dev,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
    }
