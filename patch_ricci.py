import re

with open('app/core/immune_system/topological_watcher.py', 'r') as f:
    content = f.read()

# 1. Update `__slots__` of `ImmuneWatcherMorphism` to include `_metric_tensors_state`
slots_str = """
    __slots__ = (
        "_critical",
        "_warning",
        "_hysteresis",
        "_enable_topology_monitoring",
        "_projector",
        "_previous_status",
        "_euler_history",
        "_evaluation_count",
        "_metric_tensors_state",
    )
"""
content = re.sub(r'__slots__ = \([^)]+\)', slots_str.strip('\n'), content)


# 2. Initialize metric tensor state in `__init__`
init_str = """
        # Estado mutable interno
        self._previous_status: Optional[HealthStatus] = None
        self._euler_history: List[Optional[int]] = []
        self._evaluation_count: int = 0

        # Mantenimiento de tensores evolutivos G_mu_nu(t)
        import app.core.immune_system.metric_tensors as ext_metric_tensors
        self._metric_tensors_state = {
            "G_PHYSICS": np.copy(ext_metric_tensors.G_PHYSICS),
            "G_TOPOLOGY": np.copy(ext_metric_tensors.G_TOPOLOGY),
            "G_THERMODYNAMICS": np.copy(ext_metric_tensors.G_THERMODYNAMICS),
        }

        self._projector = self._build_projector()
"""
content = re.sub(r'# Estado mutable interno.*?\n\s*self\._projector = self\._build_projector\(\)', init_str.strip('\n'), content, flags=re.DOTALL)


# 3. Update `_build_projector` to use state variables
proj_str = """
        # Usamos los tensores evolutivos G_mu_nu(t)
        # G_phys escalado: D^{-1} G_PHYSICS D^{-1} donde D = diag(scale)
        scale_phys = np.array([
            PhysicalConstants.SATURATION_CRITICAL,
            PhysicalConstants.FLYBACK_MAX_SAFE,
            _P_NOMINAL,
        ], dtype=np.float64)
        D_inv_phys = np.diag(1.0 / scale_phys)
        scaled_G_phys = D_inv_phys @ self._metric_tensors_state["G_PHYSICS"] @ D_inv_phys

        scale_topo = np.array([1.0, 1.0], dtype=np.float64)
        D_inv_topo = np.diag(1.0 / scale_topo)
        scaled_G_topo = D_inv_topo @ self._metric_tensors_state["G_TOPOLOGY"] @ D_inv_topo

        scale_thermo = np.array([0.5, 0.5], dtype=np.float64)
        D_inv_thermo = np.diag(1.0 / scale_thermo)
        scaled_G_thermo = D_inv_thermo @ self._metric_tensors_state["G_THERMODYNAMICS"] @ D_inv_thermo
"""
content = re.sub(r'# Usamos los tensores precompilados del nuevo módulo combinados con las escalas.*?scaled_G_thermo = D_inv_thermo @ ext_metric_tensors\.G_THERMODYNAMICS @ D_inv_thermo', proj_str.strip('\n'), content, flags=re.DOTALL)


# 4. Implement Ricci Flow integrators and methods
ricci_funcs = """

    def _compute_discrete_ricci_curvature(self, telemetry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        \"\"\"
        Calcula la Curvatura de Forman-Ricci/Ollivier-Ricci aproximada (Ric_mu_nu)
        basado en la topología (Laplaciano) de entrada para evolucionar la métrica.
        \"\"\"
        # Extraer características básicas del grafo desde la telemetría (Fallback a valores neutrales)
        betti_0 = telemetry.get('betti_0', 1.0)
        betti_1 = telemetry.get('betti_1', 0.0)

        # Una matriz de Ricci aproximada basada en anomalías de conectividad
        # Fiedler < epsilon => Curvatura muy negativa (Fragilidad)
        # Betti 1 > 0 => Curvatura negativa

        # Approximate Ricci curvature tensors for the 3 sub-spaces
        Ric_PHYSICS = np.zeros((3, 3))
        Ric_TOPOLOGY = np.zeros((2, 2))
        Ric_THERMO = np.zeros((2, 2))

        if betti_1 > 0:
            # Ciclos inducen curvatura negativa en la topología
            Ric_TOPOLOGY[1, 1] = -0.5 * betti_1

        if betti_0 > 1:
            # Desconexiones (islas) inducen curvatura fuertemente negativa
            Ric_TOPOLOGY[0, 0] = -1.0 * betti_0

        return Ric_PHYSICS, Ric_TOPOLOGY, Ric_THERMO

    def _evolve_metric_tensors_ricci_flow(self, telemetry: Dict[str, Any], dt: float = 0.01) -> None:
        \"\"\"
        Evoluciona los tensores G_mu_nu(t) usando Integración de Euler Hacia Atrás (Backward Euler)
        acoplado a la electrodinámica: G_mu_nu(t+dt) = G_mu_nu(t) - 2 * dt * Ric_mu_nu(t).

        Aplica Tikhonov displacement para aniquilar singularidades si es necesario.
        \"\"\"
        Ric_P, Ric_T, Ric_Th = self._compute_discrete_ricci_curvature(telemetry)

        # G(t+dt) = G(t) - 2 * dt * Ric
        self._metric_tensors_state["G_PHYSICS"] -= 2 * dt * Ric_P
        self._metric_tensors_state["G_TOPOLOGY"] -= 2 * dt * Ric_T
        self._metric_tensors_state["G_THERMODYNAMICS"] -= 2 * dt * Ric_Th

        # Síntesis Espectral Exacta (Desplazamiento de Tikhonov para mantener SPD)
        for key in self._metric_tensors_state:
            G = self._metric_tensors_state[key]
            # Eigen descomposicion
            eigvals, eigvecs = np.linalg.eigh(G)
            min_eig = np.min(eigvals)
            epsilon = MIN_EIGVAL_TOL

            # Tikhonov si G_reg deforma la métrica cerca del colapso
            if min_eig < epsilon:
                delta = max(0.0, epsilon - min_eig)
                # G_reg = Q(Lambda + delta*I)Q^T
                G_reg = eigvecs @ np.diag(eigvals + delta) @ eigvecs.T
                # Simetrizar para eliminar errores de FPU
                self._metric_tensors_state[key] = (G_reg + G_reg.T) / 2.0

        # Reconstruir proyector (Doble búfer dinámico simulado)
        self._projector = self._build_projector()
"""

# Insert ricci_funcs before `def __call__`
content = content.replace("    def __call__(self, state: CategoricalState) -> CategoricalState:", ricci_funcs + "\n    def __call__(self, state: CategoricalState) -> CategoricalState:")

call_integration = """
        try:
            telemetry = state.context.get("telemetry_metrics", {})

            # Integración Numérica del Flujo de Ricci ANTES de la proyección
            self._evolve_metric_tensors_ricci_flow(telemetry or {})

            signal = build_signal(telemetry or {}, strict=False)
"""
content = re.sub(r'        try:\n\s*telemetry = state\.context\.get\("telemetry_metrics", \{\}\)\n\s*signal = build_signal\(telemetry or \{\}, strict=False\)', call_integration.strip('\n'), content)

with open('app/core/immune_system/topological_watcher.py', 'w') as f:
    f.write(content)
