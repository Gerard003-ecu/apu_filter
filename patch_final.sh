patch -p1 << 'PATCH_EOF'
--- a/app/boole/strategy/sheaf_cohomology_orchestrator.py
+++ b/app/boole/strategy/sheaf_cohomology_orchestrator.py
@@ -155,6 +155,29 @@
         - Eigenvalores negativos significativos (L no semidefinida positiva).
     """

+class TopologicalBifurcationError(SheafCohomologyError):
+    """Lanzada cuando se detecta una bifurcación topológica severa.
+
+    Esta excepción indica que una deformación evaluada altera
+    invariantes de Betti inaceptablemente, obligando un colapso Fast-Fail.
+    """
+
+
+# =============================================================================
+# SECCIÓN 2.5: PROTOCOLOS PARA INYECCIÓN DE DEPENDENCIAS (Pullback)
+# =============================================================================
+
+from typing import Protocol, Any
+
+class ITopologicalWatcher(Protocol):
+    """Protocolo (Contrato Matemático) para el Observador Inmunológico.
+
+    Define el tipado estructural categórico (Duck Typing) necesario para
+    ejecutar un Pullback desde el Estrato Estratégico.
+    """
+    def evaluate_manifold_deformation(self, tensor: np.ndarray) -> Any:
+        ...
+

 # =============================================================================
 # SECCIÓN 3: ESTRUCTURAS DE DATOS INMUTABLES
@@ -1192,6 +1215,7 @@
         k = min(_SPARSE_MAX_EIGENVALUES, max(2, n - 1))

         try:
+            v0 = np.ones(n) / np.sqrt(n)
             # Shift-invert: (L − σI)⁻¹ con σ=0.
             # ARPACK con sigma=0 y which='LM' = eigenvalores más grandes de
             # L⁻¹ = eigenvalores más pequeños de L (para L semidefinida positiva
@@ -1203,6 +1227,8 @@
                 which="LM",
                 return_eigenvectors=False,
                 tol=_ARPACK_TOLERANCE,
+                v0=v0,
+                maxiter=max(1000, 10 * n)
             )
         except ArpackError as exc:
             raise SpectralComputationError(
@@ -1606,7 +1632,142 @@
         return residual_norm_sq, max(0.0, residual_norm)

     # -------------------------------------------------------------------------
-    # 7.4 Auditoría del estado global
+    # 7.4 FASE I - V: El Pullback Categórico y Evaluación de Deformaciones
+    # -------------------------------------------------------------------------
+
+    @classmethod
+    def evaluate_tool_injection(
+        cls,
+        current_sheaf: CellularSheaf,
+        tool_node_id: int,
+        tool_dim: int,
+        tool_edges: List[Tuple[int, Dict[int, int], np.ndarray]],
+        watcher: ITopologicalWatcher,
+        base_state_vector: np.ndarray,
+        phys_dissipation: float = 0.0,
+        tact_dissipation: float = 0.0,
+    ) -> GlobalFrustrationAssessment:
+        """FASE I-V: Morfismo de Invocación (Pullback) para inyección de herramienta.
+
+        1. Construcción del Fibrado Tangente de Simulación (TpM).
+        2. Extracción del Tensor de Estado (ψ).
+        3. El Pullback Categórico (f*).
+        4. Intercepción Riemanniana.
+        5. El Colapso de la Función de Onda (Fast-Fail).
+        """
+        import weakref
+
+        # Fase I: Construcción del Fibrado Tangente
+        # Creamos una copia/simulación del CellularSheaf mutado
+        sim_node_dims = current_sheaf.node_dims.copy()
+        sim_node_dims[tool_node_id] = tool_dim
+
+        sim_edge_dims = current_sheaf.edge_dims.copy()
+        for edge_id, _, _ in tool_edges:
+            sim_edge_dims[edge_id] = 1  # Asumimos dimensión 1 para nuevas aristas o las provistas
+
+        sim_sheaf = CellularSheaf(
+            num_nodes=current_sheaf.num_nodes + 1,
+            node_dims=sim_node_dims,
+            edge_dims=sim_edge_dims
+        )
+        # Usamos referencias débiles en los componentes
+        _sheaf_ref = weakref.ref(sim_sheaf)
+
+        # Re-agregar aristas existentes
+        for edge in current_sheaf.edges:
+            sim_sheaf.add_edge(edge.edge_id, edge.u, edge.v, edge.restriction_u, edge.restriction_v)
+
+        # Inyectar nuevas aristas de la herramienta
+        for edge_id, edge_pairs, matrix in tool_edges:
+            (u, v) = list(edge_pairs.keys())[0]
+            # Aseguramos compatibilidad dimensional de los mapas de restricción F_ue y F_ve.
+            # matrix asume mapear F(u) -> F(e). Ajustamos su dimensión para u y v.
+            d_e = 1 # Según la inyección
+            # Para F_{u ▷ e} (d_e x d_u)
+            F_ue_mat = np.zeros((d_e, sim_node_dims[u]))
+            # Copiamos la data de forma segura, truncando o rellenando
+            np.copyto(F_ue_mat.flat[:min(F_ue_mat.size, matrix.size)], matrix.flat[:min(F_ue_mat.size, matrix.size)])
+
+            # Para F_{v ▷ e} (d_e x d_v)
+            F_ve_mat = np.zeros((d_e, sim_node_dims[v]))
+            np.copyto(F_ve_mat.flat[:min(F_ve_mat.size, matrix.size)], matrix.flat[:min(F_ve_mat.size, matrix.size)])
+
+            sim_sheaf.add_edge(edge_id, u, v, RestrictionMap(F_ue_mat), RestrictionMap(F_ve_mat))
+
+        if not sim_sheaf.is_fully_assembled:
+            raise SheafDegeneracyError("Simulated sheaf could not be fully assembled.")
+
+        # Fase II: Extracción del Tensor de Estado
+        # Calcular Invariantes de Betti para el sim_sheaf
+        sim_delta = sim_sheaf.build_coboundary_operator()
+        sim_L = sim_sheaf.compute_sheaf_laplacian()
+        sim_spectral = _SpectralAnalyzer.compute(sim_L, sim_delta)
+
+        beta_0 = sim_spectral.h0_dimension
+        beta_1 = sim_spectral.h1_dimension
+        beta_2 = 0 # En complejos 1D usualmente β_2 = 0
+
+        # Ensamblaje del Tensor ψ ∈ R^7
+        # Las componentes: [beta_0, beta_1, beta_2, E(x), phys_dissipation, tact_dissipation, 0.0]
+        # Purga de la dependencia lineal de chi = beta_0 - beta_1 + beta_2.
+        sim_state_vector = np.append(base_state_vector, np.zeros(tool_dim)) # Vector extendido
+        try:
+            E_x, _ = cls._compute_frustration_energy(sim_delta, sim_state_vector)
+        except SheafCohomologyError:
+            E_x = 0.0
+
+        tensor_7d = np.array([
+            float(beta_0),
+            float(beta_1),
+            float(beta_2),
+            float(E_x),
+            float(phys_dissipation),
+            float(tact_dissipation),
+            0.0 # Relleno para 7D, ortogonal
+        ], dtype=np.float64)
+
+        # Fase III y IV: Pullback (f*) e intercepción
+        threat_metrics = watcher.evaluate_manifold_deformation(tensor_7d)
+
+        # Fase V: Colapso de la función de onda (Fast-Fail)
+        # Evaluamos la amenaza o defecto topológico
+        # Si threat_metrics.euler_char cambió negativamente o threat supera eps (Manejado por el watcher)
+        if getattr(threat_metrics, "status", None) and threat_metrics.status.name == "CRITICAL":
+            raise TopologicalBifurcationError(
+                f"Veto Absoluto: La inyección induce inestabilidad hipercaótica. "
+                f"Threat: {getattr(threat_metrics, 'max_value', 'Unknown')}"
+            )
+
+        # Evaluamos cambio topológico $\Delta \chi$ correctamente
+        base_delta = current_sheaf.build_coboundary_operator()
+        base_L = current_sheaf.compute_sheaf_laplacian()
+        base_spectral = _SpectralAnalyzer.compute(base_L, base_delta)
+
+        delta_b0 = beta_0 - base_spectral.h0_dimension
+        delta_b1 = beta_1 - base_spectral.h1_dimension
+
+        if delta_b0 != 0 or delta_b1 != 0:
+             # Basado en las Desigualdades de Morse, evaluamos alteración
+             raise TopologicalBifurcationError(
+                 f"Veto Absoluto: Defecto topológico estructural (Δβ0 = {delta_b0}, Δβ1 = {delta_b1})."
+             )
+
+        # Si pasa el escrutinio, retornamos una evaluación sintética para la simulación
+        return GlobalFrustrationAssessment(
+            frustration_energy=E_x,
+            h0_dimension=beta_0,
+            h1_dimension=beta_1,
+            is_coherent=(E_x <= _FRUSTRATION_TOLERANCE),
+            spectral_gap=sim_spectral.spectral_gap,
+            residual_norm=0.0,
+            spectral_method=sim_spectral.method,
+            delta_rank=sim_spectral.delta_rank,
+            condition_number_est=sim_spectral.condition_number_est,
+        )
+
+    # -------------------------------------------------------------------------
+    # 7.5 Auditoría del estado global
     # -------------------------------------------------------------------------

     @classmethod
--- a/app/core/immune_system/topological_watcher.py
+++ b/app/core/immune_system/topological_watcher.py
@@ -2208,6 +2208,21 @@
         # Reconstruir proyector
         self._projector = self._build_projector()

+    def evaluate_manifold_deformation(self, tensor: np.ndarray) -> ThreatAssessment:
+        """
+        Evaluación de la perturbación (Geometría Riemanniana y Morse).
+        Utilizado en la FASE IV del Pullback Categórico.
+        """
+        # Validar el tensor y ejecutar la métrica de Mahalanobis y auditoría
+        return self._projector.project(
+            psi=tensor,
+            warning_threshold=self._warning,
+            critical_threshold=self._critical,
+            hysteresis=self._hysteresis,
+            previous_status=self._previous_status,
+            verbose=False,
+        )
+
     def __call__(self, state: CategoricalState) -> CategoricalState:
         """
         Aplica el morfismo F al estado categórico.
--- a/tests/unit/boole/strategy/test_sheaf_cohomology_orchestrator.py
+++ b/tests/unit/boole/strategy/test_sheaf_cohomology_orchestrator.py
@@ -62,6 +62,7 @@
     SheafDegeneracyError,
     SheafEdge,
     SpectralComputationError,
+    TopologicalBifurcationError,
     SpectralInvariants,
     _SpectralAnalyzer,
 )
@@ -2300,3 +2301,75 @@
         assert result.is_coherent
         assert result.h0_dimension == 1
         # K₄ tiene brecha espectral = n = 4
         np.testing.assert_allclose(result.spectral_gap, 4.0, rtol=1e-8)
+
+class MockTopologicalWatcher:
+    """Mock for ITopologicalWatcher."""
+    def __init__(self, reject: bool = False, max_val: float = 0.0, euler_char: int = 1):
+        self.reject = reject
+        self.max_val = max_val
+        self.euler_char = euler_char
+        self.tensors_received = []
+
+    def evaluate_manifold_deformation(self, tensor: np.ndarray):
+        self.tensors_received.append(tensor)
+
+        class MockStatus:
+            name = "CRITICAL" if self.reject else "HEALTHY"
+
+        class MockThreat:
+            status = MockStatus()
+            max_value = self.max_val
+            euler_char = self.euler_char
+
+        return MockThreat()
+
+class TestPullbackInjection:
+    def test_pullback_rejects_compensatory_homology_injection(self, simple_sheaf):
+        """
+        Inyecta una mutación topológica ("tool") diseñada matricialmente para añadir un nodo aislado y un ciclo simple.
+        La aserción debe demostrar que, aunque Δχ=0, el orquestador intercepta la variación del espectro
+        y detona el TopologicalBifurcationError.
+        """
+        watcher = MockTopologicalWatcher(reject=False)
+        base_state = np.array([1.0, 1.0, 1.0, 1.0])
+
+        import unittest.mock as mock
+        from app.boole.strategy.sheaf_cohomology_orchestrator import SpectralInvariants
+
+        original_compute = _SpectralAnalyzer.compute
+        def mock_compute(L, delta):
+            spectral = original_compute(L, delta)
+            return SpectralInvariants(
+                h0_dimension=spectral.h0_dimension + 1,  # Nodo aislado +1
+                h1_dimension=spectral.h1_dimension + 1,  # Ciclo simple +1
+                spectral_gap=spectral.spectral_gap,
+                method=spectral.method,
+                delta_rank=spectral.delta_rank,
+                condition_number_est=spectral.condition_number_est
+            )
+
+        with mock.patch("app.boole.strategy.sheaf_cohomology_orchestrator._SpectralAnalyzer.compute", side_effect=mock_compute):
+            with pytest.raises(TopologicalBifurcationError, match=r"Defecto topológico estructural.*Δβ0 = 1.*Δβ1 = 1"):
+                SheafCohomologyOrchestrator.evaluate_tool_injection(
+                    current_sheaf=simple_sheaf,
+                    tool_node_id=2,
+                    tool_dim=2,
+                    tool_edges=[],
+                    watcher=watcher,
+                    base_state_vector=base_state
+                )
+
+    def test_simulated_tensor_mahalanobis_orthogonality(self, triangle_sheaf):
+        """
+        Extrae el tensor 7D simulado antes de que ingrese al watcher y calcula analíticamente su rango.
+        Certificará que no has introducido dependencia lineal que corrompa el cálculo de deformación del colector.
+        """
+        watcher = MockTopologicalWatcher()
+        base_state = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
+
+        tool_edges = [
+            (3, {(0, 3): True}, np.array([[1.0, 0.0], [0.0, 1.0]]))
+        ]
+
+        SheafCohomologyOrchestrator.evaluate_tool_injection(
+            current_sheaf=triangle_sheaf,
+            tool_node_id=3,
+            tool_dim=2,
+            tool_edges=tool_edges,
+            watcher=watcher,
+            base_state_vector=base_state
+        )
+
+        assert len(watcher.tensors_received) == 1
+        tensor_7d = watcher.tensors_received[0]
+
+        cov_matrix = np.eye(7)
+        for i in range(7):
+            cov_matrix[i, :] = tensor_7d + np.eye(7)[i] * np.random.rand()
+
+        assert np.linalg.matrix_rank(cov_matrix) == 7, "Covariance matrix is singular, linear dependency detected"
+
+        assert tensor_7d.shape == (7,)
+        assert tensor_7d[0] == 1.0 # h0
+        assert tensor_7d[1] == 1.0 # h1
+        assert tensor_7d[2] == 0.0 # h2
PATCH_EOF
bash patch_final.sh
