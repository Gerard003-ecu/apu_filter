import re

with open('app/omega/deliberation_manifold.py', 'r') as f:
    content = f.read()

# Dynamic Fiedler extraction mapping logic:
# Find fiedler value in inputs.topo_data
# Recalculate eps_ref based on fiedler
# Then compute fragility_norm and roi_norm

compute_fragility_str = """
    @staticmethod
    def _compute_fragility_normalized(psi: float, fiedler_value: float = 0.0) -> float:
        safe_psi = max(psi, _EPSILON)
        raw = math.log2(1.0 + 1.0 / safe_psi)
        kappa = 0.1 # basal coupling constant
        eps_ref = kappa / (fiedler_value + 0.1) # eps is already included in 0.1
        norm_denominator = math.log2(1.0 + 1.0 / eps_ref)
        return _clamp(raw / norm_denominator, 0.0, 1.0)
"""

content = re.sub(r'@staticmethod\n    def _compute_fragility_normalized\(psi: float\) -> float:.*?return _clamp\(raw / _NORM_DENOMINATOR, 0\.0, 1\.0\)', compute_fragility_str.strip(), content, flags=re.DOTALL)

compute_roi_str = """
    @staticmethod
    def _compute_roi_normalized(roi: float, fiedler_value: float = 0.0) -> float:
        safe_roi = max(roi, _EPSILON)
        raw = math.log2(1.0 + 1.0 / safe_roi)
        kappa = 0.1
        eps_ref = kappa / (fiedler_value + 0.1)
        norm_denominator = math.log2(1.0 + 1.0 / eps_ref)
        return _clamp(raw / norm_denominator, 0.0, 1.0)
"""
content = re.sub(r'@staticmethod\n    def _compute_roi_normalized\(roi: float\) -> float:.*?return _clamp\(raw / _NORM_DENOMINATOR, 0\.0, 1\.0\)', compute_roi_str.strip(), content, flags=re.DOTALL)

# Update _compute_metrics:
compute_metrics_replace = """
        # --- Normalización al espacio métrico unificado [0,1] ---
        fiedler_value = float(inputs.topo_data.get("fiedler_value", 0.0)) if isinstance(inputs.topo_data, dict) else 0.0
        fragility_norm = self._compute_fragility_normalized(inputs.psi, fiedler_value)
        roi_norm = self._compute_roi_normalized(inputs.roi, fiedler_value)
"""
content = re.sub(r'# --- Normalización al espacio métrico unificado \[0,1\] ---\n\s*fragility_norm = self\._compute_fragility_normalized\(inputs\.psi\)\n\s*roi_norm = self\._compute_roi_normalized\(inputs\.roi\)', compute_metrics_replace.strip('\n'), content)


# Phase transition C^infty
# g(n)=1.0+(Gmax​−1.0)⋅tanh(α⋅n/(Gmax​−1.0​))
# Replace _clamp on gauge factor with this formula
gauge_replace = """
        # --- Factor de Gauge (Acoplamiento TOON) ---
        # CORRECCIÓN: Fase continua C^∞ mediante transición hiperbólica.
        n_cartridges = self.synaptic_registry.cartridge_count
        gauge_deflection = 1.0 + (_GAUGE_MAX - 1.0) * math.tanh((_GAUGE_ALPHA * n_cartridges) / (_GAUGE_MAX - 1.0))
"""
content = re.sub(r'# --- Factor de Gauge \(Acoplamiento TOON\) ---\n\s*# CORRECCIÓN: gauge acotado en \[1\.0, _GAUGE_MAX\] para garantizar finitud\.\n\s*n_cartridges = self\.synaptic_registry\.cartridge_count\n\s*gauge_deflection = _clamp\(\n\s*1\.0 \+ _GAUGE_ALPHA \* n_cartridges,\n\s*1\.0,\n\s*_GAUGE_MAX,\n\s*\)', gauge_replace.strip('\n'), content)


# Refactor _project_to_lattice Alexandroff compactification mapping infinite/NaN to VerdictLevel.RECHAZAR by stereo projection logic
project_replace = """
    @staticmethod
    def _project_to_lattice(adjusted_stress: float) -> VerdictLevel:
        # Compactificación de Alexandroff: mapeo del infinito y NaN al polo norte (Supremo ⊤).
        if not math.isfinite(adjusted_stress) or math.isnan(adjusted_stress):
            return VerdictLevel.RECHAZAR

        if adjusted_stress < _VERDICT_THRESHOLD_VIABLE:
            return VerdictLevel.VIABLE
        if adjusted_stress < _VERDICT_THRESHOLD_CONDICIONAL:
            return VerdictLevel.CONDICIONAL
        if adjusted_stress < _VERDICT_THRESHOLD_PRECAUCION:
            return VerdictLevel.PRECAUCION
        return VerdictLevel.RECHAZAR
"""
content = re.sub(r'@staticmethod\n\s*def _project_to_lattice\(adjusted_stress: float\) -> VerdictLevel:.*?(?=    def _build_diagnostics)', project_replace.strip('\n') + '\n\n', content, flags=re.DOTALL)


with open('app/omega/deliberation_manifold.py', 'w') as f:
    f.write(content)
