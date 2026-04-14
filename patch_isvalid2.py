import re

with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

def repl(match):
    return '''def verify_cochain_complex(self) -> Dict[str, Any]:
        """
        Verificación formal de las propiedades del complejo de cadenas.

        Propiedades verificadas:
            1. ‖B₁B₂‖_F ≤ tol             (∂₁ ∘ ∂₂ = 0)
            2. Dimensiones consistentes
            3. rank(B₁) = n − c
            4. rank(B₂) = 0 en grafo 1D
            5. Euler–Poincaré: χ = n − m = β₀ − β₁

        Returns:
            Dict con resultados booleanos y métricas numéricas.
        """
        B1, meta1 = self.build_incidence_matrix()
        B2, meta2 = self.build_face_matrix()

        undirected = self.G.to_undirected()
        c = nx.number_connected_components(undirected)
        betti_0 = c
        betti_1 = meta2["betti_1"]

        B1B2_zero = meta2["verify_B1B2_zero"]
        B1B2_norm = meta2["B1B2_norm"]

        dims_ok = (
            B1.shape == (self.n, self.m)
            and B2.shape == (self.m, 0)
        )

        rank_B1 = meta1["rank_B1"]
        rank_B1_expected = self.n - c
        rank_B1_ok = rank_B1 == rank_B1_expected

        rank_B2 = meta2["rank_B2"]
        rank_B2_ok = rank_B2 == 0

        chi = self.n - self.m
        chi_topological = betti_0 - betti_1
        euler_ok = (chi == chi_topological)

        is_valid = B1B2_zero and dims_ok and rank_B1_ok and rank_B2_ok and euler_ok

        return {
            "is_valid": is_valid,
            "B1B2_zero": B1B2_zero,
            "B1B2_norm": B1B2_norm,
            "dimensions_consistent": dims_ok,
            "rank_B1": rank_B1,
            "rank_B1_expected": rank_B1_expected,
            "rank_B1_ok": rank_B1_ok,
            "rank_B2": rank_B2,
            "rank_B2_expected": 0,
            "rank_B2_ok": rank_B2_ok,
            "chi_geometric": chi,
            "chi_topological": chi_topological,
            "euler_poincare_ok": euler_ok,
            "beta_0": betti_0,
            "beta_1": betti_1,
            "connected_components": c,
        }'''

code = re.sub(r'def verify_cochain_complex\(self\) -> Dict\[str, Any\]:.*?return \{.*?\}', repl, code, flags=re.DOTALL)

with open('app/physics/solenoid_acustic.py', 'w') as f:
    f.write(code)
