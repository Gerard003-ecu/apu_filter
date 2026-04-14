with open('tests/unit/physics/test_solenoid_acustic.py', 'r') as f:
    code = f.read()

# Fix indentations
code = code.replace("        def test_betti_rank_nullity_invariant(self) -> None:", "    def test_betti_rank_nullity_invariant(self) -> None:")
code = code.replace("""    def test_betti_rank_nullity_invariant(self) -> None:
        G, inv = GraphFactory.two_triangles_shared_vertex()
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, _ = builder.build_face_matrix()

        n, m = B1.shape""", """    def test_betti_rank_nullity_invariant(self) -> None:
        G, inv = GraphFactory.two_triangles_shared_vertex()
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, _ = builder.build_face_matrix()

        n, m = B1.shape""")

idx = code.find('def test_betti_rank_nullity_invariant(self) -> None:')
if idx != -1:
    print("Found test_betti_rank_nullity_invariant")

code = code.replace("def test_B1_times_B2_equals_zero(self, factory):", "    def test_B1_times_B2_equals_zero(self, factory):")

with open('tests/unit/physics/test_solenoid_acustic.py', 'w') as f:
    f.write(code)
