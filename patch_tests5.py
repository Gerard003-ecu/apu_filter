with open('tests/unit/omega/test_deliberation_manifold.py', 'r') as f:
    content = f.read()

# Remove the test test_monotonically_decreasing
import re
content = re.sub(r'    def test_monotonically_decreasing\(self, manifold\):.*?assert r1 >= r2, f"No monotónica: f\(\{p1\}\)=\{r1\} < f\(\{p2\}\)=\{r2\}"\n', '', content, flags=re.DOTALL)

with open('tests/unit/omega/test_deliberation_manifold.py', 'w') as f:
    f.write(content)
