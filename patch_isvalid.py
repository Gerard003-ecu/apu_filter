with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

# Let's fix the verify_cochain_complex function so rank_B2_ok is checked properly.
# The original patch didn't change rank_B2_expected and rank_B2_ok properly for the returned dict?
# In the test output: 'dimensions_consistent': False, 'rank_B2_expected': 1, 'rank_B2_ok': False
import re
match = re.search(r'def verify_cochain_complex\(self\).*?return \{.*?\}', code, flags=re.DOTALL)
if match:
    old = match.group(0)
    print("Found verify")
