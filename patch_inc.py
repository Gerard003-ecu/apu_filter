with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

# Fix max of empty array
code = code.replace("col_sum_max = float(np.max(np.abs(col_sums)))", "col_sum_max = float(np.max(np.abs(col_sums))) if col_sums.size > 0 else 0.0")

with open('app/physics/solenoid_acustic.py', 'w') as f:
    f.write(code)
