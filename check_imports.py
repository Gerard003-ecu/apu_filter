import ast
import os

def check_imports(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=filepath)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if "estimator" in alias.name:
                                    print(f"{filepath}: import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and "estimator" in node.module:
                                print(f"{filepath}: from {node.module} import ...")
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

print("Checking app/...")
check_imports("app")
print("Checking tests/...")
check_imports("tests")
