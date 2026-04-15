with open('app/core/immune_system/topological_watcher.py', 'r') as f:
    content = f.read()

# Fix Indentation Error for __slots__ introduced by regex
content = content.replace("        __slots__ = (\n        \"_critical\",", "    __slots__ = (\n        \"_critical\",")

with open('app/core/immune_system/topological_watcher.py', 'w') as f:
    f.write(content)
