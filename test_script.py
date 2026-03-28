import re

content = open("app/agents/MIC_agent.py").read()
print(bool(re.search("class TOONCompressor", content)))
