import re

with open('app/tactics/pipeline_director.py', 'r') as f:
    content = f.read()

# Let's change STRATUM_EVIDENCE for TACTICS to only require df_apu_costos
old_evidence = '    Stratum.TACTICS: ("df_apu_costos", "df_tiempo", "df_rendimiento"),'
new_evidence = '    Stratum.TACTICS: ("df_apu_costos",),'

content = content.replace(old_evidence, new_evidence)

with open('app/tactics/pipeline_director.py', 'w') as f:
    f.write(content)
