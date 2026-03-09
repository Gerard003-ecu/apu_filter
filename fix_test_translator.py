import re
import os

content = open('tests/unit/wisdom/test_semantic_translator.py').read()

# Fix stability thresholds init
content = content.replace(
'''        config = TranslatorConfig(
            stability_thresholds=stability or StabilityThresholds(),
            topological_thresholds=topological or TopologicalThresholds(),
            thermal_thresholds=thermal or ThermalThresholds(),
            financial_thresholds=financial or FinancialThresholds(),
        )''',
'''        config = TranslatorConfig(
            stability=stability or StabilityThresholds(),
            topological=topological or TopologicalThresholds(),
            thermal=thermal or ThermalThresholds(),
            financial=financial or FinancialThresholds(),
        )'''
)

with open('tests/unit/wisdom/test_semantic_translator.py', 'w') as f:
    f.write(content)
