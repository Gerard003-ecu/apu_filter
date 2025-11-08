import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.apu_processor import APUProcessor
from tests.test_data import APUS_DATA, TEST_CONFIG

def test_apu_processor_produces_correct_columns():
    """
    Prueba que APUProcessor genere un DataFrame con las columnas esperadas.
    """
    # Simular la estructura de `raw_records` que `ReportParserCrudo` generaría
    raw_records = []
    current_apu = {}
    for line in APUS_DATA.strip().split('\n'):
        if line.startswith('ITEM:'):
            parts = [p.strip() for p in line.split(';')]
            current_apu = {
                "apu_code": parts[0].replace('ITEM:', '').strip(),
                "apu_unit": parts[1].replace('UNIDAD:', '').strip(),
                "apu_desc": "", # No es relevante
            }
        elif line.strip() and not line.isupper():
            raw_records.append({
                **current_apu,
                "insumo_line": line,
                "category": "OTRO", # No es relevante
            })

    processor = APUProcessor(raw_records, TEST_CONFIG)
    df_result = processor.process_all()

    assert not df_result.empty, "El DataFrame resultante no puede estar vacío"
    assert "descripcion_insumo" in df_result.columns, "La columna 'descripcion_insumo' es obligatoria"
