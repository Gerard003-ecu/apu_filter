import os
import sys
import unittest
import logging

# Configure logging to display debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.report_parser import ReportParser

class TestReportParserWithRealData(unittest.TestCase):
    """
    Test suite for ReportParser with the actual apus.csv file.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """
        cls.test_file_path = "apus.csv"
        cls.parser = ReportParser(cls.test_file_path)
        cls.df = cls.parser.parse()

        # Print the DataFrame for debugging if necessary
        logging.debug("Parsed DataFrame for testing with real data:")
        logging.debug(cls.df.to_string())

    def test_dataframe_is_not_empty_with_real_data(self):
        """
        Tests that the DataFrame is not empty when parsing the real apus.csv.
        """
        self.assertFalse(self.df.empty, "The DataFrame should not be empty when parsing apus.csv.")

    def test_finds_a_significant_number_of_apus(self):
        """
        Tests that the parser identifies a significant number of unique APUs.
        """
        apu_codes = self.df["CODIGO_APU"].unique()
        self.assertTrue(len(apu_codes) > 5, f"Expected more than 5 APUs, but found {len(apu_codes)}.")

    def test_specific_insumo_is_parsed_correctly(self):
        """
        Tests that a specific known insumo is parsed correctly.
        """
        # Example insumo from apus.csv
        # LAMINA DE 1.22 X 3.05 MTS CAL. 22 PINTADA INCLUIDO IVA;UND;0,33;14,04;174.928,81;65.403,35
        insumo = self.df[self.df["DESCRIPCION_INSUMO"] == "LAMINA DE 1.22 X 3.05 MTS CAL. 22 PINTADA INCLUIDO IVA"]
        self.assertTrue(not insumo.empty, "The specific insumo 'LAMINA...' was not found.")

        insumo_data = insumo.iloc[0]
        self.assertEqual(insumo_data["UNIDAD"], "UND")
        self.assertAlmostEqual(insumo_data["CANTIDAD_APU"], 0.33, places=2)
        self.assertAlmostEqual(insumo_data["PRECIO_UNIT_APU"], 174928.81, places=2)
        self.assertAlmostEqual(insumo_data["VALOR_TOTAL_APU"], 65403.35, places=2)
        self.assertEqual(insumo_data["CATEGORIA"], "MATERIALES")

if __name__ == "__main__":
    unittest.main()