import unittest
from app.utils import parse_number

class TestParseNumber(unittest.TestCase):
    """Suite de pruebas para la función parse_number en app/utils.py."""

    def test_basic_conversions(self):
        """Prueba conversiones numéricas básicas."""
        self.assertEqual(parse_number("123"), 123.0)
        self.assertEqual(parse_number("123.45"), 123.45)
        self.assertEqual(parse_number("-50"), -50.0)
        self.assertEqual(parse_number(123), 123.0)
        self.assertEqual(parse_number(123.45), 123.45)

    def test_separators(self):
        """Prueba el manejo de separadores de miles y decimales."""
        self.assertEqual(parse_number("1,234.56"), 1234.56)
        self.assertEqual(parse_number("1.234,56", decimal_separator="comma"), 1234.56)
        self.assertEqual(parse_number("1,234,567.89"), 1234567.89)
        self.assertEqual(parse_number("1.234.567,89", decimal_separator="comma"), 1234567.89)

    def test_currency_and_spaces(self):
        """Prueba el manejo de símbolos de moneda y espacios."""
        self.assertEqual(parse_number("€ 1,500.00"), 1500.0)
        self.assertEqual(parse_number("$ 1,208.12"), 1208.12)
        self.assertEqual(parse_number("   123.45   "), 123.45)

    def test_percentages(self):
        """Prueba el manejo de porcentajes."""
        self.assertAlmostEqual(parse_number("15%"), 0.15)
        self.assertAlmostEqual(parse_number("1,5%"), 0.015)
        self.assertAlmostEqual(parse_number("1.5%"), 0.015)

    def test_scientific_notation(self):
        """Prueba el manejo de notación científica."""
        self.assertAlmostEqual(parse_number("1.5e-3"), 0.0015)
        self.assertAlmostEqual(parse_number("1.5E+3"), 1500.0)

    def test_edge_cases_and_defaults(self):
        """Prueba casos límite y el valor por defecto."""
        self.assertEqual(parse_number("N/A", default_value=-1), -1.0)
        self.assertEqual(parse_number(None, default_value=-1), -1.0)
        self.assertEqual(parse_number(""), 0.0)
        self.assertEqual(parse_number("  "), 0.0)
        self.assertEqual(parse_number("not a number"), 0.0)

    def test_strict_mode(self):
        """Prueba el modo estricto que debe lanzar excepciones."""
        with self.assertRaises(ValueError):
            parse_number("not a number", strict=True)
        with self.assertRaises(ValueError):
            parse_number("N/A", strict=True)

if __name__ == "__main__":
    unittest.main()
