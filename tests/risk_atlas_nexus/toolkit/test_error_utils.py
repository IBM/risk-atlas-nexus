# Assisted by watsonx Code Assistant
import unittest

from src.risk_atlas_nexus.toolkit.error_utils import (
    _gen_new_error_code,
    type_check,
    value_check,
)


class TestErrorUtils(unittest.TestCase):

    def test_type_check(self):
        with self.assertRaises(TypeError):
            type_check("type_check_error", int, "value", value="not_an_int")

        with self.assertRaises(RuntimeError):
            type_check("type_check_error", str, None, allow_none=False)

        with self.assertRaises(RuntimeError):
            type_check("type_check_error")

        with self.assertRaises(RuntimeError):
            type_check("type_check_error", int)

    def test_value_check(self):
        val = 25
        with self.assertRaises(ValueError):
            value_check("value_check_error", val > 25, "value is not larger than 25")

        with self.assertRaises(ValueError):
            value_check("value_check_error", 0, "value is not greater than 0")

        with self.assertRaises(ValueError):
            value_check(
                "value_check_error", 0 < val < 1, "value is not between 0 and 1", val
            )

    def test_error_code_gen(self):
        new_code = _gen_new_error_code("RAN", "error")
        self.assertRegex(new_code, r"RAN\d\d\d\d\d\d\d\dE")

        new_code_2 = _gen_new_error_code("TEST", "warning")
        self.assertRegex(new_code_2, r"TEST\d\d\d\d\d\d\d\dW")


if __name__ == "__main__":
    unittest.main()
