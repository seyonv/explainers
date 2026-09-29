import unittest
from orders import total_units


class Hidden(unittest.TestCase):
    def test_units(self):
        orders = [{"sku": s, "quantity": q, "unit_price": 1.0} for s, q in [("x", 1), ("y", 10), ("z", 0)]]
        self.assertEqual(total_units(orders), 11)
        self.assertEqual(total_units([]), 0)
