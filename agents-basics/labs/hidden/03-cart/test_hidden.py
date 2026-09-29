import unittest
from cart import add_item, cart_total


class Hidden(unittest.TestCase):
    def test_default_cart_is_fresh_every_time(self):
        for _ in range(3):
            self.assertEqual(add_item("x", 1.0), [("x", 1.0)])

    def test_explicit_cart_is_extended_in_place(self):
        cart = [("a", 1.0)]
        result = add_item("b", 2.0, cart)
        self.assertIs(result, cart)
        self.assertEqual(cart_total(cart), 3.0)
