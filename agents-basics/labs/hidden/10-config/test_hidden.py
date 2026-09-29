import unittest
import config


class Hidden(unittest.TestCase):
    def test_defaults_untouched(self):
        before = dict(config.DEFAULTS)
        result = config.with_defaults({"font_size": 20, "extra": 1})
        self.assertEqual(config.DEFAULTS, before)
        self.assertEqual(result, {"theme": "light", "font_size": 20, "autosave": True, "extra": 1})
        self.assertIsNot(config.with_defaults({}), config.DEFAULTS)
