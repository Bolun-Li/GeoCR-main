"""Unit tests for the lightweight GeoCR training CLI helpers."""

import argparse
import unittest

from train import parse_args, parse_ffc_levels


class ParseFfcLevelsTests(unittest.TestCase):
    def test_accepts_valid_levels(self) -> None:
        cases = {
            "3,4,5": (3, 4, 5),
            "P5,P3": (3, 5),
            "4,4": (4,),
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_ffc_levels(value), expected)

    def test_rejects_invalid_levels(self) -> None:
        for value in ("", "2", "3,6", "Pthree"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_ffc_levels(value)


class TrainArgumentsTests(unittest.TestCase):
    def test_geocr_yaml_is_the_default_model(self) -> None:
        args = parse_args([])
        self.assertEqual(args.model, "ultralytics/cfg/geocr/geocr.yaml")


if __name__ == "__main__":
    unittest.main()
