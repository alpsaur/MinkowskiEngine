"""Optional real-data jobs must not run merely because Open3D is installed."""

import os
import unittest
from unittest.mock import patch

import pytest
from tests.python.common import load_file


class TestDataOptIn(unittest.TestCase):
    def test_default_does_not_download_or_load_data(self):
        with patch.dict(os.environ, {"ME_RUN_DATA_TESTS": "0"}):
            with patch("tests.python.common.urlretrieve") as download:
                with self.assertRaises(pytest.skip.Exception):
                    load_file("deliberately-missing-point-cloud.ply")
                download.assert_not_called()
