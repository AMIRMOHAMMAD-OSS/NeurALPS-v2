"""Static/import checks for training helpers; PyTorch tests skip if unavailable."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment.")
class TrainingTests(unittest.TestCase):
    def test_activity_and_ssl_helpers_import(self):
        from neuralps_v2.training_objectives import activity_loss, object_ssl_loss
        from neuralps_v2.phase2_training import configure_stage, config_fingerprint
        self.assertTrue(callable(activity_loss))
        self.assertTrue(callable(object_ssl_loss))
        self.assertEqual(len(config_fingerprint({"b": 2, "a": 1})), 64)


if __name__ == "__main__":
    unittest.main()
