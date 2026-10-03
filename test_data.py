"""CPU-only regression checks; no dataset downloads or ML framework needed."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from mri_common import (CLASSES, discover, make_splits, validate_manifest,
                        training_records, load_image, class_weights, report)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for label, name in enumerate(CLASSES):
            folder = self.root / name
            folder.mkdir()
            for patient in range(10):
                for section in range(2):
                    Image.new('RGB', (16, 16), (100, 100, 100)).save(
                        folder / f'OAS1_{label * 100 + patient:04}_MR1_{section}.png')
        _, self.rows = discover(self.root)
        self.rows = make_splits(self.rows, 'patient', 42, r'(OAS1_\d+)_')

    def test_split_disjoint_and_repeatable(self):
        validate_manifest(self.root, self.rows, 'patient', r'(OAS1_\d+)_')
        _, fresh = discover(self.root)
        self.assertEqual(self.rows, make_splits(fresh, 'patient', 42, r'(OAS1_\d+)_'))
        partitions = [{r['patient_id'] for r in self.rows if r['split'] == s} for s in ['train', 'val', 'test']]
        for i in range(3):
            for j in range(i):
                self.assertFalse(partitions[i] & partitions[j])

    def test_tampered_manifest_fails(self):
        self.rows[0]['patient_id'] = 'wrong'
        with self.assertRaises(ValueError):
            validate_manifest(self.root, self.rows, 'patient', r'(OAS1_\d+)_')

    def test_rare_class_fails(self):
        rows = [dict(r) for r in self.rows if r['label'] != 1 or r['patient_id'] == 'OAS1_0100']
        with self.assertRaises(ValueError):
            make_splits(rows, 'patient', 42, r'(OAS1_\d+)_')

    def test_noise_and_counts(self):
        train = [r for r in self.rows if r['split'] == 'train']
        records = training_records(train, 42)
        expected = sum({0: 6, 1: 6, 2: 1, 3: 3}[r['label']] for r in train)
        self.assertEqual(len(records), expected)
        self.assertEqual(len(training_records(train, 42, False)), len(train))
        noisy = next(r for r in records if r['noise_variant'])
        path = self.root / noisy['path']
        original = path.read_bytes()
        a = load_image(self.root, noisy, 32, 42, .1)
        b = load_image(self.root, noisy, 32, 42, .1)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(original, path.read_bytes())
        self.assertEqual(int(np.any(a != 100, axis=2).sum()), int(.1 * 32 * 32))
        weights = class_weights(records)
        counts = np.bincount([r['label'] for r in records])
        np.testing.assert_allclose(counts * weights, np.repeat(len(records) / 4, 4), rtol=1e-6)

    def test_report_mapping(self):
        rows = [r for r in self.rows if r['split'] == 'test']
        probabilities = np.eye(4)[[r['label'] for r in rows]]
        history = dict(loss=[1], val_loss=[1], accuracy=[1], val_accuracy=[1])
        report(self.root, rows, probabilities, history)
        import json
        metrics = json.loads((self.root / 'metrics.json').read_text())
        for name in CLASSES:
            self.assertEqual(metrics[name]['f1-score'], 1)


if __name__ == '__main__':
    unittest.main()
