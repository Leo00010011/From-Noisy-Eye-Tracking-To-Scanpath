import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import seq2seq_padded_collate_fn
from src.model.blocks import DeformableAttention, MultiHeadedAttention
from src.training.inference_recorder import InferenceRecorder


class TinyRecorderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = MultiHeadedAttention(
            model_dim=4,
            total_dim=4,
            n_heads=2,
            is_self_attention=False,
        )
        self.deformable_attention = DeformableAttention(
            embed_dim=4,
            num_heads=2,
            num_points=2,
            dropout=0.0,
        )

    def forward(self):
        query = torch.randn(2, 3, 4)
        key = torch.randn(2, 5, 4)
        value = torch.randn(2, 16, 4)
        reference_points = torch.rand(2, 3, 2)

        cross_output = self.cross_attention(query, key)
        deform_output = self.deformable_attention(query, reference_points, value, (4, 4))
        return {
            "coord": cross_output[..., :2],
            "dur": cross_output[..., :1],
            "denoise": deform_output[..., :2],
            "cls": cross_output[..., :1],
        }


class InferenceRecorderTests(unittest.TestCase):
    def test_seq2seq_padded_collate_fn_keeps_traceability_indices(self):
        batch = [
            {
                "x": np.array([[1.0, 2.0], [3.0, 4.0], [0.1, 0.2]], dtype=np.float32),
                "y": np.array([[5.0], [6.0], [0.3]], dtype=np.float32),
                "sample_idx": 11,
                "image_idx": 7,
            },
            {
                "x": np.array([[7.0], [8.0], [0.4]], dtype=np.float32),
                "y": np.array([[9.0, 10.0], [11.0, 12.0], [0.5, 0.6]], dtype=np.float32),
                "sample_idx": 13,
                "image_idx": 9,
            },
        ]

        output = seq2seq_padded_collate_fn(batch)

        self.assertTrue(torch.equal(output["sample_idx"], torch.tensor([11, 13], dtype=torch.long)))
        self.assertTrue(torch.equal(output["image_idx"], torch.tensor([7, 9], dtype=torch.long)))

    def test_inference_recorder_saves_inputs_outputs_and_activations(self):
        model = TinyRecorderModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = InferenceRecorder(temp_dir)
            recorder.attach(model)

            batch = {
                "sample_idx": torch.tensor([2, 4], dtype=torch.long),
                "image_idx": torch.tensor([10, 12], dtype=torch.long),
                "src": torch.randn(2, 5, 3),
                "tgt": torch.randn(2, 3, 3),
            }

            recorder.start_batch(epoch=3, phase="Combined", split="train", batch_index=1, global_step=8)
            output = model()
            recorder.record_batch(batch, output)
            saved_path = recorder.save_batch()

            payload = torch.load(saved_path)
            expected_path = Path(temp_dir) / "train_epoch_0003_batch_00001_step_000008_Combined.pt"

            self.assertEqual(saved_path, expected_path)
            self.assertEqual(payload["metadata"]["epoch"], 3)
            self.assertEqual(payload["metadata"]["phase"], "Combined")
            self.assertTrue(torch.equal(payload["data"]["sample_idx"], batch["sample_idx"]))
            self.assertTrue(torch.equal(payload["data"]["image_idx"], batch["image_idx"]))
            self.assertIn("denoise_output", payload["outputs"])
            self.assertIn("scanpath_coordinates", payload["outputs"])
            self.assertIn("scanpath_duration", payload["outputs"])
            self.assertIn("cross_attention", payload["activations"])
            self.assertIn("attention_weights", payload["activations"]["cross_attention"])
            self.assertEqual(payload["activations"]["cross_attention"]["attention_weights"].shape, (2, 2, 3, 5))
            self.assertIn("deformable_attention", payload["activations"])
            self.assertIn("sampling_offsets", payload["activations"]["deformable_attention"])
            self.assertIn("sampling_locations", payload["activations"]["deformable_attention"])


if __name__ == "__main__":
    unittest.main()
