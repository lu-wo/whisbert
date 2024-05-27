import os
import numpy as np
from pytorch_lightning import Callback


class TestPredictionLogger(Callback):
    def __init__(self, output_dir: str = "test_data"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # Save test data and predictions
        np.save(
            f"{self.output_dir}/test_input_ids_{batch_idx}.npy",
            batch["input_ids"].cpu().numpy(),
        )
        np.save(
            f"{self.output_dir}/test_attention_mask_{batch_idx}.npy",
            batch["attention_mask"].cpu().numpy(),
        )
        np.save(
            f"{self.output_dir}/test_labels_{batch_idx}.npy",
            batch["labels"].cpu().numpy(),
        )
        np.save(
            f"{self.output_dir}/test_preds_{batch_idx}.npy",
            outputs["preds"].cpu().numpy(),
        )

        np.save(
            f"{self.output_dir}/test_loss_{batch_idx}.npy",
            outputs["loss"].cpu().numpy(),
        )
