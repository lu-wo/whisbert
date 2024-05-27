import random
import torch
import pytest
from sklearn.metrics import r2_score
from src.utils.torch_metrics import MaskedR2Score, MaskedMeanSquaredError


@pytest.mark.slow
def test_masked_r2_score():
    num_tests = 1000
    list_length = 100

    for i in range(num_tests):
        y_true = [random.uniform(-10, 10) for _ in range(list_length)]
        y_pred = [random.uniform(-10, 10) for _ in range(list_length)]
        mask = [random.choice([0, 1]) for _ in range(list_length)]

        r2_metric = MaskedR2Score()

        masked_y_true = [y_true[i] for i in range(len(y_true)) if mask[i]]
        masked_y_pred = [y_pred[i] for i in range(len(y_pred)) if mask[i]]

        sklearn_r2_score = r2_score(masked_y_true, masked_y_pred)
        masked_r2_score = r2_metric(
            torch.tensor(y_pred), torch.tensor(y_true), torch.tensor(mask)
        )

        assert round(sklearn_r2_score, 3) == round(
            masked_r2_score.item(), 3
        ), f"Scores do not match in test {i+1}"


@pytest.mark.slow
def test_masked_mse():
    num_tests = 1000
    list_length = 100

    for i in range(num_tests):
        y_true = [random.uniform(-10, 10) for _ in range(list_length)]
        y_pred = [random.uniform(-10, 10) for _ in range(list_length)]
        mask = [random.choice([0, 1]) for _ in range(list_length)]

        mse_metric = MaskedMeanSquaredError()

        masked_y_true = [y_true[i] for i in range(len(y_true)) if mask[i]]
        masked_y_pred = [y_pred[i] for i in range(len(y_pred)) if mask[i]]

        sklearn_mse = sum(
            [
                (masked_y_true[i] - masked_y_pred[i]) ** 2
                for i in range(len(masked_y_true))
            ]
        ) / len(masked_y_true)
        masked_mse = mse_metric(
            torch.tensor(y_pred), torch.tensor(y_true), torch.tensor(mask)
        )

        assert round(sklearn_mse, 2) == round(
            masked_mse.item(), 2
        ), f"Scores do not match in test {i+1}"
