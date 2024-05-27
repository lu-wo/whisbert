import math
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError
from torchmetrics import MeanAbsoluteError
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.regression import R2Score
from torchmetrics.functional.regression.r2 import _r2_score_compute, _r2_score_update
from torchmetrics.regression.spearman import SpearmanCorrCoef
from torch.nn.functional import gaussian_nll_loss
from torchmetrics.functional.regression.spearman import (
    _spearman_corrcoef_compute,
    _spearman_corrcoef_update,
)
from torchmetrics.regression.pearson import PearsonCorrCoef
from torchmetrics.functional.regression.pearson import (
    _pearson_corrcoef_compute,
    _pearson_corrcoef_update,
)


class MeanMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        if torch.isnan(loss).any():
            print("Warning: Loss contains NaN values. Skipping update.")
            return
        self.sum += torch.sum(loss)
        self.total += loss.numel()

    def compute(self):
        return self.sum.float() / self.total


class MaskedPearsonCorrCoef(PearsonCorrCoef):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        # apply mask
        preds = preds[mask == 1]
        target = target[mask == 1]

        """Update state with predictions and targets."""
        (
            self.mean_x,
            self.mean_y,
            self.var_x,
            self.var_y,
            self.corr_xy,
            self.n_total,
        ) = _pearson_corrcoef_update(
            preds,
            target,
            self.mean_x,
            self.mean_y,
            self.var_x,
            self.var_y,
            self.corr_xy,
            self.n_total,
            self.num_outputs,
        )


class MaskedSpearmanCorrCoeff(SpearmanCorrCoef):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        """Update state with predictions and targets."""
        preds = preds[mask == 1]
        target = target[mask == 1]

        # check whether these are valid
        if preds.numel() == 0 or target.numel() == 0:
            return

        print(f"preds: {preds}, target: {target}")

        # check whether these are valid
        if preds.numel() == 0 or target.numel() == 0:
            return

        # check whether there is a nan in the preds and target
        if torch.isnan(preds).any():
            print("preds has nan")
            raise ValueError("preds has nan")
        if torch.isnan(target).any():
            print("target has nan")
            raise ValueError("target has nan")

        preds, target = _spearman_corrcoef_update(
            preds, target, num_outputs=self.num_outputs
        )
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes Spearman's correlation coefficient."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _spearman_corrcoef_compute(preds, target, eps=1e-4)


class MaskedR2Score(R2Score):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        """Update state with predictions, targets, and mask.

        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: Mask to apply on the loss
        """
        # Compute the element-wise squared error
        # apply the mask to the squared error
        # print(preds, target, mask)
        masked_preds = preds[mask == 1]
        masked_target = target[mask == 1]
        # print(masked_preds, masked_target)
        if len(masked_preds) < 2:
            return  # r2 not defined for 1 sample

        # Update the state
        sum_squared_error, sum_error, residual, total = _r2_score_update(
            masked_preds, masked_target
        )
        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total
        # torch_r2_score = _r2_score_compute(
        #     sum_squared_error,
        #     sum_error,
        #     residual,
        #     total,
        #     self.adjusted,
        #     self.multioutput,
        # )
        # print(f"Torchmetrics R2 score: {torch_r2_score}")

        # # sklearn r2 score
        # sklearn_r2_score = r2_score(masked_target.cpu(), masked_preds.cpu())
        # print(f"Sklearn R2 score: {sklearn_r2_score}")


class MaskedMeanSquaredError(MeanSquaredError):
    r"""Computes masked mean squared error (MSE) given a mask:

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N w_i(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions, and :math:`w_i` is the mask weight for each sample.

    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> mask = torch.tensor([1, 1, 0, 1], dtype=torch.float)
        >>> masked_mean_squared_error = MaskedMeanSquaredError()
        >>> masked_mean_squared_error(preds, target, mask)
        tensor(0.6667)
    """

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        """Update state with predictions, targets, and mask.

        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: Mask to apply on the loss
        """
        # Compute the element-wise squared error
        squared_error = torch.square(preds - target)

        # Apply the mask to the squared error
        masked_squared_error = squared_error * mask

        # Update the state
        self.sum_squared_error += torch.sum(masked_squared_error)
        self.total += torch.sum(mask).to(torch.long)


class MaskedMeanAbsoluteError(MeanAbsoluteError):
    r"""Computes masked mean absolute error (MAE) given a mask:

    .. math:: \text{MAE} = \frac{1}{N}\sum_i^N w_i|y_i - \hat{y_i}|

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions, and :math:`w_i` is the mask weight for each sample.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> mask = torch.tensor([1, 1, 0, 1], dtype=torch.float)
        >>> masked_mean_absolute_error = MaskedMeanAbsoluteError()
        >>> masked_mean_absolute_error(preds, target, mask)
        tensor(0.6667)
    """

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        """Update state with predictions, targets, and mask.

        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: Mask to apply on the loss
        """
        # Compute the element-wise absolute error
        absolute_error = torch.abs(preds - target)

        # Apply the mask to the absolute error
        masked_absolute_error = absolute_error * mask

        # Update the state
        self.sum_abs_error += torch.sum(masked_absolute_error)
        self.total += torch.sum(mask).to(torch.long)


class MaskedAccuracy(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        # process_group: Any = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            # process_group=process_group,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        preds = torch.argmax(preds, dim=-1)
        correct = (preds == target) * mask.bool()
        self.correct += torch.sum(correct)
        self.total += torch.sum(mask)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
