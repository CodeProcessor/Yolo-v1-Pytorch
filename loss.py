
import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=1, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5)
        exists_box = target[..., self.num_classes].unsqueeze(3)

        # ===================== #
        #  Calculate box loss   #
        # ===================== #

        box_predictions = exists_box * predictions[..., self.num_classes+1:self.num_classes+5]

        box_targets = exists_box * target[..., self.num_classes+1:self.num_classes+5]

        box_predictions[..., 2:4] = torch.sign(
            box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ======================================= #
        #  Calculate object loss if object exists #
        # ======================================= #

        box_pred_confidence = exists_box * predictions[..., self.num_classes:self.num_classes+1]
        box_targets_confidence = exists_box * target[..., self.num_classes:self.num_classes+1]

        object_loss = self.mse(
            torch.flatten(box_pred_confidence),
            torch.flatten(box_targets_confidence)
        )
        # ========================== #
        #  Calculate no object loss  #
        # ========================== #

        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., self.num_classes:self.num_classes+1], start_dim=1),
            torch.flatten((1-exists_box) * target[..., self.num_classes:self.num_classes+1], start_dim=1)
        )

        # ====================== #
        #  Calculate class loss  #
        # ====================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.num_classes], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.num_classes], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_no_obj * no_object_loss +
            class_loss
        )

        return loss
