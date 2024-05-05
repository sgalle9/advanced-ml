import os
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter


class WeightedCE:
    """Custom loss function."""

    def __init__(self, class_weight, data_weight):
        self.class_weight = class_weight
        self.data_weight = data_weight
        self.cross_entropy = CrossEntropyLoss(
            weight=self.class_weight, reduction="none"
        )

    def to(self, device):
        """Assigns tensors to the specified device (CPU or GPU)."""
        self.class_weight = self.class_weight.to(device)
        self.data_weight = self.data_weight.to(device)
        self.cross_entropy.to(device)
        return self

    def __call__(self, outputs, targets, data_source):
        weights = self.data_weight[data_source]
        loss = self.cross_entropy(outputs, targets).mean(dim=(1, 2))
        loss = (loss * weights).mean()
        return loss


class Trainer:
    """Trainer class for managing and executing the training of a model."""

    def __init__(
        self,
        train_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        device,
        path_to_out,
    ) -> None:

        self.device = device
        self.n_epochs = n_epochs

        self.train_loader = train_loader
        self.model = model.to(self.device)

        # If stateful loss function, move its "parameters" to `device`.
        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.path_to_out = Path(path_to_out)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        self.writer = SummaryWriter(
            self.path_to_out / f"runs/{type(self.model).__name__}_{self.timestamp}"
        )

        # Log model's architecture.
        frames, _, _ = next(iter(train_loader))
        self.writer.add_graph(self.model, frames.to(self.device))

    def _train_one_epoch(self, epoch_idx):
        self.model.train()
        # Also known as "empirical risk".
        avg_loss = 0.0

        for batch_idx, (inputs, targets, data_src) in enumerate(self.train_loader):
            inputs, targets, data_src = (
                inputs.to(self.device),
                targets.to(self.device),
                data_src.to(self.device),
            )

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            # Moving average (with "stride" `batch_size`) of the loss.
            batch_loss = self.loss_fn(outputs, targets, data_src)

            batch_loss.backward()
            self.optimizer.step()

            avg_loss += batch_loss.item() * len(inputs)
            # if batch_idx % 5 == 0:
            #     print(f"    batch {batch_idx+1} loss: {batch_loss.item()}")

        avg_loss = avg_loss / len(self.train_loader.dataset)
        print(f"EPOCH {epoch_idx}    empirical risk: {avg_loss}\n")

        self.writer.add_scalar("Loss/train", avg_loss, epoch_idx + 1)
        self.writer.add_scalar(
            "Learning Rate", self.scheduler.get_last_lr()[0], epoch_idx + 1
        )

        self.scheduler.step()

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        for epoch_idx in range(self.n_epochs):
            self._train_one_epoch(epoch_idx)
            self.writer.flush()

        self.writer.close()

    def save(self):
        """Save the model's state, optimizer state, and other relevant configurations to disk."""
        # Ensure the path exists.
        path = self.path_to_out / "models"
        os.makedirs(path, exist_ok=True)

        path_to_file = path / f"{type(self.model).__name__}_{self.timestamp}.pt"

        # Prepare dictionary (i.e., state) to save.
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "n_epochs": self.n_epochs,
            "batch_size": self.train_loader.batch_size,
            "device": self.device,
        }
        # If stateful loss, save its attributes as well.
        loss_fn_state = {
            f"loss_{k}": v
            for k, v in vars(self.loss_fn).items()
            if not k.startswith("__") and not callable(v)
        }
        save_dict.update(loss_fn_state)

        # Save trainer state.
        torch.save(save_dict, path_to_file)

        # print(f"Saved trainer state to {path_to_file}")
