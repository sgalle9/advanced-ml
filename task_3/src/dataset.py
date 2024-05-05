import gzip
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask


def to_image(frame, size):
    """
    Convert a single frame to a Torch image tensor with specified resizing.
    :param frame: Numpy array of the frame data.
    :param size: Tuple of (width, height) to resize the image.
    :return: Torch tensor representing the processed image.
    """
    # `to_image` expects the shape [Height, Width, Channels].
    frame = np.transpose(frame[:, :, np.newaxis], (1, 0, 2))
    # The two transformations below are equivalent to the `to_tensor` function.
    frame = v2.functional.to_image(frame)
    frame = v2.functional.to_dtype(frame, dtype=torch.float32, scale=True)
    frame = v2.functional.resize_image(frame, size=size)
    return frame


def to_mask(label, size):
    """
    Convert a label mask to a Torch Mask tensor with specified resizing.
    :param label: Numpy array of the label data.
    :param size: Tuple of (width, height) to resize the mask.
    :return: Mask tensor representing the processed label mask.
    """
    # `Mask` expects the shape [Height, Width].
    label = np.transpose(label, (1, 0))
    label = Mask(label)
    label = v2.functional.resize_mask(label, size=size)
    return label


class EchoDataset(Dataset):
    """Dataset class for loading and transforming echocardiogram video data with associated labels and metadata."""

    def __init__(self, path_to_data, size, transforms=None) -> None:
        self.size = size
        self.transforms = transforms

        # Given that the dataset is small (~200 frames), we can load it all into memory.
        self.inputs, self.targets, self.metadata = self._extract_data(path_to_data)

    def _extract_data(self, path_to_data):
        """
        Extracts and preprocesses the input frames, target masks, and metadata from the dataset.
        :param path_to_data: Path to the compressed dataset file containing the raw data.
        :return: Tuple of stacked input tensors, target masks, and metadata tensors.
        """
        inputs = []
        targets = []
        metadata = []
        with gzip.open(path_to_data, "rb") as f:
            observations = pickle.load(f)

            # List of dictionnaries. Each dictionnary represents a video.
            for observation in observations:
                # Not all frames in the video are labeled.
                labeled_frames = observation["frames"]

                # Only the labeled frames are use during training (supervised learning).
                frames = [
                    to_image(observation["video"][:, :, frame_id], self.size)
                    for frame_id in labeled_frames
                ]
                labels = [
                    to_mask(observation["label"][:, :, frame_id], self.size)
                    for frame_id in labeled_frames
                ]

                inputs.extend(frames)
                targets.extend(labels)
                # Some observations (videos) were labeled by experts, and have higher resolution.
                # Observations are then differentiated by this metadata ('expert' or 'amateur').
                labeler_id = 0 if observation["dataset"] == "amateur" else 1
                metadata.extend([labeler_id] * len(labeled_frames))

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0).long()
        metadata = torch.Tensor(metadata).long()

        return inputs, targets, metadata

    def __getitem__(self, idx):
        # Get the data.
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        # Metadata will be used by the loss function (to give more importance to the data labeled by experts).
        metadata = self.metadata[idx]

        if self.transforms is not None:
            inputs, targets = self.transforms(inputs, targets)

        return inputs, targets, metadata

    def __len__(self):
        return len(self.inputs)
