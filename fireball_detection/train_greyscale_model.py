import math
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.color import rgb2gray
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator


class GreyScaleValidator(DetectionValidator):
    def preprocess(self, batch):
        """Preprocesses batch of images to greyscale"""

        imgs = batch["img"]  # Retrieve the batch of images

        # Convert images to grayscale using skimage
        if imgs.shape[1] == 3:  # Check if the images are in RGB format
            imgs_np = imgs.cpu().numpy()  # Convert to numpy for skimage
            imgs_gray = [rgb2gray(img.transpose(1, 2, 0)) for img in imgs_np]  # Convert each image to grayscale
            imgs_gray = np.stack(imgs_gray)[:, None, :, :]  # Stack and add channel dimension

            # Convert back to tensor with appropriate dtype
            dtype = torch.float16 if self.args.half else torch.float32
            imgs = torch.tensor(imgs_gray, dtype=dtype)  # Convert back to tensor

        # Scale the images to [0, 1] and move them to the device
        imgs = imgs / 255.0  # Scale to [0, 1]
        imgs = imgs.to(self.device, non_blocking=True)  # Move to the specified device

        # Update the batch with preprocessed images
        batch["img"] = imgs

        # Move other batch components to the device
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

class GreyScaleTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by converting to grayscale, scaling, and converting to float."""
        
        imgs = batch["img"]  # Retrieve the batch of images

        # Convert images to grayscale using skimage
        if imgs.shape[1] == 3:  # Check if the images are in RGB format
            imgs_np = imgs.cpu().numpy()  # Convert to numpy for skimage
            imgs_gray = [rgb2gray(img.transpose(1, 2, 0)) for img in imgs_np]  # Convert each image to grayscale
            imgs_gray = np.stack(imgs_gray)[:, None, :, :]  # Stack and add channel dimension
            imgs = torch.tensor(imgs_gray, dtype=torch.float32)  # Convert back to tensor
        
        # Scale the images to [0, 1] and move them to the device
        imgs = imgs / 255.0  # Scale to [0, 1]
        imgs = imgs.to(self.device, non_blocking=True)  # Move to the specified device
        
        # Perform multi-scale training if applicable
        if self.args.multi_scale:
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        
        batch["img"] = imgs
        return batch
    
    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return GreyScaleValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

if __name__ == "__main__":
    args = dict(
        model="yolov8n.yaml",
        data=Path("yolov8_fireball_dataset", "data.yaml"),
        epochs=100,
        imgsz=640
    )

    trainer = GreyScaleTrainer(
        overrides=args
    )
    trainer.train()