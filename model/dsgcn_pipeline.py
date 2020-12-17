import csv
import json
import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from os.path import abspath, exists
from typing import Dict, List, Optional, Tuple, Union
import torch

class _ScikitCompat(ABC):
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()
class ArgumentHandler(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

class DefaultArgumentHandler(ArgumentHandler):
    def __call__(self, *args, **kwargs):
        if "X" in kwargs:
            return kwargs["X"]
        elif "data" in kwargs:
            return kwargs["data"]
        elif len(args) == 1:
            if isinstance(args[0], list):
                return args[0]
            else:
                return [args[0]]
        elif len(args) > 1:
            return list(args)
        raise ValueError("Unable to infer the format of the provided data (X=, data=, ...)")


class Pipeline(_ScikitCompat):
    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.framework = "pt"
        self.device = device
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()

        if self.framework == "pt" and self.device.type == "cuda":
            self.model = self.model.to(self.device)

    def save_pretrained(self, save_directory):
        if not os.path.isdir(save_directory):
            logger.error("Provided path ({}) should be a directory".format(save_directory))
            return

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def transform(self, X):
        return self(X=X)

    def predict(self, X):
        return self(X=X)

    @contextmanager
    def device_placement(self):
        if self.framework == "tf":
            with tf.device("/CPU:0" if self.device == -1 else "/device:GPU:{}".format(self.device)):
                yield
        else:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            yield

    def ensure_tensor_on_device(self, **inputs):
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def inputs_for_model(self, features: Union[dict, List[dict]]) -> Dict:
        args = ["input_ids", "attention_mask"]

        if isinstance(features, dict):
            return {k: features[k] for k in args}
        else:
            return {k: [feature[k] for feature in features] for k in args}

    def _parse_and_tokenize(self, *texts, **kwargs):
        inputs = self._args_parser(*texts, **kwargs)
        inputs = self.tokenizer.batch_encode_plus(
            inputs, add_special_tokens=True, return_tensors=self.framework, max_length=self.tokenizer.max_len
        )
        inputs = self.inputs_for_model(inputs)

        return inputs

    def __call__(self, *texts, **kwargs):
        inputs = self._parse_and_tokenize(*texts, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        with self.device_placement():
            with torch.no_grad():
                inputs = self.ensure_tensor_on_device(**inputs)
                predictions = self.model(**inputs)[2][1:]
                predictions = torch.stack(predictions).mean(0).squeeze()[1:-1].cpu()
        return predictions
