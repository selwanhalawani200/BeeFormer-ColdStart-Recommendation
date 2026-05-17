import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import sentence_transformers
import torch

from keras.layers import TorchModuleWrapper


# Basic ELSA model as a Keras layer
class LayerELSA(keras.layers.Layer):
    def __init__(self, n_dims, n_items, device):
        super(LayerELSA, self).__init__()

        # Save the device used for training
        self.device = device

        # Initialize the item embedding matrix
        self.A = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty([n_dims, n_items])
            )
        )

    def parameters(self, recurse=True):
        # Return the trainable parameter of the layer
        return [self.A]

    def track_module_parameters(self):
        # Track Torch parameters so Keras can update them
        for param in self.parameters():
            variable = keras.Variable(
                initializer=param,
                trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)

        self.built = True

    def build(self):
        # Move the layer to the selected device
        self.to(self.device)

        # Run one sample input to initialize the layer
        sample_input = torch.ones([self.A.shape[0]]).to(self.device)
        _ = self.call(sample_input)

        # Register parameters inside Keras
        self.track_module_parameters()

    def call(self, x):
        # Normalize the item matrix before using it
        A = torch.nn.functional.normalize(self.A, dim=-1)

        # Project the input into item space
        xA = torch.matmul(x, A)

        # Project it back to the original embedding space
        xAAT = torch.matmul(xA, A.T)

        # Return only positive reconstruction differences
        return keras.activations.relu(xAAT - x)


# Keras layer wrapper for the SentenceTransformer model
class LayerSBERT(keras.layers.Layer):
    def __init__(self, model, device, tokenized_sentences):
        super(LayerSBERT, self).__init__()

        # Save device and wrap the SBERT model for Keras
        self.device = device
        self.sbert = TorchModuleWrapper(model.to(device))

        # Keep the tokenizer from the original SentenceTransformer model
        self.tokenize_ = self.sb().tokenize

        # Save tokenized sentences used to build the layer
        self.tokenized_sentences = tokenized_sentences

        # Build the layer immediately
        self.build()

    def sb(self):
        # Get the original SentenceTransformer module from the wrapper
        for module in self.sbert.modules():
            if isinstance(module, sentence_transformers.SentenceTransformer):
                return module

    def parameters(self, recurse=True):
        # Return SBERT trainable parameters
        return self.sbert.parameters()

    def track_module_parameters(self):
        # Track Torch parameters so Keras can train them
        for param in self.parameters():
            variable = keras.Variable(
                initializer=param,
                trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)

        self.built = True

    def tokenize(self, inp):
        # Tokenize new text inputs and move them to the selected device
        return {
            k: v.to(self.device)
            for k, v in self.tokenize_(
                inp,
                padding=True,
                truncation=True,
                max_length=256,
            ).items()
        }

    def build(self):
        # Move the layer to the selected device
        self.to(self.device)

        # Use a small sample to initialize the SBERT layer
        sample_input = {
            k: v[:2].to(self.device)
            for k, v in self.tokenized_sentences.items()
        }

        _ = self.call(sample_input)

        # Register SBERT parameters inside Keras
        self.track_module_parameters()

    def call(self, x):
        # Generate sentence embeddings and normalize them
        return torch.nn.functional.normalize(
            self.sbert.forward(x)["sentence_embedding"],
            dim=-1
        )