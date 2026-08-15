import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision
from irene.utilities import Hook

from model_architectures.resnet_no_skip import resnet18_no_skip


class Privacy_head(torch.nn.Module):
    def __init__(self, bottleneck_layer, head_structure, old=False):
        super(Privacy_head, self).__init__()
        self.bottleneck = Hook(bottleneck_layer, backward=False)
        self.classifier = head_structure
        self.old = old
        self.bottleneck_layer = (
            bottleneck_layer[1] if isinstance(bottleneck_layer, (list, tuple)) else bottleneck_layer
        )

    def _move_to_classifier_device(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the input is placed on the same device as the classifier parameters."""
        try:
            device = next(self.classifier.parameters()).device
        except StopIteration:
            # If the classifier has no parameters, keep the original device
            return x
        return x.to(device, non_blocking=True)

    def forward(self):
        if type(self.bottleneck.output) == tuple:
            x = self.bottleneck.output[0].clone().detach()
        else:
            x = self.bottleneck.output.clone().detach()
        x = self._move_to_classifier_device(x)
        if self.old:
            if len(x.size()) > 2:
                x = x.reshape(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x

    def forward_attached(self):
        if type(self.bottleneck.output) == tuple:
            x = self.bottleneck.output[0]
        else:
            x = self.bottleneck.output
        x = self._move_to_classifier_device(x)
        if self.old:
            if len(x.size()) > 2:
                x = x.reshape(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x

    def forward_data(self, x):
        x = self._move_to_classifier_device(x)
        if self.old:
            if len(x.size()) > 2:
                x = x.reshape(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x

    def get_encoder_output(self):
        return self.bottleneck.output.clone().detach()

    def get_weights(self):
        weights_dict = self.classifier.state_dict()
        if self.classifier.model_type in ["resnet18", "resnet18_noskip"]:
            if self.classifier.first_layer == "Linear" or self.classifier.first_layer == "doubleLinear":
                # For linear or doubleLinear layers, all weights are already in state_dict
                pass
            else:
                # Include the fc layer weights as well if it's a ResNet with standard layers
                if hasattr(self.classifier, "fc"):
                    for key, value in self.classifier.fc.state_dict().items():
                        weights_dict[f"fc.{key}"] = value
        elif "Linear" in self.classifier.model_type:
            # For linear layers, include the linear layer weights
            for key, value in self.classifier.state_dict().items():
                weights_dict[key] = value

        # Flatten and concatenate all weights into a single tensor
        weights_tensors = []
        for key, value in weights_dict.items():
            if "weight" in key or "bias" in key:  # Only include weight and bias parameters
                weights_tensors.append(value.flatten())

        return torch.cat(weights_tensors) if weights_tensors else torch.tensor([])


class MI(torch.nn.Module):
    def __init__(self, privates = 10, device = "cpu"):
        super(MI, self).__init__()
        self.device = device
        self.privates = privates
        self.scaling = 1 / np.log(privates)

    def forward(self, private_head,yb_ethic):
        out_bias = private_head.forward_attached()
        GT =  1.0* torch.nn.functional.one_hot(yb_ethic, num_classes=self.privates)
        prob_bias = torch.nn.functional.softmax(out_bias, dim=1)
        joint = torch.clamp(torch.mm(torch.transpose(GT, 0, 1), prob_bias), min=1e-15)/ len(yb_ethic)
        marginal_bias = torch.sum(joint, dim=0, keepdim=True)
        marginal_GT = torch.sum(joint, dim=1, keepdim=True)
        marginals = torch.clamp(torch.mm(marginal_GT, marginal_bias), min=1e-15)
        return torch.sum(joint * torch.log(joint /marginals) * self.scaling)

# 10 january 2025

class HeadStructure(torch.nn.Module):
        def __init__(self, input_size, output_size, model_type="resnet18", first_layer="layer2"):
                super(HeadStructure, self).__init__()
                self.output_size = output_size
                self.input_size = input_size      # dimensione “attesa” iniziale (va bene per CelebA)
                self.model_type = model_type
                self.first_layer = first_layer
                self.layers = None                # verrà creato in _build_layers la prima volta
                self._build_layers(self.input_size)

        def _build_layers(self, input_size):
                """
                Costruisce o ricostruisce i layer interni usando la dimensione di input passata.
                Questo permette di adattarsi automaticamente a CelebA (200704) e CIFAR (4096).
                """
                self.input_size = int(input_size)

                # Caso 1: privacy head puramente lineare / doubleLinear (valido per tutti i modelli)
                if self.first_layer == "Linear":
                        linear = torch.nn.Linear(self.input_size, self.output_size)

                        # scaling più piccolo per input molto grandi
                        scaling_factor = 1.0 / np.sqrt(self.input_size)
                        torch.nn.init.normal_(linear.weight, mean=0.0, std=scaling_factor)
                        if linear.bias is not None:
                                torch.nn.init.zeros_(linear.bias)

                        self.layers = nn.Sequential(linear)
                        return

                elif self.first_layer == "doubleLinear":
                        # Dim intermediaria più piccola per input grandi
                        if self.input_size > 1000:
                                intermediate_size = 512
                        else:
                                intermediate_size = self.input_size

                        linear1 = torch.nn.Linear(self.input_size, intermediate_size)
                        linear2 = torch.nn.Linear(intermediate_size, self.output_size)

                        input_scaling_factor = 1.0 / np.sqrt(self.input_size)
                        intermediate_scaling_factor = 1.0 / np.sqrt(intermediate_size)

                        torch.nn.init.normal_(linear1.weight, mean=0.0, std=input_scaling_factor)
                        torch.nn.init.normal_(linear2.weight, mean=0.0, std=intermediate_scaling_factor)

                        if linear1.bias is not None:
                                torch.nn.init.zeros_(linear1.bias)
                        if linear2.bias is not None:
                                torch.nn.init.zeros_(linear2.bias)

                        reluLayer = torch.nn.ReLU()
                        self.layers = torch.nn.Sequential(linear1, reluLayer, linear2)
                        return

                # Caso 2: head “a blocchi ResNet”
                if self.model_type in ["resnet18", "resnet18_noskip"]:
                        if self.model_type == "resnet18_noskip":
                                m = resnet18_no_skip(weights=None)
                        else:
                                m = torchvision.models.resnet18(weights=None)

                        self.avgpool = nn.Sequential(m.avgpool, torch.nn.Identity())
                        self.fc = torch.nn.Linear(in_features=self.input_size,
                                                  out_features=self.output_size,
                                                  bias=True)

                        layers = [m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool]
                        if self.first_layer == "layer4":
                                layers = layers[3:]
                        elif self.first_layer == "layer3":
                                layers = layers[2:]
                        elif self.first_layer == "layer2":
                                layers = layers[1:]

                        self.layers = torch.nn.Sequential(*layers)
                        return

                raise NotImplementedError(
                        f"Model type {self.model_type} with first layer {self.first_layer} not supported"
                )

        def _ensure_linear_matches(self, x: torch.Tensor):
                """
                Se la dimensione delle feature non coincide con self.input_size,
                ricostruisce la/e Linear per quella dimensione.
                Serve per far funzionare CelebA (200704) e CIFAR (4096) con lo stesso codice.
                """
                if self.first_layer not in ["Linear", "doubleLinear"]:
                        return

                if x.dim() > 2:
                        feat_dim = int(np.prod(x.size()[1:]))
                else:
                        feat_dim = int(x.size(1))

                if self.layers is None or feat_dim != self.input_size:
                        # Ricostruisce i layer con la nuova dimensione
                        self._build_layers(feat_dim)
                        # Assicura che i layer ricostruiti si trovino sullo stesso
                        # device delle feature di input (e quindi del resto del modello).
                        # Questo evita mismatch CPU/GPU quando si ricreano le Linear
                        # dopo aver spostato il modello su GPU/CPU.
                        self.to(x.device)

        def load_compatible_state_dict(self, state_dict: dict):
                """Load the checkpoint while ignoring parameters with incompatible shapes.

                This prevents crashes when reusing privacy heads trained on different
                feature sizes (e.g., loading a 4,096-dim head while the current model
                exposes 200,704 features). Only parameters whose keys and shapes match
                the current instance are loaded.
                """

                current_state = self.state_dict()
                compatible_state = {}
                skipped_keys = []

                for key, tensor in state_dict.items():
                        if key in current_state and current_state[key].shape == tensor.shape:
                                compatible_state[key] = tensor
                        else:
                                skipped_keys.append(key)

                if skipped_keys:
                        print(
                                "[WARNING] Skipping incompatible parameters during privacy "
                                f"head load: {', '.join(skipped_keys)}"
                        )

                self.load_state_dict(compatible_state, strict=False)

        def forward(self, x):
                if self.first_layer == "Linear" or self.first_layer == "doubleLinear":
                        # Flatten se necessario
                        if len(x.size()) > 2:
                                x = torch.flatten(x, 1)

                        # Controlla se la dimensione combacia, altrimenti ricostruisce
                        self._ensure_linear_matches(x)

                        # Normalizzazione opzionale (come nel tuo codice originale)
                        if self.input_size > 500 or self.first_layer == "doubleLinear":
                                x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
                                x = x / (x_norm + 1e-6)

                        x = self.layers(x)
                elif self.model_type in ["resnet18", "resnet18_noskip"]:
                        # Caso head basata sui blocchi ResNet
                        x = self.layers(x)
                        x = torch.flatten(x, 1)
                        x = self.fc(x)
                else:
                        raise NotImplementedError(
                                f"Model type {self.model_type} with first layer {self.first_layer} not supported"
                        )
                return x
