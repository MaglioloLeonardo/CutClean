import math
from irene.core import HeadStructure, Privacy_head
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import torchvision.models as models

from model_architectures.resnet_no_skip import resnet18_no_skip


# ------------ Functions for training vanilla models -------------
def init_model(args, model_name, reload_config=None):
    img_size = (3, 224, 224)
    dataset = args.dataset.split("-")[0]
    if dataset == "celeba":
        img_size = (3,224,224)

    num_classes = getattr(args, "target_num_classes", None)
    if num_classes is None:
        cifar_like_defaults = {
            "cifar10c": 10,
            "corruptedcifarunbiased": 10,
        }
        num_classes = cifar_like_defaults.get(dataset, 2)
        args.target_num_classes = num_classes
    if model_name == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).to(args.device)
        model.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=512), nn.Linear(in_features=512, out_features=num_classes)).to(args.device)
    elif model_name == "vit":
        config = ViTConfig(
            image_size=img_size[1],
            patch_size=16,
            num_hidden_layers=6,  # Small for fast training
            num_attention_heads=6,
            hidden_size=384,
            intermediate_size=1536,
            num_labels=num_classes )
        model = ViTForImageClassification(config).to(args.device)
    elif model_name == "vit_b":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(args.device)
        model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=512), nn.Linear(in_features=512, out_features=num_classes)).to(args.device)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(args.device)
        model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes)).to(args.device)
    elif model_name == "resnet18_noskip":
        model = resnet18_no_skip(weights=None).to(args.device)
        model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes)).to(args.device)
    if reload_config is not None:
        print(f"Reloading model from {reload_config}")
        model.load_state_dict(torch.load(reload_config, map_location=args.device, weights_only=True))
    return model

def init_and_plug_phs(model, args, model_type="vgg11", reload_config=None):
    if model_type == "vgg11":
        bottleneck_layers = [
            ("features.2",  model.features[2]),
            ("features.5",  model.features[5]),
            ("features.10", model.features[10]),
            ("features.15", model.features[15]),
            ("features.20", model.features[20]),
        ]
    elif model_type == "vit":
        bottleneck_layers = [
            ("vit.encoder.layer.0", model.vit.encoder.layer[0]),
            ("vit.encoder.layer.1", model.vit.encoder.layer[1]),
            ("vit.encoder.layer.2", model.vit.encoder.layer[2]),
            ("vit.encoder.layer.3", model.vit.encoder.layer[3]),
            ("vit.encoder.layer.4", model.vit.encoder.layer[4]),
            ("vit.encoder.layer.5", model.vit.encoder.layer[5]), 
        ]
    elif model_type == "vit_b":
        bottleneck_layers = [
            ("encoder.layers.encoder_layer_0", model.encoder.layers.encoder_layer_0),
            ("encoder.layers.encoder_layer_1", model.encoder.layers.encoder_layer_1),
            ("encoder.layers.encoder_layer_2", model.encoder.layers.encoder_layer_2),
            ("encoder.layers.encoder_layer_3", model.encoder.layers.encoder_layer_3),
            ("encoder.layers.encoder_layer_4", model.encoder.layers.encoder_layer_4),
            ("encoder.layers.encoder_layer_5", model.encoder.layers.encoder_layer_5),
            ("encoder.layers.encoder_layer_6", model.encoder.layers.encoder_layer_6),
            ("encoder.layers.encoder_layer_7", model.encoder.layers.encoder_layer_7),
            ("encoder.layers.encoder_layer_8", model.encoder.layers.encoder_layer_8),
            ("encoder.layers.encoder_layer_9", model.encoder.layers.encoder_layer_9),
            ("encoder.layers.encoder_layer_10",model.encoder.layers.encoder_layer_10),
            ("encoder.layers.encoder_layer_11",model.encoder.layers.encoder_layer_11),
        ]
    elif model_type in ["resnet18", "resnet18_noskip"]:
        bottleneck_layers = [
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
            ("layer4", model.layer4),
        ]
    num_private_classes = getattr(args, "private_num_classes", 2)
    phs = []
    ph_structure = [
        HeadStructure(
            input_size=torch.tensor(get_layer_output_size(model, layer[0], args)).prod().item(),
            output_size=num_private_classes,
            model_type=model_type,
            first_layer="Linear",
        ).to(args.device)
        for layer in bottleneck_layers
    ]
    for i, layer in enumerate(bottleneck_layers):
        ph = Privacy_head(layer,ph_structure[i], old=True)
        if reload_config is not None:
            ph.classifier.load_state_dict(torch.load(reload_config, map_location=args.device, weights_only=True))
        phs.append(ph)
    
    return phs, bottleneck_layers 
## Train and minimize the MI of the PHs
def compute_MI(private_labels, ph, nb_classes=2, ph_output=None, args=None):
    out_bias = ph.forward_attached() if ph_output is None else ph_output
    if type(out_bias) == tuple:
        out_bias = out_bias[0]
    # The evaluation loops run under autocast, where torch.mm would compute
    # `joint` in float16. The 1e-15 clamps below are not representable in half
    # precision, so underflowing cells stay at exactly 0 and log(0) turns the
    # result into NaN. Forcing float32 leaves the value bit-identical outside
    # autocast and keeps it well defined inside it.
    with torch.autocast("cuda", enabled=False):
        out_bias = out_bias.float()
        gt = torch.zeros(len(private_labels), nb_classes, device=args.device)
        gt.scatter_(1, private_labels.unsqueeze(1), 1.0)
        prob_bias = torch.clamp(torch.nn.functional.softmax(out_bias, dim=1),  min=1e-15)
        joint = torch.clamp(torch.mm(gt.T, prob_bias) / len(private_labels), min=1e-15)
        marginal_bias = torch.clamp(joint.sum(dim=0, keepdim=True), min=1e-15)
        marginal_GT = torch.clamp(joint.sum(dim=1, keepdim=True), min=1e-15)
        marginals = torch.clamp(marginal_GT * marginal_bias, min=1e-15)
        return torch.sum(joint * torch.log(joint /marginals) / math.log(nb_classes))


def get_layer_output_size(model, layer_name, args, input_size=(3, 224, 224)):
    print(f"Getting output size of layer {layer_name} in model {args.model}")
    dummy_input = torch.randn(1, *input_size).to(args.device)
    output_size = None

    def hook(module, input, output):
        nonlocal output_size

        if type(output) is tuple:
            output = output[0]  # For models that return a tuple (e.g., ViT)

        output_size = output.shape

    hooks = []
    for name, layer in model.named_modules():
        if name == layer_name:
            hooks.append(layer.register_forward_hook(hook))
            break

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()
    print(f"-- Output size = {output_size}")

    return output_size
