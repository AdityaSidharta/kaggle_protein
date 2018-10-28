from torch import nn as nn
from utils.common import to_list, pairwise
import torch


def linear_block(
    input_tensor, output_tensor, dropout=0.0, activation=False, batchnorm=True
):
    layers = nn.ModuleList()
    layers.append(nn.Linear(input_tensor, output_tensor))
    if activation:
        layers.append(nn.ReLU())
    if batchnorm:
        layers.append(nn.BatchNorm1d(output_tensor))
    if dropout:
        layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Header(nn.Module):
    def __init__(self, n_hidden_list, dropout=0.0):
        super().__init__()
        self.input_tensor = 2048
        self.output_tensor = 28

        self.n_hidden_list = to_list(n_hidden_list)
        self.n_hidden = len(self.n_hidden_list)
        self.dropout = dropout

        self.layers = []
        for before_hidden, after_hidden in pairwise(
            [self.input_tensor] + self.n_hidden_list
        ):
            self.layers.append(
                linear_block(
                    before_hidden, after_hidden, activation=True, dropout=dropout
                )
            )
        self.layers.append(nn.Linear(self.n_hidden_list[-1], self.output_tensor))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    def __init__(self, preload_model, header_model, device):
        super().__init__()
        self.preload_model = preload_model
        self.preload_backbone, self.preload_header = self.dissect_model(
            self.preload_model
        )
        self.device = device

        self.rgb_backbone = self.preload_backbone
        self.backbone = self.create_four_channel(self.rgb_backbone)
        self.unfreeze_backbone()
        self.header = header_model

        self.backbone_children = list(self.backbone.children())
        self.header_children = list(self.header.children())
        self.backbone_layers = len(self.backbone_children)
        self.header_layers = len(self.header_children)

    def create_four_channel(self, backbone):
        rgb_layer = list(backbone.children())[0]
        wo_rgb_layer = list(backbone.children())[1:]
        rgb_weight = rgb_layer[0].weight
        y_weight = torch.zeros((64, 1, 7, 7)).to(self.device)
        rgby_layer = list(
            nn.Sequential(
                nn.Conv2d(
                    4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
            )
        )
        rgby_weight = torch.cat((rgb_weight, y_weight), dim=1)
        rgby_layer[0].weight = torch.nn.Parameter(rgby_weight)
        layers = rgby_layer + wo_rgb_layer
        new_backbone = nn.Sequential(*layers)
        return new_backbone

    @staticmethod
    def dissect_model(model):
        model_list = [nn.Sequential(*to_list(x)) for x in list(model.children())]
        if len(model_list) == 1:
            raise ValueError("preload model not suitable")
        elif len(model_list) == 2:
            return model_list[0], model_list[1]
        else:
            return nn.Sequential(*model_list[:-1]), model_list[-1]

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model):
        for param in model.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        self.freeze(self.backbone)

    def unfreeze_backbone(self):
        self.unfreeze(self.backbone)

    def forward(self, x):
        bs = x.shape[0]
        x = self.backbone(x).view(bs, -1)
        x = self.header(x)
        return x
