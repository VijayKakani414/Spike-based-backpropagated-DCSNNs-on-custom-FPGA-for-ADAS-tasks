import torch
import torch.nn as nn
import torch.nn.parallel


# 6-class classfication using KITTI dataset



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# print("device:", DEVICE)


# Surrogate gradient descent

class SurrGradSpike(torch.autograd.Function):
    # usage: spike_fn  = SurrGradSpike.apply
    scale = 3  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# Spiking Model Neuron

class IF_Neurons(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 1.0
        self.reset()

    def forward(self, x):
        v1 = x + self.v  # current is added directly to potential for simplicity
        y = SurrGradSpike.apply(v1 - self.threshold)  # Heaviside function (with surrogate gradient) is applied
        self.v = v1 * (1. - y)  # reset potential if neuron emits spike
        return y

    def reset(self):
        self.v = torch.tensor(0.)  # neuron potential


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


# Deep Spiking VGG-9 Model for six-class dataset


class VGGModel(nn.Module):
    def __init__(self, stride=1, num_classes=6, n_steps=30):
        super(VGGModel, self).__init__()
        self.n_steps = n_steps
        self.num_classes = num_classes

        self.stride = stride
        layers = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            IF_Neurons(),
            # nn.Dropout(0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            IF_Neurons(),
            # nn.Dropout(0.25),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            IF_Neurons(),
            # nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            IF_Neurons(),
            # nn.Dropout(0.25),
            nn.AvgPool2d(kernel_size=2),
            Flatten(),

            nn.Linear(4096, 512, bias=False),
            IF_Neurons(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes, bias=False),

            # nn.Linear(1024, 10, bias=False)
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        n_batch = x.shape[0]
        out = torch.zeros(n_batch, self.num_classes).to(DEVICE)

        for m in self.modules():
            if isinstance(m, IF_Neurons):
                m.reset()

        for step in range(self.n_steps):
            out += self.network(x)

        return out


model = VGGModel(n_steps=10).to(DEVICE).cuda()
print(model)
