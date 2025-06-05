# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(is_recurrent: bool, policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(is_recurrent, policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    is_recurrent: bool, policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(is_recurrent, policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, is_recurrent, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, is_recurrent, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = is_recurrent
        # copy policy parameters
        # self.actor = copy.deepcopy(policy)
        self._nn = copy.deepcopy(policy)

        # Experiment using the --- for name, module in self._nn._modules.items(): --- inside the instantiator
        # output = self._nn(torch.zeros(1, self._nn.net_container[0].in_features))
        # print(output)
    
        # Need to import the template model torch hook
        self.model = MyModel(self._nn)
        # self.model = self.get_all_layers(self._nn)
        for name, module in self._nn._modules.items():
            print(f"Submodule name: {name}, Submodule: {module}")  
        
        # self.visualisation = {}

        # if hasattr(policy, "actor"):
        #     self.actor = copy.deepcopy(policy.actor)
        #     if self.is_recurrent:
        #         self.rnn = copy.deepcopy(policy.memory_a.rnn)
        # elif hasattr(policy, "student"):
        #     self.actor = copy.deepcopy(policy.student)
        #     if self.is_recurrent:
        #         self.rnn = copy.deepcopy(policy.memory_s.rnn)
        # else:
        #     raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    # # visualisation = {}
    # def hook_fn(m, i, o):
    #     self.visualisation[m] = o 

    # def get_all_layers(net):
    #     for name, layer in net._modules.items():
    #         #If it is a sequential, don't register a hook on it
    #         # but recursively register hook on all it's module children
    #         if isinstance(layer, nn.Sequential):
    #             get_all_layers(layer)
    #         else:
    #             # it's a non sequential. Register a hook
    #             layer.register_forward_hook(hook_fn)

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in), # model input (or a tuple for multiple inputs)
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            print(f"printing the self._nn: {self._nn}")
            print(f"printing the self.model: {self.model}")
            # print(f"Zeros from the net container:{self._nn.net_container[0].in_features}")
            # obs = torch.zeros(1, 4) # self._nn.net_container[0].in_features
            obs = torch.zeros(1, self._nn.net_container[0].in_features)
            # print(obs)
            torch.onnx.export(
                self.model, # self -- this should be wrong -- self.model -- fail self._nn.forward(self._nn)
                obs, # model input (or a tuple for multiple inputs)
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"], # "taken_actions"
                dynamic_axes={},
            )

"""
printing the self: _OnnxPolicyExporter(
  (actor): SharedModel(
    (net_container): Sequential(
      (0): Linear(in_features=4, out_features=32, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ELU(alpha=1.0)
    )
    (policy_layer): Linear(in_features=32, out_features=1, bias=True)
    (value_layer): Linear(in_features=32, out_features=1, bias=True)
  )
  (normalizer): Identity()
)
"""

"""
This need to be in the template inside the SKRL
"""
class MyModel(torch.nn.Module):
    def __init__(self, policy):
        super(MyModel, self).__init__()
        # Define the sequential part
        self.sequential = policy.net_container 
        # Define the linear part
        self.linear = policy.policy_layer

    def forward(self, x):
        for name, module in self._modules.items():
            x = module(x)
        # Process the input through the sequential model
        # x = self.sequential(x)
        # Apply the linear model to the output of the sequential model
        # x = self.linear(x)
        return x