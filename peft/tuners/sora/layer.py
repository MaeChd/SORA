
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F # For F.linear

from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.other import transpose


class SoraLinear(nn.Module, LoraLayer):
    adapter_layer_names = ("lora_A", "lora_B", "lora_C" ,"lora_D" ,"lora_Diag", "lora_embedding_A", "lora_embedding_B")

    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_sora_weights: Union[str, bool] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        # Ensure base_layer is an nn.Linear instance for proper in/out features
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Base layer must be an instance of nn.Linear, but got {type(base_layer)}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.base_layer_bias = base_layer.bias is not None # Check if original layer has bias
        # Sora Parameters
        self.r_p = {}
        self.initial_s_p = {}

        self.lora_C = nn.ModuleDict({})
        self.lora_D = nn.ModuleDict({})
        self.lora_Diag = nn.ModuleDict({})

        # Extract r_p from kwargs if it exists, otherwise set a default
        r_p = kwargs.pop("r_p" ,4)  # Safely get r_p and remove it from kwargs
        initial_s_p = kwargs.pop("initial_s_p" ,1.0)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r_p=r_p,
            initial_s_p = initial_s_p,
            init_sora_weights=init_sora_weights,
        )

    def update_layer(
            self,
            adapter_name: str,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            r_p: int,
            initial_s_p: float,
            init_sora_weights: Union[str, bool],
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if r_p <= 0:
            raise ValueError(f"`r_p` should be a positive integer value but the value passed is {r_p}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.r_p[adapter_name] = r_p
        self.initial_s_p[adapter_name] = initial_s_p

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        self.lora_C[adapter_name] = nn.Linear(r_p, self.out_features, bias=False)
        self.lora_D[adapter_name] = nn.Linear(r_p, self.out_features, bias=False)
        self.lora_Diag[adapter_name] = nn.Linear(1, self.out_features, bias=False)

        # LoRA trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        self.scaling[adapter_name] = lora_alpha / r
        self.reset_sora_parameters(adapter_name, init_sora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_sora_parameters(self, adapter_name: str, init_sora_weights: Union[str, bool]):
        if init_sora_weights is False:
            return

        # Initialize LoRA A and B
        if init_sora_weights is True or init_sora_weights.lower() == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=torch.math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_sora_weights.lower() == "symmetric_gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name].weight, mean=0.0, std=0.02)
        elif init_sora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            nn.init.zeros_(self.lora_B[adapter_name].weight) # Often zeros for B
        else:
            raise ValueError(f"Unknown initialization {init_sora_weights=}")
        with torch.no_grad():
            # Initialize lora_C and lora_D for Sora
            nn.init.kaiming_uniform_(self.lora_C[adapter_name].weight, a=torch.math.sqrt(5) ,mode='fan_out')
            self.lora_D[adapter_name].weight.copy_(self.lora_C[adapter_name].weight) # lora_D initialized to lora_C
            nn.init.ones_(self.lora_Diag[adapter_name].weight)

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str):
        # This function moves the adapter weights to the same device as the base layer.
        # It's crucial for correct computation.
        base_layer_device = self.base_layer.weight.device
        self.lora_A[adapter_name].to(base_layer_device)
        self.lora_B[adapter_name].to(base_layer_device)
        self.lora_C[adapter_name].to(base_layer_device)
        self.lora_D[adapter_name].to(base_layer_device)
        self.lora_Diag[adapter_name].to(base_layer_device)



    def set_adapter(self, adapter_names: Union[list[str], str]):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        self._active_adapter = adapter_names

    def compute_s_p(self, adapter_name: str) -> torch.Tensor:
        # Compute s_P as a scaling factor for the P matrix
        # This is a key part of SoRA, preventing P from exploding.
        norm_C = torch.norm(self.lora_C[adapter_name].weight, 'fro')
        norm_D = torch.norm(self.lora_D[adapter_name].weight, 'fro')
        # Add a small epsilon to prevent division by zero
        s_p = torch.clamp(self.initial_s_p[adapter_name] / (norm_C + norm_D + 1e-6), max=1.0)
        return s_p




    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        elif self.merged:
            self.unmerge()

        if len(self.active_adapters) > 1:
            raise ValueError \
                (f"SoRA only supports one active adapter at a time, but got {len(self.active_adapters)}: {self.active_adapters}")

        if len(self.active_adapters) == 0:
            return self.base_layer(x, *args, **kwargs)

        active_adapter = self.active_adapters[0]
        if active_adapter in self.lora_Diag.keys():
            d_weight = self.lora_Diag[active_adapter].weight.squeeze()
            if d_weight.dim() != 1 or d_weight.size(0) != self.out_features:
                raise ValueError \
                    (f"D weight dimension mismatch for adapter {active_adapter}. Expected ({self.out_features},), got {d_weight.shape}")
            result = self.base_layer(x, *args, **kwargs) * d_weight  # (x @ W.T) * d_weight = x @ (W*D).T
        else:
            result = self.base_layer(x, *args, **kwargs)  # x @ W.T

        if active_adapter in self.lora_A.keys():
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x_lora = dropout(x).to(lora_A.weight.dtype)

            # --- LoRA Path: BA ---
            lora_output = lora_B(lora_A(x_lora)) * scaling
            result += lora_output

        if active_adapter in self.lora_C.keys():
            s_p = self.compute_s_p(active_adapter)
            lora_C = self.lora_C[active_adapter]
            lora_D = self.lora_D[active_adapter]

            original_shape = result.shape
            # reshape: [batch_size*seq_len, hidden_dim]
            result_2d = result.view(-1, result.size(-1))
            # result_2d @ lora_C.weight (complexity: N*out*r)
            temp_C = result_2d @ lora_C.weight  # [N, out] @ [out, r] = [N, r]
            # temp_C @ lora_D.weight.T (complexity: N*r*out)
            term1 = temp_C @ lora_D.weight.T  # [N, r] @ [r, out] = [N, out]
            temp_D = result_2d @ lora_D.weight  # [N, out] @ [out, r] = [N, r]
            term2 = temp_D @ lora_C.weight.T  # [N, r] @ [r, out] = [N, out]
            # reshape
            preconditioning = s_p * (term1 - term2)
            result = result + preconditioning.view(original_shape)


        return result


    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight
                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias


    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Computes the change in weights for a given adapter.

        This function calculates the total effective change to the base layer's weight
        by combining the LoRA update and the Sora preconditioning.
        """
        base_layer = self.get_base_layer()
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        weight_C = self.lora_C[adapter].weight
        weight_D = self.lora_D[adapter].weight
        d_weight = self.lora_Diag[adapter].weight.squeeze()  # (out_features,)
        s_p = self.compute_s_p(adapter)

        # Precondition_matrix P' dim: (out_features, out_features)
        Precondition_matrix = s_p * (weight_D @ weight_C.T - weight_C @ weight_D.T)
        # LoRA delta weight dim: (out_features, in_features)
        lora_delta_weight = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        combined_weight = base_layer.weight * d_weight.unsqueeze(1) + lora_delta_weight
        #  (out, out) @ (out, in) -> (out, in)
        preconditioning_effect = Precondition_matrix @ combined_weight
        # delta weight
        delta_layer_weight = preconditioning_effect - base_layer.weight
        return delta_layer_weight


    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sora." + rep