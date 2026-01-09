'''
Mamba block definition with NAS search space support.
'''

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple, Union

from espnet2.asr.state_spaces.bimamba import Mamba as BiMamba
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.legacy.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet2.legacy.nets.pytorch_backend.conformer.convolution import ConvolutionModule

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None


class IdentityModule(nn.Module):
    """Identity module that passes input through unchanged."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Output tensor (same as input)
            masks: Output masks (same as input)
        """
        return x, masks


class LayerNormModule(nn.Module):
    """LayerNorm module wrapper."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Normalized output tensor
            masks: Output masks (same as input)
        """
        return self.norm(x), masks


class RMSNormModule(nn.Module):
    """RMSNorm module wrapper."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        if RMSNorm is None:
            raise ImportError("RMSNorm is not available. Please install mamba_ssm.")
        self.norm = RMSNorm(d_model, eps=eps)
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Normalized output tensor
            masks: Output masks (same as input)
        """
        return self.norm(x), masks


class FFNReLUModule(nn.Module):
    """Feed-Forward Network with ReLU activation."""
    
    def __init__(
        self,
        d_model: int,
        linear_units: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.ffn = PositionwiseFeedForward(
            d_model,
            linear_units,
            dropout_rate,
            activation=nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Output tensor
            masks: Output masks (same as input)
        """
        return self.ffn(x), masks


class FFNSwishModule(nn.Module):
    """Feed-Forward Network with Swish activation."""
    
    def __init__(
        self,
        d_model: int,
        linear_units: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        from espnet2.legacy.nets.pytorch_backend.nets_utils import get_activation
        activation = get_activation("swish")
        self.ffn = PositionwiseFeedForward(
            d_model,
            linear_units,
            dropout_rate,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Output tensor
            masks: Output masks (same as input)
        """
        return self.ffn(x), masks


class ConvolutionModuleWrapper(nn.Module):
    """Convolution Module wrapper."""
    
    def __init__(
        self,
        d_model: int,
        cnn_module_kernel: int = 31,
        activation=None,
    ):
        super().__init__()
        if activation is None:
            from espnet2.legacy.nets.pytorch_backend.nets_utils import get_activation
            activation = get_activation("swish")
        self.conv = ConvolutionModule(d_model, cnn_module_kernel, activation)
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Output tensor
            masks: Output masks (same as input)
        """
        return self.conv(x), masks


class BiMambaModule(nn.Module):
    """BiMamba module wrapper."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.mamba = BiMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v2",
            layer_idx=layer_idx,
        )
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            masks: Optional mask tensor (batch, 1, seq_len)
        
        Returns:
            x: Output tensor
            masks: Output masks (same as input)
        """
        # BiMamba doesn't use masks in its forward pass
        # Apply masks after if needed
        out = self.mamba(x)
        if masks is not None:
            # masks shape: (batch, 1, seq_len), need to transpose for masked_fill
            out = out.masked_fill(~masks.squeeze(1).unsqueeze(-1), 0.0)
        return out, masks


# Module initial mapping
# NOTE:
#   We intentionally exclude 'LN' (LayerNorm) and 'RN' (RMSNorm) from the
#   NAS search space. We use a pre-LN design where normalization is applied
#   uniformly before blocks instead of being searched as modules here.
MODULE_INITIAL_MAP = {
    'I': 'identity',
    # 'LN': 'layernorm',  # disabled from search space (pre-LN is fixed)
    # 'RN': 'rmsnorm',    # disabled from search space
    # 'RFF': 'ffn_relu',
    'F': 'ffn_swish',
    'C': 'conv',
    'B': 'bimamba',
}

# Reverse mapping for validation
INITIAL_TO_TYPE = MODULE_INITIAL_MAP
TYPE_TO_INITIAL = {v: k for k, v in MODULE_INITIAL_MAP.items()}


def parse_module_config(module_config: Union[List[str], List[Dict[str, Any]], str]) -> List[str]:
    """Parse module configuration from initial list or dict list to type list.
    
    Args:
        module_config: List of module initials (e.g., ['I', 'SFF', 'M', 'RFF', 'I'])
                      or List of dicts with 'type' key (e.g., [{'type': 'identity'}, ...])
                      or String representation of list (e.g., "['I', 'SFF', 'M', 'RFF', 'I']")
    
    Returns:
        List of module type strings
    """
    # Handle string input (e.g., from YAML that was saved as string)
    if isinstance(module_config, str):
        try:
            import ast
            module_config = ast.literal_eval(module_config)
            # Ensure it's a list after parsing
            if not isinstance(module_config, list):
                raise ValueError(
                    f"module_config string must evaluate to a list, got {type(module_config)}: {module_config}"
                )
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"module_config string could not be parsed: {module_config}. Error: {e}"
            )
    
    # Ensure module_config is a list at this point
    if not isinstance(module_config, list):
        raise ValueError(
            f"module_config must be a list, got {type(module_config)}: {module_config}"
        )
    
    if len(module_config) == 0:
        return []
    
    # Check if it's a list of initials (strings that are in MODULE_INITIAL_MAP)
    if isinstance(module_config[0], str) and module_config[0] in MODULE_INITIAL_MAP:
        # Convert initials to types
        return [MODULE_INITIAL_MAP[init] for init in module_config]
    elif isinstance(module_config[0], dict):
        # Legacy format: list of dicts
        return [mod_cfg['type'] for mod_cfg in module_config]
    else:
        raise ValueError(
            f"module_config must be a list of initials (e.g., ['I', 'LN', 'M']) "
            f"or list of dicts with 'type' key. Got: {type(module_config)} with first element type: {type(module_config[0]) if len(module_config) > 0 else 'empty'}, "
            f"value: {module_config}"
        )


class MambaCellBlock(nn.Module):
    """Mamba Cell Block with configurable module sequence.
    
    This block consists of 5 sequential modules, each wrapped with residual connection.
    The modules are arranged according to the provided configuration.
    
    Args:
        d_model: Model dimension
        module_config: List of 5 module initials or dicts
            Initials: 'I' (Identity),
                     'RFF' (FFN ReLU), 'SFF' (FFN Swish),
                     'C' (Conv), 'M' (BiMamba)
            Example: ['I', 'SFF', 'M', 'RFF', 'I']
        linear_units: Hidden units for FFN modules
        dropout_rate: Dropout rate
        cnn_module_kernel: Kernel size for convolution module
        activation: Activation function for convolution module
        mamba_d_state: State dimension for BiMamba
        mamba_d_conv: Convolution kernel size for BiMamba
        mamba_expand: Expansion factor for BiMamba
        layer_idx: Layer index for BiMamba
        normalize_before: Whether to normalize before modules
    """
    
    def __init__(
        self,
        d_model: int,
        module_config: Union[List[str], List[Dict[str, Any]]],
        linear_units: int = 1024,
        dropout_rate: float = 0.1,
        cnn_module_kernel: int = 31,
        activation=None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        layer_idx: Optional[int] = None,
        normalize_before: bool = True,
        stochastic_depth_rate: float = 0.0,
        final_layer_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.normalize_before = normalize_before
        self.stochastic_depth_rate = stochastic_depth_rate
        # Optional post-block LayerNorm (does not change module_config)
        self.final_norm = LayerNorm(d_model) if final_layer_norm else torch.nn.Identity()
        
        # Parse module configuration to type list
        mod_types = parse_module_config(module_config)
        
        # if len(mod_types) != 5:
        #     raise ValueError(f"module_config must have exactly 5 modules, got {len(mod_types)}")
        
        # # Validate configuration (can be enabled if we want to enforce NAS rules)
        # # self._validate_config(mod_types)
        
        # Create modules
        self.modules_list = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        self.use_residual = []
        for i, mod_type in enumerate(mod_types):
            
            if mod_type == 'identity':
                module = IdentityModule(d_model)
            elif mod_type == 'layernorm':
                module = LayerNormModule(d_model)
            elif mod_type == 'rmsnorm':
                if RMSNorm is None:
                    raise ImportError("RMSNorm is not available. Please install mamba_ssm.")
                module = RMSNormModule(d_model)
            elif mod_type == 'ffn_relu':
                module = FFNReLUModule(d_model, linear_units, dropout_rate)
            elif mod_type == 'ffn_swish':
                module = FFNSwishModule(d_model, linear_units, dropout_rate)
            elif mod_type == 'conv':
                if activation is None:
                    from espnet2.legacy.nets.pytorch_backend.nets_utils import get_activation
                    activation = get_activation("swish")
                module = ConvolutionModuleWrapper(d_model, cnn_module_kernel, activation)
            elif mod_type == 'bimamba':
                module = BiMambaModule(
                    d_model,
                    mamba_d_state,
                    mamba_d_conv,
                    mamba_expand,
                    layer_idx,
                )
            else:
                raise ValueError(f"Unknown module type: {mod_type}")
            
            self.modules_list.append(module)
            self.use_residual.append(
                mod_type not in {'identity', 'layernorm', 'rmsnorm'}
            )
            
            if (
                self.normalize_before
                and mod_type not in {'identity', 'layernorm', 'rmsnorm'}
            ):
                self.pre_norms.append(LayerNorm(d_model))
            else:
                self.pre_norms.append(torch.nn.Identity())
    
    def _validate_config(self, mod_types: List[str]):
        """Validate that the configuration satisfies the NAS constraints.

        Rules:
            1. BiMamba ('bimamba') must appear 1–2 times.
            2. FFN modules ('ffn_relu' + 'ffn_swish') may appear 0–3 times.
            3. Identity ('identity') may appear 0–3 times.
            4. Consecutive norm layers are allowed (no check).
        """
        # 1) Check BiMamba appears 1–2 times
        bimamba_count = mod_types.count('bimamba')
        if bimamba_count < 1 or bimamba_count > 2:
            raise ValueError(
                f"BiMamba module must appear 1–2 times, got {bimamba_count}"
            )

        # 2) Check FFN appears 0–3 times
        ffn_count = mod_types.count('ffn_relu') + mod_types.count('ffn_swish')
        if ffn_count < 0 or ffn_count > 3:
            # ffn_count < 0 는 발생할 수 없지만, 규칙을 명시적으로 남겨둠
            raise ValueError(
                f"FFN modules must appear 0–3 times, got {ffn_count}"
            )

        # 3) Check identity appears 0–3 times
        identity_count = mod_types.count('identity')
        if identity_count < 0 or identity_count > 3:
            # identity_count < 0 도 이론상 불가능하지만 규칙 일관성을 위해 명시
            raise ValueError(
                f"Identity modules must appear 0–3 times, got {identity_count}"
            )

        # 4) No consecutive norm-layer check anymore (intentionally removed)
    
    def forward(
        self,
        xs_pad: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Mamba cell block.
        
        Args:
            xs_pad: Input tensor (batch, seq_len, d_model) or tuple (x, pos_emb)
            masks: Mask tensor (batch, 1, seq_len)
        
        Returns:
            xs_pad: Output tensor (batch, seq_len, d_model)
            masks: Output masks (batch, 1, seq_len)
        """
        # Handle tuple input (for positional encoding)
        if isinstance(xs_pad, tuple):
            x, pos_emb = xs_pad
        else:
            x = xs_pad
            pos_emb = None
        
        # Stochastic depth: skip layer with probability stochastic_depth_rate
        skip_layer = False
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        
        if skip_layer:
            if pos_emb is not None:
                return (x, pos_emb), masks
            return x, masks
        
        # Process through each module with residual connection
        # Each module has its own residual connection: x = x + module(x)
        for module, pre_norm, use_residual in zip(
            self.modules_list, self.pre_norms, self.use_residual
        ):
            residual = x
            if self.normalize_before:
                x = self._apply_pre_norm(x, pre_norm)
            # Apply module
            x, masks = module(x, masks)
            # Residual connection: add residual before module to output
            if use_residual:
                x = residual + stoch_layer_coeff * x
            else:
                x = x

        # Optional final LayerNorm at the end of the block
        x = self._apply_pre_norm(x, self.final_norm)
        
        # Handle positional encoding
        if pos_emb is not None:
            return (x, pos_emb), masks
        return x, masks

    def _apply_pre_norm(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], norm_module: nn.Module
    ):
        if isinstance(norm_module, torch.nn.Identity):
            return x
        if isinstance(x, tuple):
            h, pos_emb = x
            h = norm_module(h)
            return (h, pos_emb)
        return norm_module(x)


def create_mamba_cell_block(
    d_model: int,
    module_config: Union[List[str], List[Dict[str, Any]]],
    linear_units: int = 2048,
    dropout_rate: float = 0.1,
    cnn_module_kernel: int = 31,
    activation=None,
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    layer_idx: Optional[int] = None,
    normalize_before: bool = True,
    stochastic_depth_rate: float = 0.0,
    final_layer_norm: bool = False,
) -> MambaCellBlock:
    """Factory function to create a MambaCellBlock.
    
    Args:
        d_model: Model dimension
        module_config: List of 5 module initials (e.g., ['I', 'SFF', 'M', 'RFF', 'I'])
                      or List of dicts with 'type' key (legacy format)
        linear_units: Hidden units for FFN modules
        dropout_rate: Dropout rate
        cnn_module_kernel: Kernel size for convolution module
        activation: Activation function for convolution module
        mamba_d_state: State dimension for BiMamba
        mamba_d_conv: Convolution kernel size for BiMamba
        mamba_expand: Expansion factor for BiMamba
        layer_idx: Layer index for BiMamba
        normalize_before: Whether to normalize before modules
        stochastic_depth_rate: Stochastic depth rate
        final_layer_norm: Whether to apply an additional LayerNorm at block output
    
    Returns:
        MambaCellBlock instance
    """
    return MambaCellBlock(
        d_model=d_model,
        module_config=module_config,
        linear_units=linear_units,
        dropout_rate=dropout_rate,
        cnn_module_kernel=cnn_module_kernel,
        activation=activation,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        layer_idx=layer_idx,
        normalize_before=normalize_before,
        stochastic_depth_rate=stochastic_depth_rate,
        final_layer_norm=final_layer_norm,
    )
