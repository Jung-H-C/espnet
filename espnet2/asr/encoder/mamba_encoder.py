# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Mamba encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.legacy.nets.pytorch_backend.nets_utils import (
    get_activation,
    make_pad_mask,
    trim_by_ctc_posterior,
)
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
    ConvolutionalPositionalEmbedding,
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.legacy.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

from espnet2.asr.state_spaces.mamba_blocks import (
    MambaCellBlock,
    create_mamba_cell_block,
)


class MambaEncoder(AbsEncoder):
    """Mamba encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of output.
        num_blocks (int): The number of encoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        activation_type (str): Encoder activation function type.
        padding_idx (int): Padding idx for input_layer=embed.
        interctc_layer_idx (List[int]): List of layer indices for intermediate CTC.
        interctc_use_conditioning (bool): Whether to use CTC conditioning.
        ctc_trim (bool): Whether to trim CTC posterior.
        stochastic_depth_rate (Union[float, List[float]]): Stochastic depth rate.
        layer_drop_rate (float): Layer drop rate.
        max_pos_emb_len (int): Maximum positional encoding length.
        mamba_d_state (int): State dimension for Mamba.
        mamba_d_conv (int): Convolution kernel size for Mamba.
        mamba_expand (int): Expansion factor for Mamba.
        linear_units (int): Hidden units for FFN modules.
        cnn_module_kernel (int): Kernel size for convolution module.
        final_layer_norm (bool): Whether to apply an extra LayerNorm at the end of each
            MambaCellBlock, in addition to the existing pre-LN design.
        module_configs (Optional[List[List[str]]]): Module configurations for each block.
            Each block configuration is a list of 5 module initials.
            Module initials: 'I' (Identity), 'LN' (LayerNorm), 'RN' (RMSNorm),
                           'RFF' (FFN ReLU), 'SFF' (FFN Swish), 'C' (Conv), 'M' (BiMamba).
            Example: ['I', 'SFF', 'M', 'RFF', 'I']
            If None, uses default configuration.

    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        activation_type: str = "swish",
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        ctc_trim: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        # Mamba-specific parameters
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        linear_units: int = 2048,
        cnn_module_kernel: int = 31,
        final_layer_norm: bool = False,
        # Module configuration for each block
        # Each element is a list of 5 module initials: ['I', 'SFF', 'M', 'RFF', 'I']
        # Module initials: 'I' (Identity), 'LN' (LayerNorm), 'RN' (RMSNorm),
        #                  'RFF' (FFN ReLU), 'SFF' (FFN Swish), 'C' (Conv), 'M' (BiMamba)
        # If None, uses default configuration
        module_configs: Optional[List[List[str]]] = None,
    ):
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
        elif rel_pos_type == "latest":
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "conv":
            pos_enc_class = ConvolutionalPositionalEmbedding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # Get activation function
        activation = get_activation(activation_type)

        # Handle stochastic depth rate
        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        # Default module configuration if not provided
        # Example: ['SFF', 'M', 'C', 'RFF', 'I']
        if module_configs is None:
            # Default configuration: simple structure
            # [FFN(Swish), BiMamba, Conv, FFN(ReLU), Identity]
            default_config = ['B', 'C']
            module_configs = [default_config] * num_blocks
        else:
            if len(module_configs) != num_blocks:
                raise ValueError(
                    f"Length of module_configs ({len(module_configs)}) "
                    f"should be equal to num_blocks ({num_blocks})"
                )
            # Validate each module_config is a list of strings
            for idx, config in enumerate(module_configs):
                if not isinstance(config, list):
                    raise TypeError(
                        f"module_configs[{idx}] must be a list, got {type(config)}"
                    )
                if not all(isinstance(item, str) for item in config):
                    raise TypeError(
                        f"module_configs[{idx}] must be a list of strings (module initials), "
                        f"got {[type(item) for item in config]}"
                    )

        # Create MambaCellBlocks
        self.encoders = repeat(
            num_blocks,
            lambda lnum: create_mamba_cell_block(
                d_model=output_size,
                module_config=module_configs[lnum],
                linear_units=linear_units,
                dropout_rate=dropout_rate,
                cnn_module_kernel=cnn_module_kernel,
                activation=activation,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                layer_idx=lnum,
                normalize_before=normalize_before,
                stochastic_depth_rate=stochastic_depth_rate[lnum],
                final_layer_norm=final_layer_norm,
            ),
            layer_drop_rate,
        )
        
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        if self.interctc_use_conditioning:
            self.conditioning_layer = torch.nn.Linear(output_size, output_size)
        self.ctc_trim = ctc_trim

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): ctc module for intermediate CTC loss
            return_all_hs (bool): whether to return all hidden states

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
            # Handle RelPositionalEncoding that returns tuple (x, pos_emb)
            if isinstance(xs_pad, tuple):
                xs_pad, _ = xs_pad

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                # MambaCellBlock forward: (xs_pad, masks) -> (xs_pad, masks)
                xs_pad, masks = encoder_layer(xs_pad, masks)
                
                if return_all_hs:
                    if isinstance(xs_pad, tuple):
                        intermediate_outs.append(xs_pad[0])
                    else:
                        intermediate_outs.append(xs_pad)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                # MambaCellBlock forward: (xs_pad, masks) -> (xs_pad, masks)
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

                    if self.ctc_trim and ctc is not None:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x, masks, pos_emb = trim_by_ctc_posterior(
                                x, ctc_out, masks, pos_emb
                            )
                            xs_pad = (x, pos_emb)
                        else:
                            x, masks, _ = trim_by_ctc_posterior(x, ctc_out, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

