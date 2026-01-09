#!/usr/bin/env python3
"""
Update module_configs in YAML configuration file.
The same module_config will be applied to all blocks.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import List


def parse_module_config(config_str: str) -> List[str]:
    """Parse module_config from string format.
    
    Expected format: "['I','LN','SFF','M','RN','I']"
    
    Args:
        config_str: String representation of module_config (single list)
        
    Returns:
        Module config list
    """
    try:
        # Try to evaluate as Python literal
        config = eval(config_str)
        if not isinstance(config, list):
            raise ValueError("module_config must be a list")
        # if len(config) != 5:
        #     raise ValueError(f"module_config must have exactly 5 modules, got {len(config)}")
        # Check if all elements are strings
        if not all(isinstance(item, str) for item in config):
            raise ValueError("All elements in module_config must be strings")
        return config
    except Exception as e:
        raise ValueError(f"Failed to parse module_config: {e}")


def update_yaml_config(yaml_path: str, module_config: List[str], num_blocks: int = None):
    """Update module_configs in YAML file.
    
    The same module_config will be applied to all blocks.
    
    Args:
        yaml_path: Path to YAML configuration file
        module_config: Single module configuration (will be repeated for all blocks)
        num_blocks: Number of blocks (if None, will be read from YAML)
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    # Read YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if encoder_conf exists
    if 'encoder_conf' not in config:
        raise ValueError("encoder_conf not found in YAML file")
    
    encoder_conf = config['encoder_conf']
    
    # Get num_blocks if not provided
    if num_blocks is None:
        if 'num_blocks' in encoder_conf:
            num_blocks = encoder_conf['num_blocks']
        else:
            raise ValueError("num_blocks must be specified either in YAML or via --num_blocks option")
    
    # Repeat the same config for all blocks (deep copy to avoid aliasing)
    module_configs = [list(module_config) for _ in range(num_blocks)]
    
    # Update num_blocks and module_configs together
    encoder_conf['num_blocks'] = num_blocks
    encoder_conf['module_configs'] = module_configs
    
    # Write back to YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated {yaml_path} with module_config:")
    print(f"  Configuration: {module_config}")
    print(f"  Applied to {num_blocks} blocks")


def main():
    parser = argparse.ArgumentParser(
        description="Update module_configs in YAML configuration file. "
                    "The same module_config will be applied to all blocks."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--module_config",
        type=str,
        required=True,
        help="Module configuration in string format (will be applied to all blocks). "
             "Example: \"['I','LN','SFF','M','RN','I']\""
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="Number of blocks (if not provided, will use num_blocks from YAML)"
    )
    
    args = parser.parse_args()
    
    try:
        module_config = parse_module_config(args.module_config)
        update_yaml_config(args.yaml_path, module_config, args.num_blocks)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

