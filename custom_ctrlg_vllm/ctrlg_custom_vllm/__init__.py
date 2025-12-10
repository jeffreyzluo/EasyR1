"""
ctrlg_custom: Custom CTRL-G logits processors for vLLM
"""

import sys
import os
import re
from typing import TYPE_CHECKING
import importlib

# Dynamic loading of constraint dictionaries from keyphrases/v{n}.py files
def _load_constraint_dictionaries():
    """Dynamically load all CONSTRAINTS_DICT from keyphrases/v{n}.py files"""
    constraints = {}
    keyphrases_dir = os.path.join(os.path.dirname(__file__), 'keyphrases')
    
    # Find all v{n}.py files
    if os.path.exists(keyphrases_dir):
        for filename in os.listdir(keyphrases_dir):
            match = re.match(r'^v(\d+)\.py$', filename)
            if match:
                version = match.group(1)
                module_name = f'.keyphrases.v{version}'
                try:
                    module = importlib.import_module(module_name, package=__name__)
                    if hasattr(module, 'CONSTRAINTS_DICT'):
                        constraints[f'CONSTRAINTS_DICT_V{version}'] = module.CONSTRAINTS_DICT
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")
                    continue
    return constraints

# Load constraint dictionaries
_constraint_dicts = _load_constraint_dictionaries()

# For debugging, print what was loaded
if os.getenv('DEBUG_CTRLG_CUSTOM'):
    print(f"Loaded constraint dictionaries: {list(_constraint_dicts.keys())}")

# Type hints for IDEs when not actually importing
if TYPE_CHECKING:
    from . import ctrlg_variants, ctrlg_logprocs

__version__ = "0.1.0"

# Module mapping for lazy imports
_MODULE_MAP = {
    # Classes from ctrlg_variants
    'ctrlg_variants': [
        'Qwen25VLBaseCtrlgProcessorV0',
    ],
    # Classes from ctrlg_logprocs
    'ctrlg_logprocs': [
        'CtrlgWrappedLogitsProcessor',
    ]
}

# Flatten the mapping for quick lookup
_CLASS_TO_MODULE = {}
for module_name, class_names in _MODULE_MAP.items():
    for class_name in class_names:
        _CLASS_TO_MODULE[class_name] = module_name

# Dynamic lazy import mechanism
def __getattr__(name: str):
    # Check if it's a constraint dictionary
    if name in _constraint_dicts:
        return _constraint_dicts[name]
    
    # Check if it's a known class
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(f'.{module_name}', package=__name__)
        return getattr(module, name)
    
    # Fallback: try to find the attribute in any of our submodules
    for module_name in _MODULE_MAP.keys():
        try:
            module = importlib.import_module(f'.{module_name}', package=__name__)
            if hasattr(module, name):
                # Cache it for future use
                _CLASS_TO_MODULE[name] = module_name
                return getattr(module, name)
        except (ImportError, AttributeError):
            continue
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Dynamic __all__ generation
__all__ = [
    # All constraint dictionaries (dynamically discovered)
    *list(_constraint_dicts.keys()),
] + [
    # All classes from the module map
    class_name for class_names in _MODULE_MAP.values() for class_name in class_names
]
