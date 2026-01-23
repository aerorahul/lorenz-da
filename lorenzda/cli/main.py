#!/usr/bin/env python
"""
Lorenz-DA Command Line Interface

Main entry point for running Lorenz data assimilation experiments.
"""

import sys
import argparse
import importlib
import importlib.util
from pathlib import Path

from lorenzda.config.yaml_loader import load_config_from_yaml, Config


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog='lorenzda',
        description='Lorenz Data Assimilation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lorenzda truth L63_param_truthDA
  lorenzda ensemble L63_param_ensDA
  lorenzda variational L63_param_varDA
  lorenzda hybrid L63_param_hybDA
  lorenzda ensemble-var L96_param_ensvarDA
  lorenzda hybrid-ensemble L96_param_hybensvarDA
  lorenzda model-run L96_param_truthDA
  lorenzda stats L96_param_varDA

Configuration files are loaded from the current working directory or
lorenzda/config/ subdirectory.
        """
    )

    parser.add_argument(
        '--version', action='version', version='%(prog)s 1.0'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Truth generation command
    truth_parser = subparsers.add_parser(
        'truth',
        help='Generate truth trajectory for data assimilation'
    )
    truth_parser.add_argument('config', help='Configuration file/module name')
    truth_parser.set_defaults(func=cmd_truth)

    # Ensemble DA command
    ens_parser = subparsers.add_parser(
        'ensemble',
        help='Run ensemble-based data assimilation'
    )
    ens_parser.add_argument('config', help='Configuration file/module name')
    ens_parser.set_defaults(func=cmd_ensemble)

    # Variational DA command
    var_parser = subparsers.add_parser(
        'variational',
        help='Run variational data assimilation (3DVar/4DVar)'
    )
    var_parser.add_argument('config', help='Configuration file/module name')
    var_parser.set_defaults(func=cmd_variational)

    # Hybrid DA command
    hyb_parser = subparsers.add_parser(
        'hybrid',
        help='Run hybrid data assimilation (EnVar)'
    )
    hyb_parser.add_argument('config', help='Configuration file/module name')
    hyb_parser.set_defaults(func=cmd_hybrid)

    # Ensemble-Variational command
    ensvar_parser = subparsers.add_parser(
        'ensemble-var',
        help='Run ensemble-variational data assimilation'
    )
    ensvar_parser.add_argument('config', help='Configuration file/module name')
    ensvar_parser.set_defaults(func=cmd_ensemble_var)

    # Hybrid Ensemble-Variational command
    hybensvar_parser = subparsers.add_parser(
        'hybrid-ensemble',
        help='Run hybrid ensemble-variational data assimilation'
    )
    hybensvar_parser.add_argument('config', help='Configuration file/module name')
    hybensvar_parser.set_defaults(func=cmd_hybrid_ensemble_var)

    # Model integration command
    model_parser = subparsers.add_parser(
        'model-run',
        help='Run standalone model integration'
    )
    model_parser.add_argument('config', help='Configuration file/module name')
    model_parser.set_defaults(func=cmd_model_run)

    # Statistics/covariance command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Generate background error covariance statistics'
    )
    stats_parser.add_argument('config', help='Configuration file/module name')
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def load_config(config_name):
    """
    Load configuration from config_name (YAML or Python module).

    Tries multiple approaches:
    1. Load as YAML file (.yaml or .yml extension)
    2. Direct import as Python module
    3. Import from lorenzda.config subpackage
    4. Load from lorenzda/config/ directory (Python)
    5. Load from lorenzda/config/ directory (YAML)
    """

    config_path = Path(config_name)

    # Check for YAML files in current directory
    for ext in ['.yaml', '.yml']:
        if config_path.with_suffix(ext).exists():
            return load_config_from_yaml(str(config_path.with_suffix(ext)))

    # Check for YAML files in lorenzda/config/
    yaml_path = Path(f'lorenzda/config/{config_name}')
    for ext in ['.yaml', '.yml']:
        if yaml_path.with_suffix(ext).exists():
            return load_config_from_yaml(str(yaml_path.with_suffix(ext)))

    config_module = None

    # Try direct Python module import
    try:
        config_module = importlib.import_module(config_name)
        return config_module
    except ImportError:
        pass

    # Try from lorenzda.config subpackage
    try:
        config_module = importlib.import_module(f'lorenzda.config.{config_name}')
        return config_module
    except ImportError:
        pass

    # Try loading Python file from config directory
    config_file = Path(f'lorenzda/config/{config_name}.py')
    if config_file.exists():
        spec = importlib.util.spec_from_file_location(config_name, config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module

    raise ImportError(
        f"Cannot find configuration: {config_name}\n"
        f"Tried: {config_name}.yaml, {config_name}.yml, "
        f"lorenzda/config/{config_name}.yaml, lorenzda/config/{config_name}.yml, "
        f"lorenzda/config/{config_name}.py"
    )


def cmd_truth(args):
    """Execute truth generation command."""
    from lorenzda.apps import run_truth_da
    config = load_config(args.config)
    run_truth_da(config)


def cmd_ensemble(args):
    """Execute ensemble DA command."""
    from lorenzda.apps import run_ensemble_da
    config = load_config(args.config)
    run_ensemble_da(config)


def cmd_variational(args):
    """Execute variational DA command."""
    from lorenzda.apps import run_variational_da
    config = load_config(args.config)
    run_variational_da(config)


def cmd_hybrid(args):
    """Execute hybrid DA command."""
    print("Hybrid DA not yet implemented")
    sys.exit(1)


def cmd_ensemble_var(args):
    """Execute ensemble-variational DA command."""
    print("Ensemble-Variational DA not yet implemented")
    sys.exit(1)


def cmd_hybrid_ensemble_var(args):
    """Execute hybrid ensemble-variational DA command."""
    print("Hybrid Ensemble-Variational DA not yet implemented")
    sys.exit(1)


def cmd_model_run(args):
    """Execute model integration command."""
    print("Standalone model run not yet implemented")
    sys.exit(1)


def cmd_stats(args):
    """Execute statistics/covariance generation command."""
    print("Covariance statistics generation not yet implemented")
    sys.exit(1)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
