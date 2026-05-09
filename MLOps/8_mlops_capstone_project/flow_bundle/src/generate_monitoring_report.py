"""generate_monitoring_report.py

Convenience script to generate nannyml drift monitoring visualizations.

Usage:
    python generate_monitoring_report.py \
        --experiment green_taxi_monitoring \
        --tracking-uri http://localhost:5000 \
        --output-dir monitoring_reports

"""

import argparse
import logging
from pathlib import Path
from nannyml_visualization import create_monitoring_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-10s %(levelname)-7s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("report_gen")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NannyML drift monitoring visualizations from MLflow experiment"
    )
    parser.add_argument(
        "--experiment",
        default="green_taxi_monitoring",
        help="MLflow experiment name (default: green_taxi_monitoring)"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--output-dir",
        default="monitoring_reports",
        help="Directory to save report (default: monitoring_reports)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Generating monitoring report for experiment: {args.experiment}")
    
    try:
        files = create_monitoring_report(
            experiment_name=args.experiment,
            tracking_uri=args.tracking_uri,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 60)
        print("✓ Monitoring report generated successfully!")
        print("=" * 60)
        print(f"\nOutput directory: {Path(args.output_dir).absolute()}")
        print(f"\nGenerated files:")
        for key, path in sorted(files.items()):
            if Path(path).exists():
                print(f"  ✓ {key:.<30} {path}")
            else:
                print(f"  ✗ {key:.<30} {path} (not found)")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
