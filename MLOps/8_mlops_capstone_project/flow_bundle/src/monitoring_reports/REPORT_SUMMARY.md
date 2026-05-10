# NannyML Monitoring Report

**Generated:** 2026-05-10 17:43:40

**Experiment:** green_taxi_monitoring

## Summary Statistics

- **Total Runs:** 6
- **Integrity Passed:** 6
- **Integrity Failed:** 0
- **Runs with Drift Warnings:** 6
- **Average Issues per Run:** 11.0
- **Max Issues in Single Run:** 11

## Data Quality Score

**Overall Score: 45/100** - POOR

## Drift Detection - Most Frequent

- **DOLocationID**: 6 occurrences
- **PULocationID**: 6 occurrences
- **RatecodeID**: 6 occurrences
- **congestion_surcharge**: 6 occurrences
- **fare_amount**: 6 occurrences

## Generated Visualizations

1. **01_warnings_over_time.png** - Severity trend with anomaly detection
2. **02_issue_severity_scorecard.png** - Which columns have the worst issues
3. **03_drift_columns_frequency.png** - Most frequently drifted columns
4. **04_drift_heatmap.png** - Column-wise drift per run (matrix view)
5. **05_integrity_summary.png** - Overall system health dashboard

## Recommendations

SUCCESS: 6 batches passed validation - good progress

- Start by fixing the CRITICAL columns identified above
- Check data collection and preprocessing scripts
- Consider relaxing thresholds if they are too strict
- Set up data quality checks at the source
