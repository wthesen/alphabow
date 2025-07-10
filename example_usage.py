#!/usr/bin/env python3
"""
Example usage script for AlphabowRiskAssessmentV3

This script demonstrates how to use the various features of the
risk assessment tool with sample data.
"""

import pandas as pd
import json
from pathlib import Path
from AlphabowRiskAssessmentV3 import AlphabowRiskAssessmentV3

def main():
    """
    Demonstrate AlphabowRiskAssessmentV3 functionality
    """
    print("=" * 60)
    print("AlphabowRiskAssessmentV3 Example Usage")
    print("=" * 60)
    
    # Initialize the assessment tool
    print("\n1. Initializing Assessment Tool...")
    assessment = AlphabowRiskAssessmentV3('.')
    
    # Load data files
    print("\n2. Loading Data Files...")
    success = assessment.load_data_files()
    if success:
        print("   ✓ Data files loaded successfully")
        print(f"   ✓ Production records: {len(assessment.production_data) if assessment.production_data is not None else 0}")
        print(f"   ✓ Pipeline records: {len(assessment.pipeline_data) if assessment.pipeline_data is not None else 0}")
        print(f"   ✓ CP survey records: {len(assessment.cp_survey_data) if assessment.cp_survey_data is not None else 0}")
    else:
        print("   ✗ Failed to load data files")
        return
    
    # Demonstrate UWI validation
    print("\n3. UWI Validation Examples...")
    test_uwis = [
        "100/01-01-057-20W5/00",  # Valid standard format
        "100010105720W500",       # Valid alternative format
        "invalid-uwi-format",     # Invalid format
        "",                       # Empty UWI
    ]
    
    for uwi in test_uwis:
        result = assessment.validate_and_parse_uwi(uwi)
        status = "✓ VALID" if result['valid'] else "✗ INVALID"
        print(f"   {status}: '{uwi}'")
        if result['valid']:
            print(f"      Pattern: {result['pattern_type']}")
            print(f"      LSD: {result.get('lsd', 'N/A')}, Section: {result.get('section', 'N/A')}")
    
    # Risk matrix validation
    print("\n4. Risk Matrix Validation...")
    matrix_validation = assessment.validate_risk_matrix()
    if matrix_validation['valid']:
        print("   ✓ Risk matrix is valid")
    else:
        print("   ✗ Risk matrix has issues:")
        for error in matrix_validation.get('missing_entries', []):
            print(f"      - {error}")
    
    # Production statistics calculation
    print("\n5. Production Statistics Analysis...")
    if assessment.production_data is not None:
        prod_stats = assessment.calculate_production_statistics()
        print(f"   ✓ Analyzed {len(prod_stats)} wells")
        
        # Show sample statistics
        if prod_stats:
            sample_uwi = list(prod_stats.keys())[0]
            stats = prod_stats[sample_uwi]
            print(f"\n   Sample Well: {sample_uwi}")
            print(f"   - Decline Rate: {stats.decline_rate:.2f}%/year")
            print(f"   - Trend: {stats.trend_direction}")
            print(f"   - R-squared: {stats.r_squared:.3f}")
            print(f"   - Peak Production: {stats.peak_production:.2f} BOE/day")
            print(f"   - Current Production: {stats.current_production:.2f} BOE/day")
            print(f"   - Days Producing: {stats.days_producing}")
        
        # Show decline rate distribution
        decline_rates = [stats.decline_rate for stats in prod_stats.values()]
        avg_decline = sum(decline_rates) / len(decline_rates)
        print(f"\n   Average Decline Rate: {avg_decline:.2f}%/year")
        
        # Show trend distribution
        trends = [stats.trend_direction for stats in prod_stats.values()]
        trend_counts = {}
        for trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        print("   Production Trends:")
        for trend, count in trend_counts.items():
            print(f"   - {trend}: {count} wells")
    
    # Data quality assessment
    print("\n6. Data Quality Assessment...")
    quality_metrics = assessment.generate_data_quality_report()
    if quality_metrics:
        print(f"   ✓ Overall Quality Score: {quality_metrics.data_completeness_score:.2f}")
        print(f"   ✓ Production Coverage: {quality_metrics.production_data_coverage:.1%}")
        print(f"   ✓ Total Wells: {quality_metrics.total_wells}")
        print(f"   ✓ Total Pipelines: {quality_metrics.total_pipelines}")
        print(f"   - Missing Age Data: {quality_metrics.missing_age_data_count} records")
        print(f"   - Missing Pressure Data: {quality_metrics.missing_pressure_data_count} records")
        print(f"   - Zero Production Pipelines: {quality_metrics.zero_production_pipelines}")
    
    # Sensitivity analysis
    print("\n7. Sensitivity Analysis...")
    sensitivity_results = assessment.perform_sensitivity_analysis()
    if sensitivity_results:
        scenarios = sensitivity_results['scenarios']
        print(f"   ✓ Analyzed {len(scenarios)} risk weighting scenarios")
        
        # Show scenario results
        for scenario_name, scenario_data in scenarios.items():
            weights = scenario_data['weights']
            risk_scores = scenario_data['risk_scores']
            avg_risk = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0
            print(f"   - {scenario_name}: Avg Risk = {avg_risk:.3f}")
            print(f"     Weights: Prod={weights['production_risk']:.1f}, "
                  f"Tech={weights['technical_risk']:.1f}, "
                  f"Env={weights['environmental_risk']:.1f}")
        
        # Show most sensitive assets
        sensitive_assets = sensitivity_results.get('most_sensitive_assets', [])
        if sensitive_assets:
            print(f"\n   Most Sensitive Assets (top 3):")
            for i, (asset, volatility) in enumerate(sensitive_assets[:3]):
                print(f"   {i+1}. {asset}: Volatility = {volatility:.2f}")
    
    # Dashboard data generation
    print("\n8. Dashboard Data Generation...")
    dashboard_data = assessment.generate_dashboard_data()
    if dashboard_data:
        print(f"   ✓ Generated dashboard data with {len(dashboard_data)} datasets")
        
        # Show summary metrics
        summary = dashboard_data.get('summary_metrics', {})
        print("   Summary Metrics:")
        for key, value in summary.items():
            print(f"   - {key}: {value}")
        
        # Show timeline data sample
        timeline_data = dashboard_data.get('timeline_data', {})
        if 'monthly_production' in timeline_data:
            monthly_data = timeline_data['monthly_production']
            print(f"   ✓ Generated {len(monthly_data)} monthly production records")
    
    # Generate validation summary
    print("\n9. Validation Summary...")
    validation_summary = assessment.generate_validation_summary()
    overall_status = validation_summary.get('overall_status', 'unknown')
    print(f"   Overall Status: {overall_status.upper()}")
    
    # Show validation details
    for section, results in validation_summary.items():
        if section in ['data_validation', 'uwi_validation', 'risk_matrix_validation', 'production_analysis_validation']:
            status = "✓ PASS" if results.get('valid', False) else "✗ FAIL"
            print(f"   {status}: {section.replace('_', ' ').title()}")
            
            # Show warnings if any
            warnings = results.get('warnings', [])
            if warnings:
                for warning in warnings[:2]:  # Show first 2 warnings
                    print(f"      Warning: {warning}")
    
    # Production-to-pipeline mapping
    print("\n10. Production-Pipeline Mapping...")
    try:
        mapped_data = assessment.create_production_to_pipeline_mapping()
        if len(mapped_data) > 0:
            print(f"    ✓ Created {len(mapped_data)} production-pipeline mappings")
            
            # Show confidence distribution
            if 'mapping_confidence' in mapped_data.columns:
                avg_confidence = mapped_data['mapping_confidence'].mean()
                print(f"    Average mapping confidence: {avg_confidence:.2f}")
        else:
            print("    No mappings created (expected with limited sample data)")
    except Exception as e:
        print(f"    Warning: Mapping failed - {str(e)}")
    
    # Save results demonstration
    print("\n11. Saving Results...")
    output_dir = Path('./example_results')
    output_dir.mkdir(exist_ok=True)
    
    success = assessment.save_results(str(output_dir))
    if success:
        print(f"   ✓ Results saved to {output_dir}")
        
        # List generated files
        output_files = list(output_dir.glob('*'))
        print("   Generated files:")
        for file_path in output_files:
            file_size = file_path.stat().st_size
            print(f"   - {file_path.name} ({file_size:,} bytes)")
    else:
        print("   ✗ Failed to save results")
    
    print("\n" + "=" * 60)
    print("Example Usage Complete!")
    print("=" * 60)
    
    # Final recommendations
    print("\nNext Steps:")
    print("1. Review generated output files in ./example_results/")
    print("2. Customize risk matrix parameters as needed")
    print("3. Integrate dashboard data with visualization tools")
    print("4. Set up automated reporting workflows")
    print("5. Configure sensitivity analysis weights for your use case")

if __name__ == '__main__':
    main()