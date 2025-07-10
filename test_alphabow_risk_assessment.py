#!/usr/bin/env python3
"""
Test suite for AlphabowRiskAssessmentV3

This script tests the core functionality of the risk assessment tool
to ensure all features work correctly with the available data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import os
import sys

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AlphabowRiskAssessmentV3 import AlphabowRiskAssessmentV3, DataQualityMetrics, ProductionStatistics

class TestAlphabowRiskAssessmentV3:
    """Test class for AlphabowRiskAssessmentV3"""
    
    @pytest.fixture
    def assessment_tool(self):
        """Create an instance of the assessment tool for testing"""
        return AlphabowRiskAssessmentV3('.')
    
    @pytest.fixture
    def sample_production_data(self):
        """Create sample production data for testing"""
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='M')
        data = []
        
        for i, date in enumerate(dates):
            data.append({
                'wellbore_uwi': '100/01-01-057-20W5/00',
                'formatted_uwi': '100/01-01-057-20W5/00',
                'date': date,
                'oil_volume (m3)': max(0, 100 - i * 5 + np.random.normal(0, 5)),
                'gas_volume (e3m3)': max(0, 50 - i * 2 + np.random.normal(0, 2)),
                'water_volume (m3)': 10 + np.random.normal(0, 2),
                'daily_oil_volume (m3/day)': max(0, 3.3 - i * 0.15 + np.random.normal(0, 0.3)),
                'daily_gas_volume (e3m3/day)': max(0, 1.6 - i * 0.06 + np.random.normal(0, 0.1)),
                'daily_water_volume (m3/day)': 0.3 + np.random.normal(0, 0.05),
                'producing_formation': 'Cardium'
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_pipeline_data(self):
        """Create sample pipeline data for testing"""
        return pd.DataFrame([
            {
                'ID': 'AB-56757-2',
                'Licence No.': '56757',
                'Segment Line No.': '2',
                'Province': 'AB',
                'Company': 'Alphabow Energy Ltd.',
                'Substance': 'Salt Water',
                'Segment Status': 'Operating',
                'Segment Length (m)': 1350,
                'Outer Diameter (mm)': 168.3,
                'Licence Approval Date': '2005-04-20',
                'Pipe Material': 'Steel',
                'MAOP (kPa)': 9000,
                'Wall Thickness (mm)': 4.0,
                'H2S Content': 0.45
            },
            {
                'ID': 'AB-51780-1',
                'Licence No.': '51780',
                'Segment Line No.': '1',
                'Province': 'AB',
                'Company': 'Alphabow Energy Ltd.',
                'Substance': 'Natural Gas',
                'Segment Status': 'Operating',
                'Segment Length (m)': 580,
                'Outer Diameter (mm)': 88.9,
                'Licence Approval Date': '2010-10-07',
                'Pipe Material': 'Steel',
                'MAOP (kPa)': 9930,
                'Wall Thickness (mm)': 4.0,
                'H2S Content': 0
            }
        ])
    
    def test_initialization(self, assessment_tool):
        """Test that the assessment tool initializes correctly"""
        assert assessment_tool is not None
        assert assessment_tool.data_directory == Path('.')
        assert assessment_tool.uwi_patterns is not None
        assert len(assessment_tool.uwi_patterns) > 0
        assert assessment_tool.risk_matrix is not None
    
    def test_uwi_validation_standard_format(self, assessment_tool):
        """Test UWI validation with standard format"""
        # Test valid UWI
        valid_uwi = "100/01-01-057-20W5/00"
        result = assessment_tool.validate_and_parse_uwi(valid_uwi)
        
        assert result['valid'] == True
        assert result['original'] == valid_uwi
        assert result['pattern_type'] == 'standard'
        assert result['lsd'] == '100'
        assert result['section'] == '01'
        assert result['township'] == '01'
        assert result['range'] == '057'
        assert result['meridian'] == '20'
        assert result['event_sequence'] == '5'
    
    def test_uwi_validation_invalid_format(self, assessment_tool):
        """Test UWI validation with invalid format"""
        invalid_uwi = "invalid-uwi-format"
        result = assessment_tool.validate_and_parse_uwi(invalid_uwi)
        
        assert result['valid'] == False
        assert 'error' in result
        assert result['original'] == invalid_uwi
    
    def test_uwi_validation_empty_input(self, assessment_tool):
        """Test UWI validation with empty/null input"""
        result_empty = assessment_tool.validate_and_parse_uwi("")
        result_none = assessment_tool.validate_and_parse_uwi(None)
        result_nan = assessment_tool.validate_and_parse_uwi(pd.NA)
        
        assert result_empty['valid'] == False
        assert result_none['valid'] == False
        assert result_nan['valid'] == False
    
    def test_risk_matrix_validation(self, assessment_tool):
        """Test risk matrix validation"""
        validation_result = assessment_tool.validate_risk_matrix()
        
        assert 'valid' in validation_result
        assert 'warnings' in validation_result
        assert 'missing_entries' in validation_result
        assert 'recommendations' in validation_result
        
        # The default risk matrix should be valid
        assert validation_result['valid'] == True
    
    def test_production_statistics_calculation(self, assessment_tool, sample_production_data):
        """Test production statistics calculation"""
        # Set up sample data
        assessment_tool.production_data = sample_production_data
        
        # Calculate statistics
        stats = assessment_tool.calculate_production_statistics()
        
        assert len(stats) > 0
        assert '100/01-01-057-20W5/00' in stats
        
        well_stats = stats['100/01-01-057-20W5/00']
        assert isinstance(well_stats, ProductionStatistics)
        assert well_stats.uwi == '100/01-01-057-20W5/00'
        assert well_stats.decline_rate >= 0
        assert well_stats.trend_direction in ['increasing', 'decreasing', 'stable', 'insufficient_data']
        assert 0 <= well_stats.r_squared <= 1
        assert well_stats.peak_production >= 0
        assert well_stats.current_production >= 0
        assert well_stats.days_producing >= 0
    
    def test_data_quality_report_generation(self, assessment_tool, sample_production_data, sample_pipeline_data):
        """Test data quality report generation"""
        # Set up sample data
        assessment_tool.production_data = sample_production_data
        assessment_tool.pipeline_data = sample_pipeline_data
        
        # Generate report
        metrics = assessment_tool.generate_data_quality_report()
        
        assert isinstance(metrics, DataQualityMetrics)
        assert 0 <= metrics.production_data_coverage <= 1
        assert metrics.missing_age_data_count >= 0
        assert metrics.missing_pressure_data_count >= 0
        assert metrics.zero_production_pipelines >= 0
        assert metrics.total_wells >= 0
        assert metrics.total_pipelines >= 0
        assert 0 <= metrics.data_completeness_score <= 1
    
    def test_sensitivity_analysis(self, assessment_tool, sample_pipeline_data):
        """Test sensitivity analysis functionality"""
        # Set up sample data
        assessment_tool.pipeline_data = sample_pipeline_data
        
        # Run sensitivity analysis
        results = assessment_tool.perform_sensitivity_analysis()
        
        assert 'scenarios' in results
        assert 'ranking_volatility' in results
        assert 'most_sensitive_assets' in results
        assert 'recommendations' in results
        
        # Check that we have the expected scenarios
        expected_scenarios = ['baseline', 'production_focused', 'technical_focused', 'environmental_focused', 'balanced']
        for scenario in expected_scenarios:
            assert scenario in results['scenarios']
            assert 'weights' in results['scenarios'][scenario]
            assert 'risk_scores' in results['scenarios'][scenario]
            assert 'rankings' in results['scenarios'][scenario]
    
    def test_dashboard_data_generation(self, assessment_tool, sample_production_data, sample_pipeline_data):
        """Test dashboard data generation"""
        # Set up sample data
        assessment_tool.production_data = sample_production_data
        assessment_tool.pipeline_data = sample_pipeline_data
        
        # Calculate required intermediate results
        assessment_tool.calculate_production_statistics()
        assessment_tool.generate_data_quality_report()
        
        # Generate dashboard data
        dashboard_data = assessment_tool.generate_dashboard_data()
        
        assert 'summary_metrics' in dashboard_data
        assert 'location_groupings' in dashboard_data
        assert 'timeline_data' in dashboard_data
        assert 'production_risk_correlation' in dashboard_data
        assert 'risk_distribution' in dashboard_data
        assert 'data_quality_indicators' in dashboard_data
        
        # Check summary metrics
        summary = dashboard_data['summary_metrics']
        assert 'total_wells' in summary
        assert 'total_pipelines' in summary
        assert 'active_wells' in summary
        assert 'data_completeness' in summary
    
    def test_validation_summary_generation(self, assessment_tool):
        """Test validation summary generation"""
        validation_summary = assessment_tool.generate_validation_summary()
        
        assert 'timestamp' in validation_summary
        assert 'data_validation' in validation_summary
        assert 'uwi_validation' in validation_summary
        assert 'risk_matrix_validation' in validation_summary
        assert 'production_analysis_validation' in validation_summary
        assert 'overall_status' in validation_summary
        
        # Each validation section should have the required fields
        for section in ['data_validation', 'uwi_validation', 'risk_matrix_validation', 'production_analysis_validation']:
            assert 'valid' in validation_summary[section]
            assert 'errors' in validation_summary[section]
            assert 'warnings' in validation_summary[section]
    
    def test_save_results(self, assessment_tool, sample_production_data, sample_pipeline_data):
        """Test saving results to files"""
        # Set up sample data and run analysis
        assessment_tool.production_data = sample_production_data
        assessment_tool.pipeline_data = sample_pipeline_data
        assessment_tool.calculate_production_statistics()
        assessment_tool.generate_data_quality_report()
        assessment_tool.generate_dashboard_data()
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            success = assessment_tool.save_results(temp_dir)
            assert success == True
            
            # Check that files were created
            output_files = list(Path(temp_dir).glob('*'))
            assert len(output_files) > 0
            
            # Check for specific expected files
            csv_files = list(Path(temp_dir).glob('*.csv'))
            json_files = list(Path(temp_dir).glob('*.json'))
            
            assert len(csv_files) > 0  # Should have production statistics CSV
            assert len(json_files) > 0  # Should have JSON output files
    
    def test_production_pipeline_mapping(self, assessment_tool, sample_production_data, sample_pipeline_data):
        """Test production to pipeline mapping functionality"""
        # Set up sample data
        assessment_tool.production_data = sample_production_data
        assessment_tool.pipeline_data = sample_pipeline_data
        
        # Create mapping
        mapped_data = assessment_tool.create_production_to_pipeline_mapping()
        
        assert isinstance(mapped_data, pd.DataFrame)
        if len(mapped_data) > 0:
            assert 'mapping_confidence' in mapped_data.columns
            # Check that mapping confidence is between 0 and 1
            assert all(0 <= conf <= 1 for conf in mapped_data['mapping_confidence'])

def test_main_function():
    """Test the main function execution"""
    # Create a temporary directory with sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create minimal sample data files
        sample_production = pd.DataFrame([
            {
                'wellbore_uwi': '100/01-01-057-20W5/00',
                'date': '2022-01-01',
                'daily_oil_volume (m3/day)': 3.0,
                'daily_gas_volume (e3m3/day)': 1.5,
                'daily_water_volume (m3/day)': 0.3
            }
        ])
        
        sample_pipelines = pd.DataFrame([
            {
                'ID': 'AB-56757-2',
                'Licence Approval Date': '2005-04-20',
                'MAOP (kPa)': 9000
            }
        ])
        
        # Save sample data files
        sample_production.to_csv(temp_path / 'production_export.csv', index=False)
        sample_pipelines.to_csv(temp_path / 'pipelines export data.csv', index=False)
        
        # Test main function
        assessment = AlphabowRiskAssessmentV3(str(temp_path))
        success = assessment.main(str(temp_path), str(temp_path / 'output'))
        
        # The main function should complete successfully even with minimal data
        assert success == True

def run_basic_functionality_test():
    """Run a basic functionality test with actual data files if available"""
    print("Running basic functionality test...")
    
    # Initialize assessment tool
    assessment = AlphabowRiskAssessmentV3('.')
    
    # Try to load actual data files
    try:
        success = assessment.load_data_files()
        print(f"Data loading: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            # Test UWI validation with actual data
            if assessment.production_data is not None:
                sample_uwi = assessment.production_data['wellbore_uwi'].iloc[0]
                uwi_result = assessment.validate_and_parse_uwi(sample_uwi)
                print(f"UWI validation for '{sample_uwi}': {'SUCCESS' if uwi_result['valid'] else 'FAILED'}")
            
            # Test risk matrix validation
            matrix_validation = assessment.validate_risk_matrix()
            print(f"Risk matrix validation: {'SUCCESS' if matrix_validation['valid'] else 'FAILED'}")
            
            # Test production statistics calculation
            if assessment.production_data is not None:
                prod_stats = assessment.calculate_production_statistics()
                print(f"Production statistics: {'SUCCESS' if len(prod_stats) > 0 else 'FAILED'} ({len(prod_stats)} wells analyzed)")
            
            # Test data quality report
            quality_metrics = assessment.generate_data_quality_report()
            if quality_metrics:
                print(f"Data quality report: SUCCESS (score: {quality_metrics.data_completeness_score:.2f})")
            else:
                print("Data quality report: FAILED")
            
            # Test dashboard data generation
            dashboard_data = assessment.generate_dashboard_data()
            print(f"Dashboard data generation: {'SUCCESS' if dashboard_data else 'FAILED'}")
            
            print("\nBasic functionality test completed successfully!")
            return True
        else:
            print("Could not load data files - this is expected if running without actual data")
            return False
            
    except Exception as e:
        print(f"Error during basic functionality test: {str(e)}")
        return False

if __name__ == '__main__':
    # Run basic functionality test if called directly
    print("=" * 60)
    print("AlphabowRiskAssessmentV3 Test Suite")
    print("=" * 60)
    
    # Run basic functionality test
    basic_test_success = run_basic_functionality_test()
    
    print("\n" + "=" * 60)
    print("To run the full pytest suite, use: pytest test_alphabow_risk_assessment.py -v")
    print("=" * 60)