# AlphabowRiskAssessmentV3

Enhanced Pipeline Risk Assessment Tool for Alphabow Energy

## Overview

AlphabowRiskAssessmentV3 is a comprehensive risk assessment tool designed specifically for pipeline systems. It provides advanced analytics, data quality reporting, and visualization-ready outputs for pipeline risk management.

## Features

### 🔍 **Improved Production-to-Pipeline Mapping**
- Robust UWI (Unique Well Identifier) parsing with regex validation
- Support for multiple UWI formats (standard, alternative, short)
- Confidence scoring for production-pipeline associations
- Enhanced error handling and validation

### 📊 **Risk Matrix Validation**
- Comprehensive risk matrix completeness checking
- Missing entry detection and warnings
- Value range validation (0-1 bounds)
- Automated recommendations for risk matrix improvements

### 📈 **Enhanced Production Statistics**
- Exponential decline rate calculation with curve fitting
- Production trend detection (increasing/decreasing/stable)
- R-squared goodness-of-fit metrics
- Peak and current production analysis
- Time-based production analytics

### 📋 **Data Quality Reporting**
- Production data coverage assessment
- Missing age and pressure data detection
- Zero production pipeline identification
- Overall data completeness scoring
- Comprehensive quality metrics dashboard

### ⚖️ **Sensitivity Analysis**
- Multiple risk weighting scenarios
- Ranking volatility analysis
- Sensitive asset identification
- Scenario-based recommendations
- Risk factor sensitivity measurement

### 📊 **Dashboard-Ready Data Generation**
- Interactive visualization data preparation
- Location-based asset groupings
- Timeline and trend analysis
- Production-risk correlation matrices
- Risk distribution histograms

### ✅ **Enhanced Error Reporting**
- Comprehensive validation summaries
- UWI parsing success rates
- Data integrity validation
- Production analysis verification
- Detailed error and warning logs

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wthesen/alphabow.git
   cd alphabow
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python test_alphabow_risk_assessment.py
   ```

## Usage

### Command Line Interface

```bash
# Basic usage with current directory
python AlphabowRiskAssessmentV3.py

# Specify data and output directories
python AlphabowRiskAssessmentV3.py --data-dir ./data --output-dir ./results

# Enable debug logging
python AlphabowRiskAssessmentV3.py --log-level DEBUG
```

### Python API

```python
from AlphabowRiskAssessmentV3 import AlphabowRiskAssessmentV3

# Initialize assessment tool
assessment = AlphabowRiskAssessmentV3(data_directory='./data')

# Load data files
success = assessment.load_data_files()

# Run complete analysis
assessment.main(data_directory='./data', output_directory='./results')
```

### Programmatic Access to Individual Features

```python
# UWI validation and parsing
uwi_result = assessment.validate_and_parse_uwi('100/01-01-057-20W5/00')
print(f"UWI valid: {uwi_result['valid']}")

# Production statistics calculation
prod_stats = assessment.calculate_production_statistics()
for uwi, stats in prod_stats.items():
    print(f"Well {uwi}: Decline rate {stats.decline_rate:.2f}%/year")

# Data quality assessment
quality_metrics = assessment.generate_data_quality_report()
print(f"Data completeness: {quality_metrics.data_completeness_score:.2f}")

# Sensitivity analysis
sensitivity_results = assessment.perform_sensitivity_analysis()
print(f"Found {len(sensitivity_results['most_sensitive_assets'])} sensitive assets")

# Dashboard data generation
dashboard_data = assessment.generate_dashboard_data()
print(f"Generated {len(dashboard_data)} dashboard datasets")
```

## Data Requirements

### Required Files

1. **production_export.csv** - Well production data
   - Columns: `wellbore_uwi`, `date`, `daily_oil_volume (m3/day)`, `daily_gas_volume (e3m3/day)`, etc.

2. **pipelines export data.csv** - Pipeline infrastructure data
   - Columns: `ID`, `Licence No.`, `MAOP (kPa)`, `Licence Approval Date`, etc.

3. **pine_creek_cp_data.xlsx** - Cathodic Protection survey data
   - Excel format with CP measurement data

### Optional Files

- `updated_well_list.csv` - Additional well information
- `second_white_specks_well_info.csv` - Formation-specific well data
- `second_white_specks_well_production.csv` - Formation-specific production data

## Output Files

The tool generates timestamped output files in the specified results directory:

### 📊 **Production Statistics** (`production_statistics_YYYYMMDD_HHMMSS.csv`)
```csv
uwi,decline_rate,trend_direction,r_squared,peak_production,current_production,days_producing
100/01-01-057-20W5/00,1.59,decreasing,0.53,0.75,0.13,245
```

### 📋 **Data Quality Report** (`data_quality_report_YYYYMMDD_HHMMSS.json`)
```json
{
  "timestamp": "20250710_065753",
  "metrics": {
    "production_data_coverage": 1.0,
    "missing_age_data_count": 1,
    "missing_pressure_data_count": 14,
    "zero_production_pipelines": 0,
    "total_wells": 52,
    "total_pipelines": 111,
    "data_completeness_score": 0.97
  }
}
```

### 📊 **Dashboard Data** (`dashboard_data_YYYYMMDD_HHMMSS.json`)
- Summary metrics for overview dashboards
- Location-based groupings for geographic visualization
- Timeline data for trend analysis
- Production-risk correlation matrices
- Risk distribution data for histogram charts

### ✅ **Validation Summary** (`validation_summary_YYYYMMDD_HHMMSS.json`)
```json
{
  "timestamp": "2025-07-10T06:57:53.662467",
  "data_validation": {"valid": true, "errors": [], "warnings": []},
  "uwi_validation": {"valid": true, "parse_success_rate": 1.0},
  "risk_matrix_validation": {"valid": true},
  "production_analysis_validation": {"valid": true},
  "overall_status": "valid"
}
```

## Architecture

### Core Classes

- **AlphabowRiskAssessmentV3**: Main assessment class
- **DataQualityMetrics**: Data quality measurement container
- **ProductionStatistics**: Well production analysis results
- **RiskAssessmentResult**: Risk assessment outcomes

### Key Methods

- `load_data_files()`: Load and validate input data
- `validate_and_parse_uwi()`: UWI parsing and validation
- `create_production_to_pipeline_mapping()`: Asset relationship mapping
- `calculate_production_statistics()`: Production decline analysis
- `generate_data_quality_report()`: Data completeness assessment
- `perform_sensitivity_analysis()`: Risk weighting sensitivity
- `generate_dashboard_data()`: Visualization data preparation

## Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest test_alphabow_risk_assessment.py -v

# Run specific test categories
python -m pytest test_alphabow_risk_assessment.py::TestAlphabowRiskAssessmentV3::test_uwi_validation_standard_format -v

# Run with coverage
python -m pytest test_alphabow_risk_assessment.py --cov=AlphabowRiskAssessmentV3
```

### Test Coverage

The test suite covers:
- ✅ UWI validation and parsing (multiple formats)
- ✅ Risk matrix validation
- ✅ Production statistics calculation
- ✅ Data quality reporting
- ✅ Sensitivity analysis
- ✅ Dashboard data generation
- ✅ File I/O operations
- ✅ Error handling and edge cases

## Configuration

### Risk Matrix Configuration

The risk matrix can be customized by modifying the `_initialize_risk_matrix()` method:

```python
{
    'probability': {
        'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 
        'high': 0.7, 'very_high': 0.9
    },
    'consequence': {
        'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 
        'high': 0.7, 'very_high': 0.9
    },
    'risk_categories': {
        'low': (0.0, 0.3), 'medium': (0.3, 0.6),
        'high': (0.6, 0.8), 'critical': (0.8, 1.0)
    }
}
```

### UWI Pattern Configuration

Support for additional UWI formats can be added:

```python
self.uwi_patterns = {
    'standard': r'^(\d{3})\/(\d{2})-(\d{2})-(\d{3})-(\d{2})W(\d)\/(\d{2})$',
    'alternative': r'^(\d{3})(\d{2})(\d{2})(\d{3})(\d{2})W(\d)(\d{2})$',
    'short': r'^(\d{3})\/(\d{2})-(\d{2})-(\d{3})-(\d{2})W(\d)$',
    # Add custom patterns here
}
```

## Logging

The tool provides comprehensive logging:

```python
# Configure logging level
logging.getLogger().setLevel(logging.DEBUG)

# Log files are automatically created
# - alphabow_risk_assessment.log (main log file)
# - Console output (INFO level and above)
```

## Performance Considerations

- **Memory Usage**: Large datasets (>10,000 wells) may require additional memory
- **Processing Time**: Decline curve fitting is computationally intensive for large datasets
- **File I/O**: Excel files (.xlsx) are slower to process than CSV files

### Optimization Tips

1. **Use CSV format** when possible for faster loading
2. **Filter data** to relevant time periods before analysis
3. **Batch processing** for very large datasets
4. **Parallel processing** can be added for production statistics calculation

## Troubleshooting

### Common Issues

1. **UWI Parsing Failures**
   ```
   Check UWI format against supported patterns
   Review validation summary for parse success rates
   ```

2. **Missing Data Files**
   ```
   Verify file paths and names match expected format
   Check data directory permissions
   ```

3. **Memory Issues with Large Datasets**
   ```
   Consider data filtering or chunked processing
   Monitor memory usage during decline curve fitting
   ```

4. **Excel File Reading Errors**
   ```
   Ensure openpyxl dependency is installed
   Verify Excel file format compatibility
   ```

## Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/enhancement`)
3. **Add tests** for new functionality
4. **Run test suite** (`pytest test_alphabow_risk_assessment.py`)
5. **Submit pull request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Use type hints where appropriate

## License

This project is developed for Alphabow Energy. Please refer to the company's internal licensing policies.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review test cases for usage examples
- Contact the development team for Alphabow-specific questions

## Version History

- **v3.0** - Complete rewrite with enhanced features
  - Improved UWI parsing and validation
  - Risk matrix validation functionality
  - Enhanced production statistics with decline analysis
  - Comprehensive data quality reporting
  - Sensitivity analysis capabilities
  - Dashboard-ready data generation
  - Enhanced error reporting and validation

---

**AlphabowRiskAssessmentV3** - Empowering data-driven pipeline risk management