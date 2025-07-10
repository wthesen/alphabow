# CP Restoration Priority Assessment Tool
Alphabow Energy Ltd.

## URGENT: 18+ Month CP System Outage Assessment

This repository contains the emergency assessment tools and results for the critical CP (Cathodic Protection) restoration priority assessment following an 18+ month system outage.

### Critical Timeline
- **CP Systems Inactive Since**: Q4 2023 (18+ months)
- **Assessment Date**: July 10, 2025
- **Regulatory Status**: EXTREME non-compliance
- **Immediate Action Required**: AER notification within 24 hours

### Assessment Results Summary
- **Total Steel Pipelines Assessed**: 96
- **EXTREME Priority (7 days)**: 96 pipelines
- **Total Restoration Cost**: $7.7 million
- **Estimated Timeline**: 4-6 months

### Tools and Scripts

#### 1. CP Restoration Priority Assessment (`cp_restoration_priority_assessment.py`)
Main assessment tool that:
- Loads pipeline data from CSV files
- Applies risk multipliers for 18+ month CP inactivity
- Categorizes pipelines by priority (EXTREME/HIGH/MEDIUM)
- Generates comprehensive risk scores and reports

#### 2. AER Notification Assessment (`aer_notification_assessment.py`)
Regulatory compliance tool that:
- Generates Pipeline Act Section 35 notification content
- Assesses Directive 077 compliance violations
- Calculates corrosion acceleration rates
- Provides environmental impact analysis

#### 3. Executive Summary (`CP_RESTORATION_EXECUTIVE_SUMMARY.md`)
Comprehensive summary document covering:
- Immediate deliverables and priorities
- Regulatory requirements and violations
- Cost/timeline analysis
- Emergency response requirements

### Key Data Files
- `pipelines export data.csv` - Primary pipeline segment data
- `pine_creek_cp_data.xlsx` - Cathodic protection historical data
- `cp_restoration_priority_results.csv` - Assessment results (generated)

### Risk Multipliers Applied
- **CP inactive 18+ months**: 4.0x (EXTREME)
- **Steel material without CP**: 2.5x
- **High pressure (>500 kPa)**: 2.0x
- **Water body crossing**: 2.0x
- **Age >20 years**: 1.8x

### Priority Categories
- **EXTREME** (≥80% risk score): Restore within 7 days
- **HIGH** (60-79% risk score): Restore within 30 days  
- **MEDIUM** (40-59% risk score): Restore within 90 days

### Usage

1. **Run Main Assessment**:
   ```bash
   python3 cp_restoration_priority_assessment.py
   ```

2. **Generate AER Notification**:
   ```bash
   python3 aer_notification_assessment.py
   ```

3. **Review Executive Summary**:
   ```bash
   cat CP_RESTORATION_EXECUTIVE_SUMMARY.md
   ```

### Emergency Contacts
- **AER Emergency Line**: 1-800-222-6514
- **Environmental Response**: 1-800-222-6514

### Regulatory Requirements
- **Immediate**: AER Pipeline Act Section 35 notification
- **24-48 hours**: Emergency CP restoration plan submission
- **7 days**: Begin EXTREME priority pipeline restoration
- **Monthly**: Progress reports during restoration period

### Dependencies
```bash
pip install pandas openpyxl numpy datetime
```

### File Structure
```
alphabow/
├── cp_restoration_priority_assessment.py    # Main assessment tool
├── aer_notification_assessment.py           # AER compliance tool
├── CP_RESTORATION_EXECUTIVE_SUMMARY.md      # Executive summary
├── pipelines export data.csv                # Pipeline data
├── pine_creek_cp_data.xlsx                  # CP historical data
├── cp_restoration_priority_results.csv      # Generated results
└── README.md                                # This file
```

### CRITICAL NOTICE
**This assessment represents an extreme regulatory non-compliance situation requiring immediate action. All 96 steel pipelines require emergency CP restoration within 7 days to prevent potential safety and environmental incidents.**