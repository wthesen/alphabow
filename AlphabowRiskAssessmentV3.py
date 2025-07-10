#!/usr/bin/env python3
"""
Alphabow Risk Assessment V3

A comprehensive pipeline risk assessment tool for Alphabow Energy Ltd.
This script loads and analyzes data from multiple sources to assess pipeline integrity
and operational risks across the pipeline network.

Usage:
    python AlphabowRiskAssessmentV3.py [options]

Required input files (located in alphabow_files directory):
    - well_production.csv: Production data from wells
    - pipelines_list.csv: Pipeline infrastructure data
    - pipeline_cp_survey.csv: Cathodic protection survey data
    - soil_pipe_potentials.csv: Soil corrosivity and environmental data
    - gas_analysis_list.csv: Gas composition analysis
    - well_info.csv: Well metadata and configuration
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

class AlphabowRiskAssessment:
    """Main class for pipeline risk assessment analysis"""
    
    def __init__(self, data_directory="alphabow_files"):
        """
        Initialize the risk assessment tool
        
        Args:
            data_directory (str): Directory containing input data files
        """
        self.data_dir = Path(data_directory)
        self.logger = self._setup_logging()
        
        # Required input files with their expected names
        self.required_files = {
            'well_production': 'well_production.csv',
            'pipelines_list': 'pipelines_list.csv', 
            'pipeline_cp_survey': 'pipeline_cp_survey.csv',
            'soil_pipe_potentials': 'soil_pipe_potentials.csv',
            'gas_analysis_list': 'gas_analysis_list.csv',
            'well_info': 'well_info.csv'
        }
        
        # Data containers
        self.data = {}
        self.risk_scores = {}
        self.analysis_results = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'alphabow_risk_assessment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def load_data_files(self):
        """Load all required data files from the alphabow_files directory"""
        self.logger.info(f"Loading data files from {self.data_dir}")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found")
        
        # Load each required file
        for file_key, filename in self.required_files.items():
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"Required file '{filename}' not found in {self.data_dir}")
            
            try:
                self.logger.info(f"Loading {filename}...")
                self.data[file_key] = pd.read_csv(file_path)
                self.logger.info(f"Successfully loaded {filename}: {self.data[file_key].shape[0]} rows, {self.data[file_key].shape[1]} columns")
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {str(e)}")
                raise
        
        self.logger.info("All data files loaded successfully")
        return True
    
    def validate_data(self):
        """Validate loaded data for completeness and consistency"""
        self.logger.info("Validating data integrity...")
        
        validation_results = {}
        
        # Validate well production data
        if 'well_production' in self.data:
            wp_data = self.data['well_production']
            validation_results['well_production'] = {
                'total_records': len(wp_data),
                'unique_wells': wp_data['wellbore_uwi'].nunique() if 'wellbore_uwi' in wp_data.columns else 0,
                'date_range': f"{wp_data['date'].min()} to {wp_data['date'].max()}" if 'date' in wp_data.columns else "No date data"
            }
        
        # Validate pipeline data
        if 'pipelines_list' in self.data:
            pl_data = self.data['pipelines_list']
            validation_results['pipelines_list'] = {
                'total_pipelines': len(pl_data),
                'unique_licenses': pl_data['Licence No.'].nunique() if 'Licence No.' in pl_data.columns else 0,
                'operating_status': pl_data['Segment Status'].value_counts().to_dict() if 'Segment Status' in pl_data.columns else {}
            }
        
        # Validate gas analysis data
        if 'gas_analysis_list' in self.data:
            ga_data = self.data['gas_analysis_list']
            validation_results['gas_analysis_list'] = {
                'total_analyses': len(ga_data),
                'unique_wells': ga_data['UWI'].nunique() if 'UWI' in ga_data.columns else 0
            }
        
        # Log validation summary
        for file_type, results in validation_results.items():
            self.logger.info(f"{file_type} validation: {results}")
        
        return validation_results
    
    def calculate_production_risk_factors(self):
        """Calculate risk factors based on production data"""
        self.logger.info("Calculating production-based risk factors...")
        
        if 'well_production' not in self.data:
            self.logger.warning("Well production data not available for risk calculation")
            return {}
        
        wp_data = self.data['well_production'].copy()
        
        # Calculate production metrics per well
        well_metrics = wp_data.groupby('wellbore_uwi').agg({
            'oil_volume (m3)': ['sum', 'mean', 'std'],
            'gas_volume (e3m3)': ['sum', 'mean', 'std'], 
            'water_volume (m3)': ['sum', 'mean', 'std'],
            'production_hours': 'mean'
        }).reset_index()
        
        # Flatten column names
        well_metrics.columns = ['wellbore_uwi'] + ['_'.join(col).strip() for col in well_metrics.columns[1:]]
        
        # Calculate risk scores based on production variability
        production_risk = {}
        for _, row in well_metrics.iterrows():
            well_id = row['wellbore_uwi']
            
            # High variability in production indicates potential equipment issues
            gas_cv = (row['gas_volume (e3m3)_std'] / row['gas_volume (e3m3)_mean']) if row['gas_volume (e3m3)_mean'] > 0 else 0
            oil_cv = (row['oil_volume (m3)_std'] / row['oil_volume (m3)_mean']) if row['oil_volume (m3)_mean'] > 0 else 0
            
            # Water production risk (high water cut can indicate casing issues)
            water_ratio = row['water_volume (m3)_sum'] / (row['oil_volume (m3)_sum'] + row['gas_volume (e3m3)_sum'] + 1)
            
            production_risk[well_id] = {
                'gas_variability_risk': min(gas_cv * 10, 10),  # Scale to 0-10
                'oil_variability_risk': min(oil_cv * 10, 10),
                'water_production_risk': min(water_ratio * 5, 10),
                'overall_production_risk': np.mean([gas_cv * 10, oil_cv * 10, water_ratio * 5])
            }
        
        self.risk_scores['production'] = production_risk
        self.logger.info(f"Calculated production risk factors for {len(production_risk)} wells")
        
        return production_risk
    
    def calculate_pipeline_integrity_risk(self):
        """Calculate pipeline integrity risk based on infrastructure data"""
        self.logger.info("Calculating pipeline integrity risk factors...")
        
        if 'pipelines_list' not in self.data:
            self.logger.warning("Pipeline list data not available for risk calculation")
            return {}
        
        pl_data = self.data['pipelines_list'].copy()
        
        pipeline_risk = {}
        
        for _, row in pl_data.iterrows():
            pipeline_id = row.get('ID', f"Pipeline_{row.name}")
            
            # Age-based risk (older pipelines have higher risk)
            approval_date = pd.to_datetime(row.get('Licence Approval Date', '1990-01-01'), errors='coerce')
            age_years = (datetime.now() - approval_date).days / 365.25 if pd.notna(approval_date) else 30
            age_risk = min(age_years / 50 * 10, 10)  # Scale to 0-10, max at 50 years
            
            # Diameter-based risk (larger diameter = higher consequence)
            diameter = pd.to_numeric(row.get('Outer Diameter (mm)', 100), errors='coerce')
            diameter_risk = min(diameter / 1000 * 5, 10)  # Scale based on diameter
            
            # Pressure-based risk
            maop = pd.to_numeric(row.get('MAOP (kPa)', 5000), errors='coerce')
            pressure_risk = min(maop / 15000 * 10, 10)  # Scale to 0-10
            
            # Material and coating risk
            material_risk = 3 if row.get('Pipe Material', '').lower() == 'steel' else 2
            coating_risk = 2 if 'Uncoated' in str(row.get('Pipe Exterior Coating', '')) else 1
            
            # H2S risk
            h2s_risk = 5 if row.get('Has H2S', False) else 1
            
            overall_risk = np.mean([age_risk, diameter_risk, pressure_risk, material_risk, coating_risk, h2s_risk])
            
            pipeline_risk[pipeline_id] = {
                'age_risk': age_risk,
                'diameter_risk': diameter_risk,
                'pressure_risk': pressure_risk,
                'material_risk': material_risk,
                'coating_risk': coating_risk,
                'h2s_risk': h2s_risk,
                'overall_integrity_risk': overall_risk
            }
        
        self.risk_scores['pipeline_integrity'] = pipeline_risk
        self.logger.info(f"Calculated integrity risk factors for {len(pipeline_risk)} pipelines")
        
        return pipeline_risk
    
    def calculate_corrosion_risk(self):
        """Calculate corrosion risk based on soil and CP data"""
        self.logger.info("Calculating corrosion risk factors...")
        
        corrosion_risk = {}
        
        # Soil-based corrosion risk
        if 'soil_pipe_potentials' in self.data:
            soil_data = self.data['soil_pipe_potentials']
            
            for _, row in soil_data.iterrows():
                location_id = row.get('location_id', f"Location_{row.name}")
                
                # pH risk (acidic soils are more corrosive)
                ph = pd.to_numeric(row.get('soil_ph', 7), errors='coerce')
                ph_risk = max(0, (7 - ph) * 2) if ph < 7 else max(0, (ph - 7) * 1.5)
                
                # Corrosivity factor
                corr_factor = pd.to_numeric(row.get('soil_corrosivity_factor', 5), errors='coerce')
                
                # Resistivity (lower resistivity = higher corrosion risk)
                resistivity = pd.to_numeric(row.get('resistivity_ohm_cm', 1500), errors='coerce')
                resistivity_risk = max(0, 10 - (resistivity / 200))
                
                # Moisture content
                moisture = pd.to_numeric(row.get('moisture_content_pct', 10), errors='coerce')
                moisture_risk = min(moisture / 3, 10)
                
                overall_soil_risk = np.mean([ph_risk, corr_factor, resistivity_risk, moisture_risk])
                
                corrosion_risk[location_id] = {
                    'ph_risk': ph_risk,
                    'corrosivity_factor_risk': corr_factor,
                    'resistivity_risk': resistivity_risk,
                    'moisture_risk': moisture_risk,
                    'overall_soil_corrosion_risk': overall_soil_risk
                }
        
        # CP survey data integration
        if 'pipeline_cp_survey' in self.data:
            # Note: CP data structure may vary, this is a general approach
            cp_data = self.data['pipeline_cp_survey']
            self.logger.info(f"Integrating CP survey data with {len(cp_data)} records")
        
        self.risk_scores['corrosion'] = corrosion_risk
        self.logger.info(f"Calculated corrosion risk factors for {len(corrosion_risk)} locations")
        
        return corrosion_risk
    
    def calculate_gas_composition_risk(self):
        """Calculate risk factors based on gas composition"""
        self.logger.info("Calculating gas composition risk factors...")
        
        if 'gas_analysis_list' not in self.data:
            self.logger.warning("Gas analysis data not available for risk calculation")
            return {}
        
        ga_data = self.data['gas_analysis_list'].copy()
        
        gas_risk = {}
        
        for _, row in ga_data.iterrows():
            well_id = row.get('UWI', f"Well_{row.name}")
            
            # H2S risk
            h2s_pct = pd.to_numeric(row.get('H2S', 0), errors='coerce')
            h2s_risk = min(h2s_pct * 100, 10)  # Scale H2S percentage to 0-10 risk
            
            # CO2 risk
            co2_pct = pd.to_numeric(row.get('CO2', 0), errors='coerce')
            co2_risk = min(co2_pct * 20, 10)  # CO2 is less aggressive than H2S
            
            # Water content risk (if available)
            # High water content can lead to internal corrosion
            
            overall_gas_risk = np.mean([h2s_risk, co2_risk])
            
            gas_risk[well_id] = {
                'h2s_risk': h2s_risk,
                'co2_risk': co2_risk,
                'overall_gas_composition_risk': overall_gas_risk
            }
        
        self.risk_scores['gas_composition'] = gas_risk
        self.logger.info(f"Calculated gas composition risk factors for {len(gas_risk)} wells")
        
        return gas_risk
    
    def generate_comprehensive_risk_assessment(self):
        """Generate comprehensive risk assessment combining all factors"""
        self.logger.info("Generating comprehensive risk assessment...")
        
        # Calculate all individual risk components
        self.calculate_production_risk_factors()
        self.calculate_pipeline_integrity_risk()
        self.calculate_corrosion_risk()
        self.calculate_gas_composition_risk()
        
        # Combine risks for overall assessment
        comprehensive_risk = {}
        
        # Get all unique identifiers across datasets
        all_identifiers = set()
        for risk_category in self.risk_scores.values():
            all_identifiers.update(risk_category.keys())
        
        for identifier in all_identifiers:
            risk_components = {}
            
            # Collect risks for this identifier
            for category, risks in self.risk_scores.items():
                if identifier in risks:
                    risk_components[category] = risks[identifier]
            
            # Calculate weighted overall risk
            if risk_components:
                risk_values = []
                for category, risk_data in risk_components.items():
                    if isinstance(risk_data, dict):
                        # Get overall risk from each category
                        overall_key = [k for k in risk_data.keys() if 'overall' in k.lower()]
                        if overall_key:
                            risk_values.append(risk_data[overall_key[0]])
                
                if risk_values:
                    overall_risk = np.mean(risk_values)
                    risk_level = self._categorize_risk_level(overall_risk)
                    
                    comprehensive_risk[identifier] = {
                        'overall_risk_score': overall_risk,
                        'risk_level': risk_level,
                        'components': risk_components
                    }
        
        self.analysis_results['comprehensive_assessment'] = comprehensive_risk
        self.logger.info(f"Generated comprehensive risk assessment for {len(comprehensive_risk)} assets")
        
        return comprehensive_risk
    
    def _categorize_risk_level(self, risk_score):
        """Categorize numeric risk score into risk level"""
        if risk_score < 2:
            return "Low"
        elif risk_score < 5:
            return "Medium"
        elif risk_score < 7.5:
            return "High"
        else:
            return "Critical"
    
    def generate_report(self, output_file=None):
        """Generate risk assessment report"""
        if output_file is None:
            output_file = f"alphabow_risk_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        self.logger.info(f"Generating risk assessment report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("ALPHABOW ENERGY LTD. - PIPELINE RISK ASSESSMENT REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            for file_type, data in self.data.items():
                f.write(f"{file_type}: {len(data)} records\n")
            f.write("\n")
            
            # Risk summary
            if 'comprehensive_assessment' in self.analysis_results:
                comprehensive = self.analysis_results['comprehensive_assessment']
                
                # Risk level distribution
                risk_levels = [asset['risk_level'] for asset in comprehensive.values()]
                risk_distribution = pd.Series(risk_levels).value_counts()
                
                f.write("RISK LEVEL DISTRIBUTION\n")
                f.write("-" * 25 + "\n")
                for level, count in risk_distribution.items():
                    f.write(f"{level}: {count} assets\n")
                f.write("\n")
                
                # Top 10 highest risk assets
                sorted_assets = sorted(comprehensive.items(), 
                                     key=lambda x: x[1]['overall_risk_score'], 
                                     reverse=True)[:10]
                
                f.write("TOP 10 HIGHEST RISK ASSETS\n")
                f.write("-" * 30 + "\n")
                for i, (asset_id, risk_data) in enumerate(sorted_assets, 1):
                    f.write(f"{i:2d}. {asset_id}: {risk_data['overall_risk_score']:.2f} ({risk_data['risk_level']})\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Prioritize inspection and maintenance for Critical and High risk assets\n")
            f.write("2. Implement enhanced monitoring for assets with high corrosion risk\n")
            f.write("3. Review production optimization for wells with high variability\n")
            f.write("4. Consider cathodic protection upgrades for high soil corrosivity areas\n")
            f.write("5. Implement gas composition monitoring for H2S/CO2 risk management\n")
        
        self.logger.info(f"Risk assessment report saved to {output_file}")
        return output_file
    
    def run_full_assessment(self):
        """Run complete risk assessment workflow"""
        self.logger.info("Starting Alphabow Risk Assessment V3...")
        
        try:
            # Load and validate data
            self.load_data_files()
            self.validate_data()
            
            # Generate comprehensive risk assessment
            self.generate_comprehensive_risk_assessment()
            
            # Generate report
            report_file = self.generate_report()
            
            self.logger.info("Risk assessment completed successfully!")
            self.logger.info(f"Report generated: {report_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            return False


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Alphabow Pipeline Risk Assessment V3')
    parser.add_argument('--data-dir', default='alphabow_files', 
                       help='Directory containing input data files (default: alphabow_files)')
    parser.add_argument('--output-dir', default='.', 
                       help='Directory for output files (default: current directory)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Change to output directory
    if args.output_dir != '.':
        os.chdir(args.output_dir)
    
    # Run assessment
    assessment = AlphabowRiskAssessment(data_directory=args.data_dir)
    success = assessment.run_full_assessment()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()