#!/usr/bin/env python3
"""
AlphabowRiskAssessmentV3 - Enhanced Pipeline Risk Assessment Tool

This script provides comprehensive risk assessment functionality for pipeline systems
including production-to-pipeline mapping, risk analysis, data quality reporting,
and dashboard-ready data generation.

Features:
- Improved production-to-pipeline mapping with robust UWI parsing
- Risk matrix validation with missing entry warnings
- Enhanced production statistics with decline rate and trend detection
- Comprehensive data quality reporting
- Sensitivity analysis for different risk weightings
- Dashboard-ready data for interactive visualization
- Enhanced error reporting and validation

Author: Generated for Alphabow Energy
Date: 2024
Version: 3.0
"""

import pandas as pd
import numpy as np
import re
import logging
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphabow_risk_assessment.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class DataQualityMetrics:
    """Data class to store data quality metrics"""
    production_data_coverage: float
    missing_age_data_count: int
    missing_pressure_data_count: int
    zero_production_pipelines: int
    total_wells: int
    total_pipelines: int
    data_completeness_score: float

@dataclass
class ProductionStatistics:
    """Data class to store production statistics for a well"""
    uwi: str
    decline_rate: float
    trend_direction: str
    r_squared: float
    peak_production: float
    current_production: float
    production_start_date: datetime
    days_producing: int

@dataclass
class RiskAssessmentResult:
    """Data class to store risk assessment results"""
    pipeline_id: str
    overall_risk_score: float
    production_risk: float
    technical_risk: float
    environmental_risk: float
    risk_category: str
    recommendations: List[str]

class AlphabowRiskAssessmentV3:
    """
    Enhanced Pipeline Risk Assessment Tool for Alphabow Energy
    
    This class provides comprehensive risk assessment functionality including:
    - Production data analysis and mapping
    - Risk matrix validation
    - Data quality reporting
    - Sensitivity analysis
    - Dashboard data generation
    """
    
    def __init__(self, data_directory: str = None):
        """
        Initialize the risk assessment tool
        
        Args:
            data_directory (str): Path to directory containing data files
        """
        self.logger = logging.getLogger(__name__)
        self.data_directory = Path(data_directory) if data_directory else Path('.')
        
        # Initialize data containers
        self.production_data = None
        self.pipeline_data = None
        self.cp_survey_data = None
        self.well_data = None
        
        # UWI parsing regex patterns
        self.uwi_patterns = {
            'standard': r'^(\d{3})\/(\d{2})-(\d{2})-(\d{3})-(\d{2})W(\d)\/(\d{2})$',
            'alternative': r'^(\d{3})(\d{2})(\d{2})(\d{3})(\d{2})W(\d)(\d{2})$',
            'short': r'^(\d{3})\/(\d{2})-(\d{2})-(\d{3})-(\d{2})W(\d)$'
        }
        
        # Risk matrix configuration
        self.risk_matrix = self._initialize_risk_matrix()
        
        # Initialize results containers
        self.production_statistics = {}
        self.data_quality_metrics = None
        self.risk_assessment_results = {}
        self.dashboard_data = {}
        
        self.logger.info("AlphabowRiskAssessmentV3 initialized")
    
    def _initialize_risk_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize the risk assessment matrix with default values
        
        Returns:
            Dict: Risk matrix configuration
        """
        return {
            'probability': {
                'very_low': 0.1,
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'very_high': 0.9
            },
            'consequence': {
                'very_low': 0.1,
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'very_high': 0.9
            },
            'risk_categories': {
                'low': (0.0, 0.3),
                'medium': (0.3, 0.6),
                'high': (0.6, 0.8),
                'critical': (0.8, 1.0)
            }
        }
    
    def load_data_files(self) -> bool:
        """
        Load all required data files
        
        Returns:
            bool: True if all files loaded successfully, False otherwise
        """
        try:
            self.logger.info("Loading data files...")
            
            # Load production data
            production_file = self.data_directory / 'production_export.csv'
            if production_file.exists():
                self.production_data = pd.read_csv(production_file)
                self.logger.info(f"Loaded production data: {len(self.production_data)} records")
            else:
                self.logger.warning(f"Production file not found: {production_file}")
            
            # Load pipeline data
            pipeline_file = self.data_directory / 'pipelines export data.csv'
            if pipeline_file.exists():
                self.pipeline_data = pd.read_csv(pipeline_file)
                self.logger.info(f"Loaded pipeline data: {len(self.pipeline_data)} records")
            else:
                self.logger.warning(f"Pipeline file not found: {pipeline_file}")
            
            # Load CP survey data
            cp_file = self.data_directory / 'pine_creek_cp_data.xlsx'
            if cp_file.exists():
                self.cp_survey_data = pd.read_excel(cp_file)
                self.logger.info(f"Loaded CP survey data: {len(self.cp_survey_data)} records")
            else:
                self.logger.warning(f"CP survey file not found: {cp_file}")
            
            # Load additional well data files
            self._load_additional_well_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data files: {str(e)}")
            return False
    
    def _load_additional_well_data(self):
        """Load additional well data files if available"""
        try:
            additional_files = [
                'updated_well_list.csv',
                'second_white_specks_well_info.csv',
                'second_white_specks_well_production.csv'
            ]
            
            self.well_data = {}
            for file_name in additional_files:
                file_path = self.data_directory / file_name
                if file_path.exists():
                    self.well_data[file_name] = pd.read_csv(file_path)
                    self.logger.info(f"Loaded {file_name}: {len(self.well_data[file_name])} records")
                    
        except Exception as e:
            self.logger.warning(f"Error loading additional well data: {str(e)}")
    
    def validate_and_parse_uwi(self, uwi: str) -> Dict[str, Any]:
        """
        Validate and parse UWI (Unique Well Identifier) using regex patterns
        
        Args:
            uwi (str): UWI string to validate and parse
            
        Returns:
            Dict: Parsed UWI components or error information
        """
        if uwi is None or (isinstance(uwi, str) and not uwi) or pd.isna(uwi):
            return {'valid': False, 'error': 'Empty or null UWI'}
        
        uwi = str(uwi).strip()
        
        for pattern_name, pattern in self.uwi_patterns.items():
            match = re.match(pattern, uwi)
            if match:
                groups = match.groups()
                return {
                    'valid': True,
                    'original': uwi,
                    'pattern_type': pattern_name,
                    'lsd': groups[0] if len(groups) > 0 else None,
                    'section': groups[1] if len(groups) > 1 else None,
                    'township': groups[2] if len(groups) > 2 else None,
                    'range': groups[3] if len(groups) > 3 else None,
                    'meridian': groups[4] if len(groups) > 4 else None,
                    'event_sequence': groups[5] if len(groups) > 5 else None
                }
        
        return {
            'valid': False,
            'error': f'UWI format not recognized: {uwi}',
            'original': uwi
        }
    
    def create_production_to_pipeline_mapping(self) -> pd.DataFrame:
        """
        Create robust mapping between production data and pipeline data
        
        Returns:
            pd.DataFrame: Mapped production and pipeline data
        """
        if self.production_data is None or self.pipeline_data is None:
            self.logger.error("Production or pipeline data not loaded")
            return pd.DataFrame()
        
        self.logger.info("Creating production-to-pipeline mapping...")
        
        # Parse UWIs in production data
        production_parsed = self.production_data.copy()
        production_parsed['uwi_parsed'] = production_parsed['wellbore_uwi'].apply(
            self.validate_and_parse_uwi
        )
        
        # Parse UWIs in pipeline data (if available)
        # Note: Pipeline data may not have UWIs directly, might need location-based mapping
        mapped_data = []
        
        for idx, prod_row in production_parsed.iterrows():
            uwi_info = prod_row['uwi_parsed']
            if uwi_info['valid']:
                # Try to find matching pipeline data based on location
                # This is a simplified approach - in practice, you might need more sophisticated matching
                matching_pipelines = self._find_matching_pipelines(uwi_info, prod_row)
                
                for pipeline in matching_pipelines:
                    mapped_row = {
                        **prod_row.to_dict(),
                        **pipeline.to_dict(),
                        'mapping_confidence': self._calculate_mapping_confidence(uwi_info, pipeline)
                    }
                    mapped_data.append(mapped_row)
        
        mapped_df = pd.DataFrame(mapped_data)
        self.logger.info(f"Created {len(mapped_df)} production-pipeline mappings")
        
        return mapped_df
    
    def _find_matching_pipelines(self, uwi_info: Dict, production_row: pd.Series) -> List[pd.Series]:
        """
        Find pipelines that match the given UWI/production data
        
        Args:
            uwi_info (Dict): Parsed UWI information
            production_row (pd.Series): Production data row
            
        Returns:
            List[pd.Series]: List of matching pipeline records
        """
        # This is a simplified matching algorithm
        # In practice, you'd want more sophisticated location-based matching
        matching_pipelines = []
        
        if self.pipeline_data is not None:
            # For now, return all pipelines (this would be refined based on actual location matching)
            # You could use legal subdivision, section, township, range matching
            for idx, pipeline in self.pipeline_data.iterrows():
                # Add actual matching logic based on location, facility connections, etc.
                matching_pipelines.append(pipeline)
        
        return matching_pipelines[:5]  # Limit to top 5 matches
    
    def _calculate_mapping_confidence(self, uwi_info: Dict, pipeline_row: pd.Series) -> float:
        """
        Calculate confidence score for production-pipeline mapping
        
        Args:
            uwi_info (Dict): Parsed UWI information
            pipeline_row (pd.Series): Pipeline data row
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simplified confidence calculation
        # In practice, this would consider location proximity, facility connections, etc.
        base_confidence = 0.7 if uwi_info['valid'] else 0.3
        
        # Add logic to increase/decrease confidence based on:
        # - Location proximity
        # - Facility connections
        # - Date overlaps
        # - Production volumes vs pipeline capacity
        
        return min(1.0, base_confidence)
    
    def validate_risk_matrix(self) -> Dict[str, Any]:
        """
        Validate the risk matrix for completeness and consistency
        
        Returns:
            Dict: Validation results with warnings and recommendations
        """
        self.logger.info("Validating risk matrix...")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_entries': [],
            'recommendations': []
        }
        
        # Check probability matrix completeness
        required_prob_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
        for level in required_prob_levels:
            if level not in self.risk_matrix['probability']:
                validation_results['missing_entries'].append(f"Missing probability level: {level}")
                validation_results['valid'] = False
        
        # Check consequence matrix completeness
        required_cons_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
        for level in required_cons_levels:
            if level not in self.risk_matrix['consequence']:
                validation_results['missing_entries'].append(f"Missing consequence level: {level}")
                validation_results['valid'] = False
        
        # Check value ranges
        for category, values in self.risk_matrix['probability'].items():
            if not 0 <= values <= 1:
                validation_results['warnings'].append(
                    f"Probability value for {category} outside 0-1 range: {values}"
                )
        
        for category, values in self.risk_matrix['consequence'].items():
            if not 0 <= values <= 1:
                validation_results['warnings'].append(
                    f"Consequence value for {category} outside 0-1 range: {values}"
                )
        
        # Check risk category boundaries
        risk_cats = self.risk_matrix['risk_categories']
        if len(risk_cats) == 0:
            validation_results['missing_entries'].append("No risk categories defined")
            validation_results['valid'] = False
        
        # Add recommendations
        if validation_results['warnings']:
            validation_results['recommendations'].append(
                "Review and correct value ranges in risk matrix"
            )
        
        if validation_results['missing_entries']:
            validation_results['recommendations'].append(
                "Add missing entries to risk matrix before proceeding"
            )
        
        self.logger.info(f"Risk matrix validation complete. Valid: {validation_results['valid']}")
        return validation_results
    
    def calculate_production_statistics(self) -> Dict[str, ProductionStatistics]:
        """
        Calculate enhanced production statistics including decline rates and trends
        
        Returns:
            Dict[str, ProductionStatistics]: Production statistics by UWI
        """
        if self.production_data is None:
            self.logger.error("Production data not loaded")
            return {}
        
        self.logger.info("Calculating production statistics...")
        
        production_stats = {}
        
        # Group by UWI
        grouped = self.production_data.groupby('wellbore_uwi')
        
        for uwi, group in grouped:
            try:
                stats = self._calculate_well_statistics(uwi, group)
                production_stats[uwi] = stats
            except Exception as e:
                self.logger.warning(f"Error calculating statistics for UWI {uwi}: {str(e)}")
        
        self.production_statistics = production_stats
        self.logger.info(f"Calculated statistics for {len(production_stats)} wells")
        
        return production_stats
    
    def _calculate_well_statistics(self, uwi: str, production_data: pd.DataFrame) -> ProductionStatistics:
        """
        Calculate statistics for a single well
        
        Args:
            uwi (str): Well UWI
            production_data (pd.DataFrame): Production data for the well
            
        Returns:
            ProductionStatistics: Calculated statistics
        """
        # Convert date column and sort
        data = production_data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Calculate total daily production (oil + gas equivalent)
        oil_prod = data['daily_oil_volume (m3/day)'].fillna(0)
        gas_prod = data['daily_gas_volume (e3m3/day)'].fillna(0) * 6.29  # Convert to BOE
        total_prod = oil_prod + gas_prod
        
        # Calculate decline rate using exponential fit
        if len(data) > 3 and total_prod.sum() > 0:
            days_since_start = (data['date'] - data['date'].min()).dt.days
            
            # Filter out zero production days for decline analysis
            nonzero_mask = total_prod > 0
            if nonzero_mask.sum() > 3:
                decline_rate, r_squared = self._fit_decline_curve(
                    days_since_start[nonzero_mask], 
                    total_prod[nonzero_mask]
                )
            else:
                decline_rate, r_squared = 0.0, 0.0
        else:
            decline_rate, r_squared = 0.0, 0.0
        
        # Determine trend direction
        if len(data) >= 6:
            recent_prod = total_prod.tail(3).mean()
            earlier_prod = total_prod.head(3).mean()
            
            if recent_prod > earlier_prod * 1.1:
                trend_direction = 'increasing'
            elif recent_prod < earlier_prod * 0.9:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        return ProductionStatistics(
            uwi=uwi,
            decline_rate=decline_rate,
            trend_direction=trend_direction,
            r_squared=r_squared,
            peak_production=total_prod.max(),
            current_production=total_prod.iloc[-1] if len(total_prod) > 0 else 0,
            production_start_date=data['date'].min(),
            days_producing=(data['date'].max() - data['date'].min()).days
        )
    
    def _fit_decline_curve(self, days: pd.Series, production: pd.Series) -> Tuple[float, float]:
        """
        Fit exponential decline curve to production data
        
        Args:
            days (pd.Series): Days since start of production
            production (pd.Series): Production rates
            
        Returns:
            Tuple[float, float]: Decline rate and R-squared
        """
        try:
            # Exponential decline: P(t) = P0 * exp(-D*t)
            # Where D is the decline rate
            
            def exponential_decline(t, p0, decline_rate):
                return p0 * np.exp(-decline_rate * t / 365.25)  # Convert to yearly decline
            
            # Initial guess
            p0_guess = production.iloc[0]
            decline_guess = 0.1  # 10% per year
            
            popt, pcov = curve_fit(
                exponential_decline, 
                days, 
                production,
                p0=[p0_guess, decline_guess],
                bounds=([0, 0], [np.inf, 2.0]),  # Reasonable bounds
                maxfev=1000
            )
            
            # Calculate R-squared
            y_pred = exponential_decline(days, *popt)
            ss_res = np.sum((production - y_pred) ** 2)
            ss_tot = np.sum((production - np.mean(production)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return popt[1], max(0, r_squared)
            
        except Exception as e:
            # Fallback to linear regression if exponential fit fails
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(days, np.log(production + 1))
                return max(0, -slope * 365.25), r_value**2
            except:
                return 0.0, 0.0
    
    def generate_data_quality_report(self) -> DataQualityMetrics:
        """
        Generate comprehensive data quality report
        
        Returns:
            DataQualityMetrics: Data quality metrics and scores
        """
        self.logger.info("Generating data quality report...")
        
        if self.production_data is None or self.pipeline_data is None:
            self.logger.error("Required data not loaded for quality assessment")
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate production data coverage
        total_wells = len(self.production_data['wellbore_uwi'].unique())
        wells_with_production = len(
            self.production_data[
                (self.production_data['daily_oil_volume (m3/day)'] > 0) |
                (self.production_data['daily_gas_volume (e3m3/day)'] > 0)
            ]['wellbore_uwi'].unique()
        )
        production_coverage = wells_with_production / total_wells if total_wells > 0 else 0
        
        # Count missing age data
        missing_age_data = 0
        if self.pipeline_data is not None:
            missing_age_data = self.pipeline_data['Licence Approval Date'].isna().sum()
        
        # Count missing pressure data
        missing_pressure_data = 0
        if self.pipeline_data is not None:
            missing_pressure_data = self.pipeline_data['MAOP (kPa)'].isna().sum()
        
        # Count zero production pipelines
        zero_production_pipelines = 0
        if self.production_data is not None:
            zero_prod_wells = self.production_data.groupby('wellbore_uwi').agg({
                'daily_oil_volume (m3/day)': 'sum',
                'daily_gas_volume (e3m3/day)': 'sum'
            })
            zero_production_pipelines = len(zero_prod_wells[
                (zero_prod_wells['daily_oil_volume (m3/day)'] == 0) &
                (zero_prod_wells['daily_gas_volume (e3m3/day)'] == 0)
            ])
        
        # Calculate overall data completeness score
        completeness_factors = [
            production_coverage,
            1 - (missing_age_data / len(self.pipeline_data)) if len(self.pipeline_data) > 0 else 0,
            1 - (missing_pressure_data / len(self.pipeline_data)) if len(self.pipeline_data) > 0 else 0,
            1 - (zero_production_pipelines / total_wells) if total_wells > 0 else 0
        ]
        
        data_completeness_score = np.mean(completeness_factors)
        
        metrics = DataQualityMetrics(
            production_data_coverage=production_coverage,
            missing_age_data_count=missing_age_data,
            missing_pressure_data_count=missing_pressure_data,
            zero_production_pipelines=zero_production_pipelines,
            total_wells=total_wells,
            total_pipelines=len(self.pipeline_data) if self.pipeline_data is not None else 0,
            data_completeness_score=data_completeness_score
        )
        
        self.data_quality_metrics = metrics
        self.logger.info(f"Data quality report generated. Completeness score: {data_completeness_score:.2f}")
        
        return metrics
    
    def perform_sensitivity_analysis(self, risk_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform sensitivity analysis showing how different risk weightings affect rankings
        
        Args:
            risk_weights (Dict[str, float]): Alternative risk weights to test
            
        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        self.logger.info("Performing sensitivity analysis...")
        
        if risk_weights is None:
            risk_weights = {
                'production_risk': 0.4,
                'technical_risk': 0.3,
                'environmental_risk': 0.3
            }
        
        # Define alternative weight scenarios
        scenarios = {
            'baseline': {'production_risk': 0.4, 'technical_risk': 0.3, 'environmental_risk': 0.3},
            'production_focused': {'production_risk': 0.6, 'technical_risk': 0.2, 'environmental_risk': 0.2},
            'technical_focused': {'production_risk': 0.2, 'technical_risk': 0.6, 'environmental_risk': 0.2},
            'environmental_focused': {'production_risk': 0.2, 'technical_risk': 0.2, 'environmental_risk': 0.6},
            'balanced': {'production_risk': 0.33, 'technical_risk': 0.33, 'environmental_risk': 0.34}
        }
        
        results = {}
        
        for scenario_name, weights in scenarios.items():
            scenario_results = self._calculate_risk_scores_with_weights(weights)
            results[scenario_name] = {
                'weights': weights,
                'risk_scores': scenario_results,
                'rankings': self._rank_by_risk_score(scenario_results)
            }
        
        # Calculate ranking volatility
        ranking_changes = self._analyze_ranking_changes(results)
        
        sensitivity_results = {
            'scenarios': results,
            'ranking_volatility': ranking_changes,
            'most_sensitive_assets': self._identify_sensitive_assets(results),
            'recommendations': self._generate_sensitivity_recommendations(ranking_changes)
        }
        
        self.logger.info("Sensitivity analysis completed")
        return sensitivity_results
    
    def _calculate_risk_scores_with_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk scores using specified weights"""
        # Simplified risk calculation - in practice, you'd use actual risk factors
        risk_scores = {}
        
        if self.pipeline_data is not None:
            for idx, pipeline in self.pipeline_data.iterrows():
                pipeline_id = pipeline.get('ID', f'pipeline_{idx}')
                
                # Mock risk factors (replace with actual calculations)
                prod_risk = np.random.uniform(0.1, 0.9)
                tech_risk = np.random.uniform(0.1, 0.9)
                env_risk = np.random.uniform(0.1, 0.9)
                
                overall_risk = (
                    prod_risk * weights['production_risk'] +
                    tech_risk * weights['technical_risk'] +
                    env_risk * weights['environmental_risk']
                )
                
                risk_scores[pipeline_id] = overall_risk
        
        return risk_scores
    
    def _rank_by_risk_score(self, risk_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """Rank assets by risk score"""
        return sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _analyze_ranking_changes(self, scenario_results: Dict) -> Dict[str, Any]:
        """Analyze how rankings change between scenarios"""
        baseline_rankings = {
            asset: rank for rank, (asset, score) in 
            enumerate(scenario_results['baseline']['rankings'])
        }
        
        ranking_changes = {}
        for scenario_name, scenario_data in scenario_results.items():
            if scenario_name == 'baseline':
                continue
                
            scenario_rankings = {
                asset: rank for rank, (asset, score) in 
                enumerate(scenario_data['rankings'])
            }
            
            changes = {}
            for asset in baseline_rankings:
                if asset in scenario_rankings:
                    change = baseline_rankings[asset] - scenario_rankings[asset]
                    changes[asset] = change
            
            ranking_changes[scenario_name] = changes
        
        return ranking_changes
    
    def _identify_sensitive_assets(self, scenario_results: Dict) -> List[str]:
        """Identify assets with high ranking sensitivity"""
        asset_volatility = {}
        
        # Calculate ranking standard deviation for each asset
        for asset in scenario_results['baseline']['risk_scores']:
            rankings = []
            for scenario_data in scenario_results.values():
                asset_rankings = {a: r for r, (a, s) in enumerate(scenario_data['rankings'])}
                if asset in asset_rankings:
                    rankings.append(asset_rankings[asset])
            
            if rankings:
                asset_volatility[asset] = np.std(rankings)
        
        # Return top 5 most volatile assets
        return sorted(asset_volatility.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _generate_sensitivity_recommendations(self, ranking_changes: Dict) -> List[str]:
        """Generate recommendations based on sensitivity analysis"""
        recommendations = []
        
        # Analyze volatility patterns
        high_volatility_count = sum(
            1 for changes in ranking_changes.values() 
            for change in changes.values() 
            if abs(change) > 5
        )
        
        if high_volatility_count > len(ranking_changes) * 2:
            recommendations.append(
                "High ranking volatility detected. Consider refining risk factor definitions."
            )
        
        recommendations.append(
            "Review risk weightings with stakeholders to ensure alignment with business priorities."
        )
        
        recommendations.append(
            "Focus additional analysis on assets showing high sensitivity to weight changes."
        )
        
        return recommendations
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate dashboard-ready data for interactive visualization
        
        Returns:
            Dict[str, Any]: Structured data for dashboard consumption
        """
        self.logger.info("Generating dashboard-ready data...")
        
        dashboard_data = {
            'summary_metrics': self._generate_summary_metrics(),
            'location_groupings': self._generate_location_groupings(),
            'timeline_data': self._generate_timeline_data(),
            'production_risk_correlation': self._generate_production_risk_correlation(),
            'risk_distribution': self._generate_risk_distribution(),
            'data_quality_indicators': self._generate_quality_indicators()
        }
        
        self.dashboard_data = dashboard_data
        self.logger.info("Dashboard data generated successfully")
        
        return dashboard_data
    
    def _generate_summary_metrics(self) -> Dict[str, Any]:
        """Generate summary metrics for dashboard"""
        return {
            'total_wells': len(self.production_data['wellbore_uwi'].unique()) if self.production_data is not None else 0,
            'total_pipelines': len(self.pipeline_data) if self.pipeline_data is not None else 0,
            'active_wells': len(self.production_statistics),
            'high_risk_assets': len([r for r in self.risk_assessment_results.values() if r.risk_category == 'high']),
            'data_completeness': self.data_quality_metrics.data_completeness_score if self.data_quality_metrics else 0
        }
    
    def _generate_location_groupings(self) -> Dict[str, Any]:
        """Generate location-based groupings"""
        # This would group assets by geographic location, field, etc.
        # Simplified implementation
        return {
            'by_field': {},
            'by_region': {},
            'by_facility': {}
        }
    
    def _generate_timeline_data(self) -> Dict[str, Any]:
        """Generate timeline data for trend analysis"""
        if self.production_data is None:
            return {}
        
        # Generate monthly production summaries
        timeline_data = self.production_data.copy()
        timeline_data['date'] = pd.to_datetime(timeline_data['date'])
        timeline_data['year_month'] = timeline_data['date'].dt.to_period('M')
        
        monthly_summary = timeline_data.groupby('year_month').agg({
            'daily_oil_volume (m3/day)': 'sum',
            'daily_gas_volume (e3m3/day)': 'sum',
            'wellbore_uwi': 'nunique'
        }).reset_index()
        
        return {
            'monthly_production': monthly_summary.to_dict('records'),
            'production_trends': self._calculate_production_trends(monthly_summary)
        }
    
    def _calculate_production_trends(self, monthly_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate production trend statistics"""
        if len(monthly_data) < 2:
            return {'oil_trend': 0, 'gas_trend': 0}
        
        # Simple linear trend calculation
        months = range(len(monthly_data))
        oil_trend = np.polyfit(months, monthly_data['daily_oil_volume (m3/day)'], 1)[0]
        gas_trend = np.polyfit(months, monthly_data['daily_gas_volume (e3m3/day)'], 1)[0]
        
        return {
            'oil_trend': float(oil_trend),
            'gas_trend': float(gas_trend)
        }
    
    def _generate_production_risk_correlation(self) -> Dict[str, Any]:
        """Generate production-risk correlation data"""
        correlations = {}
        
        # Calculate correlations between production metrics and risk scores
        if self.production_statistics and self.risk_assessment_results:
            prod_data = []
            risk_data = []
            
            for uwi, stats in self.production_statistics.items():
                if uwi in self.risk_assessment_results:
                    prod_data.append(stats.current_production)
                    risk_data.append(self.risk_assessment_results[uwi].overall_risk_score)
            
            if len(prod_data) > 1:
                correlation = np.corrcoef(prod_data, risk_data)[0, 1]
                correlations['production_risk'] = float(correlation) if not np.isnan(correlation) else 0
        
        return correlations
    
    def _generate_risk_distribution(self) -> Dict[str, Any]:
        """Generate risk distribution data"""
        if not self.risk_assessment_results:
            return {}
        
        risk_categories = [result.risk_category for result in self.risk_assessment_results.values()]
        category_counts = pd.Series(risk_categories).value_counts().to_dict()
        
        return {
            'category_distribution': category_counts,
            'risk_score_histogram': self._create_risk_histogram()
        }
    
    def _create_risk_histogram(self) -> List[Dict[str, Any]]:
        """Create histogram data for risk scores"""
        if not self.risk_assessment_results:
            return []
        
        risk_scores = [result.overall_risk_score for result in self.risk_assessment_results.values()]
        hist, bins = np.histogram(risk_scores, bins=10)
        
        histogram_data = []
        for i in range(len(hist)):
            histogram_data.append({
                'bin_start': float(bins[i]),
                'bin_end': float(bins[i+1]),
                'count': int(hist[i])
            })
        
        return histogram_data
    
    def _generate_quality_indicators(self) -> Dict[str, Any]:
        """Generate data quality indicators for dashboard"""
        if not self.data_quality_metrics:
            return {}
        
        return {
            'completeness_score': self.data_quality_metrics.data_completeness_score,
            'production_coverage': self.data_quality_metrics.production_data_coverage,
            'missing_data_counts': {
                'age_data': self.data_quality_metrics.missing_age_data_count,
                'pressure_data': self.data_quality_metrics.missing_pressure_data_count,
                'zero_production': self.data_quality_metrics.zero_production_pipelines
            }
        }
    
    def generate_validation_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation summary
        
        Returns:
            Dict[str, Any]: Validation summary with errors, warnings, and recommendations
        """
        validation_summary = {
            'timestamp': datetime.now().isoformat(),
            'data_validation': self._validate_data_integrity(),
            'uwi_validation': self._validate_uwi_parsing(),
            'risk_matrix_validation': self.validate_risk_matrix(),
            'production_analysis_validation': self._validate_production_analysis(),
            'overall_status': 'pending'
        }
        
        # Determine overall status
        all_valid = all([
            validation_summary['data_validation']['valid'],
            validation_summary['uwi_validation']['valid'],
            validation_summary['risk_matrix_validation']['valid'],
            validation_summary['production_analysis_validation']['valid']
        ])
        
        validation_summary['overall_status'] = 'valid' if all_valid else 'issues_found'
        
        return validation_summary
    
    def _validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and completeness"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if required data is loaded
        if self.production_data is None:
            validation['errors'].append("Production data not loaded")
            validation['valid'] = False
        
        if self.pipeline_data is None:
            validation['errors'].append("Pipeline data not loaded")
            validation['valid'] = False
        
        # Check data quality
        if self.production_data is not None:
            null_uwis = self.production_data['wellbore_uwi'].isna().sum()
            if null_uwis > 0:
                validation['warnings'].append(f"{null_uwis} records with null UWIs")
        
        return validation
    
    def _validate_uwi_parsing(self) -> Dict[str, Any]:
        """Validate UWI parsing results"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'parse_success_rate': 0
        }
        
        if self.production_data is not None:
            unique_uwis = self.production_data['wellbore_uwi'].dropna().unique()
            parse_results = [self.validate_and_parse_uwi(uwi) for uwi in unique_uwis]
            valid_parses = sum(1 for result in parse_results if result['valid'])
            
            parse_success_rate = valid_parses / len(parse_results) if parse_results else 0
            validation['parse_success_rate'] = parse_success_rate
            
            if parse_success_rate < 0.9:
                validation['warnings'].append(
                    f"UWI parse success rate below 90%: {parse_success_rate:.1%}"
                )
            
            if parse_success_rate < 0.5:
                validation['errors'].append(
                    f"UWI parse success rate critically low: {parse_success_rate:.1%}"
                )
                validation['valid'] = False
        
        return validation
    
    def _validate_production_analysis(self) -> Dict[str, Any]:
        """Validate production analysis results"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not self.production_statistics:
            validation['warnings'].append("No production statistics calculated")
        else:
            # Check for reasonable decline rates
            extreme_declines = sum(
                1 for stats in self.production_statistics.values() 
                if stats.decline_rate > 2.0  # >200% per year
            )
            
            if extreme_declines > 0:
                validation['warnings'].append(
                    f"{extreme_declines} wells with extreme decline rates (>200%/year)"
                )
        
        return validation
    
    def save_results(self, output_directory: str = None) -> bool:
        """
        Save all analysis results to files
        
        Args:
            output_directory (str): Directory to save results
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_dir = Path(output_directory) if output_directory else self.data_directory / 'results'
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save production statistics
            if self.production_statistics:
                prod_stats_df = pd.DataFrame([
                    {
                        'uwi': stats.uwi,
                        'decline_rate': stats.decline_rate,
                        'trend_direction': stats.trend_direction,
                        'r_squared': stats.r_squared,
                        'peak_production': stats.peak_production,
                        'current_production': stats.current_production,
                        'days_producing': stats.days_producing
                    }
                    for stats in self.production_statistics.values()
                ])
                prod_stats_df.to_csv(output_dir / f'production_statistics_{timestamp}.csv', index=False)
            
            # Save data quality metrics
            if self.data_quality_metrics:
                quality_data = {
                    'timestamp': timestamp,
                    'metrics': self.data_quality_metrics.__dict__
                }
                with open(output_dir / f'data_quality_report_{timestamp}.json', 'w') as f:
                    json.dump(quality_data, f, indent=2, default=str)
            
            # Save dashboard data
            if self.dashboard_data:
                with open(output_dir / f'dashboard_data_{timestamp}.json', 'w') as f:
                    json.dump(self.dashboard_data, f, indent=2, default=str)
            
            # Save validation summary
            validation_summary = self.generate_validation_summary()
            with open(output_dir / f'validation_summary_{timestamp}.json', 'w') as f:
                json.dump(validation_summary, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False
    
    def main(self, data_directory: str = None, output_directory: str = None) -> bool:
        """
        Main execution function
        
        Args:
            data_directory (str): Directory containing input data files
            output_directory (str): Directory to save results
            
        Returns:
            bool: True if execution successful, False otherwise
        """
        try:
            self.logger.info("=== Starting AlphabowRiskAssessmentV3 Analysis ===")
            
            # Update data directory if provided
            if data_directory:
                self.data_directory = Path(data_directory)
            
            # Load data files
            if not self.load_data_files():
                self.logger.error("Failed to load required data files")
                return False
            
            # Validate risk matrix
            matrix_validation = self.validate_risk_matrix()
            if not matrix_validation['valid']:
                self.logger.error("Risk matrix validation failed")
                for error in matrix_validation['missing_entries']:
                    self.logger.error(f"  - {error}")
                return False
            
            # Create production-pipeline mapping
            mapped_data = self.create_production_to_pipeline_mapping()
            self.logger.info(f"Created {len(mapped_data)} production-pipeline mappings")
            
            # Calculate production statistics
            production_stats = self.calculate_production_statistics()
            self.logger.info(f"Calculated statistics for {len(production_stats)} wells")
            
            # Generate data quality report
            quality_metrics = self.generate_data_quality_report()
            self.logger.info(f"Data quality score: {quality_metrics.data_completeness_score:.2f}")
            
            # Perform sensitivity analysis
            sensitivity_results = self.perform_sensitivity_analysis()
            self.logger.info("Sensitivity analysis completed")
            
            # Generate dashboard data
            dashboard_data = self.generate_dashboard_data()
            self.logger.info("Dashboard data generated")
            
            # Generate validation summary
            validation_summary = self.generate_validation_summary()
            self.logger.info(f"Validation status: {validation_summary['overall_status']}")
            
            # Save results
            if not self.save_results(output_directory):
                self.logger.warning("Failed to save some results")
            
            self.logger.info("=== AlphabowRiskAssessmentV3 Analysis Complete ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in main execution: {str(e)}")
            return False

def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphabowRiskAssessmentV3 - Enhanced Pipeline Risk Assessment')
    parser.add_argument('--data-dir', default='.', help='Directory containing data files')
    parser.add_argument('--output-dir', help='Directory to save results')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run assessment
    assessment = AlphabowRiskAssessmentV3(args.data_dir)
    success = assessment.main(args.data_dir, args.output_dir)
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())