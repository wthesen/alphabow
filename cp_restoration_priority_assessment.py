#!/usr/bin/env python3
"""
URGENT: CP Restoration Priority Assessment Tool
Alphabow Energy Ltd.

CRITICAL TIMELINE UPDATE:
- Current Date: July 10, 2025
- Pipelines shut-in: Q4 2023 (18+ months ago)
- CP Systems inactive: 18+ months
- This represents EXTREME regulatory non-compliance and safety risk

This tool provides immediate prioritization for CP restoration based on:
1. EXTREME PRIORITY (Restore within 7 days)
2. HIGH PRIORITY (Restore within 30 days)
3. MEDIUM PRIORITY (Restore within 90 days)

Risk multipliers applied:
- CP inactive 18+ months: 4.0x multiplier (EXTREME)
- Steel material without CP: 2.5x multiplier
- High pressure (>500 kPa): 2.0x multiplier
- Water body crossing: 2.0x multiplier
- Age >20 years: 1.8x multiplier
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

class CPRestorationPriorityAssessment:
    def __init__(self):
        self.current_date = datetime(2025, 7, 10)  # Current assessment date
        self.cp_shutdown_date = datetime(2023, 10, 1)  # Q4 2023 shutdown
        self.cp_inactive_months = 18.25  # 18+ months inactive
        
        # Risk multipliers
        self.multipliers = {
            'cp_inactive_18_months': 4.0,
            'steel_without_cp': 2.5,
            'high_pressure': 2.0,  # >500 kPa
            'water_crossing': 2.0,
            'age_over_20_years': 1.8
        }
        
        # Priority thresholds
        self.priority_thresholds = {
            'extreme': 0.8,  # ≥80% risk score
            'high': 0.6,     # 60-79% risk score
            'medium': 0.4,   # 40-59% risk score
            'low': 0.0       # <40% risk score
        }
        
        self.results = None
        
    def load_pipeline_data(self, csv_file='pipelines export data.csv'):
        """Load pipeline data from CSV file"""
        try:
            self.df = pd.read_csv(csv_file)
            print(f"Loaded {len(self.df)} pipeline segments")
            return True
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_pipeline_age(self, licence_date):
        """Calculate pipeline age in years"""
        try:
            if pd.isna(licence_date):
                return 25  # Conservative estimate for missing data
            
            licence_dt = pd.to_datetime(licence_date)
            age_years = (self.current_date - licence_dt).days / 365.25
            return max(0, age_years)
        except:
            return 25  # Conservative estimate for invalid dates
    
    def is_water_crossing(self, from_location, to_location):
        """Determine if pipeline crosses water body based on location names"""
        # Simple heuristic - look for creek, river, stream indicators
        water_indicators = ['creek', 'river', 'stream', 'lake', 'pond', 'crossing']
        locations = f"{from_location} {to_location}".lower()
        return any(indicator in locations for indicator in water_indicators)
    
    def assess_corrosive_environment(self, substance, h2s_content):
        """Assess if pipeline is in corrosive environment"""
        # High corrosion risk factors
        if substance in ['Salt Water', 'Produced Water']:
            return True
        if h2s_content and h2s_content > 0:
            return True
        return False
    
    def calculate_base_risk_score(self, row):
        """Calculate base risk score (0-1) before multipliers"""
        risk_factors = []
        
        # Material factor
        if row['Pipe Material'] == 'Steel':
            risk_factors.append(0.3)  # Steel inherently higher risk
        else:
            risk_factors.append(0.1)  # Non-steel materials
        
        # Pressure factor
        maop = row.get('MAOP (kPa)', 0)
        if maop > 1000:
            risk_factors.append(0.25)
        elif maop > 500:
            risk_factors.append(0.15)
        else:
            risk_factors.append(0.05)
            
        # Age factor
        age = self.calculate_pipeline_age(row.get('Licence Issue Date'))
        if age > 25:
            risk_factors.append(0.2)
        elif age > 20:
            risk_factors.append(0.15)
        elif age > 15:
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.05)
            
        # Substance/environment factor
        if self.assess_corrosive_environment(row.get('Substance'), row.get('H2S Content')):
            risk_factors.append(0.15)
        else:
            risk_factors.append(0.05)
            
        # Water crossing factor
        if self.is_water_crossing(row.get('From Location', ''), row.get('To Location', '')):
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.02)
            
        return min(1.0, sum(risk_factors))
    
    def apply_risk_multipliers(self, base_score, row):
        """Apply risk multipliers to base score"""
        multiplier = 1.0
        applied_multipliers = []
        
        # CP inactive 18+ months (CRITICAL)
        multiplier *= self.multipliers['cp_inactive_18_months']
        applied_multipliers.append('CP inactive 18+ months (4.0x)')
        
        # Steel without CP
        if row['Pipe Material'] == 'Steel':
            multiplier *= self.multipliers['steel_without_cp']
            applied_multipliers.append('Steel without CP (2.5x)')
        
        # High pressure
        maop = row.get('MAOP (kPa)', 0)
        if maop > 500:
            multiplier *= self.multipliers['high_pressure']
            applied_multipliers.append('High pressure >500kPa (2.0x)')
        
        # Water crossing
        if self.is_water_crossing(row.get('From Location', ''), row.get('To Location', '')):
            multiplier *= self.multipliers['water_crossing']
            applied_multipliers.append('Water crossing (2.0x)')
        
        # Age over 20 years
        age = self.calculate_pipeline_age(row.get('Licence Issue Date'))
        if age > 20:
            multiplier *= self.multipliers['age_over_20_years']
            applied_multipliers.append('Age >20 years (1.8x)')
        
        final_score = min(1.0, base_score * multiplier)
        return final_score, multiplier, applied_multipliers
    
    def categorize_priority(self, risk_score):
        """Categorize priority based on risk score"""
        if risk_score >= self.priority_thresholds['extreme']:
            return 'EXTREME', 7  # 7 days
        elif risk_score >= self.priority_thresholds['high']:
            return 'HIGH', 30  # 30 days
        elif risk_score >= self.priority_thresholds['medium']:
            return 'MEDIUM', 90  # 90 days
        else:
            return 'LOW', 365  # 1 year
    
    def run_assessment(self):
        """Run the complete CP restoration priority assessment"""
        if self.df is None:
            print("Error: No pipeline data loaded")
            return False
        
        print("\n" + "="*80)
        print("URGENT: CP RESTORATION PRIORITY ASSESSMENT")
        print("Alphabow Energy Ltd.")
        print(f"Assessment Date: {self.current_date.strftime('%Y-%m-%d')}")
        print(f"CP Systems Inactive Since: Q4 2023 ({self.cp_inactive_months:.1f} months)")
        print("="*80)
        
        results = []
        
        for idx, row in self.df.iterrows():
            # Skip non-steel pipelines for CP assessment
            if row['Pipe Material'] != 'Steel':
                continue
                
            # Calculate risk scores
            base_score = self.calculate_base_risk_score(row)
            final_score, multiplier, applied_multipliers = self.apply_risk_multipliers(base_score, row)
            priority, restoration_days = self.categorize_priority(final_score)
            
            # Calculate pipeline age
            age = self.calculate_pipeline_age(row.get('Licence Issue Date'))
            
            result = {
                'Pipeline_ID': row['ID'],
                'Licence_No': row['Licence No.'],
                'Substance': row['Substance'],
                'MAOP_kPa': row.get('MAOP (kPa)', 0),
                'Material': row['Pipe Material'],
                'Length_m': row.get('Segment Length (m)', 0),
                'Age_Years': round(age, 1),
                'H2S_Content': row.get('H2S Content', 0),
                'From_Location': row.get('From Location', ''),
                'To_Location': row.get('To Location', ''),
                'Base_Risk_Score': round(base_score, 3),
                'Risk_Multiplier': round(multiplier, 2),
                'Final_Risk_Score': round(final_score, 3),
                'Priority': priority,
                'Restoration_Days': restoration_days,
                'Applied_Multipliers': '; '.join(applied_multipliers),
                'Water_Crossing': self.is_water_crossing(row.get('From Location', ''), row.get('To Location', '')),
                'Corrosive_Environment': self.assess_corrosive_environment(row.get('Substance'), row.get('H2S Content'))
            }
            results.append(result)
        
        self.results = pd.DataFrame(results)
        
        # Sort by risk score (highest first)
        self.results = self.results.sort_values('Final_Risk_Score', ascending=False)
        
        return True
    
    def generate_priority_action_list(self):
        """Generate Priority Action List - specific pipeline IDs requiring immediate CP restoration"""
        if self.results is None:
            return None
            
        print("\n" + "="*80)
        print("1. PRIORITY ACTION LIST - IMMEDIATE CP RESTORATION REQUIRED")
        print("="*80)
        
        extreme_priority = self.results[self.results['Priority'] == 'EXTREME']
        high_priority = self.results[self.results['Priority'] == 'HIGH']
        medium_priority = self.results[self.results['Priority'] == 'MEDIUM']
        
        print(f"\nEXTREME PRIORITY (Restore within 7 days): {len(extreme_priority)} pipelines")
        print("-" * 60)
        if len(extreme_priority) > 0:
            for _, row in extreme_priority.iterrows():
                print(f"• {row['Pipeline_ID']} (Licence {row['Licence_No']}) - Risk: {row['Final_Risk_Score']:.3f}")
                print(f"  MAOP: {row['MAOP_kPa']} kPa, Age: {row['Age_Years']} years, Length: {row['Length_m']} m")
                print(f"  Substance: {row['Substance']}, H2S: {row['H2S_Content']}")
                print(f"  Route: {row['From_Location']} → {row['To_Location']}")
                print(f"  Risk Factors: {row['Applied_Multipliers']}")
                print()
        
        print(f"\nHIGH PRIORITY (Restore within 30 days): {len(high_priority)} pipelines")
        print("-" * 60)
        if len(high_priority) > 0:
            for _, row in high_priority.head(10).iterrows():  # Show top 10
                print(f"• {row['Pipeline_ID']} (Licence {row['Licence_No']}) - Risk: {row['Final_Risk_Score']:.3f}")
                print(f"  MAOP: {row['MAOP_kPa']} kPa, Age: {row['Age_Years']} years")
                print()
        
        print(f"\nMEDIUM PRIORITY (Restore within 90 days): {len(medium_priority)} pipelines")
        print("-" * 60)
        if len(medium_priority) > 0:
            for _, row in medium_priority.head(5).iterrows():  # Show top 5
                print(f"• {row['Pipeline_ID']} (Licence {row['Licence_No']}) - Risk: {row['Final_Risk_Score']:.3f}")
        
        return {
            'extreme': extreme_priority,
            'high': high_priority,
            'medium': medium_priority
        }
    
    def generate_regulatory_compliance_report(self):
        """Generate Regulatory Compliance Report for AER notification requirements"""
        if self.results is None:
            return None
            
        print("\n" + "="*80)
        print("2. REGULATORY COMPLIANCE REPORT - AER NOTIFICATION REQUIREMENTS")
        print("="*80)
        
        total_steel_pipelines = len(self.results)
        extreme_count = len(self.results[self.results['Priority'] == 'EXTREME'])
        high_count = len(self.results[self.results['Priority'] == 'HIGH'])
        
        print(f"\nREGULATORY VIOLATIONS IDENTIFIED:")
        print(f"• Total Steel Pipelines Affected: {total_steel_pipelines}")
        print(f"• CP Systems Inactive: {self.cp_inactive_months:.1f} months (Q4 2023 - July 2025)")
        print(f"• EXTREME Priority Pipelines: {extreme_count}")
        print(f"• HIGH Priority Pipelines: {high_count}")
        
        print(f"\nDIRECTIVE 077 VIOLATIONS:")
        print("• Extended CP system outages exceeding acceptable maintenance windows")
        print("• Potential for accelerated external corrosion on steel pipelines")
        print("• Integrity management program failure to maintain CP protection")
        
        print(f"\nPIPELINE ACT SECTION 35 REQUIREMENTS:")
        print("• Conditions affecting pipeline safety: ✓ IDENTIFIED")
        print("• Environmental protection concerns: ✓ IDENTIFIED") 
        print("• Public safety implications: ✓ IDENTIFIED")
        
        print(f"\nIMMEDIATE ACTIONS REQUIRED:")
        print("1. AER Notification within 24 hours of this assessment")
        print("2. Emergency CP restoration plan submission within 48 hours")
        print("3. Immediate CP restoration on EXTREME priority pipelines (7 days)")
        print("4. Monthly progress reports to AER during restoration period")
        
        # Risk assessment by substance type
        substance_summary = self.results.groupby('Substance').agg({
            'Final_Risk_Score': ['count', 'mean', 'max'],
            'Priority': lambda x: (x == 'EXTREME').sum()
        }).round(3)
        
        print(f"\nRISK SUMMARY BY SUBSTANCE TYPE:")
        print(substance_summary)
        
        return substance_summary
    
    def generate_cost_timeline_matrix(self):
        """Generate Cost/Timeline Matrix for restoration resources"""
        if self.results is None:
            return None
            
        print("\n" + "="*80)
        print("3. COST/TIMELINE MATRIX - CP RESTORATION RESOURCES")
        print("="*80)
        
        # Cost estimates (conservative)
        costs = {
            'emergency_cp_restoration': 50000,  # $50k per pipeline
            'full_integrity_assessment': 25000,  # $25k per pipeline  
            'potential_abandonment': 100000,     # $100k per pipeline
            'regulatory_penalties': 500000       # $500k estimate
        }
        
        extreme_count = len(self.results[self.results['Priority'] == 'EXTREME'])
        high_count = len(self.results[self.results['Priority'] == 'HIGH'])
        medium_count = len(self.results[self.results['Priority'] == 'MEDIUM'])
        total_count = extreme_count + high_count + medium_count
        
        print(f"\nCP RESTORATION COST ESTIMATES:")
        print(f"• Emergency CP Restoration: ${costs['emergency_cp_restoration']:,} per pipeline")
        print(f"• Full Integrity Assessment: ${costs['full_integrity_assessment']:,} per pipeline")
        print(f"• Potential Abandonment: ${costs['potential_abandonment']:,} per pipeline")
        
        extreme_cost = extreme_count * (costs['emergency_cp_restoration'] + costs['full_integrity_assessment'])
        high_cost = high_count * (costs['emergency_cp_restoration'] + costs['full_integrity_assessment'])
        medium_cost = medium_count * costs['emergency_cp_restoration']
        total_technical_cost = extreme_cost + high_cost + medium_cost
        
        print(f"\nTOTAL COST BREAKDOWN:")
        print(f"• EXTREME Priority ({extreme_count} pipelines): ${extreme_cost:,}")
        print(f"• HIGH Priority ({high_count} pipelines): ${high_cost:,}")
        print(f"• MEDIUM Priority ({medium_count} pipelines): ${medium_cost:,}")
        print(f"• Regulatory Penalties/Remediation: ${costs['regulatory_penalties']:,}")
        print(f"• TOTAL ESTIMATED COST: ${total_technical_cost + costs['regulatory_penalties']:,}")
        
        print(f"\nRESTORATION TIMELINE:")
        print(f"• Days 1-7: EXTREME Priority restoration ({extreme_count} pipelines)")
        print(f"• Days 8-37: HIGH Priority restoration ({high_count} pipelines)")
        print(f"• Days 38-127: MEDIUM Priority restoration ({medium_count} pipelines)")
        print(f"• Total Restoration Period: ~4 months")
        
        return {
            'costs': costs,
            'counts': {'extreme': extreme_count, 'high': high_count, 'medium': medium_count},
            'total_cost': total_technical_cost + costs['regulatory_penalties']
        }
    
    def generate_emergency_response_plan(self):
        """Generate Emergency Response Plan for potential failures"""
        if self.results is None:
            return None
            
        print("\n" + "="*80)
        print("4. EMERGENCY RESPONSE PLAN - IMMEDIATE ACTIONS")
        print("="*80)
        
        extreme_pipelines = self.results[self.results['Priority'] == 'EXTREME']
        water_crossings = self.results[self.results['Water_Crossing'] == True]
        h2s_pipelines = self.results[self.results['H2S_Content'] > 0]
        
        print(f"\nIMMEDIATE RISK MITIGATION (Next 24-48 hours):")
        print(f"1. Emergency shutdown capability verification for {len(extreme_pipelines)} EXTREME priority pipelines")
        print(f"2. Environmental monitoring at {len(water_crossings)} water crossing locations")
        print(f"3. H2S detection systems verification for {len(h2s_pipelines)} sour pipelines")
        print(f"4. Emergency response team mobilization")
        
        print(f"\nSPILL RISK ASSESSMENT:")
        high_risk_spill = extreme_pipelines[
            (extreme_pipelines['MAOP_kPa'] > 500) & 
            (extreme_pipelines['Water_Crossing'] == True)
        ]
        print(f"• High-pressure water crossing pipelines: {len(high_risk_spill)} (CRITICAL)")
        print(f"• Unknown pipeline condition after 18+ months without CP")
        print(f"• Increased failure probability due to accelerated corrosion")
        
        print(f"\nEMERGENCY CONTACT PROTOCOL:")
        print("• AER Emergency Line: 1-800-222-6514")
        print("• Alphabow Emergency Response: [CONTACT INFO NEEDED]")
        print("• Environmental Authorities: [CONTACT INFO NEEDED]")
        
        print(f"\nCONTINGENCY MEASURES:")
        print("• Pressure reduction on high-risk pipelines")
        print("• Increased patrol frequency (daily for EXTREME priority)")
        print("• Leak detection system enhancement")
        print("• Emergency shutdown procedure validation")
        
        return {
            'extreme_count': len(extreme_pipelines),
            'water_crossings': len(water_crossings),
            'h2s_pipelines': len(h2s_pipelines),
            'high_risk_spill': len(high_risk_spill)
        }
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the assessment"""
        if self.results is None:
            return None
            
        print("\n" + "="*80)
        print("5. ASSESSMENT SUMMARY STATISTICS")
        print("="*80)
        
        priority_counts = self.results['Priority'].value_counts()
        
        print(f"\nRISK DISTRIBUTION:")
        for priority in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            count = priority_counts.get(priority, 0)
            percentage = (count / len(self.results)) * 100
            print(f"• {priority}: {count} pipelines ({percentage:.1f}%)")
        
        print(f"\nRISK SCORE STATISTICS:")
        print(f"• Mean Risk Score: {self.results['Final_Risk_Score'].mean():.3f}")
        print(f"• Maximum Risk Score: {self.results['Final_Risk_Score'].max():.3f}")
        print(f"• Median Risk Score: {self.results['Final_Risk_Score'].median():.3f}")
        
        # Age analysis
        old_pipelines = self.results[self.results['Age_Years'] > 20]
        print(f"\nAGE ANALYSIS:")
        print(f"• Pipelines >20 years old: {len(old_pipelines)} ({(len(old_pipelines)/len(self.results)*100):.1f}%)")
        print(f"• Average age: {self.results['Age_Years'].mean():.1f} years")
        
        return priority_counts
    
    def save_results_to_csv(self, filename='cp_restoration_priority_results.csv'):
        """Save detailed results to CSV file"""
        if self.results is None:
            return False
            
        try:
            self.results.to_csv(filename, index=False)
            print(f"\nDetailed results saved to: {filename}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

def main():
    """Main function to run the CP restoration priority assessment"""
    print("URGENT: CP Restoration Priority Assessment Tool")
    print("Alphabow Energy Ltd.")
    print("="*60)
    
    # Initialize assessment
    assessment = CPRestorationPriorityAssessment()
    
    # Load pipeline data
    if not assessment.load_pipeline_data():
        print("Failed to load pipeline data. Exiting.")
        sys.exit(1)
    
    # Run assessment
    if not assessment.run_assessment():
        print("Failed to run assessment. Exiting.")
        sys.exit(1)
    
    # Generate all required reports
    assessment.generate_priority_action_list()
    assessment.generate_regulatory_compliance_report()
    assessment.generate_cost_timeline_matrix()
    assessment.generate_emergency_response_plan()
    assessment.generate_summary_statistics()
    
    # Save results
    assessment.save_results_to_csv()
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("IMMEDIATE ACTION REQUIRED: Contact AER within 24 hours")
    print("="*80)

if __name__ == "__main__":
    main()