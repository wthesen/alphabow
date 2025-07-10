#!/usr/bin/env python3
"""
AER Notification Requirements Assessment
Alphabow Energy Ltd. - CP Restoration Priority Assessment

This module generates specific AER notification documentation for the
18+ month CP system outage situation.
"""

import pandas as pd
from datetime import datetime, timedelta

class AERNotificationAssessment:
    def __init__(self, results_df):
        self.results = results_df
        self.current_date = datetime(2025, 7, 10)
        self.cp_shutdown_date = datetime(2023, 10, 1)
        
    def generate_section_35_notification(self):
        """Generate Pipeline Act Section 35 notification content"""
        print("\n" + "="*80)
        print("AER PIPELINE ACT SECTION 35 NOTIFICATION REQUIRED")
        print("="*80)
        
        extreme_pipelines = self.results[self.results['Priority'] == 'EXTREME']
        high_pressure_lines = extreme_pipelines[extreme_pipelines['MAOP_kPa'] > 5000]
        sour_lines = extreme_pipelines[extreme_pipelines['H2S_Content'] > 0]
        
        print(f"\nSECTION 35 REPORTING REQUIREMENTS:")
        print(f"1. CONDITIONS AFFECTING PIPELINE SAFETY:")
        print(f"   • {len(extreme_pipelines)} steel pipelines without CP protection for 18+ months")
        print(f"   • {len(high_pressure_lines)} high-pressure pipelines (>5000 kPa) at extreme risk")
        print(f"   • {len(sour_lines)} sour gas pipelines with accelerated corrosion risk")
        print(f"   • Potential for stress-corrosion cracking and external corrosion failures")
        
        print(f"\n2. ENVIRONMENTAL PROTECTION CONCERNS:")
        print(f"   • Unknown pipeline integrity after 18+ months CP inactivity")
        print(f"   • Accelerated external corrosion rates on steel pipelines")
        print(f"   • Increased risk of product release to soil and groundwater")
        print(f"   • Potential impact on sensitive environmental areas")
        
        print(f"\n3. PUBLIC SAFETY IMPLICATIONS:")
        print(f"   • Increased risk of pipeline rupture or leak")
        print(f"   • Potential H2S exposure risk from {len(sour_lines)} sour pipelines")
        print(f"   • Risk to public from uncontrolled product releases")
        print(f"   • Emergency response capability degradation")
        
        return {
            'total_affected': len(extreme_pipelines),
            'high_pressure': len(high_pressure_lines),
            'sour_pipelines': len(sour_lines)
        }
    
    def generate_directive_077_compliance(self):
        """Generate Directive 077 compliance assessment"""
        print(f"\n" + "="*80)
        print("DIRECTIVE 077 COMPLIANCE ASSESSMENT")
        print("="*80)
        
        print(f"\nINTEGRITY MANAGEMENT PROGRAM FAILURES:")
        print(f"• CP system monitoring discontinued for 18+ months")
        print(f"• Failure to maintain adequate corrosion protection")
        print(f"• Extended outage exceeds any acceptable maintenance window")
        print(f"• No documented integrity assessment during outage period")
        
        print(f"\nEXTERNAL CORROSION PREVENTION VIOLATIONS:")
        print(f"• CSA Z662 Clause 14.6 - Cathodic protection system requirements")
        print(f"• Directive 077 Section 2.1 - Corrosion prevention coating requirements")
        print(f"• Extended exposure without primary corrosion protection")
        
        print(f"\nREPORTING TIMELINE VIOLATIONS:")
        print(f"• Failure to notify AER of extended CP system outage")
        print(f"• Missing monthly integrity management reports")
        print(f"• No corrective action plan submitted for CP restoration")
        
        aging_pipelines = self.results[self.results['Age_Years'] > 20]
        print(f"\nHIGH-RISK AGING INFRASTRUCTURE:")
        print(f"• {len(aging_pipelines)} pipelines >20 years old without CP protection")
        print(f"• Average pipeline age: {self.results['Age_Years'].mean():.1f} years")
        print(f"• Oldest pipeline: {self.results['Age_Years'].max():.1f} years")
        
        return len(aging_pipelines)
    
    def calculate_corrosion_acceleration(self):
        """Calculate expected corrosion acceleration rates"""
        print(f"\n" + "="*80)
        print("CORROSION ACCELERATION ANALYSIS")
        print("="*80)
        
        # Typical corrosion rates (mils per year)
        baseline_corrosion = 2.0  # mils/year with CP
        unprotected_corrosion = 8.0  # mils/year without CP
        sour_multiplier = 2.5  # Additional factor for H2S
        
        months_unprotected = 18.25
        
        print(f"\nCORROSION RATE ESTIMATES:")
        print(f"• Baseline with CP: {baseline_corrosion} mils/year")
        print(f"• Without CP protection: {unprotected_corrosion} mils/year")
        print(f"• Sour service multiplier: {sour_multiplier}x")
        print(f"• Exposure period: {months_unprotected:.1f} months")
        
        # Calculate metal loss
        metal_loss_standard = (unprotected_corrosion * months_unprotected) / 12
        metal_loss_sour = metal_loss_standard * sour_multiplier
        
        print(f"\nESTIMATED METAL LOSS (18+ months):")
        print(f"• Standard pipelines: {metal_loss_standard:.1f} mils")
        print(f"• Sour pipelines: {metal_loss_sour:.1f} mils")
        print(f"• Potential wall thickness reduction: 5-15%")
        
        # Risk of failure
        sour_pipelines = self.results[self.results['H2S_Content'] > 0]
        print(f"\nFAILURE RISK ASSESSMENT:")
        print(f"• {len(sour_pipelines)} sour pipelines at extreme risk")
        print(f"• Stress-corrosion cracking potential significantly increased")
        print(f"• Joint integrity may be compromised")
        print(f"• Coating failure progression accelerated")
        
        return {
            'metal_loss_standard': metal_loss_standard,
            'metal_loss_sour': metal_loss_sour,
            'sour_count': len(sour_pipelines)
        }
    
    def environmental_impact_assessment(self):
        """Assess environmental impact of 18+ month CP exposure"""
        print(f"\n" + "="*80)
        print("ENVIRONMENTAL IMPACT ASSESSMENT")
        print("="*80)
        
        # Categorize pipelines by environmental sensitivity
        water_lines = self.results[self.results['Water_Crossing'] == True]
        salt_water_lines = self.results[self.results['Substance'] == 'Salt Water']
        oil_lines = self.results[self.results['Substance'] == 'Oil Well Effluent']
        
        print(f"\nENVIRONMENTAL SENSITIVITY MAPPING:")
        print(f"• Water body crossings: {len(water_lines)} pipelines")
        print(f"• Salt water pipelines: {len(salt_water_lines)} pipelines")
        print(f"• Oil effluent pipelines: {len(oil_lines)} pipelines")
        
        print(f"\nSPILL RISK ASSESSMENT:")
        print(f"• Unknown pipeline condition after 18+ months")
        print(f"• Increased failure probability due to external corrosion")
        print(f"• Coating degradation may have occurred")
        print(f"• Stress-corrosion cracking risk elevated")
        
        # Calculate potential spill volumes
        high_pressure_lines = self.results[self.results['MAOP_kPa'] > 5000]
        total_length_high_risk = high_pressure_lines['Length_m'].sum()
        
        print(f"\nEMERGENCY RESPONSE CAPABILITY GAPS:")
        print(f"• {len(high_pressure_lines)} high-pressure lines requiring immediate monitoring")
        print(f"• Total high-risk pipeline length: {total_length_high_risk/1000:.1f} km")
        print(f"• Emergency shutdown systems may not be optimally positioned")
        print(f"• Leak detection systems may have reduced sensitivity")
        
        return {
            'water_crossings': len(water_lines),
            'salt_water': len(salt_water_lines),
            'oil_effluent': len(oil_lines),
            'high_pressure_length': total_length_high_risk
        }
    
    def generate_immediate_action_plan(self):
        """Generate immediate action plan for AER submission"""
        print(f"\n" + "="*80)
        print("IMMEDIATE ACTION PLAN FOR AER SUBMISSION")
        print("="*80)
        
        extreme_count = len(self.results[self.results['Priority'] == 'EXTREME'])
        
        print(f"\nPHASE 1: IMMEDIATE ACTIONS (0-24 hours)")
        print(f"1. AER notification via Section 35 reporting")
        print(f"2. Emergency response team mobilization")
        print(f"3. Pressure reduction on highest risk pipelines")
        print(f"4. Enhanced monitoring and patrol schedule implementation")
        
        print(f"\nPHASE 2: SHORT-TERM ACTIONS (1-7 days)")
        print(f"1. Emergency CP restoration on {extreme_count} EXTREME priority pipelines")
        print(f"2. Integrity assessment planning and resource mobilization")
        print(f"3. Environmental monitoring enhancement")
        print(f"4. Emergency response capability verification")
        
        print(f"\nPHASE 3: MEDIUM-TERM ACTIONS (1-4 months)")
        print(f"1. Complete CP system restoration across all steel pipelines")
        print(f"2. Comprehensive integrity assessment program")
        print(f"3. Enhanced integrity management program implementation")
        print(f"4. Regulatory compliance verification and reporting")
        
        print(f"\nRESOURCE REQUIREMENTS:")
        print(f"• CP restoration crews: 8-10 teams required")
        print(f"• Integrity assessment contractors: 3-4 companies")
        print(f"• Environmental monitoring: Continuous for high-risk areas")
        print(f"• Estimated timeline: 4-6 months for complete restoration")
        
        return extreme_count

def main():
    """Generate AER notification assessment"""
    try:
        # Load results from previous assessment
        results_df = pd.read_csv('cp_restoration_priority_results.csv')
        
        aer_assessment = AERNotificationAssessment(results_df)
        
        print("="*80)
        print("AER NOTIFICATION REQUIREMENTS ASSESSMENT")
        print("Alphabow Energy Ltd. - CP Restoration Priority")
        print(f"Assessment Date: July 10, 2025")
        print("="*80)
        
        # Generate all AER notification components
        aer_assessment.generate_section_35_notification()
        aer_assessment.generate_directive_077_compliance()
        aer_assessment.calculate_corrosion_acceleration()
        aer_assessment.environmental_impact_assessment()
        aer_assessment.generate_immediate_action_plan()
        
        print(f"\n" + "="*80)
        print("AER NOTIFICATION ASSESSMENT COMPLETE")
        print("SUBMIT TO AER WITHIN 24 HOURS")
        print("="*80)
        
    except FileNotFoundError:
        print("Error: cp_restoration_priority_results.csv not found")
        print("Please run cp_restoration_priority_assessment.py first")

if __name__ == "__main__":
    main()