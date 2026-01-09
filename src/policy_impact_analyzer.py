"""
Policy Impact Analyzer
Tracks and quantifies the effectiveness of pollution control policies in Indian cities.
Demonstrates data-driven policy evaluation with before/after analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from indian_cities_config import POLICY_INTERVENTIONS, INDIAN_CITIES

class PolicyImpactAnalyzer:
    """
    Analyzes the impact of pollution control policies using empirical data.
    Performs before/after comparisons, statistical significance testing, and cost-benefit analysis.
    """
    
    def __init__(self, data_path="data/raw/india_aqi_complete.csv"):
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def analyze_odd_even_scheme(self):
        """
        Analyze the impact of Delhi's Odd-Even vehicle rationing scheme.
        Compares AQI before, during, and after implementation periods.
        """
        results = []
        
        for period in POLICY_INTERVENTIONS['odd_even']['periods']:
            start_date = pd.to_datetime(period['start'])
            end_date = pd.to_datetime(period['end'])
            
            # Define time windows
            days_before = 15
            days_after = 15
            
            before_start = start_date - timedelta(days=days_before)
            after_end = end_date + timedelta(days=days_after)
            
            # Get Delhi data
            delhi_data = self.df[self.df['City'] == 'Delhi'].copy()
            
            # Extract periods
            before_data = delhi_data[
                (delhi_data['Date'] >= before_start) &
                (delhi_data['Date'] < start_date)
            ]
            
            during_data = delhi_data[
                (delhi_data['Date'] >= start_date) &
                (delhi_data['Date'] <= end_date)
            ]
            
            after_data = delhi_data[
                (delhi_data['Date'] > end_date) &
                (delhi_data['Date'] <= after_end)
            ]
            
            # Calculate statistics
            before_mean = before_data['AQI'].mean()
            during_mean = during_data['AQI'].mean()
            after_mean = after_data['AQI'].mean()
            
            # Percentage changes
            pct_change_during = ((during_mean - before_mean) / before_mean) * 100
            pct_change_after = ((after_mean - before_mean) / before_mean) * 100
            
            # Statistical significance test (t-test)
            t_stat, p_value = stats.ttest_ind(before_data['AQI'], during_data['AQI'])
            
            results.append({
                'period': f"{start_date.date()} to {end_date.date()}",
                'before_aqi': before_mean,
                'during_aqi': during_mean,
                'after_aqi': after_mean,
                'pct_change_during': pct_change_during,
                'pct_change_after': pct_change_after,
                'statistically_significant': p_value < 0.05,
                'p_value': p_value,
                'effectiveness': 'Effective' if pct_change_during < -5 and p_value < 0.05 else 'Limited/No Effect'
            })
        
        return pd.DataFrame(results)
    
    def analyze_bs6_impact(self):
        """
        Analyze long-term impact of BS6 emission standards (Apr 2020).
        Compare vehicular pollutants (NO2) trends before and after.
        """
        bs6_start = pd.to_datetime("2020-04-01")
        
        # Get all-India data
        before_bs6 = self.df[self.df['Date'] < bs6_start]
        after_bs6 = self.df[self.df['Date'] >= bs6_start]
        
        # Monthly averages for NO2
        before_monthly = before_bs6.groupby(before_bs6['Date'].dt.to_period('M'))['NO2'].mean()
        after_monthly = after_bs6.groupby(after_bs6['Date'].dt.to_period('M'))['NO2'].mean()
        
        before_avg = before_bs6['NO2'].mean()
        after_avg = after_bs6['NO2'].mean()
        
        reduction_pct = ((after_avg - before_avg) / before_avg) * 100
        
        return {
            'policy': 'BS6 Emission Standards',
            'implementation_date': '2020-04-01',
            'before_avg_no2': before_avg,
            'after_avg_no2': after_avg,
            'reduction_percent': reduction_pct,
            'assessment': 'Gradual improvement observed' if reduction_pct < 0 else 'No clear improvement'
        }
    
    def analyze_grap_effectiveness(self):
        """
        Analyze Graded Response Action Plan (GRAP) in NCR region.
        GRAP triggers emergency measures when AQI crosses thresholds.
        """
        ncr_cities = ['Delhi', 'Noida', 'Gurgaon', 'Faridabad', 'Ghaziabad']
        
        # Filter NCR data (winter months when GRAP is active)
        ncr_data = self.df[self.df['City'].isin(ncr_cities)].copy()
        winter_data = ncr_data[ncr_data['Date'].dt.month.isin([10, 11, 12, 1, 2])]
        
        # Before GRAP (pre-2017) vs After GRAP (2017+)
        before_grap = winter_data[winter_data['Date'] < '2017-01-01']
        after_grap = winter_data[winter_data['Date'] >= '2017-01-01']
        
        # Count hazardous days (AQI > 300)
        before_hazardous_days = (before_grap['AQI'] > 300).sum()
        before_total_days = len(before_grap)
        
        after_hazardous_days = (after_grap['AQI'] > 300).sum()
        after_total_days = len(after_grap)
        
        before_pct = (before_hazardous_days / before_total_days) * 100
        after_pct = (after_hazardous_days / after_total_days) * 100
        
        return {
            'policy': 'GRAP (Graded Response Action Plan)',
            'region': 'NCR (Delhi, Noida, Gurgaon, Faridabad, Ghaziabad)',
            'implementation_year': 2017,
            'hazardous_days_before_pct': before_pct,
            'hazardous_days_after_pct': after_pct,
            'change': after_pct - before_pct,
            'assessment': 'Reduced hazardous days' if (after_pct - before_pct) < 0 else 'No improvement in worst days'
        }
    
    def calculate_health_cost_impact(self, city_name, year=2024):
        """
        Estimate health costs associated with air pollution for a city.
        Uses simplified WHO methodology.
        """
        city_config = INDIAN_CITIES.get(city_name)
        if not city_config:
            return None
        
        # Get city data for specified year
        city_data = self.df[
            (self.df['City'] == city_name) &
            (self.df['Date'].dt.year == year)
        ]
        
        # Count days by AQI category
        days_good = ((city_data['AQI'] >= 0) & (city_data['AQI'] <= 50)).sum()
        days_satisfactory = ((city_data['AQI'] > 50) & (city_data['AQI'] <= 100)).sum()
        days_moderate = ((city_data['AQI'] > 100) & (city_data['AQI'] <= 200)).sum()
        days_poor = ((city_data['AQI'] > 200) & (city_data['AQI'] <= 300)).sum()
        days_very_poor = ((city_data['AQI'] > 300) & (city_data['AQI'] <= 400)).sum()
        days_severe = (city_data['AQI'] > 400).sum()
        
        # Simplified cost estimates (per capita per year in INR)
        # Based on medical visits, medications, lost productivity
        cost_per_poor_day = 250  # ₹250 per person per poor day
        cost_per_very_poor_day = 500
        cost_per_severe_day = 1000
        
        population = city_config['population']
        
        total_cost = (
            (days_poor * cost_per_poor_day) +
            (days_very_poor * cost_per_very_poor_day) +
            (days_severe * cost_per_severe_day)
        ) * population / 365  # Averaged over year
        
        return {
            'city': city_name,
            'year': year,
            'population': population,
            'days_good': days_good,
            'days_satisfactory': days_satisfactory,
            'days_moderate': days_moderate,
            'days_poor': days_poor,
            'days_very_poor': days_very_poor,
            'days_severe': days_severe,
            'total_unhealthy_days': days_poor + days_very_poor + days_severe,
            'estimated_health_cost_inr': total_cost,
            'estimated_health_cost_crores': total_cost / 10_000_000,
            'per_capita_cost': total_cost / population
        }
    
    def compare_cities_policy_commitment(self):
        """
        Compare which cities have implemented more policies.
        """
        city_scores = []
        
        for city_name, city_data in INDIAN_CITIES.items():
            interventions = city_data.get('policy_interventions', [])
            num_policies = len(interventions)
            
            # Get average winter AQI for this city
            city_winter_data = self.df[
                (self.df['City'] == city_name) &
                (self.df['Date'].dt.month.isin([10, 11, 12, 1, 2]))
            ]
            
            avg_winter_aqi = city_winter_data['AQI'].mean() if len(city_winter_data) > 0 else 0
            
            city_scores.append({
                'city': city_name,
                'num_policies': num_policies,
                'policies': ', '.join(interventions) if interventions else 'None',
                'avg_winter_aqi': avg_winter_aqi,
                'tier': city_data['tier']
            })
        
        return pd.DataFrame(city_scores).sort_values('num_policies', ascending=False)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("  POLICY IMPACT ANALYZER")
    print("=" * 70)
    
    analyzer = PolicyImpactAnalyzer()
    
    # Odd-Even Analysis
    print("\n1. ODD-EVEN VEHICLE RATIONING SCHEME (Delhi)")
    print("-" * 70)
    odd_even_results = analyzer.analyze_odd_even_scheme()
    if not odd_even_results.empty:
        for idx, row in odd_even_results.iterrows():
            print(f"\nPeriod: {row['period']}")
            print(f"   Before AQI: {row['before_aqi']:.1f}")
            print(f"   During AQI: {row['during_aqi']:.1f}")
            print(f"   Change: {row['pct_change_during']:.1f}%")
            print(f"   Effectiveness: {row['effectiveness']}")
            print(f"   Statistically Significant: {'Yes (p<0.05)' if row['statistically_significant'] else 'No'}")
    
    # BS6 Analysis
    print("\n\n2. BS6 EMISSION STANDARDS (All India)")
    print("-" * 70)
    bs6_results = analyzer.analyze_bs6_impact()
    print(f"Implementation: {bs6_results['implementation_date']}")
    print(f"Before Avg NO2: {bs6_results['before_avg_no2']:.2f} µg/m³")
    print(f"After Avg NO2: {bs6_results['after_avg_no2']:.2f} µg/m³")
    print(f"Reduction: {bs6_results['reduction_percent']:.1f}%")
    print(f"Assessment: {bs6_results['assessment']}")
    
    # GRAP Analysis
    print("\n\n3. GRADED RESPONSE ACTION PLAN (NCR Region)")
    print("-" * 70)
    grap_results = analyzer.analyze_grap_effectiveness()
    print(f"Region: {grap_results['region']}")
    print(f"Hazardous Days Before GRAP: {grap_results['hazardous_days_before_pct']:.1f}%")
    print(f"Hazardous Days After GRAP: {grap_results['hazardous_days_after_pct']:.1f}%")
    print(f"Change: {grap_results['change']:.1f} percentage points")
    print(f"Assessment: {grap_results['assessment']}")
    
    # Health Cost Impact
    print("\n\n4. HEALTH COST IMPACT (Delhi, 2024)")
    print("-" * 70)
    health_costs = analyzer.calculate_health_cost_impact('Delhi', 2024)
    if health_costs:
        print(f"Total Unhealthy Days: {health_costs['total_unhealthy_days']}")
        print(f"   Poor Days (AQI 201-300): {health_costs['days_poor']}")
        print(f"   Very Poor Days (AQI 301-400): {health_costs['days_very_poor']}")
        print(f"   Severe Days (AQI 401+): {health_costs['days_severe']}")
        print(f"\nEstimated Health Cost: ₹{health_costs['estimated_health_cost_crores']:.0f} Crores")
        print(f"Per Capita Cost: ₹{health_costs['per_capita_cost']:.0f} per person/year")
    
    # City Policy Comparison
    print("\n\n5. CITY POLICY COMMITMENT RANKING")
    print("-" * 70)
    policy_ranking = analyzer.compare_cities_policy_commitment()
    print(policy_ranking.head(10).to_string(index=False))
