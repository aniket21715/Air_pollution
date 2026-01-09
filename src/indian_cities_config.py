"""
Indian Cities Configuration
Centralized metadata for 30+ major Indian cities including coordinates,
demographics, pollution characteristics, and socioeconomic data for equity analysis.
"""

# Comprehensive city metadata for analysis and mapping
INDIAN_CITIES = {
    # NCR Region (Highest pollution zone)
    "Delhi": {
        "coords": {"lat": 28.6139, "lon": 77.2090},
        "state": "Delhi",
        "population": 32_941_000,
        "tier": 1,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "construction", "stubble_burning", "industrial"],
        "income_level": "high",
        "baseline_winter_aqi": 265,  # aqi.in verified (Jan 3, 2026: 265 Severe)
        "baseline_summer_aqi": 130,
        "policy_interventions": ["odd_even", "graded_response", "bs6"],
    },
    "Ghaziabad": {
        "coords": {"lat": 28.6692, "lon": 77.4538},
        "state": "Uttar Pradesh",
        "population": 2_358_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["industrial", "vehicular", "dust"],
        "income_level": "medium",
        "baseline_winter_aqi": 280,  # Often India's most polluted
        "baseline_summer_aqi": 140,
        "policy_interventions": ["ncr_action_plan"],
    },
    "Noida": {
        "coords": {"lat": 28.5355, "lon": 77.3910},
        "state": "Uttar Pradesh",
        "population": 810_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "construction", "industrial"],
        "income_level": "high",
        "baseline_winter_aqi": 240,
        "baseline_summer_aqi": 115,
        "policy_interventions": ["ncr_action_plan"],
    },
    "Faridabad": {
        "coords": {"lat": 28.4089, "lon": 77.3178},
        "state": "Haryana",
        "population": 1_595_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["industrial", "vehicular"],
        "income_level": "medium",
        "baseline_winter_aqi": 260,
        "baseline_summer_aqi": 130,
        "policy_interventions": ["ncr_action_plan"],
    },
    "Gurgaon": {
        "coords": {"lat": 28.4595, "lon": 77.0266},
        "state": "Haryana",
        "population": 1_153_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "construction"],
        "income_level": "very_high",
        "baseline_winter_aqi": 220,
        "baseline_summer_aqi": 105,
        "policy_interventions": ["ncr_action_plan"],
    },
    
    # Tier-1 Metropolitan Cities
    "Mumbai": {
        "coords": {"lat": 19.0760, "lon": 72.8777},
        "state": "Maharashtra",
        "population": 22_120_000,
        "tier": 1,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["vehicular", "industrial", "port_activities"],
        "income_level": "high",
        "baseline_winter_aqi": 110,  # Real-time calibrated (Jan 2026: 110)
        "baseline_summer_aqi": 85,
        "policy_interventions": ["metro_expansion", "ev_buses"],
    },
    "Bengaluru": {
        "coords": {"lat": 12.9716, "lon": 77.5946},
        "state": "Karnataka",
        "population": 13_707_000,
        "tier": 1,
        "primary_pollutants": ["PM10", "PM2.5"],  # PM2.5 is also significant
        "pollution_sources": ["vehicular", "construction", "dust"],
        "income_level": "high",
        "baseline_winter_aqi": 160,  # aqi.in verified (Jan 3, 2026: 163 Unhealthy)
        "baseline_summer_aqi": 85,
        "policy_interventions": ["metro_expansion", "tree_cover"],
    },
    "Hyderabad": {
        "coords": {"lat": 17.3850, "lon": 78.4867},
        "state": "Telangana",
        "population": 10_534_000,
        "tier": 1,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["vehicular", "construction", "dust"],
        "income_level": "high",
        "baseline_winter_aqi": 130,
        "baseline_summer_aqi": 100,
        "policy_interventions": ["metro_expansion"],
    },
    "Chennai": {
        "coords": {"lat": 13.0827, "lon": 80.2707},
        "state": "Tamil Nadu",
        "population": 11_503_000,
        "tier": 1,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["vehicular", "industrial"],
        "income_level": "high",
        "baseline_winter_aqi": 60,  # Real-time calibrated (Jan 2026: 57)
        "baseline_summer_aqi": 50,
        "policy_interventions": ["coastal_regulation"],
    },
    "Kolkata": {
        "coords": {"lat": 22.5726, "lon": 88.3639},
        "state": "West Bengal",
        "population": 15_134_000,
        "tier": 1,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "industrial", "biomass_burning"],
        "income_level": "medium",
        "baseline_winter_aqi": 360,  # Real-time calibrated (Jan 2026: 365!) - severely polluted
        "baseline_summer_aqi": 140,
        "policy_interventions": ["industrial_relocation"],
    },
    
    # Tier-2 Major Cities (High Pollution)
    "Ahmedabad": {
        "coords": {"lat": 23.0225, "lon": 72.5714},
        "state": "Gujarat",
        "population": 8_450_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["industrial", "vehicular", "dust"],
        "income_level": "medium",
        "baseline_winter_aqi": 150,
        "baseline_summer_aqi": 120,
        "policy_interventions": ["industrial_zones"],
    },
    "Pune": {
        "coords": {"lat": 18.5204, "lon": 73.8567},
        "state": "Maharashtra",
        "population": 7_764_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["vehicular", "construction"],
        "income_level": "high",
        "baseline_winter_aqi": 175,  # Real-time calibrated (Jan 2026: 175)
        "baseline_summer_aqi": 100,
        "policy_interventions": ["metro_expansion"],
    },
    "Jaipur": {
        "coords": {"lat": 26.9124, "lon": 75.7873},
        "state": "Rajasthan",
        "population": 3_971_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["vehicular", "dust", "construction"],
        "income_level": "medium",
        "baseline_winter_aqi": 160,
        "baseline_summer_aqi": 130,
        "policy_interventions": ["dust_control"],
    },
    "Lucknow": {
        "coords": {"lat": 26.8467, "lon": 80.9462},
        "state": "Uttar Pradesh",
        "population": 3_645_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "biomass_burning", "industrial"],
        "income_level": "medium",
        "baseline_winter_aqi": 220,
        "baseline_summer_aqi": 130,
        "policy_interventions": ["biomass_ban"],
    },
    "Kanpur": {
        "coords": {"lat": 26.4499, "lon": 80.3319},
        "state": "Uttar Pradesh",
        "population": 3_067_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["industrial", "vehicular", "leather_tanneries"],
        "income_level": "low",
        "baseline_winter_aqi": 270,  # Heavily polluted industrial city
        "baseline_summer_aqi": 150,
        "policy_interventions": ["industrial_cleanup"],
    },
    "Nagpur": {
        "coords": {"lat": 21.1458, "lon": 79.0882},
        "state": "Maharashtra",
        "population": 2_968_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["vehicular", "industrial"],
        "income_level": "medium",
        "baseline_winter_aqi": 140,
        "baseline_summer_aqi": 110,
        "policy_interventions": [],
    },
    "Indore": {
        "coords": {"lat": 22.7196, "lon": 75.8577},
        "state": "Madhya Pradesh",
        "population": 3_276_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["vehicular", "industrial"],
        "income_level": "medium",
        "baseline_winter_aqi": 150,
        "baseline_summer_aqi": 115,
        "policy_interventions": ["clean_city_initiative"],
    },
    "Patna": {
        "coords": {"lat": 25.5941, "lon": 85.1376},
        "state": "Bihar",
        "population": 2_356_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "biomass_burning", "construction"],
        "income_level": "low",
        "baseline_winter_aqi": 240,
        "baseline_summer_aqi": 140,
        "policy_interventions": [],
    },
    "Bhopal": {
        "coords": {"lat": 23.2599, "lon": 77.4126},
        "state": "Madhya Pradesh",
        "population": 2_371_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["vehicular", "industrial"],
        "income_level": "medium",
        "baseline_winter_aqi": 145,
        "baseline_summer_aqi": 105,
        "policy_interventions": [],
    },
    "Ludhiana": {
        "coords": {"lat": 30.9010, "lon": 75.8573},
        "state": "Punjab",
        "population": 1_800_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["industrial", "stubble_burning", "vehicular"],
        "income_level": "medium",
        "baseline_winter_aqi": 230,
        "baseline_summer_aqi": 120,
        "policy_interventions": ["stubble_burning_ban"],
    },
    "Agra": {
        "coords": {"lat": 27.1767, "lon": 78.0081},
        "state": "Uttar Pradesh",
        "population": 1_760_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "industrial", "foundries"],
        "income_level": "low",
        "baseline_winter_aqi": 210,
        "baseline_summer_aqi": 130,
        "policy_interventions": ["taj_trapezium_zone"],
    },
    "Varanasi": {
        "coords": {"lat": 25.3176, "lon": 82.9739},
        "state": "Uttar Pradesh",
        "population": 1_435_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["biomass_burning", "vehicular", "cremations"],
        "income_level": "low",
        "baseline_winter_aqi": 220,
        "baseline_summer_aqi": 135,
        "policy_interventions": [],
    },
    "Meerut": {
        "coords": {"lat": 28.9845, "lon": 77.7064},
        "state": "Uttar Pradesh",
        "population": 1_765_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["industrial", "vehicular", "stubble_burning"],
        "income_level": "medium",
        "baseline_winter_aqi": 250,
        "baseline_summer_aqi": 130,
        "policy_interventions": ["ncr_action_plan"],
    },
    "Vadodara": {
        "coords": {"lat": 22.3072, "lon": 73.1812},
        "state": "Gujarat",
        "population": 2_065_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["industrial", "vehicular"],
        "income_level": "medium",
        "baseline_winter_aqi": 140,
        "baseline_summer_aqi": 110,
        "policy_interventions": ["industrial_zones"],
    },
    "Visakhapatnam": {
        "coords": {"lat": 17.6868, "lon": 83.2185},
        "state": "Andhra Pradesh",
        "population": 2_035_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["industrial", "port_activities"],
        "income_level": "medium",
        "baseline_winter_aqi": 115,
        "baseline_summer_aqi": 85,
        "policy_interventions": [],
    },
    "Surat": {
        "coords": {"lat": 21.1702, "lon": 72.8311},
        "state": "Gujarat",
        "population": 6_564_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "PM2.5"],
        "pollution_sources": ["industrial", "vehicular"],
        "income_level": "medium",
        "baseline_winter_aqi": 135,
        "baseline_summer_aqi": 105,
        "policy_interventions": [],
    },
    "Chandigarh": {
        "coords": {"lat": 30.7333, "lon": 76.7794},
        "state": "Chandigarh",
        "population": 1_179_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "stubble_burning"],
        "income_level": "high",
        "baseline_winter_aqi": 190,
        "baseline_summer_aqi": 100,
        "policy_interventions": ["green_city_planning"],
    },
    "Amritsar": {
        "coords": {"lat": 31.6340, "lon": 74.8723},
        "state": "Punjab",
        "population": 1_183_000,
        "tier": 2,
        "primary_pollutants": ["PM2.5", "PM10"],
        "pollution_sources": ["vehicular", "stubble_burning"],
        "income_level": "medium",
        "baseline_winter_aqi": 220,
        "baseline_summer_aqi": 115,
        "policy_interventions": ["stubble_burning_ban"],
    },
    "Jodhpur": {
        "coords": {"lat": 26.2389, "lon": 73.0243},
        "state": "Rajasthan",
        "population": 1_137_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "dust"],
        "pollution_sources": ["dust", "vehicular"],
        "income_level": "low",
        "baseline_winter_aqi": 170,
        "baseline_summer_aqi": 150,  # Desert dust
        "policy_interventions": [],
    },
    "Kochi": {
        "coords": {"lat": 9.9312, "lon": 76.2673},
        "state": "Kerala",
        "population": 2_117_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["vehicular", "port_activities"],
        "income_level": "medium",
        "baseline_winter_aqi": 90,
        "baseline_summer_aqi": 70,
        "policy_interventions": [],
    },
    "Coimbatore": {
        "coords": {"lat": 11.0168, "lon": 76.9558},
        "state": "Tamil Nadu",
        "population": 2_151_000,
        "tier": 2,
        "primary_pollutants": ["PM10", "NO2"],
        "pollution_sources": ["industrial", "vehicular"],
        "income_level": "medium",
        "baseline_winter_aqi": 105,
        "baseline_summer_aqi": 85,
        "policy_interventions": [],
    },
}

# Policy intervention database
POLICY_INTERVENTIONS = {
    "odd_even": {
        "name": "Odd-Even Vehicle Rationing",
        "description": "Vehicles with odd/even number plates allowed on alternate days",
        "implemented_in": ["Delhi"],
        "periods": [
            {"start": "2016-01-01", "end": "2016-01-15"},
            {"start": "2016-04-15", "end": "2016-04-30"},
            {"start": "2019-11-04", "end": "2019-11-15"},
            {"start": "2024-11-15", "end": "2024-11-30"},
        ],
        "expected_impact": "10-15% reduction in PM2.5",
    },
    "graded_response": {
        "name": "Graded Response Action Plan (GRAP)",
        "description": "Tiered pollution control measures based on AQI levels",
        "implemented_in": ["Delhi", "Noida", "Gurgaon", "Faridabad", "Ghaziabad"],
        "start_date": "2017-01-01",
        "expected_impact": "Variable based on AQI triggers",
    },
    "bs6": {
        "name": "BS6 Emission Standards",
        "description": "Stricter vehicular emission norms",
        "implemented_in": "all",
        "start_date": "2020-04-01",
        "expected_impact": "30-40% reduction in vehicular emissions (long-term)",
    },
    "stubble_burning_ban": {
        "name": "Stubble Burning Ban",
        "description": "Ban on crop residue burning",
        "implemented_in": ["Punjab", "Haryana", "Uttar Pradesh"],
        "enforcement_period": "October-November annually",
        "compliance_rate": "variable (20-60%)",
    }
}

# AQI category thresholds (Indian standards)
AQI_CATEGORIES = {
    "Good": {"range": (0, 50), "color": "#00e400", "health_impact": "Minimal"},
    "Satisfactory": {"range": (51, 100), "color": "#ffff00", "health_impact": "Minor breathing discomfort for sensitive people"},
    "Moderate": {"range": (101, 200), "color": "#ff7e00", "health_impact": "Breathing discomfort for people with lung/heart disease"},
    "Poor": {"range": (201, 300), "color": "#ff0000", "health_impact": "Breathing discomfort to most people on prolonged exposure"},
    "Very Poor": {"range": (301, 400), "color": "#99004c", "health_impact": "Respiratory illness on prolonged exposure"},
    "Severe": {"range": (401, 500), "color": "#7e0023", "health_impact": "Affects healthy people, seriously impacts those with existing diseases"},
}


def get_city_metadata(city_name):
    """Get metadata for a specific city."""
    return INDIAN_CITIES.get(city_name, None)


def get_all_cities():
    """Get list of all configured cities."""
    return list(INDIAN_CITIES.keys())


def get_cities_by_tier(tier):
    """Get cities by tier (1, 2, or 3)."""
    return [city for city, data in INDIAN_CITIES.items() if data["tier"] == tier]


def get_cities_by_pollution_level(threshold_aqi=200):
    """Get cities with baseline winter AQI above threshold."""
    return [city for city, data in INDIAN_CITIES.items() if data["baseline_winter_aqi"] >= threshold_aqi]


def get_aqi_category(aqi_value):
    """Get AQI category and color based on value."""
    for category, info in AQI_CATEGORIES.items():
        min_val, max_val = info["range"]
        if min_val <= aqi_value <= max_val:
            return category, info["color"], info["health_impact"]
    return "Severe", "#7e0023", "Hazardous"


if __name__ == "__main__":
    print(f"Total cities configured: {len(INDIAN_CITIES)}")
    print(f"\nTier-1 cities: {len(get_cities_by_tier(1))}")
    print(f"Tier-2 cities: {len(get_cities_by_tier(2))}")
    print(f"\nHighly polluted cities (Winter AQI > 200): {len(get_cities_by_pollution_level(200))}")
    print(get_cities_by_pollution_level(200))
