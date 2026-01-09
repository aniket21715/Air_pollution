"""
Indian AQI Calculator
Converts pollutant concentrations to Indian AQI using CPCB (Central Pollution Control Board) breakpoints.

This matches the AQI shown on aqi.in and government sources.

Reference: CPCB National Air Quality Index
https://cpcb.nic.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvcHVibGljYXRpb25z
"""

# Indian AQI Breakpoints (CPCB Standard)
# Format: (min_concentration, max_concentration, min_aqi, max_aqi)
INDIAN_AQI_BREAKPOINTS = {
    'PM2.5': [
        (0, 30, 0, 50),        # Good
        (31, 60, 51, 100),     # Satisfactory
        (61, 90, 101, 200),    # Moderate
        (91, 120, 201, 300),   # Poor
        (121, 250, 301, 400),  # Very Poor
        (251, 500, 401, 500),  # Severe
    ],
    'PM10': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    'NO2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    'SO2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2400, 401, 500),
    ],
    'CO': [  # in mg/m³
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500),
    ],
    'O3': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}

# AQI Categories (Indian Standard)
INDIAN_AQI_CATEGORIES = {
    (0, 50): {"category": "Good", "color": "#00B050", "health_impact": "Minimal impact"},
    (51, 100): {"category": "Satisfactory", "color": "#92D050", "health_impact": "Minor breathing discomfort for sensitive people"},
    (101, 200): {"category": "Moderate", "color": "#FFFF00", "health_impact": "Breathing discomfort for people with lung/heart disease"},
    (201, 300): {"category": "Poor", "color": "#FF9900", "health_impact": "Breathing discomfort for most people"},
    (301, 400): {"category": "Very Poor", "color": "#FF0000", "health_impact": "Respiratory illness on prolonged exposure"},
    (401, 500): {"category": "Severe", "color": "#C00000", "health_impact": "Affects healthy people, serious impact on ill"},
}


def calculate_sub_index(concentration, pollutant):
    """
    Calculate sub-index for a pollutant using Indian AQI breakpoints.
    
    Args:
        concentration: Pollutant concentration in µg/m³ (or mg/m³ for CO)
        pollutant: One of 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'
    
    Returns:
        Sub-index value (0-500) or None if calculation fails
    """
    if concentration is None or pollutant not in INDIAN_AQI_BREAKPOINTS:
        return None
    
    breakpoints = INDIAN_AQI_BREAKPOINTS[pollutant]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            # Linear interpolation
            sub_index = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(sub_index)
    
    # Concentration exceeds all breakpoints
    if concentration > breakpoints[-1][1]:
        return 500
    
    return None


def calculate_indian_aqi(pm25=None, pm10=None, no2=None, so2=None, co=None, o3=None):
    """
    Calculate Indian AQI from pollutant concentrations.
    AQI = Maximum of all sub-indices (as per CPCB methodology).
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        pm10: PM10 concentration in µg/m³
        no2: NO2 concentration in µg/m³
        so2: SO2 concentration in µg/m³
        co: CO concentration in mg/m³
        o3: O3 concentration in µg/m³
    
    Returns:
        dict with 'aqi', 'category', 'color', 'dominant_pollutant', 'health_impact'
    """
    sub_indices = {}
    
    if pm25 is not None:
        sub_indices['PM2.5'] = calculate_sub_index(pm25, 'PM2.5')
    if pm10 is not None:
        sub_indices['PM10'] = calculate_sub_index(pm10, 'PM10')
    if no2 is not None:
        sub_indices['NO2'] = calculate_sub_index(no2, 'NO2')
    if so2 is not None:
        sub_indices['SO2'] = calculate_sub_index(so2, 'SO2')
    if co is not None:
        sub_indices['CO'] = calculate_sub_index(co, 'CO')
    if o3 is not None:
        sub_indices['O3'] = calculate_sub_index(o3, 'O3')
    
    # Remove None values
    valid_indices = {k: v for k, v in sub_indices.items() if v is not None}
    
    if not valid_indices:
        return None
    
    # AQI is the maximum sub-index
    aqi = max(valid_indices.values())
    dominant_pollutant = max(valid_indices, key=valid_indices.get)
    
    # Get category
    category_info = {"category": "Unknown", "color": "#888888", "health_impact": "N/A"}
    for (low, high), info in INDIAN_AQI_CATEGORIES.items():
        if low <= aqi <= high:
            category_info = info
            break
    
    return {
        'aqi': aqi,
        'category': category_info['category'],
        'color': category_info['color'],
        'health_impact': category_info['health_impact'],
        'dominant_pollutant': dominant_pollutant,
        'sub_indices': valid_indices,
    }


def convert_us_aqi_to_indian_aqi(us_aqi, pm25=None, pm10=None):
    """
    Approximate conversion from US EPA AQI to Indian AQI.
    
    Note: This is an approximation. For accurate results,
    raw concentration values should be used with calculate_indian_aqi().
    
    US EPA has different breakpoints than Indian CPCB:
    - US EPA: 0-50 (Good), 51-100 (Moderate), 101-150 (Unhealthy for Sensitive)
    - Indian: 0-50 (Good), 51-100 (Satisfactory), 101-200 (Moderate)
    
    Generally, Indian AQI tends to be higher for the same pollution level.
    """
    if pm25 is not None:
        # Use PM2.5 directly for more accurate conversion
        return calculate_indian_aqi(pm25=pm25, pm10=pm10)
    
    # Rough conversion factor (Indian AQI is typically 1.5-2.5x higher)
    # This is very approximate and should only be used as fallback
    if us_aqi <= 50:
        indian_aqi = us_aqi  # Similar in Good range
    elif us_aqi <= 100:
        indian_aqi = us_aqi * 1.2  # Slightly higher
    elif us_aqi <= 150:
        indian_aqi = us_aqi * 1.8  # Significantly higher
    elif us_aqi <= 200:
        indian_aqi = us_aqi * 2.0
    else:
        indian_aqi = us_aqi * 2.2
    
    return {
        'aqi': min(500, int(indian_aqi)),
        'category': 'Estimated',
        'color': '#888888',
        'health_impact': 'Approximate conversion',
        'dominant_pollutant': 'Unknown',
    }


# Test with current Delhi values from aqi.in
if __name__ == "__main__":
    print("=" * 70)
    print("  INDIAN AQI CALCULATOR TEST")
    print("=" * 70)
    
    # Test case: Delhi values from aqi.in (Jan 4, 2026)
    # PM2.5 = 187 µg/m³, PM10 = 254 µg/m³
    print("\nDelhi Test Case (from aqi.in screenshot):")
    print("  PM2.5 = 187 µg/m³")
    print("  PM10 = 254 µg/m³")
    
    result = calculate_indian_aqi(pm25=187, pm10=254)
    
    if result:
        print(f"\n  Calculated Indian AQI: {result['aqi']}")
        print(f"  Category: {result['category']}")
        print(f"  Dominant Pollutant: {result['dominant_pollutant']}")
        print(f"  Sub-indices: {result['sub_indices']}")
        print(f"\n  Expected (aqi.in): 352")
        print(f"  Match: {'✅ YES' if abs(result['aqi'] - 352) < 20 else '❌ NO'}")
    
    print("\n" + "-" * 70)
    
    # Bengaluru test case
    print("\nBengaluru Test Case (from aqi.in screenshot):")
    print("  PM2.5 = 73 µg/m³, PM10 = 97 µg/m³")
    
    result = calculate_indian_aqi(pm25=73, pm10=97)
    
    if result:
        print(f"\n  Calculated Indian AQI: {result['aqi']}")
        print(f"  Category: {result['category']}")
        print(f"  Expected (aqi.in): 163")
        print(f"  Match: {'✅ YES' if abs(result['aqi'] - 163) < 20 else '❌ NO'}")
