"""
Location Detector Module
Simple city matching utility for the Air Quality App.
"""

from math import radians, sin, cos, sqrt, atan2

# City coordinates for matching (latitude, longitude)
CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Thane": (19.2183, 72.9781),
    "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185),
    "Patna": (25.5941, 85.1376),
    "Vadodara": (22.3072, 73.1812),
    "Ghaziabad": (28.6692, 77.4538),
    "Ludhiana": (30.9010, 75.8573),
    "Agra": (27.1767, 78.0081),
    "Nashik": (19.9975, 73.7898),
    "Faridabad": (28.4089, 77.3178),
    "Meerut": (28.9845, 77.7064),
    "Rajkot": (22.3039, 70.8022),
    "Varanasi": (25.3176, 82.9739),
    "Srinagar": (34.0837, 74.7973),
    "Aurangabad": (19.8762, 75.3433),
    "Dhanbad": (23.7957, 86.4304),
    "Amritsar": (31.6340, 74.8723),
    "Allahabad": (25.4358, 81.8463),
    "Ranchi": (23.3441, 85.3096),
    "Howrah": (22.5958, 88.2636),
    "Coimbatore": (11.0168, 76.9558),
    "Jabalpur": (23.1815, 79.9864),
    "Gwalior": (26.2183, 78.1828),
    "Vijayawada": (16.5062, 80.6480),
    "Jodhpur": (26.2389, 73.0243),
    "Madurai": (9.9252, 78.1198),
    "Raipur": (21.2514, 81.6296),
    "Kota": (25.2138, 75.8648),
    "Guwahati": (26.1445, 91.7362),
    "Chandigarh": (30.7333, 76.7794),
    "Solapur": (17.6599, 75.9064),
    "Hubli-Dharwad": (15.3647, 75.1240),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Bareilly": (28.3670, 79.4304),
    "Moradabad": (28.8389, 78.7769),
    "Mysore": (12.2958, 76.6394),
    "Tiruppur": (11.1085, 77.3411),
    "Gurgaon": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def find_nearest_city(lat: float, lon: float, available_cities: list) -> tuple:
    """
    Find the nearest city from the available cities list.
    Returns (city_name, distance_km).
    """
    min_distance = float('inf')
    nearest_city = None
    
    for city in available_cities:
        if city in CITY_COORDINATES:
            city_lat, city_lon = CITY_COORDINATES[city]
            distance = haversine_distance(lat, lon, city_lat, city_lon)
            
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
    
    return nearest_city, min_distance
