import math


def semi_major_axis_au(period_days, star_mass_solar):
    # Kepler's third law -> orbital semi-major axis
    years = period_days / 365.25
    return (years**2 * star_mass_solar)**(1/3)

def transit_depth(planet_radius_jupiter, star_radius_solar):
    # Fractional brightness dip (dimensionless)
    # 1 Jupiter radius is approximately = to 0.1005 solar radii
    planet_radius_solar = planet_radius_jupiter * 0.1005
    return (planet_radius_solar / star_radius_solar)**2


def transit_duration_hours(period_days, star_radius_solar, a_au, impact_param):
    # Transit duration in hours. Return 0 (no transit) if impact_param >= 1
    if (impact_param >= 1):
        return 0
    else:
        solar_radii = a_au * 215.032
        period_hours = period_days * 24
        return (period_hours / math.pi) * math.asin((star_radius_solar / solar_radii) * math.sqrt(1-impact_param**2))

# accelerates sharply as b approaches 1
print(transit_duration_hours(4332.6, 1, 5.2, 0))
print(transit_duration_hours(4332.6, 1, 5.2, 0.5))
print(transit_duration_hours(4332.6, 1, 5.2, 0.9))
print(transit_duration_hours(4332.6, 1, 5.2, 0.99))