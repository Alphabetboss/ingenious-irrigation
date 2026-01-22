def calculate_hydration_score(ai_results, soil_moisture=None):
    """
    Returns hydration score 0–10
    0 = bone dry
    5 = optimal
    10 = oversaturated
    """

    grass_health = ai_results.get("healthy_grass", 0.5)
    water_presence = ai_results.get("water", 0.0)
    dead_grass = ai_results.get("dead_grass", 0.0)

    score = 5

    score -= (1 - grass_health) * 4
    score -= dead_grass * 3
    score += water_presence * 4

    if soil_moisture is not None:
        score += (soil_moisture - 0.5) * 4

    return max(0, min(10, round(score, 1)))