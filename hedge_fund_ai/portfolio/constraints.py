from collections import defaultdict


def apply_sector_constraints(portfolio, max_sector_weight=0.25):

    sector_weights = defaultdict(float)
    adjusted = []

    for p in sorted(portfolio, key=lambda x: x["weight"], reverse=True):

        sector = p.get("sector", "Unknown")
        w = float(p["weight"])

        if sector_weights[sector] + w > max_sector_weight:
            allowed = max(0.0, max_sector_weight - sector_weights[sector])
            w = allowed

        if w > 0:
            sector_weights[sector] += w
            p["weight"] = w
            adjusted.append(p)

    # Renormalize
    total = sum(float(p["weight"]) for p in adjusted)
    if total <= 0:
        return []

    for p in adjusted:
        p["weight"] = float(p["weight"]) / total

    return adjusted

