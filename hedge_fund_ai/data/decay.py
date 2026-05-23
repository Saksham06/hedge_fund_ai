from datetime import datetime, timezone
import math


def time_decay(published_at, half_life_hours=24):

    if not published_at:
        return 0.5  # neutral fallback

    try:
        now = datetime.now(timezone.utc)

        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        age_hours = (now - published_at).total_seconds() / 3600

        decay = math.exp(-age_hours / half_life_hours)

        return max(0.1, min(1.0, decay))

    except Exception:
        return 0.5
