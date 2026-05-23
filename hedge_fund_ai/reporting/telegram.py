import requests

from hedge_fund_ai.config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN


def send_telegram(message):

    if not TELEGRAM_TOKEN:
        print("Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message[:4000],
            },
            timeout=20,
        )
    except Exception as e:
        print("Telegram error:", e)
