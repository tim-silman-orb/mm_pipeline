import pandas as pd

import pandas as pd


async def collect_telegram_messages(
    client,
    channels,
    limit=100
):
    """
    Pull raw messages from one or more Telegram channels.

    Returns source-native Telegram data.
    """

    rows = []

    for channel_name in channels:

        channel = await client.get_entity(
            channel_name
        )

        messages = await client.get_messages(
            channel,
            limit=limit
        )

        for msg in messages:

            rows.append({
                "channel": channel_name,
                "channel_title": channel.title,
                "message_id": msg.id,
                "date": msg.date,
                "text": msg.text,
                "views": msg.views,
                "forwards": msg.forwards,
                "message_url": (
                    f"https://t.me/"
                    f"{channel_name.lstrip('@')}/"
                    f"{msg.id}"
                ),
            })

    return pd.DataFrame(rows)


def standardise_telegram(df):
    return pd.DataFrame({
        "platform": "telegram",
        "source_id": df["message_id"],
        "published_at": df["date"],
        "country": "",
        "language": "",
        "source_name": df["channel_title"],
        "author": df["channel"],
        "headline": "",
        "content": df["text"],
        "summary": "",
        "url": df["message_url"],
        "search_id": df["channel"],
    })

