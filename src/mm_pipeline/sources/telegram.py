from __future__ import annotations

import asyncio
import logging
import os

from mm_pipeline.config import SourceConfig
from mm_pipeline.models import Article, CollectionWindow, SourceResult

from .parsers import clean_text, parse_datetime


class TelegramAdapter:
    def __init__(self, config: SourceConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def collect(self, window: CollectionWindow) -> SourceResult:
        return asyncio.run(self._collect(window))

    async def _collect(self, window: CollectionWindow) -> SourceResult:
        try:
            from telethon import TelegramClient
            from telethon.sessions import StringSession
        except ImportError as exc:
            raise RuntimeError("Install mm-pipeline[telegram] to use Telegram") from exc

        api_id_env = str(self.config.options.get("api_id_env", "TELEGRAM_API_ID"))
        api_hash_env = str(self.config.options.get("api_hash_env", "TELEGRAM_API_HASH"))
        session_env = str(self.config.options.get("session_env", "TELEGRAM_SESSION_STRING"))
        api_id = os.getenv(api_id_env, "").strip()
        api_hash = os.getenv(api_hash_env, "").strip()
        session_string = os.getenv(session_env, "").strip()
        missing = [
            name
            for name, value in ((api_id_env, api_id), (api_hash_env, api_hash), (session_env, session_string))
            if not value
        ]
        if missing:
            raise ValueError(f"Required environment variable(s) missing: {', '.join(missing)}")
        channels = list(self.config.options.get("channels", []))
        limit = int(self.config.options.get("limit", 100))
        client = TelegramClient(StringSession(session_string), int(api_id), api_hash)
        await client.connect()
        try:
            if not await client.is_user_authorized():
                raise RuntimeError("Telegram session is not authorised")
            records: list[Article] = []
            for channel_name in channels:
                channel = await client.get_entity(channel_name)
                messages = await client.get_messages(channel, limit=limit)
                for message in messages:
                    published = parse_datetime(message.date)
                    if published and not (window.start <= published < window.end):
                        continue
                    username = str(channel_name).lstrip("@")
                    records.append(
                        Article(
                            platform=self.config.platform,
                            source_id=f"{username}:{message.id}",
                            published_at=published,
                            source_name=clean_text(getattr(channel, "title", channel_name)),
                            headline="",
                            content=clean_text(message.text),
                            summary="",
                            url=f"https://t.me/{username}/{message.id}",
                            author=str(channel_name),
                            search_id=str(channel_name),
                            metadata={
                                "views": message.views,
                                "forwards": message.forwards,
                            },
                        )
                    )
            return SourceResult(self.config.id, records)
        finally:
            await client.disconnect()
