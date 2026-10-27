from __future__ import annotations

import logging
import time
from typing import Any

import requests

from mm_pipeline.exceptions import SourceCollectionError


class HttpClient:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        timeout_seconds: int = 30,
        max_attempts: int = 3,
        user_agent: str = "mm-pipeline/0.1",
    ) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=kwargs.pop("timeout", self.timeout_seconds),
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "HTTP attempt %s/%s failed for %s: %s",
                    attempt,
                    self.max_attempts,
                    url,
                    exc,
                )
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise SourceCollectionError(f"HTTP request failed for {url}: {last_error}")
