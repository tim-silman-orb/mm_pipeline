import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

def clean_value(value: Any) -> str:
    """Convert a value to a stripped string."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def extract_json(text: str) -> Any:
    """Parse JSON even if the model adds a code fence."""

    if not text or not text.strip():
        raise ValueError("The model returned an empty response.")

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    object_match = re.search(
        r"\{.*\}",
        cleaned,
        re.DOTALL
    )

    if object_match:
        return json.loads(object_match.group(0))

    raise ValueError(
        "Could not parse JSON from response:\n"
        f"{cleaned[:2000]}"
    )



def call_bedrock_json(
    bedrock_runtime: Any,
    *,
    prompt: str,
    model_id: str,
    max_tokens: int = 4000,
    max_retries: int = 4,
    temperature: float = 0.0,
    operation_name: str = "Bedrock") -> Any:
    """Call Bedrock and return parsed JSON."""

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )

            stop_reason = response.get("stopReason", "")

            print(
                "Stop reason:",
                stop_reason
            )

            if stop_reason == "max_tokens":
                raise RuntimeError(
                    f"{operation_name} response reached the output-token limit."
                )
            
            usage = response.get("usage", {})

            print(
                f"Tokens: "
                f"{usage.get('inputTokens', 0):,} input, "
                f"{usage.get('outputTokens', 0):,} output"
            )

            content_blocks = (
                response
                .get("output", {})
                .get("message", {})
                .get("content", [])
            )

            response_text = "".join(
                block.get("text", "")
                for block in content_blocks
                if "text" in block
            )

            return extract_json(response_text)

        except Exception as error:
            last_error = error

            print(
                f"Bedrock attempt {attempt}/{max_retries} "
                f"failed: {error}"
            )

            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        "Bedrock request failed after all retries."
    ) from last_error


def safe_save_csv(dataframe, output_path):
    import os
    import tempfile
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{output_path.stem}_",
            dir=output_path.parent,
            delete=False,
            encoding="utf-8-sig",
            newline="",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        dataframe.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )

        try:
            os.replace(
                temporary_path,
                output_path,
            )

        except PermissionError:
            print(
                f"Atomic replace blocked for {output_path.name}; "
                "falling back to direct write."
            )

            dataframe.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )

