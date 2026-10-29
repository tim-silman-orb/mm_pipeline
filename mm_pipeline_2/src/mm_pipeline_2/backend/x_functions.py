import requests
import pandas as pd


COMMON_COLUMNS = [
    "platform",
    "source_id",
    "published_at",
    "country",
    "language",
    "source_name",
    "author",
    "headline",
    "content",
    "url",
    "search_id",
]


def _headers(bearer_token):
    return {
        "Authorization": f"Bearer {bearer_token}"
    }


def pull_recent_x(
    bearer_token,
    query,
    max_results=10,
):
    """
    Pull recent X posts matching a search query.

    Returns source-native X data.
    """

    url = "https://api.x.com/2/tweets/search/recent"

    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "id,text,created_at,lang,author_id",
        "expansions": "author_id",
        "user.fields": "id,name,username",
    }

    response = requests.get(
        url,
        headers=_headers(bearer_token),
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    posts = data.get("data", [])

    users = {
        user["id"]: user
        for user in data.get("includes", {}).get("users", [])
    }

    rows = []

    for post in posts:

        author_id = post.get("author_id", "")
        user = users.get(author_id, {})

        rows.append({
            "id": post.get("id", ""),
            "created_at": post.get("created_at", ""),
            "lang": post.get("lang", ""),
            "author_id": author_id,
            "author_name": user.get("name", ""),
            "username": user.get("username", ""),
            "text": post.get("text", ""),
            "query": query,
        })

    return pd.DataFrame(rows)


def standardise_x(df):
    """
    Convert raw X output into the common
    media-monitoring schema.
    """

    urls = []

    for _, row in df.iterrows():

        username = row.get("username", "")
        post_id = row.get("id", "")

        if username and post_id:
            urls.append(
                f"https://x.com/{username}/status/{post_id}"
            )
        else:
            urls.append("")

    return pd.DataFrame({
        "platform": "x",
        "source_id": df["id"],
        "published_at": df["created_at"],
        "country": "",
        "language": df["lang"],
        "source_name": df["author_name"],
        "author": df["username"],
        "headline": "",
        "content": df["text"],
        "summary": "",
        "url": urls,
        "search_id": df["query"],
    })

