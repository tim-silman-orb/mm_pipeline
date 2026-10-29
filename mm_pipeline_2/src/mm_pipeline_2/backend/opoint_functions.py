import time
import requests
import pandas as pd


def collect_opoint(
    token,
    stored_search_id,
    days_back=7,
    num_art=25000
):
    """
    Collect articles from the OPoint stored search API.

    Parameters
    ----------
    token : str
        OPoint API token.

    stored_search_id : int
        ID of the stored search to retrieve.

    days_back : int, default=7
        Number of days back from the current time to retrieve articles.

    num_art : int, default=25000
        Maximum number of articles requested from OPoint.

    Returns
    -------
    pandas.DataFrame
        Raw OPoint article data.
    """

    to_timestamp = int(time.time())

    from_timestamp = (
        to_timestamp
        - (days_back * 24 * 60 * 60)
    )

    response = requests.get(
        "https://api.opoint.com/storedsearch/feed/",
        headers={
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        },
        params={
            "from": from_timestamp,
            "to": to_timestamp,
            "format": "json",
            "num_art": num_art,
        },
    )

    response.raise_for_status()

    data = response.json()

    documents = (
        data
        .get("searchresult", {})
        .get("document", [])
    )

    # OPoint sometimes returns a single document
    # as a dictionary rather than a list.
    if isinstance(documents, dict):
        documents = [documents]

    df = pd.json_normalize(
        documents
    )

    # Keep only results associated with the
    # requested stored search.
    if "stored_search_id" in df.columns:
        df = df[
            df["stored_search_id"]
            .astype(str)
            == str(stored_search_id)
        ].copy()

    # Remove duplicate headlines while retaining
    # articles with missing headlines.
    if "header.text" in df.columns:

        df["header.text"] = (
            df["header.text"]
            .astype("string")
            .str.strip()
        )

        has_header = (
            df["header.text"].notna()
            & df["header.text"].ne("")
        )

        df_with_header = (
            df[has_header]
            .drop_duplicates(
                subset=["header.text"],
                keep="first"
            )
        )

        df_without_header = (
            df[~has_header]
        )

        df = pd.concat(
            [
                df_with_header,
                df_without_header
            ],
            ignore_index=True
        )

    return df


def standardise_opoint(df):
    """
    Convert raw OPoint output into the
    common media-monitoring schema.
    """

    content = (
        df["header.text"]
        .fillna("")
        .astype(str)

        + "\n\n"

        + df["summary.text"]
        .fillna("")
        .astype(str)

        + "\n\n"

        + df["body.text"]
        .fillna("")
        .astype(str)
    ).str.strip()

    return pd.DataFrame({
        "platform": "opoint",

        "source_id":
            df["id_article"],

        "published_at":
            df["local_time.text"],

        "country":
            df["countryname"],

        "language":
            df["language.text"],

        "source_name":
            df["first_source.name"],

        "author":
            df["author"],

        "headline":
            df["header.text"],

        "content":
            content,

        "summary":"",

        "url":
            df["url"],

        "search_id":
            df["stored_search_id"],
    })

