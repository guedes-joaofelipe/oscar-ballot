import pydantic
from pydantic import Field


class VoterResponse(pydantic.BaseModel):
    """
    Response model for an expert voter.
    """

    category_id: str = Field(description="The category identifier (id)")
    nominee_id: str = Field(description="The nominee identifier (id)")
    explanation: str = Field(description="The explanation for why you chose the nominee")


def get_voter_prompt(
    categories: list[dict], system_prompt_repetitions: int = 1, user_prompt_repetitions: int = 1, imdb_metadata: dict = None
) -> dict[str, str]:
    """
    Build prompts for an expert voter.

    Parameters
    ----------
    categories : list[dict]
        Categories and nominees that should be evaluated by the voter.

    Returns
    -------
    dict[str, str]
        Prompt dictionary with `system_prompt` and `user_prompt`.
    """

    system_prompt = ""

    for i in range(system_prompt_repetitions):
        if i > 0:
            system_prompt += "\n" + "--------------------------------" + "\n"

        system_prompt += """
        You are a movie expert who helps people vote for the best movies and actors in the Academy Awards.
        You will be given a list of categories and the nominees for each category.
        You will need to help the user vote for the best nominee in each category.
        The user will provide you with the following information:
        - The category identifier (id)
        - The category name
        - The category description
        - The name of the nominee (either a movie or an actor / actress)
        - The nominee's identifier (id)
        - (if available) Metadata from IMDb about the nominees

        You must return the best nominee in each category in the following json format:
        {"votes":[{"category_id":"<category_id>","nominee_id":"<nominee_id>","explanation":"<explanation>"}]}

        The explanation should include your reasoning for choosing this nominee, indicating what aspects of the nominee's
        performance you considered most important. If needed, you can browse the web to find more information about the
        nominees and categories to help you make your decision, specially on other awards websites such as:
        - Golden Globes: https://goldenglobes.com/
        - BAFTA: https://www.bafta.org/
        - Critics' Choice Awards: https://www.criticschoice.org/
        - SAG Awards: https://www.sagawards.org/
        - Cannes Film Festival: https://www.festival-cannes.com/
        """

    user_prompt = ""
    for i in range(user_prompt_repetitions):
        if i > 0:
            user_prompt += "\n" + "--------------------------------" + "\n"

        user_prompt += f"""
        Help me vote for the best nominee in the following categories:
        """

        category_prompt = ""
        for category in categories:
            nominees_prompt = ""
            for nominee in category["nominees"]:
                nominees_prompt += f"""
                <NOMINEE>
                    <NOMINEE_ID>{nominee['id']}</NOMINEE_ID>
                    <NAME>{nominee.get('name', '')}</NAME>
                    <MOVIE>{nominee.get('movie', '')}</MOVIE>
                </NOMINEE>
                """

            category_prompt += f"""
            <CATEGORY>
                <CATEGORY_ID>{category['id']}</CATEGORY_ID>
                <NAME>{category['name']}</NAME>
                <DESCRIPTION>
                    {category['description']}
                </DESCRIPTION>
                <NOMINEES>
                    {nominees_prompt}
                </NOMINEES>
            </CATEGORY>
            """

        user_prompt += category_prompt

    if imdb_metadata:
        imdb_metadata_prompt = "To help you make your decision, I provide you with the following " \
            "metadata from IMDb about the nominees:"
        for nominee_id, metadata in imdb_metadata.items():
            imdb_metadata_prompt += f"""
            <IMDB_METADATA>
                <NOMINEE_ID>{nominee_id}</NOMINEE_ID>
                <TITLE>{metadata['title']}</TITLE>
                <YEAR>{metadata['year']}</YEAR>
                <SYNOPSIS>{metadata['synopsis']}</SYNOPSIS>
                <DIRECTORS>{metadata['directors']}</DIRECTORS>
                <CAST>{metadata['cast']}</CAST>
                <IMDB_RATING>{metadata['imdb_rating']}</IMDB_RATING>
            </IMDB_METADATA>
            """
        user_prompt += imdb_metadata_prompt

    return {
        "system_prompt": system_prompt.strip(),
        "user_prompt": user_prompt.strip(),
    }
