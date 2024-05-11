import openai

from local.private.keys import OPENAI_KEY


def make_query(query_message, max_response_tokens=6000):
    openai.api_key = OPENAI_KEY

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a Python and Vision AI developer."},
            {"role": "user", "content": query_message},
        ],
        max_tokens=max_response_tokens,
    )

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    response_message = make_query(
        "Explain unit testing in Python. Tell all you know please"
    )
    print(response_message)
