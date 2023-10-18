import openai

# Set up your OpenAI API key
api_key = "YOUR_API_KEY"
openai.api_key = api_key

# Define a function to summarize an article
def summarize_article(article_text):
    prompt = f"Please summarize the following article:\n\n{article_text}\n\nSummary:"
    response = openai.Completion.create(
        engine="gpt-35-turbo",  # You can choose different engines based on your subscription
        prompt=prompt,
        max_tokens=100,  # Adjust the number of tokens for desired summary length
        temperature=0.6,  # Higher values make output more random, lower values make it more deterministic
        top_p=1.0,  # Probability mass to keep in the generated text
        frequency_penalty=0.0,  # Higher values make output more diverse
        presence_penalty=0.0,  # Higher values make output more focused
    )
    summary = response.choices[0].text.strip()
    return summary

# Example usage
article = """
    This is an example article. It contains some information that we want to summarize.
    The OpenAI API allows us to do this with ease.
    """

summary = summarize_article(article)
print(summary)


