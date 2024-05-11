def add_text_tags(start_text, end_text, text):
    """
    Adds the specified start and end text tags to the given text.

    Args:
        - start_text (str): The start text tag.
        - end_text (str): The end text tag.
        - text (str): The text to which the tags should be added.

    Returns:
        - str: The text with the start and end text tags.
    """
    stop_sep = "\n\n" + "*" * 10 + "\n\n"
    start_sep = "\n\n" + "*" * 10 + "\n\n"
    text = f"{start_text}{start_sep}{text}{stop_sep}{end_text}"
    return text
