
def add_text_tags(start_text, end_text, text):
    stop_sep = "\n\n" + "*" * 10 + "\n\n"
    start_sep = "\n\n" + "*" * 10 + "\n\n"
    text = f"{start_text}{start_sep}{text}{stop_sep}{end_text}"
    return text