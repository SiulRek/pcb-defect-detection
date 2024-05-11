def wrap_text(text, width):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if (
            sum(len(word) for word in current_line)
            + len(current_line)
            - 1
            + len(word)
            + 1
            > width
        ):
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


if __name__ == "__main__":
    sample_text = "This is a 'long text' that needs to be 'wrapped_according' \n to the maximum number of characters per line provided."
    print(wrap_text(sample_text, 50))
