def check_age(age):
    if age < 0:
        # Raising an exception with a direct simple string message
        raise ValueError("Age cannot be negative.")
    elif age > 120:
        # Raising an exception with a formatted string message using an f-string
        raise ResourceWarning(f"Age {age} is unusually high for a human.")
    else:
        print(f"Age {age} is valid.")