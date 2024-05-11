def process_start(process_title):
    msg = f"{'-'*100}\nRunned Task: {process_title}\n{'-'*100}"
    print(msg)


def process_end(additional_msg=""):
    msg = f"{'='*50} TASK EXECUTED SUCCESSFULLY {'='*50}"
    if additional_msg:
        msg += f"\n{additional_msg}"
    print(msg)
