import pandas as pd

def print_dataframe(df: pd.DataFrame):
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple pages
    pd.set_option('max_colwidth', None)  # Show full content of each cell

    formatters = {}
    len_max = df["content"].astype(str).str.len().max()
    formatters["content"] = lambda _: f"{_!s:<{len_max}s}"

    print(df.to_string(formatters=formatters, justify="left"))
    print()