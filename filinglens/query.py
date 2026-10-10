r"""Ask the database a question in SQL, straight from the terminal.

Usage (put the SQL in double quotes, text values inside it in single quotes):

    .\.venv\Scripts\python.exe -m filinglens.query "SELECT * FROM companies"
"""

import sys

import duckdb
import pandas as pd

from filinglens.config import DB_PATH


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        result = con.execute(sys.argv[1]).df()
    except duckdb.Error as exc:
        print(f"SQL error: {exc}")
        sys.exit(1)
    finally:
        con.close()

    if result.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 250):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
