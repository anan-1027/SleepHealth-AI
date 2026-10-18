"""
Download the public-use NHANES files this project needs.

Run before src/prepare_nhanes.py. Safe to re-run (existing files are kept).

DATA CHOICE — why 2017-March 2020 pre-pandemic (P_*.XPT) and not a newer cycle:
    * The NHANES 2019-2020 cycle was SUSPENDED in March 2020 after only 18 of 30
      survey locations were visited, so it was never nationally representative on
      its own and is not released as standalone public-use files.
    * NCHS therefore merged 2019-Mar 2020 into the 2017-2018 cycle and published a
      combined "2017-March 2020 pre-pandemic" sample (May 2021) with special
      weights (WTINTPRP / WTMECPRP). This SUPERSEDES the 2017-2018 (cycle J) files.
    * The next complete cycle, 2021-Aug 2023, straddles the pandemic and uses a new
      sample design, so mixing it in would require careful weighting and could
      confound the sleep-disease associations studied here.

    Using the pre-pandemic release raises our adult cohort from 5,088 to 8,282.
"""
import os
import urllib.request

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

# P_* = the 2017-March 2020 pre-pandemic combined sample
FILES = ["P_DEMO.XPT", "P_BMX.XPT", "P_SLQ.XPT", "P_DIQ.XPT", "P_BPQ.XPT", "P_MCQ.XPT"]

XPORT_MAGIC = b"HEADER RECORD*"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw")


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for f in FILES:
        path = os.path.join(RAW_DIR, f)
        if not os.path.exists(path):
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(BASE + f, path)
        with open(path, "rb") as fh:
            head = fh.read(14)
        # Guard against the failure mode this project already hit once: the CDC
        # site serves an HTML error page at bad URLs, and pandas would later fail
        # with a confusing "not an XPORT file" error. Fail loudly and early here.
        if head != XPORT_MAGIC:
            raise SystemExit(
                f"{f} is not a valid SAS XPORT file (head={head!r}).\n"
                f"It is probably an HTML error page — check the URL or delete the file.")
        print(f"  ok {f:14s} {os.path.getsize(path):>9,} bytes")


if __name__ == "__main__":
    main()
