from pathlib import Path
from extract import extract, ExtractionError

POSTINGS_DIR = Path("data/postings")
LOGS_DIR = Path("data/logs")

LOGS_DIR.mkdir(parents=True, exist_ok=True)

def run_batch():
    posting_files = sorted(POSTINGS_DIR.glob("*.txt"))

    for posting_file in posting_files:
        job_text = posting_file.read_text(encoding="utf-8")
        log_file = LOGS_DIR / f"{posting_file.stem}.log"

        try:
            result = extract(job_text)
            log_file.write_text(
                "STATUS: success\n" + result.model_dump_json(indent=2),
                encoding="utf-8",
            )
            print(f"{posting_file.name}: OK")
        except ExtractionError as e:
            log_file.write_text(
                f"STATUS: failed\nERROR: {e}",
                encoding="utf-8",
            )
            print(f"{posting_file.name}: FAILED — {e}")

if __name__ == "__main__":
    run_batch()