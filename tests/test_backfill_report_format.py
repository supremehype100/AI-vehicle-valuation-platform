"""
Locks in the exact backfill progress-report format the task spec's example
shows (comma-thousands-separated, one metric per line):

    Total records: 125,430
    Already processed: 82,100
    Remaining: 43,330
    Processed this run: 10,000
    Successful: 9,842
    Failed: 158
"""
from backfill.backfill_conditions import print_report


def test_report_uses_comma_thousands_separators(capsys):
    print_report({
        "total": 125430,
        "already_processed": 82100,
        "remaining": 43330,
        "processed_this_run": 10000,
        "successful": 9842,
        "failed": 158,
    })
    out = capsys.readouterr().out
    assert "Total records: 125,430" in out
    assert "Already processed: 82,100" in out
    assert "Remaining: 43,330" in out
    assert "Processed this run: 10,000" in out
    assert "Successful: 9,842" in out
    assert "Failed: 158" in out
