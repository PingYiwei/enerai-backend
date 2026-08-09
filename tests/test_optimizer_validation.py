from app.modules.optimizer.validation import decode_csv, validate_dataset


def test_pump_dataset_reports_exact_invalid_rows() -> None:
    content = (
        b"time,flow,eff_pump,head\n2026-01-01 00:00:00,10,0.8,12\n2026-01-01 00:01:00,-1,1.2,12\n"
    )
    header, rows = decode_csv(content)
    rules = validate_dataset("pump", header, rows)
    by_id = {rule.rule_id: rule for rule in rules}
    assert by_id["pump_efficiency"].invalid_rows == [3]
    assert by_id["pump_flow"].invalid_rows == [3]


def test_chiller_dataset_requires_exact_semantic_columns() -> None:
    header, rows = decode_csv(b"time,value\n2026-01-01 00:00:00,1\n")
    rules = validate_dataset("chiller", header, rows)
    assert rules[0].rule_id == "required_fields"
    assert rules[0].passed is False
    assert len(rules) == 1
