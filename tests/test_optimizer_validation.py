from app.modules.optimizer.validation import count_valid_rows, decode_csv, validate_dataset


def test_pump_dataset_reports_exact_invalid_rows() -> None:
    content = (
        b"time,flow,eff_pump,head\n2026-01-01 00:00:00,10,0.8,12\n2026-01-01 00:01:00,-1,1.2,12\n"
    )
    header, rows = decode_csv(content)
    rules = validate_dataset("pump", header, rows)
    assert all(rule.passed for rule in rules)
    assert count_valid_rows("pump", header, rows) == 2


def test_chiller_dataset_requires_exact_semantic_columns() -> None:
    header, rows = decode_csv(b"time,value\n2026-01-01 00:00:00,1\n")
    rules = validate_dataset("chiller", header, rows)
    assert rules[0].rule_id == "required_fields"
    assert rules[0].passed is False
    assert len(rules) == 1


def test_chiller_operating_rules_are_warnings() -> None:
    content = (
        b"time,t_chw_ret,t_chw_sup,t_cw_sup,t_cw_ret,flow_chw,flow_cw,q_cool,q_reject,"
        b"load_pct,t_evap,t_cond\n"
        b"2026-01-01 00:00:00,6,5,28,29,10,12,100,120,50,4.9,29.1\n"
    )
    header, rows = decode_csv(content)
    rules = validate_dataset("chiller", header, rows)
    failed = [rule for rule in rules if not rule.passed]
    assert failed
    assert all(rule.severity == "warning" for rule in failed)
    assert count_valid_rows("chiller", header, rows) == 1


def test_cooling_tower_uses_device_specific_rules() -> None:
    content = (
        b"time,t_cw_in,t_cw_out,t_wb_wea,air_water_ratio\n"
        b"2026-01-01 00:00:00,32,28,25,1\n"
        b"2026-01-01 00:01:00,32,29,25,3\n"
    )
    header, rows = decode_csv(content)
    rules = validate_dataset("cooling_tower", header, rows)
    by_id = {rule.rule_id: rule for rule in rules}
    assert by_id["cooling_tower_air_water_ratio"].invalid_rows == [3]
    assert by_id["cooling_tower_air_water_ratio"].severity == "warning"
