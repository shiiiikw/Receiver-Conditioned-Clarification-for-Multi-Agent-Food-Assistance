"""Offline checks of the receiver output parsing (no model, no network)."""
from rclup import roles as R
from rclup.fanout2 import RECEIVER_SCOPE, det_decision, make_receiver_canon, parse_receiver

COUNTIES = ", ".join(f'"{c}"' for c in ["ADAMS", "ALLEN", "LAKE", "MARION", "ST. JOSEPH"] * 20)
TRUNCATED = ('<tool_call>\n{"name": "report_domain_decision", "arguments": {"decisions": [{"candidate": 1, "needs": [{"need_id": "need-1", '
             '"location": {"counties": [' + COUNTIES + '], "cities": [], "zipcodes": []}, "schedule": "none",')


def test_complete_call_parses():
    text = ('<tool_call>\n{"name": "report_domain_decision", "arguments": {"decisions": [{"candidate": 1, "needs": [{"need_id": "need-1", '
            '"location": {"counties": ["Lake County"]}}], "obligations": []}]}}\n</tool_call>')
    args = parse_receiver(text)
    assert args and args["decisions"][0]["candidate"] == 1
    key, norm = make_receiver_canon(RECEIVER_SCOPE["logistics"])(args)
    assert norm["decisions"]["1"]["needs"]["need-1"]["location"]["counties"] == ["LAKE COUNTY"]
    # index normalisation of county literals happens in the deterministic admissible decision
    assert det_decision({"counties": ["Lake County"]}, "location", {"LAKE", "MARION"}) == {"counties": ["LAKE"]}


def test_truncated_call_is_salvaged():
    args = parse_receiver(TRUNCATED)
    assert args is not None, "token-limit truncation must not fail the invocation"
    key, norm = make_receiver_canon(RECEIVER_SCOPE["logistics"])(args)
    assert "1" in norm["decisions"]


def test_unknown_stays_unknown_and_is_an_obligation():
    text = '{"name": "report_domain_decision", "arguments": {"decisions": [{"candidate": 2, "needs": [{"need_id": "need-1", "schedule": "unknown"}], "obligations": []}]}}'
    key, norm = make_receiver_canon(RECEIVER_SCOPE["schedule"])(parse_receiver(text))
    assert norm["decisions"]["2"]["needs"]["need-1"]["schedule"] == R.UNKNOWN
    assert ("need-1", "schedule") in norm["decisions"]["2"]["obligations"]
