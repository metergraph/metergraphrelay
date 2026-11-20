import json

import pytest

from metergraphrelay.filtering import ExclusionPolicy, filter_jsonl


def policy(**condition):
    return {'version': 1, 'rules': [{'id': 'direct-coverage', 'conditions': {
        'provider': 'anthropic', 'route': 'audit', 'timestamp_gte': '2026-01-01T00:00:00Z',
        **condition}}]}


def row(**changes):
    return {'provider': 'anthropic', 'route': 'audit', 'ts': '2026-01-01T00:00:00Z', **changes}


def test_empty_default_and_exact_inclusive_cutover():
    assert ExclusionPolicy.from_dict(None).evaluate(row()) is None
    p = ExclusionPolicy.from_dict(policy())
    assert p.evaluate(row()) == 'direct-coverage'
    assert p.evaluate(row(ts='2025-12-31T23:59:59Z')) is None
    assert p.evaluate(row(provider='openai')) is None
    assert p.evaluate(row(route='another')) is None


def test_tag_conditions_and_upper_bound():
    p = ExclusionPolicy.from_dict(policy(tags={'team': 'test'}, timestamp_lt='2026-02-01T00:00:00Z'))
    assert p.evaluate(row(tags={'team': 'test', 'other': 'allowed'}))
    assert p.evaluate(row(tags={'team': 'other'})) is None
    assert p.evaluate(row()) is None
    assert p.evaluate(row(tags={'team': 'test'}, ts='2026-02-01T00:00:00Z')) is None


@pytest.mark.parametrize('bad', [
    {'version': 2, 'rules': []}, {'version': 1, 'rules': 'bad'},
    policy(timestamp_gte='yesterday'), policy(timestamp_gte='2026-01-01'),
    policy(timestamp_lt='2025-01-01T00:00:00Z'), policy(expression='eval(x)'),
    policy(tags={'team': ['not-a-string']}),
    {'version': 1, 'rules': [{'id': 'all', 'conditions': {'provider': 'anthropic'}}]},
])
def test_invalid_or_broad_rules_are_rejected(bad):
    with pytest.raises(ValueError):
        ExclusionPolicy.from_dict(bad)


def test_snapshot_does_not_change_when_input_is_mutated():
    source = policy()
    p = ExclusionPolicy.from_dict(source)
    source['rules'][0]['conditions']['route'] = 'changed'
    assert p.evaluate(row())
    assert ExclusionPolicy.from_dict(p.to_dict()).digest == p.digest


def test_duplicate_rule_ids_and_invalid_row_time_fail():
    source = policy()
    source['rules'] *= 2
    with pytest.raises(ValueError):
        ExclusionPolicy.from_dict(source)
    with pytest.raises(ValueError):
        ExclusionPolicy.from_dict(policy()).evaluate(row(ts='bad'))


def test_filter_preview_keeps_rows_and_reports_matches(tmp_path):
    source, output = tmp_path/'in.jsonl', tmp_path/'out.jsonl'
    rows = [row(), row(provider='openai')]
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    p = ExclusionPolicy.from_dict(policy())
    r = filter_jsonl(source, output, p, preview=True)
    assert (r.included, r.excluded, r.matched_by_rule) == (2, 0, {'direct-coverage': 1})
    assert [json.loads(x) for x in output.read_text().splitlines()] == rows
    r = filter_jsonl(source, output, p)
    assert (r.included, r.excluded) == (1, 1)
    assert json.loads(output.read_text())['provider'] == 'openai'


def test_invalid_input_preserves_output_and_never_overwrites_source(tmp_path):
    source, output = tmp_path/'in.jsonl', tmp_path/'out.jsonl'
    source.write_text(json.dumps(row())+'\nnot json\n')
    output.write_text('original')
    p = ExclusionPolicy.from_dict(policy())
    with pytest.raises(ValueError):
        filter_jsonl(source, output, p)
    assert output.read_text() == 'original'
    with pytest.raises(ValueError):
        filter_jsonl(source, source, p)


def test_cli_requires_no_credentials_or_network(tmp_path, capsys):
    from metergraphrelay.cli import main
    source, output, rules = tmp_path/'in.jsonl', tmp_path/'out.jsonl', tmp_path/'rules.json'
    source.write_text(json.dumps(row())+'\n')
    rules.write_text(json.dumps(policy()))
    assert main(['filter', str(source), '--rules', str(rules), '--output', str(output), '--preview']) == 0
    assert json.loads(capsys.readouterr().out)['included'] == 1
    assert main(['filter', str(source), '--rules', str(rules), '--output', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['excluded'] == 1


def test_bounded_row_size(tmp_path, monkeypatch):
    import metergraphrelay.filtering as filtering
    source, output = tmp_path/'in.jsonl', tmp_path/'out.jsonl'
    source.write_text(json.dumps(row())+'\n')
    monkeypatch.setattr(filtering, 'MAX_LINE_BYTES', 8)
    with pytest.raises(ValueError):
        filter_jsonl(source, output)
    assert not output.exists()
