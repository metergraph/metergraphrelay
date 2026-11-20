"""Caller-supplied exclusions for normalized records. No network or policy discovery."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

MAX_RULES = 100
MAX_LINE_BYTES = 16 * 1024 * 1024


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError('timestamps must be timezone-aware ISO 8601 strings')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('timestamps must be timezone-aware ISO 8601 strings') from None
    if result.tzinfo is None:
        raise ValueError('timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError('rule values must be nonempty strings of at most 512 characters')
    return value


@dataclass(frozen=True)
class ExclusionPolicy:
    """An immutable, validated snapshot; rules OR together, conditions AND together."""
    _json: str

    @classmethod
    def from_dict(cls, value: Any = None) -> ExclusionPolicy:
        if value is None:
            value = {'version': 1, 'rules': []}
        if not isinstance(value, dict) or set(value) != {'version', 'rules'} or type(value['version']) is not int or value['version'] != 1:
            raise ValueError('expected exclusion policy version 1 with rules')
        rules = value['rules']
        if not isinstance(rules, list) or len(rules) > MAX_RULES:
            raise ValueError('rules must be a list of at most 100 rules')
        seen = set()
        normalized = []
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {'id', 'conditions'}:
                raise ValueError('each rule needs id and conditions')
            name = _text(rule['id'])
            if name in seen:
                raise ValueError('rule IDs must be unique')
            seen.add(name)
            c = rule['conditions']
            required = {'provider', 'route', 'timestamp_gte'}
            if not isinstance(c, dict) or not required <= set(c) or set(c) - (required | {'timestamp_lt', 'tags'}):
                raise ValueError('conditions require provider, route and timestamp_gte; only timestamp_lt and tags are optional')
            c = dict(c)
            _text(c['provider'])
            _text(c['route'])
            c['timestamp_gte'] = _time(c['timestamp_gte']).isoformat()
            if 'timestamp_lt' in c:
                c['timestamp_lt'] = _time(c['timestamp_lt']).isoformat()
                if c['timestamp_lt'] <= c['timestamp_gte']:
                    raise ValueError('timestamp_lt must be after timestamp_gte')
            if 'tags' in c:
                if not isinstance(c['tags'], dict) or len(c['tags']) > 10:
                    raise ValueError('tags must contain at most 10 exact string pairs')
                for k, v in c['tags'].items():
                    _text(k)
                    _text(v)
            normalized.append({'id': name, 'conditions': c})
        return cls(json.dumps({'version': 1, 'rules': normalized}, sort_keys=True, separators=(',', ':')))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._json.encode()).hexdigest()

    def to_dict(self) -> dict:
        return json.loads(self._json)

    def evaluate(self, row: dict) -> str | None:
        """Return the first matching rule ID, or None. Never evaluate arbitrary code."""
        for rule in self.to_dict()['rules']:
            c = rule['conditions']
            if row.get('provider') != c['provider'] or row.get('route') != c['route']:
                continue
            ts = _time(row.get('ts'))
            if ts < _time(c['timestamp_gte']) or ('timestamp_lt' in c and ts >= _time(c['timestamp_lt'])):
                continue
            tags = row.get('tags') or {}
            if not isinstance(tags, dict) or any(tags.get(k) != v for k, v in c.get('tags', {}).items()):
                continue
            return rule['id']
        return None


@dataclass(frozen=True)
class FilterResult:
    included: int
    excluded: int
    matched_by_rule: dict[str, int]
    policy_digest: str


def filter_jsonl(source, output, policy: ExclusionPolicy | dict | None = None, *, preview=False) -> FilterResult:
    """Stream normalized JSONL to another file. Preview retains every valid row.

    Invalid rows abort without replacing output. File processing performs no uploads,
    checkpoint changes or rule fetching. The caller owns retention and policy selection.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('input and output must be different files')
    policy = policy if isinstance(policy, ExclusionPolicy) else ExclusionPolicy.from_dict(policy)
    included = excluded = 0
    matched: Counter = Counter()
    temporary = None
    try:
        with source.open('rb') as inp, tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as out:
            temporary = Path(out.name)
            while line := inp.readline(MAX_LINE_BYTES + 1):
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError('JSONL row exceeds 16 MiB limit')
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    raise ValueError('invalid normalized JSONL record') from None
                if not isinstance(row, dict):
                    raise ValueError('normalized JSONL records must be objects')
                rule = policy.evaluate(row)
                if rule:
                    matched[rule] += 1
                if rule and not preview:
                    excluded += 1
                else:
                    included += 1
                    out.write(line if line.endswith(b'\n') else line + b'\n')
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return FilterResult(included, excluded, dict(matched), policy.digest)
