import sqlite3

import pytest

from scripts.p6_calibration_sandbox import require_idle_account


def test_selected_account_and_aliases_are_guarded(tmp_path):
    path = tmp_path / 'relay.db'
    with sqlite3.connect(path) as db:
        db.execute('create table jobs (kaggle_key_id text, status text)')
        db.execute("insert into jobs values ('first', 'waiting_kernel')")
    auth = {'kaggle_keys': [
        {'id': 'first', 'username': 'unrelated'},
        {'id': 'cjq', 'username': 'iiiitsme'},
        {'id': 'alias', 'username': 'IIIITSME'},
    ]}
    assert require_idle_account(path, auth, 'cjq') == 'iiiitsme'
    with sqlite3.connect(path) as db:
        db.execute("insert into jobs values ('alias', 'waiting_kernel')")
    with pytest.raises(ValueError, match='active task'):
        require_idle_account(path, auth, 'cjq')
    with sqlite3.connect(path) as db:
        db.execute("update jobs set status='complete' where kaggle_key_id='alias'")
    assert require_idle_account(path, auth, 'cjq') == 'iiiitsme'


@pytest.mark.parametrize('rows', [[], [{'id': 'cjq', 'enabled': False}],
                                  [{'id': 'cjq', 'username': ''}]])
def test_invalid_selection_refused_before_database_access(tmp_path, rows):
    with pytest.raises(ValueError):
        require_idle_account(tmp_path/'missing.db', {'kaggle_keys': rows}, 'cjq')
    assert not (tmp_path/'missing.db').exists()
