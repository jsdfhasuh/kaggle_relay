"""Inspect an existing stopped P6 sandbox; --apply requires separate task authorization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def require_idle_account(database, auth, account_id):
    """Check all configured aliases of the explicitly selected account."""
    accounts = auth['kaggle_keys']
    matches = [row for row in accounts if row['id'] == account_id]
    if len(matches) != 1 or not matches[0].get('enabled', True):
        raise ValueError('selected test account is missing or disabled')
    owner = str(matches[0].get('username', '')).strip().casefold()
    if not owner:
        raise ValueError('selected test account has no configured owner')
    aliases = [row['id'] for row in accounts
               if str(row.get('username', '')).strip().casefold() == owner]
    query = ("select count(*) from jobs where kaggle_key_id in ("
             + ','.join('?' for _ in aliases)
             + ") and status not in ('complete','failed','canceled')")
    with sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True) as db:
        if db.execute(query, aliases).fetchone()[0]:
            raise ValueError('test account has an active task; defer without canceling it')
    return owner


def inspect(revision, account_id):
    root = Path('/docker_volume/kaggle_relay-p6')
    source, production = root/'source', Path('/docker_volume/kaggle_relay')
    if Path(__file__).resolve().parents[1] != source or root.resolve() != root:
        raise ValueError('unexpected sandbox path')
    if command('git', '-C', str(source), 'rev-parse', 'HEAD') != revision:
        raise ValueError('source differs from approved revision')
    if command('git', '-C', str(source), 'branch', '--show-current') != 'codex/p6-dino-cloud-onnx':
        raise ValueError('unexpected sandbox branch')
    if command('git', '-C', str(source), 'status', '--porcelain'):
        raise ValueError('dirty sandbox source; refuse overwrite')
    state = json.loads(command('docker', 'inspect', 'kaggle-relay-p6', '--format', '{{json .State}}'))
    if state['Status'] != 'exited':
        raise ValueError('sandbox must already be stopped; never interrupt live tasks')
    auth = json.loads((production/'relay-data/auth.json').read_text())
    owner = require_idle_account(production/'relay-data/relay.db', auth, account_id)
    sandbox_auth = json.loads((root/'state/auth.json').read_text())
    sandbox_owner = require_idle_account(root/'state/relay.db', sandbox_auth, account_id)
    if owner != sandbox_owner:
        raise ValueError('sandbox selected owner differs from production account configuration')
    for database, query in [
        (root/'state/relay.db', "select count(*) from jobs where status not in ('complete','failed','canceled')")]:
        with sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True) as db:
            if db.execute(query).fetchone()[0]:
                raise ValueError('test account or sandbox has an active task; defer without canceling it')
    return root, source, {
        'revision': revision, 'account_id': account_id, 'configured_owner': owner,
        'production_revision': command('git', '-C', str(production), 'rev-parse', 'HEAD'),
        'production_dirty': bool(command('git', '-C', str(production), 'status', '--porcelain')),
        'production_image': command('docker', 'inspect', 'kaggle_relay-kaggle-relay-1', '--format', '{{.Image}}'),
        'previous_image': command('docker', 'inspect', 'kaggle-relay-p6', '--format', '{{.Image}}')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--account-id', required=True, help='explicitly authorized test account')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, source, record = inspect(args.revision, args.account_id)
    if not args.apply:
        print(json.dumps({**record, 'status': 'READY', 'applied': False}))
        return
    os.umask(0o077)
    identifier = args.revision[:12]
    backup = root/('backup-calibration-'+identifier)
    context = root/('context-calibration-'+identifier)
    if backup.exists() or context.exists():
        raise ValueError('existing deployment attempt; inspect instead of replacing')
    backup.mkdir(mode=0o700)
    shutil.copytree(root/'state', backup/'state')
    shutil.copytree(root/'private', backup/'private')
    context.mkdir(mode=0o700)
    shutil.copytree(source/'app', context/'app', ignore=shutil.ignore_patterns('__pycache__'))
    base = 'kaggle-relay-p6-base:'+record['previous_image'].split(':')[1][:16]
    command('docker', 'tag', record['previous_image'], base)
    (context/'Dockerfile').write_text('FROM '+base+'\nCOPY app /app/app\nLABEL p6.revision="'+args.revision+'"\n')
    tag = 'kaggle-relay-p6:'+identifier
    subprocess.run(['docker', 'build', '--network=none', '--pull=false', '-t', tag, str(context)], check=True)
    # Repeat all guards after building; no other task may have started meanwhile.
    inspect(args.revision, args.account_id)
    previous = 'kaggle-relay-p6-before-'+identifier
    command('docker', 'rename', 'kaggle-relay-p6', previous)
    record.update(previous_container=previous, backup=str(backup), image=command('docker', 'image', 'inspect', tag, '--format', '{{.Id}}'))
    record['rollback'] = ('Stop only kaggle-relay-p6 after confirming its jobs are terminal; retain its state. '
        'Use '+record['previous_image']+' with a separate restored copy of '+str(backup/'state')+
        '. Do not overwrite live state or change production.')
    record_path = root/('deployment-calibration-'+identifier+'.json')
    record_path.write_text(json.dumps(record, indent=2))
    command('docker', 'run', '-d', '--name', 'kaggle-relay-p6', '--restart=no',
        '-p', '127.0.0.1:18006:8000', '-v', str(root/'state')+':/data',
        '-e', 'RELAY_STORAGE_DIR=/data', '-e', 'RELAY_AUTH_CONFIG=/data/auth.json',
        '-e', 'RELAY_WORKER_COUNT=1', '-e', 'RELAY_ACCOUNT_CONCURRENCY=1', tag,
        'sh', '-c', 'mkdir -p /tmp/kaggle-config && exec uvicorn app.main:app --host 0.0.0.0 --port 8000')
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((source/'app').glob('*.py'))}
    actual = json.loads(command('docker', 'exec', 'kaggle-relay-p6', 'python', '-c',
        "import pathlib,hashlib,json;print(json.dumps({str(p.relative_to('/app')):hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('/app/app').glob('*.py')}))"))
    if actual != hashes:
        raise ValueError('running container source differs from approved files; retain evidence')
    record.update(running_hashes=actual, actual_image=command('docker', 'inspect', 'kaggle-relay-p6', '--format', '{{.Image}}'))
    if record['actual_image'] != record['image']:
        raise ValueError('container image mismatch')
    record_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({'revision': args.revision, 'image': record['image'], 'applied': True,
                      'running_files_verified': True, 'production_changed': False}))


if __name__ == '__main__':
    main()
