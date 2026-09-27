"""Oracle-only isolated P6 acceptance deployment. Production is read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    production = Path('/docker_volume/kaggle_relay')
    root = Path('/docker_volume/kaggle_relay-p6')
    if source != root/'source' or production.resolve() != production or root.is_symlink():
        raise ValueError('unexpected deployment paths')
    def command(*args):
        return subprocess.check_output(args, text=True).strip()
    if command('git', '-C', str(source), 'rev-parse', 'HEAD') != args.revision:
        raise ValueError('unreviewed source revision')
    if command('git', '-C', str(source), 'status', '--porcelain'):
        raise ValueError('dirty source; refuse deployment')
    if (root/'state').exists():
        raise ValueError('existing sandbox state; inspect and resume explicitly')
    if command('git', '-C', str(production), 'status', '--porcelain'):
        raise ValueError('production is dirty; preserve and inspect')
    db = sqlite3.connect('file:'+str(production/'relay-data/relay.db')+'?mode=ro', uri=True)
    active = db.execute("select count(*) from jobs where kaggle_key_id='first' and status not in ('complete','failed','canceled')").fetchone()[0]
    db.close()
    if active:
        raise ValueError('test account has active production tasks')
    image = command('docker', 'inspect', 'kaggle_relay-kaggle-relay-1', '--format', '{{.Image}}')
    auth = json.loads((production/'relay-data/auth.json').read_text())
    keys = [key for key in auth['kaggle_keys'] if key['id'] == 'first']
    if len(keys) != 1 or keys[0].get('username') != 'jsdfhasuh' or keys[0].get('config_dir'):
        raise ValueError('explicit test credential scope unavailable')
    state, private, context = root/'state', root/'private', root/'image-context'
    for folder in (state, private, context):
        folder.mkdir(mode=0o700)
    token = secrets.token_urlsafe(48)
    test_auth = {'kaggle_keys': keys, 'relay_tokens': [{'id': 'p6-test', 'token': token,
        'allowed_kaggle_key_ids': ['first'], 'can_view_keys': True}]}
    (state/'auth.json').write_text(json.dumps(test_auth))
    os.chmod(state/'auth.json', 0o600)
    (private/'relay.json').write_text(json.dumps({'base_url': 'http://127.0.0.1:18006', 'api_token': token, 'enabled': True}))
    key = keys[0]
    (private/'kaggle.json').write_text(json.dumps({'username': key['username'], 'key': key.get('api_token') or key.get('key')}))
    for path in private.iterdir():
        os.chmod(path, 0o600)
    shutil.copytree(source/'app', context/'app', ignore=shutil.ignore_patterns('__pycache__'))
    # Dockerfile FROM interprets sha256:... as a registry name; use an exact local tag.
    base_tag = 'kaggle-relay-p6-base:'+image.split(':')[1][:16]
    subprocess.run(['docker', 'tag', image, base_tag], check=True)
    if command('docker', 'image', 'inspect', base_tag, '--format', '{{.Id}}') != image:
        raise ValueError('base image identity changed')
    (context/'Dockerfile').write_text('FROM '+base_tag+'\nCOPY app /app/app\nLABEL p6.revision="'+args.revision+'"\n')
    tag = 'kaggle-relay-p6:'+args.revision[:12]
    subprocess.run(['docker', 'build', '--network=none', '--pull=false', '-t', tag, str(context)], check=True)
    subprocess.run(['docker', 'run', '-d', '--name', 'kaggle-relay-p6', '--restart=no',
        '-p', '127.0.0.1:18006:8000', '-v', str(state)+':/data',
        '-e', 'RELAY_STORAGE_DIR=/data', '-e', 'RELAY_AUTH_CONFIG=/data/auth.json',
        '-e', 'RELAY_WORKER_COUNT=1', '-e', 'RELAY_ACCOUNT_CONCURRENCY=1',
        tag, 'sh', '-c', 'mkdir -p /tmp/kaggle-config && exec uvicorn app.main:app --host 0.0.0.0 --port 8000'], check=True)
    record = {'source_revision': args.revision, 'base_image': image,
        'production_revision': command('git', '-C', str(production), 'rev-parse', 'HEAD'),
        'image': command('docker', 'inspect', 'kaggle-relay-p6', '--format', '{{.Image}}'),
        'source_hashes': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((source/'app').glob('*.py'))},
        'rollback': 'docker stop kaggle-relay-p6; keep state/private/source and image for diagnosis; production unchanged'}
    (root/'deployment.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'revision': args.revision, 'image': record['image'], 'production_unchanged': True}))


if __name__ == '__main__':
    main()
