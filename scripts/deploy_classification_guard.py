"""Scoped Oracle deployment of the reviewed classification GPU guard."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import urllib.request


ROOT = Path("/docker_volume/kaggle_relay")
CONTAINER = "kaggle_relay-kaggle-relay-1"


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def active_jobs():
    with sqlite3.connect("file:" + str(ROOT / "relay-data/relay.db") + "?mode=ro", uri=True) as db:
        return dict(db.execute("select status,count(*) from jobs where status not in ('complete','failed','canceled') group by status"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-revision", default="")
    args = parser.parse_args()
    if ROOT.resolve() != ROOT or ROOT.is_symlink():
        raise ValueError("Unexpected deployment root")
    state = json.loads(command("docker", "inspect", CONTAINER))[0]
    branch = command("git", "-C", str(ROOT), "branch", "--show-current")
    revision = command("git", "-C", str(ROOT), "rev-parse", "HEAD")
    dirty = command("git", "-C", str(ROOT), "status", "--porcelain")
    active = active_jobs()
    print(json.dumps({"branch": branch, "revision": revision, "dirty": bool(dirty),
                      "active_jobs": active, "image": state["Image"]}), flush=True)
    if not args.deploy_revision:
        return
    if branch != "main" or dirty or revision != args.deploy_revision or active:
        raise ValueError("Deployment preflight failed; nothing restarted")
    files = state["Config"]["Labels"]["com.docker.compose.project.config_files"].split(",")
    if not files or files[0] != str(ROOT / "docker-compose.yml") or any(not Path(f).is_file() for f in files):
        raise ValueError("Unknown Compose deployment configuration")
    backup = ROOT.parent / "kaggle_relay-deploy-backups" / ("classification-" + revision[:12] + "-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    backup.mkdir(mode=0o700)
    for index, file in enumerate(files):
        shutil.copy2(file, backup / f"compose-{index}.yml")
    for filename in (".env", "relay-data/auth.json"):
        path = ROOT / filename
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    with sqlite3.connect(ROOT / "relay-data/relay.db") as source, sqlite3.connect(backup / "relay.db") as target:
        source.backup(target)
    previous_tag = "kaggle-relay-rollback:classification-" + revision[:12]
    command("docker", "tag", state["Image"], previous_tag)
    context = backup / "context"
    context.mkdir()
    shutil.copy2(ROOT / "app/yolo_gpu_guard.py", context / "yolo_gpu_guard.py")
    (context / "Dockerfile").write_text("FROM " + previous_tag + "\nCOPY yolo_gpu_guard.py /app/app/yolo_gpu_guard.py\n")
    tag = "kaggle-relay-reviewed:classification-" + revision[:12]
    subprocess.run(["docker", "build", "--pull=false", "-t", tag, str(context)], check=True)
    expected = hashlib.sha256((context / "yolo_gpu_guard.py").read_bytes()).hexdigest()
    check = "import hashlib;print(hashlib.sha256(open('/app/app/yolo_gpu_guard.py','rb').read()).hexdigest())"
    if command("docker", "run", "--rm", "--network=none", "--entrypoint", "python", tag, "-c", check) != expected:
        raise RuntimeError("Built image source mismatch")
    override = backup / "image.override.yml"
    override.write_text("services:\n  kaggle-relay:\n    image: " + tag + "\n")
    rollback_override = backup / "rollback.override.yml"
    rollback_override.write_text("services:\n  kaggle-relay:\n    image: " + previous_tag + "\n")
    base = ["docker", "compose", "--project-name", "kaggle_relay"]
    for file in files:
        base.extend(["-f", file])
    tail = ["up", "-d", "--no-build", "--force-recreate", "kaggle-relay"]
    rollback = base + ["-f", str(rollback_override)] + tail
    record = {"source_revision": revision, "previous_image": state["Image"], "image": tag,
              "guard_sha256": expected, "rollback": rollback, "backup": str(backup), "deployed": False}
    record_path = backup / "deployment.json"
    record_path.write_text(json.dumps(record, indent=2))
    if active_jobs():
        raise RuntimeError("A job arrived during build; deployment deferred")
    subprocess.run(base + ["-f", str(override)] + tail, check=True, cwd=ROOT)
    try:
        if command("docker", "exec", CONTAINER, "python", "-c", check) != expected:
            raise RuntimeError("Running source mismatch")
        auth = json.loads((ROOT / "relay-data/auth.json").read_text())
        token = next(item["token"] for item in auth["relay_tokens"] if item.get("token") and not item.get("disabled", False))
        healthy = False
        for _ in range(30):
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/v1/health", headers={"Authorization": "Bearer " + token})
                with urllib.request.urlopen(req, timeout=5) as response:
                    healthy = json.load(response).get("status") == "ok"
                if healthy:
                    break
            except Exception:
                time.sleep(1)
        if not healthy:
            raise RuntimeError("Authenticated health check failed")
        record.update(deployed=True, running_image=command("docker", "inspect", CONTAINER, "--format", "{{.Image}}"))
        record_path.write_text(json.dumps(record, indent=2))
    except Exception:
        subprocess.run(rollback, check=True, cwd=ROOT)
        raise
    print(json.dumps({"deployed": True, "source_revision": revision, "guard_sha256": expected,
                      "backup": str(backup), "health": "ok"}), flush=True)


if __name__ == "__main__":
    main()
