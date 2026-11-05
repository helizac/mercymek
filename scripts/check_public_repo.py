"""Check the files and optional Git history intended for a public source release.

Uses only the standard library. Reports locations and secret types, never values.
    python scripts/check_public_repo.py --history
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'credential in URL': rb'https?://[^/\s:@]+:[^/\s@]+@',
    'private key': rb'(?m)^-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----',
    'Hugging Face token': rb'\bhf_[A-Za-z0-9]{25,}\b',
    'GitHub token': rb'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'service key': rb'\b(?:sk_live_|sk_test_|sk_proj_|KGAT_)[A-Za-z0-9_-]{25,}\b',
}
SECRET_NAMES = {'MODEL_API_KEY', 'HF_TOKEN', 'ELEVENLABS_API_KEY', 'KAGGLE_API_TOKEN', 'MAPILLARY_TOKEN'}

def git(*args, **kwargs):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, **kwargs).stdout


def audit(history=False, expected_version=None):
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
        raise RuntimeError("Run this audit from the actual Git checkout, not an extracted copy inside another repository")
    errors = []
    def error(path, reason):
        errors.append({'location': path, 'reason': reason})
    names = sorted(set(filter(None, git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0'))))
    files = [n for n in names if (ROOT/n).is_file()]
    known = {k: v.encode() for k in SECRET_NAMES if (v := os.environ.get(k)) and len(v) >= 12}
    env = ROOT/'.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8').splitlines():
            key, _, value = line.partition('=')
            value = value.split(' #')[0].strip().strip('\"\'')
            if key.strip() in SECRET_NAMES and len(value) >= 12:
                known[key.strip()] = value.encode()
    def scan(data, where):
        for key, value in known.items():
            if value in data:
                error(where, 'exact local '+key)
        for kind, pattern in PATTERNS.items():
            if re.search(pattern, data):
                error(where, kind)
    banned_dirs = {'data', 'deliverables', '.git', 'node_modules', '__pycache__', 'build', 'dist'}
    for name in files:
        p = ROOT/name
        parts = Path(name).parts
        if any(x in banned_dirs or x.startswith('.venv') for x in parts):
            error(name, 'local data/environment/build output in source')
        if p.name.startswith('.env') and p.name != '.env.example':
            error(name, 'local environment file in source')
        if p.suffix.lower() in {'.pem', '.key', '.p12', '.pfx', '.zip', '.pt', '.pth', '.safetensors', '.onnx'}:
            error(name, 'credential/archive/model artifact in source')
        if p.stat().st_size > 95*1024*1024:
            error(name, 'file exceeds the source size limit')
        data = p.read_bytes()
        scan(data, 'working tree: '+name)
        if p.suffix == '.ipynb':
            notebook = json.loads(data)
            if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook['cells'] if c['cell_type'] == 'code'):
                error(name, 'notebook outputs/execution counts must be cleared')
        if p.suffix == '.md':
            text = re.sub(r'```.*?```', '', data.decode('utf-8'), flags=re.S)
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                link = target.split('#')[0].strip('<>')
                if link and not re.match(r'[A-Za-z]+:', link) and not (p.parent/link).exists():
                    error(name, 'missing relative link: '+target)
    for name in ['.env.production', 'data/keys/seal.pem', 'deliverables/demo.mp4', '.venv/Scripts/python.exe']:
        if subprocess.run(['git', 'check-ignore', '-q', name], cwd=ROOT).returncode != 0:
            error('.gitignore', 'does not ignore '+name)
    meta = tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']
    version = meta['version']
    init = re.search(r'__version__\s*=\s*[\"\']([^\"\']+)', (ROOT/'mercy/__init__.py').read_text())
    if not init or init[1] != version:
        error('mercy/__init__.py', 'version differs from pyproject.toml')
    if expected_version and expected_version.removeprefix('v') != version:
        error('pyproject.toml', 'version differs from the requested release tag')
    for required in ['README.md', 'CHANGELOG.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md', 'docs/releasing.md', 'requirements-dev.lock',
                     'frontend/vendor/three.LICENSE', 'frontend/vendor/spark/LICENSE', 'frontend/vendor/leaflet/LICENSE', 'third_party/NEXAR_LICENSE.txt']:
        if not (ROOT/required).is_file():
            error(required, 'required public documentation/notice is missing')
    objects_checked = 0
    if history:
        objects = []
        for line in git('rev-list', '--objects', '--all').decode().splitlines():
            pair = line.split(' ', 1)
            objects.append((pair[0], pair[1] if len(pair)>1 else '(git metadata)'))
        payload = ''.join(oid+'\n' for oid, _ in objects).encode()
        raw = git('cat-file', '--batch', input=payload)
        pos = 0
        for oid, name in objects:
            end = raw.index(b'\n', pos)
            _, kind, size = raw[pos:end].decode().split()
            pos = end+1
            size = int(size)
            value = raw[pos:pos+size]
            pos += size+1
            if kind in {'blob', 'commit', 'tag'}:
                scan(value, 'history: '+name+' ['+oid[:12]+']')
                objects_checked += 1
    return {'version': version, 'source_files': len(files), 'history_objects_checked': objects_checked,
            'findings': errors, 'passed': not errors,
            'scope': 'Pattern and known-local-key checks; not a guarantee that every possible secret is detectable.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--history', action='store_true')
    ap.add_argument('--expected-version')
    ap.add_argument('--json-out', type=Path)
    args = ap.parse_args()
    result = audit(args.history, args.expected_version)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
