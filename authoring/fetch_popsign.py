"""Build the word-sign video library from PopSign ASL v1.0 (CC BY 4.0).

Replaces fetch_videos.py, whose clips were scraped from dictionary sites that
do not license redistribution. PopSign is 250 everyday signs recorded by 47
Deaf adults whose primary language is ASL, released by Georgia Tech and
RIT/NTID under CC BY 4.0 -- redistributable, commercial use included, with
attribution (written to sign_videos/CREDITS.json and shown in the app).

The full set is 1.1 TB, so nothing is bulk-downloaded. Each sign's videos sit in
one tar on the server, which honours HTTP range requests: tar headers are read
512 bytes at a time until a clip from the chosen signer turns up, and only that
clip's bytes are fetched. One signer throughout, so a sentence does not change
person on every word. Each clip is then downscaled with macOS's avconvert.

    python3 authoring/fetch_popsign.py            # resumable
    python3 authoring/fetch_popsign.py --survey   # signer coverage only
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Raw downloads stay out of the app bundle; authoring/studio_backdrop.swift
# turns them into the finished clips in sign_videos/.
OUT_DIR = os.path.join(ROOT, 'sign_videos_raw')
APP_DIR = os.path.join(ROOT, 'sign_videos')
BASE = 'https://signdata.cc.gatech.edu'
DS = 'popsign_v1_0'
SPLITS = ('test', 'val', 'train')
MAX_HEADERS = 80
# --strict scans whole tars (the train split runs to ~550 clips per sign).
STRICT_MAX_HEADERS = 800

# PopSign gloss -> file name(s) the app looks up. The app keys clips by the
# English word, lowercased, spaces as underscores; a two-word key is matched
# as a pair ("thank you"). Glosses that cover several English words are written
# once and reached through word_map.json (see EXTRA_WORD_MAP).
RENAME = {
    'thankyou': 'thank_you', 'callonphone': 'call', 'hesheit': 'he',
    'minemy': 'my', 'weus': 'we', 'haveto': 'have_to',
    # Checked by eye: the clip is GLASS (index taps the teeth), not WINDOW.
    'glasswindow': 'glass', 'frenchfries': 'french_fries',
    'icecream': 'ice_cream', 'shhh': 'shh',
}
EXTRA_WORD_MAP = {
    'phone': 'call', 'she': 'he', 'it': 'he', 'him': 'he', 'her': 'he',
    'mine': 'my', 'us': 'we', 'hurt': 'owie',
    'pain': 'owie', 'bathroom': 'potty', 'toilet': 'potty',
    'restroom': 'potty', 'fries': 'french_fries', 'television': 'tv',
    'hi': 'hello', 'thanks': 'thank_you', 'trash': 'garbage',
    'fridge': 'refrigerator', 'present': 'gift', 'quick': 'fast',
    'angry': 'mad', 'ill': 'sick', 'tired': 'sleepy', 'kid': 'child',
    'father': 'dad', 'mother': 'mom', 'grandmother': 'grandma',
    'grandfather': 'grandpa', 'store': 'store', 'shop': 'store',
}


def curl(url, rng=None, out=None):
    cmd = ['curl', '-sL', '--fail', '--retry', '4', '--retry-all-errors',
           '--max-time', '120', url]
    if rng:
        cmd[1:1] = ['-r', f'{rng[0]}-{rng[1]}']
    if out:
        cmd += ['-o', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise IOError(f'curl {r.returncode} for {url}')
    return r.stdout


def sign_list():
    html = curl(f'{BASE}/view/datasets/{DS}/index.html').decode('utf-8', 'replace')
    return sorted(set(re.findall(rf'/view/datasets/{DS}/game/([a-z]+)/index\.html', html)))


def participants(sign):
    """split -> Counter of participant ids, from the sign's own page."""
    html = curl(f'{BASE}/view/datasets/{DS}/game/{sign}/index.html').decode('utf-8', 'replace')
    m = re.search(r"sourceMap=JSON\.parse\('(.*?)'\)", html)
    if not m:
        return {}
    data = json.loads(m.group(1))
    return {split: Counter(n.split('-')[0] for n in data[split]['orig_name'].values())
            for split in data}


def find_in_tar(url, want, limit=MAX_HEADERS):
    """-> (offset, size) of the first member whose name starts with sign/want."""
    off = 0
    for _ in range(limit):
        h = curl(url, (off, off + 511))
        if len(h) < 512 or not h.strip(b'\0'):
            return None
        name = h[:100].split(b'\0')[0].decode('utf-8', 'replace')
        size = int(h[124:136].split(b'\0')[0].strip() or b'0', 8)
        base = name.split('/')[-1]
        if base.endswith('.mp4') and base.split('-')[0] == want:
            return off + 512, size
        off += 512 + ((size + 511) // 512) * 512
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--survey', action='store_true')
    ap.add_argument('--signer', help='participant id to use (default: best coverage)')
    ap.add_argument('--strict', action='store_true',
                    help='only --signer, never another person: a word they did not '
                         'record is left to fingerspelling')
    args = ap.parse_args()

    signs = sign_list()
    print(f'{len(signs)} signs')
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = os.path.join(ROOT, 'authoring', '.popsign_participants.json')
    who = json.load(open(cache)) if os.path.exists(cache) else {}
    for s in signs:
        if s not in who:
            who[s] = {k: dict(v) for k, v in participants(s).items()}
    json.dump(who, open(cache, 'w'))

    coverage = Counter()
    for s, splits in who.items():
        for p in {p for c in splits.values() for p in c}:
            coverage[p] += 1
    print('signers with the widest coverage:', coverage.most_common(6))
    if args.survey:
        return
    ranked = [args.signer] if args.signer else []
    if not args.strict:
        ranked += [p for p, _ in coverage.most_common(8) if p not in ranked]
    limit = STRICT_MAX_HEADERS if args.strict else MAX_HEADERS

    credits, done, failed = [], 0, []
    for sign in signs:
        key = RENAME.get(sign, sign)
        dest = os.path.join(OUT_DIR, f'{key}.mp4')
        if os.path.exists(dest):
            done += 1
            credits.append({'file': f'{key}.mp4', 'gloss': sign})
            continue
        got = None
        for p in ranked:                     # preferred signer first, then fallbacks
            for split in SPLITS:
                if who[sign].get(split, {}).get(p):
                    url = f'{BASE}/data/{DS}/game/{split}/{sign}.tar'
                    try:
                        hit = find_in_tar(url, p, limit)
                    except IOError as e:     # a few tars on the server are truncated
                        print(f'  {sign}: {e}; trying elsewhere')
                        hit = None
                    if hit:
                        got = (url, hit, p)
                        break
            if got:
                break
        if not got:
            failed.append(sign)
            print(f'  {sign}: no clip found')
            continue
        url, (start, size), p = got
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, 'raw.mp4')
            curl(url, (start, start + size - 1), out=raw)
            r = subprocess.run(['avconvert', '-s', raw, '-p', 'Preset960x540',
                                '-o', dest, '--replace'], capture_output=True)
            if r.returncode != 0 or not os.path.exists(dest):
                failed.append(sign)
                print(f'  {sign}: avconvert failed')
                continue
        done += 1
        credits.append({'file': f'{key}.mp4', 'gloss': sign, 'signer': p})
        print(f'  {sign} -> {key}.mp4  ({p}, {os.path.getsize(dest) // 1024} KB)')

    os.makedirs(APP_DIR, exist_ok=True)
    with open(os.path.join(APP_DIR, 'CREDITS.json'), 'w') as f:
        json.dump({
            'dataset': 'PopSign ASL v1.0',
            'citation': 'Starner et al., "PopSign ASL v1.0: An Isolated American '
                        'Sign Language Dataset Collected via Smartphones", NeurIPS '
                        '2023 Datasets and Benchmarks Track',
            'publisher': 'Georgia Institute of Technology and RIT/NTID',
            'source': f'{BASE}/view/datasets/{DS}/index.html',
            'license': 'CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)',
            'changes': 'One clip per sign, trimmed from the dataset tar; the '
                       'signer is cut out with Apple person segmentation, placed '
                       'on a plain studio backdrop, and downscaled to 480p.',
            'clips': credits,
        }, f, indent=1)
    print(f'\n{done} clips in {OUT_DIR}, {len(failed)} failed: {failed}')

    # Extra English words that reach these clips.
    wm_path = os.path.join(ROOT, 'ios', 'SignAid', 'word_map.json')
    wm = json.load(open(wm_path))
    have = {os.path.splitext(f)[0] for f in os.listdir(OUT_DIR) if f.endswith('.mp4')}
    added = 0
    for word, gloss in EXTRA_WORD_MAP.items():
        if gloss in have and word not in have and wm['map'].get(word) != gloss:
            wm['map'][word] = gloss
            added += 1
    json.dump(wm, open(wm_path, 'w'))
    print(f'word_map.json: {added} entries pointed at PopSign clips')


if __name__ == '__main__':
    main()
