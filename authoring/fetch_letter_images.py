"""Fetch the fingerspelling alphabet from Wikimedia Commons.

Replaces the photographs of the developer's own hand. Those were consistent with
each other but not with the sign videos — a close-up of a hand in a kitchen next
to a full-body studio shot — and they put a real person's hand in the app.

The "Sign language X.svg" set on Commons is complete for A-Z, drawn in one
style, and **public domain** (wpclipart.com), so unlike the sign videos there is
no attribution obligation and no licensing risk. Line drawings also beat photos
for this specific job: a fingerspelled letter is a shape, and a drawing removes
the skin tone, lighting and background that a reader has to look past.

License provenance is written alongside the images, since "where did this come
from" is a question worth being able to answer without re-deriving it.
"""
import json
import os
import ssl
import socket
import time
import urllib.parse
import urllib.request

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'letter_photos')
# Raw downloads are cached so re-processing (background colour, size) does not
# re-hit Commons, which rate-limits hard.
CACHE = os.path.join(ROOT, '.letter_cache')
# The artwork is BLACK line drawing on transparency, so it has to sit on a light
# card. Flattening onto the app's dark background made every letter invisible.
PAPER = (245, 245, 247)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SIZE = 400
# Rendered at 2x then downsampled: the source is SVG, and asking Commons for the
# final size directly gives noticeably rougher edges.
REQUEST_WIDTH = 800

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
_HEADERS = {'User-Agent': 'SignAid-educational-project/1.0'}
# Commons rate-limits bursts. 26 files back to back trips it, so they are
# spaced out and retried with backoff rather than hammered.
PAUSE = 3.0
RETRIES = 5


def get(url, timeout=30):
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout, context=_ctx) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt * 5)
    raise RuntimeError('unreachable')


def query(titles):
    url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode({
        'action': 'query', 'titles': '|'.join(titles), 'prop': 'imageinfo',
        'iiprop': 'url|extmetadata', 'iiurlwidth': str(REQUEST_WIDTH),
        'format': 'json',
    })
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=_ctx) as r:
        return json.load(r)


def main():
    socket.setdefaulttimeout(30)
    os.makedirs(OUT, exist_ok=True)

    titles = [f'Sign language {c}.svg' for c in LETTERS]
    pages = query([f'File:{t}' for t in titles])['query']['pages']

    by_letter = {}
    for page in pages.values():
        title = page.get('title', '')
        if 'imageinfo' not in page or not title.startswith('File:Sign language '):
            continue
        letter = title[len('File:Sign language '):].split('.')[0].upper()
        if letter in LETTERS:
            by_letter[letter] = page['imageinfo'][0]

    missing = [c for c in LETTERS if c not in by_letter]
    if missing:
        raise SystemExit(f'not available on Commons: {" ".join(missing)}')

    # Download everything BEFORE touching the existing images. An earlier
    # version cleared them first and then hit a rate limit, which left the app
    # with no alphabet at all.
    os.makedirs(CACHE, exist_ok=True)
    fetched = {}
    for letter, info in sorted(by_letter.items()):
        cached = os.path.join(CACHE, f'{letter}.png')
        if os.path.exists(cached):
            with open(cached, 'rb') as f:
                fetched[letter] = f.read()
            continue
        # Strip the query string: the API appends utm tracking params, and
        # upload.wikimedia.org rejects the request with them attached.
        url = (info.get('thumburl') or info.get('url')).split('?')[0]
        fetched[letter] = get(url)
        with open(cached, 'wb') as f:
            f.write(fetched[letter])
        time.sleep(PAUSE)

    for old in os.listdir(OUT):
        os.remove(os.path.join(OUT, old))

    credits = {}
    for letter, info in sorted(by_letter.items()):
        data = fetched[letter]
        tmp = os.path.join(OUT, '_tmp')
        with open(tmp, 'wb') as f:
            f.write(data)

        im = Image.open(tmp).convert('RGBA')
        canvas = Image.new('RGBA', im.size, PAPER + (255,))
        canvas.alpha_composite(im)
        canvas = canvas.convert('RGB')
        # Square with padding, so every letter is the same size on screen.
        side = int(max(canvas.size) * 1.12)      # a little breathing room
        square = Image.new('RGB', (side, side), PAPER)
        square.paste(canvas, ((side - canvas.width) // 2,
                              (side - canvas.height) // 2))
        square.resize((SIZE, SIZE), Image.LANCZOS).save(
            os.path.join(OUT, f'{letter}.jpg'), quality=90)
        os.remove(tmp)

        md = info.get('extmetadata', {})
        credits[letter] = {
            'source': info.get('descriptionurl', ''),
            'license': md.get('LicenseShortName', {}).get('value', ''),
        }

    with open(os.path.join(ROOT, 'letter_photos_CREDITS.json'), 'w') as f:
        json.dump({'source': 'Wikimedia Commons, "Sign language X.svg" series',
                   'letters': credits}, f, indent=2)

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    licenses = {c['license'] for c in credits.values()}
    print(f'{len(credits)} letters -> {OUT} ({total/1024:.0f} KB)')
    print(f'licenses: {", ".join(sorted(licenses))}')


if __name__ == '__main__':
    main()
