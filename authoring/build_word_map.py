"""Map English words onto the signs we actually have.

Three separate problems, all of which showed up as "it spells everything":

1. GRAMMAR. ASL is not English. It has no articles and no copula, so "the" and
   "is" have no sign to find — spelling them T-H-E and I-S is not a fallback,
   it is wrong. They get dropped.

2. INFLECTION. The library is keyed on lemmas, so "books", "running" and
   "walked" all miss and fall through to fingerspelling. Stripping suffixes
   takes coverage of the 4000 commonest English words from 33% to about 51%.

3. SYNONYMY. English has roughly 170,000 words; ASL has on the order of 10,000
   signs. Many English words share one sign — "automobile" and "car", "purchase"
   and "buy" — so a word with no entry of its own often has a perfectly good
   sign under a different name.

What is left over gets fingerspelled, and that is CORRECT rather than a
shortfall: fingerspelling is what a signer does for names and words with no
sign. The goal is not to have every word, it is to stop spelling words that
have a sign and to stop spelling words that should not be there at all.

Output: ios/SignAid/word_map.json
    {"drop": [...], "map": {english: gloss}}
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = os.path.join(ROOT, 'sign_videos')
FREQ = os.path.join(ROOT, 'ios', 'SignAid', 'wordfreq_compact.json')
OUT = os.path.join(ROOT, 'ios', 'SignAid', 'word_map.json')

# Dropped entirely. ASL marks these with grammar, space and non-manual signals
# rather than with lexical items, so there is nothing to show.
DROP = {
    'a', 'an', 'the',                                   # articles
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',   # copula
    'do', 'does', 'did',                                # dummy auxiliary
    'of',                                               # possessive/partitive
    'to',                                               # infinitive marker
}

# Common English words whose sign lives under another name. Only entries whose
# target actually exists in the library survive to the output.
SYNONYMS = {
    'automobile': 'car', 'vehicle': 'car', 'purchase': 'buy', 'buy': 'buy',
    'obtain': 'get', 'receive': 'get', 'require': 'need', 'desire': 'want',
    'wish': 'want', 'speak': 'talk', 'chat': 'talk', 'converse': 'talk',
    'consume': 'eat', 'dine': 'eat', 'meal': 'food', 'beverage': 'drink',
    'residence': 'home', 'house': 'home', 'apartment': 'home',
    'automobile_': 'car', 'physician': 'doctor', 'nurse': 'nurse',
    'ill': 'sick', 'unwell': 'sick', 'ache': 'hurt', 'pain': 'hurt',
    'assist': 'help', 'aid': 'help', 'support': 'help',
    'commence': 'start', 'begin': 'start', 'complete': 'finish',
    'conclude': 'finish', 'end': 'finish', 'halt': 'stop', 'cease': 'stop',
    'depart': 'leave', 'exit': 'leave', 'arrive': 'come', 'return': 'come',
    'observe': 'see', 'view': 'see', 'watch': 'look', 'listen': 'hear',
    'comprehend': 'understand', 'realize': 'understand', 'believe': 'think',
    'consider': 'think', 'recall': 'remember', 'forget': 'forget',
    'enjoy': 'like', 'love': 'love', 'prefer': 'like', 'hate': 'hate',
    'inquire': 'ask', 'question': 'ask', 'respond': 'answer', 'reply': 'answer',
    'inform': 'tell', 'notify': 'tell', 'explain': 'explain',
    'labor': 'work', 'job': 'work', 'occupation': 'work', 'employment': 'work',
    'instruct': 'teach', 'educate': 'teach', 'study': 'learn',
    'currency': 'money', 'cash': 'money', 'cost': 'price', 'expensive': 'expensive',
    'inexpensive': 'cheap', 'restroom': 'bathroom', 'toilet': 'bathroom',
    'physician_': 'doctor', 'infant': 'baby', 'child': 'child', 'kid': 'child',
    'spouse': 'married', 'companion': 'friend', 'buddy': 'friend',
    'attractive': 'beautiful', 'pretty': 'beautiful', 'unattractive': 'ugly',
    'enormous': 'big', 'huge': 'big', 'large': 'big', 'tiny': 'small',
    'little': 'small', 'rapid': 'fast', 'quick': 'fast', 'slow': 'slow',
    'difficult': 'hard', 'simple': 'easy', 'correct': 'right', 'incorrect': 'wrong',
    'certain': 'sure', 'perhaps': 'maybe', 'possibly': 'maybe',
    'presently': 'now', 'currently': 'now', 'afterward': 'later',
    'previously': 'before', 'commence_': 'start', 'entire': 'all', 'every': 'all',
    'numerous': 'many', 'several': 'many', 'few': 'few',
    'gratitude': 'thank', 'apologize': 'sorry', 'greetings': 'hello',
    'farewell': 'goodbye', 'bye': 'goodbye', 'yes': 'yes', 'no': 'no',
}


# Irregular verbs, which no amount of suffix stripping will reach: "went" does
# not contain "go". This is the single biggest remaining source of misses, and
# they are all high-frequency words.
IRREGULAR = {
    'was': 'be', 'were': 'be', 'been': 'be', 'am': 'be', 'is': 'be', 'are': 'be',
    'has': 'have', 'had': 'have', 'having': 'have',
    'said': 'say', 'says': 'say', 'saying': 'say',
    'made': 'make', 'making': 'make', 'got': 'get', 'gotten': 'get',
    'took': 'take', 'taken': 'take', 'taking': 'take',
    'came': 'come', 'coming': 'come', 'went': 'go', 'gone': 'go', 'going': 'go',
    'thought': 'think', 'seen': 'see', 'saw': 'see', 'seeing': 'see',
    'told': 'tell', 'telling': 'tell', 'known': 'know', 'knew': 'know',
    'lost': 'lose', 'losing': 'lose', 'found': 'find', 'finding': 'find',
    'given': 'give', 'gave': 'give', 'giving': 'give',
    'ate': 'eat', 'eaten': 'eat', 'eating': 'eat',
    'drank': 'drink', 'drunk': 'drink', 'drinking': 'drink',
    'wrote': 'write', 'written': 'write', 'writing': 'write',
    'read': 'read', 'ran': 'run', 'running': 'run',
    'began': 'start', 'begun': 'start', 'left': 'leave', 'leaving': 'leave',
    'felt': 'feel', 'feeling': 'feel', 'kept': 'keep', 'keeping': 'keep',
    'held': 'have', 'brought': 'bring', 'bought': 'buy', 'taught': 'teach',
    'caught': 'catch', 'sent': 'send', 'spent': 'spend', 'built': 'build',
    'met': 'meet', 'meeting': 'meet', 'paid': 'pay', 'paying': 'pay',
    'sat': 'sit', 'sitting': 'sit', 'stood': 'stand', 'standing': 'stand',
    'understood': 'understand', 'became': 'become', 'chose': 'choose',
    'slept': 'sleep', 'sleeping': 'sleep', 'woke': 'wake',
    'children': 'child', 'people': 'person', 'men': 'man', 'women': 'woman',
    'feet': 'foot', 'teeth': 'tooth',
    # Pronouns and modals: these have signs, under other names.
    'he': 'boy', 'him': 'boy', 'his': 'boy', 'she': 'girl', 'her': 'girl',
    'us': 'we', 'our': 'we', 'ours': 'we', 'them': 'they', 'their': 'they',
    'its': 'it', 'mine': 'my', 'your': 'you', 'yours': 'you',
    'would': 'will', 'could': 'can', 'should': 'should', 'might': 'maybe',
    'may': 'maybe', 'must': 'need', 'shall': 'will',
    'thanks': 'thank', 'ok': 'yes', 'okay': 'yes', 'hi': 'hello',
    'look': 'see', 'looking': 'see', 'looked': 'see', 'looks': 'see',
}


def variants(word):
    """Plausible lemmas for an inflected form, best first."""
    out = [word]
    rules = [
        ('ies', 'y'), ('ied', 'y'), ('ier', 'y'), ('iest', 'y'),
        ('sses', 'ss'), ('shes', 'sh'), ('ches', 'ch'), ('xes', 'x'),
        ('ing', ''), ('ing', 'e'), ('ed', ''), ('ed', 'e'),
        ('est', ''), ('est', 'e'), ('er', ''), ('er', 'e'),
        ('ly', ''), ('es', ''), ('s', ''),
    ]
    for suf, rep in rules:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            out.append(word[:len(word) - len(suf)] + rep)
    # Doubled consonant before -ing/-ed: "running" -> "run", "stopped" -> "stop"
    for suf in ('ing', 'ed'):
        if word.endswith(suf):
            stem = word[:-len(suf)]
            if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
                out.append(stem[:-1])
    seen, ordered = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def main():
    have = {f[:-4].lower() for f in os.listdir(VIDEOS) if f.endswith('.mp4')}

    with open(FREQ) as f:
        freq = json.load(f)
    vocabulary = list(freq.keys() if isinstance(freq, dict) else freq)

    mapping = {}

    # Irregulars and synonyms first: an explicit choice should beat a guessed
    # lemma, and suffix rules cannot reach an irregular form at all.
    for word, target in IRREGULAR.items():
        if target in have and word not in have:
            mapping[word] = target

    for word, target in SYNONYMS.items():
        w = word.rstrip('_')
        if target in have and w not in have and w not in mapping:
            mapping[w] = target

    # Then inflections, for every word in the frequency list.
    for word in vocabulary:
        if not isinstance(word, str) or not word.isalpha():
            continue
        w = word.lower()
        if w in have or w in DROP or w in mapping:
            continue
        for v in variants(w)[1:]:
            if v in have:
                mapping[w] = v
                break

    payload = {'drop': sorted(DROP), 'map': mapping}
    with open(OUT, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

    top = [w for w in vocabulary[:4000] if isinstance(w, str) and w.isalpha()]
    covered = sum(1 for w in top
                  if w in have or w in DROP or w in mapping)
    size = os.path.getsize(OUT) / 1024
    print(f'{len(mapping)} words mapped onto {len(have)} signs, '
          f'{len(DROP)} dropped ({size:.0f} KB)')
    print(f'top 4000 English words now covered: {covered}/{len(top)} '
          f'({covered/len(top):.0%})')


if __name__ == '__main__':
    main()
