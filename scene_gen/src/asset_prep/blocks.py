"""Organize bounded tasks by source lines/paragraphs; the span is the fact, adjacent text only provides context."""

import re
from .contracts import require
from .storage import digest


def balanced_parts(source, limit):
    """Split at line breaks or whitespace where possible, so a fixed stride does not leave single-character remainders."""
    text = source['text']
    require(limit >= 2, 'structural block character limit must be at least 2')
    result, start = [], 0
    while len(text) - start > limit:
        remaining = len(text) - start
        pieces = (remaining + limit - 1) // limit
        target = start + (remaining + pieces - 1) // pieces
        low, high = start + max(2, (target-start)//2), min(start+limit, len(text)-2)
        boundaries = [m.end() for m in re.finditer(r'\n|\s+', text[start:high])
                      if start + m.end() >= low]
        cut = start + min(boundaries, key=lambda p: abs(start+p-target)) if boundaries else target
        result.append(dict(source, text=text[start:cut], offset=source.get('offset',0)+start))
        start = cut
    if start < len(text):
        result.append(dict(source, text=text[start:], offset=source.get('offset',0)+start))
    return result


def units(sources, limit):
    result = []
    # Blank lines, material-list lines and table text lines keep their original separators; content is never dropped based on character length.
    for source in sources:
        start = 0
        local = []
        for line in source['text'].splitlines(keepends=True):
            part = dict(source, text=line, offset=source.get('offset',0)+start)
            local.extend(balanced_parts(part, limit))
            start += len(line)
        if not local and source['text']:
            local = balanced_parts(source,limit)
        # Line breaks and single-character lines are merged into adjacent lines of the same source; no characters are dropped and no quotes are fabricated across sources.
        merged=[]
        for part in local:
            if merged and (len(part['text'].strip())<=1 or len(merged[-1]['text'].strip())<=1):
                previous=merged.pop()
                merged.append(dict(previous,text=previous['text']+part['text']))
            else:
                merged.append(part)
        for part in merged:
            result.extend(balanced_parts(part,limit))
    return result


def pack(items, limit):
    groups, current, size = [], [], 0
    for item in items:
        if current and size + len(item['text']) > limit and len(item['text'].strip())>1:
            groups.append(current); current=[]; size=0
        current.append(item); size += len(item['text'])
    if current: groups.append(current)
    return groups


def task(parts):
    return {'id':'block_'+digest(parts)[:20], 'units':parts}


def plan(sources, limit):
    return [task(group) for group in pack(units(sources, limit), limit)]


def excerpts(parts, documents, context=160):
    by_id={s['id']:s for s in documents['sources']}
    ranges={}
    for part in parts:
        start=part.get('offset',0); end=start+len(part['text'])
        if part['id'] in ranges:
            left,right=ranges[part['id']]; ranges[part['id']]=(min(left,start),max(right,end))
        else: ranges[part['id']]=(start,end)
    result=[]
    for sid,(start,end) in ranges.items():
        source=by_id[sid]; left=max(0,start-context); right=min(len(source['text']),end+context)
        result.append(dict(source,text=source['text'][left:right],offset=left,
                           focus_start=start,focus_end=end))
    return result


def split_task(block):
    parts=block['units']
    if len(parts)>1:
        middle=len(parts)//2
        return [task(parts[:middle]),task(parts[middle:])]
    source=parts[0]
    require(len(source['text'])>200, 'a single minimal structural block still cannot produce bounded candidates; the task is kept for inspection instead of splitting indefinitely')
    return [task([part]) for part in balanced_parts(source,(len(source['text'])+1)//2)]
