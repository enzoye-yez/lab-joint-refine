"""Read-only replay of archived model responses. Corrections are developer fixtures."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from asset_prep.errors import ValidationError
from asset_prep.storage import atomic_json
from scene_generation.review import scoped_review,resolution_feedback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.cache/review_repair_20261020'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def checksum(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    sources={};rows=[]
    for protocol in ('001','043','088'):
        root=ROOT/'runs'/('static5_sol_1019_233853_np'+protocol)
        for name in ('state.json','result.json'):
            path=root/name
            if path.exists():sources[str(path)]=checksum(path)
        for stage in sorted((root/'stages').glob('*_scene')):
            record_path=stage/'record.json'
            if not record_path.exists():continue
            record=read(record_path)
            if 'plan' not in record or 'job_versions' not in record:continue
            sources[str(record_path)]=checksum(record_path)
            deferred=[]
            for directory in record['job_versions'].values():
                path=root/directory/'record.json'
                sources[str(path)]=checksum(path)
                deferred+=read(path).get('deferred_checks',[])
            expected={d['id'] for d in deferred if d['stage']=='scene'}
            ids={i['id'] for i in record['plan']['instances']}
            for path in sorted(stage.rglob('review_format_*.json')):
                archived=read(path)
                if not isinstance(archived,dict) or 'response' not in archived:continue
                value=archived['response'];sources[str(path)]=checksum(path)
                row={'protocol':'NP-'+protocol,'source':str(path.relative_to(ROOT)),
                     'original_error':archived.get('error'),'expected_items':len(expected)}
                try:
                    scoped_review(value,'scene',deferred,preview=True,instance_ids=ids)
                    row['current_valid']=True
                except ValidationError as exc:
                    row.update(current_valid=False,current_error=str(exc))
                valid,errors=resolution_feedback(value.get('deferred_results',[]) if isinstance(value,dict) else [],expected,preview=True)
                row.update(retained=len(valid),invalid=errors,missing_ids=sorted(expected-set(valid)))
                # Validate only contract recovery: never fabricate resolved evidence.
                corrected={'verdict':'uncertain','findings':[],'unverified':['Developer correction fixture; not a model recovery result'],
                    'blocking':['Historical response needs evidence/review; no geometry acceptance'],
                    'deferred_results':list(valid.values())+[{'id':key,'status':'unresolved','evidence':'Developer replay: original answer absent or invalid'} for key in sorted(expected-set(valid))]}
                scoped_review(corrected,'scene',deferred,preview=True,instance_ids=ids)
                row['developer_correction_valid']=True
                rows.append(row)
    unchanged=all(checksum(Path(path))==old for path,old in sources.items())
    report={'provenance':'historical responses plus developer-authored contract correction fixtures',
        'formal_model_calls':0,'source_files_checked':len(sources),'sources_unchanged':unchanged,
        'responses':len(rows),'by_protocol':dict(Counter(r['protocol'] for r in rows)),
        'valid_under_current_contract':sum(r['current_valid'] for r in rows),'rows':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'historical_review_replay.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
    assert unchanged and rows

if __name__=='__main__':main()
