"""只用已存响应将明确支持的旧控制流程转换为持久化任务；禁止网络。"""

import copy
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from . import blocks
from .contracts import require, validate_inventory
from .extraction import extract_inventory, validate_unique_assets
from .providers import Model
from .storage import Store, atomic_json, digest, file_hash, safe_path
from .workflow import VERSION, new_workflow, seed_active, plan_tasks


SUPPORTED={'245ec04872e75a90240e268abc6ede9a720cf7e8df340bea7c3dbf01c8eeef5b'}


class ReplayBoundary(BaseException):
    pass


class ReadOnlyReplayStore(Store):
    def cached(self,key):
        response=super().cached(key)
        if response is not None:
            if not hasattr(self,'replayed_keys'): self.replayed_keys=set()
            self.replayed_keys.add(key)
        return response

    def reserve(self,*args,**kwargs):
        raise ReplayBoundary()


class NoNetwork:
    def post(self,*args,**kwargs):
        raise AssertionError('离线迁移禁止网络')


def migrate_run(root,identity,documents,config):
    root=Path(root); path=safe_path(root,'state.json')
    state=json.loads(path.read_text(encoding='utf-8')); old=state['identity']
    if old==identity and state.get('workflow',{}).get('version')==VERSION:
        return {'status':'already_current','network_calls':0}
    require(old.get('code') in SUPPORTED and not old.get('workflow_contract'),'不支持此历史源码版本的自动任务迁移')
    require({k:v for k,v in old.items() if k!='code'}==
            {k:v for k,v in identity.items() if k not in {'code','workflow_contract'}},'输入、解析、模型或语义配置发生变化，不能迁移')
    require({'asset_package.json','documents.json'}<=state.get('outputs',{}).keys(),'历史任务缺少完整输出摘要')
    for name,sha in state['outputs'].items():
        require(file_hash(safe_path(root,name))==sha,'历史输出被修改，不能迁移')
    for call in state['calls']:
        require(call['kind']=='model' and call['status']=='done','自动迁移只接受提取阶段且全部请求已确定的运行')
        require(file_hash(safe_path(root,call['file']))==call['sha256'],'历史响应被修改，不能迁移')
    package=json.loads(safe_path(root,'asset_package.json').read_text(encoding='utf-8'))
    require(package['identity']==old and package['status'] in {'failed','paused','interrupted'},'历史任务不处于可迁移状态')
    require(not state.get('checkpoint'),'不支持叠加其他历史迁移检查点')
    snapshot={}; progress={}
    def capture(value):
        snapshot.clear(); snapshot.update(copy.deepcopy(value))
    # 临时账本只引用只读原响应；校验事件写在临时目录，原账本最后一次性切换。
    with tempfile.TemporaryDirectory(prefix='migration-',dir=root.parent) as directory:
        temporary=Path(directory)
        replay_state=copy.deepcopy(state)
        for call in replay_state['calls']:
            target=temporary/call['file']; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(safe_path(root,call['file']).read_bytes())
        atomic_json(temporary/'state.json',replay_state)
        replay_config=replace(config,max_reserved_tokens=max(config.max_reserved_tokens,
            sum(max(c['reserved_tokens'],c.get('reported_total_tokens',0)) for c in state['calls'])))
        store=ReadOnlyReplayStore(temporary,old,replay_config,resume=True)
        model=Model(replay_config,{replay_config.api_key_env:'offline-migration-secret-not-in-documents'},store,NoNetwork())
        try:
            extract_inventory(documents,replay_config,model,progress,capture=capture)
        except ReplayBoundary:
            pass
        require({c['key'] for c in state['calls']}<=getattr(store,'replayed_keys',set()),
                '部分已付费响应未被迁移适配器消费；拒绝丢弃决定或改变请求后重算')
    require(snapshot and snapshot['catalog']==package['inventory'],'已存响应重建的总表与检查点不一致；拒绝迁移')
    validate_inventory(snapshot['catalog'],documents['sources'],config.max_assets,allow_empty=True)
    validate_unique_assets(snapshot['catalog'])
    workflow=new_workflow(documents,config)
    workflow['catalog']=snapshot['catalog']; workflow['revision']=package['extraction']['revision']
    workflow['commits']=copy.deepcopy(package['extraction']['updates'])
    for commit in workflow['commits']: commit['legacy_import']=True
    # 合并尚未处理且连续的旧区间，再按新规则成块；不能继承旧二分产生的尾片。
    remaining=[]
    pending_parts=([snapshot['segment']] if snapshot['draft'] is None else [])+snapshot['queue']
    for parts in pending_parts:
        for part in parts:
            if (remaining and remaining[-1]['id']==part['id'] and
                    remaining[-1].get('offset',0)+len(remaining[-1]['text'])==part.get('offset',0)):
                remaining[-1]['text']+=part['text']
            else:
                remaining.append(copy.deepcopy(part))
    workflow['queue']=plan_tasks(documents,config,sources=remaining) if remaining else []
    task=blocks.task(snapshot['segment'])
    if snapshot['draft'] is not None:
        active=seed_active(task,snapshot['draft'],snapshot['catalog'])
        if snapshot['batch']:
            batch=snapshot['batch']
            active.update(assets=batch['assets'],mapping=batch['mapping'],keys=batch['keys'],decisions=batch['decisions'],
                          cursor=len(snapshot['draft']['assets'])-len(batch['remaining']))
        workflow['active']=active
    workflow['operation_audit'].append({'action':'legacy_migration','calls_preserved':len(state['calls']),
        'from_code':old['code'],'note':'已确认资产和当前候选/部分合并导入；后续使用新的身份与字段操作，不承诺请求格式相同'})
    audit={'at':time.time(),'from_identity':old,'to_identity':identity,'mode':'persistent_tasks',
           'outputs_before':copy.deepcopy(state['outputs']),'calls_preserved':len(state['calls'])}
    state['identity']=identity; state['workflow']=workflow; state['workflow_sha256']=digest(workflow)
    state.setdefault('upgrades',[]).append(audit)
    atomic_json(path,state)
    return {'status':'upgraded','mode':'persistent_tasks','assets_preserved':len(workflow['catalog']['assets']),
            'calls_preserved':len(state['calls']),'pending_candidates':len(workflow['active']['draft']['assets'])-workflow['active']['cursor'] if workflow['active'] else 0,
            'network_calls':0}
