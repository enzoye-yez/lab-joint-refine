"""Resumable task understanding, local asset registration and delivery review. All requests share the original ledger."""

from copy import deepcopy

from . import blocks, prepare_prompts as prompts
from .contracts import evidence, fields, require, text, texts, validate_inventory
from .errors import OutputTooLarge, ValidationError
from .providers import Model
from .storage import atomic_json, digest, encoded
from .task_contracts import (BOUNDARY, WORKFLOW_VERSION, public_environment, source_map,
                             validate_task, validate_roles, validate_links)
from .workflow import run_workflow


def task_card(task):
    # The fixed semantic summary does not repeat each stage's long quotes; the source text is still provided by this request's source excerpts.
    return {'objective':task['objective'], 'selected_flow':task['selected_flow'],
            'excluded_branches':[{k:r[k] for k in ('name','reason')} for r in task['excluded_branches']],
            'preparations':[{k:r[k] for k in ('description','object_ids','basis')} for r in task['preparations']],
            'objects':[{**{k:r[k] for k in ('id','description','initial_state')},
                        'quantity':{k:r['quantity'][k] for k in ('value','origin','basis')}} for r in task['objects']],
            'stages':[{**{k:r[k] for k in ('id','description','depends_on','object_ids','outcome')},
                       'source_ids':list(dict.fromkeys(e['source_id'] for e in r['evidence']))} for r in task['stages']]}


def apply_patches(task, reply, sources):
    fields(reply, 'patches issues'); texts(reply['issues'])
    require(isinstance(reply['patches'], list), 'patches must be a list')
    revised = deepcopy(task); seen = set()
    keyed = {'objects', 'stages', 'preparations', 'final_conditions'}
    for patch in reply['patches']:
        fields(patch, 'section id value reason evidence')
        section = patch['section']; require(section in task, 'patch targets an unknown task field')
        text(patch['reason']); evidence(patch['evidence'], source_map(sources))
        key = (section, patch['id']); require(key not in seen, 'the same field is patched more than once'); seen.add(key)
        if section in keyed:
            require(isinstance(patch['id'], str), 'object patch is missing an id')
            rows = revised[section]; old = next((i for i,r in enumerate(rows) if r['id']==patch['id']), None)
            value = patch['value']
            if value is None:
                require(old is not None and patch['evidence'], 'a deletion must name an existing object and give source evidence'); rows.pop(old)
            else:
                require(isinstance(value, dict) and value.get('id')==patch['id'], 'a patch must not silently change the id')
                if old is None: rows.append(deepcopy(value))
                else: rows[old] = deepcopy(value)
        else:
            require(patch['id'] is None, 'a whole-field patch does not use an object id')
            revised[section] = deepcopy(patch['value'])
    validate_task(revised, sources, partial=True)
    return revised


class Preparation:
    def __init__(self, store, documents, config, keys, transport, environment, *, model=None):
        self.store, self.documents, self.config = store, documents, config
        self.model = model or Model(config, keys, store, transport)
        self.environment = public_environment(environment)
        self.sources = documents['sources']
        self.state = store.state.setdefault('preparation', {
            'version':WORKFLOW_VERSION, 'phase':'index', 'status':'running',
            'index_queue':blocks.plan(self.sources, config.chunk_chars), 'index':[],
            'selection_cursor':0, 'selection':None, 'definition_cursor':0, 'review_cursor':0,
            'task':None, 'task_revision':0, 'task_history':[], 'roles_by_asset':{},
            'links_by_stage':{}, 'reviews':{}, 'repair_round':0, 'reference_cache':{}})
        require(self.state['version']==WORKFLOW_VERSION, 'an old full-text extraction cannot be resumed directly as task preparation')
        if 'preparation_sha256' in store.state:
            require(digest(self.state)==store.state['preparation_sha256'], 'task preparation state digest mismatch')
        self.save()

    def save(self):
        self.store.state['preparation_sha256'] = digest(self.state)
        self.store.save()
        atomic_json(self.store.root/'progress.json', {
            'phase':self.state['phase'], 'status':self.state['status'],
            'indexed_blocks':len(self.state['index']), 'remaining_blocks':len(self.state['index_queue']),
            'task_revision':self.state['task_revision'],
            'asset_types':len(self.state.get('inventory',{}).get('assets',[])),
            'instance_count':sum(r['count'] for r in self.all_roles()),
            'reference_gaps':sum(a.get('status')!='prepared' for a in self.state.get('assets',[])),
            'budget':self.store.totals()})

    def phase(self, name):
        self.state.update(phase=name, status='running'); self.save()
        labels = {'index':'full-text index','select':'flow selection','define':'task definition','task_review':'task review',
                  'assets':'asset registration','roles':'instance assignment','links':'interaction requirements','references':'reference search',
                  'delivery_review':'delivery review','publish':'public delivery','complete':'complete'}
        print('Task preparation: '+labels.get(name,name)+'; model calls '+str(self.store.totals()['model_calls']), flush=True)

    def ask(self, operation, prompt, data, validate):
        payload = {'operation':operation, **data}
        if self.state['task'] is not None:
            payload['task_revision'] = self.state['task_revision']
        require(len(encoded(self.model.messages(prompt, payload))) <= min(96000, self.config.max_input_bytes),
                'task request exceeds the bounded input limit; progress saved; the task and sources are not truncated')
        return self.model.structured(prompt, payload, validate)

    def index(self):
        while self.state['index_queue']:
            block = self.state['index_queue'][0]
            sources = blocks.excerpts(block['units'], self.documents, context=0)
            def validate(value):
                if value == {'split_unit':True}: raise OutputTooLarge('index needs the source text subdivided')
                fields(value, 'entries'); require(isinstance(value['entries'],list) and len(value['entries'])<=16, 'index too long')
                for row in value['entries']:
                    fields(row, 'kind summary evidence')
                    require(row['kind'] in {'purpose','stage','branch','preparation','cross_reference','equipment'}, 'invalid index kind')
                    text(row['summary']); require(len(row['summary'])<=400, 'index summary too long')
                    evidence(row['evidence'], source_map(sources), required=True)
            try:
                reply = self.ask('protocol_index', prompts.INDEX, {'sources':sources}, validate)
            except OutputTooLarge:
                self.state['index_queue'][0:1] = blocks.split_task(block); self.save(); continue
            self.state['index'].append({'block':block, 'entries':reply['entries']})
            self.state['index_queue'].pop(0); self.save()
        self.phase('select')

    def select(self):
        while self.state['selection_cursor'] < len(self.state['index']):
            row = self.state['index'][self.state['selection_cursor']]
            def validate(value):
                if isinstance(value,dict) and value.get('pending') is True:
                    fields(value, 'pending reason'); text(value['reason'])
                    require(self.state['selection'] is None, 'an existing flow cannot be overwritten by an empty candidate')
                    return
                fields(value, 'name objective reason evidence')
                for key in ('name','objective','reason'):
                    text(value[key]); require(len(value[key])<=1500, 'flow summary too long')
                evidence(value['evidence'], source_map(self.sources), required=True)
            decision = self.ask('flow_selection', prompts.SELECT,
                {'index_page':row['entries'], 'current_best':self.state['selection'],
                 'page':self.state['selection_cursor'], 'page_count':len(self.state['index'])}, validate)
            self.state.setdefault('selection_history',[]).append({'page':self.state['selection_cursor'],'decision':decision})
            if not decision.get('pending'): self.state['selection'] = decision
            self.state['selection_cursor'] += 1; self.save()
        chosen = self.state['selection']; require(chosen, 'failed to identify the execution flow')
        self.state['task'] = {'objective':chosen['objective'], 'selected_flow':chosen['name'],
            'selection_reason':chosen['reason'], 'selection_evidence':deepcopy(chosen['evidence']), **{k:[] for k in
            ('excluded_branches','preparations','objects','stages','final_conditions','abstractions','assumptions','issues')}}
        self.phase('define')

    def record_task(self, revised, patches, reason):
        previous = self.state['task']
        if revised != previous:
            affected = {p['id'] for p in patches if p.get('id')}
            affected_assets = {r['asset_id'] for r in self.all_roles()
                               if affected.intersection(r['object_ids']+r['stage_ids'])}
            if any(p['id'] is None for p in patches):
                affected_assets.update(self.state['roles_by_asset'])
            self.state['task_revision'] += 1
            self.state['task_history'].append({'revision':self.state['task_revision'], 'before':digest(previous),
                'after':digest(revised), 'patches':deepcopy(patches), 'reason':reason,
                'affected_ids':sorted(affected), 'affected_asset_ids':sorted(affected_assets)})
            self.state['task'] = revised
    def define(self, review=False):
        cursor = 'review_cursor' if review else 'definition_cursor'
        prompt = prompts.TASK_REVIEW if review else prompts.DEFINE
        while self.state[cursor] < len(self.state['index']):
            row = self.state['index'][self.state[cursor]]
            sources = blocks.excerpts(row['block']['units'], self.documents)
            data = {'selection':self.state['selection'], 'task':self.state['task'],
                    'sources':sources, 'index_page':row['entries'], 'boundary':BOUNDARY}
            reply = self.ask('task_review' if review else 'task_definition', prompt, data,
                lambda v:apply_patches(self.state['task'], v, self.sources))
            revised = apply_patches(self.state['task'], reply, self.sources)
            self.record_task(revised, reply['patches'], 'source_review' if review else 'source_definition')
            self.state.setdefault('definition_issues', {})[row['block']['id']] = reply['issues']
            self.state[cursor] += 1; self.save()
        if review:
            validate_task(self.state['task'], self.sources)
            self.phase('assets')
        else: self.phase('task_review')

    def assets(self):
        def persist(work):
            self.state['asset_workflow'] = work; self.save()
        inv = run_workflow(self.documents, self.config, self.model, self.state.get('asset_workflow'), persist,
                           task_context=task_card(self.state['task']))
        self.state['inventory'] = inv; self.phase('roles')

    def roles(self):
        task = self.state['task']; inv = self.state['inventory']
        for asset in inv['assets']:
            if asset['id'] in self.state['roles_by_asset']: continue
            assigned = {o for rows in self.state['roles_by_asset'].values() for r in rows for o in r['object_ids']}
            def validate(value):
                fields(value, 'roles'); require(isinstance(value['roles'],list) and value['roles'], 'asset has no roles')
                for row in value['roles']:
                    fields(row, 'id asset_id object_ids count stage_ids initial_state basis')
                    require(row['asset_id']==asset['id'] and isinstance(row['object_ids'],list), 'role asset mismatch')
                    require(not assigned.intersection(row['object_ids']), 'object is already assigned to another asset')
                validate_roles(self.all_roles()+value['roles'], task, inv, partial=True)
            reply = self.ask('asset_roles', prompts.ROLES, {'asset':asset, 'task':task,
                'unassigned_objects':[o['id'] for o in task['objects'] if o['id'] not in assigned],
                'other_types':[{'id':a['id'],'name':a['name']} for a in inv['assets'] if a['id']!=asset['id']]}, validate)
            self.state['roles_by_asset'][asset['id']] = reply['roles']; self.save()
        missing = {o['id'] for o in task['objects']} - {o for r in self.all_roles() for o in r['object_ids']}
        if missing:
            def validate_missing(value):
                fields(value, 'roles')
                require({o for r in value['roles'] for o in r['object_ids']}==missing, 'supplementary assignment must cover exactly the unassigned objects')
                validate_roles(self.all_roles()+value['roles'], task, inv)
            reply = self.ask('missing_roles', prompts.ROLES+'\nThis is an allocation repair: assign ALL listed unassigned objects across the existing types. Return only additional roles with unique ids.',
                {'task':task, 'unassigned_objects':sorted(missing), 'existing_roles':self.all_roles(),
                 'types':[{'id':a['id'],'name':a['name'],'requirements':a['requirements']} for a in inv['assets']]}, validate_missing)
            for role in reply['roles']: self.state['roles_by_asset'][role['asset_id']].append(role)
            self.save()
        self.recount(); self.phase('links')

    def all_roles(self):
        return [r for rows in self.state['roles_by_asset'].values() for r in rows]

    def recount(self):
        inv = self.state['inventory']; task = self.state['task']
        counts = validate_roles(self.all_roles(), task, inv)
        for asset in inv['assets']:
            rows = self.state['roles_by_asset'][asset['id']]
            objects = [o for o in task['objects'] if any(o['id'] in r['object_ids'] for r in rows)]
            explicit = bool(objects) and all(o['quantity']['origin']=='explicit' for o in objects) and all(r['object_ids'] for r in rows)
            refs = [e for o in objects for e in o['quantity']['evidence']]
            asset['quantity'] = {'value':counts[asset['id']], 'origin':'explicit' if explicit else 'inferred',
                                 'basis':'Allocation from task instance roles: '+', '.join(r['id'] for r in rows),
                                 'evidence':list({digest(e):e for e in refs}.values())}
        # After reassignment, a quantity conflict must no longer reference the old field.
        quantity_conflicts = [c for c in inv['conflicts'] if c['target_key']=='quantity']
        if quantity_conflicts:
            self.state.setdefault('quantity_reconciliation', []).append({'task_revision':self.state['task_revision'],
                'prior_conflicts':quantity_conflicts, 'counts':counts, 'basis':'Task object allocation supersedes type-level quantity choices.'})
            inv['conflicts'] = [c for c in inv['conflicts'] if c['target_key']!='quantity']
        validate_inventory(inv, self.sources, self.config.max_assets); self.save()

    def stage_packet(self, stage):
        roles = [r for r in self.all_roles() if stage['id'] in r['stage_ids']]
        aids = {r['asset_id'] for r in roles}
        return {'stage':stage, 'roles':roles, 'assets':[a for a in self.state['inventory']['assets'] if a['id'] in aids],
                'objects':[o for o in self.state['task']['objects'] if o['id'] in stage['object_ids']],
                'task_summary':{k:self.state['task'][k] for k in ('objective','selected_flow','preparations','final_conditions')}}

    def links(self):
        for stage in self.state['task']['stages']:
            if stage['id'] in self.state['links_by_stage']: continue
            reply = self.ask('task_interactions', prompts.LINKS, self.stage_packet(stage),
                lambda v:validate_links(v, self.state['task'], self.state['inventory']))
            self.state['links_by_stage'][stage['id']] = reply; self.save()
        self.phase('references')

    def all_links(self):
        # In-stage ids carry a stage prefix so different stages do not use the same local ids.
        result = {k:[] for k in ('interactions','mating_requirements','assistance_needs')}
        for sid, links in self.state['links_by_stage'].items():
            for key, rows in links.items():
                result[key].extend(dict(r, id='link_'+digest([sid,key,r['id']])[:24]) for r in rows)
        return result

    def review_delivery(self):
        for stage in self.state['task']['stages']:
            packet = self.stage_packet(stage)
            packet['links'] = self.state['links_by_stage'][stage['id']]
            packet['references'] = [a for a in self.state.get('assets',[]) if a['asset_id'] in {r['asset_id'] for r in packet['roles']}]
            packet['open_source_concerns'] = list(dict.fromkeys(i for issues in self.state.get('definition_issues',{}).values() for i in issues))
            packet_key = digest(packet)
            if self.state['reviews'].get(stage['id'], {}).get('input_sha256') == packet_key: continue
            def validate(value):
                fields(value, 'issues'); require(isinstance(value['issues'],list), 'review issues must be a list')
                for issue in value['issues']:
                    fields(issue, 'id kind target_id description evidence')
                    require(issue['kind'] in {'task','asset','roles','links','reference'}, 'unknown review issue kind')
                    for key in ('id','target_id','description'): text(issue[key])
                    evidence(issue['evidence'], source_map(self.sources))
            reply = self.ask('delivery_review', prompts.DELIVERY_REVIEW, packet, validate)
            self.state['reviews'][stage['id']] = dict(reply, input_sha256=packet_key); self.save()
        issues = [i for r in self.state['reviews'].values() for i in r['issues'] if i['kind']!='reference']
        if issues:
            self.repair(issues)
        else: self.phase('publish')

    def repair(self, issues):
        require(self.state['repair_round'] < self.config.repair_attempts,
                'targeted revision attempts exhausted; the specific unresolved issues are saved in the review record')
        prompt = prompts.COMMON + '''Repair only evidenced issues in this task package. Return
{task_changes:{patches:[],issues:[]}, assets:[full replacement inventory asset rows],
 retire_assets:[{asset_id,replacement_id:null|existing_asset_id,reason,evidence}],
 roles:[{asset_id,roles:[complete replacement role rows]}], links:[{stage_id,value:complete link object}],reason:string}.
Use the provided task patch schema. Assets must keep existing ids; you may add necessary missing
assets using the inventory schema, with a role and a links entry. Retire only evidenced duplicate
or out-of-scope types, with explicit replacement or exclusion reasons; update every affected role
and interaction. Do not delete requirements for convenience, weaken the end state or invent
product facts. Unchanged lists must be empty. '''
        # Each time, repair the issues of only one bounded stage instead of sending all references and the full text.
        issue = issues[0]
        task = self.state['task']; inv = self.state['inventory']
        relevant = next(s for s in task['stages'] if any(i==issue for i in self.state['reviews'].get(s['id'],{}).get('issues',[])))
        data = {'issues':[issue], 'task':task, 'packet':self.stage_packet(relevant),
                'links':self.state['links_by_stage'][relevant['id']], 'task_patch_schema':prompts.PATCH_FORMAT,
                'asset_schema':__import__('asset_prep.prompts',fromlist=['INVENTORY']).INVENTORY}
        def candidate(value):
            fields(value, 'task_changes assets retire_assets roles links reason'); text(value['reason'])
            new_task = apply_patches(task, value['task_changes'], self.sources)
            require(all(p['evidence'] for p in value['task_changes']['patches']), 'task semantic patches must have source evidence')
            validate_task(new_task, self.sources)
            new_inv = deepcopy(inv); assets = {a['id']:a for a in new_inv['assets']}
            for asset in value['assets']:
                require(isinstance(asset,dict), 'an asset revision must be a complete object')
                if asset.get('id') not in assets:
                    new_inv['steps'].append({'id':'repair_'+digest(asset['id'])[:24], 'description':'Additional physical dependency identified during task review.',
                        'asset_ids':[asset['id']], 'evidence':asset['evidence']})
                assets[asset['id']] = deepcopy(asset)
            retired = {}
            require(isinstance(value['retire_assets'],list), 'retire_assets must be a list')
            for row in value['retire_assets']:
                fields(row, 'asset_id replacement_id reason evidence'); text(row['reason'])
                evidence(row['evidence'],source_map(self.sources),required=True)
                aid = row['asset_id']; replacement = row['replacement_id']
                require(aid in assets and aid not in retired, 'retiring an unknown or duplicate asset')
                require(replacement is None or (replacement in assets and replacement!=aid), 'invalid merge target')
                retired[aid] = replacement
            require(not any(a in retired for a in retired.values() if a), 'asset merge chains are forbidden; name the retained type directly')
            for aid in retired: assets.pop(aid)
            for asset in assets.values(): asset['parent_id'] = retired.get(asset['parent_id'],asset['parent_id'])
            for step in new_inv['steps']:
                step['asset_ids'] = list(dict.fromkeys(retired.get(a,a) for a in step['asset_ids'] if retired.get(a,a)))
            new_inv['steps'] = [s for s in new_inv['steps'] if s['asset_ids']]
            new_inv['relations'] = [dict(r, subject=retired.get(r['subject'],r['subject']), object=retired.get(r['object'],r['object']))
                for r in new_inv['relations'] if retired.get(r['subject'],r['subject']) and retired.get(r['object'],r['object'])]
            new_inv['conflicts'] = [dict(c, asset_id=retired.get(c['asset_id'],c['asset_id']))
                for c in new_inv['conflicts'] if retired.get(c['asset_id'],c['asset_id'])]
            new_inv['assets'] = list(assets.values()); validate_inventory(new_inv, self.sources, self.config.max_assets)
            from .extraction import validate_unique_assets
            validate_unique_assets(new_inv)
            roles = deepcopy(self.state['roles_by_asset'])
            for aid in retired: roles.pop(aid,None)
            for row in value['roles']:
                fields(row, 'asset_id roles'); require(row['asset_id'] in assets, 'unknown role asset')
                roles[row['asset_id']] = row['roles']
            validate_roles([r for rows in roles.values() for r in rows], new_task, new_inv)
            links = deepcopy(self.state['links_by_stage'])
            links = {sid:v for sid,v in links.items() if sid in {s['id'] for s in new_task['stages']}}
            for row in value['links']:
                fields(row, 'stage_id value'); require(row['stage_id'] in {s['id'] for s in new_task['stages']}, 'unknown interaction stage')
                validate_links(row['value'], new_task, new_inv); links[row['stage_id']] = row['value']
            require(set(links)=={s['id'] for s in new_task['stages']}, 'after the task stage revision, matching interactions are missing')
            for link in links.values(): validate_links(link, new_task, new_inv)
            require((new_task,new_inv,roles,links)!=(task,inv,self.state['roles_by_asset'],self.state['links_by_stage']), 'the revision makes no actual change')
            return new_task,new_inv,roles,links
        value = self.ask('targeted_repair', prompt, data, candidate)
        new_task,new_inv,roles,links = candidate(value)
        self.state.setdefault('repair_history',[]).append({'issues':[issue], 'change':value,
                                                          'prior_reviews':deepcopy(self.state['reviews'])})
        self.state.setdefault('definition_issues',{})['latest_repair'] = value['task_changes']['issues']
        self.record_task(new_task, value['task_changes']['patches'], value['reason'])
        active_stages = {s['id'] for s in new_task['stages']}
        self.state.update(inventory=new_inv, roles_by_asset=roles, links_by_stage=links,
                          reviews={k:v for k,v in self.state['reviews'].items() if k in active_stages})
        self.state['repair_round'] += 1; self.recount(); self.phase('references')

    def advance_requirements(self):
        phases = {'index':self.index, 'select':self.select, 'define':self.define,
                  'task_review':lambda:self.define(True), 'assets':self.assets, 'roles':self.roles, 'links':self.links}
        while self.state['phase'] in phases: phases[self.state['phase']]()
        return self.state
