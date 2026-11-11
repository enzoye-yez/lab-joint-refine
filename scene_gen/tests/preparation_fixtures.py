"""任务准备开发者合成规程和固定响应；不访问网络。"""

import json
from copy import deepcopy

TEXT = ('Main route: use two identical capped tubes, one sample and one receiver. '
        'Reagents are prepared beforehand. Open both caps, then place both tubes in the work area. '
        'Finish with both tubes open and their identities preserved. '
        'Alternative route: use a centrifuge; this is not the main route.')


def refs(): return [{'source_id':'pdf:p1', 'quote':TEXT}]


def documents():
    return {'files':{'pdf':{'sha256':'developer-fixture', 'name':'synthetic.pdf'}},
            'sources':[{'id':'pdf:p1','kind':'pdf','text':TEXT}], 'warnings':[]}


def task():
    objects = [{'id':name,'description':name+' tube','initial_state':'Closed, prepared contents',
                'contents':['Prepared contents'], 'quantity':{'value':1,'origin':'explicit','basis':'One per role','evidence':refs()},
                'evidence':refs()} for name in ('sample','receiver')]
    return {'objective':'Prepare two open tubes for handling.', 'selected_flow':'Main route',
            'selection_reason':'Explicit main route precedes the alternative.',
            'selection_evidence':refs(),
            'excluded_branches':[{'name':'Centrifuge route','reason':'Alternative branch','evidence':refs()}],
            'preparations':[{'id':'prepared','description':'Reagents prepared before execution',
                             'object_ids':['sample','receiver'],'basis':'Default preparation boundary','evidence':refs()}],
            'objects':objects,
            'stages':[{'id':'open','description':'Open both caps','depends_on':[], 'object_ids':['sample','receiver'],
                       'outcome':'Both caps open','evidence':refs()},
                      {'id':'place','description':'Place both tubes','depends_on':['open'], 'object_ids':['sample','receiver'],
                       'outcome':'Both tubes placed with identities preserved','evidence':refs()}],
            'final_conditions':[{'id':'ready','description':'Both tubes open, identity preserved',
                                 'object_ids':['sample','receiver'],'evidence':refs()}],
            'abstractions':['Contents represented as state, no chemical reactions.'], 'assumptions':[], 'issues':[]}


def inventory():
    return {'assets':[{'id':'tube','name':'Capped tube','category':'container','aliases':[],
            'quantity':{'value':2,'origin':'explicit','basis':'Two tubes','evidence':refs()},
            'requirements':[{'key':'cap','value':'openable','unit':'','origin':'explicit','basis':'Open caps','evidence':refs()}],
            'operations':['Open','Place'],'evidence':refs(),'parent_id':None,'queries':['manufacturer capped tube']}],
            'steps':[{'id':'handling','description':'Open and place','asset_ids':['tube'],'evidence':refs()}],
            'relations':[], 'conflicts':[], 'assumptions':[], 'issues':[]}


ENV = {'version':'synthetic', 'robot_count':1, 'robot':{'name':'developer dual-arm robot'},
       'workspace_design':{'tables':'variable'}}


class Transport:
    def __init__(self): self.operations=[]; self.requests=[]
    def post(self, url, payload, key, timeout):
        assert url.endswith('/chat/completions'), 'Unexpected network path in offline fixture'
        data = json.loads(payload['messages'][1]['content'].split('\nTask data follows:\n',1)[1])
        op = data.get('operation'); self.operations.append(op); self.requests.append(deepcopy(data))
        value = self.respond(data)
        return {'model':'developer_fixture', 'usage':{'completion_tokens':300,'total_tokens':1000},
                'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}

    def respond(self, data):
        op = data.get('operation')
        if op=='protocol_index': value={'entries':[{'kind':'branch','summary':'Explicit main route with two tubes; alternative centrifuge.','evidence':refs()}]}
        elif op=='flow_selection': value={'name':'Main route','objective':'Prepare two open tubes for handling.', 'reason':'Explicit main route','evidence':refs()}
        elif op=='task_definition':
            value={'patches':[], 'issues':[]}
            for section, rows in task().items():
                if section in {'objects','stages','preparations','final_conditions'}:
                    value['patches'] += [{'section':section,'id':row['id'],'value':row,'reason':'Source fact','evidence':refs()} for row in rows]
                else: value['patches'].append({'section':section,'id':None,'value':rows,'reason':'Source fact','evidence':refs()})
        elif op=='task_review': value={'patches':[],'issues':[]}
        elif op=='local_extract': value={'inventory':inventory(),'consumed_units':len(data['units'])}
        elif op in {'identity','field_identity'}: value={'target':data['page'][0]['id'],'reason':'Same type or field'}
        elif op=='field_value': value={'take':'existing','reason':'Same source'}
        elif op=='asset_roles':
            value={'roles':[{'id':name+'_role','asset_id':data['asset']['id'],'object_ids':[name],'count':1,
                            'stage_ids':['open','place'],'initial_state':'Closed tube','basis':'Independent '+name+' object'}
                            for name in ('sample','receiver')]}
        elif op=='task_interactions':
            sid=data['stage']['id']; value={'interactions':[{'id':'handling','description':data['stage']['description'],
                'asset_ids':[data['assets'][0]['id']], 'stage_ids':[sid], 'basis':'Source operation'}], 'mating_requirements':[], 'assistance_needs':[]}
        elif op=='delivery_review': value={'issues':[]}
        else: raise AssertionError('Unexpected fixture operation '+str(op))
        return value
    def request(self,*args,**kwargs): raise AssertionError('No web/image calls in requirements-only fixture')
