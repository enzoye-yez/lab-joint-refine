"""Limited mounting zone for table-edge clamps; exempts only declared parts from the tabletop boundary, never from collision or support checks."""

import numpy as np
from asset_generation.contracts import require

POLICY = {'max_overhang_m':.05,'max_inward_m':.05,
          'rule':'Only explicit collision geoms of a fixture may be declared as table-edge clamps; all other parts must stay within the tabletop. Collision and support checks run as usual.'}


def validate_mounts(entries, plan):
    require(isinstance(entries,list) and len(entries)<=128,'invalid edge-mount declaration')
    items={i['id']:i for i in plan['instances']}
    jobs={j['id']:j for j in plan['jobs']}
    seen=set()
    for row in entries:
        require(isinstance(row,dict) and set(row)=={'instance','geoms','edge','reason'},'invalid edge-mount fields')
        require(isinstance(row['instance'],str),'edge-mount instance id must be a string')
        item=items.get(row['instance'])
        require(item and item['placement']=='table' and jobs[item['job']]['role']=='fixture',
                'edge mounts apply only to tabletop auxiliary fixtures')
        require(isinstance(row['edge'],str) and row['edge'] in {'x+','x-','y+','y-'},'invalid table-edge direction')
        require(isinstance(row['reason'],str) and row['reason'],'edge mount must state its purpose')
        require(isinstance(row['geoms'],list) and 0<len(row['geoms'])<=128,'invalid edge-mount part list')
        for name in row['geoms']:
            require(isinstance(name,str) and name and '__' not in name,'use asset-local geom names')
            pair=(row['instance'],name)
            require(pair not in seen,'edge-mount part declared more than once')
            seen.add(pair)
    return entries


def boundary_evidence(model, geometry, item, env, entries):
    from .geometry import owner
    half=np.array(env['table']['size'][:2])/2
    selected={item['id']+'__'+name:row for row in entries if row['instance']==item['id'] for name in row['geoms']}
    owned={model.geom(g).name:g for g in geometry if owner(model,g)==item['id']}
    missing=sorted(set(selected)-set(owned))
    violations=[]
    for name,g in owned.items():
        lo,hi=geometry[g]
        if name not in selected:
            valid=bool(np.all(lo[:2]>=-half) and np.all(hi[:2]<=half))
        else:
            row=selected[name];axis=0 if row['edge'][0]=='x' else 1;other=1-axis
            sign=1 if row['edge'][1]=='+' else -1
            a,b=sorted((sign*lo[axis],sign*hi[axis]));edge=half[axis]
            valid=bool(a>=edge-POLICY['max_inward_m'] and b<=edge+POLICY['max_overhang_m']
                       and lo[other]>=-half[other] and hi[other]<=half[other]
                       and lo[2]>=env['table']['height']-env['table']['size'][2]-POLICY['max_overhang_m']
                       and hi[2]<=env['table']['height']+POLICY['max_overhang_m'])
        if not valid:
            violations.append({'geom':name,'bounds':[lo.tolist(),hi.tolist()]})
    return {'valid':not missing and not violations,'missing_mount_geoms':missing,
            'boundary_violations':violations[:32],'boundary_violation_count':len(violations),
            'edge_mount_geoms':sorted(selected)}
