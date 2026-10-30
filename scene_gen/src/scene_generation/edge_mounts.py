"""桌边夹持的有限安装区；只豁免声明部件的桌面边界，不豁免碰撞或支撑。"""

import numpy as np
from asset_generation.contracts import require

POLICY = {'max_overhang_m':.05,'max_inward_m':.05,
          'rule':'仅fixture的明确碰撞几何可声明桌边夹持；其他部件必须留在桌面内。碰撞和承托检查照常执行。'}


def validate_mounts(entries, plan):
    require(isinstance(entries,list) and len(entries)<=128,'桌边安装声明无效')
    items={i['id']:i for i in plan['instances']}
    jobs={j['id']:j for j in plan['jobs']}
    seen=set()
    for row in entries:
        require(isinstance(row,dict) and set(row)=={'instance','geoms','edge','reason'},'桌边安装字段无效')
        require(isinstance(row['instance'],str),'桌边安装实例编号须为字符串')
        item=items.get(row['instance'])
        require(item and item['placement']=='table' and jobs[item['job']]['role']=='fixture',
                '桌边安装仅适用于桌面辅助固定装置')
        require(isinstance(row['edge'],str) and row['edge'] in {'x+','x-','y+','y-'},'桌边方向无效')
        require(isinstance(row['reason'],str) and row['reason'],'桌边安装须说明用途')
        require(isinstance(row['geoms'],list) and 0<len(row['geoms'])<=128,'桌边部件列表无效')
        for name in row['geoms']:
            require(isinstance(name,str) and name and '__' not in name,'使用资产内的局部几何名')
            pair=(row['instance'],name)
            require(pair not in seen,'桌边部件重复声明')
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
