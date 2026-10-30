"""Persisted cumulative ceilings; unused allocation flows forward, never resets."""
from .errors import BudgetExceeded, ValidationError


class PhaseBudgetExceeded(BudgetExceeded):
    pass


def enter(store, phase):
    order=('scene','operation','final')
    if phase not in order:raise ValidationError('未知预算阶段')
    if 'phase_budget' not in store.state:
        totals=store.totals()
        limits={'model_calls':store.config.max_model_calls,'accounted_tokens':store.config.max_reserved_tokens}
        store.state['phase_budget']={'version':1,'entry':totals,
            'ceilings':{p:{k:totals[k]+int(max(0,v-totals[k])*n//10) for k,v in limits.items()}
                for p,n in zip(order,(4,9,10))},'phase':phase}
    allocation=store.state['phase_budget']
    if phase=='scene' and allocation['phase']!='scene':
        raise ValidationError('不能重新开启场景额度；操作阶段返修仍使用联合操作额度')
    allocation['phase']=phase
    store.save()


def check(store, tokens=0, model_call=True):
    allocation=store.state.get('phase_budget')
    if not allocation:return
    totals=store.totals();ceiling=allocation['ceilings'][allocation['phase']]
    for key,extra in (('model_calls',int(model_call)),('accounted_tokens',tokens)):
        if totals[key]+extra>ceiling[key]:
            raise PhaseBudgetExceeded('当前阶段额度已用尽；保留已保存响应及后续阶段预算：'+allocation['phase']+'/'+key)
