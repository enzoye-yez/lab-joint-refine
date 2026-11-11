"""Persisted cumulative ceilings; unused allocation flows forward, never resets."""
from .errors import BudgetExceeded, ValidationError


class PhaseBudgetExceeded(BudgetExceeded):
    pass


# Tenths of the run budget each phase may reach. Scene generation (assets and their repair)
# shares the operation ceiling; the final tenth stays reserved for the final stage.
SHARES={'scene':9,'operation':9,'final':10}


def ceilings(entry, limits):
    return {p:{k:entry[k]+int(max(0,v-entry[k])*n//10) for k,v in limits.items()} for p,n in SHARES.items()}


def enter(store, phase):
    if phase not in SHARES:raise ValidationError('unknown budget phase')
    if 'phase_budget' not in store.state:
        totals=store.totals()
        limits={'model_calls':store.config.max_model_calls,'accounted_tokens':store.config.max_reserved_tokens}
        store.state['phase_budget']={'version':1,'entry':totals,'ceilings':ceilings(totals,limits),'phase':phase}
    allocation=store.state['phase_budget']
    if phase=='scene' and allocation['phase']!='scene':
        raise ValidationError('scene allowance cannot be reopened; operation-stage repairs still use the joint operation allowance')
    allocation['phase']=phase
    store.save()


def check(store, tokens=0, model_call=True):
    allocation=store.state.get('phase_budget')
    if not allocation:return
    totals=store.totals();ceiling=allocation['ceilings'][allocation['phase']]
    for key,extra in (('model_calls',int(model_call)),('accounted_tokens',tokens)):
        if totals[key]+extra>ceiling[key]:
            raise PhaseBudgetExceeded('current phase allowance exhausted; saved responses and later-phase budgets are kept: '+allocation['phase']+'/'+key)
