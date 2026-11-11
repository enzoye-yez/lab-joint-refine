"""Follow-up budget planning based on actual assets; only states requirements and never raises limits automatically."""

import math
from .errors import BudgetExceeded


def reference_plan(assets, config, totals):
    searches = sum(min(config.queries_per_asset, len(a['queries'])) for a in assets)
    selections = len(assets)
    models = selections + (searches if config.search_provider == 'model_web' else 0)
    return {'asset_count': len(assets), 'planned_search_calls': searches, 'planned_selection_calls': selections,
            'planned_model_calls': models, 'remaining_model_calls': max(0, config.max_model_calls-totals['model_calls']),
            'suggested_max_model_calls': totals['model_calls'] + models + math.ceil(models * 0.2),
            'suggested_max_search_calls': totals['search_calls'] + searches,
            'assumptions': 'Estimated from each planned query plus one successful selection; keep an extra 20% margin of model calls; excludes unknown repairs, later-added assets and real cost.'}


def check_reference_capacity(plan, config, totals):
    if (totals['model_calls'] + plan['planned_model_calls'] > config.max_model_calls or
            totals['search_calls'] + plan['planned_search_calls'] > config.max_search_calls):
        raise BudgetExceeded('Insufficient budget for the reference stage; search has not started. See reference_budget_plan in asset_package.json; '
                             f"suggested total model calls at least {plan['suggested_max_model_calls']}, total search calls at least {plan['suggested_max_search_calls']}; limits were not changed automatically")
