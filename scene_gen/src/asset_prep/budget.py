"""基于实际资产的后续预算规划；只提出需求，不自动增加上限。"""

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
            'assumptions': '按每项计划查询及一次成功选型估算，建议另留20%模型调用余量；不含未知修复、后续新增资产及真实费用。'}


def check_reference_capacity(plan, config, totals):
    if (totals['model_calls'] + plan['planned_model_calls'] > config.max_model_calls or
            totals['search_calls'] + plan['planned_search_calls'] > config.max_search_calls):
        raise BudgetExceeded('参考资料阶段预算不足，尚未开始检索。请查看 asset_package.json 的 reference_budget_plan；'
                             f"建议模型总次数至少 {plan['suggested_max_model_calls']}，检索总次数至少 {plan['suggested_max_search_calls']}；上限未自动修改")
