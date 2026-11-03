"""Room arrangement from actual asset geometry, before fixtures and tools are designed against it."""
from asset_prep.storage import digest
from .candidate import usable
from .layout import validate_layout


def refine(run):
    """Arrange the room from the actual asset geometry once every asset can be loaded.

    The plan's semantics (bench, order, floor, supports) stay; arrangement.arrange computes the
    coordinates, docks and robot home. No model call: a deterministic arrangement replaced the
    former free-form layout refinement, which kept the first guess unchanged."""
    plan = run.state["plan"]
    if "layout" not in plan:
        return
    assets = [j for j in plan["jobs"] if j["role"]=="asset"]
    if not all(usable(run,run.state["jobs"].get(j["id"])) for j in assets):
        return
    record = run.state.setdefault("layout_refinement", {"attempts":[], "complete":False})
    input_key = digest({j["id"]:run.state["jobs"][j["id"]]["assembly_readiness"] for j in assets})
    if record.get("input_key") == input_key and record["complete"]:
        return
    from .arrangement import arrange, shapes_for
    updated = arrange(plan, shapes_for(run, plan), run.env)
    validate_layout(updated, run.env)
    run.state["plan"] = updated
    record.update(input_key=input_key, complete=True, method="deterministic_rows",
                  room=updated["layout"]["room"]["size"], tables=len(updated["layout"]["tables"]))
    run.store.save()
