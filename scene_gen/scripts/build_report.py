"""Private HTML report for one scene run: scene views, every asset with studio stills, its
turntable video, mechanism-by-mechanism check results, review verdict and rubric evaluation.

Reads reports/<run>/figures/figures.json (render_figures.py) and, if present,
reports/<run>/mechanism_eval.json (evaluate_mechanisms.py). Writes reports/<run>/index.html.
Serve it only as described in .claude/skills/np002-private-html/SKILL.md.
"""
import argparse
import html
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))

from asset_prep.state_archive import read_state  # noqa: E402

CSS = """
:root{--bg:#f4f6f7;--surface:#fff;--line:#d9dfe3;--fg:#1a2227;--muted:#5b6870;--accent:#1d6b86;
--ok:#2c7a4b;--ok-bg:#e2f1e7;--bad:#b3412d;--bad-bg:#f8e3de;--warn:#93670f;--warn-bg:#f6ecd6;
--display:"Archivo","Noto Sans SC",system-ui,sans-serif;--body:"Noto Sans SC","PingFang SC",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121719;--surface:#1a2125;--line:#2d373d;--fg:#e2e8eb;--muted:#93a1a9;
--accent:#62b2cd;--ok:#6cc290;--ok-bg:#1c3326;--bad:#ee8a74;--bad-bg:#3d231d;--warn:#dfb15b;--warn-bg:#3a2f19}}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--body);margin:0}
.wrap{max-width:1280px;margin:0 auto;padding:28px 16px 64px;display:grid;gap:32px}
h1,h2,h3{font-family:var(--display);margin:0;line-height:1.25}h1{font-size:28px}h2{font-size:21px}h3{font-size:17px}
.muted{color:var(--muted)}.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.stats{display:flex;flex-wrap:wrap;gap:8px 28px}.stats b{font:500 22px var(--mono);display:block}.stats span{font-size:13px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.grid img,.grid video{width:100%;border-radius:4px;border:1px solid var(--line);background:#eef0f2}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px;display:grid;gap:14px;min-width:0}
.head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:baseline}
.pill{font-size:12.5px;font-weight:500;padding:1px 9px;border-radius:999px;white-space:nowrap}
.ok{color:var(--ok);background:var(--ok-bg)}.bad{color:var(--bad);background:var(--bad-bg)}.warn{color:var(--warn);background:var(--warn-bg)}
.two{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:16px}@media(max-width:820px){.two{grid-template-columns:1fr}}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:500;font-size:12px}
td.id{font-family:var(--mono);font-size:12px;word-break:break-all}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}.pair img{width:100%;border:1px solid var(--line);border-radius:3px}
ul{margin:0;padding-left:18px}details summary{cursor:pointer;color:var(--accent);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px}nav a{font:12.5px var(--mono);color:var(--accent);text-decoration:none;border:1px solid var(--line);border-radius:4px;padding:2px 7px;background:var(--surface)}
"""

ROW_KINDS = (("mechanism_contract", "契约"), ("sampled_joint_sweep", "扫描"), ("mechanism_drive", "驱动/锁"),
             ("release_probe", "释放"), ("button_probe", "按钮"))


def esc(value):
    return html.escape(str(value if value is not None else ""))


def pill(status):
    cls = {"pass": "ok", "VERIFIED_PASS": "ok", "accept": "ok", "candidate_ready": "ok",
           "fail": "bad", "VERIFIED_FAIL": "bad", "repair": "bad", "failed": "bad"}.get(status, "warn")
    return f'<span class="pill {cls}">{esc(status)}</span>'


def mechanism_table(mechanisms, rows):
    by = {}
    for r in rows:
        name, _, mech = r["id"].partition(":")
        by.setdefault(mech, {})[name] = r
    out = ['<div class="scroll"><table><thead><tr><th>机构</th><th>资料要求</th><th>关节 / 锁</th>'
           + "".join(f"<th>{label}</th>" for kind, label in ROW_KINDS[1:4]) + "</tr></thead><tbody>"]
    for m in mechanisms:
        cells = by.get(m["id"], {})
        out.append(f'<tr><td class="id">{esc(m["id"])}</td><td>{esc(m.get("requirement"))}</td>'
                   f'<td class="id">{esc(", ".join(m.get("joints") or []))}<br>{esc(m.get("lock") or "")}</td>'
                   + "".join(f'<td>{pill(cells[k]["status"]) if k in cells else "—"}</td>' for k, _ in ROW_KINDS[1:4]) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def asset_card(record, job, plan_job, evaluation):
    numerical = job.get("feedback", {}).get("numerical", {})
    visual = job.get("feedback", {}).get("visual", {})
    rows = numerical.get("checks", [])
    failing = [r for r in rows if r.get("status") == "fail"]
    counts = {s: sum(r.get("status") == s for r in rows) for s in ("pass", "fail", "partial")}
    stills = "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in record.get("stills", []))
    video = (f'<video controls loop muted playsinline preload="metadata" src="{esc(record["video"])}"></video>'
             if record.get("video") else '<p class="muted">无视频</p>')
    pairs = "".join(f'<div><div class="mono muted" style="font-size:12px">{esc(name)}</div><div class="pair">'
                    + "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in paths) + "</div></div>"
                    for name, paths in record.get("mechanism_stills", {}).items())
    coverage = visual.get("motion_coverage") or []
    coverage_html = ("<ul>" + "".join(f'<li>{pill(c["verdict"])} {esc(c["requirement"])} → <span class="mono">{esc(c["mechanism"])}</span></li>'
                                      for c in coverage) + "</ul>") if coverage else '<p class="muted">无</p>'
    fail_html = ("<ul>" + "".join(f'<li class="mono">{esc(r["id"])}：{esc(str(r.get("detail",""))[:140])}</li>' for r in failing[:20]) + "</ul>"
                 if failing else '<p class="muted">无失败条目</p>')
    units = [e for e in evaluation if record["job"] in e.get("jobs", [])]
    eval_html = ("<ul>" + "".join(f'<li>{pill(e["status"])} <span class="mono">{esc(e["unit"])}</span> {esc(e["requirement"])}</li>' for e in units) + "</ul>"
                 if units else '<p class="muted">无对应 rubric 机构条目</p>')
    return f"""<article class="card" id="{esc(record['job'])}">
<div class="head"><h3>{esc(record['job'])} · {esc(', '.join(record['asset_ids']))}</h3>
<span>{pill(job.get('status'))} 审查 {pill(visual.get('verdict','—'))}</span></div>
<p class="muted" style="margin:0">{esc(plan_job.get('requirements',''))[:400]}</p>
<div class="stats"><div><b>{counts['pass']}</b><span>检查通过</span></div><div><b>{counts['fail']}</b><span>检查失败</span></div>
<div><b>{counts['partial']}</b><span>未完成</span></div><div><b>{len(record.get('mechanisms',[]))}</b><span>声明机构</span></div></div>
<div class="two"><div>{video}</div><div class="grid" style="grid-template-columns:1fr 1fr">{stills}</div></div>
<h3>机构</h3>{mechanism_table(record.get('mechanisms', []), rows) if record.get('mechanisms') else '<p class="muted">未声明机构</p>'}
<details><summary>机构两端姿态（初始 / 极限）</summary><div class="grid">{pairs}</div></details>
<div class="two"><div><h3>失败检查</h3>{fail_html}</div><div><h3>审查：运动要求覆盖</h3>{coverage_html}<h3>Rubric 机构条目</h3>{eval_html}</div></div>
</article>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = PROJECT/"reports"/run.name
    figures = json.loads((report/"figures"/"figures.json").read_text(encoding="utf-8"))
    eval_path = report/"mechanism_eval.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {"results": []}
    state = read_state(run/"state.json")["scene"]
    plan = {j["id"]: j for j in state["plan"]["jobs"]}
    assets = [a for a in figures["assets"]]
    for a in assets:
        a["stills"] = ["figures/"+p.split("figures/", 1)[-1] if not p.startswith("figures/") else p for p in a["stills"]]
    scene = figures.get("scene") or {}
    scene_html = "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in scene.get("views", []))
    ready = sum(state["jobs"][a["job"]].get("status") == "candidate_ready" for a in assets if a["job"] in state["jobs"])
    nav = "".join(f'<a href="#{esc(a["job"])}">{esc(a["job"])}</a>' for a in assets)
    body = f"""<title>NP-002 资产与机构报告</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style><div class="wrap">
<header><div class="mono muted">{esc(run.name)} · 场景状态 {esc(state.get('status'))}</div><h1>NP-002 场景、资产与机构</h1>
<div class="stats"><div><b>{len(assets)}</b><span>资产任务</span></div><div><b>{ready}</b><span>资产验收通过</span></div>
<div><b>{evaluation.get('by_status',{}).get('VERIFIED_PASS',0)}/{evaluation.get('units',0)}</b><span>rubric 机构条目通过</span></div>
<div><b>{evaluation.get('points',0):.2f}/{evaluation.get('max_points',0):.2f}</b><span>机构条目得分</span></div></div></header>
<section><h2>整体场景</h2><div class="grid">{scene_html or '<p class="muted">无场景渲染</p>'}</div></section>
<section><h2>资产</h2><nav>{nav}</nav></section>
{''.join(asset_card(a, state['jobs'].get(a['job'], {}), plan.get(a['job'], {}), evaluation.get('results', [])) for a in assets)}
</div>"""
    (report/"index.html").write_text("<!doctype html><meta charset=\"utf-8\">"+body, encoding="utf-8")
    print("wrote", report/"index.html")


if __name__ == "__main__":
    main()
