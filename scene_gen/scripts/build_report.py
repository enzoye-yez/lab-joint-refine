"""Private HTML report for one scene run: scene views, and per asset and configuration the
studio stills, turntable video, mechanism-by-mechanism check results and review findings.

Reads reports/<run>/figures/figures.json (render_figures.py), the run's check rows and review
findings, and the benchmark reference images. Writes reports/<run>/index.html.
Serve it only as described in .claude/skills/np002-private-html/SKILL.md.
"""
import argparse
import html
import json
import shutil
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


def configuration_block(entry, rows, multiple):
    prefix = entry["name"]+":" if multiple else ""
    own = [dict(r, id=r["id"][len(prefix):]) for r in rows if r["id"].startswith(prefix)] if multiple else rows
    stills = "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in entry.get("stills", []))
    video = (f'<video controls loop muted playsinline preload="metadata" src="{esc(entry["video"])}"></video>'
             if entry.get("video") else '<p class="muted">无视频</p>')
    pairs = "".join(f'<div><div class="mono muted" style="font-size:12px">{esc(name)}</div><div class="pair">'
                    + "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in paths) + "</div></div>"
                    for name, paths in entry.get("mechanism_stills", {}).items())
    mechanisms = entry.get("mechanisms") or []
    title = f'<h3>配置 <span class="mono">{esc(entry["name"])}</span></h3>' if multiple else ""
    return f"""{title}<div class="two"><div>{video}</div><div class="grid" style="grid-template-columns:1fr 1fr">{stills}</div></div>
<p class="muted mono" style="font-size:12px;margin:0">MuJoCo：mujoco/{esc(entry.get('mujoco',''))}</p>
{mechanism_table(mechanisms, own) if mechanisms else '<p class="muted">未声明机构</p>'}
<details><summary>机构两端姿态（初始 / 极限）</summary><div class="grid">{pairs}</div></details>"""


def assembled_block(record):
    """The job's instances as placed together in the scene: connections between configurations
    (hoses between a cooler and its host) are only visible here."""
    paths = record.get("assembled") or []
    if not paths:
        return ""
    images = "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in paths)
    note = "，各配置之间的连接（如软管）只在这里可见" if len(record.get("configurations", [])) > 1 else ""
    return f'<div><h3>场景中装配</h3><p class="muted" style="margin:0 0 6px">本资产在场景里的全部实例，其余物体隐藏{note}。</p><div class="grid">{images}</div></div>'


def listing(items, empty):
    return ("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>") if items else f'<p class="muted">{empty}</p>'


def asset_card(record, job, plan_job, references):
    numerical = job.get("feedback", {}).get("numerical", {})
    visual = job.get("feedback", {}).get("visual", {})
    rows = numerical.get("checks", [])
    failing = [r for r in rows if r.get("status") == "fail"]
    counts = {s: sum(r.get("status") == s for r in rows) for s in ("pass", "fail", "partial")}
    configurations = record.get("configurations", [])
    multiple = len(configurations) > 1
    refs = "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in references)
    coverage = visual.get("motion_coverage") or []
    coverage_html = ("<ul>" + "".join(f'<li>{pill(c.get("verdict"))} {esc(c.get("requirement"))} → <span class="mono">{esc(c.get("mechanism"))}</span>'
                                      f'<br><span class="muted">{esc(c.get("evidence",""))}</span></li>' for c in coverage) + "</ul>") if coverage else '<p class="muted">无</p>'
    fail_html = ("<ul>" + "".join(f'<li><span class="mono">{esc(r["id"])}</span>：{esc(str(r.get("detail",""))[:240])}</li>' for r in failing[:40]) + "</ul>"
                 if failing else '<p class="muted">无失败条目</p>')
    deferred = [f'[{d.get("stage")}] {d.get("detail")}' for d in (job.get("deferred_checks") or []) if isinstance(d, dict)]
    return f"""<article class="card" id="{esc(record['job'])}">
<div class="head"><h3>{esc(record['job'])} · {esc(', '.join(record['asset_ids']))}</h3>
<span>{pill(job.get('status'))} 视觉审查 {pill(visual.get('verdict','—'))} 数值检查 {pill(numerical.get('status','—'))}</span></div>
<details><summary>资料要求</summary><p class="muted" style="white-space:pre-wrap">{esc(plan_job.get('requirements',''))}</p></details>
<div class="stats"><div><b>{counts['pass']}</b><span>检查通过</span></div><div><b>{counts['fail']}</b><span>检查失败</span></div>
<div><b>{counts['partial']}</b><span>未完成</span></div><div><b>{len(configurations)}</b><span>配置</span></div>
<div><b>{len(visual.get('findings') or [])}</b><span>审查意见</span></div></div>
<details><summary>原始参考图（{len(references)} 张）</summary><div class="grid">{refs}</div></details>
{assembled_block(record)}
{''.join(configuration_block(e, rows, multiple) for e in configurations)}
<div class="two"><div><h3>视觉审查意见</h3>{listing(visual.get('findings') or [], '无')}
<h3>未核实</h3>{listing(visual.get('unverified') or [], '无')}<h3>留给后续阶段</h3>{listing(deferred, '无')}</div>
<div><h3>失败检查</h3>{fail_html}<h3>审查：运动要求覆盖</h3>{coverage_html}</div></div>
</article>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = PROJECT/"reports"/run.name
    figures = json.loads((report/"figures"/"figures.json").read_text(encoding="utf-8"))
    state_all = read_state(run/"state.json")
    state = state_all["scene"]
    plan = {j["id"]: j for j in state["plan"]["jobs"]}
    catalogue = state_all["identity"]["catalogue"]["assets"]
    benchmark = PROJECT/"benchmark"/"NP-002"
    references = {}
    for asset_id, entry in catalogue.items():
        for image in entry.get("images", []):
            source = benchmark/image["path"]
            if source.is_file():
                target = report/"references"/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                references.setdefault(asset_id, []).append(f"references/{source.name}")
    assets = figures["assets"]
    scene = figures.get("scene") or {}
    scene_html = "".join(f'<img loading="lazy" src="{esc(p)}" alt="">' for p in scene.get("views", []))
    statuses = [state["jobs"].get(a["job"], {}).get("status") for a in assets]
    nav = "".join(f'<a href="#{esc(a["job"])}">{esc(a["job"])}</a>' for a in assets)
    archive = figures.get("mujoco_archive")
    body = f"""<title>NP-002 资产排查</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style><div class="wrap">
<header><div class="mono muted">{esc(run.name)} · 场景状态 {esc(state.get('status'))}</div><h1>NP-002 场景、资产与机构</h1>
<div class="stats"><div><b>{len(assets)}</b><span>资产任务</span></div><div><b>{statuses.count('candidate_ready')}</b><span>资产验收通过</span></div>
<div><b>{statuses.count('failed')}</b><span>待修复</span></div>
<div><b>{sum(len(a.get('configurations', [])) for a in assets)}</b><span>配置</span></div></div>
{f'<p><a href="{esc(archive)}" download>下载全部资产的 MuJoCo 包（{esc(archive)}）</a>：每个资产一个目录，'
 '<span class="mono">&lt;配置&gt;_poses.xml</span> 含初始姿态及每个机构的极限姿态关键帧；'
 '<span class="mono">python play_mechanisms.py &lt;配置&gt;_poses.xml</span> 依次播放各机构。</p>' if archive else ''}</header>
<section><h2>整体场景</h2>{'<p class="muted">未发布的场景候选（' + esc(scene.get("directory")) + '），批次仍在修复，碰撞未清零。</p>' if scene.get("candidate") else ''}<div class="grid">{scene_html or '<p class="muted">无场景渲染</p>'}</div></section>
<section><h2>资产</h2><nav>{nav}</nav></section>
{''.join(asset_card(a, state['jobs'].get(a['job'], {}), plan.get(a['job'], {}),
                    [r for i in a['asset_ids'] for r in references.get(i, [])]) for a in assets)}
</div>"""
    (report/"index.html").write_text("<!doctype html><meta charset=\"utf-8\">"+body, encoding="utf-8")
    print("wrote", report/"index.html")


if __name__ == "__main__":
    main()
