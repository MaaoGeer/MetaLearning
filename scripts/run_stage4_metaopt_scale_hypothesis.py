"""Validation-only test of the MetaOpt early-update-scale hypothesis.

The runner changes only an in-memory diagnostic multiplier after MetaOpt has
proposed its update.  It never changes the saved optimizer, training code, or
default inference behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_meta_model, build_meta_optimizer, load_artifacts  # noqa: E402
from src.data.pipeline import build_pipeline  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.task_manifest import manifest_raw_row_ids, read_task_manifest, sha256_file, write_task_manifest  # noqa: E402
from src.meta_learning.functional import functional_forward  # noqa: E402
from src.meta_optimizer.handcrafted import HandcraftedOptimizer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


STEPS, EARLY_STEPS, TARGET = 20, 5, .8
CHECKPOINTS = (0, 1, 2, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="42,52,62")
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--shot", type=int, default=3)
    p.add_argument("--q-query", type=int, default=10)
    p.add_argument("--alphas", default="1,2,5,10,15")
    return p.parse_args()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def l2(values: Iterable[torch.Tensor]) -> float:
    return math.sqrt(sum(float(x.detach().double().pow(2).sum().cpu()) for x in values))


def metric(model: nn.Module, params: Mapping[str, torch.Tensor], task: Any, loss_fn: nn.Module) -> tuple[float, float, float]:
    with torch.no_grad():
        support = functional_forward(model, params, task.support_x)
        query = functional_forward(model, params, task.query_x)
        sl, ql = float(loss_fn(support, task.support_y).cpu()), float(loss_fn(query, task.query_y).cpu())
    f1 = float("nan") if not torch.isfinite(query).all() else float(compute_metrics(query.cpu(), task.query_y.cpu(), num_classes=2).macro_f1)
    return sl, ql, f1


def existing_manifest_rows(outputs: Path, excluded_output: Path) -> tuple[set[int], List[dict]]:
    """Read only raw-row provenance, never metrics, from existing manifests."""
    rows: set[int] = set(); index = []
    excluded = excluded_output.resolve()
    for path in outputs.rglob("*.json"):
        if excluded in path.resolve().parents: continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "tasks" not in payload or "protocol" not in payload:
            continue
        raw = manifest_raw_row_ids(payload)
        if raw:
            rows.update(raw)
            index.append({"manifest": str(path.resolve()), "split": payload.get("protocol", {}).get("split"),
                          "raw_row_count": len(raw), "manifest_sha256": sha256_file(path)})
    return rows, index


def task_raw_rows(task: Any, dataset: Any) -> set[int]:
    ids = list(task.support_window_ids) + list(task.query_window_ids)
    return {int(value) for index in ids for value in np.asarray(dataset.row_ids[int(index)]).reshape(-1).tolist()}


def fresh_disjoint_manifest(out: Path, artifact_path: Path, artifact: Mapping[str, Any], cfg: Config,
                            seed: int, task_seed: int, n_tasks: int, shot: int, q_query: int,
                            forbidden: set[int]) -> tuple[dict, Any, dict]:
    bundle = build_pipeline(cfg, seed=seed); dataset = bundle.adapt_val_dataset
    sampler = bundle.make_adaptation_sampler(
        k_shot=shot, q_query=q_query, mode=str(cfg.data.get("task_mode", "binary")), n_way=int(artifact["extra"]["n_way"]),
        seed=task_seed, disallow_support_query_overlap=bool(cfg.data.get("disallow_support_query_overlap", True)),
        disallow_internal_overlap=bool(cfg.data.get("disallow_internal_overlap", True)), split="val")
    selected, attempted = [], 0
    while len(selected) < n_tasks and attempted < 50000:
        attempted += 1; task = sampler.sample_task()
        if not (task_raw_rows(task, dataset) & forbidden): selected.append(task)
    if len(selected) != n_tasks:
        raise RuntimeError(f"could only sample {len(selected)}/{n_tasks} raw-row-disjoint validation tasks after {attempted} attempts")
    path = out / "manifests" / f"validation_disjoint_seed_{seed}_taskseed_{task_seed}.json"
    digest = write_task_manifest(
        path, selected,
        protocol={"shot": shot, "q_query": q_query, "n_way": int(artifact["extra"]["n_way"]), "split": "val", "task_seed": task_seed,
                  "data_split_source": "adapt_val; rejection-sampled to avoid raw-row IDs present in all pre-existing manifests",
                  "sampler": "AdaptationTaskSampler sequential RNG stream with external raw-row exclusion"},
        base_checkpoint_path=str(artifact_path.resolve()), base_checkpoint_sha256=sha256_file(artifact_path),
        metadata={"dataset": str(cfg.data.name), "unknown_class": str(artifact["extra"]["unknown_class"]), "stage": "stage4_validation_only",
                  "strict_adapt_test": bool(cfg.data.get("strict_adapt_test", False)), "excluded_raw_row_count": len(forbidden)}, dataset=dataset)
    payload = read_task_manifest(path)
    overlap = manifest_raw_row_ids(payload) & forbidden
    if overlap: raise RuntimeError(f"new manifest raw-row overlap: {len(overlap)}")
    return payload, dataset, {"path": str(path.resolve()), "sha256": digest, "task_seed": task_seed,
                               "attempts": attempted, "new_raw_rows": len(manifest_raw_row_ids(payload)), "overlap_with_existing": 0}


def task_from_manifest(payload: Mapping[str, Any], dataset: Any) -> List[Any]:
    from src.evaluation.task_manifest import load_tasks_from_manifest
    return load_tasks_from_manifest(payload, dataset)


def run_metaopt(task: Any, model: nn.Module, init: Mapping[str, torch.Tensor], names: Sequence[str], meta: nn.Module,
                seed: int, task_index: int, schedule: str, alpha: float) -> List[dict]:
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    full = OrderedDict((n, v.detach().clone().to(task.support_x.device).requires_grad_(True)) for n,v in init.items())
    frozen = OrderedDict((n,v) for n,v in full.items() if n not in names); params = OrderedDict((n,full[n]) for n in names)
    state = meta.init_state(params); rows = []
    sl, ql, f1 = metric(model, {**frozen, **params}, task, loss_fn)
    rows.append({"seed":seed,"task_index":task_index,"method":"MetaOpt","schedule":schedule,"alpha":alpha,"step":0,"support_loss":sl,"query_loss":ql,"query_macro_f1":f1,"raw_update_norm":0.,"applied_update_norm":0.,"status":"ok"})
    for step in range(1, STEPS + 1):
        loss = loss_fn(functional_forward(model, {**frozen, **params}, task.support_x), task.support_y)
        try: grads = torch.autograd.grad(loss, list(params.values()), create_graph=False, retain_graph=False, allow_unused=False)
        except Exception as exc:
            rows.append({**rows[-1],"step":step,"support_loss":float(loss.detach().cpu()),"query_loss":float("nan"),"query_macro_f1":float("nan"),"raw_update_norm":float("nan"),"applied_update_norm":float("nan"),"status":f"gradient_error:{type(exc).__name__}"}); break
        gd = OrderedDict(zip(params.keys(),grads)); raw_updates,state = meta.step(gd,state)
        multiplier = alpha if schedule == "all_steps" or step <= EARLY_STEPS else 1.
        updates = OrderedDict((n, value * multiplier) for n,value in raw_updates.items())
        next_params = OrderedDict((n,params[n]+updates[n]) for n in params)
        sl,ql,f1 = metric(model,{**frozen,**next_params},task,loss_fn)
        status="ok" if all(np.isfinite(x) for x in (sl,ql,f1,l2(raw_updates.values()),l2(updates.values()))) else "nonfinite"
        rows.append({"seed":seed,"task_index":task_index,"method":"MetaOpt","schedule":schedule,"alpha":alpha,"step":step,"support_loss":sl,"query_loss":ql,"query_macro_f1":f1,"raw_update_norm":l2(raw_updates.values()),"applied_update_norm":l2(updates.values()),"status":status})
        params=next_params
        if status != "ok": break
        if step != STEPS:
            params=OrderedDict((n,v.detach().clone().requires_grad_(True)) for n,v in params.items()); state=meta.detach_state(state)
    return rows


def run_adam(task: Any, model: nn.Module, init: Mapping[str, torch.Tensor], names: Sequence[str], seed: int, task_index: int) -> List[dict]:
    loss_fn=nn.CrossEntropyLoss(reduction="mean"); full=OrderedDict((n,v.detach().clone().to(task.support_x.device).requires_grad_(True)) for n,v in init.items())
    frozen=OrderedDict((n,v) for n,v in full.items() if n not in names); params=OrderedDict((n,full[n]) for n in names); opt=HandcraftedOptimizer("adam",lr=.1); state=opt.init_state(params); rows=[]
    sl,ql,f1=metric(model,{**frozen,**params},task,loss_fn); rows.append({"seed":seed,"task_index":task_index,"method":"Adam","schedule":"baseline","alpha":.1,"step":0,"support_loss":sl,"query_loss":ql,"query_macro_f1":f1,"raw_update_norm":0.,"applied_update_norm":0.,"status":"ok"})
    for step in range(1,STEPS+1):
        loss=loss_fn(functional_forward(model,{**frozen,**params},task.support_x),task.support_y); grads=torch.autograd.grad(loss,list(params.values()),create_graph=False,retain_graph=False,allow_unused=False); gd=OrderedDict(zip(params.keys(),grads)); updates,state=opt.step(gd,state); params=OrderedDict((n,params[n]+updates[n]) for n in params); sl,ql,f1=metric(model,{**frozen,**params},task,loss_fn); status="ok" if all(np.isfinite(x) for x in (sl,ql,f1,l2(updates.values()))) else "nonfinite"; rows.append({"seed":seed,"task_index":task_index,"method":"Adam","schedule":"baseline","alpha":.1,"step":step,"support_loss":sl,"query_loss":ql,"query_macro_f1":f1,"raw_update_norm":l2(updates.values()),"applied_update_norm":l2(updates.values()),"status":status})
        if status != "ok": break
        if step != STEPS: params=OrderedDict((n,v.detach().clone().requires_grad_(True)) for n,v in params.items()); state=opt.detach_state(state)
    return rows


def outcomes(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    groups: Dict[tuple,list] = {}
    for r in rows: groups.setdefault((r["seed"],r["task_index"],r["method"],r["schedule"],str(r["alpha"])),[]).append(r)
    answer=[]
    for key, trace in groups.items():
        hit=[int(r["step"]) for r in trace if np.isfinite(float(r["query_macro_f1"])) and float(r["query_macro_f1"])>=TARGET]
        answer.append({"seed":key[0],"task_index":key[1],"method":key[2],"schedule":key[3],"alpha":key[4],"steps_to_f1_0_8_capped_20":hit[0] if hit else STEPS,"reached_f1_0_8":bool(hit),"final_status":trace[-1]["status"]})
    return answer


def aggregate(rows: Sequence[Mapping[str, Any]], outcomes_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    answer=[]; keys=sorted({(r["method"],r["schedule"],str(r["alpha"])) for r in rows})
    for method,schedule,alpha in keys:
        sub=[r for r in rows if (r["method"],r["schedule"],str(r["alpha"]))==(method,schedule,alpha) and r["status"]=="ok"]
        for step in CHECKPOINTS:
            values=[float(r["query_macro_f1"]) for r in sub if int(r["step"])==step]
            answer.append({"method":method,"schedule":schedule,"alpha":alpha,"metric":"query_macro_f1","step":step,"mean":float(np.mean(values)),"std":float(np.std(values)),"n":len(values)})
        reach=[float(r["steps_to_f1_0_8_capped_20"]) for r in outcomes_rows if (r["method"],r["schedule"],str(r["alpha"]))==(method,schedule,alpha)]
        answer.append({"method":method,"schedule":schedule,"alpha":alpha,"metric":"steps_to_f1_0_8_capped_20","step":"reach","mean":float(np.mean(reach)),"std":float(np.std(reach)),"n":len(reach)})
    return answer


def paired(rows: Sequence[Mapping[str, Any]], outcomes_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    try: from scipy.stats import wilcoxon
    except ImportError: wilcoxon=None
    def values(method,schedule,alpha,step):
        return {(int(r["seed"]),int(r["task_index"])):float(r["query_macro_f1"]) for r in rows if r["method"]==method and r["schedule"]==schedule and str(r["alpha"])==str(alpha) and int(r["step"])==step and r["status"]=="ok"}
    base=values("MetaOpt","early_only",1,0); answer=[]
    configs=sorted({(r["method"],r["schedule"],str(r["alpha"])) for r in rows if not (r["method"]=="MetaOpt" and r["schedule"]=="early_only" and str(r["alpha"])=="1")})
    for method,schedule,alpha in configs:
        for ref_name, ref in (("original_metaopt", values("MetaOpt","early_only",1,0)),("Adam_lr_0.1",values("Adam","baseline",.1,0))):
            for step in (1,2,5,10,20):
                current=values(method,schedule,alpha,step); reference=values("MetaOpt","early_only",1,step) if ref_name=="original_metaopt" else values("Adam","baseline",.1,step); keys=sorted(set(current)&set(reference)); diff=np.asarray([current[k]-reference[k] for k in keys]); p=1.0
                if wilcoxon is not None and len(diff) and not np.allclose(diff,0):
                    try:p=float(wilcoxon(diff).pvalue)
                    except ValueError:p=float("nan")
                answer.append({"config_method":method,"schedule":schedule,"alpha":alpha,"reference":ref_name,"metric":f"step{step}_macro_f1","n":len(diff),"mean_paired_difference":float(np.mean(diff)) if len(diff) else float("nan"),"wilcoxon_p":p})
    return answer


def plot(summary: Sequence[Mapping[str, Any]], out: Path) -> None:
    for schedule in ("early_only","all_steps"):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for alpha in ("1","2","5","10","15"):
            sub=sorted([r for r in summary if r["method"]=="MetaOpt" and r["schedule"]==schedule and r["alpha"]==alpha and r["metric"]=="query_macro_f1"],key=lambda r:int(r["step"]))
            ax.plot([r["step"] for r in sub],[r["mean"] for r in sub],label=f"α={alpha}")
        adam=sorted([r for r in summary if r["method"]=="Adam" and r["metric"]=="query_macro_f1"],key=lambda r:int(r["step"]))
        ax.plot([r["step"] for r in adam],[r["mean"] for r in adam],color="black",linestyle="--",label="Adam lr=.1")
        ax.set_xlabel("adaptation step");ax.set_ylabel("query macro-F1");ax.set_title(schedule.replace("_"," "));ax.grid(alpha=.3);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(out/f"metaopt_scale_{schedule}_f1.png",dpi=160);plt.close(fig)


def report(out: Path, audit: Mapping[str,Any], summary: Sequence[Mapping[str,Any],], paired_rows: Sequence[Mapping[str,Any]]) -> None:
    key={(r["method"],r["schedule"],str(r["alpha"]),str(r["step"])):r for r in summary}
    original=key[("MetaOpt","early_only","1","5")]["mean"]; candidates=[]
    for alpha in ("2","5","10","15"):
        early=key[("MetaOpt","early_only",alpha,"5")]["mean"]; final=key[("MetaOpt","early_only",alpha,"20")]["mean"]
        candidates.append((alpha,early,final))
    best=max(candidates,key=lambda x:x[1]); original20=key[("MetaOpt","early_only","1","20")]["mean"]
    accepted=best[1]>original and best[2]>=original20-.01
    decision="A: early-only scale evidence supports insufficient early update scale." if accepted else "B: scaling alone does not provide a stable early improvement without unacceptable late degradation; investigate direction quality, hidden-state timing, and training-task distribution before retraining."
    lines=["# Stage 4: MetaOpt early-scale hypothesis", "", "## Static training audit", "",
           "- OuterLoop computes query cross-entropy only after `InnerLoop.adapt` returns the final (step 20) fast weights; loss weights are step 1–19 = 0 and step 20 = 1.",
           f"- `tbptt_steps={audit['tbptt_steps']}` and `first_order={audit['first_order']}`; thus training did not use truncated BPTT.",
           f"- Update is a linear LSTM output × fixed `output_scale={audit['output_scale']}` × learnable `softplus(raw_lr)`; no tanh/sigmoid/output clamp is present. Input gradients use log/sign preprocessing with p={audit['preprocess_p']}.",
           "- Retained artifacts/logs do not contain per-inner-step training-task gradient/update distributions; that sub-question cannot be confirmed retrospectively without a new instrumented training run, which this stage does not perform.",
           "", "## Fresh validation isolation", "", f"- Raw-row provenance from {audit['existing_manifest_count']} pre-existing manifests was used only to exclude rows, not to read metrics. New manifests have zero raw-row overlap with that union.",
           "", "## Scale search", "", "| Schedule | α | Step 1 | Step 2 | Step 5 | Step 10 | Step 20 | Steps to 0.8 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for schedule in ("early_only","all_steps"):
        for alpha in ("1","2","5","10","15"):
            vals=[key[("MetaOpt",schedule,alpha,str(step))]["mean"] for step in (1,2,5,10,20)]; reach=key[("MetaOpt",schedule,alpha,"reach")]["mean"]
            lines.append(f"| {schedule} | {alpha} | " + " | ".join(f"{x:.4f}" for x in vals) + f" | {reach:.2f} |")
    lines += ["", "## Decision", "", decision]
    if accepted:
        lines += ["", f"Best early-only candidate: α={best[0]} (step5 {best[1]:.4f}, versus original {original:.4f}; step20 {best[2]:.4f}, original {original20:.4f}).",
                  "Minimal retraining proposal (not executed): retain head_only/strict split; add weighted query losses at steps 1–5; make output scale learnable or validation-selectable; choose on fresh validation manifests, then evaluate once on a new test manifest."]
    else:
        lines += ["", "No retraining is recommended from this scale-only diagnostic. Do not extend the attack × shot matrix."]
    (out/"STAGE4_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    args=parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True);(out/"manifests").mkdir();seeds=[int(x) for x in args.seeds.split(",") if x.strip()];alphas=[float(x) for x in args.alphas.split(",") if x.strip()]
    forbidden,existing=existing_manifest_rows(Path("outputs"),out);dump_json(out/"existing_manifest_provenance_index.json",existing)
    root=Path(args.artifact_root);prepared=[];manifest_index=[];audit={"existing_manifest_count":len(existing),"excluded_raw_row_count":len(forbidden),"tbptt_steps":None,"first_order":None,"output_scale":None,"preprocess_p":None,"training_loss":"final_step_query_loss_only"}
    for seed in seeds:
        path=root/f"seed_{seed}"/"horizon_20"/"meta_artifacts.pt";artifact=load_artifacts(str(path));cfg=Config(artifact["config"]);set_seed(seed,bool(cfg.experiment.get("deterministic",True)))
        manifest,dataset,info=fresh_disjoint_manifest(out,path,artifact,cfg,seed,seed+5002,args.tasks,args.shot,args.q_query,forbidden);manifest_index.append(info)
        init=OrderedDict((n,v.detach().clone()) for n,v in artifact["meta_init_state"].items());names=list(artifact["extra"]["adapt_names"])
        audit.update({"tbptt_steps":int(cfg.meta.get("tbptt_steps",0)),"first_order":bool(cfg.meta.get("first_order",False)),"output_scale":float(cfg.meta_optimizer.output_scale),"preprocess_p":float(cfg.meta_optimizer.preprocess_p)})
        prepared.append((seed,artifact,cfg,manifest,dataset,init,names))
    dump_json(out/"validation_manifest_index.json",manifest_index);dump_json(out/"static_training_audit.json",audit);dump_json(out/"run_protocol.json",{"created_at_utc":datetime.now(timezone.utc).isoformat(),"validation_only":True,"alphas":alphas,"early_steps":EARLY_STEPS,"steps":STEPS,"seeds":seeds,"tasks":args.tasks})
    device=resolve_device(str(prepared[0][2].device.get("prefer","auto")));rows=[]
    for seed,artifact,cfg,manifest,dataset,init,names in prepared:
        model=build_meta_model(cfg,artifact["extra"]["feature_dim"],artifact["extra"]["window_size"]).to(device);model.load_state_dict(artifact["meta_init_state"]);model.eval();tasks=[x.to(device) for x in task_from_manifest(manifest,dataset)]
        meta=build_meta_optimizer(cfg).to(device);meta.load_state_dict(artifact["meta_opt_state"]);meta.eval()
        for p in meta.parameters():p.requires_grad_(False)
        for i,task in enumerate(tasks):rows.extend(run_adam(task,model,init,names,seed,i))
        for schedule in ("early_only","all_steps"):
            for alpha in alphas:
                for i,task in enumerate(tasks):rows.extend(run_metaopt(task,model,init,names,meta,seed,i,schedule,alpha))
    dump_csv(out/"scale_search_raw.csv",rows);outcome_rows=outcomes(rows);dump_csv(out/"scale_search_task_outcomes.csv",outcome_rows);summary=aggregate(rows,outcome_rows);dump_csv(out/"scale_search_summary.csv",summary);paired_rows=paired(rows,outcome_rows);dump_csv(out/"scale_search_paired_differences.csv",paired_rows);plot(summary,out);report(out,audit,summary,paired_rows)
    print(json.dumps({"out":str(out.resolve()),"raw_rows":len(rows),"non_ok":sum(r["status"]!="ok" for r in rows)},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
