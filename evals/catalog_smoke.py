"""Exercise the real catalog/index/tools/Agent graph with a controlled local LLM.

This is an integration smoke check, not an evaluation of a remote model's prose.
No LLM API calls or catalog-site requests are made.
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Load .env before Hugging Face-dependent imports cache their model directory.
from src.config import config
from langchain_core.messages import AIMessage, ToolMessage
from evals.runner import RecordingProfileStore
from evals.retrieval_runner import preflight_retrieval
from src.agent.shopping_agent import ShoppingGuideAgent
from src.data.catalog import DEFAULT_DATASET_DIR, load_dataset
from src.retrieval.product_retriever import ProductRetriever


class CatalogSmokeLLM:
    def __init__(self, stage, tool_name, arguments, conclusion):
        self.stage, self.tool_name, self.arguments = stage, tool_name, arguments
        self.conclusion = conclusion
        self.called = False

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        if isinstance(messages, str):
            return AIMessage(content=self.stage)
        if not self.called:
            self.called = True
            return AIMessage(content="", tool_calls=[{
                "name": self.tool_name, "args": self.arguments,
                "id": "catalog-smoke-call", "type": "tool_call",
            }])
        output = next(message.content for message in reversed(messages) if isinstance(message, ToolMessage))
        return AIMessage(content=self.conclusion + "\n\n" + output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results/catalog-smoke"))
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--live", action="store_true", help="Use the configured real model through FastAPI in isolated conversations.")
    args = parser.parse_args()
    logging.getLogger("shopsift").setLevel(logging.WARNING)
    run_token = uuid.uuid4().hex[:12]
    manifest, tables, _ = load_dataset(args.dataset_dir, require_complete=True)
    preflight = preflight_retrieval(config.PRODUCT_DB_PATH, config.PRODUCT_CHROMA_DIR,
        expected_catalog_fingerprint=manifest["catalog_fingerprint"], embedding_model=config.EMBEDDING_MODEL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "mode": "controlled-llm-real-catalog",
              "catalog_version": manifest["dataset_version"], "catalog_fingerprint": manifest["catalog_fingerprint"],
              "embedding_model": config.EMBEDDING_MODEL, "index_version": preflight.index_version,
              "external_access": False, "checks": []}
    if not preflight.passed:
        report.update(status="blocked", issues=preflight.issues)
        (args.output_dir / "catalog-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2
    products = {p["product_id"]: p for p in tables["products"]}
    reviews = tables["product_reviews"]
    if args.live:
        return run_live_service(args, manifest, tables)
    retriever = ProductRetriever(chroma_dir=config.PRODUCT_CHROMA_DIR, catalog_db=config.PRODUCT_DB_PATH,
                                 embedding_model=config.EMBEDDING_MODEL)
    pairs = [(146206509,44884558), (553355699,221703640), (47445997,155437621), (422301229,338068870)]
    for category in ('显示器','键盘','鼠标'):
        ids = [p['product_id'] for p in products.values() if p['category']==category]
        if len(ids)>=2:
            first=next((pid for pid in ids if any(r['product_id']==pid for r in reviews)),ids[0])
            pairs.append((first,next(pid for pid in ids if pid!=first)))
    try:
        for first, second in pairs:
            p, q = products[first], products[second]
            aspect = next((r["aspect"] for r in reviews if r["product_id"] == first),'文字清晰度')
            has_experience = any(r['product_id']==first and r['aspect']==aspect for r in reviews)
            workflows = [
                ("recommendation", "recommendation", "get_product_detail", {"product_id": first},
                 f"{p['category']}：请给最后的购机结论，{p['name']} 是否适合，核实这个配置的公开价格与规格。", "建议查看下列配置，价格为采集时的公开快照。"),
                ("comparison", "comparison", "compare_products", {"product_ids": f"{first},{second}"},
                 f"{p['category']}：对比 {p['name']} 和 {q['name']}。", "以下是这两个配置的目录对比。"),
                ("experience", "objection_handling", "get_reviews", {"product_id": first, "aspect": aspect, "top_k": 5},
                 f"{p['category']}：{p['name']} 的{aspect}体验怎么样？请给出来源。", "以下摘要来自具体用户体验或原创评测，不能视为总体口碑统计。"),
            ]
            for workflow, stage, tool, arguments, question, conclusion in workflows:
                agent = ShoppingGuideAgent(CatalogSmokeLLM(stage, tool, arguments, conclusion), retriever,
                    RecordingProfileStore(), catalog_db=config.PRODUCT_DB_PATH,
                    checkpoint_db_path=str(args.output_dir / f"smoke-{first}-{workflow}.db"))
                try:
                    result = agent.run(question, conv_id=f"catalog-smoke-{run_token}-{first}-{workflow}")
                    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                    failures = []
                    if result["stop_reason"] != "completed": failures.append(result["stop_reason"])
                    if result["stage"] != stage: failures.append("wrong_stage")
                    if tool not in result["executed_tools"]: failures.append("tool_not_executed")
                    if result["tool_errors"]: failures.append("tool_error")
                    if not result["retrieval_triggered"]: failures.append("retrieval_missing")
                    if not tool_messages or (workflow!='experience' or has_experience) and "https://" not in str(tool_messages[-1].content): failures.append("source_missing")
                    if workflow=='experience' and not has_experience and '暂无' not in result['answer']: failures.append('experience_unknown_missing')
                    if workflow == "experience" and aspect not in result["answer"]: failures.append("aspect_missing")
                    if workflow != "experience" and "价格快照:" not in result["answer"]: failures.append("price_snapshot_missing")
                    if "未知" not in result["product_context"]: failures.append("unknown_metadata_missing")
                    report["checks"].append({"category": p["category"], "workflow": workflow, "product_ids": [first,second] if workflow=="comparison" else [first],
                        "passed": not failures, "failures": failures, "stage": result["stage"], "executed_tools": result["executed_tools"],
                        "failure_counts": result.get("failure_counts", {}),
                        "answer": result["answer"], "tool_output": str(tool_messages[-1].content) if tool_messages else ""})
                finally:
                    agent.graph.close()
        # The real Agent extracts Chinese exclusions; the new catalog uses
        # English brand names. Check this boundary with the actual retriever.
        for category, excluded in (("笔记本电脑","荣耀"),("手机","小米"),("平板电脑","苹果"),("无线耳机","小米")):
            from src.retrieval.models import canonical_brand
            found = retriever.search(f"推荐{category}", filters={"category":category,"excluded_brands":[excluded]})
            passed = bool(found.products) and all(canonical_brand(p.brand)!=canonical_brand(excluded) for p in found.products)
            report["checks"].append({"category":category,"workflow":"chinese_brand_exclusion","passed":passed,
                                     "excluded_brand":excluded,"returned_ids":[p.product_id for p in found.products]})
    finally:
        retriever.close()
    report.update(status="passed" if all(c["passed"] for c in report["checks"]) else "failed")
    (args.output_dir / "catalog-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"catalog smoke: {report['status']}; {sum(c['passed'] for c in report['checks'])}/{len(report['checks'])} checks")
    return 0 if report["status"] == "passed" else 1


def run_live_service(args, manifest, tables):
    """Exercise real HTTP handlers, memory, tools and verifier without mocks."""
    import re
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend import dependencies
    from src.data.catalog import catalog_fingerprint
    from src.agent.langgraph_engine import build_retrieval_constraints
    products={p['product_id']:p for p in tables['products']}
    definitions=[
        ('显示器',3000,('日常办公','编程和多设备办公','游戏和影音'),[90330009,271160000],'惠普',
         '必须可升降，并支持DP接口180Hz及以上。',41771888,'文字清晰度','必须支持USB-C供电至少65W。'),
        ('键盘',800,('日常办公','编程和多设备办公','游戏'),[795300500,231384076],'罗技',
         '必须同时支持蓝牙和2.4G。',127510714,'背光','必须支持轴体热插拔。'),
        ('鼠标',600,('日常办公','便携和多设备办公','游戏'),[242667166,552898930],'罗技',
         '必须支持蓝牙。',242667166,'便携','使用时重量必须不超过60g，含需要安装的电池。')]
    report={'mode':'live_service','status':'running','catalog_version':manifest['dataset_version'],
            'catalog_fingerprint':manifest['catalog_fingerprint'],'index_version':preflight_retrieval(config.PRODUCT_DB_PATH,config.PRODUCT_CHROMA_DIR).index_version,
            'model':config.LLM_MODEL,'provider':config.LLM_PROVIDER,'scenarios':[],
            'limitations':['试点数据不足以代表每个场景的完整市场覆盖','Docker未安装，未做Docker验收']}
    names=('PROFILE_DB_PATH','PROFILE_CHROMA_DIR','AGENT_CHECKPOINT_DB_PATH','CONV_DB_PATH')
    before={name:getattr(config,name) for name in names}
    for name,filename in zip(names,('profiles.db','profile_chroma','checkpoints.db','conversations.db')):
        setattr(config,name,str(args.output_dir/filename))
    path=args.output_dir/'live-service.json'
    def save(): path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    try:
        # A new process must not reuse another user's live agent singleton.
        if dependencies._agent is not None or dependencies._conv_store is not None:
            raise RuntimeError('live smoke requires a fresh process')
        unavailable=False
        with TestClient(app,raise_server_exceptions=False) as client:
            for category,budget,scenes,pair,brand,refinement,review_pid,aspect,empty in definitions:
                for scene in scenes:
                    entry={'category':category,'scene':scene,'status':'running','turns':[]}
                    report['scenarios'].append(entry)
                    if unavailable:
                        entry.update(status='blocked',reason='真实模型服务不可用，未运行；不使用受控模型替代')
                        save();continue
                    created=client.post('/api/conversations',json={'title':f'验收/{category}/{scene}'})
                    if created.status_code!=200:
                        entry.update(status='blocked',reason='conversation_api_unavailable');unavailable=True;save();continue
                    conv_id=created.json()['id'];history=[]
                    questions=[f'预算{budget}元以内，{scene}用，想买{category}，请先给出目录内不同型号的推荐候选。',
                               refinement,f'排除{brand}品牌，按刚才的条件重新筛选。',
                               f'取消品牌排除；对比{products[pair[0]]["name"]}与{products[pair[1]]["name"]}，调用对比工具核实价格和规格。',
                               f'{products[review_pid]["name"]}的{aspect}体验如何？调用体验查询工具说明来源；缺少资料时明确说未知。',empty]
                    for number,question in enumerate(questions,1):
                        response=client.post('/api/chat',json={'conv_id':conv_id,'question':question,'chat_history':history})
                        if response.status_code!=200:
                            entry.update(status='blocked',reason=f'chat_http_{response.status_code}')
                            entry['turns'].append({'turn':number,'question':question,'http_status':response.status_code})
                            unavailable=True;break
                        result=response.json();answer=result.get('answer','');failures=[]
                        if not answer.strip(): failures.append('empty_answer')
                        if result.get('stop_reason')!='completed': failures.append(result.get('stop_reason','unknown_stop'))
                        if result.get('tool_errors'): failures.append('tool_error')
                        if result.get('failure_counts'): failures += list(result['failure_counts'])
                        profile_constraints=build_retrieval_constraints(dependencies._agent.profile_store.get_structured(conv_id))
                        # These expected fields are independent of the parser:
                        # a later USB-C condition must preserve the DP scope.
                        if number >= 2:
                            expected = {'显示器':[('dp_refresh_hz','gte',180),('stand_height_adjustable','eq',True)],
                                        '键盘':[('connections','contains_all',['bluetooth','2.4ghz'])],
                                        '鼠标':[('connections','contains_all',['bluetooth'])]}[category]
                            if number == 6:
                                expected += {'显示器':[('usb_c_pd_w','gte',65)], '键盘':[('hot_swap','eq',True)],
                                             '鼠标':[('weight_g','lte',60.0)]}[category]
                            actual=profile_constraints.get('spec_filters',[])
                            for field,op,value in expected:
                                if not any(p['field']==field and p['op']==op and p['value']==value for p in actual):
                                    failures.append('profile_condition_missing:'+field)
                            if category=='显示器' and any(p['field']=='usb_c_refresh_hz' for p in actual):
                                failures.append('refresh_port_scope_changed')
                        if number==4 and 'compare_products' not in result.get('executed_tools',[]): failures.append('comparison_tool_missing')
                        if number==5 and 'get_reviews' not in result.get('executed_tools',[]): failures.append('review_tool_missing')
                        if number==6 and not re.search(r'暂无|没有|无符合|无匹配|未找到|不足|无法|未知|不能|不满足|不符合|未能',answer): failures.append('unknown_or_empty_not_explained')
                        entry['turns'].append({'turn':number,'question':question,'answer':answer,'stage':result.get('stage'),
                            'executed_tools':result.get('executed_tools',[]),'stop_reason':result.get('stop_reason'),
                            'failure_counts':result.get('failure_counts',{}),'passed':not failures,'failures':failures,
                            'profile_constraints':profile_constraints,
                            'latency_ms':result.get('latency_ms'),'llm_calls':result.get('llm_calls')})
                        history += [{'role':'user','content':question},{'role':'assistant','content':answer}]
                        save()
                        print(f'live {category}/{scene} turn {number}: {"passed" if not failures else "failed"}',flush=True)
                    if entry['status']=='running': entry['status']='passed' if len(entry['turns'])==6 and all(t['passed'] for t in entry['turns']) else 'failed'
                    save()
        report['status']='blocked' if unavailable else 'passed' if all(s['status']=='passed' for s in report['scenarios']) else 'failed'
        if catalog_fingerprint(config.PRODUCT_DB_PATH,Path(config.PRODUCT_DB_PATH).parent/'product_reviews.db') != report['catalog_fingerprint']:
            report['status']='blocked';report['limitations'].append('测试期间目录发生变化')
    except Exception as exc:
        report.update(status='blocked',error_type=type(exc).__name__)
    finally:
        for name,value in before.items(): setattr(config,name,value)
        if dependencies._agent is not None:
            dependencies._agent.graph.close()
        save()
    print(f'live service: {report["status"]}; report: {path}')
    return 0 if report['status']=='passed' else 2 if report['status']=='blocked' else 1


if __name__ == "__main__":
    raise SystemExit(main())
