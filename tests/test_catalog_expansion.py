"""New category constraints must remain factual, compatible and auditable."""
import pytest
from pydantic import ValidationError

from src.data.categories import category_mentions, explicit_spec_filters, matches_spec_filters
from src.data.categories import port_refresh_at_resolution
from src.retrieval.models import SpecFilter, RetrievalConstraints


@pytest.mark.parametrize('clause,resolution,expected', [
    ('HDMI支持最高2560 x 1440，240 Hz', '3440x1440', None),
    ('HDMI支持最高3840 x 2160，120 Hz', '3840x2160', 120),
    ('DisplayPort支持1920×1080 100Hz', '1920x1080', 100),
    ('DisplayPort支持2 x QHD 120Hz 8位', '2560x1440', None),
    ('HDMI支持3840x2160 60Hz及1920x1080 120Hz', '3840x2160', None),
    ('HDMI支持3840x2160，线缆规格另行确认', '3840x2160', None),
])
def test_port_refresh_cannot_borrow_a_different_resolution(clause, resolution, expected):
    assert port_refresh_at_resolution(clause, resolution) == expected


def test_lower_resolution_hdmi_mode_does_not_satisfy_native_hard_requirement():
    normalized = {'resolution': '3440x1440'}
    verified = port_refresh_at_resolution('2个HDMI支持最高2560 x 1440，240 Hz', normalized['resolution'])
    if verified is not None:
        normalized['hdmi_refresh_hz'] = verified
    constraints = RetrievalConstraints(spec_filters=[
        {'field': 'resolution', 'op': 'eq', 'value': '3440x1440'},
        {'field': 'hdmi_refresh_hz', 'op': 'gte', 'value': 240},
    ])
    assert not matches_spec_filters(normalized, constraints.spec_filters)


@pytest.mark.parametrize("question,category", [
    ("给MacBook选键盘", "键盘"), ("为笔记本挑显示器", "显示器"),
    ("推荐平板电脑", "平板电脑"), ("手机配一个鼠标", "鼠标"),
])
def test_purchase_peripheral_is_distinguished_from_host(question, category):
    assert next(category_mentions(question))[1] == category


@pytest.mark.parametrize("condition", [
    {"field": "imagined", "value": True},
    {"field": "weight_g", "op": "gte", "value": True},
    {"field": "weight_g", "value": float("nan")},
    {"field": "ports", "op": "contains_all", "value": "HDMI"},
])
def test_invalid_or_unknown_specification_predicate_is_rejected(condition):
    with pytest.raises(ValidationError):
        SpecFilter.model_validate(condition)


def test_unknown_or_conditional_capability_does_not_satisfy_hard_filter():
    filters = [SpecFilter.model_validate(p) for p in explicit_spec_filters("HDMI 接口144Hz及以上显示器", "显示器")]
    assert not matches_spec_filters({"refresh_hz": 180, "dp_refresh_hz": 180}, filters)
    assert not matches_spec_filters({"hdmi_refresh_hz": 120}, filters)
    assert matches_spec_filters({"hdmi_refresh_hz": 144, "ports": ["HDMI"]}, filters)
    weight = RetrievalConstraints(spec_filters=[{"field": "weight_g", "op": "lte", "value": 60}])
    assert not matches_spec_filters({}, weight.spec_filters)


def test_absent_optional_metadata_preserves_core_serialization():
    import json
    from src.data.catalog import DEFAULT_DATASET_DIR, ProductRecord
    row = json.loads((DEFAULT_DATASET_DIR / 'products.jsonl').read_text(encoding='utf8').splitlines()[0])
    row.pop('normalized_specs', None)
    result = ProductRecord.model_validate(row).historical_dump()
    assert 'normalized_specs' not in result
    assert result == row


def test_purchase_requirements_are_not_inferred_from_comparison_titles():
    assert explicit_spec_filters('对比108键与79键键盘的规格', '键盘') == []
    assert explicit_spec_filters('这款鼠标是否支持蓝牙？', '鼠标') == []
    assert explicit_spec_filters('蓝牙还是2.4G鼠标更适合我？', '鼠标') == []


@pytest.mark.parametrize('category,text', [
    ('显示器', '推荐显示器，不需要4K'),
    ('显示器', '推荐显示器，4K不需要'),
    ('显示器', '不要求HDMI接口144Hz以上'),
    ('键盘', '推荐键盘，不要磁轴'),
    ('键盘', '推荐键盘，不想要磁轴'),
    ('键盘', '推荐非磁轴键盘'),
    ('键盘', '不需要ANSI布局，也不需要热插拔'),
    ('鼠标', '不需要蓝牙，也不用有线连接'),
    ('鼠标', '这款鼠标是否不需要蓝牙？'),
    ('显示器', '4K或2K显示器哪个好？'),
])
def test_negative_or_ambiguous_mentions_never_become_positive_hard_filters(category, text):
    assert explicit_spec_filters(text, category) == []


def test_negation_scope_preserves_a_separate_affirmative_requirement():
    filters = explicit_spec_filters('不需要4K，但需要HDMI接口144Hz以上', '显示器')
    assert {'field': 'hdmi_refresh_hz', 'op': 'gte', 'value': 144} in filters
    assert not any(p['field'] == 'resolution' for p in filters)


@pytest.mark.parametrize('category,previous,current,expected', [
    ('鼠标', '蓝牙；重量不超过100g', '现在不需要蓝牙，只需要2.4G',
     {'connections': ['2.4ghz'], 'weight_g': 100.0}),
    ('鼠标', '蓝牙；重量不超过100g', '取消重量限制', {'connections': ['bluetooth']}),
    ('鼠标', '蓝牙；2.4G', '改为有线连接', {'connections': ['wired']}),
    ('键盘', '至少84键；蓝牙', '改成108键', {'key_count': 108, 'connections': ['bluetooth']}),
    ('键盘', '磁轴；热插拔', '不要磁轴，机械轴就行', {'switch_technology': 'mechanical', 'hot_swap': True}),
    ('显示器', '4K；HDMI 144Hz及以上', '不需要4K', {'ports': ['HDMI'], 'hdmi_refresh_hz': 144}),
])
def test_requirement_cancellation_and_replacement_survive_profile_serialization(category, previous, current, expected):
    from src.data.categories import merge_requirement_text
    merged = merge_requirement_text(previous, current, category)
    assert {p['field']: p['value'] for p in explicit_spec_filters(merged, category)} == expected


def test_profile_cancellation_without_a_new_positive_filter_reaches_retrieval_constraints():
    from evals.runner import RecordingProfileStore
    from src.agent.langgraph_engine import extract_profile_signals, build_retrieval_constraints
    store = RecordingProfileStore()
    extract_profile_signals('conversation', '推荐蓝牙鼠标，重量不超过100g', store)
    extract_profile_signals('conversation', '不需要蓝牙', store)
    constraints = build_retrieval_constraints(store.get_structured('conversation'))
    assert constraints['spec_filters'] == [{'field': 'weight_g', 'op': 'lte', 'value': 100.0}]
    extract_profile_signals('conversation', '取消重量限制', store)
    assert not build_retrieval_constraints(store.get_structured('conversation')).get('spec_filters')


def test_refinements_preserve_connections_and_replace_a_weight_limit():
    from src.data.categories import merge_requirement_text
    old='蓝牙；重量不超过100g'
    combined=merge_requirement_text(old,'2.4G，重量不超过60g','鼠标')
    predicates=[SpecFilter.model_validate(p) for p in explicit_spec_filters(combined,'鼠标')]
    assert matches_spec_filters({'connections':['bluetooth','2.4ghz'],'weight_g':60},predicates)
    assert not matches_spec_filters({'connections':['2.4ghz'],'weight_g':50},predicates)
    assert not matches_spec_filters({'connections':['bluetooth','2.4ghz'],'weight_g':61},predicates)
    assert not matches_spec_filters({'connections':['bluetooth','2.4ghz'],'body_weight_g':50},predicates)


def test_power_and_hot_swap_unknown_values_fail_closed():
    for text,category,known in [('USB-C供电至少65W','显示器',{'ports':['USB-C'],'usb_c_pd_w':15}),
                                ('必须支持轴体热插拔','键盘',{})]:
        filters=[SpecFilter.model_validate(p) for p in explicit_spec_filters(text,category)]
        assert filters
        assert not matches_spec_filters(known,filters)


def test_refresh_rate_scope_survives_a_power_requirement_in_a_later_turn():
    from src.data.categories import merge_requirement_text
    assert {'field':'dp_refresh_hz','op':'gte','value':180} in explicit_spec_filters('必须可升降，并支持DP接口180Hz及以上。','显示器')
    for old,expected in [('180Hz及以上；可升降','refresh_hz'),('DP 180Hz及以上；可升降','dp_refresh_hz')]:
        merged=merge_requirement_text(old,'必须支持USB-C供电至少65W','显示器')
        filters=explicit_spec_filters(merged,'显示器')
        assert {'field':expected,'op':'gte','value':180} in filters
        assert not any(p['field']=='usb_c_refresh_hz' for p in filters)
    filters=explicit_spec_filters('HDMI 144Hz及以上；DP 180Hz及以上；USB-C供电至少65W','显示器')
    assert {p['field'] for p in filters} >= {'hdmi_refresh_hz','dp_refresh_hz','usb_c_pd_w'}


def test_minimum_key_count_remains_a_minimum_after_refinement():
    from src.data.categories import merge_requirement_text
    merged=merge_requirement_text('至少84键','必须支持蓝牙','键盘')
    filters=[SpecFilter.model_validate(p) for p in explicit_spec_filters(merged,'键盘')]
    assert matches_spec_filters({'key_count':108,'connections':['bluetooth']},filters)
    assert not matches_spec_filters({'key_count':79,'connections':['bluetooth']},filters)


def test_completed_tool_payloads_are_compacted_without_changing_checkpoint_messages():
    from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
    from src.agent.langgraph_engine import bounded_tool_history
    messages=[]
    for i in range(6):
        messages += [HumanMessage(content=f'question {i}'),AIMessage(content='',tool_calls=[{'name':'get_product_detail','args':{'product_id':1},'id':str(i),'type':'tool_call'}]),
                     ToolMessage(content='payload '+str(i)+'x'*5000,tool_call_id=str(i)),AIMessage(content=f'answer {i}')]
    compact=bounded_tool_history(messages)
    assert len([m for m in compact if isinstance(m,HumanMessage)]) == 3
    assert sum(len(m.content) for m in compact)<7000
    assert next(m for m in compact if isinstance(m,ToolMessage) and m.tool_call_id=='5').content == messages[-2].content
    assert all(len(m.content)>5000 for m in messages if isinstance(m,ToolMessage))


def _retriever_fixture(tmp_path, monkeypatch, *, variants=False):
    import json,sqlite3
    import numpy as np
    from src.retrieval.product_retriever import ProductRetriever
    path=tmp_path/'products.db'
    ids=list(range(1,22))
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE products(product_id INTEGER PRIMARY KEY,name TEXT,brand TEXT,category TEXT,subcategory TEXT,price REAL,rating REAL,sales_count INTEGER,release_date TEXT,description TEXT,specs TEXT)')
        conn.execute('CREATE TABLE product_provenance(product_id INTEGER PRIMARY KEY,metadata TEXT)')
        for pid in ids:
            price=50 if pid==21 or variants else 1000
            conn.execute('INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?)',(pid,f'fixture {pid}','test','显示器','显示器',price,None,None,'','fixture','{}'))
            model='a' if pid<20 else 'b' if pid==20 else 'c'
            conn.execute('INSERT INTO product_provenance VALUES(?,?)',(pid,json.dumps({'model_key':model})))
    class Collection:
        def query(self,query_embeddings,n_results,include,where=None):
            chosen=ids if where is None else [p for p in ids if p in where['product_id']['$in']]
            chosen=chosen[:n_results]
            return {'metadatas':[[{'product_id':p} for p in chosen]]}
        def get(self,**kwargs): return {'ids':[],'documents':[],'metadatas':[]}
    class Client:
        def get_collection(self,name): return Collection()
    class Model:
        def encode(self,query):return np.zeros(4)
    monkeypatch.setattr(ProductRetriever,'_initialize_sparse_index',lambda self:None)
    return ProductRetriever(str(tmp_path/'chroma'),catalog_db=str(path),model=Model(),client=Client())


def test_qualification_precedes_vector_limit_even_when_fts_is_unavailable(tmp_path,monkeypatch):
    retriever=_retriever_fixture(tmp_path,monkeypatch)
    try:
        result=retriever.search('显示器',top_k=1,filters={'category':'显示器','max_price':100})
        assert [p.product_id for p in result.products]==[21]
    finally:retriever.close()


def test_generic_recommendations_show_models_before_more_variants(tmp_path,monkeypatch):
    retriever=_retriever_fixture(tmp_path,monkeypatch,variants=True)
    try:
        result=retriever.search('推荐显示器',top_k=5,filters={'category':'显示器','max_price':100})
        assert len({p.provenance['model_key'] for p in result.products})>=3
    finally:retriever.close()


def test_query_text_cannot_override_explicit_structured_hard_conditions(tmp_path, monkeypatch):
    retriever = _retriever_fixture(tmp_path, monkeypatch)
    try:
        result = retriever.search('推荐4K显示器', filters={'category': '显示器', 'spec_filters': [
            {'field': 'resolution', 'value': '1920x1080'},
        ]})
        assert result.constraints.spec_filters == [SpecFilter(field='resolution', value='1920x1080')]
        assert not result.products  # Unknown resolutions cannot satisfy the explicit requirement.
    finally:
        retriever.close()


def test_legacy_filters_use_unit_qualified_facts_and_keep_unknown_gpu_types_unknown():
    from src.data.categories import catalog_filter_specs
    facts = catalog_filter_specs('笔记本电脑', {'内存': '32 GB LPDDR5', '存储': '1TB',
                               '屏幕尺寸': '14英寸', '显卡类型': '集成显卡'})
    filters = [SpecFilter.model_validate(p) for p in explicit_spec_filters(
        '32GB内存、1TB存储，不要独立显卡的笔记本', '笔记本电脑')]
    assert len(filters) == 3 and matches_spec_filters(facts, filters)
    assert not matches_spec_filters(catalog_filter_specs('笔记本电脑', {'内存': '32GB',
        '存储': '1TB', '显卡': 'Intel Arc'}), filters)
    assert catalog_filter_specs('手机', {'屏幕尺寸': '6.1英寸/6.7英寸'}) == {}
    assert catalog_filter_specs('笔记本电脑', {'内存': '16GB或32GB'}) == {}
    assert catalog_filter_specs('笔记本电脑', {'内存': '最大支持32GB', '存储': '可选1TB'}) == {}
    from src.data.categories import merge_requirement_text
    minimum = merge_requirement_text('至少32GB内存', '1TB存储', '笔记本电脑')
    assert {'field': 'memory_gb', 'op': 'gte', 'value': 32.0} in explicit_spec_filters(minimum, '笔记本电脑')


def test_small_screen_constraint_matches_tool_evidence_without_changing_catalog_storage():
    from src.agent.shopping_tools import _catalog_artifact
    filters = [SpecFilter.model_validate(p) for p in explicit_spec_filters(
        '8000元内，屏幕不超过6.3英寸的手机', '手机')]
    artifact = _catalog_artifact([{'product_id': 1, 'name': 'Source-backed phone', 'brand': 'Test',
        'category': '手机', 'subcategory': '手机', 'price': 2000, 'rating': None,
        'sales_count': None, 'description': 'Verified facts', 'specs': '{"屏幕尺寸":"6.1英寸"}'}])
    assert len(filters) == 1
    assert matches_spec_filters(artifact['products'][0]['normalized_specs'], filters)
    assert not matches_spec_filters({'screen_inches': 6.7}, filters)
    assert explicit_spec_filters('推荐蓝牙耳机', '无线耳机') == []
