"""Configuration proliferation cannot satisfy the publication model gate."""
from src.data.release import evaluate_release
from src.data.quality import SOURCE_PREFERRED_POLICY, product_issues
from src.data.categories import matches_spec_filters
from src.retrieval.models import SpecFilter


def test_500_configuration_variants_do_not_count_as_500_models():
    rows = [{'product_id':i,'model_key':'same-phone','brand':'Apple','category':'手机',
             'specs':{'处理器':'A18','存储':str(i)+'GB','屏幕尺寸':'6.1英寸','重量':'170g'},
             'price_source_id':'price'} for i in range(1,501)]
    result = evaluate_release(rows, [], [{'source_id':'price','seller_id':'seller','platform':'shop'}])
    assert result['qualified_models'] == 1
    assert result['remaining_models'] == 499
    assert result['status'] == 'blocked'


def test_models_with_missing_core_fields_do_not_contribute_to_scale():
    row = {'product_id':1,'model_key':'incomplete-phone','brand':'Apple','category':'手机',
           'specs':{'处理器':'A18','存储':'256GB','屏幕尺寸':'6.1英寸'},'price_source_id':'price'}
    result = evaluate_release([row],[],[])
    assert result['qualified_models'] == 0
    assert 'unqualified_core_fields' in result['issues']


def test_selected_source_policy_preserves_unknowns_without_accepting_bad_prices():
    row = {'name':'真实键盘', 'brand':'Logitech', 'category':'键盘', 'sku_key':'test-sku',
           'model_key':'test-model', 'configuration':{'型号':'测试型号'},
           'specs':{'商品型号':'测试型号'}, 'price':199.0,
           'price_source_id':'source', 'spec_source_ids':['source']}
    assert product_issues(row, policy=SOURCE_PREFERRED_POLICY) == []
    assert product_issues(row)  # Historical policy still requires decision fields.
    assert product_issues({**row, 'price':0}, policy=SOURCE_PREFERRED_POLICY)
    assert product_issues({**row, 'specs':{}}, policy=SOURCE_PREFERRED_POLICY)
    assert product_issues({**row, 'category':'虚构品类'}, policy=SOURCE_PREFERRED_POLICY)
    assert not matches_spec_filters({}, [SpecFilter(field='key_count', op='gte', value=87)])


def test_selected_source_policy_reports_channel_gaps_as_diagnostics():
    row = {'name':'真实键盘','brand':'Logitech','category':'键盘','sku_key':'one',
           'model_key':'one','configuration':{'型号':'one'},'specs':{'型号':'one'},
           'price':199.,'price_source_id':'source','spec_source_ids':['source']}
    result = evaluate_release([row], [], [], quality_policy=SOURCE_PREFERRED_POLICY)
    category = result['categories']['键盘']
    assert result['qualified_models'] == 1
    assert 'price_channel_identity_unknown' in category['warnings']
    assert 'price_channel_identity_unknown' not in category['failures']
    assert 'minimum_models_not_met' in result['issues']
