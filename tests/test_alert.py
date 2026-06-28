import pytest, json, tempfile, os
from pipeline.alert import AlertGenerator, AlertReport, HarmfulContentDetail, CATEGORY_DESCRIPTIONS, ACTION_TEMPLATES

def test_alert_levels():
    gen = AlertGenerator()
    assert gen._determine_level(0.1) == "SAFE"
    assert gen._determine_level(0.35) == "LOW"
    assert gen._determine_level(0.6) == "MEDIUM"
    assert gen._determine_level(0.9) == "HIGH"

def test_category_descriptions():
    assert len(CATEGORY_DESCRIPTIONS) == 7
    assert "Smoke" in CATEGORY_DESCRIPTIONS

def test_action_templates():
    assert "HIGH" in ACTION_TEMPLATES
    assert "MEDIUM" in ACTION_TEMPLATES
    assert "LOW" in ACTION_TEMPLATES
    assert "SAFE" in ACTION_TEMPLATES

def test_report_generation(sample_detection_result):
    gen = AlertGenerator()
    report = gen.generate(sample_detection_result)
    assert isinstance(report, AlertReport)
    assert report.alert_level == "HIGH"
    assert report.anomaly_score == 0.85

def test_json_export(sample_detection_result):
    gen = AlertGenerator()
    report = gen.generate(sample_detection_result)
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        gen.export_json(report, f.name)
        path = f.name
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        assert "alert_level" in data
        assert "anomaly_score" in data
    finally:
        os.unlink(path)

def test_batch_report(sample_detection_result):
    gen = AlertGenerator()
    reports = gen.generate_batch([sample_detection_result, sample_detection_result])
    assert len(reports) == 2
