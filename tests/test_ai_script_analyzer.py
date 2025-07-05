import textwrap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_graphs.ai_script_analyzer import AIScriptAnalyzer

def test_datetime_instantiation(tmp_path):
    script = textwrap.dedent(
        '''
        import datetime
        dt = datetime.datetime(2024, 1, 1)
        '''
    )
    path = tmp_path / 'script.py'
    path.write_text(script)
    analyzer = AIScriptAnalyzer()
    result = analyzer.analyze_script(str(path))
    assert any(ci.class_name == 'datetime.datetime' for ci in result.class_instantiations)
    assert all(fc.function_name != 'datetime.datetime' for fc in result.function_calls)

def test_bare_datetime_instantiation(tmp_path):
    script = textwrap.dedent(
        '''
        from datetime import datetime
        dt = datetime(2024, 1, 1)
        '''
    )
    path = tmp_path / 'script.py'
    path.write_text(script)
    analyzer = AIScriptAnalyzer()
    result = analyzer.analyze_script(str(path))
    assert any(ci.class_name == 'datetime' for ci in result.class_instantiations)
    assert not result.function_calls


def test_star_import_resolution(tmp_path):
    script = textwrap.dedent(
        '''
        from skidl import *
        n = Net('test')
        p = Part('Device', 'R')
        '''
    )
    path = tmp_path / 'script.py'
    path.write_text(script)
    analyzer = AIScriptAnalyzer()
    result = analyzer.analyze_script(str(path))
    full_names = {ci.full_class_name for ci in result.class_instantiations}
    assert 'skidl.Net' in full_names
    assert 'skidl.Part' in full_names
