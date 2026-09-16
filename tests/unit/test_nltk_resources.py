"""Exercise real IAML imports and text processing without downloaded NLTK data."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class TestNLTKResources(unittest.TestCase):
    def run_offline(self, body):
        project_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(project_root / "src"), environment.get("PYTHONPATH")])
        )
        environment["MPLBACKEND"] = "Agg"
        environment["OMP_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        bootstrap = textwrap.dedent("""\
            import sys
            from pathlib import Path
            from unittest import mock
            import nltk

            nltk.data.path = [sys.argv[1]]
            with mock.patch('nltk.download', side_effect=AssertionError(
                'IAML must not download NLTK resources automatically'
            )) as download, mock.patch('nltk.downloader.urlopen', side_effect=AssertionError(
                'NLTK network access is forbidden'
            )):
            """)
        script = bootstrap + textwrap.indent(textwrap.dedent(body), "    ")
        script += "\n    download.assert_not_called()\n"
        with tempfile.TemporaryDirectory() as data_dir:
            result = subprocess.run(
                [sys.executable, "-c", script, data_dir],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_construction_and_numeric_pipeline_need_no_corpus(self):
        self.run_offline("""\
            import numpy as np
            import pandas as pd
            from iaml import IAML, DataType, Dataset
            from iaml.iaml_pipeline import IAMLPipeline
            from iaml.actionables.cleaning.act_word2vec import ActWord2Vec
            from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
                ActDecisionTreeClassifier,
            )

            automl = IAML(max_workers=1)
            assert any(isinstance(step, ActWord2Vec)
                       for step in automl.first_step.all_steps())
            X = pd.DataFrame({'value': np.arange(12, dtype=float) + 0.5})
            dataset = Dataset(X, [0] * 6 + [1] * 6,
                              columns_types={'value': DataType.NUMERIC})
            text_step = ActWord2Vec()
            assert not text_step.suitable(dataset)
            assert text_step.fit(dataset) is text_step
            pd.testing.assert_frame_equal(text_step.transform(X.copy()), X)

            pipeline = IAMLPipeline(estimator_type='classifier')
            pipeline.set_model(ActDecisionTreeClassifier())
            pipeline.fit(dataset.X, dataset.y)
            np.testing.assert_array_equal(pipeline.predict(dataset.X), dataset.y)
            """)

    def test_text_fit_reports_how_to_install_missing_stopwords(self):
        self.run_offline("""\
            import pandas as pd
            from iaml import DataType, Dataset
            from iaml.actionables.cleaning.act_word2vec import ActWord2Vec

            dataset = Dataset(
                pd.DataFrame({'review': ['The cats and dogs.'] * 12}),
                [0, 1] * 6,
                columns_types={'review': DataType.TEXT},
            )
            step = ActWord2Vec()
            assert step.suitable(dataset)
            try:
                step.fit(dataset)
            except LookupError as error:
                assert 'python -m nltk.downloader stopwords' in str(error), str(error)
            else:
                raise AssertionError('Text processing must report missing stopwords')
            """)

    def test_real_text_processing_needs_only_local_stopwords_during_fit(self):
        self.run_offline("""\
            corpus = Path(sys.argv[1]) / 'corpora' / 'stopwords' / 'english'
            corpus.parent.mkdir(parents=True)
            corpus.write_text('the\\nand\\nis\\n', encoding='utf-8')

            import numpy as np
            import pandas as pd
            from iaml import Candidate, DataType, Dataset
            from iaml.actionables.cleaning.act_word2vec import ActWord2Vec

            X = pd.DataFrame({
                'review': ['The cats and dogs.', 'the and is', '', 'Birds are flying!'] * 3,
                'count': list(range(12)),
            })
            dataset = Dataset(
                X.copy(), [0, 1] * 6,
                columns_types={'review': DataType.TEXT, 'count': DataType.NUMERIC},
            )
            step = ActWord2Vec()
            output = step.run(Candidate(dataset))[0].dataset.X
            assert output.shape == (12, 101), output.shape
            assert 'review' not in output.columns
            vectors = output.filter(like='review_vec_').to_numpy()
            assert np.isfinite(vectors).all()
            assert not np.allclose(vectors[0], 0)
            np.testing.assert_allclose(vectors[[1, 2]], 0)
            vocabulary = step.columns[0][1].wv.key_to_index
            assert {'cat', 'dog'}.issubset(vocabulary)
            assert not {'the', 'and', 'is'}.intersection(vocabulary)

            corpus.unlink()
            nltk.data.path = []
            pd.testing.assert_frame_equal(step.transform(X.copy()), output)
            """)
