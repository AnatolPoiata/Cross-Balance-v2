# References

Source material for the Phase C research article, beyond the ~239-paper
corpus already collected in `references/` (copied from the v1 project,
gitignored here due to size — see the original `Cross-Balance/references/`
folder on disk).

## Background reading (from the original project brief)

- https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets
- https://medium.com/data-science/class-imbalance-strategies-a-visual-guide-with-code-8bc8fae71e1a
- https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf
- https://www.knime.com/blog/4-techniques-handle-imbalanced-datasets
- Jason Brownlee, *Imbalanced Classification with Python* — already present as a
  PDF in `references/` (also linked via
  https://github.com/khurrameycon/Data-Science-Books)
- https://numerous.ai/blog/imbalanced-data-classification
- https://machinelearningmastery.com/navigating-imbalanced-datasets-with-pandas-and-scikit-learn/

## Additional references (added after Phase A)

- https://data.mendeley.com/datasets/pb3jd7vz9z/1 — not yet used; reference
  material / possible additional dataset, not pulled into `data/raw/` yet.
- https://github.com/miriamspsantos/open-source-imbalance-overlap — open-source
  toolkit for class-overlap analysis in imbalanced learning. Not yet used;
  relevant to Phase C (understanding *why* Cross-Balance wins/loses on
  particular datasets — overlap-based diagnostics could explain the
  segment0/shuttle_2_vs_5 results where ensemble methods hit ceiling
  performance).
- https://github.com/antoniopedropi/Datasets-ImbalancedRegression — used as
  the source for the `ailerons` dataset (`benchmarks/real_datasets.py::load_ailerons`).
- https://sci2s.ugr.es/keel/dataset.php?cod=1342 and
  https://sci2s.ugr.es/keel/imbalanced.php#subA — KEEL imbalanced-datasets
  repository; used as the source for `segment0` and `shuttle_2_vs_5`
  (`benchmarks/real_datasets.py::load_segment0`, `load_shuttle_2_vs_5`).
