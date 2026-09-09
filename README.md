# Telecom-first anomaly detection and localisation

This repository is a notebook-first analytical pipeline for time-series
anomaly detection, incident consolidation and topology-aware localisation.
Telecom is the primary product sector. Petrobras 3W is retained as an
independent real-data qualification test for the topology-free path.

```text
Telecom source -> 00 feasibility audit -> 01A Telecom Pack --┐
Petrobras source ---------------------> 01A 3W Pack ---------┤
                                                             v
                                         01B canonical adapter
                                                             |
                     02 EDA -> 03 evaluation policy -> 04 modelling
                                                             |
                                      05 ranked incidents -> 06 demo
```

The model-visible contract contains telemetry plus optional topology.
Evaluation truth is physically separate and is loaded only after scores are
produced. Experimental partition labels remain in `SPLITS`; they are not model
features.

The reference model is deliberately interpretable:

- calibration-frozen robust self-history residuals;
- rapid-deviation and sustained-drift channels;
- peer-relative deviation when enough contemporaneous peers exist;
- topology common-mode evidence for shared faults;
- elapsed-time persistence and quiet-period incident consolidation;
- empirical block-maximum thresholds and event-level evaluation;
- typed, equivalence-aware hierarchical localisation.

Petrobras does not invent topology or peers. It exercises the same canonical
contract, quality handling, self-history model and whole-well evaluation path
on real industrial recordings.

Start with the [run guide](notebooks/drive_research/README.md). The full design
and claim boundaries are in the
[Telecom-first methodology](docs/TELECOM_FIRST_ANOMALY_LOCALISATION_APPROACH.md).

Datasets and generated outputs stay outside Git. Code is licensed under
[Apache License 2.0](LICENSE); source datasets retain their own licences.
