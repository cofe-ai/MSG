# Masked Structural Growth
We grow up language models in pre-training with efficient schedules and function-preserving operators that yields 2x speedup.

MSG paper: https://arxiv.org/abs/2305.02869

## Quick Start
The following example shows how to run MSG on public Bert Pre-training data.
1. Pre-processing
<pre>
    preprocess_bert_data.py
</pre>
This generates static masks for raw data.

2. Run MSG

For Bert-base:
<pre>
    sh grow_bert_base.sh
</pre>

For Bert-large:
<pre>
    sh grow_bert_large.sh
</pre>

3. Evaluation
<pre>
    cd glue_eval
    sh run_glue_together_with_stat.sh
</pre>

## Notes

You can modify configs/*.json and set "attention_probs_dropout_prob" and "hidden_dropout_prob" to 0.0 in order to check function preservation. However, according to different pytorch versions, there can still be negligible differences of loss before and after growth.

## References

If this project helps you, please cite us, thanks!
```
@article{DBLP:journals/corr/abs-2305-02869,
  author       = {Yiqun Yao and
                  Zheng Zhang and
                  Jing Li and
                  Yequan Wang},
  title        = {2x Faster Language Model Pre-training via Masked Structural Growth},
  journal      = {CoRR},
  volume       = {abs/2305.02869},
  year         = {2023}
}
```
