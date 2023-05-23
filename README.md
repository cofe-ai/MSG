# Masked Structural Growth
We grow up language models in pre-training with efficient schedules and function-preserving operators that yields 2x speedup.

MSG paper: https://arxiv.org/abs/2305.02869

## Quick Start
The following example shows how to run MSG on public Bert Pre-training data.
1. Pre-processing
<pre>
    python run_grow_bert.py
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
