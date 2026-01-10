Goal: 


ToDo:
* make --recompute flag work for all commands!
* make predictions changing parameters easier
* check you dont need to add the prompt filter if you dont care about the correctness property checker
* make sure there are defaults for every parameter
* create an alternative messier version if people haven't installed pygraphviz that is uglier
* make sure multirun actually runs sequentially 
* should I let people try using either sentence or llm? Like use just one stage and not both, this could be good -- yeah i should do this! NAH UNNEEDED

* ahhh algorithm grpah will include resamples which will be biased -- at least tell people that

## Installation 

```bash
git clone https://github.com/Centrattic/global-cot-analysis.git
cd global-cot-analysis
uv sync
```

To use our interactive graph visualization tool, you must [install Node.js](https://nodejs.org/en/download). Then, install the needed packages.

```bash
cd global-cot-analysis/deployment
npm install
```

Note: To embed our semantic clustering graphs, we use the package pygraphviz, which requires [installing graphviz](https://graphviz.org/download/) first. If you don't have graphviz, we'll still generate an embedding; it'll just be messier.

## Viewing existing results 

Head to our [Vercel page](https://cot-clustering.vercel.app/).

## Running your own experiments!

If you have any issues, please reach out to us at riyaty@mit.edu. If you're interested in building on this work, we'd love to support you.

### Quickstart

If you just want to jump into the codebase, you can just use the default config files. Then, to generate the semantic clustering graph for the hex example, run:

```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=rollouts,resamples,flowcharts --multirun

cd graph_layout_service
python3 -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload

python -m src.main --config-name=hex command=graphviz
```

To run predictions on your prefixes, do: 

```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=predictions
```

To build the algorithmic clustering graph, run:

```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=cues,properties --multirun
```

To view both graphs, start the server and visit the page:

```bash
cd global-cot-analysis/deployment
npm run dev
```

For more examples and details on running your own experiments, review the documentation below.

### Configuration 

We use [Hydra](https://hydra.cc/docs/intro/) to manage our configs. There are four types of configs:
- **responses** configs, in `config/r`. These configure the process of generating rollouts and resamples of a model on a fixed prompt.
- **flowchart** configs, in `config/f`. These configure the two-stage clustering process to generate semantic clustering graphs.
- **predictions** configs, in `config/p`. These configure the predictions pipeline.
- **algorithms** configs, in `config/a`. These configure the process of generating algorithm cue dictionaries.

We then recommend creating a unique config file for each prompt that uses rollout, flowchart, predictions, and algorithms config files. Full examples of config files are at the end of this README.

### Adding a new prompt

To add a new prompt:
1. Add a new prompt to `prompts/prompts.json.`
2. Create a unique config file. The minimal specification required is just the models you wish to run the prompt on. To see all supported models, check out `MODEL_CONFIGS` in `src/utils/model_utils.py`, where you can easily add additional models (information below). We provide the following example of the config file for the "hex" problem.

```yaml
hex_config_example.yaml

defaults:
  - _self_
  - r: default
  - f: default
  - p: default
  - a: default

_name_: "hex"

prompt: "hex"
models: ["gpt-oss-20b"]
```

### To run rollouts on a given prompt

To run rollouts, first create a response config. You can just use the default. Specify `num_seeds_rollouts`, the number of rollouts to generate. Then run:

```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=rollouts
```

### To run resamples with various prefixes

Something you may want to do is resample model chains of thought with various prefixes. These resamples can be displayed on the semantic clustering graph.

You may first want to generate prefixes. To randomly sample prefixes from existing rollouts, specify the `num_prefixes_to_generate` in the response config. 

Then, run:

```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=prefix
```

Additionally, you can simply add prefixes yourself by writing to `prompts/prefixes.json`. Then, to run resamples, specify `num_seeds_prefixes` and run:

```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=resamples
```

### To build your flowcharts


## Full example configs

### Full example prompt config

```yaml

# The names of the response, flowchart, prediction, and algorithm configs
defaults:
  - _self_
  - r: default
  - f: default
  - p: default
  - a: default

_name_: "hex"

# Name of prompt to use from prompts/prompts.json
prompt: "hex"

# List of models to run over this prompt. You can run generation for multiple models, but we currently only support generating graphs for a single model.
models: ["gpt-oss-20b"]

# List of prefixes to include in the semantic graph, or run predictions over
prefixes: ["prefix-1", "prefix-2", "prefix-3", "prefix-4", "prefix-5"]

# List of properties to store for each response: the three below are whether the answer is correct/not, whether the response was a resample or a rollout, and what algorithms are present in the chain of thought
property_checkers: ["correctness", "resampled", "multi_algorithm"]

command: "rollouts"

```

### Full example algorithm config

```yaml
_name_: "default"

# Number of rollouts to send to GPT-5 to generate cue dictionaries
num_rollouts_to_study: 50
```

### Full example response config

```yaml

_name_: "default"

# Number of rollouts per prompt
num_seeds_rollouts: 50

# Number of resamples per prefix
num_seeds_prefixes: 10

# Number of workers for OpenRouter generation
max_workers: 250

# Number of prefixes to select when running the 'prefixes' command
num_prefixes_to_generate: 5

```

### Full example flowchart config

```yaml

_name_: "default"

# Number of rollouts to include in the semantic clustering graph
num_seeds_rollouts: 50

# Number of resamples per prefix to include in the semantic clustering graph
num_seeds_prefixes: 10

# The Hugging Face ID of the sentence embedding model used to semantically group statements
sentence_embedding_model: "sentence-transformers/paraphrase-mpnet-base-v2"

# Whether to use sentences or our chunking algorithm in the semantic graph
sentences_instead_of_chunks: false

# The semantic similarity threshold for clustering Stage 1
sentence_similarity_threshold: 0.75

# Number of parallel jobs for computing semantic similarities in clustering Stage 1 (-1 uses all CPUs)
n_jobs: -1

# OpenRouter ID for LLM to use for merging decisions in clustering Stage 2
llm_model: "openai/gpt-4o-mini"

# Threshold for showing a pair of clusters to the LLM to ask for merging in clustering Stage 2
llm_cluster_threshold: 0.75

# Gamma for Leiden algorithm in clustering Stage 2
gamma: 0.5

# Maximum number of parallel LLM calls for clustering Stage 2
max_workers: 100

# Delay between API calls for clustering stage 2
request_delay: 0.01
```

### Full example predictions config

```yaml

_name_: "default"

# Number of top-scoring rollouts to weigh and determine the final prediction
top_rollouts: 20 

# The weight between matching score and entropy: 0 = fully entropy-weighted (1.0 - entropy), 1 = ignores entropy (always 1.0)
beta: 0  

# If true, aggregate answers by weighted scores; if false, use unweighted fraction correct/incorrect
weigh: true 

# If true, require exact positional matching; if false, allow subsequence matching (LCS)
strict: false

# If true, slide window across rollout sequence; if false, only check first position (s=0)
sliding: true

```


### To create semantic clustering graphs

Here, we rely on the **flowchart** configuration. You can simply use the default; a brief explanation of the parameters is below.



```bash
cd global-cot-analysis
python -m src.main --config-name=hex command=rollouts
```


2. Initialize a prompt response parser with the correct/incorrect answers in `src/utils/prompt_utils.py` to the `PROMPT_FILTERS` dictionary.


### Adding new prompts


### Config structure

uv pip install scikit-learn

Each flowchart should be associated with a new config, that could reuse existing f/r configs. If you want to change things in f/r configs (ex. num rollouts in the flowchart, or clustering method), create a new f/r config.

If you want to switch prompt, models, prefixes, property_checkers, create a new config file in configs/ or in some cases, you can use an override (covered later). For changing command, definitely use an override.




```yaml
defaults:
  - _self_
  - r: test
  - f: test
  - p: test
  
_name_: "binary_digits_test"
prompt: "prompt-0"
models: ["gpt-oss-20b"]
prefixes: [prefix-0]
property_checkers: ["correctness", "resampled"]

command: "rollouts"
```

### Some notes
- **IMPORTANT:** very much avoid deleting config files if flowcharts have been made with them, since flowchart naming includes the config name, so it's hard to track how it was made if the config is deleted
- If you want to rerun some seeds, just delete them from the summary file (don't worry about deleting particular jsons, they will just be overwritten)
- We only use multiruns for prompts, models and prefixes (of which there can be multiple on a graph, are just looped over)

### To run various functionality 

Everything can be run from main.py. For getting responses, you should probably use the api method (specified in the configs/r configs). You could use local model method as well, in which case activations are extracted during generation and you don't have to run activation extraction later.

#### To run rollouts/resamples

```bash
python -m src.main --config-name=binary_digits_test command=rollouts

python3 -m src.main --config-name=compute_bases_test command=rollouts
python3 -m src.main --config-name=icecube_tinytest command=rollouts

```

```bash
python -m src.main --config-name=binary_digits_test command=resamples
```

#### To extract activations (if using api response generation method)

```bash
python -m src.main --config-name=binary_digits_test command=extract-activations
```

#### To create flowcharts

Before this, you must have graph layout service started.


```python
cd graph_layout_service
python3 -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload
```

For Riya (who installed pygraphviz in weird way): 

```python
cd graph_layout_service
export LD_LIBRARY_PATH=~/local/lib:$LD_LIBRARY_PATH && python -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload
```
```bash
python3 -m src.main --config-name=prompt6_predictive command=graphviz


python -m src.main --config-name=prompt6_50 command=sentence_predictions; python -m src.main --config-name=prompt6_100 command=sentence_predictions; python -m src.main --config-name=prompt6_250 command=sentence_predictions; python -m src.main --config-name=prompt6_500 command=sentence_predictions; python -m src.main --config-name=prompt6_500 command=chunking_predictions; python -m src.main --config-name=prompt6_250 command=chunking_predictions; python -m src.main --config-name=prompt6_100 command=chunking_predictions; python -m src.main --config-name=prompt6_50 command=chunking_predictions

python -m src.main --config-name=prompt6_750 command=flowcharts
python -m src.main --config-name=prompt6_1000 command=flowcharts

python3 -m src.main --config-name=claude_eval_aware command=graphviz
python3 -m src.main --config-name=sisters_smalltest command=flowcharts

python3 -m src.main --config-name=hex_alg_label command=graphviz
```

#### To run predictions

Running the below command without any prefixes run, will just save the txt file that allows you to view prefixes. 

```bash
python3 -m src.main --config-name=icecube_tinytest command=predictions
```

Then, once you select and add prefixes to the prefixes.json and config, you should run resamples with those prefixes.

```bash
python3 -m src.main --config-name=icecube_tinytest command=resamples
```

Once you have run resamples, now you can run predictions again to produce a file that computes the class percentages for the resmapled prefixes, and then (soon) will save a plot showing ground truth vs. predicted. 

All files/plots will be saved in prompts/{prompt-index}/model.

#### To run baseline predictions

The sentence and chunking baselines use separate commands (not the regular `predictions` command):

**Sentence baseline:**
```bash
python -m src.main --config-name=well command=sentence_predictions
```

**Chunking baseline:**
```bash
python -m src.main --config-name=well command=chunking_predictions
```

These will save results to `prompts/{prompt}/{model}/predictions/sentence_baseline/` and `prompts/{prompt}/{model}/predictions/chunking_baseline/` respectively.

```bash
python3 -m src.main --config-name=bagel_100 command=gemini_predictions
```

#### To run property checkers

Apply all configured property checkers to existing rollout and resample JSON files, as well as flowchart files. This is useful for adding new property checkers to existing data without regenerating responses.

```bash
python3 -m src.main --config-name=icecube_tinytest command=properties
```

This will process all JSON files in both the rollouts directory and all subdirectories of the resamples directory for each model specified in the config, as well as all flowchart files in the flowcharts directory. The property checkers defined in the `property_checkers` list will be applied, and the updated files will be saved back to disk with the new property values.

To force recomputation of all properties (even if they already exist), use the `--recompute` flag. This recomputes for all properties. If you want to recompute only for properties which are currently set to "None," (especially useful for the algorithm property), you don't need the --reocmpute flag.

```bash
python3 -m src.main --config-name=icecube_tinytest command=properties --recompute
```

For flowcharts, the system will:
- Check the `resampled` field in each response
- Skip responses where `resampled` field is missing or `None`
- If `resampled` is `false`, look for the source file in `rollouts/{seed}.json`
- If `resampled` is a prefix string, look for the source file in `resamples/{prefix}/{seed}.json`
- Update the flowchart response properties with the values from the source file


#### To extract top n-grams from a flowchart

Compute the top-k overlapping n-grams of cluster IDs across all rollouts in a flowchart, ranked by total frequency and by class-specific scores for a given property checker. Results are printed and written to `flowcharts/ngrams/{basename}_{prop_check}_{n}_{k}.txt`.

```bash
python -m src.flowchart.top_ngrams flowcharts/sisters/config-sisters_smalltest-smalltest_with_llm_.9_gamma.5_flowchart.json --n 3 --k 10 --prop_check correctness
```

If `n` exceeds the longest rollout length, it prints `n is too big` and does not write a file.


### Advanced features

#### Overrides

To override, just specify the config value when running main.py. Like:

```bash
# Run with a different prompt index
python -m src.main --config-name=binary_digits_test command=responses prompt=prompt-1
```
```bash
# Run with the "local" method for response generation
python -m src.main --config-name=binary_digits_test command=responses r.method=local
```

However, overrides are not preferred, since it makes it harder to track which config you actually used for something (though hydra does store outputs/ by timestamp, so you could theoretically compare timestamp of the artefacts you create, like response json files, to that output). Instead of overrides, just create new configs. 

#### Multiruns

This, I think, is very useful. Mainly use this to run multiple commands in one line, like both responses and rollouts, or run over multiple prompts at once, or try out different chunk clustering methods. For running over multiple models or prefixes, just create a new config (since those are lists in the config). 

```bash
# Run both rollouts and responses at the same time
python -m src.main --config-name=binary_digits --multirun command=rollouts,responses 

# Run with multiple clustering methods
python -m src.main --config-name=binary_digits --multirun command=rollouts f.method=sentence, sentence_then_llm, activations
```

### Adding a new prompt:
- Add a new prompt to prompts/prompts.json
- Add a new prompt response parser in src/utils/prompt_utils.py and add it to PromptFilters dictionary on line 120
- Correctness checker???

### Adding a new prefix
- Add a new prefix to prompts/prefixes.json

### To run the visualization: 
``` python
cd deployment
npm run dev &
```

Since node positions are precomputed, you now don't need to start the graph visualization service to visualize your flowcharts.

### To test sentence embedder clustering:
```bash
python3 -m src.clustering.test_clustering --k 5 --seed 42
# or choose a flowchart:
python -m src.clustering.test_clustering --flowchart flowcharts/hex/config-hex_smalltest-smalltest_flowchart.json --k 10
```

### To test LLM clustering:
```bash
python3 -m src.clustering.test_llm_clustering --flowchart flowcharts/hex/config-hex_smalltest-smalltest_with_llm_flowchart.json --clusters 0,3,5,7 --llm_model openai/gpt-4o-mini

python3 -m src.clustering.test_llm_clustering --flowchart flowcharts/sisters/config-sisters_smalltest-smalltest_with_llm_.9_gamma.5_flowchart.json --clusters 30,57,90,141,147,169,179,185,205,212 --llm_model openai/gpt-4o-mini
```

Use gamma .5 to begin with and lower to .4 if some you see examples of clusters that should be merged.

### To run hyperparameter sweeps 

#### To just run the sweeps with existing prefixes

```bash
 python run_hyperparameter_sweep.py --base-config prompt6_500 --prompt prompt-6 --model gpt-oss-20b
```

#### To find more pefixes

```bash
 python run_hyperparameter_sweep.py --base-config prompt6_500 --prompt prompt-6 --model gpt-oss-20b --prefixes
```

#### Deploy to Vercel

Only Riya can, since she owns the Vercel project :(

```bash
cd deployment
vercel --prod
```

To merge several clusters in a flowchart:
```bash
python scripts/merge_clusters.py {path/to/flowchart/json} --clusters 24 26
```

### Refine a flowchart (batch merges and cleanup)
```bash
python scripts/refine.py {path/to/flowchart.json} --merges '[[1,4],[15,19,5]]'
# Prints two lines: cluster count before, then after
# Overwrites the input file and removes any graph_layout
```

### Export minimal flowchart for Gemini
```bash
python3 scripts/gemini_flo.py {path/to/flowchart.json}
# Writes to a sibling folder: for_gemini/<basename>_for_gemini.json
# Keeps only nodes with keys like cluster-<n>, and for each retains top 4 sentences
```

### Prediction sweeps (repeat predictions and aggregate correlations)
```bash
python3 scripts/predictions_sweep.py CONFIG_NAME --n 10
# Example:
python3 scripts/predictions_sweep.py bagel_2000 --n 50
```
- Runs `python3 -m src.main --config-name=CONFIG_NAME command=predictions` N times.
- Archives per-run artifacts under `prediction_sweeps/CONFIG_TOPROLLOUTS_ALPHA/`:
  - `pngs/run_{i}__*.png`, `csvs/run_{i}__*.csv`
- Writes a summary file named `avg_corr_{AVG}_n_{N}` (no extension) containing:
  - average_correlation_images, run_correlations_images (for plotted responses)
  - average_correlation_csv, run_correlations_csv (for all responses in CSV)
- Directory name encodes method params read from the p-config: `{CONFIG}_{top_rollouts}_{alpha}`.
- If `<flowchart>_fully_condensed.json` exists, a second sweep also runs on the fully-condensed flowchart and saves under `{CONFIG}_{top_rollouts}_{alpha}_fully/` with its own summary.
- Incremental: existing runs are not recomputed; the sweep resumes to reach N.
- Shows a simple terminal progress bar as runs execute.

### Recomputing
Currently, there is a --recompute flag for command=graphviz and command=properties. This will override existing values. Note that properties recompute also updates node properties.
```bash
python3 src.main --config-name {} command=graphviz --recompute
```
```bash
python3 src.main --config-name {} command=properties --recompute
```