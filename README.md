
# Global CoT analysis: Initial ideas to uncover patterns across many chains of thought

Welcome to our deployment repository! Here, you can make your own semantic and algorithm clustering graphs for reasoning traces. 

## TL;DR

We believe that more research effort should go into studying many chains of thought collectively and identifying global patterns across the collection. We argue why this direction is important and potentially tractable, and introduce two initial methods for a simpler setting: analyzing a collection of CoTs generated with a fixed model on a fixed prompt. Both of our methods—semantic step clustering and algorithmic step clustering—group recurring patterns across reasoning traces, but operate at different levels of abstraction.

Semantic step clustering leverages the fact that traces for the same prompt often repeat each other in slightly different ways. By clustering similar sentences like “1+2=3” and “2+1=3”, we compress a large set of CoTs into a detailed but messy reasoning graph. We evaluate semantic step clustering on a downstream task: predicting a model’s output based on a partial CoT using the graph. Our method outperforms naive baselines, although the predictions are imperfect.

At a higher level, the model often solves a problem by stringing together a small set of reusable strategies. This motivates algorithmic step clustering, which builds a coarser, cleaner graph that reflects the set of alternative pathways at the strategy level. We sanity-check algorithmic step clustering by building algorithm graphs on simple problems where we know the space of solution strategies.

There are probably better ways to study collections of rollouts, and we’re excited for more work in this area! Finding downstream tasks to test methods against is a priority.


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
3. If you care about running predictions, you should use the `correctness` property checker, and must set the correct answer for the prompt in `src/utils/prompt_utils.py` in `PROMPT_FILTERS`.

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
property_checkers: ["correctness"]
models: ["gpt-oss-20b"]
```
(Full configs further down.)


### To run rollouts on a given prompt

To run rollouts, first create a response config. You can just use the default. Specify `num_seeds_rollouts`, the number of rollouts to generate. Then run:

```bash
cd global-cot-analysis
python -m src.main --config-name={prompt_cfg} command=rollouts
```

Rollouts will appear in `global-cot-analysis/prompts/{prompt_name}/{model_name}/rollouts.`

### To run resamples with various prefixes

Something you may want to do is resample model chains of thought with various prefixes. These resamples can be displayed on the semantic clustering graph.

You may first want to generate prefixes. To randomly sample prefixes from existing rollouts, specify the `num_prefixes_to_generate` in the response config. 

Then, run:

```bash
cd global-cot-analysis
python -m src.main --config-name={prompt_cfg} command=prefix
```

Additionally, you can simply add prefixes yourself by writing to `prompts/prefixes.json`. Then, to run resamples, specify `num_seeds_prefixes` and run:

```bash
cd global-cot-analysis
python -m src.main --config-name={prompt_cfg} command=resamples
```
Resamples will appear in `global-cot-analysis/prompts/{prompt_name}/{model_name}/resamples.`

### To build semantic clustering graphs, and embed them

```bash
cd global-cot-analysis
python -m src.main --config-name={prompt_cfg} command=flowcharts
```

```bash
cd graph_layout_service
python3 -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload

python -m src.main --config-name={prompt_cfg} command=graphviz
```

Semantic clustering graphs will be saved to `global-cot-analysis/flowcharts/{prompt_name}.`

### To generate algorithm cue lists

This generates cue lists, and then updates them in existing flowcharts using "properties". All properties can be viewed in the flowchart files.

```bash
cd global-cot-analysis
python -m src.main --config-name={prompt_cfg} command=cues,properties --multirun
```

### To run predictions 

```bash
cd global-cot-analysis
python -m src.main --config-name={prompt_cfg} command=predictions
```

Prediction results will be saved to `global-cot-analysis/prompts/{prompt_name}/{model_name}/predictions.`

### To visualize results 

```bash
cd global-cot-analysis/deployment
npm run dev
```
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

## General Documentation/Additional Features

### Property checkers

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


### Adding a new model

### Extracting top n-grams for a semantic clustering graph

Compute the top-k overlapping n-grams of cluster IDs across all rollouts in a flowchart, ranked by total frequency and by class-specific scores for a given property checker. Results are printed and written to `flowcharts/ngrams/{basename}_{prop_check}_{n}_{k}.txt`.

```bash
python -m src.flowchart.top_ngrams flowcharts/sisters/config-sisters_smalltest-smalltest_with_llm_.9_gamma.5_flowchart.json --n 3 --k 10 --prop_check correctness
```

If `n` exceeds the longest rollout length, it prints `n is too big` and does not write a file.

### Recompute flag

### Best practices for development

Don't delete configs


#### To do

ToDo:
* make --recompute flag work for all commands!
* make predictions changing parameters easier
* check you dont need to add the prompt filter if you dont care about the correctness property checker
* make sure there are defaults for every parameter
* create an alternative messier version if people haven't installed pygraphviz that is uglier
* make sure multirun actually runs sequentially 
* check recompute flag on properties 
* maybe make it easier to visualize algs w/o having to use the node page
* add back ngrams

* ahhh algorithm grpah will include resamples which will be biased -- at least tell people that
