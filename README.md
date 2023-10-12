MathAttack: Attacking Large Language Models Towards Math Solving Ability
## Overview

MathAttack is an effective method for attacking the math solving ability of large language models, including Logical Entity Recognition, Freezing Logical Entity and text Attack.

We have integrated our MathAttack in the open-source tool OpenAttack.

## Installation
To attack victim model, you should install LLMs through HuggingFace, including Flan-T5-large; Flan-T5-xl; ChatGLM2.
For ChatGPT, you should put your openAi-key in the main.py.

## Attack
```bash
# download and preprocess child data
python main.py
--llm GPT3.5
--prompt zero
--dataset GSM8K
```
