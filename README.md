[MathAttack: Attacking Large Language Models Towards Math Solving Ability](https://ojs.aaai.org/index.php/AAAI/article/view/29949)
## Overview
<h1 align='center'>
  <br>
  <img src='png/overview.jpg'>
  <br>
</h1>

With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the robustness of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of robustness in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability.

We have integrated our MathAttack in the open-source tool OpenAttack.

## Result
<h1 align='center'>
  <br>
  <img src='png/Result.jpg'>
  <br>
</h1>

## Installation
To attack victim model, you should install LLMs through HuggingFace, including Flan-T5-large; Flan-T5-xl; ChatGLM2.
For ChatGPT, you should put your openAi-key in the main.py.

## Attack
```bash
python main.py
--llm GPT3.5
--prompt zero
--dataset GSM8K
```

## Data
* RobustMath.json consists of 300 high- quality MWP adversarial samples and could measure the robustness of LLMs math solving ability.  
* RobustBase.json is the orginal data of RobustMath 
