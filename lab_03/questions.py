"""Эталоны, зафиксированные на основе PDF до поисковых прогонов."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    expected: str
    sources: tuple[tuple[int, str], ...]
    alternative: str = ""


BANK = (
    Question("Q1", "According to the abstract, what data and learning method were used for the first GPT-3 fine-tuning stage?",
             "Labeler demonstrations of desired behavior, used for supervised learning.",
             ((1, "labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning"),)),
    Question("Q2", "Which InstructGPT and GPT-3 model sizes were compared in the human preference result, and which one was preferred?",
             "1.3B InstructGPT was preferred to 175B GPT-3.",
             ((1, "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3"),
              (2, "outputs from our 1.3B PPO-ptx model are preferred to those from the 175B GPT-3")),
             "How large were the two systems in the head-to-head judgment where the smaller one won?"),
    Question("Q3", "What confidence level do the error bars in Figure 1 represent?",
             "95% confidence intervals.", ((2, "Error bars throughout the paper are 95% confidence intervals"),),
             "What statistical coverage do the vertical uncertainty markers show in the first plot?"),
    Question("Q4", "In section 3.2, what key was used to split prompts, and what overlap between sets did this prevent?",
             "The split by user ID kept users from the training set out of validation and test.",
             ((6, "We also create our train, validation, and test splits based on user ID, so that the validation and test sets contain no data from users whose data is in the training set"),)),
    Question("Q5", "What two sources provided the training tasks in section 3.3?",
             "Prompts written by labelers and prompts submitted to early InstructGPT models through the API.",
             ((7, "Our training tasks are from two sources: (1) a dataset of prompts written by our labelers and (2) a dataset of prompts submitted to early InstructGPT models on our API"),),
             "Who composed the exercises, and what other channel supplied examples for the models to learn from?"),
    Question("Q6", "What pairwise agreement rates were reported for training and held-out labelers in section 3.4?",
             "72.6 ± 1.5% for training labelers, 77.3 ± 1.3% for held-out labelers.",
             ((8, "training labelers agree with each-other 72.6± 1.5% of the time, while for held-out labelers this number is 77.3± 1.3%"),)),
    Question("Q7", "In the PPO bandit environment, what determines the reward for a prompt and response?",
             "The reward model determines the reward.",
             ((9, "Given the prompt and response, it produces a reward determined by the reward model and ends the episode"),
              (6, "We use the output of the RM as a scalar reward")),
             "During the final tuning stage, what provides the numerical feedback for each attempted reply?"),
    Question("Q8", "What API provided automatic toxicity scores for RealToxicityPrompts?",
             "Perspective API.", ((13, "Perspective API8 to obtain automatic toxicity scores"),
             (14, "Comparing human evaluations and automatic evaluations (Perspective API scores) on RealToxicityPrompts"),
             (58, "In the RealToxicityPrompts task, we measure toxicity via the Perspective API")),
             "Which outside scoring service assessed how offensive the generated continuations were?"),
    Question("Q9", "How many contractors were hired for labeling and how were they selected?",
             "About 40 contractors, selected by their performance on a screening test.",
             ((2, "hire a team of 40 contractors to label our data, based on their performance on a screening test"),
              (19, "hired about 40 contractors, guided by their performance on a screening test"))),
    Question("Q10", "In the 5-fold cross-validation experiment on labelers, what accuracy did reward models reach for held-out labelers?",
             "69.6 ± 0.9%.", ((12, "accuracy of 69.6± 0.9% on predicting the preferences of labelers in the held-out group"),)),
)
QUESTIONS = {item.id: item for item in BANK}
EARLY = ("Q1", "Q2", "Q3")
LATE = ("Q6", "Q8", "Q10")
CITATION_IDS = ("Q5", "Q6", "Q7", "Q8", "Q10")
PROMPT_IDS = ("Q2", "Q3", "Q8", "Q9", "Q10")
BOUNDARY = (("Q1", 1, 1000), ("Q4", 6, 3000), ("Q5", 7, 2000),
            ("Q6", 8, 1000), ("D23", 8, 2000))

ABSENT = (
    ("N1", "What was the exact total electricity consumption, in kWh, of all InstructGPT training runs?",
     "kWh, electricity, energy, power"),
    ("N2", "What dollar amount did OpenAI pay the contractors in total for labeling the InstructGPT datasets?",
     "paid, payment, dollar, compensation, wage, salary, contractor, hourly"),
    ("N3", "Which city housed the datacenter used to train the 175B InstructGPT policy?",
     "datacenter, data center, cluster, location, hardware, GPU"),
    ("N4", "What exact random seed was used for the published 175B PPO-ptx policy training run?",
     "random seed, seed, hyperparameters, PPO-ptx"),
    ("N5", "What percentage of InstructGPT training prompts were written in Japanese specifically?",
     "Japanese, languages, language, non-English, langid"),
)

DIALOGUE_QUESTIONS = (
    Question("D11", "What is InstructGPT in Ouyang et al.?",
             "The series of GPT-3 models fine-tuned to follow instructions via human feedback.",
             ((1, "reinforcement learning from human feedback. We call the resulting models InstructGPT"),
              (2, "We call the resulting models InstructGPT"))),
    Question("D12", "How was its first stage trained?",
             BANK[0].expected, (BANK[0].sources[0],
             (8, "We fine-tune GPT-3 on our labeler demonstrations using supervised learning"))),
    Question("D21", "In InstructGPT, what did the reward model output for a prompt and response?",
             "A scalar reward.", ((8, "take in a prompt and response, and output a scalar reward"),
                                  (6, "We use the output of the RM as a scalar reward"))),
    Question("D22", "What size did they use for it?",
             "6 billion parameters.", ((8, "we only use 6B RMs"),)),
    Question("D23", "Which model did it start from, and what layer was removed?",
             "The SFT model with its final unembedding layer removed.",
             ((8, "Starting from the SFT model with the final unembedding layer removed"),)),
    Question("D31", "What is RealToxicityPrompts used for in the paper?",
             "Evaluation of toxicity.", ((3, "To measure toxicity, we use the RealToxicityPrompts dataset"),
                                       (13, "We first evaluate our models on the RealToxicityPrompts dataset"),
                                       (58, "In the RealToxicityPrompts task, we measure toxicity via the Perspective API"))),
    Question("D32", "Which API did they score it with?",
             BANK[7].expected, BANK[7].sources),
    Question("D33", "How many prompts were labeled for the human evaluation on that dataset in Figure 7?",
             "1,729 prompts.", ((14, "A total of 1,729 prompts were labeled"),)),
)
QUESTIONS.update((item.id, item) for item in DIALOGUE_QUESTIONS)
DIALOGUES = (("D11", "D12", "Q2"), ("D21", "D22", "D23"), ("D31", "D32", "D33"))
FOLLOW_UPS = {"Q2": "Which sizes were compared when people preferred it to GPT-3?",
              "D33": "How many prompts did humans label on that dataset in Figure 7?"}
