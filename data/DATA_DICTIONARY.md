# Data Dictionary

## Dataset: `data/qa_dataset.csv` (evaluation set)
| Column | Type | Description |
|---|---|---|
| id | integer | Unique question ID |
| question | string | Natural-language question a user might ask |
| ground_truth_answer | string | Correct short answer, taken from the source document |
| source_file | string | Document in `data/documents/` that contains the answer |
| answer_keyword | string | Phrase that must appear in a retrieved chunk for it to count as a correct retrieval |
| category | string | Topic label (leave, remote_work, security, equipment) |

## Corpus: `data/documents/`
| File | Description |
|---|---|
| employee_leave_policy.txt | Fictional HR leave policy (annual, sick, parental leave, holidays) |
| remote_work_and_it_security.txt | Fictional remote work and IT security rules |

## Internal chunk record (created at runtime)
| Field | Type | Description |
|---|---|---|
| source | string | Source file name |
| chunk_id | integer | Index of the chunk within its document |
| text | string | Chunk text (~120 words, 30-word overlap) |
| score | float | Similarity of the chunk to the question (0-1) |

## Evaluation metrics (`outputs/evaluation_results.csv`)
| Metric | Meaning |
|---|---|
| Hit@k | Share of questions where a correct chunk appears in the top-k results |
| MRR | Mean Reciprocal Rank: average of 1/rank of the first correct chunk |
