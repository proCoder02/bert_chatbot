from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
import torch

# Load a dataset - SQuAD is suitable for QA fine-tuning
dataset = load_dataset("squad")

# Use a pretrained model and tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=True,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        answer_start = answer["answer_start"][0]
        answer_text = answer["text"][0]
        answer_end = answer_start + len(answer_text)

        # Find token start and end
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer is out of span
        if not (offsets[token_start_index][0] <= answer_start and offsets[token_end_index][1] >= answer_end):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_pos = cls_index
            end_pos = cls_index
            for idx in range(token_start_index, token_end_index + 1):
                if offsets[idx][0] <= answer_start < offsets[idx][1]:
                    start_pos = idx
                if offsets[idx][0] < answer_end <= offsets[idx][1]:
                    end_pos = idx
                    break
            start_positions.append(start_pos)
            end_positions.append(end_pos)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples



tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Training Arguments
args = TrainingArguments(
    "qa-finetuned-model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")
