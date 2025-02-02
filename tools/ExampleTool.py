import pandas as pd
import os

class ExampleTool:
    def __init__(self, df_train):
        self.df_train = df_train
        self.added_examples = {
            "positive": [],
            "negative": []
        }
        
    @staticmethod
    def make_answer_instruction(n_words=50):
        return f'Start your answer with "yes" or "no" and then justify your response in no more than {n_words} words.'

    def sample_example(self, filtered_set, cat):
        example = filtered_set['text'].sample(n=1).iloc[0]
        return example

    def add_positive_example(self, part_of_the_prompt, cat):
        example_type = "positive"
        mode = "add"

        if mode == "add":
            if example_type == "positive":
                filtered_set = self.df_train[self.df_train[cat] == 1]
            elif example_type == "negative":
                filtered_set = self.df_train[self.df_train[cat] == 0]

            if filtered_set.empty:
                return "No positive clause found of the category in data set."

            sampled_example = self.sample_example(filtered_set, cat)
            self.added_examples[example_type].append(sampled_example)

            examples_text = ""
            for example_type, examples in self.added_examples.items():
                for example in examples:
                    examples_text += f"For example, consider this clause of the same category: \"{example}\""


    def add_negative_example(self, part_of_the_prompt, cat):
        example_type = "negative"
        mode = "add"

        if mode == "add":
            if example_type == "positive":
                filtered_set = self.df_train[self.df_train[cat] == 1]
            elif example_type == "negative":
                filtered_set = self.df_train[self.df_train[cat] == 0]

            if filtered_set.empty:
                return "No negative clause found of the category in data set."

            sampled_example = self.sample_example(filtered_set, cat)
            self.added_examples[example_type].append(sampled_example)

            examples_text = ""
            for example_type, examples in self.added_examples.items():
                for example in examples:
                    examples_text += f"For example, consider this clause which is not of this category: \"{example}\""

    def remove_positive_example(self, part_of_the_prompt, cat):
        example_type = "positive"
        mode = "remove"

        if mode == "remove":
            if not self.added_examples[example_type]:
                print(f"No {example_type} example to remove. \n")
                examples_text = ""
                for example_type, examples in self.added_examples.items():
                    for example in examples:
                        examples_text += f"For example, consider this clause of the same category: \"{example}\""

                return examples_text + part_of_the_prompt

            removed_example = self.added_examples[example_type].pop(0)

            examples_text = ""
            for example_type, examples in self.added_examples.items():
                for example in examples:
                    examples_text += f"For example, consider this clause of the same category: \"{example}\""


    def remove_negative_example(self, part_of_the_prompt, cat):
        example_type = "negative"
        mode = "remove"

        if mode == "remove":
            if not self.added_examples[example_type]:
                print(f"No {example_type} example to remove. \n")
                examples_text = ""
                for example_type, examples in self.added_examples.items():
                    for example in examples:
                        examples_text += f"For example, consider this clause which is not of this category: \"{example}\""

                return examples_text + part_of_the_prompt

            removed_example = self.added_examples[example_type].pop(0)

            examples_text = ""
            for example_type, examples in self.added_examples.items():
                for example in examples:
                    examples_text += f"For example, consider this clause which is not of this category: \"{example}\""
