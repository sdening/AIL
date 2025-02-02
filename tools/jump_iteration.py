#import tools.ExampleTool

class jump_iteration:
    def __init__(self, examples): #examples is class instance of example tool
        self.examples = examples
 
    def jump_back_to_iteration(self, iteration_number, optimization_history, cat):
        if 0 <= iteration_number < len(optimization_history):

            part_of_prompt_to_modify = optimization_history[iteration_number]["prompt"]
               
            #handle examples   
            current_pos_ex = len(self.examples.added_examples["positive"])
            current_neg_ex = len(self.examples.added_examples["negative"])
            
            target_pos_ex = optimization_history[iteration_number]["positive_examples_count"]
            target_neg_ex = optimization_history[iteration_number]["negative_examples_count"]

            if current_pos_ex > target_pos_ex:
                for _ in range(current_pos_ex - target_pos_ex):
                    self.examples.remove_positive_example(part_of_prompt_to_modify, cat)
            elif current_pos_ex < target_pos_ex:
                for _ in range(target_pos_ex - current_pos_ex):
                    self.examples.add_positive_example(part_of_prompt_to_modify, cat)

            # Adjust negative examples
            if current_neg_ex > target_neg_ex:
                for _ in range(current_neg_ex - target_neg_ex):
                    self.examples.remove_negative_example(part_of_prompt_to_modify, cat)
            elif current_neg_ex < target_neg_ex:
                for _ in range(target_neg_ex - current_neg_ex):
                    self.examples.add_negative_example(part_of_prompt_to_modify, cat)


        
            print("Jumped back to the following 'part_of_prompt_to_modify': ", part_of_prompt_to_modify)
            return part_of_prompt_to_modify
        else:
            raise ValueError(f"Iteration number {iteration_number} is out of range.")
            return None