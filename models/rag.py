from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" 

model_gen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")


class R:
    def __init__(self, lsh, model, tokenizer, model_gen, dataset, k=3):
        self.messages = []
        self.k = k
        self.model = model
        self.lsh = lsh
        self.tokenizer = tokenizer
        self.model_gen = model_gen
        self.dataset = dataset

    def complete_prompt(self, relevant_chunks, ingredients):
        recipe_text = '\n'.join(relevant_chunks)
        relevant = f"\nAdvices: Ingredients  {ingredients}"
        system_prompt = f"""You are assistant of professional chef.
    You get several relevant advices from a recipe of a good cook book {recipe_text}
    ###GOAL
    Answer user's question using provided advices. 
    Be careful, sometimes advices are irrelevant.
    
    ###OBEY AND NEVER DO
    Answer questions only about culinary topics. 
    Other topics are not supported for conversation."""
        return system_prompt, relevant
        
    def find_relevant(self, question):
        q_vec = self.model.encode(question)
        results = self.lsh.query(q_vec, top_k=self.k)
        results.sort(key=lambda x: x[1], reverse=True)
        print(results)
        index_relevant = results[0][0]
        print("results", index_relevant)
        ingredients = self.dataset[index_relevant]["ingredients"]
        print("Candidates:", results)
        relevant_chunks = [text for (_, _,_, text) in results]
        return relevant_chunks, ingredients

    
    def generate_answer(self, question):
        relevant_chunks, ingredients = self.find_relevant(question)
        system_prompt, user_prompt = self.complete_prompt(relevant_chunks, ingredients)
        self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": question + user_prompt})
        text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model_gen.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
        generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

# your code here
class RH:
    def __init__(self, lsh, model, tokenizer, model_gen, dataset, k=3):
        self.messages = []
        self.k = k
        self.model = model
        self.lsh = lsh
        self.tokenizer = tokenizer
        self.model_gen = model_gen
        self.dataset = dataset

    def reformulate(self, question):
        history_text = ""
        for msg in self.messages[-6:]: 
            role = msg["role"]
            if role!= 'system':
                content = msg["content"]
                history_text += f"{role}: {content}\n"

        prompt = f"""
            You are assistant chef. User wants to continue to dialogue.
            
            History of dialogue:
            {history_text}
            
            Current user's query:
            "{question}"
            
            Reformulate question:
            - It should be short.
            - It should include info from the history.
            - Don't create facts. Use the history and the question as a base.
            
            Return only reformulated question.
            """

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        gen = self.model_gen.generate(model_inputs.input_ids, max_new_tokens=128)
        new_ids = gen[0][len(model_inputs.input_ids[0]):]
        reformulated = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return reformulated.strip()


    def complete_prompt(self, relevant_chunks, ingredients):
        recipe_text = '\n'.join(relevant_chunks)
        relevant = f"\nAdvices: Ingredients  {ingredients} Recipe: {recipe_text}"
        system_prompt = f"""You are assistant of professional chef.
    You get several relevant advices from a recipe of a good cook book
    ###GOAL
    Answer user's question using provided advices. 
    Be careful, sometimes advices are irrelevant.
    
    ###OBEY AND NEVER DO
    Answer questions only about culinary topics. 
    Other topics are not supported for conversation."""
        return system_prompt, relevant
        
    def find_relevant(self, question):
        q_vec = self.model.encode(question)
        results = self.lsh.query(q_vec, top_k=self.k)
        results.sort(key=lambda x: x[1], reverse=True)
        index_relevant = results[0][0]
        print("results", index_relevant)
        ingredients = self.dataset[index_relevant]["ingredients"]
        print("Candidates:", results)
        relevant_chunks = [text for (_, _,_, text) in results]
        return relevant_chunks, ingredients

    
    def generate_answer(self, question):

        full_question = self.reformulate(question)
        print("Reformulated:", full_question)
        relevant_chunks, ingredients = self.find_relevant(full_question)
        system_prompt, user_prompt = self.complete_prompt(relevant_chunks, ingredients)
        if len(self.messages) >= 2:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": full_question + user_prompt})
        text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
    
        generated_ids = self.model_gen.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
        generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
    
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def test_rag(rag):
    while True:
        input_query = input()
        if input_query == "stop":
            break
        recipe = rag.generate_answer(input_query)
        print(recipe)

