import json, sys, os, re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
from vllm import SamplingParams
#from utils import extract_reasoning_and_text

class LLMAgent:
    
    def __init__(self, name, model, model_name, instruction_format='reasoning', reasoning_instruction="COT"):
        self.name = name
        self.model = model # model = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=2048)
        self.model_name = model_name
        self.hand = []
        self.reasoning_instruction = reasoning_instruction
        self.instruction_format = instruction_format
    
    def update_system_message(self, new_message):
        self.PLAYER_SYSTEM_MSG = new_message
    
    def update_BS_rules(self, new_rules):
        self.BS_RULES = new_rules

    def add_cards(self, cards):
        self.hand.extend(cards)

    def remove_cards(self, cards):
        for c in cards:
            if c in self.hand:
                self.hand.remove(c)

    def chat(self, 
             messages,
             num_responses=10,
             temperature=1.0,
             top_p=1.0,
             max_tokens=1024,
             debug=True):


        # add system message
        # we assume these are already added
        #messages.insert(0, {'role':'system', 'content':self.PLAYER_SYSTEM_MSG + self.BS_RULES})
        
        if debug:
            print("=== Prompt ===")
            for msg in messages:
                print(f"Role: {msg['role']}\nContent: {msg['content']}\n{'-'*40}")

        params_list = [
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=i
                )
                for i in range(num_responses)
            ]
        
        # vLLM requires 1 prompt per SamplingParams → duplicate prompt 10 times
        prompt_batch = [messages for _ in range(num_responses)]

        # Run batch inference
        results = self.model.chat(
            messages=prompt_batch,
            sampling_params=params_list
        )
        # Collect outputs
        outputs = []
        for i, r in enumerate(results):
            if 'gpt' in self.model_name:
                # step one take: r.text and extract reasoning
                reasoning = extract_GPT_reasoning_and_text(r, self.model)
                print(results[4].finished)
                print(reasoning)
                # step two: take "text" and parse json
                parsed = LLMAgent.parse_action(reasoning['text'])

                if parsed.get('Parse_fail', False):
                    # skip if there's a parse fail
                    continue

                parsed.update({'reasoning':reasoning['reasoning'],
                               'seed':i,
                               'prompt': messages[-1]['content']})
                outputs.append(parsed)

            else:
                parsed = LLMAgent.parse_action(r.outputs[0].text)
                parsed.update({'seed':i,
                            'prompt': messages[-1]['content']})
                outputs.append(parsed)

        # Print results
        if debug:
            for o in outputs:
                print(f"=== Seed {o['seed']} ===")
                print(o)
                print()

        return outputs


    @staticmethod
    def parse_action(raw_text):
        try:
            return json.loads(raw_text)
        except:
            pass

        try:
            # Extract the first {...} block
            m = re.search(r"\{.*?\}", raw_text, flags=re.S)
            if not m:
                raise ValueError("No JSON object found")
            js_text = m.group()

            # Remove JS // comments
            js_text = re.sub(r'//.*?(?=\n|$)', '', js_text)

            # Remove JS /* */ comments
            js_text = re.sub(r'/\*.*?\*/', '', js_text, flags=re.S)

            # Remove Python-style trailing comments ( # ... ) outside of strings
            def remove_trailing_hash(line):
                in_str = False
                escaped = False
                for i, ch in enumerate(line):
                    if ch == '\\' and not escaped:
                        escaped = True
                        continue
                    if ch in ('"', "'") and not escaped:
                        in_str = not in_str
                    if ch == '#' and not in_str:
                        return line[:i].rstrip()
                    escaped = False
                return line

            js_text = "\n".join(remove_trailing_hash(l) for l in js_text.splitlines())

            # Remove trailing commas
            js_text = re.sub(r',\s*}', '}', js_text)
            js_text = re.sub(r',\s*\]', ']', js_text)

            # Replace smart quotes
            js_text = js_text.replace('\u201c', '"').replace('\u201d', '"')
            js_text = js_text.replace('\u2018', "'").replace('\u2019', "'")

            # Collapse new lines
            js_text = re.sub(r'\n+', ' ', js_text)

            # Ensure keys are quoted
            js_text = re.sub(r'(\w+)\s*:', r'"\1":', js_text)

            js_text = js_text.strip()

            return json.loads(js_text)

        except Exception as e:
            print("COULD NOT PARSE JSON:", e)
            print(raw_text)
            return {
                "Reasoning": raw_text,
                "Action": "PLAY",
                "Declared_Rank": None,
                "Card_idx": [],
                "Parse_fail": True
            }
