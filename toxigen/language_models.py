import requests
import json
from toxigen.alice import beam_search
import os
import openai
import google.generativeai as genai

class GPT3(object):
    def __init__(self, endpoint_url, apikey):
        self.apikey = apikey
        self.endpoint_url = endpoint_url

    def __call__(self, prompt, topk=1, max_tokens=1):
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = [p.replace("'", "").replace('"', "") for p in prompt]
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.9,
            "n": 1,
            "stream": False,
            "logprobs": topk,
            "stop": ["<|endoftext|>", "\\n"]
        }
        r = requests.post(self.endpoint_url,
            headers={
                "Authorization": f"Bearer {self.apikey}",
                "Content-Type": "application/json"
            },
            json = payload
        )       
        output = json.loads(r.content)
        return output

    def from_prompt(self, prompt, topk=10, max_tokens=10):
        output = self.__call__(prompt, topk, max_tokens)
        return output["choices"][0]["text"]

class GPT4(object):
    def __init__(self, model_name, apikey):
        self.model = model_name
        self.apikey = apikey

    #def __call__(self, prompt, top_p=1, max_tokens=1):
    def __call__(self, prompt, topk=1, max_tokens=1):
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = [p.replace("'", "").replace('"', "") for p in prompt]
        client = openai.OpenAI(api_key=self.apikey)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt[0]},
        ]
        response = client.chat.completions.create(
            model=self.model,
            #model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.9,
            n=1,
            stream=False,
            logprobs=True,
            top_logprobs=topk,
            #top_p=top_p,
            stop=["<|endoftext|>", "\\n", "\n"]
        )

        return response

    def from_prompt(self, prompt, topk=10, max_tokens=10):
        output = self.__call__(prompt, topk, max_tokens)
        return output.choices[0].message.content
    
class LLaMA3(object):
    pass

class Gemini(object):
    pass

class ALICE(object):
    def __init__(self, model_name, language_model, classifier, mode, device="cpu"):
        self.classifier = classifier
        self.model_name = model_name
        self.language_model = language_model
        self.device = device
        self.mode = mode

    def __call__(self, prompt):
        return self.generate(prompt)

    def generate(self, prompt):
        if self.mode == "neutral":
            flag = 0
        else:
            flag = 1
        return beam_search(prompt, self.model_name, self.language_model, self.classifier, flag, self.device)
