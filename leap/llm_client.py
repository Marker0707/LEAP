from abc import ABC, abstractmethod
import json
from pydantic import BaseModel
import requests
import re

from torch import Size

class LLMClient(ABC):
    @abstractmethod
    def extract_phenotypes(self, content: str) -> list[str]:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.llm_name = "OpenAI-GPT-4o"
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    class GPT_Phenotype_Format(BaseModel):
        Phenotypes: list[str]

    def extract_phenotypes(self, content: str) -> list[str]:
        completion = self.client.responses.parse(
            model = "gpt-4o",
            input = [
                {
                    "role" : "system",
                    "content" : "You are a medical expert specialized in human phenotypes. I will provide you with a medical record. Your task is to extract all of the patient’s abnormal phenotypes and symptoms, including anatomical anomalies, clinical symptoms, diagnostic findings, test results, and specific conditions or syndromes, and convert them into professional medical terminology. Please completely ignore negative findings, normal findings, procedures and family history."
                },
                {
                    "role": "user",
                    "content": content
                }
                        ],
            # temperature = 0.7,  # 使用cot模型时候不能设置温度
            text_format=self.GPT_Phenotype_Format
            )
        
        return completion.output_parsed.Phenotypes


class Gemma3Client(LLMClient):
    def __init__(self, endpoint: str ="http://localhost:11433/api/generate", size: str = "27b"):
        assert size in {"4b", "12b", "27b"}, "size must be '4b', '12b' or '27b'"
        self.llm_name = f"Ollama-Gemma3:{size}"
        self.modelsize = size
        self.endpoint = endpoint

    def extract_phenotypes(self, content: str) -> list[str]:
        try:
            response = requests.post(
                url= self.endpoint,
                json={
                    "model": f"gemma3:{self.modelsize}",
                    "prompt": f"You are a medical expert specialized in human phenotypes. I will provide you with a medical record. Your task is to extract all of the patient’s abnormal phenotypes and symptoms, including anatomical anomalies, clinical symptoms, diagnostic findings, test results, and specific conditions or syndromes, and convert them into professional medical terminology. Please completely ignore negative findings, normal findings, procedures and family history. Please output the phenotypes with '@' symbols at intervals. Be careful not to output any text other than the above. Below is the medical record:\n {content}",
                    "stream": False,
                    # "options": {"temperature": 0.7}
                    },
                timeout=240
                )
            response.raise_for_status()
            return [item.strip() for item in response.json()["response"].split("@") if item.strip()]

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")


class DeepSeekClient(LLMClient):
    def __init__(self, endpoint: str ="http://localhost:11433/api/generate", size: str = "32b"):
        assert size in {"1.5b", "7b", "14b", "32b"}, "size must be '1.5b', '7b', '14b', '32b'"
        self.llm_name = f"Ollama-DeepSeek-R1:{size}"
        self.modelsize = size
        self.endpoint = endpoint

    def extract_phenotypes(self, content: str) -> list[str]:
        try:
            response = requests.post(
                url= self.endpoint,
                json={
                    "model": f"deepseek-r1:{self.modelsize}",
                    "prompt": f"You are a medical expert specialized in human phenotypes. I will provide you with a medical record. Your task is to extract all of the patient’s abnormal phenotypes and symptoms, including anatomical anomalies, clinical symptoms, diagnostic findings, test results, and specific conditions or syndromes, and convert them into professional medical terminology. Please completely ignore negative findings, normal findings, procedures and family history. Please output the phenotypes with '@' symbols at intervals. Be careful not to output any text other than the above. Below is the medical record:\n {content}",
                    "stream": False,
                    "think": False
                    # "options": {"temperature": 0.7}
                    },
                timeout=240
                )
            response.raise_for_status()
            rm_think = re.sub(r'<think>.*?</think>\s*', '', response.json()["response"], flags=re.DOTALL).split("@")
            return [pheno for pheno in rm_think if pheno]
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")


class QwenClient(LLMClient):
    def __init__(self, endpoint: str ="http://localhost:11433/api/generate", size: str = "32b"):
        assert size in {"0.6b", "1.7b", "4b", "8b", "14b", "32b", "30b", "235b"}, "size must be '0.6b', '1.7b', '4b', '8b', '14b', '32b', '30b-a3b', and '235b-a22b'"
        self.llm_name = f"Ollama-Qwen3:{size}"
        self.modelsize = size
        self.endpoint = endpoint

    def extract_phenotypes(self, content: str) -> list[str]:
        try:
            response = requests.post(
                url= self.endpoint,
                json={
                    "model": f"qwen3:{self.modelsize}",
                    "prompt": f"You are a medical expert specialized in human phenotypes. I will provide you with a medical record. Your task is to extract all of the patient’s abnormal phenotypes and symptoms, including anatomical anomalies, clinical symptoms, diagnostic findings, test results, and specific conditions or syndromes, and convert them into professional medical terminology. Please completely ignore negative findings, normal findings, procedures and family history. Please output the phenotypes with '@' symbols at intervals. Be careful not to output any text other than the above. Below is the medical record:\n {content}",
                    "stream": False,
                    # "options": {"temperature": 0.7}
                    },
                timeout=240
                )
            response.raise_for_status()
            rm_think = re.sub(r'<think>.*?</think>\s*', '', response.json()["response"], flags=re.DOTALL).split("@")
            return [pheno for pheno in rm_think if pheno]
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")


class LlamaClient(LLMClient):
    def __init__(self, endpoint: str ="http://localhost:11433/api/generate", model="llama3.3:70b"):
        assert model in {"llama3.3:70b", "llama3.1:8b", "llama3.1:70b"}, "size must be 'llama3.3:70b', 'llama3.1:8b' or 'llama3.1:70b'"
        self.llm_name = f"Ollama-{model}" # 用于存储在LEAP_Result中的模型名
        self.modelname = model # Ollama的标准模型名
        self.endpoint = endpoint

    def extract_phenotypes(self, content: str) -> list[str]:
        try:
            response = requests.post(
                url= self.endpoint,
                json={
                    "model": self.modelname,
                    "prompt": f"You are a medical expert specialized in human phenotypes. I will provide you with a medical record. Your task is to extract all of the patient’s abnormal phenotypes and symptoms, including anatomical anomalies, clinical symptoms, diagnostic findings, test results, and specific conditions or syndromes, and convert them into professional medical terminology. Please completely ignore negative findings, normal findings, procedures and family history. Please output the phenotypes with '@' symbols at intervals. Be careful not to output any text other than the above. Below is the medical record:\n {content}",
                    "stream": False,
                    # "options": {"temperature": 0.7}
                    },
                timeout=240
                )
            response.raise_for_status()
            return [item.strip() for item in response.json()["response"].split("@") if item.strip()]
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")


# 新增：根据模型名自动实例化对应的 client
# 暂时还没开始用这个
# def get_llm_client(model_name: str, **kwargs) -> LLMClient:
#     """
#     根据模型名自动实例化对应的 LLMClient。
#     model_name: 支持 "openai-gpt-4o", "gemma3:27b" 等
#     kwargs: 传递给 client 的参数，如 api_key, endpoint 等
#     """
#     model_name = model_name.lower()
#     if model_name in ["openai-gpt-4o", "gpt-4o"]:
#         return OpenAIClient(api_key=kwargs.get("api_key"))
#     elif model_name in ["ollama-gemma3:27b", "gemma3:27b"]:
#         return Gemma3Client(endpoint=kwargs.get("endpoint", "http://localhost:11433/api/generate"))
#     else:
#         raise ValueError(f"Unsupported model name: {model_name}")
