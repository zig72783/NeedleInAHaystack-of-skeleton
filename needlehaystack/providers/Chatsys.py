import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration, AutoModel,pipeline, TextStreamer
import logging
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

from langchain_community.document_loaders import TextLoader
#from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from operator import itemgetter
from typing import Optional

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tiktoken

from .model import ModelProvider
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from .skeleton import TextChunker, SummaryGenerator, Skeleton

#llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
#llm_model = AutoModelForCausalLM.from_pretrained(
#        "Qwen/Qwen2-7B",
#        device_map="auto",
#        torch_dtype='auto'
#    )
#pipe = pipeline(
#        "text-generation",
#        model=llm_model,
#        tokenizer=llm_tokenizer,
#        max_new_tokens=512,
#        temperature=0
#    )
#llm = HuggingFacePipeline(pipeline=pipe)

#model_name = "sentence-transformers/sentence-t5-large"

#embeddings = HuggingFaceEmbeddings(model_name=model_name)

#skeleton = Skeleton(llm, "facebook/bart-large-cnn", 512, 3)

device_map = 'auto'
device = "cuda"
# device_gpu = "cuda:0"
# device_cpu = "cpu"
torch_dtype = torch.float16

class Chatsys(ModelProvider):

    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 512,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "Qwen/Qwen2-7B",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS, device='cuda'):
        
        # api_key = os.getenv('NIAH_MODEL_API_KEY')
        # if (not api_key):
        #     raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = None
        model_path = "Qwen/Qwen2-7B"
        

        self.llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-7B",
                device_map="auto",
                torch_dtype='auto'
            )
        pipe = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                max_new_tokens=512,
                temperature=0
            )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        e_model_name = "sentence-transformers/sentence-t5-large"

        self.skeleton = Skeleton(self.llm, "facebook/bart-large-cnn", 512, 3)
        self.embeddings = HuggingFaceEmbeddings(model_name=e_model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map=device_map,
                                                          torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    #@abstractmethod
    async def evaluate_model(self, prompt: str) -> str: #...
        # 使用 await 等待异步生成结果
        text = self.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True, 
            **self.model_kwargs
            )
    
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
    
        # 生成 attention mask
        attention_mask = (model_inputs.input_ids != self.tokenizer.pad_token_id).long()
    
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id  # 设置 pad_token_id
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


    #@abstractmethod
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]: #...

        #loader = TextLoader(Document(page_content=context))
        pages=[Document(page_content=context)]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50,)

        #pages = loader.load()

        docs = text_splitter.split_documents(pages)

        db = FAISS.from_documents(docs, self.embeddings)

        retriever = db.similarity_search(retrieval_question , k = 3)

        re_context = "\n".join([x.page_content for x in retriever])

        chunker = TextChunker(max_chunk_length=500, chunk_overlap=20) 

        chunks = chunker.chunk_text(context) 

        summaries = self.skeleton.process_and_summarize(chunks) 

        layer_vector_databases = self.skeleton.build_layers(summaries) 

        all_retrieved_summaries = self.skeleton.retrieve_from_layers(retrieval_question, k=3)

        layer_3_summaries = all_retrieved_summaries['layer_3']

        layer_2_summaries = all_retrieved_summaries['layer_2']
        
        layer_1_summaries = all_retrieved_summaries['layer_1']

        return [{
                "role": "system",
                "content": "You are an intelligent AI assistant skilled in answering user questions. \
                You obtain three levels of summaries by the way of hierarchical summarization, and \
                three summary blocks are selected for each layer as contexts. Layer 3 represents the \
                general summary. Layer 2 represents the sub-summary. Layer 1 represents the detailed\
                summary. Context is considered to be the original text of the summary of details. \
                Please fully consider the relationship between the different levels of summary and \
                context, and the granularity of the information they represent. Then use these summaries \
                and context below, answer the question.Please keep your answers concise and clear.\
                Do not talk about irrelevant topic sorrepe at your answers.\
                Summaries:\
                    Layer 3 retrieved summaries: {layer_3_summaries}\
                    Layer 2 retrieved summaries: {layer_2_summaries}\
                    Layer 1 retrieved summaries: {layer_1_summaries}\
                Context: {re_context}"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]

    #@abstractmethod
    def encode_text_to_tokens(self, text: str) -> list[int]: #...
        # 检查输入类型，如果是list则拼接为str
        #if isinstance(text, list):
        #    text = " ".join(text)  # 使用空格连接列表中的字符串
        #return self.tokenizer(text)
        return self.tokenizer.encode(text)

    #def encode_text_to_tokens(self, text: str) -> list[list[int]]:

        #chunk_size = 30000  # 设置每个块的最大长度
        #tokens = []

        #for i in range(0, len(text), chunk_size):
        #    chunk = text[i:i + chunk_size]
        #    tokens.append(self.tokenizer.encode(chunk, truncation=True))
        #return tokens

    #@abstractmethod
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str: #...

        return self.tokenizer.decode(tokens[:context_length])

    #def decode_tokens(self, tokens, context_length=None):
        # 如果 tokens 是 torch.Tensor，则转换为列表
        #if isinstance(tokens, torch.Tensor):
        #    tokens = tokens.tolist()
    
        # 确保 tokens 是列表或元组
        #if not isinstance(tokens, (list, tuple)):
        #    raise TypeError(f"Expected tokens to be a list or tuple, but got {type(tokens)}")
    
        # 如果 context_length 被指定，则切片
        #if context_length is not None:
        #    tokens = tokens[:context_length]
    
        #return self.tokenizer.decode(tokens)
        #return self.tokenizer.decode(tokens[:context_length])

        # 如果 tokens 是 BatchEncoding 对象，提取出 token IDs
        #if isinstance(tokens, tokenizers.BatchEncoding):  # 使用正确的类名
        #tokens = tokens['input_ids']  # 假设 'input_ids' 存储的是 token IDs

        #if not isinstance(tokens, (list, tuple)):
        #    raise TypeError(f"Expected tokens to be a list or tuple, but got {type(tokens)}")

        #return self.tokenizer.decode(tokens)  # 确保这里返回的是字符串

    #def decode_tokens(self, token_lists: list[list[int]]) -> str:
        # 逐块解码并合并结果
    #    decoded_text = []
    #    for tokens in token_lists:
    #        decoded_text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
    #    return ' '.join(decoded_text)  # 合并成一个完整的文本