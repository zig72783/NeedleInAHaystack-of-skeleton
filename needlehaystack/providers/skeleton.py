import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration, pipeline
#import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
def Text_context(query, embeddings, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50,)
        docs = text_splitter.split_documents(text)
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.similarity_search(query , k = 3)
        context = "\n".join([x.page_content for x in retriever])
        return context

class TextChunker:
    def __init__(self, max_chunk_length, chunk_overlap, min_chunk_length=1):
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def chunk_text(self, text):
        chunks = []
        start = 0
        text_length = len(text)     
        while start < text_length:
            end = min(start + self.max_chunk_length, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == text_length:
                break
            start += self.max_chunk_length - self.chunk_overlap      
        # 检查最后一个 chunk 是否小于 min_chunk_length，并决定是否合并
        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_length:
            last_chunk = chunks.pop()
            chunks[-1] += last_chunk
        return chunks

class SummaryGenerator:
    def __init__(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to('cuda')
        self.max_summary_length = 150  # 最大摘要长度 之前500

    def generate_summary(self, chunk):
        prompt = "<s>摘要：</s>\n\n" + chunk  # 添加生成摘要的提示
        max_length = self.tokenizer.model_max_length if self.tokenizer.model_max_length > 0 else 512 # 获取模型的最大长度，如果没有预定义最大长度，设置为默认值512
        # 使用 tokenizer 对带有 prompt 的 chunk 进行编码，并指定 max_length
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to('cuda')
        
        # 提取 input_ids 和 attention_mask
        #input_ids = inputs['input_ids'].to('cuda')
        #attention_mask = inputs['attention_mask'].to('cuda')
        input_ids = self.tokenizer.encode(inputs,return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=device)

        summary_ids = self.model.generate(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # 确保提供 attention_mask
            max_length=self.max_summary_length,
            min_length=10,
            length_penalty=1.0,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id  # 显式指定 pad_token_id
        )

        ## 生成摘要
        #summary_ids = self.model.generate(
        #    inputs.input_ids,
        #    max_length=self.max_summary_length,
        #    min_length=10,
        #    length_penalty=1.0,
        #    num_beams=4,
        #    no_repeat_ngram_size=3,  # 避免重复生成相同的内容
        #    early_stopping=True
        #)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True) # 解码生成的 token IDs 为文本
        return summary

class Skeleton:
    def __init__(self, llm, summary_model_name, chunk_size, num_layer):
        self.chunk_size = chunk_size
        self.num_layer = num_layer
        self.summary_generator = SummaryGenerator(summary_model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")
        self.vector_database = {}
        self.layer_vector_databases = {}
        self.chunker = TextChunker(max_chunk_length=500, chunk_overlap=20)

    def process_and_summarize(self, chunks):
        summaries = self.generate_summaries(chunks)
        return summaries

    def tokenize_chunks(self, chunks):
        token_ids = []
        for chunk in chunks:
            ids = self.tokenizer.encode(chunk, truncation=True, padding='max_length', max_length=self.chunk_size)
            token_ids.append(ids)
        return token_ids

    def generate_summaries(self, chunks):
        summary = []
        summaries = []
        for chunk in chunks:
            summary = self.summary_generator.generate_summary(chunk)
            summaries.append(summary)
        return summaries

    def build_layers(self, summaries): 
        layer_vector_databases = {}
        model_name = "sentence-transformers/sentence-t5-large"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)    
        for j in range(self.num_layer):
            chunks = self.chunker.chunk_text(''.join(summaries))
            #summaries = self.summary_generator.generate_summary(chunks)
            summaries = self.generate_summaries(chunks)     
            #layer_summary = ''.join(summaries)                    # 检查是否实现分层summary
            #print(f"   ***Layer {j+1}***:\n {layer_summary}")     # 检查是否实现分层summary          
            vector_database = FAISS.from_texts(summaries, embeddings)
            self.layer_vector_databases[j] = vector_database
            #vector = embeddings.embed_query(summaries[0])                 # 获取摘要的嵌入向量
            #print(f"Vector_database {j+1} - Vector shape: {len(vector)}") # 获取摘要的嵌入向量和向量维度
        return self.layer_vector_databases

    def retrieve_from_layers(self, query, k=3):
        all_retrieved_summaries = {}
        for j in range(self.num_layer - 1, -1, -1):
            vector_database = self.layer_vector_databases[j]
            retrieved_summaries = vector_database.similarity_search(query, k=k)
            # 检查每层的检索结果
            #print(f"*** Layer {j+1} retrieved summaries: ***")
            #for idx, summary in enumerate(retrieved_summaries):
            #    print(f"  Summary {idx+1}: {summary.page_content}")
            concatenated_content = ' '.join(
                doc.page_content for doc in retrieved_summaries if hasattr(doc, 'page_content') and doc.page_content
            )
            all_retrieved_summaries[f'layer_{j+1}'] = concatenated_content
        return all_retrieved_summaries

def LLM_inference(llm, all_retrieved_summaries, question, context):
    layer_3_summaries = all_retrieved_summaries['layer_3']
    layer_2_summaries = all_retrieved_summaries['layer_2']
    layer_1_summaries = all_retrieved_summaries['layer_1']

    prompt = """
    <|system|>
    You are an intelligent AI assistant skilled in answering user questions. 
    You obtain three levels of summaries by the way of hierarchical summarization, 
    and three summary blocks are selected for each layer as contexts. Layer 3 represents 
    the general summary. Layer 2 represents the sub-summary. Layer 1 represents the 
    detailed summary. Context is considered to be the original text of the summary of 
    details. Please fully consider the relationship between the different levels of summary and context, 
    and the granularity of the information they represent. Then use these summaries and context below, 
    answer the question.Please keep your answers concise and clear. 
    Do not talk about irrelevant topic sorrepe at your answers.
    Summaries:
        Layer 3 retrieved summaries: {layer_3_summaries}
        Layer 2 retrieved summaries: {layer_2_summaries}
        Layer 1 retrieved summaries: {layer_1_summaries}
    Context: {context}
    </s>
    <|user|>
    question: {question}"""

    prompt_template = PromptTemplate(template=prompt, input_variables=["layer_3_summaries", 
    "layer_2_summaries", "layer_1_summaries", "context", "question"], )
    #llm_chain = LLMChain(prompt=prompt_template, llm=llm, llm_kwargs = {"temperature":0, "max_length":1024})
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    
    input_data = {
        "layer_1_summaries": layer_1_summaries,
        "layer_2_summaries": layer_2_summaries,
        "layer_3_summaries": layer_3_summaries,
        "context": context,
        "question": question
    }

    resulte = llm_chain.run(input_data)
    #print(f"     ###The answer is: {resulte}")
    return resulte

#query = "What is HMT?"
#query = "What are the advantages of HMT?"

# 输出摘要
#for i, summary in enumerate(summaries):
    #print(f"Summary {i+1}: {summary}")
# 打印最终检索到的摘要
#for idx, summary in enumerate(all_retrieved_summaries):
#    print(f"Final Summary {idx+1}: {summary.page_content}")