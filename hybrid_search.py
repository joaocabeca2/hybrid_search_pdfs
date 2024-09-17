import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import lancedb
from langchain_community.vectorstores import LanceDB
from lancedb.rerankers import LinearCombinationReranker
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

embed_func = get_registry().get("sentence-transformers").create(device="cpu")
class Schema(LanceModel):
    text: str = embed_func.SourceField()
    vector: Vector(embed_func.ndims()) = embed_func.VectorField()
    start_index: int 

def create_index_chunks(table):
    start_index = [indice for indice, _ in enumerate(table['metadata'])]
    table['start_index'] = start_index
    table = table.drop('metadata', axis=1)
    return table

def create_full_text_search_index(table):
    try:
        table.create_fts_index(['text'], replace=True)
        return table
    except Exception as e:
        raise f'Não foi possível criar o indice fts: {e}'

def semantic_search(query, table, reranker, k=4):
    try:
        result = table.search(query, query_type='hybrid', vector_column_name='vector').rerank(reranker=reranker).limit(4)
        return result
    except Exception as e:
        raise f'Não foi possível realizar a busca hibrida: {e}'

def create_embedding_func(model_name, model_kwargs, encode_kwargs):
    try:
        embed_func = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        return embed_func
    except Exception as e:
        raise f'Não foi possível criar a função de embeddings: {e}'

def read_file(path_file):
    if path_file.endswith('.pdf'):
        textLoader = PyPDFLoader(path_file)
    return textLoader.load()

def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0, add_start_index=True)
    return text_splitter.split_documents(docs)

def create_lance_table(docsearch, table_name, table_doc):
    db = lancedb.connect("~/langchain")
    table = db.create_table(
        table_name,
        schema=Schema,
        mode="overwrite",
    )

    table = db.open_table('test_table')
    table_doc = create_index_chunks(table_doc)
    table.add(table_doc)
    return table

def create_modelai(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={'temperature': 0.2})
    return model

def main():
    #Estabelecendo configurações iniciais
    load_dotenv()
    api_gemini = os.getenv('API_KEY')

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    path = os.path.join(os.path.dirname((__file__)), 'inputs')
    editais = os.listdir(path)
    edital = os.path.join(path, editais[0])

    #Criando a função de embedding
    embeddings = create_embedding_func('sentence-transformers/all-MiniLM-L6-v2', model_kwargs, encode_kwargs)

    #Lendo o arquivo e transformando em um documento langchain
    docs = read_file(edital)

    #Criando os chunks
    chunks = create_chunks(docs)
    
    #Tranformando os chunks em vetores de embeddigns
    docsearch = LanceDB.from_documents(chunks, embeddings)

    table_doc = docsearch.get_table().to_pandas()
    #Criando uma instancia lance para armazenar os dados do docsearch
    table = create_lance_table(docsearch, 'lancetb', table_doc)

    #Criar um indice para a busca full text
    if create_full_text_search_index(table):
        query = 'quais os sintomas do tremor essencial?'
        reranker = LinearCombinationReranker(weight=0.5)

        print(f'Realizando a busca hibrida com ranqueamento dos resultados: {query}')
        result_4_df = semantic_search(query, table, reranker).to_pandas()
        
        #Para cada um dos 4 resultados pegar o chunk anterior e o posterior e truncar com o chunk encontrado
        semantic_texts = []
        for index in result_4_df['start_index']:
            semantic_chunk = table_doc.loc[table_doc['start_index'] == index]['text'][index]
            if index == (table_doc.shape[0] - 1):
                semantic_chunk_pos = ''
            else:
                semantic_chunk_pos = table_doc.loc[table_doc['start_index'] == (index+1)]['text'][index+1]
            if index != 0:
                semantic_chunk_ant = table_doc.loc[table_doc['start_index'] == (index-1)]['text'][index-1]
            else:
                semantic_chunk_ant = ''

            text = semantic_chunk_ant + semantic_chunk + semantic_chunk_pos
            semantic_texts.append(text)
        
        #Criando instancia no gemini
        gemini_responses = []
        model = create_modelai(api_gemini)
        for semantic_text in semantic_texts:
            prompt = f""""
            Você é um especialista em análise semantica. Analise o texto abaixo e verifique se o mesmo responde a 
            seguinte pergunta: {query}\n
            texto: {semantic_text}
            """
            response = model.generate_content(prompt)
            gemini_responses.append(response.text)

        df = pd.DataFrame({'semantic_chunks': semantic_texts, 'gemini_responses': gemini_responses})
        print()
if __name__ == '__main__':
    main()