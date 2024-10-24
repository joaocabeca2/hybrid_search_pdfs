import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import lancedb
from lancedb.rerankers import LinearCombinationReranker
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import pandas as pd
import re
from unidecode import unidecode
import unicodedata
from getpass import getpass
from dotenv import load_dotenv

def create_index_chunks(table):
    start_index = [indice for indice, _ in enumerate(table['context_text'])]
    table['index'] = start_index
    return table

def preprocessar_texto(texto):
    # Normaliza o texto para lidar com caracteres Unicode
    texto = unicodedata.normalize('NFKD', texto)
    
    # Remove quebras de linha extras que possam estar fragmentando o texto
    texto = re.sub(r'\n+', ' ', texto)

    # Remove quebras de página visíveis
    texto = texto.replace('\f', '')

    # Remove tabulações e caracteres não ASCII
    texto = texto.replace('\t', ' ')

    # Remove hifens de quebra de linha e une as palavras
    texto = re.sub(r'-\s+', '', texto)

    # Remove espaços antes de pontuação
    texto = re.sub(r'\s+([.,;?!])', r'\1', texto)

    # Remove acentos
    texto = unidecode(texto)

    # Regex para remover sequências de 2 ou mais underscores (traços)
    texto = re.sub(r'_+', '', texto)

    # Remover numerações de marcação ao iniciar uma nova linha
    texto = re.sub(r'\d+(\.\d+)+\s*', '', texto)
    
    # Remove espaços duplos resultantes da remoção de quebras de linha
    texto = re.sub(r'\s{2,}', '', texto)

    return texto.strip().lower()
    
def create_full_text_search_index(table):
    try:
        table.create_fts_index(['text'], replace=True)
        return table
    except ValueError as e:
         print(f'Não foi possível criar o indice fts: {e}')
    except Exception as e :
        print(f'Erro desconhecido: {e}')

def semantic_search(query, table, reranker, k=4):
    try:
        result = table.search(query, query_type='hybrid', vector_column_name='vector').rerank(reranker=reranker).limit(4)
        return result
    except Exception as e:
        raise Exception(f'Não foi possível realizar a busca hibrida: {e}')

def create_embedding_func(model_name, api_key):
    try:
        embed_func = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        return embed_func
    except Exception as e:
        raise Exception(f'Não foi possível criar a função de embeddings: {e}')

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    return text_splitter.split_text(text)

def create_lance_table(chunks, table_name, df, schema):
    db = lancedb.connect("~/langchain")
    table = db.create_table(
        table_name,
        schema=schema,
        mode="overwrite",
    )

    #table = db.open_table(table_name)
    table.add(df)
    return table

def create_genai_llm(api_key):
    return ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.2, google_api_key=api_key)


def main():
    #Estabelecendo algumas configurações
    load_dotenv()
    
    #Estabelecendo configurações iniciais
    if "GOOGLE_API_KEY" not in os.environ:
         os.environ["GOOGLE_API_KEY"] = getpass("AIzaSyDfzFvmb_TxWXrncVJQRtln9D6NZmamcZE")
         
    model = get_registry().get("gemini-text").create(name="models/embedding-001")

    class Schema(LanceModel):
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()
        page: int
        index: int 
    
    """ Primeira tarefa - Você receberá um arquivo “Teste_Q&A.csv”, as colunas nesse arquivo 
    serão: contexto, pergunta, Classe_verdadeira, Classe_predita. Sua anotação deve seguir a 
    seguinte lógica: existem dois tipos de Classe_verdadeira, 0 e 1, nas linhas marcadas com 1 
    na Classe_verdadeira, você deve ler a pergunta e verificar se a resposta na mesma linha 
    está dentro do contexto fornecido nessa linha, de forma que nenhuma informação 
    contida na resposta possa ser de fora do contexto, ou seja, apenas informações que estão 
    dentro do contexto podem ser encontradas na resposta. Caso esse padrão aconteça, 
    marque a coluna Classe_predita com 1, caso esse padrão não aconteça marque com 0.
    O segundo tipo de Classe_verdadeira é o 0, nesse caso você deve olhar a resposta e 
    verificar se a resposta indica que não foi possível responder a pergunta, exemplo: “Meu 
    conhecimento é restrito ao contexto fornecido, e não posso fornecer informações além 
    disso. Perdão pela limitação.”, nesse caso marque a coluna Classe_predita como 0. Se a 
    resposta conter qualquer informação que tenta responder a pergunta, estando certo ou 
    não, estando dentro do contexto ou não, marque como 1."""

    print('\nLendo o arquivo...')
    df = pd.read_csv('teste_qa_portugus.csv', encoding='utf8')
    for _, linha in df.iterrows():
        context_text = linha['contexto']
        chunks = create_chunks(context_text)

        #Criando um dataframe com os dados dos chunks
        df = pd.DataFrame({'context_text': chunks,})
        df['context_text'] = df['context_text'].apply(preprocessar_texto)
        table_doc = create_index_chunks(df)
        #table_doc.head()
    
    print('\nCriando uma tabela lancedb e estabelecendo um indice para a full-text-search...')
    table = create_lance_table(chunks, 'lancetb', df, schema=Schema)

    create_full_text_search_index(table)
    query = ' é possível mapear os principios da lgpd com os princípios éticos no contexto de inteligência artificial?'
    reranker = LinearCombinationReranker()
    
    print('\nRealizando a busca hibrida...')
    result_4_df = semantic_search(query, table, reranker).to_pandas()

    #Para cada um dos 4 resultados pegar o chunk anterior e o posterior e truncar com o chunk encontrado
    semantic_texts = []
    for index in result_4_df['index']:
        semantic_chunk = table_doc.loc[table_doc['index'] == index]['text'][index]
        if index == (table_doc.shape[0] - 1):
            semantic_chunk_pos = ''
        else:
            semantic_chunk_pos = table_doc.loc[table_doc['index'] == (index+1)]['text'][index+1]
        if index != 0:
            semantic_chunk_ant = table_doc.loc[table_doc['index'] == (index-1)]['text'][index-1]
        else:
            semantic_chunk_ant = ''

        text = semantic_chunk_ant + semantic_chunk + semantic_chunk_pos
        semantic_texts.append(text)

        print(index, text)
    gemini_responses = []
    model = create_genai_llm(os.getenv('GOOGLE_API_KEY'))
    for semantic_text in semantic_texts:
        prompt = f"""
        Você é um especialista em análise semantica. Analise o texto abaixo e verifique se o mesmo responde a 
        seguinte pergunta: {query}\n
        texto: {semantic_text}.
        Se o chunk não responder a pergunta apenas diga: 'O texto não responde a pergunta'
        """
        response = model.invoke(prompt)
        gemini_responses.append(response.content)

    df = pd.DataFrame({'semantic_chunks': semantic_texts, 'gemini_responses': gemini_responses})
    df.to_excel('hybrid-search.xlsx')

if __name__ == '__main__':
    main()
