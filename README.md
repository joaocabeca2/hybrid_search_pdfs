No meu estágio no Tribunal de Contas da União estamos trabalhando com análise de documentos jurídicos. Com isso, para me aperfeiçoar um pouco nas ferramentas langchain e lancedb fiz um projeto semelhante para a realização de uma busca hibrida em um trabalho realizado por mim e mais dois colegas na disciplina de Computação Experimental na UnB. o artigo "É possível mapear os princípios da LGPD com os princípios éticos no contexto de Inteligência Artificial?".



Langchain é uma biblioteca projetada para facilitar a construção de aplicações que utilizam modelos de linguagem, como GPT-3 e similares. Ela fornece ferramentas e abstrações para integrar esses modelos de forma eficiente em fluxos de trabalho mais complexos, como consultas baseadas em linguagem natural.



LanceDB é um banco de dados otimizado para armazenamento, indexação e consulta de vetores de embeddings. Ele é especialmente útil em aplicações de machine learning e inteligência artificial, como a busca semântica.

O modelo Gemini é utilizado para gerar embeddings, que são representações numéricas dos textos, permitindo que as consultas sejam comparadas de forma semântica aos documentos armazenados no banco de dados.



Busca semântica: Utilizando embeddings gerados com o modelo Gemini, permitimos que a consulta busque por similaridade de significados, encontrando documentos ou trechos que sejam semanticamente próximos ao título da query.



Busca lexical (regex): Além da busca semântica, integramos uma busca por padrões lexicais específicos, utilizando expressões regulares (regex) para encontrar correspondências exatas ou padrões textuais dentro do documento.
