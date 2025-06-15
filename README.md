README

# Lexperito - Professor de Direito Digital

## Descrição

**Lexperito** é um assistente de IA desenvolvido com Streamlit, projetado para auxiliar no ensino e capacitação em provas digitais e forense computacional. Ele utiliza modelos de linguagem avançados (como GPT-4o) e técnicas de RAG (Retrieval-Augmented Generation) com ChromaDB para fornecer respostas baseadas em uma base de conhecimento de documentos relevantes (por exemplo, normas técnicas, legislação, artigos, relatórios de perícia).

O sistema permite aos usuários fazer perguntas em linguagem natural, anexar documentos temporariamente para análise contextual e fornecer feedback sobre as respostas, que pode ser incorporado à base de conhecimento para melhorias futuras.

## Funcionalidades Principais

* **Consulta em Linguagem Natural:** Faça perguntas sobre direito digital, forense computacional, procedimentos de perícia, ferramentas, etc.
* **Respostas Baseadas em RAG:** O sistema busca informações relevantes em uma base de documentos indexados (arquivos TXT, preferencialmente de OCR) para fundamentar as respostas.
* **Seleção de Modelo de IA:** Escolha entre diferentes modelos da OpenAI (GPT-4o, GPT-4 Turbo, etc.) para gerar as respostas.
* **Parâmetros Configuráveis:** Ajuste a "criatividade" (temperatura) e o tamanho máximo da resposta.
* **Anexos Temporários:** Envie arquivos PDF, DOCX ou TXT para serem considerados *apenas* na consulta atual, sem indexação permanente.
* **Indexação de Documentos:** Indexe arquivos de texto (.txt) de uma pasta específica (geralmente contendo resultados de OCR) para construir ou atualizar a base de conhecimento RAG.
* **Gerenciamento da Base RAG:**
    Liste os arquivos fontes indexados.
    Diagnostique o estado da base de dados vetorial (ChromaDB).
    Repare/Recrie a base de dados (ação destrutiva!).
* **Sistema de Feedback:**
    Avalie a qualidade das respostas geradas (Ótima a Inútil).
    Adicione comentários e análises complementares, que podem ser indexados no RAG.
    Anexe arquivos para justificar o feedback, cujo conteúdo também pode ser indexado.
    Visualize o histórico de feedbacks salvos em um arquivo CSV.
* **Histórico de Interação:** As perguntas, respostas e fontes RAG consultadas são salvas em um arquivo de log (`historico_respostas.txt`).
* **Inicialização com Exemplos:** Popula uma base de dados vazia com exemplos de documentos sobre forense e direito digital.
* **Teste de API:** Verifica a conexão com a API da OpenAI.

## Requisitos e Dependências

O código utiliza as seguintes bibliotecas Python principais (instale via `pip`):

```bash
pip install streamlit langchain langchain-community langchain-openai openai chromadb pypdf2 python-docx pandas tiktoken chardet unicodedata2