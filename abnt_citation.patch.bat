--- lexperito.py
+++ lexperito.py
@@ -3,6 +3,7 @@
 import streamlit as st
 import PyPDF2
 from io import StringIO
+import re
 import datetime
 import pandas as pd
 from docx import Document
@@ -80,6 +81,84 @@ def carregar_arquivo(path):
         st.warning(f"Formato não suportado: {ext}")
         return [{"page_content": f"[Formato não suportado: {ext}]", "metadata": {"source": path}}]

+# Nova função para extrair metadados ABNT
+def extrair_metadados_abnt(filepath, content=None):
+    """
+    Extrai ou infere metadados para citação ABNT de um arquivo.
+    
+    Args:
+        filepath: Caminho completo do arquivo
+        content: Conteúdo do arquivo, se disponível (opcional)
+    
+    Returns:
+        Dicionário com metadados no formato ABNT (autor, título, ano, etc.)
+    """
+    metadata = {
+        "autor": "",
+        "titulo": "",
+        "ano": "",
+        "local": "",
+        "editora": ""
+    }
+    
+    # Obter nome base do arquivo e extensão
+    basename = os.path.basename(filepath)
+    filename_noext = os.path.splitext(basename)[0]
+    ext = os.path.splitext(basename)[1].lower()
+    
+    # 1. Tentar extrair metadados do conteúdo, se disponível
+    if content:
+        # Para PDFs, tentar extrair metadados do documento
+        if ext == ".pdf" and isinstance(content, PyPDF2.PdfReader):
+            try:
+                info = content.metadata
+                if info:
+                    metadata["autor"] = info.get('/Author', "")
+                    metadata["titulo"] = info.get('/Title', "")
+                    # Extrair ano da data de criação ou modificação
+                    created_date = info.get('/CreationDate', "")
+                    if created_date and 'D:' in created_date:
+                        year_match = re.search(r'D:(\d{4})', created_date)
+                        if year_match:
+                            metadata["ano"] = year_match.group(1)
+            except:
+                pass  # Falha silenciosa, continuamos com outras estratégias
+                
+        # Para DOCX, tentar extrair propriedades do documento
+        elif ext == ".docx" and isinstance(content, Document):
+            try:
+                core_props = content.core_properties
+                metadata["autor"] = core_props.author or ""
+                metadata["titulo"] = core_props.title or ""
+                if core_props.created:
+                    metadata["ano"] = str(core_props.created.year)
+            except:
+                pass  # Falha silenciosa
+    
+    # 2. Se ainda não temos metadados suficientes, inferir do nome do arquivo
+    if not metadata["autor"] or not metadata["titulo"]:
+        # Verificar se o nome do arquivo segue padrão "Autor - Título"
+        parts = filename_noext.split(" - ", 1)
+        if len(parts) == 2:
+            if not metadata["autor"]:
+                metadata["autor"] = parts[0].strip().title()
+            if not metadata["titulo"]:
+                metadata["titulo"] = parts[1].strip().title()
+        else:
+            # Se não segue o padrão, usar nome do arquivo como título
+            if not metadata["titulo"]:
+                metadata["titulo"] = filename_noext.replace("_", " ").title()
+    
+    # 3. Se ainda não temos ano, usar a data de modificação do arquivo
+    if not metadata["ano"] and os.path.exists(filepath):
+        try:
+            mtime = os.path.getmtime(filepath)
+            metadata["ano"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y")
+        except:
+            metadata["ano"] = "s.d."  # sem data
+    
+    return metadata
+
+# Nova função para formatar referência ABNT
+def formatar_referencia_abnt(metadata):
+    """
+    Formata uma referência bibliográfica no padrão ABNT.
+    
+    Args:
+        metadata: Dicionário com metadados do documento
+    
+    Returns:
+        String com a referência formatada no padrão ABNT
+    """
+    source = metadata.get('source', '')
+    autor = metadata.get('autor', '')
+    titulo = metadata.get('titulo', '')
+    ano = metadata.get('ano', '')
+    local = metadata.get('local', '')
+    editora = metadata.get('editora', '')
+    
+    # Identificar tipo de documento especial
+    if "analise_complementar" in source.lower() or metadata.get('tipo') == 'feedback':
+        return f"ANÁLISE COMPLEMENTAR. {titulo}. {ano}."
+    
+    # Para autores, usar formato SOBRENOME, Nome.
+    if autor:
+        # Converter para maiúsculas apenas o último nome (sobrenome)
+        partes_nome = autor.split()
+        if len(partes_nome) > 1:
+            # Último nome em maiúsculas + vírgula + restante do nome
+            autor_formatado = f"{partes_nome[-1].upper()}, {' '.join(partes_nome[:-1])}"
+        else:
+            autor_formatado = autor.upper()
+    else:
+        autor_formatado = "AUTOR DESCONHECIDO"
+    
+    # Construir referência no formato ABNT
+    referencia = f"{autor_formatado}. {titulo}"
+    if ano:
+        referencia += f". {ano}"
+    if local and editora:
+        referencia += f". {local}: {editora}"
+    
+    referencia += "."
+    return referencia
+
 # FUNÇÃO CORRIGIDA: Splitter e vetorização
 def dividir_documentos(docs):
     """
@@ -392,20 +471,32 @@ def main():
 
                 # Construir contexto
                 contexto_docs = ""
+                referencias_abnt = []
+                
                 for i, doc in enumerate(docs_combinados):
-                    fonte_origem = os.path.basename(doc.metadata.get('source', 'desconhecido'))
-                    # Renomear fontes de feedback/análise complementar
-                    if "analise_complementar" in fonte_origem or doc.metadata.get('tipo') == 'feedback':
-                         fonte_display = f"Análise Complementar Relacionada #{i+1}"
-                    else:
-                         fonte_display = fonte_origem
-                    contexto_docs += f"\n--- Trecho {i+1} (Fonte: {fonte_display}) ---\n{doc.page_content}\n---\n"
+                    # Adicionar metadados ABNT ao documento se não existirem
+                    if not doc.metadata.get('abnt_metadata'):
+                        # Extrair metadados ABNT básicos
+                        source_path = doc.metadata.get('source', 'desconhecido')
+                        abnt_metadata = extrair_metadados_abnt(source_path)
+                        
+                        # Mesclar com metadados existentes
+                        for key, value in abnt_metadata.items():
+                            if key not in doc.metadata:
+                                doc.metadata[key] = value
+                    
+                    # Formatar referência ABNT
+                    ref_abnt = formatar_referencia_abnt(doc.metadata)
+                    ref_id = i + 1  # Índice da referência, começando em 1
+                    
+                    # Adicionar à lista de referências (evitar duplicatas)
+                    if ref_abnt not in referencias_abnt:
+                        referencias_abnt.append(ref_abnt)
+                    
+                    # Adicionar ao contexto com o identificador da referência
+                    contexto_docs += f"\n--- Documento [{ref_id}] ---\n{doc.page_content}\n---\n"
 
-                # Construir prompt final
-                prompt = f"""
-                Você é um assistente especializado em analisar provas jurídicas digitais e normas técnicas relacionadas. Analise criticamente a conformidade de métodos e resultados com normas aplicáveis, focando na auditabilidade e no contraditório judicial. Baseie-se EXCLUSIVAMENTE nas informações fornecidas abaixo.
-
-                ### Informações Disponíveis (Trechos de Documentos e Análises) ###
+                # Formatar lista de referências para o prompt
+                referencias_formatadas = "\n".join([f"{i+1}. {ref}" for i, ref in enumerate(referencias_abnt)])
+                
+                # Construir prompt com instruções ABNT
+                prompt = f"""
+                Você é um assistente especializado em analisar provas jurídicas digitais e normas técnicas relacionadas. Analise criticamente a conformidade de métodos e resultados com normas aplicáveis, focando na auditabilidade e no contraditório judicial. Baseie-se EXCLUSIVAMENTE nas informações fornecidas abaixo.
+
+                ### Informações Disponíveis (Documentos e Análises) ###
                 {contexto_docs}
                 ################################################################
 
                 Pergunta do Usuário: {query}
+                
+                ### REFERÊNCIAS BIBLIOGRÁFICAS ###
+                {referencias_formatadas}
+                ################################################################
 
                 ### INSTRUÇÕES DETALHADAS PARA RESPOSTA ###
                 1.  **Análise Técnica e Crítica:** Forneça uma resposta técnica, detalhada e crítica, abordando diretamente a pergunta do usuário.
-                2.  **Base nas Informações:** Fundamente TODA a sua análise nos trechos fornecidos. NÃO use conhecimento externo.
-                3.  **Contexto Jurídico/Normativo:** Se aplicável e presente nos trechos, relacione a análise com o contexto jurídico (validade, admissibilidade, CPP) e normativo (ABNT, ISO, etc.).
+                2.  **Base nas Informações:** Fundamente TODA a sua análise nos documentos fornecidos. NÃO use conhecimento externo.
+                3.  **Contexto Jurídico/Normativo:** Se aplicável e presente nos documentos, relacione a análise com o contexto jurídico (validade, admissibilidade, CPP) e normativo (ABNT, ISO, etc.).
                 4.  **Implicações:** Explore as implicações práticas dos pontos analisados (ex: para a cadeia de custódia, para o contraditório).
-                5.  **Citação Indireta:** Refira-se aos trechos de forma indireta ao construir sua argumentação (ex: "Um dos documentos menciona que...", "Conforme uma análise complementar..."). NÃO use "Trecho 1", "Fonte X".
+                5.  **Citação no Padrão ABNT:** Ao citar informações dos documentos, use o padrão ABNT: 
+                    - Para citação direta curta (até 3 linhas): "texto entre aspas" (SOBRENOME, ano, p. X).
+                    - Para citação indireta: Conforme Sobrenome (ano), a análise indica que...
+                    - Use os números de referência entre colchetes: [1], [2], etc.
                 6.  **Estrutura Clara:** Organize a resposta de forma lógica (parágrafos, marcadores).
                 7.  **Linguagem Precisa:** Use termos técnicos corretamente.
-                8.  **Insuficiência de Dados:** Se os trechos não contêm informação suficiente para responder algum aspecto, declare isso explicitamente (ex: "As informações fornecidas não detalham o método X.").
-                9.  **Não Invente:** Jamais adicione informações não presentes nos trechos.
+                8.  **Insuficiência de Dados:** Se os documentos não contêm informação suficiente para responder algum aspecto, declare isso explicitamente.
+                9.  **Não Invente:** Jamais adicione informações não presentes nos documentos.
                 10. **Foco:** Mantenha o foco na pergunta do usuário.
+                11. **Referências Bibliográficas:** Ao final da resposta, inclua uma seção intitulada "REFERÊNCIAS BIBLIOGRÁFICAS" listando todas as fontes citadas no formato ABNT completo.
                 """

                 # AJUSTE 2: Cálculo dinâmico de max_tokens
                 model_max_context = 4097  # Limite do text-davinci-003 (ajustar se mudar modelo)