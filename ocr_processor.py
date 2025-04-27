# -*- coding: utf-8 -*-
import os
import fitz  # PyMuPDF library
import pytesseract
from PIL import Image
import io
import traceback

# --- Configuration ---
# ATENÇÃO: Ajuste o caminho para a pasta que contém seus PDFs
PDF_FOLDER = r"C:\Users\pbm_s\OneDrive\Prova Digital"
# Cria uma subpasta dentro da pasta principal para armazenar os arquivos de texto
OUTPUT_FOLDER = os.path.join(PDF_FOLDER, "ocr_texts")
# Limiar: Se o texto extraído tiver menos caracteres que isso, assume-se que o OCR é necessário.
# Ajuste com base nos resultados típicos de extração de texto de PDFs vazios/digitalizados.
MIN_TEXT_LENGTH_THRESHOLD = 150
# Idioma para Tesseract OCR (Português)
OCR_LANGUAGE = 'por'

# --- !!! IMPORTANTE: CONFIGURAÇÃO DO TESSERACT !!! ---
# Remova o comentário e defina o caminho correto se o Tesseract não estiver no PATH do seu sistema
# --- Exemplo Windows ---
# Se o Tesseract não for encontrado, remova o '#' da linha abaixo e ajuste o caminho
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# --- Exemplo Linux/macOS (geralmente não necessário se instalado via gerenciador de pacotes) ---
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# --- Fim da Configuração do Tesseract ---

# --- Funções Auxiliares ---

def check_tesseract():
    """Verifica se o Tesseract está acessível."""
    try:
        tess_version = pytesseract.get_tesseract_version()
        print(f"Tesseract versão {tess_version} encontrado e acessível.")
        return True
    except pytesseract.TesseractNotFoundError:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!! TESSERACT NÃO ENCONTRADO !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Certifique-se de que o motor Tesseract OCR está instalado E seu caminho está definido corretamente.")
        print("1. Instale o Tesseract (incluindo pacotes de idioma como 'por').")
        print("2. Certifique-se de que ele esteja na variável de ambiente PATH do seu sistema, OU")
        print("3. Remova o comentário e defina a linha 'pytesseract.pytesseract.tesseract_cmd'")
        print("   neste script para o caminho completo do seu executável 'tesseract.exe' (Windows)")
        print("   ou 'tesseract' (Linux/macOS).")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return False
    except Exception as e:
        print(f"Ocorreu um erro ao verificar o Tesseract: {e}")
        return False

def needs_ocr(pdf_path, threshold):
    """
    Verifica se um PDF provavelmente precisa de OCR tentando extrair texto.
    Retorna True se o comprimento do texto estiver abaixo do limiar, False caso contrário ou em erro.
    """
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        # Verifica apenas as primeiras páginas para eficiência, ajuste conforme necessário
        num_pages_to_check = min(doc.page_count, 5)
        for i in range(num_pages_to_check):
            page = doc.load_page(i)
            extracted_text += page.get_text("text") # Extrai texto plano
            # Otimização: Se já encontramos texto suficiente, não há necessidade de verificar mais
            if len(extracted_text) >= threshold:
                doc.close()
                # print(f"  Texto suficiente encontrado ({len(extracted_text)} chars), OCR provavelmente não necessário.")
                return False
        doc.close()
        # print(f"  Verificação de texto concluída. Encontrados {len(extracted_text)} caracteres.")
        return len(extracted_text) < threshold
    # Bloco except fitz.PasswordException removido pois não existe mais nesse formato
    except Exception as e: # Captura erros gerais (incluindo senha, corrompido, etc.)
        print(f"  Erro durante a verificação inicial de texto para '{os.path.basename(pdf_path)}': {e}")
        # traceback.print_exc() # Descomente para erro detalhado
        # Cautela: se a extração de texto falhar, talvez *precise* de OCR,
        # mas também pode ser um PDF corrompido. Retornar False evita tentar OCR em erros de leitura.
        return False

def perform_ocr_on_pdf(pdf_path, output_txt_path, language):
    """
    Realiza OCR em cada página do PDF e salva o texto combinado em um arquivo.
    Retorna True em sucesso, False em falha.
    """
    all_text = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        print(f"    Iniciando OCR ({language}) para {num_pages} página(s)...")

        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            # Determina um DPI dinâmico com base no tamanho da página, visando qualidade vs desempenho
            # Abordagem simples: usa DPI maior para páginas menores, padrão para maiores
            rect = page.rect
            diag = (rect.width**2 + rect.height**2)**0.5
            dpi = 300 if diag < 1000 else 200 # Limiar de exemplo, ajuste conforme necessário

            print(f"      - Processando página {page_num + 1}/{num_pages} (usando DPI: {dpi})")

            # Renderiza a página para uma imagem (Pixmap)
            pix = page.get_pixmap(dpi=dpi)

            # Converte Pixmap para Imagem PIL
            try:
                img_bytes = pix.tobytes("png") # Usa PNG para conversão sem perdas
                img = Image.open(io.BytesIO(img_bytes))
            except Exception as img_conv_e:
                print(f"      Erro ao converter página {page_num + 1} para imagem: {img_conv_e}")
                all_text += f"\n\n--- ERRO AO CONVERTER PÁGINA {page_num + 1} ---\n\n"
                continue # Pula para a próxima página

            # Realiza OCR na imagem
            try:
                # Configuração Tesseract: --psm 3 é frequentemente bom para layout geral de página
                custom_config = r'--oem 3 --psm 3'
                page_text = pytesseract.image_to_string(img, lang=language, config=custom_config)
                all_text += page_text + "\n\n--- Quebra de Página ---\n\n"
            except pytesseract.TesseractError as tess_e:
                print(f"      Erro durante o OCR Tesseract na página {page_num + 1}: {tess_e}")
                all_text += f"\n\n--- ERRO TESSERACT NA PÁGINA {page_num + 1} ---\n\n"
            except Exception as ocr_e:
                print(f"      Erro inesperado durante o OCR na página {page_num + 1}: {ocr_e}")
                all_text += f"\n\n--- ERRO INESPERADO DE OCR NA PÁGINA {page_num + 1} ---\n\n"

        doc.close()

        # Salva o texto extraído no arquivo de saída
        try:
            # Garante que o diretório para o arquivo de saída exista
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(all_text)
            print(f"    OCR realizado com sucesso e texto salvo em '{os.path.basename(output_txt_path)}'")
            return True
        except Exception as write_e:
             print(f"    Erro ao escrever texto OCR no arquivo '{os.path.basename(output_txt_path)}': {write_e}")
             return False

    except Exception as e:
        print(f"  Erro geral durante o processo de OCR para '{os.path.basename(pdf_path)}': {e}")
        traceback.print_exc() # Imprime erro detalhado para depuração
        # Limpa arquivo parcial se erro ocorreu durante processamento
        if os.path.exists(output_txt_path):
            try: os.remove(output_txt_path)
            except OSError: pass
        return False

# --- NOVA FUNÇÃO AUXILIAR para extrair texto existente ---
def extract_existing_text(pdf_path, output_txt_path):
    """
    Extrai o texto existente de um PDF e salva em um arquivo de texto.
    Retorna True em sucesso, False em falha.
    """
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        print(f"    Extraindo texto existente de {doc.page_count} página(s)...")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text("text") # Extrai texto plano
            extracted_text += page_text
            # Adiciona uma quebra de página simbólica se desejar (opcional)
            if page_num < doc.page_count - 1:
                 extracted_text += "\n\n--- Quebra de Página ---\n\n" # Mesma quebra do OCR para consistência

        doc.close()

        # Salva o texto extraído
        try:
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"    Texto existente extraído com sucesso para '{os.path.basename(output_txt_path)}'")
            return True
        except Exception as write_e:
             print(f"    Erro ao escrever texto existente no arquivo '{os.path.basename(output_txt_path)}': {write_e}")
             return False

    except Exception as e:
        # Captura erros ao abrir/ler o PDF (incluindo os protegidos por senha que não foram detectados antes)
        print(f"  Erro ao extrair texto existente de '{os.path.basename(pdf_path)}': {e}")
        # Limpa arquivo parcial se erro ocorreu durante processamento
        if os.path.exists(output_txt_path):
            try: os.remove(output_txt_path)
            except OSError: pass
        return False
# --- Fim da Nova Função Auxiliar ---


# --- Lógica Principal de Execução (MODIFICADA) ---
def main():
    """Função principal para orquestrar a verificação de PDF, OCR e extração de texto, incluindo subpastas."""
    print("-" * 60)
    print("Iniciando Script de Processamento de PDF (OCR e Extração de Texto)") # Mensagem atualizada
    print(f"Pasta Fonte dos PDFs: {PDF_FOLDER}")
    print(f"Pasta de Saída dos Textos: {OUTPUT_FOLDER}")
    print(f"Limiar de Texto (mín chars): {MIN_TEXT_LENGTH_THRESHOLD}")
    print(f"Idioma do OCR: {OCR_LANGUAGE}")
    print("-" * 60)

    if not check_tesseract():
        print("Saindo do script porque o Tesseract não está configurado corretamente.")
        return # Para a execução se o Tesseract não estiver disponível

    if not os.path.isdir(PDF_FOLDER):
        print(f"Erro: Pasta fonte '{PDF_FOLDER}' não encontrada.")
        return

    # Garante que o diretório base de saída exista
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Diretório base de saída garantido: '{OUTPUT_FOLDER}'")

    processed_files = 0
    ocr_performed_count = 0
    text_extracted_count = 0 # Novo contador
    skipped_output_exists_count = 0 # Contador renomeado para clareza
    error_count = 0

    # Itera por todos os arquivos no diretório fonte e subdiretórios
    for root, dirs, files in os.walk(PDF_FOLDER):

        # --- Modificação: Evita descer na própria pasta de saída ---
        output_folder_basename = os.path.basename(OUTPUT_FOLDER)
        if output_folder_basename in dirs:
            # print(f"  Prevenindo descida no diretório de saída: {os.path.join(root, output_folder_basename)}")
            dirs.remove(output_folder_basename)
        # --- Fim da Modificação ---

        # Ignora diretórios vazios ou sem PDFs
        pdf_files_in_dir = [f for f in files if f.lower().endswith(".pdf")]
        if not pdf_files_in_dir:
             continue # Pula para o próximo diretório se não houver PDFs aqui

        print(f"\nEscaneando diretório: {root}")
        print(f"  Encontrados {len(pdf_files_in_dir)} arquivo(s) PDF.")


        for filename in pdf_files_in_dir:
            processed_files += 1
            pdf_path = os.path.join(root, filename)
            base_name = os.path.splitext(filename)[0]
            txt_filename = f"{base_name}.txt"

            # --- Estrutura de Saída Espelhada ---
            relative_dir = os.path.relpath(root, PDF_FOLDER)
            if relative_dir == ".":
                current_output_dir = OUTPUT_FOLDER
            else:
                current_output_dir = os.path.join(OUTPUT_FOLDER, relative_dir)
            os.makedirs(current_output_dir, exist_ok=True)
            output_txt_path = os.path.join(current_output_dir, txt_filename)
            # --- Fim da Estrutura Espelhada ---

            print(f"\n[{processed_files}] Processando PDF: {filename}") # Mais curto para clareza
            # print(f"  Caminho completo: {pdf_path}")
            # print(f"  Destino TXT: {output_txt_path}")


            # 1. Verifica se o arquivo .txt de saída já existe (Evitar Duplicação)
            if os.path.exists(output_txt_path):
                relative_output_location = os.path.relpath(current_output_dir, OUTPUT_FOLDER)
                relative_output_location = "." if not relative_output_location else relative_output_location
                print(f"  Pulando: Arquivo de saída '{txt_filename}' já existe em '{relative_output_location}'.")
                skipped_output_exists_count += 1 # Usa contador específico
                continue

            # 2. Verifica se o PDF precisa de OCR
            #    (needs_ocr retorna False também se houver erro na leitura inicial)
            ocr_needed = needs_ocr(pdf_path, MIN_TEXT_LENGTH_THRESHOLD)

            if ocr_needed:
                print(f"  Verificação indica que OCR pode ser necessário para '{filename}'.")
                # 3. Realiza OCR
                if perform_ocr_on_pdf(pdf_path, output_txt_path, OCR_LANGUAGE):
                    ocr_performed_count += 1
                else:
                    print(f"  Falha no OCR para '{filename}'.")
                    error_count += 1
            else:
                # Se o OCR não for necessário (texto existe ou erro na verificação inicial),
                # tenta extrair o texto existente diretamente.
                print(f"  Tentando extrair texto existente de '{filename}' (OCR não considerado necessário).")
                if extract_existing_text(pdf_path, output_txt_path):
                    text_extracted_count += 1 # Incrementa novo contador em sucesso
                else:
                    print(f"  Falha ao extrair texto existente de '{filename}'.")
                    error_count += 1 # Incrementa contador de erro em falha

    print("\n" + "=" * 60)
    print("Resumo do Processamento:")
    print(f"  Total de arquivos PDF encontrados: {processed_files}")
    print(f"  PDFs processados com OCR:        {ocr_performed_count}")
    print(f"  PDFs com texto existente extraído:{text_extracted_count}") # Novo contador
    print(f"  PDFs pulados (saída já existe):  {skipped_output_exists_count}") # Contador renomeado
    print(f"  Erros durante o processamento:   {error_count}") # Inclui erros de OCR e extração
    print(f"  Arquivos de texto salvos em:     '{OUTPUT_FOLDER}' (preservando estrutura de subpastas)")
    print("=" * 60)


if __name__ == "__main__":
    # Verifica o Tesseract na inicialização antes de chamar a lógica principal
    if check_tesseract():
        main() # Chama a função principal apenas se o Tesseract for encontrado
    else:
        print("\nPor favor, configure o Tesseract e tente novamente.")