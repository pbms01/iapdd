@echo off
echo Ativando ambiente virtual...
call .\venv\Scripts\activate

echo Iniciando o Agente GPT via Streamlit...
streamlit run lexperito.py

pause
