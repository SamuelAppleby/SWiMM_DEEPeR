@echo off
call ..\.venv\Scripts\activate
cd ..\cmvae_scripts
start python eval_cmvae.py
exit