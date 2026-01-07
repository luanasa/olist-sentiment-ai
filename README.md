# 🤖 Olist AI: Auditoria de Sentimentos com NLP
<img width="1920" height="1670" alt="screencapture-olist-sentiment-ai-phq2wzzeunmuxoexxb8usj-streamlit-app-2026-01-07-19_11_10" src="https://github.com/user-attachments/assets/067517fe-4519-468c-baef-9e0c3b51d88e" />

![Python](https://img.shields.io/badge/Python-AI-blue.svg)
![BERT](https://img.shields.io/badge/Model-BERT%20Transformer-yellow.svg)
![NLP](https://img.shields.io/badge/Tech-NLP-green.svg)

> **Teste a IA ao vivo:** [🔗(https://olist-sentiment-ai-phq2wzzeunmuxoexxb8usj.streamlit.app/)]

## 🧠 O Problema de Negócio
A Olist recebe milhares de avaliações mensais. Analisar notas (1 a 5 estrelas) é fácil, mas entender **o motivo** por trás da nota exige ler comentários, o que é inviável humanamente nessa escala. A empresa precisava de uma forma automatizada de classificar o sentimento dos textos para agir rápido sobre críticas.

## 💡 A Solução
Desenvolvi um sistema de **Auditoria Automatizada de Sentimentos** utilizando Processamento de Linguagem Natural (NLP) e Deep Learning.

O sistema utiliza o modelo **BERT (Bidirectional Encoder Representations from Transformers)** pré-treinado em múltiplos idiomas para ler o comentário do cliente e classificar a intenção (Negativa, Neutra ou Positiva) com grau de confiança.

## ⚙️ Tecnologias Utilizadas
* **Transformers (Hugging Face):** Implementação do modelo `bert-base-multilingual`.
* **PyTorch:** Backend de processamento tensorial para a IA.
* **Streamlit:** Interface Front-end para demonstrar a IA em funcionamento.
* **Python:** Processamento de dados e integração.

## 🚀 Como usar
No aplicativo, você pode:
1.  **Digitar um texto:** Teste a capacidade de interpretação da IA em tempo real.
2.  **Auditoria em Lote:** O sistema puxa uma amostra aleatória do dataset real da Olist e gera um relatório gráfico de sentimentos e nuvem de palavras.

---
## 👩‍💻 Autora
**Luana Sá**
*Dev & Data Scientist*
[LinkedIn](SEU_LINK) | [Portfólio](SEU_PORTFOLIO)
