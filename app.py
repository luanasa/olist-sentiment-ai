# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# Configuração da Página
st.set_page_config(layout="wide", page_title="Olist AI Review Audit")

st.title("Olist AI: Auditoria de Sentimentos com NLP")
st.markdown("""
**Problema:** Ler milhares de comentários manualmente é impossível.
**Solução:** Uma IA baseada em BERT que lê, interpreta e classifica o sentimento do cliente automaticamente.
""")

# ---------------------------------------------------------
# CARREGAMENTO DE DADOS E IA
# ---------------------------------------------------------

@st.cache_resource # O Streamlit guarda o modelo na memória para não carregar toda hora
def load_model():
    # Baixando um modelo BERT treinado especificamente para reviews (multilíngue)
    # Ele classifica de 1 a 5 estrelas
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_data
def load_data():
    df = pd.read_csv('olist_order_reviews_dataset.csv')
    # Remover linhas sem comentários escritos
    df = df.dropna(subset=['review_comment_message'])
    return df

with st.spinner("Carregando o Cérebro da IA (Modelo BERT)..."):
    classifier = load_model()

df = load_data()

# ---------------------------------------------------------
# INTERFACE DE TESTE 
# ---------------------------------------------------------
st.sidebar.header("Teste a IA você mesmo")
user_input = st.sidebar.text_area("Digite um comentário fictício:", "O produto chegou quebrado e atrasado!")

if st.sidebar.button("Analisar Sentimento"):
    result = classifier(user_input)[0]
    label = result['label'] # Ex: '1 star'
    score = result['score'] # Confiança da IA
    
    # Traduzindo 'X stars' para Positivo/Negativo
    stars = int(label.split()[0])
    if stars <= 2:
        sentiment = "🔴 Negativo"
    elif stars == 3:
        sentiment = "🟡 Neutro"
    else:
        sentiment = "🟢 Positivo"
        
    st.sidebar.success(f"Classificação: {sentiment} ({stars} estrelas)")
    st.sidebar.info(f"Confiança da IA: {score:.2f}")

# ---------------------------------------------------------
# ANÁLISE EM MASSA (Amostragem)
# ---------------------------------------------------------
st.divider()
st.subheader("Análise Real: Amostra dos Comentários da Olist")
st.info("Nota: Como processar NLP é pesado para CPU, vamos analisar apenas uma amostra aleatória de 50 comentários.")

# Botão para iniciar processamento
if st.button("Ler e Classificar 50 Comentários Aleatórios"):
    
    # Pegar 50 comentários aleatórios
    sample_df = df.sample(50, random_state=42).copy()
    
    # Função para aplicar a IA em cada linha
    def analyze_text(text):
        # A IA tem limite de tamanho de texto, então cortei em 512 caracteres
        result = classifier(text[:512])[0]
        return int(result['label'].split()[0])

    # Barra de progresso visual
    progress_bar = st.progress(0)
    results = []
    
    for i, row in enumerate(sample_df.iterrows()):
        texto = row[1]['review_comment_message']
        stars = analyze_text(texto)
        results.append(stars)
        progress_bar.progress((i + 1) / 50)
        
    sample_df['ia_score'] = results
    
    # Categorizar
    def categorize(star):
        if star <= 2: return 'Negativo'
        if star == 3: return 'Neutro'
        return 'Positivo'
    
    sample_df['sentimento'] = sample_df['ia_score'].apply(categorize)
    
    # ---------------------------------------------------------
    # DASHBOARD DOS RESULTADOS
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Distribuição de Sentimentos (IA)")
        fig = px.pie(sample_df, names='sentimento', color='sentimento', 
                     color_discrete_map={'Negativo':'red', 'Positivo':'green', 'Neutro':'gold'})
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write("### O que os clientes estão falando?")
        # Nuvem de palavras
        all_text = " ".join(sample_df['review_comment_message'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
    st.write("### Dados Processados")
    st.dataframe(sample_df[['review_comment_message', 'ia_score', 'sentimento']])