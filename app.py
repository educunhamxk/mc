import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

#configuração da página
st.set_page_config(page_title="BicMac", page_icon="🍔", layout="centered", initial_sidebar_state="collapsed")

#definição do tema
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4F4F4F;
}
</style>
    """, unsafe_allow_html=True)

#título
st.title("Projeção de preço em dólar e por país do ...")

#exibir imagem tema do Mc
st.image("mc.JPG")

#texto
st.markdown("Para realizar a projeção é bem simples, preencha o campo de data e selecione um dos países disponíveis na lista que estimaremos o preço do BicMac para o país e data selecionada.")

#1º Bloco************************************************************************************************************************
df_original = pd.read_csv("BigmacPrice.csv",delimiter=";")
df = df_original.copy()
unique_paises = df['name'].unique().tolist()

#caixa de entrada para a data
data = st.date_input("Escolha uma data")

#lista suspensa para os países
pais = st.selectbox("Escolha um país", unique_paises)

if st.button('Carregar modelo e fazer previsão'):

    #abaixo são tratamentos que foram feitos na base de treinamento para que possamos ter a mesma estrutura de base para fazer a projeção
    
    #caputando a moeda do país escolhido
    df_pais= df[df['name'] == pais]
    st.markdown(df_pais.shape)
    moeda = df_pais['currency_code'].iloc[0]

    df['registro_projecao'] = 0
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    #df['date'] = pd.to_datetime(df['date']) 

    #criação do novo registro referente a projeção para ser empilhado na base principal
    novo_registro = {'name': pais, 'date': data, 'registro_projecao': 1, 'currency_code' : moeda}
    novo_registro = pd.DataFrame([novo_registro])
    novo_registro['date'] = pd.to_datetime(novo_registro['date'], format='%d/%m/%Y')
    df = pd.concat([df, novo_registro])

    #preenchendo com 0s nas colunas que não são referentes ao país e nem a moeda do país
    df = df.fillna(0)

    #ordenando o df por data e referenciando a distância entre a primeira data e a data da projeção
    df = df.sort_values(by=['date'])
    df['data_numerica'] = (df['date'] - df['date'].min()).dt.days
    df = df.drop(columns=['date'])

    #criando as dummies
    dummy_moeda = pd.get_dummies(df['currency_code'])
    df = pd.concat([dummy_moeda,df],axis=1)

    dummy_pais = pd.get_dummies(df['name'])
    df = pd.concat([dummy_pais,df],axis=1)

    #dropando colunas que originaram as dummies
    df.drop(columns=['currency_code','name'],inplace=True)

    #capturando apenas o registro da projeção, dropando as últimas colunas e normalizando a base
    df = df[df['registro_projecao']==1]
    df.drop(columns=['dollar_ex','dollar_price','local_price','registro_projecao'],inplace=True)
    scaler = joblib.load('scaler.pkl')
    X_test_scaled = scaler.transform(df)

    #carregamento / instanciamento do modelo pkl
    model = load_model('model')

    #usando o modelo para fazer previsões
    projecao = model.predict(X_test_scaled)
    projecao = np.round(projecao, 2)[0]

    st.markdown(f"A projeção de preço do BigMac para {pais} na data {data} é de U$ {projecao}.")

    novo_registro['dollar_price'] = projecao

    df_pais = df_original[df_original['name'] == pais]
    #empilhar os dados de projeção no df original
    df_pais_full = df_pais.append(novo_registro,ignore_index=True)

    #criar uma nova coluna para marcar o ponto de projeção
    df_pais_full['projecao'] = df_pais_full['date'] == data

    #plotar o gráfico de linha
    df_pais_full['date'] = pd.to_datetime(df_pais_full['date'])
    df_pais_full['date'] = df_pais_full['date'].apply(lambda x: x.to_pydatetime())
    
    #plotando a série histórica
    plt.plot(df_pais_full['date'], df_pais_full['dollar_price'], color='blue', linestyle='solid', marker='')

    #adicionando o ponto de projeção com uma cor diferente
    plt.plot(df_pais_full[df_pais_full['projecao']]['date'], df_pais_full[df_pais_full['projecao']]['dollar_price'], color='red', marker='o')

    plt.title('Projeção do Preço do BigMac')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.grid(True)
    plt.show()
    st.pyplot(plt)
    st.markdown("*O modelo foi treinado com dados até Julho de 2022, se a data de projeção for muito distante deste período pode impactar na performance do modelo.")






    

        
