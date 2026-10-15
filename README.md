# CDP Mutation Analysis

O **CDP Mutation Analysis** é um projeto de Ciência de Dados e Aprendizado de Máquina desenvolvido para processar, analisar e extrair insights preditivos a partir dos dados coletados pelo [CDP Mutation Collector]. O objetivo principal é avaliar a capacidade de predição sobre as mutações do DOM em aplicações web, baseando-se nas ricas características espaciais, geométricas e semânticas extraídas durante as interações automatizadas.

## 🚀 Funcionalidades Principais

* **Ingestão e Pré-processamento de Dados:** Utiliza a biblioteca **Pandas** para carregar os complexos arquivos JSON gerados pelo coletor, normalizando atributos aninhados, tratando valores ausentes e estruturando os dados tabulares (DataFrames) ideais para o treinamento.
* **Engenharia de Features (Feature Engineering):** Seleção e transformação das métricas calculadas (como distâncias espaciais, densidade de nós, `tagNames`, ancestrais semânticos e tipos de eventos) para maximizar o poder preditivo.
* **Treinamento Multimodelo:** Implementa e treina múltiplos algoritmos de aprendizado de máquina supervisionado utilizando a biblioteca **Scikit-Learn**.
* **Avaliação de Desempenho:** Compara o desempenho dos diferentes algoritmos utilizando métricas padrão de classificação/regressão (como Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão).

## 🧠 Algoritmos de Aprendizado Supervisionado

O pipeline de análise realiza experimentos e compara os seguintes modelos preditivos:

* **Support Vector Machines (SVM):** Eficaz na separação de classes em espaços de alta dimensionalidade gerados pelas inúmeras *features* do DOM.
* **Árvores de Decisão (Decision Trees):** Permite uma interpretação visual e lógica de quais atributos (ex: dimensão de um elemento, contagem de palavras) são mais determinantes para uma mutação.
* **Random Forest:** Um método *ensemble* robusto que reduz o sobreajuste (*overfitting*) das Árvores de Decisão, lidando muito bem com as variações não lineares da interface web.
* **Outros (expansível):** Regressão Logística, K-Nearest Neighbors (KNN), entre outros.

## 🛠️ Tecnologias e Dependências

* **Python 3.x**
* **[Pandas](https://pandas.pydata.org/):** Para manipulação, limpeza e análise exploratória de dados.
* **[Scikit-Learn](https://scikit-learn.org/):** Para todo o pipeline de Machine Learning (divisão de treino/teste, normalização, modelagem e métricas).
