# Detector de SPAM em mensagens SMS

## Objetivo do projeto

Este projeto tem o objetivo de mostrar como o **Multilayer Perceptron** pode ser utilizado para a detecção de SPAM. Embora o projeto esteja em Língua Portuguesa do Brasil, a base de SMS utilizada foi [encontrada no Kaggle](https://www.kaggle.com/code/dhgupta/bag-of-words-model/input), em Língua Inglesa, com o original disponibilizado em `data/spam.csv`.

Muitos projetos para a avaliação de SPAM em Língua Inglesa existem, assim deseja-se que este projeto faça tal classificação em Língua Portuguesa sendo necessário, para tanto, a tradução de toda a base de dados que será feita da seguinte forma:

1. Utilização de *small language models* no [Ollama](https://ollama.com) para rodar *small language models* local;
1. Utilização do modelo [Quen2.5 Translator](https://ollama.com/lauchacarro/qwen2.5-translator) para a tradução do texto para a língua portuguesa, com revisão humana superficial (arquivo gerado em `data/spam_br.csv`).

## Tradução da base de dados em para português

Nesta parte vamos gerar o arquivo `data/spam_en.csv` que será o resultado da correção gramatical aplicada no arquivo original `data/spam.csv`.

## Análise dos dados

O primeiro passo em um projeto dessa natureza é analisar os dados. Para tanto, faremos uma leitura da base original e vamos contar o número palavras e frases. Ao final dessa tarefa, vamos decidir por executar ou não algum pré-processamento.

### Histograma

Podemos ver no histograma que as mensagens de SPAM são muito menores que as HAM na distribuição.

![Histogram](./figures/histograma.png)

### Nuvem de palavras

Na nuvem de palavras ficará claro que as mensagens de SPAM realmente são aquelas que estamos acostumados a ver, o que indica que a tradução ficou razoável.

![HAM](./figures/nuvem_palavras_ham.png)

![SPAM](./figures/nuvem_palavras_spam.png)

### Análise de frequência das palavras

Por fim, procedeu-se com uma análise das 25 palavras mais frequentes em cada tipo de mensagem.

![Frequency](./figures/top_25_palavras.png)

## Resultado da análise

### Total de mensagens

O total de mensagens na base de dados é apresentado a seguir, separadas pelo grupo Label.

| **Label** | **Número de SMS** | **Média de palavras por SMS** |
|-----------|-------------------|------------------------------|
| ham       | 4825              | 14.33                        |
| spam      | 747               | 27.55                        |
| **Total** | **5572**          | **16.38**                    |

### Desbalanceamento nas classes

De acordo com a análise até o momento, temos um claro desbalanceamento nas classes.

| Classe | Nº de amostras | Palavras por mensagem (média) |
| ------ | -------------- | ----------------------------- |
| ham    | 4.825          | \~14                          |
| spam   | 747            | \~27                          |

A classe `ham` representa cerca de 87% dos dados. Um modelo treinado sem cuidado pode aprender a simplesmente prever tudo como `ham` e ainda parecer `preciso`.
    
**Foi o que aconteceu no projeto do diabetes**, onde as classes estavam demasiadamente desbalanceadas. O Pedro, como bom estatístico, faz os dados falarem o que ele quer de forma convincente, só eu não fiquei convencido daqueles resultados: **Os modelos apresentados, tanto a versão MLP quanto a logística, NÃO FUNCIONAVAM!**
    
Tentei insistir para corrigir, mas fui voto vencido.

Como este projeto estou fazendo sozinho, não vou permitir desbalanceamentos. Para corrigir vou usar uma ou mais das possibilidades:

1. Na hora de chamar o treinador MLP, dar peso maior para a classe 'desbalanceada':
    ```
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42, class_weight='balanced')
    ```

1. Usar o `SMOTE` para fazer criar amostras sintéticas da classe `spam`
    ```
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    ```

1. Remover dados da classe `ham`, mas não vou seguir essa abordagem dado que a quantidade de dados já está baixa.

1. Uma outra atividade que podemos fazer é a produção de novos dados com base na correção gramatical do inglês e na tradução para o português, o que seria desejável.