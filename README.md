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

![Hstograma](./figures/histograma.png)

### Nuvem de palavras

Na nuvem de palavras ficará claro que as mensagens de SPAM realmente são aquelas que estamos acostumados a ver, o que indica que a tradução ficou razoável.

| ![HAM](./figures/nuvem_palavras_ham.png) | ![SPAM](./figures/nuvem_palavras_spam.png) |
|-----------------------------|-----------------------------|

