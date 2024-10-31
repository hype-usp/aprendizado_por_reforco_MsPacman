# 💻 Configurando o Ambiente do Pac-Man

Antes de começar, certifique-se de que você atendeu aos seguintes requisitos e siga as etapas abaixo para configurar o ambiente Conda necessário para o projeto.

## Passo 1: Criar e Ativar o Ambiente Conda

Primeiro, crie o ambiente com Python 3.8 e ative-o:

```bash
conda create --name pacman python=3.8
conda activate pacman
```

Com o ambiente ativo, instale os pacotes essenciais:

```bash
pip install numpy
pip install opencv-python
pip install gym
pip install "gym[atari, accept-rom-license]"
```
