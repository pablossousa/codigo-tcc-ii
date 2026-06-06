# Documentação Técnica do Sistema de Reconhecimento Facial

## 1. Visão geral do sistema

Este projeto implementa um sistema local de reconhecimento facial em tempo real, executado em Python no Windows e utilizando webcam. A aplicação foi desenvolvida com foco em uso acadêmico e prototipagem para TCC, mantendo todo o processamento no computador do usuário, sem dependência de servidores externos, serviços web ou frameworks adicionais.

O sistema possui um único módulo funcional: reconhecimento facial. A interface permite cadastrar pessoas, reconhecer rostos pela câmera, excluir pessoas cadastradas e, quando configurado, digitar automaticamente o código do cartão reconhecido no campo atualmente selecionado no sistema operacional. A digitação automática é feita como se fosse um teclado virtual, utilizando a biblioteca `pynput`.

A solução usa uma arquitetura baseada em classes, concentrada atualmente no arquivo principal `face_recognition_app.py`. Embora o código esteja em um único arquivo, as responsabilidades internas estão separadas em componentes como gerenciamento de câmera, banco de dados, criptografia, motor facial, cache de embeddings, votação temporal, interface gráfica e digitação automática.

O reconhecimento facial é feito com a biblioteca `facenet-pytorch`, utilizando `MTCNN` para detecção facial e landmarks, e `InceptionResnetV1(pretrained="vggface2")` para geração dos embeddings faciais. O sistema não treina um modelo próprio de classificação; em vez disso, cadastra embeddings criptografados no banco SQLite e compara novos embeddings contra as amostras cadastradas.

## 2. Objetivos implementados

O programa foi estruturado para resolver os principais problemas de um reconhecimento facial simples baseado em poucos frames. As principais melhorias implementadas são:

- reconhecimento por múltiplos frames, usando votação temporal;
- cadastro multi-pose, com amostras de frente, esquerda e direita;
- estimativa simples de pose facial com landmarks do MTCNN;
- comparação contra múltiplas amostras por usuário;
- maior tolerância a variações moderadas de distância da câmera;
- avaliação básica de qualidade do frame, considerando tamanho do rosto, confiança de detecção e nitidez;
- armazenamento local em SQLite;
- criptografia dos embeddings faciais com Fernet;
- cache em memória para evitar consultas ao banco a cada frame;
- interface Tkinter organizada em cabeçalho, área central de câmera e painel lateral;
- exclusão de usuário pelo código do cartão;
- digitação automática do código reconhecido com controle de repetição, cooldown e opção de pressionar Enter.

## 3. Estrutura atual do projeto

A estrutura observada no projeto é:

```text
TCC_teste_simples/
├── face_recognition_app.py
├── requirements.txt
├── setup.ps1
├── run.ps1
├── data/
│   ├── faces.db
│   └── fernet.key
├── .venv/
└── .vscode/
```

O arquivo `face_recognition_app.py` é o ponto principal da aplicação. Ele contém a configuração, as classes do sistema, a interface gráfica, o fluxo de câmera, o cadastro, a exclusão, o reconhecimento e a inicialização do programa.

O arquivo `requirements.txt` lista as dependências necessárias:

```text
python
opencv-python<4.12
tkinter
facenet-pytorch
SQLite
numpy<2
cryptography (Fernet)
Pillow
pynput
torch
```

O diretório `data/` armazena os dados persistentes do sistema:

- `faces.db`: banco SQLite com usuários e embeddings faciais;
- `fernet.key`: chave local usada para criptografar e descriptografar os embeddings.

Os scripts `setup.ps1` e `run.ps1` foram criados para simplificar a instalação e execução no Windows. O `setup.ps1` cria o ambiente virtual `.venv`, instala as dependências e valida os principais imports. O `run.ps1` executa o sistema usando o Python do ambiente virtual.

## 4. Tecnologias utilizadas

### 4.1 Python

Toda a aplicação é escrita em Python. A linguagem é usada tanto para a lógica de reconhecimento quanto para interface gráfica, banco de dados, criptografia e automação de teclado.

### 4.2 OpenCV

O OpenCV (`cv2`) é usado para:

- abrir e ler frames da webcam;
- configurar resolução e backend da câmera;
- converter imagens entre BGR e RGB;
- desenhar caixas, textos e overlays no vídeo;
- calcular nitidez do rosto com variância do Laplaciano;
- alinhar a face com transformação afim;
- liberar janelas e recursos da câmera ao encerrar.

A câmera é aberta preferencialmente com `cv2.CAP_DSHOW`, que costuma funcionar melhor no Windows. Caso falhe, o sistema tenta outros backends, como `CAP_MSMF` e o backend padrão.

### 4.3 Tkinter

O Tkinter é usado para a interface gráfica. A janela principal possui:

- cabeçalho superior com título e status geral;
- área central para exibição do vídeo ao vivo;
- painel lateral direito com ações do operador e status do sistema;
- janelas de diálogo para cadastro, exclusão e confirmação de digitação.

A atualização da câmera ocorre pelo loop do Tkinter com `after()`, evitando um loop infinito bloqueante na interface.

### 4.4 Torch

O `torch` é usado para executar o modelo neural `InceptionResnetV1`. O sistema detecta automaticamente se há GPU CUDA disponível:

```python
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Se não houver placa de vídeo dedicada compatível com CUDA, o sistema roda em CPU. Nesse caso, o funcionamento é possível, porém o processamento pode ser mais lento.

### 4.5 facenet-pytorch

A biblioteca `facenet-pytorch` fornece os dois principais modelos do sistema:

- `MTCNN`: detector facial e extrator de landmarks;
- `InceptionResnetV1`: modelo que gera embeddings faciais.

O modelo de reconhecimento facial usado é:

```python
InceptionResnetV1(pretrained="vggface2")
```

Portanto, o sistema utiliza FaceNet/InceptionResnetV1 pré-treinado no conjunto VGGFace2. Ele não usa ArcFace.

### 4.6 NumPy

O NumPy é usado para manipulação dos embeddings e cálculos matemáticos, como:

- conversão para `float32`;
- normalização vetorial;
- média de embeddings capturados;
- cálculo de distância L2;
- organização de matrizes e arrays.

### 4.7 SQLite

O SQLite é usado como banco local. Ele permite armazenar usuários e embeddings sem instalar um servidor de banco de dados. O banco fica em `data/faces.db`.

### 4.8 cryptography.fernet

A biblioteca `cryptography`, por meio do Fernet, é usada para criptografar os embeddings antes de salvá-los no banco. Isso evita armazenar os vetores faciais em texto puro.

### 4.9 Pillow

O Pillow (`PIL`) é usado para converter frames do OpenCV em imagens compatíveis com o Tkinter (`ImageTk.PhotoImage`), permitindo exibir a câmera dentro da interface.

### 4.10 pynput

O `pynput` é usado para simular digitação no sistema operacional. Quando uma pessoa é reconhecida, o sistema pode digitar automaticamente o código do cartão em outro campo selecionado pelo operador.

## 5. Arquitetura interna

### 5.1 AppConfig

A classe `AppConfig` concentra os principais parâmetros ajustáveis da aplicação. Entre eles estão:

- índice da câmera;
- resolução da câmera;
- quantidade de frames de aquecimento;
- thresholds de reconhecimento;
- tamanho da janela temporal;
- quantidade mínima de frames válidos;
- quantidade de votos para confirmar reconhecimento;
- quantidade de amostras por pose no cadastro;
- parâmetros de qualidade do rosto;
- tempo de cooldown da digitação automática;
- limite de digitações consecutivas;
- cores da interface;
- caminhos do banco e da chave Fernet.

Essa centralização facilita testes e calibrações sem espalhar valores fixos pelo código.

### 5.2 Estruturas de dados

O programa usa `dataclasses` para representar dados importantes:

- `FaceSample`: uma amostra facial carregada do banco;
- `FaceDetection`: resultado da análise de um frame;
- `MatchCandidate`: melhor candidato encontrado para um embedding;
- `FrameVote`: voto produzido por um frame válido;
- `TemporalDecision`: decisão final da janela temporal;
- `RegistrationSession`: estado atual do cadastro multi-pose.

Essas estruturas ajudam a manter o fluxo organizado e reduzem a chance de confundir dados de detecção, cadastro e reconhecimento.

### 5.3 CryptoManager

A classe `CryptoManager` é responsável pela criptografia dos embeddings. Ao iniciar, ela carrega a chave Fernet de `data/fernet.key`. Se a chave não existir, uma nova chave é gerada.

Para salvar um embedding:

1. o vetor NumPy é convertido para `float32`;
2. o vetor é transformado em bytes;
3. os bytes são criptografados com Fernet;
4. o resultado criptografado é salvo como BLOB no SQLite.

Para carregar um embedding:

1. o BLOB criptografado é lido do banco;
2. o Fernet descriptografa os bytes;
3. os bytes são convertidos novamente para `np.float32`;
4. o sistema valida se o embedding tem tamanho compatível, normalmente 512 dimensões.

Um cuidado importante é que a chave `fernet.key` precisa ser preservada. Se ela for apagada ou substituída, os embeddings antigos não poderão ser descriptografados.

### 5.4 SecureFaceDB

A classe `SecureFaceDB` gerencia o banco SQLite. Ela cria e migra as tabelas necessárias, salva usuários, salva embeddings, lista usuários e remove usuários cadastrados.

O banco possui duas tabelas principais.

Tabela `users`:

```text
user_id INTEGER PRIMARY KEY AUTOINCREMENT
registration_id TEXT UNIQUE
name TEXT NOT NULL
created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
```

No código, o campo `registration_id` é usado internamente, mas na interface e na documentação ele representa o código do cartão.

Tabela `face_embeddings`:

```text
sample_id INTEGER PRIMARY KEY AUTOINCREMENT
user_id INTEGER NOT NULL
pose TEXT NOT NULL
embedding BLOB NOT NULL
created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
```

A tabela `face_embeddings` permite múltiplas amostras por usuário, inclusive separadas por pose. Isso é essencial para o cadastro multi-pose.

A exclusão de usuário é feita pelo código do cartão. Ao excluir, o sistema remove o usuário da tabela `users` e também remove os embeddings correspondentes da tabela `face_embeddings`, evitando registros órfãos.

### 5.5 EmbeddingCache

A classe `EmbeddingCache` mantém os embeddings carregados em memória. O cache é organizado por usuário:

```text
user_id -> lista de FaceSample
```

O objetivo é evitar consultas ao banco a cada frame da câmera. O cache é carregado ao iniciar o sistema e atualizado após cadastros e exclusões. Também existe um TTL configurável para atualização periódica.

### 5.6 FaceEngine

A classe `FaceEngine` é o núcleo de visão computacional e reconhecimento. Ela inicializa:

- `MTCNN`, para detecção facial;
- `InceptionResnetV1(pretrained="vggface2")`, para embeddings.

O fluxo de análise de frame é:

1. converter o frame BGR do OpenCV para RGB;
2. detectar rostos com MTCNN;
3. selecionar o melhor rosto, caso exista mais de um;
4. estimar a pose facial a partir dos landmarks;
5. avaliar a qualidade do frame;
6. alinhar a face com base em landmarks de referência;
7. gerar o embedding com InceptionResnetV1;
8. normalizar o embedding.

O sistema seleciona o melhor rosto combinando a confiança do MTCNN e a proporção da área do rosto no frame. Assim, quando há mais de um rosto, ele tende a priorizar o rosto mais confiável e mais relevante no enquadramento.

### 5.7 Estimativa de pose

A estimativa de pose é feita pela função `estimate_pose`. Ela usa os landmarks retornados pelo MTCNN, principalmente olhos, nariz e boca.

A heurística calcula deslocamentos horizontais do nariz em relação aos olhos e à boca. Com isso, classifica a pose em:

- `front`: rosto de frente;
- `left`: rosto virado para um lado;
- `right`: rosto virado para o outro lado.

Essa estimativa não é uma reconstrução 3D da cabeça, mas é suficiente para separar poses principais e orientar tanto o cadastro quanto a comparação de embeddings.

Na interface, essas poses são exibidas em português, como frente, esquerda e direita. Como a câmera pode estar espelhada visualmente, o código trata os rótulos de forma adequada para o operador durante o cadastro.

### 5.8 Avaliação de qualidade

Antes de aceitar um frame para reconhecimento ou cadastro, o sistema verifica alguns critérios:

- confiança do MTCNN;
- tamanho relativo do rosto na imagem;
- nitidez do recorte facial;
- possibilidade de alinhamento facial.

O tamanho do rosto é calculado pela razão entre a área da caixa facial e a área total do frame. Se o rosto estiver muito pequeno, o sistema orienta o usuário a se aproximar. Se estiver muito grande, orienta a se afastar.

A nitidez é estimada pela variância do Laplaciano no recorte do rosto. Esse valor ajuda a rejeitar frames muito borrados por movimento ou falta de foco.

O sistema diferencia limites rígidos e limites suaves. Limites rígidos impedem o uso do frame; limites suaves apenas geram mensagens de orientação, como aproximar um pouco, afastar um pouco ou manter o rosto mais firme.

### 5.9 Alinhamento facial

Antes de gerar o embedding, a face é alinhada usando os landmarks do MTCNN. O código usa `cv2.estimateAffinePartial2D` para calcular uma transformação entre os landmarks detectados e landmarks de referência.

Depois, `cv2.warpAffine` gera uma face alinhada no tamanho configurado, atualmente 160x160 pixels. Esse alinhamento melhora a estabilidade dos embeddings quando há pequenas variações de posição, inclinação ou distância da câmera.

### 5.10 Comparação de embeddings

Os embeddings são normalizados e comparados usando distância L2:

```text
distância = ||embedding_atual - embedding_cadastrado||
```

Para cada usuário, o sistema compara o embedding atual com todas as amostras cadastradas daquele usuário. A melhor distância encontrada representa o melhor candidato para aquele usuário.

Também existe um peso baseado na pose:

- mesma pose: distância levemente favorecida;
- pose frontal envolvida: peso neutro;
- poses laterais diferentes: distância levemente penalizada.

Isso permite priorizar amostras da mesma pose sem descartar completamente outras poses. Essa abordagem reduz falsos negativos quando o rosto está um pouco virado, mas ainda mantém alguma proteção contra falsos positivos.

### 5.11 TemporalVoteWindow

A classe `TemporalVoteWindow` implementa a votação temporal. Ela mantém uma fila de votos recentes com tamanho configurável, por padrão 20 frames.

Cada frame válido gera um `FrameVote`, contendo:

- usuário candidato;
- código do cartão;
- nome;
- distância;
- melhor distância;
- pose detectada;
- pose da amostra usada;
- timestamp.

Quando a janela atinge o tamanho necessário, o sistema decide se o reconhecimento foi confirmado. A confirmação exige:

- quantidade mínima de frames válidos;
- quantidade mínima de votos para o mesmo usuário;
- proporção mínima de votos;
- distância média abaixo do threshold final;
- melhor distância dentro da margem permitida.

Se não houver consenso suficiente, o sistema informa baixa confiança. Se as distâncias indicarem que o rosto não parece pertencer a nenhum usuário cadastrado, o sistema informa nova face detectada.

Essa etapa é importante porque evita decidir com base em apenas um ou dois frames, tornando o reconhecimento mais estável.

## 6. Fluxo de reconhecimento facial

O fluxo principal de reconhecimento ocorre dentro da classe `MainApp`, no método `_video_loop`, chamado repetidamente por `root.after()`.

O processo pode ser descrito assim:

1. abrir a câmera;
2. ler um frame;
3. analisar o frame com `FaceEngine`;
4. atualizar o painel de status;
5. se houver cadastro em andamento, processar o frame como cadastro;
6. caso contrário, processar como reconhecimento;
7. desenhar overlays no vídeo;
8. exibir o frame no Tkinter;
9. agendar a próxima iteração com `after()`.

Durante o reconhecimento:

1. se nenhum rosto for encontrado, o sistema volta ao estado de espera;
2. se o rosto for detectado, mas o frame for ruim, o sistema orienta o usuário;
3. se o embedding for gerado, ele é comparado com o cache de embeddings;
4. o resultado do frame é adicionado à janela temporal;
5. ao completar a janela, o sistema decide por votação;
6. se confirmado, atualiza interface e aciona a digitação automática, se aplicável.

## 7. Fluxo de cadastro multi-pose

O cadastro é iniciado pelo botão "Cadastrar pessoa". O operador informa:

- código do cartão;
- nome da pessoa.

O código do cartão é validado como hexadecimal, com no máximo 8 dígitos. São aceitos caracteres de `0` a `9` e de `A` a `F`. O valor é normalizado para letras maiúsculas.

Depois da validação, o sistema inicia o cadastro guiado por pose. As poses capturadas são:

1. frente;
2. esquerda;
3. direita.

Antes de cada etapa, o sistema mostra um pop-up com orientação para o operador ou para a pessoa cadastrada:

- "Olhe para frente para iniciar a captura.";
- "Agora vire o rosto levemente para a esquerda.";
- "Agora vire o rosto levemente para a direita.".

Durante cada pose, o sistema captura vários frames válidos. Cada frame precisa passar pela detecção, estimativa de pose e avaliação de qualidade. Frames ruins ou de pose incorreta não são usados.

Quando a quantidade configurada de amostras é atingida, os embeddings daquela pose são agregados por média. Essa média também é normalizada. O resultado final é um embedding representativo da pose.

Ao final das três poses, o sistema:

1. cria ou atualiza o usuário no banco;
2. salva uma amostra criptografada para cada pose;
3. atualiza o cache;
4. limpa a janela temporal;
5. exibe mensagem de sucesso.

Se não conseguir capturar amostras suficientes dentro do tempo limite, o cadastro falha e uma mensagem clara é exibida.

## 8. Fluxo de exclusão de usuário

A exclusão é feita pelo botão "Excluir pessoa". O sistema abre uma janela própria com:

- campo de busca por código do cartão;
- lista de usuários cadastrados;
- campo preenchido automaticamente ao selecionar um usuário;
- botão "Excluir usuário selecionado".

Cada item da lista mostra o nome e o código do cartão. A busca filtra os usuários conforme o código digitado.

Antes de excluir, o sistema mostra uma confirmação com o nome e o código do cartão. Após a confirmação, a exclusão é feita pelo código do cartão, não pelo nome. O banco remove o usuário e todos os embeddings relacionados.

Após excluir, o sistema:

- atualiza o cache;
- atualiza o status do banco;
- limpa usuário reconhecido no painel;
- atualiza a lista visual;
- mostra mensagem de sucesso.

## 9. Interface gráfica

A interface principal é implementada em Tkinter e organizada em três áreas.

### 9.1 Cabeçalho

O cabeçalho superior mostra:

- título do sistema;
- subtítulo informando que é um módulo único de reconhecimento facial local;
- status geral, como câmera ativa, banco conectado e reconhecimento em execução.

### 9.2 Área central da câmera

A área central exibe o feed da webcam. O vídeo é capturado pelo OpenCV, convertido para RGB, transformado em imagem Pillow e exibido em um `Label` do Tkinter.

O sistema desenha overlays diretamente no frame:

- caixa delimitadora do rosto;
- pose detectada;
- status do reconhecimento;
- qualidade do frame;
- quantidade de frames na janela temporal;
- mensagens de orientação.

A área da câmera se ajusta ao espaço disponível na janela. Ao reduzir a largura, a barra lateral pode ser ocultada automaticamente para preservar a câmera visível.

### 9.3 Painel lateral direito

O painel lateral contém ações do operador:

- cadastrar pessoa;
- excluir pessoa;
- perguntar antes de digitar o código do cartão;
- digitar Enter automaticamente;
- fechar app.

Também possui painel de status com:

- câmera;
- banco;
- usuário reconhecido;
- código do cartão;
- pose;
- distância média;
- frames válidos;
- digitação;
- operação atual.

Visualmente, a coluna lateral usa a mesma cor do cabeçalho, com textos claros para contraste. Os botões principais usam cores pastéis para facilitar a leitura e manter aparência adequada para apresentação acadêmica.

## 10. Digitação automática do código do cartão

A classe `VirtualKeyboardTyper` implementa a digitação automática via `pynput`. Quando um reconhecimento é confirmado, o sistema pode digitar o código do cartão em outro campo selecionado no sistema operacional.

Existem dois controles principais:

- "Perguntar antes de digitar o código do cartão";
- "Digitar Enter automaticamente".

Quando a opção de perguntar antes está desmarcada, o sistema digita o código diretamente no campo que estiver focado no momento da confirmação do reconhecimento.

Quando a opção de perguntar antes está marcada, o sistema abre um pop-up de confirmação. O operador confirma, clica no campo de destino e o sistema digita o código após um pequeno atraso, permitindo que o foco do mouse seja aplicado corretamente.

A opção de Enter define se o sistema deve pressionar Enter depois de digitar o código. Essa configuração é aplicada tanto no modo direto quanto no modo com confirmação.

Para evitar repetição indevida, há três controles:

- cooldown entre digitações;
- limite de digitação consecutiva por pessoa;
- reset da sessão quando o sistema volta a aguardar ou perde o rosto.

Atualmente, o limite configurado é de uma digitação por pessoa enquanto ela continua sendo reconhecida. Se a pessoa sai da câmera, fica irreconhecível ou o sistema volta para "aguardando", o estado interno é zerado. Assim, se a mesma pessoa aparecer novamente depois, o código pode ser digitado outra vez.

## 11. Segurança e privacidade

O sistema armazena os dados localmente. Não há envio de imagens ou embeddings para servidores externos.

Os embeddings faciais são criptografados antes de serem salvos no banco. Isso melhora a proteção dos dados biométricos armazenados. Mesmo assim, alguns cuidados são importantes:

- proteger o arquivo `data/faces.db`;
- proteger o arquivo `data/fernet.key`;
- não publicar a chave Fernet junto com dados reais;
- fazer backup conjunto do banco e da chave se for necessário migrar de computador;
- obter autorização dos participantes cadastrados, pois reconhecimento facial envolve dados biométricos.

O sistema salva embeddings, não imagens faciais brutas. Ainda assim, embeddings faciais são informações sensíveis.

## 16. Limitações atuais

O sistema é adequado como protótipo local e acadêmico, mas possui limitações:

- não possui prova de vida avançada;
- não impede ataques com foto ou vídeo;
- depende de iluminação razoável;
- thresholds podem precisar de calibração conforme câmera e ambiente;
- a estimativa de pose é heurística, não 3D;
- o reconhecimento pode ficar mais lento em CPU;
- a interface e a lógica estão concentradas em um único arquivo grande, embora divididas internamente por classes.

Essas limitações não invalidam o projeto, mas devem ser reconhecidas no TCC como pontos de melhoria e escopo.

## 17. Possíveis expansões futuras

A estrutura atual permite expansão em algumas direções:

- dividir `face_recognition_app.py` em módulos separados;
- adicionar tela de calibração de thresholds;
- criar relatório de eventos de reconhecimento;
- implementar backup e restauração do banco e da chave Fernet;
- adicionar autenticação do operador;
- melhorar estimativa de pose com métodos geométricos mais robustos;
- implementar detecção de vivacidade;
- adicionar logs estruturados para auditoria;
- criar testes automatizados para banco, validação de código, votação temporal e digitação;
- permitir exportação segura de usuários cadastrados.

## 18. Conclusão

O sistema desenvolvido combina visão computacional, redes neurais pré-treinadas, banco de dados local, criptografia e interface gráfica desktop. A aplicação usa MTCNN para detectar rostos e landmarks, InceptionResnetV1 pré-treinado no VGGFace2 para gerar embeddings faciais e votação temporal para confirmar reconhecimentos com maior estabilidade.

O cadastro multi-pose melhora a robustez em situações em que o rosto não está perfeitamente de frente. A criptografia dos embeddings aumenta a segurança dos dados armazenados. A interface Tkinter organiza o uso do sistema de forma clara para o operador, e a digitação automática via `pynput` permite integrar o reconhecimento facial a outros sistemas sem necessidade de API.

Como resultado, o projeto atende ao objetivo de um sistema local de reconhecimento facial para identificação por código do cartão, com funcionamento em webcam, persistência segura dos dados e fluxo adequado para demonstração acadêmica.
