# Towards Automatic Generation of Fine-Grained Source Code Templates
This repository contains the dataset and code for our work: **Towards Automatic Generation of Fine-Grained Source Code Templates**.


### General Info
- **data** folder contains the dataset and folds information.
- **github** folder contains the scripts to mine raw projects from GitHub.
- **parser** folder contains scripts to parse the raw dataset.
- **tokenizer** folder contains a general-purpose tokenizer.
- **models** folder contains model implementations and other utile scripts.
- Rest of the scripts provide related functionalities.



### Models Info

The following hyperparameters are used for our implementations:

#### N-gram
- n = [3; 5; 7; 10]

#### RNN

The overall model structure is as under;

recurrentModels(
  (embed): Embedding(502, 300)
  (gru|lstm): GRU|LSTM(300, 300, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.5, inplace=False)
  (linear): Linear(in_features=300, out_features=502, bias=True)
)

Following hyperparameters are utilized for RNN based models;

- Max Epochs = 20
- embedding & hidden units = 300
- dropout = 0.5
- lr = 0.001
- vocab = 502
- beam size = [1,3,5,7]
- topK = [1,3,5]

#### GPT2

Following hyperparameters are utilized for RNN based models;

- Max Epochs = 20
- n_layer = 6
- n_embd = 300
- n_head = 6
- layer_norm_epsilon = 1e-6,
- lr = 1e-3
- dropout = 0.5
- vocab = 502



#### RLSTM

Following hyperparameters are utilized for RNN based models;

- Max Epochs = 20
- embedding & hidden units = 300
- n_layers = 6
- n_heads = 6
- lr = 0.001
- dropout = 0.05
- vocab = 502

we refer the readers to our paper for more details.

### General approach and results

###### General Framework
<br><br><img src="https://github.com/yaxirhuxxain/Template-Generation/blob/main/images/framework.png" width="70%" style="margin: auto;">

###### Model Architecture
<br><br><img src="https://github.com/yaxirhuxxain/Template-Generation/blob/main/images/architecture.png" width="25%" style="margin: auto;">

##### Top-k Beam Search (LSTM)
<br><br><img src="https://github.com/yaxirhuxxain/Template-Generation/blob/main/images/beamsearch.png" width="70%" style="margin: auto;">


##### Automatic Generation example With RLSTM (Example 1)

<pre>
if
├── ('(', 14.078824996948242)
│   ├── ('!', 6.605443954467773)
│   │   ├── ('(', 7.305898666381836)
│   │   │   ├── ('&LT;idf&GT;', 7.073834419250488)
│   │   │   ├── ('o', 7.187504768371582)
│   │   │   └── ('other', 7.514697551727295)
│   │   ├── ('&LT;idf&GT;', 9.010215759277344)
│   │   │   ├── ('(', 7.722522258758545)
│   │   │   ├── (')', 9.553977966308594)
│   │   │   │   ├── ('extends', 4.182424545288086)
│   │   │   │   ├── ('implements', 5.645623683929443)
│   │   │   │   └── ('{', 13.302828788757324)
│   │   │   └── ('.', 9.740118026733398)
│   │   │       ├── ('&LT;idf&GT;', 9.738948822021484)
│   │   │       ├── ('hasNext', 6.0530924797058105)
│   │   │       └── ('isEmpty', 9.61611557006836)
│   │   └── ('this', 5.561826705932617)
│   │       ├── ('(', 6.0947065353393555)
│   │       ├── (')', 6.659730911254883)
│   │       └── ('.', 11.770003318786621)
│   ├── ('&LT;idf&GT;', 8.549917221069336)
│   │   ├── ('!=', 8.935043334960938)
│   │   │   ├── ('&LT;idf&GT;', 7.5662360191345215)
│   │   │   ├── ('IntLiteral', 7.1024250984191895)
│   │   │   └── ('NullLiteral', 10.63170051574707)
│   │   │       ├── ('&&', 4.599068641662598)
│   │   │       ├── (')', 13.275740623474121)
│   │   │       │   ├── (',', 3.8449294567108154)
│   │   │       │   ├── ('throws', 3.9283547401428223)
│   │   │       │   └── ('{', 13.518827438354492)
│   │   │       └── (',', 5.644671440124512)
│   │   ├── ('.', 8.362428665161133)
│   │   │   ├── ('&LT;idf&GT;', 9.711922645568848)
│   │   │   │   ├── ('!=', 7.069042205810547)
│   │   │   │   ├── ('(', 11.226736068725586)
│   │   │   │   └── ('==', 6.921487808227539)
│   │   │   ├── ('equals', 7.482246398925781)
│   │   │   └── ('isEmpty', 8.441835403442383)
│   │   │       ├── ('(', 12.850786209106445)
│   │   │       ├── ('<', 4.745397090911865)
│   │   │       └── ('==', 4.797292709350586)
│   │   └── ('==', 8.562299728393555)
│   │       ├── ('&LT;idf&GT;', 7.947545051574707)
│   │       ├── ('IntLiteral', 7.457759857177734)
│   │       └── ('NullLiteral', 9.594560623168945)
│   │           ├── (')', 13.382678985595703)
│   │           │   ├── ('implements', 3.9990921020507812)
│   │           │   ├── ('throws', 4.432741165161133)
│   │           │   └── ('{', 13.861785888671875)
│   │           ├── (',', 5.074149131774902)
│   │           └── ('.', 5.087129592895508)
│   └── ('this', 6.889849662780762)
│       ├── ('!=', 7.022278785705566)
│       │   ├── ('o', 8.379844665527344)
│       │   ├── ('obj', 6.610993385314941)
│       │   └── ('other', 7.150498390197754)
│       ├── ('.', 10.927289009094238)
│       │   ├── ('&LT;idf&GT;', 8.909322738647461)
│       │   │   ├── ('!=', 9.208897590637207)
│       │   │   ├── (')', 8.078985214233398)
│       │   │   └── ('==', 8.87732219696045)
│       │   ├── ('context', 5.777798652648926)
│       │   └── ('headers', 4.372982978820801)
│       └── ('==', 9.836767196655273)
│           ├── ('o', 9.827459335327148)
│           │   ├── ('!=', 4.471717834472656)
│           │   ├── (')', 12.871601104736328)
│           │   │   ├── (')', 3.789093017578125)
│           │   │   ├── ('throws', 3.495553493499756)
│           │   │   └── ('{', 11.649490356445312)
│           │   └── ('.', 4.454247951507568)
│           ├── ('obj', 7.622432708740234)
│           └── ('other', 7.833925724029541)
├── ('<', 4.476499080657959)
│   ├── ('&LT;idf&GT;', 8.211740493774414)
│   ├── ('IntLiteral', 9.276811599731445)
│   │   ├── ('-', 5.6372480392456055)
│   │   ├── (';', 5.720039367675781)
│   │   └── ('{', 5.424778461456299)
│   └── ('this', 5.1236467361450195)
└── ('[', 4.347280979156494)
    ├── ('&LT;idf&GT;', 8.065448760986328)
    ├── ('i', 6.747056007385254)
    └── ('this', 5.707931041717529)
</pre>


##### Automatic Generation example With RLSTM (Example 2)

<pre>
int
├── ('&LT;idf&GT;', 9.134084701538086)
│   ├── ('(', 9.053149223327637)
│   │   ├── (')', 9.523963928222656)
│   │   ├── ('&LT;idf&GT;', 7.676253318786621)
│   │   └── ('String', 6.598589897155762)
│   ├── (';', 8.536705017089844)
│   └── ('=', 11.452676773071289)
│       ├── ('&LT;idf&GT;', 8.203071594238281)
│       ├── ('IntLiteral', 8.34068775177002)
│       └── ('this', 5.02174186706543)
├── ('count', 5.708941459655762)
│   ├── ('(', 6.719473361968994)
│   ├── (';', 8.919132232666016)
│   └── ('=', 11.442024230957031)
│       ├── ('&LT;idf&GT;', 8.14459228515625)
│       ├── ('IntLiteral', 8.767810821533203)
│       └── ('this', 5.030771255493164)
└── ('i', 5.902296543121338)
    ├── ('(', 6.6471171379089355)
    ├── (';', 9.344289779663086)
    └── ('=', 11.106783866882324)
        ├── ('&LT;idf&GT;', 7.567047119140625)
        ├── ('IntLiteral', 8.861430168151855)
        └── ('this', 5.267561912536621)
</pre>
<br>
