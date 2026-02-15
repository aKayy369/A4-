import dash
from dash import dcc, html, Input, Output, State
import torch
import torch.nn as nn

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CONFIG =================
max_len = 128
hidden_dim = 256

# ================= LOAD VOCAB =================
word2id = torch.load("word2id.pth", map_location=device)
vocab_size = len(word2id)

# ================= ATTENTION MASK =================
def get_attn_pad_mask(seq):
    return seq.eq(0).unsqueeze(1).expand(seq.size(0), seq.size(1), seq.size(1))

# ================= EMBEDDING =================
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.seg = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        pos = torch.arange(x.size(1), device=device).unsqueeze(0)
        return self.norm(self.tok(x) + self.pos(pos) + self.seg(seg))

# ================= CUSTOM ATTENTION =================
class ScaledDotAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size = x.size(0)

        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context = ScaledDotAttention(self.d_k)(Q, K, V, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        return self.fc(context)

# ================= ENCODER =================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mask):
        x = x + self.attn(x, mask)
        x = x + self.ff(x)
        return x

# ================= BERT =================
class BERT(nn.Module):
    def __init__(self, vocab_size, max_len):
        super().__init__()
        d_model = 256
        n_heads = 4
        n_layers = 4

        self.embed = Embedding(vocab_size, max_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

        # heads kept for loading weights
        self.mlm = nn.Linear(d_model, vocab_size)
        self.nsp = nn.Linear(d_model, 2)

    def forward(self, input_ids, segment_ids):
        x = self.embed(input_ids, segment_ids)
        mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ================= POOLING =================
def mean_pool(token_embeddings, input_ids):
    mask = (input_ids != 0).unsqueeze(-1)
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1)
    return summed / counts

# ================= SBERT =================
class SentenceBERT(nn.Module):
    def __init__(self, bert, hidden_dim):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_dim * 3, 3)

    def encode(self, input_ids):
        segment_ids = torch.zeros_like(input_ids).to(device)
        x = self.bert(input_ids, segment_ids)
        return mean_pool(x, input_ids)

    def forward(self, a_ids, b_ids):
        u = self.encode(a_ids)
        v = self.encode(b_ids)
        uv = torch.abs(u - v)
        x = torch.cat([u, v, uv], dim=-1)
        return self.classifier(x)

# ================= LOAD TRAINED MODELS =================
bert = BERT(vocab_size, max_len).to(device)
bert.load_state_dict(torch.load("bert_task1_weights.pth", map_location=device))

model = SentenceBERT(bert, hidden_dim).to(device)
model.load_state_dict(torch.load("sbert_finetuned_model.pth", map_location=device))
model.eval()

# ================= TOKENIZATION =================
def encode_sentence(sentence):
    words = sentence.lower().split()
    ids = [word2id.get(w, 0) for w in words]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return ids

# ================= PREDICTION =================
def predict_nli(premise, hypothesis):
    a_ids = torch.tensor([encode_sentence(premise)]).to(device)
    b_ids = torch.tensor([encode_sentence(hypothesis)]).to(device)

    with torch.no_grad():
        outputs = model(a_ids, b_ids)
        pred = torch.argmax(outputs, dim=1).item()

    labels = ["Entailment", "Neutral", "Contradiction"]
    return labels[pred]

# ================= DASH APP =================
app = dash.Dash(__name__)

app.layout = html.Div(

    style={
        "background": "linear-gradient(135deg, #667eea, #764ba2)",
        "height": "100vh",
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "fontFamily": "Arial"
    },

    children=[

        html.Div(

            style={
                "background": "white",
                "padding": "40px",
                "borderRadius": "15px",
                "boxShadow": "0px 10px 30px rgba(0,0,0,0.2)",
                "width": "700px",
                "textAlign": "center"
            },

            children=[

                html.H1(" Natural Language Inference "),

                html.Label("Premise"),
                dcc.Input(id="premise", type="text", style={"width": "100%"}),

                html.Br(), html.Br(),

                html.Label("Hypothesis"),
                dcc.Input(id="hypothesis", type="text", style={"width": "100%"}),

                html.Br(), html.Br(),

                html.Button("Predict", id="predict-btn"),

                html.H3(id="output")
            ]
        )
    ]
)

@app.callback(
    Output("output", "children"),
    Input("predict-btn", "n_clicks"),
    State("premise", "value"),
    State("hypothesis", "value"),
)
def update_output(n_clicks, premise, hypothesis):
    if n_clicks is None:
        return ""

    result = predict_nli(premise, hypothesis)
    return f"Prediction: {result}"

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
