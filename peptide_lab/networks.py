"""Two compact neural networks: a sequence classifier and an LSTM VAE."""
import torch
from torch import nn
from .config import START, END

class PeptideCNN(nn.Module):
    def __init__(self, filters=32, dropout=.3):
        super().__init__()
        self.embedding = nn.Embedding(21, 24, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(24, filters, k, padding=k//2) for k in (3, 5, 7)])
        self.head = nn.Sequential(nn.Linear(filters*3 + 25, 48), nn.ReLU(), nn.Dropout(dropout), nn.Linear(48, 1))

    def forward(self, tokens, descriptors):
        x = self.embedding(tokens).transpose(1, 2)
        mask = tokens.eq(0).unsqueeze(1)
        pooled = [torch.relu(conv(x)).masked_fill(mask, -1e4).amax(dim=2) for conv in self.convs]
        return self.head(torch.cat(pooled + [descriptors], dim=1)).squeeze(1)

class PeptideVAE(nn.Module):
    def __init__(self, embedding=32, hidden=96, latent=24, word_dropout=.2):
        super().__init__()
        self.latent_dim = latent
        self.word_dropout = word_dropout
        self.embedding = nn.Embedding(23, embedding, padding_idx=0)
        self.encoder = nn.LSTM(embedding, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.init_h = nn.Linear(latent, hidden)
        self.init_c = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(embedding + latent, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 23)

    def encode(self, tokens):
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, tokens.ne(0).sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.encoder(packed)
        return self.mu(h[-1]), self.logvar(h[-1]).clamp(-8, 8)

    def decode(self, tokens, z, state=None):
        if state is None:
            state = (torch.tanh(self.init_h(z)).unsqueeze(0), torch.tanh(self.init_c(z)).unsqueeze(0))
        z_steps = z.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        out, state = self.decoder(torch.cat([self.embedding(tokens), z_steps], dim=2), state)
        return self.output(out), state

    def forward(self, inputs):
        mu, logvar = self.encode(inputs)
        z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar) if self.training else mu
        decoder_inputs = inputs.clone()
        if self.training:
            # Word dropout encourages latent use; diagnostics must verify its effect.
            drop = torch.rand_like(inputs.float()).lt(self.word_dropout) & inputs.ne(0) & inputs.ne(START)
            decoder_inputs[drop] = 0
        logits, _ = self.decode(decoder_inputs, z)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, count, seed=42, temperature=.9, max_length=60):
        self.eval()
        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(count, self.latent_dim, generator=generator)
        token = torch.full((count, 1), START, dtype=torch.long)
        state, sequences = None, [[] for _ in range(count)]
        finished = torch.zeros(count, dtype=torch.bool)
        from .config import AA
        for position in range(max_length + 1):
            logits, state = self.decode(token, z, state)
            logits = logits[:, 0] / temperature
            logits[:, [0, START]] = -1e9
            if position < 5:
                logits[:, END] = -1e9
            token = torch.multinomial(torch.softmax(logits, 1), 1, generator=generator)
            for i, value in enumerate(token[:, 0].tolist()):
                if not finished[i] and value != END and position < max_length:
                    sequences[i].append(AA[value - 1])
            finished |= token[:, 0].eq(END)
            if finished.all():
                break
        return ["".join(s) for s in sequences], finished.tolist()
