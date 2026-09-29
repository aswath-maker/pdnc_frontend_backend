# Disease NER with an embedding projection head

Our entry for the **PDNC hackathon, Problem Statement 02**: find disease mentions in biomedical text and tag them in BIO format (`O`, `B-Disease`, `I-Disease`).

We fine-tuned two taggers, a SapBERT token classifier and a SmolLM generative tagger, and added a small **projection head** that checks every span a tagger predicts. The head embeds the span with SapBERT, projects it into a space where disease names cluster together, and drops spans that don't look like diseases. Because it scores meaning rather than memorised strings, it's designed to judge disease names the taggers never saw in training.

The repo also includes a web app for trying the models: tag your own text, score a test set, and compare results with and without the head.

## Models

All three models are published on Hugging Face.

| Model | What it does | Size |
|---|---|---|
| [sapBERT_NER_finetune](https://huggingface.co/Aswath220/sapBERT_NER_finetune) | SapBERT fine-tuned as a BIO token classifier | ~0.1B |
| [biomed_ner_smollm](https://huggingface.co/Aswath220/biomed_ner_smollm) | SmolLM fine-tuned on NCBI-disease to list the disease mentions in a sentence, which are then mapped back to BIO tags | ~0.4B |
| [sapbert-disease-projection-head](https://huggingface.co/Aswath220/sapbert-disease-projection-head) | Projection head over frozen SapBERT (layer 8) that scores whether a span is a disease | 213k params |

## How the head works

1. A tagger (SapBERT or SmolLM) predicts disease spans for a sentence.
2. For each predicted span, the head takes the frozen SapBERT representation from layer 8 and passes it through the projection layer to get a disease score.
3. Spans that score below a threshold are removed. The thresholds come from our threshold study: **0.526 for SmolLM** and **0.4 for SapBERT**.

The head only removes spans; it never adds new ones. So its job is to raise precision without costing too much recall.

## Results on NCBI-disease test (940 sentences)

| Model | Span P | Span R | Span F1 | Token macro-F1 | Token acc |
|---|---|---|---|---|---|
| SmolLM | 0.8187 | 0.8656 | 0.8415 | 0.9042 | 0.9794 |
| SmolLM + head | 0.8502 | 0.8510 | **0.8506** | 0.9081 | 0.9807 |
| SapBERT | 0.8502 | 0.8990 | 0.8739 | 0.9318 | 0.9858 |
| SapBERT + head | 0.8532 | 0.8958 | **0.8740** | 0.9312 | 0.9857 |

Span scores use strict entity-level matching: a prediction counts only if its start and end both match the gold span exactly. Token macro-F1 averages F1 across the three BIO labels.

The head improves SmolLM's span F1 by **+0.009** (paired bootstrap over sentences, 95% CI +0.001 to +0.017), so the gain is unlikely to be noise. On SapBERT the difference is negligible. SapBERT is already the more precise tagger, so it leaves the head less to fix.

## Web app

```
Browser (Vercel)  ──HTTPS──▶  Cloudflare tunnel  ──▶  Google Colab (FastAPI + T4 GPU)
                                                       SapBERT tagger, SmolLM tagger, projection head
```

The frontend is a single static page hosted on Vercel. The models run in a Colab notebook on a free GPU, served by FastAPI and exposed to the internet through a Cloudflare tunnel (ngrok also works).

What it does:

- **Tag your own text.** Type sentences and see them highlighted, with the BIO tag under every word. You can copy the tags as `word<TAB>tag` lines.
- **Score a test set.** Run on the built-in NCBI-disease test set, or upload a CSV. Labelled files show all five metrics and the bootstrap result. Unlabelled files show span counts and which sentences the head changed.
- **Toggle model and head.** Switch between SapBERT and SmolLM, and between with and without the head. Each run computes both head variants, so the head toggle switches instantly.
- **Inspect changes.** See the sentences where the head removed spans, with removed spans struck through and gold spans underlined.
- **Download predictions.** Save a submission CSV in `id,ner_tags` format.

### Running it

**1. Start the backend in Colab**

1. Open `pdnc_backend_colab.ipynb` in Google Colab.
2. Go to Runtime > Change runtime type, and choose **T4 GPU**.
3. Click Runtime > Run all. The first load takes a few minutes.
4. Copy the URL printed under "Start the server". It ends in `trycloudflare.com`.

If your network blocks Cloudflare tunnels, set `TUNNEL = "ngrok"` in the settings cell and add a free ngrok auth token.

**2. Deploy the frontend on Vercel**

1. Import this repo in Vercel (Add New > Project).
2. Set the framework preset to **Other**. There's no build step, and the root directory is the folder that holds `index.html`.
3. Open the deployed site, paste the backend URL into the connection box, and click Connect.

The Colab URL changes every time the runtime restarts. To skip pasting it, put your Vercel URL in `FRONTEND_URL` in the notebook's settings cell; the notebook then prints a link that opens the app already connected. You can also run the frontend locally by opening `index.html` in a browser.

### CSV format

| Column | Required | Format |
|---|---|---|
| `tokens` | yes (or `text`) | Python-style list, e.g. `['Breast', 'cancer', 'risk']` |
| `text` | if no `tokens` | Plain sentence, split on whitespace |
| `id` | no | Any value; defaults to the row number |
| `ner_tags` | no | List of `O` / `B-Disease` / `I-Disease`, or `0` / `1` / `2`, one per token |

With `ner_tags`, the app computes metrics. Without it, the app only predicts.

### API

The Colab backend exposes these endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | GPU name, loaded models and thresholds |
| `POST /jobs` | Start a run: form fields `model` (`smollm` or `sapbert`) plus either `dataset=ncbi` or a `file` |
| `GET /jobs/{id}` | Job status, progress and results |
| `GET /jobs/{id}/download?variant=base\|head` | Submission CSV |
| `POST /tag` | Tag free text: JSON `{"text": "...", "model": "sapbert"}` |

File runs are queued and processed one at a time on the GPU. Results are cached, so rerunning the same file with the same model returns immediately.

## Repository layout

```
├── all_test.ipynb              # evaluation: both taggers, with and without the head, plus the bootstrap
├── pdnc_backend_colab.ipynb    # Colab backend for the web app
└── pdnc-frontend/
    └── index.html              # web app (static, deploys to Vercel)
```

## Limitations

- The hackathon's final test set is synthetic and generated by the organisers, so scores on it may differ from the NCBI numbers above.
- The head can occasionally drop a real disease. For example, SmolLM + head removed "von Willebrand disease type B" on the hackathon test set. The threshold trades a little recall for precision.
- The free Colab backend disconnects after a period of inactivity or when it hits its session limit, and its URL changes on every restart. That's fine for a demo but not for permanent hosting.

## Acknowledgements

- [SapBERT](https://github.com/cambridgeltl/sapbert) (Liu et al., 2021) for the biomedical entity encoder
- [SmolLM](https://huggingface.co/HuggingFaceTB) from Hugging Face
- [NCBI-disease corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) (Doğan et al., 2014), used via the [CoNLL version](https://github.com/spyysalo/ncbi-disease)
