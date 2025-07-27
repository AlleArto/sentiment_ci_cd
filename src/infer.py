"""Pipeline di inferenza + demo Gradio."""
from transformers import pipeline
from .config import HF_REPO_ID

def sentiment_pipeline():
    return pipeline("sentiment-analysis", model=HF_REPO_ID)

def predict(text: str):
    pipe = sentiment_pipeline()
    out = pipe(text)[0]
    return f"{out['label']} ({out['score']:.3f})"

# --- Gradio demo ---
def demo_gradio():
    import gradio as gr

    with gr.Blocks() as demo:
        gr.Markdown("# Twitter Sentiment (fine‑tuned)")
        inp = gr.Textbox(lines=3, placeholder="Scrivi un tweet…")
        out = gr.Textbox()
        btn = gr.Button("Analizza")
        btn.click(lambda t: predict(t), inp, out)
    demo.launch()
