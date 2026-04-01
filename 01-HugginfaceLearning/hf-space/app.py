# app.py
import gradio as gr

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    
    text_lower = text.lower()
    positive_words = ["good", "great", "happy", "love", "awesome", "excellent", "wonderful"]
    negative_words = ["bad", "hate", "sad", "terrible", "awful", "horrible", "worst"]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        return f"Positive  (score: +{pos_count})"
    elif neg_count > pos_count:
        return f"Negative  (score: -{neg_count})"
    else:
        return "Neutral"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter your text", placeholder="Type something..."),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="Simple Sentiment Analyzer",
    description="Type a sentence and I'll tell you if it's Positive, Negative, or Neutral.",
    examples=[
        ["I love this, it's great!"],
        ["This is terrible and awful."],
        ["The weather is okay today."],
    ]
)

if __name__ == "__main__":
    demo.launch()