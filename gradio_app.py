import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

# ── Data ──────────────────────────────────────────────────────────────────────
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ── Vector DB ─────────────────────────────────────────────────────────────────
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding_model)

# ── Recommendation Logic ───────────────────────────────────────────────────────
def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=50)

    books_list = []
    for doc in recs:
        lines = doc.page_content.strip('"').split("\n")
        for line in lines:
            line = line.strip()
            if line:
                try:
                    isbn = int(line.split()[0])
                    books_list.append(isbn)
                except:
                    continue

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return []

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_split[0]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# ── UI ────────────────────────────────────────────────────────────────────────
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,400;1,600&family=Jost:wght@300;400;500;600&display=swap');

/* ── Global ── */
* { box-sizing: border-box; }

.gradio-container {
    background: #0d0e14 !important;
    font-family: 'Jost', sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── Hero ── */
.bk-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}

.bk-eyebrow {
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #d4a847;
    margin-bottom: 0.8rem;
    font-weight: 500;
}

.bk-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 4rem;
    font-weight: 300;
    color: #edeaf3;
    line-height: 1;
    letter-spacing: -1px;
    margin-bottom: 1rem;
}

.bk-title em {
    color: #d4a847;
    font-style: italic;
    font-weight: 600;
}

.bk-rule {
    width: 140px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d4a847, transparent);
    margin: 1.2rem auto 1rem;
}

.bk-sub {
    color: #7a7890;
    font-size: 0.95rem;
    font-weight: 300;
    line-height: 1.7;
    max-width: 420px;
    margin: 0 auto;
}

/* ── Search card ── */
.bk-card {
    background: #13141a !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px !important;
    padding: 1.8rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5) !important;
}

/* ── All input/textarea/select elements ── */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #1c1e27 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #edeaf3 !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 300 !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: rgba(212,168,71,0.5) !important;
    box-shadow: 0 0 0 3px rgba(212,168,71,0.07) !important;
    outline: none !important;
}

/* ── Labels ── */
.gradio-container label > span,
.gradio-container .label-wrap span {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #7a7890 !important;
}

/* ── Button ── */
.gradio-container button.primary,
.gradio-container button[variant="primary"] {
    background: linear-gradient(135deg, #d4a847, #b87d30) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #080810 !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(212,168,71,0.3) !important;
    transition: all 0.2s ease !important;
    min-height: 44px !important;
}

.gradio-container button.primary:hover,
.gradio-container button[variant="primary"]:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(212,168,71,0.45) !important;
}

/* ── Divider ── */
.bk-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0 0 1.5rem;
}

.bk-divider-line {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

.bk-divider-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-style: italic;
    color: #d4a847;
    white-space: nowrap;
}

/* ── Gallery ── */
.gradio-container .gallery-item,
.gradio-container [data-testid="gallery"] > div > div {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.gradio-container .gallery-item:hover,
.gradio-container [data-testid="gallery"] > div > div:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(212,168,71,0.2) !important;
}

/* ── Footer ── */
.bk-footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 2rem;
    color: #4a4860;
    font-size: 0.73rem;
    letter-spacing: 0.08em;
    border-top: 1px solid rgba(255,255,255,0.05);
}

footer, .show-api, .built-with { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as dashboard:

    # Hero
    gr.HTML("""
    <div class="bk-hero">
        <div class="bk-eyebrow">✦ &nbsp; Semantic Discovery &nbsp; ✦</div>
        <h1 class="bk-title">Book<em>Sense</em></h1>
        <div class="bk-rule"></div>
        <p class="bk-sub">Describe a feeling, a world, or a story —<br>
        and we'll find books that match the meaning behind your words.</p>
    </div>
    """)

    # Search card
    with gr.Group(elem_classes="bk-card"):
        with gr.Row(equal_height=True):
            user_query = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="e.g., A haunting story about grief and memory in post-war Japan...",
                lines=3,
                scale=3,
            )
            with gr.Column(scale=1, min_width=180):
                category_dropdown = gr.Dropdown(
                    choices=categories,
                    label="Category",
                    value="All",
                )
                tone_dropdown = gr.Dropdown(
                    choices=tones,
                    label="Emotional Tone",
                    value="All",
                )
                submit_button = gr.Button(
                    "✦  Discover Books",
                    variant="primary",
                )

    # Divider
    gr.HTML("""
    <div class="bk-divider">
        <div class="bk-divider-line"></div>
        <div class="bk-divider-label">Curated for you</div>
        <div class="bk-divider-line"></div>
    </div>
    """)

    # Gallery
    output = gr.Gallery(
        label="",
        columns=8,
        rows=2,
        height=520,
        object_fit="cover",
        show_label=False,
    )

    # Footer
    gr.HTML("""
    <div class="bk-footer">
        BookSense &nbsp;·&nbsp; Semantic Recommendations &nbsp;·&nbsp; Powered by all-MiniLM-L6-v2
    </div>
    """)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()