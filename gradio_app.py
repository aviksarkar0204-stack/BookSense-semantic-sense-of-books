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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400;1,600&family=Jost:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #0b0c10;
    --surface:   #13141a;
    --card:      #1c1e27;
    --border:    rgba(255,255,255,0.07);
    --gold:      #d4a847;
    --gold-dim:  rgba(212,168,71,0.15);
    --text:      #edeaf3;
    --muted:     #7a7890;
    --radius:    14px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Jost', sans-serif !important;
    color: var(--text) !important;
}

/* Ambient glow background */
.gradio-container {
    background:
        radial-gradient(ellipse 70% 40% at 15% 5%, rgba(212,168,71,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 85% 85%, rgba(91,188,176,0.05) 0%, transparent 55%),
        #0b0c10 !important;
}

/* ── Hero ── */
#booksense-hero {
    text-align: center;
    padding: 4.5rem 2rem 3rem;
    position: relative;
}

#booksense-hero::after {
    content: '';
    display: block;
    width: 180px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 2.5rem auto 0;
}

.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3.5rem, 8vw, 6rem);
    font-weight: 300;
    line-height: 1;
    color: var(--text);
    margin-bottom: 1.2rem;
    letter-spacing: -2px;
}

.hero-title em {
    color: var(--gold);
    font-style: italic;
    font-weight: 600;
}

.hero-sub {
    font-size: 0.97rem;
    font-weight: 300;
    color: var(--muted);
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.8;
}

/* ── Search card ── */
#search-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 2rem 2.2rem !important;
    margin: 0 auto 3rem !important;
    box-shadow:
        0 25px 70px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.04) !important;
    position: relative;
}

#search-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,168,71,0.4), transparent);
}

/* Labels */
label > span {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* Inputs */
textarea, input[type="text"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

textarea:focus, input:focus {
    border-color: rgba(212,168,71,0.45) !important;
    box-shadow: 0 0 0 3px rgba(212,168,71,0.07) !important;
    outline: none !important;
}

/* Dropdowns */
select {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Button */
#find-btn {
    background: linear-gradient(135deg, #d4a847, #b87d30) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #080810 !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 1.5rem !important;
    width: 100% !important;
    margin-top: 1.7rem !important;
    cursor: pointer !important;
    box-shadow: 0 6px 24px rgba(212,168,71,0.3) !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
}

#find-btn:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 36px rgba(212,168,71,0.45) !important;
}

#find-btn:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Results divider */
#results-divider {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: 0 auto 1.8rem;
    padding: 0 0.5rem;
}

.div-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}

.div-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-style: italic;
    color: var(--gold);
    font-weight: 400;
    white-space: nowrap;
}

/* Gallery */
.gr-gallery { background: transparent !important; border: none !important; }

.gallery-item {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    transition: transform 0.35s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s ease !important;
}

.gallery-item:hover {
    transform: translateY(-8px) scale(1.025) !important;
    box-shadow:
        0 24px 60px rgba(0,0,0,0.7),
        0 0 0 1px rgba(212,168,71,0.25) !important;
}

.gallery-item img {
    object-fit: cover !important;
    width: 100% !important;
    height: 100% !important;
}

/* Footer */
#booksense-footer {
    text-align: center;
    padding: 2rem 1rem 3rem;
    color: var(--muted);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

footer, .show-api { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as dashboard:

    gr.HTML("""
    <div id="booksense-hero">
        <div class="hero-eyebrow">✦ &nbsp; Semantic Discovery &nbsp; ✦</div>
        <h1 class="hero-title">Book<em>Sense</em></h1>
        <p class="hero-sub">
            Describe a feeling, a world, or a story —<br>
            and we'll find books that match the meaning behind your words.
        </p>
    </div>
    """)

    with gr.Group(elem_id="search-panel"):
        with gr.Row():
            user_query = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="e.g., A haunting story about grief and memory in post-war Japan...",
                lines=3,
                scale=3,
            )
            with gr.Column(scale=1, min_width=210):
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
                    elem_id="find-btn",
                )

    gr.HTML("""
    <div id="results-divider">
        <div class="div-line"></div>
        <div class="div-label">Curated for you</div>
        <div class="div-line"></div>
    </div>
    """)

    output = gr.Gallery(
        label="",
        columns=8,
        rows=2,
        height=520,
        object_fit="cover",
        show_label=False,
    )

    gr.HTML("""
    <div id="booksense-footer">
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