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

# Use a custom theme that works reliably on HF Spaces
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fdf6e3", c100="#f5e6b8", c200="#ecd48a",
        c300="#e2c15c", c400="#d4a847", c500="#c8973a",
        c600="#b07828", c700="#8a5c1a", c800="#63400e",
        c900="#3d2506", c950="#1e1003",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f5f4f8", c100="#e8e6f0", c200="#ccc8e0",
        c300="#aba6cc", c400="#8a84b8", c500="#6b659f",
        c600="#524d86", c700="#3d396d", c800="#2a2754",
        c900="#19173d", c950="#0d0c26",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f0eef8", c100="#dbd8ee", c200="#b8b4d8",
        c300="#9490c0", c400="#726da8", c500="#554f8e",
        c600="#3e3974", c700="#2c275a", c800="#1c1940",
        c900="#0f0d28", c950="#07061a",
    ),
    font=[gr.themes.GoogleFont("Jost"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0d0e14",
    body_background_fill_dark="#0d0e14",
    body_text_color="#edeaf3",
    body_text_color_dark="#edeaf3",
    body_text_color_subdued="#7a7890",
    body_text_color_subdued_dark="#7a7890",
    background_fill_primary="#13141a",
    background_fill_primary_dark="#13141a",
    background_fill_secondary="#1c1e27",
    background_fill_secondary_dark="#1c1e27",
    border_color_primary="rgba(255,255,255,0.08)",
    border_color_primary_dark="rgba(255,255,255,0.08)",
    border_color_accent="#d4a847",
    border_color_accent_dark="#d4a847",
    input_background_fill="#1c1e27",
    input_background_fill_dark="#1c1e27",
    input_border_color="rgba(255,255,255,0.1)",
    input_border_color_dark="rgba(255,255,255,0.1)",
    input_border_color_focus="#d4a847",
    input_border_color_focus_dark="#d4a847",
    input_placeholder_color="#4a4860",
    input_placeholder_color_dark="#4a4860",
    input_shadow_focus="0 0 0 3px rgba(212,168,71,0.12)",
    input_shadow_focus_dark="0 0 0 3px rgba(212,168,71,0.12)",
    button_primary_background_fill="linear-gradient(135deg, #d4a847, #b07828)",
    button_primary_background_fill_dark="linear-gradient(135deg, #d4a847, #b07828)",
    button_primary_background_fill_hover="linear-gradient(135deg, #e0b84f, #c08030)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #e0b84f, #c08030)",
    button_primary_text_color="#080810",
    button_primary_text_color_dark="#080810",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",
    button_primary_shadow="0 4px 18px rgba(212,168,71,0.35)",
    button_primary_shadow_dark="0 4px 18px rgba(212,168,71,0.35)",
    button_primary_shadow_hover="0 8px 28px rgba(212,168,71,0.5)",
    button_primary_shadow_hover_dark="0 8px 28px rgba(212,168,71,0.5)",
    block_background_fill="#13141a",
    block_background_fill_dark="#13141a",
    block_border_color="rgba(255,255,255,0.07)",
    block_border_color_dark="rgba(255,255,255,0.07)",
    block_border_width="1px",
    block_shadow="0 20px 60px rgba(0,0,0,0.5)",
    block_shadow_dark="0 20px 60px rgba(0,0,0,0.5)",
    block_radius="16px",
    container_radius="16px",
    button_large_radius="10px",
    button_large_padding="14px 28px",
    button_large_text_size="0.82rem",
    button_large_text_weight="700",
    input_radius="10px",
    input_text_size="0.93rem",
    section_header_text_size="0.68rem",
    section_header_text_weight="600",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,400;1,600&family=Jost:wght@300;400;500;600;700&display=swap');

/* Force dark background everywhere */
html, body, .gradio-container, #root, main {
    background: #0d0e14 !important;
    color: #edeaf3 !important;
}

/* ── Hero ── */
.bk-hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
}

.bk-eyebrow {
    font-size: 0.62rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #d4a847;
    margin-bottom: 1rem;
    font-weight: 600;
    font-family: 'Jost', sans-serif;
}

.bk-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(3rem, 6vw, 5rem);
    font-weight: 300;
    color: #edeaf3;
    line-height: 1;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.bk-title em {
    color: #d4a847;
    font-style: italic;
    font-weight: 600;
}

.bk-rule {
    width: 120px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d4a847, transparent);
    margin: 1.4rem auto;
}

.bk-sub {
    color: #7a7890;
    font-size: 0.93rem;
    font-weight: 300;
    line-height: 1.75;
    max-width: 400px;
    margin: 0 auto;
    font-family: 'Jost', sans-serif;
}

/* ── Remove Gradio's default padding and borders ── */
.gradio-container > .main > .wrap {
    padding: 0 1rem !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ── Kill double borders from gr.Group ── */
.gr-group, .gradio-group {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* ── Search panel ── */
.bk-search-wrap {
    background: #13141a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
}

.bk-search-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,168,71,0.35), transparent);
}

/* ── Labels ── */
label span,
.label-wrap > span {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: #5a5870 !important;
}

/* ── Inputs & Textareas ── */
input, textarea {
    background: #1c1e27 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #edeaf3 !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 300 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input:focus, textarea:focus {
    border-color: rgba(212,168,71,0.6) !important;
    box-shadow: 0 0 0 3px rgba(212,168,71,0.1) !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder {
    color: #3a3850 !important;
}

/* ── Dropdowns ── */
select, .gr-dropdown select {
    background: #1c1e27 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #edeaf3 !important;
    font-family: 'Jost', sans-serif !important;
}

/* ── Button override ── */
button.primary, button[variant="primary"], .gr-button-primary {
    background: linear-gradient(135deg, #d4a847 0%, #b07828 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #06060e !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 14px 24px !important;
    box-shadow: 0 4px 20px rgba(212,168,71,0.3) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    width: 100% !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #e0b84f 0%, #c08030 100%) !important;
    box-shadow: 0 8px 28px rgba(212,168,71,0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── Divider ── */
.bk-divider {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: 0.5rem 0 1.8rem;
}

.bk-div-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
}

.bk-div-text {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.25rem;
    font-style: italic;
    font-weight: 400;
    color: #d4a847;
    white-space: nowrap;
    letter-spacing: 0.02em;
}

/* ── Gallery ── */
.gradio-gallery, [data-testid="gallery"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="gallery"] > div {
    gap: 12px !important;
}

[data-testid="gallery"] img {
    border-radius: 8px !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="gallery"] img:hover {
    transform: translateY(-6px) scale(1.02) !important;
    box-shadow: 0 20px 48px rgba(0,0,0,0.7), 0 0 0 1px rgba(212,168,71,0.25) !important;
}

/* ── Footer ── */
.bk-footer {
    text-align: center;
    padding: 2rem 1rem;
    margin-top: 1rem;
    color: #35334a;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    font-family: 'Jost', sans-serif;
    border-top: 1px solid rgba(255,255,255,0.04);
}

/* ── Hide Gradio branding ── */
footer, .footer, .built-with, .show-api,
.svelte-1ipelgc, .version-info {
    display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0e14; }
::-webkit-scrollbar-thumb { background: #2a2840; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #3a3860; }
"""

with gr.Blocks(theme=theme, css=custom_css) as dashboard:

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="bk-hero">
        <div class="bk-eyebrow">✦ &nbsp; Semantic Discovery &nbsp; ✦</div>
        <h1 class="bk-title">Book<em>Sense</em></h1>
        <div class="bk-rule"></div>
        <p class="bk-sub">
            Describe a feeling, a world, or a story —<br>
            and we'll find books that match the meaning behind your words.
        </p>
    </div>
    """)

    # ── Search Panel ──────────────────────────────────────────────────────────
    gr.HTML('<div class="bk-search-wrap">')
    with gr.Row(equal_height=False):
        user_query = gr.Textbox(
            label="What kind of book are you looking for?",
            placeholder="e.g., A haunting story about grief and memory in post-war Japan...",
            lines=4,
            scale=3,
            container=True,
        )
        with gr.Column(scale=1, min_width=200):
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
                size="lg",
            )
    gr.HTML('</div>')

    # ── Divider ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="bk-divider">
        <div class="bk-div-line"></div>
        <div class="bk-div-text">Curated for you</div>
        <div class="bk-div-line"></div>
    </div>
    """)

    # ── Gallery ───────────────────────────────────────────────────────────────
    output = gr.Gallery(
        label="",
        columns=8,
        rows=2,
        height=520,
        object_fit="cover",
        show_label=False,
        elem_classes="gradio-gallery",
    )

    # ── Footer ────────────────────────────────────────────────────────────────
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
