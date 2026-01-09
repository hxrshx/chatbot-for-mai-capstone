#!/usr/bin/env python3
"""
CAIRA - CAIRO AI Research Assistant Frontend
Enhanced Professional UI with THWS branding and fixed visibility issues
"""

import gradio as gr
import requests
import json
from typing import List, Tuple, Optional
import time
import os
import base64

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_base64_image(image_path):
    """Convert image to base64 for inline display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Backend API configuration
API_URL = "http://localhost:8000"

# THWS brand colors
THWS_ORANGE = "#FF6600"
THWS_DARK = "#1a1a1a"
THWS_CARD = "#2d2d2d"
THWS_BORDER = "#444"
THWS_TEXT = "#ffffff"
THWS_LIGHT_ORANGE = "#FF8833"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_welcome_message():
    """Return welcome message for initial chat state"""
    return [(None, """👋 Hey! I'm CAIRA – your AI senior for all things MAI at THWS.

Got questions about courses, exams, accommodation, or surviving Würzburg? I'm your bot.
Want Käsespätzle recipes or life advice? Sorry, I'm just here for the MAI stuff. 🧀

Fire away! 🚀""")]


def get_technical_info():
    """Return technical details about the chatbot system"""
    return """🔧 **CAIRA's Technical Specifications**

**Architecture: RAG (Retrieval-Augmented Generation)**
A two-stage system that first retrieves relevant documents, then generates answers based on them.

---

**📚 Knowledge Base:**
• **Documents:** 46 source documents (PDFs & DOCX)
• **Chunks:** 271 text chunks (750 tokens each, 150 overlap)
• **Topics:** Housing, Admission, Courses, Campus Life, Würzburg Guide

---

**🧠 AI Components:**

**1. Embedding Model** 🎯
   • **Model:** `intfloat/multilingual-e5-small`
   • **Dimensions:** 384-dimensional vectors
   • **Purpose:** Converts text into numerical representations for similarity search
   • **Language:** Multilingual (English & German)

**2. Vector Database** 💾
   • **Technology:** FAISS (Facebook AI Similarity Search)
   • **Index Type:** IndexFlatIP (Inner Product - cosine similarity)
   • **Size:** 271 vectors indexed
   • **Search:** Semantic search with query expansion & disambiguation

**3. Language Model** 🤖
   • **Model:** Qwen 2.5-1.5B-Instruct
   • **Size:** ~3GB (1.5 billion parameters)
   • **Device:** CPU inference (optimized for local deployment)
   • **Max Tokens:** 150 tokens per response
   • **Context:** 600 characters from retrieved chunks

---

**⚙️ Advanced Features:**
✅ **Cross-topic Search** - Searches across all document categories
✅ **Ambiguity Handling** - Disambiguates unclear queries
✅ **Query Expansion** - Adds synonyms for better retrieval
✅ **Keyword Boosting** - Prioritizes important terms

---

**🚀 Performance:**
• **Response Time:** 30-60 seconds (local CPU)
• **Accuracy:** Top-k retrieval (k=2-3)
• **Optimization:** Token limits, context reduction, repetition penalty

---

**👨‍💻 Built by MAI students for MAI students**
Using Python, FastAPI, Gradio, Sentence Transformers, FAISS, and Qwen LLM.

*Want to know more? Check out the GitHub repo or ask the developers!* 🔗
"""


def check_technical_query(question: str) -> bool:
    """Check if user is asking about technical details"""
    tech_keywords = [
        'tech details', 'technical details', 'about you', 'how do you work',
        'how are you built', 'what model', 'which model', 'your architecture',
        'how you work', 'behind the scenes', 'your brain', 'your technology',
        'embedding', 'vector', 'llm', 'rag', 'how it works'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in tech_keywords)


def add_message(question: str, chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """
    Immediately add user message to chat (step 1)
    """
    if not question.strip():
        return chat_history, ""
    
    chat_history = chat_history or []
    # Add user message with thinking indicator
    chat_history.append((question, "🤔 *Analyzing your question and searching through documents...*"))
    return chat_history, ""


def get_bot_response(chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """
    Get bot response from backend (step 2)
    """
    if not chat_history:
        return chat_history, ""
    
    # Get the last user question
    question = chat_history[-1][0]
    
    # Check backend
    if not check_backend_health():
        error_msg = "⚠️ **Backend Connection Error**\n\nThe backend server is not running. Please start it first:\n\n```bash\npython backend/main.py\n```"
        chat_history[-1] = (question, error_msg)
        return chat_history, ""
    
    try:
        # Call backend API
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": 3},
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['answer']
            sources = result['sources']
            
            # Update with real answer
            chat_history[-1] = (question, answer)
            
            # Format sources beautifully
            if sources:
                sources_text = "📚 **Retrieved Sources:**\n\n"
                for i, source in enumerate(sources, 1):
                    sources_text += f"**[{i}]** `{source['source_file']}`\n"
                    sources_text += f"   • **Topic:** {source['topic']}\n"
                    sources_text += f"   • **Relevance Score:** {source['similarity_score']:.3f}\n\n"
            else:
                sources_text = "No sources retrieved for this query."
            
            return chat_history, sources_text
        else:
            error_msg = f"❌ **API Error:** {response.json().get('detail', 'Unknown error')}"
            chat_history[-1] = (question, error_msg)
            return chat_history, ""
            
    except requests.exceptions.Timeout:
        error_msg = "⏱️ **Timeout Notice**\n\nThe request timed out. Response generation can take 2-3 minutes on CPU. The model may still be processing your request in the background."
        chat_history[-1] = (question, error_msg)
        return chat_history, ""
    except Exception as e:
        error_msg = f"❌ **Error:** `{str(e)}`\n\nPlease check your backend connection and try again."
        chat_history[-1] = (question, error_msg)
        return chat_history, ""


def clear_chat():
    """Clear chat history and return welcome message"""
    return get_welcome_message(), ""


# Enhanced Custom CSS with better visibility and modern dark theme
custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {{
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}}

/* Main container */
.gradio-container {{
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: {THWS_DARK} !important;
}}

body {{
    margin: 0;
    padding: 0;
    background: {THWS_DARK} !important;
}}

/* Header - Modern and compact */
.caira-header {{
    display: flex;
    align-items: center;
    padding: 16px 40px;
    background: {THWS_CARD};
    border-bottom: 2px solid {THWS_ORANGE};
    gap: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}}

.thws-logo-box {{
    background: {THWS_ORANGE};
    padding: 12px 24px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    box-shadow: 0 4px 15px rgba(255,102,0,0.4);
}}

.thws-logo-text {{
    color: white;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.caira-title-section {{
    flex: 1;
}}

.caira-main-title {{
    color: {THWS_TEXT};
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 5px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.caira-subtitle {{
    color: {THWS_ORANGE};
    font-size: 14px;
    font-weight: 500;
    margin: 0;
    letter-spacing: 0.5px;
}}

/* Chat container - Full height and proper visibility */
.chat-wrapper {{
    max-width: 1400px;
    margin: 20px auto;
    padding: 0 40px;
    height: calc(100vh - 280px);
    min-height: 600px;
}}

/* Fix Gradio's chatbot container */
.chatbot {{
    border: none !important;
    background: transparent !important;
    height: 100% !important;
    overflow: visible !important;
}}

/* Fix message container - TRANSPARENT */
.chatbot .overflow-y-auto {{
    background: transparent !important;
    border-radius: 0 !important;
    padding: 20px !important;
    height: 100% !important;
    overflow-y: auto !important;
}}

/* Message styling - FIXED VISIBILITY */
.chatbot .message {{
    margin: 16px 0 !important;
    padding: 0 !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
    align-items: flex-start !important;
    gap: 12px !important;
    width: 100% !important;
}}

/* User messages - right aligned */
.chatbot .message.user {{
    justify-content: flex-end !important;
    flex-direction: row-reverse !important;
}}

/* Bot messages - left aligned */
.chatbot .message.bot {{
    justify-content: flex-start !important;
}}

/* Message content bubbles */
.chatbot .message .message-content {{
    background: {THWS_CARD} !important;
    border: 1px solid {THWS_BORDER} !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    max-width: 75% !important;
    color: {THWS_TEXT} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    opacity: 1 !important;
    visibility: visible !important;
}}

/* User message bubble - orange accent */
.chatbot .message.user .message-content {{
    background: linear-gradient(135deg, {THWS_ORANGE} 0%, {THWS_LIGHT_ORANGE} 100%) !important;
    border: none !important;
    color: white !important;
}}

/* Ensure all text is visible and white */
.chatbot .message p,
.chatbot .message span,
.chatbot .message div,
.chatbot .message code,
.chatbot .message pre {{
    color: inherit !important;
    opacity: 1 !important;
    visibility: visible !important;
    margin: 0 0 8px 0 !important;
    line-height: 1.6 !important;
}}

.chatbot .message p:last-child {{
    margin-bottom: 0 !important;
}}

/* Code blocks in messages */
.chatbot .message code {{
    background: rgba(0,0,0,0.3) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
}}

.chatbot .message pre {{
    background: rgba(0,0,0,0.3) !important;
    padding: 12px !important;
    border-radius: 6px !important;
    overflow-x: auto !important;
}}

/* Avatar styling */
.chatbot .avatar-container {{
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: {THWS_CARD} !important;
    border: 2px solid {THWS_ORANGE} !important;
    box-shadow: 0 2px 6px rgba(255,102,0,0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}}

/* Input area */
.input-wrapper {{
    max-width: 1400px;
    margin: 20px auto;
    padding: 0 40px;
}}

/* Input field */
.input-wrapper input[type="text"] {{
    background: {THWS_CARD} !important;
    color: {THWS_TEXT} !important;
    border: 2px solid {THWS_BORDER} !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
}}

.input-wrapper input[type="text"]:focus {{
    border-color: {THWS_ORANGE} !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,102,0,0.2) !important;
}}

.input-wrapper input[type="text"]::placeholder {{
    color: #999 !important;
}}

/* Send button */
.send-btn {{
    background: linear-gradient(135deg, {THWS_ORANGE} 0%, {THWS_LIGHT_ORANGE} 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 14px 30px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(255,102,0,0.3) !important;
    transition: all 0.3s ease !important;
}}

.send-btn:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255,102,0,0.5) !important;
}}

.send-btn:active {{
    transform: translateY(0) !important;
}}

/* Quick Questions section */
.quick-questions {{
    max-width: 1400px;
    margin: 15px auto 30px auto;
    padding: 0 40px;
}}

.quick-questions-title {{
    color: {THWS_TEXT};
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* Example buttons */
.quick-questions button {{
    background: {THWS_CARD} !important;
    color: {THWS_TEXT} !important;
    border: 1px solid {THWS_BORDER} !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    margin: 5px !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    text-align: left !important;
}}

.quick-questions button:hover {{
    background: {THWS_ORANGE} !important;
    border-color: {THWS_ORANGE} !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255,102,0,0.4) !important;
}}

/* Clear button */
.clear-btn {{
    background: {THWS_CARD} !important;
    color: {THWS_TEXT} !important;
    border: 1px solid {THWS_BORDER} !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}}

.clear-btn:hover {{
    background: {THWS_BORDER} !important;
    border-color: {THWS_TEXT} !important;
}}

/* Sources panel */
.sources-panel {{
    max-width: 1400px;
    margin: 20px auto;
    padding: 0 40px;
}}

.sources-content {{
    background: {THWS_CARD} !important;
    border: 1px solid {THWS_BORDER} !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: {THWS_TEXT} !important;
}}

/* Scrollbar styling */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: {THWS_DARK};
    border-radius: 10px;
}}

::-webkit-scrollbar-thumb {{
    background: {THWS_ORANGE};
    border-radius: 10px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {THWS_LIGHT_ORANGE};
}}

/* Responsive design */
@media (max-width: 1024px) {{
    .caira-header {{
        padding: 12px 20px;
    }}
    
    .chat-wrapper,
    .input-wrapper,
    .quick-questions,
    .sources-panel {{
        padding: 0 20px;
    }}
    
    .chatbot .message .message-content {{
        max-width: 85% !important;
    }}
}}

/* Smooth animations */
@keyframes fadeIn {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.chatbot .message {{
    animation: fadeIn 0.3s ease-out !important;
}}

/* Status indicator */
.status-indicator {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {THWS_ORANGE};
    margin-left: 8px;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="CAIRA - CAIRO AI Research Assistant", theme=gr.themes.Base()) as demo:
    
    # Load logo as base64
    logo_path = os.path.join(BASE_DIR, "Thws-logo_English.png")
    logo_base64 = get_base64_image(logo_path)
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 24px; filter: brightness(0) invert(1);">' if logo_base64 else ""
    
    # Header
    gr.HTML(f"""
        <div class="caira-header">
            <div class="thws-logo-box">
                <div class="thws-logo-text">THWS {logo_html}</div>
            </div>
            <div class="caira-title-section">
                <div class="caira-main-title">
                    CAIRA 🏛️
                </div>
                <div class="caira-subtitle">MAI Program Guide • Your seniors' brain, now in bot form </div>
            </div>
        </div>
    """)
    
    # Chat interface
    chat_thumbnail_path = os.path.join(BASE_DIR, "chat_thumbnail.jpeg")
    
    with gr.Column(elem_classes="chat-wrapper"):
        chatbot = gr.Chatbot(
            label="Chat",
            height=700,
            show_label=False,
            avatar_images=(None, chat_thumbnail_path if os.path.exists(chat_thumbnail_path) else None),
            bubble_full_width=False,
            container=False,
            elem_classes="chatbot",
            value=get_welcome_message(),
            show_copy_button=True,
            latex_delimiters=[{"left": "$$", "right": "$$", "display": True}]
        )
    
    # Input area with clear button
    with gr.Column(elem_classes="input-wrapper"):
        with gr.Row():
            question_input = gr.Textbox(
                placeholder="Ask about courses, campus life, exams, accommodation, or anything about THWS MAI...",
                show_label=False,
                scale=8,
                container=False,
                lines=1,
                max_lines=3
            )
            submit_btn = gr.Button(
                "Send 📤",
                variant="primary",
                scale=1,
                elem_classes="send-btn"
            )
            clear_btn = gr.Button(
                "🔄 Clear",
                variant="secondary",
                scale=1,
                elem_classes="clear-btn"
            )
    
    # Quick Questions
    with gr.Column(elem_classes="quick-questions"):
        gr.HTML('<div class="quick-questions-title">⚡ Quick Questions</div>')
        examples = gr.Examples(
            examples=[
                "What courses are offered in the first semester of MAI?",
                "Tell me about living in Würzburg as a student",
                "How do I register for exams at THWS?",
                "What are the admission requirements for the MAI program?",
                "Where can I find student accommodation in Würzburg?",
                "What is the application deadline for international students?"
            ],
            inputs=question_input,
            label="",
            examples_per_page=6
        )
    
    # Sources display (collapsible)
    with gr.Column(elem_classes="sources-panel"):
        with gr.Accordion("📚 Retrieved Sources", open=False):
            sources_display = gr.Markdown(
                value="Sources will appear here after you ask a question.",
                elem_classes="sources-content"
            )
    
    # Footer with technical details and copyright
    gr.HTML(f"""
        <div style="max-width: 1200px; margin: 30px auto 20px auto; padding: 20px 30px; background: #2d2d2d; border-radius: 10px; border-top: 2px solid {THWS_ORANGE};">
            
            <!-- Technical Details -->
            <div style="margin-bottom: 25px;">
                <h3 style="color: {THWS_ORANGE}; font-size: 14px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    🔧 Technical Specifications
                </h3>
                <div style="font-size: 11px; color: #ccc; line-height: 1.6;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px;">
                        
                        <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; border-left: 3px solid {THWS_ORANGE};">
                            <strong style="color: white; font-size: 12px;">🧠 Language Model</strong><br>
                            <span style="color: #999;">Model:</span> Qwen 2.5-1.5B-Instruct<br>
                            <span style="color: #999;">Size:</span> ~3GB (1.5B parameters)<br>
                            <span style="color: #999;">Device:</span> CPU inference
                        </div>
                        
                        <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; border-left: 3px solid {THWS_ORANGE};">
                            <strong style="color: white; font-size: 12px;">🎯 Embedding Model</strong><br>
                            <span style="color: #999;">Model:</span> multilingual-e5-small<br>
                            <span style="color: #999;">Dimensions:</span> 384-dimensional vectors<br>
                            <span style="color: #999;">Language:</span> Multilingual (EN/DE)
                        </div>
                        
                        <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; border-left: 3px solid {THWS_ORANGE};">
                            <strong style="color: white; font-size: 12px;">💾 Vector Database</strong><br>
                            <span style="color: #999;">Technology:</span> FAISS (Meta AI)<br>
                            <span style="color: #999;">Index:</span> IndexFlatIP (cosine similarity)<br>
                            <span style="color: #999;">Vectors:</span> 271 chunks indexed
                        </div>
                        
                        <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; border-left: 3px solid {THWS_ORANGE};">
                            <strong style="color: white; font-size: 12px;">📚 Knowledge Base</strong><br>
                            <span style="color: #999;">Documents:</span> 46 source files<br>
                            <span style="color: #999;">Chunks:</span> 271 text segments<br>
                            <span style="color: #999;">Topics:</span> Housing, Courses, Campus Life
                        </div>
                    </div>
                    
                    <div style="margin-top: 12px; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                        <strong style="color: white; font-size: 11px;">Architecture:</strong> 
                        <span style="color: #ccc;">RAG (Retrieval-Augmented Generation) with semantic search, query expansion, and cross-topic retrieval</span>
                    </div>
                </div>
            </div>
            
            <!-- Copyright and License -->
            <div style="border-top: 1px solid #444; padding-top: 15px;">
                <div style="font-size: 11px; color: #999; line-height: 1.8;">
                    <div style="margin-bottom: 8px;">
                        <strong style="color: white;">© 2025-2026 CAIRA - CAIRO AI Research Assistant</strong>
                    </div>
                    <div style="margin-bottom: 8px;">
                        Built with ❤️ by MAI students for MAI students at <strong style="color: {THWS_ORANGE};">THWS Würzburg-Schweinfurt</strong>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: white;">📄 License:</strong> All content and responses generated by CAIRA are freely available for personal and educational use. 
                        Users are free to copy, share, and utilize the information provided. However, please verify critical information with official THWS sources.
                    </div>
                    <div style="font-size: 10px; color: #666; font-style: italic; margin-top: 8px;">
                        ⚠️ Disclaimer: CAIRA is an unofficial AI assistant. Always consult official THWS channels for binding information regarding admissions, deadlines, and regulations.
                    </div>
                </div>
            </div>
            
        </div>
    """)
    
    # Event handlers - Two-step process for immediate message display
    # Step 1: Add user message immediately
    # Step 2: Get bot response
    
    submit_btn.click(
        fn=add_message,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
        queue=False
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot, sources_display],
        show_progress=True
    )
    
    question_input.submit(
        fn=add_message,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
        queue=False
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot, sources_display],
        show_progress=True
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, sources_display]
    )


if __name__ == "__main__":
    print("=" * 70)
    print("🏛️  CAIRA - CAIRO AI Research Assistant")
    print("=" * 70)
    print(f"Frontend URL:  http://localhost:7860")
    print(f"Backend URL:   {API_URL}")
    print("=" * 70)
    
    # Check backend
    if check_backend_health():
        print("✅ Backend is running and healthy")
    else:
        print("⚠️  Backend is not responding")
        print("   Start backend: python backend/main.py")
    
    print("=" * 70)
    print("Starting Gradio interface...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )