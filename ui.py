import gradio as gr
from core import CombinedGitHubRepoChat

def create_interface():
    """Create the Gradio interface with elegant Claude-inspired design"""
    
    # Elegant Claude-inspired CSS with sky blue theme
    custom_css = """
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles with sky blue theme */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 50%, #81d4fa 100%) !important;
        min-height: 100vh !important;
        color: #1e293b !important;
    }
    
    /* Main container with glass effect */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Header with centered title */
    .header-section {
        text-align: center;
        padding: 32px 0 24px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 32px;
        position: relative;
    }
    
    .main-title {
        font-size: 42px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #0ea5e9, #3b82f6, #6366f1) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 0 0 12px 0 !important;
        line-height: 1.1 !important;
        letter-spacing: -0.02em !important;
    }
    
    .subtitle {
        font-size: 16px !important;
        color: #64748b !important;
        font-weight: 400 !important;
        margin: 0 !important;
        max-width: 600px !important;
        margin: 0 auto !important;
        line-height: 1.6 !important;
    }
    
    /* Developer card */
    .developer-card {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        z-index: 999;
    }
    
    .developer-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .developer-name {
        font-size: 14px;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 4px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .developer-name::before {
        content: "👨‍💻";
        font-size: 16px;
    }
    
    .developer-link {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #0ea5e9 !important;
        text-decoration: none !important;
        font-size: 13px;
        font-weight: 500;
        transition: color 0.2s ease;
        padding: 4px 8px;
        border-radius: 8px;
        background: rgba(14, 165, 233, 0.1);
    }
    
    .developer-link:hover {
        color: #0284c7 !important;
        background: rgba(14, 165, 233, 0.2);
    }
    
    .developer-link::before {
        content: "🔗";
        font-size: 12px;
    }
    
    /* Status indicators with modern design */
    .status-section {
        display: flex;
        justify-content: center;
        gap: 16px;
        margin-bottom: 32px;
        flex-wrap: wrap;
    }
    
    .status-indicator {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 13px;
        font-weight: 500;
        color: #334155;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .status-indicator:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab navigation with modern styling */
    .tab-nav {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 6px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    }
    
    .tab-nav button {
        background: transparent !important;
        border: none !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #64748b !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 2px !important;
    }
    
    .tab-nav button.selected {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #0ea5e9 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        transform: translateY(-1px) !important;
    }
    
    .tab-nav button:hover:not(.selected) {
        background: rgba(255, 255, 255, 0.7) !important;
        color: #475569 !important;
    }
    
    /* Chat container with modern styling */
    .chat-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px;
    }
    
    /* Chatbot styling */
    .chatbot-container {
        border: none !important;
        background: transparent !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    /* Enhanced message styling */
    .message {
        margin: 16px 0 !important;
        animation: fadeInUp 0.3s ease-out !important;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message .user {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 14px 18px !important;
        margin: 8px 0 8px auto !important;
        font-size: 14px !important;
        max-width: 75% !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        border: none !important;
    }
    
    .message .bot {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 16px 20px !important;
        margin: 8px auto 8px 0 !important;
        font-size: 14px !important;
        color: #1e293b !important;
        line-height: 1.6 !important;
        max-width: 85% !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
    }
    
    /* Input area with floating design */
    .input-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-top: 16px;
    }
    
    .input-row {
        display: flex;
        gap: 12px;
        align-items: end;
    }
    
    /* Enhanced text input */
    .chat-input {
        flex: 1;
        min-height: 48px !important;
        max-height: 120px !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        font-size: 14px !important;
        color: #1e293b !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        resize: none !important;
        transition: all 0.3s ease !important;
        line-height: 1.5 !important;
        font-family: inherit !important;
    }
    
    .chat-input:focus {
        outline: none !important;
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
        background: rgba(255, 255, 255, 1) !important;
    }
    
    .chat-input::placeholder {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }
    
    /* Enhanced send button */
    .send-button {
        min-width: 48px !important;
        height: 48px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 0 16px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    .send-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        background: linear-gradient(135deg, #0284c7, #2563eb) !important;
    }
    
    .send-button:active {
        transform: translateY(0) !important;
    }
    
    .send-button:disabled {
        background: #94a3b8 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Form elements with modern styling */
    .form-section {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 24px;
    }
    
    .form-group {
        margin-bottom: 24px;
    }
    
    .form-label {
        display: block;
        font-size: 14px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 8px;
    }
    
    .form-input {
        width: 100% !important;
        padding: 14px 18px !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        color: #1e293b !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.3s ease !important;
        font-family: inherit !important;
    }
    
    .form-input:focus {
        outline: none !important;
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
        background: rgba(255, 255, 255, 1) !important;
    }
    
    /* Enhanced buttons */
    .primary-btn {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        background: linear-gradient(135deg, #0284c7, #2563eb) !important;
    }
    
    .secondary-btn {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #475569 !important;
        border: 2px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .secondary-btn:hover {
        background: rgba(255, 255, 255, 1) !important;
        border-color: #94a3b8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Status text area */
    .status-text {
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace !important;
        font-size: 13px !important;
        background: rgba(248, 250, 252, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        color: #475569 !important;
        line-height: 1.6 !important;
    }
    
    /* Dropdown styling */
    .dropdown select {
        width: 100% !important;
        padding: 14px 18px !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        color: #1e293b !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        cursor: pointer !important;
        appearance: none !important;
        transition: all 0.3s ease !important;
    }
    
    .dropdown select:focus {
        outline: none !important;
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
    }
    
    /* Clear button */
    .clear-btn {
        background: transparent !important;
        color: #64748b !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-size: 13px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin-top: 12px !important;
        font-weight: 500 !important;
    }
    
    .clear-btn:hover {
        background: rgba(248, 250, 252, 0.8) !important;
        color: #475569 !important;
        border-color: #94a3b8 !important;
    }
    
    /* Info sections */
    .info-section {
        background: rgba(248, 250, 252, 0.8);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(226, 232, 240, 0.5);
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        font-size: 13px;
        color: #475569;
        line-height: 1.6;
    }
    
    .info-section h4 {
        font-size: 15px;
        font-weight: 600;
        color: #334155;
        margin: 0 0 12px 0;
    }
    
    .info-section ul {
        margin: 12px 0;
        padding-left: 20px;
    }
    
    .info-section li {
        margin-bottom: 6px;
    }
    
    .info-section code {
        background: rgba(226, 232, 240, 0.8);
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 12px;
        color: #1e293b;
    }
    
    /* Accordion styling */
    .accordion {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        margin-top: 16px !important;
        overflow: hidden !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            margin: 10px;
            padding: 16px;
            border-radius: 20px;
        }
        
        .main-title {
            font-size: 32px !important;
        }
        
        .developer-card {
            position: static;
            margin: 16px auto 0 auto;
            max-width: 280px;
        }
        
        .header-section {
            padding: 24px 0 20px 0;
        }
        
        .status-section {
            flex-direction: column;
            align-items: center;
        }
        
        .input-row {
            flex-direction: column;
            gap: 12px;
        }
        
        .send-button {
            width: 100% !important;
            justify-content: center !important;
        }
    }
    """
    
    app = CombinedGitHubRepoChat()
    
    with gr.Blocks(title="Repository AI Assistant", css=custom_css, theme=gr.themes.Base()) as interface:
        
        with gr.Column(elem_classes=["main-container"]):
            # Header with centered title and developer card
            with gr.Row(elem_classes=["header-section"]):
                with gr.Column():
                    
                    
                    # Centered title
                    gr.HTML("""
                    <h1 class="main-title">Github AI Assistant</h1>
                    <p class="subtitle">A Learning tool to Accelerate the understanding of large Github Reposositories. Get instant insights into code functionality and code structure.</p>
                    """)
            
            # API Status with modern indicators
            api_status = []
            if hasattr(app, 'gemini_api_key') and app.gemini_api_key:
                api_status.append("✅ LLM Connected")
            else:
                api_status.append("❌ LLM Not Connected")
            if hasattr(app, 'pinecone_api_key') and app.pinecone_api_key:
                api_status.append("✅ Database  Connected")
            else:
                api_status.append("❌ Database  Connected")
            
            with gr.Row(elem_classes=["status-section"]):
                for status in api_status:
                    gr.HTML(f'<div class="status-indicator">{status}</div>')
            
            with gr.Tabs(elem_classes=["tab-nav"]) as tabs:
                
                # Main Chat Tab
                with gr.Tab("💬 Chat", elem_classes=["tab-content"]):
                    with gr.Column(elem_classes=["chat-container"]):
                        chatbot = gr.Chatbot(
                            height=500,
                            show_label=False,
                            show_copy_button=True,
                            elem_classes=["chatbot-container"],
                            placeholder="👋 Hi! I'm your repository AI assistant. Connect to a repository to start chatting about its code, structure, and functionality.",
                            avatar_images=("https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/themes/utils/icons/user.svg", 
                                         "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/themes/utils/icons/bot.svg")
                        )
                        
                        with gr.Column(elem_classes=["input-container"]):
                            with gr.Row(elem_classes=["input-row"]):
                                msg = gr.Textbox(
                                    placeholder="Ask me anything about the repository...",
                                    show_label=False,
                                    container=False,
                                    scale=4,
                                    elem_classes=["chat-input"],
                                    lines=1,
                                    max_lines=4
                                )
                                send_btn = gr.Button(
                                    "Send",
                                    scale=0,
                                    elem_classes=["send-button"],
                                    variant="primary"
                                )
                        
                        clear_btn = gr.Button(
                            "Clear conversation",
                            elem_classes=["clear-btn"],
                            variant="secondary",
                            size="sm"
                        )
                        
                        with gr.Accordion("💡 Example questions ", open=False, elem_classes=["accordion"]):
                            gr.Markdown("""
                            - What is the main purpose of this repository?
                            - **Ask by filename**:explain filename.py or filename.ipynb 
                            - **Ask by filepath(useful in repos where same filenames in multiple folders):** explain filepath
                            - **Ask by cell**:Explain Cell 3 in filename.ipynb or filepath 
                            - **Ask by Line Range**:explain line 55 to line 65 in filename.py or filepath
                            - **Ask by function or class**:explain some_function or class class_name
                            - **Ask by function or class(if same class or function_names in multiple files)**:explain some_function or class class_name in filename.py/filename.ipynb/filepath 
                            """)
                
                # Process Repository Tab
                with gr.Tab("🚀 Process Repository", elem_classes=["tab-content"]):
                    with gr.Column(elem_classes=["form-section"]):
                        gr.Markdown("### Process a New Repository")
                        gr.Markdown("Index a GitHub repository to enable AI-powered chat and analysis.")
                        
                        with gr.Column(elem_classes=["form-group"]):
                            github_url = gr.Textbox(
                                label="GitHub Repository URL",
                                placeholder="https://github.com/username/repository",
                                elem_classes=["form-input"]
                            )
                            
                            index_name = gr.Textbox(
                                label="Index Name ",
                                placeholder="Enter Index name",
                                elem_classes=["form-input"]
                            )
                            
                            process_btn = gr.Button(
                                "🚀 Start Processing",
                                elem_classes=["primary-btn"],
                                variant="primary"
                            )
                            
                            processing_status = gr.Textbox(
                                label="Processing Status",
                                interactive=False,
                                lines=10,
                                elem_classes=["status-text"]
                            )
                            
                            refresh_btn = gr.Button(
                                "🔄 Refresh Status",
                                elem_classes=["secondary-btn"],
                                variant="secondary"
                            )
                        
                        with gr.Accordion("ℹ️ Processing Information", open=False, elem_classes=["accordion"]):
                            gr.HTML("""
                            <div class="info-section">
                                <h4>What happens during processing:</h4>
                                <ul>
                                    <li><strong>Repository Analysis:</strong> Clone and analyze repository structure</li>
                                    <li><strong>Code Embedding:</strong> Generate AI embeddings for code chunks</li>
                                    <li><strong>Vector Storage:</strong> Store embeddings in Vector database</li>
                                    <li><strong>Index Creation:</strong> Create searchable index for fast retrieval</li>
                                </ul>
                                
                                
                            </div>
                            """)
                
                # Use Existing Index Tab
                with gr.Tab("📂 Connect to Repository", elem_classes=["tab-content"]):
                    with gr.Column(elem_classes=["form-section"]):
                        gr.Markdown("### Connect to Previously Processed Repository")
                        gr.Markdown("Instantly connect to repositories you've already processed.")
                        
                        with gr.Row():
                            refresh_indexes_btn = gr.Button(
                                "🔄 Load Available Repositories",
                                elem_classes=["secondary-btn"],
                                variant="secondary"
                            )
                        
                        index_info_text = gr.Textbox(
                            label="Available Repositories",
                            value="Click 'Load Available Repositories' to see your processed repositories",
                            interactive=False,
                            lines=3,
                            elem_classes=["status-text"]
                        )
                        
                        existing_index_dropdown = gr.Dropdown(
                            label="Select Repository",
                            choices=[],
                            value=None,
                            interactive=True,
                            elem_classes=["dropdown"],
                            #allow_custom_value=False
                        )
                        
                        connect_btn = gr.Button(
                            "🔗 Connect to Repository",
                            elem_classes=["primary-btn"],
                            variant="primary"
                        )
                        
                        existing_status = gr.Textbox(
                            label="Connection Status",
                            interactive=False,
                            lines=3,
                            elem_classes=["status-text"]
                        )
                        
                        with gr.Accordion("✨ Benefits of using existing repositories", open=False, elem_classes=["accordion"]):
                            gr.HTML("""
                            <div class="info-section">
                                <ul>
                                    <li><strong>⚡ Instant Access:</strong> No waiting for processing</li>
                                    <li><strong>💰 Time and effective:</strong> No re-embedding required saves the most valuable resource: TIME</li>
                                    <li><strong>🔄 Previous files:</strong> All your previous files that are processed preserved</li>
                                    <li><strong>🚀 Fast Startup:</strong> Begin chatting immediately</li>
                                </ul>
                            </div>
                            """)
        
        # Event Handlers
        def handle_chat_submit(message, history):
            if not message.strip():
                return history, ""
            return app.chat_with_repo(message, history)
        
        # Chat events
        msg.submit(handle_chat_submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
        send_btn.click(handle_chat_submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        # Processing events
        process_btn.click(
            app.process_repository,
            inputs=[github_url, index_name],
            outputs=[processing_status, process_btn]
        )
        
        refresh_btn.click(
            app.get_processing_status,
            outputs=[processing_status]
        )
        
        # Existing index events
        def refresh_indexes_handler():
            try:
                # Call the app method and ensure it returns the expected format
                result = app.refresh_indexes()
                if isinstance(result, tuple) and len(result) == 2:
                    choices, info = result
                    # Ensure choices is a list
                    if not isinstance(choices, list):
                        choices = []
                    return gr.Dropdown(choices=choices, value=None), info
                else:
                    return gr.Dropdown(choices=[], value=None), "Error loading indexes"
            except Exception as e:
                print(f"Error refreshing indexes: {e}")
                return gr.Dropdown(choices=[], value=None), f"Error: {str(e)}"
        
        refresh_indexes_btn.click(
            refresh_indexes_handler,
            inputs=[],
            outputs=[existing_index_dropdown, index_info_text]
        )
        
        def connect_to_index_handler(selected_index):
            try:
                if not selected_index:
                    return "Please select a repository from the dropdown first.", gr.Button(interactive=True)
                
                result = app.initialize_existing_index(selected_index)
                if isinstance(result, tuple) and len(result) == 2:
                    status, button_state = result
                    return status, button_state
                else:
                    return str(result), gr.Button(interactive=True)
            except Exception as e:
                print(f"Error connecting to index: {e}")
                return f"Error connecting to repository: {str(e)}", gr.Button(interactive=True)
        
        connect_btn.click(
            connect_to_index_handler,
            inputs=[existing_index_dropdown],
            outputs=[existing_status, connect_btn]
        )
        gr.HTML("""
                <div class="developer-card">
                    <div class="developer-name">Somshekar M</div>
                    <a href="https://www.linkedin.com/in/somshekar-m" target="_blank" class="developer-link">
                        LinkedIn Profile
                    </a>
                </div>
                """)
    return interface

def main():
    """Launch the interface"""
    print("🚀 Starting Repository AI Assistant...")
    print("✨ Created by Somshekar M")
    print("🔗 LinkedIn: https://www.linkedin.com/in/somshekar-m")
    interface = create_interface()
    print("🌐 Launching elegant Claude-inspired interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        favicon_path=None,
    )

if __name__ == "__main__":
    main()