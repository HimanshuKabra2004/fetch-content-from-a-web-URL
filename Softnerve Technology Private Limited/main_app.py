import asyncio
import base64
import datetime
import os
import uuid
from typing import List, Dict, Optional, Tuple

import aiohttp
import chromadb
import gradio as gr
import numpy as np
import playwright.async_api
import torch
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
from pydantic import BaseModel
from torch import nn
from torch.optim import Adam


## ======== Configuration ======== ##
class Config:
    CHROMA_DB_PATH = "chroma_content_db"
    SCREENSHOTS_DIR = "screenshots"
    DEFAULT_MODEL = "text-davinci-003"  # Using OpenAI as placeholder, switch to Gemini API when needed
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    @staticmethod
    def setup_directories():
        os.makedirs(Config.SCREENSHOTS_DIR, exist_ok=True)


## ======== Database & Vector Storage ======== ##
class ContentDatabase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name="content_versions",
            embedding_function=self.embedding_func
        )
    
    def save_version(self, content: str, metadata: dict, spun_content: str = None) -> str:
        """Save content version with metadata and embeddings"""
        version_id = str(uuid.uuid4())
        embeddings = self.embedding_func([content])
        
        self.collection.add(
            documents=[content],
            embeddings=embeddings,
            metadatas=[metadata],
            ids=[version_id]
        )
        
        if spun_content:
            spun_id = f"{version_id}_spun"
            spun_embeddings = self.embedding_func([spun_content])
            self.collection.add(
                documents=[spun_content],
                embeddings=spun_embeddings,
                metadatas=[{**metadata, "type": "spun"}],
                ids=[spun_id]
            )
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[dict]:
        """Retrieve content version by ID"""
        result = self.collection.get(ids=[version_id])
        if not result["documents"]:
            return None
            
        return {
            "content": result["documents"][0],
            "metadata": result["metadatas"][0]
        }
    
    def search_similar(self, query: str, n_results: int = 5) -> List[dict]:
        """Search for similar content using semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return [{
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        } for i in range(len(results["ids"][0]))]


## ======== Web Scraper with Screenshot ======== ##
class ContentScraper:
    def __init__(self):
        self.session = None
    
    async def scrape_url(self, url: str) -> Tuple[str, str]:
        """Scrape content and take screenshot from webpage"""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            try:
                await page.goto(url, timeout=30000)
                
                # Get page content
                content = await page.content()
                
                # Extract text content (simple example)
                text_content = await page.evaluate("""() => {
                    return document.body.innerText;
                }""")
                
                # Take screenshot
                screenshot_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                screenshot_path = os.path.join(Config.SCREENSHOTS_DIR, screenshot_filename)
                await page.screenshot(path=screenshot_path)
                
                with open(screenshot_path, "rb") as f:
                    screenshot_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                return text_content, screenshot_b64
            finally:
                await browser.close()


## ======== AI Agents ======== ##
class LLMWrapper:
    def __init__(self):
        self.session = aiohttp.ClientSession()
    
    async def generate_content(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text using LLM API (placeholder for Gemini)"""
        # In production, replace with actual Gemini API call
        url = "https://api.openai.com/v1/completions"  # Using OpenAI as placeholder
        headers = {
            "Authorization": f"Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model or Config.DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": kwargs.get("temperature", 0.7),
            **kwargs
        }
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                return data["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            raise
    
    async def spin_chapter(self, content: str) -> str:
        """Apply AI spinning to chapter content"""
        prompt = f"""
        Rewrite the following content while maintaining its core meaning but using different phrasing, structure, and style.
        Keep all key information and facts the same, but express them in fresh new ways.
        Avoid plagiarism while ensuring the rewritten version is high quality and readable.
        
        Original content:
        {content}
        
        Rewritten version:
        """
        return await self.generate_content(prompt, temperature=0.8)
    
    async def review_chapter(self, original: str, spun: str) -> str:
        """Generate review feedback for spun content"""
        prompt = f"""
        Compare these two versions of content and provide constructive feedback on the rewritten version.
        Highlight what was improved, what might need revision, and any potential issues with accuracy, style, or flow.
        
        Original:
        {original}
        
        Rewritten:
        {spun}
        
        Professional feedback:
        """
        return await self.generate_content(prompt, temperature=0.5)


## ======== RL Search Model ======== ##
class RLSearchModel:
    """Reinforcement Learning model for content retrieval optimization"""
    def __init__(self, input_dim=512, hidden_dim=256):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def predict_relevance(self, embedding: np.array) -> float:
        """Predict relevance score for content embedding"""
        with torch.no_grad():
            tensor = torch.FloatTensor(embedding)
            return self.model(tensor).item()
    
    def train_step(self, embeddings: List[np.array], rewards: List[float]):
        """Train model with reinforcement learning"""
        predicted = self.model(torch.FloatTensor(np.array(embeddings)))
        loss = self.loss_fn(predicted.squeeze(), torch.FloatTensor(rewards))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


## ======== API & Web Interface ======== ##
class ContentSpinnerAPI:
    def __init__(self):
        Config.setup_directories()
        self.db = ContentDatabase()
        self.scraper = ContentScraper()
        self.llm = LLMWrapper()
        self.rl_model = RLSearchModel()
        
        # Initialize FastAPI app
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.app.post("/scrape")(self.api_scrape_url)
        self.app.post("/spin")(self.api_spin_content)
        self.app.post("/review")(self.api_review_content)
        self.app.get("/version/{version_id}")(self.api_get_version)
        self.app.post("/train")(self.api_train_rl)
    
    async def api_scrape_url(self, url: str):
        """API endpoint to scrape content from URL"""
        try:
            content, screenshot = await self.scraper.scrape_url(url)
            return {
                "success": True,
                "content": content,
                "screenshot": screenshot
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def api_spin_content(self, content: str):
        """API endpoint to spin content"""
        try:
            spun_content = await self.llm.spin_chapter(content)
            return {
                "success": True,
                "original": content,
                "spun_content": spun_content
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def api_review_content(self, original: str, spun: str):
        """API endpoint to review content"""
        try:
            feedback = await self.llm.review_chapter(original, spun)
            return {
                "success": True,
                "feedback": feedback
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def api_get_version(self, version_id: str):
        """API endpoint to get content version"""
        version = self.db.get_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        return version
    
    async def api_train_rl(self, embeddings: List[List[float]], rewards: List[float]):
        """API endpoint to train RL model"""
        try:
            loss = self.rl_model.train_step(embeddings, rewards)
            return {
                "success": True,
                "loss": loss
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def run_api(self, host="0.0.0.0", port=8000):
        """Run the FastAPI server"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


## ======== Gradio Interface ======== ##
def create_gradio_interface(api: ContentSpinnerAPI):
    """Create Gradio web interface for human-in-the-loop"""
    with gr.Blocks(title="AI Content Spinner") as demo:
        gr.Markdown("""
        # AI Content Spinner with Human-in-the-Loop
        Fetch, spin, review, and refine content with AI assistance
        """)
        
        with gr.Tabs():
            # Tab 1: Scraping
            with gr.Tab("Fetch Content"):
                with gr.Row():
                    url_input = gr.Textbox(label="URL", placeholder="Enter URL to scrape")
                    scrape_btn = gr.Button("Scrape Content")
                
                with gr.Row():
                    original_content = gr.Textbox(label="Original Content", lines=15, interactive=False)
                    screenshot = gr.Image(label="Screenshot", interactive=False)
                
                scrape_btn.click(
                    api.scraper.scrape_url,
                    inputs=[url_input],
                    outputs=[original_content, screenshot]
                )
            
            # Tab 2: Spinning
            with gr.Tab("Spin Content"):
                with gr.Row():
                    text_to_spin = gr.Textbox(label="Content to Spin", lines=15)
                    spin_btn = gr.Button("Spin with AI")
                
                spun_content = gr.Textbox(label="Spun Content", lines=15, interactive=True)
                
                spin_btn.click(
                    api.llm.spin_chapter,
                    inputs=[text_to_spin],
                    outputs=[spun_content]
                )
            
            # Tab 3: Review
            with gr.Tab("Review Content"):
                with gr.Row():
                    original_review = gr.Textbox(label="Original Content", lines=10)
                    spun_review = gr.Textbox(label="Spun Content", lines=10)
                review_btn = gr.Button("Generate AI Review")
                
                feedback = gr.Textbox(label="AI Feedback", lines=5)
                
                review_btn.click(
                    api.llm.review_chapter,
                    inputs=[original_review, spun_review],
                    outputs=[feedback]
                )
            
            # Tab 4: Version Management
            with gr.Tab("Version Control"):
                version_content = gr.Textbox(label="Content", lines=15)
                version_id = gr.Textbox(label="Version ID")
                
                with gr.Row():
                    save_btn = gr.Button("Save Version")
                    load_btn = gr.Button("Load Version")
                
                save_btn.click(
                    lambda content: api.db.save_version(content, {"source": "manual"}),
                    inputs=[version_content],
                    outputs=[version_id]
                )
                
                load_btn.click(
                    lambda version_id: api.db.get_version(version_id)["content"],
                    inputs=[version_id],
                    outputs=[version_content]
                )
        
        return demo


## ======== Main Execution ======== ##
async def main():
    api = ContentSpinnerAPI()
    
    # Start the API in background
    import threading
    api_thread = threading.Thread(target=api.run_api, daemon=True)
    api_thread.start()
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(api)
    interface.launch(server_port=7860)  # Removed share=True

if __name__ == "__main__":
    asyncio.run(main())
