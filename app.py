import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from ddgs import DDGS
import wikipedia
from typing import List, Dict, Any
import re
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="Multi-Tool Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ResearchAgent:
    def __init__(self, hf_token: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.hf_token = hf_token
        self.model_name = model_name
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        
        # Available models that work without special permissions
        self.available_models = {
            "microsoft/DialoGPT-medium": "DialoGPT Medium (Recommended)",
            "microsoft/DialoGPT-large": "DialoGPT Large",
            "facebook/blenderbot-400M-distill": "BlenderBot 400M",
            "google/flan-t5-base": "FLAN-T5 Base",
            "huggingface/CodeBERTa-small-v1": "CodeBERTa Small"
        }
        
    def call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Call Hugging Face LLM API with better error handling"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.hf_api_url, headers=self.headers, json=payload)
            
            if response.status_code == 403:
                return f"Model access denied. Try a different model or check your HF token permissions."
            elif response.status_code == 503:
                return f"Model is loading. Please wait a moment and try again."
            elif response.status_code != 200:
                return f"API Error {response.status_code}: {response.text}"
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].strip()
            else:
                return f"Unexpected response format: {result}"
            
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform web search using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'source': 'Web Search'
                    })
                return results
        except Exception as e:
            return [{'title': 'Search Error', 'url': '', 'snippet': f'Error: {str(e)}', 'source': 'Error'}]
    
    def search_wikipedia(self, query: str) -> Dict:
        """Search Wikipedia for detailed information"""
        try:
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return {'title': 'No results', 'content': 'No Wikipedia results found', 'url': '', 'source': 'Wikipedia'}
            
            page = wikipedia.page(search_results[0])
            return {
                'title': page.title,
                'content': page.summary[:1000] + '...' if len(page.summary) > 1000 else page.summary,
                'url': page.url,
                'source': 'Wikipedia'
            }
        except Exception as e:
            return {'title': 'Wikipedia Error', 'content': f'Error: {str(e)}', 'url': '', 'source': 'Wikipedia'}
    
    def analyze_data(self, data: List[Dict]) -> Dict:
        """Analyze search results and create insights"""
        if not data:
            return {'stats': "No data to analyze", 'chart_data': None}
        
        analysis = {
            'total_sources': len(data),
            'sources_with_content': len([d for d in data if d.get('snippet', '').strip()]),
            'average_snippet_length': sum(len(d.get('snippet', '')) for d in data) / len(data) if data else 0,
            'sources_by_type': {}
        }
        
        # Count sources by type
        for item in data:
            source_type = item.get('source', 'Unknown')
            analysis['sources_by_type'][source_type] = analysis['sources_by_type'].get(source_type, 0) + 1
        
        # Prepare chart data
        chart_data = []
        for item in data[:10]:  # Top 10 for chart
            chart_data.append({
                'title': item.get('title', 'Unknown')[:40] + '...' if len(item.get('title', '')) > 40 else item.get('title', 'Unknown'),
                'content_length': len(item.get('snippet', '') + item.get('content', '')),
                'source': item.get('source', 'Unknown')
            })
        
        return {
            'stats': analysis,
            'chart_data': pd.DataFrame(chart_data) if chart_data else None
        }

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    if 'current_research' not in st.session_state:
        st.session_state.current_research = None

def main():
    initialize_session_state()
    
    # Header
    st.title("🔍 Multi-Tool Research Agent")
    st.markdown("*An AI-powered research assistant that searches the web, analyzes data, and creates comprehensive reports*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # HF Token input
        hf_token = st.text_input(
            "Hugging Face API Token",
            type="password",
            help="Get your free token from https://huggingface.co/settings/tokens"
        )
        
        # Model selection
        st.header("🤖 Model Selection")
        
        if st.session_state.agent:
            available_models = st.session_state.agent.available_models
        else:
            available_models = {
                "microsoft/DialoGPT-medium": "DialoGPT Medium (Recommended)",
                "microsoft/DialoGPT-large": "DialoGPT Large",
                "facebook/blenderbot-400M-distill": "BlenderBot 400M",
                "google/flan-t5-base": "FLAN-T5 Base",
                "huggingface/CodeBERTa-small-v1": "CodeBERTa Small"
            }
        
        selected_model = st.selectbox(
            "Choose LLM Model",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            help="Some models may require special permissions"
        )
        
        if st.button("Initialize Agent", type="primary"):
            if hf_token:
                try:
                    st.session_state.agent = ResearchAgent(hf_token, selected_model)
                    st.success("✅ Agent initialized successfully!")
                    st.info(f"Using model: {available_models[selected_model]}")
                except Exception as e:
                    st.error(f"❌ Error initializing agent: {str(e)}")
            else:
                st.error("❌ Please provide your Hugging Face API token")
        
        # Status indicator
        if st.session_state.agent:
            st.success("🟢 Agent Ready")
        else:
            st.warning("🟡 Agent Not Initialized")
        
        st.divider()
        
        # Research settings
        st.header("🎛️ Research Settings")
        max_search_results = st.slider("Max Search Results", 3, 10, 5)
        include_wikipedia = st.checkbox("Include Wikipedia", True)
        
        st.divider()
        
        # Quick examples
        st.header("💡 Example Queries")
        example_queries = [
            "Latest developments in quantum computing",
            "Impact of AI on healthcare 2024",
            "Sustainable energy solutions",
            "Space exploration recent missions",
            "Cryptocurrency market trends"
        ]
        
        for query in example_queries:
            if st.button(f"📝 {query}", key=f"example_{query}"):
                st.session_state.example_query = query
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔎 Research Query")
        
        # Use example query if selected
        default_query = st.session_state.get('example_query', '')
        research_query = st.text_area(
            "Enter your research question:",
            value=default_query,
            height=100,
            placeholder="e.g., 'What are the latest developments in artificial intelligence?'"
        )
        
        # Clear example query after use
        if 'example_query' in st.session_state:
            del st.session_state.example_query
        
        if st.button("🚀 Start Research", type="primary", disabled=not st.session_state.agent):
            if not research_query:
                st.error("❌ Please enter a research query")
            else:
                conduct_research(research_query, max_search_results, include_wikipedia)
    
    with col2:
        st.header("📊 Research Stats")
        
        if st.session_state.research_history:
            st.metric("Total Researches", len(st.session_state.research_history))
            
            # Show recent research topics
            st.subheader("Recent Topics")
            for research in st.session_state.research_history[-3:]:
                st.write(f"• {research['query'][:50]}...")
        else:
            st.info("No research conducted yet")
    
    # Research results area
    if st.session_state.current_research:
        display_research_results()
    
    # Research history
    display_research_history()

def conduct_research(query: str, max_results: int, include_wikipedia: bool):
    """Conduct research workflow with progress tracking"""
    
    agent = st.session_state.agent
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    research_data = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'web_results': [],
        'wikipedia_result': None,
        'analysis': None,
        'summary': None
    }
    
    try:
        # Step 1: Plan research
        status_text.text("🧠 Planning research approach...")
        progress_bar.progress(10)
        
        planning_prompt = f"""
You are a research assistant. Create a research plan for the query: "{query}"
Provide 3-4 specific search terms that would help gather comprehensive information.
Format your response as a simple list of search terms, one per line.
"""
        
        search_terms_response = agent.call_llm(planning_prompt)
        search_terms = [term.strip('- ').strip() for term in search_terms_response.split('\n') if term.strip()]
        
        if not search_terms:
            search_terms = [query]
        
        st.info(f"🎯 Search strategy: {', '.join(search_terms[:3])}")
        
        # Step 2: Web search
        status_text.text("🌐 Searching the web...")
        progress_bar.progress(30)
        
        all_results = []
        for i, term in enumerate(search_terms[:3]):
            results = agent.web_search(term, max_results=max_results//3 + 1)
            all_results.extend(results)
            progress_bar.progress(30 + (i+1) * 15)
        
        research_data['web_results'] = all_results
        
        # Step 3: Wikipedia search
        if include_wikipedia:
            status_text.text("📚 Searching Wikipedia...")
            progress_bar.progress(60)
            
            wiki_result = agent.search_wikipedia(query)
            research_data['wikipedia_result'] = wiki_result
        
        # Step 4: Analyze data
        status_text.text("📊 Analyzing information...")
        progress_bar.progress(75)
        
        analysis = agent.analyze_data(all_results + ([wiki_result] if include_wikipedia and wiki_result else []))
        research_data['analysis'] = analysis
        
        # Step 5: Generate summary
        status_text.text("✍️ Generating comprehensive summary...")
        progress_bar.progress(90)
        
        # Prepare context for LLM
        context = f"Research Query: {query}\n\n"
        
        if include_wikipedia and wiki_result:
            context += f"Wikipedia Summary: {wiki_result['content'][:500]}...\n\n"
        
        context += "Web Search Results:\n"
        for result in all_results[:5]:
            context += f"- {result['title']}: {result['snippet'][:200]}...\n"
        
        summary_prompt = f"""
Based on the research query and gathered information, provide a comprehensive summary and analysis.

{context}

Please provide:
1. A clear, concise summary of the main findings
2. Key insights or patterns discovered
3. Any limitations or areas needing further research
4. Practical implications or recommendations

Be factual and cite that the information comes from web searches and Wikipedia.
"""
        
        summary = agent.call_llm(summary_prompt, max_tokens=800)
        research_data['summary'] = summary
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Research completed!")
        
        # Store results
        st.session_state.current_research = research_data
        st.session_state.research_history.append(research_data)
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success("🎉 Research completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error during research: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_research_results():
    """Display current research results"""
    research = st.session_state.current_research
    
    st.header("📋 Research Results")
    
    # Query and timestamp
    st.subheader(f"Query: {research['query']}")
    st.caption(f"Completed: {datetime.fromisoformat(research['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🌐 Web Results", "📊 Analysis", "📚 Wikipedia"])
    
    with tab1:
        st.markdown("### 🧠 AI-Generated Summary")
        if research['summary']:
            st.markdown(research['summary'])
        else:
            st.info("No summary available")
    
    with tab2:
        st.markdown("### 🌐 Web Search Results")
        if research['web_results']:
            for i, result in enumerate(research['web_results'], 1):
                with st.expander(f"{i}. {result['title']}", expanded=i<=3):
                    st.markdown(f"**Source:** {result['url']}")
                    st.markdown(f"**Content:** {result['snippet']}")
        else:
            st.info("No web results available")
    
    with tab3:
        st.markdown("### 📊 Data Analysis")
        if research['analysis'] and research['analysis']['chart_data'] is not None:
            stats = research['analysis']['stats']
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sources", stats['total_sources'])
            with col2:
                st.metric("Sources with Content", stats['sources_with_content'])
            with col3:
                st.metric("Avg Content Length", f"{stats['average_snippet_length']:.0f}")
            
            # Content length chart
            chart_data = research['analysis']['chart_data']
            if not chart_data.empty:
                fig = px.bar(
                    chart_data, 
                    x='title', 
                    y='content_length',
                    color='source',
                    title="Content Length by Source"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Source distribution
                source_counts = chart_data['source'].value_counts()
                if len(source_counts) > 1:
                    fig2 = px.pie(
                        values=source_counts.values,
                        names=source_counts.index,
                        title="Sources Distribution"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No analysis data available")
    
    with tab4:
        st.markdown("### 📚 Wikipedia Context")
        if research['wikipedia_result']:
            wiki = research['wikipedia_result']
            st.markdown(f"**Title:** {wiki['title']}")
            if wiki['url']:
                st.markdown(f"**URL:** {wiki['url']}")
            st.markdown(f"**Content:** {wiki['content']}")
        else:
            st.info("No Wikipedia data available")

def display_research_history():
    """Display research history"""
    if st.session_state.research_history:
        st.header("📚 Research History")
        
        # Show recent research in expandable format
        for i, research in enumerate(reversed(st.session_state.research_history[-10:]), 1):
            timestamp = datetime.fromisoformat(research['timestamp']).strftime('%Y-%m-%d %H:%M')
            
            with st.expander(f"{i}. {research['query']} ({timestamp})"):
                if research['summary']:
                    st.markdown("**Summary Preview:**")
                    st.markdown(research['summary'][:300] + "...")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Web Results", len(research['web_results']))
                with col2:
                    wiki_available = "Yes" if research['wikipedia_result'] else "No"
                    st.metric("Wikipedia", wiki_available)
                
                if st.button(f"Load Research {i}", key=f"load_{i}"):
                    st.session_state.current_research = research
                    st.rerun()

if __name__ == "__main__":
    main()