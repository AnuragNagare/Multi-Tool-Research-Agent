Multi-Tool Research Agent

This Multi-Tool Research Agent is a comprehensive Streamlit-based application that serves as an AI-powered research assistant. The system combines web searching capabilities, Wikipedia integration, and Large Language Model analysis to automatically gather, process, and synthesize information from multiple sources. Users can input research queries and receive detailed, analyzed reports that include web search results, Wikipedia summaries, visual data analysis, and AI-generated comprehensive summaries. The application features a clean, intuitive interface with progress tracking, research history management, and interactive data visualizations using Plotly charts.

FEATURES

Intelligent Research Planning
The system automatically generates multiple search terms based on your query to ensure comprehensive coverage of the topic. It creates a strategic approach to information gathering by analyzing your research question and determining the most effective search strategies.

Multi-Source Information Gathering
The application searches multiple information sources simultaneously including web search through DuckDuckGo, Wikipedia articles, and processes all results through advanced language models. This ensures you get both current web information and authoritative encyclopedic content.

AI-Powered Analysis and Summarization
All gathered information is processed through Hugging Face language models to create coherent, comprehensive summaries. The system identifies key insights, patterns, and provides practical recommendations based on the research findings.

Interactive Data Visualization
Research results are presented through interactive Plotly charts showing content distribution, source analysis, and statistical breakdowns of the information gathered. Users can explore data through various visual representations including bar charts and pie charts.

Research History Management
The application maintains a complete history of all research sessions, allowing users to revisit previous queries, compare results, and build upon earlier research. Each session is timestamped and can be reloaded at any time.

Progress Tracking
Real-time progress indicators show the current status of research operations, from initial planning through final summary generation. Users can see exactly what stage their research is at during processing.

INSTALLATION

Required Dependencies
You need to install the following Python packages before running the application:

streamlit
requests
pandas
matplotlib
plotly
duckduckgo-search
wikipedia

Installation Command
Install all required packages using pip:
pip install streamlit requests pandas matplotlib plotly duckduckgo-search wikipedia

SETUP

Hugging Face API Token
Before using the application, you need to obtain a free Hugging Face API token from https://huggingface.co/settings/tokens. This token is required for accessing the language models that power the AI analysis features.

Running the Application
Navigate to the directory containing the application file and run:
streamlit run your_application_filename.py

The application will open in your default web browser, typically at http://localhost:8501.

USAGE

Initial Configuration
Enter your Hugging Face API token in the sidebar configuration section. Select your preferred language model from the dropdown menu. The application includes several pre-configured models including DialoGPT Medium, DialoGPT Large, BlenderBot, FLAN-T5 Base, and CodeBERTa Small.

Research Settings
Configure your research preferences including the maximum number of search results to retrieve and whether to include Wikipedia content in your research. The application allows you to adjust these settings based on your specific research needs.

Conducting Research
Enter your research query in the main text area. The system accepts natural language questions and automatically optimizes the search strategy. Click the "Start Research" button to begin the comprehensive research process.

Viewing Results
Results are organized into four main sections: AI-Generated Summary provides a comprehensive analysis of all gathered information, Web Results shows individual search results with sources and content, Analysis displays visual charts and statistics about the research data, and Wikipedia presents relevant encyclopedic information when available.

Managing Research History
All research sessions are automatically saved and can be accessed through the Research History section. You can review previous queries, compare results across different research sessions, and reload any previous research for further analysis.

SUPPORTED MODELS

The application supports several Hugging Face language models:

DialoGPT Medium: Recommended for general research tasks, provides good balance of speed and quality
DialoGPT Large: Higher quality responses but may be slower to load
BlenderBot 400M: Good for conversational research queries
FLAN-T5 Base: Effective for instruction-following and analysis tasks
CodeBERTa Small: Suitable for technical and code-related research



TECHNICAL ARCHITECTURE

The application is built using a modular architecture with the ResearchAgent class handling all core functionality. The Streamlit framework provides the user interface, while external APIs handle search and language model operations. Data visualization is powered by Plotly, and all session management is handled through Streamlit's built-in state management system.


SUPPORT

For technical issues related to the application, check the Streamlit documentation at https://docs.streamlit.io. For Hugging Face API issues, refer to https://huggingface.co/docs. For general Python package issues, consult the respective package documentation.

This research agent represents a powerful tool for automated information gathering and analysis, suitable for students, researchers, professionals, and anyone needing comprehensive research capabilities with AI-powered insights.






