"""
Simplified Streamlit interface for showcasing the LLM-Powered Q&A System.

This version uses mock data and simplified components for easy deployment
on Streamlit Cloud without requiring heavy ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="LLM-Powered Q&A System - Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .demo-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Mock data for demonstration
MOCK_ANSWERS = {
    "What is machine learning?": {
        "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions. Key types include supervised learning, unsupervised learning, and reinforcement learning.",
        "sources": [
            {"title": "Machine Learning Basics", "source": "Wikipedia", "similarity": 0.95},
            {"title": "ML Fundamentals", "source": "Documentation", "similarity": 0.88},
            {"title": "AI and ML Guide", "source": "News", "similarity": 0.82}
        ],
        "confidence": 0.92,
        "processing_time": 1.8
    },
    "How does BERT work?": {
        "answer": "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that uses bidirectional training to understand context from both directions. It uses attention mechanisms to process text and can be fine-tuned for various NLP tasks like question answering, text classification, and named entity recognition.",
        "sources": [
            {"title": "BERT Architecture", "source": "Documentation", "similarity": 0.94},
            {"title": "Transformer Models", "source": "Wikipedia", "similarity": 0.89},
            {"title": "NLP Advances", "source": "News", "similarity": 0.85}
        ],
        "confidence": 0.89,
        "processing_time": 2.1
    },
    "What are the latest trends in AI?": {
        "answer": "Current AI trends include large language models (LLMs), multimodal AI, AI ethics and governance, edge AI deployment, and AI-powered automation. There's also growing focus on responsible AI, explainable AI, and AI for sustainability. The field is rapidly evolving with new breakthroughs in generative AI and foundation models.",
        "sources": [
            {"title": "AI Trends 2024", "source": "News", "similarity": 0.96},
            {"title": "Latest in AI", "source": "News", "similarity": 0.91},
            {"title": "AI Development", "source": "GitHub", "similarity": 0.87}
        ],
        "confidence": 0.94,
        "processing_time": 1.6
    },
    "Explain the RAG architecture": {
        "answer": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It works by first retrieving relevant documents from a knowledge base using semantic search, then using a language model to generate answers based on the retrieved context. This approach provides more accurate, up-to-date, and verifiable answers compared to traditional language models.",
        "sources": [
            {"title": "RAG Architecture", "source": "Documentation", "similarity": 0.97},
            {"title": "Retrieval Systems", "source": "Wikipedia", "similarity": 0.90},
            {"title": "AI Systems", "source": "GitHub", "similarity": 0.84}
        ],
        "confidence": 0.91,
        "processing_time": 2.3
    },
    "What is FAISS and how is it used?": {
        "answer": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It's commonly used in recommendation systems, image search, and NLP applications for finding similar items quickly. FAISS supports various index types and can handle billions of vectors efficiently.",
        "sources": [
            {"title": "FAISS Documentation", "source": "Documentation", "similarity": 0.93},
            {"title": "Vector Search", "source": "Wikipedia", "similarity": 0.88},
            {"title": "ML Libraries", "source": "GitHub", "similarity": 0.86}
        ],
        "confidence": 0.88,
        "processing_time": 1.9
    },
    "How do transformers work in NLP?": {
        "answer": "Transformers use self-attention mechanisms to process sequences of data, allowing them to capture relationships between different parts of the input. They consist of encoder and decoder layers with multi-head attention, enabling parallel processing and better understanding of context. This architecture has revolutionized NLP tasks.",
        "sources": [
            {"title": "Transformer Architecture", "source": "Documentation", "similarity": 0.95},
            {"title": "NLP Models", "source": "Wikipedia", "similarity": 0.89},
            {"title": "AI Research", "source": "News", "similarity": 0.83}
        ],
        "confidence": 0.90,
        "processing_time": 2.0
    }
}

def generate_mock_answer(question):
    """Generate a mock answer for demonstration."""
    # Check if we have a predefined answer
    for key in MOCK_ANSWERS:
        if question.lower() in key.lower() or key.lower() in question.lower():
            return MOCK_ANSWERS[key]
    
    # Generate a generic answer
    return {
        "answer": f"This is a demonstration of the RAG system. Your question about '{question}' would be processed by retrieving relevant documents and generating an answer using the Mistral-7B language model. In a real deployment, this would provide accurate, source-attributed answers based on the latest information from our knowledge base.",
        "sources": [
            {"title": "Relevant Documentation", "source": "Documentation", "similarity": 0.85},
            {"title": "Related Article", "source": "Wikipedia", "similarity": 0.78},
            {"title": "Latest Information", "source": "News", "similarity": 0.72}
        ],
        "confidence": 0.82,
        "processing_time": 1.5 + np.random.random()
    }

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LLM-Powered Multi-Source Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced MLOps Demo - Automated Data Freshness Pipeline</p>', unsafe_allow_html=True)
    
    # Demo notice
    with st.container():
        st.markdown("""
        <div class="demo-box">
            <strong>🎯 Demo Mode:</strong> This is a demonstration of the MLOps-powered Q&A system. 
            The actual system would use real BERT embeddings, FAISS vector search, and Mistral-7B LLM 
            with automated data freshness management.
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # System status
        st.subheader("System Status")
        st.success("✅ Demo System Ready")
        
        # Mock system info
        st.write("**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2")
        st.write("**LLM Model:** mistralai/Mistral-7B-Instruct-v0.2")
        st.write("**Index Documents:** 1,247")
        st.write("**Last Update:** 2 hours ago")
        
        # MLOps features
        st.subheader("🔧 MLOps Features")
        features = [
            "✅ Automated Data Freshness",
            "✅ Multi-Source Collection", 
            "✅ CI/CD Pipeline",
            "✅ Production Monitoring",
            "✅ Zero-Downtime Deployment"
        ]
        for feature in features:
            st.write(feature)
        
        # About section
        st.subheader("ℹ️ About This Demo")
        st.markdown("""
        This demonstrates advanced MLOps skills:
        - **Automated Data Freshness**: Keeps knowledge current
        - **Multi-Source Collection**: Documentation, Wikipedia, News, GitHub
        - **Production RAG Pipeline**: BERT + FAISS + Mistral-7B
        - **MLOps Best Practices**: CI/CD, monitoring, deployment
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🤔 Ask Questions", "📊 System Info", "🔧 MLOps Pipeline", "📈 Performance"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Ask Questions</h2>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="Ask about machine learning, AI, programming, or any topic...",
            height=100
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                k_results = st.slider("Number of sources to retrieve", 1, 10, 5)
                temperature = st.slider("Response creativity", 0.1, 1.0, 0.7)
            with col2:
                include_sources = st.checkbox("Show source details", True)
                show_confidence = st.checkbox("Show confidence score", True)
        
        # Submit button
        if st.button("🚀 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Processing your question..."):
                    # Simulate processing time
                    time.sleep(1.5)
                    
                    # Get mock answer
                    result = generate_mock_answer(question)
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.write(result["answer"])
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col2:
                        st.metric("Sources Retrieved", len(result["sources"]))
                    with col3:
                        if show_confidence:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                    
                    # Display sources
                    if include_sources and result.get("sources"):
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}: {source['title']}"):
                                st.write(f"**Source:** {source['source']}")
                                st.write(f"**Similarity:** {source['similarity']:.1%}")
            else:
                st.warning("Please enter a question.")
        
        # Example questions
        st.markdown("### 💡 Example Questions")
        example_questions = list(MOCK_ANSWERS.keys())
        
        cols = st.columns(3)
        for i, example in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(example[:30] + "..." if len(example) > 30 else example, key=f"example_{i}"):
                    st.session_state.question = example
                    st.rerun()
    
    with tab2:
        st.markdown('<h2 class="sub-header">System Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Pipeline Components")
            
            pipeline_info = {
                "embedding_model": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384,
                    "device": "auto"
                },
                "vector_database": {
                    "type": "FAISS IndexFlatIP",
                    "documents": 1247,
                    "dimension": 384
                },
                "llm_model": {
                    "name": "mistralai/Mistral-7B-Instruct-v0.2",
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            }
            
            st.json(pipeline_info)
        
        with col2:
            st.markdown("### 📊 System Statistics")
            
            stats = {
                "Total Documents": "1,247",
                "Embedding Dimension": "384",
                "Index Type": "FAISS IndexFlatIP",
                "Max Context Length": "4,000",
                "System Uptime": "24 hours",
                "Last Data Update": "2 hours ago"
            }
            
            for key, value in stats.items():
                st.metric(key, value)
    
    with tab3:
        st.markdown('<h2 class="sub-header">MLOps Pipeline</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔄 Automated Data Freshness Pipeline
        
        This system demonstrates advanced MLOps with automated data freshness management:
        """)
        
        # Pipeline steps
        steps = [
            {
                "step": "1. Data Collection",
                "description": "Automated scraping from multiple sources every 6 hours",
                "sources": ["Documentation", "Wikipedia", "News Feeds", "GitHub"]
            },
            {
                "step": "2. Data Processing", 
                "description": "Cleaning, chunking, and embedding generation",
                "sources": ["BERT Embeddings", "Text Chunking", "Metadata Extraction"]
            },
            {
                "step": "3. Index Building",
                "description": "FAISS vector database construction and optimization",
                "sources": ["Vector Indexing", "Similarity Search", "Performance Optimization"]
            },
            {
                "step": "4. Quality Testing",
                "description": "Automated validation of new models and data",
                "sources": ["Accuracy Tests", "Performance Benchmarks", "Quality Gates"]
            },
            {
                "step": "5. Deployment",
                "description": "Zero-downtime model updates with rollback capability",
                "sources": ["Blue-Green Deployment", "Health Checks", "Monitoring"]
            }
        ]
        
        for step in steps:
            with st.expander(f"🔧 {step['step']}"):
                st.write(f"**Description:** {step['description']}")
                st.write(f"**Components:** {', '.join(step['sources'])}")
        
        # GitHub Actions workflow
        st.markdown("### 📋 CI/CD Pipeline")
        st.code("""
name: Data Freshness Pipeline
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
jobs:
  check-data-freshness:
    runs-on: ubuntu-latest
    steps:
      - name: Check data freshness
        run: python scripts/check_data_freshness.py
  collect-data:
    needs: check-data-freshness
    runs-on: ubuntu-latest
    steps:
      - name: Collect fresh data
        run: python scripts/collect_data.py --all-sources
  rebuild-index:
    needs: collect-data
    runs-on: ubuntu-latest
    steps:
      - name: Rebuild vector index
        run: python scripts/rebuild_index.py
  deploy:
    needs: rebuild-index
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: python scripts/deploy_index.py
        """, language="yaml")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Performance Metrics</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⏱️ Response Times")
            
            response_times = {
                "Average": "1.8s",
                "95th Percentile": "2.5s", 
                "99th Percentile": "4.2s"
            }
            
            for metric, value in response_times.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("### 📈 Accuracy Metrics")
            
            accuracy_metrics = {
                "Overall Accuracy": "95.2%",
                "Source Relevance": "92.8%",
                "Answer Quality": "94.1%"
            }
            
            for metric, value in accuracy_metrics.items():
                st.metric(metric, value)
        
        # Performance chart
        st.markdown("### 📊 Recent Performance")
        
        # Generate mock performance data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Response Time (s)': np.random.normal(1.8, 0.3, 30),
            'Accuracy (%)': np.random.normal(95, 2, 30),
            'Requests': np.random.poisson(150, 30)
        })
        
        st.line_chart(performance_data.set_index('Date'))
        
        # MLOps metrics
        st.markdown("### 🔧 MLOps Metrics")
        
        mlops_metrics = {
            "Data Freshness": "2 hours",
            "Model Version": "v1.2.3",
            "Deployment Success Rate": "99.8%",
            "System Uptime": "99.9%",
            "Automated Tests Pass Rate": "100%"
        }
        
        cols = st.columns(3)
        for i, (metric, value) in enumerate(mlops_metrics.items()):
            with cols[i % 3]:
                st.metric(metric, value)

if __name__ == "__main__":
    main() 