#!/usr/bin/env node

const { Command } = require('commander');
const fs = require('fs');
const path = require('path');
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { OllamaEmbeddings } = require('@langchain/ollama');
const { QdrantVectorStore } = require('@langchain/qdrant');
const { Ollama } = require('@langchain/ollama');
const { RetrievalQAChain } = require('langchain/chains');
const { PromptTemplate } = require('@langchain/core/prompts');
const { MemoryVectorStore } = require("langchain/vectorstores/memory")


class RAGSystem {
  constructor() {
    this.embeddings = new OllamaEmbeddings({
      baseUrl: 'http://127.0.0.1:11434',
      model: 'nomic-embed-text',
    });
    
    this.llm = new Ollama({
      baseUrl: 'http://127.0.0.1:11434',
      model: 'llama3.1',
    });
    
    this.vectorStore = null;
    this.collectionName = 'rag-documents';

    this.qdrantConfig = {
      url: 'http://127.0.0.1:6333',
      collectionName: this.collectionName,
    };
  }

  async initializeVectorStore() {
    try {
      console.log('🤖 Connecting to Qdrant')
      this.vectorStore = await QdrantVectorStore.fromExistingCollection(this.embeddings, this.qdrantConfig);
      console.log('✅ Connected to existing Qdrant collection');
    } catch (error) {
      console.log('📁 Creating new Qdrant collection...');
      this.vectorStore = new MemoryVectorStore(this.embeddings);
    }
  }

  async ingestPDF(filePath) {
    try {
      console.log(`📖 Loading PDF: ${filePath}`);
      
      // Check if file exists
      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
      }

      // Load PDF
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      console.log(`📄 Loaded ${docs.length} pages from PDF`);

      // Split documents into chunks
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      
      const splitDocs = await textSplitter.splitDocuments(docs);
      console.log(`✂️  Split into ${splitDocs.length} chunks`);

      // Add metadata
      splitDocs.forEach((doc, index) => {
        doc.metadata = {
          ...doc.metadata,
          source: path.basename(filePath),
          chunk_id: index,
          ingested_at: new Date().toISOString(),
        };
      });

      // Initialize vector store if not already done
      if (!this.vectorStore) {
        await this.initializeVectorStore();
      }

      // Add documents to vector store
      console.log('🔄 Creating embeddings and storing in Qdrant...');
      await this.vectorStore.addDocuments(splitDocs);
      
      console.log('✅ PDF ingestion completed successfully!');
      console.log(`📊 Total chunks stored: ${splitDocs.length}`);
      
    } catch (error) {
      console.error('❌ Error during PDF ingestion:', error.message);
      throw error;
    }
  }

  async query(question, numResults = 4) {
    try {
      console.log(`🔍 Searching for: "${question}"`);
      
      // Initialize vector store if not already done
      if (!this.vectorStore) {
        await this.initializeVectorStore();
      }

      // Create retriever
      const retriever = this.vectorStore.asRetriever({
        k: numResults,
      });

      // Create custom prompt template
      const promptTemplate = new PromptTemplate({
        template: `Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:`,
        inputVariables: ['context', 'question'],
      });

      // Create QA chain
      const qaChain = RetrievalQAChain.fromLLM(
        this.llm,
        retriever,
        {
          prompt: promptTemplate,
          returnSourceDocuments: true,
        }
      );

      // Get answer
      console.log('🤖 Generating answer...');
      const response = await qaChain.call({
        query: question,
      });

      // Display results
      console.log('\n📝 Answer:');
      console.log('=' .repeat(50));
      console.log(response.text);
      
      console.log('\n📚 Sources:');
      console.log('=' .repeat(50));
      response.sourceDocuments.forEach((doc, index) => {
        console.log(`${index + 1}. Source: ${doc.metadata.source}`);
        console.log(`   Page: ${doc.metadata.loc?.pageNumber || 'Unknown'}`);
        console.log(`   Chunk: ${doc.metadata.chunk_id}`);
        console.log(`   Content preview: ${doc.pageContent.substring(0, 150)}...`);
        console.log('');
      });

      return response;
      
    } catch (error) {
      console.error('❌ Error during query:', error.message);
      throw error;
    }
  }

  async listDocuments() {
    try {
      if (!this.vectorStore) {
        await this.initializeVectorStore();
      }

      // Get all documents (this is a simplified approach)
      const results = await this.vectorStore.similaritySearch('', 100);
      
      const sources = new Set();
      results.forEach(doc => {
        if (doc.metadata.source) {
          sources.add(doc.metadata.source);
        }
      });

      console.log('\n📚 Ingested Documents:');
      console.log('=' .repeat(30));
      if (sources.size === 0) {
        console.log('No documents found in the database.');
      } else {
        Array.from(sources).forEach((source, index) => {
          console.log(`${index + 1}. ${source}`);
        });
      }
      
      console.log(`\nTotal unique documents: ${sources.size}`);
      console.log(`Total chunks: ${results.length}`);
      
    } catch (error) {
      console.error('❌ Error listing documents:', error.message);
    }
  }

  async clearDatabase() {
    try {
      console.log('🗑️  Clearing Qdrant collection...');
      
      if (!this.vectorStore) {
        await this.initializeVectorStore();
      }
      
      // Delete all points in the collection
      const qdrantClient = this.vectorStore.client;
      await qdrantClient.delete(this.collectionName, {
        filter: {} // Empty filter deletes all points
      });
      
      console.log('✅ Qdrant collection cleared successfully');
      
    } catch (error) {
      console.error('❌ Error clearing database:', error.message);
      console.log('💡 You can also restart Qdrant service to clear all data');
    }
  }
}

// CLI Setup
const program = new Command();
const rag = new RAGSystem();

program
  .name('rag-cli')
  .description('RAG (Retrieval Augmented Generation) CLI using LangChain, Ollama, and Qdrant')
  .version('1.0.0');

// Ingest command
program
  .command('ingest')
  .description('Ingest a PDF file into the RAG system')
  .argument('<file>', 'Path to the PDF file')
  .action(async (file) => {
    try {
      await rag.ingestPDF(file);
    } catch (error) {
      process.exit(1);
    }
  });

// Query command
program
  .command('query')
  .description('Query the RAG system')
  .argument('<question>', 'Question to ask')
  .option('-n, --num-results <number>', 'Number of results to retrieve', '4')
  .action(async (question, options) => {
    try {
      await rag.query(question, parseInt(options.numResults));
    } catch (error) {
      process.exit(1);
    }
  });

// List command
program
  .command('list')
  .description('List all ingested documents')
  .action(async () => {
    try {
      await rag.listDocuments();
    } catch (error) {
      process.exit(1);
    }
  });

// Clear command
program
  .command('clear')
  .description('Clear the document database')
  .action(async () => {
    try {
      await rag.clearDatabase();
    } catch (error) {
      process.exit(1);
    }
  });

// Setup command
program
  .command('setup')
  .description('Display setup instructions')
  .action(() => {
    console.log(`
🚀 RAG CLI Setup Instructions
============================

1. Install and start Ollama:
   - Download from: https://ollama.ai
   - Pull required models:
     ollama pull llama3.1
     ollama pull nomic-embed-text

2. Install and start Qdrant:
   
   Option A - Using Docker (Recommended):
   docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

   
   Option B - Using Binary:
   - Download from: https://github.com/qdrant/qdrant/releases
   - Run: ./qdrant --config-path config/config.yaml
   
   Option C - Using Snap:
   sudo snap install qdrant
   qdrant

3. Install Node.js dependencies:
   pnpm install

4. Usage Examples:
   # Ingest a PDF
   node index.js ingest ./document.pdf
   
   # Query the system
   node index.js query "What is the main topic of the document?"
   
   # List ingested documents
   node index.js list
   
   # Clear database
   node index.js clear

📝 Prerequisites:
- Node.js (v18+)
- Ollama running on localhost:11434
- Qdrant running on localhost:6333
- PDF files for ingestion

🔧 Configuration:
- Default embedding model: nomic-embed-text
- Default LLM model: llama3.1
- Default chunk size: 1000 characters
- Default chunk overlap: 200 characters
- Qdrant collection: rag-documents

🌐 Qdrant Web UI:
- Access at: http://localhost:6333/dashboard
- Monitor collections and search performance
    `);
  });

// Parse CLI arguments
program.parse();

module.exports = { RAGSystem };
