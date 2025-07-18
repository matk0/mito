# FastAPI + React Migration Plan

## 🏗️ New Architecture Overview

```
slovak-health-assistant/
├── services/                    # Microservices Architecture
│   ├── ai-engine/              # AI/ML Processing Service
│   ├── api-gateway/            # Main API Gateway
│   ├── knowledge-service/      # Knowledge Base Management
│   └── analytics-service/      # Analytics & Reporting
├── frontend/                   # React Application
├── shared/                     # Shared Libraries
├── infrastructure/             # Infrastructure as Code
├── data/                       # Data Storage (unchanged)
└── deploy/                     # Deployment Configuration
```

## 🚀 Production-Ready Directory Structure

```
slovak-health-assistant/
├── services/
│   ├── ai-engine/                     # Core AI Processing Service
│   │   ├── src/
│   │   │   ├── graphrag/             # GraphRAG system (migrate existing)
│   │   │   ├── llm/                  # LLM integrations
│   │   │   ├── search/               # Vector/graph search
│   │   │   ├── entities/             # Entity extraction
│   │   │   └── pipeline/             # Processing pipeline
│   │   ├── api/                      # FastAPI endpoints
│   │   ├── models/                   # Pydantic models
│   │   ├── config/                   # Configuration
│   │   ├── tests/                    # Unit tests
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── api-gateway/                   # Main API Gateway
│   │   ├── src/
│   │   │   ├── auth/                 # Authentication
│   │   │   ├── middleware/           # CORS, rate limiting
│   │   │   ├── routes/               # API routing
│   │   │   └── websocket/            # Real-time features
│   │   ├── models/                   # Database models
│   │   ├── database/                 # Database config
│   │   ├── migrations/               # Alembic migrations
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── knowledge-service/             # Knowledge Base Management
│   │   ├── src/
│   │   │   ├── discovery/            # KB discovery
│   │   │   ├── metadata/             # KB metadata
│   │   │   ├── versioning/           # KB versions
│   │   │   └── validation/           # KB validation
│   │   ├── api/                      # FastAPI endpoints
│   │   ├── models/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── analytics-service/             # Analytics & Reporting
│       ├── src/
│       │   ├── metrics/              # Metrics collection
│       │   ├── feedback/             # Feedback system
│       │   ├── reports/              # Report generation
│       │   └── visualization/        # Data visualization
│       ├── api/
│       ├── models/
│       ├── tests/
│       ├── Dockerfile
│       └── requirements.txt
│
├── frontend/                          # React Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── chat/                 # Chat interface
│   │   │   ├── pipeline/             # Pipeline visualization
│   │   │   ├── knowledge/            # Knowledge base browser
│   │   │   ├── analytics/            # Analytics dashboard
│   │   │   └── common/               # Shared components
│   │   ├── hooks/                    # Custom React hooks
│   │   ├── services/                 # API client
│   │   ├── store/                    # State management (Zustand)
│   │   ├── utils/                    # Utilities
│   │   └── types/                    # TypeScript types
│   ├── public/
│   ├── tests/
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
│
├── shared/                            # Shared Libraries
│   ├── models/                       # Shared data models
│   ├── utils/                        # Common utilities
│   └── types/                        # TypeScript type definitions
│
├── infrastructure/                    # Infrastructure as Code
│   ├── terraform/                    # Terraform configs
│   ├── kubernetes/                   # K8s manifests
│   ├── docker-compose/               # Local development
│   └── monitoring/                   # Monitoring configs
│
├── data/                             # Data Storage (unchanged)
│   ├── knowledge_graphs/
│   ├── embeddings/
│   └── raw/
│
└── deploy/                           # Deployment Configuration
    ├── environments/                 # Environment configs
    ├── scripts/                      # Deployment scripts
    └── ci-cd/                        # CI/CD pipelines
```

## 🔧 Technology Stack

**Backend Services:**
- **FastAPI**: High-performance async API framework
- **SQLAlchemy**: Database ORM with async support
- **Alembic**: Database migrations
- **Pydantic**: Data validation and serialization
- **Redis**: Caching and session management
- **Celery**: Async task processing
- **PostgreSQL**: Primary database
- **Neo4j**: Knowledge graph
- **ChromaDB**: Vector database

**Frontend:**
- **React 18**: Modern React with concurrent features
- **TypeScript**: Type safety
- **Vite**: Fast build tool
- **Zustand**: Lightweight state management
- **React Query**: Server state management
- **Socket.io**: Real-time communication
- **Tailwind CSS**: Utility-first CSS
- **Recharts**: Data visualization

**DevOps & Infrastructure:**
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Nginx**: Reverse proxy
- **Terraform**: Infrastructure as Code
- **GitHub Actions**: CI/CD
- **Prometheus**: Metrics
- **Grafana**: Monitoring
- **ELK Stack**: Logging

## 📊 Migration Strategy

### Phase 1: Core API Migration (Weeks 1-2)
- Set up FastAPI project structure
- Migrate database models to SQLAlchemy
- Create basic CRUD endpoints
- Implement authentication system
- Set up WebSocket for real-time features

### Phase 2: AI Engine Service (Weeks 3-4)
- Migrate GraphRAG system to standalone service
- Create FastAPI wrapper for GraphRAG
- Implement async processing pipeline
- Add LLM integration endpoints
- Create search breakdown API

### Phase 3: Frontend Development (Weeks 5-6)
- Set up React application with TypeScript
- Implement chat interface with real-time updates
- Create pipeline visualization components
- Build knowledge base browser
- Add analytics dashboard

### Phase 4: Integration & Testing (Week 7)
- Integrate frontend with backend services
- End-to-end testing
- Performance optimization
- Security hardening
- Load testing

### Phase 5: Production Deployment (Week 8)
- Set up production infrastructure
- Deploy to staging environment
- Production deployment
- Monitoring and alerting setup

## 🌐 Production Deployment Architecture

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │     (Nginx)     │
                    └─────────────────┘
                             │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │    (FastAPI)    │
                    └─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ AI Engine   │ │ Knowledge   │ │ Analytics   │
    │  Service    │ │  Service    │ │  Service    │
    └─────────────┘ └─────────────┘ └─────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │  PostgreSQL     │
                    │    Neo4j        │
                    │   ChromaDB      │
                    │    Redis        │
                    └─────────────────┘
```

## 🔐 Security & Scalability Features

**Security:**
- JWT authentication with refresh tokens
- API rate limiting per user/IP
- Input validation and sanitization
- CORS configuration
- SQL injection protection
- Container security scanning

**Scalability:**
- Horizontal scaling with load balancing
- Database connection pooling
- Redis caching layer
- Async processing with Celery
- CDN for static assets
- Auto-scaling based on metrics

**Monitoring:**
- Health checks for all services
- Structured logging with correlation IDs
- Metrics collection (Prometheus)
- Error tracking (Sentry)
- Performance monitoring (APM)
- Alert management

## 📈 Production Benefits

1. **Performance**: FastAPI's async nature + React's efficient rendering
2. **Scalability**: Microservices architecture allows independent scaling
3. **Developer Experience**: TypeScript, modern tooling, hot reload
4. **Maintainability**: Clear separation of concerns, modular architecture
5. **Production Ready**: Built-in monitoring, logging, and security
6. **Cloud Native**: Kubernetes-ready, container-first approach

This architecture provides a robust, scalable, and maintainable foundation for the Slovak Health Assistant application.

## 🔄 Current Rails System Analysis

Based on analysis of the existing Rails 8 application, here's what will be migrated:

### Current Rails Controllers
- **ChatController**: Main chat interface with query processing
- **KnowledgeBasesController**: Knowledge base discovery and management
- **ReportsController**: Analytics and feedback system

### Current Database Models
- **Query**: Chat queries with metadata and responses
- **KnowledgeBase**: Knowledge base metadata and versioning
- **UserPreference**: User settings and preferences
- **FeedbackReport**: User feedback and ratings
- **QueryKnowledgeBase**: Query-KB relationships
- **QueryPipelineStep**: Processing pipeline steps

### Current GraphRAG Integration
- **graphrag_interface.py**: Command-line interface for Rails integration
- **graphrag_system.py**: Core GraphRAG system with 625 lines of code
- **Multi-database architecture**: PostgreSQL + Neo4j + ChromaDB
- **Multi-LLM support**: Ollama, OpenAI, template fallback

### Key Features to Migrate
- Real-time pipeline visualization
- Knowledge base selection and management
- Search breakdown with vector and graph results
- Session-based user preferences
- Comprehensive analytics and feedback
- Multi-source content integration
- Slovak language optimization

## 📝 Migration Checklist

### Data Migration
- [ ] Export existing PostgreSQL data
- [ ] Preserve knowledge graph data in Neo4j
- [ ] Maintain vector embeddings in ChromaDB
- [ ] Migrate user preferences and feedback

### Feature Parity
- [ ] Chat interface with real-time updates
- [ ] Pipeline visualization with search breakdown
- [ ] Knowledge base browser and selection
- [ ] Analytics dashboard
- [ ] Feedback and reporting system
- [ ] Multi-LLM integration
- [ ] Slovak language support

### Performance Optimization
- [ ] Async processing pipeline
- [ ] Connection pooling for databases
- [ ] Caching layer with Redis
- [ ] WebSocket for real-time features
- [ ] Optimized GraphRAG queries

### Production Readiness
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline setup
- [ ] Monitoring and alerting
- [ ] Security hardening
- [ ] Load testing and optimization