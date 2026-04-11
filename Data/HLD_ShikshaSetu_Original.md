# shikshasetu - High-Level Design (HLD)

**Version:** 1.0
**Date:** February 2026
**Status:** Design Phase

---

## 1. Executive Summary

shikshasetu is a multi-tenant SaaS platform designed to digitize offline coaching institutes (Classes 8-12, JEE/NEET). The platform provides:
- Tenant-isolated infrastructure with subdomain routing
- Structured test engine (Objective + Subjective)
- AI-powered analytics and performance diagnostics
- Content management system
- Role-based dashboards

**Core Differentiator:** AI-powered concept-level diagnostic engine, not just a content platform.

---

## 2. System Architecture

### 2.1 Architecture Pattern

**Multi-Tenant Architecture with Shared Infrastructure**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Load Balancer / CDN                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ coaching1      │  │ coaching2       │  │ admin          │
│ .shikshasetu    │  │ .shikshasetu     │  │ .shikshasetu    │
│ .com           │  │ .com            │  │ .com           │
└───────┬────────┘  └────────┬────────┘  └───────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────▼────────────────────────────┐
        │          API Gateway / Tenant Router            │
        │   - Subdomain parsing                           │
        │   - Tenant context injection                    │
        │   - Rate limiting per tenant                    │
        └────────────────────┬────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ Tenant Mgmt    │  │ Auth Service    │  │ Content Mgmt   │
│ Service        │  │                 │  │ Service        │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ Test Engine    │  │ Analytics       │  │ AI/ML Engine   │
│ Service        │  │ Service         │  │                │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────▼────────────────────────────┐
        │              Data Layer                          │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
        │  │MongoDB   │  │PostgreSQL│  │Redis     │      │
        │  │(Content) │  │(Tenant/  │  │(Cache)   │      │
        │  │          │  │ Analytics)│  │          │      │
        │  └──────────┘  └──────────┘  └──────────┘      │
        │                                                  │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
        │  │S3/Blob   │  │ElasticS. │  │RabbitMQ  │      │
        │  │(Files)   │  │(Search)  │  │(Queue)   │      │
        │  └──────────┘  └──────────┘  └──────────┘      │
        └──────────────────────────────────────────────────┘
```

### 2.2 Microservices Architecture

**Core Services:**

1. **Tenant Management Service**
   - Tenant onboarding and provisioning
   - Subscription management
   - Billing and payment processing
   - Subdomain management
   - Tenant configuration

2. **Authentication & Authorization Service**
   - Multi-tenant authentication
   - Role-based access control (Student/Teacher/Admin)
   - JWT token management
   - Session management
   - SSO integration (future)

3. **Content Management Service**
   - Question bank management
   - Content versioning
   - Tagging system (subject/chapter/topic/difficulty)
   - Publishing workflow
   - Media storage integration

4. **Test Engine Service**
   - Test creation and configuration
   - Test delivery and rendering
   - Answer submission and storage
   - Timer management
   - Section-wise test logic
   - Subjective answer upload

5. **Evaluation Service**
   - Auto-evaluation (Objective)
   - Manual evaluation interface (Subjective)
   - Rubric-based scoring
   - AI-assisted evaluation (future)
   - Grade calculation

6. **Analytics Engine Service**
   - Performance metrics calculation
   - Rank computation
   - Weak topic detection
   - Concept gap analysis
   - Trend analysis
   - Comparison metrics

7. **AI/ML Service**
   - Concept clustering
   - Difficulty calibration
   - Rank prediction
   - Performance modeling
   - LLM-based evaluation (Phase 2)
   - Adaptive testing (Phase 2)

8. **Notification Service**
   - Email notifications
   - SMS alerts
   - In-app notifications
   - Push notifications
   - Event-driven triggers

9. **Reporting Service**
   - Dashboard data aggregation
   - Report generation
   - Export functionality
   - Visualization data preparation

---

## 3. Data Architecture

### 3.1 Database Strategy

**Hybrid Database Approach:**

**PostgreSQL (Relational):**
- Tenant metadata
- User accounts
- Subscriptions and billing
- Test attempts and scores
- Analytics aggregations
- Strong ACID compliance needed

**MongoDB (Document):**
- Content and questions
- Test configurations
- Study materials
- Flexible schema for content
- Fast reads for content delivery

**Redis (Cache + Session):**
- Session management
- Rate limiting
- Real-time test state
- Leaderboard caching
- Analytics cache

**Elasticsearch:**
- Question search
- Content discovery
- Full-text search
- Advanced filtering

**S3/Blob Storage:**
- Study material files
- Subjective answer uploads
- Profile images
- Generated reports

### 3.2 Data Isolation Strategy

**Tenant Isolation Model: Shared Database with Tenant ID**

```sql
-- Every table has tenant_id
CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,  -- Tenant isolation
    email VARCHAR(255) NOT NULL,
    role VARCHAR(50),
    created_at TIMESTAMP,
    CONSTRAINT fk_tenant FOREIGN KEY (tenant_id)
        REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX idx_users_tenant ON users(tenant_id);

-- Row-level security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_policy ON users
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

**Benefits:**
- Cost-effective (shared infrastructure)
- Easy backups and maintenance
- Efficient resource utilization
- Tenant-level data export capability

**Security Measures:**
- Row-level security (RLS)
- Application-level tenant filtering
- Encrypted tenant_id in JWT
- Audit logging per tenant

### 3.3 Core Data Models

**Tenant Model:**
```javascript
{
  id: UUID,
  name: String,
  subdomain: String (unique),
  plan: 'basic' | 'pro' | 'enterprise',
  status: 'active' | 'suspended' | 'trial',
  subscription: {
    startDate: Date,
    endDate: Date,
    studentLimit: Number,
    currentStudents: Number
  },
  branding: {
    logo: String,
    primaryColor: String,
    customDomain: String (optional)
  },
  settings: {
    allowSubjectiveTests: Boolean,
    enableAI: Boolean,
    retentionDays: Number
  },
  createdAt: Date,
  updatedAt: Date
}
```

**User Model (Multi-tenant):**
```javascript
{
  id: UUID,
  tenantId: UUID,
  email: String,
  passwordHash: String,
  role: 'student' | 'teacher' | 'admin',
  profile: {
    name: String,
    class: String,
    stream: String,
    phone: String,
    avatar: String
  },
  status: 'active' | 'inactive',
  createdAt: Date,
  lastLogin: Date
}
```

**Question Model:**
```javascript
{
  id: UUID,
  tenantId: UUID,  // null for shikshasetu default content
  type: 'MCQ' | 'subjective',
  subject: String,
  chapter: String,
  topic: String,
  subtopic: String,
  difficulty: 'easy' | 'medium' | 'hard',
  examType: 'JEE' | 'NEET' | 'Board' | 'Other',
  question: String,
  options: [String],  // For MCQ
  correctAnswer: String,  // For MCQ
  explanation: String,
  marks: Number,
  negativeMarks: Number,
  tags: [String],
  metadata: {
    bloomLevel: String,
    conceptId: UUID,
    averageTime: Number,
    successRate: Number,
    attemptCount: Number
  },
  isActive: Boolean,
  publishStatus: 'draft' | 'review' | 'published',
  createdBy: UUID,
  createdAt: Date
}
```

**Test Model:**
```javascript
{
  id: UUID,
  tenantId: UUID,
  title: String,
  description: String,
  type: 'mock' | 'section' | 'practice',
  subjects: [{
    subject: String,
    sections: [{
      name: String,
      questionIds: [UUID],
      timeLimit: Number,
      marksPerQuestion: Number,
      negativeMarking: Number
    }]
  }],
  totalMarks: Number,
  duration: Number,  // minutes
  startTime: Date,
  endTime: Date,
  isLive: Boolean,
  settings: {
    shuffleQuestions: Boolean,
    showResults: 'immediate' | 'after_end',
    allowReview: Boolean,
    calculatorAllowed: Boolean
  },
  targetAudience: {
    classes: [String],
    batches: [UUID]
  },
  createdBy: UUID,
  createdAt: Date
}
```

**TestAttempt Model:**
```javascript
{
  id: UUID,
  tenantId: UUID,
  testId: UUID,
  studentId: UUID,
  status: 'ongoing' | 'submitted' | 'evaluated',
  answers: {
    questionId: {
      answer: String,
      timeSpent: Number,
      markedForReview: Boolean,
      visitCount: Number
    }
  },
  subjectiveUploads: {
    questionId: {
      fileUrl: String,
      uploadedAt: Date
    }
  },
  evaluation: {
    totalMarks: Number,
    marksObtained: Number,
    correctAnswers: Number,
    incorrectAnswers: Number,
    unattempted: Number,
    rank: Number,
    percentile: Number,
    subjectWise: {
      subject: {
        marks: Number,
        obtained: Number,
        accuracy: Number
      }
    }
  },
  analytics: {
    topicWise: {
      topic: {
        attempted: Number,
        correct: Number,
        avgTime: Number
      }
    },
    difficultyWise: {
      difficulty: {
        attempted: Number,
        correct: Number
      }
    }
  },
  startedAt: Date,
  submittedAt: Date,
  evaluatedAt: Date
}
```

**Concept Model (for AI):**
```javascript
{
  id: UUID,
  name: String,
  subject: String,
  chapter: String,
  parentConcept: UUID,
  difficulty: Number,  // 1-10 scale
  prerequisites: [UUID],  // Other concepts needed first
  relatedConcepts: [UUID],
  learningOutcome: String,
  createdAt: Date
}
```

---

## 4. Technology Stack

### 4.1 Backend

**Core Framework:**
- **Node.js** (Runtime)
- **NestJS** (Microservices framework with TypeScript)
  - Why: Built-in microservices support, dependency injection, modular architecture
- **Express.js** (Alternative for simpler services)

**Databases:**
- **PostgreSQL 15+** (Primary relational DB)
- **MongoDB 6+** (Content storage)
- **Redis 7+** (Cache, sessions, queues)

**Message Queue:**
- **RabbitMQ** or **Apache Kafka** (Event-driven communication)

**Search:**
- **Elasticsearch 8+** (Content search)

**Storage:**
- **AWS S3** or **MinIO** (File storage)

**Authentication:**
- **Passport.js** (Auth strategies)
- **JWT** (Token-based auth)

**API:**
- **GraphQL** (Flexible queries for dashboards)
- **REST** (CRUD operations)
- **WebSocket** (Real-time test updates)

### 4.2 Frontend

**Framework:**
- **Next.js 14+** (React framework with SSR)
  - Why: SSR for SEO, API routes, easy multi-tenant routing

**UI Library:**
- **React 18+**
- **TypeScript**
- **Tailwind CSS** (Styling)
- **shadcn/ui** (Component library)

**State Management:**
- **Zustand** (Lightweight state management)
- **React Query** (Server state management)

**Charts/Analytics:**
- **Recharts** or **Chart.js** (Visualization)
- **D3.js** (Advanced visualizations)

**Real-time:**
- **Socket.io** (WebSocket client)

### 4.3 AI/ML Stack

**Framework:**
- **Python 3.11+** (ML/AI service)
- **FastAPI** (Python API framework)

**ML Libraries:**
- **scikit-learn** (Traditional ML)
- **TensorFlow/PyTorch** (Deep learning)
- **LangChain** (LLM orchestration)
- **OpenAI API** or **Azure OpenAI** (LLM for evaluation)

**MLOps:**
- **MLflow** (Model tracking)
- **DVC** (Data versioning)

### 4.4 DevOps & Infrastructure

**Containerization:**
- **Docker** (Containerization)
- **Docker Compose** (Local development)

**Orchestration:**
- **Kubernetes** (Production orchestration)
- **Helm** (K8s package manager)

**CI/CD:**
- **GitHub Actions** or **GitLab CI**
- **ArgoCD** (GitOps)

**Cloud Provider:**
- **AWS** (Primary)
  - EC2/ECS/EKS (Compute)
  - RDS (Managed PostgreSQL)
  - DocumentDB (Managed MongoDB)
  - ElastiCache (Redis)
  - S3 (Storage)
  - CloudFront (CDN)
  - Route53 (DNS)
  - SES (Email)

**Monitoring:**
- **Prometheus** + **Grafana** (Metrics)
- **ELK Stack** (Logging)
- **Sentry** (Error tracking)
- **DataDog** (APM - optional)

**Security:**
- **AWS WAF** (Web firewall)
- **AWS Secrets Manager** (Secret management)
- **Vault** (Alternative secret management)

---

## 5. Multi-Tenant Implementation

### 5.1 Subdomain Routing

**DNS Configuration:**
```
*.shikshasetu.com → Load Balancer

coaching1.shikshasetu.com → API Gateway
coaching2.shikshasetu.com → API Gateway
admin.shikshasetu.com → Admin Portal
```

**Next.js Middleware (Tenant Resolution):**
```typescript
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  const hostname = request.headers.get('host');
  const subdomain = hostname?.split('.')[0];

  if (subdomain && subdomain !== 'www' && subdomain !== 'admin') {
    // Inject tenant context
    const url = request.nextUrl.clone();
    url.searchParams.set('tenant', subdomain);

    // Rewrite to tenant-specific route
    return NextResponse.rewrite(url);
  }

  return NextResponse.next();
}
```

**API Gateway Tenant Context:**
```typescript
// Tenant middleware
async function tenantMiddleware(req, res, next) {
  const subdomain = req.hostname.split('.')[0];

  // Fetch tenant from database
  const tenant = await Tenant.findOne({ subdomain });

  if (!tenant) {
    return res.status(404).json({ error: 'Tenant not found' });
  }

  if (tenant.status !== 'active') {
    return res.status(403).json({ error: 'Tenant suspended' });
  }

  // Inject tenant context
  req.tenant = tenant;
  req.tenantId = tenant.id;

  // Set PostgreSQL session variable for RLS
  await pool.query(`SET app.current_tenant = '${tenant.id}'`);

  next();
}
```

### 5.2 Data Isolation

**Application-Level Filtering:**
```typescript
// Every query automatically includes tenant filter
class TenantAwareRepository<T> {
  constructor(private tenantId: string) {}

  async find(filters: any): Promise<T[]> {
    return this.model.find({
      ...filters,
      tenantId: this.tenantId  // Always filter by tenant
    });
  }

  async create(data: any): Promise<T> {
    return this.model.create({
      ...data,
      tenantId: this.tenantId  // Always set tenant
    });
  }
}
```

**Database-Level RLS (PostgreSQL):**
```sql
-- Automatic row-level filtering
CREATE POLICY tenant_isolation ON test_attempts
  USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

---

## 6. API Design

### 6.1 API Gateway Pattern

**Subdomain-based routing:**
```
GET coaching1.shikshasetu.com/api/tests
  → API Gateway
  → Tenant Context Injection (coaching1)
  → Test Service
  → Filter by tenant_id
```

### 6.2 Key API Endpoints

**Tenant Management:**
```
POST   /api/admin/tenants              # Create tenant
GET    /api/admin/tenants/:id          # Get tenant
PUT    /api/admin/tenants/:id          # Update tenant
DELETE /api/admin/tenants/:id          # Suspend tenant
POST   /api/admin/tenants/:id/activate # Activate tenant
```

**Authentication:**
```
POST   /api/auth/login                 # Login (tenant-aware)
POST   /api/auth/register              # Register student
POST   /api/auth/refresh               # Refresh token
POST   /api/auth/logout                # Logout
GET    /api/auth/me                    # Get current user
```

**Content Management:**
```
GET    /api/content/questions          # List questions (tenant + global)
POST   /api/content/questions          # Create question
PUT    /api/content/questions/:id      # Update question
DELETE /api/content/questions/:id      # Delete question
POST   /api/content/questions/import   # Bulk import
GET    /api/content/questions/search   # Search questions
```

**Test Engine:**
```
GET    /api/tests                      # List tests
POST   /api/tests                      # Create test
GET    /api/tests/:id                  # Get test details
PUT    /api/tests/:id                  # Update test
POST   /api/tests/:id/start            # Start test attempt
POST   /api/tests/:id/submit           # Submit answers
GET    /api/tests/:id/results          # Get results
POST   /api/tests/:id/review           # Review test
```

**Analytics:**
```
GET    /api/analytics/student/:id      # Student performance
GET    /api/analytics/test/:id         # Test analytics
GET    /api/analytics/batch/:id        # Batch analytics
GET    /api/analytics/weak-topics      # Weak topics
GET    /api/analytics/trends           # Performance trends
GET    /api/analytics/comparison       # Peer comparison
```

**AI/ML:**
```
POST   /api/ai/evaluate                # AI evaluation
GET    /api/ai/predict-rank            # Rank prediction
GET    /api/ai/weak-concepts           # Concept gap detection
POST   /api/ai/adaptive-test           # Adaptive test generation
```

---

## 7. Security Architecture

### 7.1 Authentication Flow

```
User Login (coaching1.shikshasetu.com)
    ↓
API Gateway extracts subdomain → coaching1
    ↓
Fetch tenant_id from subdomain
    ↓
Authenticate user (email + password)
    ↓
Generate JWT with:
  - userId
  - tenantId (encrypted)
  - role
  - permissions
    ↓
Return JWT + Refresh Token
    ↓
Subsequent requests include JWT
    ↓
Middleware validates JWT & extracts tenantId
    ↓
All DB queries filtered by tenantId
```

### 7.2 Authorization

**Role-Based Access Control (RBAC):**

| Role    | Can Create Test | Can View Analytics | Can Manage Content | Can Manage Users |
|---------|-----------------|--------------------|--------------------|------------------|
| Student | ❌               | Self Only          | ❌                  | ❌                |
| Teacher | ✅               | Batch Level        | ✅                  | ❌                |
| Admin   | ✅               | Tenant Level       | ✅                  | ✅                |

**Permission Matrix:**
```typescript
const permissions = {
  student: ['view_own_tests', 'submit_test', 'view_own_results'],
  teacher: ['create_test', 'evaluate', 'view_batch_analytics', 'manage_content'],
  admin: ['manage_users', 'manage_subscription', 'view_all_analytics', '*']
};
```

### 7.3 Security Measures

1. **Data Encryption:**
   - TLS 1.3 for data in transit
   - AES-256 for data at rest
   - Field-level encryption for PII

2. **Input Validation:**
   - Request validation with Joi/Zod
   - SQL injection prevention
   - XSS protection

3. **Rate Limiting:**
   - Per tenant: 1000 req/min
   - Per user: 100 req/min
   - Per IP: 500 req/min

4. **Audit Logging:**
   - All tenant operations logged
   - User activity tracking
   - Data access logging

5. **Compliance:**
   - GDPR-ready (data portability, right to deletion)
   - Student data auto-purge after 3 months
   - Anonymized analytics retention

---

## 8. Scalability Strategy

### 8.1 Horizontal Scaling

**Service-Level Scaling:**
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-engine-service
spec:
  replicas: 3  # Auto-scale based on CPU/memory
  template:
    spec:
      containers:
      - name: test-engine
        image: shikshasetu/test-engine:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

**Database Scaling:**
- PostgreSQL: Read replicas for analytics queries
- MongoDB: Sharding by tenant_id
- Redis: Redis Cluster for high availability

### 8.2 Caching Strategy

**Multi-Level Caching:**

1. **CDN Level (CloudFront):**
   - Static assets
   - Public content
   - TTL: 24 hours

2. **Application Level (Redis):**
   - Tenant configuration (TTL: 1 hour)
   - Question cache (TTL: 30 mins)
   - User sessions
   - Leaderboards (real-time)

3. **Database Level:**
   - Materialized views for analytics
   - Query result caching

**Cache Invalidation:**
```typescript
// Event-driven cache invalidation
eventBus.on('question.updated', async (questionId, tenantId) => {
  await redis.del(`question:${questionId}`);
  await redis.del(`tenant:${tenantId}:questions`);
});
```

### 8.3 Performance Optimization

1. **Database:**
   - Proper indexing (tenant_id, user_id, test_id)
   - Connection pooling
   - Query optimization
   - Partitioning by tenant_id (future)

2. **API:**
   - GraphQL for dashboard (reduce over-fetching)
   - Pagination for all list endpoints
   - Field selection (sparse fieldsets)
   - Compression (gzip)

3. **Frontend:**
   - Code splitting
   - Lazy loading
   - Image optimization (Next.js Image)
   - Service Workers for offline support

---

## 9. Monitoring & Observability

### 9.1 Metrics to Track

**Business Metrics:**
- Active tenants
- Total students per tenant
- Tests created/attempted per day
- Revenue per tenant
- Churn rate

**Technical Metrics:**
- API response times (P50, P95, P99)
- Error rates per service
- Database query performance
- Cache hit rates
- Queue depths

**Tenant Metrics:**
- Per-tenant API usage
- Storage consumption
- Test attempt volume
- Peak concurrent users

### 9.2 Alerting

**Critical Alerts:**
- Service down (>5 min)
- Error rate >5%
- Response time >2s (P95)
- Database connection pool exhausted
- Queue backlog >1000 messages

**Warning Alerts:**
- CPU >80%
- Memory >85%
- Disk >90%
- Cache hit rate <70%

---

## 10. AI/ML Architecture

### 10.1 ML Pipeline

```
Data Collection → Feature Engineering → Model Training → Model Serving → Monitoring
      ↓                   ↓                   ↓                ↓              ↓
Test Attempts      Topic vectors      Rank prediction    FastAPI        Performance
Question logs      Difficulty scores  Concept clustering  Endpoints      Drift detection
                   Time features      Weak topic detection
```

### 10.2 AI Services Architecture

**Async Processing:**
```
Test Submitted
    ↓
Event published to RabbitMQ
    ↓
AI Service consumes event
    ↓
Perform analysis:
  - Weak topic detection
  - Concept gap clustering
  - Rank prediction
    ↓
Store results in PostgreSQL
    ↓
Trigger notification
```

**LLM Integration (Phase 2):**
```
Subjective Answer Upload
    ↓
Queue for AI evaluation
    ↓
LangChain orchestration:
  - Answer extraction
  - Rubric comparison
  - Scoring
  - Feedback generation
    ↓
Store evaluation + feedback
    ↓
Notify teacher for final review
```

---

## 11. Deployment Architecture

### 11.1 Infrastructure as Code

**Terraform:**
```hcl
# VPC, subnets, security groups
# RDS (PostgreSQL)
# DocumentDB (MongoDB)
# ElastiCache (Redis)
# EKS (Kubernetes)
# S3 buckets
# CloudFront distribution
```

### 11.2 Environment Strategy

**Environments:**
1. **Local:** Docker Compose
2. **Development:** Single K8s cluster (dev namespace)
3. **Staging:** Production-like K8s cluster
4. **Production:** Multi-AZ K8s cluster with auto-scaling

### 11.3 CI/CD Pipeline

```
Code Push to GitHub
    ↓
GitHub Actions triggered
    ↓
Run tests (unit, integration)
    ↓
Build Docker images
    ↓
Push to Container Registry
    ↓
Update Helm values
    ↓
ArgoCD detects changes
    ↓
Deploy to K8s cluster
    ↓
Run smoke tests
    ↓
Notify team
```

---

## 12. Phase-wise Implementation

### Phase 1: Foundation (Months 1-3)

**Services:**
1. Tenant Management Service
2. Authentication Service
3. Content Management Service (Basic)
4. Test Engine (Objective MCQ only)

**Features:**
- Tenant onboarding
- Subdomain provisioning
- User management (Student/Teacher/Admin)
- Question bank with tagging
- MCQ test creation
- Test attempt and auto-evaluation
- Basic analytics (score, rank, accuracy)

**Infrastructure:**
- Single-region deployment
- Basic monitoring
- PostgreSQL + MongoDB + Redis

### Phase 2: Analytics & Differentiation (Months 4-6)

**Services:**
1. Analytics Engine
2. AI/ML Service (Basic)
3. Notification Service

**Features:**
- Weak topic detection
- Concept gap clustering
- Performance trends
- Batch analytics
- Teacher dashboard
- Email notifications
- Rank prediction (AI)

### Phase 3: Subjective Tests & Advanced AI (Months 7-12)

**Services:**
1. Evaluation Service (Manual + AI-assisted)
2. Advanced AI/ML

**Features:**
- Subjective test support
- Answer upload (text/PDF/image)
- Manual evaluation interface
- Rubric-based scoring
- LLM-assisted evaluation (beta)
- Adaptive testing (early version)

### Phase 4: Scale & Expansion (Year 2)

**Features:**
- Multi-region deployment
- Advanced caching
- Personalized study paths
- AI doubt assistant
- Mobile apps (iOS/Android)
- Offline mode
- Advanced analytics
- Marketplace for content creators

---

## 13. Cost Estimation

### Infrastructure (Monthly - Year 1)

| Component          | Service           | Estimated Cost |
|--------------------|-------------------|----------------|
| Compute            | EKS (5 nodes)     | $300           |
| Database           | RDS PostgreSQL    | $200           |
| Database           | DocumentDB        | $150           |
| Cache              | ElastiCache       | $100           |
| Storage            | S3                | $50            |
| CDN                | CloudFront        | $100           |
| Monitoring         | Prometheus/Grafana| $50            |
| Misc               | DNS, LB, etc.     | $50            |
| **Total**          |                   | **$1,000/mo**  |

**Per Tenant Costs:** ~$10-20/mo (at 50+ tenants)

---

## 14. Risk Mitigation

### Technical Risks

| Risk                          | Mitigation Strategy                                      |
|-------------------------------|----------------------------------------------------------|
| Database bottleneck           | Read replicas, caching, query optimization               |
| Tenant data leak              | RLS, app-level filtering, audit logs                     |
| Service downtime              | K8s auto-restart, health checks, multi-replica           |
| AI over-promise               | Phased rollout, clear disclaimers, teacher review layer  |
| Scalability issues            | Horizontal scaling, load testing, performance monitoring |

### Business Risks

| Risk                          | Mitigation Strategy                                      |
|-------------------------------|----------------------------------------------------------|
| Low adoption                  | Pilot program, demo-driven sales, ROI focus              |
| Coaching resistance           | Change management, training, proven analytics value      |
| Feature creep                 | Strict MVP scope, phased roadmap                         |
| Engineering bandwidth         | Focused sprints, no side projects, hire as needed        |

---

## 15. Success Metrics

**Year 1 Targets:**
- 20+ active coaching tenants
- 5,000+ total students
- 95%+ uptime
- <500ms API response time (P95)
- 10,000+ tests attempted/month

**Technical KPIs:**
- Code coverage >80%
- Security audit passed
- Zero critical security incidents
- <5 production bugs per month

---

## Conclusion

This HLD provides a comprehensive blueprint for building shikshasetu as a scalable, secure, multi-tenant SaaS platform. The architecture emphasizes:

1. **Strong tenant isolation** for data security
2. **Microservices** for independent scaling
3. **AI-first approach** for differentiation
4. **Cloud-native** for reliability and scalability
5. **Phased implementation** to manage complexity

**Next Steps:**
1. Review and approve HLD
2. Create detailed service design for Phase 1
3. Set up infrastructure
4. Begin development sprint

---

**Document Control:**
- **Author:** Architecture Team
- **Reviewers:** Engineering Lead, Product Lead, CTO
- **Approval Date:** TBD
- **Next Review:** After Phase 1 completion
