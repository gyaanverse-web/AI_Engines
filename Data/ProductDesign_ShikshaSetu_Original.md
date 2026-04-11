# ShikhaSetu - MVP Product Design Document
## Service 1: Tenant Management + Test Engine (Objective MCQs)

**Version:** 1.0
**Date:** February 2026
**Phase:** MVP / Phase 1

---

## 1. Executive Summary

**Recommendation: Start with Tenant Management + Test Engine (Objective MCQs) as your MVP**

### Why This Combination?

1. **Tenant Management** is foundational - everything else depends on it
2. **Test Engine** is your core value proposition - the primary reason coachings will pay
3. Together they form a **complete MVP** that can be sold and generates revenue
4. Validates your business model before investing in complex AI features

### MVP Scope (3 months)

**What's Included:**
✅ Tenant onboarding and provisioning
✅ Subdomain setup (coaching1.shikhasetu.com)
✅ User management (Student/Teacher/Admin roles)
✅ Question bank with tagging
✅ MCQ test creation and delivery
✅ Auto-evaluation and basic results
✅ Simple analytics (score, rank, accuracy)

**What's Deferred:**
❌ Advanced AI features
❌ Subjective tests
❌ Complex analytics dashboards
❌ LLM-based evaluation

---

## 2. User Stories & Use Cases

### 2.1 Primary User Flows

**Flow 1: Coaching Admin Onboarding**
```
As a coaching institute admin,
I want to sign up for ShikhaSetu,
So that I can digitize my test infrastructure.

Steps:
1. Visit shikhasetu.com
2. Click "Start Free Trial"
3. Fill onboarding form (coaching name, subdomain, contact)
4. Verify email
5. Set up admin account
6. Get access to coaching1.shikhasetu.com
7. See onboarding wizard
```

**Flow 2: Teacher Creates a Test**
```
As a teacher,
I want to create a mock test,
So that students can practice before exams.

Steps:
1. Login to coaching1.shikhasetu.com
2. Navigate to "Tests" → "Create New Test"
3. Enter test details (name, duration, subjects)
4. Add questions from question bank
5. Configure sections and marking scheme
6. Set test schedule (start/end time)
7. Publish test
8. Share test link with students
```

**Flow 3: Student Takes a Test**
```
As a student,
I want to take a mock test,
So that I can assess my preparation.

Steps:
1. Login to coaching1.shikhasetu.com
2. See available tests on dashboard
3. Click "Start Test"
4. Read instructions
5. Begin test (timer starts)
6. Answer questions section-wise
7. Mark questions for review
8. Submit test
9. View immediate results
10. See answers and explanations
```

**Flow 4: Teacher Reviews Results**
```
As a teacher,
I want to see test results,
So that I can identify weak students.

Steps:
1. Login to teacher dashboard
2. Navigate to "Tests" → Select test
3. View results summary:
   - Average score
   - Rank list
   - Question-wise accuracy
4. Download result PDF
5. Identify weak performers
```

---

## 3. Detailed Feature Specifications

## 3.1 Tenant Management Module

### 3.1.1 Tenant Onboarding

**API Endpoint:** `POST /api/admin/tenants`

**Request:**
```json
{
  "coachingName": "Agrawal Classes",
  "subdomain": "agrawal",
  "adminEmail": "admin@agrawalclasses.com",
  "adminName": "Rajesh Agrawal",
  "phone": "+91-9876543210",
  "address": {
    "city": "Kota",
    "state": "Rajasthan"
  },
  "planType": "trial"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "tenantId": "uuid-123",
    "subdomain": "agrawal",
    "url": "https://agrawal.shikhasetu.com",
    "status": "active",
    "trialEndsAt": "2026-03-15",
    "adminAccount": {
      "email": "admin@agrawalclasses.com",
      "temporaryPassword": "generated-password"
    }
  }
}
```

**Business Logic:**
1. Validate subdomain uniqueness
2. Check subdomain format (alphanumeric, no spaces)
3. Create tenant record in PostgreSQL
4. Create admin user account
5. Send welcome email with credentials
6. Provision subdomain routing
7. Set trial period (14 days)
8. Initialize tenant settings with defaults

**Database Schema:**
```sql
CREATE TABLE tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  subdomain VARCHAR(50) UNIQUE NOT NULL,
  status VARCHAR(20) DEFAULT 'trial', -- trial, active, suspended
  plan_type VARCHAR(20) DEFAULT 'basic', -- basic, pro, enterprise
  student_limit INTEGER DEFAULT 100,
  current_students INTEGER DEFAULT 0,
  trial_ends_at TIMESTAMP,
  subscription_starts_at TIMESTAMP,
  subscription_ends_at TIMESTAMP,
  settings JSONB DEFAULT '{
    "allow_subjective": false,
    "enable_ai": false,
    "branding": {
      "logo": null,
      "primaryColor": "#3b82f6"
    }
  }',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tenants_subdomain ON tenants(subdomain);
CREATE INDEX idx_tenants_status ON tenants(status);
```

### 3.1.2 Subdomain Provisioning

**Technical Implementation:**

**Option 1: Wildcard DNS (Recommended for MVP)**
```
*.shikhasetu.com → Load Balancer IP
```

**Next.js Middleware:**
```typescript
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';

export async function middleware(request: NextRequest) {
  const hostname = request.headers.get('host') || '';
  const subdomain = extractSubdomain(hostname);

  if (!subdomain || subdomain === 'www') {
    // Main landing page
    return NextResponse.next();
  }

  // Fetch tenant from cache or DB
  const tenant = await getTenant(subdomain);

  if (!tenant) {
    return new NextResponse('Coaching not found', { status: 404 });
  }

  if (tenant.status !== 'active' && tenant.status !== 'trial') {
    return new NextResponse('Subscription expired', { status: 403 });
  }

  // Inject tenant context in request
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('x-tenant-id', tenant.id);
  requestHeaders.set('x-tenant-name', tenant.name);

  return NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
}

function extractSubdomain(hostname: string): string | null {
  const parts = hostname.split('.');
  if (parts.length >= 3) {
    return parts[0];
  }
  return null;
}
```

### 3.1.3 User Management

**Roles:**
1. **Admin** - Full access to tenant
2. **Teacher** - Create tests, view batch analytics
3. **Student** - Take tests, view own results

**API Endpoints:**
```
POST   /api/users                  # Create user (Admin only)
GET    /api/users                  # List users (Admin/Teacher)
GET    /api/users/:id              # Get user details
PUT    /api/users/:id              # Update user
DELETE /api/users/:id              # Deactivate user
POST   /api/users/bulk-import      # Bulk import via CSV
```

**User Schema:**
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  email VARCHAR(255) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL, -- admin, teacher, student
  profile JSONB DEFAULT '{
    "name": "",
    "phone": "",
    "class": "",
    "stream": "",
    "batch": "",
    "avatar": null
  }',
  status VARCHAR(20) DEFAULT 'active', -- active, inactive, suspended
  last_login TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT unique_tenant_email UNIQUE(tenant_id, email)
);

CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- Row-level security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
```

**Bulk Import Feature:**
```
Teacher uploads CSV:
student_name, email, class, stream, batch
Rohan Sharma, rohan@example.com, 12, Science, JEE-A
Priya Gupta, priya@example.com, 11, Science, NEET-B

System:
1. Validate CSV format
2. Check for duplicates
3. Generate random passwords
4. Create user accounts
5. Send welcome emails
6. Return import summary
```

---

## 3.2 Test Engine Module (Objective MCQs)

### 3.2.1 Question Bank

**Question Schema:**
```sql
CREATE TABLE questions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id), -- NULL for ShikhaSetu default content
  type VARCHAR(20) DEFAULT 'MCQ',
  subject VARCHAR(100) NOT NULL,
  chapter VARCHAR(255) NOT NULL,
  topic VARCHAR(255) NOT NULL,
  subtopic VARCHAR(255),
  difficulty VARCHAR(20) NOT NULL, -- easy, medium, hard
  exam_type VARCHAR(50), -- JEE, NEET, Board
  question_text TEXT NOT NULL,
  options JSONB NOT NULL, -- ["Option A", "Option B", "Option C", "Option D"]
  correct_answer VARCHAR(10) NOT NULL, -- "A", "B", "C", "D"
  explanation TEXT,
  marks DECIMAL(5,2) DEFAULT 4.0,
  negative_marks DECIMAL(5,2) DEFAULT 1.0,
  tags TEXT[],
  metadata JSONB DEFAULT '{
    "averageTime": 120,
    "successRate": 0,
    "attemptCount": 0
  }',
  is_active BOOLEAN DEFAULT true,
  publish_status VARCHAR(20) DEFAULT 'draft', -- draft, review, published
  created_by UUID REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_questions_tenant ON questions(tenant_id);
CREATE INDEX idx_questions_subject ON questions(subject);
CREATE INDEX idx_questions_difficulty ON questions(difficulty);
CREATE INDEX idx_questions_tags ON questions USING GIN(tags);
```

**Question CRUD APIs:**

**Create Question:**
```typescript
POST /api/content/questions

Request:
{
  "subject": "Physics",
  "chapter": "Mechanics",
  "topic": "Newton's Laws",
  "difficulty": "medium",
  "examType": "JEE",
  "questionText": "A block of mass 2 kg is placed on a frictionless surface...",
  "options": [
    "10 N",
    "20 N",
    "30 N",
    "40 N"
  ],
  "correctAnswer": "B",
  "explanation": "Using F = ma, we get...",
  "marks": 4,
  "negativeMarks": 1,
  "tags": ["mechanics", "force", "acceleration"]
}

Response:
{
  "success": true,
  "data": {
    "questionId": "uuid-456",
    "publishStatus": "draft"
  }
}
```

**Search Questions:**
```typescript
GET /api/content/questions/search?subject=Physics&difficulty=medium&tags=mechanics&page=1&limit=20

Response:
{
  "success": true,
  "data": {
    "questions": [...],
    "pagination": {
      "total": 150,
      "page": 1,
      "pages": 8
    }
  }
}
```

### 3.2.2 Test Creation

**Test Schema:**
```sql
CREATE TABLE tests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  type VARCHAR(20) DEFAULT 'mock', -- mock, section, practice
  duration INTEGER NOT NULL, -- minutes
  total_marks DECIMAL(10,2),
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  is_published BOOLEAN DEFAULT false,
  settings JSONB DEFAULT '{
    "shuffleQuestions": false,
    "shuffleOptions": true,
    "showResults": "immediate",
    "allowReview": true,
    "calculatorAllowed": false,
    "negativeMarking": true
  }',
  target_classes TEXT[], -- ["11", "12"]
  target_batches TEXT[], -- ["JEE-A", "NEET-B"]
  created_by UUID REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE test_sections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  test_id UUID NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
  section_number INTEGER NOT NULL,
  name VARCHAR(255) NOT NULL, -- "Physics", "Chemistry", "Maths"
  time_limit INTEGER, -- minutes (optional section time)
  instructions TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE test_questions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  test_id UUID NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
  section_id UUID REFERENCES test_sections(id) ON DELETE CASCADE,
  question_id UUID NOT NULL REFERENCES questions(id),
  question_number INTEGER NOT NULL,
  marks DECIMAL(5,2),
  negative_marks DECIMAL(5,2),
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(test_id, question_number)
);

CREATE INDEX idx_tests_tenant ON tests(tenant_id);
CREATE INDEX idx_tests_published ON tests(is_published);
CREATE INDEX idx_test_questions_test ON test_questions(test_id);
```

**Create Test API:**
```typescript
POST /api/tests

Request:
{
  "title": "JEE Advanced Mock Test #1",
  "description": "Full syllabus mock test for JEE Advanced",
  "type": "mock",
  "duration": 180,
  "startTime": "2026-02-20T09:00:00Z",
  "endTime": "2026-02-20T21:00:00Z",
  "targetClasses": ["11", "12"],
  "targetBatches": ["JEE-A"],
  "settings": {
    "shuffleQuestions": true,
    "showResults": "immediate",
    "allowReview": true,
    "negativeMarking": true
  },
  "sections": [
    {
      "name": "Physics",
      "timeLimit": 60,
      "instructions": "Answer all questions",
      "questions": [
        {
          "questionId": "uuid-q1",
          "marks": 4,
          "negativeMarks": 1
        },
        // ... 29 more questions
      ]
    },
    {
      "name": "Chemistry",
      "timeLimit": 60,
      "questions": [...]
    },
    {
      "name": "Mathematics",
      "timeLimit": 60,
      "questions": [...]
    }
  ]
}

Response:
{
  "success": true,
  "data": {
    "testId": "uuid-789",
    "totalQuestions": 90,
    "totalMarks": 360,
    "status": "draft"
  }
}
```

**Publish Test:**
```typescript
POST /api/tests/:testId/publish

Business Logic:
1. Validate all questions exist
2. Validate sections are complete
3. Calculate total marks
4. Set is_published = true
5. Send notifications to target students
```

### 3.2.3 Test Delivery

**Test Attempt Schema:**
```sql
CREATE TABLE test_attempts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  test_id UUID NOT NULL REFERENCES tests(id),
  student_id UUID NOT NULL REFERENCES users(id),
  status VARCHAR(20) DEFAULT 'not_started', -- not_started, ongoing, submitted, evaluated
  current_section INTEGER DEFAULT 1,
  answers JSONB DEFAULT '{}', -- { "q1": "A", "q2": "C", ... }
  marked_for_review UUID[], -- [question_ids]
  visit_counts JSONB DEFAULT '{}', -- { "q1": 3, "q2": 1 }
  time_spent JSONB DEFAULT '{}', -- { "q1": 120, "q2": 95 } in seconds
  started_at TIMESTAMP,
  submitted_at TIMESTAMP,
  auto_submitted BOOLEAN DEFAULT false,
  created_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT unique_test_student UNIQUE(test_id, student_id)
);

CREATE INDEX idx_attempts_tenant ON test_attempts(tenant_id);
CREATE INDEX idx_attempts_test ON test_attempts(test_id);
CREATE INDEX idx_attempts_student ON test_attempts(student_id);
CREATE INDEX idx_attempts_status ON test_attempts(status);
```

**Start Test API:**
```typescript
POST /api/tests/:testId/start

Response:
{
  "success": true,
  "data": {
    "attemptId": "uuid-attempt-1",
    "test": {
      "title": "JEE Advanced Mock Test #1",
      "duration": 180,
      "totalQuestions": 90,
      "totalMarks": 360,
      "startedAt": "2026-02-20T10:30:00Z",
      "endsAt": "2026-02-20T13:30:00Z"
    },
    "sections": [
      {
        "sectionId": "uuid-s1",
        "name": "Physics",
        "timeLimit": 60,
        "questions": [
          {
            "questionId": "uuid-q1",
            "questionNumber": 1,
            "questionText": "...",
            "options": ["A", "B", "C", "D"],
            "marks": 4,
            "negativeMarks": 1
          },
          // ... more questions
        ]
      },
      // ... more sections
    ]
  }
}
```

**Submit Answer API:**
```typescript
PUT /api/tests/:testId/answer

Request:
{
  "attemptId": "uuid-attempt-1",
  "questionId": "uuid-q1",
  "answer": "B",
  "markedForReview": false,
  "timeSpent": 120
}

Business Logic:
1. Validate attempt is ongoing
2. Update answers JSONB
3. Update time_spent
4. Update visit_counts
5. Auto-save every 30 seconds on client side
```

**Submit Test API:**
```typescript
POST /api/tests/:testId/submit

Request:
{
  "attemptId": "uuid-attempt-1"
}

Business Logic:
1. Validate attempt exists
2. Set status = 'submitted'
3. Set submitted_at = NOW()
4. Trigger auto-evaluation (async job)
5. Return immediate confirmation

Response:
{
  "success": true,
  "message": "Test submitted successfully. Results will be available shortly."
}
```

### 3.2.4 Auto-Evaluation

**Evaluation Schema:**
```sql
CREATE TABLE test_evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  attempt_id UUID UNIQUE NOT NULL REFERENCES test_attempts(id),
  total_questions INTEGER,
  attempted INTEGER,
  correct INTEGER,
  incorrect INTEGER,
  unattempted INTEGER,
  total_marks DECIMAL(10,2),
  marks_obtained DECIMAL(10,2),
  percentage DECIMAL(5,2),
  rank INTEGER,
  percentile DECIMAL(5,2),
  section_wise JSONB, -- {"Physics": {"marks": 120, "obtained": 80, ...}}
  topic_wise JSONB, -- {"Mechanics": {"attempted": 10, "correct": 7}}
  difficulty_wise JSONB, -- {"easy": {...}, "medium": {...}, "hard": {...}}
  evaluated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_evaluations_attempt ON test_evaluations(attempt_id);
```

**Auto-Evaluation Process:**

```typescript
// Async job triggered after test submission
async function evaluateTest(attemptId: string) {
  const attempt = await getTestAttempt(attemptId);
  const test = await getTest(attempt.testId);
  const questions = await getTestQuestions(test.id);

  const evaluation = {
    totalQuestions: questions.length,
    attempted: 0,
    correct: 0,
    incorrect: 0,
    unattempted: 0,
    totalMarks: 0,
    marksObtained: 0,
    sectionWise: {},
    topicWise: {},
    difficultyWise: {}
  };

  // Evaluate each question
  for (const question of questions) {
    const userAnswer = attempt.answers[question.id];
    const correctAnswer = question.correctAnswer;

    evaluation.totalMarks += question.marks;

    if (!userAnswer) {
      evaluation.unattempted++;
      continue;
    }

    evaluation.attempted++;

    if (userAnswer === correctAnswer) {
      evaluation.correct++;
      evaluation.marksObtained += question.marks;

      // Update question metadata
      await updateQuestionStats(question.id, true);
    } else {
      evaluation.incorrect++;
      evaluation.marksObtained -= question.negativeMarks;

      await updateQuestionStats(question.id, false);
    }

    // Aggregate by section
    updateSectionWise(evaluation, question, userAnswer === correctAnswer);

    // Aggregate by topic
    updateTopicWise(evaluation, question, userAnswer === correctAnswer);

    // Aggregate by difficulty
    updateDifficultyWise(evaluation, question, userAnswer === correctAnswer);
  }

  // Calculate percentage
  evaluation.percentage = (evaluation.marksObtained / evaluation.totalMarks) * 100;

  // Calculate rank and percentile
  const { rank, percentile } = await calculateRank(test.id, evaluation.marksObtained);
  evaluation.rank = rank;
  evaluation.percentile = percentile;

  // Save evaluation
  await saveEvaluation(attemptId, evaluation);

  // Update test attempt status
  await updateAttemptStatus(attemptId, 'evaluated');

  // Send notification to student
  await notifyStudent(attempt.studentId, {
    message: 'Your test has been evaluated',
    score: evaluation.marksObtained,
    rank: rank
  });
}
```

**Calculate Rank:**
```typescript
async function calculateRank(testId: string, marksObtained: number) {
  // Get all evaluated attempts for this test
  const attempts = await db.query(`
    SELECT e.marks_obtained
    FROM test_evaluations e
    JOIN test_attempts a ON a.id = e.attempt_id
    WHERE a.test_id = $1 AND a.status = 'evaluated'
    ORDER BY e.marks_obtained DESC
  `, [testId]);

  let rank = 1;
  for (const attempt of attempts) {
    if (attempt.marks_obtained > marksObtained) {
      rank++;
    } else {
      break;
    }
  }

  const totalAttempts = attempts.length;
  const percentile = ((totalAttempts - rank + 1) / totalAttempts) * 100;

  return { rank, percentile: Math.round(percentile * 100) / 100 };
}
```

### 3.2.5 Results & Analytics

**Get Results API:**
```typescript
GET /api/tests/:testId/results/:attemptId

Response:
{
  "success": true,
  "data": {
    "test": {
      "title": "JEE Advanced Mock Test #1",
      "totalMarks": 360,
      "duration": 180
    },
    "performance": {
      "marksObtained": 240,
      "totalMarks": 360,
      "percentage": 66.67,
      "rank": 5,
      "percentile": 87.5,
      "attempted": 85,
      "correct": 60,
      "incorrect": 25,
      "unattempted": 5
    },
    "sectionWise": {
      "Physics": {
        "totalMarks": 120,
        "obtained": 80,
        "accuracy": 66.67,
        "attempted": 28,
        "correct": 20
      },
      "Chemistry": { ... },
      "Mathematics": { ... }
    },
    "topicWise": {
      "Mechanics": {
        "attempted": 10,
        "correct": 7,
        "accuracy": 70
      },
      "Thermodynamics": { ... }
    },
    "difficultyWise": {
      "easy": {
        "attempted": 30,
        "correct": 28,
        "accuracy": 93.33
      },
      "medium": { ... },
      "hard": { ... }
    },
    "weakTopics": [
      "Thermodynamics",
      "Organic Chemistry",
      "Calculus"
    ]
  }
}
```

**Answer Review API:**
```typescript
GET /api/tests/:testId/review/:attemptId

Response:
{
  "success": true,
  "data": {
    "questions": [
      {
        "questionNumber": 1,
        "questionText": "...",
        "options": ["A", "B", "C", "D"],
        "correctAnswer": "B",
        "userAnswer": "B",
        "isCorrect": true,
        "marksAwarded": 4,
        "explanation": "Using F = ma...",
        "timeSpent": 120
      },
      {
        "questionNumber": 2,
        "questionText": "...",
        "options": ["A", "B", "C", "D"],
        "correctAnswer": "C",
        "userAnswer": "A",
        "isCorrect": false,
        "marksAwarded": -1,
        "explanation": "...",
        "timeSpent": 95
      },
      // ... more questions
    ]
  }
}
```

---

## 4. Frontend Design

### 4.1 Page Structure

**Landing Page (shikhasetu.com):**
- Hero section with value proposition
- Features overview
- Pricing plans
- "Start Free Trial" CTA

**Coaching Admin Portal (coaching.shikhasetu.com/admin):**
- Dashboard (stats, recent tests, student count)
- User Management
- Subscription & Billing
- Settings

**Teacher Dashboard (coaching.shikhasetu.com/teacher):**
- Test Management
  - Create Test
  - Manage Tests
  - View Results
- Question Bank
  - Browse Questions
  - Add Questions
  - Import Questions
- Analytics
  - Test-wise analytics
  - Batch analytics

**Student Dashboard (coaching.shikhasetu.com/student):**
- Available Tests
- Test History
- Performance Summary
- Weak Topics

**Test Taking Interface:**
- Question palette (navigator)
- Timer
- Section tabs
- Mark for review
- Submit confirmation

**Results Page:**
- Score summary
- Rank and percentile
- Section-wise breakdown
- Topic-wise analysis
- Answer review

### 4.2 Key UI Components

**Test Creation Form:**
```typescript
// TestCreationForm.tsx
interface TestCreationFormProps {
  onSubmit: (testData: TestData) => void;
}

const TestCreationForm = ({ onSubmit }: TestCreationFormProps) => {
  // Form state management
  const [testInfo, setTestInfo] = useState({
    title: '',
    duration: 180,
    startTime: null,
    endTime: null
  });

  const [sections, setSections] = useState([
    { name: 'Physics', questions: [] },
    { name: 'Chemistry', questions: [] },
    { name: 'Mathematics', questions: [] }
  ]);

  // Question selection modal
  const [showQuestionBank, setShowQuestionBank] = useState(false);
  const [currentSection, setCurrentSection] = useState(null);

  // Render form fields...
};
```

**Question Palette (Test Interface):**
```typescript
// QuestionPalette.tsx
const QuestionPalette = ({ questions, currentQuestion, onQuestionClick }) => {
  const getQuestionStatus = (q) => {
    if (q.answered) return 'answered';
    if (q.markedForReview) return 'review';
    if (q.visited) return 'not-answered';
    return 'not-visited';
  };

  return (
    <div className="question-palette">
      <div className="legend">
        <span className="answered">Answered</span>
        <span className="not-answered">Not Answered</span>
        <span className="review">Marked for Review</span>
        <span className="not-visited">Not Visited</span>
      </div>
      <div className="question-grid">
        {questions.map((q, idx) => (
          <button
            key={q.id}
            className={`q-btn ${getQuestionStatus(q)} ${currentQuestion === idx ? 'active' : ''}`}
            onClick={() => onQuestionClick(idx)}
          >
            {idx + 1}
          </button>
        ))}
      </div>
    </div>
  );
};
```

---

## 5. Technical Implementation Details

### 5.1 Tech Stack (MVP)

**Backend:**
- **Framework:** NestJS with TypeScript
- **Database:** PostgreSQL 15
- **Cache:** Redis 7
- **Queue:** Bull (Redis-based)
- **Storage:** AWS S3 (for future subjective uploads)

**Frontend:**
- **Framework:** Next.js 14 with TypeScript
- **UI:** Tailwind CSS + shadcn/ui
- **State:** Zustand
- **Forms:** React Hook Form + Zod
- **API:** React Query

**Infrastructure:**
- **Hosting:** AWS (EC2 or ECS)
- **Database:** AWS RDS (PostgreSQL)
- **Cache:** AWS ElastiCache (Redis)
- **CDN:** CloudFront
- **DNS:** Route53

### 5.2 Development Timeline (12 weeks)

**Weeks 1-2: Foundation**
- Set up monorepo (Nx or Turborepo)
- Configure PostgreSQL with tenant schema
- Implement authentication (JWT)
- Set up Next.js with subdomain routing
- Basic tenant onboarding flow

**Weeks 3-4: User Management**
- User CRUD APIs
- Role-based access control
- Bulk import feature
- Admin dashboard UI
- User management UI

**Weeks 5-7: Question Bank**
- Question CRUD APIs
- Question search and filtering
- Question tagging system
- Question bank UI
- Import from Excel/CSV

**Weeks 8-10: Test Engine**
- Test creation APIs
- Test delivery APIs
- Test attempt tracking
- Auto-evaluation service
- Test creation UI
- Test taking interface

**Weeks 11-12: Results & Polish**
- Results calculation
- Rank computation
- Analytics dashboards
- Results UI
- Performance optimization
- Bug fixes and testing

### 5.3 API Rate Limiting

```typescript
// Per tenant rate limiting
const tenantRateLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 1000, // 1000 requests per minute per tenant
  keyGenerator: (req) => req.tenantId,
  message: 'Too many requests from this coaching'
});

// Per user rate limiting
const userRateLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100, // 100 requests per minute per user
  keyGenerator: (req) => req.userId,
  message: 'Too many requests from this user'
});
```

---

## 6. Testing Strategy

### 6.1 Testing Pyramid

**Unit Tests (60%):**
- Service methods
- Utility functions
- Validation logic
- Business rules

**Integration Tests (30%):**
- API endpoints
- Database operations
- Authentication flow
- Evaluation logic

**E2E Tests (10%):**
- Critical user flows:
  - Tenant onboarding
  - Test creation
  - Test taking
  - Results viewing

### 6.2 Test Cases

**Tenant Management:**
- ✅ Create tenant with unique subdomain
- ✅ Reject duplicate subdomain
- ✅ Set trial period correctly
- ✅ Send welcome email
- ✅ Provision subdomain routing

**Test Engine:**
- ✅ Create test with multiple sections
- ✅ Add questions to test
- ✅ Publish test
- ✅ Start test creates attempt
- ✅ Submit answers
- ✅ Auto-save answers
- ✅ Submit test
- ✅ Auto-evaluate correctly
- ✅ Calculate rank accurately
- ✅ Identify weak topics

**Security:**
- ✅ Tenant data isolation (can't access other tenant data)
- ✅ Role-based access (student can't create tests)
- ✅ JWT validation
- ✅ Rate limiting works

---

## 7. Success Metrics (MVP)

**Technical Metrics:**
- API response time < 200ms (P95)
- 99.5% uptime
- Zero critical bugs in production
- Auto-evaluation accuracy 100%

**Business Metrics:**
- 5 coaching institutes onboarded
- 500+ students using platform
- 100+ tests created
- 1000+ test attempts
- 80%+ customer satisfaction

**User Experience:**
- Test interface load time < 2s
- Zero data loss during test
- Smooth test taking experience
- Clear results presentation

---

## 8. Go-to-Market Plan

### 8.1 Pilot Program

**Target:** 3-5 Tier 2 coaching institutes in Kota/Delhi

**Offer:**
- Free 3-month trial
- Onboarding support
- Dedicated account manager
- Feedback sessions

**Success Criteria:**
- 80%+ feature adoption
- 10+ tests created per coaching
- Positive feedback on analytics
- Willingness to pay post-trial

### 8.2 Pricing (Post-MVP)

**Basic Plan:** ₹5,000/month
- Up to 100 students
- Unlimited tests
- Basic analytics
- Email support

**Pro Plan:** ₹10,000/month
- Up to 300 students
- Advanced analytics
- Weak topic detection
- Priority support

**Enterprise Plan:** Custom
- Unlimited students
- Custom branding
- API access
- Dedicated support

---

## 9. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Slow coaching adoption | High | Pilot program, proven ROI demo |
| Technical bugs in test engine | Critical | Extensive testing, gradual rollout |
| Performance issues with scale | High | Load testing, caching, horizontal scaling |
| Data loss during test | Critical | Auto-save every 30s, database backups |
| Competitor launch similar product | Medium | Focus on differentiation (analytics) |

---

## 10. Next Steps After MVP

**Phase 2 (Months 4-6):**
1. Advanced analytics dashboard
2. AI-based weak topic detection
3. Batch comparison features
4. Mobile app (React Native)

**Phase 3 (Months 7-9):**
1. Subjective test support
2. Manual evaluation interface
3. Rubric-based scoring

**Phase 4 (Months 10-12):**
1. LLM-assisted subjective evaluation (beta)
2. Adaptive testing
3. Personalized study recommendations

---

## Conclusion

**This MVP focuses on:**
✅ Core value proposition (test infrastructure)
✅ Revenue generation (subscription model)
✅ Tenant isolation (multi-tenant architecture)
✅ User experience (smooth test taking)
✅ Basic analytics (differentiation starter)

**Deliberately excludes:**
❌ Complex AI features (defer to Phase 2)
❌ Subjective tests (defer to Phase 3)
❌ Advanced dashboards (defer to Phase 2)
❌ Mobile apps (defer to Phase 2)

**Timeline:** 12 weeks to MVP
**Investment:** 2-3 full-stack developers + 1 DevOps
**Go-to-Market:** Pilot with 3-5 coaching institutes

---

**Approval Required From:**
- [ ] Product Lead
- [ ] Engineering Lead
- [ ] Founder/CEO

**Next Document:** Low-Level Design (LLD) for Tenant Management Service
