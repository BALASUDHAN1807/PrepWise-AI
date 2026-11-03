# From idea to MVP: PrepWise AI (Unit 5)

> Market statements below come from general knowledge of the space, not from fresh research or scraping done for this project.
> Verify product features, pricing and market sizes before using them in a pitch.

## 1. Problem statement
Final-year students in India - especially outside metro colleges - prepare for campus placements with scattered resources: coding
sites for DSA, YouTube for HR tips, a generic resume template. Three gaps recur:
1. **No feedback loop.** Practice sites tell you pass/fail on code but not *why your approach is weak*, and say nothing about how you
   *speak* in an HR round.
2. **Preparation is generic.** It ignores the student's resume, target role (service vs product vs startup) and weak areas.
3. **Placement cells cannot scale mentoring.** A few officers serve hundreds of students and lack data on which skills the batch is missing.

## 2. Target users
* **Primary:** B.Tech/BCA/MCA final- and pre-final-year students preparing for campus drives (English, Tamil, Hindi feedback matters).
* **Secondary:** placement cells / training-and-placement officers who want batch-level readiness insight (anonymised).
* **Tertiary:** early-career job seekers, bootcamps and upskilling programmes.

## 3. Vertical landscape: EdTech x HR-Tech
* **EdTech** (test prep, upskilling, coding practice) has large Indian players and many free resources; differentiation is
  *personalised, explainable feedback* rather than content volume.
* **HR-Tech** (ATS, assessments, video-interview screening) serves employers; some vendors score candidates automatically, which is
  where bias and regulation concerns concentrate. PrepWise sits on the *candidate-practice* side and explicitly refuses to be a screening tool.
* **Enablers:** cheap LLM APIs, on-device speech models, multilingual models. **Headwinds:** LLM cost/latency, trust in AI feedback,
  data-protection obligations (DPDP Act), institutions' procurement cycles.

## 4. Competitor analysis (existing AI interview-prep and adjacent tools)
| Category (examples) | What they do well | Gaps PrepWise targets |
|---|---|---|
| Coding practice (LeetCode-style sites, GeeksforGeeks, HackerRank) | Huge problem sets, community, judge infrastructure | Little personalised feedback on approach; no HR/resume integration; (we do *not* copy their content) |
| Peer/expert mock interviews (Pramp-style, interviewing.io-style) | Realistic human conversations | Scheduling and cost; limited availability and languages |
| AI mock-interview apps (Google Interview Warmup-style, various "AI interview coach" startups) | Instant speech/text practice, some analytics | Often generic questions, English-only, little code execution, little transparency of scoring |
| Real-time "interview copilots" | Live prompts during interviews | Ethically contentious (assist during real interviews); not a learning tool - PrepWise is the opposite |
| Resume/ATS checkers (Jobscan-style, Resume Worded-style) | Keyword match, formatting | Single-purpose; no interview practice or roadmap |
| Indian placement-prep platforms (Scaler/InterviewBit-style, PrepInsta-style) | Structured courses, mentors, company-specific prep | Paid/cohort-based; heavy on content, lighter on automated personalised feedback |

**Positioning:** *one place that links resume -> DSA -> HR -> feedback -> roadmap, explains every score, runs code for real, supports
Tamil/Hindi feedback, and gives placement cells privacy-preserving batch insight.*

## 5. Lean Canvas
| Block | Content |
|---|---|
| **Problem** | Generic prep; no actionable feedback on approach or communication; mentors don't scale; students unsure what to learn next |
| **Customer segments** | Final-year engineering students (Tier-2/3 colleges first), placement cells, training institutes |
| **Unique value proposition** | Personalised mock interviews with *explainable* scores, real code execution and a roadmap built from your own gaps - in English, Tamil, Hindi |
| **Solution** | Resume analyzer + JD match; adaptive DSA with sandbox and hints; HR practice with dual evaluation; feedback chain, PDF report, roadmap with a verified course catalog; admin dashboard |
| **Channels** | College placement cells and clubs, student ambassadors, YouTube/Instagram placement-prep content, open-source/GitHub, hackathons |
| **Revenue streams** | Free student tier (limited sessions); paid premium (unlimited sessions, voice analytics); institutional licence per batch/seat; paid add-ons (company-specific packs) |
| **Cost structure** | LLM API + speech compute per session, hosting, content curation and human review, support, compliance |
| **Key metrics** | Weekly active practisers, sessions per student, readiness-score uplift between first and latest session, % students who finish a roadmap week, placement-cell renewal, cost per session |
| **Unfair advantage** | (Aspirational) a reviewed, multilingual, India-specific question/answer corpus and institution relationships; open, auditable scoring |

## 6. MVP scope vs future scope
**In this MVP (built):** auth + consent + deletion; resume parsing/ATS/JD match/rewrites/cover letter; DSA with RAG, hallucination guard,
sandbox, hints and adaptive difficulty; HR practice with text/voice, follow-ups, rubric + classifier; feedback chain, multilingual,
PDF, dashboard; roadmap with catalog-validated courses; Prompt Playground; admin dashboard; experiments and docs.

**Explicitly out of the MVP:** payments, email verification and password reset, mobile app, hardened multi-tenant code sandbox,
human-reviewed datasets, company-specific question packs, video analysis, SSO/LMS integration, proper scalability/ops (Postgres,
queues), accessibility audit, formal bias audit.

**Validation experiments before scaling:** (1) 20-30 students use it for two weeks - does their self-rated confidence and mock score
improve? (2) Have 2-3 placement officers grade 50 answers blind, compare with the system (inter-rater agreement). (3) Measure cost per
session with a real provider. (4) Interview placement cells about willingness to pay and data-sharing constraints.

## 7. Phased roadmap
| Phase | Timeline (indicative) | Goals |
|---|---|---|
| 0 - Course MVP | now | Working end-to-end prototype, documented limits |
| 1 - Pilot | 1-3 months | Human-review datasets; harden sandbox (Judge0/containers); Postgres; email verification; pilot with one college; collect agreement data |
| 2 - Product | 3-9 months | Company-specific packs, better multilingual (native-speaker reviewed), mobile-friendly UI, billing, placement-cell analytics v2, SOC2-style practices |
| 3 - Platform | 9-18 months | Video/multimodal practice, LMS/ATS integrations, API for training institutes, fine-tuned local models for privacy-sensitive customers |

## 8. Risks
Trust in AI scores (mitigate: transparency, human mentors in the loop); LLM cost and vendor dependence (provider-agnostic wrapper,
local models); data-protection compliance (consent, deletion, DPDP work in `docs/responsible_ai.md`); content quality (human review);
a crowded market (differentiate on explainability, language and institution analytics).
