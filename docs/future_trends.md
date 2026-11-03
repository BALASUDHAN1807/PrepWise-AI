# Future trends (Unit 4: where this could go)

## 1. Video interview analysis
Adding the camera enables feedback on eye contact, posture, pace and facial expression. Technically straightforward with
pose/face models plus a vision-language model, but **risky**: emotion and "confidence" inference from faces is scientifically contested and
can encode bias around disability, culture and skin tone. If added, keep it opt-in, focus on controllable behaviours (looking at the
camera, speaking pace, background/lighting), never output a hidden "hireability" score, and evaluate by subgroup.

## 2. Avatar and voice interviewers
Real-time speech-to-speech models and animated avatars make practice feel like a real call (interruptions, tone, follow-ups). Our
current pipeline - Whisper for STT, optional gTTS for TTS - is turn-based; the next step is streaming audio with low latency and
an avatar layer. Considerations: cost per minute, latency on weak networks, voice consent, and not impersonating real people.

## 3. Multimodal LLMs
Models that read screenshots, whiteboard photos, diagrams and resumes as images could (a) parse visually complex resumes where PDF text
extraction fails, (b) review hand-drawn system-design diagrams, and (c) give feedback on slides or portfolios. They would
replace brittle layout parsing in `resume_service.py` but need the same schema validation and fallbacks.

## 4. Enterprise integration
* **Placement-cell / LMS integration:** SSO (SAML/OIDC), roster import, cohort analytics in the college's existing tools, LTI plug-ins for Moodle/Canvas.
* **Employer side (carefully):** verified skill badges that students choose to share; no automated screening from practice data.
* **APIs and webhooks** for training institutes; **audit logs and data-residency options** for compliance; **on-prem or local-model** deployment for privacy-sensitive institutions.
* **Governance:** per-tenant retention policies, DPDP-aligned consent management, and model/prompt version pinning so scores are reproducible.

## 5. Technical directions
* Agentic tutors that plan multi-week coaching *with human oversight* - deliberately excluded from this project, but a natural research direction once evaluation and safety practices mature.
* Better evaluation: human-rated gold sets, calibration of LLM judges, uncertainty estimates shown to users.
* Personalisation from longitudinal data (spaced repetition of weak topics) with privacy-preserving synthetic data (VAE/GAN-style generators; see `docs/generative_vs_discriminative.md`).
* Smaller on-device models (distilled/quantised) for offline practice on low-end laptops.
* Multilingual speech and feedback reviewed by native speakers (Tamil, Hindi, and more Indian languages).
