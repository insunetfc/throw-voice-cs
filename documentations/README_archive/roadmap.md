# 📌 Voice Bot Improvement Roadmap

Now that the base system is complete (Lambda + Lex + Connect + OpenVoice TTS with fillers and warmup), the next improvements can be grouped into **Near-term polish** and **Long-term enhancements**.

---

## ✅ Near-term (1–4 weeks)
These are small, contained tasks that polish reliability and user experience.

### 🎧 Call Quality
- Ensure **consistent silence trimming/padding** in fillers and TTS parts (avoid clicks, smoother flow).
- Introduce **progressive stall fillers** (1st: “잠시만요…”, 2nd: “금방 확인해드리겠습니다”) to reduce monotony.

### ⚙️ Engineering
- Add a **CloudWatch dashboard** with metrics:
  - Avg latency (filler vs. TTS).
  - Number of barge-ins handled.
  - Warmup success/fail counts.
- Improve **error fallback**: if TTS server fails, use Polly (or Lex built-in voice) for that turn.
- Pre-register filler WAVs as **Connect Prompts** (stable ARN references instead of presigned URLs).

### 🧠 NLU / Logic
- Expand filler library to **20–30 variants per category** for variety.
- Add simple **sentiment detection** (regex or AWS Comprehend) → route to `공감` fillers when frustration is detected.

---

## 🚀 Long-term (1–3 months)
These are bigger features for robustness, intelligence, or differentiation.

### 🎧 Advanced User Experience
- **Emotion-aware TTS**: dynamically pick `happy/sad/angry` presets depending on context.
- Support **multi-voice campaigns** (male/female, formal/casual personas).

### ⚙️ Robustness / Scaling
- Run TTS server **inside VPC with Lambda** for lower latency (avoid NAT cold starts).
- Enable **Provisioned Concurrency** for Lambda (guaranteed warm containers).
- Add **autoscaling** for TTS server (horizontal pods if deployed on EKS/EC2 ASG).

### 🧠 Intelligence
- Replace regex filler routing with a **lightweight classifier** (or LLM prompt) to predict `확인` / `설명` / `공감` / `시간벌기형`.
- Experiment with **dynamic phrasing generation** (template-based or fine-tuned small model) to keep conversations fresh.

### 📊 Analytics & Manager-facing Tools
- Auto-generate a **per-call summary**: number of fillers, avg stall duration, latency stats.
- Export daily/weekly reports as CSV/PDF for managers.

### 🌐 Integrations
- Add **Twilio/WebRTC bridge** for low-cost testing without Connect charges.
- Explore integration with CRM (e.g., Salesforce) → log conversation summaries.

---

## 🎯 Prioritization

**Phase 1 (Immediate polish):**
- Silence trimming, filler variety, CloudWatch dashboard, fallback voice.  

**Phase 2 (Production readiness):**
- VPC integration, provisioned concurrency, Connect Prompt ARNs.  

**Phase 3 (Differentiators):**
- Emotion-aware fillers, dynamic phrasing, analytics reports.  

---

This roadmap ensures the system stays reliable while also adding *wow* features that differentiate it from a simple Connect+Polly setup.
