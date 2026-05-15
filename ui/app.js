const sampleMessage =
  "Urgent security notice: your account is under review. Confirm your OTP immediately and transfer the holding amount to avoid permanent suspension.";

const stages = [
  {
    title: "BERT Semantic Parsing",
    detail: "Scanning urgency markers, OTP prompts, and social-engineering cues.",
  },
  {
    title: "BiLSTM Behavioral Analysis",
    detail: "Evaluating sequential transaction context and sender activity patterns.",
  },
  {
    title: "GNN Network Mapping",
    detail: "Tracing relational risk and mule-account cluster probability.",
  },
];

const messageInput = document.getElementById("messageInput");
const accountInput = document.getElementById("accountInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const sampleBtn = document.getElementById("sampleBtn");
const errorBox = document.getElementById("errorBox");
const statusChip = document.getElementById("statusChip");
const riskPercent = document.getElementById("riskPercent");
const networkRisk = document.getElementById("networkRisk");
const muleSignal = document.getElementById("muleSignal");
const recommendations = document.getElementById("recommendations");
const phaseBox = document.getElementById("phaseBox");
const phaseTitle = document.getElementById("phaseTitle");
const phaseDetail = document.getElementById("phaseDetail");
const gaugeValue = document.getElementById("gaugeValue");
const stageCards = Array.from(document.querySelectorAll(".stage-card"));
const counters = Array.from(document.querySelectorAll("[data-counter]"));

const gaugeCircumference = 339.292;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setGauge(probability, fraud) {
  const bounded = Math.max(0, Math.min(1, probability));
  const offset = gaugeCircumference - gaugeCircumference * bounded;
  gaugeValue.style.strokeDashoffset = `${offset}`;
  gaugeValue.classList.toggle("danger-stroke", fraud);
  gaugeValue.classList.toggle("safe-stroke", !fraud);
}

function setStatus(isFraud) {
  statusChip.textContent = isFraud ? "FRAUD DETECTED" : "SECURE";
  statusChip.classList.toggle("danger", isFraud);
  statusChip.classList.toggle("safe", !isFraud);
}

function renderRecommendations(result) {
  const items = [];

  if (result.is_fraud && result.is_mule_account) {
    items.push("Cross-modal consensus indicates an active scam and a mule-linked sender.");
    items.push("Recommended action: freeze transaction flow and escalate to fraud operations.");
  } else if (result.is_fraud) {
    items.push("Language risk is elevated and model confidence crossed the fraud threshold.");
    items.push("Recommended action: hold funds and require secondary verification.");
  } else if (result.is_mule_account) {
    items.push("Network behavior is suspicious even though message fraud confidence is lower.");
    items.push("Recommended action: step-up monitoring and verify beneficiary identity.");
  } else if (result.fraud_probability >= 0.4) {
    items.push("No hard fraud trigger, but the event sits in the watch zone.");
    items.push("Recommended action: continue enhanced monitoring for repeat activity.");
  } else {
    items.push("Message and account signals remain within the secure operating range.");
    items.push("Recommended action: standard monitoring is sufficient.");
  }

  if (result.warning) {
    items.push(result.warning);
  }

  recommendations.innerHTML = items
    .map((item) => `<div class="recommendation">${item}</div>`)
    .join("");
}

function renderResult(result) {
  const fraudProbability = Number(result.fraud_probability || 0);
  const senderRisk = Number(result.sender_risk_score || 0);
  const isFraud = Boolean(result.is_fraud);
  const isMule = Boolean(result.is_mule_account);

  riskPercent.textContent = `${Math.round(fraudProbability * 100)}%`;
  networkRisk.textContent = `${Math.round(senderRisk * 100)}%`;
  muleSignal.textContent = isMule ? "Clustered" : "Clear";
  setGauge(fraudProbability, isFraud);
  setStatus(isFraud);
  renderRecommendations(result);
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function setStageState(activeIndex) {
  stageCards.forEach((card, index) => {
    card.classList.remove("active", "done");
    if (index < activeIndex) {
      card.classList.add("done");
    } else if (index === activeIndex) {
      card.classList.add("active");
    }
  });
}

async function runStages() {
  phaseBox.classList.remove("hidden");
  for (let index = 0; index < stages.length; index += 1) {
    const stage = stages[index];
    setStageState(index);
    phaseTitle.textContent = stage.title;
    phaseDetail.textContent = stage.detail;
    await sleep(850);
  }
  setStageState(stages.length);
}

async function analyze() {
  const message = messageInput.value.trim();
  const account = accountInput.value.trim();

  if (!message || !account) {
    showError("Message analysis and account ID are both required.");
    return;
  }

  clearError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const fetchPromise = fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        sender_account: account,
        transaction_sent: false,
        threshold: 0.5,
      }),
    }).then(async (response) => {
      const payload = await response.json().catch(() => ({ detail: "Analysis failed." }));
      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }
      return payload;
    });

    const [payload] = await Promise.all([fetchPromise, runStages()]);
    renderResult(payload);
  } catch (error) {
    showError(error.message || "Unable to analyze this request.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
}

function animateCounters() {
  counters.forEach((counter) => {
    const target = Number(counter.dataset.counter);
    const suffix = counter.dataset.suffix || "";
    const start = performance.now();
    const duration = 1100;

    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      const value = target * progress;
      counter.textContent = suffix === "%" ? `${value.toFixed(2)}${suffix}` : `${Math.round(value)}${suffix}`;
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  });
}

sampleBtn.addEventListener("click", () => {
  messageInput.value = sampleMessage;
  accountInput.value = "TXN-2389-ALPHA";
  clearError();
});

analyzeBtn.addEventListener("click", analyze);

setGauge(0.17, false);
setStatus(false);
animateCounters();
