<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form answer-format="flatten-objects" style="padding-bottom: 40px;">

  <!-- Hidden source fields from CSV -->
  <div id="mturk-data" style="display:none;">
    <div id="instruction-src">${instruction}</div>
    <div id="scenario-src">${scenario_description}</div>
    <div id="question-src">${question}</div>
    <div id="reasoning-src">${reasoning_snippet}</div>

    <div class="option-src" data-index="1">
      <span class="opt-value">${option_1_value}</span>
      <span class="opt-label">${option_1_label}</span>
    </div>
    <div class="option-src" data-index="2">
      <span class="opt-value">${option_2_value}</span>
      <span class="opt-label">${option_2_label}</span>
    </div>
    <div class="option-src" data-index="3">
      <span class="opt-value">${option_3_value}</span>
      <span class="opt-label">${option_3_label}</span>
    </div>
    <div class="option-src" data-index="4">
      <span class="opt-value">${option_4_value}</span>
      <span class="opt-label">${option_4_label}</span>
    </div>
    <div class="option-src" data-index="5">
      <span class="opt-value">${option_5_value}</span>
      <span class="opt-label">${option_5_label}</span>
    </div>
  </div>

  <!-- Instructions -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Instructions</h3>
    <div id="instruction-box" style="font-size: 15px; line-height: 1.6;"></div>

    <div style="margin-top: 18px; color: #b22222; font-weight: bold; background: #fff0f0; padding: 12px; border-left: 4px solid #b22222; border-radius: 6px;">
      ⚠️ Some questions may have known answers. Low accuracy may affect payment.
    </div>
  </div>

  <!-- Scenario -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Scenario</h3>
    <div id="scenario-box" style="font-size: 15px; line-height: 1.6;"></div>
  </div>

  <!-- Reasoning snippet -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Reasoning snippet</h3>
    <div id="reasoning-box" style="white-space: pre-wrap; line-height: 1.7; font-size: 15px; background: #f8f8f8; border: 1px solid #e8e8e8; border-radius: 8px; padding: 16px;"></div>
  </div>

  <!-- Question -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Question</h3>
    <div id="question-box" style="font-size: 16px; font-weight: 600; line-height: 1.6; margin-bottom: 18px;"></div>

    <div style="font-weight: bold; margin-bottom: 10px;">Answer choices</div>
    <div id="choices-container" style="display: flex; flex-direction: column; gap: 12px;"></div>

    <div id="validation-error" style="display:none; margin-top: 14px; color: #b22222; font-weight: bold; background: #fff0f0; padding: 12px; border-left: 4px solid #b22222; border-radius: 6px;">
      Please select one answer before submitting.
    </div>
  </div>

  <!-- Hidden outputs -->
  <input type="hidden" name="selected_label" id="selected_label" value="">
  <input type="hidden" name="task_id_echo" value="${task_id}">
  <input type="hidden" name="pair_id_echo" value="${pair_id}">
  <input type="hidden" name="pair_role_echo" value="${pair_role}">
  <input type="hidden" name="environment_echo" value="${environment}">

  <!-- Optional feedback -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto;">
    <h4>Optional Feedback</h4>
    <p>If anything was confusing or unclear, please let us know.</p>
    <textarea
      name="worker_feedback"
      rows="5"
      style="width: 100%; padding: 12px; font-size: 14px; border-radius: 6px; border: 1px solid #ccc; resize: vertical;"
      placeholder="Enter your feedback here..."></textarea>
  </div>

</crowd-form>

<script>
  function cleanText(text) {
    if (!text) return "";
    const trimmed = text.trim();
    if (
      trimmed === "nan" ||
      trimmed === "None" ||
      trimmed === "undefined" ||
      trimmed === "null"
    ) {
      return "";
    }
    return trimmed;
  }

  function buildChoices() {
    const container = document.getElementById("choices-container");
    container.innerHTML = "";

    const optionNodes = document.querySelectorAll(".option-src");
    let first = true;

    optionNodes.forEach((node, idx) => {
      const value = cleanText(node.querySelector(".opt-value").textContent);
      const label = cleanText(node.querySelector(".opt-label").textContent);

      if (!value || !label) return;

      const optionId = `worker_answer_${idx + 1}`;

      const wrapper = document.createElement("label");
      wrapper.setAttribute("for", optionId);
      wrapper.style.display = "flex";
      wrapper.style.alignItems = "flex-start";
      wrapper.style.gap = "12px";
      wrapper.style.padding = "14px 16px";
      wrapper.style.border = "1px solid #ddd";
      wrapper.style.borderRadius = "10px";
      wrapper.style.background = "#fafafa";
      wrapper.style.cursor = "pointer";

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "worker_answer";
      radio.id = optionId;
      radio.value = value;
      radio.style.marginTop = "3px";
      if (first) {
        radio.required = true;
        first = false;
      }

      radio.addEventListener("change", function () {
        document.getElementById("selected_label").value = label;
        document.getElementById("validation-error").style.display = "none";
      });

      const textWrap = document.createElement("div");
      textWrap.style.lineHeight = "1.5";

      const mainText = document.createElement("div");
      mainText.textContent = label;
      mainText.style.fontSize = "15px";

      textWrap.appendChild(mainText);
      wrapper.appendChild(radio);
      wrapper.appendChild(textWrap);
      container.appendChild(wrapper);
    });
  }

  function fillStaticText() {
    document.getElementById("instruction-box").textContent =
      cleanText(document.getElementById("instruction-src").textContent);

    document.getElementById("scenario-box").textContent =
      cleanText(document.getElementById("scenario-src").textContent);

    document.getElementById("question-box").textContent =
      cleanText(document.getElementById("question-src").textContent);

    document.getElementById("reasoning-box").textContent =
      cleanText(document.getElementById("reasoning-src").textContent);
  }

  function attachValidation() {
    const form = document.querySelector("crowd-form");
    form.addEventListener("submit", function (e) {
      const checked = document.querySelector('input[name="worker_answer"]:checked');
      if (!checked) {
        e.preventDefault();
        document.getElementById("validation-error").style.display = "block";
      }
    });
  }

  fillStaticText();
  buildChoices();
  attachValidation();
</script>