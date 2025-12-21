## **1. MODEL CONFIGURATION (1B Parameters)**

For a 1B param model with your vocab size targets, use this optimal configuration:

```python
# Model Architecture Config
model_config = {
    "vocab_size": 65536,
    "hidden_size": 2048,  # d_model
    "intermediate_size": 8192,  # MLP hidden dim (4x hidden_size)
    "num_hidden_layers": 24,  # depth
    "num_attention_heads": 16,  # heads
    "num_key_value_heads": 4,  # GQA for efficiency (grouped-query attention)
    "max_position_embeddings": 8192,  # target context length
    "rope_theta": 10000.0,  # RoPE base
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "bfloat16",
}
```

---

## **2. DATASET STRATEGY**

### **Phase 1: Pretraining (50B tokens, 2048 ctx)**
**Source**: `HuggingFaceTB/smollm-corpus`  
**Subsets to use**:
- `fineweb-edu-dedup`: 190M examples, 220B tokens total → **sample 30B tokens**
- `cosmopedia-v2`: 39M examples, 28B tokens total → **use all 28B tokens**
- `python-edu`: 7.7M examples, 4B tokens → **use all 4B tokens**

**Total**: ~62B tokens available, you'll use **50B** (80% of it)

**Sampling code**:
```python
from datasets import load_dataset

# Load and stream samples
ds = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", streaming=True, split="train")
# Filter for quality score > 0.7 (high-quality subset)
filtered = ds.filter(lambda x: x['metadata']['score'] > 0.7)
# Take ~30B tokens (approx 60M examples at ~500 tokens each)
pretrain_data = filtered.take(60_000_000)
```

### **Phase 2: Mid-training (50B tokens, 2048 ctx) - REASONING + PLANNING**

This is where you need **curated reasoning/planning data**. Here's how to get it:

#### **Option A: Existing Datasets (Start Here)**

**Reasoning Datasets** (all on HuggingFace):
1. **`HuggingFaceTB/FineMath`** - 50B tokens of math problems
   - Use 15B tokens for math reasoning
2. **`codeparrot/apps`** - 10K programming problems with solutions
   - Use 10B tokens for code reasoning
3. **`gsm8k`** - 8.5K grade school math with reasoning chains
   - Use 10B tokens for step-by-step reasoning
4. **`HuggingFaceTB/stack-edu`** - Educational code
   - Use 10B tokens for structured reasoning

**Planning Datasets** (harder to find, but these work):
1. **`openai/gsm8k`** - The "thought" field contains planning steps
2. **`BAAI/COIG`** - Complex instruction following with multi-step plans
3. **`HuggingFaceTB/SmolLM3-blueprint`** - Contains planning traces

**Total existing**: ~45B tokens available. You need **50B**, so...

#### **Option B: Synthetic Planning Data Generation**

Since planning datasets are scarce, **generate 5B tokens** using Kimi K2 and Gemini 3.0 Pro:

First, burn as many Azure credits on Gemini 3.0 Pro token generation:
- Generate 800k planning examples (~4B tokens)
- Cost: ~$80 on Gemini 3.0 Pro (0.0001 per 1K tokens)

Then for the remaining 1B 

**Synthetic Generation Script** (run this on your RTX 5080):
```python
import openai  # use kimi client instead
import json

# Planning prompt template
PLANNING_PROMPT = """
Generate a complex planning problem with step-by-step reasoning.

Problem: {domain} - {task_description}
Required: Provide <problem>, <thinking>, and <answer> fields.

Example domains: project management, software architecture, research planning, logistics.

Format:
<problem>[Clear problem statement]</problem>
<thinking>[Step-by-step plan: 1. Analyze constraints, 2. Identify resources, 3. Create timeline, 4. Evaluate risks]</thinking>
<answer>[Final plan/solution]</answer>
"""

domains = [
    "software_system_design",
    "research_project_timeline", 
    "logistics_optimization",
    "resource_allocation",
    "multi_agent_coordination"
]

def generate_planning_data(api_client, domain, num_samples=1000):
    data = []
    for i in range(num_samples):
        task = f"Create a plan for {domain.replace('_', ' ')} scenario #{i}"
        
        response = api_client.chat.completions.create(
            model="kimi-k2-thinking",  # or gemini-3.0-pro
            messages=[{
                "role": "user",
                "content": PLANNING_PROMPT.format(domain=domain, task_description=task)
            }],
            temperature=0.7,
            max_tokens=2048
        )
        
        data.append({
            "domain": domain,
            "problem": extract_tag(response.choices[0].message.content, "problem"),
            "thinking": extract_tag(response.choices[0].message.content, "thinking"),
            "answer": extract_tag(response.choices[0].message.content, "answer")
        })
    
    return data
```

**Cost estimate**: 10B tokens × $0.0001/1K tokens = **$1000** (ouch)

**Better approach**: Use **Azure credits** for this generation:
- Azure GPT-4o: $0.03/1K tokens → $300 for 10B tokens
- Your $100 credit gets you **3.3B tokens** of synthetic planning data

**So realistic plan**: Generate **3B tokens** synthetic planning data + use **47B** from existing datasets = **50B total**

---

## **3. TRAINING PIPELINE**

### **Context Length Strategy: Gradual vs Direct**

**Direct to 2048 is fine.** Modern models don't "shock" - they just work. But for stability:

```python
# In your training config
"seq_len_schedule": {
    "phase1": 1024,  # First 10B tokens
    "phase2": 2048,  # Remaining 40B tokens
}
```

This is optional but recommended for loss stability.

### **Phase 1: Pretraining (50B tokens, 2048 ctx)**
- **Data**: smollm-corpus (50B tokens)
- **Hardware**: RTX 5080 (16GB)
- **Time**: ~3-4 days at 200 tokens/sec
- **LR**: 3e-4 (cosine decay)
- **Batch**: 2 per GPU (gradient accumulation = 8 → effective batch 16)
- **Optimizer**: AdaFactor

### **Phase 2: Mid-training (50B tokens, 2048 ctx)**
- **Data**: 47B existing reasoning + 3B synthetic planning
- **Hardware**: RTX 5080
- **Time**: ~3-4 days
- **LR**: 1e-4 (lower for fine-tuning)
- **Batch**: 2 per GPU
- **Optimizer**: AdaFactor

### **Phase 3: Long-context + SFT/DPO (25B tokens, 8192 ctx)**
**This is where you use Azure credits**

**Why 25B tokens?** Because:
- Long-context training is expensive (slower throughput)
- SFT + DPO can be combined here
- 25B is enough to learn 8K context extension
- Use SaladCloud which ranges from $0.25-$1.50 for RTX 5090. Rent 2 of them. 

**Time**: 25B tokens / (400 tokens/sec on A100) = **17.4 hours**  
**Cost**: 17.4h × $0.50/h = **$9.80** 

**What you learn**:
- Cloud orchestration 
- Multi-GPU parallelism (even with 1 GPU, you learn distributed training patterns)
- Long-context training dynamics
- SFT/DPO implementation

---

## **4. SYNTHETIC PLANNING DATA GENERATION (DETAILED)**

Since planning datasets are scarce, here's the **exact pipeline**:

### **Step 1: Use Azure Credits for Generation**

```bash
# Azure OpenAI Service (GPT-4o) is cheaper than direct API
# Price: $0.015/1K input tokens, $0.06/1K output tokens
# Average: ~$0.03/1K tokens generated

# Your $100 credit = 3.3B tokens
# Use 2B for SFT, 1B for DPO pairs
```

### **Step 2: Generation Script**

```python
import os
import json
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)

PLANNING_DOMAINS = {
    "software_architecture": [
        "Design a microservices system for an e-commerce platform",
        "Plan a database schema for a social media app",
        "Create a CI/CD pipeline for ML model deployment"
    ],
    "research_planning": [
        "Design an experiment to test a new drug",
        "Plan a literature review on LLM reasoning",
        "Create a data collection strategy for user studies"
    ],
    "logistics": [
        "Optimize delivery routes for 100 packages",
        "Schedule workers for a 24/7 operation",
        "Plan inventory for seasonal demand"
    ],
    "multi_agent": [
        "Coordinate 5 robots in a warehouse",
        "Plan a debate between AI assistants",
        "Design a collaborative task for virtual agents"
    ]
}

def generate_planning_example(domain, task):
    prompt = f"""Create a detailed planning problem with step-by-step reasoning.

Domain: {domain}
Task: {task}

Format:
<problem>Clear problem statement</problem>
<thinking>Step-by-step plan:
1. Analyze constraints
2. Identify resources needed
3. Create timeline
4. Evaluate risks
5. Finalize plan</thinking>
<answer>Optimized solution</answer>"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048
    )
    
    return parse_response(response.choices[0].message.content)

# Generate 500K examples (~2B tokens)
# This will cost ~$60 of your Azure credit
```

### **Step 3: DPO Pair Generation**

```python
def generate_dpo_pair(problem):
    # Generate good solution
    good = generate_planning_example(domain, task)
    
    # Generate bad solution (add noise)
    bad_prompt = prompt + "\n\nIMPORTANT: Make a calculation error in step 3."
    bad = generate_with_prompt(bad_prompt)
    
    return {
        "problem": problem,
        "chosen": good["thinking"] + good["answer"],
        "rejected": bad["thinking"] + bad["answer"]
    }
```

---

## **5. MERGING LONG-CONTEXT WITH SFT/DPO**

You're right that they're orthogonal. Here's how to **combine them efficiently**:

**Combined Phase 3 Training Loop**:
```python
# Pseudo-code for merged training
for batch in data_loader:
    # Long-context data: [doc1, doc2, problem, thinking, answer]
    # Shape: (batch, seq_len=8192)
    
    # Forward pass
    outputs = model(input_ids)
    
    # SFT loss: next-token prediction on thinking + answer
    sft_loss = cross_entropy(outputs.logits, labels)
    
    # DPO loss: contrastive loss on chosen vs rejected
    # Use same context, different thinking chains
    chosen_logits = model(chosen_input_ids)
    rejected_logits = model(rejected_input_ids)
    dpo_loss = -F.logsigmoid(beta * (chosen_logps - rejected_logps)).mean()
    
    # Combined loss
    total_loss = sft_loss + 0.5 * dpo_loss
    
    # Backward pass
    total_loss.backward()
    
    # Long-context is learned implicitly through 8192 seq_len
    # SFT/DPO are learned explicitly through loss terms
```

**Why this works**:
- Single forward pass computes both losses
- No separate training phases
- Model learns long-context attention patterns while optimizing reasoning quality
- **25B tokens is enough** because you're not teaching reasoning from scratch, just refining it for long context

---

## **6. AZURE INFRASTRUCTURE SETUP**

**Step-by-step for Gemini 3.0 Pro**:

```bash
# 1. Create Azure VM with A100
az vm create \
  --resource-group myResourceGroup \
  --name llm-training-vm \
  --image nvidia:ngc_azure_ubuntu_22_04:latest \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_rsa.pub

# 2. Install dependencies
ssh azureuser@<vm-ip>
sudo apt update && sudo apt install -y python3-pip docker nvidia-container-toolkit
pip install torch transformers datasets deepspeed flash-attn

# 3. Mount storage for data
az storage account create --name llmdatastore --resource-group myResourceGroup
az storage share create --name datasets --account-name llmdatastore
# Upload your data to Azure Files

# 4. Run training with tmux
tmux new -s training
python train.py --config config_azure.yaml
# Detach: Ctrl+B then D

# 5. Monitor with wandb
# Logs automatically sync to cloud
```

**Cost tracking**:
```bash
# Check spend in real-time
az consumption usage list --start-date 2024-12-01 --end-date 2024-12-31
```

---

## **7. FINAL TOKEN ALLOCATION & COST**

| Phase | Tokens | Hardware | Time | Cost | What You Learn |
|-------|--------|----------|------|------|----------------|
| **Pretrain** | 50B | RTX 5080 | 4 days | $0 | Data pipeline, checkpointing |
| **Mid-train** | 47B existing + 3B planning | RTX 5080 | 4 days | $0 | Reasoning patterns, loss curves |
| **Long-context + SFT/DPO** | 25B | Azure A100 | 18 hours | $55 | Cloud, parallelism, long ctx |
| **C++ Inference** | - | RTX 5080 | 2 days | $0 | Deployment, quantization |
| **TOTAL** | **125B** | - | **~12 days** | **$55** | **Full stack** |

**Remaining Azure credit**: $45 for debugging/extra runs

---

## **8. PLANNING DATASET SOURCES (CONCRETE)**

**Existing datasets you can use TODAY**:
1. **`allenai/ai2_arc`**: 7,800 science problems with reasoning
2. **`google/boolq`**: 15K yes/no questions requiring reasoning
3. **`microsoft/orca-math-word-problems`**: 36K math problems
4. **`HuggingFaceTB/FineMath`**: 50B tokens of math reasoning

**For planning specifically**:
- **`BAAI/COIG`**: Complex instruction following (multi-step plans)
- **`openai/gsm8k`**: Use the "solution" field as planning traces
- **`deepmind/mathematics_dataset`**: Algorithmic planning in math

**If you need more**: Generate synthetic data using the script above. **3B tokens is realistic** with your Azure credit.

---

## **9. FEEDING THIS TO GEMINI 3.0 PRO**

**Prompt for Gemini**:
```
Create a production-ready training infrastructure for a 1B parameter LLM with the following specs:

MODEL CONFIG:
- 1B params, vocab_size=65536, hidden_size=2048, num_layers=24, num_heads=16, GQA=4
- RoPE theta=10000, max_ctx=8192, bfloat16

TRAINING PIPELINE:
- Phase 1: 50B tokens from smollm-corpus (fineweb-edu-dedup, cosmopedia-v2, python-edu) on RTX 5080, 2048 ctx, AdamW, 4 days
- Phase 2: 50B tokens (47B from FineMath + Stack-Edu + 3B synthetic planning) on RTX 5080, 2048 ctx, AdamW, 4 days
- Phase 3: 25B tokens on Azure A100 80GB, 8192 ctx, combined SFT/DPO + long-context, Adafactor, 18 hours

DATA GENERATION:
- Use Azure OpenAI GPT-4o to generate 3B tokens of synthetic planning data
- Domains: software_architecture, research_planning, logistics, multi_agent_coordination
- Format: <problem>...</problem><thinking>...</thinking><answer>...</answer>

INFRASTRUCTURE:
- Azure VM: Standard_NC24ads_A100_v4
- Storage: Azure Files for datasets
- Monitoring: Weights & Biases
- Orchestration: DeepSpeed ZeRO Stage 2

OUTPUT:
- Provide complete train.py, data generation scripts, azure-deployment.sh, and monitoring setup
- Include cost tracking and checkpointing logic
```

This gives Gemini everything it needs to generate the full infrastructure.

---

## **10. REALISTIC EXPECTATIONS**

**Your model will be good at**:
- Step-by-step reasoning (math, code, logic)
- Planning within 2K context (project plans, research designs)
- Following complex instructions

**It won't be**:
- GPT-4 level planner (you need 70B+ params for that)
- Perfect at 8K context (25B tokens is minimal for long-context)
- A "god-like planner" - but it will be **impressive for 1B**

**The win**: You'll have trained a **specialized reasoning/planning model** that punches above its weight on structured tasks, with full cloud training experience.

**Ship it.**

**IF YOU BELIEVE THAT THERES A BETTER SOLUTION, YOU ARE ALLOWED TO DEVIATE FROM THE PLAN---GIVE ME THE REASON WHY THOUGH. HOWEVER THE REQUIREMENTS OF THE PLAN MUST BE COMPLETED.**

IN ADDITION, I HAVE A REPO THAT ALREADY HAS THE PIPELINE BUT IT NEEDS EXTENSIVE UPDATES. GIVE ME THOSE UPDATES. 