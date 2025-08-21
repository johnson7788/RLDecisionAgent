
import os

from dotenv import load_dotenv

load_dotenv()

# Required
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is required for RULER functionality when using openai/o4-mini."
    )

# Optional
# os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"

if not os.environ.get("WANDB_API_KEY"):
    print("WANDB_API_KEY is not set. We'll skip logging metrics to Weights & Biases.")


# <a name="Environment"></a>
# 
# ### Email Search Environment
# 
# ART allows your agent to learn by interacting with its environment. In this example, we'll create an environment where the agent can search through emails and answer questions about them.
# 
# The agent will have access to three tools:
# 
# 1. `search_inbox` - Search for emails by keywords
# 2. `read_email` - Read a specific email by message ID
# 3. `return_final_answer` - Return the final answer with source email IDs
# 

# In[3]:


import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from textwrap import dedent
from typing import List, Literal, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from pydantic import BaseModel, Field
from tqdm import tqdm


# Email and Scenario data models
class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []  # Populated from recipients table
    cc_addresses: List[str] = []  # Populated from recipients table
    bcc_addresses: List[str] = []  # Populated from recipients table
    body: Optional[str] = None
    file_name: Optional[str] = None


class Scenario(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids (strings) of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


@dataclass
class SearchResult:
    message_id: str
    snippet: str


class FinalAnswer(BaseModel):
    answer: str
    source_ids: list[str]


# Database configuration
DB_PATH = "./enron_emails.db"
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"
SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"

# Global database connection
db_conn = None


def create_email_database():
    """Create the email database from Hugging Face dataset"""
    print("Creating email database from Hugging Face dataset...")
    print(
        "This will download and process the full Enron email dataset - this may take several minutes..."
    )

    # Database schema
    SQL_CREATE_TABLES = """
    DROP TABLE IF EXISTS recipients;
    DROP TABLE IF EXISTS emails_fts;
    DROP TABLE IF EXISTS emails;

    CREATE TABLE emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        subject TEXT,
        from_address TEXT,
        date TEXT,
        body TEXT,
        file_name TEXT
    );

    CREATE TABLE recipients (
        email_id TEXT,
        recipient_address TEXT,
        recipient_type TEXT
    );
    """

    SQL_CREATE_INDEXES_TRIGGERS = """
    CREATE INDEX idx_emails_from ON emails(from_address);
    CREATE INDEX idx_emails_date ON emails(date);
    CREATE INDEX idx_emails_message_id ON emails(message_id);
    CREATE INDEX idx_recipients_address ON recipients(recipient_address);
    CREATE INDEX idx_recipients_type ON recipients(recipient_type);
    CREATE INDEX idx_recipients_email_id ON recipients(email_id);
    CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

    CREATE VIRTUAL TABLE emails_fts USING fts5(
        subject,
        body,
        content='emails',
        content_rowid='id'
    );

    CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts (rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
        DELETE FROM emails_fts WHERE rowid=old.id;
    END;

    CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
        UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
    END;
    """

    # Create database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    # Load dataset
    print("Loading full email dataset...")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )

    dataset = load_dataset(
        EMAIL_DATASET_REPO_ID, features=expected_features, split="train"
    )
    print(f"Dataset contains {len(dataset)} total emails")

    # Populate database with ALL emails (not limited to 1000)
    print("Populating database with all emails...")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()  # Track (subject, body, from) tuples for deduplication

    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        # Apply the same filters as the original project
        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)

        # Filter out very long emails and those with too many recipients
        if len(body) > 5000:
            skipped_count += 1
            continue

        if total_recipients > 30:
            skipped_count += 1
            continue

        # Deduplication check (same as original project)
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        else:
            processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (message_id, subject, from_address, date_str, body, file_name),
        )

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((message_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((message_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((message_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
            """,
                recipient_data,
            )

        record_count += 1

    conn.commit()

    # Create indexes and triggers
    print("Creating indexes and FTS...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"Successfully created database with {record_count} emails.")
    print(f"Skipped {skipped_count} emails due to length/recipient limits.")
    print(f"Skipped {duplicate_count} duplicate emails.")
    return conn


def get_db_connection():
    """Get database connection"""
    global db_conn
    if db_conn is None:
        if os.path.exists(DB_PATH):
            print(f"Loading existing database from {DB_PATH}")
            db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        else:
            db_conn = create_email_database()
    return db_conn


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """Search the email database based on keywords and filters"""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[str | int] = []

    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter
    where_clauses.append("""
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.message_id
        ))
    """)
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append("""
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.message_id
            )
        """)
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]


def read_email(message_id: str) -> Optional[Email]:
    """Retrieve a single email by its message_id"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get email details
    cursor.execute(
        "SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
        (message_id,),
    )
    email_row = cursor.fetchone()

    if not email_row:
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Get recipients
    cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?",
        (message_id,),
    )
    recipient_rows = cursor.fetchall()

    to_addresses = []
    cc_addresses = []
    bcc_addresses = []

    for addr, type_val in recipient_rows:
        if type_val.lower() == "to":
            to_addresses.append(addr)
        elif type_val.lower() == "cc":
            cc_addresses.append(addr)
        elif type_val.lower() == "bcc":
            bcc_addresses.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )


def load_training_scenarios(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Scenario]:
    """Load training scenarios from Hugging Face dataset"""
    print(f"Loading {split} scenarios from Hugging Face...")
    dataset: Dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split=split)

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if shuffle or (seed is not None):
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    # Convert each row to a Scenario object
    scenarios = [Scenario(**row, split=split) for row in dataset]

    if max_messages is not None:
        scenarios = [s for s in scenarios if len(s.message_ids) <= max_messages]

    if shuffle:
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(scenarios)
        else:
            random.shuffle(scenarios)

    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"Loaded {len(scenarios)} scenarios.")
    return scenarios


# Load training scenarios
training_scenarios = load_training_scenarios(
    split="train", limit=50, max_messages=1, shuffle=True, seed=42
)

print("Email search environment created with full Enron dataset!")
print(
    f"Database contains the complete email dataset, loaded {len(training_scenarios)} training scenarios."
)

# print first scenario
print("\nSample scenario")
print("id:", training_scenarios[0].id)
print("question:", training_scenarios[0].question)
print("answer:", training_scenarios[0].answer)
print("message_ids:", training_scenarios[0].message_ids)
print("how_realistic:", training_scenarios[0].how_realistic)
print("inbox_address:", training_scenarios[0].inbox_address)
print("query_date:", training_scenarios[0].query_date)
print("split:", training_scenarios[0].split)


# ### Creating a Model
# 
# Now that we've defined the rules of our environment, we can create a model that will learn to search emails effectively. We'll use a Qwen 2.5 7B model for this example.
# 

# In[ ]:


import art
from art.local import LocalBackend

random.seed(42)

# Declare the model
model = art.TrainableModel(
    name="email-agent-001",
    project="email-search-agent",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

# To run on a T4, we need to override some config defaults.
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=8192,
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    ),
)

# Initialize the server
backend = LocalBackend(
    # Normally we don't want to run the server in-process, but for the output
    # to show up properly on Google Colab we'll enable this.
    in_process=True,
    path="./.art",
)

# Register the model with the local Backend (sets up logging, inference, and training)
await model.register(backend)


# <a name="Rollout"></a>
# 
# ### Defining a Rollout
# 
# A rollout is a single episode of an agent performing its task. In this example, the rollout function presents the agent with an email search scenario, and the agent uses the available tools to search for emails and answer the question.
# 
# When the agent provides a final answer, the `correct` metric is calculated based on whether the answer is correct.
# 

# In[ ]:


import weave
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm import acompletion
from tenacity import retry, stop_after_attempt

import art
from art.utils.litellm import convert_litellm_choice_to_openai

if os.getenv("WANDB_API_KEY", ""):
    weave.init(model.project, settings={"print_call_link": False})

MAX_TURNS = 10


class CorrectnessJudgeResponse(BaseModel):
    reasoning: str = Field(description="Explanation of the reasoning process.")
    accept: bool = Field(description="Whether the AI answer should be accepted.")


@retry(stop=stop_after_attempt(3))
async def judge_correctness(
    scenario: Scenario, answer: str
) -> CorrectnessJudgeResponse:
    system_prompt = dedent(
        """
        You are given a question, the reference answer (labelled **Reference answer**), and an answer generated by an AI assistant (labelled **AI answer**).

        Your task is to decide whether the AI answer is correct and should be accepted. You should accept the answer if it contains the relevant information from the reference answer. You should not accept the answer if it is missing information relevant to the question, or if it contradicts the reference answer.
        """
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {scenario.question}\n"
                f"Reference answer: {scenario.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    response = await acompletion(
        model="openai/gpt-4.1",
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )

    first_choice = response.choices[0]
    raw_content = first_choice.message.content or "{}"

    try:
        return CorrectnessJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        return CorrectnessJudgeResponse(
            reasoning=f"Parse error: {e}\nRaw: {raw_content}", accept=False
        )


class ProjectTrajectory(art.Trajectory):
    final_answer: FinalAnswer | None = None


class EmailScenario(BaseModel):
    step: int
    scenario: Scenario


@weave.op
async def rollout(model: art.Model, email_scenario: EmailScenario) -> ProjectTrajectory:
    scenario = email_scenario.scenario

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": email_scenario.step,
        },
    )

    system_prompt = dedent(
        f"""
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {MAX_TURNS} turns to find the answer, so if your first search doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
        """
    )

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]

    def search_inbox(keywords: list[str]) -> list[dict]:
        """Search the inbox for emails matching the given keywords and return
        a list of dictionaries so the LLM can easily consume them."""
        results = search_emails(
            inbox=scenario.inbox_address,
            keywords=keywords,
            sent_before=scenario.query_date,
        )
        return [asdict(result) for result in results]

    def return_final_answer(
        answer: str, reference_message_ids: list[str]
    ) -> FinalAnswer:
        """Return the final answer and the message IDs of the emails that were used to generate the answer."""
        return FinalAnswer(answer=answer, source_ids=reference_message_ids)

    tools = [search_inbox, read_email, return_final_answer]
    tools_by_name = {t.__name__: t for t in tools}
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    if model.trainable:
        litellm_model_name = f"hosted_vllm/{model.name}"
    else:
        litellm_model_name = model.name

    for _ in range(MAX_TURNS):
        response = await acompletion(
            model=litellm_model_name,
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
            temperature=1,
            messages=traj.messages(),
            caching=False,
            tools=traj.tools,
        )

        response_message = response.choices[0].message
        traj.messages_and_choices.append(
            convert_litellm_choice_to_openai(response.choices[0])
        )

        if not response_message.tool_calls:
            return traj

        try:
            for tool_call in response_message.tool_calls:
                tool_name: str = tool_call.function.name
                if tool_name in tools_by_name:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_to_call = tools_by_name[tool_name]
                    result = tool_to_call(**tool_args)
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(result),
                        }
                    )

                    if tool_name == "return_final_answer":
                        traj.final_answer = result
                        # Score the trajectory
                        if traj.final_answer:
                            correctness_judge_response = await judge_correctness(
                                scenario, traj.final_answer.answer
                            )
                            traj.metrics["correct"] = float(
                                correctness_judge_response.accept
                            )
                        return traj
        except Exception as e:
            print(f"Error parsing tool calls: {e}")
            return traj

    return traj


print("Rollout function defined!")


# <a name="ruler"></a>
# 
# ### How RULER works
# 
# **RULER** leverages two key insights:
# 
# 1. Relative scoring is easier than absolute scoring: It's easier for an LLM to rank several solutions relative to each other than to score them in isolation
# 2. GRPO only needs relative scores: Since GRPO normalizes scores within each group, only the relative rankings matter, not absolute values
# 
# The process:
# 
# 1. Generate N trajectories for a given scenario
# 2. Pass all N trajectories to **RULER**
# 3. **RULER** deduplicates common prefixes (e.g., identical system messages)
# 4. An LLM judge scores each trajectory from 0 to 1 based on goal achievement
# 5. These scores are used directly as rewards in GRPO training
# 
# To learn more about **RULER**, check out the [RULER docs](https://art.openpipe.ai/fundamentals/ruler).
# 

# In[6]:


import art
from art.rewards import ruler_score_group

# Test RULER with a simple example
base_messages = [
    {"role": "system", "content": "You count numbers using numeric symbols."},
    {"role": "user", "content": "Count to 10."},
]

good_trajectory = art.Trajectory(
    messages_and_choices=[
        *base_messages,
        {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
    ],
    reward=0,
)

mediocre_trajectory = art.Trajectory(
    messages_and_choices=[
        *base_messages,
        {
            "role": "assistant",
            "content": "one, two, three, four, five, six, seven, eight, nine, ten",
        },
    ],
    reward=0,
)

bad_trajectory = art.Trajectory(
    messages_and_choices=[
        *base_messages,
        {"role": "assistant", "content": "a, b, c, d, e, f, g, h, i, j"},
    ],
    reward=0,
)

sample_group = art.TrajectoryGroup(
    trajectories=[
        good_trajectory,
        mediocre_trajectory,
        bad_trajectory,
    ]
)

judged_group = await ruler_score_group(sample_group, "openai/o4-mini", debug=True)
assert judged_group is not None

# Display rankings
sorted_trajectories = sorted(
    judged_group.trajectories, key=lambda t: t.reward, reverse=True
)
for rank, traj in enumerate(sorted_trajectories, 1):
    messages = traj.messages()
    print(f"\nRank {rank}: Score {traj.reward:.3f}")
    print(f"  Response: {messages[-1]['content'][:50]}...")


# <a name="Loop"></a>
# 
# ### Training Loop
# 
# The training loop is where the magic happens. For each of the 10 steps defined below, the rollout function will be called multiple times in parallel. Each scenario will produce a trajectory, which will be used to update the model.
# 
# The `gather` step will wait for all of the trajectories to be generated, then it will use RULER to assign relative scores to each trajectory.
# 
# Our notebook will then delete all but the most recent checkpoint and train the model on the scored trajectories.
# 

# In[ ]:


# Training configuration
from art.utils import iterate_dataset

training_config = {
    "groups_per_step": 2,
    "num_epochs": 20,
    "rollouts_per_group": 4,
    "learning_rate": 1e-5,
    "max_steps": 20,
}

# Use iterate_dataset with real training scenarios (similar to train.py)
training_iterator = iterate_dataset(
    training_scenarios,  # Use real scenarios from Hugging Face
    groups_per_step=training_config["groups_per_step"],
    num_epochs=training_config["num_epochs"],
    initial_step=await model.get_step(),
)

for batch in training_iterator:
    print(
        f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
    )
    print(f"Batch contains {len(batch.items)} scenarios")

    # Create trajectory groups for this batch (similar to train.py)
    groups = []
    for scenario in batch.items:
        groups.append(
            art.TrajectoryGroup(
                (
                    rollout(model, EmailScenario(step=batch.step, scenario=scenario))
                    for _ in range(training_config["rollouts_per_group"])
                )
            )
        )

    # Gather all trajectory groups
    finished_groups = await art.gather_trajectory_groups(
        groups,
        pbar_desc="gather",
        max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
    )

    judged_groups = []
    for group in finished_groups:
        # Use RULER to assign relative scores to each trajectory
        judged_group = await ruler_score_group(group, "openai/o4-mini", debug=True)
        judged_groups.append(judged_group)

    await model.delete_checkpoints()
    await model.train(
        judged_groups,
        config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
        # Lowering the logprob_calculation_chunk_size is a memory saving measure
        # to allow longer sequences (up to 8192 tokens) to be processed on a T4.
        _config={"logprob_calculation_chunk_size": 8},
    )

    print(f"Completed training step {batch.step}")

    # Stop after max_steps for demo purposes (adjust as needed)
    if batch.step >= training_config["max_steps"]:
        break


# ### Using the Model
# 
# Just like that, you've trained an agent to search emails and answer questions! Now it's time to use your model outside of the training loop.
# 
# Check out the code below for a small demo of the model you just trained!
# 

# In[ ]:


# Test the trained model using the rollout function
# This avoids memory issues and uses the same inference path as training

print("Testing the trained model with a real scenario...\n")


# Use a scenario from our training set
test_scenario = training_scenarios[1]

print(f"Test scenario ID: {test_scenario.id}")
print(f"Question: {test_scenario.question}")
print(f"Expected answer: {test_scenario.answer}")
print(f"Reference message IDs: {test_scenario.message_ids}")
print(f"Inbox: {test_scenario.inbox_address}")
print(f"Query date: {test_scenario.query_date}")
print("-" * 50)

# Run the rollout function with the trained model
test_email_scenario = EmailScenario.model_validate(
    {"step": 0, "scenario": test_scenario.model_dump()}
)
result_trajectory = await rollout(model, test_email_scenario)

print("Agent's trajectory:")
print("-" * 20)

# Display the conversation
messages = result_trajectory.messages()
for i, msg in enumerate(messages):
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])

    if role == "system":
        print(
            f"[SYSTEM]: {content[:100]}..."
            if len(content) > 100
            else f"[SYSTEM]: {content}"
        )
    elif role == "user":
        print(f"[USER]: {content}")
    elif role == "assistant":
        if tool_calls:
            print(f"[ASSISTANT]: {tool_calls}")
        if content:
            print(f"[ASSISTANT]: {content}")
    elif role == "tool":
        tool_name = msg.get("name", "unknown_tool")
        print(
            f"[TOOL - {tool_name}]: {content[:200]}..."
            if len(content) > 200
            else f"[TOOL - {tool_name}]: {content}"
        )

    print()

print("-" * 50)
if result_trajectory.final_answer:
    print(f"Agent's Final Answer: {result_trajectory.final_answer.answer}")
    print(f"Source IDs Used: {result_trajectory.final_answer.source_ids}")
else:
    print("No final answer provided by the agent")

print(f"\nExpected Answer: {test_scenario.answer}")
print(f"Expected Source IDs: {test_scenario.message_ids}")

print("\n🎉 Email search agent testing completed!")
print(
    "The agent used the same inference path as during training, avoiding memory issues."
)


# <div class="align-center">
# <a href="https://github.com/openpipe/art"><img src="https://github.com/openpipe/art/raw/main/assets/ART_pill.png" height="50"></a>
# <a href="https://discord.gg/zbBHRUpwf4"><img src="https://github.com/openpipe/art/raw/main/assets/Discord_pill.png" height="50"></a>
# <a href="https://art.openpipe.ai"><img src="https://github.com/openpipe/art/raw/main/assets/Documentation_pill.png" height="50"></a>
# 
# Questions? Join the Discord and ask away! For feature requests or to leave a star, visit our [Github](https://github.com/openpipe/art).
# 
# </div>
# 
